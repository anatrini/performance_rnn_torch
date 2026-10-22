"""Training functions for Performance RNN."""

import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from performance_rnn_torch.config import ModelConfig, TrainConfig
from performance_rnn_torch.core import Dataset, PerformanceRNN
from performance_rnn_torch.training.checkpoint import load_checkpoint, save_checkpoint
from performance_rnn_torch.training.early_stopping import EarlyStopping
from performance_rnn_torch.utils import helpers
from performance_rnn_torch.utils.logger import setup_logger

logger = setup_logger('trainer')


def set_seed(seed: int) -> None:
    """Same random choices at every run: windows, initial states, dropout."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def last_session_path(sess_path) -> Path:
    """Where the most recent state is saved: next to the best one, with '-last'."""
    sess_path = Path(sess_path)
    return sess_path.with_name(f'{sess_path.stem}-last{sess_path.suffix}')


def load_session(
    sess_path, model_config: ModelConfig, device: torch.device, learning_rate: float, reset_optimizer: bool = False
) -> tuple[PerformanceRNN, torch.optim.Optimizer, ModelConfig]:
    """Continue the training saved in sess_path, or start a new model.

    An existing checkpoint keeps its own model configuration.
    """
    sess_path = Path(sess_path)
    state = load_checkpoint(sess_path, device) if sess_path.exists() else None
    if state is not None:
        saved = ModelConfig(**state['model_config'])
        if saved != model_config:
            logger.info(f'Continuing {sess_path} with its own model configuration: {saved}')
        model_config = saved

    model = PerformanceRNN(**asdict(model_config)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if state is not None:
        model.load_state_dict(state['model_state'])
        if not reset_optimizer and state.get('optimizer_state'):
            optimizer.load_state_dict(state['optimizer_state'])
        logger.info(f'Session loaded from {sess_path}')
    else:
        logger.info('New session')
    return model, optimizer, model_config


def load_dataset(data_path) -> Dataset:
    """Load the preprocessed files (.npz) under data_path."""
    dataset = Dataset(str(data_path), verbose=True)
    assert len(dataset.samples) > 0, f'No preprocessed files in {data_path}: run `prnn preprocess`'
    return dataset


def batch_loss(
    model: PerformanceRNN,
    events: torch.Tensor,
    controls: torch.Tensor | None,
    init: torch.Tensor,
    teacher_forcing_ratio: float = 1.0,
) -> torch.Tensor:
    """Cross entropy of every predicted event of a batch [window, batch]."""
    logits = model(events, controls, init, teacher_forcing_ratio)
    return nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), events.reshape(-1))


def evaluate(model: PerformanceRNN, data: Dataset, cfg: TrainConfig, device: torch.device, seed: int = 0) -> float:
    """
    Mean loss on held-out data: no dropout, no gradients, true previous events
    as input, the same initial states at every call.

    Returns:
        float: Mean validation loss (inf when the data give no complete batch)
    """
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    with torch.no_grad():
        for events, controls in data.batches(cfg.batch_size, cfg.window_size, cfg.stride_size, shuffle=False):
            events = torch.as_tensor(events, device=device)
            controls = torch.as_tensor(controls, device=device)
            init = torch.randn(events.shape[1], model.init_dim, generator=generator).to(device)
            losses.append(batch_loss(model, events, controls, init).item())
    model.train(was_training)
    return float(np.mean(losses)) if losses else math.inf


def train_model(
    model: PerformanceRNN,
    optimizer: torch.optim.Optimizer,
    dataset: Dataset,
    cfg: TrainConfig,
    model_config: ModelConfig,
    sess_path,
    device: torch.device,
) -> float:
    """
    Train the model.

    The model with the lowest validation loss is saved to sess_path; the most
    recent state is saved next to it with '-last' (every cfg.saving_interval
    seconds, at the end, and on ctrl-C).

    Returns:
        float: Lowest validation loss (inf if no validation batch was available)
    """
    writer = None
    if cfg.enable_logging:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter()
        except ImportError:
            logger.warning('TensorBoard is not installed (uv sync --extra tensorboard): logging disabled')

    early_stopping = EarlyStopping(cfg.early_stopping_patience) if cfg.early_stopping_patience > 0 else None

    train_data, test_data = dataset.train_test_split(cfg.train_test_ratio, seed=cfg.seed)
    logger.info(f'Training on {len(train_data.samples)} files, validating on {len(test_data.samples)}')
    if test_data.get_length(cfg.batch_size, cfg.window_size, cfg.stride_size) == 0:
        logger.warning(
            'The validation files are too short for one batch: no validation loss, the best model cannot be chosen'
        )

    last_path = last_session_path(sess_path)
    best_val_loss = math.inf
    epoch = step = 0
    last_saving_time = time.time()
    model.train()

    def save(path):
        save_checkpoint(
            path, model, model_config, optimizer, epoch=epoch, best_val_loss=best_val_loss, train_config=asdict(cfg)
        )
        logger.info(f'Saved {path}')

    try:
        for epoch in range(1, cfg.num_epochs + 1):
            batches = train_data.batches(cfg.batch_size, cfg.window_size, cfg.stride_size)
            total = train_data.get_length(cfg.batch_size, cfg.window_size, cfg.stride_size)
            pbar = tqdm(batches, total=total, desc=f'Epoch {epoch}')

            for events, controls in pbar:
                if cfg.use_transposition:
                    offset = np.random.randint(-6, 6)
                    events, controls = helpers.transposition(events, controls, offset)

                events = torch.as_tensor(events, device=device)
                controls = torch.as_tensor(controls, device=device) if np.random.random() < cfg.control_ratio else None
                init = torch.randn(cfg.batch_size, model.init_dim, device=device)

                loss = batch_loss(model, events, controls, init, cfg.teacher_forcing_ratio)
                optimizer.zero_grad()
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
                optimizer.step()

                pbar.set_postfix({'loss': loss.item()}, refresh=False)
                if writer is not None:
                    writer.add_scalar('model/train_loss', loss.item(), step)
                    # gradient norm before clipping, to spot exploding or vanishing gradients
                    writer.add_scalar('model/norm', norm.item(), step)
                step += 1

                if time.time() - last_saving_time > cfg.saving_interval:
                    save(last_path)
                    last_saving_time = time.time()

            pbar.close()

            val_loss = evaluate(model, test_data, cfg, device)
            logger.info(f'Epoch {epoch}: validation loss {val_loss:.4f}')
            if writer is not None:
                writer.add_scalar('model/val_loss', val_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save(sess_path)

            if early_stopping is not None and early_stopping(val_loss):
                logger.info('Early stopping')
                break

    except KeyboardInterrupt:
        logger.info('Training interrupted')
    finally:
        save(last_path)
        if writer is not None:
            writer.close()

    if best_val_loss == math.inf:
        # no validation was possible: keep the last state as the result
        save(sess_path)
    return best_val_loss
