"""Training functions for Performance RNN."""

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from performance_rnn_torch.core import Dataset, PerformanceRNN
from performance_rnn_torch.core.sequence import FORMAT_VERSION
from performance_rnn_torch.training.early_stopping import EarlyStopping
from performance_rnn_torch.utils.logger import setup_logger
from performance_rnn_torch.utils import helpers

logger = setup_logger('trainer')


def load_session(sess_path, model_config, device, learning_rate, reset_optimizer, strict_config=False):
    """
    Load a training session from a checkpoint file.

    Args:
        sess_path: Path to the session checkpoint file
        model_config: Model configuration dictionary
        device: PyTorch device (cpu, cuda, or mps)
        learning_rate: Learning rate for optimizer
        reset_optimizer: Whether to reset optimizer state
        strict_config: If True, only load session if config matches exactly (for optimization)

    Returns:
        tuple: (model, optimizer)
    """
    sess_loaded = False
    model_state = None
    optimizer_state = None

    try:
        sess = torch.load(sess_path, map_location=device, weights_only=True)
        check_format(sess, sess_path)

        # Check if model configuration matches
        if 'model_config' in sess:
            if sess['model_config'] != model_config:
                if strict_config:
                    # For optimization: don't load incompatible sessions, start fresh
                    logger.info(f'Session config mismatch - starting fresh model for this trial')
                    logger.info(f'  Session: {helpers.dict2params(sess["model_config"])}')
                    logger.info(f'  Trial:   {helpers.dict2params(model_config)}')
                else:
                    # For regular training: use session config to continue training
                    model_config = sess['model_config']
                    logger.info(f'Use session config instead: {helpers.dict2params(model_config)}')
                    model_state = sess['model_state']
                    optimizer_state = sess['model_optimizer_state']
                    logger.info(f'Session is loaded from {sess_path}')
                    sess_loaded = True
            else:
                # Configs match - load normally
                model_state = sess['model_state']
                optimizer_state = sess['model_optimizer_state']
                logger.info(f'Session is loaded from {sess_path}')
                sess_loaded = True
    except FileNotFoundError:
        logger.info('New session')

    # Initialize the model and optimizer
    model = PerformanceRNN(**model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # If a session was loaded, load the states of the model and optimizer
    if sess_loaded and model_state is not None:
        try:
            model.load_state_dict(model_state)
            if not reset_optimizer and optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
        except RuntimeError as e:
            logger.warning(f'Failed to load session state: {e}')
            logger.info('Starting with fresh model instead')
            # Model and optimizer are already initialized, just continue with fresh state

    return model, optimizer


def check_format(state, path):
    """Refuse checkpoints written for another event vocabulary."""
    version = state.get('format_version')
    if version != FORMAT_VERSION:
        raise ValueError(f'{path}: checkpoint format {version}, expected {FORMAT_VERSION}. '
                         'Checkpoints of older versions cannot be used: train a new model.')


def load_dataset(data_path):
    """
    Load preprocessed dataset from disk.

    Args:
        data_path: Path to directory containing preprocessed .data files

    Returns:
        Dataset: Loaded dataset object
    """
    dataset = Dataset(data_path, verbose=True)
    dataset_size = len(dataset.samples)
    assert dataset_size > 0, f"Dataset is empty at {data_path}"
    return dataset


def save_model(model, model_config, optimizer, sess_path):
    """
    Save model and optimizer state to checkpoint file.

    Args:
        model: PerformanceRNN model
        model_config: Model configuration dictionary
        optimizer: PyTorch optimizer
        sess_path: Path to save checkpoint
    """
    logger.info(f'Saving to {sess_path}')
    torch.save({
        'format_version': FORMAT_VERSION,
        'model_config': dict(model_config),  # a plain dict loads with weights_only=True
        'model_state': model.state_dict(),
        'model_optimizer_state': optimizer.state_dict()
    }, sess_path)
    logger.info('Done saving')


def last_session_path(sess_path):
    """Where the most recent state is saved: next to the best one, with '-last'."""
    sess_path = Path(sess_path)
    return sess_path.with_name(f'{sess_path.stem}-last{sess_path.suffix}')


def loss_update(init, window_size, events, event_dim, controls, model, loss_function, teacher_forcing_ratio):
    """
    Compute the loss of one batch: cross entropy of every predicted event.

    Args:
        init: Initial hidden state
        window_size: Sequence window size
        events: Event indices tensor [window_size, batch_size]
        event_dim: Event vocabulary dimension
        controls: Control sequence tensor, or None
        model: PerformanceRNN model
        loss_function: Loss criterion (e.g., CrossEntropyLoss)
        teacher_forcing_ratio: Probability of using teacher forcing

    Returns:
        torch.Tensor: Computed loss value
    """
    outputs = model.generate(
        init, window_size,
        events=events[:-1],
        controls=controls,
        teacher_forcing_ratio=teacher_forcing_ratio,
        output_type='logit'
    )

    assert outputs.shape[:2] == events.shape[:2]
    # Every event counts: there is no padding in the batches.
    return loss_function(outputs.reshape(-1, event_dim), events.reshape(-1))


def evaluate(model, data, batch_size, window_size, stride_size, event_dim, loss_function, device, seed=0):
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
        for events, controls in data.batches(batch_size, window_size, stride_size, shuffle=False):
            events = torch.as_tensor(events, dtype=torch.long, device=device)
            controls = torch.as_tensor(controls, dtype=torch.float32, device=device)
            init = torch.randn(events.shape[1], model.init_dim, generator=generator).to(device)
            loss = loss_update(init, window_size, events, event_dim, controls, model, loss_function, 1.0)
            losses.append(loss.item())
    model.train(was_training)
    return float(np.mean(losses)) if losses else math.inf


def train_model(model, optimizer, dataset, batch_size, num_epochs, window_size, stride_size,
                train_test_ratio, early_stopping_patience, event_dim, control_ratio,
                teacher_forcing_ratio, enable_logging, saving_interval, use_transposition,
                device, model_config, sess_path):
    """
    Train the PerformanceRNN model.

    The model with the lowest validation loss is saved to sess_path; the most
    recent state is saved next to it with '-last' (every saving_interval
    seconds, at the end, and on ctrl-C).

    Args:
        model: PerformanceRNN model
        optimizer: PyTorch optimizer
        dataset: Training dataset
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        window_size: Sequence window size
        stride_size: Stride for creating windows
        train_test_ratio: Ratio of data to use for validation
        early_stopping_patience: Epochs to wait before early stopping
        event_dim: Event vocabulary dimension
        control_ratio: Probability of using control conditioning
        teacher_forcing_ratio: Probability of using teacher forcing
        enable_logging: Whether to enable TensorBoard logging
        saving_interval: Interval (seconds) between checkpoints
        use_transposition: Whether to use transposition augmentation
        device: PyTorch device
        model_config: Model configuration dictionary
        sess_path: Path of the best checkpoint

    Returns:
        float: Lowest validation loss (inf if no validation batch was available)
    """
    writer = None
    if enable_logging:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
        except ImportError:
            logger.warning('TensorBoard is not installed (uv sync --extra tensorboard): logging disabled')

    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True) \
        if early_stopping_patience > 0 else None

    train_data, test_data = dataset.train_test_split(test_size=train_test_ratio)
    logger.info(f'Training on {len(train_data.samples)} files, validating on {len(test_data.samples)}')
    if test_data.get_length(batch_size, window_size, stride_size) == 0:
        logger.warning('The validation files are too short for one batch: '
                       'no validation loss, the best model cannot be chosen')

    Path(sess_path).parent.mkdir(parents=True, exist_ok=True)
    last_path = last_session_path(sess_path)
    best_val_loss = math.inf
    step = 0
    last_saving_time = time.time()
    loss_function = nn.CrossEntropyLoss()
    model.train()

    try:
        for epoch in range(num_epochs):
            batch_gen = train_data.batches(batch_size, window_size, stride_size)
            num_batches = train_data.get_length(batch_size, window_size, stride_size)
            pbar = tqdm(batch_gen, total=num_batches, desc=f'Epoch {epoch + 1}')

            for events, controls in pbar:
                if use_transposition:
                    offset = np.random.choice(np.arange(-6, 6))
                    events, controls = helpers.transposition(events, controls, offset)

                events = torch.as_tensor(events, dtype=torch.long, device=device)
                assert events.shape[0] == window_size

                if np.random.random() < control_ratio:
                    controls = torch.as_tensor(controls, dtype=torch.float32, device=device)
                    assert controls.shape[0] == window_size
                else:
                    controls = None

                init = torch.randn(batch_size, model.init_dim, device=device)

                train_loss = loss_update(init, window_size, events, event_dim, controls, model,
                                         loss_function, teacher_forcing_ratio)
                optimizer.zero_grad()
                train_loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                pbar.set_postfix({'loss': train_loss.item()}, refresh=True)
                if writer is not None:
                    writer.add_scalar('model/train_loss', train_loss.item(), step)
                    # gradient norm before clipping, to spot exploding or vanishing gradients
                    writer.add_scalar('model/norm', norm.item(), step)
                step += 1

                if time.time() - last_saving_time > saving_interval:
                    save_model(model, model_config, optimizer, last_path)
                    last_saving_time = time.time()

            pbar.close()

            val_loss = evaluate(model, test_data, batch_size, window_size, stride_size,
                                event_dim, loss_function, device)
            logger.info(f'Epoch {epoch + 1}: validation loss {val_loss:.4f}')
            if writer is not None:
                writer.add_scalar('model/val_loss', val_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, model_config, optimizer, sess_path)

            if early_stopping is not None:
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    logger.info('Early stopping!')
                    break

    except KeyboardInterrupt:
        logger.info('Training interrupted')
    finally:
        save_model(model, model_config, optimizer, last_path)
        if writer is not None:
            writer.close()

    if best_val_loss == math.inf:
        # no validation was possible: keep the last state as the result
        save_model(model, model_config, optimizer, sess_path)
    return best_val_loss
