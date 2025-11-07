"""Training functions for Performance RNN."""

import time
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from performance_rnn_torch.core import Dataset, PerformanceRNN
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
        sess = torch.load(sess_path, map_location=device, weights_only=False)

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
        'model_config': model_config,
        'model_state': model.state_dict(),
        'model_optimizer_state': optimizer.state_dict()
    }, sess_path)
    logger.info('Done saving')


def loss_update(init, window_size, events, event_dim, controls, model, loss_function, teacher_forcing_ratio):
    """
    Compute loss for a single training batch.

    Args:
        init: Initial hidden state
        window_size: Sequence window size
        events: Event indices tensor
        event_dim: Event vocabulary dimension
        controls: Control sequence tensor
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

    # Create padding mask
    mask = (events.view(-1) != 0).float()
    # Apply mask to loss calculation and normalize loss
    loss = loss_function(outputs.view(-1, event_dim), events.view(-1)) * mask
    loss = loss.sum() / mask.sum()

    return loss


def train_model(model, optimizer, dataset, batch_size, num_epochs, window_size, stride_size,
                train_test_ratio, early_stopping_patience, event_dim, control_ratio,
                teacher_forcing_ratio, enable_logging, saving_interval, use_transposition,
                device, model_config, sess_path):
    """
    Train the PerformanceRNN model.

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
        sess_path: Path to save checkpoints

    Returns:
        float: Average validation loss
    """
    if enable_logging:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

    if early_stopping_patience > 0:
        early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    else:
        early_stopping = None

    train_data, test_data = dataset.train_test_split(test_size=train_test_ratio)

    avg_val_loss = None
    last_saving_time = time.time()
    loss_function = nn.CrossEntropyLoss()

    try:
        for epoch in range(num_epochs):
            # Create a progress bar for this epoch
            batch_gen = train_data.batches(batch_size, window_size, stride_size)
            num_batches = train_data.get_length(batch_size, window_size, stride_size)

            # Create a progress bar
            pbar = tqdm(batch_gen, total=num_batches, desc=f'Progressing Epoch {epoch + 1}')

            for iteration, (events, controls) in enumerate(pbar):
                if use_transposition:
                    offset = np.random.choice(np.arange(-6, 6))
                    events, controls = helpers.transposition(events, controls, offset)

                events = torch.LongTensor(events).to(device)
                assert events.shape[0] == window_size

                if np.random.random() < control_ratio:
                    controls = torch.FloatTensor(controls).to(device)
                    assert controls.shape[0] == window_size
                else:
                    controls = None

                init = torch.randn(batch_size, model.init_dim).to(device)

                train_loss = loss_update(init, window_size, events, event_dim, controls, model, loss_function, teacher_forcing_ratio)
                model.zero_grad()
                train_loss.backward()

                norm = helpers.compute_gradient_norm(model.parameters())
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                # Update the tqdm bar with the current loss
                pbar.set_postfix({'loss': train_loss.item()}, refresh=True)

                if enable_logging:
                    # Check loss on train data on iteration-basis
                    writer.add_scalar('model/train_loss', train_loss.item(), iteration)
                    # Check gradient normalization to spot exploding or vanishing gradients
                    writer.add_scalar('model/norm', norm.item(), iteration)

                if time.time() - last_saving_time > saving_interval:
                    save_model(model, model_config, optimizer, sess_path)
                    last_saving_time = time.time()

            pbar.close()

            val_losses = []
            for test_events, test_controls in test_data.batches(batch_size, window_size, stride_size):
                test_events = torch.LongTensor(test_events).to(device)
                test_controls = torch.FloatTensor(test_controls).to(device)

                # Compute validation loss
                val_loss = loss_update(init, window_size, test_events, event_dim, test_controls, model, loss_function, teacher_forcing_ratio)
                val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses)
            if enable_logging:
                # Check validation loss on epoch-basis
                writer.add_scalar('model/val_loss', avg_val_loss, epoch)

            # Check if early stopping has been called
            if early_stopping is not None:
                early_stopping(avg_val_loss, model)
                if early_stopping.early_stop:
                    logger.info('Early stopping!')
                    save_model(model, model_config, optimizer, sess_path)
                    break

    except KeyboardInterrupt:
        save_model(model, model_config, optimizer, sess_path)
    finally:
        if enable_logging:
            writer.close()

        return avg_val_loss
