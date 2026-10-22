"""Checkpoints: one format for training, generation and the OSC server."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from performance_rnn_torch.config import ModelConfig
from performance_rnn_torch.core.model import PerformanceRNN
from performance_rnn_torch.core.sequence import FORMAT_VERSION


def save_checkpoint(path, model: PerformanceRNN, model_config: ModelConfig,
                    optimizer: torch.optim.Optimizer | None = None, **info: Any) -> None:
    """Write the model, its configuration, the optimizer state and any extra
    information (epoch, validation loss, training settings).

    Only plain Python values and tensors are stored, so the file loads with
    torch.load(weights_only=True).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'format_version': FORMAT_VERSION,
        'model_config': asdict(model_config),
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        **info,
    }, path)


def load_checkpoint(path, device: torch.device | str = 'cpu') -> dict:
    """Read a checkpoint and check that it uses the current event vocabulary."""
    state = torch.load(path, map_location=device, weights_only=True)
    version = state.get('format_version')
    if version != FORMAT_VERSION:
        raise ValueError(f'{path}: checkpoint format {version}, expected {FORMAT_VERSION}. '
                         'Checkpoints of older versions cannot be used: train a new model.')
    return state


def load_model(path, device: torch.device | str = 'cpu') -> tuple[PerformanceRNN, ModelConfig, dict]:
    """The model of a checkpoint, in evaluation mode, with its configuration."""
    state = load_checkpoint(path, device)
    config = ModelConfig(**state['model_config'])
    model = PerformanceRNN(**asdict(config)).to(device)
    model.load_state_dict(state['model_state'])
    model.eval()
    return model, config, state
