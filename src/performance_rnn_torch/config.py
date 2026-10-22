"""Default settings of the model, of training and of generation.

The model and training defaults are those of Magenta's Performance RNN
(magenta/models/performance_rnn/performance_model.py): three GRU layers of
512 units, no dropout, batches of 64, gradient norm clipped at 3, learning
rate 0.001.

They suit a large dataset such as MAESTRO and a GPU. On a laptop, or when
training is too slow, use a smaller model, e.g. -M "hidden_dim=256,gru_layers=2"
and -b 32. With few training files the model may learn them by heart (the
validation loss rises while the training loss keeps falling): add dropout,
e.g. -M "gru_dropout=0.3", or use a smaller model.
"""

import ast
from dataclasses import dataclass, fields, replace

import torch

from performance_rnn_torch.core.sequence import ControlSeq, EventSeq


def default_device(name: str = 'auto') -> torch.device:
    """'auto' picks CUDA, then Apple's MPS, then the CPU."""
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@dataclass(frozen=True)
class ModelConfig:
    event_dim: int = EventSeq.dim()
    control_dim: int = ControlSeq.dim()
    init_dim: int = 32
    hidden_dim: int = 512
    gru_layers: int = 3
    gru_dropout: float = 0.0

    def with_params(self, params: str) -> 'ModelConfig':
        """Override fields from text such as "hidden_dim=512,gru_layers=3"."""
        names = {f.name for f in fields(self)}
        values = {}
        for item in filter(None, (p.strip() for p in params.split(','))):
            key, _, value = item.partition('=')
            if key not in names:
                raise ValueError(f'unknown model parameter "{key}"; known: {", ".join(sorted(names))}')
            values[key] = ast.literal_eval(value)
        return replace(self, **values)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    num_epochs: int = 15
    window_size: int = 100  # events per training window
    stride_size: int = 10  # events between the starts of two windows
    learning_rate: float = 1e-3
    train_test_ratio: float = 0.3  # share of files used for validation
    early_stopping_patience: int = 5  # epochs without improvement; 0 = off
    use_transposition: bool = False  # random transposition by -6..5 semitones
    control_ratio: float = 1.0  # share of batches trained with controls
    teacher_forcing_ratio: float = 1.0
    clip_norm: float = 3.0
    saving_interval: float = 180.0  # seconds between saves of the last state
    enable_logging: bool = True  # TensorBoard
    seed: int = 0


@dataclass(frozen=True)
class GenerateConfig:
    batch_size: int = 8
    max_len: int = 1000
    greedy_ratio: float = 0.0  # share of steps that take the most likely event
    beam_size: int = 0  # 0 = sampling instead of beam search
    temperature: float = 1.0
    top_k: int = 0  # 0 = all events
    top_p: float = 1.0  # 1 = all events
    stochastic_beam_search: bool = False
    init_zero: bool = False
