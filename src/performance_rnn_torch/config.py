"""Default settings of the model, of training and of generation."""

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
    hidden_dim: int = 256     # Magenta: 512
    gru_layers: int = 2       # Magenta: 3
    gru_dropout: float = 0.3

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
    batch_size: int = 32
    num_epochs: int = 15
    window_size: int = 100            # events per training window
    stride_size: int = 10             # events between the starts of two windows
    learning_rate: float = 1e-3
    train_test_ratio: float = 0.3     # share of files used for validation
    early_stopping_patience: int = 5  # epochs without improvement; 0 = off
    use_transposition: bool = False   # random transposition by -6..5 semitones
    control_ratio: float = 1.0        # share of batches trained with controls
    teacher_forcing_ratio: float = 1.0
    clip_norm: float = 1.0
    saving_interval: float = 180.0    # seconds between saves of the last state
    enable_logging: bool = True       # TensorBoard
    seed: int = 0


@dataclass(frozen=True)
class GenerateConfig:
    batch_size: int = 8
    max_len: int = 1000
    greedy_ratio: float = 0.0         # share of steps that take the most likely event
    beam_size: int = 0                # 0 = sampling instead of beam search
    temperature: float = 1.0
    top_k: int = 0                    # 0 = all events
    top_p: float = 1.0                # 1 = all events
    stochastic_beam_search: bool = False
    init_zero: bool = False
