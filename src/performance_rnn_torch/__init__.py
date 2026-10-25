"""
Performance RNN in PyTorch
==========================

Expressive piano performances with timing and dynamics, after Magenta's
Performance RNN (Simon & Oore, 2017).

Example:
    >>> import torch
    >>> from performance_rnn_torch import load_model
    >>> model, config, _ = load_model('models/train.sess')
    >>> events = model.generate(torch.randn(1, model.init_dim), 1000)

The command line tool is `prnn` (see `prnn --help`).
"""

from importlib.metadata import version as _version

from performance_rnn_torch.config import GenerateConfig, ModelConfig, TrainConfig, default_device
from performance_rnn_torch.core import Control, ControlSeq, Dataset, EventSeq, NoteSeq, PerformanceRNN
from performance_rnn_torch.training import load_model, save_checkpoint

__version__ = _version('performance-rnn-torch')

__all__ = [
    '__version__',
    'Control',
    'ControlSeq',
    'Dataset',
    'EventSeq',
    'GenerateConfig',
    'ModelConfig',
    'NoteSeq',
    'PerformanceRNN',
    'TrainConfig',
    'default_device',
    'load_model',
    'save_checkpoint',
]
