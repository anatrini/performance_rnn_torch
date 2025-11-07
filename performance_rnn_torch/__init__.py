"""
Performance RNN in PyTorch
===========================

A PyTorch implementation of Google's Performance RNN for generating expressive
piano performances with dynamics and timing.

This package provides tools for:
- Training Performance RNN models on MIDI data
- Generating new piano performances from trained models
- Preprocessing MIDI files into training data
- Hyperparameter optimization

Example usage:
    >>> from performance_rnn_torch import PerformanceRNN, Dataset, config
    >>> from performance_rnn_torch.utils import paths
    >>>
    >>> # Load a dataset
    >>> dataset = Dataset(str(paths.processed_dir))
    >>>
    >>> # Create a model
    >>> model = PerformanceRNN(**config.model).to(config.device)
    >>>
    >>> # Train the model (see scripts/train.py for full example)
    >>> # Generate music (see scripts/generate.py for full example)
"""

__version__ = '1.0.0'
__author__ = 'Alessandro Anatrini'
__license__ = 'MIT'

# Import configuration
from performance_rnn_torch import config

# Import core modules
from performance_rnn_torch.core import (
    EventSeq,
    ControlSeq,
    NoteSeq,
    Event,
    Control,
    PerformanceRNN,
    Dataset,
)

# Import training utilities
from performance_rnn_torch.training import EarlyStopping

# Import utilities
from performance_rnn_torch.utils import paths
from performance_rnn_torch.utils.logger import setup_logger
from performance_rnn_torch.utils.helpers import (
    find_files_by_extensions,
    event_indices_to_midi_file,
    transposition,
    dict2params,
    params2dict,
    compute_gradient_norm,
)

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__license__',
    # Configuration
    'config',
    # Core classes
    'EventSeq',
    'ControlSeq',
    'NoteSeq',
    'Event',
    'Control',
    'PerformanceRNN',
    'Dataset',
    # Training
    'EarlyStopping',
    # Utilities
    'paths',
    'setup_logger',
    'find_files_by_extensions',
    'event_indices_to_midi_file',
    'transposition',
    'dict2params',
    'params2dict',
    'compute_gradient_norm',
]