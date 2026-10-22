"""Configuration settings for Performance RNN."""

import platform
import torch


# ========================================================================
# Set device according to OS and available devices
# ========================================================================

if platform.system() == 'Darwin':  # Mac OS
    if torch.backends.mps.is_available():  # Check if Metal is available
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
else:  # Linux or Windows
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========================================================================
# Model configuration factory function
# ========================================================================

def get_model_config():
    """
    Get model configuration with dimensions from EventSeq and ControlSeq.

    This function lazily imports EventSeq and ControlSeq to avoid circular imports.

    Returns:
        dict: Model configuration dictionary
    """
    from performance_rnn_torch.core.sequence import EventSeq, ControlSeq

    return {
        'init_dim': 32,
        'event_dim': EventSeq.dim(),
        'control_dim': ControlSeq.dim(),
        'hidden_dim': 256,  # Smaller for faster training (512 for better quality)
        'gru_layers': 2,    # Fewer layers = faster training (3 for better quality)
        'gru_dropout': 0.3,
    }


# For backward compatibility, create a lazy-loading dict-like class
class _LazyModelConfig(dict):
    """Dictionary that lazily computes model dimensions when first accessed."""

    def __init__(self):
        super().__init__()
        self._initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            self.update(get_model_config())
            self._initialized = True

    def __getitem__(self, key):
        self._ensure_initialized()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure_initialized()
        return super().__contains__(key)

    def get(self, key, default=None):
        self._ensure_initialized()
        return super().get(key, default)

    def items(self):
        self._ensure_initialized()
        return super().items()

    def keys(self):
        self._ensure_initialized()
        return super().keys()

    def values(self):
        self._ensure_initialized()
        return super().values()

    def copy(self):
        """Return a copy of the config, ensuring initialization first."""
        self._ensure_initialized()
        return super().copy()


model = _LazyModelConfig()

# ========================================================================
# Training configuration
# ========================================================================

train = {
    'batch_size': 32,  # Smaller batch for faster epochs (64 for better stability)
    'num_epochs': 15,  # Quick test (50+ for production)
    'window_size': 100,  # Shorter sequences = faster training (200 for better context)
    'stride_size': 10,  # Stride for creating training windows
    'learning_rate': 0.001,  # Learning rate for optimizer
    'train_test_ratio': 0.3,  # Ratio of data to use for testing
    'early_stopping_patience': 5,  # Epochs to wait before early stopping
    'use_transposition': False,  # Whether to use data augmentation via transposition
    'control_ratio': 1.0,  # Ratio for control conditioning
    'teacher_forcing_ratio': 0.5,  # Balanced teaching (1.0 for more guidance)
    'saving_interval': 180  # Saving interval in seconds
}


# ========================================================================
# Generation configuration
# ========================================================================

generate = {
    'batch_size': 8,  # Number of sequences to generate in parallel
    'max_len': 1000,  # Maximum length of generated sequences
    'greedy_ratio': 1.0,  # Ratio for greedy sampling (vs. probabilistic)
    'beam_size': 0,  # Beam size for beam search (0 = disabled)
    'temperature': 1.0,  # Temperature for probabilistic sampling
    'stochastic_beam_search': False,  # Whether to use stochastic beam search
    'init_zero': False  # Whether to initialize with zeros
}