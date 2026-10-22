"""Early stopping utilities for training."""

import math

from performance_rnn_torch.utils.logger import setup_logger

logger = setup_logger('early_stopping')


class EarlyStopping:
    """Stop when the validation loss has not improved by more than delta for
    `patience` epochs in a row."""

    def __init__(self, patience: int, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.best = math.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Record one validation loss; True when training should stop."""
        if val_loss < self.best - self.delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f'No improvement for {self.counter} of {self.patience} epochs')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
