"""Early stopping utilities for training."""

from typing import Optional
import numpy as np

from performance_rnn_torch import config
from performance_rnn_torch.utils.logger import setup_logger

logger = setup_logger('early_stopping')


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.

    Attributes:
        patience: Number of epochs to wait for improvement before stopping
        verbose: Whether to print messages
        counter: Current count of epochs without improvement
        best_score: Best validation score seen so far
        early_stop: Whether early stopping has been triggered
        val_loss_min: Minimum validation loss seen
        delta: Minimum change in score to qualify as an improvement
    """

    def __init__(
        self,
        patience: int = None,
        verbose: bool = False,
        delta: float = 0.0
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement after which training
                     will be stopped. If None, uses value from config.
            verbose: If True, prints messages when validation loss improves
            delta: Minimum change in the monitored quantity to qualify as an
                  improvement
        """
        if patience is None:
            patience = config.train['early_stopping_patience']

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss: float, model=None) -> None:
        """
        Check if training should stop based on validation loss.

        Args:
            val_loss: Current validation loss
            model: PyTorch model (optional, for future use with checkpointing)
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                logger.info(f'Validation loss initialized: {val_loss:.6f}')
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info('Early stopping triggered!')
        else:
            if self.verbose:
                logger.info(
                    f'Validation loss improved ({self.val_loss_min:.6f} → {val_loss:.6f})'
                )
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0