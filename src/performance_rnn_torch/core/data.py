"""Dataset class for loading and batching training data."""

import os
import torch
import numpy as np
from tqdm import tqdm

from performance_rnn_torch import config
from performance_rnn_torch.utils import helpers
from performance_rnn_torch.core.sequence import ControlSeq


class Dataset:
    def __init__(self, root, verbose=False):
        assert os.path.isdir(root), root
        paths = helpers.find_files_by_extensions(root, ['.data'])
        if verbose:
            paths = tqdm(list(paths), desc=root)
        samples = []
        for path in paths:
            # pickled numpy arrays written by `prnn preprocess`
            eventseq, controlseq = torch.load(path, weights_only=False)
            controlseq = ControlSeq.recover_compressed_array(controlseq)
            assert len(eventseq) == len(controlseq)
            samples.append((eventseq, controlseq))
        self._set(root, samples)

    @classmethod
    def from_samples(cls, samples, root=''):
        """A dataset made of (events, controls) pairs already in memory."""
        dataset = cls.__new__(cls)
        dataset._set(root, samples)
        return dataset

    def _set(self, root, samples):
        self.root = root
        self.samples = list(samples)
        self.seqlens = [len(events) for events, _ in self.samples]
        self.avglen = float(np.mean(self.seqlens)) if self.seqlens else 0.0

    def train_test_split(self, test_size=config.train['train_test_ratio'], seed=0):
        """Split by file, in random order: at least one file on each side."""
        n = len(self.samples)
        if n < 2:
            raise ValueError(f'{n} file(s) in the dataset: at least two are needed, '
                             'one for training and one for validation')
        num_test = min(max(1, round(test_size * n)), n - 1)
        order = np.random.default_rng(seed).permutation(n)
        test = [self.samples[i] for i in order[:num_test]]
        train = [self.samples[i] for i in order[num_test:]]
        return Dataset.from_samples(train, self.root), Dataset.from_samples(test, self.root)

    def _windows(self, window_size, stride_size):
        return [(i, start)
                for i, seqlen in enumerate(self.seqlens)
                for start in range(0, seqlen - window_size + 1, stride_size)]

    def batches(self, batch_size, window_size, stride_size, shuffle=True):
        """Batches of windows, [window_size, batch_size] events and controls.

        The last windows that do not fill a whole batch are left out.
        """
        windows = self._windows(window_size, stride_size)
        order = np.random.permutation(len(windows)) if shuffle else np.arange(len(windows))
        for first in range(0, len(windows) - batch_size + 1, batch_size):
            chosen = [windows[k] for k in order[first:first + batch_size]]
            yield (np.stack([self.samples[i][0][s:s + window_size] for i, s in chosen], axis=1),
                   np.stack([self.samples[i][1][s:s + window_size] for i, s in chosen], axis=1))

    def get_length(self, batch_size, window_size, stride_size):
        """Number of batches that batches() yields."""
        return len(self._windows(window_size, stride_size)) // batch_size

    def __repr__(self):
        return (f'Dataset(root="{self.root}", '
                f'samples={len(self.samples)}, '
                f'avglen={self.avglen})')
