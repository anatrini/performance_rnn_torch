"""Dataset class for loading and batching training data."""

import os
from collections.abc import Iterator

import numpy as np
from tqdm import tqdm

from performance_rnn_torch.core.sequence import ControlSeq, load_processed
from performance_rnn_torch.utils import helpers


class Dataset:
    """Preprocessed performances, kept compact in memory.

    Events are stored as uint16 and controls as a density class and a
    histogram in uint8 (13 bytes per event instead of 80 for float32 vectors);
    the control vectors are rebuilt batch by batch.
    """

    def __init__(self, root: str, verbose: bool = False):
        assert os.path.isdir(root), root
        paths = sorted(helpers.find_files_by_extensions(root, ['.npz']))
        if verbose:
            paths = tqdm(paths, desc=root)
        self._set(root, [load_processed(path, compressed=True) for path in paths])

    @classmethod
    def from_samples(cls, samples: list[tuple[np.ndarray, np.ndarray]], root: str = '') -> 'Dataset':
        """A dataset made of (events [n], controls [n, control_dim]) pairs in memory."""
        dens = ControlSeq.feat_ranges()['note_density']
        hist = ControlSeq.feat_ranges()['pitch_histogram']
        compressed = [
            (
                np.asarray(events, np.uint16),
                np.asarray(controls)[:, dens].argmax(1).astype(np.uint8),
                np.round(np.asarray(controls)[:, hist] * 255).astype(np.uint8),
            )
            for events, controls in samples
        ]
        return cls._from_compressed(compressed, root)

    @classmethod
    def _from_compressed(cls, samples, root):
        dataset = cls.__new__(cls)
        dataset._set(root, samples)
        return dataset

    def _set(self, root, samples):
        self.root = root
        self.samples = list(samples)  # (events, density, histogram) per file
        self.seqlens = [len(events) for events, _, _ in self.samples]
        self.avglen = float(np.mean(self.seqlens)) if self.seqlens else 0.0
        self.offsets = np.concatenate([[0], np.cumsum(self.seqlens)]).astype(np.int64)
        if self.samples:
            self.events = np.concatenate([s[0] for s in self.samples])
            self.density = np.concatenate([s[1] for s in self.samples])
            self.histogram = np.concatenate([s[2] for s in self.samples])

    def train_test_split(self, test_size: float = 0.3, seed: int = 0) -> tuple['Dataset', 'Dataset']:
        """Split by file, in random order: at least one file on each side."""
        n = len(self.samples)
        if n < 2:
            raise ValueError(
                f'{n} file(s) in the dataset: at least two are needed, one for training and one for validation'
            )
        num_test = min(max(1, round(test_size * n)), n - 1)
        order = np.random.default_rng(seed).permutation(n)
        test = [self.samples[i] for i in order[:num_test]]
        train = [self.samples[i] for i in order[num_test:]]
        return Dataset._from_compressed(train, self.root), Dataset._from_compressed(test, self.root)

    def _windows(self, window_size: int, stride_size: int) -> np.ndarray:
        """Start of every window, as a position in the concatenated arrays."""
        starts = [
            self.offsets[i] + np.arange(0, seqlen - window_size + 1, stride_size)
            for i, seqlen in enumerate(self.seqlens)
            if seqlen >= window_size
        ]
        return np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)

    def batches(
        self, batch_size: int, window_size: int, stride_size: int, shuffle: bool = True
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Batches of windows: events [window_size, batch_size] (int64) and
        controls [window_size, batch_size, control_dim] (float32).

        The last windows that do not fill a whole batch are left out.
        """
        starts = self._windows(window_size, stride_size)
        if shuffle:
            starts = np.random.permutation(starts)
        steps = np.arange(window_size)
        for first in range(0, len(starts) - batch_size + 1, batch_size):
            index = (starts[first : first + batch_size, None] + steps).T  # [window, batch]
            events = self.events[index].astype(np.int64)
            controls = ControlSeq.recover_compressed_arrays(
                self.density[index].reshape(-1), self.histogram[index].reshape(-1, 12)
            )
            yield events, controls.reshape(window_size, batch_size, -1)

    def get_length(self, batch_size: int, window_size: int, stride_size: int) -> int:
        """Number of batches that batches() yields."""
        return len(self._windows(window_size, stride_size)) // batch_size

    def __repr__(self):
        return f'Dataset(root="{self.root}", samples={len(self.samples)}, avglen={self.avglen})'
