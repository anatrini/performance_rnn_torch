import copy
import os
import torch
import numpy as np

import utils
from sequence import ControlSeq
from tqdm import tqdm



class Dataset:
    def __init__(self, root, verbose=False):
        assert os.path.isdir(root), root
        paths = utils.find_files_by_extensions(root, ['.data'])
        self.root = root
        self.samples = []
        self.seqlens = []
        if verbose:
            paths = tqdm(list(paths), desc=root)
        for path in paths:
            eventseq, controlseq = torch.load(path)
            controlseq = ControlSeq.recover_compressed_array(controlseq)
            assert len(eventseq) == len(controlseq)
            self.samples.append((eventseq, controlseq))
            self.seqlens.append(len(eventseq))
        self.avglen = np.mean(self.seqlens)


    def train_test_split(self, test_size=0.2):
        # Check test length
        num_test = int(test_size * len(self.samples))

        # Split data
        train_data = Dataset(self.root)
        train_data.samples = self.samples[:-num_test]
        train_data.seqlens = self.seqlens[:-num_test]

        test_data = Dataset(self.root)
        test_data.samples = self.samples[-num_test:]
        test_data.seqlens = self.seqlens[-num_test:]

        return train_data, test_data

 
    def batches(self, batch_size, window_size, stride_size):
        indeces = [(i, range(j, j + window_size))
                   for i, seqlen in enumerate(self.seqlens) 
                   for j in range(0, seqlen - window_size, stride_size)]
        eventseq_batch = []
        controlseq_batch = []
        n = 0
        for ii in np.random.permutation(len(indeces)):
            i, r = indeces[ii]
            eventseq, controlseq = self.samples[i]
            eventseq = eventseq[r.start:r.stop]
            controlseq = controlseq[r.start:r.stop]
            eventseq_batch.append(eventseq)
            controlseq_batch.append(controlseq)
            n += 1
            if n == batch_size:
                if n == batch_size or ii == len(indeces) -1:
                    if n < batch_size:
                        # Padding if batch size is not as multiple of available batches number
                        padding = batch_size - n
                        eventseq_batch.extend([np.zeros_like(eventseq)] * padding)
                        controlseq_batch.extend([np.zeros_like(controlseq)] * padding)
                yield (np.stack(eventseq_batch, axis=1),
                       np.stack(controlseq_batch, axis=1))
                eventseq_batch.clear()
                controlseq_batch.clear()
                n = 0
    

    def get_length(self, batch_size, window_size, stride_size):
        total_windows = sum((seqlen - window_size) // stride_size + 1 for seqlen in self.seqlens)
        num_batches = total_windows // batch_size
        return num_batches


    def __repr__(self):
        return (f'Dataset(root="{self.root}", '
                f'samples={len(self.samples)}, '
                f'avglen={self.avglen})')
