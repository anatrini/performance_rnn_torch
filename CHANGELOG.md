# Changelog

## 2.0.0 (2026-10-22)

A rewrite. Data files and checkpoints of 1.x cannot be used: preprocess the MIDI files again and train new models.

### Changed

- **Event vocabulary and controls as in Magenta's Performance RNN**: 128 note-on, 128 note-off, 100 time shifts of 10 ms, 32 velocities (388 events); note density in notes per second over the next 3 s (8 classes) and pitch class histogram weighted by sounding time over the next 5 s. The sustain pedal is applied when MIDI files are read.
- **Default settings are Magenta's**: 3 GRU layers of 512 units, no dropout, batch 64, gradient clipping at 3. The README explains how to reduce them.
- **Installation with uv** instead of conda: `uv sync`, Python 3.12 or later, NumPy 2, current PyTorch.
- **One command, `prnn`**, with the subcommands `prepare`, `preprocess`, `train`, `generate` and `optimize`, instead of separate scripts.
- **Faster**: a training window runs in one call of the GRU (on an M1 Pro: 2.4× on the CPU, 1.4× with MPS); generation without per-step overhead; data kept compact in memory (15 bytes per event).
- Preprocessed data are `.npz` files and checkpoints contain only plain values and tensors: both load without pickle.
- The model reads the top GRU layer and has its own start event.

### Added

- Top-k and top-p sampling (`prnn generate -k`, `-p`).
- `--seed` and `--device` options; `--logging/--no-logging` for TensorBoard; `--clip-norm`.
- The best model and the last training state are saved separately; training continues an existing checkpoint with its configuration and optimizer.
- Tests (67), ruff, GitHub Actions on Linux and macOS.

### Fixed

- Transposition rotated the note density instead of the pitch histogram.
- The hyperparameter search looked for the highest validation loss.
- Validation ran with dropout, gradients and random teacher forcing.
- The loss ignored note A0 (event 0) as if it were padding.
- With few files the training set could be empty; the split was not shuffled.
- The model was not saved at the end of training; early stopping kept the last model instead of the best; errors during training were hidden.
- Model parameters given with `-M` were ignored.
- Generation always took the most likely event, so the temperature had no effect.
- Beam search added raw scores instead of log probabilities.
- Stochastic beam search failed on GPUs.
- Several crashes with NumPy 2 and PyTorch 2.6 or later.

### Removed

- Conda environment files, `setup.py`, `requirements*.txt`.
- The 2018 download scripts for MIDI websites; MAESTRO is read with `prnn prepare`.
- Velocity rescaling when writing MIDI files.

## 1.0.0 (2025-11-07)

Restructured package with command line scripts, Optuna hyperparameter search and TensorBoard logging, based on the PyTorch port by Yuankui Lee (djosix, 2018).
