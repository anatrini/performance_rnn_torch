# Changelog

## Unreleased

Models trained with 2.0.0 still load; train new ones to get the new defaults.

### Changed

- **Training windows of 512 events** (`-w`, was 100), as Magenta's training sequences, with a stride of 64 (`-s`, was 10) and at most 50 epochs (`-e`, was 15). With 100 events the model did not learn to use a longer context, and the first events of every window, predicted without context, weighed on the loss five times more.
- **Transposition on by default**, by up to 3 semitones as in Magenta (was −6 to +5, only with `-T`); `--no-transposition` turns it off.
- **Half of the batches are trained without controls** (`-c 0.5`, was 1.0), as Magenta's optional conditioning: before, a model had never learnt to play without controls, although `prnn generate` without `-c` and `/controls 0` in `prnn serve` ask for exactly that.
- **Validation counts every event once**: its windows do not overlap, the last smaller batch is kept, and it measures the loss with and without controls in the same proportion as the training. Small validation sets now give a loss instead of none.
- The hyperparameter search no longer varies the window, the share of controls and the teacher forcing: they change what the validation loss measures, so the trials could not be compared. Its ranges were narrowed to useful values.
- `prnn preprocess` options are `--midi-root`, `--save-dir`, `--num-workers`, like the other commands; the names of the data files no longer depend on where the MIDI folder is.

### Fixed

- Continuing a training replaced the best model with the first epoch's, even when it was worse; early stopping started from zero. Both now start from the checkpoint's best validation loss (when the validation settings are the same).
- The log file of `prnn train` and `prnn optimize` lacked the validation losses and the warnings of the training loop.
- `prnn generate` failed with a traceback for `-T 0`, `-k` or `-B` above 388, `-p 0`, `-b 0` and for a control file shorter than `-l`; it now says what is wrong.
- `prnn preprocess` reported success for a missing or empty MIDI folder; a missing data folder in `prnn train` gave a traceback.
- `prnn prepare` ignored `PERFORMANCE_RNN_MAESTRO_DIR`.
- `PerformanceRNN` used dropout 0.3 by default, unlike `ModelConfig`; `from performance_rnn_torch.core import *` failed.
- Unused code from 1.x removed (`NoteSeq.trim_overlapped_notes`, `NoteSeq.copy`, `get_package_root`).

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

- **Real-time play over OSC**, `prnn serve`, with example patches for Max and for Pd and plugdata that start the server with the spawn object.
- A model trained on the Bach pieces of MAESTRO, attached to the release as `bach.sess`.
- Top-k and top-p sampling (`prnn generate -k`, `-p`).
- `--seed` and `--device` options; `--logging/--no-logging` for TensorBoard; `--clip-norm`.
- The best model and the last training state are saved separately; training continues an existing checkpoint with its configuration and optimizer.
- Tests (74), ruff, GitHub Actions on Linux and macOS.

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
