# Performance RNN in PyTorch

![score](./imgs/score.png)

A modern PyTorch implementation of Google's [Performance RNN](https://magenta.tensorflow.org/performance-rnn) for generating expressive piano performances with dynamics and timing.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Preprocessing](#preprocessing)
  - [Hyperparameter Optimization](#hyperparameter-optimization-recommended)
  - [Training](#training)
  - [Generation](#generation)
- [Configuration](#configuration)
- [Development](#development)
- [Citation](#citation)

## Overview

This repository contains a PyTorch implementation of Performance RNN, inspired by the work of Ian Simon and Sageev Oore on "Performance RNN: Generating Music with Expressive Timing and Dynamics" ([Magenta Blog, 2017](https://magenta.tensorflow.org/performance-rnn)).

This implementation was developed as part of the educational activities for the "Artificial Models for Music Creativity" class at Hochschule für Musik und Theater Hamburg (Winter Semester 2023/2024). For more resources, see the [class repository](https://github.com/anatrini/Artificial-Models-Music-Creativity).

## Features

- ✨ **Modern PyTorch 2.0+** implementation with full GPU support
- 🎹 **Expressive generation** with dynamics and timing control
- 🎯 **Flexible training** with configurable batch sizes, early stopping, and checkpointing
- 📊 **TensorBoard integration** for training visualization
- 🔧 **Hyperparameter optimization** using Optuna
- 🏃 **Easy-to-use CLI** scripts for all operations
- 📦 **Proper Python packaging** with pip installability
- 🌍 **Cross-platform** support (Linux, macOS, Windows)
- 🍎 **Apple Silicon** support with MPS acceleration

## Installation

### Requirements

- Python 3.10 or higher
- Miniconda or Anaconda ([download here](https://docs.conda.io/en/latest/miniconda.html))

### Quick Install

```bash
# 1. Clone the repository
git clone https://github.com/anatrini/performance_rnn_torch.git
cd performance_rnn_torch

# 2. Create conda environment based on your hardware:
#    Choose ONE of the following:

# For CPU-only systems:
conda env create -f environment-cpu.yml

# For NVIDIA GPU with CUDA:
conda env create -f environment-cuda.yml

# For Apple Silicon (M1/M2/M3) with MPS:
conda env create -f environment-mps.yml

# 3. Activate the environment
conda activate py_magenta

# 4. Install the package
pip install -e .
```

That's it! You're ready to use Performance RNN.

### Development Installation

If you want to contribute to the project:

```bash
# Clone the repository
git clone https://github.com/anatrini/performance_rnn_torch.git
cd performance_rnn_torch

# Create development environment (includes testing, linting, documentation tools)
conda env create -f environment-dev.yml
conda activate py_magenta

# Install the package in development mode
pip install -e ".[dev]"
```

**Note:** For GPU development, use `environment-cuda.yml` or `environment-mps.yml` instead:
```bash
conda env create -f environment-cuda.yml  # or environment-mps.yml
conda activate py_magenta
pip install -e ".[dev]"
```

### Verifying Installation

```bash
# The installation test will show the version and detected device
python -c "import performance_rnn_torch as prnn; print(f'Version: {prnn.__version__}'); print(f'Device: {prnn.config.device}')"
```

Expected output:
```
Version: 1.0.0
Device: cpu  # or 'cuda:0' for NVIDIA GPU, or 'mps' for Apple Silicon
```

## Quick Start

Complete workflow from dataset to music generation:

```bash
# Activate your conda environment
conda activate py_magenta

# 1. Get MIDI files from Maestro dataset (example: Claude Debussy)
python scripts/prepare_data.py --composer "Claude Debussy"

# 2. Preprocess MIDI files into training data
python scripts/preprocess.py --num_workers -1

# 3. Optimize hyperparameters (optional but recommended, ~1-2 hours)
python scripts/optimization_routine.py --n-trials 20

# 4. Train the model with optimized settings
python scripts/train.py --session models/optimization.sess --num-epochs 50

# 5. Generate new music
python scripts/generate.py --session models/optimization.sess --num-samples 3
```

**Skip optimization?** Use default hyperparameters instead:
```bash
python scripts/train.py --num-epochs 50
python scripts/generate.py
```

## Project Structure

```
performance_rnn_torch/              # Project root directory
│
├── performance_rnn_torch/          # Python package (source code)
│   ├── core/                       # Core models and data handling
│   │   ├── model.py               # PerformanceRNN neural network
│   │   ├── sequence.py            # MIDI-to-event conversion
│   │   └── data.py                # Dataset class
│   ├── training/                   # Training utilities
│   │   ├── trainer.py             # Training loop and helpers
│   │   └── early_stopping.py      # Early stopping callback
│   ├── utils/                      # Utility functions
│   │   ├── paths.py               # Path management
│   │   ├── logger.py              # Logging setup
│   │   └── helpers.py             # Helper functions
│   ├── config.py                   # Global configuration
│   └── __init__.py                 # Package initialization
│
├── scripts/                        # Command-line scripts
│   ├── prepare_data.py            # Extract MIDI from Maestro dataset
│   ├── preprocess.py              # Convert MIDI to training data
│   ├── train.py                   # Train the model
│   ├── generate.py                # Generate music
│   └── optimization_routine.py    # Hyperparameter tuning
│
├── data/                           # Data directory (gitignored)
│   ├── maestro-v3.0.0/            # Maestro dataset (download separately)
│   ├── midi/                      # Your MIDI files (organized by composer/folder)
│   │   ├── claude_debussy/        # Example: Debussy's pieces
│   │   ├── bach/                  # Example: Bach's pieces
│   │   └── ...
│   ├── processed/                 # Preprocessed training data (mirrors midi/ structure)
│   │   ├── claude_debussy/        # Processed Debussy files
│   │   ├── bach/                  # Processed Bach files
│   │   └── ...
│   └── scripts/                   # Dataset download scripts (.sh)
│
├── models/                         # Trained model checkpoints (gitignored)
├── output/                         # Generated MIDI files (gitignored)
├── logs/                           # Training logs (gitignored)
├── runs/                           # TensorBoard logs (gitignored)
│
├── environment-*.yml               # Conda environment files
├── setup.py                        # Package installation script
├── pyproject.toml                  # Python project metadata
├── requirements.txt                # Python dependencies (pip fallback)
└── README.md                       # This file
```

**Key points:**
- **`performance_rnn_torch/`** (inner) = Python package with source code
- **`scripts/`** = Command-line tools you run
- **`data/`** = All your data files (MIDI, preprocessed, datasets)
  - **Folder structure is preserved**: `data/midi/composer/` → `data/processed/composer/`
- **`models/`** = Trained models
- **`output/`** = Generated music
- **`.egg-info/`** = Build artifact (auto-generated, gitignored)

## Usage

### Data Preparation

**Option A: Use Maestro Dataset (Recommended)**

Download [Maestro v3.0.0](https://magenta.tensorflow.org/datasets/maestro) and extract to `data/maestro-v3.0.0/`.

```bash
# List available composers
python scripts/prepare_data.py --list

# Extract MIDI files for a composer (auto-saves to data/midi/{composer}/)
python scripts/prepare_data.py --composer "Claude Debussy"
```

**Option B: Use Your Own MIDI Files**

Place MIDI files in `data/midi/` organized by subdirectories:
```
data/midi/
├── composer_1/
│   ├── piece1.mid
└── composer_2/
    └── piece2.mid
```

### Preprocessing

Convert MIDI files to training data (reads from `data/midi/`, saves to `data/processed/`):

```bash
# Use all CPU cores (recommended)
python scripts/preprocess.py --num_workers -1

# Or use specific number of workers
python scripts/preprocess.py --num_workers 4
```

**Key Parameters:**
- `--midi_root`: Source directory (default: `data/midi/`)
- `--save_dir`: Output directory (default: `data/processed/`)
- `--num_workers`: Parallel workers (default: 1, use -1 for all cores)

### Hyperparameter Optimization (Recommended)

Find optimal hyperparameters using Optuna (tests model architecture, batch size, learning rate, etc.):

```bash
# Quick optimization (~1-2 hours)
python scripts/optimization_routine.py --n-trials 20

# Thorough optimization (~5-10 hours, best results)
python scripts/optimization_routine.py --n-trials 100
```

Best parameters are saved to `models/optimization.sess` for use in training.

**Key Parameters:**
- `-n`, `--n-trials`: Number of optimization trials (default: 20)
- `-d`, `--dataset`: Preprocessed data path (default: `data/processed/`)
- `-S`, `--session`: Where to save results (default: `models/optimization.sess`)
- `-L`, `--enable-logging`: Enable TensorBoard logging for each trial

### Training

Train using optimized hyperparameters:

```bash
# Use optimized settings
python scripts/train.py --session models/optimization.sess --num-epochs 50

# Or use default hyperparameters
python scripts/train.py --num-epochs 50
```

**Monitor training:** `tensorboard --logdir runs/` (open http://localhost:6006)

**Key Parameters:**
- `--session, -S`: Model checkpoint path (default: `models/train.sess`)
- `--dataset, -d`: Preprocessed data path (default: `data/processed/`)
- `--batch-size, -b`: Batch size (default: 64)
- `--num-epochs, -e`: Number of epochs (default: 24)
- `--learning-rate, -l`: Learning rate (default: 0.001)
- `--window-size, -w`: Sequence length (default: 200)

### Generation

Generate new piano pieces from your trained model:

```bash
# Use the optimized model
python scripts/generate.py --session models/optimization.sess --num-samples 3

# Use default model
python scripts/generate.py
```

Generated MIDI files are saved to `output/`.

**Key Parameters:**
- `--session, -S`: Trained model path (default: `models/train.sess`)
- `--output, -O`: Output directory (default: `output/`)
- `--num-samples, -n`: Number of pieces (default: 1)
- `--max-len, -l`: Sequence length (default: 1000)
- `--temperature, -t`: Randomness/creativity (default: 1.0, range: 0.1-2.0)

## Configuration

Default settings are in `performance_rnn_torch/config.py`. Key configurations:

- **Model**: `hidden_dim` (512), `gru_layers` (3), `gru_dropout` (0.3)
- **Training**: `batch_size` (64), `num_epochs` (24), `window_size` (200), `learning_rate` (0.001)
- **Generation**: `max_len` (1000), `temperature` (1.0)

**Environment Variables** (optional path customization):
```bash
export PERFORMANCE_RNN_DATA_DIR=/path/to/data
export PERFORMANCE_RNN_MODELS_DIR=/path/to/models
export PERFORMANCE_RNN_OUTPUT_DIR=/path/to/output
```

## Development

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Code quality
black performance_rnn_torch/ scripts/
flake8 performance_rnn_torch/ scripts/

# Run tests
pytest

# Build docs
cd docs/ && make html
```

## Citation

```bibtex
@misc{anatrini2024performancernn,
  author = {Anatrini, Alessandro},
  title = {Performance RNN in PyTorch},
  year = {2024},
  url = {https://github.com/anatrini/performance_rnn_torch}
}
```

Original work: [Performance RNN: Generating Music with Expressive Timing and Dynamics](https://magenta.tensorflow.org/performance-rnn) by Simon & Oore (2017)

## License & Contact

MIT License - [Alessandro Anatrini](mailto:alessandro.anatrini@hfmt-hamburg.de) - Hochschule für Musik und Theater Hamburg

For issues or contributions: [GitHub Issues](https://github.com/anatrini/performance_rnn_torch/issues)

---

