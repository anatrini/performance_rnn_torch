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
  - [Training](#training)
  - [Generation](#generation)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
- [Configuration](#configuration)
- [Environment Variables](#environment-variables)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

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

# 2. Preprocess MIDI files into training data (use all CPU cores)
python scripts/preprocess.py --num_workers -1

# 3. Find optimal hyperparameters (recommended for best results)
python scripts/optimization_routine.py --n-trials 20

# 4. Train the model with optimized settings
python scripts/train.py --session models/optimization.sess

# 5. Generate new music
python scripts/generate.py --session models/optimization.sess
```

**That's it!** The scripts automatically handle file organization and paths.

### Quick Start (Skip Optimization)

If you want to train quickly without optimization:

```bash
conda activate py_magenta
python scripts/prepare_data.py --composer "Claude Debussy"
python scripts/preprocess.py --num_workers -1
python scripts/train.py  # Uses default hyperparameters
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

### Complete Example Workflow

Here's a complete example from start to finish with hyperparameter optimization:

```bash
# Activate environment
conda activate py_magenta

# 1. List available composers in Maestro dataset
python scripts/prepare_data.py --list

# 2. Extract MIDI files for Claude Debussy
#    (automatically saves to data/midi/claude_debussy/)
python scripts/prepare_data.py --composer "Claude Debussy"

# 3. Preprocess MIDI files into training data
#    (reads from data/midi/, saves to data/processed/)
#    Use -1 to utilize all CPU cores for faster processing
python scripts/preprocess.py --num_workers -1

# 4. Find optimal hyperparameters (20 trials for quick test, 100+ for best results)
#    (tests different model architectures and training settings)
#    (saves best config to models/optimization.sess)
python scripts/optimization_routine.py --n-trials 20

# 5. Train the model with optimized hyperparameters
#    (reads from data/processed/, uses models/optimization.sess for best settings)
python scripts/train.py --session models/optimization.sess --num-epochs 50

# 6. Generate 3 new piano pieces using the optimized model
#    (reads optimized model, saves MIDI to output/)
python scripts/generate.py --session models/optimization.sess --num-samples 3
```

**That's the complete workflow!** All paths are automatic - no manual file organization needed.

---

### Detailed Steps

### Step 1: Prepare MIDI Data

You have three options for getting MIDI data:

#### Option A: Use Maestro Dataset (Recommended)

The easiest way is to use the Maestro dataset. First, download it:

```bash
# Download Maestro v3.0.0 from:
# https://magenta.tensorflow.org/datasets/maestro

# Extract to data/maestro-v3.0.0/
```

Then list available composers:

```bash
python scripts/prepare_data.py --list
```

Extract MIDI files for your chosen composer (files are automatically saved to `data/midi/{composer}/`):

```bash
# Example: Claude Debussy
python scripts/prepare_data.py --composer "Claude Debussy"

# Example: Johann Sebastian Bach
python scripts/prepare_data.py --composer "Johann Sebastian Bach"

# Example: Franz Schubert
python scripts/prepare_data.py --composer "Franz Schubert"
```

**That's it!** The script automatically:
- Finds the Maestro dataset
- Extracts all MIDI files for the composer
- Saves them to `data/midi/{composer_name}/`
- No manual file organization needed!

#### Option B: Use Your Own MIDI Files

Simply place your MIDI files in `data/midi/` organized by subdirectories:

```
data/midi/
├── composer_1/
│   ├── piece1.mid
│   ├── piece2.mid
│   └── piece3.mid
├── composer_2/
│   ├── piece1.mid
│   └── piece2.mid
```

The preprocessing script will:
- Automatically find and process all `.mid` and `.midi` files in subdirectories
- **Preserve the folder structure** in `data/processed/`:
  ```
  data/processed/
  ├── composer_1/
  │   ├── piece1-<hash>.data
  │   ├── piece2-<hash>.data
  │   └── piece3-<hash>.data
  ├── composer_2/
  │   ├── piece1-<hash>.data
  │   └── piece2-<hash>.data
  ```

### Step 2: Preprocess MIDI Files

Convert MIDI files to training data (uses defaults - reads from `data/midi/`, saves to `data/processed/`):

```bash
# Use default (1 worker)
python scripts/preprocess.py

# Use all CPU cores for faster processing (recommended)
python scripts/preprocess.py --num_workers -1

# Use specific number of workers
python scripts/preprocess.py --num_workers 4
```

**What it does:**
The script automatically:
- Finds all MIDI files in the source directory
- Converts each MIDI file to training data (`.data` files)
- **Preserves the composer folder name** in the output

**Behavior:**
```bash
# Process all composers (default)
python scripts/preprocess.py
# Finds: data/midi/*/piece.mid
# Saves: data/processed/*/piece-<hash>.data

# Process specific composer
python scripts/preprocess.py --midi_root data/midi/johann_sebastian_bach
# Finds: data/midi/johann_sebastian_bach/piece.mid
# Saves: data/processed/johann_sebastian_bach/piece-<hash>.data
```

**Example:**
```
Input:  data/midi/claude_debussy/piece1.mid
Output: data/processed/claude_debussy/piece1-<hash>.data

Input:  data/midi/bach/piece2.mid
Output: data/processed/bach/piece2-<hash>.data
```

**Parameters:**
- `--midi_root`: Source directory (default: `data/midi/`, processes all subfolders)
- `--save_dir`: Output directory (default: `data/processed/`)
- `--num_workers`: Number of parallel workers (default: 1, use -1 for all CPU cores)

### Step 3: Optimize Hyperparameters (Recommended)

**Before training, find the best hyperparameters for your dataset using Optuna:**

```bash
# Quick optimization (20 trials, ~1-2 hours)
python scripts/optimization_routine.py --n-trials 20

# Thorough optimization (100 trials, ~5-10 hours, best results)
python scripts/optimization_routine.py --n-trials 100

# With all options specified
python scripts/optimization_routine.py \
  --n-trials 50 \
  --dataset data/processed/ \
  --session models/my_optimization.sess \
  --enable-logging
```

**What it does:**
The optimization script automatically tests different combinations of:
- **Model architecture**: hidden dimensions, GRU layers, dropout rates
- **Training settings**: batch size, learning rate, window size, stride size
- **Control parameters**: control ratio, teacher forcing ratio

**Results:**
- Best parameters are saved to `models/optimization.sess`
- You can then use this for training with optimal settings

**Parameters:**
- `-n`, `--n-trials`: Number of optimization trials (default: 20, recommended: 100+)
- `-d`, `--dataset`: Preprocessed data path (default: `data/processed/`)
- `-S`, `--session`: Where to save results (default: `models/optimization.sess`)
- `-R`, `--reset-optimizer`: Reset optimizer state (use when starting fresh)
- `-L`, `--enable-logging`: Enable TensorBoard logging for each trial (creates many log files)

**Skip optimization?** You can skip this step and use default hyperparameters, but optimization usually gives much better results.

### Step 4: Train the Model

Train using the optimized hyperparameters:

```bash
# Use optimized hyperparameters (recommended)
python scripts/train.py --session models/optimization.sess --num-epochs 50

# Or use default hyperparameters
python scripts/train.py --num-epochs 50
```

**Common options:**
```bash
# Train for more epochs
python scripts/train.py --session models/optimization.sess --num-epochs 100

# Use larger batch size (if you have GPU memory)
python scripts/train.py --batch-size 128

# Train from scratch with custom session name
python scripts/train.py --session models/my_model.sess --num-epochs 50
```

**Monitor training in real-time:**
```bash
# In another terminal
tensorboard --logdir runs/
# Then open http://localhost:6006 in your browser
```

**All parameters:**
- `--session, -S`: Model checkpoint path (default: `models/train.sess`)
- `--dataset, -d`: Preprocessed data path (default: `data/processed/`)
- `--batch-size, -b`: Batch size (default: 64)
- `--num-epochs, -e`: Number of epochs (default: 24)
- `--learning-rate, -l`: Learning rate (default: 0.001)
- `--window-size, -w`: Sequence length (default: 200)
- `--enable-logging`: Enable TensorBoard (default: True)

### Step 5: Generate Music

Generate new piano pieces from your trained model:

```bash
# Use the optimized model
python scripts/generate.py --session models/optimization.sess

# Or use default model
python scripts/generate.py
```

This generates 1 MIDI file and saves to `output/`.

**Common options:**
```bash
# Generate 5 pieces from optimized model
python scripts/generate.py --session models/optimization.sess --num-samples 5

# Generate longer pieces
python scripts/generate.py --session models/optimization.sess --max-len 2000

# More creative/random output (higher temperature)
python scripts/generate.py --session models/optimization.sess --temperature 1.5

# More conservative output (lower temperature)
python scripts/generate.py --session models/optimization.sess --temperature 0.8

# Use custom model
python scripts/generate.py --session models/my_model.sess
```

Generated MIDI files will be saved to `output/` and can be played with any MIDI player or imported into music software.

**All parameters:**
- `--session, -S`: Trained model path (default: `models/train.sess`)
- `--output, -O`: Output directory (default: `output/`)
- `--num-samples, -n`: Number of pieces (default: 1)
- `--max-len, -l`: Sequence length (default: 1000)
- `--temperature, -t`: Randomness (default: 1.0, range: 0.1-2.0)

---

## Configuration

### Default Configuration

Edit `performance_rnn_torch/config.py` to change default settings:

```python
# Model configuration (lazily loaded)
model = {
    'init_dim': 32,
    'event_dim': EventSeq.dim(),  # Automatically computed
    'control_dim': ControlSeq.dim(),  # Automatically computed
    'hidden_dim': 512,
    'gru_layers': 3,
    'gru_dropout': 0.3,
}

# Training configuration
train = {
    'batch_size': 64,
    'num_epochs': 24,
    'window_size': 200,
    'stride_size': 10,
    'learning_rate': 0.001,
    'train_test_ratio': 0.3,
    'early_stopping_patience': 5,
    'use_transposition': False,
    'control_ratio': 1.0,
    'teacher_forcing_ratio': 1.0,
    'saving_interval': 180,  # seconds
}

# Generation configuration
generate = {
    'batch_size': 8,
    'max_len': 1000,
    'greedy_ratio': 1.0,
    'beam_size': 0,
    'temperature': 1.0,
    'stochastic_beam_search': False,
    'init_zero': False,
}
```

### Programmatic Usage

```python
from performance_rnn_torch import PerformanceRNN, Dataset, config
from performance_rnn_torch.utils import paths
import torch

# Load dataset
dataset = Dataset(str(paths.processed_dir), verbose=True)
train_data, test_data = dataset.train_test_split()

# Create model
model = PerformanceRNN(**config.model).to(config.device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=config.train['learning_rate'])
for epoch in range(config.train['num_epochs']):
    for events, controls in train_data.batches(
        config.train['batch_size'],
        config.train['window_size'],
        config.train['stride_size']
    ):
        # Training step
        output, losses = model(events, controls)
        loss = losses['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save checkpoint
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
    }, paths.get_model_path('checkpoint'))
```

## Environment Variables

Customize paths using environment variables:

```bash
# Data directories
export PERFORMANCE_RNN_DATA_DIR=/path/to/data
export PERFORMANCE_RNN_MIDI_DIR=/path/to/midi
export PERFORMANCE_RNN_PROCESSED_DIR=/path/to/processed

# Output directories
export PERFORMANCE_RNN_MODELS_DIR=/path/to/models
export PERFORMANCE_RNN_OUTPUT_DIR=/path/to/output
export PERFORMANCE_RNN_LOGS_DIR=/path/to/logs
export PERFORMANCE_RNN_RUNS_DIR=/path/to/runs
```

## Development

### Setting Up Development Environment

```bash
# Activate your conda environment
conda activate py_magenta

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality

```bash
# Activate your conda environment
conda activate py_magenta

# Format code
black performance_rnn_torch/ scripts/
isort performance_rnn_torch/ scripts/

# Lint code
flake8 performance_rnn_torch/ scripts/

# Type checking
mypy performance_rnn_torch/
```

### Running Tests

```bash
# Activate your conda environment
conda activate py_magenta

# Run all tests
pytest

# Run with coverage
pytest --cov=performance_rnn_torch --cov-report=html

# Run specific test file
pytest tests/test_model.py
```

### Building Documentation

```bash
# Activate your conda environment
conda activate py_magenta

cd docs/
make html
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{anatrini2024performancernn,
  author = {Anatrini, Alessandro},
  title = {Performance RNN in PyTorch},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/anatrini/performance_rnn_torch}
}
```

Original Performance RNN paper:

```bibtex
@inproceedings{simon2017performance,
  title={Performance RNN: Generating Music with Expressive Timing and Dynamics},
  author={Simon, Ian and Oore, Sageev},
  year={2017},
  organization={Magenta}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original Performance RNN implementation by Google Magenta
- Hochschule für Musik und Theater Hamburg for supporting this educational project
- All contributors and students from the "Artificial Models for Music Creativity" class

## Contact

For questions, issues, or contributions, please:
- Open an issue on [GitHub](https://github.com/anatrini/performance_rnn_torch/issues)
- Contact: alessandro.anatrini@hfmt-hamburg.de

---

