# Installation Guide

This guide explains how to install Performance RNN for different hardware configurations.

## Quick Reference

| Your Hardware | Commands |
|--------------|----------|
| **CPU only** | `conda env create -f environment-cpu.yml`<br>`conda activate py_magenta`<br>`pip install -e .` |
| **NVIDIA GPU (CUDA)** | `conda env create -f environment-cuda.yml`<br>`conda activate py_magenta`<br>`pip install -e .` |
| **Apple Silicon (M1/M2/M3)** | `conda env create -f environment-mps.yml`<br>`conda activate py_magenta`<br>`pip install -e .` |
| **Development** | `conda env create -f environment-dev.yml`<br>`conda activate py_magenta`<br>`pip install -e ".[dev]"` |

## Detailed Installation Steps

### 1. Prerequisites

- Install Miniconda or Anaconda from: https://docs.conda.io/en/latest/miniconda.html
- Git (to clone the repository)

### 2. Clone the Repository

```bash
git clone https://github.com/anatrini/performance_rnn_torch.git
cd performance_rnn_torch
```

### 3. Create the Conda Environment

Choose the appropriate environment file based on your hardware:

#### For CPU-only systems (no GPU)

```bash
conda env create -f environment-cpu.yml
conda activate py_magenta
pip install -e .
```

This installs PyTorch with CPU-only support, which works on all systems but is slower for training.

#### For systems with NVIDIA GPU

```bash
conda env create -f environment-cuda.yml
conda activate py_magenta
pip install -e .
```

This installs PyTorch with CUDA 11.8 support for GPU acceleration on NVIDIA graphics cards.

**Requirements:**
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed
- Linux or Windows

#### For Apple Silicon Macs (M1/M2/M3)

```bash
conda env create -f environment-mps.yml
conda activate py_magenta
pip install -e .
```

This installs PyTorch with MPS (Metal Performance Shaders) support for GPU acceleration on Apple Silicon.

**Requirements:**
- Mac with M1, M2, or M3 chip
- macOS 12.3 or later

### 4. Verify Installation

```bash
python -c "import performance_rnn_torch as prnn; print(f'Version: {prnn.__version__}'); print(f'Device: {prnn.config.device}')"
```

Expected output:
- `Device: cpu` for CPU installation
- `Device: cuda:0` for NVIDIA GPU
- `Device: mps` for Apple Silicon

## Development Installation

If you want to contribute to the project or modify the code:

```bash
# Use the development environment (includes testing, linting, documentation tools)
conda env create -f environment-dev.yml
conda activate py_magenta

# Install the package in development mode
pip install -e ".[dev]"
```

**Note:** The development environment uses CPU-only PyTorch by default. If you need GPU support for development:

```bash
# For NVIDIA GPU development:
conda env create -f environment-cuda.yml
conda activate py_magenta
pip install -e ".[dev]"

# For Apple Silicon development:
conda env create -f environment-mps.yml
conda activate py_magenta
pip install -e ".[dev]"
```

**Optional:** Install pre-commit hooks for automatic code formatting:
```bash
pre-commit install
```

## Alternative: pip Installation

If you prefer not to use conda, you can install with pip (not recommended):

```bash
# 1. Install PyTorch first (choose your platform):
# Visit: https://pytorch.org/get-started/locally/

# 2. Install the package
pip install -r requirements.txt
pip install -e .

# For development:
pip install -r requirements-dev.txt
```

## What Gets Installed?

### Step 1: Conda environment creation

All environment files install these dependencies:
- **Python 3.10** - Programming language
- **PyTorch 2.0+** - Deep learning framework (CPU/CUDA/MPS variant)
- **NumPy** - Numerical computing
- **pandas** - Data manipulation
- **pretty_midi** - MIDI file processing
- **tqdm** - Progress bars
- **tensorboard** - Training visualization
- **optuna** - Hyperparameter optimization

The development environment additionally includes:
- **pytest** - Testing framework
- **black, isort** - Code formatting
- **flake8** - Linting
- **mypy** - Type checking
- **pre-commit** - Git hooks for code quality
- **sphinx** - Documentation generation
- **jupyter** - Notebooks for experimentation

### Step 2: Package installation (`pip install -e .`)

This installs:
- **Performance RNN package** - Installed in editable mode so code changes take effect immediately

## Troubleshooting

### Environment already exists

If you get an error that the environment already exists:

```bash
# Remove the old environment
conda env remove -n py_magenta

# Create the new environment
conda env create -f environment-<your-system>.yml
```

### Import errors

If you get import errors after installation:

```bash
# Make sure you're in the right environment
conda activate py_magenta

# Reinstall the package
pip install -e .
```

### GPU not detected

**For NVIDIA GPU:**
```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If False, you may need to install NVIDIA drivers
```

**For Apple Silicon:**
```bash
# Check if MPS is available
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# If False, update macOS to 12.3 or later
```

## Updating the Environment

To update packages in your environment:

```bash
# Activate the environment
conda activate py_magenta

# Update all packages
conda update --all

# Or recreate the environment
conda env remove -n py_magenta
conda env create -f environment-<your-system>.yml
```

## Uninstalling

To completely remove the installation:

```bash
# Remove the conda environment
conda env remove -n py_magenta

# Optionally, delete the repository
cd ..
rm -rf performance_rnn_torch
```
