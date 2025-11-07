# Contributing to Performance RNN in PyTorch

Thank you for your interest in contributing to Performance RNN! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and professional in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/performance_rnn_torch.git
   cd performance_rnn_torch
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/anatrini/performance_rnn_torch.git
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Miniconda or Anaconda ([download here](https://docs.conda.io/en/latest/miniconda.html))

### Installation

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/performance_rnn_torch.git
   cd performance_rnn_torch
   ```

2. Create the development environment:
   ```bash
   # This installs all dependencies including dev tools
   conda env create -f environment-dev.yml

   # Activate the environment
   conda activate py_magenta

   # Install the package in development mode
   pip install -e ".[dev]"
   ```

   **Note**: If you need GPU support for development:
   ```bash
   # For NVIDIA GPU:
   conda env create -f environment-cuda.yml
   conda activate py_magenta
   pip install -e ".[dev]"

   # For Apple Silicon:
   conda env create -f environment-mps.yml
   conda activate py_magenta
   pip install -e ".[dev]"
   ```

3. (Optional) Install pre-commit hooks for automatic code formatting:
   ```bash
   pre-commit install
   ```

### Verify Installation

```bash
# Activate your conda environment
conda activate py_magenta

# Run tests
pytest

# Check code formatting
black --check performance_rnn_torch/ scripts/
isort --check-only performance_rnn_torch/ scripts/

# Run linting
flake8 performance_rnn_torch/ scripts/

# Type checking
mypy performance_rnn_torch/
```

## Project Structure

```
performance_rnn_torch/
├── performance_rnn_torch/  # Main package
│   ├── core/              # Core functionality (model, sequence, data)
│   ├── training/          # Training utilities
│   ├── utils/             # Utility modules
│   └── config.py          # Configuration
├── scripts/               # CLI scripts
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── data/                  # Data directory (not in git)
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters (enforced by Black)
- **Import order**: Managed by `isort` with Black profile
- **Docstrings**: Google style docstrings for all public functions and classes
- **Type hints**: Use type hints for function signatures

### Code Formatting

We use the following tools (automatically run by pre-commit hooks):

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Format your code before committing:

```bash
# Activate your conda environment
conda activate py_magenta

# Format code
black performance_rnn_torch/ scripts/
isort performance_rnn_torch/ scripts/

# Check linting
flake8 performance_rnn_torch/ scripts/

# Type check
mypy performance_rnn_torch/
```

### Docstring Example

```python
def train_model(model, dataset, epochs=100):
    """
    Train a Performance RNN model.

    Args:
        model: The PerformanceRNN model to train
        dataset: Dataset instance containing training data
        epochs: Number of training epochs (default: 100)

    Returns:
        dict: Training history with loss values

    Raises:
        ValueError: If dataset is empty

    Example:
        >>> model = PerformanceRNN(**config.model)
        >>> dataset = Dataset('data/processed/')
        >>> history = train_model(model, dataset, epochs=50)
    """
    pass
```

### Import Organization

Imports should be organized in the following order:

1. Standard library imports
2. Third-party imports
3. Local application imports

```python
# Standard library
import os
from pathlib import Path
from typing import Optional, List

# Third-party
import numpy as np
import torch
from tqdm import tqdm

# Local
from performance_rnn_torch import config
from performance_rnn_torch.core import PerformanceRNN
from performance_rnn_torch.utils import paths
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names

Example test:

```python
# tests/test_model.py
import pytest
import torch
from performance_rnn_torch.core import PerformanceRNN
from performance_rnn_torch import config


def test_model_creation():
    """Test that model can be created with default config."""
    model = PerformanceRNN(**config.model)
    assert isinstance(model, PerformanceRNN)
    assert model.hidden_dim == config.model['hidden_dim']


def test_model_forward_pass():
    """Test model forward pass with dummy data."""
    model = PerformanceRNN(**config.model)
    batch_size, seq_len = 4, 100

    # Create dummy inputs
    events = torch.randint(0, config.model['event_dim'], (seq_len, batch_size))
    controls = torch.randn(seq_len, batch_size, config.model['control_dim'])

    # Forward pass
    output, losses = model(events, controls)

    assert output.shape == (seq_len, batch_size, config.model['event_dim'])
    assert 'loss' in losses
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

# Run specific test
pytest tests/test_model.py::test_model_creation

# Run tests in parallel
pytest -n auto
```

### Test Coverage

- Aim for at least 80% code coverage
- Write tests for all public APIs
- Include edge cases and error conditions

## Pull Request Process

### Before Submitting

1. **Update documentation** if you've changed APIs
2. **Add tests** for new functionality
3. **Update README** if needed
4. **Activate your conda environment**:
   ```bash
   conda activate py_magenta
   ```
5. **Run the test suite**:
   ```bash
   pytest
   ```
6. **Check code quality**:
   ```bash
   black --check performance_rnn_torch/ scripts/
   flake8 performance_rnn_torch/ scripts/
   mypy performance_rnn_torch/
   ```

### Submitting a Pull Request

1. **Push your changes** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a pull request** on GitHub with:
   - Clear title describing the change
   - Detailed description of what changed and why
   - Link to any related issues
   - Screenshots/examples if applicable

3. **Wait for review** and address any feedback

### Pull Request Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Code coverage maintained

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-reviewed the code
- [ ] Commented complex code sections
- [ ] Updated documentation
- [ ] No new warnings generated
```

## Reporting Bugs

### Before Reporting

1. **Check existing issues** to avoid duplicates
2. **Update to the latest version** to see if the bug persists
3. **Gather information** about your environment

### Bug Report Template

```markdown
**Describe the Bug**
Clear description of what the bug is

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Run '...'
3. See error

**Expected Behavior**
What you expected to happen

**Screenshots/Logs**
If applicable, add screenshots or error logs

**Environment:**
 - OS: [e.g., Ubuntu 22.04]
 - Python Version: [e.g., 3.10.5]
 - PyTorch Version: [e.g., 2.0.1]
 - Package Version: [e.g., 1.0.0]

**Additional Context**
Any other context about the problem
```

## Suggesting Enhancements

### Enhancement Proposal Template

```markdown
**Is your feature request related to a problem?**
Clear description of the problem

**Describe the solution you'd like**
What you want to happen

**Describe alternatives you've considered**
Alternative solutions or features

**Additional context**
Any other context, screenshots, or examples
```

## Development Guidelines

### Adding New Features

1. **Discuss first**: Open an issue to discuss major features
2. **Keep it focused**: One feature per pull request
3. **Maintain compatibility**: Don't break existing APIs without discussion
4. **Update documentation**: Document new features thoroughly

### Code Review Checklist

- [ ] Code is readable and well-documented
- [ ] Follows project coding standards
- [ ] Tests are comprehensive
- [ ] No unnecessary dependencies added
- [ ] Performance impact is minimal
- [ ] Security implications considered

## Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Email**: alessandro.anatrini@hfmt-hamburg.de
- **Documentation**: Check the [README](README.md) and inline docs

## Recognition

Contributors will be recognized in:
- Release notes
- README acknowledgments
- Git commit history

Thank you for contributing to Performance RNN!

---

**Happy Coding! 🎵🎹**
