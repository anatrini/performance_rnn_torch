# Repository Structure Cleanup

## Problems Fixed

### 1. Confusing Duplicate Directories
**Problem:** Two locations for data files
- ❌ `performance_rnn_torch/dataset/` (old, inside package)
- ❌ `data/` (new, at root level)

**Solution:** Consolidated everything to `data/` at root level

### 2. Build Artifacts Visible
**Problem:** `.egg-info/` directory visible in repo
**Solution:** Added to `.gitignore` with clear comment

### 3. Unclear README Structure
**Problem:** Project structure showed old paths
**Solution:** Updated with clean, clear structure diagram

## Changes Made

### Files Moved
```
FROM: performance_rnn_torch/dataset/maestro-v3.0.0/
TO:   data/maestro-v3.0.0/

FROM: performance_rnn_torch/dataset/scripts/*.sh
TO:   data/scripts/*.sh
```

### Directories Removed
```
✗ performance_rnn_torch/dataset/          # Completely removed
✗ performance_rnn_torch/dataset/midi/     # Old MIDI location
✗ performance_rnn_torch/dataset/processed/ # Old processed data
```

### Files Updated
1. **`.gitignore`**
   - Added clear comment for `.egg-info/`
   - Removed old dataset directory references
   - Simplified to only reference `data/` at root

2. **`README.md`**
   - New clean structure diagram
   - Clear explanation of directory purposes
   - Added note about `.egg-info` being auto-generated

3. **`scripts/prepare_data.py`**
   - Removed search for old `performance_rnn_torch/dataset/` location
   - Updated error messages to show only `data/maestro-v3.0.0/`

## New Clean Structure

```
performance_rnn_torch/              # Project root
│
├── performance_rnn_torch/          # Python package (source code only)
│   ├── core/                       # Models and data handling
│   ├── training/                   # Training utilities
│   ├── utils/                      # Helper functions
│   └── config.py                   # Configuration
│
├── scripts/                        # CLI tools you run
│   ├── prepare_data.py
│   ├── preprocess.py
│   ├── train.py
│   └── generate.py
│
├── data/                           # ALL data files here
│   ├── maestro-v3.0.0/            # Dataset
│   ├── midi/                      # Your MIDI files
│   ├── processed/                 # Preprocessed data
│   └── scripts/                   # Download scripts
│
├── models/                         # Trained models
├── output/                         # Generated MIDI
├── logs/                           # Logs
└── runs/                           # TensorBoard
```

## What Each Directory Is For

| Directory | Purpose | Gitignored? |
|-----------|---------|-------------|
| `performance_rnn_torch/` | Python package source code | No |
| `scripts/` | Command-line tools | No |
| `data/` | All data files | Contents ignored |
| `models/` | Trained model checkpoints | Contents ignored |
| `output/` | Generated MIDI files | Contents ignored |
| `logs/` | Training logs | Contents ignored |
| `runs/` | TensorBoard logs | Contents ignored |
| `*.egg-info/` | Build artifact (from `pip install`) | Yes |

## Why `.egg-info` Exists

The `.egg-info` directory is automatically created when you run:
```bash
pip install -e .
```

It contains metadata about the installed package. It's:
- **Auto-generated** - Created by pip
- **Gitignored** - Not committed to repository
- **Safe to delete** - Will be recreated on next `pip install -e .`
- **Normal** - All Python packages have this when installed in editable mode

## Benefits of New Structure

1. **Clear separation**: Source code vs. data vs. generated files
2. **Single data location**: Everything in `data/` at root level
3. **No confusion**: Package name vs. project root is now clear
4. **Clean git status**: Build artifacts properly ignored
5. **Easier to understand**: New users can immediately see what's what

## Migration

If you have old data in `performance_rnn_torch/dataset/`, it has been moved to `data/`.

**Nothing for you to do** - all paths automatically updated!

## Verification

Check the clean structure:
```bash
ls -la
# Should see:
# - performance_rnn_torch/  (package)
# - scripts/                (CLI tools)
# - data/                   (all data)
# - models/                 (trained models)
# - output/                 (generated MIDI)
# - environment-*.yml       (conda environments)
# - *.egg-info/             (build artifact - gitignored)
```

All scripts automatically use the correct paths:
```bash
python scripts/prepare_data.py --list  # Finds data/maestro-v3.0.0/
python scripts/preprocess.py           # Reads data/midi/, writes data/processed/
python scripts/train.py                # Reads data/processed/, writes models/
python scripts/generate.py             # Reads models/, writes output/
```

---

**Summary:** Repository structure is now clean, logical, and follows Python best practices!
