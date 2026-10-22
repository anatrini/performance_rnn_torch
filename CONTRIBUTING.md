# Contributing

Bug reports, questions and pull requests are welcome on [GitHub](https://github.com/anatrini/performance_rnn_torch/issues).

## Setup

```bash
git clone https://github.com/anatrini/performance_rnn_torch.git
cd performance_rnn_torch
uv sync --all-extras
```

## Before a pull request

```bash
uv run ruff format      # formatting
uv run ruff check       # lint
uv run pytest           # tests
```

GitHub Actions runs the same checks on Linux and macOS. New behaviour comes with a test; a bug fix with a test that fails without it.

## Layout

```
src/performance_rnn_torch/
  core/sequence.py     events, controls, MIDI reading and writing, data files
  core/model.py        the network, generation, beam search
  core/data.py         the dataset and its batches
  training/            training loop, checkpoints, early stopping
  generation.py        controls for generation, writing MIDI files
  server.py            real-time play over OSC (prnn serve)
  config.py            default settings
  cli/                 the prnn command, one module per subcommand
  utils/               working folders, logging, file helpers
tests/
examples/              patches for Max and for Pd and plugdata
```

## Conventions

- The event vocabulary and the controls follow Magenta's code; a change to them changes `FORMAT_VERSION` in `core/sequence.py`, so that old data files and checkpoints are refused.
- Checkpoints contain only plain values and tensors, so they load with `torch.load(weights_only=True)`.
- Command line options have a default from `config.py` and a help text with that default.
