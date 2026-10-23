"""Command line interface: `prnn <command>`.

Each command is a module of this package with HELP, EPILOG, add_arguments(parser)
and main(args). Modules are imported only when their command runs.
"""

import argparse
import importlib
import sys

COMMANDS = {
    'prepare': 'Copy the MIDI files of one composer, or of all, from the MAESTRO dataset',
    'preprocess': 'Preprocess MIDI files into training data',
    'train': 'Train a model on preprocessed data',
    'generate': 'Generate MIDI files with a trained model',
    'optimize': 'Search hyperparameters with Optuna',
    'serve': 'Play a model in real time, controlled over OSC (from Max or Pd)',
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog='prnn', description='Performance RNN in PyTorch')
    commands = parser.add_subparsers(dest='command', metavar='command', required=True)
    for name, summary in COMMANDS.items():
        commands.add_parser(name, help=summary, add_help=False)

    # Only the chosen command's module is imported and builds its own options.
    if argv and argv[0] in COMMANDS:
        name = argv[0]
        module = importlib.import_module(f'performance_rnn_torch.cli.{name}')
        sub = argparse.ArgumentParser(
            prog=f'prnn {name}',
            description=module.HELP,
            epilog=module.EPILOG,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        module.add_arguments(sub)
        args = sub.parse_args(argv[1:])
        try:
            return module.main(args)
        except (FileNotFoundError, ValueError) as error:
            print(f'prnn {name}: {error}', file=sys.stderr)
            return 1
    parser.parse_args(argv)
