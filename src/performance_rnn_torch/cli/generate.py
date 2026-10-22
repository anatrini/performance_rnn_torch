"""`prnn generate`: write MIDI files with a trained model."""

import torch

from performance_rnn_torch.config import GenerateConfig, default_device
from performance_rnn_torch.generation import controls_from_spec, model_name, write_midi_files
from performance_rnn_torch.training import load_model, set_seed
from performance_rnn_torch.utils import paths
from performance_rnn_torch.utils.logger import setup_logger

logger = setup_logger('generate')

HELP = 'Generate MIDI files with a trained model'
EPILOG = """
Examples:
  # 8 performances of 1000 events with the model models/train.sess
  prnn generate

  # another model, less random, only C major pitch classes, 2-4 notes per second
  prnn generate -s models/bach.sess -T 0.9 -c "1,0,1,0,1,1,0,1,0,1,0,1;2"

  # the controls of a preprocessed piece, beam search with 5 candidates
  prnn generate -c data/processed/bach/piece.npz -B 5
"""

DEFAULTS = GenerateConfig()


def add_arguments(parser):
    parser.add_argument(
        '-s',
        '--session',
        dest='sess_path',
        default=str(paths.get_model_path('train')),
        help='Checkpoint of the trained model (default: %(default)s)',
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        dest='output_dir',
        default=str(paths.output_dir),
        help='Folder of the MIDI files (default: %(default)s)',
    )
    parser.add_argument(
        '-c',
        '--control',
        dest='control',
        default=None,
        help='"PITCH_HISTOGRAM;NOTE_DENSITY", e.g. "2,0,1,1,0,1,0,1,1,0,0,1;4" '
        '(";3" gives all pitch classes the same weight). NOTE_DENSITY is a class '
        '0-7: fewer than 1, 1-2, 2-4, 4-8, 8-16, 16-32, 32-64, 64 or more notes '
        'per second. Or a preprocessed file (.npz), or a folder of them.',
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        dest='batch_size',
        type=int,
        default=DEFAULTS.batch_size,
        help='Number of performances generated together (default: %(default)s)',
    )
    parser.add_argument(
        '-l',
        '--max-length',
        dest='max_len',
        type=int,
        default=DEFAULTS.max_len,
        help='Events per performance; 0 = the length of the control file (default: %(default)s)',
    )
    parser.add_argument(
        '-T',
        '--temperature',
        dest='temperature',
        type=float,
        default=DEFAULTS.temperature,
        help='Below 1 more predictable, above 1 more surprising (default: %(default)s)',
    )
    parser.add_argument(
        '-k',
        '--top-k',
        dest='top_k',
        type=int,
        default=DEFAULTS.top_k,
        help='Sample only among the k most likely events; 0 = all (default: %(default)s)',
    )
    parser.add_argument(
        '-p',
        '--top-p',
        dest='top_p',
        type=float,
        default=DEFAULTS.top_p,
        help='Sample only among the most likely events whose probabilities add up to p; 1 = all (default: %(default)s)',
    )
    parser.add_argument(
        '-g',
        '--greedy-ratio',
        dest='greedy_ratio',
        type=float,
        default=DEFAULTS.greedy_ratio,
        help='Share of steps that take the most likely event (default: %(default)s)',
    )
    parser.add_argument(
        '-B',
        '--beam-size',
        dest='beam_size',
        type=int,
        default=DEFAULTS.beam_size,
        help='Beam search with this many candidates instead of sampling; 0 = off',
    )
    parser.add_argument(
        '-S',
        '--stochastic-beam-search',
        dest='stochastic_beam_search',
        action='store_true',
        help='Add random noise to the beam search scores',
    )
    parser.add_argument(
        '-z',
        '--init-zero',
        dest='init_zero',
        action='store_true',
        help='Start from a zero hidden state instead of a random one',
    )
    parser.add_argument('--seed', type=int, default=None, help='Random seed, for repeatable results')
    parser.add_argument('--device', default='auto', help='auto, cpu, mps or cuda (default: %(default)s)')


def main(args):
    if args.seed is not None:
        set_seed(args.seed)
    device = default_device(args.device)
    model, _, _ = load_model(args.sess_path, device)
    controls, description = controls_from_spec(args.control, args.batch_size, device)

    steps = args.max_len
    if steps == 0 and controls is not None and controls.shape[0] > 1:
        steps = controls.shape[0]
    if steps <= 0:
        raise ValueError('give a length with -l, or a control file whose length is used')

    logger.info(f'Model {args.sess_path} on {device}; controls: {description}')
    init = (
        torch.zeros(args.batch_size, model.init_dim, device=device)
        if args.init_zero
        else torch.randn(args.batch_size, model.init_dim, device=device)
    )

    if args.beam_size > 0:
        events = model.beam_search(
            init,
            steps,
            args.beam_size,
            controls=controls,
            temperature=args.temperature,
            stochastic=args.stochastic_beam_search,
            verbose=True,
        )
    else:
        events = model.generate(
            init,
            steps,
            controls=controls,
            greedy=args.greedy_ratio,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            verbose=True,
        )

    for path, notes in write_midi_files(events, args.output_dir, model_name(args.sess_path)):
        logger.info(f'===> {path} ({notes} notes)')
