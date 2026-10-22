"""`prnn train`: train a Performance RNN model."""

import argparse

from performance_rnn_torch.config import ModelConfig, TrainConfig, default_device
from performance_rnn_torch.training import load_dataset, load_session, set_seed, train_model
from performance_rnn_torch.utils import paths
from performance_rnn_torch.utils.logger import setup_logger

HELP = 'Train a model on preprocessed data'
EPILOG = """
Examples:
  # train models/train.sess on data/processed/, or continue it if it exists
  prnn train

  # a larger model, saved as models/bach.sess, with random transposition
  prnn train -S models/bach.sess -M "hidden_dim=512,gru_layers=3" -T
"""

DEFAULTS = TrainConfig()


def add_arguments(parser):
    parser.add_argument('-S', '--session', dest='sess_path', default=str(paths.get_model_path('train')),
                        help='Checkpoint to write, or to continue if it exists (default: %(default)s)')
    parser.add_argument('-d', '--dataset', dest='data_path', default=str(paths.processed_dir),
                        help='Folder of the preprocessed files (default: %(default)s)')
    parser.add_argument('-M', '--model-params', dest='model_params', default='',
                        help='Model settings, e.g. "hidden_dim=512,gru_layers=3"; defaults: '
                             + ', '.join(f'{k}={v}' for k, v in vars(ModelConfig()).items()
                                         if k not in ('event_dim', 'control_dim')))
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=DEFAULTS.batch_size,
                        help='Windows per gradient update (default: %(default)s)')
    parser.add_argument('-e', '--num-epochs', dest='num_epochs', type=int, default=DEFAULTS.num_epochs,
                        help='Passes over the training data (default: %(default)s)')
    parser.add_argument('-l', '--learning-rate', dest='learning_rate', type=float, default=DEFAULTS.learning_rate,
                        help='Learning rate of the optimizer (default: %(default)s)')
    parser.add_argument('-w', '--window-size', dest='window_size', type=int, default=DEFAULTS.window_size,
                        help='Events per training window (default: %(default)s)')
    parser.add_argument('-s', '--stride-size', dest='stride_size', type=int, default=DEFAULTS.stride_size,
                        help='Events between the starts of two windows (default: %(default)s)')
    parser.add_argument('-r', '--train-test-ratio', dest='train_test_ratio', type=float,
                        default=DEFAULTS.train_test_ratio,
                        help='Share of the files used for validation (default: %(default)s)')
    parser.add_argument('-p', '--early-stopping-patience', dest='early_stopping_patience', type=int,
                        default=DEFAULTS.early_stopping_patience,
                        help='Stop after this many epochs without improvement; 0 = never (default: %(default)s)')
    parser.add_argument('-c', '--control-ratio', dest='control_ratio', type=float, default=DEFAULTS.control_ratio,
                        help='Share of batches trained with controls; the others learn to play '
                             'without them (default: %(default)s)')
    parser.add_argument('-f', '--teacher-forcing-ratio', dest='teacher_forcing_ratio', type=float,
                        default=DEFAULTS.teacher_forcing_ratio,
                        help="Share of steps that receive the true previous event as input. 1 is fastest: "
                             "the whole window runs at once. Below 1 the other steps receive the model's "
                             "own prediction (scheduled sampling), computed step by step and several "
                             "times slower (default: %(default)s)")
    parser.add_argument('--clip-norm', dest='clip_norm', type=float, default=DEFAULTS.clip_norm,
                        help='Largest gradient norm; larger gradients are scaled down (default: %(default)s)')
    parser.add_argument('-T', '--use-transposition', dest='use_transposition', action='store_true',
                        help='Transpose every batch by a random interval of -6 to +5 semitones')
    parser.add_argument('-i', '--saving-interval', dest='saving_interval', type=float,
                        default=DEFAULTS.saving_interval,
                        help='Seconds between saves of the last state (default: %(default)s)')
    parser.add_argument('-R', '--reset-optimizer', dest='reset_optimizer', action='store_true',
                        help='When continuing a checkpoint, start the optimizer again')
    parser.add_argument('--logging', dest='enable_logging', action=argparse.BooleanOptionalAction,
                        default=DEFAULTS.enable_logging,
                        help='TensorBoard logging of the training progress (--no-logging to turn it off)')
    parser.add_argument('--seed', type=int, default=DEFAULTS.seed, help='Random seed (default: %(default)s)')
    parser.add_argument('--device', default='auto', help='auto, cpu, mps or cuda (default: %(default)s)')


def main(args):
    logger = setup_logger('training', file=True)
    cfg = TrainConfig(**{name: getattr(args, name) for name in vars(DEFAULTS)})
    set_seed(cfg.seed)
    device = default_device(args.device)

    model, optimizer, model_config = load_session(args.sess_path, ModelConfig().with_params(args.model_params),
                                                  device, cfg.learning_rate, args.reset_optimizer)
    logger.info(f'Session: {args.sess_path}, device: {device}')
    logger.info(f'{model_config}')
    logger.info(f'{cfg}')

    dataset = load_dataset(args.data_path)
    logger.info(dataset)

    best = train_model(model, optimizer, dataset, cfg, model_config, args.sess_path, device)
    logger.info(f'Best validation loss: {best:.4f}, model saved to {args.sess_path}')
