"""`prnn optimize`: search hyperparameters with Optuna."""

import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from performance_rnn_torch.config import ModelConfig, TrainConfig, default_device
from performance_rnn_torch.core import PerformanceRNN
from performance_rnn_torch.training import last_session_path, load_dataset, set_seed, train_model
from performance_rnn_torch.utils import paths
from performance_rnn_torch.utils.logger import setup_logger

HELP = 'Search hyperparameters with Optuna (needs the "optimize" extra)'
EPILOG = """
Examples:
  # Quick search (10 trials)
  prnn optimize --n-trials 10

  # Thorough search (100 trials)
  prnn optimize --n-trials 100

Each trial trains a new model with a different combination of model size,
dropout, batch size, number of epochs, window, stride, learning rate,
control ratio and teacher forcing ratio. Only the best trial's checkpoint is
kept, in models/optimization/, with its hyperparameters in best-params.json.
"""


def add_arguments(parser):
    parser.add_argument(
        '-S',
        '--session',
        dest='sess_path',
        default=str(paths.get_model_path('optimization')),
        help='The trials are saved in a folder with this name (default: %(default)s)',
    )
    parser.add_argument(
        '-d',
        '--dataset',
        dest='data_path',
        default=str(paths.processed_dir),
        help='Folder of the preprocessed files (default: %(default)s)',
    )
    parser.add_argument(
        '-n',
        '--n-trials',
        dest='n_trials',
        type=int,
        default=20,
        help='Number of trials (default: %(default)s; 100 or more for a thorough search)',
    )
    parser.add_argument(
        '--logging',
        dest='enable_logging',
        action='store_true',
        help='TensorBoard logging of every trial (many log files)',
    )
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: %(default)s)')
    parser.add_argument('--device', default='auto', help='auto, cpu, mps or cuda (default: %(default)s)')


def trial_session_path(sess_path, number: int) -> Path:
    """Checkpoint of one trial: models/optimization/trial-<n>.sess for models/optimization.sess."""
    sess_path = Path(sess_path)
    return sess_path.with_suffix('') / f'trial-{number}{sess_path.suffix}'


def objective(
    trial, sess_path, model_config: ModelConfig, train_config: TrainConfig, device: torch.device, dataset
) -> float:
    """Train a fresh model with the trial's hyperparameters; return its validation loss.

    Only the checkpoints of the best trial so far are kept.
    """
    model_config = replace(
        model_config,
        init_dim=trial.suggest_categorical('init_dim', [32, 64]),
        hidden_dim=trial.suggest_categorical('hidden_dim', [256, 512, 1024]),
        gru_layers=trial.suggest_int('gru_layers', 2, 5),
        gru_dropout=trial.suggest_float('gru_dropout', 0.0, 0.5),
    )
    cfg = replace(
        train_config,
        batch_size=trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        num_epochs=trial.suggest_categorical('num_epochs', [5, 10, 25, 50]),
        window_size=trial.suggest_categorical('window_size', [50, 100, 200]),
        stride_size=trial.suggest_categorical('stride_size', [5, 10, 20, 50]),
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        control_ratio=trial.suggest_float('control_ratio', 0.1, 1.0),
        teacher_forcing_ratio=trial.suggest_float('teacher_forcing_ratio', 0.1, 1.0),
    )

    model = PerformanceRNN(**vars(model_config)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    trial_path = trial_session_path(sess_path, getattr(trial, 'number', 0))
    val_loss = train_model(model, optimizer, dataset, cfg, model_config, trial_path, device)

    study = getattr(trial, 'study', None)
    if study is not None:
        if val_loss < study.user_attrs.get('best_loss', math.inf):
            old = study.user_attrs.get('best_path')
            study.set_user_attr('best_loss', val_loss)
            study.set_user_attr('best_path', str(trial_path))
            if old:
                _remove_session(old)
        else:
            _remove_session(trial_path)
    return val_loss


def _remove_session(path):
    for p in (Path(path), last_session_path(path)):
        p.unlink(missing_ok=True)


def main(args):
    import optuna

    logger = setup_logger('optimize', file=True)
    set_seed(args.seed)
    device = default_device(args.device)
    train_config = TrainConfig(enable_logging=args.enable_logging, seed=args.seed)

    logger.info(
        f'Dataset: {args.data_path}; {args.n_trials} trials, '
        f'checkpoints in {Path(args.sess_path).with_suffix("")}; device: {device}'
    )
    dataset = load_dataset(args.data_path)
    logger.info(dataset)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(
        lambda trial: objective(trial, args.sess_path, ModelConfig(), train_config, device, dataset),
        n_trials=args.n_trials,
    )

    if 'best_path' not in study.user_attrs:
        logger.error(
            'No trial produced a validation loss: the validation files are too short '
            'for one batch. Use more data or a smaller window.'
        )
        return 1
    best_path = Path(study.user_attrs['best_path'])
    params_path = best_path.with_name('best-params.json')
    params_path.write_text(
        json.dumps(
            {'validation_loss': study.best_value, 'trial': study.best_trial.number, 'params': study.best_params},
            indent=2,
        )
    )

    logger.info(f'Best validation loss: {study.best_value:.4f} (trial {study.best_trial.number})')
    for param, value in study.best_params.items():
        logger.info(f'  {param:25} = {value}')
    logger.info(f'Hyperparameters written to {params_path}')
    logger.info(f'Best model: {best_path}. To train it further: prnn train --session {best_path}')
