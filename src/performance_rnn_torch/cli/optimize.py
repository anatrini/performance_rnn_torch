"""
`prnn optimize`: search hyperparameters with Optuna.

This command uses Optuna to perform hyperparameter optimization for the
PerformanceRNN model. It systematically searches for the best combination
of model and training hyperparameters.
"""

import json
import math
from pathlib import Path

from torch import optim

from performance_rnn_torch import config
from performance_rnn_torch.core import PerformanceRNN
from performance_rnn_torch.utils.logger import setup_logger
from performance_rnn_torch.utils import paths
from performance_rnn_torch.training import load_dataset, last_session_path, train_model

logger = setup_logger('Hyperparameters optimization routine')


HELP = 'Search hyperparameters with Optuna (needs the "optimize" extra)'
EPILOG = """
Examples:
  # Quick optimization (10 trials)
  prnn optimize --n-trials 10

  # Thorough optimization (100 trials, recommended)
  prnn optimize --n-trials 100

  # With TensorBoard logging for each trial
  prnn optimize --n-trials 50 --enable-logging

This command tests different combinations of:
  - Model architecture (hidden_dim, gru_layers, dropout)
  - Training parameters (batch_size, learning_rate, window_size)
  - Control parameters (control_ratio, teacher_forcing_ratio)

Each trial trains a new model; only the best trial's checkpoint is kept,
with its hyperparameters in best-params.json next to it.
"""


def add_arguments(parser):
    parser.add_argument('-S', '--session',
                      dest='sess_path',
                      type=str,
                      default=str(paths.get_model_path('optimization')),
                      help=f'Path to save optimization results (default: {paths.get_model_path("optimization")})')

    parser.add_argument('-d', '--dataset',
                      dest='data_path',
                      type=str,
                      default=str(paths.processed_dir),
                      help=f'Preprocessed dataset path (default: {paths.processed_dir})')

    parser.add_argument('-n', '--n-trials',
                      dest='n_trials',
                      type=int,
                      default=20,
                      help='Number of optimization trials (default: 20, use 100+ for thorough search)')

    parser.add_argument('-L', '--enable-logging',
                      dest='enable_logging',
                      action='store_true',
                      default=False,
                      help='Enable TensorBoard logging for each trial (creates many log files)')



def trial_session_path(sess_path, number):
    """Checkpoint of one trial: models/optimization/trial-<n>.sess for models/optimization.sess."""
    sess_path = Path(sess_path)
    return sess_path.with_suffix('') / f'trial-{number}{sess_path.suffix}'


def objective(trial, sess_path, model_config, train_config, device, dataset, enable_logging):
    """Train a fresh model with the trial's hyperparameters; return its validation loss.

    Only the checkpoints of the best trial so far are kept.
    """
    # Model hyperparameters
    init_dim = trial.suggest_categorical('init_dim', [32, 64])
    hidden_dim = trial.suggest_categorical('hidden_dim', [256, 512, 1024])
    gru_layers = trial.suggest_int('gru_layers', 2, 5)
    gru_dropout = trial.suggest_float('gru_dropout', 0.0, 0.5)

    # Training hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    num_epochs = trial.suggest_categorical('num_epochs', [5, 10, 25, 50])
    window_size = trial.suggest_categorical('window_size', [50, 100, 200])
    stride_size = trial.suggest_categorical('stride_size', [5, 10, 20, 50])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    control_ratio = trial.suggest_float('control_ratio', 0.1, 1.0)
    teacher_forcing_ratio = trial.suggest_float('teacher_forcing_ratio', 0.1, 1.0)

    trial_model_config = dict(model_config)
    trial_model_config.update(init_dim=init_dim, hidden_dim=hidden_dim,
                              gru_layers=gru_layers, gru_dropout=gru_dropout)

    model = PerformanceRNN(**trial_model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trial_path = trial_session_path(sess_path, getattr(trial, 'number', 0))

    val_loss = train_model(model, optimizer, dataset, batch_size, num_epochs, window_size, stride_size,
                           train_config['train_test_ratio'], train_config['early_stopping_patience'],
                           trial_model_config['event_dim'], control_ratio, teacher_forcing_ratio,
                           enable_logging, train_config['saving_interval'],
                           train_config['use_transposition'], device, trial_model_config, trial_path)

    study = getattr(trial, 'study', None)
    if study is not None:
        best = study.user_attrs.get('best_loss', math.inf)
        if val_loss < best:
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

    global logger
    logger = setup_logger('Hyperparameters optimization routine', file=True)

    sess_path = args.sess_path
    data_path = args.data_path
    n_trials = args.n_trials
    enable_logging = args.enable_logging

    model_config = dict(config.model)
    train_config = config.train
    device = config.device

    logger.info(f'Dataset: {data_path}')
    logger.info(f'Trials: {n_trials}, checkpoints in {Path(sess_path).with_suffix("")}')
    logger.info(f'Device: {device}')

    dataset = load_dataset(data_path)
    logger.info(f'Dataset loaded: {dataset}')

    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, sess_path, model_config, train_config, device, dataset, enable_logging),
        n_trials=n_trials
    )

    if 'best_path' not in study.user_attrs:
        logger.error('No trial produced a validation loss: the validation files are too short '
                     'for one batch. Use more data or a smaller window.')
        return 1
    best_path = Path(study.user_attrs['best_path'])
    params_path = best_path.with_name('best-params.json')
    params_path.write_text(json.dumps({'validation_loss': study.best_value,
                                       'trial': study.best_trial.number,
                                       'params': study.best_params}, indent=2))

    logger.info(f'Best validation loss: {study.best_value:.4f} (trial {study.best_trial.number})')
    for param, value in study.best_params.items():
        logger.info(f'  {param:25} = {value}')
    logger.info(f'Hyperparameters written to {params_path}')
    logger.info(f'Best model: {best_path}. To train it further:')
    logger.info(f'  prnn train --session {best_path}')
