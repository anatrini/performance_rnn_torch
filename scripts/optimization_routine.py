#!/usr/bin/env python3
"""
Hyperparameter Optimization Script

This script uses Optuna to perform hyperparameter optimization for the
PerformanceRNN model. It systematically searches for the best combination
of model and training hyperparameters.
"""
import argparse
import optuna

from performance_rnn_torch import config
from performance_rnn_torch.utils.logger import setup_logger
from performance_rnn_torch.utils import paths
from performance_rnn_torch.training import load_session, load_dataset, train_model

logger = setup_logger('Hyperparameters optimization routine', file=True)


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Find optimal hyperparameters using Optuna',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick optimization (10 trials)
  python scripts/optimization_routine.py --n-trials 10

  # Thorough optimization (100 trials, recommended)
  python scripts/optimization_routine.py --n-trials 100

  # With TensorBoard logging for each trial
  python scripts/optimization_routine.py --n-trials 50 --enable-logging

This script will test different combinations of:
  - Model architecture (hidden_dim, gru_layers, dropout)
  - Training parameters (batch_size, learning_rate, window_size)
  - Control parameters (control_ratio, teacher_forcing_ratio)

The best parameters are saved and can be used for training.
        """
    )

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

    parser.add_argument('-R', '--reset-optimizer',
                      dest='reset_optimizer',
                      action='store_true',
                      default=False,
                      help='Reset optimizer state (use when starting fresh)')

    parser.add_argument('-L', '--enable-logging',
                      dest='enable_logging',
                      action='store_true',
                      default=False,
                      help='Enable TensorBoard logging for each trial (creates many log files)')

    return parser.parse_args()



def objective(trial, 
              sess_path, 
              model_config, 
              train_config, 
              device,
              dataset, 
              reset_optimizer, 
              enable_logging
              ):
    
    # Set model hyperparameters values to be tested using the trial object
    init_dim = trial.suggest_categorical('init_dim', [32, 64])
    hidden_dim = trial.suggest_categorical('hidden_dim', [256, 512, 1024])
    gru_layers = trial.suggest_int('gru_layers', 2, 5)
    gru_dropout = trial.suggest_float('gru_dropout', 1e-3, 1.0)

    # Set training hyperparameters values to be tested using the trial object
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    num_epochs = trial.suggest_categorical('num_epochs', [5, 10, 25, 50, 100, 150])
    window_size = trial.suggest_categorical('window_size', [10, 20, 50, 100, 200])
    stride_size = trial.suggest_categorical('stride_size', [1, 2, 5, 10, 20])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    control_ratio = trial.suggest_float('control_ratio', 1e-1, 1.0)
    teacher_forcing_ratio = trial.suggest_float('teacher_forcing_ratio', 1e-1, 1)

    # Create a copy of model to test the hyperparameters
    trial_model_config = model_config.copy()
    trial_model_config['init_dim'] = init_dim
    trial_model_config['hidden_dim'] = hidden_dim
    trial_model_config['gru_layers'] = gru_layers
    trial_model_config['gru_dropout'] = gru_dropout
    # Unchanged model's hyperparameters
    event_dim = trial_model_config['event_dim']

    # Create a copy of train to test the hyperparameters
    trial_train_config = train_config.copy()
    trial_train_config['learning_rate'] = learning_rate
    # Unchanged training's hyperparameters
    train_test_ratio = trial_train_config['train_test_ratio']
    early_stopping_patience = trial_train_config['early_stopping_patience']
    saving_interval = trial_train_config['saving_interval']
    use_transposition = trial_train_config['use_transposition']

    model, optimizer = load_session(sess_path, trial_model_config, device, learning_rate, reset_optimizer, strict_config=True)

    validation_accuracy = train_model(model, optimizer, dataset, batch_size, num_epochs, window_size, stride_size, train_test_ratio, early_stopping_patience, event_dim, control_ratio, teacher_forcing_ratio, enable_logging, saving_interval, use_transposition, device, trial_model_config, sess_path)
    
    return 1.0 - validation_accuracy



def main(args=None):
    if args is None:
        args = get_arguments()

    sess_path = args.sess_path
    data_path = args.data_path
    n_trials = args.n_trials
    reset_optimizer = args.reset_optimizer
    enable_logging = args.enable_logging

    model_config = config.model
    train_config = config.train
    device = config.device

    logger.info('='*70)
    logger.info('HYPERPARAMETER OPTIMIZATION')
    logger.info('='*70)
    logger.info(f'Dataset: {data_path}')
    logger.info(f'Session: {sess_path}')
    logger.info(f'Number of trials: {n_trials}')
    logger.info(f'Device: {device}')
    logger.info('='*70)

    logger.info('Loading dataset...')
    dataset = load_dataset(data_path)
    logger.info(f'Dataset loaded: {dataset}')

    logger.info(f'\nStarting optimization with {n_trials} trials...')
    logger.info('This may take a long time depending on your dataset size and number of trials.')

    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, sess_path, model_config, train_config, device, dataset, reset_optimizer, enable_logging),
        n_trials=n_trials
    )

    logger.info('\n' + '='*70)
    logger.info('OPTIMIZATION COMPLETE')
    logger.info('='*70)

    best_params = study.best_params
    best_value = study.best_value

    logger.info(f'\nBest validation loss: {best_value:.4f}')
    logger.info(f'Best validation accuracy: {1.0 - best_value:.4f}')
    logger.info('\nBest hyperparameters:')
    for param, value in best_params.items():
        logger.info(f'  {param:25} = {value}')

    logger.info(f'\nOptimization results saved to: {sess_path}')
    logger.info('\nTo use these hyperparameters for training, load the session file:')
    logger.info(f'  python scripts/train.py --session {sess_path}')
    logger.info('='*70)



if __name__ == '__main__':
    main()