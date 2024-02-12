import argparse
import config
import optuna

from logger import setup_logger
from train import *

logger = setup_logger('Hyperparameters optimization routine', file=True)


def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('-S', '--session',
                      dest='sess_path',
                      type=str,
                      default='save/optimization.sess',
                      help='The path to save the optimization session file. Using the -Sddment in the training script you can use this file to read hyperparameters from during the training session')

    parser.add_argument('-d', '--dataset',
                      dest='data_path',
                      type=str,
                      default='dataset/processed',
                      help='The dataset to be used for the optimization routine')
    
    parser.add_argument('-R', '--reset-optimizer',
                      dest='reset_optimizer',
                      action='store_true',
                      default=False,
                      help='Reset the optimizer to its initial state. Useful when starting a new type of training or training on a new dataset')
    
    parser.add_argument('-L', '--enable-logging',
                      dest='enable_logging',
                      action='store_true',
                      default=False,
                      help='Create an optimization log file')
    
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

    model, optimizer = load_session(sess_path, trial_model_config, device, learning_rate, reset_optimizer)

    validation_accuracy = train_model(model, optimizer, dataset, batch_size, num_epochs, window_size, stride_size, train_test_ratio, early_stopping_patience, event_dim, control_ratio, teacher_forcing_ratio, enable_logging, saving_interval, use_transposition, device, trial_model_config, sess_path)
    
    return 1.0 - validation_accuracy



def main(args=None):
    if args is None:
        args = get_arguments()

    sess_path = args.sess_path
    data_path = args.data_path
    reset_optimizer = args.reset_optimizer
    enable_logging = args.enable_logging

    model_config = config.model
    train_config = config.train
    device = config.device

    dataset = load_dataset(data_path)

    study = optuna.create_study()
    study.optimize(lambda trial: objective(trial, sess_path, model_config, train_config, device, dataset, reset_optimizer, enable_logging), n_trials=100)

    best_params = study.best_params
    logger.info(f'Best hyperparameters combination: {best_params}')



if __name__ == '__main__':
    main()