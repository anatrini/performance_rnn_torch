#!/usr/bin/env python3
"""Training script for Performance RNN."""

import argparse
import numpy as np
import time
import torch
from torch import nn
from torch import optim
from tqdm import tqdm

from performance_rnn_torch import config
from performance_rnn_torch.core import Dataset, PerformanceRNN, EventSeq
from performance_rnn_torch.training import (
    EarlyStopping,
    load_session,
    load_dataset,
    save_model,
    loss_update,
    train_model,
)
from performance_rnn_torch.utils.logger import setup_logger
from performance_rnn_torch.utils import helpers, paths


logger = setup_logger('training', file=True)



#========================================================================
# Settings
#========================================================================


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-S', '--session',
                      dest='sess_path',
                      type=str,
                      default=str(paths.get_model_path('train')),
                      help='Path to the session file for training. You can read an existing optim.sess file to use the best hyperparameters combination found during the optimization session')

    parser.add_argument('-d', '--dataset',
                      dest='data_path',
                      type=str,
                      default=str(paths.processed_dir),
                      help='Path to the processed dataset for training')

    parser.add_argument('-i', '--saving-interval',
                      dest='saving_interval',
                      type=int,
                      default=config.train['saving_interval'],
                      help='Interval (in seconds) at which the model is saved during training')

    parser.add_argument('-b', '--batch-size',
                      dest='batch_size',
                      type=int,
                      default=config.train['batch_size'],
                      help='Number of samples per gradient update')
    
    parser.add_argument('-e', '--num_epochs',
                      dest='num_epochs',
                      type=int,
                      default=config.train['num_epochs'],
                      help='Number of epochs to train the model')

    parser.add_argument('-l', '--learning-rate',
                      dest='learning_rate',
                      type=float,
                      default=config.train['learning_rate'],
                      help='Learning rate for the optimizer')

    parser.add_argument('-w', '--window-size',
                      dest='window_size',
                      type=int,
                      default=config.train['window_size'],
                      help='Number of consecutive MIDI events to consider in a sequence')

    parser.add_argument('-s', '--stride-size',
                      dest='stride_size',
                      type=int,
                      default=config.train['stride_size'],
                      help='Number of steps to advance in the input data for each new sequence in the batch')
    
    parser.add_argument('-r', '--train_test_ratio',
                      dest='train_test_ratio',
                      type=float,
                      default=config.train['train_test_ratio'],
                      help='Ratio of data to use for training vs testing')
    
    parser.add_argument('-p', '--early_stopping_patience',
                      dest='early_stopping_patience',
                      type=int,
                      default=config.train['early_stopping_patience'],
                      help='Number of epochs with no improvement after which training will be stopped')

    parser.add_argument('-c', '--control-ratio',
                      dest='control_ratio',
                      type=float,
                      default=config.train['control_ratio'],
                      help='Ratio for controlling the balance between exploration and exploitation during training')

    parser.add_argument('-f', '--teacher-forcing-ratio',
                      dest='teacher_forcing_ratio',
                      type=float,
                      default=config.train['teacher_forcing_ratio'],
                      help='Ratio for controlling the use of teacher forcing during training. A higher ratio means more often using the true previous output instead of the predicted output from the last time step as input for the current time step')

    parser.add_argument('-T', '--use-transposition',
                      dest='use_transposition',
                      action='store_true',
                      default=config.train['use_transposition'],
                      help='Enable transposition of MIDI events during training')

    parser.add_argument('-M', '--model-params',
                      dest='model_params',
                      type=str,
                      default='',
                      help='Additional parameters for the model')
                      
    parser.add_argument('-R', '--reset-optimizer',
                      dest='reset_optimizer',
                      action='store_true',
                      default=False,
                      help='Reset the optimizer to its initial state before training')
                      
    parser.add_argument('-L', '--enable-logging',
                      dest='enable_logging',
                      action='store_true',
                      default=True,
                      help='Enable logging of training progress for metrics visualization')

    return parser.parse_args()



def main(args=None):
    if args is None:
        args = get_arguments()

    sess_path = args.sess_path
    data_path = args.data_path
    saving_interval = args.saving_interval

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    window_size = args.window_size
    stride_size = args.stride_size
    train_test_ratio = args.train_test_ratio
    early_stopping_patience = args.early_stopping_patience
    use_transposition = args.use_transposition
    control_ratio = args.control_ratio
    teacher_forcing_ratio = args.teacher_forcing_ratio
    reset_optimizer = args.reset_optimizer
    enable_logging = args.enable_logging

    event_dim = EventSeq.dim()
    model_config = config.model
    model_params = helpers.params2dict(args.model_params)
    model_config.update(model_params)
    device = config.device

    logger.info(f'Session path: {sess_path}')
    logger.info(f'Dataset path: {data_path}')
    logger.info(f'Saving interval: {saving_interval}')

    logger.info(f'Hyperparameters: {helpers.dict2params(model_config)}')
    logger.info(f'Learning rate: {learning_rate}')
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Window size: {window_size}')
    logger.info(f'Stride size: {stride_size}')
    logger.info(f'Control ratio: {control_ratio}')
    logger.info(f'Teacher forcing ratio: {teacher_forcing_ratio}')
    logger.info(f'Random transposition: {use_transposition}')
    logger.info(f'Reset optimizer: {reset_optimizer}')
    logger.info(f'Enabling logging: {enable_logging}')
    logger.info(f'Device: {device}')

    logger.info('Loading session')
    model, optimizer = load_session(sess_path, model_config, device, learning_rate, reset_optimizer)
    logger.info(model)
    
    logger.info('Loading dataset')
    dataset = load_dataset(data_path)
    logger.info(dataset)

    validation_accuracy = train_model(model, optimizer, dataset, batch_size, num_epochs, window_size, stride_size, train_test_ratio, early_stopping_patience, event_dim, control_ratio, teacher_forcing_ratio, enable_logging, saving_interval, use_transposition, device, model_config, sess_path)
    logger.info(f'Validation accuracy: {validation_accuracy}')



if __name__ == '__main__':
    main()