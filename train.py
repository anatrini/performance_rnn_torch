import config
import numpy as np
import optparse
import time
import torch
from torch import nn
from torch import optim
import utils

from data import Dataset
from early_stopping import EarlyStopping
from logger import setup_logger
from model import PerformanceRNN
from sequence import EventSeq
from tqdm import tqdm


logger = setup_logger('Training logger', file=True)



#========================================================================
##### Settings
#========================================================================


def get_options():
    parser = optparse.OptionParser()

    parser.add_option('-S', '--session',
                      dest='sess_path',
                      type='string',
                      default='save/train.sess')

    parser.add_option('-d', '--dataset',
                      dest='data_path',
                      type='string',
                      default='dataset/processed/')

    parser.add_option('-i', '--saving-interval',
                      dest='saving_interval',
                      type='float',
                      default=60.)

    parser.add_option('-b', '--batch-size',
                      dest='batch_size',
                      type='int',
                      default=config.train['batch_size'])
    
    parser.add_option('-e', '--num_epochs',
                      dest='num_epochs',
                      type='int',
                      default=config.train['num_epochs'])

    parser.add_option('-l', '--learning-rate',
                      dest='learning_rate',
                      type='float',
                      default=config.train['learning_rate'])

    parser.add_option('-w', '--window-size',
                      dest='window_size',
                      type='int',
                      default=config.train['window_size'])

    parser.add_option('-s', '--stride-size',
                      dest='stride_size',
                      type='int',
                      default=config.train['stride_size'])
    
    parser.add_option('-r', '--train_test_ratio',
                      dest='train_test_ratio',
                      type='float',
                      default=config.train['train_test_ratio'])
    
    parser.add_option('-p', '--early_stopping_patience',
                      dest='early_stopping_patience',
                      type='int',
                      default=config.train['early_stopping_patience'])

    parser.add_option('-c', '--control-ratio',
                      dest='control_ratio',
                      type='float',
                      default=config.train['control_ratio'])

    parser.add_option('-f', '--teacher-forcing-ratio',
                      dest='teacher_forcing_ratio',
                      type='float',
                      default=config.train['teacher_forcing_ratio'])

    parser.add_option('-T', '--use-transposition',
                      dest='use_transposition',
                      action='store_true',
                      default=config.train['use_transposition'])

    parser.add_option('-M', '--model-params',
                      dest='model_params',
                      type='string',
                      default='')
                      
    parser.add_option('-R', '--reset-optimizer',
                      dest='reset_optimizer',
                      action='store_true',
                      default=False)
                      
    parser.add_option('-L', '--enable-logging',
                      dest='enable_logging',
                      action='store_true',
                      default=False)

    return parser.parse_args()[0]



#========================================================================
##### Load session and dataset
#========================================================================

# Function to load the model and optimizer states if a session exists
def load_session(sess_path,
                model_config,
                device,
                learning_rate,
                reset_optimizer):

    try:
        sess = torch.load(sess_path)

        # If the model configuration in the session is different from the current configuration, use the session's configuration
        if 'model_config' in sess and sess['model_config'] != model_config:
            model_config = sess['model_config']
            logger.info(f'Use session config instead: {utils.dict2params(model_config)}')
        model_state = sess['model_state']
        optimizer_state = sess['model_optimizer_state']
        logger.info(f'Session is loaded from {sess_path}')
        sess_loaded = True
    except FileNotFoundError:
        logger.info('New session')
        sess_loaded = False

    # Initialize the model and optimizer
    model = PerformanceRNN(**model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # If a session was loaded, load the states of the model and optimizer
    if sess_loaded:
        model.load_state_dict(model_state)
        if not reset_optimizer:
            optimizer.load_state_dict(optimizer_state)
    return model, optimizer


# Function to load the dataset
def load_dataset(data_path):
    #global data_path
    dataset = Dataset(data_path, verbose=True)
    dataset_size = len(dataset.samples)
    assert dataset_size > 0
    return dataset


# Function to save the model and optimizer states
def save_model(model, model_config, optimizer, sess_path):
    logger.info(f' Saving to {sess_path}')
    torch.save({'model_config': model_config,
                'model_state': model.state_dict(),
                'model_optimizer_state': optimizer.state_dict()}, sess_path)
    logger.info('Done saving')



#========================================================================
##### Training
#========================================================================

def loss_update(init, 
                window_size, 
                events, 
                event_dim,
                controls, 
                model, 
                loss_function,
                teacher_forcing_ratio
                ):

    outputs = model.generate(init, window_size, events=events[:-1], controls=controls, teacher_forcing_ratio=teacher_forcing_ratio, output_type='logit')
    
    assert outputs.shape[:2] == events.shape[:2]

    # Create padding mask
    mask = (events.view(-1) != 0).float()
    # Apply mask to loss calculation and normalize loss
    loss = loss_function(outputs.view(-1, event_dim), events.view(-1)) * mask
    loss = loss.sum() / mask.sum()

    return loss
    


def train_model(model, 
                optimizer, 
                dataset, 
                batch_size,
                num_epochs, 
                window_size, 
                stride_size,
                train_test_ratio,
                early_stopping_patience, 
                event_dim, 
                control_ratio, 
                teacher_forcing_ratio, 
                enable_logging, 
                saving_interval,
                use_transposition,
                device,
                model_config,
                sess_path
                ):

    if enable_logging:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
    if early_stopping_patience > 0:
        early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

    train_data, test_data = dataset.train_test_split(test_size=train_test_ratio)

    last_saving_time = time.time()
    loss_function = nn.CrossEntropyLoss()

    try:
        for epoch in range(num_epochs):

            # Create a progress bar for this epoch
            batch_gen = train_data.batches(batch_size, window_size, stride_size)
            num_batches = train_data.get_length(batch_size, window_size, stride_size)

            # Create a progress bar
            pbar = tqdm(batch_gen, total=num_batches, desc=f'Progressing Epoch {epoch + 1}')
        
            for iteration, (events, controls) in enumerate(pbar):
                # if np.isnan(events).any() or (controls is not None and np.isnan(controls).any()):
                #     print(f'nan found in training sample {iteration}')
                if use_transposition:
                    offset = np.random.choice(np.arange(-6, 6))
                    events, controls = utils.transposition(events, controls, offset)

                events = torch.LongTensor(events).to(device)
                assert events.shape[0] == window_size

                if np.random.random() < control_ratio:
                    controls = torch.FloatTensor(controls).to(device)
                    assert controls.shape[0] == window_size
                else:
                    controls = None
                    

                init = torch.randn(batch_size, model.init_dim).to(device)
                
                train_loss = loss_update(init, window_size, events, event_dim, controls, model, loss_function, teacher_forcing_ratio)
                model.zero_grad()
                train_loss.backward()

                norm = utils.compute_gradient_norm(model.parameters())
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                # Update the tqdm bar with the current loss
                pbar.update()
                pbar.set_postfix({'loss': train_loss.item()}, refresh=True)

                if enable_logging:
                    writer.add_scalar('model/loss', train_loss.item(), iteration)
                    writer.add_scalar('model/norm', norm.item(), iteration)

                logger.info(f'iter {iteration}')

                if time.time() - last_saving_time > saving_interval:
                    save_model(model, model_config, optimizer, sess_path)
                    last_saving_time = time.time()
            
            pbar.close()

            # Create a validation batch and convert values to tensors
            # After each epoch, evaluate the model on the entire test set
            val_losses = []
            for test_events, test_controls in test_data.batches(batch_size, window_size, stride_size):
                #test_events, test_controls = next(iter(test_data.batches(batch_size, window_size, stride_size)))
                test_events = torch.LongTensor(test_events).to(device)
                test_controls = torch.LongTensor(test_controls).to(device)
                
                # Compute validation loss
                val_loss = loss_update(init, window_size, test_events, event_dim, test_controls, model, loss_function, teacher_forcing_ratio)
                val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses)

            # Check if early stopping has been called during iterations
            if early_stopping is not None:
                early_stopping(avg_val_loss, model)
                if early_stopping.early_stop:
                    logger.info(f'Early stopping!')
                    break

            # # If early stopping has been called stop epochs as well
            # if early_stopping.early_stop:
            #     break

    except KeyboardInterrupt:
        save_model(model, model_config, optimizer, sess_path)



#========================================================================
##### Main
#========================================================================

def main():

    options = get_options()

    sess_path = options.sess_path
    data_path = options.data_path
    saving_interval = options.saving_interval

    learning_rate = options.learning_rate
    batch_size = options.batch_size
    num_epochs = options.num_epochs
    window_size = options.window_size
    stride_size = options.stride_size
    train_test_ratio = options.train_test_ratio
    early_stopping_patience = options.early_stopping_patience
    use_transposition = options.use_transposition
    control_ratio = options.control_ratio
    teacher_forcing_ratio = options.teacher_forcing_ratio
    reset_optimizer = options.reset_optimizer
    enable_logging = options.enable_logging

    event_dim = EventSeq.dim()
    model_config = config.model
    model_params = utils.params2dict(options.model_params)
    model_config.update(model_params)
    device = config.device

    logger.info(f'Session path: {sess_path}')
    logger.info(f'Dataset path: {data_path}')
    logger.info(f'Saving interval: {saving_interval}')

    logger.info(f'Hyperparameters: {utils.dict2params(model_config)}')
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

    train_model(model, optimizer, dataset, batch_size, num_epochs, window_size, stride_size, train_test_ratio, early_stopping_patience, event_dim, control_ratio, teacher_forcing_ratio, enable_logging, saving_interval, use_transposition, device, model_config, sess_path)



if __name__ == '__main__':
    main()