import time
import logging

### Set a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set a handler to write info on a file with a timestamp
now = time.time()
timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
handler = logging.FileHandler(f'logs/model_info_{timestamp}.log')

# Set a formatter and add it to the logger
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

#========================================================================

import config
import numpy as np
import optparse
import torch
from torch import nn
from torch import optim
import utils

from data import Dataset
from model import PerformanceRNN
#from progress.bar import Bar
from sequence import EventSeq
from tqdm import tqdm
#========================================================================
# Settings
#========================================================================

def get_options():
    parser = optparse.OptionParser()

    parser.add_option('-s', '--session',
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

    parser.add_option('-S', '--stride-size',
                      dest='stride_size',
                      type='int',
                      default=config.train['stride_size'])

    parser.add_option('-c', '--control-ratio',
                      dest='control_ratio',
                      type='float',
                      default=config.train['control_ratio'])

    parser.add_option('-T', '--teacher-forcing-ratio',
                      dest='teacher_forcing_ratio',
                      type='float',
                      default=config.train['teacher_forcing_ratio'])

    parser.add_option('-t', '--use-transposition',
                      dest='use_transposition',
                      action='store_true',
                      default=config.train['use_transposition'])

    parser.add_option('-p', '--model-params',
                      dest='model_params',
                      type='string',
                      default='')
                      
    parser.add_option('-r', '--reset-optimizer',
                      dest='reset_optimizer',
                      action='store_true',
                      default=False)
                      
    parser.add_option('-L', '--enable-logging',
                      dest='enable_logging',
                      action='store_true',
                      default=False)

    return parser.parse_args()[0]


#========================================================================
# Load session and dataset
#========================================================================

# Function to load the model and optimizer states if a session exists
def load_session(sess_path,
                model_config,
                device,
                learning_rate,
                reset_optimizer):
    #global sess_path, model_config, device, learning_rate, reset_optimizer

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

#------------------------------------------------------------------------

# Function to save the model and optimizer states
def save_model(model, model_config, optimizer, sess_path):
    logger.info(f' Saving to {sess_path}')
    torch.save({'model_config': model_config,
                'model_state': model.state_dict(),
                'model_optimizer_state': optimizer.state_dict()}, sess_path)
    logger.info('Done saving')


#========================================================================
# Training
#========================================================================

def train_model(model, 
                optimizer, 
                dataset, 
                batch_size,
                num_epochs, 
                window_size, 
                stride_size, 
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
    print(f'Num epochs: {num_epochs}')
    if enable_logging:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

    last_saving_time = time.time()
    loss_function = nn.CrossEntropyLoss()

    # Calculate number of batches per epoch
    num_samples = len(dataset.samples)
    num_batches = num_samples // batch_size
    last_batch_size = num_samples % batch_size
    #print(f'Batch stuff: {num_samples}, {num_batches, {last_batch_size}}')

    try:
        for epoch in range(num_epochs):
            # Create a progress bar for this epoch
            print(f'Questa è epoch: {epoch + 1}')
            batch_gen = dataset.batches(batch_size, window_size, stride_size)
            #bar = Bar(f'Processing Epoch {epoch + 1}', max=num_batches)

            # Create a progress bar
            pbar = tqdm(batch_gen, total=num_batches, desc=f'Progressing Epoch {epoch + 1}')
        
            for iteration, (events, controls) in enumerate(pbar):
                print(f'events shape: {events.shape}')
                print(f'controls shape: {controls.shape}')
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
                
                # Check if current batch is the last and if it has less samples
                if iteration == num_batches - 1 and last_batch_size > 0:
                    print(f'Condition met!')
                    # Adjust your events and controls here to account for the smaller batch size
                    events = events[:last_batch_size]
                    print(f'events shape last batch: {events.shape}')
                    if controls is not None:
                        controls = controls[:last_batch_size]

                init = torch.randn(batch_size, model.init_dim).to(device)
                steps_to_generate = min(window_size, events.shape[0])
                print(f'steps_to_generate: {steps_to_generate}')
                #print(f'controls shape: {controls.shape}')
                outputs = model.generate(init, 
                                        steps_to_generate, 
                                        events=events[:-1], 
                                        controls=controls,
                                        teacher_forcing_ratio=teacher_forcing_ratio, 
                                        output_type='logit'
                                        )
        
                assert outputs.shape[:2] == events.shape[:2]

                loss = loss_function(outputs.view(-1, event_dim), events.view(-1))
                model.zero_grad()
                loss.backward()

                norm = utils.compute_gradient_norm(model.parameters())
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
                optimizer.step()
                #bar.next() # Update the progress bar

                if enable_logging:
                    writer.add_scalar('model/loss', loss.item(), iteration)
                    writer.add_scalar('model/norm', norm.item(), iteration)

                logger.info(f'iter {iteration}, loss: {loss.item()}')

                if time.time() - last_saving_time > saving_interval:
                    save_model(model, model_config, optimizer, sess_path)
                    last_saving_time = time.time()
            
            pbar.close()

    except KeyboardInterrupt:
        save_model(model, model_config, optimizer, sess_path)



#========================================================================
# Main
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

    print('-' * 70)

    logger.info(f'Session path: {sess_path}')
    logger.info(f'Dataset path: {data_path}')
    logger.info(f'Saving interval: {saving_interval}')
    print('-' * 70)

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
    print('-' * 70)

    load_dataset(data_path)

    logger.info('Loading session')
    model, optimizer = load_session(sess_path, model_config, device, learning_rate, reset_optimizer)
    logger.info(model)
    print('-' * 70)

    logger.info('Loading dataset')
    dataset = load_dataset(data_path)
    logger.info(dataset)
    print('-' * 70)

    train_model(model, optimizer, dataset, batch_size, num_epochs, window_size, stride_size, event_dim, control_ratio, teacher_forcing_ratio, enable_logging, saving_interval, use_transposition, device, model_config, sess_path)


if __name__ == '__main__':
    main()