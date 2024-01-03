import platform
import torch
from sequence import EventSeq, ControlSeq



#========================================================================
##### Set device according to OS and available devices
#========================================================================

if platform.system() == 'Darwin':  # Mac OS
    if torch.backends.mps.is_available():  # Check if Metal is available
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
else:  # Linux o Windows
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#========================================================================
##### Model's and train settings
#========================================================================

model = {
    'init_dim': 32,
    'event_dim': EventSeq.dim(),
    'control_dim': ControlSeq.dim(),
    'hidden_dim': 512,
    'gru_layers': 3,
    'gru_dropout': 0.3,
}

train = {
    'batch_size': 64,
    'num_epochs': 100,
    'window_size': 200,
    'stride_size': 10,
    'learning_rate': 0.001,
    'train_test_ratio': 0.3,
    'early_stopping_patience': 10,
    'use_transposition': False,
    'control_ratio': 1.0,
    'teacher_forcing_ratio': 1.0
}