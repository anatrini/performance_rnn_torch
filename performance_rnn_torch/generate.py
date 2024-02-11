import argparse
import config
import numpy as np
import os
import torch
import utils

from config import device, model as model_config
from logger import setup_logger
from model import PerformanceRNN
from sequence import Control, ControlSeq


logger = setup_logger('Generator logger')


# ========================================================================
# Settings
# ========================================================================

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--control',
                      dest='control',
                      type='string',
                      default=None,
                      help=('control or a processed data file path, '
                            'e.g., "PITCH_HISTOGRAM;NOTE_DENSITY" like '
                            '"2,0,1,1,0,1,0,1,1,0,0,1;4", or '
                            '";3" (which gives all pitches the same probability), '
                            'or "/path/to/processed/midi/file.data" '
                            '(uses control sequence from the given processed data)'))

    parser.add_argument('-b', '--batch-size',
                      dest='batch_size',
                      type='int',
                      default=config.generate['batch_size'])

    parser.add_argument('-s', '--session',
                      dest='sess_path',
                      type='string',
                      default='save/train.sess',
                      help='session file containing the trained model')

    parser.add_argument('-o', '--output-dir',
                      dest='output_dir',
                      type='string',
                      default='output/')

    parser.add_argument('-l', '--max-length',
                      dest='max_len',
                      type='int',
                      default=config.generate['max_len'])

    parser.add_argument('-g', '--greedy-ratio',
                      dest='greedy_ratio',
                      type='float',
                      default=config.generate['greedy_ratio'])

    parser.add_argument('-B', '--beam-size',
                      dest='beam_size',
                      type='int',
                      default=config.generate['beam_size'])

    parser.add_argument('-S', '--stochastic-beam-search',
                      dest='stochastic_beam_search',
                      action='store_true',
                      default=config.generate['stochastic_beam_search'])

    parser.add_argument('-T', '--temperature',
                      dest='temperature',
                      type='float',
                      default=config.generate['temperature'])

    parser.add_argument('-z', '--init-zero',
                      dest='init_zero',
                      action='store_true',
                      default=config.generate['init_zero'])

    return parser.parse_args()


# ========================================================================
# Generating
# ========================================================================

def generate(model,
             init, 
             max_len,  
             controls, 
             greedy_ratio, 
             temperature, 
             output_dir, 
             use_beam_search, 
             beam_size, 
             stochastic_beam_search
             ):
    
    with torch.no_grad():
        if use_beam_search:
            outputs = model.beam_search(init, max_len, beam_size, controls=controls, temperature=temperature, stochastic=stochastic_beam_search, verbose=True)
        else:
            outputs = model.generate(init, max_len, controls=controls, greedy=greedy_ratio, temperature=temperature, verbose=True)

    outputs = outputs.cpu().numpy().T

    os.makedirs(output_dir, exist_ok=True)
    for i, output in enumerate(outputs):
        name = f'output-{i:03d}.mid'
        path = os.path.join(output_dir, name)
        n_notes = utils.event_indeces_to_midi_file(output, path)
        logger.info(f'===> {path} ({n_notes} notes)')



#========================================================================
# Main
#========================================================================

def main(args=None):
    if args is None:
        args = get_arguments()

    sess_path = args.sess_path
    output_dir = args.output_dir
    batch_size = args.batch_size
    max_len = args.max_len
    greedy_ratio = args.greedy_ratio
    control = args.control
    use_beam_search = args.beam_size > 0
    stochastic_beam_search = args.stochastic_beam_search
    beam_size = args.beam_size
    temperature = args.temperature
    init_zero = args.init_zero
    device = config.device

    logger.info(f'Session path: {sess_path}')
    logger.info(f'Output directory: {output_dir}') ### modificala
    logger.info(f'Batch size: {batch_size}')
    logger.info(f'Max length: {max_len}')
    logger.info(f'Greedy ratio: {greedy_ratio}')
    logger.info(f'Beam size: {beam_size}')
    logger.info(f'Beam search stochastic: {stochastic_beam_search}')
    logger.info(f'Controls: {control}')
    logger.info(f'Temperature: {temperature}')
    logger.info(f'Init zero: {init_zero}')
    logger.info(f'Device: {device}')

    logger.info('Loading session')

    if use_beam_search:
        greedy_ratio = 'DISABLED'
    else:
        beam_size = 'DISABLED'

    assert os.path.isfile(sess_path), f'"{sess_path}" is not a file'

    if control is not None:
        if os.path.isfile(control) or os.path.isdir(control):
            if os.path.isdir(control):
                files = list(utils.find_files_by_extensions(control))
                assert len(files) > 0, f'no file in "{control}"'
                control = np.random.choice(files)
            _, compressed_controls = torch.load(control)
            controls = ControlSeq.recover_compressed_array(compressed_controls)
            if max_len == 0:
                max_len = controls.shape[0]
            controls = torch.tensor(controls, dtype=torch.float32)
            controls = controls.unsqueeze(1).repeat(1, batch_size, 1).to(device)
            control = f'control sequence from "{control}"'
        else:
            pitch_histogram, note_density = control.split(';')
            pitch_histogram = list(filter(len, pitch_histogram.split(',')))
            if len(pitch_histogram) == 0:
                pitch_histogram = np.ones(12) / 12
            else:
                pitch_histogram = np.array(list(map(float, pitch_histogram)))
                assert pitch_histogram.size == 12
                assert np.all(pitch_histogram >= 0)
                pitch_histogram = pitch_histogram / pitch_histogram.sum() \
                    if pitch_histogram.sum() else np.ones(12) / 12
            note_density = int(note_density)
            assert note_density in range(len(ControlSeq.note_density_bins))
            control = Control(pitch_histogram, note_density)
            controls = torch.tensor(control.to_array(), dtype=torch.float32)
            controls = controls.repeat(1, batch_size, 1).to(device)
            control = repr(control)
    else:
        controls = None
        control = 'NONE'

    assert max_len > 0, 'either max length or control sequence length should be given'

    state = torch.load(sess_path, map_location=device)
    model = PerformanceRNN(**state['model_config']).to(device)
    model.load_state_dict(state['model_state'])
    model.eval()
    logger.info(model)
    if init_zero:
        init = torch.zeros(batch_size, model.init_dim).to(device)
    else:
        init = torch.randn(batch_size, model.init_dim).to(device)

    generate(model, init, max_len, controls, greedy_ratio, temperature, output_dir, use_beam_search, beam_size, stochastic_beam_search)



if __name__ == '__main__':
    main()