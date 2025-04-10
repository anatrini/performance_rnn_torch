import argparse
import hashlib
import os
import torch
import utils

from concurrent.futures import ProcessPoolExecutor
from logger import setup_logger
from sequence import NoteSeq, EventSeq, ControlSeq
from tqdm import tqdm


logger = setup_logger('Preprocess logger')

#========================================================================
# Settings
#========================================================================

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--midi_root',
                      dest='midi_root',
                      type=str,
                      default=None,
                      help='The root directory of MIDI files')
    
    parser.add_argument('-s', '--save_dir',
                      dest='save_dir',
                      type=str,
                      default=None,
                      help='The directory to save the processed data')
    
    parser.add_argument('-w', '--num_workers',
                      dest='num_workers',
                      default=0,
                      type=int,
                      help='The number of worker processes to use. Default 0, preprocessing is executing on a single thread')
    
    return parser.parse_args()

def preprocess_midi(path):
    note_seq = NoteSeq.from_midi_file(path)
    note_seq.adjust_time(-note_seq.notes[0].start)
    event_seq = EventSeq.from_note_seq(note_seq)
    control_seq = ControlSeq.from_event_seq(event_seq)
    return event_seq.to_array(), control_seq.to_compressed_array()

def preprocess_midi_files_under(midi_root, save_dir, num_workers):
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}-{}.data'
    
    results = []
    executor = ProcessPoolExecutor(num_workers)

    for path in midi_paths:
        try:
            results.append((path, executor.submit(preprocess_midi, path)))
        except KeyboardInterrupt:
            logger.info(' Abort')
            return
        except:
            logger.error(' Error')
            continue
    
    for path, future in tqdm(results, desc='Processing'):
        logger.info(f' [{path}]')
        name = os.path.basename(path)
        code = hashlib.md5(path.encode()).hexdigest()
        save_path = os.path.join(save_dir, f'{name}-{code}.data')
        torch.save(future.result(), save_path)

    logger.info('Done')

def main(args=None):
    if args is None:
        args = get_arguments()

    midi_root = args.midi_root
    save_dir = args.save_dir
    num_workers = args.num_workers

    preprocess_midi_files_under(midi_root, save_dir, num_workers)



if __name__ == '__main__':
    main()