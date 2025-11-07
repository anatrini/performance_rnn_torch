#!/usr/bin/env python3
"""
MIDI Preprocessing Script

This script preprocesses MIDI files into a format suitable for training
the PerformanceRNN model. It converts MIDI files into sequences of events
and control sequences.
"""
import argparse
import hashlib
import os
import torch

from concurrent.futures import ProcessPoolExecutor
from performance_rnn_torch.utils.logger import setup_logger
from performance_rnn_torch.core.sequence import NoteSeq, EventSeq, ControlSeq
from performance_rnn_torch.utils import helpers, paths
from tqdm import tqdm


logger = setup_logger('Preprocess logger')

#========================================================================
# Settings
#========================================================================

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Preprocess MIDI files into training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults (reads from data/midi/, saves to data/processed/)
  python scripts/preprocess.py

  # Use all available CPU cores
  python scripts/preprocess.py --num_workers -1

  # Use 4 workers
  python scripts/preprocess.py --num_workers 4

  # Custom paths
  python scripts/preprocess.py --midi_root custom/midi/ --save_dir custom/processed/
        """
    )

    parser.add_argument('-m', '--midi_root',
                      dest='midi_root',
                      type=str,
                      default=str(paths.midi_dir),
                      help=f'Root directory of MIDI files (default: {paths.midi_dir})')

    parser.add_argument('-s', '--save_dir',
                      dest='save_dir',
                      type=str,
                      default=str(paths.processed_dir),
                      help=f'Directory to save processed data (default: {paths.processed_dir})')

    parser.add_argument('-w', '--num_workers',
                      dest='num_workers',
                      default=1,
                      type=int,
                      help='Number of worker processes (default: 1, use -1 for all CPU cores)')

    return parser.parse_args()

def preprocess_midi(path):
    note_seq = NoteSeq.from_midi_file(path)
    note_seq.adjust_time(-note_seq.notes[0].start)
    event_seq = EventSeq.from_note_seq(note_seq)
    control_seq = ControlSeq.from_event_seq(event_seq)
    return event_seq.to_array(), control_seq.to_compressed_array()

def preprocess_midi_files_under(midi_root, save_dir, num_workers):
    from pathlib import Path

    midi_paths = list(helpers.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    midi_root = Path(midi_root).resolve()  # Get absolute path
    save_dir = Path(save_dir).resolve()

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

        # Preserve directory structure from midi_root
        path = Path(path).resolve()
        relative_path = path.relative_to(midi_root)

        # Determine the target subdirectory
        if relative_path.parent == Path('.'):
            # File is directly in midi_root (e.g., midi_root/file.mid)
            # Use the last component of midi_root as the subdirectory
            composer_folder = midi_root.name
            target_dir = save_dir / composer_folder
        else:
            # File is in a subdirectory (e.g., midi_root/composer/file.mid)
            # Preserve the subdirectory structure
            target_dir = save_dir / relative_path.parent

        target_dir.mkdir(parents=True, exist_ok=True)

        # Save with hash to avoid name collisions
        name = path.stem  # filename without extension
        code = hashlib.md5(str(path).encode()).hexdigest()
        save_path = target_dir / f'{name}-{code}.data'

        torch.save(future.result(), save_path)

    logger.info('Done')

def main(args=None):
    if args is None:
        args = get_arguments()

    midi_root = args.midi_root
    save_dir = args.save_dir
    num_workers = args.num_workers

    # Handle -1 to use all CPU cores
    if num_workers == -1:
        num_workers = os.cpu_count()
        logger.info(f'Using all available CPU cores: {num_workers}')
    elif num_workers < 1:
        logger.error('num_workers must be >= 1 (or -1 for all CPU cores)')
        return 1

    logger.info(f'MIDI root: {midi_root}')
    logger.info(f'Save directory: {save_dir}')
    logger.info(f'Number of workers: {num_workers}')

    preprocess_midi_files_under(midi_root, save_dir, num_workers)



if __name__ == '__main__':
    main()