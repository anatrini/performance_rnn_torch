"""Helper utility functions for Performance RNN."""

import os
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np


def find_files_by_extensions(
    root: str,
    exts: Optional[List[str]] = None
) -> Generator[str, None, None]:
    """
    Recursively find all files with specified extensions.

    Args:
        root: Root directory to search
        exts: List of file extensions (e.g., ['.mid', '.midi']).
              If None or empty, returns all files.

    Yields:
        str: Full path to each matching file
    """
    exts = exts or []

    def _has_ext(name: str) -> bool:
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext.lower()):
                return True
        return False

    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)


def event_indices_to_midi_file(event_indices: np.ndarray, midi_file_name: str) -> int:
    """
    Convert event indices to a MIDI file.

    Args:
        event_indices: Array of event indices
        midi_file_name: Output MIDI file path

    Returns:
        int: Number of notes in the generated MIDI file
    """
    # Import here to avoid circular import
    from performance_rnn_torch.core.sequence import EventSeq

    note_seq = EventSeq.from_array(event_indices).to_note_seq()
    note_seq.to_midi_file(midi_file_name)
    return len(note_seq.notes)


def transposition(
    events: np.ndarray,
    controls: np.ndarray,
    offset: int = 0
) -> tuple:
    """
    Transpose events and controls by a number of semitones (at most an octave).

    note_on and note_off events move by the offset; a pitch that would leave
    the MIDI range 0-127 moves an octave less. The pitch histogram of the
    controls rotates by the same offset.

    Args:
        events: Event array of shape [steps, batch_size]
        controls: Control array of shape [steps, batch_size, control_dim]
        offset: Number of semitones, -12 to 12

    Returns:
        tuple: (transposed_events, transposed_controls)
    """
    # Import here to avoid circular import
    from performance_rnn_torch.core.sequence import NUM_PITCHES, ControlSeq, EventSeq

    assert -12 <= offset <= 12, offset
    events = np.array(events, dtype=np.int64)
    controls = np.array(controls, dtype=np.float32)
    ranges = EventSeq.feat_ranges()

    for name in ('note_on', 'note_off'):
        start = ranges[name].start
        mask = (events >= start) & (events < start + NUM_PITCHES)
        pitch = events[mask] - start + offset
        pitch = np.where(pitch >= NUM_PITCHES, pitch - 12, pitch)
        pitch = np.where(pitch < 0, pitch + 12, pitch)
        events[mask] = start + pitch

    hist = ControlSeq.feat_ranges()['pitch_histogram']
    controls[..., hist.start:hist.stop] = np.roll(controls[..., hist.start:hist.stop], offset, -1)

    return events, controls
