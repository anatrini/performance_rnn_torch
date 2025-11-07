"""Helper utility functions for Performance RNN."""

import os
from pathlib import Path
from typing import Generator, List, Optional, Dict, Any

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


def event_indices_to_midi_file(
    event_indices: np.ndarray,
    midi_file_name: str,
    velocity_scale: float = 0.8
) -> int:
    """
    Convert event indices to a MIDI file.

    Args:
        event_indices: Array of event indices
        midi_file_name: Output MIDI file path
        velocity_scale: Scale factor for note velocity (default: 0.8)

    Returns:
        int: Number of notes in the generated MIDI file
    """
    # Import here to avoid circular import
    from performance_rnn_torch.core.sequence import EventSeq

    event_seq = EventSeq.from_array(event_indices)
    note_seq = event_seq.to_note_seq()

    # Apply velocity scaling
    for note in note_seq.notes:
        note.velocity = int((note.velocity - 64) * velocity_scale + 64)

    note_seq.to_midi_file(midi_file_name)
    return len(note_seq.notes)


def transposition(
    events: np.ndarray,
    controls: np.ndarray,
    offset: int = 0
) -> tuple:
    """
    Transpose events and controls by a given offset.

    This function shifts note_on and note_off events by the specified offset,
    wrapping around octaves to maintain valid note ranges. It also rotates
    the pitch histogram in the control sequence accordingly.

    Args:
        events: Event array of shape [steps, batch_size, event_dim]
        controls: Control array of shape [steps, batch_size, control_dim]
        offset: Number of semitones to transpose (can be negative)

    Returns:
        tuple: (transposed_events, transposed_controls)
    """
    # Import here to avoid circular import
    from performance_rnn_torch.core.sequence import EventSeq, ControlSeq

    events = np.array(events, dtype=np.int64)
    controls = np.array(controls, dtype=np.float32)
    event_feat_ranges = EventSeq.feat_ranges()

    on = event_feat_ranges['note_on']
    off = event_feat_ranges['note_off']

    if offset > 0:
        # For positive offset, shift notes up
        # Notes that would go out of range wrap down by one octave
        indeces0 = (((on.start <= events) & (events < on.stop - offset)) |
                    ((off.start <= events) & (events < off.stop - offset)))
        indeces1 = (((on.stop - offset <= events) & (events < on.stop)) |
                    ((off.stop - offset <= events) & (events < off.stop)))
        events[indeces0] += offset
        events[indeces1] += offset - 12
    elif offset < 0:
        # For negative offset, shift notes down
        # Notes that would go out of range wrap up by one octave
        indeces0 = (((on.start - offset <= events) & (events < on.stop)) |
                    ((off.start - offset <= events) & (events < off.stop)))
        indeces1 = (((on.start <= events) & (events < on.start - offset)) |
                    ((off.start <= events) & (events < off.start - offset)))
        events[indeces0] += offset
        events[indeces1] += offset + 12

    # Validate all events are in valid range
    assert ((0 <= events) & (events < EventSeq.dim())).all(), \
        "Transposed events out of valid range"

    # Rotate pitch histogram in controls
    histr = ControlSeq.feat_ranges()['pitch_histogram']
    controls[:, :, histr.start:histr.stop] = np.roll(
        controls[:, :, histr.start:histr.stop], offset, -1)

    return events, controls


def dict2params(d: Dict[str, Any], separator: str = ',') -> str:
    """
    Convert a dictionary to a parameter string.

    Args:
        d: Dictionary to convert
        separator: Character to separate key-value pairs (default: ',')

    Returns:
        str: Parameter string in format 'key1=value1,key2=value2'

    Example:
        >>> dict2params({'lr': 0.001, 'batch': 32})
        'lr=0.001,batch=32'
    """
    return separator.join(f'{k}={v}' for k, v in d.items())


def params2dict(
    params_str: str,
    separator: str = ',',
    equals: str = '='
) -> Dict[str, Any]:
    """
    Convert a parameter string to a dictionary.

    WARNING: This function uses eval() which can be a security risk.
    Only use with trusted input.

    Args:
        params_str: Parameter string to parse
        separator: Character separating key-value pairs (default: ',')
        equals: Character separating keys and values (default: '=')

    Returns:
        dict: Parsed dictionary

    Example:
        >>> params2dict('lr=0.001,batch=32')
        {'lr': 0.001, 'batch': 32}
    """
    d = {}
    for item in params_str.split(separator):
        item = item.split(equals)
        if len(item) < 2:
            continue
        k, *v = item
        # WARNING: eval is used here - potential security risk
        # Consider using ast.literal_eval for safer parsing
        d[k] = eval(equals.join(v))
    return d


def compute_gradient_norm(parameters, norm_type: float = 2.0) -> float:
    """
    Compute the norm of gradients across all parameters.

    Args:
        parameters: Iterable of parameters (e.g., model.parameters())
        norm_type: Type of norm to compute (default: 2.0 for L2 norm)

    Returns:
        float: The computed gradient norm
    """
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
