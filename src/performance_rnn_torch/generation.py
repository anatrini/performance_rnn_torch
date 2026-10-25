"""Controls for generation and writing the generated performances."""

import re
from pathlib import Path

import numpy as np
import torch

from performance_rnn_torch.core.sequence import Control, ControlSeq, load_processed
from performance_rnn_torch.utils import helpers


def parse_control(spec: str) -> Control:
    """A control from "PITCH_HISTOGRAM;NOTE_DENSITY", e.g. "2,0,1,0,1,1,0,1,0,1,0,1;3".

    The histogram has 12 non-negative weights (C to B), normalised here; an
    empty histogram gives all pitch classes the same weight. The note density
    is a class 0-7 (see ControlSeq).
    """
    histogram_text, _, density_text = spec.partition(';')
    weights = [float(w) for w in histogram_text.split(',') if w.strip()]
    if not weights:
        weights = [1.0] * 12
    histogram = np.asarray(weights, dtype=np.float64)
    if histogram.size != 12 or (histogram < 0).any():
        raise ValueError(f'pitch histogram needs 12 non-negative values, got "{histogram_text}"')
    histogram = histogram / histogram.sum() if histogram.sum() else np.full(12, 1 / 12)
    classes = ControlSeq.feat_dims()['note_density']
    density = int(density_text)
    if not 0 <= density < classes:
        raise ValueError(f'note density class must be 0-{classes - 1}, got {density}')
    return Control(histogram, density)


def controls_from_spec(spec: str | None, batch_size: int, device) -> tuple[torch.Tensor | None, str]:
    """Controls [steps or 1, batch, control_dim] and a description, from a control
    text, a preprocessed file (.npz), a folder of them (one chosen at random) or None.
    """
    if spec is None:
        return None, 'none'
    path = Path(spec)
    if path.exists():
        if path.is_dir():
            files = sorted(helpers.find_files_by_extensions(path, ['.npz']))
            if not files:
                raise ValueError(f'no preprocessed file (.npz) in {path}')
            path = Path(np.random.choice(files))
        _, controls = load_processed(path)
        tensor = torch.as_tensor(controls, dtype=torch.float32)[:, None].expand(-1, batch_size, -1)
        return tensor.contiguous().to(device), f'controls of {path}'
    control = parse_control(spec)
    tensor = torch.as_tensor(control.to_array())[None, None].expand(1, batch_size, -1)
    return tensor.contiguous().to(device), repr(control)


def model_name(sess_path) -> str:
    """File-system-safe name of a checkpoint, e.g. "jsbach" for models/jsbach.sess."""
    return re.sub(r'[^\w\-]', '_', Path(sess_path).stem)


def next_file_number(output_dir, name: str) -> int:
    """1 + the highest number of the files <name>_<number>.mid in output_dir."""
    numbers = [
        int(m.group(1))
        for p in Path(output_dir).glob(f'{name}_*.mid')
        if (m := re.fullmatch(rf'{re.escape(name)}_(\d+)\.mid', p.name))
    ]
    return max(numbers, default=0) + 1


def write_midi_files(events: torch.Tensor, output_dir, name: str) -> list[tuple[Path, int]]:
    """Write each sequence of events [steps, batch] to <name>_<number>.mid; returns (path, notes)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first = next_file_number(output_dir, name)
    written = []
    for i, sequence in enumerate(events.cpu().numpy().T):
        path = output_dir / f'{name}_{first + i:03d}.mid'
        written.append((path, helpers.event_indices_to_midi_file(sequence, path)))
    return written
