"""Synthetic MIDI pieces and model inputs used by the baseline tests.

Everything is built from fixed numbers or a seeded generator, so the same
pieces come out on every machine and no MIDI files need to be stored.
"""

import numpy as np
from pretty_midi import Instrument, Note, PrettyMIDI


def _midi(notes, program=0):
    midi = PrettyMIDI(resolution=480, initial_tempo=120)
    inst = Instrument(program=program)
    inst.notes = [Note(velocity=v, pitch=p, start=s, end=e) for p, v, s, e in notes]
    midi.instruments.append(inst)
    return midi


def scale():
    # C major scale up and down, crescendo then diminuendo
    pitches = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60]
    return _midi([(p, 30 + 6 * min(i, 14 - i), 0.25 * i, 0.25 * i + 0.2)
                  for i, p in enumerate(pitches)])


def chords_and_overlaps():
    # simultaneous notes, a repeated pitch that overlaps itself, a sustained bass
    return _midi([
        (48, 80, 0.0, 2.0),
        (60, 70, 0.0, 0.5), (64, 70, 0.0, 0.5), (67, 70, 0.0, 0.5),
        (62, 90, 0.5, 1.2), (62, 50, 0.9, 1.5),
        (65, 60, 1.0, 1.5), (69, 60, 1.0, 1.5),
    ])


def rests():
    # pauses shorter and longer than one second, up to five seconds
    return _midi([
        (60, 64, 0.0, 0.1),
        (62, 64, 0.6, 0.7),
        (64, 64, 2.0, 2.1),
        (65, 64, 4.5, 4.6),
        (67, 64, 9.6, 9.7),
    ])


def extremes():
    # pitches outside the piano range, velocities 1 and 127, very short notes
    return _midi([
        (12, 1, 0.0, 0.3), (21, 127, 0.0, 0.3), (108, 64, 0.3, 0.6),
        (120, 64, 0.3, 0.6), (60, 100, 0.6, 0.605), (61, 100, 0.607, 0.61),
        (62, 100, 0.612, 0.62),
    ])


def random_piece(seed=0, n=300):
    rng = np.random.default_rng(seed)
    starts = np.cumsum(rng.exponential(0.12, n))
    notes = []
    for s in starts:
        d = float(rng.uniform(0.03, 1.5))
        notes.append((int(rng.integers(30, 100)), int(rng.integers(20, 120)), float(s), float(s) + d))
    return _midi(notes)


PIECES = {
    "scale": scale,
    "chords_and_overlaps": chords_and_overlaps,
    "rests": rests,
    "extremes": extremes,
    "random_piece": random_piece,
}

# Small model for comparing outputs step by step
MODEL_CONFIG = dict(init_dim=8, event_dim=388, control_dim=24, hidden_dim=32,
                    gru_layers=3, gru_dropout=0.3)
STEPS, BATCH = 40, 3


def model_inputs(seed=1):
    rng = np.random.default_rng(seed)
    events = rng.integers(0, MODEL_CONFIG["event_dim"], (STEPS, BATCH))
    controls = rng.random((STEPS, BATCH, MODEL_CONFIG["control_dim"]))
    init = rng.standard_normal((BATCH, MODEL_CONFIG["init_dim"]))
    return events, controls.astype(np.float32), init.astype(np.float32)
