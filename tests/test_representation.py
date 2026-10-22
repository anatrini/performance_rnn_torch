"""Event vocabulary, controls, sustain and files, checked against Magenta's definitions."""

import numpy as np
import pytest
import torch
from pretty_midi import ControlChange, Instrument, Note, PrettyMIDI

from performance_rnn_torch.core import PerformanceRNN
from performance_rnn_torch.core.sequence import (
    DENSITY_BIN_RANGES, FORMAT_VERSION, ControlSeq, EventSeq, NoteSeq, apply_sustain,
    bin_to_velocity, load_processed, save_processed, velocity_to_bin)
from performance_rnn_torch.training import check_format, load_session, save_model

R = EventSeq.feat_ranges()
ON, OFF, SHIFT, VEL = (R[k].start for k in ('note_on', 'note_off', 'time_shift', 'velocity'))


def seq(*notes):
    """notes as (pitch, velocity, start, end)"""
    return NoteSeq([Note(v, p, s, e) for p, v, s, e in notes])


def random_notes(seed, n=200):
    rng = np.random.default_rng(seed)
    starts = np.cumsum(rng.exponential(0.15, n))
    return seq(*[(int(rng.integers(21, 109)), int(rng.integers(1, 128)), float(s),
                  float(s + rng.uniform(0.005, 2.0))) for s in starts])


# ------------------------------------------------------------------ vocabulary

def test_vocabulary_is_magentas():
    assert dict(EventSeq.feat_dims()) == {'note_on': 128, 'note_off': 128, 'time_shift': 100, 'velocity': 32}
    assert EventSeq.dim() == 388
    assert (ON, OFF, SHIFT, VEL) == (0, 128, 256, 356)


def test_velocity_bins():
    assert [velocity_to_bin(v) for v in (1, 4, 5, 80, 127)] == [1, 1, 2, 20, 32]
    assert [bin_to_velocity(b) for b in (1, 2, 20, 32)] == [1, 5, 77, 125]


def test_encoding_of_two_notes():
    # C4 then E4, 0.5 s each, velocity 80 (bin 20): the velocity is written once,
    # and at 0.5 s the note_off of C4 comes before the note_on of E4
    events = EventSeq.from_note_seq(seq((60, 80, 0.0, 0.5), (64, 80, 0.5, 1.0))).indices
    assert events.tolist() == [VEL + 19, ON + 60, SHIFT + 49, OFF + 60, ON + 64, SHIFT + 49, OFF + 64]


def test_long_pauses_are_split_into_shifts_of_one_second():
    events = EventSeq.from_note_seq(seq((60, 64, 0.0, 0.1), (62, 64, 2.6, 2.7))).indices
    shifts = [i - SHIFT + 1 for i in events if SHIFT <= i < VEL]
    assert shifts == [10, 100, 100, 50, 10]


def test_very_short_notes_last_one_step():
    events = EventSeq.from_note_seq(seq((60, 64, 0.0, 0.001))).indices
    assert events.tolist() == [VEL + velocity_to_bin(64) - 1, ON + 60, SHIFT, OFF + 60]


@pytest.mark.parametrize('seed', range(5))
def test_round_trip_keeps_quantised_notes(seed):
    original = random_notes(seed)
    decoded = EventSeq.from_note_seq(original).to_note_seq(max_note_duration=None)

    def quantised(note_seq):
        notes = []
        for n in note_seq.notes:
            s, e = round(n.start * 100), round(n.end * 100)
            e = max(e, s + 1)
            notes.append((n.pitch, s, e, bin_to_velocity(velocity_to_bin(n.velocity))))
        return sorted(notes)

    expected = quantised(original)
    # overlapping notes of the same pitch pair up first-in first-out, as in Magenta,
    # so compare per pitch the multiset of onsets and of offsets and the velocities
    got = sorted((n.pitch, round(n.start * 100), round(n.end * 100), n.velocity) for n in decoded.notes)
    assert sorted((p, s, v) for p, s, _, v in got) == sorted((p, s, v) for p, s, _, v in expected)
    assert sorted((p, e) for p, _, e, _ in got) == sorted((p, e) for p, _, e, _ in expected)


def test_decoding_limits_notes_without_note_off():
    notes = EventSeq([ON + 60, SHIFT + 99] * 8).to_note_seq().notes
    assert len(notes) == 8
    assert max(n.end - n.start for n in notes) == pytest.approx(5.0)


# ------------------------------------------------------------------ sustain

def test_sustain_extends_until_the_pedal_is_released():
    notes = [Note(64, 60, 0.0, 0.5), Note(64, 64, 1.0, 1.2)]
    pedal = [ControlChange(64, 127, 0.2), ControlChange(64, 0, 2.0)]
    out = apply_sustain(notes, pedal)
    assert [(n.start, n.end) for n in out] == [(0.0, 2.0), (1.0, 2.0)]


def test_sustain_ends_a_held_note_when_the_pitch_is_played_again():
    notes = [Note(64, 60, 0.0, 0.5), Note(64, 60, 1.0, 1.2)]
    pedal = [ControlChange(64, 100, 0.1), ControlChange(64, 10, 3.0)]
    out = apply_sustain(notes, pedal)
    assert [(n.start, n.end) for n in out] == [(0.0, 1.0), (1.0, 3.0)]


def test_notes_without_pedal_are_unchanged():
    notes = [Note(64, 60, 0.0, 0.5), Note(64, 62, 0.5, 0.7)]
    assert [(n.start, n.end) for n in apply_sustain(notes, [])] == [(0.0, 0.5), (0.5, 0.7)]


# ------------------------------------------------------------------ controls
# Straight translations of Magenta's NoteDensityPerformanceControlSignal.extract
# and PitchHistogramPerformanceControlSignal.extract, as reference.

def magenta_density(indices):
    window, out, prev_type, prev = 300, [], None, 0.0
    kind = lambda i: 'on' if i < OFF else 'off' if i < SHIFT else 'shift' if i < VEL else 'vel'
    for i in range(len(indices)):
        if prev_type is not None and prev_type != 'shift':
            out.append(prev)
            prev_type = kind(indices[i])
            continue
        j, offset, count = i, 0, 0
        while offset < window and j < len(indices):
            if kind(indices[j]) == 'on':
                count += 1
            elif kind(indices[j]) == 'shift':
                offset += indices[j] - SHIFT + 1
            j += 1
        actual = min(offset, window)
        density = count * 100 / actual if actual > 0 else 0.0
        out.append(density)
        prev_type, prev = kind(indices[i]), density
    return out


def magenta_histogram(indices):
    window, out, prev_type, prev = 500, [], None, None
    kind = lambda i: 'on' if i < OFF else 'off' if i < SHIFT else 'shift' if i < VEL else 'vel'
    base = set()
    for i in range(len(indices)):
        if kind(indices[i]) == 'on':
            base.add(indices[i] - ON)
        elif kind(indices[i]) == 'off':
            base.discard(indices[i] - OFF)
        if prev_type is not None and prev_type != 'shift':
            out.append(prev)
            prev_type = kind(indices[i])
            continue
        j, offset, active, hist = i, 0, set(base), [0.01] * 12
        while offset < window and j < len(indices):
            k = kind(indices[j])
            if k == 'on':
                active.add(indices[j] - ON)
            elif k == 'off':
                active.discard(indices[j] - OFF)
            elif k == 'shift':
                for p in active:
                    hist[p % 12] += (indices[j] - SHIFT + 1) / 100
                offset += indices[j] - SHIFT + 1
            j += 1
        total = sum(hist)
        out.append([h / total for h in hist])
        prev_type, prev = kind(indices[i]), out[-1]
    return out


@pytest.mark.parametrize('seed', range(4))
def test_controls_match_magenta(seed):
    events = EventSeq.from_note_seq(random_notes(seed))
    controls = ControlSeq.from_event_seq(events)
    indices = events.indices.tolist()

    expected_class = np.searchsorted(DENSITY_BIN_RANGES, magenta_density(indices), side='right')
    np.testing.assert_array_equal(controls.density, expected_class)
    np.testing.assert_allclose(controls.histogram, magenta_histogram(indices), rtol=1e-9, atol=1e-12)


def test_density_classes():
    # four notes per second for five seconds: class 3 (4-8 per second) at the start
    notes = seq(*[(60, 64, i / 4, i / 4 + 0.1) for i in range(20)])
    controls = ControlSeq.from_event_seq(EventSeq.from_note_seq(notes))
    assert controls.density[0] == 3


def test_histogram_of_a_held_chord():
    notes = seq((60, 64, 0.0, 4.0), (64, 64, 0.0, 4.0), (67, 64, 0.0, 4.0))
    histogram = ControlSeq.from_event_seq(EventSeq.from_note_seq(notes)).histogram[0]
    assert histogram[[0, 4, 7]] == pytest.approx([1 / 3] * 3, abs=0.01)
    assert histogram.sum() == pytest.approx(1.0)


def test_control_vectors():
    controls = ControlSeq.from_event_seq(EventSeq.from_note_seq(random_notes(0))).to_array()
    dens = ControlSeq.feat_ranges()['note_density']
    hist = ControlSeq.feat_ranges()['pitch_histogram']
    assert controls.shape[1] == ControlSeq.dim() == 20
    assert (controls[:, dens].sum(axis=1) == 1).all()
    assert np.allclose(controls[:, hist].sum(axis=1), 1, atol=0.03)


# ------------------------------------------------------------------ files

def test_processed_file_round_trip(tmp_path):
    events = EventSeq.from_note_seq(random_notes(1))
    controls = ControlSeq.from_event_seq(events)
    save_processed(tmp_path / 'piece.npz', events, controls)
    loaded_events, loaded_controls = load_processed(tmp_path / 'piece.npz')
    np.testing.assert_array_equal(loaded_events, events.indices)
    np.testing.assert_allclose(loaded_controls, controls.to_array(), atol=1e-6)


def test_files_of_another_version_are_refused(tmp_path):
    np.savez(tmp_path / 'old.npz', events=np.zeros(3, np.uint16))
    with pytest.raises(ValueError, match='preprocess'):
        load_processed(tmp_path / 'old.npz')


def test_checkpoints_carry_the_format_version(tmp_path):
    cfg = dict(init_dim=4, event_dim=EventSeq.dim(), control_dim=ControlSeq.dim(),
               hidden_dim=8, gru_layers=1, gru_dropout=0.0)
    model = PerformanceRNN(**cfg)
    path = tmp_path / 'm.sess'
    save_model(model, cfg, torch.optim.Adam(model.parameters()), path)
    state = torch.load(path, weights_only=True)
    assert state['format_version'] == FORMAT_VERSION
    check_format(state, path)

    del state['format_version']
    torch.save(state, tmp_path / 'old.sess')
    with pytest.raises(ValueError, match='train a new model'):
        load_session(tmp_path / 'old.sess', cfg, torch.device('cpu'), 1e-3, False)


def test_midi_file_with_pedal(tmp_path):
    midi = PrettyMIDI()
    piano = Instrument(0)
    piano.notes = [Note(70, 60, 0.0, 0.3)]
    piano.control_changes = [ControlChange(64, 127, 0.1), ControlChange(64, 0, 1.5)]
    midi.instruments.append(piano)
    drums = Instrument(0, is_drum=True)
    drums.notes = [Note(100, 36, 0.0, 0.1)]
    midi.instruments.append(drums)
    midi.write(str(tmp_path / 'p.mid'))

    notes = NoteSeq.from_midi_file(tmp_path / 'p.mid').notes
    assert [(n.pitch, round(n.end, 3)) for n in notes] == [(60, 1.5)]
    assert NoteSeq.from_midi_file(tmp_path / 'p.mid', sustain=False).notes[0].end == pytest.approx(0.3)
