"""MIDI performances as event sequences, as in Magenta's Performance RNN.

A performance is a sequence of events from a vocabulary of 388:

    note_on     128   a pitch starts (MIDI 0-127)
    note_off    128   a pitch ends
    time_shift  100   time moves forward by 10 ms to 1 s
    velocity     32   velocity of the following notes, in 32 bins

Every event comes with a control vector describing the music that follows it:
the note density (note onsets per second in the next 3 s, in 8 classes) and
the pitch class histogram (share of time each pitch class sounds in the next
5 s).

The definitions follow magenta/note-seq (performance_lib.py,
performance_controls.py, sequences_lib.py) and the Performance RNN
configurations in magenta/models/performance_rnn/performance_model.py.
"""

import collections
import copy

import numpy as np
from pretty_midi import Instrument, Note, PrettyMIDI

# Version of the event vocabulary and of the preprocessed files; stored in
# data files and checkpoints, which are refused when it differs.
FORMAT_VERSION = 2

STEPS_PER_SECOND = 100
MAX_SHIFT_STEPS = 100
NUM_PITCHES = 128
NUM_VELOCITY_BINS = 32
MIN_VELOCITY, MAX_VELOCITY = 1, 127
VELOCITY_BIN_SIZE = -(-(MAX_VELOCITY - MIN_VELOCITY + 1) // NUM_VELOCITY_BINS)   # 4

DENSITY_WINDOW_SECONDS = 3.0
DENSITY_BIN_RANGES = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])   # onsets per second
HISTOGRAM_WINDOW_SECONDS = 5.0
HISTOGRAM_PRIOR_COUNT = 0.01

SUSTAIN_CONTROL = 64
DEFAULT_VELOCITY = 64          # before the first velocity event
MAX_NOTE_DURATION = 5.0        # seconds; notes decoded without a note_off end here
DEFAULT_PROGRAM = 0            # acoustic grand piano
DEFAULT_TEMPO = 120


def velocity_to_bin(velocity):
    """MIDI velocity 1-127 to bin 1-32."""
    velocity = min(max(int(velocity), MIN_VELOCITY), MAX_VELOCITY)
    return (velocity - MIN_VELOCITY) // VELOCITY_BIN_SIZE + 1


def bin_to_velocity(velocity_bin):
    """Bin 1-32 to the lowest MIDI velocity of the bin."""
    return MIN_VELOCITY + (velocity_bin - 1) * VELOCITY_BIN_SIZE


def seconds_to_step(seconds):
    """Nearest step of 10 ms."""
    return int(seconds * STEPS_PER_SECOND + 0.5)


# ==================================================================================
# Notes
# ==================================================================================

def apply_sustain(notes, control_changes, control=SUSTAIN_CONTROL):
    """Extend notes held by the sustain pedal (one instrument).

    A note that ends while the pedal is down lasts until the pedal is released
    or the same pitch is played again, whichever comes first.
    """
    SUSTAIN_ON, SUSTAIN_OFF, NOTE_ON, NOTE_OFF = range(4)
    events = [(n.start, NOTE_ON, n) for n in notes] + [(n.end, NOTE_OFF, n) for n in notes]
    events += [(cc.time, SUSTAIN_ON if cc.value >= 64 else SUSTAIN_OFF, None)
               for cc in control_changes if cc.number == control]
    events.sort(key=lambda e: (e[0], e[1]))

    active, sustain, removed = [], False, set()
    time = 0.0
    for time, kind, note in events:
        if kind == SUSTAIN_ON:
            sustain = True
        elif kind == SUSTAIN_OFF:
            sustain = False
            still = []
            for held in active:
                if held.end < time:
                    held.end = time
                else:
                    still.append(held)
            active = still
        elif kind == NOTE_ON:
            if sustain:
                still = []
                for held in active:
                    if held.pitch == note.pitch:
                        held.end = time
                        if held.start == held.end:
                            removed.add(id(held))
                    else:
                        still.append(held)
                active = still
            active.append(note)
        elif not sustain:
            active = [held for held in active if held is not note]
    for held in active:
        held.end = time
    return [n for n in notes if id(n) not in removed]


class NoteSeq:

    @staticmethod
    def from_midi(midi, programs=range(128), sustain=True):
        """Notes of all non-drum instruments, with the sustain pedal applied."""
        notes = []
        for inst in midi.instruments:
            if inst.is_drum or inst.program not in programs:
                continue
            inst_notes = [copy.copy(note) for note in inst.notes]
            if sustain:
                inst_notes = apply_sustain(inst_notes, inst.control_changes)
            notes.extend(inst_notes)
        return NoteSeq(notes)

    @staticmethod
    def from_midi_file(path, *args, **kwargs):
        return NoteSeq.from_midi(PrettyMIDI(str(path)), *args, **kwargs)

    def __init__(self, notes=()):
        self.notes = sorted((n for n in notes if n.end >= n.start), key=lambda n: n.start)

    def copy(self):
        return copy.deepcopy(self)

    def to_midi(self, program=DEFAULT_PROGRAM, tempo=DEFAULT_TEMPO):
        midi = PrettyMIDI(initial_tempo=tempo)
        inst = Instrument(program, False, 'Performance RNN')
        inst.notes = copy.deepcopy(self.notes)
        midi.instruments.append(inst)
        return midi

    def to_midi_file(self, path, *args, **kwargs):
        self.to_midi(*args, **kwargs).write(str(path))

    def adjust_time(self, offset):
        for note in self.notes:
            note.start += offset
            note.end += offset

    def trim_overlapped_notes(self, min_interval=0):
        """Merge notes of the same pitch that start within min_interval of each
        other; otherwise end a note where the next one of the same pitch starts."""
        last_notes = {}
        kept = []
        for note in self.notes:
            last_note = last_notes.get(note.pitch)
            if last_note is not None and note.start - last_note.start <= min_interval:
                last_note.end = max(note.end, last_note.end)
                last_note.velocity = max(note.velocity, last_note.velocity)
                continue
            if last_note is not None and note.start < last_note.end:
                last_note.end = note.start
            last_notes[note.pitch] = note
            kept.append(note)
        self.notes = kept


# ==================================================================================
# Events
# ==================================================================================

class EventSeq:

    @staticmethod
    def feat_dims():
        return collections.OrderedDict([('note_on', NUM_PITCHES), ('note_off', NUM_PITCHES),
                                        ('time_shift', MAX_SHIFT_STEPS),
                                        ('velocity', NUM_VELOCITY_BINS)])

    @staticmethod
    def feat_ranges():
        offset = 0
        feat_ranges = collections.OrderedDict()
        for name, dim in EventSeq.feat_dims().items():
            feat_ranges[name] = range(offset, offset + dim)
            offset += dim
        return feat_ranges

    @staticmethod
    def dim():
        return sum(EventSeq.feat_dims().values())

    def __init__(self, indices=()):
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def from_note_seq(note_seq):
        """Quantise to 10 ms steps and encode, as Magenta's Performance."""
        ranges = EventSeq.feat_ranges()
        notes = sorted(note_seq.notes, key=lambda n: (n.start, n.pitch))
        marks = []
        for i, note in enumerate(notes):
            start, end = seconds_to_step(note.start), seconds_to_step(note.end)
            if end == start:
                end += 1
            marks.append((start, i, False))
            marks.append((end, i, True))
        marks.sort()

        events, step, velocity_bin = [], 0, 0
        for mark_step, i, is_off in marks:
            while mark_step > step:
                shift = min(mark_step - step, MAX_SHIFT_STEPS)
                events.append(ranges['time_shift'].start + shift - 1)
                step += shift
            note = notes[i]
            if is_off:
                events.append(ranges['note_off'].start + note.pitch)
            else:
                b = velocity_to_bin(note.velocity)
                if b != velocity_bin:
                    velocity_bin = b
                    events.append(ranges['velocity'].start + b - 1)
                events.append(ranges['note_on'].start + note.pitch)
        return EventSeq(events)

    @staticmethod
    def from_array(indices):
        return EventSeq(indices)

    def to_array(self):
        return self.indices.astype(np.uint16)

    def steps(self):
        """Step of every event, and the time shift (in steps) each one adds."""
        start = EventSeq.feat_ranges()['time_shift'].start
        shift = np.where((self.indices >= start) & (self.indices < start + MAX_SHIFT_STEPS),
                         self.indices - start + 1, 0)
        after = np.cumsum(shift)
        return after - shift, shift

    def to_note_seq(self, max_note_duration=MAX_NOTE_DURATION):
        """Decode; a note_off ends the earliest open note of its pitch."""
        ranges = EventSeq.feat_ranges()
        on, off = ranges['note_on'].start, ranges['note_off'].start
        shift, vel = ranges['time_shift'].start, ranges['velocity'].start
        step, velocity = 0, DEFAULT_VELOCITY
        open_notes = collections.defaultdict(collections.deque)
        notes = []

        def close(pitch, start_step, note_velocity, end_step):
            if end_step == start_step:
                return
            start, end = start_step / STEPS_PER_SECOND, end_step / STEPS_PER_SECOND
            if max_note_duration:
                end = min(end, start + max_note_duration)
            notes.append(Note(note_velocity, pitch, start, end))

        for index in self.indices.tolist():
            if index < off:
                open_notes[index - on].append((step, velocity))
            elif index < shift:
                pitch = index - off
                if open_notes[pitch]:
                    start_step, note_velocity = open_notes[pitch].popleft()
                    close(pitch, start_step, note_velocity, step)
            elif index < vel:
                step += index - shift + 1
            else:
                velocity = bin_to_velocity(index - vel + 1)
        for pitch, pending in open_notes.items():
            for start_step, note_velocity in pending:
                close(pitch, start_step, note_velocity, step)
        return NoteSeq(notes)


# ==================================================================================
# Controls
# ==================================================================================

def density_to_bin(notes_per_second):
    """Note density (onsets per second) to its class 0-7."""
    return int(np.searchsorted(DENSITY_BIN_RANGES, notes_per_second, side='right'))


class Control:
    """One control vector: a pitch class histogram and a note density class."""

    def __init__(self, pitch_histogram, note_density):
        self.pitch_histogram = pitch_histogram  # 12 values
        self.note_density = note_density        # class 0-7

    def __repr__(self):
        return f'Control(pitch_histogram={list(self.pitch_histogram)}, note_density={self.note_density})'

    def to_array(self):
        density = np.zeros(ControlSeq.feat_dims()['note_density'], dtype=np.float32)
        density[self.note_density] = 1.0
        histogram = np.asarray(self.pitch_histogram, dtype=np.float32)
        return np.concatenate([density, histogram])


class ControlSeq:

    @staticmethod
    def feat_dims():
        return collections.OrderedDict([('note_density', len(DENSITY_BIN_RANGES) + 1),
                                        ('pitch_histogram', 12)])

    @staticmethod
    def feat_ranges():
        offset = 0
        feat_ranges = collections.OrderedDict()
        for name, dim in ControlSeq.feat_dims().items():
            feat_ranges[name] = range(offset, offset + dim)
            offset += dim
        return feat_ranges

    @staticmethod
    def dim():
        return sum(ControlSeq.feat_dims().values())

    def __init__(self, density, histogram):
        self.density = np.asarray(density, dtype=np.int64)       # [n] class 0-7
        self.histogram = np.asarray(histogram, dtype=np.float64)  # [n, 12], rows sum to 1

    def __len__(self):
        return len(self.density)

    @staticmethod
    def from_event_seq(event_seq):
        """Controls of every event, computed as Magenta's control signals.

        The values are computed at the first event and after every time shift,
        over a window that starts there; the other events repeat them.
        """
        indices = event_seq.indices
        n = len(indices)
        if n == 0:
            return ControlSeq(np.zeros(0), np.zeros((0, 12)))
        before, shift = event_seq.steps()
        after = before + shift
        total = int(after[-1])

        recompute = np.ones(n, dtype=bool)
        recompute[1:] = shift[:-1] > 0
        starts = np.flatnonzero(recompute)
        t0 = before[starts]

        # note density: onsets in the window, per second of window
        window = round(DENSITY_WINDOW_SECONDS * STEPS_PER_SECOND)
        is_on = indices < EventSeq.feat_ranges()['note_off'].start
        onsets = np.concatenate([[0], np.cumsum(is_on)])
        end = np.searchsorted(before, t0 + window, side='left')
        count = onsets[end] - onsets[starts]
        span = np.minimum(window, total - t0)
        density = np.where(span > 0, count * STEPS_PER_SECOND / np.maximum(span, 1), 0.0)
        density_class = np.searchsorted(DENSITY_BIN_RANGES, density, side='right')

        # pitch class histogram: time each pitch class sounds in the window
        sounding = np.zeros((total + 1, 12))
        active = set()
        on_start, off_start = EventSeq.feat_ranges()['note_on'].start, EventSeq.feat_ranges()['note_off'].start
        for k, index in enumerate(indices.tolist()):
            if index < off_start:
                active.add(index - on_start)
            elif index < off_start + NUM_PITCHES:
                active.discard(index - off_start)
            elif shift[k] and active:
                counts = np.bincount([p % 12 for p in active], minlength=12)
                sounding[before[k]:after[k]] += counts
        cumulative = np.concatenate([np.zeros((1, 12)), np.cumsum(sounding, axis=0)])
        window = round(HISTOGRAM_WINDOW_SECONDS * STEPS_PER_SECOND)
        end = np.searchsorted(before, t0 + window, side='left')
        span_end = after[end - 1]
        histogram = HISTOGRAM_PRIOR_COUNT + (cumulative[span_end] - cumulative[t0]) / STEPS_PER_SECOND
        histogram /= histogram.sum(axis=1, keepdims=True)

        source = np.maximum.accumulate(np.where(recompute, np.arange(n), 0))
        position = np.searchsorted(starts, source)
        return ControlSeq(density_class[position], histogram[position])

    def to_compressed_arrays(self):
        """Density class and histogram (0-255) as uint8, for the data files."""
        return (self.density.astype(np.uint8),
                np.round(self.histogram * 255).astype(np.uint8))

    @staticmethod
    def recover_compressed_arrays(density, histogram):
        """Control vectors [n, 20] from the arrays of to_compressed_arrays."""
        one_hot = np.zeros((len(density), ControlSeq.feat_dims()['note_density']), dtype=np.float32)
        one_hot[np.arange(len(density)), density] = 1.0
        return np.concatenate([one_hot, histogram.astype(np.float32) / 255], axis=1)

    def to_array(self):
        return ControlSeq.recover_compressed_arrays(self.density, np.round(self.histogram * 255))


# ==================================================================================
# Files
# ==================================================================================

def save_processed(path, event_seq, control_seq):
    """Write one preprocessed performance (.npz)."""
    density, histogram = control_seq.to_compressed_arrays()
    np.savez_compressed(path, format_version=FORMAT_VERSION, events=event_seq.to_array(),
                        density=density, histogram=histogram)


def load_processed(path, compressed=False):
    """Events [n] and control vectors [n, 20] of a preprocessed file.

    With compressed=True: events, density classes and histograms (uint8) as stored.
    """
    with np.load(path) as data:
        version = int(data['format_version']) if 'format_version' in data else None
        if version != FORMAT_VERSION:
            raise ValueError(f'{path}: format version {version}, expected {FORMAT_VERSION}; '
                             'run `prnn preprocess` again')
        events, density, histogram = data['events'], data['density'], data['histogram']
    if compressed:
        return events, density, histogram
    return events.astype(np.int64), ControlSeq.recover_compressed_arrays(density, histogram)
