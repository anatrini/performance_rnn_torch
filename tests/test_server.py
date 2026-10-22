"""The real-time performer and the OSC server."""

import threading
import time

import pytest
import torch

from performance_rnn_torch.config import ModelConfig
from performance_rnn_torch.core import PerformanceRNN
from performance_rnn_torch.core.sequence import MAX_NOTE_DURATION
from performance_rnn_torch.server import Performer, serve

CPU = torch.device('cpu')


def small_model(seed=0):
    torch.manual_seed(seed)
    return PerformanceRNN(**vars(ModelConfig(hidden_dim=16, gru_layers=2))).eval()


class Recorder:
    """Collects (clock time, address, arguments) of every message sent."""

    def __init__(self):
        self.now = 0.0
        self.messages = []

    def __call__(self, address, args):
        self.messages.append((self.now, address, list(args)))

    def run(self, performer, until, step=0.01):
        while self.now < until:
            performer.tick(self.now)
            self.now = round(self.now + step, 6)

    def notes(self):
        return [m for m in self.messages if m[1] in ('/note_on', '/note_off')]


def test_notes_are_sent_in_order_and_paired():
    torch.manual_seed(1)
    rec = Recorder()
    performer = Performer(small_model(), CPU, rec, lookahead=0.5)
    performer.play(0.0)
    rec.run(performer, 20.0)

    notes = rec.notes()
    assert sum(a == '/note_on' for _, a, _ in notes) > 10
    sounding = set()
    for _, address, args in notes:
        if address == '/note_on':
            assert args[0] not in sounding and 1 <= args[1] <= 127
            sounding.add(args[0])
        else:
            assert args[0] in sounding  # never a note-off without its note-on
            sounding.remove(args[0])
    times = [t for t, _, _ in rec.messages]
    assert times == sorted(times)


def test_notes_leave_at_their_time():
    torch.manual_seed(2)
    rec = Recorder()
    performer = Performer(small_model(), CPU, rec, lookahead=0.5)
    scheduled = []
    schedule = performer._schedule
    performer._schedule = lambda at, address, args: (scheduled.append(at), schedule(at, address, args))
    performer.play(0.0)
    rec.run(performer, 5.0, step=0.001)

    sent = [t for t, _, _ in rec.notes()]
    assert any(a == '/note_on' for _, a, _ in rec.notes())
    # every message leaves on the first tick at or after a scheduled time: never early, at most one tick late
    assert all(any(0 <= t - at <= 0.0011 for at in scheduled) for t in sent)
    assert 5.0 <= performer.time <= 5.0 + 0.5 + 1.0  # generated ahead of the clock by about the lookahead


def test_stop_releases_every_note_and_reports():
    torch.manual_seed(3)
    rec = Recorder()
    performer = Performer(small_model(), CPU, rec)
    performer.play(0.0)
    rec.run(performer, 3.0)
    performer.stop()
    assert not performer.sounding and not performer.pending
    assert rec.messages[0][1:] == ('/playing', [1]) and rec.messages[-1][1:] == ('/playing', [0])
    on = sum(a == '/note_on' for _, a, _ in rec.notes())
    off = sum(a == '/note_off' for _, a, _ in rec.notes())
    assert on == off


def test_no_note_lasts_longer_than_the_limit():
    torch.manual_seed(4)
    rec = Recorder()
    performer = Performer(small_model(), CPU, rec)
    performer.play(0.0)
    rec.run(performer, 30.0)
    started = {}
    for t, address, args in rec.notes():
        if address == '/note_on':
            started[args[0]] = t
        else:
            assert t - started.pop(args[0]) <= MAX_NOTE_DURATION + 0.02


def test_a_late_clock_does_not_rush():
    rec = Recorder()
    performer = Performer(small_model(), CPU, rec)
    performer.play(0.0)
    performer.tick(0.0)
    performer.tick(60.0)  # one minute without ticks
    late = [t for t, a, _ in rec.notes() if a == '/note_on']
    assert performer.start + performer.time >= 60.0
    assert len(late) < 200


def test_controls():
    performer = Performer(small_model(), CPU, lambda *_: None)
    performer.set_density(10)
    assert performer.density == 4  # 8-16 notes per second
    performer.set_histogram([1] + [0] * 11)
    assert performer.histogram[0] == 1.0
    performer.set_histogram([0] * 12)
    assert performer.histogram.tolist() == [1 / 12] * 12
    with pytest.raises(ValueError):
        performer.set_histogram([1, 2, 3])
    performer.set_use_controls(False)
    assert performer._control() is None
    performer.set_temperature(0)
    assert performer.temperature == 0.05


def test_osc_round_trip():
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import ThreadingOSCUDPServer
    from pythonosc.udp_client import SimpleUDPClient

    received = []
    dispatcher = Dispatcher()
    dispatcher.set_default_handler(lambda address, *args: received.append((address, args)))
    listener = ThreadingOSCUDPServer(('127.0.0.1', 0), dispatcher)
    threading.Thread(target=listener.serve_forever, daemon=True).start()

    torch.manual_seed(5)
    ports, ready, stop = [], threading.Event(), threading.Event()
    thread = threading.Thread(
        target=serve,
        daemon=True,
        kwargs=dict(
            model=small_model(),
            device=CPU,
            port=0,
            send_port=listener.server_address[1],
            lookahead=0.2,
            log=lambda _: None,
            on_ready=lambda port: (ports.append(port), ready.set()),
            stop=stop,
        ),
    )
    thread.start()
    assert ready.wait(5)

    client = SimpleUDPClient('127.0.0.1', ports[0])
    client.send_message('/temperature', 1.1)
    client.send_message('/density', 8.0)
    client.send_message('/histogram', [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    client.send_message('/play', 1)
    time.sleep(1.5)
    stop.set()
    thread.join(5)
    listener.shutdown()

    addresses = [a for a, _ in received]
    assert addresses[0] == '/playing' and received[-1] == ('/playing', (0,))
    assert '/note_on' in addresses
    assert addresses.count('/note_on') == addresses.count('/note_off')
