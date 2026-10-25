"""Real-time generation controlled over OSC (`prnn serve`).

The performer generates events slightly ahead of the clock (the lookahead)
and sends every note-on and note-off when its time comes. Control changes
apply to the events generated after them, so they are heard after about the
lookahead.

Messages received:

    /play 1|0             start or stop (stop releases every sounding note)
    /reset                stop and start the next performance from scratch
    /temperature <f>      below 1 more predictable, above 1 more surprising
    /density <f>          notes per second (converted to Magenta's 8 classes)
    /histogram <12 f>     weights of the pitch classes C to B
    /controls 1|0         generate with or without the controls

Messages sent:

    /note_on <pitch> <velocity>
    /note_off <pitch>
    /playing 1|0
"""

import heapq
import itertools
import os
import threading
import time
from collections.abc import Callable

import numpy as np
import torch

from performance_rnn_torch.core.model import PerformanceRNN
from performance_rnn_torch.core.sequence import (
    MAX_NOTE_DURATION,
    STEPS_PER_SECOND,
    Control,
    EventSeq,
    bin_to_velocity,
    density_to_bin,
)

Send = Callable[[str, list], None]


class Performer:
    """A model that plays in real time.

    tick(now) advances the performance to the time `now` (seconds, any clock);
    it is called often by the server, and directly by the tests with a
    simulated clock.
    """

    def __init__(
        self,
        model: PerformanceRNN,
        device: torch.device,
        send: Send,
        lookahead: float = 0.5,
        temperature: float = 1.0,
        density: float = 6.0,
    ):
        self.model = model.eval()
        self.device = device
        self.send = send
        self.lookahead = lookahead
        self.temperature = temperature
        self.use_controls = True
        self.density = density_to_bin(density)
        self.histogram = np.full(12, 1 / 12)
        self.playing = False
        self.lock = threading.Lock()
        ranges = EventSeq.feat_ranges()
        self._on, self._off = ranges['note_on'].start, ranges['note_off'].start
        self._shift, self._velocity = ranges['time_shift'].start, ranges['velocity'].start
        self._order = itertools.count()
        self._reset_state()

    # ------------------------------------------------------------ controls

    def set_density(self, notes_per_second: float) -> None:
        with self.lock:
            self.density = density_to_bin(max(0.0, notes_per_second))

    def set_histogram(self, weights) -> None:
        weights = np.clip(np.asarray(weights, dtype=np.float64), 0, None)
        if weights.size != 12:
            raise ValueError(f'/histogram needs 12 values, got {weights.size}')
        with self.lock:
            self.histogram = weights / weights.sum() if weights.sum() > 0 else np.full(12, 1 / 12)

    def set_temperature(self, temperature: float) -> None:
        with self.lock:
            self.temperature = max(0.05, float(temperature))

    def set_use_controls(self, on: bool) -> None:
        with self.lock:
            self.use_controls = bool(on)

    def _control(self) -> torch.Tensor | None:
        if not self.use_controls:
            return None
        vector = Control(self.histogram, self.density).to_array()
        return torch.as_tensor(vector, device=self.device)[None, None]

    # ------------------------------------------------------------ transport

    def play(self, now: float) -> None:
        with self.lock:
            if not self.playing:
                self.playing = True
                self.start = now - self.time  # continue from where it stopped
                self.send('/playing', [1])

    def stop(self) -> None:
        with self.lock:
            self._stop()

    def reset(self) -> None:
        with self.lock:
            self._stop()
            self._reset_state()

    def _stop(self) -> None:
        self.pending.clear()
        for pitch in list(self.sounding):
            self.send('/note_off', [pitch])
        self.sounding.clear()
        if self.playing:
            self.playing = False
            self.send('/playing', [0])

    def _reset_state(self) -> None:
        with torch.no_grad():
            self.hidden = self.model.init_to_hidden(torch.zeros(1, self.model.init_dim, device=self.device))
        self.event = torch.full((1, 1), self.model.start_event, dtype=torch.long, device=self.device)
        self.time = 0.0  # performance time of the next event, in seconds
        self.start = 0.0  # clock time at which the performance time was 0
        self.velocity = 64
        self.pending = []  # heap of (clock time, order, address, arguments)
        self.sounding = {}  # pitch -> clock time of its note-on

    # ------------------------------------------------------------ the clock

    def tick(self, now: float) -> None:
        """Generate up to now + lookahead, then send the messages that are due."""
        with self.lock:
            if self.playing:
                if self.start + self.time < now - self.lookahead:
                    # generation fell behind the clock: continue from now instead of rushing
                    self.start = now - self.time
                while self.start + self.time < now + self.lookahead:
                    self._step()
            self._send_due(now)

    def next_due(self) -> float | None:
        with self.lock:
            return self.pending[0][0] if self.pending else None

    def _step(self) -> None:
        with torch.no_grad():
            logits, self.hidden = self.model.step(self.event, self._control(), self.hidden)
            self.event = PerformanceRNN.sample(logits, self.temperature)
        index = int(self.event)
        at = self.start + self.time
        if index < self._off:
            self._schedule(at, '/note_on', [index - self._on, self.velocity])
        elif index < self._shift:
            self._schedule(at, '/note_off', [index - self._off])
        elif index < self._velocity:
            self.time += (index - self._shift + 1) / STEPS_PER_SECOND
        else:
            self.velocity = bin_to_velocity(index - self._velocity + 1)

    def _schedule(self, at: float, address: str, args: list) -> None:
        heapq.heappush(self.pending, (at, next(self._order), address, args))

    def _send_due(self, now: float) -> None:
        while self.pending and self.pending[0][0] <= now:
            at, _, address, args = heapq.heappop(self.pending)
            pitch = args[0]
            if address == '/note_on':
                if pitch in self.sounding:  # the same key again: release it first
                    self.send('/note_off', [pitch])
                self.sounding[pitch] = at
                self.send(address, args)
            elif pitch in self.sounding:
                del self.sounding[pitch]
                self.send(address, args)
        for pitch, started in list(self.sounding.items()):
            if now - started >= MAX_NOTE_DURATION:  # a note-off that never came
                del self.sounding[pitch]
                self.send('/note_off', [pitch])


def serve(
    model: PerformanceRNN,
    device: torch.device,
    port: int = 9010,
    send_host: str = '127.0.0.1',
    send_port: int = 9011,
    lookahead: float = 0.5,
    temperature: float = 1.0,
    density: float = 6.0,
    log: Callable[[str], None] = print,
    on_ready: Callable[[int], None] | None = None,
    stop: threading.Event | None = None,
) -> None:
    """Run until interrupted (ctrl-C, SIGTERM, the `stop` event), or until the program
    that started it (Max, Pd) ends: a patch closed by force cannot stop it otherwise."""
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import ThreadingOSCUDPServer
    from pythonosc.udp_client import SimpleUDPClient

    client = SimpleUDPClient(send_host, send_port)
    performer = Performer(model, device, client.send_message, lookahead, temperature, density)

    def handler(function):
        def handle(address, *args):
            try:
                function(*args)
            except (TypeError, ValueError) as error:
                log(f'{address}: {error}')

        return handle

    dispatcher = Dispatcher()
    dispatcher.map('/play', handler(lambda on=1: performer.play(time.monotonic()) if on else performer.stop()))
    dispatcher.map('/reset', handler(lambda *_: performer.reset()))
    dispatcher.map('/temperature', handler(lambda t: performer.set_temperature(float(t))))
    dispatcher.map('/density', handler(lambda d: performer.set_density(float(d))))
    dispatcher.map('/histogram', handler(lambda *w: performer.set_histogram(w)))
    dispatcher.map('/controls', handler(lambda on=1: performer.set_use_controls(bool(on))))
    dispatcher.set_default_handler(lambda address, *args: log(f'unknown message {address}'))

    try:
        server = ThreadingOSCUDPServer(('127.0.0.1', port), dispatcher)
    except OSError as error:
        raise ValueError(f'port {port} is in use (is another prnn serve running?): {error}') from error
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(f'listening on port {server.server_address[1]}, notes to {send_host}:{send_port}')
    if on_ready is not None:
        on_ready(server.server_address[1])
    parent, next_check = os.getppid(), 0.0
    try:
        while stop is None or not stop.is_set():
            now = time.monotonic()
            if now >= next_check:
                if os.getppid() != parent:
                    log('the program that started the server has ended')
                    break
                next_check = now + 0.5
            performer.tick(now)
            due = performer.next_due()
            wait = 0.002 if due is None else min(0.002, max(0.0, due - time.monotonic()))
            time.sleep(wait)
    except KeyboardInterrupt:
        pass
    finally:
        performer.stop()
        server.shutdown()
        server.server_close()
        log('stopped')
