"""The model's whole-window computation, sampling, beam search and batching."""

import numpy as np
import pytest
import torch

import performance_rnn_torch.core.model as model_module
from performance_rnn_torch.core import Dataset, PerformanceRNN
from performance_rnn_torch.core.sequence import ControlSeq, EventSeq

E, C = EventSeq.dim(), ControlSeq.dim()


def make_model(layers=3, dropout=0.3, seed=0):
    torch.manual_seed(seed)
    return PerformanceRNN(E, C, 6, 24, gru_layers=layers, gru_dropout=dropout)


def inputs(steps=30, batch=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    events = torch.randint(0, E, (steps, batch), generator=g)
    controls = torch.rand(steps, batch, C, generator=g)
    init = torch.randn(batch, 6, generator=g)
    return events, controls, init


def step_by_step(model, events, controls, init):
    hidden = model.init_to_hidden(init)
    event = torch.full((1, events.shape[1]), model.start_event)
    outputs = []
    for t in range(events.shape[0]):
        control = None if controls is None else controls[t:t + 1]
        logits, hidden = model.step(event, control, hidden)
        outputs.append(logits)
        event = events[t:t + 1]
    return torch.cat(outputs)


@pytest.mark.parametrize('with_controls', [True, False])
def test_whole_window_equals_step_by_step(with_controls):
    model = make_model().eval()
    events, controls, init = inputs()
    controls = controls if with_controls else None
    with torch.no_grad():
        torch.testing.assert_close(model(events, controls, init), step_by_step(model, events, controls, init))


def test_scheduled_sampling_with_only_true_events_equals_the_whole_window(monkeypatch):
    model = make_model().eval()
    events, controls, init = inputs()
    monkeypatch.setattr(model_module.random, 'random', lambda: 0.0)   # always the true event
    with torch.no_grad():
        torch.testing.assert_close(model(events, controls, init, teacher_forcing_ratio=0.5),
                                   model(events, controls, init))


def test_gradients_reach_every_layer():
    model = make_model().train()
    events, controls, init = inputs()
    loss = torch.nn.functional.cross_entropy(model(events, controls, init).reshape(-1, E), events.reshape(-1))
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.named_parameters() if 'embedding' not in name)


def test_controls_of_one_step_are_repeated():
    model = make_model().eval()
    events, controls, init = inputs()
    one = controls[:1]
    with torch.no_grad():
        torch.testing.assert_close(model(events, one, init), model(events, one.expand(30, -1, -1), init))


def test_generate_returns_valid_events():
    model = make_model().eval()
    out = model.generate(torch.randn(3, 6), 50, temperature=1.2)
    assert out.shape == (50, 3)
    assert out.dtype == torch.long and int(out.min()) >= 0 and int(out.max()) < E


@pytest.mark.parametrize('kwargs', [dict(top_k=1), dict(top_p=1e-6), dict(greedy=1.0)])
def test_narrow_sampling_is_greedy(kwargs):
    model = make_model().eval()
    init = torch.randn(2, 6)
    torch.manual_seed(5)
    sampled = model.generate(init, 40, **kwargs)
    reference = greedy(model, init, 40)
    assert torch.equal(sampled, reference)


def greedy(model, init, steps):
    hidden, event, out = model.init_to_hidden(init), torch.full((1, init.shape[0]), model.start_event), []
    with torch.no_grad():
        for _ in range(steps):
            logits, hidden = model.step(event, None, hidden)
            event = logits.argmax(-1)
            out.append(event)
    return torch.cat(out)


def test_top_k_and_top_p_keep_only_likely_events():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]])).repeat(2000, 1)
    torch.manual_seed(0)
    assert set(PerformanceRNN.sample(logits, top_k=2).tolist()) == {0, 1}
    assert set(PerformanceRNN.sample(logits, top_p=0.7).tolist()) == {0, 1}   # 0.5 + 0.3 crosses 0.7
    assert set(PerformanceRNN.sample(logits).tolist()) == {0, 1, 2, 3}


def test_beam_search_of_width_one_is_greedy():
    model = make_model().eval()
    init = torch.randn(2, 6)
    assert torch.equal(model.beam_search(init, 25, 1), greedy(model, init, 25))


def test_beam_search_returns_events_without_the_start_event():
    model = make_model().eval()
    out = model.beam_search(torch.randn(2, 6), 20, 4, controls=torch.rand(1, 2, C), stochastic=True)
    assert out.shape == (20, 2) and int(out.max()) < E


def test_batches_are_the_windows_of_the_files():
    rng = np.random.default_rng(0)
    samples = []
    for n in (120, 75, 300):
        density = rng.integers(0, 8, n)
        controls = np.zeros((n, C), np.float32)
        controls[np.arange(n), density] = 1
        controls[:, 8:] = rng.integers(0, 256, (n, 12)) / 255
        samples.append((rng.integers(0, E, n), controls))
    data = Dataset.from_samples(samples)

    expected = [(i, s) for i, (e, _) in enumerate(samples) for s in range(0, len(e) - 50 + 1, 20)]
    batches = list(data.batches(4, 50, 20, shuffle=False))
    assert len(batches) == len(expected) // 4
    for b, (events, controls) in enumerate(batches):
        for k in range(4):
            i, s = expected[b * 4 + k]
            np.testing.assert_array_equal(events[:, k], samples[i][0][s:s + 50])
            np.testing.assert_allclose(controls[:, k], samples[i][1][s:s + 50], atol=1e-6)
