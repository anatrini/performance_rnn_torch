"""Tests for the bugs fixed in phase 3 (numbers as in the development notes)."""

import argparse
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from pretty_midi import Instrument, Note, PrettyMIDI
from torch import nn, optim

from performance_rnn_torch.config import GenerateConfig, ModelConfig, TrainConfig
from performance_rnn_torch.core import Dataset, PerformanceRNN
from performance_rnn_torch.core.sequence import ControlSeq, EventSeq
from performance_rnn_torch.training import batch_loss, evaluate, last_session_path, load_model, train_model
from performance_rnn_torch.utils import helpers

CPU = torch.device('cpu')
ROOT = Path(__file__).parent.parent


def small_model(seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(init_dim=4, hidden_dim=16, gru_layers=2, gru_dropout=0.5)
    return PerformanceRNN(**vars(cfg)), cfg


def train_config(**kwargs):
    defaults = dict(
        batch_size=4,
        num_epochs=3,
        window_size=20,
        stride_size=10,
        early_stopping_patience=0,
        enable_logging=False,
        saving_interval=1e9,
    )
    return TrainConfig(**{**defaults, **kwargs})


def random_samples(n, length=200, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        events = rng.integers(0, EventSeq.dim(), length)
        controls = rng.random((length, ControlSeq.dim()))
        samples.append((events, controls))
    return samples


# 1 -------------------------------------------------------------------------


def test_transposition_rotates_the_pitch_histogram_and_keeps_the_density():
    ranges = ControlSeq.feat_ranges()
    hist, dens = ranges['pitch_histogram'], ranges['note_density']
    controls = np.zeros((1, 1, ControlSeq.dim()))
    controls[0, 0, dens.start + 3] = 1.0  # density bin 3
    controls[0, 0, hist.start + 0] = 1.0  # all notes are C
    events = np.array([[EventSeq.feat_ranges()['note_on'].start + 60]])  # C4

    _, transposed = helpers.transposition(events, controls, 7)
    np.testing.assert_array_equal(transposed[0, 0, dens], controls[0, 0, dens])
    assert transposed[0, 0, hist.start + 7] == 1.0  # now all notes are G


def test_control_layout_matches_the_arrays():
    ranges = ControlSeq.feat_ranges()
    array = ControlSeq.recover_compressed_arrays(np.array([5]), np.zeros((1, 12), np.uint8))
    assert array[0, ranges['note_density'].start + 5] == 1.0


# 2 -------------------------------------------------------------------------


def test_optimize_minimises_the_validation_loss_and_keeps_only_the_best_trial(tmp_path, monkeypatch):
    optuna = pytest.importorskip('optuna')
    from performance_rnn_torch.cli import optimize

    losses = iter([3.0, 1.0, 2.0])

    def fake_train_model(model, optimizer, dataset, cfg, model_config, sess_path, device):
        sess_path = Path(sess_path)
        sess_path.parent.mkdir(parents=True, exist_ok=True)
        sess_path.write_bytes(b'x')
        last_session_path(sess_path).write_bytes(b'x')
        return next(losses)

    monkeypatch.setattr(optimize, 'train_model', fake_train_model)
    _, cfg = small_model()
    sess = tmp_path / 'optimization.sess'
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: optimize.objective(t, sess, cfg, TrainConfig(), CPU, None), n_trials=3)

    assert study.best_value == 1.0
    kept = sorted(p.name for p in (tmp_path / 'optimization').iterdir())
    assert kept == ['trial-1-last.sess', 'trial-1.sess']


# 3 -------------------------------------------------------------------------


def test_evaluate_is_deterministic_and_leaves_no_gradients():
    model, cfg = small_model()
    data = Dataset.from_samples(random_samples(2))
    cfg = train_config(stride_size=20)

    first = evaluate(model, data, cfg, CPU)
    second = evaluate(model, data, cfg, CPU)

    assert first == second  # dropout off, same initial states
    assert all(p.grad is None for p in model.parameters())
    assert model.training  # training mode restored


# 4 -------------------------------------------------------------------------


def test_loss_counts_every_event_including_index_0():
    model, cfg = small_model()
    model.eval()
    events = torch.zeros(10, 2, dtype=torch.long)  # only event 0
    init = torch.zeros(2, cfg.init_dim)
    loss = batch_loss(model, events, None, init)
    logits = model(events, None, init)
    expected = nn.functional.cross_entropy(logits.reshape(-1, cfg.event_dim), events.reshape(-1))
    assert torch.isclose(loss, expected)


# 5, 14 ---------------------------------------------------------------------


def test_split_keeps_at_least_one_file_on_each_side():
    train, test = Dataset.from_samples(random_samples(3)).train_test_split(0.3)
    assert (len(train.samples), len(test.samples)) == (2, 1)


def test_split_needs_two_files():
    with pytest.raises(ValueError):
        Dataset.from_samples(random_samples(1)).train_test_split(0.3)


def test_split_is_shuffled_and_reproducible():
    samples = random_samples(10)
    data = Dataset.from_samples(samples)
    _, test_a = data.train_test_split(0.3, seed=1)
    _, test_b = data.train_test_split(0.3, seed=1)
    ids = lambda d: [next(i for i, s in enumerate(samples) if np.array_equal(s[0], e)) for e, _, _ in d.samples]
    assert ids(test_a) == ids(test_b)
    assert ids(test_a) != [7, 8, 9]  # not simply the last files


# 6 -------------------------------------------------------------------------


@pytest.mark.parametrize('drop_last', [True, False])
@pytest.mark.parametrize('batch, window, stride', [(4, 20, 10), (3, 50, 7), (8, 199, 1)])
def test_get_length_matches_the_batches(batch, window, stride, drop_last):
    data = Dataset.from_samples(random_samples(3))
    batches = list(data.batches(batch, window, stride, drop_last=drop_last))
    assert data.get_length(batch, window, stride, drop_last) == len(batches)
    assert all(events.shape[0] == window and events.shape[1] <= batch for events, _ in batches)


# 7, 15 ---------------------------------------------------------------------


def test_training_saves_the_best_model_and_the_last_state(tmp_path):
    model, cfg = small_model()
    data = Dataset.from_samples(random_samples(4, length=300))
    sess = tmp_path / 'nested' / 'model.sess'  # folder does not exist yet
    tc = train_config()
    best = train_model(model, optim.Adam(model.parameters(), 1e-2), data, tc, cfg, sess, CPU)

    assert sess.exists() and last_session_path(sess).exists()
    saved, saved_config, state = load_model(sess)
    assert saved_config == cfg and state['best_val_loss'] == best
    _, test = data.train_test_split(tc.train_test_ratio, seed=tc.seed)
    assert math.isclose(evaluate(saved, test, tc, CPU), best, rel_tol=1e-6)


# 8 -------------------------------------------------------------------------


def test_tensorboard_steps_keep_counting_across_epochs(tmp_path, monkeypatch):
    tensorboard = pytest.importorskip('torch.utils.tensorboard')
    steps = []

    class FakeWriter:
        def add_scalar(self, tag, value, step):
            if tag == 'model/train_loss':
                steps.append(step)

        def close(self):
            pass

    monkeypatch.setattr(tensorboard, 'SummaryWriter', FakeWriter)
    model, cfg = small_model()
    data = Dataset.from_samples(random_samples(3, length=100))
    tc = train_config(batch_size=2, num_epochs=2, stride_size=20, enable_logging=True)
    train_model(model, optim.Adam(model.parameters()), data, tc, cfg, tmp_path / 'm.sess', CPU)
    assert steps == list(range(len(steps))) and len(steps) > 4


# 9 -------------------------------------------------------------------------


def test_generation_samples_by_default():
    assert GenerateConfig().greedy_ratio == 0.0


# 11 ------------------------------------------------------------------------


def test_preprocess_skips_empty_files(tmp_path):
    from performance_rnn_torch.cli.preprocess import preprocess_midi_files_under

    midi_root = tmp_path / 'midi'
    midi_root.mkdir()
    PrettyMIDI().write(str(midi_root / 'empty.mid'))
    full = PrettyMIDI()
    piano = Instrument(0)
    piano.notes = [Note(80, 60 + i, i * 0.25, i * 0.25 + 0.2) for i in range(8)]
    full.instruments.append(piano)
    full.write(str(midi_root / 'full.mid'))

    saved = preprocess_midi_files_under(midi_root, tmp_path / 'processed', 1)
    assert saved == 1
    assert len(list((tmp_path / 'processed').rglob('*.npz'))) == 1


# 12 ------------------------------------------------------------------------


def test_licence_keeps_the_original_notice():
    assert 'Copyright (c) 2018 Yuankui Lee' in (ROOT / 'LICENSE').read_text()


# 16 ------------------------------------------------------------------------


def test_model_parameters_from_the_command_line_are_applied():
    cfg = ModelConfig().with_params('hidden_dim=64, gru_dropout=0.1')
    assert (cfg.hidden_dim, cfg.gru_dropout, cfg.event_dim) == (64, 0.1, EventSeq.dim())
    with pytest.raises(ValueError, match='unknown model parameter'):
        ModelConfig().with_params('hiden_dim=64')


# 17 ------------------------------------------------------------------------


def test_tensorboard_logging_can_be_turned_off():
    from performance_rnn_torch.cli import train

    parser = argparse.ArgumentParser()
    train.add_arguments(parser)
    assert parser.parse_args([]).enable_logging is True
    assert parser.parse_args(['--no-logging']).enable_logging is False
