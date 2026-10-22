"""Validation, continuing a training, logging and argument checks (fixed in phase 11)."""

import argparse
import math

import numpy as np
import pytest
import torch
from torch import optim

from performance_rnn_torch.config import MAX_TRANSPOSITION, ModelConfig, TrainConfig
from performance_rnn_torch.core import Dataset, PerformanceRNN
from performance_rnn_torch.core.sequence import ControlSeq, EventSeq
from performance_rnn_torch.training import batch_loss, evaluate, load_checkpoint, load_session, train_model

CPU = torch.device('cpu')
MODEL = ModelConfig(init_dim=4, hidden_dim=16, gru_layers=2)


def model():
    torch.manual_seed(0)
    return PerformanceRNN(**vars(MODEL))


def samples(lengths, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for n in lengths:
        controls = np.zeros((n, ControlSeq.dim()), np.float32)
        controls[np.arange(n), rng.integers(0, 8, n)] = 1
        controls[:, 8:] = 1 / 12
        out.append((rng.integers(0, EventSeq.dim(), n), controls))
    return out


def config(**kwargs):
    defaults = dict(
        batch_size=4,
        num_epochs=2,
        window_size=20,
        stride_size=10,
        early_stopping_patience=0,
        enable_logging=False,
        saving_interval=1e9,
    )
    return TrainConfig(**{**defaults, **kwargs})


def test_defaults_are_magentas():
    cfg, m = TrainConfig(), ModelConfig()
    assert (cfg.window_size, cfg.batch_size, cfg.clip_norm, cfg.learning_rate) == (512, 64, 3.0, 1e-3)
    assert cfg.use_transposition and MAX_TRANSPOSITION == 3
    assert cfg.control_ratio == 0.5  # optional conditioning: half of the examples without controls
    assert (m.hidden_dim, m.gru_layers, m.gru_dropout) == (512, 3, 0.0)
    assert PerformanceRNN(388, 20, 32, 16).gru.dropout == 0.0  # same default as ModelConfig


def test_validation_counts_every_event_once_with_and_without_controls():
    data = Dataset.from_samples(samples([50, 45]))  # 4 windows of 20: one batch of 3 and one of 1 for batch 3
    m = model().eval()
    cfg = config(batch_size=3, stride_size=1, control_ratio=0.25)

    def loss(with_controls):
        generator = torch.Generator().manual_seed(0)
        total = count = 0
        for events, controls in data.batches(3, 20, 20, shuffle=False, drop_last=False):
            events = torch.as_tensor(events)
            controls = torch.as_tensor(controls) if with_controls else None
            init = torch.randn(events.shape[1], m.init_dim, generator=generator)
            total += batch_loss(m, events, controls, init, reduction='sum').item()
            count += events.numel()
        assert count == 4 * 20  # the stride of training does not matter, the last batch is kept
        return total / count

    expected = 0.25 * loss(True) + 0.75 * loss(False)
    assert math.isclose(evaluate(m, data, cfg, CPU), expected, rel_tol=1e-6)


def test_validation_with_files_shorter_than_a_batch_still_gives_a_loss():
    data = Dataset.from_samples(samples([30]))  # one window: less than a batch
    assert math.isfinite(evaluate(model(), data, config(), CPU))


def test_continuing_a_training_keeps_the_best_model_unless_it_improves(tmp_path):
    path = tmp_path / 'm.sess'
    data = Dataset.from_samples(samples([200] * 4))
    cfg = config()
    m = model()
    train_model(m, optim.Adam(m.parameters(), 1e-3), data, cfg, MODEL, path, CPU)
    best = load_checkpoint(path)['best_val_loss']

    # pretend the saved model is much better than anything the next run can reach
    state = torch.load(path, weights_only=True)
    state['best_val_loss'] = 0.001
    torch.save(state, path)
    m, opt, model_config, previous = load_session(path, MODEL, CPU, 1e-3, train_config=cfg)
    assert previous == 0.001
    assert train_model(m, opt, data, cfg, model_config, path, CPU, previous) == 0.001
    assert load_checkpoint(path)['best_val_loss'] == 0.001  # not overwritten by a worse model
    assert best > 0.001

    # with other validation settings the old loss is not comparable
    *_, previous = load_session(path, MODEL, CPU, 1e-3, train_config=config(window_size=30))
    assert previous == math.inf


def test_training_refuses_data_that_give_no_batch():
    data = Dataset.from_samples(samples([30, 30]))
    m = model()
    with pytest.raises(ValueError, match='no batch'):
        train_model(m, optim.Adam(m.parameters()), data, config(window_size=40), MODEL, 'unused.sess', CPU)


def train_and_record(monkeypatch, tmp_path, module, name, record):
    """Train a small model; record(*args) sees every call of module.name during training."""
    original = getattr(module, name)

    def spy(*args, **kwargs):
        record(*args)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    data = Dataset.from_samples(samples([400] * 4))
    m = model()
    train_model(m, optim.Adam(m.parameters()), data, config(num_epochs=3, stride_size=2), MODEL, tmp_path / 'm', CPU)


def test_transposition_stays_within_magentas_range(monkeypatch, tmp_path):
    from performance_rnn_torch.training import trainer

    offsets = []
    record = lambda events, controls, offset: offsets.append(offset)
    train_and_record(monkeypatch, tmp_path, trainer.helpers, 'transposition', record)
    assert min(offsets) >= -MAX_TRANSPOSITION and max(offsets) <= MAX_TRANSPOSITION
    assert len(set(offsets)) > 3


def test_half_of_the_batches_are_trained_without_controls(monkeypatch, tmp_path):
    from performance_rnn_torch.training import trainer

    without = []

    def record(m, events, controls, *args):
        if m.training:  # not the validation
            without.append(controls is None)

    train_and_record(monkeypatch, tmp_path, trainer, 'batch_loss', record)
    assert 0.3 < np.mean(without) < 0.7


def test_the_log_file_has_the_validation_losses(tmp_path):
    from performance_rnn_torch.utils.logger import setup_logger

    log = tmp_path / 'train.log'
    setup_logger('training', file=True, log_path=log)
    data = Dataset.from_samples(samples([200] * 4))
    m = model()
    train_model(m, optim.Adam(m.parameters()), data, config(), MODEL, tmp_path / 'm.sess', CPU)
    text = log.read_text()
    assert 'Epoch 1: validation loss' in text and 'Training on' in text


def test_transposition_option():
    from performance_rnn_torch.cli import train

    parser = argparse.ArgumentParser()
    train.add_arguments(parser)
    assert parser.parse_args([]).use_transposition is True
    assert parser.parse_args(['-T']).use_transposition is True
    assert parser.parse_args(['--no-transposition']).use_transposition is False


@pytest.mark.parametrize(
    'option, message',
    [('-T 0', '-T must'), ('-k 400', '-k must'), ('-p 0', '-p must'), ('-B 400', '-B must'), ('-b 0', '-b must')],
)
def test_generate_checks_its_arguments(option, message):
    from performance_rnn_torch.cli import generate

    parser = argparse.ArgumentParser()
    generate.add_arguments(parser)
    with pytest.raises(ValueError, match=message):
        generate.check_arguments(parser.parse_args(option.split()), 388)


def test_star_import_of_core():
    namespace = {}
    exec('from performance_rnn_torch.core import *', namespace)
    assert 'PerformanceRNN' in namespace


def test_missing_data_folder_is_a_clear_error(tmp_path):
    from performance_rnn_torch.cli.preprocess import preprocess_midi_files_under

    with pytest.raises(FileNotFoundError):
        Dataset(str(tmp_path / 'missing'))
    with pytest.raises(FileNotFoundError):
        preprocess_midi_files_under(tmp_path / 'missing', tmp_path / 'out', 1)
    (tmp_path / 'empty').mkdir()
    with pytest.raises(ValueError, match='no MIDI files'):
        preprocess_midi_files_under(tmp_path / 'empty', tmp_path / 'out', 1)
