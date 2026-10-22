"""Configuration, checkpoints, seeds and generation helpers."""

import numpy as np
import pytest
import torch

from performance_rnn_torch.config import ModelConfig, TrainConfig, default_device
from performance_rnn_torch.core import Dataset, PerformanceRNN
from performance_rnn_torch.core.sequence import ControlSeq, EventSeq
from performance_rnn_torch.generation import (
    controls_from_spec,
    model_name,
    next_file_number,
    parse_control,
    write_midi_files,
)
from performance_rnn_torch.training import load_session, set_seed, train_model

CPU = torch.device('cpu')
MODEL = ModelConfig(init_dim=4, hidden_dim=16, gru_layers=2, gru_dropout=0.2)
TRAIN = TrainConfig(
    batch_size=4,
    num_epochs=2,
    window_size=20,
    stride_size=10,
    early_stopping_patience=0,
    enable_logging=False,
    saving_interval=1e9,
)


def dataset(seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(4):
        n = 200
        controls = np.zeros((n, ControlSeq.dim()), np.float32)
        controls[np.arange(n), rng.integers(0, 8, n)] = 1
        controls[:, 8:] = 1 / 12
        samples.append((rng.integers(0, EventSeq.dim(), n), controls))
    return Dataset.from_samples(samples)


def run(path, seed):
    set_seed(seed)
    model, optimizer, cfg = load_session(path, MODEL, CPU, TRAIN.learning_rate)
    return train_model(model, optimizer, dataset(), TRAIN, cfg, path, CPU)


def test_same_seed_same_result(tmp_path):
    assert run(tmp_path / 'a.sess', 3) == run(tmp_path / 'b.sess', 3)


def test_training_continues_a_checkpoint_with_its_own_configuration(tmp_path):
    path = tmp_path / 'm.sess'
    run(path, 0)
    model, optimizer, cfg = load_session(path, ModelConfig(hidden_dim=999), CPU, 1e-3)
    assert cfg == MODEL and model.hidden_dim == 16
    assert optimizer.state_dict()['state']  # optimizer state restored


def test_default_device_names():
    assert default_device('cpu') == CPU
    assert default_device().type in ('cpu', 'mps', 'cuda')


def test_parse_control():
    control = parse_control('2,0,1,0,1,1,0,1,0,1,0,1;3')
    assert control.note_density == 3
    assert control.pitch_histogram.sum() == pytest.approx(1.0)
    assert control.pitch_histogram[0] == pytest.approx(2 / 8)
    assert parse_control(';0').pitch_histogram.tolist() == [1 / 12] * 12
    with pytest.raises(ValueError):
        parse_control('1,2,3;2')
    with pytest.raises(ValueError):
        parse_control(';8')


def test_controls_from_spec_shapes():
    controls, _ = controls_from_spec(';2', 3, CPU)
    assert controls.shape == (1, 3, ControlSeq.dim())
    assert controls_from_spec(None, 3, CPU) == (None, 'none')


def test_files_are_numbered_after_the_existing_ones(tmp_path):
    (tmp_path / 'bach_007.mid').write_bytes(b'')
    assert next_file_number(tmp_path, 'bach') == 8
    assert model_name('models/my model.sess') == 'my_model'

    model = PerformanceRNN(**vars(MODEL)).eval()
    events = model.generate(torch.randn(2, 4), 30)
    written = write_midi_files(events, tmp_path, 'bach')
    assert [p.name for p, _ in written] == ['bach_008.mid', 'bach_009.mid']
