"""The prnn command, from MIDI files to generated MIDI files."""

import csv

import numpy as np
import pytest
from pretty_midi import Instrument, Note, PrettyMIDI

from performance_rnn_torch.cli import main as prnn
from performance_rnn_torch.training import load_checkpoint


def write_piece(path, seed, notes=120):
    rng = np.random.default_rng(seed)
    midi = PrettyMIDI()
    piano = Instrument(0)
    t = 0.0
    for _ in range(notes):
        duration = float(rng.uniform(0.05, 0.6))
        piano.notes.append(Note(int(rng.integers(40, 100)), int(rng.integers(48, 84)), t, t + duration))
        t += float(rng.uniform(0.05, 0.3))
    midi.instruments.append(piano)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


@pytest.fixture
def project(tmp_path, monkeypatch):
    """An empty working folder: data, models, output and logs are created here."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('PERFORMANCE_RNN_HOME', raising=False)
    return tmp_path


def test_preprocess_train_generate(project):
    for i in range(4):
        write_piece(project / 'data' / 'midi' / 'demo' / f'piece{i}.mid', i)

    assert prnn(['preprocess', '-w', '1']) is None
    assert len(list((project / 'data' / 'processed').rglob('*.npz'))) == 4

    train = 'train -e 2 -b 4 -w 40 -s 20 --no-logging --device cpu -M hidden_dim=32,gru_layers=2'
    assert prnn(train.split()) is None
    state = load_checkpoint(project / 'models' / 'train.sess')
    assert state['model_config']['hidden_dim'] == 32 and state['epoch'] >= 1
    assert (project / 'models' / 'train-last.sess').exists()
    assert list((project / 'logs').glob('*.log'))

    generate = 'generate -b 2 -l 100 --device cpu --seed 0 -c 1,0,1,0,1,1,0,1,0,1,0,1;3'
    assert prnn(generate.split()) is None
    assert sorted(p.name for p in (project / 'output').iterdir()) == ['train_001.mid', 'train_002.mid']

    control_file = next((project / 'data' / 'processed').rglob('*.npz'))
    assert prnn(['generate', '-b', '1', '-l', '0', '-c', str(control_file), '-B', '3', '--device', 'cpu']) is None
    assert (project / 'output' / 'train_003.mid').exists()


def test_prepare_copies_one_composer(project):
    maestro = project / 'maestro-v3.0.0'
    rows = [
        ('Johann Sebastian Bach', '2004/bach1.midi'),
        ('Johann Sebastian Bach', '2006/bach2.midi'),
        ('Claude Debussy', '2004/debussy.midi'),
    ]
    maestro.mkdir()
    with open(maestro / 'maestro-v3.0.0.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['canonical_composer', 'midi_filename'])
        writer.writerows(rows)
    for i, (_, name) in enumerate(rows):
        write_piece(maestro / name, i, notes=5)

    assert prnn(['prepare', '--maestro-dir', str(maestro), '--list']) == 0
    assert prnn(['prepare', '--maestro-dir', str(maestro), '--composer', 'Johann Sebastian Bach']) == 0
    copied = sorted(p.name for p in (project / 'data' / 'midi' / 'johann_sebastian_bach').iterdir())
    assert copied == ['bach1.midi', 'bach2.midi']
    assert prnn(['prepare', '--maestro-dir', str(maestro), '--composer', 'Bach']) == 1


def test_errors_are_one_line(project, capsys):
    assert prnn(['generate', '-s', 'missing.sess', '--device', 'cpu']) == 1
    assert prnn(['train', '-M', 'hiden_dim=3', '--device', 'cpu']) == 1
    err = capsys.readouterr().err.strip().splitlines()
    assert err[0].startswith('prnn generate: ') and 'missing.sess' in err[0]
    assert err[1].startswith('prnn train: unknown model parameter "hiden_dim"')


def test_help_lists_the_commands(capsys):
    with pytest.raises(SystemExit):
        prnn(['--help'])
    out = capsys.readouterr().out
    assert all(name in out for name in ('prepare', 'preprocess', 'train', 'generate', 'optimize'))
