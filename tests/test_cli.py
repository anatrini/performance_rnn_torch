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
        ('Frédéric Chopin', '2004/chopin.midi'),
    ]
    maestro.mkdir()
    with open(maestro / 'maestro-v3.0.0.csv', 'w', newline='', encoding='utf-8') as f:
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

    # everything except Bach, with the year folders, outside data/midi/
    assert prnn(['prepare', '--maestro-dir', str(maestro), '--all', '--exclude', 'Johann Sebastian Bach']) == 0
    everything = project / 'data' / 'midi-maestro'
    assert sorted(p.relative_to(everything).as_posix() for p in everything.rglob('*.midi')) == ['2004/chopin.midi']
    assert prnn(['prepare', '--maestro-dir', str(maestro), '--all', '--exclude', 'Bach']) == 1
    assert prnn(['prepare', '--maestro-dir', str(maestro), '--exclude', 'Frédéric Chopin']) == 1  # needs --all


def test_fine_tuning_starts_from_the_weights_of_another_model(project):
    for i in range(4):
        write_piece(project / 'data' / 'midi' / 'demo' / f'piece{i}.mid', i)
    assert prnn(['preprocess', '-w', '1']) is None
    train = ['train', '-e', '1', '-b', '4', '-w', '40', '-s', '20', '--no-logging', '--device', 'cpu']
    assert prnn([*train, '-S', 'models/base.sess', '-M', 'hidden_dim=32,gru_layers=2']) is None
    base = load_checkpoint(project / 'models' / 'base.sess')

    # zero epochs: only the starting state is saved
    assert (
        prnn([*train, '-e', '0', '-S', 'models/ft.sess', '--init-from', 'models/base.sess', '-M', 'gru_dropout=0.3'])
        is None
    )
    last = load_checkpoint(project / 'models' / 'ft-last.sess')
    assert last['model_config'] == {**base['model_config'], 'gru_dropout': 0.3}
    assert all(last['model_state'][k].equal(v) for k, v in base['model_state'].items())
    assert last['best_val_loss'] == float('inf')  # measured on other data: chosen again

    assert prnn([*train, '-S', 'models/base.sess', '--init-from', 'models/base.sess']) == 1  # exists
    assert prnn([*train, '-S', 'models/ft2.sess', '--init-from', 'models/base.sess', '-M', 'hidden_dim=64']) == 1


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
