"""Compare the current code with the baseline recorded before the rewrite.

A test that fails here means that behaviour changed. When the change is
intended (a bug fix), the test is updated in the same commit with the reason
next to it. The whole folder is removed when the new event vocabulary and the
final tests replace it.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import performance_rnn_torch.core.model as model_module
from performance_rnn_torch.core.model import PerformanceRNN
from performance_rnn_torch.core.sequence import ControlSeq, EventSeq

from fixtures import MODEL_CONFIG, PIECES, model_inputs
from make_reference import encode, logits, notes_array

HERE = Path(__file__).parent
REF = np.load(HERE / "reference.npz")
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch):
    monkeypatch.setattr(model_module, "device", CPU)


@pytest.mark.parametrize("name", PIECES)
def test_encoding(name):
    events, controls = encode(PIECES[name]())
    np.testing.assert_array_equal(events, REF[f"{name}/events"])
    np.testing.assert_array_equal(controls, REF[f"{name}/controls"])


@pytest.mark.parametrize("name", PIECES)
def test_decoding(name):
    decoded = notes_array(EventSeq.from_array(REF[f"{name}/events"]).to_note_seq())
    np.testing.assert_allclose(decoded, REF[f"{name}/decoded"], atol=1e-9)


@pytest.mark.parametrize("name", PIECES)
def test_recovered_controls(name):
    recovered = ControlSeq.recover_compressed_array(REF[f"{name}/controls"])
    np.testing.assert_allclose(recovered, REF[f"{name}/recovered_controls"])


@pytest.mark.parametrize("offset", (-5, 7))
@pytest.mark.parametrize("name", PIECES)
def test_transposition(name, offset):
    from performance_rnn_torch.utils import helpers
    events = REF[f"{name}/events"][:, None]
    controls = REF[f"{name}/recovered_controls"][:, None]
    e, c = helpers.transposition(events, controls, offset)
    np.testing.assert_array_equal(e[:, 0], REF[f"{name}/transposed{offset:+d}/events"])
    np.testing.assert_allclose(c[:, 0], REF[f"{name}/transposed{offset:+d}/controls"])


def _model():
    model = PerformanceRNN(**MODEL_CONFIG)
    model.load_state_dict(torch.load(HERE / "model.pt", map_location=CPU))
    return model


def test_model_outputs():
    events, controls, init = model_inputs()
    np.testing.assert_allclose(logits(_model(), events, controls, init),
                               REF["model/logits"], rtol=1e-5, atol=1e-5)


def test_model_outputs_without_controls():
    events, _, init = model_inputs()
    np.testing.assert_allclose(logits(_model(), events, None, init),
                               REF["model/logits_no_control"], rtol=1e-5, atol=1e-5)

