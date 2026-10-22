"""Record the behaviour of the code before the rewrite, as a temporary baseline.

Run once, from the repository root, on the code as it was at commit 0332797:

    python tests/baseline/make_reference.py

It writes tests/baseline/reference.npz and model.pt. While the code is being
restructured, the tests in this folder check that nothing changes by accident.
They are removed once the new event vocabulary and the final tests replace them.
"""

from pathlib import Path

import numpy as np
import torch

import performance_rnn_torch.core.model as model_module
from performance_rnn_torch.core.model import PerformanceRNN
from performance_rnn_torch.core.sequence import ControlSeq, EventSeq, NoteSeq
from performance_rnn_torch.utils import helpers

from fixtures import MODEL_CONFIG, PIECES, model_inputs

HERE = Path(__file__).parent
CPU = torch.device("cpu")


def notes_array(note_seq):
    return np.array([[n.pitch, n.velocity, n.start, n.end] for n in note_seq.notes],
                    dtype=np.float64).reshape(-1, 4)


def encode(midi):
    # as in scripts/preprocess.py
    note_seq = NoteSeq.from_midi(midi)
    note_seq.adjust_time(-note_seq.notes[0].start)
    event_seq = EventSeq.from_note_seq(note_seq)
    control_seq = ControlSeq.from_event_seq(event_seq)
    return event_seq.to_array(), control_seq.to_compressed_array()


def logits(model, events, controls, init):
    """Outputs of every step, the true previous event given as input (teacher forcing)."""
    model.eval()
    with torch.no_grad():
        out = model.generate(torch.from_numpy(init), events.shape[0],
                             events=torch.from_numpy(events[:-1]),
                             controls=None if controls is None else torch.from_numpy(controls),
                             teacher_forcing_ratio=1.0, output_type="logit")
    return out.numpy()


def main():
    if hasattr(model_module, "device"):  # the old model read a global device chosen at import
        model_module.device = CPU
    ref = {}

    for name, make in PIECES.items():
        events, controls = encode(make())
        ref[f"{name}/events"] = events
        ref[f"{name}/controls"] = controls
        ref[f"{name}/decoded"] = notes_array(EventSeq.from_array(events).to_note_seq())
        recovered = ControlSeq.recover_compressed_array(controls)
        ref[f"{name}/recovered_controls"] = recovered
        for offset in (-5, 7):
            e, c = helpers.transposition(events[:, None], recovered[:, None], offset)
            ref[f"{name}/transposed{offset:+d}/events"] = e[:, 0]
            ref[f"{name}/transposed{offset:+d}/controls"] = c[:, 0]

    events, controls, init = model_inputs()
    torch.manual_seed(0)
    model = PerformanceRNN(**MODEL_CONFIG)
    torch.save(model.state_dict(), HERE / "model.pt")
    ref["model/logits"] = logits(model, events, controls, init)
    ref["model/logits_no_control"] = logits(model, events, None, init)

    np.savez_compressed(HERE / "reference.npz", **ref)
    print(f"{len(ref)} arrays written to {HERE / 'reference.npz'}")


if __name__ == "__main__":
    main()
