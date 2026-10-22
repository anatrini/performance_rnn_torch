"""Performance RNN model implementation."""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gumbel
from tqdm import tqdm


class PerformanceRNN(nn.Module):
    """A stack of GRUs that predicts the next event of a performance.

    The input of every step is the previous event, a flag that says whether
    controls are given, and the control vector. The output layer reads the
    top GRU layer, as in Magenta. A separate start event, never predicted,
    is the input of the first step.
    """

    def __init__(self, event_dim, control_dim, init_dim, hidden_dim,
                 gru_layers=3, gru_dropout=0.3):
        super().__init__()

        self.event_dim = event_dim
        self.control_dim = control_dim
        self.init_dim = init_dim
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.start_event = event_dim

        self.inithid_fc = nn.Linear(init_dim, gru_layers * hidden_dim)
        self.event_embedding = nn.Embedding(event_dim + 1, event_dim)
        self.concat_input_fc = nn.Linear(event_dim + 1 + control_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=gru_layers,
                          dropout=gru_dropout if gru_layers > 1 else 0.0)
        self.output_fc = nn.Linear(hidden_dim, event_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.event_embedding.weight)
        nn.init.xavier_normal_(self.inithid_fc.weight)
        nn.init.zeros_(self.inithid_fc.bias)
        nn.init.xavier_normal_(self.concat_input_fc.weight)
        nn.init.xavier_normal_(self.output_fc.weight)
        nn.init.zeros_(self.output_fc.bias)

    # ------------------------------------------------------------------ pieces

    def init_to_hidden(self, init):
        """[batch, init_dim] -> initial hidden states [gru_layers, batch, hidden_dim]."""
        out = torch.tanh(self.inithid_fc(init))
        return out.view(init.shape[0], self.gru_layers, self.hidden_dim).transpose(0, 1).contiguous()

    def expand_controls(self, controls, steps):
        """[1 or steps, batch, control_dim] -> [steps, batch, control_dim]."""
        assert controls.dim() == 3 and controls.shape[2] == self.control_dim
        if controls.shape[0] > 1:
            assert controls.shape[0] >= steps
            return controls[:steps]
        return controls.expand(steps, -1, -1)

    def _inputs(self, events, controls):
        """Previous events [T, batch] and controls [T, batch, control_dim] or None."""
        steps, batch = events.shape
        x = self.event_embedding(events)
        if controls is None:
            flag = x.new_ones(steps, batch, 1)
            controls = x.new_zeros(steps, batch, self.control_dim)
        else:
            flag = x.new_zeros(steps, batch, 1)
        return F.leaky_relu(self.concat_input_fc(torch.cat([x, flag, controls], -1)), 0.1)

    def step(self, event, control, hidden):
        """One step: event [1, batch], control [1, batch, control_dim] or None."""
        out, hidden = self.gru(self._inputs(event, control), hidden)
        return self.output_fc(out), hidden

    def _start(self, batch, device):
        return torch.full((1, batch), self.start_event, dtype=torch.long, device=device)

    # ------------------------------------------------------------------ training

    def forward(self, events, controls=None, init=None, teacher_forcing_ratio=1.0):
        """Logits [T, batch, event_dim] that predict events[t] from the events before it.

        With teacher_forcing_ratio 1 the true previous events are the inputs and
        the whole window runs at once. Below 1, each step uses the true previous
        event with that probability and the model's own most likely event
        otherwise (scheduled sampling), which needs a step-by-step loop.
        """
        steps, batch = events.shape
        hidden = self.init_to_hidden(init) if init is not None else \
            events.new_zeros(self.gru_layers, batch, self.hidden_dim, dtype=torch.float32)
        if controls is not None:
            controls = self.expand_controls(controls, steps)

        if teacher_forcing_ratio >= 1.0:
            previous = torch.cat([self._start(batch, events.device), events[:-1]], 0)
            out, _ = self.gru(self._inputs(previous, controls), hidden)
            return self.output_fc(out)

        event, outputs = self._start(batch, events.device), []
        for t in range(steps):
            control = controls[t:t + 1] if controls is not None else None
            logits, hidden = self.step(event, control, hidden)
            outputs.append(logits)
            use_true = random.random() < teacher_forcing_ratio
            event = events[t:t + 1] if use_true else logits.argmax(-1).detach()
        return torch.cat(outputs, 0)

    # ------------------------------------------------------------------ generation

    @staticmethod
    def sample(logits, temperature=1.0, top_k=0, top_p=1.0):
        """Draw events from logits [..., event_dim].

        top_k keeps the k most likely events, top_p the smallest set whose
        probability reaches p; 0 and 1 turn them off.
        """
        logits = logits / temperature
        if top_k:
            kth = logits.topk(top_k, -1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, float('-inf'))
        if top_p < 1.0:
            sorted_logits, order = logits.sort(-1, descending=True)
            cumulative = sorted_logits.softmax(-1).cumsum(-1)
            remove = cumulative - sorted_logits.softmax(-1) >= top_p   # keep the event that crosses p
            sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
            logits = logits.scatter(-1, order, sorted_logits)
        probs = logits.softmax(-1)
        return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).view(probs.shape[:-1])

    @torch.no_grad()
    def generate(self, init, steps, controls=None, greedy=0.0, temperature=1.0,
                 top_k=0, top_p=1.0, verbose=False):
        """Generate events [steps, batch].

        greedy: share of steps that take the most likely event instead of sampling.
        """
        batch = init.shape[0]
        assert init.shape[1] == self.init_dim and steps > 0
        if controls is not None:
            controls = self.expand_controls(controls, steps)
        hidden = self.init_to_hidden(init)
        event = self._start(batch, init.device)

        outputs = []
        step_iter = tqdm(range(steps), desc='Generating') if verbose else range(steps)
        for t in step_iter:
            control = controls[t:t + 1] if controls is not None else None
            logits, hidden = self.step(event, control, hidden)
            if greedy and random.random() < greedy:
                event = logits.argmax(-1)
            else:
                event = self.sample(logits, temperature, top_k, top_p)
            outputs.append(event)
        return torch.cat(outputs, 0)

    @torch.no_grad()
    def beam_search(self, init, steps, beam_size, controls=None,
                    temperature=1.0, stochastic=False, verbose=False):
        """The most likely sequences [steps, batch], keeping beam_size candidates.

        stochastic adds Gumbel noise to the scores when choosing the candidates.
        """
        assert len(init.shape) == 2 and init.shape[1] == self.init_dim
        assert self.event_dim >= beam_size > 0 and steps > 0

        batch_size = init.shape[0]
        current_beam_size = 1
        device = init.device

        if controls is not None:
            controls = self.expand_controls(controls, steps)  # [steps, batch_size, control_dim]

        hidden = self.init_to_hidden(init)[:, :, None, :]       # [layers, batch, 1, hidden]
        event = self._start(batch_size, device)[:, :, None]     # [1, batch, 1]
        beam_events = event[0, :, None, :]                       # [batch, 1, 1]
        beam_log_prob = torch.zeros(batch_size, current_beam_size, device=device)

        if stochastic:
            gumbel_dist = Gumbel(torch.tensor(0., device=device), torch.tensor(1., device=device))

        step_iter = range(steps)
        if verbose:
            step_iter = tqdm(step_iter, desc=('Stochastic ' if stochastic else '') + 'Beam Search')

        for step in step_iter:
            if controls is not None:
                control = controls[step, None, :, None, :].expand(1, batch_size, current_beam_size, -1)
                control = control.reshape(1, batch_size * current_beam_size, self.control_dim)
            else:
                control = None

            event = event.reshape(1, batch_size * current_beam_size)
            hidden = hidden.reshape(self.gru_layers, batch_size * current_beam_size, self.hidden_dim)

            logits, hidden = self.step(event, control, hidden)
            hidden = hidden.view(self.gru_layers, batch_size, current_beam_size, self.hidden_dim)
            log_probs = (logits / temperature).log_softmax(-1)
            log_probs = log_probs.view(1, batch_size, current_beam_size, self.event_dim)

            expanded = (log_probs + beam_log_prob[None, :, :, None]).view(1, batch_size, -1)
            if stochastic:
                perturbed = expanded + gumbel_dist.sample(expanded.shape)
                _, top_indices = perturbed.topk(beam_size, -1)
            else:
                _, top_indices = expanded.topk(beam_size, -1)        # [1, batch, beam]

            beam_log_prob = torch.gather(expanded, -1, top_indices)[0]  # [batch, beam]
            beam_index = torch.div(top_indices, self.event_dim, rounding_mode='floor')
            event = top_indices % self.event_dim                     # [1, batch, beam]

            hidden = torch.gather(hidden, 2, beam_index[:, :, :, None].expand(self.gru_layers, -1, -1, self.hidden_dim))
            beam_events = torch.gather(beam_events, 1, beam_index[0, :, :, None].expand(-1, -1, beam_events.shape[-1]))
            beam_events = torch.cat([beam_events, event[0, :, :, None]], -1)
            current_beam_size = beam_size

        best = beam_events[torch.arange(batch_size, device=device), beam_log_prob.argmax(-1)]
        return best[:, 1:].t().contiguous()   # without the start event
