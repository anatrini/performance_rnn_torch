"""`prnn serve`: play a trained model in real time, controlled over OSC (from Max or Pd)."""

import signal

from performance_rnn_torch import server
from performance_rnn_torch.config import default_device
from performance_rnn_torch.training import load_model, set_seed
from performance_rnn_torch.utils import paths

HELP = 'Play a model in real time, controlled over OSC (from Max or Pd)'
EPILOG = """
Messages received (port 9010):
  /play 1|0  /reset  /temperature <f>  /density <notes per second>
  /histogram <12 weights, C to B>  /controls 1|0
Messages sent (port 9011):
  /note_on <pitch> <velocity>  /note_off <pitch>  /playing 1|0

Example, from the repository folder:
  .venv/bin/prnn serve -s models/bach.sess
From Max or Pd, start it with the spawn object; stop or close the patch to end it.
"""


def add_arguments(parser):
    parser.add_argument(
        '-s',
        '--session',
        dest='sess_path',
        default=str(paths.get_model_path('train')),
        help='Checkpoint of the trained model (default: %(default)s)',
    )
    parser.add_argument('--port', type=int, default=9010, help='UDP port for the controls (default: %(default)s)')
    parser.add_argument('--send-host', default='127.0.0.1', help='Where the notes go (default: %(default)s)')
    parser.add_argument('--send-port', type=int, default=9011, help='UDP port for the notes (default: %(default)s)')
    parser.add_argument(
        '--lookahead',
        type=float,
        default=0.5,
        help='Seconds generated ahead of the clock; more is safer on slow computers, '
        'less makes the controls respond sooner (default: %(default)s)',
    )
    parser.add_argument(
        '-T', '--temperature', type=float, default=1.0, help='Initial temperature (default: %(default)s)'
    )
    parser.add_argument(
        '--density', type=float, default=6.0, help='Initial note density, notes per second (default: %(default)s)'
    )
    parser.add_argument('--seed', type=int, default=None, help='Random seed, for repeatable performances')
    parser.add_argument('--device', default='cpu', help='cpu, mps, cuda or auto (default: %(default)s)')


def main(args):
    if args.seed is not None:
        set_seed(args.seed)
    device = default_device(args.device)
    model, _, _ = load_model(args.sess_path, device)

    def log(text):
        print(f'prnn serve: {text}', flush=True)

    # spawn and other process managers end a program with SIGTERM: stop the notes first
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    log(f'model {args.sess_path} on {device}')
    server.serve(
        model, device, args.port, args.send_host, args.send_port, args.lookahead, args.temperature, args.density, log
    )
