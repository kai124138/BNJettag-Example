"""Gate G40 (PREFLIGHT-PLAN section 3 row G40; STUDY v0.10 section 8.8; decision U4: the MHA re-expression with
injection points equals the layer's own inference forward bitwise; module M11 cbl/ablate.py). PC CPU, venv of
record (a GPU run is an extra row).

For CH-8 (Chang's builder), F000, F100, F111-FB and F111-BF, warmed by two training-mode calls on constructed
jets, chunk by chunk (2790 and 210 jets): the eager re-expression equals model(x, training=False) bytewise;
mha_forward equals the MHA layer's call and its probabilities the layer's returned attention scores; the taps Q,
K, V, O and FFN equal the sublayers called directly; the jit pass (StatelessScope, variables as arguments)
equals the eager forward bytewise and follows a changed variable without recompiling. Keras's own predict is
compared as a reading. A foreign graph and a mis-shaped injection are refused. One child, `gates/b8_child.py
g40`, under the open-file audit. Constructed jets only. Engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g40_mha_reexpression.py --device cpu --expect-sha12 SHA12 --work DIR \\
        [--data-manifest PATH] [--json PATH] [--children-out DIR]
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import b8gate  # noqa: E402

if __name__ == '__main__':
    sys.exit(b8gate.main('G40', 'g40', __doc__, devices=('cpu', 'gpu')))
