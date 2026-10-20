"""Gate G39 (PREFLIGHT-PLAN section 3 row G39; STUDY v0.10 section 8.1, structure record; arbiter v0.5 section 6
item 4: per-head softmax zero-width fractions and, for a degenerate Q, whether Q_t is identical across t and
whether Q is 0 at every element; module M10 cbl/structure.py). PC CPU, venv of record.

For F000 (HGQ Q) and F100 (binary Q): the softmax output quantizer with 16 of head 0's 64 entries at zero width
gives fractions [0.25, 0.0], head 1 at zero width gives [0.0, 1.0], the unmodified model [0.0, 0.0]; a zero Q
kernel with bias 0.3 gives a degenerate Q with Q_t identical across t in both heads and Q not all zero, equal to
the record computed from the taps; a different output-quantizer format at position 3 of head 0 makes head 0's Q_t
differ; bias 0 gives Q all zero; an unmodified Q records null. One child, `gates/b8_child.py g39`, under the
open-file audit. Constructed jets only. Engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g39_structure_record.py --device cpu --expect-sha12 SHA12 --work DIR \\
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
    sys.exit(b8gate.main('G39', 'g39', __doc__, devices=('cpu', 'gpu')))
