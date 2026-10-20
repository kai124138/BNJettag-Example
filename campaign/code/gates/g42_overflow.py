"""Gate G42 (PREFLIGHT-PLAN section 3 row G42; arbiter v0.5 section 6 item 12; STUDY v0.10 section 8.8: a
constructed overflowing case is counted at the downstream WRAP quantizers; rows with a negative entry counted;
module M11 cbl/ablate.py). PC CPU, venv of record.

A constructed KIF WRAP quantizer with six lanes of known formats (a masked lane among them) and inputs at half a
step around each bound: the counter equals an exact count in fractions, its range check equals it, and the
in-pass counter agrees. On F000: probabilities replaced by 3/8 per key with V constant overflow O's input
quantizer at exactly jets x 8 x 16 entries (exact and analytic), with 0 unablated; the branch replaced by 5
overflows the second tanh table's input quantizer at the exact count and not QAdd's; the branch replaced by 100
overflows QAdd's branch input quantizer at every entry (a reading for the STUDY C item on the named quantizer);
peaked rows under a softmax output range of [-0.5, 0.5) give rows with a negative entry equal to an exact count
from QSoftmax's unwrapped call, left out of H_attn. One child, `gates/b8_child.py g42`, under the open-file audit.
Constructed jets only. Engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g42_overflow.py --device cpu --expect-sha12 SHA12 --work DIR \\
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
    sys.exit(b8gate.main('G42', 'g42', __doc__, devices=('cpu', 'gpu')))
