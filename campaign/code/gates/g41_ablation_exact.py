"""Gate G41 (PREFLIGHT-PLAN section 3 row G41; STUDY v0.10 sections 8.8 and 13: Delta_attn, Delta_static and
Delta_unif exactly 0 when the ablated tensor is replaced by itself; Delta_unif and Delta_static exactly 0 on a
constructed K == 0 model, the mean accumulated in float64; H_attn = log 8 on uniform rows; module M11
cbl/ablate.py). PC CPU, venv of record.

For CH-8, F000 and F111-FB: identity injections at the three points, and each ablation's replacement code path
given the tensor itself per jet, reproduce the baseline logits, correct count and overflow counts; the in-loop
attention diagnostics of a saved checkpoint equal the checkpoint audit of its file, and eager equals jit. For
F000 (K kernel 0) and F111-FB (K scale pruned): K is 0, the probabilities are constant, their float64 mean equals
each jet bytewise, and Delta_unif and Delta_static are exactly 0. Injected uniform rows give H_attn = log 8
exactly; constructed rows give log 8, log 2 and log 4 with an all-zero and a negative row left out. A timing
reading on 62,000 constructed jets. One child, `gates/b8_child.py g41`, under the open-file audit. Engineering,
not quotable. Run:

    <venv>/bin/python -I -B gates/g41_ablation_exact.py --device cpu --expect-sha12 SHA12 --work DIR \\
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
    sys.exit(b8gate.main('G41', 'g41', __doc__, devices=('cpu', 'gpu')))
