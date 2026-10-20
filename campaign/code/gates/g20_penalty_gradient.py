"""Gate G20 (PREFLIGHT-PLAN section 3 row G20; STUDY v0.10 section 13 "Binary-layer gates": the penalty gradient reaches q_g's b;
module M6 cbl/binary.py with M5 cbl/model.py). PC CPU and PC GPU (RTX 4060 Ti), venv of record.

F100, F111 (FB, BF), BE (FB, BF) and GR-FB under headline H1 and H2, beta = 2^-10 at every binary
layer. For each binary site and q_g states b in {0, 1, 4} (per column at fused-BN sites, cycled), the
gradient of beta * _compute_ebops with respect to q_g's b equals, exactly: beta * sum_b per column under H1
(sum_b the sum of the input widths, as integers); under H2 0 on a column with rounded bits 0, beta * sum_b
on the H1 branch and beta * sum_u on the sign-plus-scale branch, with sum_u from cbl/cost_pure.py's u_bound
(B3a, exact); times the number of columns at per-layer sites. End to end: in the model's training loss,
beta = 2^-10 against beta = 0 adds the penalty gradient (evaluated on the state the same forward leaves)
to every q_g b within 1e-5 of the largest magnitude, and it is nonzero.

One child, `gates/binary_child.py g20`, in a fresh process under the open-file audit (gates/binarygate.py);
on the GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true (decision B5-1). Constructed tensors only. Every
reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g20_penalty_gradient.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
        [--data-manifest PATH] [--json PATH] [--children-out DIR]
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import binarygate  # noqa: E402

if __name__ == '__main__':
    sys.exit(binarygate.main('G20', 'g20', __doc__))
