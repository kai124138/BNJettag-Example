"""Gate G24 (PREFLIGHT-PLAN section 3 row G24; STUDY v0.10 section 13 "Binary-layer gates": q_g's float32 output exact on its grid at b <= 23; a forced b = 24 fails;
module M6 cbl/binary.py with M5 cbl/model.py). PC CPU and PC GPU (RTX 4060 Ti), venv of record.

q_g as a fused-BN binary layer builds it (32 columns), at (b, i) for b in {1, 2, 4, 8, 12, 16, 20, 22, 23}
and i in {-2, 0, 3}, in training and inference mode: every output lies exactly on q_g's grid, and every
on-grid input (the extremes and odd constants near the top and middle of the range) comes back unchanged.
A forced b = 24 (set past bc = MinMax(0, 23)) changes on-grid inputs, so the check can fail. Beside it,
the comparison with exact RND of off-grid inputs, whose only deviation (inputs within one float32 ulp
below half a step) is recorded as a finding (owner decision B6-9). At b = 23 the binary kernel's q is on
its grid within half a step of g and K = +-q.

One child, `gates/binary_child.py g24`, in a fresh process under the open-file audit (gates/binarygate.py);
on the GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true (decision B5-1). Constructed tensors only. Every
reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g24_qg_exact.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
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
    sys.exit(binarygate.main('G24', 'g24', __doc__))
