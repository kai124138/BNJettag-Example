"""Gate G23 (PREFLIGHT-PLAN section 3 row G23; STUDY v0.10 section 13 "Binary-layer gates": binary edge cases;
module M6 cbl/binary.py with M5 cbl/model.py). PC CPU and PC GPU (RTX 4060 Ti), venv of record.

Constructed standalone layers in Chang's scopes: Sign(0) = -1 (entries equal to alpha get S = -1);
beta = 0 (K = 0, rounded-to-zero route, H1 = H2 = 0, NR > 0); negative and zero s_c in FB and BF (signs
carried, zero columns, K equal to STUDY section 2.3's formula; BF flips the column); q_g pruned (b = 0) and
rounded to zero (b > 0, |g| below half a step) in both forms and at a non-BN site, each with H1, H2 and NR
from cbl/cost_pure.py on the layer's SiteInput and the penalty's kernel term under both branches, as STUDY
section 2.2 states them; a saturated column (beta |s_c| above 2^i - 2^-f), where BF's direct gamma gradient
is 0 and FB's is not; FB columns single-magnitude.

One child, `gates/binary_child.py g23`, in a fresh process under the open-file audit (gates/binarygate.py);
on the GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true (decision B5-1). Constructed tensors only. Every
reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g23_edge_cases.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
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
    sys.exit(binarygate.main('G23', 'g23', __doc__))
