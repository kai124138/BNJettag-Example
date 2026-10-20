"""Gate G21 (PREFLIGHT-PLAN section 3 row G21; STUDY v0.10 section 13 "Binary-layer gates": qkernel computable at the end of build;
module M6 cbl/binary.py with M5 cbl/model.py). PC CPU and PC GPU (RTX 4060 Ti), venv of record.

The three binary classes built standalone in Chang's scopes (non-BN, FB, BF, and the binary MHA with its
four projections): build completes, and a counting stand-in of the qkernel property shows that hgq2 0.1.9's
build wrapper (installed base.py:118-140, check_collapsed_weight) evaluated qkernel once at the end of each
build; qkernel then equals the map's kernel. Controls: a binary subclass whose qkernel is not computable
until a flag set after its build fails inside the build wrapper; the same with the flag set at the end of
build builds.

One child, `gates/binary_child.py g21`, in a fresh process under the open-file audit (gates/binarygate.py);
on the GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true (decision B5-1). Constructed tensors only. Every
reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g21_qkernel_build.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
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
    sys.exit(binarygate.main('G21', 'g21', __doc__))
