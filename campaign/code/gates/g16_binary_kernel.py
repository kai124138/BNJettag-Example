"""Gate G16 (PREFLIGHT-PLAN section 3 row G16; STUDY v0.10 section 13 "Binary-layer gates": the training-mode forward kernel holds exactly +-q_g(g) per layer or +-q_g(g_c) per column, or zero;
module M6 cbl/binary.py with M5 cbl/model.py). PC CPU and PC GPU (RTX 4060 Ti), venv of record.

For each of the 17 binary configurations B6 builds (F100; F001, F010, F011, F101, F110, F111 in FB and
BF; GR and BE in FB and BF), built by M5 at seed 1 with headline H1, one training-mode forward on a
constructed batch of 64 jets (private generator) records every call of the binary map (a spy on
`cbl.binary.SCALE.parts` that returns its result unchanged). Checks: the map runs once per binary site,
in training mode; each kernel K the forward used equals q * S exactly (BF: q * sign(s) * S), holds only
+q and -q per layer (non-BN sites) or per column (fused-BN sites), or 0; S is +-1; q >= 0 lies on q_g's
grid (q 2^f an integer, |q| <= 2^i - 2^-f, exact rationals); q_g is per layer at non-BN and per column at
fused-BN sites; and qkernel (moving moments) satisfies the same and equals the map's kernel.

One child, `gates/binary_child.py g16`, in a fresh process under the open-file audit (gates/binarygate.py);
on the GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true (decision B5-1). Constructed tensors only. Every
reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g16_binary_kernel.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
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
    sys.exit(binarygate.main('G16', 'g16', __doc__))
