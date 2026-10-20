"""Gate G17 (PREFLIGHT-PLAN section 3 row G17; STUDY v0.10 section 13 "Binary-layer gates": trainable variables equal CH-8's, minus kq's b and i at binary sites, plus q_g's b and i;
module M6 cbl/binary.py with M5 cbl/model.py). PC CPU and PC GPU (RTX 4060 Ti), venv of record.

For each of the 17 binary configurations: the set of trainable roles equals CH-8's (Chang's get_model,
seed 1) minus `<site>/kq/FixedPointQuantizerKBI[0]/{b,i}` plus `<site>/qg/FixedPointQuantizerKBI[0]/{b,i}`
at its binary sites; every shared trainable role has CH-8's shape and dtype; q_g's b and i have shape ()
at non-BN sites and (n_out,) at fused-BN sites, and q_g's k is a non-trainable uint8 of the same shape.
Control: CH-8's own set fails the F111 rule.

One child, `gates/binary_child.py g17`, in a fresh process under the open-file audit (gates/binarygate.py);
on the GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true (decision B5-1). Constructed tensors only. Every
reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g17_trainable.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
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
    sys.exit(binarygate.main('G17', 'g17', __doc__))
