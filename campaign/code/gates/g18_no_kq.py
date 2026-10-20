"""Gate G18 (PREFLIGHT-PLAN section 3 row G18; STUDY v0.10 section 13 "Binary-layer gates": no kq built and no kq regularization term at binary sites;
module M6 cbl/binary.py with M5 cbl/model.py). PC CPU and PC GPU (RTX 4060 Ti), venv of record.

For each of the 17 binary configurations and each binary site: no `_kq` attribute, no sublayer named
`*_kq`, no role under `<site>/kq/`, and `layer.kq` raises AttributeError; the regularized trainable
variables at the site are CH-8's with kq's b and i replaced by q_g's (MonoL1 1e-8 each); a training-mode
call adds as many losses as CH-8's (89). Control: the same test finds kq at CH-8's ten kernel sites.

One child, `gates/binary_child.py g18`, in a fresh process under the open-file audit (gates/binarygate.py);
on the GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true (decision B5-1). Constructed tensors only. Every
reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g18_no_kq.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
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
    sys.exit(binarygate.main('G18', 'g18', __doc__))
