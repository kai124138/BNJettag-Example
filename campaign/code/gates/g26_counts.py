"""Gate G26 (PREFLIGHT-PLAN section 3 row G26; STUDY v0.10 sections 3.1-3.2 and 13 cost row; module M7 cbl/cost.py
with cbl/cost_pure.py): H1, H2 and NR per STUDY 3.2 in the native dtype. An independent implementation equals
cbl.cost per layer, group and total; native terms are the uint32 truncation of each layer's float32
_compute_ebops; the HGQ NR kernel term is the layer's own float32 einsum (native order).

One child, `gates/cost_child.py g26` (its docstring states the checks), in a fresh process under the open-file
audit (gates/costgate.py); on the GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true (decision B5-1). Constructed
tensors only. Every reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g26_counts.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
        [--data-manifest PATH] [--json PATH] [--children-out DIR]
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import costgate  # noqa: E402

if __name__ == '__main__':
    sys.exit(costgate.main('G26', 'g26', __doc__))
