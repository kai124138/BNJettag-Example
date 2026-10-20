"""Gate G31 (PREFLIGHT-PLAN section 3 row G31; STUDY v0.10 sections 3.1-3.2 and 13 cost row; module M7 cbl/cost.py
with cbl/cost_pure.py): the stock top-level public aggregate equals the stock flattened private total on CH-8,
F000 and every binary configuration; the flattened public sum double-counts (control).

One child, `gates/cost_child.py g31` (its docstring states the checks), in a fresh process under the open-file
audit (gates/costgate.py); on the GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true (decision B5-1). Constructed
tensors only. Every reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g31_aggregates.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
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
    sys.exit(costgate.main('G31', 'g31', __doc__))
