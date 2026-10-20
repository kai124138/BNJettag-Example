"""Gate G33 (PREFLIGHT-PLAN section 3 row G33; STUDY v0.10 sections 3.1-3.2 and 13 cost row; module M7 cbl/cost.py
with cbl/cost_pure.py): a binary layer's _compute_ebops equals penalty (b) of its branch; under H1 its NR is the
truncated penalty; NR comes only from the evaluator (junk counters change nothing).

One child, `gates/cost_child.py g33` (its docstring states the checks), in a fresh process under the open-file
audit (gates/costgate.py); on the GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true (decision B5-1). Constructed
tensors only. Every reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g33_binary_penalty.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
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
    sys.exit(costgate.main('G33', 'g33', __doc__))
