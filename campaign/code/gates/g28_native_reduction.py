"""Gate G28 (PREFLIGHT-PLAN section 3 row G28; STUDY v0.10 sections 3.1-3.2 and 13 cost row; module M7 cbl/cost.py
with cbl/cost_pure.py): the counts reduce exactly to native on F000 and CH-8. NR per layer equals the counter a
jitted training-mode call writes at the same quantizer state and the eager _compute_ebops; H1 = NR when every
constant has its full width; F000 holding CH-8's variables gives CH-8's counts.

One child, `gates/cost_child.py g28` (its docstring states the checks), in a fresh process under the open-file
audit (gates/costgate.py); on the GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true (decision B5-1). Constructed
tensors only. Every reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g28_native_reduction.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
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
    sys.exit(costgate.main('G28', 'g28', __doc__))
