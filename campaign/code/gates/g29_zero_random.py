"""Gate G29 (PREFLIGHT-PLAN section 3 row G29; STUDY v0.10 sections 3.1-3.2 and 13 cost row; module M7 cbl/cost.py
with cbl/cost_pure.py): zero- and random-input agreement for H and NR. Identical counts with no forward and
after inference forwards on zero, unit and large random inputs; a training-mode call assigned back changes them
(control).

One child, `gates/cost_child.py g29` (its docstring states the checks), in a fresh process under the open-file
audit (gates/costgate.py); on the GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true (decision B5-1). Constructed
tensors only. Every reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g29_zero_random.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
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
    sys.exit(costgate.main('G29', 'g29', __doc__))
