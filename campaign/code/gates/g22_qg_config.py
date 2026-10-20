"""Gate G22 (PREFLIGHT-PLAN section 3 row G22; STUDY v0.10 section 13 "Binary-layer gates": q_g's configuration equals _qkvo_kq_conf field by field and survives a serialized reload;
module M6 cbl/binary.py with M5 cbl/model.py). PC CPU and PC GPU (RTX 4060 Ti), venv of record.

The weight configuration inside Chang's six scopes is STUDY section 2.2's q_g configuration, field by
field. For each of the 17 binary configurations: at the MHA projections q_g's configuration is the MHA's
`_qkvo_kq_conf` object and equal to it field by field; elsewhere it equals the scope's weight
configuration; at every site it equals CH-8's kq configuration. A .keras save and load_model restores each
class, q_g configuration, form, penalty branch and GR flag, the link of the projections' q_g to the reloaded
`_qkvo_kq_conf`, all 244 variables and the inference logits bytewise. BinaryMultiHeadAttention.build and
_make_output_dense equal the installed QMultiHeadAttention.build and _make_output_dense by syntax tree with
QEinsumDense replaced by BinaryEinsumDense (decision U3), and differ without the replacement (control).

One child, `gates/binary_child.py g22`, in a fresh process under the open-file audit (gates/binarygate.py);
on the GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true (decision B5-1). Constructed tensors only. Every
reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g22_qg_config.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
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
    sys.exit(binarygate.main('G22', 'g22', __doc__))
