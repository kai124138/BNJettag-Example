"""Gate G19 (PREFLIGHT-PLAN section 3 row G19; STUDY v0.10 section 13 "Binary-layer gates": training-mode autodiff gradients equal Chang's own layer with kq replaced by the map;
module M6 cbl/binary.py with M5 cbl/model.py). PC CPU and PC GPU (RTX 4060 Ti), venv of record.

Five sites (the FFN contraction, a Q/V projection with its output quantizer, the O projection, the FFN
expansion and a head layer), the fused-BN ones in FB and BF, each with the main map and GR: 14 cases.
Chang's own layer (QEinsumDense or QEinsumDenseBatchnorm, released code) is built in the same scopes with
its kernel quantizer replaced, before build, by a gate-written map written from STUDY sections 2.2, 2.3
and 5 (`RefMap` in gates/binary_child.py); for BF, Chang's get_fused_qkernel_and_qbias is copied with lines
167-168 replaced by the BF map, because the stock method hands kq only U = W*s (owner decision B6-8).
Both layers get the same constructed values; one training-mode stateless call each (batch moments,
moving-moment updates, the EBOPs counter) on the same 64 inputs. Checks, exact numeric equality of every
element: outputs; the gradient of sum(output * w) with respect to every trainable variable (W, gamma, the
bias, the input, bias and q_g quantizers' b, i and f) and to x; the updated non-trainable state, which
includes the EBOPs counter (the H1 penalty equals Chang's contraction with kq.bits_ replaced by q_g's);
the task gradient reaches q_g's b. GR: at non-BN sites dL/dW equals the main map's plus the analytic added
derivative (sum S dL/dK) dq_g/dg sign(W)/n within 1e-5 of the largest magnitude; at fused-BN sites GR's
dL/dW differs from the main map's while the forward is the same. Never finite differences of Sign.

One child, `gates/binary_child.py g19`, in a fresh process under the open-file audit (gates/binarygate.py);
on the GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true (decision B5-1). Constructed tensors only. Every
reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g19_gradients.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
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
    sys.exit(binarygate.main('G19', 'g19', __doc__))
