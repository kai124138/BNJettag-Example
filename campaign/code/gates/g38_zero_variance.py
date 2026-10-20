"""Gate G38 (PREFLIGHT-PLAN section 3 row G38; STUDY v0.10 sections 2.2 and 13, structure-record row: the
zero-variance detector flags each named route on constructed cases for a binary and an HGQ layer, and nothing on
a non-degenerate case; module M10 cbl/structure.py with M11's re-expression). PC CPU and PC GPU, venv of record.

For F000 (HGQ at Q, K, V, O and the FFN contraction) and F111-FB (binary there), warmed by two training-mode
calls on constructed jets: the unmodified model flags nothing; at each of Q, K, V, O and the FFN contraction a
pruned kernel (widths 0), a kernel rounded to zero (constants 0, widths nonzero) and an all-zero-width input
quantizer give exactly that route, and at Q, K and V an all-zero-width output quantizer does (V's cases also make
O constant, route other, because V's bias is 0 at initialization); O and the FFN contraction made constant by an
upstream layer give route other; the mask-versus-bits boundary (k = 1, i + f = 0) names no route; zero columns
with routes at a subset of columns (HGQ K; the binary fused-BN FFN expansion). Each case also runs eagerly and
must agree with the jit pass. One child, `gates/b8_child.py g38`, under the open-file audit (gates/b8gate.py); on
the GPU with XLA_FLAGS=--xla_gpu_deterministic_ops=true. Constructed jets only. Engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g38_zero_variance.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
        [--data-manifest PATH] [--json PATH] [--children-out DIR]
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import b8gate  # noqa: E402

if __name__ == '__main__':
    sys.exit(b8gate.main('G38', 'g38', __doc__, devices=('cpu', 'gpu')))
