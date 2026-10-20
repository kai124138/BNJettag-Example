"""M7, model part: the counts' inputs read from the layers at the frozen state, native terms from each
layer's own `_compute_ebops`, H1, H2 and NR through `cbl.cost_pure`, the independent recompute of
penalty (b) from quantizer variables, and the stock public and private EBOPs aggregates.

STUDY means `review/snapshots/v0.10/STUDY.md` (sha256 7180c24f6b55807e...): section 3.1 (a), (a'), (b),
(c); 3.2 (width rules, per-position widths, parallelization, dtype and rounding); 3.3 (frozen
evaluation row: moving moments, current weights and quantizer parameters, no layer call in training or
tracing mode); 7.2 (aggregation); 13 (cost row). "plan" means PREFLIGHT-PLAN.md (sha256 2f1be9f3...30ff8),
section 2 row M7 and section 3 rows G26-G33. Interface notes: docs/interfaces.md sections 7 (the per-site
record) and 16 (B7). Installed means hgq2 0.1.9 in `~/lab/venvs/cbl1016-jax2`.

What it reads (no layer is called; only quantizer properties, `qkernel` in inference mode and the
layers' own `_compute_ebops`, which reads quantizer bits and shapes):
- the layers: every `QLayerBase` with `enable_ebops` in the model, walked from `model.layers` through
  `Layer._layers` with the role segment rule of `cbl.rng` (`segment_name`); a record name is the role
  prefix with its segments joined by `.` (`mha.query`; interfaces.md section 8: single path segments);
- the call shapes: `_build_shapes_dict` (what Keras's build wrapper recorded) mapped as QLayerMeta's call
  wrapper maps call arguments to `_compute_ebops` arguments (installed base.py:97-105): one shape per
  input tensor, batch axis 1; a list of input shapes (QAdd) gives several, the MHA gives
  (query_shape, value_shape). G28 checks these against counters written by real training-mode calls;
- constant-kernel sites (`QEinsumDense` and its subclasses, the binary layers of M6 included): the
  `SiteInput` of interfaces.md section 7 in canonical (P, n_in) x (n_in, n_out) form, by
  `site_input`. HGQ sites: K = `qkernel` (moving moments at fused-BN sites, einsum_dense_batchnorm.py:
  176-178), `kernel_f` = kq's f broadcast to the kernel, `kernel_bits` = `kq.bits_(kernel shape)`;
  binary sites: `cbl.binary.SCALE.record(layer)` (interfaces.md section 15). `b`, `I`, `F` from the input
  quantizer (`bits_` and `kif` broadcast to the call shape). `rest_f32` is the bias term computed with the
  layer's own expression (einsum_dense.py:73-76) in the backend, converted to a NumPy float32 scalar;
- every other counted layer (the tanh tables, QAdd, pooling, the MHA's own QK and AV counter, QSoftmax and
  its two tables): `NativeLayer` with native = the layer's `_compute_ebops(*shapes) * ebops_factor` cast to
  uint32 by the backend, as base.py:104-105 writes the counter.

What it returns (`frozen_counts`): `cost_pure.aggregate`'s H1, H2, NR, per layer and per group, and per
layer also `native` (the layer's own `_compute_ebops` at the frozen state, uint32; null at a binary site,
which is not an HGQ layer, STUDY 3.1(a')), `penalty_b` (binary sites: the same call, which there returns
penalty (b)) and `counter` (the layer's uint32 counter as the last training-mode call wrote it: the
training count's part, one optimizer step older than the frozen state; owner decision B7-3).

Groups (owner decision B7-4): `attention` holds the MHA's four projections, `ffn` the expansion and the
contraction, `head` the three hidden-head layers, `embedding` the embedding (STUDY 2.2 "Sites"); every
other layer (the classifier, the MHA's own counter, QSoftmax and its tables, the tanh tables, QAdd,
pooling) is `other`.

Penalty (b), independently (`penalty_recompute`, gates G30 and G33): from quantizer variables only, never
from a kernel tensor and never through `cbl.binary.penalty_b`: q_g's rounded b per layer or per column,
the input quantizer's rounded bits, integer and fractional bits as plain integers, `cost_pure.u_bound`
for u(t) over the inputs of nonzero width, the column test from the rounded bits (STUDY 3.1(b)), exact
integer kernel terms, then one float32 addition of the backend's bias term (owner decision B3a-3). It
returns both branches; the layer's `penalty` attribute names the one its `_compute_ebops` uses.

Aggregates (STUDY 7.2): `public_aggregate` is the stock BetaPID sensor (`BaseBetaPID.get_ebops`,
beta_pid.py:41-47, over the top-level layers' public `ebops`), `private_total` the stock FreeEBOPs sum
(ebops.py:7-15, over the flattened layers' counters); both call the installed code itself.

Keras, JAX and hgq are imported inside functions (cbl.env.setup() first); NumPy and cost_pure at import.
Nothing here is quotable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from cbl import cost_pure

GROUP_OF_TOP = {'embed': 'embedding', 'ffn1': 'ffn', 'ffn2': 'ffn', 'head1': 'head', 'head2': 'head', 'head3': 'head'}
MHA_TOP = 'mha'
MHA_PROJECTIONS = ('query', 'key', 'value', 'attention_output')
KINDS = ('hgq', 'binary', 'native')


class CostModelError(RuntimeError):
    """The model does not have the structure this module reads (shapes, equations, quantizers)."""


@dataclass(frozen=True)
class Entry:
    """One counted layer: record name, role prefix, group, kind, class name, the layer and its call shapes."""
    name: str
    role: str
    group: str
    kind: str
    cls: str
    layer: Any
    shapes: tuple


def num(x) -> np.ndarray:
    import keras

    return np.asarray(keras.ops.convert_to_numpy(x))


def _ints(name, a) -> np.ndarray:
    """An array of integer values (rounded quantizer bits held as floats) as int64; refuses anything else."""
    a = np.asarray(a)
    if a.dtype.kind == 'f':
        if not np.all(np.isfinite(a)) or not np.all(a == np.round(a)):
            raise CostModelError(f'{name}: not integer-valued')
    elif a.dtype.kind not in 'iu':
        raise CostModelError(f'{name}: dtype {a.dtype}')
    return a.astype(np.int64)


# ----------------------------------------------------------------------------------------------
# the layers and their call shapes


def ebops_shapes(layer) -> tuple:
    """The shapes QLayerMeta's call wrapper passes to `_compute_ebops` (installed base.py:97-104): one per
    input tensor, `(1,) + shape[1:]`, read from Keras's recorded build shapes."""
    d = getattr(layer, '_build_shapes_dict', None)
    if not d:
        raise CostModelError(f'{layer.name}: no recorded build shapes')
    vals = list(d.values())
    if len(vals) == 1 and isinstance(vals[0], (list, tuple)) and vals[0] and isinstance(vals[0][0], (list, tuple)):
        tensors = vals[0]                               # one list of input shapes (QAdd)
    else:
        tensors = vals                                  # one shape per argument (single input; MHA query, value)
    out = []
    for s in tensors:
        s = tuple(s)
        if len(s) < 2 or any(x is None for x in s[1:]):
            raise CostModelError(f'{layer.name}: build shape {s} is not a batched static shape')
        out.append((1,) + tuple(int(x) for x in s[1:]))
    return tuple(out)


def _is_binary(layer) -> bool:
    from cbl import binary as B

    return isinstance(layer, (B.BinaryEinsumDense, B.BinaryEinsumDenseBatchnorm))


def layer_entries(model, top_names) -> list:
    """Every counted layer of the model (QLayerBase with enable_ebops), in walk order (module docstring)."""
    from hgq.layers import QEinsumDense, QLayerBase

    from cbl import rng

    layers = list(model.layers)
    top_names = list(top_names)
    if len(top_names) != len(layers) or len(set(top_names)) != len(top_names):
        raise CostModelError(f'{len(top_names)} top-level names for {len(layers)} layers')
    out = []

    def group_of(role, kernel_site):
        if not kernel_site:
            return 'other'
        parts = role.split('/')
        if parts[0] == MHA_TOP and len(parts) == 2 and parts[1] in MHA_PROJECTIONS:
            return 'attention'
        return GROUP_OF_TOP.get(role, 'other')

    def visit(layer, role):
        if isinstance(layer, QLayerBase) and layer.enable_ebops:
            site = isinstance(layer, QEinsumDense)
            kind = ('binary' if _is_binary(layer) else 'hgq') if site else 'native'
            out.append(Entry(name=role.replace('/', '.'), role=role, group=group_of(role, site), kind=kind,
                             cls=type(layer).__name__, layer=layer, shapes=ebops_shapes(layer)))
        counts = {}
        for sub in layer._layers:
            visit(sub, role + '/' + rng.segment_name(layer.name, sub.name, type(sub).__name__, counts))

    for name, layer in zip(top_names, layers):
        visit(layer, name)
    names = [e.name for e in out]
    if len(set(names)) != len(names):
        raise CostModelError(f'record names are not unique: {names}')
    return out


# ----------------------------------------------------------------------------------------------
# native terms


def native_value(layer, shapes) -> tuple:
    """(uint32 value, float32 value) of the layer's own `_compute_ebops(*shapes) * ebops_factor`, cast to
    uint32 by the backend as base.py:104-105 writes the counter."""
    from keras import ops

    v = layer._compute_ebops(*shapes) * layer.ebops_factor
    f = num(v)
    if f.dtype != np.float32 or f.shape != ():
        raise CostModelError(f'{layer.name}: _compute_ebops gave {f.dtype} {f.shape}, not a float32 scalar')
    c = num(ops.cast(v, layer._ebops.dtype))
    return int(c), np.float32(f)


def rest_f32(layer):
    """The bias term of einsum_dense.py:73-76 as the layer computes it, in the backend; None without a bias."""
    from keras import ops

    if layer.bq is None:
        return None
    bw_bias = layer.bq.bits_(ops.shape(layer.bias))
    size = ops.cast(ops.prod(layer.full_output_shape[1:]), layer.dtype)
    a = num(ops.mean(bw_bias) * size)
    if a.dtype != np.float32 or a.shape != ():
        raise CostModelError(f'{layer.name}: bias term is {a.dtype} {a.shape}, not a float32 scalar')
    return np.float32(a)


# ----------------------------------------------------------------------------------------------
# constant-kernel sites


def contraction(layer, shape) -> tuple:
    """(n_contracted, n_in, kernel letters) of an einsum layer: the contracted axes must be the input's
    trailing axes and the kernel's leading axes, in the same order (interfaces.md section 7: the Q, K and
    V kernels (24, 2, 16) flatten to (24, 32), the O kernel (2, 16, 24) to (32, 24))."""
    in_spec, rest = layer.equation.split(',')
    k_spec = rest.split('->')[0]
    letters = in_spec.replace('...', '')
    contracted = ''.join(ch for ch in letters if ch in k_spec)
    n_c = len(contracted)
    if n_c == 0 or letters[len(letters) - n_c:] != contracted or k_spec[:n_c] != contracted:
        raise CostModelError(f'{layer.name}: equation {layer.equation} is not ...<positions><contracted>, '
                             f'<contracted><outputs>')
    n_in = int(math.prod(shape[len(shape) - n_c:]))
    return n_c, n_in, k_spec


def _input_grids(layer, shape, n_in):
    """b, I, F of the input quantizer at the call shape, as int64 (P, n_in) arrays."""
    iq = layer.iq
    qz = iq.quantizer
    _, i, f = qz.kif
    b = _ints(f'{layer.name}.iq.bits_', num(iq.bits_(shape)))
    I = _ints(f'{layer.name}.iq.i', num(qz.bw_mapper.bw_to_x(i, shape)))
    F = _ints(f'{layer.name}.iq.f', num(qz.bw_mapper.bw_to_x(f, shape)))
    if b.size % n_in:
        raise CostModelError(f'{layer.name}: input shape {shape} does not split into n_in = {n_in}')
    return b.reshape(-1, n_in), I.reshape(-1, n_in), F.reshape(-1, n_in)


def site_input(entry, record=None) -> cost_pure.SiteInput:
    """The interfaces.md section 7 record of one constant-kernel site at the frozen state. At a binary
    site `record` may pass the `SCALE.record(layer)` already read at this state."""
    layer = entry.layer
    if entry.kind not in ('hgq', 'binary') or len(entry.shapes) != 1:
        raise CostModelError(f'{entry.name}: not a single-input constant-kernel site')
    shape = entry.shapes[0]
    if not layer.enable_iq:
        raise CostModelError(f'{entry.name}: no input quantizer')
    _, n_in, _ = contraction(layer, shape)
    b, I, F = _input_grids(layer, shape, n_in)
    if entry.kind == 'binary':
        from cbl import binary as B

        rec = B.SCALE.record(layer) if record is None else record
        K = np.asarray(rec['K'])
        kf = np.broadcast_to(np.asarray(rec['f']), K.shape)
        kb = np.broadcast_to(np.asarray(rec['b']), K.shape)
    else:
        K = num(layer.qkernel)
        kq = layer.kq
        kshape = tuple(K.shape)
        kf = num(kq.quantizer.bw_mapper.bw_to_x(kq.quantizer.f, kshape))
        kb = num(kq.bits_(kshape))
    if K.dtype != np.float32:
        raise CostModelError(f'{entry.name}: exported kernel dtype {K.dtype}, not float32')
    if K.size % n_in:
        raise CostModelError(f'{entry.name}: kernel {K.shape} does not split into n_in = {n_in}')
    n_out = K.size // n_in
    return cost_pure.SiteInput(
        name=entry.name, group=entry.group, kind=entry.kind, b=b, I=I, F=F, K=K.reshape(n_in, n_out),
        kernel_f=_ints(f'{entry.name}.kernel_f', kf).reshape(n_in, n_out),
        kernel_bits=_ints(f'{entry.name}.kernel_bits', kb).reshape(n_in, n_out), rest_f32=rest_f32(layer))


# ----------------------------------------------------------------------------------------------
# the counts


def frozen_counts(entries, *, binary_records=None, with_sites=False) -> dict:
    """H1, H2 and NR of the model at its current (frozen) state, per layer and per group (module docstring).
    Raises cost_pure.SnapError or WidthBoundError rather than returning a count from off-grid constants.
    `binary_records` may pass {record name: SCALE.record(layer)} read at this state."""
    sites, natives, info = [], [], {}
    for e in entries:
        counter = int(num(e.layer._ebops))
        if e.kind == 'native':
            v, _ = native_value(e.layer, e.shapes)
            natives.append(cost_pure.NativeLayer(name=e.name, group=e.group, native=v))
            info[e.name] = {'native': v, 'counter': counter}
        else:
            sites.append(site_input(e, (binary_records or {}).get(e.name)))
            v, _ = native_value(e.layer, e.shapes)
            info[e.name] = ({'native': None, 'penalty_b': v, 'counter': counter} if e.kind == 'binary'
                            else {'native': v, 'counter': counter})
    agg = cost_pure.aggregate(sites, natives)
    layers = {}
    for e in entries:
        lay = dict(agg['layers'][e.name])
        lay.update(info[e.name])
        lay['class'] = e.cls
        layers[e.name] = lay
    groups = {}
    for g in cost_pure.GROUPS:
        members = [layers[e.name] for e in entries if e.group == g]
        groups[g] = {'H1': agg['groups'][g]['H1'], 'H2': agg['groups'][g]['H2'], 'NR': agg['groups'][g]['NR'],
                     'native': None if any(m['native'] is None for m in members) else sum(m['native'] for m in members)}
    out = {'H1': agg['H1'], 'H2': agg['H2'], 'NR': agg['NR'], 'snap_changed': agg['snap_changed'], 'layers': layers,
           'groups': groups}
    if with_sites:
        out['sites'] = agg['sites']
    return out


# ----------------------------------------------------------------------------------------------
# penalty (b), recomputed from quantizer variables (gates G30, G33)


def penalty_recompute(layer, shape=None) -> dict:
    """Penalty (b) of a binary layer under both branches, from quantizer variables only (module docstring).
    `shape` is the call shape (batch axis 1); by default the recorded build shape."""
    if not _is_binary(layer):
        raise CostModelError(f'{layer.name}: not a binary layer')
    shape = ebops_shapes(layer)[0] if shape is None else tuple(shape)
    _, n_in, _ = contraction(layer, shape)
    b, I, F = _input_grids(layer, shape, n_in)
    n_out = int(math.prod(int(d) for d in layer.kernel.shape)) // n_in        # the shape only, never the values
    qb = _ints(f'{layer.name}.qg.b', num(layer.qg.quantizer.b))
    bits = np.broadcast_to(qb, (n_out,)) if qb.ndim == 0 else qb.reshape(-1)
    if bits.shape != (n_out,):
        raise CostModelError(f'{layer.name}: q_g bits {qb.shape} against {n_out} output columns')
    P = b.shape[0]
    sum_b = int(b.sum())
    sum_u = sum(cost_pure.u_bound(I[t].tolist(), F[t].tolist(), b[t].tolist(), [True] * n_in) for t in range(P))
    bits = [int(x) for x in bits]
    terms = {'H1': sum_b * sum(bits),
             'H2': sum(0 if bc == 0 else min(sum_b * bc, sum_b + sum_u * bc) for bc in bits)}
    rest = rest_f32(layer)
    out = {'branch': layer.penalty, 'sum_b': sum_b, 'sum_u': sum_u, 'bits': bits, 'rest_f32': None if rest is None else float(rest)}
    for k, kt in terms.items():
        if kt > cost_pure.FLOAT32_EXACT_INT_MAX:
            raise cost_pure.Float32RangeError(f'{layer.name}: penalty kernel term {kt} exceeds 2**24')
        v = np.float32(kt) if rest is None else np.float32(np.float32(kt) + rest)
        out[k] = {'kernel_term': kt, 'value_f32': float(v), 'counter': cost_pure.layer_value(kt, rest)}
    out['value_f32'] = out[layer.penalty]['value_f32']
    out['counter'] = out[layer.penalty]['counter']
    return out


# ----------------------------------------------------------------------------------------------
# the stock aggregates (STUDY 7.2)


def public_aggregate(model) -> float:
    """The stock BetaPID sensor: the sum of the top-level layers' public `ebops` (beta_pid.py:41-47)."""
    from hgq.utils.sugar.beta_pid import BaseBetaPID

    return float(BaseBetaPID.get_ebops(SimpleNamespace(model=model)))


def private_total(model) -> int:
    """The stock FreeEBOPs sum of the flattened layers' counters (ebops.py:7-15), as the int it logs."""
    from hgq.utils.sugar import FreeEBOPs

    logs = {}
    FreeEBOPs.on_epoch_end(SimpleNamespace(model=model), 0, logs)
    return logs['ebops']


def counters(entries) -> dict:
    """{record name: the layer's uint32 counter} (the values FreeEBOPs sums)."""
    return {e.name: int(num(e.layer._ebops)) for e in entries}


__all__ = ['Entry', 'CostModelError', 'layer_entries', 'ebops_shapes', 'native_value', 'rest_f32', 'contraction',
           'site_input', 'frozen_counts', 'penalty_recompute', 'public_aggregate', 'private_total', 'counters',
           'GROUP_OF_TOP', 'MHA_PROJECTIONS']
