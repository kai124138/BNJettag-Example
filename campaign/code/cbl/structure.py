"""M10: the structure record of STUDY sections 2.2 and 8.1: degenerate layers by effect with their route, zero
columns with theirs, zero-width input fractions, per-head softmax zero-width fractions, and for a degenerate Q
whether Q_t is identical across t and whether Q is 0 at every element (arbiter section 6 item 4).

STUDY means `review/snapshots/v0.10/STUDY.md` (sha256 7180c24f6b55807e...); "plan" means PREFLIGHT-PLAN.md
(sha256 2f1be9f3...30ff8), section 2 row M10 and section 3 rows G38-G39. Installed means hgq2 0.1.9 in
`~/lab/venvs/cbl1016-jax2`. Owner decisions B8-n are in `preflight/gates/G37/<sha12>/PREFLIGHT-rows.md`.
Nothing here is quotable.

Degenerate, by effect (STUDY 2.2): Q, K, V, O or the FFN contraction, binary or HGQ, whose output after its
output quantizer where it has one (Q, K and V have one, mha.py:162,181,200; O and the FFN contraction have
none under Chang's scopes) has zero variance over the validation jets at every element. The test is exact:
every jet's output equals the first jet's at every element, in float32, the layers called in inference mode
through the U4 re-expression (cbl/ablate.py); a NaN never counts as constant (owner decision B8-7).
The route of a degenerate layer is the first of these that holds, in the STUDY's order, and every one that
holds is listed beside it (owner decision B8-6):
  kernel_pruned           every kernel width is 0 (q_g's b at a binary site, kq's b at an HGQ site);
  kernel_rounded_to_zero  not every width is 0, and every exported constant (qkernel, moving moments) is 0;
  input_zero_width        the input quantizer has zero width at every position and channel: k + i + f <= 0,
                          the inference mask of fixed_point_quantizer.py:148-149 (STUDY 2.2);
  output_zero_width       the output quantizer has zero width at every element (same mask);
  other                   none of these.
Zero width is the inference mask, as the STUDY defines it; zero bits (`bits_` = 0, which for a KIF WRAP
quantizer is relu(i + f) = 0) differs from it where i + f = 0 and k = 1 (a sign-only lane holding 0 or -2^i).
Both are recorded; routes read the mask (owner decision B8-6).
Zero columns (STUDY 8.1): at every constant-kernel site (embedding, Q, K, V, O, FFN expansion and
contraction, the three head layers, the classifier), a column (the kernel's output axes: (head, dim) for Q,
K and V, the output feature elsewhere) whose exported constants are all 0; route `pruned` when every width in
the column is 0, else `rounded_to_zero` (owner decision B8-8).
Zero-width input fraction: per projection, the fraction of its inputs (contracted channels) whose input
quantizer has zero width at every position. Softmax: per head, the fraction of (query, key) entries of the
softmax output quantizer (one format per (head, query, key), mha.py:72,327-341) with zero width.

`quantizer_state(model)` needs no data pass (the per-epoch `quantizer_structure` field of interfaces section
1, STUDY section 12); `structure_record(model, X)` adds the effect test (the front sidecar's
`structure_record`, interfaces section 2, with the key names M16 reads, interfaces section 11).
Imports keras and hgq inside the functions, after `cbl.env.setup()`.
"""
from __future__ import annotations

import time

import numpy as np

SCHEMA = 'cbl1016-structure-record/1'
LAYERS = ('Q', 'K', 'V', 'O', 'FFN')
ROUTES = ('kernel_pruned', 'kernel_rounded_to_zero', 'input_zero_width', 'output_zero_width', 'other')
NAMED_ROUTES = ROUTES[:4]
KERNEL_SITES = ('embed', 'Q', 'K', 'V', 'O', 'ffn1', 'FFN', 'head1', 'head2', 'head3', 'logits')


def num(x):
    from keras import ops

    return np.asarray(ops.convert_to_numpy(x))


def _axes(layer, shape):
    from cbl import binary

    return binary._einsum_axes(layer, shape)       # (contracted input axes, position axes, kernel output axes)


def _input_shape(layer):
    return (1,) + tuple(layer._build_shapes_dict['input_shape'][1:])


def mask_zero_width(quantizer, shape):
    """Lanes of zero width under the inference mask k + i + f <= 0 (fixed_point_quantizer.py:148-149)."""
    qz = quantizer.quantizer
    k, i, f = qz.kif
    s = qz.bw_mapper.bw_to_x(k + i + f, shape)
    return num(s) <= 0


def zero_bits(quantizer, shape):
    return num(quantizer.bits_(shape)) == 0


def kernel_widths(layer):
    """Per-coefficient kernel widths: q_g's bits broadcast over the kernel (binary), kq's bits (HGQ)."""
    from keras import ops

    from cbl import ablate

    shp = tuple(ops.shape(layer.kernel))
    return num(layer.qg.bits_(shp)) if ablate.is_binary(layer) else num(layer.kq.bits_(shp))


def kernel_state(layer) -> dict:
    """Kernel routes and zero columns of one constant-kernel layer at the frozen state."""
    from cbl import ablate

    K = num(layer.qkernel)
    W = np.broadcast_to(kernel_widths(layer), K.shape)
    _, _, k_out = _axes(layer, _input_shape(layer))
    other = tuple(a for a in range(K.ndim) if a not in k_out)
    col_zero = np.all(K == 0, axis=other).ravel()
    col_pruned = np.all(W == 0, axis=other).ravel()
    pruned = bool(np.all(W == 0))
    all_zero = bool(np.all(K == 0))
    return {'kind': 'binary' if ablate.is_binary(layer) else 'hgq', 'kernel_shape': list(K.shape),
            'pruned': pruned, 'all_constants_zero': all_zero, 'rounded_to_zero': all_zero and not pruned,
            'n_columns': int(col_zero.size),
            'zero_columns': {'pruned': [int(c) for c in np.flatnonzero(col_zero & col_pruned)],
                             'rounded_to_zero': [int(c) for c in np.flatnonzero(col_zero & ~col_pruned)]},
            'zero_width_coefficients': int(np.sum(W == 0)), 'coefficients': int(W.size)}


def quantizer_io_state(layer) -> dict:
    """Input and output quantizer zero width (mask and bits) of one projection or the FFN contraction."""
    out = {}
    shp = _input_shape(layer)
    if layer.enable_iq:
        m, b = mask_zero_width(layer.iq, shp), zero_bits(layer.iq, shp)
        contracted, positions, _ = _axes(layer, shp)
        red = tuple(positions) + (0,)
        ch_m = np.all(m, axis=red).ravel()
        ch_b = np.all(b, axis=red).ravel()
        out['iq'] = {'present': True, 'zero_width_all': bool(np.all(m)), 'zero_bits_all': bool(np.all(b)),
                     'zero_width_lanes': int(m.sum()), 'zero_bits_lanes': int(b.sum()), 'lanes': int(m.size),
                     'input_zero_width_fraction': float(ch_m.mean()), 'input_zero_bits_fraction': float(ch_b.mean()),
                     'inputs': int(ch_m.size)}
    else:
        out['iq'] = {'present': False}
    if layer.enable_oq:
        oshp = (1,) + tuple(layer.full_output_shape[1:])
        m, b = mask_zero_width(layer.oq, oshp), zero_bits(layer.oq, oshp)
        out['oq'] = {'present': True, 'zero_width_all': bool(np.all(m)), 'zero_bits_all': bool(np.all(b)),
                     'zero_width_lanes': int(m.sum()), 'zero_bits_lanes': int(b.sum()), 'lanes': int(m.size)}
    else:
        out['oq'] = {'present': False}
    return out


def named_routes(kstate, io) -> list:
    """The STUDY 2.2 routes that hold from quantizer state alone, in the STUDY's order."""
    r = []
    if kstate['pruned']:
        r.append('kernel_pruned')
    if kstate['rounded_to_zero']:
        r.append('kernel_rounded_to_zero')
    if io['iq'].get('present') and io['iq']['zero_width_all']:
        r.append('input_zero_width')
    if io['oq'].get('present') and io['oq']['zero_width_all']:
        r.append('output_zero_width')
    return r


def softmax_state(g) -> dict:
    sm = g.softmax
    shp = (1,) + tuple(sm._build_shapes_dict['input_shape'][1:])          # (1, heads, query, key)
    m, b = mask_zero_width(sm.oq, shp), zero_bits(sm.oq, shp)
    heads = m.shape[1]
    return {'shape': list(shp), 'zero_width_fraction': [float(m[0, h].mean()) for h in range(heads)],
            'zero_bits_fraction': [float(b[0, h].mean()) for h in range(heads)]}


def quantizer_state(model, g=None) -> dict:
    """The record that needs no data pass (interfaces section 1 `quantizer_structure`; STUDY section 12): per
    layer of Q, K, V, O and the FFN contraction the kernel and quantizer-width routes; zero columns at every
    constant-kernel site; zero-width input fractions; per-head softmax zero-width fractions."""
    from cbl import ablate

    g = g or ablate.Graph(model)
    layers = {}
    for name in LAYERS:
        ly = g.layer(name)
        ks = kernel_state(ly)
        io = quantizer_io_state(ly)
        layers[name] = {'kind': ks['kind'], 'kernel': {k: ks[k] for k in ('pruned', 'all_constants_zero', 'rounded_to_zero',
                                                                          'zero_width_coefficients', 'coefficients')},
                        **io, 'routes': named_routes(ks, io)}
    cols = {}
    for name in KERNEL_SITES:
        ks = kernel_state(g.layer(name))
        cols[name] = {'kind': ks['kind'], 'n_columns': ks['n_columns'], **ks['zero_columns']}
    sm = softmax_state(g)
    return {'schema': SCHEMA, 'layers': layers, 'zero_columns': cols,
            'input_zero_width_fraction': {n: layers[n]['iq'].get('input_zero_width_fraction') for n in LAYERS},
            'input_zero_bits_fraction': {n: layers[n]['iq'].get('input_zero_bits_fraction') for n in LAYERS},
            'softmax_zero_width_fraction': sm['zero_width_fraction'], 'softmax_zero_bits_fraction': sm['zero_bits_fraction']}


class ZeroVariance:
    """The effect test (owner decision B8-7), fed chunk by chunk with the `zv` product of a cbl.ablate pass:
    per layer, whether every jet of the chunk equals the chunk's first jet, and that first jet. A layer is
    constant when every chunk is and every chunk's first jet equals the first chunk's."""

    def __init__(self):
        self.first, self.constant, self.n = {}, {n: True for n in LAYERS}, 0
        self.lo, self.hi = {}, {}

    def update_products(self, n_chunk, zv):
        for name in LAYERS:
            eq, row, lo, hi = zv[name]
            row, lo, hi = np.asarray(row), np.asarray(lo, np.float64), np.asarray(hi, np.float64)
            if name not in self.first:
                self.first[name] = row.copy()
                self.lo[name], self.hi[name] = lo, hi
            else:
                self.lo[name] = np.minimum(self.lo[name], lo)
                self.hi[name] = np.maximum(self.hi[name], hi)
            if self.constant[name]:
                self.constant[name] = bool(eq) and bool(np.all(row == self.first[name]))
        self.n += int(n_chunk)

    def spread(self, name):
        """The largest per-element range (max - min over the jets), float64; 0 for a constant layer."""
        return float(np.max(self.hi[name] - self.lo[name])) if name in self.hi else None


def zero_variance_pass(ex, X):
    """The effect test alone: one pass of a cbl.ablate.Exec with the `zv` and `logits` products. Returns the
    ZeroVariance and the logits."""
    zv = ZeroVariance()
    logits = []
    for sl, p in ex.passes('base', X, None, ('zv', 'logits')):
        zv.update_products(sl.stop - sl.start, p['zv'])
        logits.append(p['logits'])
    return zv, np.concatenate(logits)


def record_from(state: dict, zv: ZeroVariance, *, batch=None, seconds=None) -> dict:
    """The sidecar's `structure_record` from a quantizer state and a completed effect test."""
    degenerate, routes, anomalies = {}, {}, []
    for name in LAYERS:
        held = state['layers'][name]['routes']
        routes[name] = list(held)
        if zv.constant[name]:
            degenerate[name] = held[0] if held else 'other'
        else:
            degenerate[name] = None
            if held:
                anomalies.append(f'{name}: route(s) {held} hold from quantizer state but the output varies over the jets')
    q_ident = q_zero = None
    if degenerate['Q'] is not None:
        qc = zv.first['Q']                                       # (T, heads, key_dim): mha.py:463
        q_ident = [bool(np.all(qc[:, h, :] == qc[:1, h, :])) for h in range(qc.shape[1])]
        q_zero = bool(np.all(qc == 0))
    constant_zero = {n: (bool(np.all(zv.first[n] == 0)) if degenerate[n] is not None else None) for n in LAYERS}
    return {'schema': SCHEMA, 'label': 'validation jets, inference mode', 'n': zv.n, 'batch': batch,
            'inference_mode': True, 'degenerate': degenerate, 'routes': routes, 'constant': dict(zv.constant),
            'max_spread': {n: zv.spread(n) for n in LAYERS},
            'constant_zero': constant_zero, 'q_identical_across_t': q_ident, 'q_all_zero': q_zero,
            'zero_columns': state['zero_columns'], 'input_zero_width_fraction': state['input_zero_width_fraction'],
            'input_zero_bits_fraction': state['input_zero_bits_fraction'],
            'softmax_zero_width_fraction': state['softmax_zero_width_fraction'],
            'softmax_zero_bits_fraction': state['softmax_zero_bits_fraction'],
            'layers': state['layers'], 'anomalies': anomalies, 'seconds': seconds}


def structure_record(model, X, *, ex=None, zv=None, jit=True) -> dict:
    """quantizer_state plus the effect test on the validation jets X (float32). `ex` reuses a cbl.ablate.Exec
    of the model; `zv` a ZeroVariance that a pass of the same state has fed (cbl.ablate.BaselinePass.zv)."""
    from cbl import ablate

    t0 = time.time()
    ex = ex or ablate.Exec(ablate.Graph(model), jit=jit)
    state = quantizer_state(model, ex.g)
    if zv is None:
        zv, _ = zero_variance_pass(ex, X)
    if zv.n != len(X):
        raise ValueError(f'the effect test saw {zv.n} jets, the validation set has {len(X)}')
    return record_from(state, zv, batch=ex.batch, seconds=time.time() - t0)


__all__ = ['LAYERS', 'ROUTES', 'KERNEL_SITES', 'quantizer_state', 'structure_record', 'ZeroVariance',
           'zero_variance_pass', 'record_from', 'kernel_state', 'quantizer_io_state', 'softmax_state',
           'mask_zero_width', 'zero_bits']


def epoch_structure(state):
    """M10's no-data routes in the M15 catalogue spelling (keep the richer sidecar unchanged)."""
    out = {name: {'kernel_zero_pruned': row['kernel']['pruned'],
                  'kernel_zero_rounded': row['kernel']['rounded_to_zero'],
                  'input_all_zero_width': row['iq'].get('zero_width_all', False),
                  'output_all_zero_width': row['oq'].get('zero_width_all', False)}
           for name, row in state['layers'].items()}
    out['softmax_heads'] = {str(h): {'zero_width_frac': value}
                            for h, value in enumerate(state['softmax_zero_width_fraction'])}
    return out


def pruning_channels(entries):
    """M10: fractions at every constant-kernel site. No invented unpruned-cost counterfactual."""
    out = {}
    for e in entries:
        if e.kind not in ('hgq', 'binary'):
            continue
        ks = kernel_state(e.layer)
        row = {'zero_width_frac': ks['zero_width_coefficients'] / ks['coefficients']}
        if e.kind == 'binary':
            io = quantizer_io_state(e.layer)
            row['input_zero_width_frac'] = io['iq'].get('input_zero_width_fraction', 0.0)
        out[e.name] = row
    return out
