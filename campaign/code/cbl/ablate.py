"""M11: attention ablations, H_attn and downstream WRAP overflow counts on validation (STUDY section 8.8),
with a lab re-expression of the model's inference forward that has injection points (decision U4).

STUDY means `review/snapshots/v0.10/STUDY.md` (sha256 7180c24f6b55807e...); "plan" means PREFLIGHT-PLAN.md
(sha256 2f1be9f3...30ff8), section 2 row M11, section 3 rows G40-G42 and section 8 decision U4;
"arbiter" means `review/arbiter-v0.5.md` section 6 items 12 and 13. Installed means hgq2 0.1.9 and Keras
3.14.0 in `~/lab/venvs/cbl1016-jax2`. Owner decisions B8-n are in
`preflight/gates/G37/<sha12>/PREFLIGHT-rows.md`. Nothing here is quotable.

The re-expression (decision U4). `forward` runs the 14 top-level layers of Chang's xfm graph
(`cbl.rng.CH8_TOP`, chang/model.py:190-209) one by one in inference mode, each layer called as Keras's
Functional graph calls it (`training=False` passed to every layer that takes it, functional.py:683-697).
The MHA is replaced by `mha_forward`, which restates the installed `QMultiHeadAttention.call` and
`_compute_attention` (mha.py:431-534) step by step:
    attention mask from `_compute_attention_mask` (None: no mask is passed; checked);
    Q, K, V = the three projections (each with its output quantizer, mha.py:162,181,200);
    query * _inverse_sqrt_key_dim (1.0 after `_build_attention`, mha.py:344; the 1/sqrt(d) scale is
      QSoftmax's `input_scaler`, softmax.py:44,51), then the K.Q einsum (mha.py:517-521): the logits;
    QSoftmax called through its layer (iq disabled in the MHA, exp table, inv table, output quantizer
      applied by the QLayerMeta wrapper, base.py:107-112): the probabilities;
    dropout skipped (`if self.dropout:` with dropout 0.0, mha.py:527);
    the A.V einsum (mha.py:533): O's input; O (no output quantizer: `enable_oq=self.enable_oq`, which
      Chang's scopes leave False, mha.py:291); the MHA's own output quantizer if enabled (False here).
Injection points: `logits` (the softmax input), `probs` (after the softmax output quantizer, before A.V,
STUDY 8.8), `branch` (the MHA output, the attention branch of model.py:195-197).

Execution (owner decision B8-13). `Exec` runs a pass of the re-expression over chunks of `VAL_BATCH` = 2790
validation jets (Chang's validation batch, run_train.py:74), either eagerly (`jit=False`, the reference) or
compiled with `jax.jit` inside Keras's `StatelessScope`, every model variable passed as an argument so that one
compilation serves every state of the model (`jit=True`, the default for the saver). A pass returns
"products" computed from the taps inside the pass (on the device): the logits, the zero-variance test of
cbl/structure.py, the probabilities and the branch (for the float64 means, H_attn and the branch norm, taken
to the host), and the overflow counts. Gate G40 checks both modes against the layer's own forward
(`mha(xa, xa, training=False)`, `model(x, training=False)`, `model.predict`) bitwise.

Ablations (STUDY 8.8), each a top-1 difference on the validation jets, baseline minus ablated, from the
same Exec and the same batching; the exact correct counts are kept beside the float:
- delta_attn_mean: the branch replaced by its validation mean per (position, channel), accumulated in
  float64 in NumPy and cast back to float32 (B8-11; the STUDY states the float64 rule for Delta_static);
- delta_attn_zero: the branch replaced by 0 (zero-ablation context);
- delta_static: the probabilities replaced by their validation mean per (head, query, key), float64 then
  float32 (arbiter item 13), so the mean of a constant equals it bitwise;
- delta_unif: the logits set to 0 and passed through the live QSoftmax (B8-12).
H_attn: the entropy in nats of the post-quantizer probabilities, each row (jet, head, query) renormalized
to sum to 1, averaged over rows; all-zero rows and rows with a negative entry are counted and left out of
the average (B8-9). Computed on the host in float64: rows are summed by a fixed pairwise tree and the mean is
shifted by the first row's value, so constructed uniform rows give log 8 exactly (G41).
Overflow (arbiter item 12): at a datalane WRAP quantizer, an entry overflows when its lane has nonzero
width (k + i + f > 0) and the wrap changes the quantizer's own rounded value, i.e. the rounded input lies
outside [-k 2^i', 2^i' - 2^-f] with i' = max(i, -f) (quantizers 1.2.2 _fixed_point_ops.py:94-98,
183-188, 198); masked lanes are counted apart (B8-10). `overflow` holds the count at the quantizer the
STUDY names for each ablation (O's input quantizer for delta_static and delta_unif, the second tanh
table's input quantizer for the branch ablations); `overflow_downstream` holds every downstream datalane
WRAP quantizer on the path, QAdd's input quantizer of the branch included (B8-10, a STUDY C item).

Hgq, keras and jax are imported inside the functions, after `cbl.env.setup()`; the pure helpers
(`attention_entropy`, `pairwise_sum`, `overflow_from_arrays`, `shifted_mean`) need NumPy only.
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np

SCHEMA = 'cbl1016-attention-diagnostics/1'
VAL_BATCH = 2790                                   # run_train.py:74, Chang's validation batch (B8-13)
LAYERS = ('Q', 'K', 'V', 'O', 'FFN')               # the structure record's layers (cbl/structure.py)
DELTA_KEYS = ('delta_attn_mean', 'delta_attn_zero', 'delta_static', 'delta_unif')
# the quantizer the STUDY names for each ablation (STUDY 8.8; arbiter section 6 item 12)
NAMED_OVERFLOW = {'delta_attn_mean': 'tanh2.iq', 'delta_attn_zero': 'tanh2.iq',
                  'delta_static': 'O.iq', 'delta_unif': 'O.iq'}
# every downstream datalane WRAP quantizer on the path of each ablation, in path order (B8-10)
DOWNSTREAM = {'delta_attn_mean': ('add1.iq[1]', 'tanh2.iq'), 'delta_attn_zero': ('add1.iq[1]', 'tanh2.iq'),
              'delta_static': ('O.iq', 'add1.iq[1]', 'tanh2.iq'), 'delta_unif': ('O.iq', 'add1.iq[1]', 'tanh2.iq')}
OVERFLOW_TAPS = ('O.iq', 'add1.iq[1]', 'tanh2.iq')
COUNT_FIELDS = ('overflow', 'range_check', 'masked', 'live', 'entries', 'nonfinite')
# pass kinds: (injection point, how)
KINDS = {'base': (None, None),
         'branch_const': ('branch', 'const'), 'branch_zero': ('branch', 'zero'), 'branch_identity': ('branch', 'identity'),
         'probs_const': ('probs', 'const'), 'probs_identity': ('probs', 'identity'),
         'logits_zero': ('logits', 'zero'), 'logits_const': ('logits', 'const'), 'logits_identity': ('logits', 'identity')}
ABLATION_KIND = {'delta_attn_mean': 'branch_const', 'delta_attn_zero': 'branch_zero', 'delta_static': 'probs_const',
                 'delta_unif': 'logits_zero'}
PRODUCTS = ('logits', 'zv', 'probs', 'branch', 'overflow')
TENSOR_NDIM = {'branch': 3, 'probs': 4, 'logits': 4}   # (batch, ...) rank of the tensor at each injection point
# graph roles and their inputs (chang/model.py:190-209)
PARENTS = {'embed': ('input',), 'tanh1': ('embed',), 'mha': ('tanh1', 'tanh1'), 'add1': ('embed', 'mha'),
           'tanh2': ('add1',), 'ffn1': ('tanh2',), 'ffn2': ('ffn1',), 'add2': ('add1', 'ffn2'), 'pool': ('add2',),
           'head1': ('pool',), 'head2': ('head1',), 'head3': ('head2',), 'logits': ('head3',)}


class GraphError(ValueError):
    """The model is not Chang's xfm graph (or its lab rebuild), so the re-expression does not apply."""


# ----------------------------------------------------------------------------------------------
# pure helpers (NumPy only)


def pairwise_sum(a):
    """Sum over the last axis by a fixed pairwise tree: pad with zeros to a power of two, then add adjacent
    pairs level by level. Identical terms sum exactly (each level doubles), and the order does not depend on
    NumPy's reduction internals."""
    a = np.asarray(a, np.float64)
    n = a.shape[-1]
    if n == 0:
        return np.zeros(a.shape[:-1], np.float64)
    m = 1
    while m < n:
        m *= 2
    if m != n:
        a = np.concatenate([a, np.zeros(a.shape[:-1] + (m - n,), np.float64)], axis=-1)
    while a.shape[-1] > 1:
        a = a[..., 0::2] + a[..., 1::2]
    return a[..., 0]


def shifted_mean(values):
    """Mean of a float64 vector as v0 + fsum(v - v0) / n: exact when every value equals v0, accurate otherwise.
    None for an empty vector."""
    v = np.asarray(values, np.float64).ravel()
    if v.size == 0:
        return None
    ref = float(v[0])
    return ref + math.fsum((v - ref).tolist()) / v.size


def attention_entropy(p):
    """Per-row entropy (nats) of probabilities p (..., n_keys), each row renormalized to sum to 1 (STUDY 8.8).
    Returns (entropies of the rows used, counts): rows with a negative entry, all-zero rows and rows holding
    a NaN are counted and not used (owner decision B8-9); 0 log 0 = 0."""
    a = np.asarray(p, np.float64)
    a = a.reshape(-1, a.shape[-1])
    nan = np.isnan(a).any(axis=1)
    neg = (a < 0).any(axis=1) & ~nan
    zero = (a == 0).all(axis=1)
    use = ~(nan | neg | zero)
    q = a[use]
    s = pairwise_sum(q)
    q = q / s[:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0)), 0.0)
    h = -pairwise_sum(t)
    return h, {'rows': int(a.shape[0]), 'used': int(use.sum()), 'all_zero_rows': int(zero.sum()),
               'negative_rows': int(neg.sum()), 'nan_rows': int(nan.sum())}


def overflow_from_arrays(x, rounded, wrapped, k, i, f):
    """Counts at one WRAP quantizer from NumPy arrays of one shape: x (its input), rounded (the quantizer's own
    rounding of x), wrapped (its wrap of `rounded`), and k, i, f broadcast to x (owner decision B8-10).
    overflow: live lanes (k + i + f > 0) where the wrap changed the rounded value (a NaN counts, NaN != NaN);
    range_check: live lanes whose rounded value lies outside [-k 2^i', 2^i' - 2^-f], i' = max(i, -f), or is not
    finite (must equal overflow); nonfinite: entries of x that are not finite, counted also apart;
    masked: lanes of zero width, which inference sets to 0 (fixed_point_quantizer.py:148-149)."""
    x, r, w = (np.asarray(v, np.float64) for v in (x, rounded, wrapped))
    k, i, f = (np.asarray(v, np.float64) for v in (k, i, f))
    live = (k + i + f) > 0
    i2 = np.maximum(i, -f)
    lo = -k * np.exp2(i2)
    hi = np.exp2(i2) - np.exp2(-f)
    outside = (r < lo) | (r > hi) | ~np.isfinite(r)
    return {'overflow': int((live & (w != r)).sum()), 'range_check': int((live & outside).sum()),
            'masked': int((~live).sum()), 'live': int(live.sum()), 'entries': int(x.size),
            'nonfinite': int((~np.isfinite(x)).sum())}


def add_counts(acc, c):
    for key, v in c.items():
        acc[key] = acc.get(key, 0) + int(v)
    return acc


def sha256_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def chunks(n, batch):
    if not isinstance(batch, int) or isinstance(batch, bool) or batch <= 0:
        raise ValueError(f'batch must be a positive int, not {batch!r}')
    for start in range(0, n, batch):
        yield slice(start, min(n, start + batch))


# ----------------------------------------------------------------------------------------------
# the graph


def _role_classes():
    from cbl import model as lab_model
    from cbl import rng

    out = {}
    for cls, role in rng.CH8_TOP:
        out[role] = {cls} | ({lab_model.BINARY_CLASS[role]} if role in lab_model.BINARY_CLASS else set())
    return out


def is_binary(layer) -> bool:
    """A binary kernel layer of cbl/binary.py (it has a scale quantizer q_g and no kernel quantizer)."""
    return type(layer).__name__ in ('BinaryEinsumDense', 'BinaryEinsumDenseBatchnorm')


class Graph:
    """The 14 top-level layers of a CH-8 or lab model by role (cbl.rng.CH8_TOP), the MHA's parts, and the
    downstream quantizers. Refuses any other graph: class by role, the functional wiring of PARENTS, and the
    MHA settings the re-expression assumes (fuse none, dropout 0, stable softmax without iq, Q, K and V with
    output quantizers, counted quantizers WRAP without scaler or affine map)."""

    def __init__(self, model):
        from cbl import rng

        layers = list(model.layers)
        roles = [r for _, r in rng.CH8_TOP]
        if len(layers) != len(roles):
            raise GraphError(f'{len(layers)} top-level layers, expected the {len(roles)} of CH-8')
        allowed = _role_classes()
        self.model = model
        self.L = {}
        for role, ly in zip(roles, layers):
            if type(ly).__name__ not in allowed[role]:
                raise GraphError(f'role {role}: class {type(ly).__name__} is not one of {sorted(allowed[role])}')
            self.L[role] = ly
        by_id = {id(ly): role for role, ly in self.L.items()}
        for role, want in PARENTS.items():
            node = self.L[role]._inbound_nodes[0]
            got = tuple(by_id.get(id(t._keras_history.operation), '?') for t in node.input_tensors)
            if got != want:
                raise GraphError(f'role {role}: inputs from {got}, expected {want}')
        mha = self.L['mha']
        self.mha = mha
        self.q, self.k, self.v, self.o = mha._query_dense, mha._key_dense, mha._value_dense, mha._output_dense
        self.softmax = mha._softmax
        problems = []
        if mha._fuse != 'none':
            problems.append(f'fuse {mha._fuse}')
        if mha.dropout:
            problems.append(f'dropout {mha.dropout}')
        if not self.softmax.stable or self.softmax.enable_iq:
            problems.append('softmax not stable or with an input quantizer')
        if not all(ly.enable_oq for ly in (self.q, self.k, self.v)):
            problems.append('Q, K or V without an output quantizer')
        if problems:
            raise GraphError('the MHA differs from Chang\'s: ' + '; '.join(problems))
        self.quantizers = {'O.iq': self.o.iq, 'add1.iq[1]': self.L['add1'].iq[1], 'tanh2.iq': self.L['tanh2'].iq}
        for name, qz in self.quantizers.items():
            if qz.scaler is not None or qz.affine is not None:
                raise GraphError(f'{name}: a quantizer with a scaler or affine map is not counted')
            if qz.quantizer.overflow_mode != 'WRAP':
                raise GraphError(f'{name}: overflow mode {qz.quantizer.overflow_mode}, not WRAP')

    def layer(self, name):
        """The layer of a structure-record name (Q, K, V, O, FFN) or of a top-level role."""
        if name in ('Q', 'K', 'V', 'O'):
            return {'Q': self.q, 'K': self.k, 'V': self.v, 'O': self.o}[name]
        return self.L['ffn2' if name == 'FFN' else name]

    @property
    def mha_has_oq(self) -> bool:
        return bool(self.mha.enable_oq)


def _call(layer, *args):
    """A layer called as Keras's Functional graph calls it in inference (functional.py:683-697)."""
    kw = {'training': False} if 'training' in getattr(layer, '_call_context_args', {}) else {}
    return layer(*args, **kw)


def _inject(inject, point, t, sl):
    if not inject or point not in inject:
        return t
    from keras import ops

    r = inject[point](t, sl)
    if tuple(ops.shape(r)) != tuple(ops.shape(t)) or str(r.dtype) != str(t.dtype):
        raise ValueError(f'injection at {point}: {tuple(ops.shape(r))} {r.dtype} for {tuple(ops.shape(t))} {t.dtype}')
    return r


def mha_forward(g: Graph, query, value=None, key=None, *, inject=None, taps=None, sl=None):
    """QMultiHeadAttention.call and _compute_attention in inference mode (module docstring), with the
    injection points `logits` and `probs`. `taps` (a dict) receives Q, K, V, logits_raw, logits, probs,
    O.iq (the A.V output, O's input) and O."""
    from keras import ops

    mha = g.mha
    taps = {} if taps is None else taps
    value = query if value is None else value
    key = value if key is None else key
    mask = mha._compute_attention_mask(query, value, query_mask=None, value_mask=None, key_mask=None,
                                       attention_mask=None, use_causal_mask=False)
    if mask is not None:
        raise GraphError('an attention mask is not expected')
    q = _call(g.q, query)                                                        # mha.py:464
    k = _call(g.k, key)                                                          # :467
    v = _call(g.v, value)                                                        # :470
    taps['Q'], taps['K'], taps['V'] = q, k, v
    qs = ops.multiply(q, ops.cast(mha._inverse_sqrt_key_dim, q.dtype))          # :517
    scores = ops.einsum(mha._dot_product_equation, k, qs)                       # :521
    taps['logits_raw'] = scores
    scores = _inject(inject, 'logits', scores, sl)
    taps['logits'] = scores
    probs = mha._softmax(scores, mask=None, training=False)                      # :523 (_masked_softmax)
    probs = _inject(inject, 'probs', probs, sl)
    taps['probs'] = probs
    av = ops.einsum(mha._combine_equation, probs, v)                             # :533
    taps['O.iq'] = av
    out = _call(g.o, av)                                                         # :473
    taps['O'] = out
    if mha.enable_oq:                                                            # :475-476
        out = mha.oq(out, training=False)
    return out


def forward(g: Graph, x, *, inject=None, taps=None, sl=None):
    """The model's inference forward by role (chang/model.py:190-209), the MHA through `mha_forward`, with
    the injection point `branch`. `taps` also receives embed, branch, add1.iq[0], add1.iq[1], tanh2.iq,
    FFN (the contraction's output) and the logits under `out`."""
    L = g.L
    taps = {} if taps is None else taps
    h = _call(L['embed'], x)                                                     # :192
    xa = _call(L['tanh1'], h)                                                    # :194
    a = mha_forward(g, xa, xa, inject=inject, taps=taps, sl=sl)                  # :195
    a = _inject(inject, 'branch', a, sl)
    taps['embed'], taps['branch'] = h, a
    taps['add1.iq[0]'], taps['add1.iq[1]'] = h, a
    s = _call(L['add1'], [h, a])                                                 # :197
    taps['tanh2.iq'] = s
    t = _call(L['tanh2'], s)                                                     # :198
    f1 = _call(L['ffn1'], t)                                                     # :199
    f2 = _call(L['ffn2'], f1)                                                    # :200
    taps['FFN'] = f2
    s2 = _call(L['add2'], [s, f2])                                               # :202
    z = _call(L['pool'], s2)                                                     # :204
    z = _call(L['head1'], z)                                                     # :205
    z = _call(L['head2'], z)                                                     # :206
    z = _call(L['head3'], z)                                                     # :207
    out = _call(L['logits'], z)                                                  # :208
    taps['out'] = out
    return out


def num(x):
    from keras import ops

    return np.asarray(ops.convert_to_numpy(x))


def overflow_counts_t(quantizer, x):
    """The overflow counts of `overflow_from_arrays` in backend ops (inside a pass), as an int32 vector in
    COUNT_FIELDS order, with the quantizer's own k, i, f (bw_mapper broadcast), rounding and wrap."""
    from keras import ops

    qz = quantizer.quantizer
    shape = ops.shape(x)
    k, i, f = qz.kif
    k = qz.bw_mapper.bw_to_x(k, shape)
    i = qz.bw_mapper.bw_to_x(i, shape)
    f = qz.bw_mapper.bw_to_x(f, shape)
    sq = qz.stateless_quantizer
    i2 = ops.maximum(i, -f)
    r = sq.round(x, f, False, None)                                              # _fixed_point_ops.py:141-151
    w = sq.saturate(r, k, i2, f)                                                 # :94-98 with :198
    live = (k + i + f) > 0
    lo = -k * 2.0 ** i2
    hi = 2.0 ** i2 - 2.0 ** (-f)
    outside = (r < lo) | (r > hi) | ~ops.isfinite(r)
    one = lambda b: ops.sum(ops.cast(b, 'int32'))                                # noqa: E731
    size = ops.convert_to_tensor(int(np.prod(tuple(x.shape))), 'int32')
    return ops.stack([one(live & (w != r)), one(live & outside), one(~live), one(live), size,
                      one(~ops.isfinite(x))])


def wrap_overflow(quantizer, x) -> dict:
    """`overflow_from_arrays` for one hgq datalane WRAP Quantizer receiving tensor x (eager, NumPy counts)."""
    from keras import ops

    if quantizer.scaler is not None or quantizer.affine is not None:
        raise GraphError('a quantizer with a scaler or affine map is not counted')
    qz = quantizer.quantizer
    if qz.overflow_mode != 'WRAP':
        raise GraphError(f'overflow mode {qz.overflow_mode} is not WRAP')
    shape = ops.shape(x)
    k, i, f = qz.kif
    k = qz.bw_mapper.bw_to_x(k, shape)
    i = qz.bw_mapper.bw_to_x(i, shape)
    f = qz.bw_mapper.bw_to_x(f, shape)
    sq = qz.stateless_quantizer
    rounded = sq.round(x, f, False, None)
    wrapped = sq.saturate(rounded, k, ops.maximum(i, -f), f)
    return overflow_from_arrays(num(x), num(rounded), num(wrapped), num(k), num(i), num(f))


# ----------------------------------------------------------------------------------------------
# passes


def _products(g, out, taps, products):
    from keras import ops

    res = {}
    if 'logits' in products:
        res['logits'] = out
    if 'zv' in products:
        res['zv'] = {n: (ops.all(taps[n] == taps[n][:1]), taps[n][0], ops.min(taps[n], axis=0), ops.max(taps[n], axis=0))
                     for n in LAYERS}
    if 'probs' in products:
        res['probs'] = taps['probs']
    if 'branch' in products:
        res['branch'] = taps['branch']
    if 'overflow' in products:
        res['overflow'] = {q: overflow_counts_t(g.quantizers[q], taps[q]) for q in OVERFLOW_TAPS}
    return res


def _make_fn(g, kind, products):
    from keras import ops

    point, how = KINDS[kind]

    def fn(x, const):
        inj = None
        if point is not None:
            if how == 'zero':
                f = lambda t, sl: ops.zeros_like(t)                                              # noqa: E731
            elif how == 'identity':
                f = lambda t, sl: t                                                              # noqa: E731
            else:
                f = lambda t, sl: ops.broadcast_to(ops.cast(const, t.dtype), ops.shape(t))       # noqa: E731
            inj = {point: f}
        taps = {}
        out = forward(g, x, inject=inj, taps=taps)
        return _products(g, out, taps, products)
    return fn


class Exec:
    """Passes of the re-expression for one Graph (module docstring, "Execution"): eager, or jax.jit inside
    Keras's StatelessScope with every model variable an argument (one compilation per kind, products and
    chunk shape, reused for every later state of the same model)."""

    def __init__(self, g: Graph, *, jit=True, batch=VAL_BATCH):
        if not isinstance(jit, bool):
            raise ValueError('jit must be a bool')
        self.g, self.jit, self.batch = g, jit, batch
        self.variables = list(g.model.variables)
        self._cache = {}
        self.compiles = 0

    def run(self, kind, x, const=None, products=('logits',)):
        """One chunk: products as NumPy (nested dicts and tuples kept)."""
        import jax
        from keras import ops

        if kind not in KINDS:
            raise ValueError(f'kind {kind!r}')
        bad = set(products) - set(PRODUCTS)
        if bad:
            raise ValueError(f'products {sorted(bad)}')
        x = ops.convert_to_tensor(np.asarray(x, np.float32))
        c = None if const is None else ops.convert_to_tensor(np.asarray(const, np.float32))
        if KINDS[kind][1] == 'const' and c is None:
            raise ValueError(f'{kind} needs a constant')
        key = (kind, tuple(sorted(products)))
        fn = self._cache.get(key)
        if fn is None:
            raw = _make_fn(self.g, kind, tuple(products))
            if self.jit:
                import keras
                variables = self.variables

                def stateless(values, xx, cc):
                    with keras.StatelessScope(state_mapping=list(zip(variables, values))):
                        return raw(xx, cc)
                fn = jax.jit(stateless)
            else:
                fn = lambda values, xx, cc: raw(xx, cc)                                         # noqa: E731
            self._cache[key] = fn
            self.compiles += 1
        values = [v.value for v in self.variables] if self.jit else None
        out = fn(values, x, c)
        return jax.tree_util.tree_map(lambda a: np.asarray(a), out)

    def passes(self, kind, X, const=None, products=('logits',)):
        """Chunks of the validation jets X; `const` broadcast over the batch, or per jet when its first axis is
        len(X) (sliced per chunk). Yields (slice, products)."""
        X = np.asarray(X)
        if X.dtype != np.float32:
            raise ValueError(f'validation inputs must be float32, not {X.dtype}')
        point = KINDS[kind][0]
        per_jet = (const is not None and np.ndim(const) == TENSOR_NDIM.get(point, -1)
                   and np.shape(const)[0] == len(X))
        for sl in chunks(len(X), self.batch):
            yield sl, self.run(kind, X[sl], np.asarray(const)[sl] if per_jet else const, products)


def correct_of(logits, y):
    return int(np.sum(np.argmax(logits, axis=-1) == np.asarray(y).ravel()))


class BaselinePass:
    """One pass of the unablated model (or of a gate's injection `kind`) over the validation jets: correct
    count, logits hash, float64 sums of the branch and of the probabilities, H_attn rows, overflow counts,
    branch norms (float64, on the host) and the zero-variance test (a cbl.structure.ZeroVariance in `zv`)."""

    def __init__(self, ex: Exec, X, y, *, kind='base', const=None, keep_logits=True):
        from cbl import structure

        t0 = time.time()
        y = np.asarray(y).ravel()
        if len(X) != len(y) or len(y) == 0:
            raise ValueError('validation inputs and labels must be non-empty and of one length')
        self.n, self.batch, self.kind = len(y), ex.batch, kind
        self.correct = 0
        self.branch_sum = self.probs_sum = None
        h_rows, self.h_counts, self.overflow = [], {}, {}
        nb, logits_all = [], []
        hh = hashlib.sha256()
        self.zv = structure.ZeroVariance()
        for sl, p in ex.passes(kind, X, const, PRODUCTS):
            out = p['logits']
            hh.update(np.ascontiguousarray(out).tobytes())
            if keep_logits:
                logits_all.append(out)
            self.correct += correct_of(out, y[sl])
            br = p['branch'].astype(np.float64)
            pr = p['probs'].astype(np.float64)
            self.branch_sum = br.sum(axis=0) if self.branch_sum is None else self.branch_sum + br.sum(axis=0)
            self.probs_sum = pr.sum(axis=0) if self.probs_sum is None else self.probs_sum + pr.sum(axis=0)
            rows, counts = attention_entropy(pr)
            h_rows.append(rows)
            add_counts(self.h_counts, counts)
            for q, c in p['overflow'].items():
                add_counts(self.overflow.setdefault(q, {}), dict(zip(COUNT_FIELDS, c.tolist())))
            nb.append(np.sqrt(pairwise_sum((br * br).reshape(len(br), -1))))
            self.zv.update_products(sl.stop - sl.start, p['zv'])
        self.logits_sha256 = hh.hexdigest()
        self.logits = np.concatenate(logits_all) if keep_logits else None
        self.branch_mean = (self.branch_sum / self.n).astype(np.float32)
        self.probs_mean = (self.probs_sum / self.n).astype(np.float32)
        self.h_all = np.concatenate(h_rows) if h_rows else np.zeros(0)
        self.norm_branch = np.concatenate(nb)
        self.seconds = time.time() - t0

    def H_attn(self):
        return shifted_mean(self.h_all)


def ablated_pass(ex: Exec, X, y, kind, const=None):
    """Correct count of an injected pass and its overflow counts at every counted quantizer."""
    y = np.asarray(y).ravel()
    correct, over = 0, {}
    for sl, p in ex.passes(kind, X, const, ('logits', 'overflow')):
        correct += correct_of(p['logits'], y[sl])
        for q, c in p['overflow'].items():
            add_counts(over.setdefault(q, {}), dict(zip(COUNT_FIELDS, c.tolist())))
    return correct, over


def ablations(ex: Exec, X, y, base: BaselinePass) -> dict:
    """The four ablations of STUDY 8.8 against `base` (module docstring)."""
    consts = {'delta_attn_mean': base.branch_mean, 'delta_attn_zero': None, 'delta_static': base.probs_mean,
              'delta_unif': None}
    out = {}
    for key in DELTA_KEYS:
        t0 = time.time()
        correct, over = ablated_pass(ex, X, y, ABLATION_KIND[key], consts[key])
        out[key] = {'correct': correct, 'n': base.n,
                    'overflow': {q: over[q] for q in DOWNSTREAM[key]}, 'seconds': time.time() - t0}
    return out


def _widths(g: Graph) -> dict:
    """Mean and maximum widths of the attention quantizers at the frozen state (bits_ at the built shapes)."""
    from keras import ops

    def stats(qz, shape):
        b = num(qz.bits_(shape))
        return {'mean_bits': float(np.mean(b)), 'max_bits': float(np.max(b)), 'zero_bits': int(np.sum(b == 0)),
                'entries': int(b.size)}

    out = {}
    for name, ly in (('Q', g.q), ('K', g.k), ('V', g.v), ('O', g.o)):
        shp = (1,) + tuple(ly._build_shapes_dict['input_shape'][1:])
        rec = {'iq': stats(ly.iq, shp) if ly.enable_iq else None,
               'oq': stats(ly.oq, (1,) + tuple(ly.full_output_shape[1:])) if ly.enable_oq else None}
        kshape = tuple(ops.shape(ly.kernel))
        binary = is_binary(ly)
        b = num(ly.qg.bits_(kshape)) if binary else num(ly.kq.bits_(kshape))
        rec['kernel'] = {'mean_bits': float(np.mean(b)), 'max_bits': float(np.max(b)), 'zero_bits': int(np.sum(b == 0)),
                         'entries': int(b.size), 'kind': 'binary' if binary else 'hgq'}
        out[name] = rec
    sm_shape = (1,) + tuple(g.softmax._build_shapes_dict['input_shape'][1:])
    b = num(g.softmax.oq.bits_(sm_shape))
    out['softmax_oq'] = {'mean_bits_per_head': [float(np.mean(b[0, h])) for h in range(b.shape[1])],
                         'max_bits': float(np.max(b)), 'entries': int(b.size)}
    return out


def _qk_av_native(g: Graph):
    """The MHA's own QK and AV term (mha.py:383-413) at the frozen state, float32, as `_compute_ebops` gives it."""
    shp = (1,) + tuple(g.mha._build_shapes_dict['query_shape'][1:])
    return float(num(g.mha._compute_ebops(shp)))


def diagnostics_from(g: Graph, base: BaselinePass, abl: dict, *, record_correct=None) -> dict:
    """The sidecar's `attention_diagnostics` object (interfaces section 2; the names M16 reads, interfaces
    section 11) from a baseline pass and the four ablations."""
    n = base.n
    out = {'schema': SCHEMA, 'label': 'validation only', 'n': n, 'batch': base.batch, 'inference_mode': True,
           'baseline': {'correct': base.correct, 'n': n, 'top1': base.correct / n, 'logits_sha256': base.logits_sha256},
           'baseline_equals_record': (None if record_correct is None else bool(record_correct == base.correct)),
           'ablated': {}, 'delta_correct': {}, 'overflow': {}, 'overflow_quantizer': dict(NAMED_OVERFLOW),
           'overflow_downstream': {}, 'overflow_baseline': {k: dict(v) for k, v in base.overflow.items()},
           'seconds': {'baseline': base.seconds}}
    for key in DELTA_KEYS:
        a = abl[key]
        out['ablated'][key] = {'correct': a['correct'], 'n': n, 'top1': a['correct'] / n}
        out['delta_correct'][key] = base.correct - a['correct']
        out[key] = (base.correct - a['correct']) / n
        out['overflow'][key] = int(a['overflow'][NAMED_OVERFLOW[key]]['overflow'])
        out['overflow_downstream'][key] = {q: dict(c) for q, c in a['overflow'].items()}
        out['seconds'][key] = a['seconds']
    out['H_attn'] = base.H_attn()
    out['H_attn_rows'] = dict(base.h_counts)
    out['all_zero_rows'] = base.h_counts.get('all_zero_rows', 0)
    out['negative_rows'] = base.h_counts.get('negative_rows', 0)
    out['log8'] = math.log(8.0)
    out['branch_norm'] = {'mean_frobenius': shifted_mean(base.norm_branch),
                          'label': 'per-jet Frobenius norm of the attention branch output, float64 on the host'}
    out['means_sha256'] = {'branch': sha256_array(base.branch_mean), 'probs': sha256_array(base.probs_mean)}
    out['widths'] = _widths(g)
    out['qk_av_native'] = _qk_av_native(g)
    out['mha_has_output_quantizer'] = g.mha_has_oq
    return out


def attention_diagnostics(model, X, y, *, ex=None, record_correct=None, base=None, jit=True, batch=VAL_BATCH):
    """Every M11 quantity at the model's current (frozen) state on the validation jets X (float32) and labels
    y; `ex` reuses an Exec of the same model, `base` a BaselinePass of the same state.
    Returns (diagnostics dict, the BaselinePass)."""
    ex = ex or Exec(Graph(model), jit=jit, batch=batch)
    base = base or BaselinePass(ex, X, y)
    return diagnostics_from(ex.g, base, ablations(ex, X, y, base), record_correct=record_correct), base


def audit_checkpoint(ckpt_path, X, y, *, record_correct=None, jit=True, batch=VAL_BATCH):
    """The checkpoint-audit entry point: reload a saved checkpoint and compute its attention diagnostics
    (the `run_end` mode of cbl.front, owner decision B8-17)."""
    import keras

    from cbl import binary  # noqa: F401  registers the lab classes for deserialization

    model = keras.saving.load_model(str(ckpt_path), compile=False)
    diag, _ = attention_diagnostics(model, X, y, record_correct=record_correct, jit=jit, batch=batch)
    diag['source'] = 'checkpoint audit'
    return diag


__all__ = ['Graph', 'GraphError', 'Exec', 'forward', 'mha_forward', 'BaselinePass', 'ablated_pass', 'ablations',
           'attention_diagnostics', 'audit_checkpoint', 'diagnostics_from', 'wrap_overflow', 'overflow_counts_t',
           'overflow_from_arrays', 'attention_entropy', 'pairwise_sum', 'shifted_mean', 'DELTA_KEYS',
           'NAMED_OVERFLOW', 'DOWNSTREAM', 'VAL_BATCH', 'KINDS', 'is_binary']
