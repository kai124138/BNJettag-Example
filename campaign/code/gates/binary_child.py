"""Child process of gates G16-G24 (STUDY v0.10 section 13 "Binary-layer gates"; PREFLIGHT-PLAN section 3
rows G16-G24; module M6 cbl/binary.py, with M5 cbl/model.py). Gate code, not campaign code.

One invocation is one fresh process on one device (cpu or gpu). It builds the binary layers inside
Chang's scopes, alone or in the lab builder's models, on constructed tensors from private NumPy
generators (no data archive is opened), runs one gate's checks and writes one JSON record: a list of
checks {what, ok} and readings. The parent gate script prints the checks through gatelib.

    g16  the training-mode forward kernel (not only qkernel) of every binary layer of every binary
         configuration holds exactly +-q per layer or +-q_c per column, or 0, and q is on its grid
    g17  trainable variables by role: CH-8's, minus kq's b and i at binary sites, plus q_g's b and i
    g18  no kq built and no kq regularization term at binary sites; the losses of a training call
    g19  training-mode autodiff gradients equal Chang's own layer with kq replaced by a gate-written map
    g20  the penalty gradient reaches q_g's b, with the exact value of each branch
    g21  qkernel computable at the end of build (counted), with a control the build wrapper catches
    g22  q_g's configuration (= _qkvo_kq_conf, = the scope's weight configuration), serialized reload,
         the binary MHA's build against the installed source by syntax tree
    g23  edge cases: Sign(0), beta = 0, negative and zero s_c, pruned and rounded-to-zero q_g with H1,
         H2, NR (cbl/cost_pure.py) and the penalty, a saturated column, FB single magnitude
    g24  q_g's float32 output exact on its grid for b <= 23 (outputs on the grid, grid values unchanged);
         a forced b = 24 fails; deviations from exact RND of off-grid inputs recorded

Comparisons fixed before any run (B6, PREFLIGHT.md): forward kernels, gradients and penalty values are
compared for exact numeric equality of every element (no tolerance; -0 equals +0, and byte equality is
recorded beside it); the two derived quantities that this child recomputes analytically in float64
(GR's added derivative at non-BN sites, G19; the end-to-end penalty gradient in a model, G20) must
agree within 1e-5 of the largest magnitude involved. Every reading is engineering, not quotable.

    <venv>/bin/python -I -B gates/binary_child.py CMD --device cpu|gpu --out JSON --work DIR
"""

import argparse
import ast
import contextlib
import functools
import inspect
import json
import math
import random
import sys
import textwrap
import traceback
import types
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))

GROUPS = ('attention', 'ffn', 'head')
PRIVATE_SEED = 20261018
BETA_TEST = 2.0 ** -10
TOL_ANALYTIC = 1e-5


def arm_list():
    """(label, arm_id, groups, embedding, form, control): every binary configuration B6 builds."""
    out = []
    for code in ('F100', 'F001', 'F010', 'F011', 'F101', 'F110', 'F111'):
        groups = {g: 'B' if c == '1' else 'Q' for g, c in zip(GROUPS, code[1:])}
        for form in ([None] if code == 'F100' else ['FB', 'BF']):
            out.append((code if form is None else f'{code}-{form}', code, groups, 'Q', form, None))
    allb = {g: 'B' for g in GROUPS}
    for form in ('FB', 'BF'):
        out.append((f'GR-{form}', 'GR', dict(allb), 'Q', form, 'GR'))
        out.append((f'BE-{form}', 'BE', dict(allb), 'B', form, 'BE'))
    return out


# ----------------------------------------------------------------------------------------------
# shared


class Rec:
    def __init__(self):
        self.checks, self.info = [], {}

    def ck(self, what, ok):
        self.checks.append({'what': what, 'ok': bool(ok)})
        return bool(ok)


def setup(device):
    from cbl import env

    st = env.setup(mem_fraction='0.5', jax_platforms='cpu' if device == 'cpu' else 'cuda')
    return st


def num(x):
    import keras
    import numpy as np

    return np.asarray(keras.ops.convert_to_numpy(x))


@contextlib.contextmanager
def chang_scopes():
    """get_model's three scopes, then get_transformer's three (chang/model.py:278-296, 185-189)."""
    from cbl import model as M

    s0, s1, s2 = M.release_scopes()
    with s0, s1, s2:
        t0, t1, t2 = M.transformer_scopes()
        with t0, t1, t2:
            yield


def build_arm(label, arm_id, groups, embedding, form, control, headline='H1', seed=1):
    import keras
    from cbl import model as M

    spec = M.arm_spec(arm_id, groups=groups, embedding=embedding, form=form, control=control, headline=headline)
    random.seed(seed)
    keras.utils.set_random_seed(seed)
    from cbl import lossguard

    model = M.build(spec)
    lossguard.guard(model)
    return spec, model


def build_ch8(seed=1):
    import keras
    import model as chang_model
    from cbl import rng

    random.seed(seed)
    keras.utils.set_random_seed(seed)
    from cbl import lossguard

    model = chang_model.get_model(*rng.CH8_BUILDER_ARGS)
    lossguard.guard(model)
    return model


def layer_roles(layer):
    """{role: variable} of one layer, with cbl/rng.py's segment rule, relative to the layer."""
    from cbl import rng

    out = {}

    def walk(ly, prefix):
        for v in list(ly._trainable_variables) + list(ly._non_trainable_variables):
            out[f'{prefix}{v.name}'] = v
        counts = {}
        for sub in ly._layers:
            walk(sub, f'{prefix}{rng.segment_name(ly.name, sub.name, type(sub).__name__, counts)}/')

    walk(layer, '')
    return out


def site_layers(model):
    """{role prefix: binary kernel layer} of a lab model (CH-8's top names; MHA projections nested)."""
    from cbl import binary as B
    from cbl import rng

    out = {}
    for (_, top), ly in zip(rng.CH8_TOP, model.layers):
        if isinstance(ly, B.BinaryMultiHeadAttention):
            for attr, seg in (('_query_dense', 'query'), ('_key_dense', 'key'), ('_value_dense', 'value'),
                              ('_output_dense', 'attention_output')):
                out[f'mha/{seg}'] = getattr(ly, attr)
        elif isinstance(ly, (B.BinaryEinsumDense, B.BinaryEinsumDenseBatchnorm)):
            out[top] = ly
    return out


def kernel_sites(model):
    """{role prefix: kernel layer (Chang's or binary)} at the nine binarizable sites and the embedding."""
    from cbl import rng

    out = {}
    for (_, top), ly in zip(rng.CH8_TOP, model.layers):
        if top == 'mha':
            for attr, seg in (('_query_dense', 'query'), ('_key_dense', 'key'), ('_value_dense', 'value'),
                              ('_output_dense', 'attention_output')):
                out[f'mha/{seg}'] = getattr(ly, attr)
        elif top in ('embed', 'ffn1', 'ffn2', 'head1', 'head2', 'head3'):
            out[top] = ly
    return out


def frac(v):
    return Fraction(float(v))


def on_grid(q, b, i):
    """Every q lies exactly on its quantizer's SAT_SYM grid: q * 2^f is an integer, |q| <= 2^i - 2^-f.
    q, b, i broadcast together (b and i are the rounded bits). Returns (all ok, number off)."""
    import numpy as np

    q, b, i = np.broadcast_arrays(np.asarray(q, np.float64), np.asarray(b, np.float64), np.asarray(i, np.float64))
    off = 0
    for qv, bv, iv in zip(q.ravel(), b.ravel(), i.ravel()):
        f = int(bv) - int(iv)
        x = frac(qv) * (Fraction(2) ** f)
        top = Fraction(2) ** int(iv) - Fraction(2) ** (-f)
        if x.denominator != 1 or abs(frac(qv)) > top:
            off += 1
    return off == 0, off


def eq_exact(a, b):
    """(numerically equal everywhere, byte-equal)."""
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False, False
    return bool(np.array_equal(a, b)), a.dtype == b.dtype and a.tobytes() == b.tobytes()


# ----------------------------------------------------------------------------------------------
# G16: the training-mode forward kernel


class PartsSpy:
    """Records every SCALE.parts call (cbl/binary.py) without changing its result."""

    def __init__(self):
        from cbl import binary as B

        self.B, self.orig, self.calls = B, B.SCALE.parts, []

    def __enter__(self):
        orig = self.orig

        def parts(layer, kernel, scaler=None, *, training=None):
            p = orig(layer, kernel, scaler, training=training)
            self.calls.append({'layer': layer, 'training': training,
                               **{k: (None if v is None else num(v)) for k, v in p.items()}})
            return p
        self.B.SCALE.parts = parts
        return self

    def __exit__(self, *exc):
        del self.B.SCALE.parts            # the instance attribute; the class method is back
        return False


def kernel_checks(p, layer):
    """For one recorded map call: K == q * S (BF: q * sign(s) * S) exactly, |K| == q (one magnitude per
    layer or column), q >= 0 and on q_g's grid. Returns a dict of booleans and counts."""
    import numpy as np

    K, q, S = p['K'], p['q'], p['S']
    qb = np.broadcast_to(q, K.shape)
    want = qb * S if p['sign_s'] is None else qb * (np.broadcast_to(p['sign_s'], K.shape) * S)
    qz = layer.qg.quantizer
    grid_ok, off = on_grid(q, num(qz.b), num(qz.i))
    zero_cols = int(np.sum(np.asarray(q) == 0)) if np.ndim(q) else int(q == 0)
    return {'K_eq_qS': eq_exact(K, want.astype(K.dtype))[0], 'abs_K_eq_q': bool(np.array_equal(np.abs(K), qb)),
            'two_values': bool(np.all((K == qb) | (K == -qb))), 'S_pm1': bool(np.all(np.abs(S) == 1)),
            'q_nonneg': bool(np.all(np.asarray(q) >= 0)), 'q_on_grid': grid_ok, 'q_off_grid': off,
            'zero_columns': zero_cols, 'per_column': bool(np.ndim(q) == 1)}


def cmd_g16(a, r):
    import numpy as np

    from cbl import binary as B

    x = np.random.default_rng(PRIVATE_SEED).standard_normal((64, 8, 3)).astype(np.float32)
    per_arm = {}
    for label, arm_id, groups, emb, form, ctl in arm_list():
        spec, m = build_arm(label, arm_id, groups, emb, form, ctl)
        sites = site_layers(m)
        names = {id(ly): s for s, ly in sites.items()}
        with PartsSpy() as spy:
            m(x, training=True)
        calls = spy.calls
        seen = [names.get(id(c['layer'])) for c in calls]
        res = {'n_sites': len(sites), 'n_calls': len(calls), 'sites': sorted(s for s in seen if s)}
        r.ck(f'{label}: the training-mode forward calls the map once per binary site, in training mode '
             f'({len(calls)} calls, {len(sites)} sites)',
             len(calls) == len(sites) and sorted(seen) == sorted(sites) and all(c['training'] is True for c in calls))
        bad = {}
        for c in calls:
            k = kernel_checks(c, c['layer'])
            if not all(k[x_] for x_ in ('K_eq_qS', 'abs_K_eq_q', 'two_values', 'S_pm1', 'q_nonneg', 'q_on_grid')):
                bad[names.get(id(c['layer']))] = k
            res.setdefault('train', {})[names.get(id(c['layer']))] = k
        r.ck(f'{label}: every training-mode kernel equals q*S exactly and holds only +-q per '
             f'{"layer or column"} (or 0), q on its grid{"" if not bad else f"; differ: {sorted(bad)}"}', not bad)
        bad_inf = {}
        for s, ly in sites.items():
            rec = B.SCALE.record(ly)
            k = kernel_checks(rec, ly)
            qk_eq = eq_exact(num(ly.qkernel), rec['K'])[0]
            res.setdefault('inference', {})[s] = dict(k, qkernel_eq_record=qk_eq)
            if not (qk_eq and all(k[x_] for x_ in ('K_eq_qS', 'abs_K_eq_q', 'two_values', 'q_on_grid'))):
                bad_inf[s] = k
        r.ck(f'{label}: qkernel (moving moments) equals the map\'s kernel and holds only +-q or 0, q on its grid'
             f'{"" if not bad_inf else f"; differ: {sorted(bad_inf)}"}', not bad_inf)
        r.ck(f'{label}: q_g is per layer at non-BN sites and per column at fused-BN sites',
             all(res['train'][s]['per_column'] == (s in ('embed', 'ffn1', 'head1', 'head2', 'head3'))
                 for s in res['train'] if s))
        per_arm[label] = {k: v for k, v in res.items() if k != 'inference'}
        per_arm[label]['zero_columns_train'] = {s: v['zero_columns'] for s, v in res['train'].items()}
    r.info['arms'] = per_arm
    r.ck(f'all {len(arm_list())} binary configurations checked', len(per_arm) == len(arm_list()))


# ----------------------------------------------------------------------------------------------
# G17 and G18: variables by role, no kq


def cmd_g17(a, r):
    import keras

    from cbl import model as M
    from cbl import rng

    ch8 = build_ch8()
    ch8_roles = rng.roles_of(ch8, rng.CH8_ROLE_SPEC)
    tr = {id(v) for v in ch8.trainable_variables}
    ch8_train = {ro for ro, v in ch8_roles.items() if id(v) in tr}
    r.info['ch8_trainable'] = len(ch8_train)
    for label, arm_id, groups, emb, form, ctl in arm_list():
        spec, m = build_arm(label, arm_id, groups, emb, form, ctl)
        roles = rng.roles_of(m, M.role_spec(spec))
        trm = {id(v) for v in m.trainable_variables}
        got = {ro for ro, v in roles.items() if id(v) in trm}
        sites = M.binary_sites(spec)
        kq_bi = {f'{s}/kq/FixedPointQuantizerKBI[0]/{v}' for s in sites for v in ('b', 'i')}
        qg_bi = {f'{s}/qg/FixedPointQuantizerKBI[0]/{v}' for s in sites for v in ('b', 'i')}
        want = (ch8_train - kq_bi) | qg_bi
        r.ck(f'{label}: {len(got)} trainable variables by role = CH-8\'s {len(ch8_train)} - kq b,i + q_g b,i at '
             f'{len(sites)} sites', got == want and kq_bi <= ch8_train)
        shapes_ok = all(tuple(roles[ro].shape) == tuple(ch8_roles[ro].shape)
                        and keras.backend.standardize_dtype(roles[ro].dtype) == keras.backend.standardize_dtype(ch8_roles[ro].dtype)
                        for ro in got & ch8_train)
        r.ck(f'{label}: every shared trainable role has CH-8\'s shape and dtype', shapes_ok)
        sh = {}
        for s in sites:
            ncol = {'embed': 24, 'ffn1': 32, 'head1': 32, 'head2': 32, 'head3': 32}.get(s)
            want_shape = () if ncol is None else (ncol,)
            for v in ('k', 'b', 'i'):
                var = roles[f'{s}/qg/FixedPointQuantizerKBI[0]/{v}']
                sh[f'{s}/{v}'] = (tuple(var.shape), keras.backend.standardize_dtype(var.dtype), id(var) in trm)
            ok_s = (sh[f'{s}/b'][0] == want_shape and sh[f'{s}/i'][0] == want_shape and sh[f'{s}/k'][0] == want_shape
                    and sh[f'{s}/k'][1] == 'uint8' and not sh[f'{s}/k'][2] and sh[f'{s}/b'][2] and sh[f'{s}/i'][2])
            if not ok_s:
                r.ck(f'{label}: {s} q_g variables {sh}', False)
        r.ck(f'{label}: q_g b and i have shape () at non-BN sites and (n_out,) at fused-BN sites; q_g k is a '
             f'non-trainable uint8 of the same shape', all(
                 sh[f'{s}/b'][0] == (() if s not in ('embed', 'ffn1', 'head1', 'head2', 'head3') else
                                     ({'embed': 24}.get(s, 32),)) for s in sites))
        r.info.setdefault('arms', {})[label] = {'n_trainable': len(got), 'sites': sites}
    # control: CH-8 does not pass the binary rule for F111's sites (the check can fail)
    spec = M.arm_spec('F111', groups={g: 'B' for g in GROUPS}, form='FB', headline='H1')
    sites = M.binary_sites(spec)
    qg_bi = {f'{s}/qg/FixedPointQuantizerKBI[0]/{v}' for s in sites for v in ('b', 'i')}
    r.ck('control: CH-8\'s own trainable set is not the F111 set (it holds kq b,i and no q_g b,i)',
         ch8_train != ((ch8_train - {x.replace('/qg/', '/kq/') for x in qg_bi}) | qg_bi))


def regularized(layer_role_map, model):
    tr = {id(v) for v in model.trainable_variables}
    return {ro: (type(v.regularizer).__name__, getattr(v.regularizer, 'l1', None))
            for ro, v in layer_role_map.items() if id(v) in tr and v.regularizer is not None}


def cmd_g18(a, r):
    import jax
    import numpy as np

    from cbl import binary as B
    from cbl import model as M
    from cbl import rng

    x = np.random.default_rng(PRIVATE_SEED).standard_normal((16, 8, 3)).astype(np.float32)

    def n_losses(m):
        tv = [v.value for v in m.trainable_variables]
        ntv = [v.value for v in m.non_trainable_variables]
        _, _, losses = m.stateless_call(tv, ntv, x, training=True, return_losses=True)
        return len(losses)

    ch8 = build_ch8()
    ch8_roles = rng.roles_of(ch8, rng.CH8_ROLE_SPEC)
    ch8_reg = regularized(ch8_roles, ch8)
    n_ch8 = n_losses(ch8)
    r.info['ch8_losses_training_call'] = n_ch8
    for label, arm_id, groups, emb, form, ctl in arm_list():
        spec, m = build_arm(label, arm_id, groups, emb, form, ctl)
        roles = rng.roles_of(m, M.role_spec(spec))
        reg = regularized(roles, m)
        bad = []
        for s, ly in site_layers(m).items():
            raised = False
            try:
                ly.kq
            except AttributeError:
                raised = True
            if '_kq' in ly.__dict__ or any(sub.name.endswith('_kq') for sub in ly._layers) or not raised:
                bad.append(f'{s}: a kq object or attribute')
            if any(ro.startswith(f'{s}/kq/') for ro in roles):
                bad.append(f'{s}: a kq role')
            want = {ro.replace(f'{s}/kq/', f'{s}/qg/'): v for ro, v in ch8_reg.items() if ro.startswith(f'{s}/')}
            got = {ro: v for ro, v in reg.items() if ro.startswith(f'{s}/')}
            if got != want:
                bad.append(f'{s}: regularized {sorted(got)} != {sorted(want)}')
            qreg = {ro: v for ro, v in got.items() if '/qg/' in ro}
            if sorted(qreg) != sorted(f'{s}/qg/FixedPointQuantizerKBI[0]/{v}' for v in ('b', 'i')) or \
                    any(v != ('MonoL1', 1e-8) for v in qreg.values()):
                bad.append(f'{s}: q_g regularizers {qreg}')
        r.ck(f'{label}: no kq built at any binary site (no attribute, sublayer or role; kq raises), and the regularized '
             f'variables there are CH-8\'s with kq\'s b and i replaced by q_g\'s (MonoL1 1e-8)'
             f'{"" if not bad else "; " + "; ".join(bad[:4])}', not bad)
        nl = n_losses(m)
        r.ck(f'{label}: a training-mode call adds {nl} losses, as CH-8 ({n_ch8})', nl == n_ch8)
        r.info.setdefault('arms', {})[label] = {'n_regularized': len(reg), 'n_losses': nl}
    # control: the same test on CH-8's layers at F111's sites finds kq
    found = [s for s, ly in kernel_sites(ch8).items() if '_kq' in ly.__dict__]
    r.ck(f'control: the test finds kq at every one of CH-8\'s {len(found)} kernel sites', len(found) == 10)
    del jax, B


# ----------------------------------------------------------------------------------------------
# G19: gradients against Chang's own layer with kq replaced by a gate-written map


def ref_map_class():
    """A gate-written stand-in for kq (STUDY sections 2.2, 2.3, 5 GR), written from the STUDY formulas
    independently of cbl/binary.py. `call` is the non-BN and FB map of T; `bf` the BF map of (W, s);
    `bits_` what the stock _compute_ebops reads from kq (einsum_dense.py:71)."""
    import keras
    from keras import ops

    from hgq.quantizer import Quantizer

    class RefMap(keras.layers.Layer):
        def __init__(self, conf, kind, gr, name):
            super().__init__(name=name)
            self.qg = Quantizer(conf, name=f'{name}_qg')
            self.kind, self.gr = kind, gr

        def build(self, kernel_shape):
            self.qg.build(() if self.kind == 'layer' else (int(kernel_shape[-1]),))
            super().build(kernel_shape)

        def call(self, T, training=None):
            S = ops.cast(ops.where(T > ops.mean(T), 1.0, -1.0), T.dtype)
            beta = ops.mean(ops.abs(T)) if self.kind == 'layer' else ops.mean(ops.abs(T), axis=0)
            q = self.qg(beta if self.gr else ops.stop_gradient(beta), training=training)
            return q * ops.stop_gradient(S) + (T - ops.stop_gradient(T))

        def bf(self, W, s, training=None):
            S = ops.cast(ops.where(W > ops.mean(W), 1.0, -1.0), W.dtype)
            beta = ops.mean(ops.abs(W))
            q = self.qg((beta if self.gr else ops.stop_gradient(beta)) * ops.abs(s), training=training)
            return q * ops.stop_gradient(ops.sign(s) * S) + s * (W - ops.stop_gradient(W))

        def bits_(self, shape):
            return self.qg.bits_(shape)

    return RefMap


def ref_bf_fused(self, training, mean, var):
    """A copy of einsum_dense_batchnorm.py:158-173 with lines 167-168 replaced by the BF map."""
    from keras import ops
    from keras.src.backend.config import epsilon

    assert self.bias is not None
    assert self.bq is not None
    if self.scale:
        bn_gamma = self.bn_gamma
    else:
        bn_gamma = 1
    scaler = bn_gamma / ops.sqrt(var + epsilon())  # type: ignore
    kernel = self.kernel
    fused_qkernel = self.kq.bf(kernel, scaler, training=training)

    offset = -mean * scaler
    offset = ops.reshape(offset, self._norm_offset_cast_shape)
    fused_qbias = self.bq(self.bias + offset, training=training)
    return fused_qkernel, fused_qbias


G19_CASES = (
    # name, kind, equation, output shape, bias axes, activation, input shape, enable_oq
    ('dense', 'dense', '...c,cC->...C', 24, None, None, (8, 32), False),            # the FFN contraction
    ('proj', 'dense', 'abc,cde->abde', (8, 2, 16), 'de', None, (8, 24), True),      # Q and V projections
    ('out', 'dense', 'abcd,cde->abe', (8, 24), 'e', None, (8, 2, 16), False),       # the O projection
    ('ffn1', 'bn', '...c,cC->...C', 32, 'C', 'relu', (8, 24), False),               # the FFN expansion
    ('head', 'bn', '...c,cC->...C', 32, 'C', 'relu', (32,), False),                 # a head layer
)


def make_pair(case, form, gr, seed):
    """The binary layer L and Chang's layer R with kq replaced, both built inside Chang's scopes, with the
    same constructed values in every variable."""
    import numpy as np
    from keras import ops

    from cbl import binary as B
    from hgq.layers import QEinsumDense, QEinsumDenseBatchnorm
    from hgq.quantizer import QuantizerConfig

    name, kind, eq, out, bias_axes, act, in_shape, oq = case
    RefMap = ref_map_class()
    with chang_scopes():
        if kind == 'dense':
            L = B.BinaryEinsumDense(eq, out, bias_axes=bias_axes, activation=act, enable_oq=oq, penalty='H1', gr=gr)
            R = QEinsumDense(eq, out, bias_axes=bias_axes, activation=act, enable_oq=oq)
            mk = 'layer'
        else:
            L = B.BinaryEinsumDenseBatchnorm(eq, out, bias_axes=bias_axes, activation=act, form=form, penalty='H1', gr=gr)
            R = QEinsumDenseBatchnorm(eq, out, bias_axes=bias_axes, activation=act)
            mk = 'column' if form == 'FB' else 'bf'
        conf = QuantizerConfig('default', 'weight')
    old = R._kq
    R._layers.remove(old)
    R._kq = RefMap(conf, mk, gr, name=f'{R.name}_kq')
    if mk == 'bf':
        R.get_fused_qkernel_and_qbias = types.MethodType(ref_bf_fused, R)
    shape = (None,) + tuple(in_shape)
    L.build(shape)
    R.build(shape)
    rg = np.random.default_rng(seed)
    lr = layer_roles(L)
    rr = layer_roles(R)
    mapping = {ro: ('kq/' + ro) if ro.startswith('qg/') else ro for ro in lr}
    for ro, v in lr.items():
        a = num(v)
        if ro == 'kernel':
            a = (rg.standard_normal(a.shape) * 0.3).astype(a.dtype)
        elif ro == 'bias':
            a = (rg.standard_normal(a.shape) * 0.1).astype(a.dtype)
        elif ro == 'gamma':
            a = (rg.standard_normal(a.shape) * 1.5).astype(a.dtype)
        elif ro == 'moving_mean':
            a = (rg.standard_normal(a.shape) * 0.1).astype(a.dtype)
        elif ro == 'moving_variance':
            a = rg.uniform(0.5, 2.0, a.shape).astype(a.dtype)
        v.assign(a)
    for ro, v in lr.items():
        rr[mapping[ro]].assign(num(v))
    x = rg.standard_normal((64,) + tuple(in_shape)).astype(np.float32)
    del ops
    return L, R, lr, rr, mapping, x, rg


def train_grads(layer, x, w):
    """Training-mode stateless call of one layer: output, updated non-trainable state, and the gradients
    of sum(output * w) with respect to every trainable variable and to x."""
    import jax
    import jax.numpy as jnp

    tvars, ntvars = layer.trainable_variables, layer.non_trainable_variables

    def loss(tv, xx):
        y, new = layer.stateless_call(tv, [v.value for v in ntvars], xx, training=True)
        return jnp.sum(y * w), (y, new)

    (val, (y, new)), (gtv, gx) = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)([v.value for v in tvars], x)
    roles = {id(v): ro for ro, v in layer_roles(layer).items()}
    return ({'loss': float(val), 'y': num(y), 'new': {roles[id(v)]: num(n) for v, n in zip(ntvars, new)},
             'grads': {roles[id(v)]: num(g) for v, g in zip(tvars, gtv)}, 'gx': num(gx)})


def cmd_g19(a, r):
    import numpy as np

    from cbl import binary as B

    for ci, case in enumerate(G19_CASES):
        forms = [None] if case[1] == 'dense' else ['FB', 'BF']
        for form in forms:
            for gr in (False, True):
                tag = f'{case[0]}{"-" + form if form else ""}{"-GR" if gr else ""}'
                L, R, lr, rr, mapping, x, rg = make_pair(case, form, gr, PRIVATE_SEED + 7 * ci)
                yshape = (x.shape[0],) + tuple(int(d) for d in L.full_output_shape[1:])
                w = rg.standard_normal(yshape).astype(np.float32)
                gl, gref = train_grads(L, x, w), train_grads(R, x, w)
                out_eq = eq_exact(gl['y'], gref['y'])
                r.ck(f'{tag}: training-mode outputs equal Chang\'s layer with kq replaced by the map (byte-equal '
                     f'{out_eq[1]})', out_eq[0])
                miss = sorted(set(mapping[k] for k in gl['grads']) ^ set(gref['grads']) - {'kq/qg/FixedPointQuantizerKBI[0]/k'})
                diff, byte = {}, {}
                for ro, g in gl['grads'].items():
                    e = eq_exact(g, gref['grads'][mapping[ro]])
                    byte[ro] = e[1]
                    if not e[0]:
                        diff[ro] = float(np.max(np.abs(g.astype(np.float64) - gref['grads'][mapping[ro]])))
                ex = eq_exact(gl['gx'], gref['gx'])
                want_roles = {'kernel', 'qg/FixedPointQuantizerKBI[0]/b', 'qg/FixedPointQuantizerKBI[0]/i'} | (
                    {'gamma', 'bias'} if case[1] == 'bn' else ({'bias'} if case[4] else set()))
                r.ck(f'{tag}: the gradient of every trainable variable ({", ".join(sorted(gl["grads"]))}) and dL/dx '
                     f'equals the reference exactly{"" if not diff else f"; differ {diff}"} (byte-equal '
                     f'{sum(byte.values())}/{len(byte)}, dx byte-equal {ex[1]})',
                     not diff and ex[0] and not miss and want_roles <= set(gl['grads']))
                nb = {ro: eq_exact(v, gref['new'][mapping[ro]])[0] for ro, v in gl['new'].items()}
                r.ck(f'{tag}: the updated non-trainable state equals the reference (moving moments, WRAP integer bits, '
                     f'the EBOPs counter: the H1 penalty equals the stock contraction with kq.bits_ replaced by '
                     f'q_g\'s)', all(nb.values()))
                qg_nonzero = bool(np.any(gl['grads']['qg/FixedPointQuantizerKBI[0]/b'] != 0))
                r.info.setdefault('cases', {})[tag] = {
                    'loss': gl['loss'], 'grad_roles': sorted(gl['grads']), 'qg_b_grad_nonzero': qg_nonzero,
                    'ebops': int(gl['new'].get('ebops', -1)), 'saturated_cols': int(np.sum(B.SCALE.record(L)['saturated']))}
                r.ck(f'{tag}: the task-loss gradient reaches q_g\'s b (nonzero somewhere)', qg_nonzero)
                if gr and case[1] == 'bn':
                    L0, _, _, _, _, _, _ = make_pair(case, form, False, PRIVATE_SEED + 7 * ci)
                    g0 = train_grads(L0, x, w)
                    nz = float(np.max(np.abs(gl['grads']['kernel'] - g0['grads']['kernel'])))
                    r.ck(f'{tag}: GR adds a derivative: its dL/dW differs from the main map\'s (max {nz:.3e}) on the same '
                         f'values and batch, while the forward is unchanged', nz > 0 and eq_exact(gl['y'], g0['y'])[0])
                if gr and case[1] == 'dense':
                    L0, _, lr0, _, _, _, _ = make_pair(case, None, False, PRIVATE_SEED + 7 * ci)
                    g0 = train_grads(L0, x, w)
                    rec = B.SCALE.record(L0)
                    W = num(L0.kernel).astype(np.float64)
                    G0 = g0['grads']['kernel'].astype(np.float64)
                    unsat = float(rec['g'] <= rec['max'])
                    added = float(np.sum(rec['S'] * G0)) * unsat * np.sign(W) / W.size
                    dev = float(np.max(np.abs(gl['grads']['kernel'] - (G0 + added))))
                    scale_ = max(float(np.max(np.abs(G0))), float(np.max(np.abs(added))))
                    r.ck(f'{tag}: GR\'s dL/dW equals the main map\'s plus the analytic added derivative '
                         f'stop_gradient(S) dq_g/dg sign(W)/n (max deviation {dev:.3e}, scale {scale_:.3e}, '
                         f'unsaturated {bool(unsat)})', dev <= TOL_ANALYTIC * scale_ and float(np.max(np.abs(added))) > 0)
                    del lr0


# ----------------------------------------------------------------------------------------------
# G20: the penalty gradient


def canonical(layer, shape, arr):
    """An input-shaped array (batch 1) as (P, n_in): contracted axes last (true at every site here)."""
    import numpy as np

    in_spec, rest = layer.equation.split(',')
    k_spec = rest.split('->')[0]
    letters = in_spec.replace('...', '')
    n_c = sum(1 for ch in letters if ch in k_spec)
    if any(ch in k_spec for ch in letters[:len(letters) - n_c]):
        raise AssertionError(f'{layer.name}: contracted axes are not the last ones ({layer.equation})')
    a = np.broadcast_to(np.asarray(arr), shape)
    n_in = int(np.prod(shape[len(shape) - n_c:]))
    return a.reshape(-1, n_in), n_in


def expected_penalty_grads(layer, shape, bits, beta):
    """Per column: dPenalty/dbits_c for H1 and H2 from the input quantizer's state, with u(t) from
    cbl/cost_pure.py (B3a, exact) and sum_b as integers (STUDY section 3.1(b), 3.2)."""
    import numpy as np

    from cbl import cost_pure

    qz = layer.iq.quantizer
    _, I, F = qz.kif
    b2, n_in = canonical(layer, shape, num(layer.iq.bits_(shape)))
    I2, _ = canonical(layer, shape, num(I))
    F2, _ = canonical(layer, shape, num(F))
    b2, I2, F2 = b2.astype(np.int64), I2.astype(np.int64), F2.astype(np.int64)
    sum_b = int(b2.sum())
    sum_u = sum(cost_pure.u_bound([int(v) for v in I2[t]], [int(v) for v in F2[t]], [int(v) for v in b2[t]],
                                  [True] * n_in) for t in range(b2.shape[0]))
    h1, h2, branch = [], [], []
    for bc in np.ravel(bits):
        bc = int(bc)
        h1.append(beta * sum_b)
        if bc == 0:
            h2.append(0.0)
            branch.append('zero')
        elif sum_b * bc <= sum_b + sum_u * bc:
            h2.append(beta * sum_b)
            branch.append('h1')
        else:
            h2.append(beta * sum_u)
            branch.append('sign_scale')
    return np.asarray(h1), np.asarray(h2), branch, sum_b, sum_u


def penalty_grad(layer, shape, b_value, state=()):
    """beta * penalty (b) of one layer and its gradient with respect to q_g's b, the other variables read
    from `state` (pairs of variable and value) where given, else their current values."""
    import jax
    import jax.numpy as jnp
    import keras

    b_var = layer.qg.quantizer._b

    def pen(bv):
        with keras.StatelessScope(state_mapping=[(b_var, bv)] + [kv for kv in state if kv[0] is not b_var]):
            return layer.beta * layer._compute_ebops(shape)

    v, g = jax.value_and_grad(pen)(jnp.asarray(b_value, jnp.float32))
    return float(v), num(g)


def cmd_g20(a, r):
    import jax
    import keras
    import numpy as np

    from cbl import binary as B
    from cbl import model as M
    from cbl import rng

    arms = [x for x in arm_list() if x[0] in ('F100', 'F111-FB', 'F111-BF', 'BE-FB', 'BE-BF', 'GR-FB')]
    x = np.random.default_rng(PRIVATE_SEED).standard_normal((16, 8, 3)).astype(np.float32)
    yv = np.arange(16) % 5
    for label, arm_id, groups, emb, form, ctl in arms:
        for hl in ('H1', 'H2'):
            spec, m = build_arm(label, arm_id, groups, emb, form, ctl, headline=hl)
            for s, ly in site_layers(m).items():
                ly._beta.assign(BETA_TEST)
                shape = (1,) + tuple(int(d) for d in ly._build_shapes_dict['input_shape'][1:])
                per_col = isinstance(ly, B.BinaryEinsumDenseBatchnorm)
                ncols = int(np.prod(num(ly.qg.bits).shape)) if per_col else None
                states = ([np.resize(np.asarray([0.0, 1.0, 4.0], np.float32), (ncols,))] if per_col
                          else [np.float32(0.0), np.float32(1.0), np.float32(4.0)])
                for bv in states:
                    val, g = penalty_grad(ly, shape, bv)
                    h1, h2, branch, sb, su = expected_penalty_grads(ly, shape, np.round(bv), BETA_TEST)
                    want = h1 if hl == 'H1' else h2
                    if not per_col:
                        k_spec = ly.equation.split(',')[1].split('->')[0]
                        letters = ly.equation.split(',')[0].replace('...', '')
                        n_cols = int(np.prod([int(d) for d, ch in zip(ly.kernel.shape, k_spec) if ch not in letters]))
                        want = want * n_cols
                    ok = bool(np.array_equal(np.asarray(g, np.float64).ravel(), np.asarray(want, np.float64).ravel()))
                    r.ck(f'{label} {hl} {s} b={np.unique(bv).tolist()}: d(beta*penalty)/d(q_g b) equals the {hl} value '
                         f'exactly (sum_b {sb}, sum_u {su}, branches {sorted(set(branch))}); nonzero where b > 0'
                         + ('' if ok else f'; got {np.ravel(g)[:4]} want {np.ravel(want)[:4]}'),
                         ok and (bool(np.all(np.ravel(g)[np.ravel(np.round(bv)) > 0] != 0)) or hl == 'H2'))
            # end to end: the model's training loss with beta > 0 against beta = 0. The penalty that loss holds
            # is computed after the forward has moved the WRAP integer bits of the input quantizers, so the
            # expected gradient is evaluated on the state that forward leaves (beta does not change the forward)
            roles = rng.roles_of(m, M.role_spec(spec))
            loss_obj = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            tv = [v.value for v in m.trainable_variables]
            beta_ids = {id(v) for ro, v in roles.items() if ro.endswith('/beta')}

            def total(tv_, ntv_):
                yp, new_, losses = m.stateless_call(tv_, ntv_, x, training=True, return_losses=True)
                return loss_obj(yv, yp) + sum(losses), new_

            grads, after = {}, None
            for bval in (BETA_TEST, 0.0):
                ntv = [np.float32(bval) if id(v) in beta_ids else v.value for v in m.non_trainable_variables]
                grads[bval], new_ntv = jax.grad(total, has_aux=True)(tv, ntv)
                if after is None:
                    after = list(zip(m.non_trainable_variables, new_ntv))
            idx = {id(v): i for i, v in enumerate(m.trainable_variables)}
            bad = []
            for s, ly in site_layers(m).items():
                i = idx[id(ly.qg.quantizer._b)]
                d = num(grads[BETA_TEST][i]).astype(np.float64) - num(grads[0.0][i]).astype(np.float64)
                shape = (1,) + tuple(int(dd) for dd in ly._build_shapes_dict['input_shape'][1:])
                _, gpen = penalty_grad(ly, shape, num(ly.qg.quantizer._b), state=after)
                scale_ = max(float(np.max(np.abs(num(grads[BETA_TEST][i])))), float(np.max(np.abs(gpen))))
                if not (np.max(np.abs(d - gpen)) <= TOL_ANALYTIC * scale_ and np.any(d != 0)):
                    bad.append(f'{s} ({np.max(np.abs(d - gpen)):.2e})')
            r.ck(f'{label} {hl}: in the model\'s training loss, beta > 0 adds the penalty gradient to every '
                 f'q_g b (within {TOL_ANALYTIC} of the largest magnitude), nonzero{"" if not bad else "; " + ", ".join(bad)}',
                 not bad)


# ----------------------------------------------------------------------------------------------
# G21: qkernel at the end of build


def cmd_g21(a, r):
    import numpy as np

    from cbl import binary as B

    counts = {}
    originals = {}

    def counting(cls):
        orig = next(base.__dict__['qkernel'] for base in cls.__mro__ if 'qkernel' in base.__dict__)
        originals[cls] = cls.__dict__.get('qkernel')

        def get(self):
            counts[self.name] = counts.get(self.name, 0) + 1
            return orig.fget(self)
        cls.qkernel = property(get)

    for cls in (B.BinaryEinsumDense, B.BinaryEinsumDenseBatchnorm):
        counting(cls)
    built = {}
    try:
        with chang_scopes():
            layers = {'dense': B.BinaryEinsumDense('...c,cC->...C', 24, penalty='H1'),
                      'fb': B.BinaryEinsumDenseBatchnorm('...c,cC->...C', 32, bias_axes='C', activation='relu', form='FB', penalty='H1'),
                      'bf': B.BinaryEinsumDenseBatchnorm('...c,cC->...C', 32, bias_axes='C', activation='relu', form='BF', penalty='H2'),
                      'mha': B.BinaryMultiHeadAttention(2, 16, dropout=0.0, penalty='H1')}
            shapes = {'dense': (None, 8, 32), 'fb': (None, 8, 24), 'bf': (None, 32)}
            for k, ly in layers.items():
                before = dict(counts)
                if k == 'mha':
                    ly.build((None, 8, 24), (None, 8, 24))
                else:
                    ly.build(shapes[k])
                built[k] = {n: counts.get(n, 0) - before.get(n, 0) for n in set(counts) | set(before)}
    finally:
        for cls, o in originals.items():
            if o is None:
                del cls.qkernel
            else:
                cls.qkernel = o
    for k in ('dense', 'fb', 'bf'):
        ly = layers[k]
        r.ck(f'{k}: build completed and the 0.1.9 build wrapper evaluated qkernel once at its end '
             f'({built[k]})', ly.built and built[k].get(ly.name) == 1)
        r.ck(f'{k}: qkernel at the end of build equals the map\'s kernel (moving moments)',
             eq_exact(num(ly.qkernel), B.SCALE.record(ly)['K'])[0])
    mha = layers['mha']
    proj = [mha._query_dense, mha._key_dense, mha._value_dense, mha._output_dense]
    r.ck(f'mha: build completed; each of the four binary projections\' qkernel was evaluated once at the end of its '
         f'own build ({ {p.name: built["mha"].get(p.name) for p in proj} })',
         mha.built and all(built['mha'].get(p.name) == 1 for p in proj)
         and all(isinstance(p, B.BinaryEinsumDense) for p in proj))
    # controls: a binary layer whose qkernel is not ready at the end of build fails in the wrapper
    orig_build = B._unwrapped(B.BinaryEinsumDense, 'build')
    orig_call = B._unwrapped(B.BinaryEinsumDense, 'call')

    class NotReady(B.BinaryEinsumDense):
        @functools.wraps(orig_build)
        def build(self, input_shape):
            orig_build(self, input_shape)

        @functools.wraps(orig_call)
        def call(self, inputs, training=None):
            return orig_call(self, inputs, training=training)

        @property
        def qkernel(self):
            if not getattr(self, '_ready', False):
                raise RuntimeError('control sentinel: qkernel is not computable yet')
            return B.SCALE.kernel(self, self._kernel)

    class ReadyAtEnd(NotReady):
        @functools.wraps(orig_build)
        def build(self, input_shape):
            orig_build(self, input_shape)
            self._ready = True

        @functools.wraps(orig_call)
        def call(self, inputs, training=None):
            return orig_call(self, inputs, training=training)

    outcome = {}
    for cls in (NotReady, ReadyAtEnd):
        with chang_scopes():
            ly = cls('...c,cC->...C', 24, penalty='H1')
        try:
            ly.build((None, 8, 32))
            outcome[cls.__name__] = 'built'
        except RuntimeError as e:
            outcome[cls.__name__] = f'raised: {e}'
    r.info['controls'] = outcome
    r.ck(f'control: a layer whose qkernel is not computable at the end of build fails inside the build wrapper '
         f'({outcome["NotReady"]})', outcome['NotReady'].startswith('raised: control sentinel'))
    r.ck(f'control: the same layer made ready at the end of its build builds ({outcome["ReadyAtEnd"]})',
         outcome['ReadyAtEnd'] == 'built')
    del np


# ----------------------------------------------------------------------------------------------
# G22: q_g's configuration, serialized reload, the binary MHA build by syntax tree


def ser(obj):
    import re

    if isinstance(obj, dict):
        return {str(k): ser(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [ser(x) for x in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, 'get_config'):
        try:
            return {'class': type(obj).__name__, 'config': ser(obj.get_config())}
        except NotImplementedError:
            pass
    return re.sub(r' at 0x[0-9a-f]+', '', repr(obj))


STUDY_QG = {'q_type': 'kbi', 'place': 'weight', 'k0': True, 'b0': 4, 'i0': 0, 'round_mode': 'RND', 'overflow_mode': 'SAT_SYM',
            'bc': {'class': 'MinMax', 'config': {'max_value': 23.0, 'min_value': 0.0}},
            'br': {'class': 'MonoL1', 'config': {'l1': 1e-08}}, 'ir': {'class': 'MonoL1', 'config': {'l1': 1e-08}},
            'ic': None, 'i_decay_speed': 0.001, 'homogeneous_axis': [], 'heterogeneous_axis': None, 'trainable': True,
            'is_weight': True}


def strip_doc(fn_node):
    body = fn_node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        fn_node.body = body[1:]
    return fn_node


def fn_ast(fn, rename=None):
    node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    node = strip_doc(node)
    node.decorator_list = []
    if rename:
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id in rename:
                n.id = rename[n.id]
    return ast.dump(node, include_attributes=False)


def cmd_g22(a, r):
    import keras
    import numpy as np

    from cbl import binary as B
    from cbl import model as M
    from cbl import rng
    from hgq.layers import QMultiHeadAttention
    from hgq.quantizer import QuantizerConfig

    with chang_scopes():
        scope_conf = ser(QuantizerConfig('default', 'weight').get_config())
    study_ok = all(scope_conf.get(k) == v for k, v in STUDY_QG.items())
    r.info['scope_weight_conf'] = scope_conf
    r.ck('the weight configuration inside Chang\'s six scopes is STUDY section 2.2\'s q_g configuration (KBI, k0 1, '
         'b0 4, i0 0, RND, SAT_SYM, bc MinMax(0,23), br and ir MonoL1(1e-8), i_decay_speed 1e-3, per value)', study_ok)
    ch8 = build_ch8()
    ch8_kq = {s: ser(ly.kq.config.get_config()) for s, ly in kernel_sites(ch8).items()}
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    x = np.random.default_rng(PRIVATE_SEED).standard_normal((32, 8, 3)).astype(np.float32)
    for label, arm_id, groups, emb, form, ctl in arm_list():
        spec, m = build_arm(label, arm_id, groups, emb, form, ctl, headline='H2' if 'BF' in label else 'H1')
        sites = site_layers(m)
        bad = []
        for s, ly in sites.items():
            got = ser(ly.qg.config.get_config())
            if s.startswith('mha/'):
                mha = m.layers[3]
                if not (ly.qg.config is mha._qkvo_kq_conf and got == ser(mha._qkvo_kq_conf.get_config())):
                    bad.append(f'{s}: not _qkvo_kq_conf')
            elif got != scope_conf:
                bad.append(f'{s}: differs from the scope\'s weight configuration')
            if got != ch8_kq[s]:
                bad.append(f'{s}: differs from CH-8\'s kq configuration there')
        r.ck(f'{label}: q_g\'s configuration is _qkvo_kq_conf itself at the MHA projections and the scope\'s weight '
             f'configuration elsewhere, field by field, and equals CH-8\'s kq configuration at every site'
             f'{"" if not bad else "; " + "; ".join(bad)}', not bad)
        path = work / f'{label}.keras'
        m.save(path)
        m2 = keras.models.load_model(path)
        roles1 = rng.roles_of(m, M.role_spec(spec))
        roles2 = rng.roles_of(m2, M.role_spec(spec))
        var_eq = set(roles1) == set(roles2) and all(eq_exact(num(roles1[k]), num(roles2[k]))[1] for k in roles1)
        s2 = site_layers(m2)
        conf_eq = set(s2) == set(sites) and all(
            ser(s2[k].qg.config.get_config()) == ser(sites[k].qg.config.get_config())
            and type(s2[k]) is type(sites[k]) and (s2[k].form, s2[k].penalty, s2[k].gr) == (sites[k].form, sites[k].penalty, sites[k].gr)
            for k in sites)
        mha2 = m2.layers[3]
        mha_link = all(s2[k].qg.config is mha2._qkvo_kq_conf for k in s2 if k.startswith('mha/')) if \
            isinstance(mha2, B.BinaryMultiHeadAttention) else True
        y1, y2 = num(m(x, training=False)), num(m2(x, training=False))
        r.ck(f'{label}: a serialized reload (.keras) restores every class, q_g configuration (with form, penalty branch '
             f'and GR flag), all {len(roles1)} variables bytewise and the inference logits bytewise',
             conf_eq and var_eq and eq_exact(y1, y2)[1] and mha_link)
    # the binary MHA build and output projection against the installed source
    rename = {'QEinsumDense': 'BinaryEinsumDense'}
    for meth in ('build', '_make_output_dense'):
        inst = fn_ast(B._unwrapped(QMultiHeadAttention, meth), rename)
        ours = fn_ast(B._unwrapped(B.BinaryMultiHeadAttention, meth))
        r.ck(f'BinaryMultiHeadAttention.{meth} equals installed 0.1.9 QMultiHeadAttention.{meth} by syntax tree, '
             f'with QEinsumDense replaced by BinaryEinsumDense ({inspect.getsourcefile(QMultiHeadAttention)})', inst == ours)
        bad_ctl = fn_ast(B._unwrapped(QMultiHeadAttention, meth))
        r.ck(f'control: without the replacement the {meth} trees differ', bad_ctl != ours)


# ----------------------------------------------------------------------------------------------
# G23: edge cases


def site_input(layer, shape, name):
    """cbl/cost_pure.py's SiteInput of one binary layer at its frozen state (interfaces.md section 7)."""
    import numpy as np

    from cbl import binary as B
    from cbl import cost_pure

    rec = B.SCALE.record(layer)
    qz = layer.iq.quantizer
    _, I, F = qz.kif
    b2, n_in = canonical(layer, shape, num(layer.iq.bits_(shape)))
    I2, _ = canonical(layer, shape, num(I))
    F2, _ = canonical(layer, shape, num(F))
    K = rec['K'].reshape(n_in, -1)
    kf = np.broadcast_to(rec['f'], K.shape[1:]) if rec['per_column'] else rec['f']
    kb = np.broadcast_to(rec['b'], K.shape[1:]) if rec['per_column'] else rec['b']
    return cost_pure.SiteInput(name=name, group='other', kind='binary', b=b2.astype(int).tolist(),
                               I=I2.astype(int).tolist(), F=F2.astype(int).tolist(), K=K.astype(np.float32),
                               kernel_f=np.broadcast_to(kf, K.shape).astype(int).tolist(),
                               kernel_bits=np.broadcast_to(kb, K.shape).astype(int).tolist(), rest_f32=None)


def penalty_kernel_terms(layer, shape):
    """The penalty's kernel term under both branches (cbl/binary.py: the H1 einsum and h2_kernel_term)."""
    from keras import ops

    from cbl import binary as B

    bw_inp = layer.iq.bits_(shape)
    ks = ops.shape(layer.kernel)
    h1 = ops.einsum(layer.ebops_equation, bw_inp, B.SCALE.kernel_bits(layer, ks))
    h2 = B.h2_kernel_term(layer, shape, bw_inp, B.SCALE.bits(layer), ks)
    return float(num(h1)), float(num(h2))


def cmd_g23(a, r):
    import jax
    import jax.numpy as jnp
    import keras
    import numpy as np

    from cbl import binary as B
    from cbl import cost_pure

    rg = np.random.default_rng(PRIVATE_SEED + 23)

    def dense(penalty='H1'):
        with chang_scopes():
            ly = B.BinaryEinsumDense('...c,cC->...C', 24, penalty=penalty)
        ly.build((None, 8, 32))
        return ly

    def bn(form, penalty='H1'):
        with chang_scopes():
            ly = B.BinaryEinsumDenseBatchnorm('...c,cC->...C', 32, bias_axes='C', activation='relu', form=form, penalty=penalty)
        ly.build((None, 8, 24))
        return ly

    # 1. Sign(0) = -1
    ly = dense()
    W = np.zeros((32, 24), np.float32)
    W[0::3] = 0.5
    W[1::3] = -0.5
    ly._kernel.assign(W)
    rec = B.SCALE.record(ly)
    zero = W == float(np.mean(W))
    r.ck(f'Sign(0) = -1: mean(W) = {float(np.mean(W))} exactly; the {int(zero.sum())} entries equal to it have S = -1 '
         f'and K = -q ({float(rec["q"])})', float(np.mean(W)) == 0.0 and bool(np.all(rec['S'][zero] == -1))
         and bool(np.all(rec['K'][zero] == -rec['q'])) and float(rec['q']) > 0)
    # 2. beta = 0
    ly = dense()
    ly._kernel.assign(np.zeros((32, 24), np.float32))
    rec = B.SCALE.record(ly)
    shape = (1, 8, 32)
    terms = cost_pure.site_terms(site_input(ly, shape, 'beta0'))
    p1, p2 = penalty_kernel_terms(ly, shape)
    r.ck(f'beta = 0: g = 0, every K = 0 (S = -1), route rounded to zero (b = {float(rec["b"])}); H1 = H2 = 0, '
         f'NR = {terms.nr_kernel} = penalty H1 {p1:.0f}, penalty H2 {p2:.0f} > 0',
         float(rec['g']) == 0 and bool(np.all(rec['K'] == 0)) and bool(np.all(rec['S'] == -1))
         and bool(rec['rounded_to_zero']) and terms.h1_kernel == 0 and terms.h2_kernel == 0
         and terms.nr_kernel > 0 and p1 == terms.nr_kernel and p2 > 0)
    # 3. negative and zero s_c, both forms
    for form in ('FB', 'BF'):
        ly = bn(form)
        W = (rg.standard_normal((24, 32)) * 0.3).astype(np.float32)
        gamma = (rg.uniform(0.5, 1.5, 32)).astype(np.float32)
        gamma[0::4] *= -1
        gamma[1::4] = 0.0
        ly._kernel.assign(W)
        ly.bn_gamma.assign(gamma)
        rec = B.SCALE.record(ly)
        s = num(ly.frozen_scaler())
        if form == 'FB':
            U = W * s[None, :]
            S_want = np.where(U - U.mean() > 0, 1.0, -1.0)
            beta_c = np.abs(U.astype(np.float64)).mean(axis=0)
            ok = bool(np.all(rec['S'] == S_want)) and \
                float(np.max(np.abs(rec['g'] - beta_c))) <= TOL_ANALYTIC * float(np.max(beta_c))
            K_want = rec['q'][None, :] * S_want
        else:
            S_want = np.where(W - W.mean() > 0, 1.0, -1.0)
            ok = bool(np.all(rec['S'] == S_want))
            K_want = rec['q'][None, :] * np.sign(s)[None, :] * S_want
        zero_c = gamma == 0
        neg_c = gamma < 0
        r.ck(f'{form}: negative s_c ({int(neg_c.sum())} columns) are carried by the signs, zero s_c ({int(zero_c.sum())}) '
             f'give zero columns; K equals the formula of STUDY section 2.3 exactly',
             ok and eq_exact(rec['K'], K_want.astype(np.float32))[0] and bool(np.all(rec['K'][:, zero_c] == 0))
             and bool(np.all(rec['q'][zero_c] == 0)) and bool(np.any(rec['K'][:, neg_c] != 0)))
        if form == 'BF':
            r.ck('BF: on a negative s_c the exported column is -q_c S_ic (signs flipped)',
                 bool(np.all(rec['K'][:, neg_c] == -(rec['q'][None, neg_c] * S_want[:, neg_c]))))
    # 4. pruned and rounded-to-zero q_g, with H1, H2, NR and the penalty
    for form in ('FB', 'BF'):
        for route in ('pruned', 'rounded_to_zero'):
            ly = bn(form)
            W = (rg.standard_normal((24, 32)) * 0.3).astype(np.float32)
            ly._kernel.assign(W)
            if route == 'pruned':
                ly.qg.quantizer._b.assign(np.zeros(32, np.float32))
            else:
                ly.bn_gamma.assign(np.full(32, 1e-3, np.float32))       # beta |s| < 1/32: rounds to 0 at f = 4
            rec = B.SCALE.record(ly)
            shape = (1, 8, 24)
            terms = cost_pure.site_terms(site_input(ly, shape, f'{form}-{route}'))
            p1, p2 = penalty_kernel_terms(ly, shape)
            exp_h1, exp_h2, branch, sb, su = expected_penalty_grads(ly, shape, num(ly.qg.bits), 1.0)
            route_ok = bool(np.all(rec[route]))
            if route == 'pruned':
                ok = terms.h1_kernel == terms.h2_kernel == terms.nr_kernel == 0 and p1 == 0 and p2 == 0
            else:
                h2_want = sum(min(sb * int(bb), sb + su * int(bb)) for bb in np.ravel(num(ly.qg.bits)))
                ok = (terms.h1_kernel == 0 and terms.h2_kernel == 0 and terms.nr_kernel > 0 and p1 == terms.nr_kernel
                      and p2 == h2_want and p2 > 0)
            r.ck(f'{form} {route}: every column {route.replace("_", " ")} (b {np.unique(rec["b"]).tolist()}, q '
                 f'{np.unique(rec["q"]).tolist()}); H1 {terms.h1_kernel}, H2 {terms.h2_kernel}, NR {terms.nr_kernel}, '
                 f'penalty H1 {p1:.0f}, H2 {p2:.0f} as STUDY section 2.2 states', route_ok and bool(np.all(rec['K'] == 0)) and ok)
    ly = dense()
    ly.qg.quantizer._b.assign(np.float32(0.0))
    rec = B.SCALE.record(ly)
    terms = cost_pure.site_terms(site_input(ly, (1, 8, 32), 'dense-pruned'))
    p1, p2 = penalty_kernel_terms(ly, (1, 8, 32))
    r.ck(f'non-BN pruned (b = 0): K = 0; H1 = H2 = NR = 0 and both penalty branches 0 (got {terms.h1_kernel}, '
         f'{terms.h2_kernel}, {terms.nr_kernel}, {p1}, {p2})', bool(rec['pruned']) and bool(np.all(rec['K'] == 0))
         and terms.h1_kernel == terms.h2_kernel == terms.nr_kernel == 0 and p1 == p2 == 0)
    # 5. a saturated column: the kernel-path gamma gradient
    for form in ('FB', 'BF'):
        ly = bn(form)
        W = (rg.standard_normal((24, 32)) * 0.3).astype(np.float32)
        gamma = np.full(32, 0.5, np.float32)
        gamma[::2] = 40.0                                      # beta |s_c| far above 15/16
        ly._kernel.assign(W)
        ly.bn_gamma.assign(gamma)
        rec = B.SCALE.record(ly)
        sat = rec['saturated']
        mean = np.zeros(32, np.float32)
        var = np.ones(32, np.float32)
        Rw = rg.standard_normal((24, 32)).astype(np.float32)

        def kpath(gm):
            with keras.StatelessScope(state_mapping=[(ly.bn_gamma, gm)]):
                K, _ = ly.get_fused_qkernel_and_qbias(True, jnp.asarray(mean), jnp.asarray(var))
                return jnp.sum(K * Rw)

        gg = num(jax.grad(kpath)(jnp.asarray(gamma)))
        if form == 'BF':
            ok = bool(np.all(gg[sat] == 0)) and bool(np.all(gg[~sat] != 0))
            what = 'zero on every saturated column, nonzero on the others'
        else:
            ok = bool(np.all(gg[sat] != 0))
            what = 'nonzero on the saturated columns'
        r.ck(f'{form}: {int(sat.sum())} saturated columns (beta |s_c| > 2^i - 2^-f); the direct gamma gradient is {what}',
             ok and int(sat.sum()) == 16)
    # 6. FB columns single-magnitude
    ly = bn('FB')
    ly._kernel.assign((rg.standard_normal((24, 32)) * 0.3).astype(np.float32))
    ly.bn_gamma.assign(rg.uniform(-1.5, 1.5, 32).astype(np.float32))
    rec = B.SCALE.record(ly)
    mags = [np.unique(np.abs(rec['K'][:, c])).tolist() for c in range(32)]
    r.ck('FB: every column holds one magnitude q_c (|K_ic| = q_c for all i)',
         all(len(mg) == 1 and mg[0] == float(rec['q'][c]) for c, mg in enumerate(mags)))


# ----------------------------------------------------------------------------------------------
# G24: exactness of q_g's float32 output


def exact_q(x, b, i):
    """RND with SAT_SYM in exact arithmetic: clip to +-(2^i - 2^-f), then floor(x 2^f + 1/2) / 2^f."""
    f = b - i
    top = Fraction(2) ** i - Fraction(2) ** (-f)
    v = min(max(frac(x), -top), top)
    return Fraction(math.floor(v * Fraction(2) ** f + Fraction(1, 2))) / Fraction(2) ** f


def g24_inputs(b, i, n, rg):
    """float32 test values for (b, i): on-grid values k 2^-f from k = 0 to the top of the range, half-grid
    points (when representable), their float32 neighbours, values beyond the bound, and random values."""
    import numpy as np

    f = b - i
    step = 2.0 ** (-f)
    ks = [0, 1, 2, 3, 2 ** (b - 1) - 1, 2 ** (b - 1), 2 ** (b - 1) + 1, 2 ** b - 3, 2 ** b - 2, 2 ** b - 1]
    vals = []
    for k in ks:
        for d in (0.0, 0.5):
            v = np.float32((k + d) * step)
            vals += [v, np.nextafter(v, np.float32(0)), np.nextafter(v, np.float32(2 ** 30))]
    vals += [np.float32(2.0 ** i), np.float32(2.0 ** (i + 1))]
    vals += list(rg.uniform(0, 2.0 ** i, n).astype(np.float32))
    vals = np.asarray(vals, np.float32)
    return np.concatenate([vals, -vals])


def grid_inputs(b, i):
    """Every on-grid value k 2^-f, |k| <= 2^b - 1, sampled: the extremes, the middle and odd k near both."""
    import numpy as np

    f = b - i
    top = 2 ** b - 1
    ks = sorted({0, 1, 2, 3, top, top - 1, top - 2, top - 3, 2 ** (b - 1), 2 ** (b - 1) - 1, 2 ** (b - 1) + 1} & set(range(0, top + 1)))
    v = np.asarray([k * 2.0 ** (-f) for k in ks], np.float32)
    return np.concatenate([v, -v])


def first_half_step_case(xv, qv, b, i):
    """The one deviation of HGQ's float32 RND from exact RND seen in development: an input within one float32
    ulp below 2^-f / 2, for which the float32 sum x 2^f + 0.5 rounds to 1, so the output is 2^-f, not 0."""
    f = b - i
    half = Fraction(1, 2) * Fraction(2) ** (-f)
    x = frac(xv)
    return abs(x) < half and abs(x) >= half - Fraction(2) ** (-f - 25) and abs(frac(qv)) == Fraction(2) ** (-f) \
        and (frac(qv) > 0) == (x > 0)


def cmd_g24(a, r):
    """STUDY section 2.2: "q_g's own output is exact on its grid because bc = MinMax(0, 23) keeps |q 2^f|
    below 2^23"; critical C16 and the frozen test script (stage_b_build.py:45-48): at 24 bits HGQ's float32
    RND changes odd constants. The exactness check is therefore: every output lies on q_g's grid, and every
    on-grid input comes back unchanged. The comparison with exact RND of off-grid inputs is recorded beside
    it, with the one kind of deviation development found (owner decision B6-9)."""
    import numpy as np

    from cbl import binary as B

    rg = np.random.default_rng(PRIVATE_SEED + 24)
    with chang_scopes():
        ly = B.BinaryEinsumDenseBatchnorm('...c,cC->...C', 32, bias_axes='C', activation='relu', form='FB', penalty='H1')
    ly.build((None, 32))
    qz = ly.qg.quantizer

    def run(b, i, xs, training):
        X = np.tile(np.asarray(xs, np.float32)[:, None], (1, 32))
        qz._b.assign(np.full(32, float(b), np.float32))                 # b = 24 bypasses bc = MinMax(0, 23)
        qz._i.assign(np.full(32, float(i), np.float32))
        return num(ly.qg(X, training=training))[:, 0]

    cases = [(b, i) for b in (1, 2, 4, 8, 12, 16, 20, 22, 23) for i in (-2, 0, 3)]
    off_grid, changed, rnd_dev, other_dev = {}, {}, {}, {}
    for training in (True, None):
        for (b, i) in cases:
            tag = f'b={b} i={i} training={training}'
            xs = g24_inputs(b, i, 200, rg)
            q = run(b, i, xs, training)
            off_grid[tag] = on_grid(q, b, i)[1]
            gx = grid_inputs(b, i)
            changed[tag] = int(np.sum(run(b, i, gx, training) != gx))
            dev = [(xv, qv) for xv, qv in zip(xs, q) if frac(qv) != exact_q(xv, b, i)]
            rnd_dev[tag] = len(dev)
            other_dev[tag] = sum(1 for xv, qv in dev if not first_half_step_case(xv, qv, b, i))
    r.info.update(off_grid_b_le_23=off_grid, grid_values_changed_b_le_23=changed, exact_rnd_deviations_b_le_23=rnd_dev,
                  other_deviations_b_le_23=other_dev)
    r.ck(f'b <= 23 (b = 23 included), {len(off_grid)} (b, i, mode) cases: every q_g output lies exactly on its grid '
         f'(q 2^f an integer, |q| <= 2^i - 2^-f)', all(v == 0 for v in off_grid.values()))
    r.ck(f'b <= 23: q_g returns every on-grid input unchanged, odd constants up to 2^b - 1 steps included',
         all(v == 0 for v in changed.values()))
    forced = {}
    for i in (-2, 0, 3):
        gx = grid_inputs(24, i)
        forced[f'b=24 i={i}'] = int(np.sum(run(24, i, gx, True) != gx))
    r.info['forced_b24_grid_values_changed'] = forced
    r.ck(f'a forced b = 24 fails the same check: on-grid inputs changed {forced}', all(v > 0 for v in forced.values()))
    r.ck(f'finding (owner decision B6-9): against exact RND of off-grid inputs, the only deviations at b <= 23 are '
         f'inputs within one float32 ulp below 2^-f/2 that come out as 2^-f '
         f'({sum(rnd_dev.values())} deviations in {sum(1 for v in rnd_dev.values() if v)} cases, all of that kind)',
         all(v == 0 for v in other_dev.values()))
    # the binary kernel at b = 23: q on its grid, and K = +-q_c
    qz._b.assign(np.full(32, 23.0, np.float32))
    qz._i.assign(np.zeros(32, np.float32))
    ly._kernel.assign((rg.standard_normal((32, 32)) * 0.3).astype(np.float32))
    rec = B.SCALE.record(ly)
    grid_ok, _ = on_grid(rec['q'], rec['b'], rec['i'])
    near = all(abs(frac(qv) - frac(gv)) <= Fraction(1, 2) * Fraction(2) ** (-23) for qv, gv in zip(rec['q'], rec['g']))
    r.ck('at b = 23 the binary kernel\'s q_g(g_c) lies on its grid within half a step of g_c, and K = +-q_c exactly',
         grid_ok and near and bool(np.all(np.abs(rec['K']) == rec['q'][None, :])))


# ----------------------------------------------------------------------------------------------


COMMANDS = {'g16': cmd_g16, 'g17': cmd_g17, 'g18': cmd_g18, 'g19': cmd_g19, 'g20': cmd_g20, 'g21': cmd_g21,
            'g22': cmd_g22, 'g23': cmd_g23, 'g24': cmd_g24}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('cmd', choices=sorted(COMMANDS))
    ap.add_argument('--device', choices=['cpu', 'gpu'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    a = ap.parse_args()
    r = Rec()
    rec = {'cmd': a.cmd, 'args': vars(a)}
    code = 0
    try:
        st = setup(a.device)
        import jax
        import os
        rec['setup'] = {'matmul_precision': str(jax.config.jax_default_matmul_precision),
                        'setup_precision': st.get('matmul_precision'), 'jax_backend': jax.default_backend(),
                        'devices': [str(d) for d in jax.devices()], 'XLA_FLAGS': os.environ.get('XLA_FLAGS')}
        COMMANDS[a.cmd](a, r)
        rec['setup']['matmul_precision_after'] = str(jax.config.jax_default_matmul_precision)
    except Exception as e:  # noqa: BLE001 - recorded for the parent, then exit 1
        rec['error'] = f'{type(e).__name__}: {e}'
        rec['traceback'] = traceback.format_exc()
        code = 1
    rec['checks'], rec['info'], rec['exit'] = r.checks, r.info, code
    Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + '\n', encoding='utf-8')
    return code


if __name__ == '__main__':
    sys.exit(main())
