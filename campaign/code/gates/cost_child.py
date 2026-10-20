"""Child process of gates G26-G34 and the audit part of G34 (PREFLIGHT-PLAN section 3 rows G26-G34; STUDY
v0.10 sections 3.1-3.3 and 13 cost row; modules M7 cbl/cost.py, cbl/cost_pure.py and M8 cbl/frozen_eval.py,
with M5 and M6). Gate code, not campaign code.

One invocation is one fresh process on one device (cpu or gpu). It builds models as the runner does
(Chang's get_model for CH-8, cbl.model for the lab arms, in Chang's scopes), puts them in constructed
states from private NumPy generators (no data archive is opened, except by `audit`, which reads the smoke
archive's validation split through M3's reader), runs one gate's checks and writes one JSON record: checks
{what, ok} and readings.

States (`STATES`, applied in this order to one model): `init` as built; `trained` five Adam steps
(learning rate 1e-3, steps_per_execution 1, beta 1e-6 in every layer) on a constructed batch from `init`;
`rand1`-`rand3` every quantizer variable set to integers drawn in ranges that include zero widths and pruned
scales, and kernels, BN gammas (some negative or zero) and moving moments redrawn.

    g26  H1, H2 and NR per layer recomputed by an independent vectorized implementation in this file
         (integer widths from the float32 bits, the column rule with Python integers) equal cbl.cost's
         per layer, per group and in total, for every configuration and state; every native value is the
         uint32 truncation of the layer's own float32 `_compute_ebops`; the NR kernel term at an HGQ site
         equals the layer's own float32 einsum of its `ebops_equation` on (iq.bits_, kq.bits_)
    g27  H1 <= NR and H2 <= H1 at every column, site and layer and in total, every configuration and state;
         a constructed constant wider than its quantizer's bits raises WidthBoundError
    g28  on F000 and CH-8: NR per layer equals the native counter that a jitted training-mode call writes at
         the same quantizer state (a fixed point of two calls), and the layer's own eager `_compute_ebops`;
         the limit case where every constant has its quantizer's full width gives H1 = NR = native at every
         HGQ site; F000 loaded with CH-8's variables gives CH-8's counts; mismatches recorded per layer
    g29  zero- and random-input agreement: the counts are identical after inference forwards on zeros, on
         unit and on large random inputs, and the variables unchanged; a training-mode call assigned back
         (positive control) changes them
    g30  the per-step penalty equals penalty (b) recomputed from quantizer variables (cbl.cost
         penalty_recompute), under both branches: in an eager stateless training-mode call (the value each
         binary layer's `_compute_ebops` returned, and its counter) and after a jitted Adam step (counters);
         redrawing every kernel changes neither the recompute nor `_compute_ebops`
    g31  the stock public aggregate (BetaPID sensor) equals the stock flattened private total (FreeEBOPs)
         on CH-8, F000 and every binary configuration, both > 0, after training-mode calls; a sum of the
         flattened public aggregates (the double count) differs (control)
    g32  the snap of every exported constant to its quantizer's f changes nothing, every configuration
         and state; a constant moved by one float32 ulp raises SnapError
    g33  every binary layer's `_compute_ebops` equals penalty (b) of its branch exactly; under H1 its NR
         equals the truncated penalty; the frozen counts read no counter: junk written to every counter
         leaves H1, H2 and NR unchanged
    audit  G34: cbl.frozen_eval.checkpoint_audit of a saved .keras checkpoint against the frozen record of
         its epoch (exact), and, with --mutate, the same after one q_g or kq bit changed (must differ)

Comparisons are exact; every reading is engineering, not quotable.

    <venv>/bin/python -I -B gates/cost_child.py CMD --device cpu|gpu --out JSON --work DIR [...]
"""

import argparse
import contextlib
import json
import math
import random
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import binary_child as BC  # noqa: E402  (arm_list, build helpers; heavy imports inside its functions)

PRIVATE_SEED = 20261026
STATES = ('init', 'trained', 'rand1', 'rand2', 'rand3')
BATCH = 64


class Rec(BC.Rec):
    pass


def setup(device):
    from cbl import env

    return env.setup(mem_fraction='0.5', jax_platforms='cpu' if device == 'cpu' else 'cuda')


num = BC.num


def configurations():
    """(label, arm_id, groups, embedding, form, control): CH-8, F000 and the 17 binary configurations of B6."""
    out = [('CH-8', 'CH-8', None, None, None, None),
           ('F000', 'F000', {'attention': 'Q', 'ffn': 'Q', 'head': 'Q'}, 'Q', None, None)]
    return out + list(BC.arm_list())


def build(cfg, headline='H1', seed=1):
    """(model, top_names, role_spec); BC builders install the B10-F1 loss guard."""
    from cbl import model as M
    from cbl import rng

    label, arm, groups, emb, form, ctl = cfg
    if arm == 'CH-8':
        m = BC.build_ch8(seed)
        return m, rng.ch8_top_names(m), rng.CH8_ROLE_SPEC
    spec, m = BC.build_arm(label, arm, groups, emb, form, ctl, headline=headline if arm != 'F000' else None, seed=seed)
    rs = M.role_spec(spec)
    return m, rs.top_names(m), rs


# ----------------------------------------------------------------------------------------------
# constructed states


def quantizers(model):
    """Every hgq Quantizer in the model with its owning layer's name, in _flatten_layers order."""
    from hgq.quantizer import Quantizer

    out = []
    for ly in model._flatten_layers(include_self=False):
        if isinstance(ly, Quantizer):
            out.append(ly)
    return out


def _place(q):
    return q.config.config.get('place') if hasattr(q.config, 'config') else None


def randomize(model, rg):
    """Redraw every quantizer variable (integers, zero widths and pruned scales included), every kernel, the BN
    gammas (some negative or zero) and the moving moments. Constant initializers of k are kept."""
    import numpy as np
    from hgq.layers import QEinsumDenseBatchnorm

    from cbl import binary as B

    qg_ids = {id(ly.qg) for ly in model._flatten_layers() if isinstance(ly, (B.BinaryEinsumDense, B.BinaryEinsumDenseBatchnorm))}
    for q in quantizers(model):
        iq = q.quantizer
        cls = type(iq).__name__
        if not hasattr(iq, '_i'):
            continue
        shp = tuple(iq._i.shape)
        if cls == 'FixedPointQuantizerKBI':
            if id(q) in qg_ids:
                b = rg.integers(0, 9, shp)
                b = np.where(rg.random(shp) < 0.15, 0, b)
                i = rg.integers(-3, 2, shp)
            elif iq.overflow_mode == 'WRAP':                         # tables (KBI WRAP)
                b = rg.integers(3, 9, shp)
                i = rg.integers(0, 3, shp)
            else:                                                    # kernel and bias quantizers (SAT_SYM)
                b = rg.integers(0, 9, shp)
                i = rg.integers(-2, 3, shp)
            iq._b.assign(b.astype(np.float32))
            iq._i.assign(i.astype(np.float32))
        elif cls == 'FixedPointQuantizerKIF':
            iq._i.assign(rg.integers(0, 7, shp).astype(np.float32))
            iq._f.assign(rg.integers(-2, 9, shp).astype(np.float32))
    for ly in model._flatten_layers(include_self=False):
        if hasattr(ly, '_kernel') and ly._kernel is not None:
            k = ly._kernel
            ly._kernel.assign((rg.standard_normal(tuple(k.shape)) * 0.5).astype(np.float32))
        if isinstance(ly, QEinsumDenseBatchnorm):
            n = tuple(ly.moving_mean.shape)
            g = rg.standard_normal(n).astype(np.float32)
            g = np.where(rg.random(n) < 0.1, 0.0, g).astype(np.float32)
            ly.bn_gamma.assign(g)
            ly.moving_mean.assign(rg.standard_normal(n).astype(np.float32))
            ly.moving_variance.assign(rg.uniform(0.1, 4.0, n).astype(np.float32))


def train_steps(model, rg, steps=5):
    """Five Adam steps on a constructed batch (a gate compile; the release's recipe is G13's)."""
    import keras
    import numpy as np

    for ly in model._flatten_layers():
        if getattr(ly, '_beta', None) is not None:
            ly._beta.assign(np.float32(1e-6))
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  steps_per_execution=1)
    x = rg.standard_normal((BATCH, 8, 3)).astype(np.float32)
    y = rg.integers(0, 5, (BATCH,))
    for _ in range(steps):
        model.train_on_batch(x, y)


def put_state(model, state, seed):
    import numpy as np

    rg = np.random.default_rng(PRIVATE_SEED + 1000 * seed + STATES.index(state))
    if state.startswith('rand'):
        randomize(model, rg)
    elif state == 'trained':
        train_steps(model, rg)


def models(headlines=('H1',), only=None, seed=1):
    """Yield (label, headline, cfg, model, top_names, role_spec) for every configuration (or `only`)."""
    for cfg in configurations():
        if only is not None and cfg[0] not in only:
            continue
        binary = cfg[1] not in ('CH-8', 'F000')
        for h in (headlines if binary else ('H1',)):
            m, tn, rs = build(cfg, h, seed)
            yield cfg[0], (h if binary else None), cfg, m, tn, rs


def variables_hash(model, rs):
    from cbl import rng

    return rng.variables_sha256(rng.roles_of(model, rs))


# ----------------------------------------------------------------------------------------------
# an independent implementation of the counts (G26)


def width_int(K, f):
    """w(K) of STUDY 3.2 for float32 constants on the grid of step 2^-f, from integer arithmetic: m = |K| 2^f
    (an integer), w = bit length of m without its trailing zeros; 0 for K = 0."""
    import numpy as np

    K = np.asarray(K, np.float64)
    f = np.broadcast_to(np.asarray(f, np.int64), K.shape)
    m = np.abs(K) * np.exp2(f.astype(np.float64))
    if not np.all(m == np.round(m)):
        raise AssertionError('constant off its grid (independent check)')
    out = np.zeros(K.shape, np.int64)
    for idx, v in np.ndenumerate(m.astype(np.int64)):
        v = int(v)
        if v:
            while v % 2 == 0:
                v //= 2
            out[idx] = v.bit_length()
    return out


def ceil_log2_int(x):
    return (x - 1).bit_length()


def independent_site(si):
    """(H1, H2, NR) kernel terms of a cost_pure.SiteInput by the independent implementation."""
    import numpy as np

    b = np.asarray(si.b, np.int64)
    I = np.asarray(si.I, np.int64)
    F = np.asarray(si.F, np.int64)
    K = np.asarray(si.K, np.float32)
    kf = np.broadcast_to(np.asarray(si.kernel_f, np.int64), K.shape)
    kb = np.broadcast_to(np.asarray(si.kernel_bits, np.int64), K.shape)
    w = width_int(K, kf)
    B = b.sum(axis=0)                                   # per input, summed over positions
    h1c = (B[:, None] * w).sum(axis=0)
    nrc = (B[:, None] * kb).sum(axis=0)
    h2 = 0
    for c in range(K.shape[1]):
        col = K[:, c]
        nz = col != 0
        if not nz.any():
            continue
        mags = set(np.abs(col[nz]).tolist())
        if len(mags) != 1:
            h2 += int(h1c[c])
            continue
        g = np.float32(mags.pop())
        wg = int(width_int(np.array([g]), np.array([int(kf[np.argmax(nz), c])]))[0])
        sb, su = 0, 0
        for t in range(b.shape[0]):
            inc = [i for i in range(b.shape[1]) if nz[i] and b[t, i] > 0]
            if not inc:
                continue
            sb += int(sum(int(b[t, i]) for i in inc))
            m = min(int(I[t, i]) for i in inc)
            s = sum(1 << (int(I[t, i]) - m) for i in inc)
            su += m + ceil_log2_int(s) + max(int(F[t, i]) for i in inc)
        h2 += min(int(h1c[c]), sb + su * wg)
    return int(h1c.sum()), int(h2), int(nrc.sum())


def layer_value_np(kt, rest):
    import numpy as np

    v = np.float32(kt) if rest is None else np.float32(np.float32(kt) + rest)
    return int(math.trunc(float(v)))


# ----------------------------------------------------------------------------------------------
# commands


def cmd_g26(a, r):
    import jax
    import numpy as np
    from keras import ops

    from cbl import cost

    n_layers = n_states = 0
    trunc_examples = []
    for label, h, cfg, m, tn, rs in models():
        for state in STATES:
            put_state(m, state, 1)
            ents = cost.layer_entries(m, tn)
            got = cost.frozen_counts(ents)
            bad, tot = [], {'H1': 0, 'H2': 0, 'NR': 0}
            groups = {}
            for e in ents:
                lay = got['layers'][e.name]
                if e.kind == 'native':
                    v32 = num(e.layer._compute_ebops(*e.shapes) * e.layer.ebops_factor)
                    want = {k: int(math.trunc(float(v32))) for k in ('H1', 'H2', 'NR')}
                    if v32.dtype != np.float32:
                        bad.append((e.name, 'dtype', str(v32.dtype)))
                    if float(v32) != math.floor(float(v32)) and len(trunc_examples) < 5:
                        trunc_examples.append({'config': label, 'state': state, 'layer': e.name, 'float32': float(v32),
                                               'value': lay['NR']})
                else:
                    si = cost.site_input(e)
                    k1, k2, kn = independent_site(si)
                    want = {'H1': layer_value_np(k1, si.rest_f32), 'H2': layer_value_np(k2, si.rest_f32),
                            'NR': layer_value_np(kn, si.rest_f32)}
                    if e.kind == 'hgq':
                        shape = e.shapes[0]
                        ein = num(ops.einsum(e.layer.ebops_equation, e.layer.iq.bits_(shape),
                                             e.layer.kq.bits_(ops.shape(e.layer.kernel))))
                        if ein.dtype != np.float32 or int(ein) != kn or float(ein) != kn:
                            bad.append((e.name, 'native-order einsum', float(ein), kn))
                for k in want:
                    if lay[k] != want[k]:
                        bad.append((e.name, k, lay[k], want[k]))
                    tot[k] += want[k]
                    groups.setdefault(e.group, {'H1': 0, 'H2': 0, 'NR': 0})[k] += want[k]
                if num(e.layer._ebops).dtype != np.uint32:
                    bad.append((e.name, 'counter dtype', str(num(e.layer._ebops).dtype)))
                n_layers += 1
            gbad = [g for g, v in groups.items() if any(got['groups'][g][k] != v[k] for k in v)]
            ok = not bad and not gbad and all(got[k] == tot[k] for k in tot)
            r.ck(f'{label} {state}: H1 {got["H1"]}, H2 {got["H2"]}, NR {got["NR"]} equal the independent implementation per '
                 f'layer ({len(ents)} layers), per group and in total; native values are the uint32 truncation of the '
                 f'float32 _compute_ebops; the HGQ NR kernel terms equal the layers\' own float32 einsum'
                 + ('' if ok else f'; differences {bad[:4]} groups {gbad}'), ok)
            n_states += 1
    r.info['n_layer_states'] = n_layers
    r.info['n_config_states'] = n_states
    r.info['truncation_examples'] = trunc_examples
    r.ck(f'non-integer native values are truncated toward zero, not rounded ({len(trunc_examples)} examples recorded: '
         f'{trunc_examples[:2]})', len(trunc_examples) > 0 and all(x['value'] == math.floor(x['float32']) for x in trunc_examples))
    r.info['jax_backend'] = jax.default_backend()


def cmd_g27(a, r):
    import numpy as np

    from cbl import cost, cost_pure

    worst = {}
    for label, h, cfg, m, tn, rs in models():
        for state in STATES:
            put_state(m, state, 1)
            ents = cost.layer_entries(m, tn)
            got = cost.frozen_counts(ents, with_sites=True)
            bad = []
            for name, st in got['sites'].items():
                for c in st.columns:
                    if not (c['H2_c'] <= c['H1_c'] <= c['NR_c']):
                        bad.append((name, c['column']))
                if not (st.h2_kernel <= st.h1_kernel <= st.nr_kernel):
                    bad.append((name, 'site'))
            for name, lay in got['layers'].items():
                if not (lay['H2'] <= lay['H1'] <= lay['NR']):
                    bad.append((name, 'layer'))
            ok = not bad and got['H2'] <= got['H1'] <= got['NR']
            worst[f'{label}/{state}'] = {'H1': got['H1'], 'H2': got['H2'], 'NR': got['NR']}
            r.ck(f'{label} {state}: H2 <= H1 <= NR at every column and site ({len(got["sites"])} sites), every layer and in '
                 f'total ({got["H2"]} <= {got["H1"]} <= {got["NR"]})' + ('' if ok else f'; violations {bad[:4]}'), ok)
    r.info['totals'] = worst
    # control: a constant wider than its quantizer's bits
    si = cost_pure.SiteInput(name='ctl', group='other', kind='hgq', b=[[4, 4]], I=[[1, 1]], F=[[3, 3]],
                             K=np.array([[0.375], [0.5]], np.float32), kernel_f=3, kernel_bits=1)
    try:
        cost_pure.site_terms(si)
        raised = False
    except cost_pure.WidthBoundError:
        raised = True
    r.ck('control: a constant 0.375 (w = 2) under 1-bit quantizer bits raises WidthBoundError (H1 <= NR is enforced)', raised)


def _train_fixed_point(m, x, jitted):
    """Two training-mode stateless calls from the model's state; returns (ntv2, rounded state equal between the
    two calls). The second call's counters are then native at the state (tv, ntv2) (module docstring, g28)."""
    import numpy as np

    tv = [v.value for v in m.trainable_variables]
    ntv = [v.value for v in m.non_trainable_variables]
    _, ntv1 = jitted(tv, ntv, x)
    _, ntv2 = jitted(tv, ntv1, x)
    ok = True
    for v, a1, a2 in zip(m.non_trainable_variables, ntv1, ntv2):
        if v.name in ('i', 'k', 'f', 'b'):
            x1, x2 = np.round(num(a1)), np.round(num(a2))
            if not np.array_equal(x1, x2):
                ok = False
    return ntv2, ok


def cmd_g28(a, r):
    import jax
    import numpy as np

    from cbl import cost, rng

    rg = np.random.default_rng(PRIVATE_SEED + 28)
    x = rg.standard_normal((BATCH, 8, 3)).astype(np.float32)
    mism = []
    for label in ('CH-8', 'F000'):
        cfg = [c for c in configurations() if c[0] == label][0]
        for state in STATES + ('limit',):
            m, tn, rs = build(cfg)
            put_state(m, 'rand1' if state == 'limit' else state, 1)
            if state == 'limit':
                from hgq.layers import QEinsumDenseBatchnorm
                for ly in m._flatten_layers(include_self=False):
                    if hasattr(ly, '_kernel') and ly._kernel is not None:
                        k = num(ly._kernel)
                        ly._kernel.assign(np.where(k >= 0, 1e3, -1e3).astype(np.float32))
                    if isinstance(ly, QEinsumDenseBatchnorm):          # |gamma| >= 1: every W s saturates
                        g = num(ly.bn_gamma)
                        ly.bn_gamma.assign((np.where(g >= 0, 1.0, -1.0) * (1.0 + np.abs(g))).astype(np.float32))
            jitted = jax.jit(lambda tv, ntv, xx, mm=m: mm.stateless_call(tv, ntv, xx, training=True))
            ntv2, fixed = _train_fixed_point(m, x, jitted)
            if not fixed:
                ntv2, fixed = _train_fixed_point(m, x, jitted)
            for v, val in zip(m.non_trainable_variables, ntv2):
                v.assign(val)
            if state == 'limit':
                # the training-mode calls moved the BN moving variance (huge x.W moments); native counts do not read
                # the moments, so set them back to keep every W s saturated for H1 (only qkernel reads them)
                from hgq.layers import QEinsumDenseBatchnorm
                for ly in m._flatten_layers(include_self=False):
                    if isinstance(ly, QEinsumDenseBatchnorm):
                        ly.moving_variance.assign(np.full(tuple(ly.moving_variance.shape), 0.01, np.float32))
            ents = cost.layer_entries(m, tn)
            got = cost.frozen_counts(ents)
            bad = []
            for e in ents:
                lay = got['layers'][e.name]
                eager, _ = cost.native_value(e.layer, e.shapes)
                if not (lay['NR'] == lay['counter'] == eager == lay['native']):
                    bad.append({'layer': e.name, 'NR': lay['NR'], 'counter_jit_training': lay['counter'], 'eager': eager})
                if state == 'limit' and e.kind == 'hgq' and lay['H1'] != lay['NR']:
                    bad.append({'layer': e.name, 'limit_H1': lay['H1'], 'NR': lay['NR']})
            mism += [dict(b, config=label, state=state, device=a.device) for b in bad]
            r.ck(f'{label} {state}: NR per layer equals the native counter of a jitted training-mode call at the same '
                 f'quantizer state and the layer\'s eager _compute_ebops ({len(ents)} layers; NR {got["NR"]}, private total '
                 f'{cost.private_total(m)}; fixed point {fixed})'
                 + (', and H1 = NR at every HGQ site (every constant at its full width)' if state == 'limit' else '')
                 + ('' if not bad else f'; MISMATCH {bad[:3]}'),
                 fixed and not bad and got['NR'] == cost.private_total(m))
            r.info[f'{label}/{state}'] = {'H1': got['H1'], 'H2': got['H2'], 'NR': got['NR']}
    r.info['mismatches'] = mism
    # F000 with CH-8's variables gives CH-8's counts
    for state in ('init', 'rand2'):
        ch, tch, rch = build(configurations()[0])
        f0, tf, rf = build(configurations()[1])
        put_state(ch, state, 1)
        rc, rfr = rng.roles_of(ch, rch), rng.roles_of(f0, rf)
        for k, v in rc.items():
            rfr[k].assign(v.value)
        a1 = cost.frozen_counts(cost.layer_entries(ch, tch))
        a2 = cost.frozen_counts(cost.layer_entries(f0, tf))
        same = a1['layers'] == a2['layers'] and a1['groups'] == a2['groups']
        r.ck(f'F000 holding CH-8\'s 244 variables ({state}) has CH-8\'s H1, H2, NR, native and counters per layer and in total '
             f'({a1["H1"]}, {a1["H2"]}, {a1["NR"]})', same and all(a1[k] == a2[k] for k in ('H1', 'H2', 'NR')))


def cmd_g29(a, r):
    import numpy as np

    from cbl import cost

    rg = np.random.default_rng(PRIVATE_SEED + 29)
    xs = {'zeros': np.zeros((BATCH, 8, 3), np.float32), 'unit': rg.standard_normal((BATCH, 8, 3)).astype(np.float32),
          'large': (rg.standard_normal((BATCH, 8, 3)) * 100).astype(np.float32)}
    for label, h, cfg, m, tn, rs in models():
        for state in ('init', 'rand1'):
            put_state(m, state, 1)
            ents = cost.layer_entries(m, tn)
            h0 = variables_hash(m, rs)
            c0 = cost.frozen_counts(ents)
            same = True
            for k, xv in xs.items():
                m(xv, training=False)
                c = cost.frozen_counts(ents)
                same = same and c == c0
            h1 = variables_hash(m, rs)
            r.ck(f'{label} {state}: H1, H2, NR and every per-layer value identical with no forward and after inference '
                 f'forwards on zero, unit and large random inputs; variables unchanged', same and h0 == h1)
        import jax
        jitted = jax.jit(lambda tv, ntv, xx, mm=m: mm.stateless_call(tv, ntv, xx, training=True))
        tv = [v.value for v in m.trainable_variables]
        ntv = [v.value for v in m.non_trainable_variables]
        _, ntv1 = jitted(tv, ntv, xs['large'])
        for v, val in zip(m.non_trainable_variables, ntv1):
            v.assign(val)
        c1 = cost.frozen_counts(ents)
        r.ck(f'{label} control: a training-mode call on the large inputs, assigned back, moves WRAP integer bits and '
             f'changes the counts (NR {c0["NR"]} -> {c1["NR"]})', c1['NR'] != c0['NR'] and variables_hash(m, rs) != h1)


@contextlib.contextmanager
def capture_compute_ebops(layers):
    """Record the value each layer's `_compute_ebops` returns (eager calls), unchanged."""
    caps = {id(ly): [] for ly in layers}
    saved = []
    for ly in layers:
        orig = ly._compute_ebops

        def spy(*args, _orig=orig, _ly=ly, **kw):
            v = _orig(*args, **kw)
            caps[id(_ly)].append(num(v))
            return v
        ly._compute_ebops = spy
        saved.append(ly)
    try:
        yield caps
    finally:
        for ly in saved:
            del ly._compute_ebops


def cmd_g30(a, r):
    import jax
    import keras
    import numpy as np

    from cbl import binary as B
    from cbl import cost

    rg = np.random.default_rng(PRIVATE_SEED + 30)
    x = rg.standard_normal((BATCH, 8, 3)).astype(np.float32)
    y = rg.integers(0, 5, (BATCH,))
    n_layers = 0
    for label, h, cfg, m, tn, rs in models(headlines=('H1', 'H2')):
        if h is None:
            continue
        bl = B.binary_kernel_layers(m)
        for state in ('init', 'rand1', 'rand2'):
            put_state(m, state, 1)
            # (a) an eager stateless training-mode call: the value each binary _compute_ebops returned, and its counter
            tv = [v.value for v in m.trainable_variables]
            ntv = [v.value for v in m.non_trainable_variables]
            with capture_compute_ebops(bl) as caps:
                _, ntv1 = m.stateless_call(tv, ntv, x, training=True)
            for v, val in zip(m.non_trainable_variables, ntv1):
                v.assign(val)
            bad = []
            for ly in bl:
                p = cost.penalty_recompute(ly)
                got = caps[id(ly)]
                if len(got) != 1 or got[0].dtype != np.float32 or float(got[0]) != p['value_f32']:
                    bad.append((ly.name, [float(g) for g in got], p['value_f32']))
                if int(num(ly._ebops)) != p['counter']:
                    bad.append((ly.name, 'counter', int(num(ly._ebops)), p['counter']))
                n_layers += 1
            r.ck(f'{label} {h} {state}: in an eager training-mode call every binary layer\'s penalty ({len(bl)} layers) and '
                 f'counter equal penalty (b) recomputed from the quantizer variables, branch {h}' + ('' if not bad else f'; {bad[:3]}'),
                 not bad)
        # (b) a jitted Adam step: counters against the recompute at the pre-step trainable state
        m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  steps_per_execution=1)
        bad = []
        for step in range(2):
            before = [num(v.value) for v in m.trainable_variables]
            m.train_on_batch(x, y)
            after = [num(v.value) for v in m.trainable_variables]
            for v, val in zip(m.trainable_variables, before):
                v.assign(val)
            for ly in bl:
                p = cost.penalty_recompute(ly)
                if int(num(ly._ebops)) != p['counter']:
                    bad.append((step, ly.name, int(num(ly._ebops)), p['counter']))
            for v, val in zip(m.trainable_variables, after):
                v.assign(val)
        r.ck(f'{label} {h}: after each of two jitted Adam steps every binary counter equals penalty (b) recomputed at the '
             f'step\'s quantizer state' + ('' if not bad else f'; {bad[:3]}'), not bad)
        # (c) the recompute and _compute_ebops never read the kernel
        p0 = {ly.name: cost.penalty_recompute(ly) for ly in bl}
        e0 = {ly.name: float(num(ly._compute_ebops(*cost.ebops_shapes(ly)))) for ly in bl}
        for ly in bl:
            ly._kernel.assign((rg.standard_normal(tuple(ly._kernel.shape)) * 3).astype(np.float32))
        p1 = {ly.name: cost.penalty_recompute(ly) for ly in bl}
        e1 = {ly.name: float(num(ly._compute_ebops(*cost.ebops_shapes(ly)))) for ly in bl}
        r.ck(f'{label} {h}: redrawing every binary kernel changes neither the recompute nor _compute_ebops', p0 == p1 and e0 == e1)
    r.info['n_layer_checks'] = n_layers
    r.info['jax_backend'] = jax.default_backend()


def cmd_g31(a, r):
    import jax
    import numpy as np

    from cbl import cost

    rg = np.random.default_rng(PRIVATE_SEED + 31)
    x = rg.standard_normal((BATCH, 8, 3)).astype(np.float32)
    for label, h, cfg, m, tn, rs in models():
        for state in ('init', 'rand1'):
            put_state(m, state, 1)
            jitted = jax.jit(lambda tv, ntv, xx, mm=m: mm.stateless_call(tv, ntv, xx, training=True))
            tv = [v.value for v in m.trainable_variables]
            ntv = [v.value for v in m.non_trainable_variables]
            _, ntv1 = jitted(tv, ntv, x)
            for v, val in zip(m.non_trainable_variables, ntv1):
                v.assign(val)
            pub, priv = cost.public_aggregate(m), cost.private_total(m)
            flat_pub = sum(float(ly.ebops) for ly in m._flatten_layers(include_self=False) if hasattr(ly, 'ebops'))
            r.ck(f'{label} {state}: stock public aggregate {pub:.0f} equals the flattened private total {priv}, both > 0; the sum '
                 f'of flattened public aggregates ({flat_pub:.0f}) double-counts (control)',
                 pub == float(priv) and priv > 0 and flat_pub != pub)


def cmd_g32(a, r):
    import numpy as np

    from cbl import cost, cost_pure

    n_const = 0
    for label, h, cfg, m, tn, rs in models():
        for state in STATES:
            put_state(m, state, 1)
            ents = cost.layer_entries(m, tn)
            try:
                got = cost.frozen_counts(ents, with_sites=True)
                ok = got['snap_changed'] == 0 and all(st.snap_changed == 0 for st in got['sites'].values())
                k = sum(st.n_in * st.n_out for st in got['sites'].values())
                err = None
            except cost_pure.SnapError as e:
                ok, k, err = False, 0, str(e)
            n_const += k
            r.ck(f'{label} {state}: the snap of every exported constant to its quantizer\'s f changes nothing ({k} constants)'
                 + ('' if err is None else f'; {err}'), ok)
    r.info['n_constants_snapped'] = n_const
    si = cost_pure.SiteInput(name='ctl', group='other', kind='hgq', b=[[4]], I=[[1]], F=[[3]],
                             K=np.array([[np.nextafter(np.float32(0.375), np.float32(1))]], np.float32), kernel_f=3, kernel_bits=8)
    try:
        cost_pure.site_terms(si)
        raised = False
    except cost_pure.SnapError:
        raised = True
    r.ck('control: 0.375 moved up by one float32 ulp, on the grid of step 2^-3, raises SnapError', raised)


def cmd_g33(a, r):
    import numpy as np

    from cbl import binary as B
    from cbl import cost

    for label, h, cfg, m, tn, rs in models(headlines=('H1', 'H2')):
        if h is None:
            continue
        bl = B.binary_kernel_layers(m)
        for state in ('init', 'rand1', 'rand2', 'trained'):
            put_state(m, state, 1)
            ents = cost.layer_entries(m, tn)
            bad = []
            got = cost.frozen_counts(ents)
            for e in ents:
                if e.kind != 'binary':
                    continue
                p = cost.penalty_recompute(e.layer)
                v = num(e.layer._compute_ebops(*e.shapes))
                if v.dtype != np.float32 or float(v) != p['value_f32']:
                    bad.append((e.name, float(v), p['value_f32']))
                lay = got['layers'][e.name]
                if lay['penalty_b'] != p['counter']:
                    bad.append((e.name, 'penalty_b', lay['penalty_b'], p['counter']))
                if h == 'H1' and lay['NR'] != p['H1']['counter']:
                    bad.append((e.name, 'NR != penalty H1', lay['NR'], p['H1']['counter']))
            saved = [(ly, num(ly._ebops)) for ly in m._flatten_layers() if getattr(ly, '_ebops', None) is not None]
            for ly, _ in saved:
                ly._ebops.assign(np.uint32(123456789))
            junk = cost.frozen_counts(ents)
            for ly, val in saved:
                ly._ebops.assign(val)
            unread = (all(junk['layers'][n][k] == got['layers'][n][k] for n in got['layers'] for k in ('H1', 'H2', 'NR'))
                      and all(junk[k] == got[k] for k in ('H1', 'H2', 'NR'))
                      and all(junk['layers'][n]['counter'] == 123456789 for n in got['layers']))
            r.ck(f'{label} {h} {state}: every binary _compute_ebops ({len(bl)} layers) equals penalty (b) of branch {h} '
                 f'recomputed from quantizer variables' + (', NR equals the truncated H1 penalty' if h == 'H1' else '')
                 + '; junk in every counter leaves H1, H2 and NR unchanged (NR comes only from the evaluator)'
                 + ('' if not bad else f'; {bad[:3]}'), not bad and unread)


def cmd_audit(a, r):
    """G34 audit side: the checkpoint audit of one saved checkpoint against its epoch's frozen record."""
    import keras
    import numpy as np

    from cbl import binary  # noqa: F401
    from cbl import data, frozen_eval, rng

    expect = json.loads(Path(a.expect).read_text(encoding='utf-8'))
    np.random.seed(rng.SPLIT_SEED)                                    # section 6 step 1 (the reader checks it)
    tv = data.read_train_val(a.archive, rng.CH8_N_CONSTITUENTS, rng.CH8_PTETAPHI, expect_sha256=a.archive_sha)
    dev = 'cpu:0' if a.device == 'cpu' else 'gpu:0'
    with keras.device(dev):
        xv, yv = keras.ops.convert_to_tensor(tv.X_val), keras.ops.convert_to_tensor(tv.y_val)
    spec = role_spec_of(a)
    if a.mutate:
        model = keras.models.load_model(a.ckpt)
        roles = rng.roles_of(model, spec)
        key = sorted(k for k in roles if k.endswith('/qg/FixedPointQuantizerKBI[0]/b') or k.endswith('/kq/FixedPointQuantizerKBI[0]/b'))[0]
        v = roles[key]
        arr = num(v.value).copy()
        arr.flat[0] = arr.flat[0] + 1.0
        v.assign(arr)
        ev = frozen_eval.FrozenEvaluator(model, roles=roles, top_names=spec.top_names(model), x_val=xv, y_val=yv)
        core = ev.evaluate()
        mism = frozen_eval.audit_mismatches(expect, core)
        r.info['mutated_role'] = key
        r.info['mismatches'] = mism
        r.ck(f'control: the reloaded checkpoint with {key} moved by one bit differs from the frozen record ({mism})', bool(mism))
        return
    out = frozen_eval.checkpoint_audit(a.ckpt, role_spec=spec, x_val=xv, y_val=yv, expect=expect)
    r.info['audit'] = {k: out['audit'][k] for k in ('val_correct', 'val_n', 'H1', 'H2', 'NR', 'vars_sha256_after')}
    r.info['checkpoint_sha256'] = out['checkpoint_sha256']
    r.info['mismatches'] = out['mismatches']
    r.ck(f'checkpoint audit of {Path(a.ckpt).name} equals the frozen record of its epoch exactly (val {out["audit"]["val_correct"]}/'
         f'{out["audit"]["val_n"]}, H1 {out["audit"]["H1"]}, H2 {out["audit"]["H2"]}, NR {out["audit"]["NR"]}, every per-layer '
         f'subtotal, the binary-site record, the variable hash)' + ('' if out['equal'] else f'; differs in {out["mismatches"]}'),
         out['equal'])


def role_spec_of(a):
    from cbl import model as M
    from cbl import rng

    if a.arm == 'CH-8':
        return rng.CH8_ROLE_SPEC
    import runner_child as RC
    groups, emb, control, code = RC.lab_fields(a.arm, a.form)
    spec = M.arm_spec(a.arm, groups=groups, embedding=emb, form=a.form, control=control,
                      headline=a.headline if a.arm != 'F000' else None)
    return M.role_spec(spec)


COMMANDS = {'g26': cmd_g26, 'g27': cmd_g27, 'g28': cmd_g28, 'g29': cmd_g29, 'g30': cmd_g30, 'g31': cmd_g31,
            'g32': cmd_g32, 'g33': cmd_g33, 'audit': cmd_audit}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('cmd', choices=sorted(COMMANDS))
    ap.add_argument('--device', choices=['cpu', 'gpu'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--ckpt')
    ap.add_argument('--expect')
    ap.add_argument('--archive')
    ap.add_argument('--archive-sha')
    ap.add_argument('--arm')
    ap.add_argument('--form', default=None)
    ap.add_argument('--headline', default=None)
    ap.add_argument('--mutate', action='store_true')
    a = ap.parse_args()
    r = Rec()
    rec = {'cmd': a.cmd, 'args': vars(a)}
    code = 0
    try:
        st = setup(a.device)
        import os

        import jax
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
