"""Child process of gates G37-G42 (STUDY v0.10 sections 2.2, 8.1 and 8.8, the section 13 structure-record and
ablation row; PREFLIGHT-PLAN section 3 rows G37-G42; modules M9 cbl/front.py, M10 cbl/structure.py, M11
cbl/ablate.py). Gate code, not campaign code.

One invocation is one fresh process on one device (cpu or gpu). It builds CH-8 (Chang's builder) or lab models
(cbl/model.py) at seed 1, warms their WRAP quantizers with two training-mode calls on constructed jets, and
runs one gate's checks on constructed jets and labels from a private NumPy generator (no data archive is
opened). Constructed cases set quantizer variables, kernels and biases directly. It writes one JSON record: a
list of checks {what, ok} and readings. The parent (gates/b8gate.py, or g37_front.py) prints the checks.

    g37  the front saver bound to real models: no random draw, no NumPy global draw and no variable change
         around a save; ckpt and state hashes; reload of every live checkpoint; sidecar metrics against an
         independent eager forward; .keras byte determinism (a finding either way); register() on a stand-in
         runner context holding float16 validation inputs, as the reader returns them; the run-end audit; M16's
         reader; Chang's ParetoFront parsed
    g38  the zero-variance detector: each named route on a binary and an HGQ layer at Q, K, V, O and the FFN
         contraction (pruned kernel, kernel rounded to zero, all-zero-width input quantizer, all-zero-width
         output quantizer at Q, K, V), `other` by upstream degeneracy, nothing on the unmodified models; the
         mask-versus-bits boundary; zero columns with routes at a non-BN and a fused-BN site
    g39  per-head softmax zero-width fractions; for a degenerate Q, Q_t identical across t and Q all zero
    g40  the U4 re-expression against the layer's own inference forward, bitwise: MHA and taps, whole model,
         eager and jit; refusal of a foreign graph and of a mis-shaped injection
    g41  Delta_attn, Delta_static, Delta_unif exactly 0 when the ablated tensor is replaced by itself; Delta_unif
         and Delta_static exactly 0 on constructed K == 0 models; H_attn = log 8 on uniform rows; in-loop
         diagnostics equal the checkpoint audit; eager equals jit
    g42  the WRAP overflow counter on a constructed quantizer against exact fractions; constructed overflowing
         cases at O's input quantizer, QAdd's input quantizer and the second tanh table's input quantizer;
         rows with a negative entry counted against an independent count

Comparisons fixed before any run (B8, PREFLIGHT rows): logits, taps, counts and Deltas are compared exactly
(bitwise for tensors, integer equality for counts, float equality for H_attn against math.log(8)); timings are
readings. Every reading is engineering, not quotable.

    <venv>/bin/python -I -B gates/b8_child.py CMD --device cpu|gpu --out JSON --work DIR
"""

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
import types
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))

GROUPS = ('attention', 'ffn', 'head')
PRIVATE_SEED = 20261019
N_JETS = 3000                      # two chunk shapes at batch 2790: 2790 and 210
LAYERS = ('Q', 'K', 'V', 'O', 'FFN')


class Rec:
    def __init__(self):
        self.checks, self.info = [], {}

    def ck(self, what, ok):
        self.checks.append({'what': what, 'ok': bool(ok)})
        return bool(ok)


def setup(device):
    from cbl import env

    return env.setup(mem_fraction='0.5', jax_platforms='cpu' if device == 'cpu' else 'cuda')


def num(x):
    import keras
    import numpy as np

    return np.asarray(keras.ops.convert_to_numpy(x))


def jets(n=N_JETS, seed=PRIVATE_SEED):
    import numpy as np

    g = np.random.default_rng(seed)
    return g.standard_normal((n, 8, 3)).astype(np.float32), g.integers(0, 5, n)


def build(label, seed=1):
    """(model, role spec) of CH-8 or a lab label F<abc>[-FB|-BF], seed 1, headline H1 for binary arms."""
    import keras
    import model as chang_model

    from cbl import model as M
    from cbl import rng, lossguard

    random.seed(seed)
    keras.utils.set_random_seed(seed)
    if label == 'CH-8':
        model = chang_model.get_model(*rng.CH8_BUILDER_ARGS)
        lossguard.guard(model)
        return model, rng.CH8_ROLE_SPEC
    code, _, form = label.partition('-')
    groups = {g: 'B' if c == '1' else 'Q' for g, c in zip(GROUPS, code[1:])}
    binary = 'B' in groups.values()
    spec = M.arm_spec(code, groups=groups, form=form or None, headline='H1' if binary else None)
    model = M.build(spec)
    lossguard.guard(model)
    return model, M.role_spec(spec)


def warm(m, X):
    for k in range(2):
        m(X[k * 256:(k + 1) * 256], training=True)


def snapshot(m):
    import numpy as np

    return [np.array(num(v.value)) for v in m.variables]


def restore(m, snap):
    for v, a in zip(m.variables, snap):
        v.assign(a)


def set_var(var, value, index=None):
    import numpy as np

    a = np.array(num(var.value))
    if index is None:
        a[...] = value
    else:
        a[index] = value
    var.assign(a.astype(a.dtype))


def set_q(quantizer, index=None, **vals):
    """Set the internal variables (_k, _b, _i, _f) of an hgq Quantizer, everywhere or at `index`."""
    for name, value in vals.items():
        set_var(getattr(quantizer.quantizer, '_' + name), value, index)


def zero_width(quantizer, index=None):
    set_q(quantizer, index, k=1, i=0, f=-1)          # k + i + f = 0: the inference mask fires


def kernel_widths_zero(layer, index=None):
    from cbl import ablate

    q = layer.qg if ablate.is_binary(layer) else layer.kq
    set_q(q, index, b=0)


def states():
    from cbl import rng

    return rng.py_state_sha256(), rng.np_state_sha256()


def strip(o, drop=('seconds', 'source', 'epoch', 'ckpt_sha256')):
    if isinstance(o, dict):
        return {k: strip(v, drop) for k, v in o.items() if not any(d in k for d in drop)}
    if isinstance(o, list):
        return [strip(v, drop) for v in o]
    return o


def diff_paths(a, b, p=''):
    """Key paths at which two JSON-like objects differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff_paths(a.get(k), b.get(k), f'{p}/{k}')
        return out
    return [] if a == b else [p or '/']


def eager_logits(m, X, batch=2790):
    import numpy as np

    from cbl import ablate

    return np.concatenate([num(m(X[sl], training=False)) for sl in ablate.chunks(len(X), batch)])


def eager_taps(g, X, names, inject=None, batch=2790):
    """Concatenated eager taps of the re-expression (gate-side reference)."""
    import keras
    import numpy as np

    from cbl import ablate

    out = {n: [] for n in names}
    for sl in ablate.chunks(len(X), batch):
        t = {}
        ablate.forward(g, keras.ops.convert_to_tensor(X[sl]), inject=inject, taps=t, sl=sl)
        for n in names:
            out[n].append(num(t[n]))
    return {n: np.concatenate(v) for n, v in out.items()}


# ----------------------------------------------------------------------------------------------
# G37


def cmd_g37(a, r):
    import keras
    import numpy as np
    from hgq.utils.sugar import ParetoFront

    from cbl import ablate, confirm, front as F, rng, runner

    X16, y = jets()
    X16 = X16.astype(np.float16)                    # the reader's dtype (chang/data.py:33; cbl/data.py)
    X = X16.astype(np.float32)                      # what the model's float32 input conversion gives it
    work = Path(a.work)
    sidecars_for_c6 = []
    for label in ('F000', 'F111-FB', 'CH-8'):
        m, spec = build(label)
        m.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        roles = rng.roles_of(m, spec)
        run_dir = work / label / 'run'
        cfg = {'headline': 'H1', 'schedule': 'ramp', 'code_sha': 'd' * 40}
        ctx = types.SimpleNamespace(cfg=cfg, model=m, roles=roles, record={'config_sha256': 'c' * 64}, wandb_run=None,
                                    out_dir=run_dir, flow=types.SimpleNamespace(tv=types.SimpleNamespace(X_val=X16, y_val=y)))
        saver = F.register(ctx, ablations='run_end', keep_superseded=False)
        records = [(600, 400000), (620, 380000), (550, 300000), (500, 200000), (640, 390000), (560, 290000),
                   (450, 100000), (650, 395000)]
        t_save, ok_rng, ok_vars, ok_side, ok_metrics = [], True, True, True, True
        for e, (corr, cost) in enumerate(records):
            m.train_on_batch(X[e * 300:(e + 1) * 300], y[e * 300:(e + 1) * 300])
            before_rng, before_vars = states(), rng.variables_sha256(roles)
            rec = {'epoch': e, 'val_correct': corr, 'val_n': 1000, 'H1': cost, 'H2': cost - 1000, 'NR': cost + 1000,
                   'lr': 0.003, 'beta': 2e-8}
            t0 = time.time()
            ev = saver.on_frozen_record(e, rec)
            t_save.append(time.time() - t0)
            ok_rng &= states() == before_rng
            ok_vars &= rng.variables_sha256(roles) == before_vars
            if ev['saved']:
                sc = json.loads(saver.sidecar_path(e).read_text())
                path = run_dir / sc['ckpt_path']
                ok_side &= (path.exists() and F.sha256_file(path) == sc['ckpt_sha256'] and sc['state_sha256'] == before_vars
                            and sc['ckpt_bytes'] == path.stat().st_size)
                lg = eager_logits(m, X)
                met = F.val_metrics(lg, y)
                ok_metrics &= (met['correct'] == sc['m11_correct'] and met['logits_sha256'] == sc['val_logits_sha256']
                               and met['val_auc_per_class'] == sc['val_auc_per_class'] and met['val_auc_macro'] == sc['val_auc_macro']
                               and set(sc['structure_record']['degenerate']) == set(LAYERS)
                               and sc['structure_record']['n'] == len(X) and sc['structure_record']['inference_mode'] is True)
        r.ck(f'{label}: no module-level random draw and no NumPy global draw during any save (py and NumPy state hashes)', ok_rng)
        r.ck(f'{label}: no variable changes during any save, structure pass included (variables hash by role)', ok_vars)
        r.ck(f'{label}: every saved sidecar names an existing .keras file whose sha256 and size it records, and the '
             f'variables hash of the state saved', ok_side)
        r.ck(f'{label}: register widened the float16 validation inputs exactly: sidecar top-1 count, logits hash and AUC '
             f'equal an independent eager forward (model(x, training=False)) on the float32 cast; structure record over '
             f'all {len(X)} jets in inference mode', ok_metrics)
        # reload every live checkpoint
        ok_reload = True
        for e in sorted(saver.state.live_epochs()):
            sc = json.loads(saver.sidecar_path(e).read_text())
            m2 = keras.saving.load_model(str(run_dir / sc['ckpt_path']), compile=False)
            roles2 = rng.roles_of(m2, spec)
            ok_reload &= rng.variables_sha256(roles2) == sc['state_sha256']
            ok_reload &= F.val_metrics(eager_logits(m2, X), y)['logits_sha256'] == sc['val_logits_sha256']
        r.ck(f'{label}: every live checkpoint reloads to the recorded state hash and to the recorded logits', ok_reload)
        # .keras determinism (a finding either way; the state hash is the deterministic identity)
        p1, p2 = work / label / 'det1.keras', work / label / 'det2.keras'
        m.save(str(p1))
        time.sleep(1.1)
        m.save(str(p2))
        same_bytes = F.sha256_file(p1) == F.sha256_file(p2)
        r1 = rng.variables_sha256(rng.roles_of(keras.saving.load_model(str(p1), compile=False), spec))
        r2 = rng.variables_sha256(rng.roles_of(keras.saving.load_model(str(p2), compile=False), spec))
        r.info.setdefault('keras_save_bytes_identical', {})[label] = same_bytes
        r.ck(f'{label}: two saves of one state reload to one state hash (bytes identical: {same_bytes}; Keras writes '
             f'the save time, so ckpt_sha256 identifies the file and state_sha256 the state)', r1 == r2 == rng.variables_sha256(roles))
        # Chang's ParetoFront as released, parsed
        pf = ParetoFront(run_dir / 'pareto', runner.PARETO_METRICS, runner.PARETO_SIDES, fname_format=runner.PARETO_FNAME,
                         enable_if=runner.pareto_enable_if)
        pf.set_model(m)
        pf.on_train_begin()
        for e, (acc, eb) in enumerate([(0.61, 420000.0), (0.63, 410000.0), (0.40, 100000.0)]):
            pf.on_epoch_end(e, {'val_accuracy': acc, 'ebops': eb, 'accuracy': acc, 'val_loss': 1.0, 'loss': 1.0})
        rf = F.released_front_of(pf, run_dir)
        summ = saver.finalize(end='completed', epochs_completed=len(records), released_front=rf)
        r.ck(f'{label}: released front parsed from Chang\'s ParetoFront ({[(p["epoch"], p["train_count"]) for p in rf]})',
             [p['epoch'] for p in rf] == [1] and rf[0]['train_count'] == 410000.0 and summ['released_front'] == rf)
        r.ck(f'{label}: M16\'s reader accepts the summary ({summ["m16_reader"]})', summ['m16_reader'] == {'ok': True})
        live = sorted(saver.state.live_epochs())
        files = sorted(int(p.name[6:-6]) for p in (saver.dir / 'ckpt').glob('*.keras'))
        att = F.load_sidecars(saver.dir)
        have = sorted(s['epoch'] for s in att.values() if s['attention_diagnostics'] is not None)
        keys_ok = all(all(k in s['attention_diagnostics'] for k in ('delta_attn_mean', 'delta_static', 'delta_unif', 'H_attn', 'overflow'))
                      and s['attention_diagnostics']['source'] == 'checkpoint audit'
                      and set(s['attention_diagnostics']['overflow']) == set(ablate.DELTA_KEYS)
                      for s in att.values() if s['attention_diagnostics'] is not None)
        r.ck(f'{label}: keep_superseded false leaves exactly the live epochs {live} on disk; the run-end audit gave each its '
             f'attention diagnostics with the names M16 reads', files == live and have == live and keys_ok)
        r.info.setdefault('save_seconds', {})[label] = [round(t, 3) for t in t_save]
        r.info.setdefault('saved', {})[label] = {'saved': sorted(saver.saved), 'live': live,
                                                 'bytes': [s['bytes'] for s in saver.saved.values()]}
        sidecars_for_c6.append(att[saver.sha_of(live[0])])
    c6 = confirm.c6_point(sidecars_for_c6)
    r.ck(f'M16\'s c6_point reads three sidecars (one per model) without error (label {c6["label"]!r}; engineering)',
         isinstance(c6.get('label'), str))


# ----------------------------------------------------------------------------------------------
# G38


def cmd_g38(a, r):
    from cbl import ablate, structure

    X, _ = jets()
    for label in ('F000', 'F111-FB'):
        m, _ = build(label)
        warm(m, X)
        g = ablate.Graph(m)
        ex, ee = ablate.Exec(g, jit=True), ablate.Exec(g, jit=False)
        snap = snapshot(m)
        kind = 'binary' if label != 'F000' else 'HGQ'
        cases = 0

        def run_case(name, mutate, expect, check=None):
            restore(m, snap)
            mutate()
            rec = structure.structure_record(m, X, ex=ex)
            rec_e = structure.structure_record(m, X, ex=ee)
            got = {k: v for k, v in rec['degenerate'].items() if v}
            ok = got == expect and not rec['anomalies'] and rec_e['degenerate'] == rec['degenerate']
            if check is not None:
                ok = ok and check(rec)
            r.ck(f'{label} ({kind}) {name}: degenerate {got}, expected {expect}; jit and eager agree', ok)
            r.info.setdefault(label, {})[name] = {'degenerate': rec['degenerate'], 'routes': rec['routes'],
                                                  'max_spread': rec['max_spread'], 'anomalies': rec['anomalies']}
            return rec

        run_case('unmodified', lambda: None, {})
        for L in LAYERS:
            ly = g.layer(L)
            binary = ablate.is_binary(ly)
            r.ck(f'{label}: layer {L} is {"binary" if binary else "HGQ"}', binary == (label != 'F000'))
            # V is 0 when its kernel or a quantizer route removes it (its bias is 0 at initialization), so O, fed
            # A.V = 0, is constant too: route `other`
            also = {'O': 'other'} if L == 'V' else {}
            rec = run_case(f'{L} kernel pruned', lambda ly=ly: kernel_widths_zero(ly), {L: 'kernel_pruned', **also})
            ncol = rec['zero_columns'][L]['n_columns']
            r.ck(f'{label} {L} kernel pruned: all {ncol} columns zero with route pruned',
                 rec['zero_columns'][L]['pruned'] == list(range(ncol)) and rec['zero_columns'][L]['rounded_to_zero'] == [])
            rec = run_case(f'{L} kernel rounded to zero', lambda ly=ly: set_var(ly.kernel, 1e-6),
                           {L: 'kernel_rounded_to_zero', **also})
            r.ck(f'{label} {L} kernel rounded to zero: all {ncol} columns zero with route rounded_to_zero',
                 rec['zero_columns'][L]['rounded_to_zero'] == list(range(ncol)) and rec['zero_columns'][L]['pruned'] == [])
            run_case(f'{L} input quantizer zero width', lambda ly=ly: zero_width(ly.iq), {L: 'input_zero_width', **also},
                     lambda rec, L=L: rec['input_zero_width_fraction'][L] == 1.0)
            if L in ('Q', 'K', 'V'):
                run_case(f'{L} output quantizer zero width', lambda ly=ly: zero_width(ly.oq), {L: 'output_zero_width', **also})
            cases += 1
        run_case('O by upstream (K and V kernels pruned)', lambda: (kernel_widths_zero(g.k), kernel_widths_zero(g.v)),
                 {'K': 'kernel_pruned', 'V': 'kernel_pruned', 'O': 'other'})
        run_case('FFN by upstream (FFN expansion kernel pruned)', lambda: kernel_widths_zero(g.L['ffn1']), {'FFN': 'other'})
        # the mask-versus-bits boundary: k = 1, i + f = 0 (bits 0, mask not fired)
        rec = run_case('Q input quantizer with k = 1 and i + f = 0', lambda: set_q(g.q.iq, k=1, i=0, f=0), {})
        r.ck(f'{label}: at k = 1, i + f = 0 the input is zero bits but not zero width (input_zero_bits_fraction '
             f'{rec["input_zero_bits_fraction"]["Q"]}, input_zero_width_fraction {rec["input_zero_width_fraction"]["Q"]}); '
             f'no route named', rec['input_zero_bits_fraction']['Q'] == 1.0 and rec['input_zero_width_fraction']['Q'] == 0.0
             and rec['routes']['Q'] == [])
        # zero columns at a column subset
        if label == 'F000':
            def cols():
                set_q(g.k.kq, (slice(None), 0, 2), b=0)
                set_var(g.k.kernel, 1e-6, (slice(None), 0, 7))
            rec = run_case('K columns 2 (widths 0) and 7 (rounded)', cols, {})
            r.ck(f'{label}: HGQ K zero columns pruned [2], rounded_to_zero [7]',
                 rec['zero_columns']['K']['pruned'] == [2] and rec['zero_columns']['K']['rounded_to_zero'] == [7])
        else:
            def cols():
                set_q(g.L['ffn1'].qg, slice(0, 4), b=0)
                set_var(g.L['ffn1'].bn_gamma, 1e-6, 5)
            rec = run_case('FFN expansion columns 0-3 (q_g b = 0) and 5 (gamma tiny)', cols, {})
            r.ck(f'{label}: binary fused-BN FFN expansion zero columns pruned [0, 1, 2, 3], rounded_to_zero [5]',
                 rec['zero_columns']['ffn1']['pruned'] == [0, 1, 2, 3] and rec['zero_columns']['ffn1']['rounded_to_zero'] == [5])
        r.ck(f'{label}: every layer had its named-route cases ({cases} layers)', cases == len(LAYERS))


# ----------------------------------------------------------------------------------------------
# G39


def cmd_g39(a, r):
    import numpy as np

    from cbl import ablate, structure

    X, _ = jets()
    for label in ('F000', 'F100'):
        m, _ = build(label)
        warm(m, X)
        g = ablate.Graph(m)
        ex = ablate.Exec(g, jit=True)
        snap = snapshot(m)
        sm = g.softmax
        restore(m, snap)
        zero_width(sm.oq, (0, 0, slice(0, 2), slice(None)))
        rec = structure.structure_record(m, X, ex=ex)
        r.ck(f'{label}: softmax zero-width fractions per head {rec["softmax_zero_width_fraction"]} = [0.25, 0.0] with '
             f'16 of 64 head-0 entries at zero width', rec['softmax_zero_width_fraction'] == [0.25, 0.0])
        restore(m, snap)
        zero_width(sm.oq, (0, 1))
        rec = structure.structure_record(m, X, ex=ex)
        r.ck(f'{label}: softmax zero-width fractions {rec["softmax_zero_width_fraction"]} = [0.0, 1.0] with head 1 at '
             f'zero width', rec['softmax_zero_width_fraction'] == [0.0, 1.0])
        restore(m, snap)
        rec = structure.structure_record(m, X, ex=ex)
        r.ck(f'{label}: unmodified: softmax fractions [0.0, 0.0]; Q not degenerate, q_identical_across_t and q_all_zero '
             f'null', rec['softmax_zero_width_fraction'] == [0.0, 0.0] and rec['degenerate']['Q'] is None
             and rec['q_identical_across_t'] is None and rec['q_all_zero'] is None)
        binary = ablate.is_binary(g.q)
        route = 'kernel_pruned' if binary else 'kernel_rounded_to_zero'

        def zero_q():
            if binary:
                kernel_widths_zero(g.q)
            else:
                set_var(g.q.kernel, 0.0)

        restore(m, snap)
        zero_q()
        set_var(g.q.bias, 0.3)
        rec = structure.structure_record(m, X, ex=ex)
        q = eager_taps(g, X[:7], ('Q',))['Q']
        ident = [bool(np.all(q[:, :, h, :] == q[:, :1, h, :])) for h in range(q.shape[2])]
        r.ck(f'{label}: zero Q kernel, bias 0.3: Q degenerate ({rec["degenerate"]["Q"]}), q_identical_across_t '
             f'{rec["q_identical_across_t"]} = {ident} from the taps, q_all_zero {rec["q_all_zero"]}',
             rec['degenerate']['Q'] == route and rec['q_identical_across_t'] == [True, True] == ident and rec['q_all_zero'] is False)
        set_q(g.q.oq, (0, 3, 0, slice(None)), f=1)
        rec = structure.structure_record(m, X, ex=ex)
        q = eager_taps(g, X[:7], ('Q',))['Q']
        ident = [bool(np.all(q[:, :, h, :] == q[:, :1, h, :])) for h in range(q.shape[2])]
        r.ck(f'{label}: and Q output format f = 1 at position 3, head 0: q_identical_across_t {rec["q_identical_across_t"]} '
             f'= [False, True] = {ident}', rec['q_identical_across_t'] == [False, True] == ident and rec['q_all_zero'] is False)
        restore(m, snap)
        zero_q()
        set_var(g.q.bias, 0.0)
        rec = structure.structure_record(m, X, ex=ex)
        r.ck(f'{label}: zero Q kernel, bias 0: q_all_zero {rec["q_all_zero"]}, q_identical_across_t {rec["q_identical_across_t"]}',
             rec['q_all_zero'] is True and rec['q_identical_across_t'] == [True, True] and rec['constant_zero']['Q'] is True)


# ----------------------------------------------------------------------------------------------
# G40


def cmd_g40(a, r):
    import keras
    import numpy as np
    from keras import ops

    from cbl import ablate

    X, y = jets()
    for label in ('CH-8', 'F000', 'F100', 'F111-FB', 'F111-BF'):
        m, _ = build(label)
        warm(m, X)
        g = ablate.Graph(m)
        eq = {k: True for k in ('model', 'mha', 'probs', 'Q', 'K', 'V', 'O', 'FFN', 'byte')}
        for sl in ablate.chunks(len(X), 2790):
            x = ops.convert_to_tensor(X[sl])
            ref = num(m(x, training=False))
            t = {}
            out = num(ablate.forward(g, x, taps=t))
            eq['model'] &= np.array_equal(ref, out)
            eq['byte'] &= ref.tobytes() == out.tobytes()
            h = g.L['embed'](x, training=False)
            xa = g.L['tanh1'](h, training=False)
            a_ref, sc_ref = m.layers[3](xa, xa, training=False, return_attention_scores=True)
            t2 = {}
            a_my = ablate.mha_forward(g, xa, xa, taps=t2)
            eq['mha'] &= num(a_ref).tobytes() == num(a_my).tobytes()
            eq['probs'] &= num(sc_ref).tobytes() == num(t2['probs']).tobytes()
            eq['Q'] &= num(g.q(xa, training=False)).tobytes() == num(t2['Q']).tobytes()
            eq['K'] &= num(g.k(xa, training=False)).tobytes() == num(t2['K']).tobytes()
            eq['V'] &= num(g.v(xa, training=False)).tobytes() == num(t2['V']).tobytes()
            eq['O'] &= num(a_ref).tobytes() == num(t2['O']).tobytes() and not g.mha_has_oq
            s = g.L['add1']([h, a_ref], training=False)
            f2 = g.L['ffn2'](g.L['ffn1'](g.L['tanh2'](s, training=False), training=False), training=False)
            eq['FFN'] &= num(f2).tobytes() == num(t['FFN']).tobytes()
        r.ck(f'{label}: eager re-expression equals model(x, training=False) bytewise in every chunk', eq['model'] and eq['byte'])
        r.ck(f'{label}: mha_forward equals the MHA layer\'s inference call bytewise, and its probabilities equal the '
             f'layer\'s returned attention scores', eq['mha'] and eq['probs'])
        r.ck(f'{label}: taps Q, K, V equal the projections called directly; O equals the MHA output (no MHA output '
             f'quantizer); FFN equals the contraction called on the layer chain', all(eq[k] for k in ('Q', 'K', 'V', 'O', 'FFN')))
        ref_all = eager_logits(m, X)
        ex = ablate.Exec(g, jit=True)
        jit_all = np.concatenate([p['logits'] for _, p in ex.passes('base', X)])
        r.ck(f'{label}: the jit pass (StatelessScope, variables as arguments) equals the eager forward bytewise',
             jit_all.tobytes() == ref_all.tobytes())
        g.L['embed'].kernel.assign(num(g.L['embed'].kernel) * 1.25)
        jit2 = np.concatenate([p['logits'] for _, p in ex.passes('base', X)])
        r.ck(f'{label}: after a variable changes, the same compiled pass follows it (equals the eager forward; '
             f'{ex.compiles} compilation)', jit2.tobytes() == eager_logits(m, X).tobytes() and ex.compiles == 1)
        pr = m.predict(X, batch_size=2790, verbose=0)
        r.info.setdefault('predict_equals_eager', {})[label] = bool(np.asarray(pr).tobytes() == eager_logits(m, X).tobytes())
    bad = keras.Sequential([keras.Input((8, 3)), keras.layers.Dense(5)])
    try:
        ablate.Graph(bad)
        refused = False
    except ablate.GraphError:
        refused = True
    r.ck('a foreign graph is refused (GraphError)', refused)
    m, _ = build('F000')
    g = ablate.Graph(m)
    try:
        ablate.forward(g, ops.convert_to_tensor(X[:10]), inject={'probs': lambda t, sl: t[:, :1]})
        refused = False
    except ValueError:
        refused = True
    r.ck('an injection of another shape is refused (ValueError)', refused)


# ----------------------------------------------------------------------------------------------
# G41


def cmd_g41(a, r):
    import numpy as np

    from cbl import ablate, front as F, rng

    X, y = jets()
    for label in ('CH-8', 'F000', 'F111-FB'):
        m, spec = build(label)
        warm(m, X)
        g = ablate.Graph(m)
        ex = ablate.Exec(g, jit=True)
        base = ablate.BaselinePass(ex, X, y)
        ok = True
        for kind in ('logits_identity', 'probs_identity', 'branch_identity'):
            bp = ablate.BaselinePass(ex, X, y, kind=kind)
            ok &= bp.logits_sha256 == base.logits_sha256 and bp.correct == base.correct and bp.overflow == base.overflow
        r.ck(f'{label}: identity injections at the logits, the probabilities and the branch give the baseline logits '
             f'bytewise, its correct count ({base.correct}) and its overflow counts', ok)
        T = eager_taps(g, X, ('branch', 'probs', 'logits_raw'))
        exact = {}
        for key, kind, tensor in (('delta_attn_mean', 'branch_const', 'branch'), ('delta_static', 'probs_const', 'probs'),
                                  ('delta_unif', 'logits_const', 'logits_raw')):
            c, over = ablate.ablated_pass(ex, X, y, kind, T[tensor])
            exact[key] = (c == base.correct, all(over[q] == base.overflow[q] for q in ablate.DOWNSTREAM[key]))
        r.ck(f'{label}: the replacement code paths of Delta_attn, Delta_static and Delta_unif, given each tensor itself '
             f'per jet, give Delta exactly 0 and the baseline overflow counts ({exact})',
             all(v[0] and v[1] for v in exact.values()))
        r.info.setdefault('baseline', {})[label] = {'correct': base.correct, 'n': base.n, 'H_attn': base.H_attn(),
                                                    'overflow': base.overflow, 'seconds': base.seconds}
        # in-loop equals the checkpoint audit
        d = Path(a.work) / label / 'run' / 'front'
        sv_fns = F.bind(m, rng.roles_of(m, spec), X, y, ablations='every_save')
        saver = F.FrontSaver(d, run_id='g41', config_sha256='c' * 64, code_sha='d' * 40, headline='H1', schedule='ramp',
                             save_fn=sv_fns[0], state_fn=sv_fns[1], inloop_fn=sv_fns[2], audit_fn=sv_fns[3],
                             ablations='every_save', keep_superseded=False)
        saver.on_frozen_record(0, {'epoch': 0, 'val_correct': 600, 'val_n': 1000, 'H1': 300000, 'H2': 300000, 'NR': 300000})
        inloop = json.loads(saver.attention_path(0).read_text())
        audit = ablate.audit_checkpoint(saver.dir.parent / saver.saved[0]['path'], X, y)
        dd = diff_paths(strip(inloop), strip(audit))
        r.ck(f'{label}: the in-loop attention diagnostics equal the checkpoint audit of the saved file exactly '
             f'(Deltas, counts, H_attn, overflow, widths){"" if not dd else f"; differ at {dd}"}', not dd)
        dj, _ = ablate.attention_diagnostics(m, X, y, ex=ex)
        de, _ = ablate.attention_diagnostics(m, X, y, ex=ablate.Exec(g, jit=False))
        dd = diff_paths(strip(dj), strip(de))
        r.ck(f'{label}: eager and jit attention diagnostics are equal exactly{"" if not dd else f"; differ at {dd}"}', not dd)
    # constructed K == 0 models
    for label in ('F000', 'F111-FB'):
        m, _ = build(label)
        warm(m, X)
        g = ablate.Graph(m)
        if ablate.is_binary(g.k):
            kernel_widths_zero(g.k)
        else:
            set_var(g.k.kernel, 0.0)
        ex = ablate.Exec(g, jit=True)
        diag, base = ablate.attention_diagnostics(m, X, y, ex=ex)
        P = eager_taps(g, X, ('probs', 'K'))
        const = bool(np.all(P['probs'] == P['probs'][:1])) and bool(np.all(P['K'] == 0))
        r.ck(f'{label} with K == 0: K is 0 and the probabilities are constant over the jets; their float64 mean equals '
             f'each jet bytewise', const and base.probs_mean.tobytes() == P['probs'][0].tobytes())
        r.ck(f'{label} with K == 0: Delta_unif and Delta_static exactly 0 (delta_correct {diag["delta_correct"]})',
             diag['delta_correct']['delta_unif'] == 0 and diag['delta_correct']['delta_static'] == 0
             and diag['delta_unif'] == 0.0 and diag['delta_static'] == 0.0)
    # H_attn on uniform rows
    m, _ = build('F000')
    warm(m, X)
    ex = ablate.Exec(ablate.Graph(m), jit=True)
    bp = ablate.BaselinePass(ex, X, y, kind='probs_const', const=np.full((2, 8, 8), 0.125, np.float32))
    r.ck(f'H_attn of the injected uniform rows is log 8 exactly ({bp.H_attn()!r} vs {math.log(8.0)!r}); every row used, '
         f'none all zero or negative ({bp.h_counts})', bp.H_attn() == math.log(8.0)
         and bp.h_counts['used'] == len(X) * 16 and bp.h_counts['all_zero_rows'] == 0 and bp.h_counts['negative_rows'] == 0
         and bool(np.all(bp.h_all == math.log(8.0))))
    rows = np.array([[0.125] * 8, [0.0] * 8, [0.5, 0.5] + [0.0] * 6, [-0.1, 1.1] + [0.0] * 6, [0.25] * 4 + [0.0] * 4])
    h, c = ablate.attention_entropy(rows)
    r.ck(f'attention_entropy on constructed rows: entropies {h.tolist()} = [log 8, log 2, log 4], one all-zero and one '
         f'negative row counted and left out ({c})', h.tolist() == [math.log(8.0), math.log(2.0), math.log(4.0)]
         and c == {'rows': 5, 'used': 3, 'all_zero_rows': 1, 'negative_rows': 1, 'nan_rows': 0})
    # timing reading on the full validation size (constructed jets)
    from cbl import structure

    Xb, yb = jets(62000, PRIVATE_SEED + 1)
    t0 = zv_t0 = time.time()
    structure.zero_variance_pass(ex, Xb)
    zv_t1 = time.time()
    structure.zero_variance_pass(ex, Xb)
    zv_t2 = time.time()
    dg, _ = ablate.attention_diagnostics(m, Xb, yb, ex=ex)
    t3 = time.time()
    r.info['timing_62000_jets'] = {'structure_pass_first': zv_t1 - zv_t0, 'structure_pass_second': zv_t2 - zv_t1,
                                   'attention_diagnostics': t3 - zv_t2, 'compiles': ex.compiles, 'since': t3 - t0,
                                   'label': 'engineering timing, not quotable'}


# ----------------------------------------------------------------------------------------------
# G42


def frac_round(x, f):
    """RND (round half up) of x at f fractional bits, exactly."""
    v = Fraction(float(x)) * (Fraction(2) ** int(f))
    return Fraction(math.floor(v + Fraction(1, 2))) / (Fraction(2) ** int(f))


def frac_overflow(x, k, i, f):
    """(live, overflow) of one entry, exactly (owner decision B8-10)."""
    k, i, f = int(k), int(i), int(f)
    if k + i + f <= 0:
        return False, False
    if not math.isfinite(float(x)):
        return True, True
    rr = frac_round(x, f)
    i2 = max(i, -f)
    lo = -k * Fraction(2) ** i2
    hi = Fraction(2) ** i2 - Fraction(2) ** (-f)
    return True, (rr < lo or rr > hi)


def frac_counts(x, k, i, f):
    import numpy as np

    x, k, i, f = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(k), np.asarray(i), np.asarray(f))
    live = over = 0
    for xv, kv, iv, fv in zip(x.ravel(), k.ravel(), i.ravel(), f.ravel()):
        lv, ov = frac_overflow(xv, kv, iv, fv)
        live += lv
        over += ov
    return {'live': live, 'overflow': over, 'masked': int(x.size) - live}


def kif_of(quantizer, shape):
    qz = quantizer.quantizer
    k, i, f = qz.kif
    return tuple(num(qz.bw_mapper.bw_to_x(v, shape)) for v in (k, i, f))


def cmd_g42(a, r):
    import numpy as np
    from hgq.quantizer import Quantizer, QuantizerConfig
    from keras import ops

    from cbl import ablate

    # unit: one KIF WRAP quantizer with six lanes of known formats
    conf = QuantizerConfig('kif', 'datalane', k0=1, i0=0, f0=3, round_mode='RND', overflow_mode='WRAP', homogeneous_axis=(0,))
    qq = Quantizer(conf)
    qq.build((None, 6))
    lanes = [(1, 0, 3), (0, 1, 2), (1, 2, 0), (1, -1, 2), (1, 0, -1), (1, 3, -2)]
    set_q(qq, k=np.array([[l[0] for l in lanes]], np.uint8), i=np.array([[l[1] for l in lanes]], np.float32),
          f=np.array([[l[2] for l in lanes]], np.float32))
    vals = []
    for k, i, f in lanes:
        i2 = max(i, -f)
        hi, lo, h = 2.0 ** i2 - 2.0 ** -f, -k * 2.0 ** i2, 2.0 ** (-f - 1)
        vals.append([hi, hi + h * 0.999, hi + h, lo, lo - h * 0.999, lo - h * 1.001, 0.0, 3.3, -3.3, 1e3, -1e3, hi + 1e-3])
    xs = np.array(vals, np.float32).T                       # (12, 6)
    xs = np.concatenate([xs, np.full((1, 6), np.nan, np.float32)])
    x = ops.convert_to_tensor(xs)
    want = frac_counts(xs, *kif_of(qq, x.shape))
    got_np = ablate.wrap_overflow(qq, x)
    got_t = dict(zip(ablate.COUNT_FIELDS, num(ablate.overflow_counts_t(qq, x)).tolist()))
    r.info['unit'] = {'want': want, 'numpy': got_np, 'traced': got_t}
    r.ck(f'unit quantizer (6 lanes, boundaries at half a step, a masked lane, NaN): overflow {got_np["overflow"]} = '
         f'range check {got_np["range_check"]} = exact {want["overflow"]}; masked {got_np["masked"]} = {want["masked"]}; '
         f'the in-pass counter agrees ({got_t["overflow"]})',
         got_np['overflow'] == got_np['range_check'] == want['overflow'] == got_t['overflow'] == got_t['range_check']
         and got_np['masked'] == want['masked'] == got_t['masked'] and got_np['nonfinite'] == 6 and want['overflow'] > 0)
    X, y = jets(600, PRIVATE_SEED + 2)
    # O's input quantizer: constant V, probabilities replaced by 3/8 per key
    m, _ = build('F000')
    warm(m, X)
    g = ablate.Graph(m)
    set_var(g.v.kernel, 0.0)
    pattern = np.where(np.arange(32).reshape(2, 16) % 2 == 0, 0.2, 0.5).astype(np.float32)
    set_var(g.v.bias, pattern)
    set_q(g.o.iq, k=1, i=0, f=7)
    ex = ablate.Exec(g, jit=True)
    base = ablate.BaselinePass(ex, X, y)
    c38 = np.full((2, 8, 8), 3 / 8, np.float32)
    corr, over = ablate.ablated_pass(ex, X, y, 'probs_const', c38)
    T = eager_taps(g, X, ('O.iq',), inject={'probs': lambda t, sl: ops.broadcast_to(ops.convert_to_tensor(c38), ops.shape(t))})
    exact = frac_counts(T['O.iq'], *kif_of(g.o.iq, T['O.iq'].shape))
    analytic = len(X) * 8 * 16
    r.ck(f'O\'s input quantizer, unablated: 0 overflow ({base.overflow["O.iq"]["overflow"]}); with probabilities 3/8 per '
         f'key: {over["O.iq"]["overflow"]} = exact {exact["overflow"]} = analytic {analytic} (jets x 8 positions x 16 of 32 '
         f'(head, dim) lanes at 3 x 0.5)', base.overflow['O.iq']['overflow'] == 0
         and over['O.iq']['overflow'] == over['O.iq']['range_check'] == exact['overflow'] == analytic)
    # the second tanh table's input quantizer and QAdd's input quantizer under branch replacement
    m, _ = build('F000')
    warm(m, X)
    g = ablate.Graph(m)
    set_q(g.L['tanh2'].iq, k=1, i=0, f=7)
    set_q(g.L['add1'].iq[1], k=1, i=4, f=7)
    ex = ablate.Exec(g, jit=True)
    c5 = np.full((8, 24), 5.0, np.float32)
    corr, over = ablate.ablated_pass(ex, X, y, 'branch_const', c5)
    T = eager_taps(g, X, ('tanh2.iq',), inject={'branch': lambda t, sl: ops.broadcast_to(ops.convert_to_tensor(c5), ops.shape(t))})
    exact = frac_counts(T['tanh2.iq'], *kif_of(g.L['tanh2'].iq, T['tanh2.iq'].shape))
    r.ck(f'branch replaced by 5: the second tanh table\'s input quantizer counts {over["tanh2.iq"]["overflow"]} = exact '
         f'{exact["overflow"]} (> 0); QAdd\'s branch input quantizer counts 0 ({over["add1.iq[1]"]["overflow"]})',
         over['tanh2.iq']['overflow'] == exact['overflow'] > 0 and over['add1.iq[1]']['overflow'] == 0)
    c100 = np.full((8, 24), 100.0, np.float32)
    corr, over = ablate.ablated_pass(ex, X, y, 'branch_const', c100)
    r.info['branch_100'] = over
    r.ck(f'branch replaced by 100: QAdd\'s branch input quantizer (range [-16, 16)) counts every entry '
         f'({over["add1.iq[1]"]["overflow"]} = {len(X) * 192}); the second tanh table\'s counts '
         f'{over["tanh2.iq"]["overflow"]} (the STUDY-named quantizer sees the wrapped value)',
         over['add1.iq[1]']['overflow'] == len(X) * 192)
    # rows with a negative entry: softmax output range [-0.5, 0.5), peaked rows on even jets (logit 10 on key 0, so
    # the peak probability is about 1 / (1 + 7 exp(-10/4)) = 0.63, which wraps), zero logits on odd jets
    m, _ = build('F000')
    warm(m, X)
    g = ablate.Graph(m)
    set_q(g.softmax.oq, k=1, i=-1, f=7)
    set_q(g.softmax.exp_table.iq, i=4)          # the exp table's SAT input range reaches the peak logit 10
    ex = ablate.Exec(g, jit=True)
    L = np.zeros((len(X), 2, 8, 8), np.float32)
    L[0::2, :, :, 0] = 10.0
    bp = ablate.BaselinePass(ex, X, y, kind='logits_const', const=L)
    sm = g.softmax
    pre = num(type(sm).call.__wrapped__(sm, ops.convert_to_tensor(L), training=False))
    post = num(sm.oq(ops.convert_to_tensor(pre), training=False))
    P = eager_taps(g, X, ('probs',), inject={'logits': lambda t, sl: ops.convert_to_tensor(L[sl])})['probs']
    k_, i_, f_ = kif_of(sm.oq, pre.shape)
    neg_exact = 0
    for rr in range(pre.shape[0]):
        for h in range(2):
            for qi in range(8):
                row = False
                for kk in range(8):
                    lv, ov = frac_overflow(pre[rr, h, qi, kk], k_[rr, h, qi, kk], i_[rr, h, qi, kk], f_[rr, h, qi, kk])
                    rv = frac_round(pre[rr, h, qi, kk], f_[rr, h, qi, kk])
                    row |= lv and ov and rv > 0
                neg_exact += row
    r.ck(f'negative rows: the output quantizer applied to QSoftmax\'s unwrapped call equals the probabilities of the pass '
         f'bytewise; {bp.h_counts["negative_rows"]} rows with a negative entry = exact count of rows whose rounded value '
         f'wraps past 0.5 ({neg_exact}) = {len(X) // 2 * 16} (even jets x 2 heads x 8 queries)',
         post.tobytes() == P.tobytes() and bp.h_counts['negative_rows'] == neg_exact == len(X) // 2 * 16
         and bool(np.all((P < 0).any(axis=-1)[0::2])) and not bool((P < 0).any(axis=-1)[1::2].any()))
    r.ck(f'negative rows are left out of H_attn; the remaining uniform rows give log 8 exactly ({bp.H_attn()!r})',
         bp.h_counts['used'] == len(X) // 2 * 16 and bp.H_attn() == math.log(8.0))


COMMANDS = {'g37': cmd_g37, 'g38': cmd_g38, 'g39': cmd_g39, 'g40': cmd_g40, 'g41': cmd_g41, 'g42': cmd_g42}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('cmd', choices=sorted(COMMANDS))
    ap.add_argument('--device', choices=['cpu', 'gpu'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    a = ap.parse_args()
    Path(a.work).mkdir(parents=True, exist_ok=True)
    r = Rec()
    rec = {'cmd': a.cmd, 'args': vars(a)}
    code = 0
    try:
        st = setup(a.device)
        import jax
        rec['setup'] = {'matmul_precision': str(jax.config.jax_default_matmul_precision),
                        'setup_precision': st.get('matmul_precision'), 'jax_backend': jax.default_backend(),
                        'devices': [str(d) for d in jax.devices()], 'XLA_FLAGS': os.environ.get('XLA_FLAGS')}
        t0 = time.time()
        COMMANDS[a.cmd](a, r)
        rec['seconds'] = time.time() - t0
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
