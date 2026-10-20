"""Gate G03 (PREFLIGHT-PLAN section 3; STUDY v0.8 section 7.3 TF32 override and section 13 image
row; module M1 cbl/env.py). Run once with --device cpu (PCC) and once with --device gpu (PCG).

Checks, in the venv of record:
1. Precision record of `cbl.env.setup()`: `jax_default_matmul_precision` is None before
   `import model`, 'tensorfloat32' after it (model.py:20-23), 'float32' after the override.
2. The backend is the one asked for (JAX_PLATFORMS passed by this gate; on the GPU the device is
   the RTX 4060 Ti) and the stack is jax 0.9.2, keras 3.14.0, hgq2 0.1.9.
3. CH-8 is built with Chang's `get_model('xfm', 7, 7, 1e-8, 8, True)` (run_train.py:76 with
   -m xfm -n 8 -f 3), compiled as run_train.py:97-100, and fitted for --epochs epochs of 4 steps
   on synthetic data of Chang's input shape (8 constituents, 3 features, 5 classes; a private
   NumPy generator) through hgq's Dataset with the device this gate passes (decision U2). The
   precision is 'float32' after the build, at the start of fit, at every epoch end, at the end of
   fit and after fit returns; each reading is logged.
4. Lowered StableHLO of the model's own jitted computations (the inference forward, and the
   training-mode loss with its gradient): every dot_general carries precision HIGHEST, the
   lowering of 'float32'. Negative control: the same lowerings traced inside an explicit
   'tensorfloat32' context carry HIGH, so the check can fail.
5. Numerical check on the model's own einsum path, for every QEinsumDense in the built model
   (the FFN contraction, the logits layer and the four MHA projections). The path checked is the
   layer's own `QEinsumDense.call` (kernel quantizer, input quantizer, `ops.einsum`, bias), taken
   from under hgq's class-level wrapper (`base.py:57-114`), which adds only the EBOPs bookkeeping
   in training mode and the output quantizer. Its output on fixed inputs is compared with a
   float64 NumPy einsum of the layer's own quantized input and quantized kernel (plus quantized
   bias), cast to float32. A structural check confirms that the full layer call equals the output
   quantizer applied to that path (Q, K and V have one; the others return it unchanged).
   E1, the layer as fitted: max relative error <= 1e-5. Its quantized operands carry few
     significant bits, so TF32 and float32 may agree exactly there; the bit counts are recorded.
   E2, the same layer with its kernel quantizer set to b=22, i=2 and its input quantizer to
     i=4, f=18, so that both operands carry more than 11 significant bits (TF32 keeps 11):
     max relative error <= 1e-5 under the active precision. Negative control on the GPU: the
     same path traced inside an explicit 'tensorfloat32' context must, for each layer, either be
     at least 10 times worse and >= 1e-5, or be bitwise equal to the active output (XLA then
     computes that dot in full float32 whatever the precision asks, as it does for the small
     64x32x5 logits contraction); at least one layer must separate. On the CPU there are no TF32
     units, and the control is recorded only. (Changed during development, before any run of
     record: the first draft required every layer to separate, which assumed that XLA honours
     a TF32 request for every dot shape.)
   The tolerance 1e-5 is a float32 accumulation bound for contractions of length <= 32
   (32 * 2**-24 * a factor of 5 for cancellation), fixed before the first run.
6. The 512 x 512 jnp.matmul check of `cbl.env.matmul_check()` (context only).
7. `git status --porcelain` is empty at start and at end.

Run:  <venv>/bin/python -I -B gates/g03_precision.py --device cpu|gpu --expect-sha12 SHA12 [--json PATH]
"""

import argparse
import logging
import random
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import gatelib  # noqa: E402

GATE = 'G03'
DATA_SEED = 20261018
BATCH = 2790  # run_train.py:67, :74
TOL_ACTIVE = 1e-5
CONTROL_RATIO = 10.0
CONTROL_MIN = 1e-5
E2_KQ = {'b': 22.0, 'i': 2.0}
E2_IQ = {'i': 4.0, 'f': 18.0}
TF32_SIG_BITS = 11
EXPECT = {'jax': '0.9.2', 'jaxlib': '0.9.2', 'keras': '3.14.0', 'hgq2': '0.1.9'}

DOT_RE = re.compile(r'(stablehlo\.dot_general|stablehlo\.dot\b|"stablehlo\.dot_general"|"stablehlo\.dot")')
PREC_RE = re.compile(r'precision = \[([A-Z_]+), ([A-Z_]+)\]|#stablehlo<precision ([A-Z_]+)>')


def dot_precisions(text):
    """Precision of every dot op in a StableHLO module text, as a sorted list of strings."""
    out = []
    for line in text.splitlines():
        if not DOT_RE.search(line):
            continue
        m = PREC_RE.findall(line)
        if not m:
            out.append('none' if 'algorithm' not in line else 'algorithm')
            continue
        vals = []
        for a, b, c in m:
            vals += [v for v in (a, b, c) if v]
        out.append('/'.join(vals))
    return out


def sig_bits(arr):
    """Largest number of significant bits of any nonzero float32 value in arr."""
    import numpy as np

    v = np.asarray(arr, dtype=np.float64).ravel()
    v = v[v != 0]
    if v.size == 0:
        return 0
    m, _ = np.frexp(np.abs(v))
    n = np.round(m * 2.0 ** 24).astype(np.int64)
    low = n & -n
    return int(np.max(24 - np.log2(low).astype(np.int64)))


def input_shape_for(layer, batch=64):
    shape = None
    d = getattr(layer, '_build_shapes_dict', None)
    if isinstance(d, dict) and len(d) == 1:
        shape = tuple(next(iter(d.values())))
    if shape is None:
        raise RuntimeError('no build shape for %s' % layer.name)
    return tuple(batch if s is None else int(s) for s in shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--device', choices=['cpu', 'gpu'], required=True)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--expect-sha12', default=None)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    g = gatelib.Gate(GATE)
    rec = {'header': g.header(a.expect_sha12), 'device': a.device}
    gatelib.check_venv_of_record(g)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='LOG %(name)s %(message)s', force=True)

    from cbl import env

    jp = 'cpu' if a.device == 'cpu' else 'cuda'
    st = env.setup(mem_fraction='0.5', jax_platforms=jp)
    mp = st['matmul_precision']
    rec['setup'] = st
    g.info('matmul precision record', mp)
    g.check(mp['before_import_model'] is None, 'precision before import model is None')
    g.check(mp['after_import_model'] == 'tensorfloat32', "precision after import model is 'tensorfloat32' (model.py:20-23)")
    g.check(mp['after_override'] == 'float32', "precision after the override is 'float32'")

    prov = env.provenance()
    rec['provenance'] = prov
    g.info('provenance', {k: prov[k] for k in ('jax_backend', 'jax_devices', 'device_kind', 'env', 'packages')})
    if a.device == 'cpu':
        g.check(prov['jax_backend'] == 'cpu', 'JAX backend is cpu')
    else:
        g.check(prov['jax_backend'] == 'gpu' and all('4060 Ti' in (k or '') for k in prov['device_kind']),
                'JAX backend is gpu on the RTX 4060 Ti %s' % prov['device_kind'])
    for k, v in EXPECT.items():
        g.check(prov['packages'].get(k) == v, '%s %s' % (k, v))

    import jax
    import keras
    import numpy as np
    from hgq.layers import QEinsumDense
    from hgq.utils.sugar import Dataset

    chang_model = sys.modules['model']
    np.random.seed(42)  # run_train.py:41-42
    random.seed(42)
    from cbl import lossguard

    model = chang_model.get_model('xfm', 7, 7, 1e-8, 8, True)
    lossguard.guard(model)
    g.check(env.log_matmul_precision('after get_model')['jax_default_matmul_precision'] == 'float32', "precision after get_model is 'float32'")

    rng = np.random.default_rng(DATA_SEED)
    n_train, n_val = 4 * BATCH, BATCH
    x_tr = rng.standard_normal((n_train, 8, 3)).astype(np.float32)
    y_tr = rng.integers(0, 5, n_train).astype(np.int32)
    x_va = rng.standard_normal((n_val, 8, 3)).astype(np.float32)
    y_va = rng.integers(0, 5, n_val).astype(np.int32)
    ds_device = 'cpu:0' if a.device == 'cpu' else 'gpu:0'  # decision U2: the caller passes it, as run_train.py:73-74
    d_tr = Dataset(x_tr, y_tr, BATCH, ds_device, shuffle=True)
    d_va = Dataset(x_va, y_va, BATCH, ds_device)
    g.info('dataset device', ds_device)

    readings = []

    class PrecisionLog(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            readings.append(env.log_matmul_precision('fit on_train_begin'))

        def on_epoch_end(self, epoch, logs=None):
            readings.append(env.log_matmul_precision('fit on_epoch_end %d' % epoch))

        def on_train_end(self, logs=None):
            readings.append(env.log_matmul_precision('fit on_train_end'))

    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'], steps_per_execution=4)  # run_train.py:97-100
    hist = model.fit(d_tr, epochs=a.epochs, validation_data=d_va, callbacks=[PrecisionLog()], verbose=0)
    rec['fit_history'] = {k: [float(x) for x in v] for k, v in hist.history.items()}
    g.info('fit history (synthetic data, engineering only)', rec['fit_history'])
    rec['precision_readings'] = readings
    g.check(len(readings) == a.epochs + 2 and all(r['jax_default_matmul_precision'] == 'float32' for r in readings),
            "precision is 'float32' at fit begin, at each of %d epoch ends and at fit end" % a.epochs)
    after = env.log_matmul_precision('after fit')
    g.check(after['jax_default_matmul_precision'] == 'float32', "precision after fit is 'float32'")
    devs = sorted({str(d) for v in model.variables for d in v.value.devices()})
    g.info('variable devices after fit', devs)
    g.check(len(devs) > 0 and all((d.startswith('cuda') if a.device == 'gpu' else 'cpu' in d.lower()) for d in devs),
            'all model variables live on the %s %s' % (a.device, devs))

    # 4. StableHLO precision of the model's own jitted computations
    tv = [v.value for v in model.trainable_variables]
    ntv = [v.value for v in model.non_trainable_variables]
    xb, yb = x_va[:64], y_va[:64]
    loss_obj = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def fwd(tv, ntv, x):
        y, _ = model.stateless_call(tv, ntv, x, training=False)
        return y

    def train_loss(tv, ntv, x, y):
        logits, ntv2 = model.stateless_call(tv, ntv, x, training=True)
        return loss_obj(y, logits), ntv2

    def lowerings():
        return {
            'inference_forward': jax.jit(fwd).lower(tv, ntv, xb).as_text(),
            'training_value_and_grad': jax.jit(jax.value_and_grad(train_loss, has_aux=True)).lower(tv, ntv, xb, yb).as_text(),
        }

    low = lowerings()
    with jax.default_matmul_precision('tensorfloat32'):
        low_tf32 = lowerings()
    hlo = {}
    for key in low:
        p, q = dot_precisions(low[key]), dot_precisions(low_tf32[key])
        hlo[key] = {'n_dots': len(p), 'active': sorted(set(p)), 'explicit_tf32': sorted(set(q)),
                    'sample_dot': next((ln.strip() for ln in low[key].splitlines() if DOT_RE.search(ln)), None)}
        g.info('stablehlo ' + key, hlo[key])
        g.check(len(p) > 0 and set(p) == {'HIGHEST/HIGHEST'}, '%s: all %d dot ops carry precision HIGHEST' % (key, len(p)))
        g.check(len(q) == len(p) and set(q) == {'HIGH/HIGH'}, '%s, negative control: under an explicit tensorfloat32 context all dot ops carry HIGH' % key)
    rec['stablehlo'] = hlo

    # 5. numerical check on every QEinsumDense of the built model
    layers = [ly for ly in model._flatten_layers() if type(ly) is QEinsumDense]
    g.info('QEinsumDense layers', [(ly.name, ly.equation, tuple(ly.kernel.shape)) for ly in layers])
    g.check(len(layers) == 6, 'the built CH-8 has 6 QEinsumDense layers (FFN, logits, query, key, value, attention_output)')
    xr = np.random.default_rng(DATA_SEED + 1)
    linear = keras.activations.get(None)

    def conv(t):
        return np.asarray(keras.ops.convert_to_numpy(t))

    def reference(ly, x):
        qx = conv(ly.iq(x, training=False)) if ly.enable_iq else x
        qk = conv(ly.qkernel)
        ref = np.einsum(ly.equation, qx.astype(np.float64), qk.astype(np.float64))
        if ly.bias is not None:
            ref = ref + conv(ly.qbias).astype(np.float64)
        return ref, qx, qk

    def rel(y, ref):
        return float(np.max(np.abs(np.asarray(y, np.float64) - ref.astype(np.float32).astype(np.float64))) / np.max(np.abs(ref)))

    num = []
    for ly in layers:
        row = {'layer': ly.path if hasattr(ly, 'path') else ly.name, 'equation': ly.equation}
        g.check(ly.activation in (None, linear), '%s has no activation' % row['layer'])
        shape = input_shape_for(ly)
        x = np.clip(xr.standard_normal(shape) * 2.0, -15.0, 15.0).astype(np.float32)
        row['input_shape'] = shape
        raw = getattr(type(ly).call, '__wrapped__', None)
        g.check(raw is not None and raw.__qualname__ == 'QEinsumDense.call',
                "%s: the layer's own QEinsumDense.call is reachable under hgq's wrapper" % row['layer'])
        row['enable_oq'] = bool(ly.enable_oq)
        # E1: as fitted
        ref, qx, qk = reference(ly, x)
        y_eager = conv(raw(ly, x, training=False))
        y_jit = np.asarray(jax.jit(lambda t: raw(ly, t, training=False))(x))
        full = conv(ly(x, training=False))
        want = conv(ly.oq(y_eager, training=False)) if ly.enable_oq else y_eager
        row['full_call_equals_oq_of_path'] = bool(np.array_equal(full, want))
        g.check(row['full_call_equals_oq_of_path'], '%s: full layer call equals %s of the checked path' % (
            row['layer'], 'the output quantizer' if ly.enable_oq else 'the unchanged output'))
        row['E1'] = {'rel_eager': rel(y_eager, ref), 'rel_jit': rel(y_jit, ref), 'sig_bits_input': sig_bits(qx),
                     'sig_bits_kernel': sig_bits(qk), 'eager_equals_jit': bool(np.array_equal(y_eager, y_jit))}
        g.check(row['E1']['rel_eager'] <= TOL_ACTIVE and row['E1']['rel_jit'] <= TOL_ACTIVE,
                '%s E1 (as fitted): rel err eager %.3g, jit %.3g <= %g' % (row['layer'], row['E1']['rel_eager'], row['E1']['rel_jit'], TOL_ACTIVE))
        # E2: widened quantizers
        kq, iq = ly.kq.quantizer, ly.iq.quantizer
        kinds = (type(kq).__name__, type(iq).__name__)
        row['quantizer_types'] = kinds
        ok_types = g.check(kinds == ('FixedPointQuantizerKBI', 'FixedPointQuantizerKIF'), '%s quantizers are KBI (kernel) and KIF (input) %s' % (row['layer'], kinds))
        if ok_types:
            for var, val in ((kq._b, E2_KQ['b']), (kq._i, E2_KQ['i']), (iq._i, E2_IQ['i']), (iq._f, E2_IQ['f'])):
                var.assign(np.full(var.shape, val, dtype=keras.backend.standardize_dtype(var.dtype)))
            ref, qx, qk = reference(ly, x)
            y_act = np.asarray(jax.jit(lambda t: raw(ly, t, training=False))(x))
            with jax.default_matmul_precision('tensorfloat32'):
                y_tf = np.asarray(jax.jit(lambda t: raw(ly, t, training=False) + 0.0)(x))
            row['E2'] = {'rel_active': rel(y_act, ref), 'rel_explicit_tf32': rel(y_tf, ref), 'sig_bits_input': sig_bits(qx),
                         'sig_bits_kernel': sig_bits(qk)}
            e2 = row['E2']
            g.check(e2['sig_bits_input'] > TF32_SIG_BITS and e2['sig_bits_kernel'] > TF32_SIG_BITS,
                    '%s E2 operands carry more than %d significant bits (input %d, kernel %d)' % (row['layer'], TF32_SIG_BITS, e2['sig_bits_input'], e2['sig_bits_kernel']))
            g.check(e2['rel_active'] <= TOL_ACTIVE, '%s E2 active precision: rel err %.3g <= %g' % (row['layer'], e2['rel_active'], TOL_ACTIVE))
            e2['tf32_equals_active'] = bool(np.array_equal(y_tf, y_act))
            e2['tf32_separates'] = e2['rel_explicit_tf32'] >= max(CONTROL_RATIO * e2['rel_active'], CONTROL_MIN)
            if a.device == 'gpu':
                g.check(e2['tf32_separates'] or e2['tf32_equals_active'],
                        '%s E2 negative control: explicit tensorfloat32 rel err %.3g %s' % (
                            row['layer'], e2['rel_explicit_tf32'],
                            '>= max(%g x %.3g, %g)' % (CONTROL_RATIO, e2['rel_active'], CONTROL_MIN) if e2['tf32_separates']
                            else 'is bitwise equal to the active output (this dot runs in float32 under either request)'
                            if e2['tf32_equals_active'] else 'neither separates nor equals the active output'))
            else:
                g.info('%s E2 explicit tensorfloat32 on CPU (no TF32 units; recorded only)' % row['layer'], e2['rel_explicit_tf32'])
        num.append(row)
        g.info('einsum ' + row['layer'], row)
    rec['einsum_checks'] = num
    if a.device == 'gpu':
        sep = [r['layer'] for r in num if r.get('E2', {}).get('tf32_separates')]
        g.check(len(sep) >= 1, 'E2 negative control separates for at least one layer on the GPU: %s' % sep)

    mc = env.matmul_check()
    rec['matmul_check'] = mc
    g.info('jnp.matmul 512 check (context)', mc)
    rec['precision_final'] = env.log_matmul_precision('gate end')
    g.check(rec['precision_final']['jax_default_matmul_precision'] == 'float32', "precision at gate end is 'float32'")
    g.check_clean('at end')
    return g.finish(rec, a.json)


if __name__ == '__main__':
    sys.exit(main())
