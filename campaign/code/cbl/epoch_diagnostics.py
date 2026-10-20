"""M8/M13 per-epoch diagnostics, separate from the M7 frozen-count evaluation.

All forward passes use explicit stateless variable inputs. Training-mode validation
and the last-step FB observation discard all returned updates. No model variables,
metrics, optimizer slots or random streams are added. Python call observers exist
only while tracing this diagnostic function and are restored even on an exception.
The real trainer step is delegated unchanged; FB replay uses its PRE-update state.
"""
from contextlib import contextmanager
import functools
import numpy as np


def width_summary(bits, fractional_bits):
    b, f = np.asarray(bits), np.asarray(fractional_bits)
    steps = np.exp2(-f.astype(np.float64))
    return {'width_mean': float(b.mean()), 'width_max': float(b.max()),
            'step_min': float(steps.min()), 'step_max': float(steps.max())}


def bn_summary(gamma, mean, variance):
    g, m, v = map(np.asarray, (gamma, mean, variance))
    return {'gamma_min': float(g.min()), 'gamma_max': float(g.max()),
            'moving_mean_absmax': float(np.abs(m).max()),
            'moving_var_min': float(v.min()), 'moving_var_max': float(v.max())}


class Moments:
    """Float64 parallel population moments, preserving channels and combining uneven chunks."""
    def __init__(self):
        self.n, self.mean, self.m2 = 0, None, None

    def add(self, n, mean, variance):
        mean, variance = np.asarray(mean, np.float64), np.asarray(variance, np.float64)
        if self.n == 0:
            self.n, self.mean, self.m2 = n, mean.copy(), variance * n
        else:
            d = mean - self.mean
            total = self.n + n
            self.m2 += variance * n + d * d * (self.n * n / total)
            self.mean += d * (n / total)
            self.n = total

    @property
    def variance(self):
        return self.m2 / self.n


def moment_difference(raw, binary):
    return {'moment_mismatch_mean': float(np.mean(np.abs(raw.mean - binary.mean))),
            'moment_mismatch_var': float(np.mean(np.abs(raw.variance - binary.variance)))}


@contextmanager
def observed_calls(observers):
    """Instance-local wrappers, removed after the stateless trace; no persistent model patch."""
    saved = []
    try:
        for layer, observer in observers:
            own = layer.__dict__.get('call')
            original = layer.call
            @functools.wraps(original)
            def call(*args, _original=original, _observer=observer, **kwargs):
                value = _original(*args, **kwargs)
                _observer(args, kwargs, value)
                return value
            saved.append((layer, own))
            object.__setattr__(layer, 'call', call)
        yield
    finally:
        for layer, own in reversed(saved):
            if own is None:
                object.__delattr__(layer, 'call')
            else:
                object.__setattr__(layer, 'call', own)


def quantizer_entries(model, top_names):
    from hgq.quantizer import Quantizer
    from cbl.rng import segment_name
    out = []
    def walk(layer, path, activation=False):
        if isinstance(layer, Quantizer):
            if layer.built:
                out.append((path, layer, activation))
            return
        counts = {}
        for sub in layer._layers:
            seg = segment_name(layer.name, sub.name, type(sub).__name__, counts)
            walk(sub, path + '.' + seg, activation or sub is getattr(layer, '_iq', None)
                 or sub is getattr(layer, '_oq', None) or seg in ('iq', 'oq') or seg.startswith('iq['))
    for name, ly in zip(top_names, model.layers):
        walk(ly, name)
    return out


class EpochDiagnostics:
    """M8 state summaries and validation observations; M13 last-step FB log producer."""
    def __init__(self, model, entries, top_names, x_val, y_val, batch_size):
        import jax
        self.model, self.entries = model, entries
        self.quantizers = quantizer_entries(model, top_names)
        self.x, self.y, self.batch = x_val, y_val, batch_size
        self.binary = [e for e in entries if e.kind == 'binary']
        self.fused = [e for e in self.binary if hasattr(e.layer, 'moving_variance')]
        self.fb = [e for e in self.fused if e.layer.form == 'FB']
        self.infer = jax.jit(self._inference)
        self.train = jax.jit(lambda tv, ntv, x: model.stateless_call(tv, ntv, x, training=True)[0])

    def state(self):
        from cbl.structure import num, quantizer_state, epoch_structure, pruning_channels, mask_zero_width
        widths, bn, zero = {}, {}, {}
        for path, q, activation in self.quantizers:
            layer, name = path.rsplit('.', 1)
            widths.setdefault(layer, {})[name] = width_summary(num(q.bits), num(q.fbits))
            if activation:
                shape = tuple(q._build_shapes_dict['input_shape'])
                shape = tuple(1 if d is None else d for d in shape)
                zero[path] = float(mask_zero_width(q, shape).mean())
        for e in self.entries:
            ly = e.layer
            if hasattr(ly, 'moving_variance'):
                bn[e.name] = bn_summary(num(ly.bn_gamma) if ly.scale else np.ones(1),
                                        num(ly.moving_mean), num(ly.moving_variance))
        return {'quantizers': widths, 'bn': bn,
                'quantizer_structure': epoch_structure(quantizer_state(self.model)),
                'pruning_channel': pruning_channels(self.entries)}, zero

    def _inference(self, tv, ntv, x):
        import keras
        from keras import ops
        stats, observers = {}, []
        for path, q, activation in self.quantizers:
            if activation:
                def obs(args, kwargs, out, path=path):
                    stats[path] = (ops.min(out), ops.max(out))
                observers.append((q, obs))
        binary = {}
        # Read the same inference inputs, qkernel and qbias inside the stateless scope.
        for e in self.binary:
            ly = e.layer
            def obs(args, kwargs, out, e=e, ly=ly):
                xx = ly.iq(args[0], training=False) if ly.enable_iq else args[0]
                w = ops.einsum(ly.equation, xx, ly.kernel)
                k = ops.einsum(ly.equation, xx, ly.qkernel)
                # Kernel output axes are channels; all leading batch/position axes are observations.
                from cbl.cost import contraction
                nc, _, _ = contraction(ly, e.shapes[0])
                nout = len(ly.kernel.shape) - nc
                axes = tuple(range(len(k.shape) - nout))
                count = int(np.prod([k.shape[a] for a in axes]))
                wm, wv = ops.moments(w, axes)
                km, kv = ops.moments(k, axes)
                pre = k + ly.qbias if ly.bias is not None else k
                binary[e.name] = (count, wm, wv, km, kv, ops.all(pre <= 0, axis=axes))
            observers.append((ly, obs))
        mapping = list(zip(self.model.trainable_variables, tv)) + list(zip(self.model.non_trainable_variables, ntv))
        with keras.StatelessScope(state_mapping=mapping), observed_calls(observers):
            logits = self.model(x, training=False)
        return logits, stats, binary

    def fb_replay(self, tv, ntv, x):
        """Last training batch, original pre-update weights; compare batch vs updated moving moments."""
        import keras
        from keras import ops
        from cbl.binary import SCALE
        from keras.src.backend.config import epsilon
        values, observers = {}, []
        for e in self.fb:
            ly = e.layer
            def obs(args, kwargs, out, ly=ly, name=e.name):
                xx = ly.iq(args[0], training=True) if ly.enable_iq else args[0]
                raw = ops.einsum(ly.equation, xx, ly.kernel)
                _, var = ops.moments(raw, ly._reduction_axes, keepdims=False, synchronized=ly.synchronized)
                gamma = ly.bn_gamma if ly.scale else 1
                batch = SCALE.parts(ly, ly.kernel, gamma / ops.sqrt(var + epsilon()), training=True)['S']
                moving = SCALE.parts(ly, ly.kernel, ly.frozen_scaler(), training=False)['S']
                values['b18_fb/' + name] = ops.mean(ops.cast(batch != moving, 'float32'))
            observers.append((ly, obs))
        mapping = list(zip(self.model.trainable_variables, tv)) + list(zip(self.model.non_trainable_variables, ntv))
        with keras.StatelessScope(state_mapping=mapping), observed_calls(observers):
            self.model(x, training=True)
        return values

    def install_train_observer(self):
        """M13 delegates the real step byte-for-byte; replay adds scalar last-batch logs only."""
        if not self.fb:
            return
        from keras.src.trainers.data_adapters.data_adapter_utils import unpack_x_y_sample_weight
        original = self.model.train_step
        def train_step(state, data):
            logs, new_state = original(state, data)
            x, _, _ = unpack_x_y_sample_weight(data)
            return {**logs, **self.fb_replay(state[0], state[1], x)}, new_state
        object.__setattr__(self.model, 'train_step', train_step)

    def evaluate(self, frozen_top1, logs):
        from cbl.structure import num
        result, zero = self.state()
        tv = [v.value for v in self.model.trainable_variables]
        ntv = [v.value for v in self.model.non_trainable_variables]
        ranges, moments, dead = {}, {}, {}
        train_correct, frozen_correct = 0, 0
        for lo in range(0, len(self.x), self.batch):
            hi = min(lo + self.batch, len(self.x))
            x = self.x[lo:hi]
            pred, stats, bs = self.infer(tv, ntv, x)
            labels = num(self.y[lo:hi])
            if labels.ndim == 2:
                labels = labels.argmax(-1)
            frozen_correct += int(np.sum(num(pred).argmax(-1) == labels))
            if self.fused:
                train_correct += int(np.sum(num(self.train(tv, ntv, x)).argmax(-1) == labels))
            for name, (low, high) in stats.items():
                low, high = float(low), float(high)
                old = ranges.get(name, (low, high))
                ranges[name] = min(low, old[0]), max(high, old[1])
            for name, (n, wm, wv, km, kv, d) in bs.items():
                a, b = moments.setdefault(name, (Moments(), Moments()))
                a.add(int(n), num(wm), num(wv)); b.add(int(n), num(km), num(kv))
                dead[name] = num(d) if name not in dead else dead[name] & num(d)
        if frozen_correct / len(self.x) != frozen_top1:
            raise ValueError('diagnostic inference disagrees with frozen validation')
        result['activations'] = {n: {'min': v[0], 'max': v[1], 'zero_width_frac': zero[n]} for n, v in ranges.items()}
        sites = {n: moment_difference(*v) for n, v in moments.items()}
        for e in self.fused:
            sites[e.name]['dead_channel_frac'] = float(dead[e.name].mean())
        for e in self.fb:
            sites[e.name]['fb_sign_diff_frac'] = float(logs['b18_fb/' + e.name])
        result['binary_sites'] = sites
        if self.fused:
            result['train_frozen_top1_gap'] = train_correct / len(self.x) - frozen_top1
        return result
