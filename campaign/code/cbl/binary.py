"""M6: the binary layers of STUDY sections 2.2 and 2.3, with their scale quantizer q_g and penalty (b).

STUDY means `review/snapshots/v0.10/STUDY.md` (sha256 7180c24f6b55807e...); "plan" means PREFLIGHT-PLAN.md
(sha256 2f1be9f3...30ff8), section 2 row M6, section 8 decision U3 and section 9 item 8. Gates G16-G24,
with G15 (single wrap), G10 and G11 (binary arms) and B7's G26-G34 reading these layers. Installed means
hgq2 0.1.9 in `~/lab/venvs/cbl1016-jax2`.

Design status. The scale treatment is STUDY decision OD-13 (a learned-width HGQ quantizer q_g on the
exported magnitude; in BF quantized after the fold), adopted under Kai's delegation and not yet
confirmed by him (STUDY section 19.2 item 2). Everything that depends on that choice is in one class,
`LearnedWidthScale`, used through the module object `SCALE`: how q_g is created and built (configuration
and granularity), how the exported magnitude is formed and quantized in each of the three maps (non-BN,
FB, BF), its bits for the penalty, and the per-site record. A different scale treatment replaces that
class and nothing else.

The map (STUDY section 2.2), for the tensor T presented to it:
    alpha = mean(T); S = Sign(T - alpha) with Sign(0) = -1; beta = mean(|T|), uncentered;
    K = q_g(stop_gradient(g)) * S + (T - stop_gradient(T))       (OD-16, exact forward, dK/dT = I)
- non-BN sites (Q, K, V, O, the FFN contraction): T = W, g = beta per layer, q_g scalar;
- FB (fold then binarize): T = U = W * s (s_c = gamma_c / sqrt(v_c + eps), Chang's
  einsum_dense_batchnorm.py:165-167), g_c = beta_c = mean_i |U_ic|, q_g per output column;
- BF (binarize then fold): S and beta from W, g_c = beta * |s_c|, and
    K_ic = q_g(stop_gradient(beta) * |s_c|) * stop_gradient(sign(s_c) * S_ic) + s_c * (W_ic - stop_gradient(W_ic));
- GR (STUDY section 5, OD-7): the same maps without the stop_gradient on the scale's input (on beta
  in BF), so dK/dT gains stop_gradient(S) * dq_g/dg * dg/dT.
Chang's moment path (batch moments of x.W in training, moving moments otherwise), statistics updates,
bias quantizer and activation are his code, called unchanged (`QEinsumDenseBatchnorm.call`, unwrapped).

No kernel quantizer. `QEinsumDense.__init__` always creates `_kq` (installed einsum_dense.py:32-33), so
both kernel layers re-express that constructor without it (`_init_without_kq`); `kq` raises. q_g is
created in the layer's `__init__`, inside whatever scopes are active (Chang's six in the lab builder),
from the configuration Chang's kernel quantizer would get at the site (`kq_conf`, or
`QuantizerConfig('default', 'weight')` as einsum_dense.py:32): at the MHA projections that is the MHA's
`_qkvo_kq_conf` object itself (mha.py:61, :84). Only the granularity differs: built with shape () at a
non-BN site and (n_out,) at a fused-BN site (STUDY section 2.2). q_g carries the KBI quantizer's three
variables k (uint8, constant 1), b and i; STUDY section 6 step 4 names only b and i (a finding, PREFLIGHT B6).

hgq2 0.1.9 wrappers (base.py:59-142). QLayerMeta wraps a class's `call` (EBOPs, output quantizer) and
`build` (check_collapsed_weight, which evaluates `qkernel` right after build) at the class's first
instantiation, and wraps again what a subclass inherits. Every class here defines its own `call` and
`build`, and calls only unwrapped ancestors (`_unwrapped`), so each is wrapped once (G15) and `qkernel`
is computable at the end of build (G21). QLayerMeta registers and serializes `get_config` only for
classes in `hgq` modules (base.py:47-57), so these are registered in package `cbl1016` and serialize
their quantizer configurations themselves (G22).

Binary attention (decision U3). `BinaryMultiHeadAttention.build` and `_make_output_dense` are the
installed 0.1.9 `QMultiHeadAttention.build` (mha.py:98-247) and `_make_output_dense` (mha.py:249-293)
with `QEinsumDense` replaced by `BinaryEinsumDense`; G22 checks the two against the installed source by
syntax tree. The projections receive `penalty` and `gr` through `_get_common_kwargs_for_sublayer`.

Penalty (b) (STUDY section 3.1(b)). A binary layer's `_compute_ebops` returns it, so HGQ's wrapper adds
beta * (b) to the loss and writes (b) to the layer's training counter (base.py:102-106):
- H1 branch: Chang's contraction (einsum_dense.py:68-77) with kq.bits_ replaced by q_g's bits_
  broadcast over the input axis, which is also the native rule (a'), plus the native bias term;
- H2 branch: per output column c, min(H1_c, sum_{t,i} b_{t,i} + sum_t u(t) * bits_c) with H1_c =
  sum_{t,i} b_{t,i} * bits_c, a column with rounded bits 0 costing 0, the branch and the zero test
  detached, u(t) = ceil(log2 sum_i 2^I_{t,i}) + max_i F_{t,i} over inputs of nonzero width (owner
  decisions B3a-1 and B3a-2), its gradient reaching max F only; plus the native bias term.
The branch is the config's `headline`, which has no default (interfaces.md section 5): a binary layer
refuses `penalty=None`. B7 (M7) recomputes (b) independently from the quantizer variables (G30, G33).

hgq, keras and jax are imported at module import time, so this module is imported only after
`cbl.env.setup()` (the lab builder imports it inside its hooks). Nothing here is quotable.
"""
from __future__ import annotations

import functools
import math
from collections.abc import Sized

from keras import ops
from keras.initializers import Constant
from keras.saving import register_keras_serializable, serialize_keras_object
from keras.src import constraints, initializers, regularizers
from keras.src.backend.config import epsilon
from keras.src.layers.attention.multi_head_attention import _build_proj_equation

from hgq.layers import QEinsumDense, QEinsumDenseBatchnorm, QMultiHeadAttention
from hgq.layers.attn.mha import _get_output_shape
from hgq.layers.core.base import QLayerBaseSingleInput, QLayerMeta
from hgq.quantizer import Quantizer, QuantizerConfig

PACKAGE = 'cbl1016'
FORMS = ('FB', 'BF')
PENALTY_RULES = ('H1', 'H2')
U_SPREAD_EXACT = 24            # u(t): integer-bit spread below which the power-of-two sum is exact (below)
_SENTINEL = -2.0 ** 30


# ----------------------------------------------------------------------------------------------
# QLayerMeta's wrappers (installed base.py:59-142)


def _wrapper_codes():
    consts = {c.co_name: c for c in QLayerMeta.__call__.__code__.co_consts if hasattr(c, 'co_name')}
    return consts['call'], consts['build']


_WRAP_CALL, _WRAP_BUILD = _wrapper_codes()


def _unwrapped(cls, name):
    """`cls.<name>` without QLayerMeta's call and build wrappers (identified by their code objects)."""
    fn = getattr(cls, name)
    while getattr(fn, '__code__', None) in (_WRAP_CALL, _WRAP_BUILD):
        fn = fn.__wrapped__
    return fn


_SINGLE_INPUT_BUILD = _unwrapped(QLayerBaseSingleInput, 'build')     # base.py:296-299
_SINGLE_INPUT_GET_CONFIG = QLayerBaseSingleInput.get_config          # base.py:301-304, serializing (base.py:50-56)
_BN_CALL = _unwrapped(QEinsumDenseBatchnorm, 'call')                  # einsum_dense_batchnorm.py:185-211
_MHA_CALL = _unwrapped(QMultiHeadAttention, 'call')                   # mha.py:431-480


# ----------------------------------------------------------------------------------------------
# the binary map's statistics (STUDY section 2.2)


def sign_m1(x):
    """Sign with Sign(0) = -1 (STUDY section 2.2): +1 where x > 0, -1 elsewhere."""
    return ops.where(x > 0, ops.ones_like(x), -ops.ones_like(x))


def _check_binary_args(penalty, gr, form=None, fused=False):
    if penalty not in PENALTY_RULES:
        raise ValueError(f'penalty must be one of {PENALTY_RULES} (the config headline; no default), not {penalty!r}')
    if not isinstance(gr, bool):
        raise ValueError(f'gr must be a bool, not {gr!r}')
    if fused and form not in FORMS:
        raise ValueError(f'a fused-BN binary site needs form FB or BF, not {form!r}')
    if not fused and form is not None:
        raise ValueError('form applies only at fused-BN sites')


# ----------------------------------------------------------------------------------------------
# the scale treatment: the one place of decision OD-13


class LearnedWidthScale:
    """OD-13 as adopted: q_g is an HGQ2 `Quantizer` built in the layer's `__init__` from Chang's kernel
    configuration at the site, one (b, i) per layer at non-BN sites and per output column at fused-BN
    sites; it quantizes the exported magnitude (BF: after the fold, g_c = beta * |s_c|). Replacing this
    class changes the scale treatment everywhere and nothing else (module docstring)."""

    label = 'OD-13: learned-width HGQ quantizer q_g on the exported magnitude; BF quantized after the fold'

    def create(self, layer, kq_conf):
        """q_g, built from the configuration Chang's kq would get here (einsum_dense.py:32-33)."""
        conf = kq_conf or QuantizerConfig('default', 'weight')
        return Quantizer(conf, name=f'{layer.name}_qg')

    def build(self, layer, column_shape):
        """Granularity: shape () (one b, i per layer) or the fused-BN norm shape (one per column)."""
        layer.qg.build(() if column_shape is None else tuple(column_shape))

    def bits(self, layer):
        """q_g's differentiable bits (KBI SAT_SYM: b, quantizer.py:72-73, fixed_point_quantizer.py:92-97)."""
        return layer.qg.bits

    def kernel_bits(self, layer, kernel_shape):
        """q_g's bits_ broadcast over the input axis: NR's kernel widths and the H1 penalty's (STUDY 3.1 a')."""
        return layer.qg.bits_(kernel_shape)

    def magnitude(self, layer, x, *, training=None):
        """q_g applied to the scale's input (an HGQ quantizer call; SAT_SYM never moves its state)."""
        return layer.qg(x, training=training)

    def parts(self, layer, kernel, scaler=None, *, training=None) -> dict:
        """The map of the layer's site and form. `scaler` is s (fused-BN sites) or None (non-BN).
        Returns K (the kernel every consumer reads), q (q_g's output), S, g (the scale's input before
        stop_gradient), and sign_s (BF) or None."""
        gr = layer.gr
        if scaler is None:                                          # non-BN: T = W, g = beta per layer
            T = kernel
            S = sign_m1(T - ops.mean(T))
            beta = ops.mean(ops.abs(T))
            q = self.magnitude(layer, beta if gr else ops.stop_gradient(beta), training=training)
            K = q * ops.stop_gradient(S) + (T - ops.stop_gradient(T))
            return {'K': K, 'q': q, 'S': S, 'g': beta, 'sign_s': None}
        if layer.form == 'FB':                                      # fold, then binarize; beta per column
            U = ops.einsum(layer.fused_kernel_equation, kernel, scaler)          # einsum_dense_batchnorm.py:167
            S = sign_m1(U - ops.mean(U))
            beta_c = ops.mean(ops.abs(U), axis=layer._binary_reduce_axes)
            q = self.magnitude(layer, beta_c if gr else ops.stop_gradient(beta_c), training=training)
            K = q * ops.stop_gradient(S) + (U - ops.stop_gradient(U))
            return {'K': K, 'q': q, 'S': S, 'g': beta_c, 'sign_s': None}
        if layer.form == 'BF':                                      # binarize, then fold; beta per layer
            S = sign_m1(kernel - ops.mean(kernel))
            beta = ops.mean(ops.abs(kernel))
            g = (beta if gr else ops.stop_gradient(beta)) * ops.abs(scaler)
            q = self.magnitude(layer, g, training=training)
            sign_s = ops.sign(scaler)
            K = (q * ops.stop_gradient(sign_s * S)
                 + ops.einsum(layer.fused_kernel_equation, kernel - ops.stop_gradient(kernel), scaler))
            return {'K': K, 'q': q, 'S': S, 'g': beta * ops.abs(scaler), 'sign_s': sign_s}
        raise ValueError(f'{layer.name}: form {layer.form!r}')

    def kernel(self, layer, kernel, scaler=None, *, training=None):
        return self.parts(layer, kernel, scaler, training=training)['K']

    def record(self, layer) -> dict:
        """Plain numbers of the site at the frozen state (moving moments, inference-mode q_g call): q_g's
        b, i and f, q_g(g), g, S, the exported kernel, zero columns by route and saturated columns
        (STUDY sections 2.2, 2.3, 12). For B7 (counts) and B8 (structure); the gates use it too."""
        import numpy as np

        scaler = layer.frozen_scaler() if isinstance(layer, BinaryEinsumDenseBatchnorm) else None
        p = self.parts(layer, layer.kernel, scaler, training=False)
        qz = layer.qg.quantizer
        num = lambda x: np.asarray(ops.convert_to_numpy(x))          # noqa: E731
        b, i, f, top = num(qz.b), num(qz.i), num(qz.f), num(qz.max)
        q, g = num(p['q']), num(p['g'])
        return {'per_column': scaler is not None, 'form': layer.form, 'gr': layer.gr, 'penalty': layer.penalty,
                'b': b, 'i': i, 'f': f, 'max': top, 'g': g, 'q': q, 'S': num(p['S']), 'K': num(p['K']),
                'sign_s': None if p['sign_s'] is None else num(p['sign_s']),
                'pruned': b == 0, 'rounded_to_zero': (b > 0) & (q == 0), 'saturated': g > top}


SCALE = LearnedWidthScale()


# ----------------------------------------------------------------------------------------------
# penalty (b), STUDY section 3.1(b)


def _einsum_axes(layer, shape):
    """Contracted input axes, position axes (neither batch nor contracted) and kernel output axes."""
    in_spec, rest = layer.equation.split(',')
    k_spec = rest.split('->')[0]
    letters = in_spec.replace('...', '')
    rank = len(shape)
    first = rank - len(letters)
    contracted = tuple(first + j for j, ch in enumerate(letters) if ch in k_spec)
    positions = tuple(a for a in range(1, rank) if a not in contracted)
    k_out = tuple(j for j, ch in enumerate(k_spec) if ch not in letters)
    return contracted, positions, k_out


def u_bound(I, F, included, contracted, n_in):
    """u(t) = ceil(log2 sum_i 2^I_{t,i}) + max_i F_{t,i} over the included inputs, 0 where none is
    included (STUDY section 3.2; owner decisions B3a-1, B3a-2), for every position of an input-shaped
    tensor. The integer part is computed from exact powers of two, M + ceil(log2 sum_i 2^(I_i - M)) with
    M the largest I, and carries no gradient; F's maximum carries it. Exact when every included input's
    I lies within U_SPREAD_EXACT - ceil(log2 n_in) of M; Chang's datalane constraint ic = MinMax(0, 12)
    (model.py:292) bounds the spread by 12."""
    Im = ops.where(included, I, ops.full_like(I, _SENTINEL))
    M = ops.max(Im, axis=contracted, keepdims=True)
    d = Im - M
    p = ops.zeros_like(I)
    for k in range(-U_SPREAD_EXACT, 1):
        p = p + ops.where(d == float(k), ops.full_like(I, 2.0 ** k), ops.zeros_like(I))
    r = ops.sum(p, axis=contracted)
    e = ops.zeros_like(r)
    for k in range(0, max(1, math.ceil(math.log2(max(n_in, 1)))) + 1):
        e = e + ops.cast(r > 2.0 ** k, r.dtype)
    any_inc = ops.any(included, axis=contracted)
    maxF = ops.max(ops.where(included, F, ops.full_like(F, _SENTINEL)), axis=contracted)
    u = ops.stop_gradient(ops.max(Im, axis=contracted) + e) + maxF
    return ops.where(any_inc, u, ops.zeros_like(u))


def h2_kernel_term(layer, shape, bw_inp, bits, kernel_shape):
    """The H2 branch's kernel term (module docstring). At a binary site every coefficient of a nonzero
    column is +-q, so each column's included inputs are all inputs of nonzero width, and sum_b and u(t)
    are the same for every column."""
    contracted, _, k_out = _einsum_axes(layer, shape)
    n_in = int(math.prod(int(shape[a]) for a in contracted))
    out_shape = tuple(int(kernel_shape[j]) for j in k_out)
    qz = layer.iq.quantizer
    _, I, F = qz.kif
    I = qz.bw_mapper.bw_to_x(I, shape)
    F = qz.bw_mapper.bw_to_x(F, shape)
    included = bw_inp > 0
    sum_b = ops.sum(bw_inp)
    sum_u = ops.sum(u_bound(I, F, included, contracted, n_in))
    bits_c = ops.reshape(ops.broadcast_to(bits, out_shape), (-1,))
    h1_c = sum_b * bits_c
    ss_c = sum_b + sum_u * bits_c
    chosen = ops.where(h1_c <= ss_c, h1_c, ss_c)                      # ties to H1, as cbl/cost_pure.py
    cost = ops.where(bits_c == 0, ops.zeros_like(chosen), chosen)     # a column with rounded bits 0 costs 0
    return ops.sum(cost)


def penalty_b(layer, shape):
    """Penalty (b) of a binary layer for an input of `shape` (batch axis 1), as HGQ's wrapper calls
    `_compute_ebops` (base.py:102-104)."""
    bw_inp = layer.iq.bits_(shape)                                     # einsum_dense.py:70
    kernel_shape = ops.shape(layer.kernel)
    if layer.penalty == 'H1':
        bw_ker = SCALE.kernel_bits(layer, kernel_shape)                # einsum_dense.py:71, kq -> q_g
        ebops = ops.einsum(layer.ebops_equation, bw_inp, bw_ker)       # :72
    elif layer.penalty == 'H2':
        ebops = h2_kernel_term(layer, shape, bw_inp, SCALE.bits(layer), kernel_shape)
    else:
        raise ValueError(f'{layer.name}: penalty {layer.penalty!r}')
    if layer.bq is not None:                                           # :73-76, native bias term
        bw_bias = layer.bq.bits_(ops.shape(layer.bias))
        size = ops.cast(ops.prod(layer.full_output_shape[1:]), layer.dtype)
        ebops = ebops + ops.mean(bw_bias) * size  # type: ignore
    return ebops


# ----------------------------------------------------------------------------------------------
# the kernel layers


def _init_without_kq(self, *, equation, output_shape, activation, bias_axes, kernel_initializer, bias_initializer,
                     kernel_regularizer, bias_regularizer, kernel_constraint, bias_constraint, kq_conf, iq_conf,
                     bq_conf, parallelization_factor, kwargs):
    """`QEinsumDense.__init__` (einsum_dense.py:29-36) with q_g in place of the kernel quantizer. Its
    `gather_vars_to_kwargs` passes every argument but kq_conf, bq_conf and parallelization_factor on to
    the base, with lora_rank=None; so does this."""
    super(QEinsumDense, self).__init__(
        equation=equation, output_shape=output_shape, activation=activation, bias_axes=bias_axes,
        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, iq_conf=iq_conf,
        lora_rank=None, **kwargs)
    self._qg = SCALE.create(self, kq_conf)                                       # in place of :32-33
    bq_conf = bq_conf or QuantizerConfig('default', 'bias')                       # :34
    self._bq = None if bias_axes is None else Quantizer(bq_conf, name=f'{self.name}_bq')   # :35
    self.parallelization_factor = parallelization_factor                         # :36


def _binary_config(self, extra=None):
    """get_config without kq: the base's serialized config (base.py:245-257, 301-304; EinsumDense's),
    then q_g's configuration under the constructor argument it is built from, and the binary fields."""
    config = _SINGLE_INPUT_GET_CONFIG(self)
    config.update({
        'kq_conf': serialize_keras_object(self.qg.config),
        'bq_conf': serialize_keras_object(self.bq.config) if self.bq is not None else None,
        'penalty': self.penalty,
        'gr': self.gr,
    })
    config.update(extra or {})
    return config


class _BinaryKernelMixin:
    """What both kernel layers share (not a Keras class; the concrete classes define call and build)."""

    @property
    def kq(self):
        raise AttributeError(f'{self.name} is a binary site: it has no kernel quantizer (STUDY section 2.2); '
                             'read qkernel, qg and SCALE.record(layer)')

    @property
    def qg(self):
        return self._qg

    def _set_binary(self, form, penalty, gr):
        self.form = form
        self.penalty = penalty
        self.gr = gr


@register_keras_serializable(package=PACKAGE)
class BinaryEinsumDense(_BinaryKernelMixin, QEinsumDense):
    """The binary map at a non-BN site (Q, K, V, O, the FFN contraction): one q_g per layer."""

    def __init__(self, equation, output_shape, activation=None, bias_axes=None, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, kq_conf=None, iq_conf=None, bq_conf=None, parallelization_factor=-1,
                 penalty=None, gr=False, **kwargs):
        _check_binary_args(penalty, gr)
        _init_without_kq(self, equation=equation, output_shape=output_shape, activation=activation,
                         bias_axes=bias_axes, kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint, kq_conf=kq_conf, iq_conf=iq_conf, bq_conf=bq_conf,
                         parallelization_factor=parallelization_factor, kwargs=kwargs)
        self._set_binary(None, penalty, gr)

    def build(self, input_shape):
        """einsum_dense.py:46-53 with q_g built per layer in place of kq."""
        _SINGLE_INPUT_BUILD(self, input_shape)                                    # :47
        SCALE.build(self, None)                                                   # :49, kq -> q_g
        if self.bias is not None:                                                 # :50-52
            assert self.bq is not None
            self.bq.build(ops.shape(self.bias))
        self.ebops_equation = self.equation.split('->')[0] + '->'                 # :53

    def call(self, inputs, training=None):
        """einsum_dense.py:55-66 with the binary kernel in place of kq(W)."""
        qkernel = SCALE.kernel(self, self._kernel, training=training)             # :56
        if self.enable_iq:
            inputs = self.iq(inputs, training=training)
        x = ops.einsum(self.equation, inputs, qkernel)
        if self.bias is not None:
            assert self.bq is not None
            x += self.bq(self.bias, training=training)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def _compute_ebops(self, shape):
        return penalty_b(self, shape)

    def get_config(self):
        return _binary_config(self)

    @property
    def qkernel(self):
        return SCALE.kernel(self, self._kernel)                                   # einsum_dense.py:90-91


@register_keras_serializable(package=PACKAGE)
class BinaryEinsumDenseBatchnorm(_BinaryKernelMixin, QEinsumDenseBatchnorm):
    """The binary map at a fused-BN site (FFN expansion, head layers, the BE embedding), form FB or BF:
    one q_g per output column, at Chang's quantizer site after the fold."""

    def __init__(self, equation, output_shape, activation=None, bias_axes=None, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, kq_conf=None, iq_conf=None, bq_conf=None, normalize_axes=None, momentum=0.99,
                 epsilon=1e-3, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones',
                 moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None,
                 gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, synchronized=False, form=None,
                 penalty=None, gr=False, **kwargs):
        _check_binary_args(penalty, gr, form, fused=True)
        _init_without_kq(self, equation=equation, output_shape=output_shape, activation=activation,   # :43-58
                         bias_axes=bias_axes, kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer, kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint, kq_conf=kq_conf, iq_conf=iq_conf, bq_conf=bq_conf,
                         parallelization_factor=-1, kwargs=kwargs)
        self.synchronized = synchronized                                          # :59-79, as released
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(
            moving_variance_initializer,
        )
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.supports_masking = False

        normalize_axes = normalize_axes or bias_axes
        assert normalize_axes is not None, 'Either normalize_axes or bias_axes must be provided.'
        self.normalize_axes = normalize_axes
        self._check_normalize_axes()
        self._set_binary(form, penalty, gr)

    def build(self, input_shape):
        """einsum_dense_batchnorm.py:126-156 with QEinsumDense.build's kq.build replaced by q_g's per-column
        build; the fused specs are computed first, since q_g's shape is the norm shape."""
        _SINGLE_INPUT_BUILD(self, input_shape)                                    # einsum_dense.py:47
        norm_shape = self._compute_fused_einsum_specs(input_shape)                # einsum_dense_batchnorm.py:128
        kernel_axes = self.fused_kernel_equation.split(',')[0]
        norm = ''.join(c for c in kernel_axes if c in self.normalize_axes)
        if not kernel_axes.endswith(norm):
            raise ValueError(f'{self.name}: normalize axes {norm} are not the trailing kernel axes {kernel_axes}')
        self._binary_reduce_axes = tuple(range(len(kernel_axes) - len(norm)))
        SCALE.build(self, norm_shape)                                             # einsum_dense.py:49, kq -> q_g
        if self.bias is not None:                                                 # einsum_dense.py:50-52
            assert self.bq is not None
            self.bq.build(ops.shape(self.bias))
        self.ebops_equation = self.equation.split('->')[0] + '->'                 # einsum_dense.py:53

        if self.scale:                                                            # einsum_dense_batchnorm.py:130-156
            self.bn_gamma = self.add_weight(
                shape=norm_shape,
                name='gamma',
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                autocast=False,
            )
        assert self.bias is not None
        self.bias.trainable = self.center

        self.moving_mean = self.add_weight(
            shape=norm_shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False,
            autocast=False,
        )
        self.moving_variance = self.add_weight(
            shape=norm_shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False,
            autocast=False,
        )

    @functools.wraps(_BN_CALL)
    def call(self, inputs, training=None):
        """Chang's call (einsum_dense_batchnorm.py:185-211), unwrapped and unchanged: moments, statistics
        updates, then get_fused_qkernel_and_qbias, which is the binary one here."""
        return _BN_CALL(self, inputs, training=training)

    def get_fused_qkernel_and_qbias(self, training, mean, var):
        """einsum_dense_batchnorm.py:158-173 with the binary map in place of kq(W * s) at :167-168."""
        assert self.bias is not None
        assert self.bq is not None
        if self.scale:
            bn_gamma = self.bn_gamma
        else:
            bn_gamma = 1
        scaler = bn_gamma / ops.sqrt(var + epsilon())  # type: ignore
        kernel = self.kernel
        fused_qkernel = SCALE.kernel(self, kernel, scaler, training=training)    # :167-168

        offset = -mean * scaler
        offset = ops.reshape(offset, self._norm_offset_cast_shape)
        fused_qbias = self.bq(self.bias + offset, training=training)
        return fused_qkernel, fused_qbias

    def frozen_scaler(self):
        """s on the moving moments, as qkernel uses it (einsum_dense_batchnorm.py:176-178, 165)."""
        bn_gamma = self.bn_gamma if self.scale else 1
        return bn_gamma / ops.sqrt(self.moving_variance + epsilon())

    def _compute_ebops(self, shape):
        return penalty_b(self, shape)

    def get_config(self):
        return _binary_config(self, {'form': self.form})


# ----------------------------------------------------------------------------------------------
# binary attention (decision U3)


@register_keras_serializable(package=PACKAGE)
class BinaryMultiHeadAttention(QMultiHeadAttention):
    """QMultiHeadAttention whose build creates binary Q, K, V and O projections (BinaryEinsumDense),
    configured from the captured `_qkvo_kq_conf` as Chang's projections are (mha.py:78-92)."""

    def __init__(self, num_heads, key_dim, *args, penalty=None, gr=False, **kwargs):
        _check_binary_args(penalty, gr)
        super().__init__(num_heads, key_dim, *args, **kwargs)
        self.penalty = penalty
        self.gr = gr

    def _get_common_kwargs_for_sublayer(self):
        common_kwargs = super()._get_common_kwargs_for_sublayer()                # mha.py:78-92
        common_kwargs.update({'penalty': self.penalty, 'gr': self.gr})
        return common_kwargs

    @functools.wraps(_MHA_CALL)
    def call(self, *args, **kwargs):
        """Chang's MHA call (mha.py:431-480), unwrapped and unchanged."""
        return _MHA_CALL(self, *args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({'penalty': self.penalty, 'gr': self.gr})
        return config

    # ------------------------------------------------------------------------------------------
    # installed 0.1.9 mha.py:98-247 and :249-293, with QEinsumDense replaced by BinaryEinsumDense
    # (G22 compares both with the installed source by syntax tree)

    def build(
        self,
        query_shape,
        value_shape=None,
        key_shape=None,
    ):
        """Builds layers and variables.

        Parameters
        ----------
        query_shape : tuple
            Shape of the `query` tensor.
        value_shape : tuple, optional
            Shape of the `value` tensor.
        key_shape : tuple, optional
            Shape of the `key` tensor.
        """

        # Copied and modified from keras MultiHeadAttention, substituted
        # EinsumDense with QEinsumDense and added sequence length (shape) to its
        # output shape when initializing, if known.
        value_shape = query_shape if value_shape is None else value_shape
        key_shape = value_shape if key_shape is None else key_shape

        if self._fuse == 'qkv':
            assert query_shape == value_shape == key_shape, (
                'For fuse=qkv, query, key and value must have the same shape',
                f'but got query_shape={query_shape}, value_shape={value_shape} and key_shape={key_shape}.',
            )
        if self._fuse == 'kv':
            assert value_shape == key_shape, (
                'For fuse=kv, key and value must have the same shape',
                f'but got value_shape={value_shape} and key_shape={key_shape}.',
            )

        if value_shape[1:-1] != key_shape[1:-1]:
            raise ValueError(
                'All dimensions of `value` and `key`, except the last one, '
                f'must be equal. Received: value_shape={value_shape} and '
                f'key_shape={key_shape}',
            )

        query_rank = len(query_shape)
        key_rank = len(key_shape)
        value_rank = len(value_shape)

        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            query_rank - 1,
            bound_dims=1,
            output_dims=2,
        )
        self._query_dense = BinaryEinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1,
                [self._num_heads, self._key_dim],
                query_shape,
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name='query',
            enable_iq=self.enable_iq and not self._fuse == 'qkv',
            enable_oq=True,
            **self._get_common_kwargs_for_sublayer(),
        )

        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            key_rank - 1,
            bound_dims=1,
            output_dims=2,
        )
        self._key_dense = BinaryEinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1,
                [self._num_heads, self._key_dim],
                key_shape,
            ),
            bias_axes=None,  # Useless as it will be directly fed to softmax on seq axis
            name='key',
            enable_iq=self.enable_iq and self._fuse not in ('qkv', 'kv'),
            enable_oq=True,
            **self._get_common_kwargs_for_sublayer(),
        )

        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            value_rank - 1,
            bound_dims=1,
            output_dims=2,
        )
        self._value_dense = BinaryEinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1,
                [self._num_heads, self._value_dim],
                value_shape,
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name='value',
            enable_iq=self.enable_iq,
            enable_oq=True,
            **self._get_common_kwargs_for_sublayer(),
        )
        self._value_dense.build(value_shape)

        if self._fuse == 'qkv' and self.enable_iq:
            self._query_dense._iq = self._value_dense._iq
            self._query_dense._enable_iq = True
        if self._fuse in ('qkv', 'kv') and self.enable_iq:
            self._key_dense._iq = self._value_dense._iq
            self._key_dense._enable_iq = True

        self._query_dense.build(query_shape)
        self._key_dense.build(key_shape)

        # Builds the attention computations for multi-head dot product
        # attention.  These computations could be wrapped into the keras
        # attention layer once it supports multi-head einsum computations.
        self._build_attention(output_rank, (query_shape, value_shape, key_shape))
        self._output_dense = self._make_output_dense(
            query_shape,
            self._get_common_kwargs_for_sublayer(),
            'attention_output',
        )
        output_dense_input_shape = list(
            self._query_dense.compute_output_shape(query_shape),
        )
        output_dense_input_shape[-1] = self._value_dim
        self._output_dense.build(tuple(output_dense_input_shape))

        if self.enable_ebops:
            self._beta = self.add_weight(
                name='beta',
                shape=(),
                initializer=self._beta0,
                trainable=False,
            )
            self._ebops = self.add_weight(
                name='ebops',
                shape=(),
                initializer=Constant(0.0),
                trainable=False,
                dtype='uint32',
            )
        else:
            self._beta = None
            self._ebops = None

        self._dot_product_ebops_equation = self._dot_product_equation.split('->', 1)[0] + '->'
        self._combine_ebops_equation = self._combine_equation.split('->', 1)[0] + '->'
        self.built = True

    def _make_output_dense(self, query_shape, common_kwargs, name=None):
        """Builds the output projection matrix.

        Parameters
        ----------
        query_shape : tuple
            Shape of the query tensor.
        common_kwargs : dict
            Common keyword arguments for the einsum layer.
        name : str, optional
            Name for the projection layer.

        Returns
        -------
        QEinsumDense

        Notes
        -----
        This method is copied and modified from Keras MultiHeadAttention. It substitutes
        EinsumDense with QEinsumDense and adds sequence length (shape) to its output shape
        when initializing, if known.
        """

        query_rank = len(query_shape)
        if self._output_shape:
            if not isinstance(self._output_shape, Sized):
                output_shape = [self._output_shape]
            else:
                output_shape = self._output_shape
        else:
            output_shape = [query_shape[-1]]
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            query_rank - 1,
            bound_dims=2,
            output_dims=len(output_shape),
        )
        return BinaryEinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(output_rank - 1, output_shape, query_shape),
            bias_axes=bias_axes if self._use_bias else None,
            name=name,
            enable_iq=True,
            enable_oq=self.enable_oq,
            **common_kwargs,
        )


BINARY_CLASSES = (BinaryEinsumDense, BinaryEinsumDenseBatchnorm, BinaryMultiHeadAttention)


def binary_kernel_layers(model):
    """Every binary kernel layer of a model (MHA projections included), in `_flatten_layers` order."""
    return [ly for ly in model._flatten_layers(include_self=False)
            if isinstance(ly, (BinaryEinsumDense, BinaryEinsumDenseBatchnorm))]


__all__ = ['SCALE', 'LearnedWidthScale', 'BinaryEinsumDense', 'BinaryEinsumDenseBatchnorm', 'BinaryMultiHeadAttention',
           'BINARY_CLASSES', 'penalty_b', 'h2_kernel_term', 'u_bound', 'sign_m1', 'binary_kernel_layers', 'FORMS',
           'PENALTY_RULES']
