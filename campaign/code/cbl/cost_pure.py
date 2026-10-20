"""M7, pure part: coefficient width, snap, the u_c bound, the H2 column rule and the H1/H2/NR
aggregation over per-site inputs given as plain numbers.

Sources: STUDY v0.8 (`review/snapshots/v0.8/STUDY.md`, sha256 290022fa41a726b8...98ca8) §3.1, §3.2,
§3.5; PREFLIGHT-PLAN §2 row M7, §3 row G25, §4 row B3a. The model part (batch B7, `cbl/cost.py`)
extracts the per-site inputs from the layers, computes the penalty (b) inside the trace and gates
the native-order equality (G26-G34). This module decides nothing the STUDY leaves open beyond the
owner decisions named below and recorded in the G25 rows file.

Imports: the standard library and numpy only. No Keras, JAX or TensorFlow module is imported,
directly or indirectly; gate G25 checks this statically and in a fresh interpreter.

Arithmetic
----------
Every finite binary floating-point number is a dyadic rational, so a float16/32/64 value converts
to `fractions.Fraction` without rounding. w(.), snap, u_c, the column rule and every kernel term
are computed in exact integer or rational arithmetic.

Floating point enters at one place only: `layer_value`, which combines a layer's exact integer
kernel term with the layer's native non-kernel contribution `rest_f32` (the bias term
`mean(bq.bits_) * size` of `QEinsumDense._compute_ebops`, computed by the layer in float32) in one
IEEE float32 addition and truncates toward zero to uint32, as the native counter does
(`base.py:102-103`; installed hgq2 0.1.9 returns contraction plus bias term with no
parallelization factor, STUDY §3.1(a); `ebops_factor` is 1.0). The kernel term is required to be at
most 2**24. All its partial sums are then non-negative integers no larger than the total, every one
exactly representable in float32, so any float32 summation order (the native einsum's included)
gives the exact integer. Whether the backend fuses the bias multiply and the add (one rounding
instead of two) is outside this module; G26 and G28 (B7) gate equality with native.

Per-layer values are integers in [0, 2**32 - 1]; totals are Python-int sums of per-layer values,
as `FreeEBOPs` sums the per-layer uint32 counters (`hgq/utils/sugar/ebops.py:7-15`).

Owner decisions (B3a, recorded in `preflight/gates/G25/<sha12>/PREFLIGHT-rows.md`)
-------------------------------------------------------------------------------
B3a-1  In u_c(t), max F runs over the same inputs as the sum: K_ic != 0 and b_{t,i} > 0.
B3a-2  u_c(t) = 0 at a position where no input is included (empty sum, nothing to accumulate).
B3a-3  Layer value = trunc_uint32(float32(kernel term) + rest_f32), one float32 addition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

STUDY_SHA256 = "290022fa41a726b8ae66ff302f7910a69c4c148ec27b15c2448cab83fb898ca8"

FLOAT32_EXACT_INT_MAX = 2 ** 24
UINT32_MAX = 2 ** 32 - 1
GROUPS = ("attention", "ffn", "head", "embedding", "other")  # docs/interfaces.md section 1, subtotals
SITE_KINDS = ("hgq", "binary")
COLUMN_ZERO, COLUMN_SINGLE, COLUMN_OTHER = "zero", "single", "other"
BRANCH_H1, BRANCH_SIGN_SCALE = "h1", "sign_scale"

_HALF = Fraction(1, 2)
_FLOAT_DTYPES = (np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64))


class CostInputError(ValueError):
    """An input is malformed: wrong type, shape, non-integer width, NaN or infinity."""


class SnapError(AssertionError):
    """Snapping a value to its quantizer's f changed it (STUDY §3.2, grid exactness)."""


class WidthBoundError(AssertionError):
    """w(K_ic) exceeds the coefficient's quantizer bits, so H1 <= NR would fail (STUDY §3.2)."""


class Float32RangeError(ArithmeticError):
    """A kernel term exceeds 2**24, so a float32 sum of it is no longer order-independent."""


# ---------------------------------------------------------------------------------------------
# exact values


def pow2(e: int) -> Fraction:
    """2**e as an exact Fraction, for any integer e."""
    e = _as_int("exponent", e)
    return Fraction(1 << e) if e >= 0 else Fraction(1, 1 << -e)


def exact(q: Any) -> Fraction:
    """Exact rational value of q.

    Accepts int, a dyadic Fraction, a Python float, and numpy float16/32/64 or integer scalars.
    Refuses bool, NaN, infinity, a non-dyadic Fraction and any other type. Conversion is exact:
    float32 widens to float64 exactly and `Fraction(float)` is exact.
    """
    if isinstance(q, (bool, np.bool_)):
        raise CostInputError("bool is not a coefficient value: %r" % (q,))
    if isinstance(q, np.floating):
        if q.dtype not in _FLOAT_DTYPES:
            raise CostInputError("unsupported float dtype %s" % q.dtype)
        q = float(q)
    elif isinstance(q, np.integer):
        q = int(q)
    if isinstance(q, float):
        if not math.isfinite(q):
            raise CostInputError("value is not finite: %r" % (q,))
        return Fraction(q)
    if isinstance(q, int):
        return Fraction(q)
    if isinstance(q, Fraction):
        d = q.denominator
        if d & (d - 1):
            raise CostInputError("value is not dyadic: %s" % q)
        return q
    raise CostInputError("unsupported value type %s" % type(q).__name__)


def is_float32_value(q: Any) -> bool:
    """True when q is exactly representable as a finite float32."""
    x = exact(q)
    try:
        f = float(x)
    except OverflowError:
        return False
    if not math.isfinite(f) or Fraction(f) != x:
        return False
    with np.errstate(over="ignore", under="ignore"):
        f32 = np.float32(f)
    return bool(np.isfinite(f32)) and Fraction(float(f32)) == x


def _odd_part(n: int) -> int:
    """n with its trailing zero bits removed (n > 0)."""
    return n >> ((n & -n).bit_length() - 1)


def w(q: Any) -> int:
    """Coefficient width of STUDY §3.2.

    w(0) = 0; for q != 0, w(q) = floor(log2|q|) - nu(q) + 1, where nu(q) is the exponent of the
    lowest set bit of |q|. Writing |q| = m * 2**e with m odd, floor(log2|q|) = e + bitlen(m) - 1
    and nu(q) = e, so w(q) = bitlen(m): the number of bits from the highest to the lowest set bit.
    It is the smallest KBI bit count that holds q exactly under SAT_SYM, sign excluded.
    """
    x = exact(q)
    if x == 0:
        return 0
    n, d = abs(x.numerator), x.denominator
    if d == 1:
        n = _odd_part(n)
    # d > 1 is a power of two and the fraction is reduced, so n is odd
    return n.bit_length()


def floor_log2(q: Any) -> int:
    """floor(log2 |q|) for q != 0, exact."""
    x = abs(exact(q))
    if x == 0:
        raise CostInputError("floor_log2 of zero")
    n, d = x.numerator, x.denominator
    e = n.bit_length() - d.bit_length()
    while pow2(e) > x:
        e -= 1
    while pow2(e + 1) <= x:
        e += 1
    return e


def lowest_set_bit_exponent(q: Any) -> int:
    """nu(q) of STUDY §3.2: the exponent of the lowest set bit of |q| (q != 0)."""
    x = abs(exact(q))
    if x == 0:
        raise CostInputError("nu of zero")
    n, d = x.numerator, x.denominator
    if d > 1:
        return -(d.bit_length() - 1)
    return (n & -n).bit_length() - 1


def w_literal(q: Any) -> int:
    """w(q) evaluated literally as floor(log2|q|) - nu(q) + 1 (w(0) = 0); a cross-check of w."""
    x = exact(q)
    if x == 0:
        return 0
    return floor_log2(x) - lowest_set_bit_exponent(x) + 1


# ---------------------------------------------------------------------------------------------
# snap to the quantizer's f (STUDY §3.2, grid exactness, chang C27)


def snap(q: Any, f: Any) -> Fraction:
    """q rounded to the grid of step 2**-f with HGQ's RND rule, floor(q * 2**f + 1/2) / 2**f."""
    x = exact(q)
    s = pow2(_as_int("f", f))
    return Fraction(math.floor(x * s + _HALF)) / s


def snap_exact(q: Any, f: Any) -> Fraction:
    """The snap of q to f, asserting that snapping changes nothing; raises SnapError otherwise."""
    x = exact(q)
    s = snap(x, f)
    if s != x:
        raise SnapError("snap to f=%s changes %s to %s" % (f, x, s))
    return s


def snap_changed_count(values: Any, f: Any) -> int:
    """Number of entries of `values` whose snap to the broadcast `f` changes them (record field
    `snap_changed`, docs/interfaces.md section 1; the evaluator requires 0)."""
    vals = np.asarray(values)
    fs = np.broadcast_to(np.asarray(f), vals.shape)
    n = 0
    for v, fi in zip(vals.reshape(-1).tolist(), fs.reshape(-1).tolist()):
        x = exact(v)
        if snap(x, _as_int("f", fi)) != x:
            n += 1
    return n


# ---------------------------------------------------------------------------------------------
# the u_c bound (STUDY §3.2, H2)


def ceil_log2(q: Any) -> int:
    """ceil(log2 q) for q > 0, exact: the smallest integer k with 2**k >= q."""
    x = exact(q)
    if x <= 0:
        raise CostInputError("ceil_log2 needs a positive value, got %s" % x)
    k = floor_log2(x)
    return k if pow2(k) == x else k + 1


def included_inputs(b_t: Sequence[int], nonzero_col: Sequence[bool]) -> List[int]:
    """Inputs i that enter H2's two terms at one position: K_ic != 0 and b_{t,i} > 0
    ("inputs of zero width are excluded from both terms", STUDY §3.2)."""
    if len(b_t) != len(nonzero_col):
        raise CostInputError("b_t and nonzero_col differ in length")
    return [i for i, (bi, nz) in enumerate(zip(b_t, nonzero_col)) if nz and bi > 0]


def u_bound(I_t: Sequence[int], F_t: Sequence[int], b_t: Sequence[int],
            nonzero_col: Sequence[bool]) -> int:
    """u_c(t) = ceil(log2 sum_i 2**I_{t,i}) + max_i F_{t,i} over the included inputs (STUDY §3.2).

    An upper bound on the width, sign excluded, of the signed +-1 sum of column c at position t;
    it ignores the sign pattern and unsigned inputs (critical C6). Both the sum and the max run
    over inputs with K_ic != 0 and b_{t,i} > 0 (owner decision B3a-1); with no such input the
    position contributes 0 (owner decision B3a-2).
    """
    if not (len(I_t) == len(F_t) == len(b_t) == len(nonzero_col)):
        raise CostInputError("I_t, F_t, b_t and nonzero_col differ in length")
    inc = included_inputs(b_t, nonzero_col)
    if not inc:
        return 0
    total = sum((pow2(I_t[i]) for i in inc), Fraction(0))
    return ceil_log2(total) + max(_as_int("F", F_t[i]) for i in inc)


# ---------------------------------------------------------------------------------------------
# the H2 column rule (STUDY §3.2; §3.5 is its homogeneous special case)


def column_class(col: Sequence[Fraction]) -> Tuple[str, Optional[Fraction]]:
    """Classify an exported kernel column: ('zero', None) if every coefficient is 0;
    ('single', g_c) if its nonzero coefficients share one magnitude g_c; ('other', None) else."""
    mags = {abs(exact(v)) for v in col if exact(v) != 0}
    if not mags:
        return COLUMN_ZERO, None
    if len(mags) == 1:
        return COLUMN_SINGLE, mags.pop()
    return COLUMN_OTHER, None


def h2_column(h1_c: int, cls: str, g_c: Optional[Fraction], b: Sequence[Sequence[int]],
              I: Sequence[Sequence[int]], F: Sequence[Sequence[int]],
              nonzero_col: Sequence[bool]) -> Tuple[int, Optional[str], int, int]:
    """H2 charge of one column (STUDY §3.2).

    A zero column costs 0. A column whose nonzero coefficients share one magnitude g_c costs
    min(H1_c, sum_{t,i: K_ic != 0} b_{t,i} + sum_t u_c(t) * w(g_c)), zero-width inputs excluded
    from both terms. Any other column costs H1_c.

    Returns (charge, branch, sum_b, sum_u): branch is 'h1' or 'sign_scale' for a single-magnitude
    column (ties go to 'h1'; both give the same charge) and None otherwise.
    """
    if cls == COLUMN_ZERO:
        return 0, None, 0, 0
    if cls == COLUMN_OTHER:
        return h1_c, None, 0, 0
    if cls != COLUMN_SINGLE or g_c is None:
        raise CostInputError("unknown column class %r" % (cls,))
    sum_b = 0
    sum_u = 0
    for b_t, I_t, F_t in zip(b, I, F):
        inc = included_inputs(b_t, nonzero_col)
        sum_b += sum(b_t[i] for i in inc)
        sum_u += u_bound(I_t, F_t, b_t, nonzero_col)
    alt = sum_b + sum_u * w(g_c)
    if h1_c <= alt:
        return h1_c, BRANCH_H1, sum_b, sum_u
    return alt, BRANCH_SIGN_SCALE, sum_b, sum_u


# ---------------------------------------------------------------------------------------------
# per-site record and terms (docs/interfaces.md section 7)


@dataclass(frozen=True)
class SiteInput:
    """Plain-number inputs of one constant-kernel site, in canonical 2-D form.

    name         layer name, unique in the model
    group        one of GROUPS
    kind         'hgq' (kernel from an HGQ kernel quantizer) or 'binary' (K = q_g(g)*S); the
                 rules are identical (STUDY §3.2); 'binary' adds structural checks
    b            (P, n_in) native input widths b_{t,i} (input quantizer bits_), integers >= 0
    I, F         (P, n_in) input quantizer integer and fractional bits, integers
    K            (n_in, n_out) exported constants (`qkernel`, or q_g(g)*S), float32 values
    kernel_f     broadcastable to (n_in, n_out): the f of each coefficient's quantizer (b - i)
    kernel_bits  broadcastable to (n_in, n_out): kq.bits_ at an HGQ site, q_g bits_ at a binary
                 site (per layer or per column); the widths of NR (STUDY §3.1(a), (a'))
    rest_f32     numpy float32 scalar: the layer's native non-kernel contribution (bias term) as
                 the layer computes it, or None when the layer has none (no bias)
    """

    name: str
    group: str
    kind: str
    b: Any
    I: Any
    F: Any
    K: Any
    kernel_f: Any
    kernel_bits: Any
    rest_f32: Any = None


@dataclass(frozen=True)
class NativeLayer:
    """A layer with no constant kernel (QAdd, tables, softmax, the MHA's QK and AV counter,
    pooling): its count is native under H1, H2 and NR alike (STUDY §3.2)."""

    name: str
    group: str
    native: int


@dataclass(frozen=True)
class SiteTerms:
    name: str
    group: str
    kind: str
    P: int
    n_in: int
    n_out: int
    h1_kernel: int
    h2_kernel: int
    nr_kernel: int
    H1: int
    H2: int
    NR: int
    snap_changed: int
    columns: Tuple[Dict[str, Any], ...]


def _as_int(name: str, v: Any) -> int:
    if isinstance(v, (bool, np.bool_)):
        raise CostInputError("%s: bool is not an integer width" % name)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        if math.isfinite(fv) and fv.is_integer():
            return int(fv)
    if isinstance(v, Fraction) and v.denominator == 1:
        return int(v)
    raise CostInputError("%s: not an integer value: %r" % (name, v))


def _int_grid(name: str, a: Any, shape: Tuple[int, ...]) -> List[List[int]]:
    arr = np.asarray(a)
    if arr.dtype == object:
        raise CostInputError("%s: object arrays are not accepted" % name)
    try:
        arr = np.broadcast_to(arr, shape)
    except ValueError:
        raise CostInputError("%s: shape %s does not broadcast to %s" % (name, np.shape(a), shape))
    return [[_as_int(name, v) for v in row] for row in arr.tolist()]


def _validate_site(site: SiteInput):
    if not isinstance(site.name, str) or not site.name:
        raise CostInputError("site name must be a non-empty string")
    if site.group not in GROUPS:
        raise CostInputError("%s: group %r not in %s" % (site.name, site.group, GROUPS))
    if site.kind not in SITE_KINDS:
        raise CostInputError("%s: kind %r not in %s" % (site.name, site.kind, SITE_KINDS))
    Karr = np.asarray(site.K)
    if Karr.ndim != 2 or 0 in Karr.shape:
        raise CostInputError("%s: K must be a non-empty (n_in, n_out) array" % site.name)
    n_in, n_out = Karr.shape
    barr = np.asarray(site.b)
    if barr.ndim != 2 or barr.shape[1] != n_in or barr.shape[0] == 0:
        raise CostInputError("%s: b must be (P, n_in) with n_in = %d, got %s" % (site.name, n_in, barr.shape))
    P = barr.shape[0]
    b = _int_grid(site.name + ".b", site.b, (P, n_in))
    I = _int_grid(site.name + ".I", site.I, (P, n_in))
    F = _int_grid(site.name + ".F", site.F, (P, n_in))
    for t in range(P):
        for i in range(n_in):
            if b[t][i] < 0:
                raise CostInputError("%s: negative input width at (%d, %d)" % (site.name, t, i))
            # KIF: b = relu(i + f); KBI: f = b - i (fixed_point_quantizer.py, KBI f and KIF b)
            if b[t][i] != max(0, I[t][i] + F[t][i]):
                raise CostInputError("%s: b != max(0, I + F) at (%d, %d): %d, %d, %d"
                                     % (site.name, t, i, b[t][i], I[t][i], F[t][i]))
    kf = _int_grid(site.name + ".kernel_f", site.kernel_f, (n_in, n_out))
    kb = _int_grid(site.name + ".kernel_bits", site.kernel_bits, (n_in, n_out))
    if Karr.dtype == object:
        Kvals = Karr.tolist()
    else:
        if Karr.dtype.kind == "f" and Karr.dtype not in _FLOAT_DTYPES:
            raise CostInputError("%s: unsupported K dtype %s" % (site.name, Karr.dtype))
        Kvals = Karr.tolist()
    K = [[exact(v) for v in row] for row in Kvals]
    for i in range(n_in):
        for c in range(n_out):
            if kb[i][c] < 0:
                raise CostInputError("%s: negative kernel bits at (%d, %d)" % (site.name, i, c))
            if not is_float32_value(K[i][c]):
                raise CostInputError("%s: K[%d][%d] = %s is not a float32 value" % (site.name, i, c, K[i][c]))
    rest = site.rest_f32
    if rest is not None:
        if not isinstance(rest, np.float32):
            raise CostInputError("%s: rest_f32 must be a numpy float32 scalar or None" % site.name)
        if not np.isfinite(rest) or rest < 0:
            raise CostInputError("%s: rest_f32 must be finite and >= 0" % site.name)
    return P, n_in, n_out, b, I, F, K, kf, kb


def site_terms(site: SiteInput) -> SiteTerms:
    """H1, H2 and NR of one constant-kernel site (STUDY §3.1-§3.2).

    Each exported constant is snapped to its quantizer's f and must be unchanged (SnapError);
    w(K_ic) must not exceed the coefficient's quantizer bits (WidthBoundError). Kernel terms:
      H1 = sum_{t,i,c} b_{t,i} * w(K_ic)
      NR = sum_{t,i,c} b_{t,i} * kernel_bits_ic
      H2 = sum_c H2_c (the column rule, `h2_column`)
    Each is an exact integer; the per-layer value adds `rest_f32` by `layer_value`.
    """
    P, n_in, n_out, b, I, F, K, kf, kb = _validate_site(site)

    changed = 0
    for i in range(n_in):
        for c in range(n_out):
            if snap(K[i][c], kf[i][c]) != K[i][c]:
                changed += 1
    if changed:
        raise SnapError("%s: snap to the quantizer's f changes %d of %d constants"
                        % (site.name, changed, n_in * n_out))

    wk = [[w(K[i][c]) for c in range(n_out)] for i in range(n_in)]
    for i in range(n_in):
        for c in range(n_out):
            if wk[i][c] > kb[i][c]:
                raise WidthBoundError("%s: w(K[%d][%d]) = %d exceeds the quantizer bits %d"
                                      % (site.name, i, c, wk[i][c], kb[i][c]))

    bsum = [sum(b[t][i] for t in range(P)) for i in range(n_in)]  # sum over positions
    columns = []
    h1_kernel = h2_kernel = nr_kernel = 0
    for c in range(n_out):
        col = [K[i][c] for i in range(n_in)]
        cls, g_c = column_class(col)
        if site.kind == "binary":
            nonzero = [v != 0 for v in col]
            if cls == COLUMN_OTHER or (cls == COLUMN_SINGLE and not all(nonzero)):
                raise CostInputError("%s: binary column %d is neither zero nor +-g_c throughout"
                                     % (site.name, c))
        h1_c = sum(bsum[i] * wk[i][c] for i in range(n_in))
        nr_c = sum(bsum[i] * kb[i][c] for i in range(n_in))
        nonzero_col = [v != 0 for v in col]
        h2_c, branch, sum_b, sum_u = h2_column(h1_c, cls, g_c, b, I, F, nonzero_col)
        if not (0 <= h2_c <= h1_c <= nr_c):
            raise AssertionError("%s: column %d violates 0 <= H2_c <= H1_c <= NR_c (%d, %d, %d)"
                                 % (site.name, c, h2_c, h1_c, nr_c))
        columns.append({
            "column": c,
            "class": cls,
            "g_c": g_c,
            "w_g": None if g_c is None else w(g_c),
            "H1_c": h1_c,
            "H2_c": h2_c,
            "NR_c": nr_c,
            "branch": branch,
            "sum_b": sum_b,
            "sum_u": sum_u,
        })
        h1_kernel += h1_c
        h2_kernel += h2_c
        nr_kernel += nr_c

    rest = site.rest_f32
    return SiteTerms(
        name=site.name, group=site.group, kind=site.kind, P=P, n_in=n_in, n_out=n_out,
        h1_kernel=h1_kernel, h2_kernel=h2_kernel, nr_kernel=nr_kernel,
        H1=layer_value(h1_kernel, rest), H2=layer_value(h2_kernel, rest),
        NR=layer_value(nr_kernel, rest),
        snap_changed=changed, columns=tuple(columns),
    )


def layer_value(kernel_term: int, rest_f32: Any = None) -> int:
    """Per-layer count: trunc_uint32(float32(kernel_term) + rest_f32) (owner decision B3a-3).

    kernel_term must be an integer in [0, 2**24], so that float32(kernel_term) is exact and equal
    to a float32 sum of its terms in any order. rest_f32 is a numpy float32 scalar or None. The
    addition is one IEEE float32 addition (numpy float32 scalars); the result is truncated toward
    zero, as the float32 -> uint32 cast of the native counter does (`base.py:102-103`).
    """
    if isinstance(kernel_term, (bool, np.bool_)) or not isinstance(kernel_term, (int, np.integer)):
        raise CostInputError("kernel_term must be an integer, got %r" % (kernel_term,))
    kt = int(kernel_term)
    if kt < 0:
        raise CostInputError("kernel_term must be >= 0, got %d" % kt)
    if kt > FLOAT32_EXACT_INT_MAX:
        raise Float32RangeError("kernel term %d exceeds 2**24; a float32 sum is order-dependent" % kt)
    k32 = np.float32(kt)
    if int(k32) != kt:
        raise AssertionError("float32(%d) is not exact" % kt)
    if rest_f32 is None:
        v = k32
    else:
        if not isinstance(rest_f32, np.float32):
            raise CostInputError("rest_f32 must be a numpy float32 scalar or None")
        if not np.isfinite(rest_f32) or rest_f32 < 0:
            raise CostInputError("rest_f32 must be finite and >= 0, got %r" % (rest_f32,))
        v = k32 + rest_f32
    if not isinstance(v, np.float32):
        raise AssertionError("float32 addition did not stay float32: %r" % type(v))
    out = math.trunc(float(v))
    if not 0 <= out <= UINT32_MAX:
        raise Float32RangeError("layer value %d outside uint32" % out)
    return out


# ---------------------------------------------------------------------------------------------
# aggregation


def aggregate(sites: Sequence[SiteInput], native_layers: Sequence[NativeLayer] = ()) -> Dict[str, Any]:
    """Model totals and subtotals of H1, H2 and NR over per-site inputs (STUDY §3.1(c), §3.2).

    Constant-kernel sites go through `site_terms`; layers without a constant kernel enter with
    their native count under all three. Returns
      {"H1", "H2", "NR": int totals,
       "snap_changed": int (always 0 on return; a change raises),
       "layers": {name: {"group", "kind", "H1", "H2", "NR"}},
       "groups": {group: {"H1", "H2", "NR"}} for every group in GROUPS,
       "sites": {name: SiteTerms}}
    """
    layers: Dict[str, Dict[str, Any]] = {}
    site_out: Dict[str, SiteTerms] = {}
    groups = {g: {"H1": 0, "H2": 0, "NR": 0} for g in GROUPS}

    def add(name, group, kind, h1, h2, nr):
        if name in layers:
            raise CostInputError("duplicate layer name %r" % name)
        if group not in GROUPS:
            raise CostInputError("%s: group %r not in %s" % (name, group, GROUPS))
        layers[name] = {"group": group, "kind": kind, "H1": h1, "H2": h2, "NR": nr}
        groups[group]["H1"] += h1
        groups[group]["H2"] += h2
        groups[group]["NR"] += nr

    for s in sites:
        if not isinstance(s, SiteInput):
            raise CostInputError("sites must be SiteInput records")
        t = site_terms(s)
        add(t.name, t.group, t.kind, t.H1, t.H2, t.NR)
        site_out[t.name] = t
    for n in native_layers:
        if not isinstance(n, NativeLayer):
            raise CostInputError("native_layers must be NativeLayer records")
        v = n.native
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not 0 <= int(v) <= UINT32_MAX:
            raise CostInputError("%s: native count must be an integer in uint32 range, got %r" % (n.name, v))
        v = int(v)
        add(n.name, n.group, "native", v, v, v)

    return {
        "H1": sum(l["H1"] for l in layers.values()),
        "H2": sum(l["H2"] for l in layers.values()),
        "NR": sum(l["NR"] for l in layers.values()),
        "snap_changed": sum(t.snap_changed for t in site_out.values()),
        "layers": layers,
        "groups": groups,
        "sites": site_out,
    }
