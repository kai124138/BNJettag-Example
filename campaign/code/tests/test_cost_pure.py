"""Unit tests for cbl/cost_pure.py (gate G25; STUDY v0.8 §3.2, §3.5).

Run with the Mac CPU interpreter, from anywhere:

    python -I -B -m pytest -q -p no:cacheprovider tests/test_cost_pure.py

Every reference here is computed independently of the module's own route: w(.) against the
IEEE float32 bit pattern, against the literal formula floor(log2|q|) - nu(q) + 1, and against a
search for the smallest SAT_SYM grid that holds the value; the site counts against a direct
triple loop over (t, i, c) of the §3.2 definitions. Every number is an engineering check, not a
quotable result.
"""

import itertools
import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

FORK = Path(__file__).resolve().parents[1]
if str(FORK) not in sys.path:
    sys.path.insert(0, str(FORK))

from cbl import cost_pure as cp  # noqa: E402

BANNED = ("keras", "jax", "jaxlib", "tensorflow", "torch", "hgq", "qkeras", "hls4ml", "da4ml")


# ---------------------------------------------------------------------------------------------
# independent references


def w_float32_bits(x):
    """w of a finite float32 from its IEEE bit pattern: width of the significand between its
    highest and lowest set bit."""
    u = int(np.array([x], dtype=np.float32).view(np.uint32)[0])
    e = (u >> 23) & 0xFF
    frac = u & 0x7FFFFF
    assert e != 0xFF, "not finite"
    m = frac if e == 0 else (frac | (1 << 23))
    if m == 0:
        return 0
    tz = (m & -m).bit_length() - 1
    return m.bit_length() - tz


def on_sat_sym_grid(x, b, i):
    """x lies exactly on the KBI SAT_SYM grid with b bits and i integer bits (f = b - i):
    a multiple of 2**-f with |x| <= 2**i - 2**-f (fixed_point_quantizer.py max for SAT_SYM)."""
    f = b - i
    step = Fraction(2) ** (-f)
    q = x / step
    return q.denominator == 1 and abs(x) <= Fraction(2) ** i - step


def w_semantic(x):
    """Smallest b such that some integer i puts x on the b-bit SAT_SYM grid (STUDY §3.2)."""
    x = Fraction(x)
    if x == 0:
        return 0
    a = abs(x)
    e = 0
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    for b in range(0, 31):
        for i in range(e - 2, e + 40):
            if on_sat_sym_grid(x, b, i):
                return b
    raise AssertionError("no grid found for %s" % x)


def ceil_log2_ref(x):
    x = Fraction(x)
    k = -400
    while Fraction(2) ** k < x:
        k += 1
    return k


def random_finite_float32(rng, n):
    out = []
    while len(out) < n:
        u = int(rng.integers(0, 2 ** 32, dtype=np.uint64))
        if (u >> 23) & 0xFF == 0xFF:
            continue
        out.append(np.array([u], dtype=np.uint32).view(np.float32)[0])
    return out


# ---------------------------------------------------------------------------------------------
# w(.)


def test_w_study_checks():
    # STUDY §3.2 "Check (derived): w(1) = 1 and w(0.375) = w(3·2⁻³) = 2."
    assert cp.w(1) == 1
    assert cp.w(0.375) == 2
    assert cp.w(Fraction(3, 8)) == 2
    assert cp.w(np.float32(0.375)) == 2


def test_w_zero_and_negative():
    assert cp.w(0) == 0
    assert cp.w(0.0) == 0
    assert cp.w(-0.0) == 0
    assert cp.w(np.float32(-0.0)) == 0
    for v in [1, 0.375, 3, 5, 0.1, 1e-30, 123456.75, Fraction(7, 64)]:
        assert cp.w(-v) == cp.w(v)
    for v in [0.1, 0.375, 3e38, 1e-45]:
        assert cp.w(np.float32(-v)) == cp.w(np.float32(v))


def test_w_powers_of_two_float32_range():
    for k in range(-149, 128):  # every power of two float32 holds, subnormals included
        v = np.float32(2.0 ** k)
        assert float(v) == 2.0 ** k
        assert cp.w(v) == 1
        assert cp.w(-v) == 1
    assert cp.w(2 ** 200) == 1
    assert cp.w(Fraction(1, 2 ** 300)) == 1


def test_w_largest_widths():
    # the kernel quantizers allow b <= 23 (bc = MinMax(0, 23), STUDY §2.2; config.py:75-91)
    for i in (-8, 0, 3, 12, 30):
        v = (2 ** 23 - 1) * Fraction(2) ** (i - 23)
        assert cp.w(v) == 23
        assert cp.is_float32_value(v)
    # 24 significant bits: the largest any float32 can have (q_g cannot produce it)
    v24 = np.float32((2 ** 24 - 1) / 2 ** 24)
    assert cp.w(v24) == 24
    assert cp.w(np.float32(np.finfo(np.float32).max)) == 24
    assert cp.w(np.float32(np.finfo(np.float32).tiny)) == 1  # smallest normal
    assert cp.w(np.float32(2.0 ** -149)) == 1  # smallest subnormal
    assert cp.w(np.float32(3 * 2.0 ** -149)) == 2
    assert cp.w((1 << 23) - 1) == 23
    assert cp.w((1 << 24) - 1) == 24
    assert cp.w((1 << 24) + 1) == 25  # an int beyond float32 precision still has a width


def test_w_against_ieee_bits_and_literal_formula():
    rng = np.random.default_rng(20261018)
    vals = random_finite_float32(rng, 3000)
    vals += [np.float32(v) for v in (1, 0.375, 0.5, 0.75, 15 / 16, 0.1, 1 / 3, 1e-45, 3e38)]
    for v in vals:
        wv = cp.w(v)
        assert wv == w_float32_bits(v), v
        assert wv == cp.w_literal(v), v
        assert 0 <= wv <= 24
        assert cp.w(np.float32(-v)) == wv


def test_w_against_semantic_definition():
    rng = np.random.default_rng(7)
    vals = [np.float32(v) for v in rng.uniform(-4, 4, 150)]
    vals += [np.float32(v) for v in (1, 0.375, 0.5, 15 / 16, 2.0 ** -20, 3 * 2.0 ** -20, 1023, 2.0 ** 20)]
    for b in range(0, 9):
        for k in (1, 3, 2 ** b - 1 if b else 0):
            vals.append(Fraction(k, 2 ** 5))
    for v in vals:
        assert cp.w(v) == w_semantic(cp.exact(v)), v


def test_w_bounded_by_bits_on_every_small_grid_exhaustive():
    # every value on a b-bit SAT_SYM grid with integer i has w <= b, negative f included
    for b in range(0, 7):
        for i in range(-3, 10):
            f = b - i
            for k in range(-(2 ** b - 1), 2 ** b):
                v = Fraction(k) * Fraction(2) ** (-f)
                assert on_sat_sym_grid(v, b, i)
                assert cp.w(v) <= b
                assert cp.snap_exact(v, f) == v


def test_w_bounded_by_bits_large_grids():
    rng = np.random.default_rng(3)
    for b in (8, 12, 16, 20, 22, 23):
        for i in (-10, 0, 4, 12, 25, 40):  # i > b gives f < 0
            f = b - i
            ks = [2 ** b - 1, -(2 ** b - 1), 2 ** (b - 1), 1, 0] + [int(k) for k in rng.integers(-(2 ** b - 1), 2 ** b, 20)]
            for k in ks:
                v = Fraction(k) * Fraction(2) ** (-f)
                assert cp.w(v) <= b
                assert cp.snap_exact(v, f) == v
            assert cp.w(Fraction(2 ** b - 1) * Fraction(2) ** (-f)) == b


def test_exact_refusals():
    for bad in (float("nan"), float("inf"), -float("inf"), np.float32("nan"), np.float32("inf"),
                True, np.bool_(False), Fraction(1, 3), "1", None, 1 + 0j):
        with pytest.raises(cp.CostInputError):
            cp.exact(bad)
    assert cp.exact(np.float32(0.1)) == Fraction(float(np.float32(0.1)))
    assert cp.exact(np.int64(-7)) == -7


def test_is_float32_value():
    assert cp.is_float32_value(np.float32(0.1))
    assert not cp.is_float32_value(0.1)  # float64 0.1 is not a float32
    assert cp.is_float32_value(Fraction(3, 8))
    assert not cp.is_float32_value(2 ** 128)
    assert not cp.is_float32_value(Fraction(1, 2 ** 150))
    assert cp.is_float32_value(Fraction(1, 2 ** 149))
    assert not cp.is_float32_value((1 << 24) + 1)


# ---------------------------------------------------------------------------------------------
# snap


def test_snap_rounding_rule():
    # RND: floor(x * 2**f + 1/2) / 2**f (STUDY §2.2 "Forward")
    assert cp.snap(0.375, 2) == Fraction(1, 2)
    assert cp.snap(-0.375, 2) == Fraction(-1, 4)
    assert cp.snap(0.3, 2) == Fraction(1, 4)
    assert cp.snap(12, -3) == 16
    assert cp.snap(11, -3) == 8
    assert cp.snap(0, 5) == 0


def test_snap_exact_unchanged_and_raises():
    assert cp.snap_exact(0.375, 3) == Fraction(3, 8)
    assert cp.snap_exact(0.375, 10) == Fraction(3, 8)
    assert cp.snap_exact(16, -3) == 16
    assert cp.snap_exact(0, -7) == 0  # pruned: b = 0 gives f = -i
    with pytest.raises(cp.SnapError):
        cp.snap_exact(0.375, 2)
    with pytest.raises(cp.SnapError):
        cp.snap_exact(12, -3)
    # a value one float32 ulp off a 2**-4 grid (an inexact pow would produce this, chang C27)
    off = np.nextafter(np.float32(15 / 16), np.float32(1))
    with pytest.raises(cp.SnapError):
        cp.snap_exact(off, 4)
    assert cp.w(off) == 24  # what w would read without the snap check (STUDY §3.2)
    with pytest.raises(cp.CostInputError):
        cp.snap(0.5, 1.5)


def test_snap_off_grid_by_half_and_quarter_step():
    for b, i in ((4, 0), (8, 2), (23, 0), (23, 30)):
        f = b - i
        step = Fraction(2) ** (-f)
        v = Fraction(5) * step
        with pytest.raises(cp.SnapError):
            cp.snap_exact(v + step / 2, f)
        with pytest.raises(cp.SnapError):
            cp.snap_exact(v + step / 4, f)


def test_snap_changed_count():
    vals = np.array([0.375, 0.5, 0.3, 0.0], dtype=np.float64)
    assert cp.snap_changed_count(vals, 3) == 1
    assert cp.snap_changed_count(vals, np.array([3, 0, 2, 0])) == 2
    assert cp.snap_changed_count(np.float32([15 / 16] * 5), 4) == 0


# ---------------------------------------------------------------------------------------------
# ceil_log2 and the u_c bound


def test_ceil_log2():
    cases = {1: 0, 2: 1, 3: 2, 63: 6, 64: 6, 65: 7, Fraction(1, 2): -1, Fraction(3, 4): 0,
             Fraction(1, 1024): -10, Fraction(1, 1024) + Fraction(1, 2 ** 30): -9, 48: 6, 2 ** 17: 17}
    for x, k in cases.items():
        assert cp.ceil_log2(x) == k == ceil_log2_ref(x), x
    for bad in (0, -1, Fraction(-1, 2)):
        with pytest.raises(cp.CostInputError):
            cp.ceil_log2(bad)


def test_u_bound_section35():
    # STUDY §3.5: homogeneous 4-bit inputs, I = 1, F = 3: u = ceil(log2(n_in * 2)) + 3 = 9
    for n in (24, 32):
        assert cp.u_bound([1] * n, [3] * n, [4] * n, [True] * n) == 9
    assert cp.u_bound([1] * 3, [3] * 3, [4] * 3, [True] * 3) == 6


def test_u_bound_heterogeneous_and_negative_I():
    assert cp.u_bound([0, 3], [4, 1], [4, 4], [True, True]) == 4 + 4  # 1 + 8 = 9 -> 4
    assert cp.u_bound([-2, -2], [5, 5], [3, 3], [True, True]) == -1 + 5  # sum 1/2
    assert cp.u_bound([5], [-2], [3], [True]) == 3  # one input: u = I + F = b
    assert cp.u_bound([12] * 32, [20] * 32, [32] * 32, [True] * 32) == 17 + 20  # largest datalane I


def test_u_bound_zero_width_excluded_from_both_terms():
    n = 24
    I, F, b = [1] * n, [3] * n, [4] * n
    base = cp.u_bound(I, F, b, [True] * n)
    # add a zero-width input (b = relu(I + F) = 0) with a large F and a large I
    I2, F2, b2 = I + [-5], F + [5], b + [0]
    assert cp.u_bound(I2, F2, b2, [True] * (n + 1)) == base
    I3, F3, b3 = I + [9], F + [-9], b + [0]
    assert cp.u_bound(I3, F3, b3, [True] * (n + 1)) == base


def test_u_bound_zero_coefficient_excluded_owner_decision_b3a_1():
    # max F over the same inputs as the sum (owner decision B3a-1)
    I, F, b = [1, 1, 0], [3, 3, 10], [4, 4, 10]
    assert cp.u_bound(I, F, b, [True, True, False]) == 2 + 3
    assert cp.u_bound(I, F, b, [True, True, True]) == cp.ceil_log2(5) + 10


def test_u_bound_empty_owner_decision_b3a_2():
    assert cp.u_bound([1, 1], [3, 3], [4, 4], [False, False]) == 0
    assert cp.u_bound([1, -3], [3, 3], [4, 0], [False, True]) == 0
    with pytest.raises(cp.CostInputError):
        cp.u_bound([1], [3, 3], [4, 4], [True, True])


# ---------------------------------------------------------------------------------------------
# column rule


def test_column_class():
    F_ = Fraction
    assert cp.column_class([0, 0.0, -0.0]) == ("zero", None)
    assert cp.column_class([0.375, -0.375, 0, 0.375]) == ("single", F_(3, 8))
    assert cp.column_class([0, 0, -0.5]) == ("single", F_(1, 2))
    assert cp.column_class([0.375, 0.5]) == ("other", None)


def test_h2_column_cases():
    b = [[4, 4], [4, 4]]
    I = [[1, 1], [1, 1]]
    F = [[3, 3], [3, 3]]
    assert cp.h2_column(37, "zero", None, b, I, F, [False, False]) == (0, None, 0, 0)
    assert cp.h2_column(37, "other", None, b, I, F, [True, True]) == (37, None, 0, 0)
    # single magnitude g = 15/16 (w = 4): H1_c = 16 * 4 = 64; alt = 16 + 2 * (2 + 3) * 4 = 56
    g = Fraction(15, 16)
    assert cp.h2_column(64, "single", g, b, I, F, [True, True]) == (56, "sign_scale", 16, 10)
    # w = 1: H1_c = 16 <= 16 + 10 -> H1 (the floor of STUDY §3.5)
    assert cp.h2_column(16, "single", Fraction(1, 2), b, I, F, [True, True]) == (16, "h1", 16, 10)


# ---------------------------------------------------------------------------------------------
# sites


SECTION35 = {
    # STUDY v0.8 §3.5 table (review/snapshots/v0.8/STUDY.md:610-613):
    # site: (native 4-bit, H1 w=24, H1 w=4/2/1, H2 w=24, H2 w=4/2/1)
    "FFN expansion 24→32": (98304, 589824, (98304, 49152, 24576), 79872, (33792, 29184, 24576)),
    "FFN contraction 32→24": (98304, 589824, (98304, 49152, 24576), 66048, (31488, 28032, 24576)),
    "head 32→32, pooled": (16384, 98304, (16384, 8192, 4096), 11008, (5248, 4672, 4096)),
    "Q, K, V and O together": (393216, 2359296, (393216, 196608, 98304), 305664, (132864, 115584, 98304)),
}
SHAPES = {  # (P, n_in, n_out) of each site in the row
    "FFN expansion 24→32": [(8, 24, 32)],
    "FFN contraction 32→24": [(8, 32, 24)],
    "head 32→32, pooled": [(1, 32, 32)],
    "Q, K, V and O together": [(8, 24, 32)] * 3 + [(8, 32, 24)],  # Q, K, V 24×(2·16); O (2·16)×24
}
G_BY_W = {24: np.float32((2 ** 24 - 1) / 2 ** 24), 4: np.float32(15 / 16), 2: np.float32(0.375), 1: np.float32(0.5)}


def _homog_site(name, P, n_in, n_out, K, kernel_f, kernel_bits, kind, rest=None, group="ffn"):
    return cp.SiteInput(name=name, group=group, kind=kind, b=np.full((P, n_in), 4), I=np.full((P, n_in), 1),
                        F=np.full((P, n_in), 3), K=K, kernel_f=kernel_f, kernel_bits=kernel_bits, rest_f32=rest)


def _row_counts(row, w_value, rng):
    h1 = h2 = nr = 0
    for P, n_in, n_out in SHAPES[row]:
        g = G_BY_W[w_value]
        S = np.where(rng.random((n_in, n_out)) < 0.5, -1.0, 1.0).astype(np.float32)
        f = -cp.lowest_set_bit_exponent(g)
        t = cp.site_terms(_homog_site(row, P, n_in, n_out, S * g, f, 24, "binary"))
        h1, h2, nr = h1 + t.H1, h2 + t.H2, nr + t.NR
    return h1, h2


def _row_native(row, rng):
    total = 0
    for P, n_in, n_out in SHAPES[row]:
        # a 4-bit learned HGQ kernel: per-value values on 4-bit grids, NR = native at an HGQ site
        K = (rng.integers(-15, 16, (n_in, n_out)) / 16.0).astype(np.float32)
        t = cp.site_terms(_homog_site(row, P, n_in, n_out, K, 4, 4, "hgq"))
        total += t.NR
    return total


@pytest.mark.parametrize("row", list(SECTION35))
def test_section35_table_from_general_functions(row):
    native, h1_24, h1_q, h2_24, h2_q = SECTION35[row]
    rng = np.random.default_rng(35)
    assert _row_native(row, rng) == native
    assert _row_counts(row, 24, rng) == (h1_24, h2_24)
    for wv, h1e, h2e in zip((4, 2, 1), h1_q, h2_q):
        assert _row_counts(row, wv, rng) == (h1e, h2e)


def test_section35_derived_prose():
    # STUDY §3.5 prose: "At w = 4 it is 288 of a 1,056-unit column at the 24→32 site";
    # 6,144 = P * n_in * n_out / b-bar-free factor, 350,000 / (6,144 * 24) = 2.37, / (6,144 * 4) = 14.24
    P, n_in, u, wv = 8, 24, 9, 4
    assert P * u * wv == 288
    assert P * n_in * 4 + P * u * wv == 1056
    assert 8 * 24 * 32 == 6144
    assert f"{350000 / (6144 * 24):.2f}" == "2.37"
    assert f"{350000 / (6144 * 4):.2f}" == "14.24"


def _brute(site):
    b = np.broadcast_to(np.asarray(site.b), np.shape(site.b)).tolist()
    I = np.asarray(site.I).tolist()
    F = np.asarray(site.F).tolist()
    K = np.asarray(site.K, dtype=np.float32)
    n_in, n_out = K.shape
    P = len(b)
    kb = np.broadcast_to(np.asarray(site.kernel_bits), (n_in, n_out)).tolist()
    wk = [[w_float32_bits(K[i, c]) for c in range(n_out)] for i in range(n_in)]
    H1 = sum(b[t][i] * wk[i][c] for t in range(P) for i in range(n_in) for c in range(n_out))
    NR = sum(b[t][i] * kb[i][c] for t in range(P) for i in range(n_in) for c in range(n_out))
    H2 = 0
    for c in range(n_out):
        col = [Fraction(float(K[i, c])) for i in range(n_in)]
        h1_c = sum(b[t][i] * wk[i][c] for t in range(P) for i in range(n_in))
        mags = {abs(v) for v in col if v != 0}
        if not mags:
            continue
        if len(mags) > 1:
            H2 += h1_c
            continue
        g = mags.pop()
        A = sum(b[t][i] for t in range(P) for i in range(n_in) if col[i] != 0 and b[t][i] > 0)
        U = 0
        for t in range(P):
            inc = [i for i in range(n_in) if col[i] != 0 and b[t][i] > 0]
            if inc:
                U += ceil_log2_ref(sum(Fraction(2) ** I[t][i] for i in inc)) + max(F[t][i] for i in inc)
        H2 += min(h1_c, A + U * w_semantic(g))
    return H1, H2, NR


def _random_site(rng, kind, P, n_in, n_out):
    I = rng.integers(-3, 6, (P, n_in))
    F = rng.integers(-4, 7, (P, n_in))
    b = np.maximum(0, I + F)
    K = np.zeros((n_in, n_out), dtype=np.float32)
    kf = np.zeros((n_in, n_out), dtype=np.int64)
    kb = np.zeros((n_in, n_out), dtype=np.int64)
    for c in range(n_out):
        typ = rng.integers(0, 4)  # 0 zero, 1 single, 2 single with zeros (hgq only), 3 other (hgq only)
        if kind == "binary":
            typ = 0 if rng.random() < 0.2 else 1
        bc, ic = int(rng.integers(1, 9)), int(rng.integers(-2, 4))
        fc = bc - ic
        kf[:, c], kb[:, c] = fc, bc
        if typ == 0:
            if rng.random() < 0.5:
                kb[:, c] = 0  # pruned
                kf[:, c] = -ic
            continue
        if typ in (1, 2):
            k = int(rng.integers(1, 2 ** bc))
            g = k * 2.0 ** (-fc)
            S = np.where(rng.random(n_in) < 0.5, -1.0, 1.0)
            if typ == 2:
                S = S * (rng.random(n_in) < 0.6)
            K[:, c] = (S * g).astype(np.float32)
        else:
            K[:, c] = (rng.integers(-(2 ** bc - 1), 2 ** bc, n_in) * 2.0 ** (-fc)).astype(np.float32)
    return cp.SiteInput(name="s", group="ffn", kind=kind, b=b, I=I, F=F, K=K, kernel_f=kf, kernel_bits=kb)


@pytest.mark.parametrize("kind", ["hgq", "binary"])
def test_site_terms_against_brute_force(kind):
    rng = np.random.default_rng(1018 if kind == "hgq" else 1019)
    for _ in range(60):
        P = int(rng.choice([1, 3, 8]))
        site = _random_site(rng, kind, P, int(rng.integers(1, 7)), int(rng.integers(1, 6)))
        t = cp.site_terms(site)
        assert (t.h1_kernel, t.h2_kernel, t.nr_kernel) == _brute(site)
        assert 0 <= t.H2 <= t.H1 <= t.NR
        assert t.snap_changed == 0


def test_binary_zero_scale_routes():
    # pruned q_g (b = 0) and q_g rounded to zero (b > 0, constant 0), STUDY §2.2
    P, n_in, n_out = 8, 24, 3
    g = np.float32(15 / 16)
    K = np.zeros((n_in, n_out), dtype=np.float32)
    K[:, 0] = g * np.where(np.arange(n_in) % 2, 1, -1)
    kb = np.array([4, 0, 4])  # column 1 pruned, column 2 rounded to zero
    kf = np.array([4, 0, 4])
    t = cp.site_terms(_homog_site("x", P, n_in, n_out, K, kf, kb, "binary"))
    cols = t.columns
    assert [c["class"] for c in cols] == ["single", "zero", "zero"]
    assert [c["H1_c"] for c in cols] == [8 * 24 * 4 * 4, 0, 0]
    assert [c["H2_c"] for c in cols] == [8 * 24 * 4 + 8 * 9 * 4, 0, 0]
    assert [c["NR_c"] for c in cols] == [8 * 24 * 4 * 4, 0, 8 * 24 * 4 * 4]  # NR still charges b


def test_site_rest_and_float32_combination():
    K = np.full((4, 2), np.float32(0.5))
    s = cp.SiteInput("x", "head", "hgq", b=[[4] * 4], I=[[1] * 4], F=[[3] * 4], K=K, kernel_f=1, kernel_bits=4,
                     rest_f32=np.float32(np.mean(np.float32([3, 4, 4, 5]))) * np.float32(2))
    t = cp.site_terms(s)
    assert (t.h1_kernel, t.nr_kernel) == (32, 128)
    assert (t.H1, t.NR) == (40, 136)


def test_site_refusals():
    good = dict(name="x", group="ffn", kind="hgq", b=[[4, 4]], I=[[1, 1]], F=[[3, 3]],
                K=np.float32([[0.5], [0.25]]), kernel_f=2, kernel_bits=4)
    cp.site_terms(cp.SiteInput(**good))
    for key, bad, exc in [
        ("K", np.float32([[0.5], [0.3]]), cp.SnapError),
        ("kernel_bits", 0, cp.WidthBoundError),
        ("b", [[4, 3]], cp.CostInputError),  # b != relu(I + F)
        ("b", [[4.5, 4]], cp.CostInputError),
        ("I", [[1.5, 1]], cp.CostInputError),
        ("b", [[4, 4, 4]], cp.CostInputError),
        ("K", np.array([[0.1], [0.25]]), cp.CostInputError),  # float64 0.1 is not a float32 value
        ("K", np.float32([[np.nan], [0.25]]), cp.CostInputError),
        ("group", "attn", cp.CostInputError),
        ("kind", "ternary", cp.CostInputError),
        ("rest_f32", 3.0, cp.CostInputError),
        ("rest_f32", np.float32(-1), cp.CostInputError),
        ("kernel_f", [1, 2, 3], cp.CostInputError),
    ]:
        args = dict(good)
        args[key] = bad
        with pytest.raises(exc):
            cp.site_terms(cp.SiteInput(**args))
    # a binary site whose column is not +-g_c throughout
    args = dict(good, kind="binary", K=np.float32([[0.5], [0.25]]))
    with pytest.raises(cp.CostInputError):
        cp.site_terms(cp.SiteInput(**args))
    args = dict(good, kind="binary", K=np.float32([[0.5], [0.0]]))
    with pytest.raises(cp.CostInputError):
        cp.site_terms(cp.SiteInput(**args))


def test_site_float32_range_guard():
    # b = 90 (I = 12, F = 78), w = 23 at 8 x 32 x 32 exceeds 2**24 in H1
    g = np.float32((2 ** 23 - 1) * 2.0 ** -23)
    K = np.full((32, 32), g)
    s = cp.SiteInput("x", "ffn", "binary", b=np.full((8, 32), 90), I=12, F=78, K=K, kernel_f=23, kernel_bits=23)
    with pytest.raises(cp.Float32RangeError):
        cp.site_terms(s)
    s = cp.SiteInput("x", "ffn", "binary", b=np.full((8, 32), 32), I=12, F=20, K=K, kernel_f=23, kernel_bits=23)
    t = cp.site_terms(s)
    assert t.H1 == t.NR == 8 * 32 * 32 * 32 * 23


# ---------------------------------------------------------------------------------------------
# layer_value: where floating point enters


def test_layer_value_study_float32_example():
    # STUDY §3.2: "0.65 × 180 truncates to 116 in float32 and 117 in float64"
    rest = np.float32(0.65) * np.float32(180)
    assert isinstance(rest, np.float32)
    assert cp.layer_value(0, rest) == 116
    assert math.trunc(0.65 * 180) == 117


def test_layer_value_semantics():
    assert cp.layer_value(12345) == 12345
    assert cp.layer_value(2 ** 24) == 2 ** 24
    with pytest.raises(cp.Float32RangeError):
        cp.layer_value(2 ** 24 + 1)
    assert cp.layer_value(100, np.float32(0.75)) == 100
    # one float32 addition, round to nearest even: 2**23 + 1 + 0.75 rounds to 2**23 + 2
    assert cp.layer_value(2 ** 23 + 1, np.float32(0.75)) == 2 ** 23 + 2
    for bad in (-1, 1.0, True):
        with pytest.raises(cp.CostInputError):
            cp.layer_value(bad)
    for bad in (0.5, np.float64(0.5), np.float32(np.nan), np.float32(-0.5)):
        with pytest.raises(cp.CostInputError):
            cp.layer_value(3, bad)


# ---------------------------------------------------------------------------------------------
# aggregation


def test_aggregate_totals_and_groups():
    rng = np.random.default_rng(11)
    s1 = _random_site(rng, "hgq", 8, 5, 4)
    s1 = cp.SiteInput("emb", "embedding", s1.kind, s1.b, s1.I, s1.F, s1.K, s1.kernel_f, s1.kernel_bits)
    s2 = _random_site(rng, "binary", 8, 6, 3)
    s2 = cp.SiteInput("q", "attention", s2.kind, s2.b, s2.I, s2.F, s2.K, s2.kernel_f, s2.kernel_bits,
                      np.float32(17.5))
    natives = [cp.NativeLayer("add0", "other", 116), cp.NativeLayer("mha_qkav", "attention", 4000)]
    out = cp.aggregate([s1, s2], natives)
    t1, t2 = cp.site_terms(s1), cp.site_terms(s2)
    for key in ("H1", "H2", "NR"):
        assert out[key] == getattr(t1, key) + getattr(t2, key) + 116 + 4000
        assert out[key] == sum(v[key] for v in out["groups"].values())
        assert out["groups"]["attention"][key] == getattr(t2, key) + 4000
        assert out["groups"]["ffn"][key] == 0 and out["groups"]["head"][key] == 0
    assert set(out["groups"]) == set(cp.GROUPS)
    assert out["snap_changed"] == 0
    assert out["H2"] <= out["H1"] <= out["NR"]
    with pytest.raises(cp.CostInputError):
        cp.aggregate([s1, s1])
    with pytest.raises(cp.CostInputError):
        cp.aggregate([s1], [cp.NativeLayer("emb", "other", 1)])
    for bad in (-1, 2 ** 32, 1.5, True):
        with pytest.raises(cp.CostInputError):
            cp.aggregate([], [cp.NativeLayer("n", "other", bad)])
    with pytest.raises(cp.CostInputError):
        cp.aggregate([], [cp.NativeLayer("n", "nowhere", 1)])


def test_aggregate_section35_qkvo_row_as_model():
    # the "Q, K, V and O together" row as four attention sites of one model, w = 4
    rng = np.random.default_rng(4)
    g = G_BY_W[4]
    sites = []
    for name, (P, n_in, n_out) in zip("QKVO", SHAPES["Q, K, V and O together"]):
        S = np.where(rng.random((n_in, n_out)) < 0.5, -1.0, 1.0).astype(np.float32)
        sites.append(_homog_site(name, P, n_in, n_out, S * g, 4, 4, "binary", group="attention"))
    out = cp.aggregate(sites)
    assert (out["H1"], out["H2"]) == (393216, 132864)
    assert out["groups"]["attention"]["H2"] == 132864


# ---------------------------------------------------------------------------------------------
# import isolation


def test_no_backend_import_in_fresh_interpreter():
    code = (
        "import sys, json; sys.path.insert(0, %r); import cbl.cost_pure as m; "
        "print(json.dumps({'file': m.__file__, 'loaded': sorted({k.split('.')[0] for k in sys.modules})}))"
        % str(FORK)
    )
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, check=True)
    info = json.loads(r.stdout)
    assert Path(info["file"]).resolve() == (FORK / "cbl" / "cost_pure.py").resolve()
    assert not set(BANNED) & set(info["loaded"])
