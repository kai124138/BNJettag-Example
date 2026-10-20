"""Unit tests for cbl/stats.py (gates G54 and G55; STUDY v0.8 sections 6, 8.6, 8.7).

Run with the Mac CPU interpreter, from anywhere:

    python -I -B -m pytest -q -p no:cacheprovider tests/test_stats.py

References are independent of the module's route where one exists: t quantiles by numerical
integration-free identities (df = 1, 2 have closed forms), the chi-square band at df = 2 in closed
form, the joint simulation at rho = 0 against the closed form p^|D|, and the bootstrap and the
background rejection against direct loops. Every number is an engineering check, not a quotable
result.
"""

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

FORK = Path(__file__).resolve().parents[1]
if str(FORK) not in sys.path:
    sys.path.insert(0, str(FORK))

from cbl import stats as st  # noqa: E402


# ---------------------------------------------------------------------------------------------
# t quantiles and intervals


def test_t_crit_closed_forms():
    # df = 1: t_p = tan(pi (p - 1/2)); df = 2: t_p = (2p - 1) sqrt(2 / (4 p (1 - p)))
    for level in (0.95, 0.975, 0.99):
        p = 1 - (1 - level) / 2
        assert st.t_crit(2, level) == pytest.approx(math.tan(math.pi * (p - 0.5)), rel=1e-12)
        a = 4 * p * (1 - p)
        assert st.t_crit(3, level) == pytest.approx((2 * p - 1) * math.sqrt(2 / a), rel=1e-12)


def test_threshold_n3_matches_closed_form():
    p = 0.975
    t2 = (2 * p - 1) * math.sqrt(2 / (4 * p * (1 - p)))
    assert st.threshold(3, 0.95) == pytest.approx(t2 / math.sqrt(3), rel=1e-12)
    assert round(st.threshold(3, 0.95), 3) == 2.484


def test_levels():
    assert st.two_sided_quantile(0.95) == pytest.approx(0.975)
    assert st.two_sided_quantile(0.975) == pytest.approx(0.9875)
    assert st.bonferroni_level(0.05, 7) == pytest.approx(1 - 0.05 / 7)
    with pytest.raises(st.StatsError):
        st.two_sided_quantile(1.0)
    with pytest.raises(st.StatsError):
        st.t_crit(1, 0.95)


def test_paired_interval_identical_arms_is_zero_point():
    iv = st.paired_interval([0.0, 0.0, 0.0])
    assert iv.mean == 0.0 and iv.sd == 0.0 and iv.half_width == 0.0
    assert iv.lower == 0.0 and iv.upper == 0.0 and iv.covers(0.0)
    assert (iv.positive, iv.negative, iv.zero) == (0, 0, 3)


def test_paired_interval_constant_nonzero():
    iv = st.paired_interval([0.01] * 5, level=0.975)
    assert iv.sd == 0.0 and iv.lower == iv.upper == 0.01
    assert iv.positive == 5


def test_paired_interval_values():
    d = [0.010, -0.002, 0.006]
    iv = st.paired_interval(d, 0.95)
    m = sum(d) / 3
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / 2)
    t = st.t_crit(3, 0.95)
    assert iv.mean == pytest.approx(m, abs=1e-15)
    assert iv.sd == pytest.approx(sd, rel=1e-12)
    assert iv.lower == pytest.approx(m - t * sd / math.sqrt(3), rel=1e-12)
    assert iv.upper == pytest.approx(m + t * sd / math.sqrt(3), rel=1e-12)
    assert (iv.positive, iv.negative, iv.zero) == (2, 1, 0)


def test_paired_interval_symmetric_around_zero():
    iv = st.paired_interval([0.003, -0.003, 0.001, -0.001])
    assert iv.mean == pytest.approx(0.0, abs=1e-18)
    assert iv.lower < 0 < iv.upper and iv.lower == pytest.approx(-iv.upper)


def test_paired_interval_refuses():
    with pytest.raises(st.StatsError):
        st.paired_interval([0.1])
    with pytest.raises(st.StatsError):
        st.paired_interval([0.1, float("nan")])


def test_exact_mean():
    assert st.exact_mean([Fraction(1, 3), Fraction(-1, 3)]) == 0
    with pytest.raises(st.StatsError):
        st.exact_mean([])


# ---------------------------------------------------------------------------------------------
# chi-square bands


def test_chi2_band_df2_closed_form():
    # chi2 with 2 df: quantile q_p = -2 ln(1 - p)
    lo = math.sqrt(2 / (-2 * math.log(0.025)))
    hi = math.sqrt(2 / (-2 * math.log(0.975)))
    got = st.chi2_band(2)
    assert got[0] == pytest.approx(lo, rel=1e-10) and got[1] == pytest.approx(hi, rel=1e-10)
    assert round(got[0], 2) == 0.52 and round(got[1], 2) == 6.28
    assert st.screen_sd_upper_factor() == pytest.approx(hi, rel=1e-10)


def test_chi2_band_df16_df14():
    lo, hi = st.chi2_band(16)
    assert round(lo, 3) == 0.745 and round(hi, 3) == 1.522
    lo, hi = st.chi2_band(14)
    assert round(lo, 3) == 0.732 and round(hi, 3) == 1.577


# ---------------------------------------------------------------------------------------------
# single-allowance power


def test_single_power_at_zero_is_one_sided_level():
    for n, level in ((5, 0.975), (8, 0.975), (8, 0.95)):
        assert st.single_power(0.0, n, level) == pytest.approx((1 - level) / 2, rel=1e-9)


def test_single_power_monotone_and_limits():
    ps = [st.single_power(r, 5, 0.975) for r in (-1.0, 0.0, 0.5, 1.0, 2.0, 4.0)]
    assert all(a < b for a, b in zip(ps, ps[1:]))
    assert st.single_power(math.inf, 5, 0.975) == 1.0 and st.single_power(-math.inf, 5, 0.975) == 0.0


def test_single_power_root_inverts():
    for n, level in ((5, 0.975), (8, 0.975), (8, 0.95)):
        r = st.single_power_root(n, level)
        assert st.single_power(r, n, level) == pytest.approx(0.8, abs=1e-10)


# ---------------------------------------------------------------------------------------------
# the joint simulation


@pytest.fixture(scope="module")
def sim5():
    return st.JointSim(5, nsim=200_000)


def test_jointsim_is_deterministic():
    a = st.JointSim(5, nsim=1000)
    b = st.JointSim(5, nsim=1000)
    assert np.array_equal(a.critical(0.9, 0.975), b.critical(0.9, 0.975))
    rng = np.random.default_rng(12345)
    w = rng.standard_normal((1000, 5))
    e = rng.standard_normal((1000, 5, 6))
    z = math.sqrt(0.9) * w[:, :, None] + math.sqrt(0.1) * e
    crit = st.t_crit(5, 0.975) * z.std(axis=1, ddof=1) / math.sqrt(5) - z.mean(axis=1)
    assert np.allclose(a.critical(0.9, 0.975), crit, rtol=0, atol=1e-14)


def test_jointsim_root_is_the_crossing(sim5):
    for rho in (0.0, 0.9):
        for d in (2, 6):
            r = sim5.root(rho, 0.975, d)
            assert sim5.joint_power([r] * d, rho, 0.975) < 0.8
            assert sim5.joint_power([r + 1e-9] * d, rho, 0.975) >= 0.8


def test_jointsim_rho0_matches_closed_form(sim5):
    # rho = 0: the allowances are independent; the simulated root lies within 4 Monte Carlo
    # standard errors of the closed form
    for d in range(2, 7):
        exact = st.independent_joint_root(5, 0.975, d)
        got = sim5.root(0.0, 0.975, d)
        se = sim5.root_se(0.0, 0.975, d)
        assert abs(got - exact) < 4 * se, (d, got, exact, se)


def test_jointsim_power_at_zero_gain_below_one_sided_level(sim5):
    # P(ABOVE ON GRID) <= the one-sided level when the gain is 0 at one allowance (STUDY :765)
    for rho in (0.0, 0.9):
        p = sim5.joint_power([0.0, 5.0, 5.0], rho, 0.975)
        assert p <= 0.0125 + 4 * math.sqrt(0.0125 * 0.9875 / sim5.nsim)


def test_jointsim_nested_domains_monotone(sim5):
    for rho in (0.0, 0.5, 0.9):
        roots = [sim5.root(rho, 0.975, d) for d in range(1, 7)]
        assert all(a <= b for a, b in zip(roots, roots[1:]))


def test_jointsim_correlation_reduces_root(sim5):
    assert sim5.root(0.9, 0.975, 6) < sim5.root(0.0, 0.975, 6)


def test_jointsim_refuses():
    s = st.JointSim(3, nsim=100)
    with pytest.raises(st.StatsError):
        s.joint_power([], 0.0, 0.95)
    with pytest.raises(st.StatsError):
        s.joint_power([1.0] * 7, 0.0, 0.95)
    with pytest.raises(st.StatsError):
        s.critical(1.0, 0.95)


# ---------------------------------------------------------------------------------------------
# plug-in power


def test_plugin_sd_zero_is_na():
    p = st.plugin_power([0.0, 0.0], [0.0, 0.0], 5, 0.975, 0.0, sim=st.JointSim(5, nsim=1000))
    assert p.status == "NA" and p.joint is None and "sd is 0" in p.reason


def test_plugin_equal_gains_equal_table():
    # rho > 0: the simulation; at its own root the simulated probability reaches 0.8
    sim = st.JointSim(8, nsim=20_000)
    r = sim.root(0.9, 0.975, 3)
    p = st.plugin_power([r * 0.01 + 1e-12] * 3, [0.01] * 3, 8, 0.975, 0.9, sim=sim)
    assert p.status == "ok" and p.joint >= 0.8
    assert p.smallest_single == pytest.approx(st.single_power(p.ratios[0], 8, 0.975))
    # rho = 0: the closed form, exactly the product of the single-allowance powers
    r0 = st.independent_joint_root(8, 0.975, 3)
    q = st.plugin_power([r0 * 0.01] * 3, [0.01] * 3, 8, 0.975, 0.0)
    assert q.joint == pytest.approx(0.8, abs=1e-9)
    q2 = st.plugin_power([0.004, 0.01, 0.02], [0.003, 0.004, 0.004], 5, 0.975, 0.0)
    want = 1.0
    for m, s in zip([0.004, 0.01, 0.02], [0.003, 0.004, 0.004]):
        want *= st.single_power(m / s, 5, 0.975)
    assert q2.joint == want


def test_power_table_entry_by_rho():
    for n, level in ((5, 0.975), (8, 0.975), (8, 0.95)):
        assert st.power_table_entry(n, level, 0.9, 1) == st.single_power_root(n, level)
        for d in range(2, 7):
            assert st.power_table_entry(n, level, 0.0, d) == st.independent_joint_root(n, level, d)
            assert st.power_table_entry(n, level, 0.9, d) == st.joint_sim(n).root(0.9, level, d)


def test_joint_power_closed_form_at_rho0_matches_simulation():
    sim = st.JointSim(5, nsim=200_000)
    r = [1.5, 2.0, 2.5]
    exact = st.joint_power(r, 5, 0.975, 0.0)
    simulated = sim.joint_power(r, 0.0, 0.975)
    assert abs(exact - simulated) < 4 * math.sqrt(exact * (1 - exact) / sim.nsim)
    with pytest.raises(st.StatsError):
        st.joint_power([], 5, 0.975, 0.0)


def test_plugin_smallest_single_bounds_joint():
    sim = st.JointSim(5, nsim=50_000)
    for rho in (0.0, 0.9):
        p = st.plugin_power([0.004, 0.01, 0.02], [0.003, 0.004, 0.004], 5, 0.975, rho, sim=sim)
        assert p.joint <= p.smallest_single + 4 * math.sqrt(0.25 / sim.nsim)


def test_plugin_scale_invariant():
    sim = st.JointSim(5, nsim=20_000)
    a = st.plugin_power([0.004, 0.01], [0.003, 0.004], 5, 0.975, 0.9, sim=sim)
    b = st.plugin_power([4.0, 10.0], [3.0, 4.0], 5, 0.975, 0.9, sim=sim)
    assert a.joint == b.joint


# ---------------------------------------------------------------------------------------------
# two-way tables


def test_two_way_additive_table_has_zero_residual():
    rows = np.array([0.60, 0.61, 0.63])[:, None]
    cols = np.array([0.000, 0.002, -0.001])[None, :]
    tw = st.two_way(rows + cols)
    assert tw.ss_residual == 0.0 and tw.ms_residual == 0.0 and tw.df_residual == 4


def test_two_way_identical_rows_zero_residual():
    y = np.tile(np.array([0.65, 0.66, 0.64]), (9, 1))
    tw = st.two_way(y)
    assert tw.ms_residual == 0.0 and tw.df_residual == 16 and tw.ms_rows == 0.0


def test_two_way_against_loop():
    rng = np.random.default_rng(7)
    y = rng.normal(size=(8, 3))
    tw = st.two_way(y)
    r, c = y.shape
    g = y.mean()
    ss = 0.0
    for i in range(r):
        for j in range(c):
            ss += (y[i, j] - y[i].mean() - y[:, j].mean() + g) ** 2
    assert tw.ss_residual == pytest.approx(ss, rel=1e-12)
    assert tw.df_residual == 14
    assert tw.ms_columns == pytest.approx(r * sum((y[:, j].mean() - g) ** 2 for j in range(c)) / 2, rel=1e-12)


def test_two_way_refuses():
    with pytest.raises(st.StatsError):
        st.two_way([[1.0, 2.0]])


# ---------------------------------------------------------------------------------------------
# bootstrap and background rejection


def test_bootstrap_identical_arms_is_zero():
    rng = np.random.default_rng(3)
    a = rng.random((3, 500)) < 0.66
    out = st.paired_bootstrap(a, a.copy(), replicates=200)
    assert out["seed_mean"] == {"observed": 0.0, "lower": 0.0, "upper": 0.0}
    assert all(p == {"observed": 0.0, "lower": 0.0, "upper": 0.0} for p in out["per_seed"])


def test_bootstrap_observed_and_coverage():
    rng = np.random.default_rng(4)
    a = rng.random((2, 4000)) < 0.70
    b = rng.random((2, 4000)) < 0.66
    out = st.paired_bootstrap(a, b, replicates=2000)
    for s in range(2):
        obs = float(a[s].mean() - b[s].mean())
        ps = out["per_seed"][s]
        assert ps["observed"] == pytest.approx(obs, abs=1e-15)
        assert ps["lower"] < obs < ps["upper"]
    assert out["seed_mean"]["observed"] == pytest.approx(float((a.mean(1) - b.mean(1)).mean()), abs=1e-15)


def test_bootstrap_pattern_encoding_matches_direct_resample():
    # the multinomial over patterns gives the same distribution as resampling jets directly: check
    # the bootstrap sd against a direct resample with a generous tolerance
    rng = np.random.default_rng(5)
    a = rng.random((3, 3000)) < 0.7
    b = rng.random((3, 3000)) < 0.65
    out = st.paired_bootstrap(a, b, replicates=4000, level=0.6827)
    d = (a.astype(int) - b.astype(int)).mean(axis=0)
    idx = rng.integers(0, 3000, size=(4000, 3000))
    direct = d[idx].mean(axis=1)
    half_direct = (np.quantile(direct, 0.84135) - np.quantile(direct, 0.15865)) / 2
    half = (out["seed_mean"]["upper"] - out["seed_mean"]["lower"]) / 2
    assert half == pytest.approx(half_direct, rel=0.1)


def _rejection_loop(x, sig, eps_s):
    # direct: thresholds at each distinct score, descending; first point reaching eps_s
    pts = [(0.0, 0.0)]
    for t in sorted(set(x.tolist()), reverse=True):
        sel = x >= t
        pts.append((float((sel & sig).sum() / sig.sum()), float((sel & ~sig).sum() / (~sig).sum())))
    for k, (tp, fp) in enumerate(pts):
        if tp >= eps_s:
            if tp == eps_s:
                return fp
            tp0, fp0 = pts[k - 1]
            return fp0 + (eps_s - tp0) / (tp - tp0) * (fp - fp0)


def test_background_rejection_against_loop():
    rng = np.random.default_rng(6)
    n = 600
    labels = rng.integers(0, 5, size=n)
    scores = rng.random((n, 5))
    scores[np.arange(n), labels] += 0.3
    scores = np.round(scores, 2)  # ties
    for k in range(5):
        r = st.background_rejection(scores, labels, k)
        want = _rejection_loop(scores[:, k], labels == k, 0.5)
        assert r["eps_b"] == pytest.approx(want, abs=1e-15)
        assert r["rejection"] == pytest.approx(1 / want)


def test_background_rejection_perfect_separation_is_inf():
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    scores = np.eye(5)[labels]
    r = st.background_rejection(scores, labels, 0)
    assert r["eps_b"] == 0.0 and r["rejection"] == math.inf


def test_normal_sign_probability():
    assert st.normal_sign_probability(0.0) == 0.5
    assert round(st.normal_sign_probability(0.5), 2) == 0.81
    assert round(st.normal_sign_probability(1.0), 2) == 0.96
