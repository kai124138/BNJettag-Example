"""M16 statistics: intervals, thresholds, power and the shared simulation (PREFLIGHT-PLAN section 2
row M16; STUDY v0.8 sections 6, 8.6 and 8.7).

STUDY means `review/snapshots/v0.8/STUDY.md` (sha256 290022fa...98ca8). Line numbers below refer to
it unless marked v0.9. STUDY v0.9 (`review/snapshots/v0.9/STUDY.md`, sha256 61e68527...89df4) is
the version of record; for this module it differs from v0.8 only in section 6's power table
(v0.9 :749-760): at rho = 0 the joint power is the closed form p(delta)^|D|, and at rho > 0 it is
the simulation of `JointSim`, whose layout v0.9 :749 now states (owner decision B3b-30). Every number this module returns is arithmetic on its inputs or a simulation with fixed
settings; on PREFLIGHT data it is an engineering reading, not a quotable result.

Contents:
  - t intervals on paired seed differences, with the sign counts (section 6 "Reporting", :737);
  - the threshold an observed mean must clear, in observed s_d (section 6 "Thresholds", :739-745);
  - the chi-square band of sigma/sigma-hat (section 6 :760; section 8.6 :971-976);
  - single-allowance power from the noncentral t (section 6 :747);
  - `JointSim`, the one simulation of "ABOVE ON GRID" under the section 6 model (normal paired
    differences, equicorrelation rho between the allowances of one seed, each allowance's t from its
    own n differences), used for the power table (G54) and for the S2 plug-in powers (section 8.6
    :965-970, "the same simulation");
  - the two-way residual mean square behind the pooled sigma-hat_pool (section 8.6 :971) and the
    factorial's randomized-block sigma-hat_F (section 8.6 :1001);
  - the paired bootstrap over held-out jets (section 8.6 :988) and the per-class background
    rejection at signal efficiency 0.5 (section 8 :864; section 8.6 :990).

Owner decisions taken here are recorded in `preflight/gates/G53/<sha12>/PREFLIGHT-rows.md`:
B3b-1 (the simulation layout and the root of the power table), B3b-2 (s_d = 0 in the plug-in
power), B3b-8 (bootstrap replicates and generator), B3b-9 (background rejection interpolation).

Imports: the standard library, numpy and scipy only. No Keras, JAX or backend import (gate G53).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Sequence

import numpy as np
from scipy import stats as _st

NSIM = 200_000  # STUDY :747, "200,000 simulated confirmations per cell"
SEED = 12345  # STUDY :747, "NumPy default_rng(12345)"
D_MAX = 6  # |G|, the grid of section 8.1 (:886)
TARGET_POWER = 0.8  # STUDY :747, "80 % probability of ABOVE ON GRID"
BOOT_REPLICATES = 10_000  # owner decision B3b-8
BOOT_SEED = 12345  # owner decision B3b-8


class StatsError(ValueError):
    """An input outside what a rule defines (too few seeds, a negative sd, a bad level)."""


# ---------------------------------------------------------------------------------------------
# t quantiles, levels, thresholds


def two_sided_quantile(level: float) -> float:
    """The t-distribution probability point of a two-sided interval at `level` (0.95 -> 0.975)."""
    if not 0.0 < level < 1.0:
        raise StatsError("level must lie in (0, 1), got %r" % (level,))
    return 1.0 - (1.0 - level) / 2.0


def bonferroni_level(alpha: float, m: int) -> float:
    """Simultaneous two-sided level of each of m Bonferroni intervals at family alpha."""
    if m < 1:
        raise StatsError("m must be at least 1")
    return 1.0 - alpha / m


def t_crit(n: int, level: float) -> float:
    """Two-sided t critical value at `level` with df = n - 1."""
    if n < 2:
        raise StatsError("a t interval needs at least two seeds, got n = %d" % n)
    return float(_st.t.ppf(two_sided_quantile(level), n - 1))


def threshold(n: int, level: float) -> float:
    """The bar for the observed mean in units of the observed s_d: t * s_d / sqrt(n) / s_d (:739)."""
    return t_crit(n, level) / math.sqrt(n)


# ---------------------------------------------------------------------------------------------
# paired intervals


@dataclass(frozen=True)
class Interval:
    """A two-sided t interval on n paired seed differences (or on n per-seed values)."""

    n: int
    values: tuple
    mean: float
    sd: float  # sample sd, ddof = 1
    level: float
    t: float
    half_width: float
    lower: float
    upper: float
    positive: int  # seeds with a positive difference
    negative: int
    zero: int

    def covers(self, x: float = 0.0) -> bool:
        return self.lower <= x <= self.upper

    def as_dict(self) -> dict:
        return {"n": self.n, "values": list(self.values), "mean": self.mean, "sd": self.sd,
                "level": self.level, "t": self.t, "half_width": self.half_width, "lower": self.lower,
                "upper": self.upper, "positive": self.positive, "negative": self.negative, "zero": self.zero}


def paired_interval(diffs: Sequence[float], level: float = 0.95) -> Interval:
    """Mean, sample sd (ddof = 1), the two-sided t interval at `level` (df = n - 1) and the sign
    counts of n paired differences (STUDY section 6 "Reporting", :737). When every difference is
    equal, the sd is 0 and the interval is the single point [mean, mean]: for identical arms that is
    [0, 0], which covers 0 (G55)."""
    v = np.asarray(list(diffs), dtype=np.float64)
    n = int(v.size)
    if n < 2:
        raise StatsError("an interval needs at least two seeds, got %d" % n)
    if not np.all(np.isfinite(v)):
        raise StatsError("non-finite difference in %r" % (v.tolist(),))
    mean = float(v.mean())
    sd = float(v.std(ddof=1))
    if np.all(v == v[0]):
        sd = 0.0  # exact: avoid a rounding residue when every value is equal
        mean = float(v[0])
    t = t_crit(n, level)
    half = t * sd / math.sqrt(n)
    return Interval(n=n, values=tuple(float(x) for x in v), mean=mean, sd=sd, level=level, t=t,
                    half_width=half, lower=mean - half, upper=mean + half,
                    positive=int(np.sum(v > 0)), negative=int(np.sum(v < 0)), zero=int(np.sum(v == 0)))


def exact_mean(values: Sequence[Fraction]) -> Fraction:
    """Exact mean of Fractions, for rule decisions (> 0, = 0, ties) that must not depend on float
    rounding (interfaces section 0, "Accuracy")."""
    vals = list(values)
    if not vals:
        raise StatsError("mean of an empty set")
    return sum(vals, Fraction(0)) / len(vals)


# ---------------------------------------------------------------------------------------------
# chi-square bands


def chi2_band(df: int, level: float = 0.95) -> tuple:
    """The two-sided `level` band of sigma / sigma-hat for a variance estimate with `df` degrees of
    freedom: [sqrt(df / chi2_{1-a/2}), sqrt(df / chi2_{a/2})]. df = 2 gives [0.52, 6.28] (STUDY :760),
    df = 16 [0.745, 1.522] and df = 14 [0.732, 1.577] (:976)."""
    if df < 1:
        raise StatsError("df must be at least 1")
    a = 1.0 - level
    lo = math.sqrt(df / float(_st.chi2.ppf(1.0 - a / 2.0, df)))
    hi = math.sqrt(df / float(_st.chi2.ppf(a / 2.0, df)))
    return lo, hi


def screen_sd_upper_factor() -> float:
    """The upper 95 % bound of sigma_d / s_d at df = 2 (three screen seeds), printed as 6.28 in the
    STUDY (:760, :964, :969). The code uses the exact value (owner decision B3b-3)."""
    return chi2_band(2)[1]


# ---------------------------------------------------------------------------------------------
# single-allowance power


def single_power(ratio: float, n: int, level: float) -> float:
    """P(lower bound of the two-sided `level` t interval > 0) at true gain / true sd = `ratio`, with
    n paired seeds: the noncentral t survival function at the t critical value (STUDY :747)."""
    c = t_crit(n, level)
    if ratio == math.inf:
        return 1.0
    if ratio == -math.inf:
        return 0.0
    if not math.isfinite(ratio):
        raise StatsError("ratio must be a number, got %r" % (ratio,))
    return float(_st.nct.sf(c, n - 1, ratio * math.sqrt(n)))


def single_power_root(n: int, level: float, target: float = TARGET_POWER) -> float:
    """The true gain / true sd at which the single-allowance power equals `target` (the |D| = 1
    column of the section 6 power table)."""
    from scipy.optimize import brentq

    return float(brentq(lambda r: single_power(r, n, level) - target, 0.0, 50.0, xtol=1e-13, rtol=1e-15,
                        maxiter=500))


def independent_joint_root(n: int, level: float, d: int, target: float = TARGET_POWER) -> float:
    """At rho = 0 the allowances are independent, so P(ABOVE ON GRID) = p^|D| with p the
    single-allowance power, and the root solves p = target^(1/|D|). A closed-form cross-check of
    the simulation (G54), not a value the STUDY prints."""
    from scipy.optimize import brentq

    tgt = target ** (1.0 / d)
    return float(brentq(lambda r: single_power(r, n, level) - tgt, 0.0, 50.0, xtol=1e-13, rtol=1e-15,
                        maxiter=500))


def normal_sign_probability(ratio: float, n: int = 3) -> float:
    """Phi(ratio * sqrt(n)): the normal-approximation probability that the mean of n paired gains is
    positive at true gain / true sd = `ratio` (STUDY :902, the g* entry probability)."""
    return float(_st.norm.cdf(ratio * math.sqrt(n)))


# ---------------------------------------------------------------------------------------------
# the joint simulation


class JointSim:
    """Simulated confirmations under the section 6 model, for one seed count n (owner decision
    B3b-1).

    Layout, fixed before any comparison with the STUDY:
      - one generator `numpy.random.default_rng(seed)` per instance (seed 12345, STUDY :747);
      - draws, in this order: w of shape (nsim, n), one common factor per (confirmation, seed); then
        e of shape (nsim, n, d_max), one term per (confirmation, seed, allowance);
      - z[k, s, j] = sqrt(rho) * w[k, s] + sqrt(1 - rho) * e[k, s, j], so that allowances of one
        seed have correlation rho and seeds are independent;
      - an allowance j of a domain with |D| allowances is column j < |D| (nested domains), and the
        same draws serve every rho, every level and every |D| (common random numbers);
      - with true gain delta_j and true sd sigma_j, the paired differences are
        d = delta_j + sigma_j * z; the t statistic depends only on r_j = delta_j / sigma_j, and
        allowance j passes iff r_j > crit[k, j] = c * s_z[k, j] / sqrt(n) - zbar[k, j], with c the
        two-sided t critical value at the C2 level and s_z the sample sd (ddof = 1) over seeds.

    ABOVE ON GRID passes iff every allowance passes. For a common ratio r, P(r) is the fraction of
    confirmations with r > max_j crit[k, j]; its root at `target` is the smallest r with
    P(r) >= target, which is the ceil(target * nsim)-th smallest of max_j crit[k, j] (the limit of a
    bisection on the same draws).
    """

    def __init__(self, n: int, nsim: int = NSIM, seed: int = SEED, d_max: int = D_MAX):
        if n < 2:
            raise StatsError("n must be at least 2")
        self.n, self.nsim, self.seed, self.d_max = int(n), int(nsim), int(seed), int(d_max)
        rng = np.random.default_rng(self.seed)
        self._w = rng.standard_normal((self.nsim, self.n))
        self._e = rng.standard_normal((self.nsim, self.n, self.d_max))
        self._cache = {}

    def _moments(self, rho: float):
        key = ("m", float(rho))
        if key not in self._cache:
            if not 0.0 <= rho < 1.0:
                raise StatsError("rho must lie in [0, 1), got %r" % (rho,))
            z = math.sqrt(rho) * self._w[:, :, None] + math.sqrt(1.0 - rho) * self._e
            zbar = z.mean(axis=1)
            sz = z.std(axis=1, ddof=1)
            self._cache[key] = (zbar, sz)
        return self._cache[key]

    def critical(self, rho: float, level: float) -> np.ndarray:
        """crit[k, j] for every confirmation k and allowance column j < d_max."""
        key = ("c", float(rho), float(level))
        if key not in self._cache:
            zbar, sz = self._moments(rho)
            self._cache[key] = t_crit(self.n, level) * sz / math.sqrt(self.n) - zbar
        return self._cache[key]

    def joint_power(self, ratios: Sequence[float], rho: float, level: float) -> float:
        """P(ABOVE ON GRID) with ratio r_j at allowance column j (j = 0 .. |D| - 1)."""
        r = np.asarray(list(ratios), dtype=np.float64)
        d = int(r.size)
        if not 1 <= d <= self.d_max:
            raise StatsError("|D| must lie in 1..%d, got %d" % (self.d_max, d))
        if np.any(np.isnan(r)):
            raise StatsError("a ratio is NaN")
        crit = self.critical(rho, level)[:, :d]
        return float(np.mean(np.all(r[None, :] > crit, axis=1)))

    def root(self, rho: float, level: float, d: int, target: float = TARGET_POWER) -> float:
        """The common ratio delta / sigma_d that gives `target` probability of ABOVE ON GRID over d
        allowances (one entry of the section 6 power table)."""
        if not 1 <= d <= self.d_max:
            raise StatsError("d must lie in 1..%d" % self.d_max)
        m = self.critical(rho, level)[:, :d].max(axis=1)
        k = int(math.ceil(target * self.nsim))
        return float(np.partition(m, k - 1)[k - 1])

    def root_se(self, rho: float, level: float, d: int, target: float = TARGET_POWER, h: float = 0.01) -> float:
        """Monte Carlo standard error of `root`: sqrt(p (1 - p) / nsim) divided by the slope of the
        simulated P at the root, from a central difference of half-width h. A reading for the gate
        record, not an input to any rule."""
        r0 = self.root(rho, level, d, target)
        lo = self.joint_power([r0 - h] * d, rho, level)
        hi = self.joint_power([r0 + h] * d, rho, level)
        slope = (hi - lo) / (2.0 * h)
        if slope <= 0:
            return math.inf
        return math.sqrt(target * (1.0 - target) / self.nsim) / slope


_SIMS = {}


def joint_sim(n: int) -> JointSim:
    """The shared JointSim for n seeds at the STUDY's settings (one instance per process)."""
    if n not in _SIMS:
        _SIMS[n] = JointSim(n)
    return _SIMS[n]


def joint_power(ratios: Sequence[float], n: int, level: float, rho: float, sim: Optional[JointSim] = None) -> float:
    """P(ABOVE ON GRID) under the section 6 model with ratio delta_j / sigma_j at allowance j: at
    rho = 0 the allowances of a seed are independent and the probability is exactly the product of
    the single-allowance powers (v0.9 :749); at rho > 0 it is the simulation (`JointSim`)."""
    r = [float(x) for x in ratios]
    if not 1 <= len(r) <= D_MAX:
        raise StatsError("|D| must lie in 1..%d, got %d" % (D_MAX, len(r)))
    if any(math.isnan(x) for x in r):
        raise StatsError("a ratio is NaN")
    if rho == 0.0:
        out = 1.0
        for x in r:
            out *= single_power(x, n, level)
        return out
    sim = sim or joint_sim(n)
    if sim.n != n:
        raise StatsError("simulation n %d differs from n %d" % (sim.n, n))
    return sim.joint_power(r, rho, level)


def power_table_entry(n: int, level: float, rho: float, d: int) -> float:
    """One entry of the section 6 power table: delta / sigma_d for 80 % ABOVE ON GRID over d
    allowances (v0.9 :749-760). The |D| = 1 column is the noncentral-t root; at rho = 0 the root of
    p(delta)^d = 0.8 (closed form); at rho > 0 the simulated root of `JointSim` (layout B3b-1)."""
    if d == 1:
        return single_power_root(n, level)
    if rho == 0.0:
        return independent_joint_root(n, level, d)
    return joint_sim(n).root(rho, level, d)


@dataclass(frozen=True)
class PlugIn:
    """The S2 plug-in power at observed gains over one domain (STUDY section 8.6 :965-970)."""

    status: str  # "ok", or "NA" with a reason
    reason: Optional[str]
    joint: Optional[float]
    smallest_single: Optional[float]
    ratios: tuple

    def as_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason, "joint": self.joint,
                "smallest_single": self.smallest_single, "ratios": list(self.ratios)}


def plugin_ratio(mean: float, sd: float):
    """delta / sigma for the plug-in power. An sd of 0 has no ratio (owner decision B3b-2): it
    returns None, and the caller reports the power as NA with the reason."""
    if sd < 0 or not math.isfinite(sd) or not math.isfinite(mean):
        raise StatsError("bad mean or sd: %r, %r" % (mean, sd))
    if sd == 0.0:
        return None
    return mean / sd


def plugin_power(means: Sequence[float], sds: Sequence[float], n: int, level: float, rho: float,
                 sim: Optional[JointSim] = None) -> PlugIn:
    """Joint plug-in probability of ABOVE ON GRID with the true gain at each allowance set to the
    observed mean and the true sd to `sds` there, and the smallest single-allowance power, which
    bounds the joint power for any rho (STUDY :966-967). The joint value comes from `joint_power`:
    closed form at rho = 0, the simulation at rho > 0 (owner decision B3b-30)."""
    means, sds = list(means), list(sds)
    if len(means) != len(sds) or not means:
        raise StatsError("means and sds must be non-empty and of equal length")
    ratios = []
    for m, s in zip(means, sds):
        r = plugin_ratio(m, s)
        if r is None:
            return PlugIn("NA", "sd is 0 at an allowance: the ratio delta/sigma is undefined (owner decision B3b-2)",
                          None, None, tuple())
        ratios.append(r)
    joint = joint_power(ratios, n, level, rho, sim=sim)
    single = min(single_power(r, n, level) for r in ratios)
    return PlugIn("ok", None, joint, single, tuple(ratios))


# ---------------------------------------------------------------------------------------------
# two-way tables


@dataclass(frozen=True)
class TwoWay:
    """Rows x columns table with row and column effects; residual = interaction (error)."""

    rows: int
    cols: int
    ss_residual: float
    df_residual: int
    ms_residual: float
    ms_columns: float
    ms_rows: float

    @property
    def sigma(self) -> float:
        return math.sqrt(self.ms_residual)

    def as_dict(self) -> dict:
        return {"rows": self.rows, "cols": self.cols, "ss_residual": self.ss_residual,
                "df_residual": self.df_residual, "ms_residual": self.ms_residual,
                "ms_columns": self.ms_columns, "ms_rows": self.ms_rows,
                "sigma": self.sigma,
                "ms_columns_over_ms_residual": (self.ms_columns / self.ms_residual) if self.ms_residual > 0 else None}


def two_way(y) -> TwoWay:
    """Residual mean square after row and column effects of a complete rows x columns table, with
    df = (rows - 1)(cols - 1); the column (seed) and row mean squares beside it. Used for
    sigma-hat_pool (arms x seeds, STUDY :971) and sigma-hat_F (cells x seeds, :1001)."""
    a = np.asarray(y, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 2:
        raise StatsError("a two-way table needs at least 2 rows and 2 columns, got shape %r" % (a.shape,))
    if not np.all(np.isfinite(a)):
        raise StatsError("non-finite entry in the two-way table")
    r, c = a.shape
    grand = a.mean()
    rm = a.mean(axis=1, keepdims=True)
    cm = a.mean(axis=0, keepdims=True)
    res = a - rm - cm + grand
    ss = float(np.sum(res * res))
    if ss < 1e-30 * max(1.0, float(np.sum(a * a))):
        ss = 0.0  # an additive table: rounding residue only
    df = (r - 1) * (c - 1)
    ms_c = float(r * np.sum((cm - grand) ** 2) / (c - 1))
    ms_r = float(c * np.sum((rm - grand) ** 2) / (r - 1))
    return TwoWay(rows=r, cols=c, ss_residual=ss, df_residual=df, ms_residual=ss / df, ms_columns=ms_c, ms_rows=ms_r)


# ---------------------------------------------------------------------------------------------
# held-out jets: paired bootstrap and background rejection


def paired_bootstrap(correct_a, correct_b, level: float = 0.95, replicates: int = BOOT_REPLICATES,
                     seed: int = BOOT_SEED) -> dict:
    """Paired bootstrap over held-out jets of the top-1 difference a - b (STUDY :988).

    `correct_a`, `correct_b`: boolean arrays of shape (S, n_jets), seed s's per-jet argmax agreement
    of the two arms on the same jets in the same order. Jets are resampled with replacement, the
    same indices for every seed and both arms, so the seed-mean difference is resampled jointly.
    Exactly: only the per-jet pattern of differences across seeds matters, so the resample is a
    multinomial draw over the observed patterns (owner decision B3b-8). Returns per seed and for the
    seed mean: the observed difference and the percentile interval at `level`."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.ndim == 1:
        a, b = a[None, :], b[None, :]
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] < 1:
        raise StatsError("correct_a and correct_b must have equal shape (S, n_jets)")
    S, n = a.shape
    d = a.astype(np.int8) - b.astype(np.int8)  # in {-1, 0, 1}
    code = np.zeros(n, dtype=np.int64)
    for s in range(S):
        code = code * 3 + (d[s].astype(np.int64) + 1)
    uniq, counts = np.unique(code, return_counts=True)
    # pattern -> per-seed difference vector
    pat = np.zeros((uniq.size, S), dtype=np.float64)
    rem = uniq.copy()
    for s in range(S - 1, -1, -1):
        pat[:, s] = (rem % 3) - 1
        rem //= 3
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, counts / n, size=replicates).astype(np.float64)  # (B, K)
    per_seed = draws @ pat / n  # (B, S)
    mean = per_seed.mean(axis=1)
    q = ((1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0)
    obs = d.mean(axis=1).astype(np.float64)
    out = {"replicates": replicates, "seed": seed, "level": level, "n_jets": n, "seeds": S,
           "per_seed": [], "seed_mean": None}
    for s in range(S):
        lo, hi = np.quantile(per_seed[:, s], q)
        out["per_seed"].append({"observed": float(obs[s]), "lower": float(lo), "upper": float(hi)})
    lo, hi = np.quantile(mean, q)
    out["seed_mean"] = {"observed": float(obs.mean()), "lower": float(lo), "upper": float(hi)}
    return out


def background_rejection(scores, labels, cls: int, eps_s: float = 0.5) -> dict:
    """One-vs-rest background rejection 1/eps_B at signal efficiency eps_s for class `cls`
    (STUDY :864, :990; docs/conventions/jet-tagging-metrics.md:35-36).

    Owner decision B3b-9: the ROC curve with tied scores grouped (one point per distinct score,
    thresholds descending, starting at (0, 0)); at the first point whose eps_S reaches eps_s, eps_B
    is that point's value if its eps_S equals eps_s, and otherwise the linear interpolation between
    it and the point before; rejection = 1 / eps_B, and +inf when eps_B = 0 there."""
    sc = np.asarray(scores, dtype=np.float64)
    lab = np.asarray(labels)
    if sc.ndim != 2 or lab.ndim != 1 or sc.shape[0] != lab.shape[0]:
        raise StatsError("scores must be (n, classes) and labels (n,)")
    if not 0 <= cls < sc.shape[1]:
        raise StatsError("class index out of range")
    x = sc[:, cls]
    sig = lab == cls
    n_s, n_b = int(sig.sum()), int((~sig).sum())
    if n_s == 0 or n_b == 0:
        raise StatsError("class %d has no signal or no background jets" % cls)
    order = np.argsort(-x, kind="mergesort")
    xs, ss = x[order], sig[order]
    tp = np.cumsum(ss)
    fp = np.cumsum(~ss)
    last = np.r_[np.nonzero(np.diff(xs))[0], xs.size - 1]  # last index of each distinct score
    tpr = np.r_[0.0, tp[last] / n_s]
    fpr = np.r_[0.0, fp[last] / n_b]
    if not 0.0 < eps_s <= 1.0:
        raise StatsError("eps_s must lie in (0, 1]")
    i = int(np.argmax(tpr >= eps_s))  # first point reaching eps_s; tpr[-1] = 1 >= eps_s
    if tpr[i] == eps_s:
        eps_b = float(fpr[i])
    else:
        f = (eps_s - tpr[i - 1]) / (tpr[i] - tpr[i - 1])
        eps_b = float(fpr[i - 1] + f * (fpr[i] - fpr[i - 1]))
    rej = math.inf if eps_b == 0.0 else 1.0 / eps_b
    return {"class": cls, "eps_s": eps_s, "eps_b": eps_b, "rejection": rej, "n_signal": n_s, "n_background": n_b}
