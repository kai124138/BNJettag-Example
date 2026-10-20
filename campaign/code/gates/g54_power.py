"""Gate G54 (PREFLIGHT-PLAN section 3 row G54, section 4 row B3b; STUDY v0.9 section 6), restated
for STUDY v0.9 (rule written 2026-10-18T09:11:03Z in the B3b rows file before the rerun).

The section 6 thresholds and power table, recomputed by `cbl/stats.py`, agree with STUDY v0.9
(`review/snapshots/v0.9/STUDY.md`, sha256 61e68527...89df4) as v0.9 implies. Checks, on the Mac CPU
with no backend:
  - repository: HEAD equals --expect-sha12 and `git status --porcelain` is empty;
  - the STUDY snapshot hashes to the v0.9 sha256;
  - the v0.9 power paragraph (:749) states the closed form at rho = 0 and the simulation layout and
    root rule of owner decision B3b-1, and the note below the table (:760) labels the rows;
  - the threshold table (:743-747), parsed from the STUDY text: every entry equals the t-quantile
    threshold at three decimals;
  - the power table (:751-758), parsed from the STUDY text: every |D| = 1 entry equals the
    noncentral-t root and every rho = 0 entry the closed-form root of p(delta)^|D| = 0.8, each at
    two decimals; every rho = 0.9 entry at |D| = 2-6, with r the code's simulated root and SE its
    Monte Carlo standard error, satisfies (a) round(r, 2) within one printed unit (0.01) of the
    entry and (b) |r - entry| <= 0.005 + 4 SE (the entry's rounding interval meets r +- 4 SE);
    exact agreement at printed precision is printed as information;
  - at rho = 0, the simulation (JointSim) within four Monte Carlo standard errors of the closed form,
    a check of the simulation the rho = 0.9 rows use;
  - the other printed section 6 numbers at printed precision, in closed form at rho = 0 and with the
    stated layout where simulated: the single-allowance and joint probabilities at the threshold
    (:763), the rho = 0.5 entries at |D| = 6 (:765), the chi-square band at df = 2 and the pooled
    factor at df = 16 (:764), the one-sided bound at zero gain (:770), the power at
    delta = 0.5 sigma_d (:771), the sign-pattern bounds (:774-780), the "about" ranges of the option
    paragraph (:784-786) and the n = 3 whole-front bars (:782);
  - the unit tests `tests/test_stats.py`, as a sub-step with the same interpreter.

Run as

    python -I -B gates/g54_power.py --expect-sha12 SHA12 [--study PATH]

Prints one line per check and ends with "G54 PASS" (exit 0) or "G54 FAIL" (exit 1). Writes
nothing. Every number printed is an engineering reading, not a quotable result.
"""

import math
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I does not add the script directory
import m16gate as mg  # noqa: E402

sys.path.insert(0, str(mg.FORK))
from cbl import stats as st  # noqa: E402

LEVEL_OF = {"97.5 %": 0.975, "95 %": 0.95}
OPTION_ROWS = {"A, as adopted": "A", "A": "A", "B": "B", "C": "C"}


def table_after(text, header_prefix):
    lines = text.splitlines()
    idx = [i for i, l in enumerate(lines) if l.startswith(header_prefix)]
    if len(idx) != 1:
        raise ValueError("table header %r found %d times" % (header_prefix, len(idx)))
    rows = []
    for l in lines[idx[0] + 2:]:
        if not l.startswith("|"):
            break
        rows.append([c.strip() for c in l.strip().strip("|").split("|")])
    return idx[0] + 1, rows


def fmt(x, d):
    return ("%." + str(d) + "f") % x


def main():
    args = mg.parse_args(__doc__)
    gate = mg.start("G54", args)
    text = mg.study_text(gate, args.study)

    # ---- thresholds
    line, rows = table_after(text, "| n | 95 % two-sided (t quantile 0.975) |")
    gate.info("threshold table", "STUDY line %d, %d rows" % (line, len(rows)))
    cols = [("95 %", 0.95), ("97.5 %", 0.975), ("6 contrasts", st.bonferroni_level(0.05, 6)),
            ("7 contrasts", st.bonferroni_level(0.05, 7)), ("13 contrasts", st.bonferroni_level(0.05, 13)),
            ("91 contrasts", st.bonferroni_level(0.05, 91))]
    gate.check(len(rows) == 3 and [r[0] for r in rows] == ["3", "5", "8"], "threshold table has rows n = 3, 5, 8")
    for r in rows:
        n = int(r[0])
        for (name, level), cell in zip(cols, r[1:]):
            printed = cell.replace("s_d", "").strip()
            got = st.threshold(n, level)
            gate.check(fmt(got, 3) == printed, "threshold n = %d, %s: %s s_d (STUDY %s)" % (n, name, fmt(got, 3), printed))

    # ---- the v0.9 statement of the computation
    s749a = "At ρ = 0 the allowances of a seed are independent, so the joint power is exactly p(δ)^|D|"
    s749b = "draws in order w (200,000 × n), then e (200,000 × n × 6); z = √ρ·w + √(1 − ρ)·e"
    s749c = "the root is the ⌈0.8 × 200,000⌉-th smallest of max_{j<|D|} crit_j"
    s760note = "The ρ = 0 rows are exact (closed form). The ρ = 0.9 rows are Monte Carlo, 200,000 confirmations per cell"
    gate.check(all(s in text for s in (s749a, s749b, s749c, s760note)),
               "STUDY :749 states the closed form at rho = 0 and the B3b-1 layout and root rule; :760 labels the rows")

    # ---- power table
    line, rows = table_after(text, "| option (§8.7) | n, C2 level | ρ |")
    gate.info("power table", "STUDY line %d, %d rows" % (line, len(rows)))
    gate.check(len(rows) == 6 and [r[2] for r in rows] == ["0, exact (closed form)", "0.9, Monte Carlo"] * 3,
               "power table has six rows, rho = 0 labelled exact and rho = 0.9 labelled Monte Carlo")
    mc = []
    for r in rows:
        opt = OPTION_ROWS[r[0]]
        m = re.match(r"(\d+), (97\.5 %|95 %)", r[1])
        n, level = int(m.group(1)), LEVEL_OF[m.group(2)]
        rho = float(r[2].split(",")[0])
        sim = st.joint_sim(n)
        for d, cell in zip(range(1, 7), r[3:9]):
            got = st.power_table_entry(n, level, rho, d)
            printed = float(cell)
            if d == 1 or rho == 0.0:
                how = "noncentral t" if d == 1 else "closed form"
                gate.check(fmt(got, 2) == cell, "power %s (n = %d, level %s), rho = %s, |D| = %d: %.4f -> %s (%s; STUDY %s)"
                           % (opt, n, level, rho, d, got, fmt(got, 2), how, cell))
                if rho == 0.0 and d > 1:
                    simr, se0 = sim.root(0.0, level, d), sim.root_se(0.0, level, d)
                    gate.check(abs(simr - got) < 4 * se0,
                               "simulation check: rho = 0, %s, |D| = %d: simulated %.4f within 4 MC se (%.4f) of the closed form %.4f"
                               % (opt, d, simr, se0, got))
                continue
            se = sim.root_se(rho, level, d)
            unit_ok = abs(int(round(got * 100)) - int(round(printed * 100))) <= 1
            se_ok = abs(got - printed) <= 0.005 + 4 * se
            exact = fmt(got, 2) == cell
            mc.append((opt, d, got, cell, se, exact))
            gate.check(unit_ok and se_ok,
                       "power %s (n = %d, level %s), rho = %s, |D| = %d: simulated %.4f (MC se %.4f) -> %s; STUDY %s: within one "
                       "printed unit %s, |r - STUDY| = %.4f <= 0.005 + 4 se = %.4f %s; equal at printed precision: %s"
                       % (opt, n, level, rho, d, got, se, fmt(got, 2), cell, unit_ok, abs(got - printed), 0.005 + 4 * se,
                          se_ok, "yes" if exact else "no"))
    gate.info("rho = 0.9 cells equal at printed precision", "%d of %d" % (sum(1 for x in mc if x[5]), len(mc)))

    # ---- the other printed section 6 numbers
    opts = (("A", 5, 0.975), ("B", 8, 0.975), ("C", 8, 0.95))
    s759 = "A true gain equal to the threshold passes at one allowance with probability 0.566 (A), 0.536 (B) or 0.531 (C)"
    gate.check(s759 in text, "STUDY :763 text present")
    for (o, n, L), want in zip(opts, ("0.566", "0.536", "0.531")):
        got = st.single_power(st.threshold(n, L), n, L)
        gate.check(fmt(got, 3) == want, "single-allowance power at the threshold, %s: %s (STUDY %s)" % (o, fmt(got, 3), want))
    s759b = ("At six allowances all pass jointly with probability 0.033, 0.133 or 0.363 under A at ρ = 0, 0.5 or 0.9; "
             "0.024, 0.135, 0.350 under B; 0.022, 0.139, 0.351 under C.")
    gate.check(s759b in text, "STUDY :763 joint-probability text present")
    joint_want = {"A": ("0.033", "0.133", "0.363"), "B": ("0.024", "0.135", "0.350"), "C": ("0.022", "0.139", "0.351")}
    for o, n, L in opts:
        for rho, want in zip((0.0, 0.5, 0.9), joint_want[o]):
            got = st.joint_power([st.threshold(n, L)] * 6, n, L, rho)
            how = "closed form p^6" if rho == 0.0 else "simulated, MC se %.4f" % math.sqrt(got * (1 - got) / st.NSIM)
            gate.check(fmt(got, 3) == want, "joint probability at the threshold, %s, rho = %s, |D| = 6: %s (STUDY %s, %s)"
                       % (o, rho, fmt(got, 3), want, how))
    s761 = "for |D| = 6: 2.67 under A, 1.70 under B, 1.49 under C"
    gate.check(s761 in text, "STUDY :765 text present")
    for (o, n, L), want in zip(opts, ("2.67", "1.70", "1.49")):
        got = st.joint_sim(n).root(0.5, L, 6)
        gate.check(fmt(got, 2) == want, "rho = 0.5, |D| = 6, %s: %.4f -> %s (STUDY %s, MC se %.4f)"
                   % (o, got, fmt(got, 2), want, st.joint_sim(n).root_se(0.5, L, 6)))
    s760 = "With df = 2 the 95 % interval of σ_d/s_d is [0.52, 6.28] (χ²)"
    gate.check(s760 in text, "STUDY :764 text present")
    lo, hi = st.chi2_band(2)
    gate.check((fmt(lo, 2), fmt(hi, 2)) == ("0.52", "6.28"), "chi-square band of sigma_d/s_d at df = 2: [%s, %s]" % (fmt(lo, 2), fmt(hi, 2)))
    gate.check("narrows that factor to about 1.5 (df = 16" in text and fmt(st.chi2_band(16)[1], 1) == "1.5",
               "pooled factor at df = 16: %.3f, about 1.5" % st.chi2_band(16)[1])
    s765 = "0.0125 under A and B, 0.025 under C"
    gate.check(s765 in text, "STUDY :770 text present")
    for (o, n, L), want in zip(opts, (0.0125, 0.0125, 0.025)):
        got = st.single_power(0.0, n, L)
        gate.check(abs(got - want) < 1e-12, "power at zero gain, %s: %.6f = one-sided level %s" % (o, got, want))
    s767 = "At δ = 0.5 σ_d the single-allowance power is 0.077 (A), 0.143 (B) or 0.231 (C)"
    gate.check(s767 in text, "STUDY :771 text present")
    for (o, n, L), want in zip(opts, ("0.077", "0.143", "0.231")):
        got = st.single_power(0.5, n, L)
        gate.check(fmt(got, 3) == want, "single-allowance power at 0.5 sigma_d, %s: %s (STUDY %s)" % (o, fmt(got, 3), want))
    for n, one, two in ((3, "1/8", "1/4"), (5, "1/32", "1/16"), (8, "1/256", "1/128")):
        gate.check(Fraction(1, 2 ** n) == Fraction(one) and Fraction(2, 2 ** n) == Fraction(two),
                   "sign-pattern bound n = %d: one-sided %s, two-sided %s" % (n, one, two))
    tail = sum(math.comb(8, k) for k in (7, 8))
    gate.check("seven of eight gives 9/256 = 0.035" in text and Fraction(tail, 256) == Fraction(9, 256)
               and fmt(9 / 256, 3) == "0.035" and fmt(1 / 256, 4) == "0.0039" and fmt(1 / 32, 3) == "0.031",
               "sign counts at n = 8: P(>= 7 of 8) = 9/256 = 0.035; 1/256 = 0.0039; 1/32 = 0.031")
    about = {"A": ("2.4", "2.8"), "B": ("1.5", "1.8"), "C": ("1.3", "1.6")}
    gate.check("about 2.4–2.8 σ_d" in text and "about 1.5–1.8 σ_d" in text and "about 1.3–1.6 σ_d" in text,
               "STUDY :784-786 'about' ranges present")
    for o, n, L in opts:
        lo_, hi_ = st.power_table_entry(n, L, 0.9, 6), st.power_table_entry(n, L, 0.0, 6)  # simulated; closed form
        gate.check((fmt(lo_, 1), fmt(hi_, 1)) == about[o],
                   "80 %% power over six allowances, %s: %.2f (rho 0.9) to %.2f (rho 0) -> about %s-%s"
                   % (o, lo_, hi_, fmt(lo_, 1), fmt(hi_, 1)))
    gate.check("about 25 s_d (91 contrasts) or 9 s_d (13 contrasts)" in text
               and fmt(st.threshold(3, st.bonferroni_level(0.05, 91)), 0) == "25"
               and fmt(st.threshold(3, st.bonferroni_level(0.05, 13)), 0) == "9",
               "n = 3 whole-front bars: %.3f and %.3f s_d -> about 25 and 9" % (
                   st.threshold(3, st.bonferroni_level(0.05, 91)), st.threshold(3, st.bonferroni_level(0.05, 13))))
    gate.info("simulation", {"nsim": st.NSIM, "seed": st.SEED, "layout": "JointSim (cbl/stats.py), owner decision B3b-1"})

    mg.run_pytest(gate, ["tests/test_stats.py"])
    return mg.finish(gate)


if __name__ == "__main__":
    sys.exit(main())
