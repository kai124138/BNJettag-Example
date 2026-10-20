"""Unit tests for cbl/confirm.py (gate G55; STUDY v0.8 sections 8.7, 8.8 "Attention use", 8.10).

Run with the Mac CPU interpreter, from anywhere:

    python -I -B -m pytest -q -p no:cacheprovider tests/test_confirm.py

Constructed cases only (tests/fixtures/m16_synth.py); held-out values are invented counts out of
260,000 jets. Every number is an engineering check, not a quotable result.
"""

import math
import sys
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[1]
for p in (str(FORK), str(FORK / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cbl import confirm as cf  # noqa: E402
from cbl import select as sel  # noqa: E402
from cbl import stats as st  # noqa: E402
from fixtures import m16_synth as syn  # noqa: E402

G = sel.GRID
NT = 260_000
CLAIM_A = sel.claim_domain_config({"base": "a", "c3r": False})
CLAIM_B = sel.claim_domain_config({"base": "b", "band_rule": "default", "c3r": False})


def rising_env(lowest=100_000, base=40_000):
    """strictly rising validation envelope: one checkpoint per g"""
    return {g: (base + g // 1000 if g >= lowest else None) for g in G}


class S4:
    """BEST-C and CH-8-C runs over the seeds of an option, with held-out counts per checkpoint."""

    def __init__(self, seeds, best_lowest=None, ch_lowest=None, ends=None):
        self.seeds = tuple(seeds)
        self.best, self.ch8, self.heldout = {}, {}, {}
        for s in self.seeds:
            bl = (best_lowest or {}).get(s, 100_000)
            cl = (ch_lowest or {}).get(s, 100_000)
            c, sm = syn.run_records("F011-FB", s, rising_env(bl), stage="S4", arm_id="BEST-C", end=(ends or {}).get(s, "completed"))
            self.best[s] = add_sidecars(sel.run_from_records(c, sm))
            c, sm = syn.run_records("CH-8", s, rising_env(cl, base=40_010), stage="S4", arm_id="CH-8-C")
            self.ch8[s] = add_sidecars(sel.run_from_records(c, sm))
        for s in self.seeds:
            for c in sel.COUNTS:
                for g in G:
                    for r in (self.best[s], self.ch8[s]):
                        p = r.point(c, g)
                        if p is not None:
                            self.heldout.setdefault(p.ckpt_sha256, {"correct": 170_000, "n": NT})

    def set_gap(self, g, gaps, count="H1"):
        """held-out top-1(BEST) - top-1(CH-8) at g, in jets of 260,000, per seed"""
        for s, d in zip(self.seeds, gaps):
            q = self.ch8[s].point(count, g)
            p = self.best[s].point(count, g)
            if p is None or q is None:
                continue
            self.heldout[p.ckpt_sha256] = {"correct": self.heldout[q.ckpt_sha256]["correct"] + d, "n": NT}


def add_sidecars(run):
    """a front sidecar with a non-degenerate structure record for every front checkpoint"""
    for c in sel.COUNTS:
        for p in run.front[c]:
            run.sidecars.setdefault(p.ckpt_sha256, {"epoch": p.epoch, "structure_record": {
                "degenerate": {"Q": None, "K": None, "V": None, "O": None, "FFN": None}}})
    return run


def design(option, claim=CLAIM_A, contains=None):
    return sel.s4_design(option, claim, contains)


def dom(*gs):
    return sel.Domain("ok", tuple(gs))


# ---------------------------------------------------------------------------------------------
# identical arms


def test_identical_arms_zero_differences_interval_around_zero():
    d = design("A")
    x = S4(d.seeds)
    for s in d.seeds:  # BEST-C equals CH-8-C jet for jet: same held-out value at every point
        for g in G:
            x.heldout[x.best[s].point("H1", g).ckpt_sha256] = dict(x.heldout[x.ch8[s].point("H1", g).ckpt_sha256])
    rec = cf.c2(d, CLAIM_A, dom(*G), dom(*G), x.best, x.ch8, x.heldout, "H1")
    assert rec["verdict"] == cf.FLAT
    for row in rec["per_g"].values():
        assert row["per_seed"] == [0.0] * 5
        iv = row["interval"]
        assert iv["mean"] == 0.0 and iv["lower"] == 0.0 == iv["upper"] and iv["lower"] <= 0 <= iv["upper"]


# ---------------------------------------------------------------------------------------------
# C2


def test_c2_above_on_grid_and_naming():
    d = design("C")  # n = 8, L = 95 %
    x = S4(d.seeds)
    for g in G:
        for c in ("H1", "H2"):  # each count has its own checkpoints in the fixture
            x.set_gap(g, [900, 1100, 1000, 950, 1050, 1000, 980, 1020], count=c)
    rec = cf.c2(d, CLAIM_A, dom(*G), dom(300_000, 400_000), x.best, x.ch8, x.heldout, "H1")
    assert rec["verdict"] == cf.ABOVE_ON_GRID and rec["level"] == 0.95 and rec["naming"] == "front result"
    assert rec["size"] == 2 and rec["per_g"]["100000"]["note"] == "outside the claim"
    assert rec["per_g"]["300000"]["interval"]["positive"] == 8  # the sign count beside each interval
    assert rec["distinct_pairs_per_seed"] == {str(s): 2 for s in d.seeds}
    assert rec["non_headline"]["verdict"] == cf.ABOVE_ON_GRID and rec["accounting_dependent"] is False
    one = cf.c2(d, CLAIM_A, dom(*G), dom(450_000), x.best, x.ch8, x.heldout, "H1")
    assert one["naming"] == "single-allowance comparison"


def test_c2_threshold_matches_table():
    # the observed mean must exceed threshold * s_d: just above and just below, on gaps with a known sd
    d = design("A")
    thr = st.threshold(5, 0.975)
    base = [-2, -1, 0, 1, 2]  # sd = sqrt(2.5) jets
    sd = math.sqrt(2.5)
    for mean, want in ((thr * sd * 1.01, cf.ABOVE_ON_GRID), (thr * sd * 0.99, cf.FLAT)):
        x = S4(d.seeds)
        # integer jets cannot carry a fractional mean: scale everything by 1000
        gaps = [int(round(1000 * (b + mean))) for b in base]
        x.set_gap(450_000, gaps)
        rec = cf.c2(d, CLAIM_A, dom(450_000), dom(450_000), x.best, x.ch8, x.heldout, "H1")
        iv = rec["per_g"]["450000"]["interval"]
        assert (iv["mean"] / iv["sd"] > thr) == (want == cf.ABOVE_ON_GRID)
        assert rec["verdict"] == want


def test_c2_below_on_grid_with_cost_infeasible():
    d = design("A")
    x = S4(d.seeds, best_lowest={6: 400_000})  # BEST infeasible below 400,000 in seed 6
    for g in (400_000, 450_000):
        x.set_gap(g, [-900, -1100, -1000, -950, -1050])
    rec = cf.c2(d, CLAIM_A, dom(300_000, 400_000, 450_000), dom(300_000, 400_000, 450_000), x.best, x.ch8, x.heldout, "H1")
    assert rec["per_g"]["300000"]["status"] == sel.COST_INFEASIBLE
    assert rec["verdict"] == cf.BELOW_ON_GRID


def test_c2_cost_infeasible_blocks_above():
    d = design("A")
    x = S4(d.seeds, best_lowest={5: 400_000})
    for g in G:
        x.set_gap(g, [900, 1100, 1000, 950, 1050])
    rec = cf.c2(d, CLAIM_A, dom(300_000, 400_000), dom(300_000, 400_000), x.best, x.ch8, x.heldout, "H1")
    assert rec["verdict"] == cf.FLAT  # one allowance COST INFEASIBLE (a deficit), the other above zero


def test_c2_incomplete_cases():
    d = design("A")
    x = S4(d.seeds)
    full = dom(*G)
    assert cf.c2(d, CLAIM_A, full, full, x.best, x.ch8, x.heldout, "H1", gate_failures=["G33 failed"])["verdict"] == sel.INCOMPLETE
    best = dict(x.best)
    del best[7]
    assert "seed 7 missing" in cf.c2(d, CLAIM_A, full, full, best, x.ch8, x.heldout, "H1")["reason"]
    y = S4(d.seeds, ends={4: "failed"})
    assert cf.c2(d, CLAIM_A, full, full, y.best, y.ch8, y.heldout, "H1")["verdict"] == sel.INCOMPLETE
    empty = sel.Domain(sel.INCOMPLETE, (), "D_conf is empty")
    assert cf.c2(d, CLAIM_A, empty, empty, x.best, x.ch8, x.heldout, "H1")["verdict"] == sel.INCOMPLETE
    noband = sel.Domain(sel.INCOMPLETE_NO_BAND, (), "D_band is empty")
    assert cf.c2(d, CLAIM_B, full, noband, x.best, x.ch8, x.heldout, "H1")["verdict"] == sel.INCOMPLETE_NO_BAND
    with pytest.raises(sel.UnsetDecision):
        cf.c2(d, CLAIM_A, full, full, x.best, x.ch8, x.heldout, None)


def test_c2_strong_form_nested_in_band():
    d = design("B")
    x = S4(d.seeds)
    for g in G:
        x.set_gap(g, [800, 1200, 1000, 900, 1100, 1000, 950, 1050] if g >= 300_000 else [-50, 60, -40, 30, 10, -20, 0, 5])
    band = dom(300_000, 350_000, 400_000, 450_000)
    rec = cf.c2(d, CLAIM_B, dom(*G), band, x.best, x.ch8, x.heldout, "H1")
    assert rec["verdict"] == cf.ABOVE_ON_GRID and rec["strong_form"] == "not ABOVE ON ALL OF D_conf"
    for g in (100_000, 200_000):
        assert rec["per_g"][str(g)]["in_claim"] is False and rec["per_g"][str(g)]["note"] == "outside the claim"
    for g in (100_000, 200_000):
        x.set_gap(g, [800, 1200, 1000, 900, 1100, 1000, 950, 1050])
    rec = cf.c2(d, CLAIM_B, dom(*G), band, x.best, x.ch8, x.heldout, "H1")
    assert rec["strong_form"] == "ABOVE ON ALL OF D_conf"  # its rejection implies the band's


def test_c2_accounting_dependent():
    d = design("A")
    x = S4(d.seeds)
    for g in G:
        x.set_gap(g, [900, 1100, 1000, 950, 1050], count="H1")
    # under H2 the envelopes pick the same checkpoints (the builder copies the front), so make H2's
    # contrast differ by pointing its BEST points at checkpoints with a deficit
    for s in d.seeds:
        for g in G:
            p2 = x.best[s].point("H2", g)
            q2 = x.ch8[s].point("H2", g)
            x.heldout[p2.ckpt_sha256] = {"correct": x.heldout[q2.ckpt_sha256]["correct"] - 1000, "n": NT}
    rec = cf.c2(d, CLAIM_A, dom(*G), dom(*G), x.best, x.ch8, x.heldout, "H1")
    assert rec["verdict"] == cf.ABOVE_ON_GRID and rec["non_headline"]["verdict"] == cf.BELOW_ON_GRID
    assert rec["accounting_dependent"] is True


def test_c2_cost_reach_record():
    d = design("A")
    x = S4(d.seeds)
    rec = cf.c2(d, CLAIM_A, dom(300_000, 350_000, 400_000, 450_000), dom(400_000, 450_000), x.best, x.ch8, x.heldout, "H1")
    assert rec["cost_reach"]["allowances"] == [100_000, 200_000]


def test_degenerate_labels():
    d = design("A")
    x = S4(d.seeds)
    for g in G:
        x.set_gap(g, [900, 1100, 1000, 950, 1050])
    p = x.best[6].point("H1", 400_000)
    x.best[6].sidecars[p.ckpt_sha256]["structure_record"]["degenerate"]["K"] = "kernel_pruned"
    rec = cf.c2(d, CLAIM_A, dom(*G), dom(400_000, 450_000), x.best, x.ch8, x.heldout, "H1")
    assert rec["degenerate_labels"]["400000"] == ["with degenerate K (kernel_pruned)"]
    assert rec["verdict_labels"] == ["with degenerate K (kernel_pruned)"] and rec["ch8_degenerate_beside"]["400000"] == []


# ---------------------------------------------------------------------------------------------
# descriptive labels


def _pid_ok(per_seed):
    return {"status": "ok", "per_seed": per_seed}


@pytest.mark.parametrize("option,m", [("A", 7), ("C", 6)])
def test_descriptive_labels(option, m):
    d = design(option)
    n = d.n
    big = [3000 + 10 * i for i in range(n)]
    neg = [-3000 - 10 * i for i in range(n)]
    noise = [(-1) ** i * 200 for i in range(n)]
    pid = _pid_ok([0.0] * n) if d.pid else None
    x = S4(d.seeds)
    for g in G:
        x.set_gap(g, big if g >= 400_000 else neg)
    rec = cf.c2(d, CLAIM_A, dom(*G), dom(*G), x.best, x.ch8, x.heldout, "H1")
    lab = cf.descriptive_labels(d, rec, dom(*G), x.best, x.ch8, x.heldout, "H1", pid=pid)
    assert lab["contrasts"] == m and lab["level"] == pytest.approx(1 - 0.05 / m) and lab["label"] == cf.CROSSING
    x = S4(d.seeds)
    for g in G:
        x.set_gap(g, big if g >= 400_000 else noise)
    rec = cf.c2(d, CLAIM_A, dom(*G), dom(*G), x.best, x.ch8, x.heldout, "H1")
    assert rec["verdict"] == cf.FLAT
    assert cf.descriptive_labels(d, rec, dom(*G), x.best, x.ch8, x.heldout, "H1", pid=pid)["label"] == cf.LOCAL_ADVANTAGE
    x = S4(d.seeds)
    for g in G:
        x.set_gap(g, neg if g >= 400_000 else noise)
    rec = cf.c2(d, CLAIM_A, dom(*G), dom(*G), x.best, x.ch8, x.heldout, "H1")
    assert cf.descriptive_labels(d, rec, dom(*G), x.best, x.ch8, x.heldout, "H1", pid=pid)["label"] == cf.LOCAL_DEFICIT
    # the 7- and 6-contrast thresholds of section 6 at n = 5 and 8
    want = {("A", 7): 2.266, ("C", 6): 1.285}[(option, m)]
    assert round(lab["threshold_in_s_d"], 3) == want


def test_descriptive_labels_whole_front_gives_no_local_label():
    d = design("C")
    x = S4(d.seeds)
    for g in G:
        x.set_gap(g, [3000 + 10 * i for i in range(8)])
    rec = cf.c2(d, CLAIM_A, dom(*G), dom(*G), x.best, x.ch8, x.heldout, "H1")
    assert rec["verdict"] == cf.ABOVE_ON_GRID
    assert cf.descriptive_labels(d, rec, dom(*G), x.best, x.ch8, x.heldout, "H1")["label"] is None


# ---------------------------------------------------------------------------------------------
# C3 and C3r


def _pid_runs(seeds, best_env=None, ch_env=None, gaps=None):
    pb, pc, h = {}, {}, {}
    for s in seeds:
        c, sm = syn.run_records("F011-FB", s, best_env or rising_env(300_000), stage="S4", schedule="pid", arm_id="PID-BEST-C")
        pb[s] = add_sidecars(sel.run_from_records(c, sm))
        c, sm = syn.run_records("CH-8", s, ch_env or rising_env(300_000, base=40_010), stage="S4", schedule="pid", arm_id="PID-CH-C")
        pc[s] = add_sidecars(sel.run_from_records(c, sm))
    for i, s in enumerate(seeds):
        for r in (pb[s], pc[s]):
            for p in r.front["H1"]:
                h.setdefault(p.ckpt_sha256, {"correct": 170_000, "n": NT})
        a, b = sel.pid_selection(pb[s], "H1")["point"], sel.pid_selection(pc[s], "H1")["point"]
        if a and b and gaps:
            h[a["ckpt_sha256"]] = {"correct": 170_000 + gaps[i], "n": NT}
    return pb, pc, h


def test_c3_labels():
    d = design("A")
    pb, pc, h = _pid_runs(d.seeds, gaps=[900, 1100, 1000, 950, 1050])
    r = cf.c3(d, pb, pc, h, "H1")
    assert r["verdict"] == cf.ABOVE and r["level"] == 0.975
    pb, pc, h = _pid_runs(d.seeds, gaps=[-900, -1100, -1000, -950, -1050])
    assert cf.c3(d, pb, pc, h, "H1")["verdict"] == cf.BELOW
    pb, pc, h = _pid_runs(d.seeds, gaps=[900, -1100, 1000, -950, 50])
    assert cf.c3(d, pb, pc, h, "H1")["verdict"] == cf.FLAT
    pb, pc, h = _pid_runs(d.seeds, best_env=rising_env(400_000), gaps=[1] * 5)
    assert cf.c3(d, pb, pc, h, "H1")["verdict"] == sel.COST_INFEASIBLE
    pb, pc, h = _pid_runs(d.seeds, ch_env=rising_env(400_000, base=40_010), gaps=[1] * 5)
    assert cf.c3(d, pb, pc, h, "H1")["verdict"] == sel.INCOMPLETE
    pb, pc, h = _pid_runs(d.seeds, best_env={g: None for g in G}, gaps=[1] * 5)
    pb[5].filter_outcome["H1"][350_000] = {"outcome": sel.NO_ACCEPTED}
    for s in d.seeds:
        if s != 5:
            pb[s].filter_outcome["H1"][350_000] = {"outcome": sel.NO_ACCEPTED}
    assert cf.c3(d, pb, pc, h, "H1")["verdict"] == cf.NO_ACCEPTED_LABEL
    assert cf.c3(design("C"), pb, pc, h, "H1")["tested"] is False


def test_c3r_levels_and_labels():
    cb = sel.claim_domain_config({"base": "a", "c3r": True, "c3r_level": "bonferroni"})
    cn = sel.claim_domain_config({"base": "a", "c3r": True, "c3r_level": "nested"})
    d = sel.s4_design("C", cb)
    x = S4(d.seeds)
    x.set_gap(350_000, [900, 1100, 1000, 950, 1050, 1000, 980, 1020])
    r = cf.c3r(d, x.best, x.ch8, x.heldout, "H1")
    assert r["verdict"] == cf.ABOVE and r["level"] == 0.975
    dn = sel.s4_design("C", cn, contains_350k=True)
    assert cf.c3r(dn, x.best, x.ch8, x.heldout, "H1")["level"] == 0.95
    y = S4(d.seeds, best_lowest={9: 400_000})
    assert cf.c3r(d, y.best, y.ch8, y.heldout, "H1")["verdict"] == sel.COST_INFEASIBLE
    z = S4(d.seeds, ch_lowest={9: 400_000})
    assert cf.c3r(d, z.best, z.ch8, z.heldout, "H1")["verdict"] == sel.INCOMPLETE
    assert cf.c3r(design("C"), x.best, x.ch8, x.heldout, "H1")["tested"] is False


# ---------------------------------------------------------------------------------------------
# C4


def _pairs(n, lut_gap, acc_gap, **kw):
    out = []
    for i in range(n):
        p = {"lut_best": 50_000 + lut_gap + 37 * ((-1) ** i), "lut_ch8": 50_000, "part": "xczu7ev", "stage": "synth+opt",
             "settings": "io_parallel RF1 2.5ns", "status": "ok", "allowance_exceeded_after_trace": False,
             "post_trace_top1_best": 0.66 + acc_gap + 0.0001 * ((-1) ** i), "post_trace_top1_ch8": 0.66, "function_changed": False}
        p.update(kw)
        out.append(p)
    return out


def test_c4_labels():
    d = design("A")
    above = {"verdict": cf.ABOVE_ON_GRID, "d_c2": {"allowances": [400_000, 450_000]}}
    dec = {str(g): _pairs(5, -2000, 0.004) for g in (400_000, 450_000)}
    assert cf.c4(d, above, dec)["verdict"] == "SUPPORTED"
    dec["450000"] = _pairs(5, 2000, 0.004)
    r = cf.c4(d, above, dec)
    assert r["verdict"] == "BLOCKED" and r["per_g"]["450000"]["label"] == cf.NOT_LUT_ADVANTAGE
    assert r["per_g"]["450000"]["per_pair_labels"] == ["not a LUT advantage"] * 5
    dec["450000"] = _pairs(5, 0, 0.004)
    assert cf.c4(d, above, dec)["verdict"] == "UNRESOLVED"  # equal LUT with higher accuracy: dominance not shown
    dec["450000"] = _pairs(5, -2000, 0.004, part="xcvu13p")[:4] + _pairs(5, -2000, 0.004)[:1]
    assert cf.c4(d, above, dec)["verdict"] == cf.HW_INCOMPLETE
    dec["450000"] = _pairs(5, -2000, 0.004, allowance_exceeded_after_trace=True)
    assert cf.c4(d, above, dec)["per_g"]["450000"]["label"] == cf.HW_INCOMPLETE
    dec["450000"] = _pairs(5, -2000, -0.004)
    assert cf.c4(d, above, dec)["verdict"] == "BLOCKED"  # post-trace accuracy upper bound < 0
    dec["450000"] = _pairs(5, -2000, 0.004, function_changed=True)
    assert cf.c4(d, above, dec)["verdict"] == "BLOCKED"
    assert cf.c4(d, {"verdict": cf.FLAT}, dec)["verdict"].startswith("NOT EVALUATED")
    r = cf.c4(d, above, {str(g): _pairs(5, -2000, 0.004) for g in (400_000, 450_000)} | {"PID": _pairs(5, -2000, 0.004)},
              c3_record={"verdict": cf.ABOVE, "level": 0.975})
    assert r["pid_point"]["label"] == cf.LUT_ADVANTAGE and r["pid_point"]["point"] == "PID"


# ---------------------------------------------------------------------------------------------
# C6


def _side(da=0.02, ds=0.01, du=0.015, h=1.6, deg=None, q_zero=False, q_same=None, overflow=None):
    return {"attention_diagnostics": {"delta_attn_mean": da, "delta_static": ds, "delta_unif": du, "H_attn": h,
                                      "overflow": overflow or {}},
            "structure_record": {"degenerate": deg or {}, "q_all_zero": q_zero, "q_identical_across_t": q_same}}


def _seeds(n, **kw):
    out = []
    for i in range(n):
        s = _side(**kw)
        for k in ("delta_attn_mean", "delta_static", "delta_unif"):
            s["attention_diagnostics"][k] += 0.001 * ((-1) ** i)
        out.append(s)
    return out


def test_c6_labels():
    used = cf.c6_point(_seeds(5))
    assert used["label"] == "used" and all(used["conditions"].values())
    assert cf.c6_point(_seeds(5, ds=0.0))["label"] == "not demonstrated"  # a fixed pattern is not use
    assert cf.c6_point(_seeds(5, h=math.log(8)))["label"] == "not demonstrated"
    k = _seeds(5)
    k[2]["structure_record"]["degenerate"]["K"] = "kernel_pruned"
    assert cf.c6_point(k)["label"] == "not demonstrated"
    q = _seeds(5)
    q[1]["structure_record"]["degenerate"]["Q"] = "kernel_rounded_to_zero"
    assert cf.c6_point(q)["label"] == "used, with input-independent queries"
    q[1]["structure_record"]["q_identical_across_t"] = [True, True]
    assert cf.c6_point(q)["label"] == "used, as query-independent attention pooling"
    q[1]["structure_record"]["q_all_zero"] = True
    assert cf.c6_point(q)["label"] == "not demonstrated"  # reported as for a degenerate K
    o = _seeds(5, overflow={"delta_static": 3})
    assert cf.c6_point(o)["label"] == "used (overflow-affected)"
    overall = cf.c6({400_000: _seeds(5), 450_000: q[1:2] + _seeds(4)}, dom(400_000, 450_000))
    assert overall["overall"] == "not demonstrated"
    q2 = _seeds(5)
    q2[0]["structure_record"]["degenerate"]["Q"] = "other"
    assert cf.c6({400_000: _seeds(5), 450_000: q2}, dom(400_000, 450_000))["overall"] == "used, with input-independent queries"
    assert cf.c6({400_000: _seeds(5)}, dom(400_000, 450_000))["overall"] == "not demonstrated"  # missing g of D_C2
    assert cf.LOG8 == pytest.approx(2.0794, abs=5e-5)


# ---------------------------------------------------------------------------------------------
# section 8.10


def _ctx_runs(vals):
    runs, h = {}, {}
    for s, v in enumerate(vals, start=1):
        rf = [{"epoch": 100, "train_count": 340_000.0, "log_val_accuracy": 0.66, "path": "s%d/a.keras" % s}]
        runs[s] = sel.run_from_records(*syn.run_records("CH-8", s, syn.flat_env(40_000), released_front=rf))
        h["s%d/a.keras" % s] = {"correct": int(round(v * NT)), "n": NT}
    return runs, h


def test_pull_8_10():
    runs, h = _ctx_runs([0.660, 0.662, 0.664])
    p = cf.pull_8_10(runs, h, "S1")
    assert p["label"] == sel.LABEL_CONTEXT_8_10 and p["difference"] == pytest.approx(-0.001, abs=2e-6)
    assert p["sd"] == pytest.approx(0.002, abs=1e-5) and p["pull"] == pytest.approx(-0.5, abs=0.01)
    assert not p["triggered"] and p["trigger_bar"] == 0.01
    runs, h = _ctx_runs([0.640, 0.642, 0.644])
    p = cf.pull_8_10(runs, h, "S1")
    assert p["triggered"] and "quantitative cause" in p["consequence"]
    runs, h = _ctx_runs([0.66, 0.66, 0.66])
    p = cf.pull_8_10(runs, h, "S1", pid_ch_heldout=[0.65, 0.66, 0.67])
    assert p["sd"] == 0.0 and p["pull"] is None and p["pid_ch_seed_mean"] == pytest.approx(0.66)


def test_c3_refuses_disagreeing_pid_summary():
    d = design("A")
    pb, pc, h = _pid_runs(d.seeds, gaps=[900, 1100, 1000, 950, 1050])
    pc[6] = sel.Run(**{**pc[6].__dict__, "pid_selection": {"ckpt_sha256": "e" * 64}})
    with pytest.raises(sel.SelectError, match="pid_selection"):
        cf.c3(d, pb, pc, h, "H1")


def test_descriptive_labels_read_at_grid_points_only():
    # v0.9 :1039: the PID contrast is in the Bonferroni family but is not read for the labels
    d = design("A")
    x = S4(d.seeds)
    for g in G:
        x.set_gap(g, [(-1) ** i * 200 for i in range(5)])
    rec = cf.c2(d, CLAIM_A, dom(*G), dom(*G), x.best, x.ch8, x.heldout, "H1")
    lab = cf.descriptive_labels(d, rec, dom(*G), x.best, x.ch8, x.heldout, "H1",
                                pid={"status": "ok", "per_seed": [3000.0 + i for i in range(5)]})
    assert lab["label"] is None and lab["pid_row"]["interval"]["lower"] > 0 and lab["positive"] == []
    lab2 = cf.descriptive_labels(d, rec, dom(*G), x.best, x.ch8, x.heldout, "H1", pid={"status": sel.COST_INFEASIBLE})
    assert lab2["label"] is None and lab2["negative"] == []


def test_descriptive_labels_on_d_conf_with_empty_band():
    # under (b) an empty band leaves C2 INCOMPLETE (no band); the labels are still read on D_conf
    d = design("C", CLAIM_B)
    x = S4(d.seeds)
    for g in G:
        x.set_gap(g, [3000 + 10 * i for i in range(8)] if g == 450_000 else [(-1) ** i * 200 for i in range(8)])
    noband = sel.Domain(sel.INCOMPLETE_NO_BAND, (), "D_band is empty")
    rec = cf.c2(d, CLAIM_B, dom(*G), noband, x.best, x.ch8, x.heldout, "H1")
    assert rec["verdict"] == sel.INCOMPLETE_NO_BAND
    lab = cf.descriptive_labels(d, rec, dom(*G), x.best, x.ch8, x.heldout, "H1")
    assert lab["label"] == cf.LOCAL_ADVANTAGE and lab["positive"] == ["450000"]
    missing = cf.c2(d, CLAIM_A, dom(*G), dom(*G), x.best, x.ch8, x.heldout, "H1", gate_failures=["a failed gate"])
    assert cf.descriptive_labels(d, missing, dom(*G), x.best, x.ch8, x.heldout, "H1")["label"] is None


def test_descriptive_labels_cost_infeasible_is_a_deficit_and_accounting():
    d = design("C")
    x = S4(d.seeds, best_lowest={6: 300_000})  # BEST infeasible at 100,000 and 200,000 in seed 6
    for g in G:
        x.set_gap(g, [3000 + 10 * i for i in range(8)], count="H1")
        x.set_gap(g, [(-1) ** i * 200 for i in range(8)], count="H2")
    rec = cf.c2(d, CLAIM_A, dom(*G), dom(*G), x.best, x.ch8, x.heldout, "H1")
    lab = cf.descriptive_labels(d, rec, dom(*G), x.best, x.ch8, x.heldout, "H1")
    assert lab["negative"] == ["100000", "200000"] and lab["label"] == cf.CROSSING
    assert lab["non_headline"]["label"] == cf.LOCAL_DEFICIT and lab["accounting_dependent"] is True
