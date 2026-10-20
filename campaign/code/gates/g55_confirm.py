"""Gate G55 (PREFLIGHT-PLAN section 3 row G55, section 4 row B3b; STUDY v0.9 sections 8.6, 8.7, 13
selection and evaluation row).

Identical arms give zero differences and an interval around 0; the D_C2 test, the nested
whole-D_conf form, the 7- and 6-contrast labels, C3, C3r, C4 and C6 on constructed cases; the
plug-in power at the observed s_d and at 6.28 s_d; and, as the v0.8 gate, "with one candidate in
one form, the simulated null-selection median is 0". Checks, on the Mac CPU with no backend:
  - repository: HEAD equals --expect-sha12 and `git status --porcelain` is empty;
  - the STUDY v0.9 snapshot hashes to its recorded sha256 (61e68527...89df4);
  - identical arms: C2 contrasts, the screen held-out gaps, the factorial effects and their
    intervals, the paired bootstrap and the pooled residual are exactly 0, every interval [0, 0];
  - every confirmation label on a constructed case whose answer is known by construction, the
    descriptive labels read at the grid points of D_conf as STUDY v0.9 :1039-1042 defines them;
  - the levels of every S4 option and every claim-domain option (item 11, item 12);
  - the plug-in power at the observed s_d and at the upper factor, against the closed form at rho = 0
    and the simulation at rho = 0.9 directly (STUDY v0.9 :749);
  - the null selection reference with one candidate in one form, at the section 6 settings
    (default_rng(12345), 200,000 simulations), rho = 0 and 0.9: the median lies within four
    Monte Carlo standard errors of 0 (tolerance fixed here before the run; owner decision B3b-19),
    and its sd and 95th percentile match the closed form of a mean of normal differences;
  - the unit tests `tests/test_confirm.py`, as a sub-step.

Run as

    python -I -B gates/g55_confirm.py --expect-sha12 SHA12 [--study PATH]

Prints one line per check and ends with "G55 PASS" (exit 0) or "G55 FAIL" (exit 1). Writes
nothing. Every number printed is an engineering reading, not a quotable result.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I does not add the script directory
import m16gate as mg  # noqa: E402

sys.path.insert(0, str(mg.FORK))
sys.path.insert(0, str(mg.FORK / "tests"))
import numpy as np  # noqa: E402

from cbl import confirm as cf  # noqa: E402
from cbl import s2  # noqa: E402
from cbl import select as sel  # noqa: E402
from cbl import stats as st  # noqa: E402
from fixtures import m16_synth as syn  # noqa: E402

G = sel.GRID
NT = 260_000


def rising(lowest=100_000, base=40_000):
    return {g: (base + g // 1000 if g >= lowest else None) for g in G}


def sidecars(run):
    for c in sel.COUNTS:
        for p in run.front[c]:
            run.sidecars.setdefault(p.ckpt_sha256, {"epoch": p.epoch, "structure_record": {"degenerate": {}}})
    return run


class S4:
    def __init__(self, seeds, best_lowest=None):
        self.seeds, self.best, self.ch8, self.h = tuple(seeds), {}, {}, {}
        for s in self.seeds:
            self.best[s] = sidecars(sel.run_from_records(*syn.run_records(
                "F011-FB", s, rising((best_lowest or {}).get(s, 100_000)), stage="S4", arm_id="BEST-C")))
            self.ch8[s] = sidecars(sel.run_from_records(*syn.run_records("CH-8", s, rising(base=40_010), stage="S4", arm_id="CH-8-C")))
            for r in (self.best[s], self.ch8[s]):
                for c in sel.COUNTS:
                    for p in r.front[c]:
                        self.h.setdefault(p.ckpt_sha256, {"correct": 170_000, "n": NT})

    def gap(self, g, gaps, counts=("H1", "H2")):
        for c in counts:
            for s, d in zip(self.seeds, gaps):
                p, q = self.best[s].point(c, g), self.ch8[s].point(c, g)
                if p is not None and q is not None:
                    self.h[p.ckpt_sha256] = {"correct": self.h[q.ckpt_sha256]["correct"] + d, "n": NT}


def dom(*gs):
    return sel.Domain("ok", tuple(gs))


def main():
    args = mg.parse_args(__doc__)
    gate = mg.start("G55", args)
    text = mg.study_text(gate, args.study)
    ck = gate.check
    CA = sel.claim_domain_config({"base": "a", "c3r": False})
    CB = sel.claim_domain_config({"base": "b", "band_rule": "default", "c3r": False})

    # ---- identical arms
    d = sel.s4_design("A", CA)
    x = S4(d.seeds)
    rec = cf.c2(d, CA, dom(*G), dom(*G), x.best, x.ch8, x.h, "H1")
    zero = all(r["per_seed"] == [0.0] * 5 and r["interval"]["lower"] == 0.0 == r["interval"]["upper"] for r in rec["per_g"].values())
    ck(zero and rec["verdict"] == cf.FLAT, "identical arms: every C2 contrast is 0 and every interval is [0, 0], which covers 0; FLAT/UNRESOLVED")
    iv = st.paired_interval([0.0, 0.0, 0.0])
    ck(iv.lower == 0.0 == iv.upper and iv.covers(0.0), "identical arms: the ordinary 95 % paired interval at n = 3 is [0, 0]")
    scr = sel.Screen([sel.run_from_records(c, s) for c, s in syn.build_runs(syn.screen_spec())], "H1")
    ho = {p.ckpt_sha256: {"correct": int(round(float(p.top1) * NT)), "n": NT}
          for r in scr.runs.values() for c in sel.COUNTS for p in r.front[c]}
    t = s2.screen_heldout_tables(scr, "FB", "F011", ho, True)
    ck(all(row["interval"]["mean"] == 0.0 and row["interval"]["lower"] == 0.0 == row["interval"]["upper"]
           for rows in t["candidates"].values() for row in rows.values()),
       "identical arms: every screen held-out gap is 0 with interval [0, 0]")
    ck(all(v == 0 for v in t["factorial"]["effects"].values())
       and all(e["block"]["lower"] == 0.0 == e["block"]["upper"] and e["per_seed"]["lower"] == 0.0 == e["per_seed"]["upper"]
               for row in t["factorial"]["per_g"].values() if row for e in row["effects"].values()),
       "identical arms: zero factorial effects with block and per-seed intervals [0, 0]")
    p = s2.pooled_variance(scr, "FB", 300_000, False)
    ck(p["sigma_pool"] == 0.0 and p["two_way"]["ss_residual"] == 0.0, "identical arms: pooled residual 0 (sigma-hat_pool = 0, df %d)" % p["df"])
    rng = np.random.default_rng(3)
    a = rng.random((3, 1000)) < 0.66
    bt = st.paired_bootstrap(a, a.copy(), replicates=500)
    ck(bt["seed_mean"] == {"observed": 0.0, "lower": 0.0, "upper": 0.0}, "identical arms: the paired bootstrap over jets is exactly 0")

    # ---- C2 labels, the strong form and naming
    dC = sel.s4_design("C", CA)
    x = S4(dC.seeds)
    for g in G:
        x.gap(g, [900, 1100, 1000, 950, 1050, 1000, 980, 1020])
    r = cf.c2(dC, CA, dom(*G), dom(300_000, 400_000), x.best, x.ch8, x.h, "H1")
    ck(r["verdict"] == cf.ABOVE_ON_GRID and r["naming"] == "front result" and r["size"] == 2,
       "C2 ABOVE ON GRID at level 95 % (option C alone), |D_C2| = 2 named a front result")
    r1 = cf.c2(dC, CA, dom(*G), dom(450_000), x.best, x.ch8, x.h, "H1")
    ck(r1["naming"] == "single-allowance comparison", "C2 with |D_C2| = 1 is named a single-allowance comparison")
    y = S4(d.seeds, best_lowest={6: 400_000})
    for g in (400_000, 450_000):
        y.gap(g, [-900, -1100, -1000, -950, -1050])
    rb = cf.c2(d, CA, dom(300_000, 400_000, 450_000), dom(300_000, 400_000, 450_000), y.best, y.ch8, y.h, "H1")
    ck(rb["verdict"] == cf.BELOW_ON_GRID and rb["per_g"]["300000"]["status"] == sel.COST_INFEASIBLE,
       "C2 BELOW ON GRID: upper bounds < 0, and COST INFEASIBLE counts as a deficit")
    z = S4(d.seeds, best_lowest={5: 400_000})
    for g in G:
        z.gap(g, [900, 1100, 1000, 950, 1050])
    rz = cf.c2(d, CA, dom(300_000, 400_000), dom(300_000, 400_000), z.best, z.ch8, z.h, "H1")
    ck(rz["verdict"] == cf.FLAT, "C2: one allowance COST INFEASIBLE blocks ABOVE ON GRID (FLAT/UNRESOLVED)")
    inc = [cf.c2(d, CA, dom(*G), dom(*G), z.best, z.ch8, z.h, "H1", gate_failures=["a failed gate"])["verdict"],
           cf.c2(d, CA, sel.Domain(sel.INCOMPLETE, (), "empty"), sel.Domain(sel.INCOMPLETE, (), "empty"), z.best, z.ch8, z.h, "H1")["verdict"],
           cf.c2(d, CB, dom(*G), sel.Domain(sel.INCOMPLETE_NO_BAND, (), "empty"), z.best, z.ch8, z.h, "H1")["verdict"]]
    ck(inc == [sel.INCOMPLETE, sel.INCOMPLETE, sel.INCOMPLETE_NO_BAND], "C2 INCOMPLETE: failed gate, empty D_conf; INCOMPLETE (no band)")
    dB = sel.s4_design("B", CB)
    w = S4(dB.seeds)
    for g in G:
        w.gap(g, [800, 1200, 1000, 900, 1100, 1000, 950, 1050] if g >= 300_000 else [-50, 60, -40, 30, 10, -20, 0, 5])
    rs = cf.c2(dB, CB, dom(*G), dom(300_000, 350_000, 400_000, 450_000), w.best, w.ch8, w.h, "H1")
    ck(rs["verdict"] == cf.ABOVE_ON_GRID and rs["strong_form"] == "not ABOVE ON ALL OF D_conf"
       and rs["per_g"]["100000"]["note"] == "outside the claim",
       "nested whole-D_conf form under (b): band ABOVE ON GRID, strong form not; allowances below g* printed outside the claim")
    for g in (100_000, 200_000):
        w.gap(g, [800, 1200, 1000, 900, 1100, 1000, 950, 1050])
    rs2 = cf.c2(dB, CB, dom(*G), dom(300_000, 350_000, 400_000, 450_000), w.best, w.ch8, w.h, "H1")
    ck(rs2["strong_form"] == "ABOVE ON ALL OF D_conf" and rs2["verdict"] == cf.ABOVE_ON_GRID,
       "strong form ABOVE ON ALL OF D_conf implies the band's ABOVE ON GRID")

    # ---- 7- and 6-contrast descriptive labels
    for opt, m, want_thr in (("A", 7, "2.266"), ("B", 7, "1.327"), ("C", 6, "1.285")):
        dd = sel.s4_design(opt, CA)
        n = dd.n
        pid = {"status": "ok", "per_seed": [0.0] * n} if dd.pid else None
        labs = []
        for hi, lo in (([3000 + 10 * i for i in range(n)], [-3000 - 10 * i for i in range(n)]),
                       ([3000 + 10 * i for i in range(n)], [(-1) ** i * 200 for i in range(n)]),
                       ([-3000 - 10 * i for i in range(n)], [(-1) ** i * 200 for i in range(n)])):
            q = S4(dd.seeds)
            for g in G:
                q.gap(g, hi if g >= 400_000 else lo)
            rc = cf.c2(dd, CA, dom(*G), dom(*G), q.best, q.ch8, q.h, "H1")
            lb = cf.descriptive_labels(dd, rc, dom(*G), q.best, q.ch8, q.h, "H1", pid=pid)
            labs.append(lb["label"])
        ck(labs == [cf.CROSSING, cf.LOCAL_ADVANTAGE, cf.LOCAL_DEFICIT] and lb["contrasts"] == m
           and ("%.3f" % lb["threshold_in_s_d"]) == want_thr,
           "option %s: %d-contrast labels CROSSING, LOCAL ADVANTAGE, LOCAL DEFICIT at %s s_d" % (opt, m, want_thr))

    # ---- the v0.9 label definitions (:1039-1042) and the grid-points-only reading
    defs = ["read on those simultaneous intervals at the grid points g ∈ D_conf, where a COST INFEASIBLE g counts as a deficit",
            "`CROSSING`: at least one significantly positive and one significantly negative g.",
            "`LOCAL ADVANTAGE`: some lower bounds > 0 and no upper bound < 0, but C2's `ABOVE ON GRID` fails.",
            "`LOCAL DEFICIT`: the mirror case of LOCAL ADVANTAGE: some upper bounds < 0 and no lower bound > 0, "
            "but C2's `BELOW ON GRID` fails."]
    ck(all(s in text for s in defs), "STUDY v0.9 :1039-1042 defines the three labels as cbl.confirm.descriptive_labels reads them")
    q = S4(d.seeds)
    for g in G:
        q.gap(g, [(-1) ** i * 200 for i in range(5)])
    rc = cf.c2(d, CA, dom(*G), dom(*G), q.best, q.ch8, q.h, "H1")
    l1 = cf.descriptive_labels(d, rc, dom(*G), q.best, q.ch8, q.h, "H1", pid={"status": "ok", "per_seed": [3000.0 + i for i in range(5)]})
    l2 = cf.descriptive_labels(d, rc, dom(*G), q.best, q.ch8, q.h, "H1", pid={"status": sel.COST_INFEASIBLE})
    ck(l1["label"] is None and l1["pid_row"]["interval"]["lower"] > 0 and l2["label"] is None,
       "labels are read at the grid points only: a significant or COST INFEASIBLE PID contrast labels nothing")
    for g in G:
        q.gap(g, [3000 + 10 * i for i in range(5)] if g == 450_000 else [(-1) ** i * 200 for i in range(5)])
    rnb = cf.c2(d, CB, dom(*G), sel.Domain(sel.INCOMPLETE_NO_BAND, (), "empty"), q.best, q.ch8, q.h, "H1")
    l3 = cf.descriptive_labels(d, rnb, dom(*G), q.best, q.ch8, q.h, "H1", pid={"status": "ok", "per_seed": [0.0] * 5})
    ck(rnb["verdict"] == sel.INCOMPLETE_NO_BAND and l3["label"] == cf.LOCAL_ADVANTAGE,
       "with an empty band the labels are still read on D_conf (LOCAL ADVANTAGE: C2's ABOVE ON GRID fails)")

    # ---- C3 and C3r
    def pid_runs(seeds, gaps, best_env=None, ch_env=None):
        pb, pc, h = {}, {}, {}
        for i, s in enumerate(seeds):
            pb[s] = sidecars(sel.run_from_records(*syn.run_records("F011-FB", s, best_env or rising(300_000), stage="S4",
                                                                   schedule="pid", arm_id="PID-BEST-C")))
            pc[s] = sidecars(sel.run_from_records(*syn.run_records("CH-8", s, ch_env or rising(300_000, base=40_010), stage="S4",
                                                                   schedule="pid", arm_id="PID-CH-C")))
            for rr in (pb[s], pc[s]):
                for pt in rr.front["H1"]:
                    h.setdefault(pt.ckpt_sha256, {"correct": 170_000, "n": NT})
            a_, b_ = sel.pid_selection(pb[s], "H1")["point"], sel.pid_selection(pc[s], "H1")["point"]
            if a_ and b_:
                h[a_["ckpt_sha256"]] = {"correct": 170_000 + gaps[i], "n": NT}
        return pb, pc, h
    v = [cf.c3(d, *pid_runs(d.seeds, [900, 1100, 1000, 950, 1050]), "H1")["verdict"],
         cf.c3(d, *pid_runs(d.seeds, [-900, -1100, -1000, -950, -1050]), "H1")["verdict"],
         cf.c3(d, *pid_runs(d.seeds, [900, -1100, 1000, -950, 50]), "H1")["verdict"],
         cf.c3(d, *pid_runs(d.seeds, [1] * 5, best_env=rising(400_000)), "H1")["verdict"],
         cf.c3(d, *pid_runs(d.seeds, [1] * 5, ch_env=rising(400_000, base=40_010)), "H1")["verdict"]]
    ck(v == [cf.ABOVE, cf.BELOW, cf.FLAT, sel.COST_INFEASIBLE, sel.INCOMPLETE],
       "C3 at the PID selections: ABOVE, BELOW, FLAT/UNRESOLVED, COST INFEASIBLE, INCOMPLETE")
    pb, pc, h = pid_runs(d.seeds, [1] * 5, best_env={g: None for g in G})
    for s in d.seeds:
        pb[s].filter_outcome["H1"][350_000] = {"outcome": sel.NO_ACCEPTED}
    ck(cf.c3(d, pb, pc, h, "H1")["verdict"] == cf.NO_ACCEPTED_LABEL and cf.c3(sel.s4_design("C", CA), pb, pc, h, "H1")["tested"] is False,
       "C3: NO ACCEPTED CHECKPOINT / ACCURACY FILTER FAILURE; not tested under option C")
    cbon = sel.claim_domain_config({"base": "a", "c3r": True, "c3r_level": "bonferroni"})
    cnest = sel.claim_domain_config({"base": "a", "c3r": True, "c3r_level": "nested"})
    dcb = sel.s4_design("C", cbon)
    q = S4(dcb.seeds)
    q.gap(350_000, [900, 1100, 1000, 950, 1050, 1000, 980, 1020])
    r3 = cf.c3r(dcb, q.best, q.ch8, q.h, "H1")
    q2 = S4(dcb.seeds, best_lowest={9: 400_000})
    ck(r3["verdict"] == cf.ABOVE and r3["level"] == 0.975 and cf.c3r(dcb, q2.best, q2.ch8, q2.h, "H1")["verdict"] == sel.COST_INFEASIBLE
       and cf.c3r(sel.s4_design("C", cnest, contains_350k=True), q.best, q.ch8, q.h, "H1")["level"] == 0.95,
       "C3r at 350,000: ABOVE at 97.5 % (Bonferroni), COST INFEASIBLE, 95 % under the nested variant when 350,000 is in D_C2")

    # ---- levels of every option
    lv = {(o, k): (sel.s4_design(o, cfg_, c35).c2_level, sel.s4_design(o, cfg_, c35).n)
          for o in ("A", "B", "C") for k, cfg_, c35 in (("a", CA, None), ("c-bonf", cbon, None), ("c-nest-in", cnest, True),
                                                       ("c-nest-out", cnest, False))}
    want = {("A", "a"): (0.975, 5), ("B", "a"): (0.975, 8), ("C", "a"): (0.95, 8), ("C", "c-bonf"): (0.975, 8),
            ("C", "c-nest-in"): (0.95, 8), ("C", "c-nest-out"): (0.975, 8), ("A", "c-bonf"): (0.975, 5), ("B", "c-nest-in"): (0.975, 8)}
    ck(all(lv[k] == v_ for k, v_ in want.items()), "levels: A and B 97.5 %; C alone 95 %; C with (c) 97.5 %; nested 95 % only with 350,000 in D_C2")
    ck(["%.3f" % st.threshold(n_, L_) for _, n_, L_ in sel.S2_OPTIONS] == ["1.563", "1.005", "0.836", "1.005"],
       "S2 thresholds per option: 1.563 (A), 1.005 (B), 0.836 (C alone), 1.005 (C with (c))")

    # ---- C4 and C6
    def pairs(n, lut, acc, **kw):
        out = []
        for i in range(n):
            pp = {"lut_best": 50_000 + lut + 37 * (-1) ** i, "lut_ch8": 50_000, "part": "xczu7ev", "stage": "synth+opt",
                  "settings": "s", "status": "ok", "allowance_exceeded_after_trace": False,
                  "post_trace_top1_best": 0.66 + acc + 1e-4 * (-1) ** i, "post_trace_top1_ch8": 0.66, "function_changed": False}
            pp.update(kw)
            out.append(pp)
        return out
    above = {"verdict": cf.ABOVE_ON_GRID, "d_c2": {"allowances": [400_000, 450_000]}}
    lab4 = []
    for dec450 in (pairs(5, -2000, 0.004), pairs(5, 2000, 0.004), pairs(5, 0, 0.004), pairs(5, -2000, 0.004, status="missing"),
                   pairs(5, -2000, -0.004)):
        lab4.append(cf.c4(d, above, {"400000": pairs(5, -2000, 0.004), "450000": dec450})["verdict"])
    ck(lab4 == ["SUPPORTED", "BLOCKED", "UNRESOLVED", cf.HW_INCOMPLETE, "BLOCKED"]
       and cf.c4(d, {"verdict": cf.FLAT}, {})["verdict"].startswith("NOT EVALUATED"),
       "C4: SUPPORTED, BLOCKED (not a LUT advantage), UNRESOLVED (equal LUT), HARDWARE INCOMPLETE, BLOCKED (accuracy); not evaluated without C2")

    def side(ds_=0.01, deg=None, qz=False, qs=None, ov=None, i=0):
        return {"attention_diagnostics": {"delta_attn_mean": 0.02 + 1e-3 * (-1) ** i, "delta_static": ds_ + 1e-3 * (-1) ** i,
                                          "delta_unif": 0.015 + 1e-3 * (-1) ** i, "H_attn": 1.6, "overflow": ov or {}},
                "structure_record": {"degenerate": deg or {}, "q_all_zero": qz, "q_identical_across_t": qs}}
    base = [side(i=i) for i in range(5)]
    c6l = [cf.c6_point(base)["label"], cf.c6_point([side(ds_=0.0, i=i) for i in range(5)])["label"],
           cf.c6_point(base[:4] + [side(deg={"K": "kernel_pruned"})])["label"],
           cf.c6_point(base[:4] + [side(deg={"Q": "other"})])["label"],
           cf.c6_point(base[:4] + [side(deg={"Q": "other"}, qs=[True, True])])["label"],
           cf.c6_point(base[:4] + [side(deg={"Q": "other"}, qz=True)])["label"],
           cf.c6_point([side(ov={"delta_unif": 2}, i=i) for i in range(5)])["label"]]
    ck(c6l == ["used", "not demonstrated", "not demonstrated", "used, with input-independent queries",
               "used, as query-independent attention pooling", "not demonstrated", "used (overflow-affected)"],
       "C6 labels: used; not demonstrated (static pattern; degenerate K; Q equal to 0); the two Q qualifiers; overflow-affected")

    # ---- plug-in power at the observed s_d and at the upper factor
    means, sds = [0.004, 0.006, 0.009], [0.003, 0.003, 0.004]
    up = st.screen_sd_upper_factor()
    for name, n, L in sel.S2_OPTIONS:
        for rho in (0.0, 0.9):
            p1 = st.plugin_power(means, sds, n, L, rho)
            p2 = st.plugin_power(means, [up * s for s in sds], n, L, rho)
            def direct(rs):
                if rho == 0.0:
                    return math.prod(st.single_power(x, n, L) for x in rs)
                return st.joint_sim(n).joint_power(rs, rho, L)
            direct1 = direct([m_ / s_ for m_, s_ in zip(means, sds)])
            direct2 = direct([m_ / (up * s_) for m_, s_ in zip(means, sds)])
            ok = (p1.joint == direct1 and p2.joint == direct2 and p2.joint <= p1.joint
                  and p1.joint <= p1.smallest_single + 4 * math.sqrt(0.25 / st.NSIM))
            ck(ok, "plug-in power %s, rho %s: %.4f at s_d, %.4f at %.4f s_d (smallest single %.4f, %.4f)"
               % (name, rho, p1.joint, p2.joint, up, p1.smallest_single, p2.smallest_single))
    ck("%.2f" % up == "6.28", "upper factor of sigma_d / s_d at df = 2: %.6f, printed 6.28" % up)
    ck(st.plugin_power([0.0, 0.0], [0.0, 0.0], 5, 0.975, 0.0).status == "NA", "plug-in power at s_d = 0 is NA with its reason, never inf or NaN")

    # ---- the v0.8 gate: one candidate in one form, the null-selection median is 0
    labels = (sel.CH8, "F100")
    dm = (300_000, 350_000, 400_000, 450_000)
    feas = {(l, s, g): True for l in labels for s in (1, 2, 3) for g in dm}
    pan = sel.Panel(labels, (1, 2, 3), dm, feas, {k: k[2] - 1000 for k in feas})
    sig = 0.004
    for rho in (0.0, 0.9):
        r0 = s2.null_selection_reference(pan, {g: sig for g in dm}, rho)
        se = math.sqrt(math.pi / 2) * r0["sd"] / math.sqrt(r0["nsim"])
        want_sd = sig * math.sqrt(2 * (1 + (len(dm) - 1) * rho) / (3 * len(dm)))
        ck(r0["nsim"] == st.NSIM and r0["seed"] == st.SEED and abs(r0["median"]) <= 4 * se,
           "one candidate (F100) in one form, rho = rho_f = %s: null-selection median %.3e within 4 MC se (%.3e) of 0" % (rho, r0["median"], se))
        ck(abs(r0["sd"] / want_sd - 1) < 0.01 and abs(r0["p95"] / (1.6448536 * want_sd) - 1) < 0.02,
           "one candidate, rho = %s: sd %.4e (closed form %.4e), 95th percentile %.4e (closed form %.4e)"
           % (rho, r0["sd"], want_sd, r0["p95"], 1.6448536 * want_sd))

    mg.run_pytest(gate, ["tests/test_confirm.py"])
    return mg.finish(gate)


if __name__ == "__main__":
    sys.exit(main())
