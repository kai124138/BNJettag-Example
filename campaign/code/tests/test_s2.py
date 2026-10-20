"""Unit tests for cbl/s2.py (gates G53 and G55; STUDY v0.8 section 8.6).

Run with the Mac CPU interpreter, from anywhere:

    python -I -B -m pytest -q -p no:cacheprovider tests/test_s2.py

Constructed cases only (tests/fixtures/m16_synth.py). Every number is an engineering check, not a
quotable result.
"""

import math
import random
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

FORK = Path(__file__).resolve().parents[1]
for p in (str(FORK), str(FORK / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cbl import s2  # noqa: E402
from cbl import select as sel  # noqa: E402
from cbl import stats as st  # noqa: E402
from fixtures import m16_synth as syn  # noqa: E402

G = sel.GRID
N = syn.VAL_N
CLAIM_B = {"base": "b", "band_rule": "default", "c3r": False}


def jitter(spec, seed=1, k=3):
    rnd = random.Random(seed)
    for lab in spec:
        for s in spec[lab]:
            m = None
            for g in G:
                v = spec[lab][s][g]
                if v is None:
                    continue
                v = v + rnd.randint(0, k)
                m = v if m is None else max(m, v)
                spec[lab][s][g] = m
    return spec


def make_screen(spec, headline="H1"):
    return sel.Screen([sel.run_from_records(c, s) for c, s in syn.build_runs(spec)], headline)


def test_s2_record_has_no_verdict_and_follows_rules():
    spec = jitter(syn.screen_spec(gains={"F011-FB": {g: (12, 9, 15) for g in G}}))
    scr = make_screen(spec)
    rec = s2.s2_record(scr, sel.claim_domain_config(CLAIM_B), False, nsim=2_000)
    assert s2.verdict_tokens(rec) == []
    assert rec["status"] == "ok" and rec["best"]["best"] == "F011" and rec["form_choice"]["form"] in ("FB", "BF")
    assert rec["label"] == sel.LABEL_OPTIMISTIC
    for g, row in rec["per_g"].items():
        for name, n, level in sel.S2_OPTIONS:
            o = row["options"][name]
            assert o["threshold_in_s_d"] == st.threshold(n, level)
            assert o["gain_for_80pct"]["0.9"]["at_upper_s_d"] == pytest.approx(
                o["gain_for_80pct"]["0.9"]["at_s_d"] * st.screen_sd_upper_factor())
    assert set(rec["plug_in"]) == {"D_screen", "screen_band"}
    nd = len(rec["d_screen"]["allowances"])
    assert all(1 <= v <= nd for v in rec["plug_in"]["D_screen"]["distinct_pairs_per_seed"].values())
    assert rec["validation_vs_heldout_se_ratio"] == pytest.approx(math.sqrt(260_000 / 62_000))
    assert round(rec["validation_vs_heldout_se_ratio"], 2) == 2.05


def test_s2_incomplete_screen():
    spec = syn.screen_spec()
    for g in G:
        spec["CH-8"][1][g] = None
    rec = s2.s2_record(make_screen(spec), sel.claim_domain_config(CLAIM_B), False, nsim=100)
    assert rec["status"] == sel.INCOMPLETE and "form_choice" not in rec and "best" not in rec


def test_s2_requires_decisions():
    scr = make_screen(syn.screen_spec())
    with pytest.raises(sel.UnsetDecision):
        s2.s2_record(scr, None, False, nsim=100)
    with pytest.raises(sel.UnsetDecision):
        s2.s2_record(scr, sel.claim_domain_config(CLAIM_B), None, nsim=100)


def test_pooled_variance_rows_and_df():
    spec = jitter(syn.screen_spec(), seed=2)
    scr = make_screen(spec)
    a = s2.pooled_variance(scr, "FB", 300_000, False)
    b = s2.pooled_variance(scr, "FB", 300_000, True)
    assert a["df"] == 16 and len(a["rows"]) == 9 and a["f000_row"]
    assert b["df"] == 14 and "F000" not in b["rows"]
    lo, hi = a["band_sigma_d_over_hat"]
    assert round(lo, 3) == 0.745 and round(hi, 3) == 1.522
    assert a["sigma_d"] == pytest.approx(math.sqrt(2) * a["sigma_pool"])
    y = np.array([[float(scr.point(l, s, 300_000).top1) for s in (1, 2, 3)] for l in a["rows"]])
    res = y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + y.mean()
    assert a["sigma_pool"] == pytest.approx(math.sqrt((res ** 2).sum() / 16), rel=1e-12)
    with pytest.raises(sel.UnsetDecision):
        s2.pooled_variance(scr, "FB", 300_000, None)


def test_pooled_identical_arms_zero_residual():
    scr = make_screen(syn.screen_spec())  # every arm equals CH-8 at every (seed, g)
    p = s2.pooled_variance(scr, "FB", 300_000, False)
    assert p["sigma_pool"] == 0.0 and p["sigma_d_upper"] == 0.0
    assert all(v == 0.0 for v in p["own_paired_sd"].values())
    assert p["ms_seed_over_ms_residual"] is None  # residual 0: the ratio is undefined


def test_pooled_drops_infeasible_arm():
    spec = syn.screen_spec(lowest={"F111-FB": 400_000})
    p = s2.pooled_variance(make_screen(jitter(spec)), "FB", 300_000, False)
    assert "F111-FB" not in p["rows"] and p["df"] == 14


def _one_candidate_panel(domain=(300_000, 400_000, 450_000)):
    labels = (sel.CH8, "F100")
    feas = {(l, s, g): True for l in labels for s in (1, 2, 3) for g in domain}
    cost = {k: k[2] - 1000 for k in feas}
    return sel.Panel(labels, (1, 2, 3), domain, feas, cost)


def test_null_reference_one_candidate_median_zero():
    pan = _one_candidate_panel()
    sig = {g: 0.004 for g in pan.domain}
    for rho in (0.0, 0.9):
        r = s2.null_selection_reference(pan, sig, rho, nsim=40_000)
        se = math.sqrt(math.pi / 2) * r["sd"] / math.sqrt(r["nsim"])
        assert abs(r["median"]) < 4 * se, (rho, r)
        # analytic sd of the mean gain over S seeds and D allowances: sigma sqrt(2 (1 + (D-1) rho) / (S D))
        D, S = len(pan.domain), 3
        want = 0.004 * math.sqrt(2 * (1 + (D - 1) * rho) / (S * D))
        assert r["sd"] == pytest.approx(want, rel=0.02)
        assert r["p95"] == pytest.approx(1.6448536 * want, rel=0.03)


def test_null_reference_selection_is_positive_with_many_candidates():
    spec = jitter(syn.screen_spec(), seed=3)
    scr = make_screen(spec)
    ds = sel.d_screen(scr)
    pan = sel.Panel.from_screen(scr, ds, labels=[l for l in sel.screen_labels() if l != sel.F000])
    r = s2.null_selection_reference(pan, {g: 0.004 for g in ds.allowances}, 0.0, nsim=4_000)
    assert r["median"] > 0  # the maximum over seven candidates in a chosen form is biased upward


def test_null_reference_zero_sigma_is_zero():
    pan = _one_candidate_panel()
    r = s2.null_selection_reference(pan, {g: 0.0 for g in pan.domain}, 0.9, nsim=1_000)
    assert r["median"] == 0.0 and r["p95"] == 0.0


def test_null_reference_layout_is_deterministic():
    pan = _one_candidate_panel()
    a = s2.null_selection_reference(pan, {g: 0.004 for g in pan.domain}, 0.9, nsim=3_000, chunk=1_000)
    b = s2.null_selection_reference(pan, {g: 0.004 for g in pan.domain}, 0.9, nsim=3_000, chunk=1_000)
    assert a == b


def test_plugin_na_when_best_sd_zero():
    # BEST's gain identical in every seed: s_d = 0, the plug-in power is NA with its reason
    spec = syn.screen_spec(gains={"F011-FB": {g: (10, 10, 10) for g in G}})
    rec = s2.s2_record(make_screen(spec), sel.claim_domain_config(CLAIM_B), False, nsim=500)
    blk = rec["plug_in"]["D_screen"]["at_s_d"]["A"]["0.0"]
    assert blk["status"] == "NA" and "sd is 0" in blk["reason"]


def test_plugin_zero_when_best_infeasible():
    # every other candidate is feasible only at 450,000, so F011-FB (missing 100,000 and 200,000 in
    # seed 2) is BEST by coverage; the form goes to FB on Delta_form at the common 450,000 cells
    others = ["F100"] + ["%s-%s" % (c, f) for c in syn.FUSED for f in ("FB", "BF") if c != "F011" or f == "BF"]
    spec = jitter(syn.screen_spec(gains={"F011-FB": {g: (40, 25, 55) for g in G}}, lowest={l: 450_000 for l in others}))
    spec["F011-FB"][2][100_000] = None
    spec["F011-FB"][2][200_000] = None
    rec = s2.s2_record(make_screen(spec), sel.claim_domain_config({"base": "a", "c3r": False}), False, nsim=500)
    assert rec["form_choice"]["form"] == "FB" and rec["best"]["best"] == "F011"
    blk = rec["plug_in"]["D_screen"]["at_s_d"]["C alone"]["0.9"]
    assert blk["status"] == "zero" and blk["joint"] == 0.0 and blk["over_feasible_allowances_only"]["status"] == "ok"
    assert rec["g_nonpositive_or_infeasible"][:2] == [100_000, 200_000]
    assert rec["g_star"] is None  # option (a): no band


def _heldout_for(scr, shift=None):
    """held-out counts at every frozen point: 260,000 jets, top-1 = validation top-1 + shift"""
    out = {}
    for (lab, s), r in scr.runs.items():
        for c in sel.COUNTS:
            for g in G:
                p = r.point(c, g)
                if p is not None:
                    d = (shift or {}).get(lab, 0)
                    out[p.ckpt_sha256] = {"correct": int(round(float(p.top1) * 260_000)) + d, "n": 260_000}
    return out


def test_heldout_identical_arms_zero_gaps():
    scr = make_screen(syn.screen_spec())
    t = s2.screen_heldout_tables(scr, "FB", "F011", _heldout_for(scr), True)
    assert s2.verdict_tokens(t) == []
    assert t["label"] == sel.LABEL_SCREEN_HELDOUT
    for lab, rows in t["candidates"].items():
        for g, row in rows.items():
            assert row["per_seed"] == [0.0, 0.0, 0.0]
            iv = row["interval"]
            assert iv["mean"] == 0.0 and iv["lower"] == 0.0 and iv["upper"] == 0.0
    assert "bitwise identical" in t["f000_minus_ch8_reading"]
    assert all(v is False for rows in t["dominance"].values() for v in rows.values())
    assert all(v == 0 for v in t["factorial"]["effects"].values())


def test_heldout_cost_infeasible_and_reach():
    spec = syn.screen_spec(lowest={"F100": 300_000, "CH-8": 200_000})
    scr = make_screen(spec)
    t = s2.screen_heldout_tables(scr, "FB", "F011", _heldout_for(scr), True)
    assert t["candidates"]["F100"]["200000"]["status"] == sel.COST_INFEASIBLE
    assert t["cost_reach"]["F011-FB"]["allowances"] == [100_000]
    assert "run-to-run divergence" not in t["f000_minus_ch8_reading"]


def test_heldout_refuses_unfrozen_checkpoint():
    scr = make_screen(syn.screen_spec())
    h = _heldout_for(scr)
    manifest = {"grid_checkpoints": sel.grid_checkpoint_table(scr)}
    s2.screen_heldout_tables(scr, "FB", "F011", h, False, manifest=manifest)
    for g in G:  # F100 seed 1 has one checkpoint at every g (flat envelope): unfreeze it everywhere
        manifest["grid_checkpoints"]["H1"]["F100"]["1"][str(g)] = "NA"
    with pytest.raises(sel.FreezeRefused):
        s2.screen_heldout_tables(scr, "FB", "F011", h, False, manifest=manifest)


def test_heldout_bootstrap_and_rejection_with_jets():
    spec = syn.screen_spec()
    scr = make_screen(spec)
    rng = np.random.default_rng(9)
    n = 400
    labels = rng.integers(0, 5, size=n)
    jets = {"labels": labels}
    for (lab, s), r in scr.runs.items():
        for g in G:
            p = r.point("H1", g)
            if p is not None and p.ckpt_sha256 not in jets:
                sc = rng.random((n, 5))
                jets[p.ckpt_sha256] = {"correct": sc.argmax(1) == labels, "scores": sc}
    # make one arm identical to CH-8 jet by jet: its bootstrap and rejection gaps are 0
    for s in (1, 2, 3):
        for g in G:
            jets[scr.point("F100", s, g).ckpt_sha256] = jets[scr.point(sel.CH8, s, g).ckpt_sha256]
    t = s2.screen_heldout_tables(scr, "FB", "F011", _heldout_for(scr), False, jets=jets)
    row = t["candidates"]["F100"]["300000"]
    assert row["bootstrap"]["seed_mean"] == {"observed": 0.0, "lower": 0.0, "upper": 0.0}
    for k in range(5):
        assert row["background_rejection_eps_s_0.5"][str(k)]["paired"]["mean"] == 0.0


def test_pid_screen_comparison():
    spec = {"CH-8": syn.screen_spec()["CH-8"], "F011-FB": syn.screen_spec(gains={"F011-FB": {g: (5, 6, 7) for g in G}})["F011-FB"]}
    runs = [sel.run_from_records(c, s) for c, s in syn.build_runs(spec, stage="S3", schedule="pid",
                                                                  arm_ids={"CH-8": "PID-CH", "F011-FB": "PID-BEST"})]
    ps = sel.Screen(runs, "H1", schedule="pid")
    out = s2.pid_screen_comparison(ps, "F011-FB")
    assert out["per_seed"] == pytest.approx([5 / N, 6 / N, 7 / N]) and out["interval"]["positive"] == 3
    assert s2.verdict_tokens(out) == []


def test_verdict_tokens_detects():
    assert s2.verdict_tokens({"a": "ABOVE ON GRID"}) == ["ABOVE ON GRID"]
    assert s2.verdict_tokens({"LOCAL DEFICIT": 1}) == ["LOCAL DEFICIT"]
    assert s2.verdict_tokens(["COST INFEASIBLE", "NO ACCEPTED CHECKPOINT"]) == []


def test_control_rows():
    spec = syn.screen_spec(gains={"F111-FB": {g: (4, 4, 4) for g in G}})
    scr = make_screen(spec)
    ctl = {}
    for name, lab, env_gain in (("DS-HGQ", "CH-8", -30), ("GR", "F111-FB", 2), ("BE", "F111-FB", -2)):
        ctl[name] = {}
        for s in (1, 2, 3):
            env = {g: (v + env_gain if v is not None else None) for g, v in spec[lab][s].items()}
            cfg, sm = syn.run_records(lab, s, env, stage="S3", arm_id=name)
            cfg["control"] = name
            ctl[name][s] = sel.run_from_records(cfg, sm)
    for name in ("FP32", "W8A8"):
        ctl[name] = {}
        for s in (1, 2, 3):
            cfg, sm = syn.run_records("CH-8", s, syn.flat_env(40_000), stage="S3", arm_id=name)
            ep = syn.epochs_trace([(0, 0.60, 900_000), (1, 0.67, 900_000), (2, 0.66, 900_000)])
            for e in ep:
                e["ckpt_sha256"] = syn.sha(name, s, e["epoch"])
            ctl[name][s] = sel.run_from_records(cfg, sm, epochs=ep)
    h = _heldout_for(scr)
    for name in ("DS-HGQ", "GR", "BE"):
        for s in (1, 2, 3):
            for c in sel.COUNTS:
                for g in G:
                    p = ctl[name][s].point(c, g)
                    if p is not None:
                        h[p.ckpt_sha256] = {"correct": int(round(float(p.top1) * 260_000)), "n": 260_000}
    for name in ("FP32", "W8A8"):
        for s in (1, 2, 3):
            h[syn.sha(name, s, 1)] = {"correct": 172_000, "n": 260_000}
    out = s2.control_heldout_rows(scr, "FB", ctl, h)
    assert s2.verdict_tokens(out) == [] and out["status"] == "ok"
    assert out["controls"]["DS-HGQ"]["300000"]["reference"] == "CH-8"
    assert out["controls"]["GR"]["300000"]["reference"] == "F111-FB"
    assert out["controls"]["GR"]["300000"]["interval"]["mean"] > 0 > out["controls"]["BE"]["300000"]["interval"]["mean"]
    assert out["controls"]["W8A8"]["per_seed"]["1"]["epoch"] == 1 and out["controls"]["W8A8"]["per_seed"]["2"]["H1"] == 900_000
    assert out["controls"]["FP32"]["mean"] == pytest.approx(172_000 / 260_000)


def test_pid_screen_comparison_refuses_disagreeing_summary():
    spec = {"CH-8": syn.screen_spec()["CH-8"], "F011-FB": syn.screen_spec()["F011-FB"]}
    runs = []
    for c, s in syn.build_runs(spec, stage="S3", schedule="pid", arm_ids={"CH-8": "PID-CH", "F011-FB": "PID-BEST"}):
        if c["arm_id"] == "PID-BEST" and c["seed"] == 2:
            s["pid_selection"] = dict(s["pid_selection"], ckpt_sha256="e" * 64)
        runs.append(sel.run_from_records(c, s))
    with pytest.raises(sel.SelectError, match="pid_selection"):
        s2.pid_screen_comparison(sel.Screen(runs, "H1", schedule="pid"), "F011-FB")
