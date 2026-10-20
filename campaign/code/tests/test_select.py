"""Unit tests for cbl/select.py (gate G53; STUDY v0.8 sections 8.1-8.6, 8.8, 8.9, 19.2 items 11-12).

Run with the Mac CPU interpreter, from anywhere:

    python -I -B -m pytest -q -p no:cacheprovider tests/test_select.py

Every case is constructed (tests/fixtures/m16_synth.py); every number is an engineering check,
not a quotable result.
"""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[1]
for p in (str(FORK), str(FORK / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cbl import select as sel  # noqa: E402
from fixtures import m16_synth as syn  # noqa: E402

G = sel.GRID
N = syn.VAL_N


def make_screen(spec, headline="H1", schedule="ramp", stage="S1"):
    runs = [sel.run_from_records(c, s) for c, s in syn.build_runs(spec, stage=stage, schedule=schedule)]
    return sel.Screen(runs, headline, schedule=schedule)


def panel_of(screen, labels=None):
    dom = sel.d_screen(screen)
    pan = sel.Panel.from_screen(screen, dom, labels=labels)
    return dom, pan, pan.values(screen)


def gains_all(lab, d):
    """gain d (correct jets) at every g and seed for one label"""
    return {lab: {g: (d, d, d) for g in G}}


# ---------------------------------------------------------------------------------------------
# records


def test_run_from_records_roundtrip_and_envelope():
    cfg, s = syn.run_records("F011-FB", 1, syn.flat_env(40_000, lowest=200_000))
    r = sel.run_from_records(cfg, s)
    assert r.label == "F011-FB" and r.point("H1", 100_000) is None
    p = r.point("H1", 200_000)
    assert p.top1 == Fraction(40_000, N) and p.cost == 199_000
    assert r.point("H1", 450_000) == p  # flat: the lowest-cost point keeps winning


def test_envelope_na_is_never_zero():
    cfg, s = syn.run_records("F100", 1, syn.flat_env(40_000, lowest=300_000))
    s["envelope"]["H1"]["100000"] = 0
    with pytest.raises(sel.SelectError, match="never 0"):
        sel.run_from_records(cfg, s)


def test_envelope_must_match_front():
    cfg, s = syn.run_records("F100", 1, {100_000: 40_000, 200_000: 40_100, 300_000: 40_100, 350_000: 40_100,
                                         400_000: 40_200, 450_000: 40_200})
    s["envelope"]["H2"]["450000"] = dict(s["front"]["H2"][0], slack=450_000 - s["front"]["H2"][0]["cost"])
    with pytest.raises(sel.SelectError, match="differs from the step envelope"):
        sel.run_from_records(cfg, s)


def test_front_point_failing_filter_refused():
    cfg, s = syn.run_records("F100", 1, syn.flat_env(30_000))  # 30,000 / 62,000 < 0.5
    with pytest.raises(sel.SelectError, match="Chang's filter"):
        sel.run_from_records(cfg, s)


def test_step_envelope_ties():
    a = sel.Point(10, "a" * 64, 40_000, N, 90_000)
    b = sel.Point(20, "b" * 64, 40_000, N, 80_000)  # equal accuracy, lower cost wins
    c = sel.Point(5, "c" * 64, 40_000, N, 80_000)  # equal accuracy and cost, earlier epoch wins
    d = sel.Point(5, "0" * 64, 40_000, N, 80_000)  # then the hash
    assert sel.step_envelope([a, b], 100_000) == b
    assert sel.step_envelope([a, b, c], 100_000) == c
    assert sel.step_envelope([a, b, c, d], 100_000) == d
    assert sel.step_envelope([a], 50_000) is None


def test_missing_run_is_an_input_error():
    spec = syn.screen_spec()
    del spec["F011-BF"][2]
    scr = make_screen(spec)
    with pytest.raises(sel.SelectError, match="missing"):
        sel.Panel.from_screen(scr, sel.d_screen(scr))


# ---------------------------------------------------------------------------------------------
# domains


def test_d_screen_from_ch8_alone():
    spec = syn.screen_spec(lowest={"CH-8": 300_000, "F100": 400_000})
    spec["CH-8"][2][300_000] = None  # seed 2 infeasible at 300,000
    scr = make_screen(spec)
    dom = sel.d_screen(scr)
    assert dom.status == "ok" and dom.allowances == (350_000, 400_000, 450_000)


def test_d_screen_empty_is_incomplete():
    spec = syn.screen_spec()
    for g in G:
        spec["CH-8"][3][g] = None
    scr = make_screen(spec)
    dom = sel.d_screen(scr)
    assert dom.status == sel.INCOMPLETE and dom.allowances == ()
    with pytest.raises(sel.SelectError):
        sel.Panel.from_screen(scr, dom)


def test_d_conf():
    scr = make_screen(syn.screen_spec(lowest={"CH-8": 200_000}))
    ds = sel.d_screen(scr)
    conf = {}
    for s in range(4, 9):
        cfg, sm = syn.run_records("CH-8", s, syn.flat_env(40_300, lowest=300_000 if s == 6 else 100_000), stage="S4",
                                  arm_id="CH-8-C")
        conf[s] = sel.run_from_records(cfg, sm)
    dc = sel.d_conf(ds, conf, "H1", range(4, 9))
    assert dc.allowances == (300_000, 350_000, 400_000, 450_000)
    assert sel.d_conf(ds, {4: conf[4]}, "H1", range(4, 9)).status == sel.INCOMPLETE
    empty = {s: sel.run_from_records(*syn.run_records("CH-8", s, {g: None for g in G}, stage="S4", arm_id="CH-8-C"))
             for s in range(4, 9)}
    assert sel.d_conf(ds, empty, "H1", range(4, 9)).status == sel.INCOMPLETE


# ---------------------------------------------------------------------------------------------
# form rule


def test_form_step1_coverage_count():
    # BF infeasible at 100,000 in two configurations: FB covers 6, BF 4
    scr = make_screen(syn.screen_spec(lowest={"F011-BF": 200_000, "F111-BF": 200_000},
                                      gains={"F001-BF": {g: (500, 500, 500) for g in G}}))
    _, pan, vals = panel_of(scr)
    fc = sel.form_choice(pan, vals)
    assert fc["step"] == 1 and fc["form"] == "FB" and fc["counts"] == {"FB": 6, "BF": 4}
    assert fc["delta_form"] < 0  # BF better where both are feasible: logged, not decisive


def test_form_step2_sign():
    scr = make_screen(syn.screen_spec(gains={"F010-BF": {g: (30, 30, 30) for g in G}}))
    _, pan, vals = panel_of(scr)
    fc = sel.form_choice(pan, vals)
    assert fc["step"] == 2 and fc["form"] == "BF" and fc["delta_form"] == Fraction(-30 * 18, N * 108)
    scr = make_screen(syn.screen_spec(gains={"F010-FB": {g: (1, 0, 0) for g in G}}))
    _, pan, vals = panel_of(scr)
    assert sel.form_choice(pan, vals)["form"] == "FB"


def test_form_step3_exact_zero():
    # +7 and -7 jets in two cells: exactly zero in Fractions; float sums of the top-1 values need not be
    gains = {"F001-FB": {g: (7, 0, 0) for g in G}, "F110-FB": {g: (0, 0, -7) for g in G}}
    scr = make_screen(syn.screen_spec(gains=gains))
    _, pan, vals = panel_of(scr)
    fc = sel.form_choice(pan, vals)
    assert fc["delta_form"] == 0 and fc["step"] == 3 and fc["form"] == "FB"


def test_form_step3_no_common_cell():
    # no fused-BN configuration is feasible anywhere in either form: equal counts (0), no common cell
    spec = syn.screen_spec()
    for c in syn.FUSED:
        for f in ("FB", "BF"):
            for s in (1, 2, 3):
                spec["%s-%s" % (c, f)][s] = {g: None for g in G}
    scr = make_screen(spec)
    _, pan, vals = panel_of(scr)
    fc = sel.form_choice(pan, vals)
    assert fc["counts"] == {"FB": 0, "BF": 0} and fc["common_cells"] == 0 and fc["step"] == 3 and fc["form"] == "FB"


# ---------------------------------------------------------------------------------------------
# BEST


def test_best_coverage_first():
    # F111-FB has the largest gain but misses 100,000 in seed 1; F001-FB has a small gain everywhere
    gains = {"F111-FB": {g: (900, 900, 900) for g in G}, "F001-FB": {g: (10, 10, 10) for g in G}}
    spec = syn.screen_spec(gains=gains)
    spec["F111-FB"][1][100_000] = None
    scr = make_screen(spec)
    _, pan, vals = panel_of(scr)
    b = sel.best_choice(pan, vals, "FB")
    assert b["best"] == "F001" and b["ranking"][-1]["code"] == "F111"
    assert b["ranking"][-1]["coverage"] == 17 and b["ranking"][0]["coverage"] == 18


def test_best_mean_then_cost_then_id():
    gains = {"F100": {g: (20, 20, 20) for g in G}, "F010-FB": {g: (20, 20, 20) for g in G}}
    spec = syn.screen_spec(gains=gains)
    scr = make_screen(spec)
    _, pan, vals = panel_of(scr)
    b = sel.best_choice(pan, vals, "FB")
    assert [e["code"] for e in b["ranking"][:2]] == ["F010", "F100"]  # equal gain and cost: id ascending
    # lower mean actual H breaks the tie before the id: rebuild F100 at lower cost
    runs = []
    for c, s in syn.build_runs(spec):
        if c["config_code"] == "F100":
            c, s = syn.run_records("F100", c["seed"], spec["F100"][c["seed"]], cost_offset=5_000)
        runs.append(sel.run_from_records(c, s))
    scr2 = sel.Screen(runs, "H1")
    _, pan2, vals2 = panel_of(scr2)
    assert sel.best_choice(pan2, vals2, "FB")["best"] == "F100"


def test_best_excludes_f000_and_may_be_negative():
    gains = {"F000": {g: (999, 999, 999) for g in G}}
    gains.update({lab: {g: (-50, -50, -50) for g in G} for lab in ["F100"] + ["%s-FB" % c for c in syn.FUSED]})
    gains["F101-FB"] = {g: (-10, -10, -10) for g in G}
    scr = make_screen(syn.screen_spec(gains=gains))
    _, pan, vals = panel_of(scr)
    b = sel.best_choice(pan, vals, "FB")
    assert b["best"] == "F101" and b["ranking"][0]["mean_gain"] == Fraction(-10, N)
    assert all(e["code"] != "F000" for e in b["ranking"]) and len(b["ranking"]) == 7


def test_best_none_feasible_still_ranked():
    lowest = {lab: 10 ** 9 for lab in ["F100"] + ["%s-FB" % c for c in syn.FUSED]}
    spec = syn.screen_spec()
    for lab in lowest:
        for s in (1, 2, 3):
            spec[lab][s] = {g: None for g in G}
    scr = make_screen(spec)
    _, pan, vals = panel_of(scr)
    b = sel.best_choice(pan, vals, "FB")
    assert b["best"] == "F001" and b["best_has_no_feasible_cell"]


def test_best_descriptives_values():
    # F011-FB gains (40, 30, -10) and F100 (10, 20, 30): both mean 20 at equal cost, so the id decides
    # (F011 < F100); seed 3 ranks F100 first
    gains = {"F011-FB": {g: (40, 30, -10) for g in G}, "F100": {g: (10, 20, 30) for g in G}}
    scr = make_screen(syn.screen_spec(gains=gains))
    _, pan, vals = panel_of(scr)
    b = sel.best_choice(pan, vals, "FB")
    assert b["best"] == "F011" and b["ranking"][1]["code"] == "F100"
    assert b["ranking"][0]["mean_gain"] == b["ranking"][1]["mean_gain"] == Fraction(20, N)
    d = sel.best_descriptives(scr, pan, vals, "FB", b)
    assert [d["per_seed_rank"][s][0] for s in (1, 2, 3)] == ["F011", "F011", "F100"]
    assert d["seeds_best_first"] == 2
    assert d["best_vs_runner_up"]["per_seed"] == pytest.approx([30 / N, 10 / N, -40 / N])
    iv = d["best_vs_runner_up"]["interval"]
    assert iv["n"] == 3 and iv["lower"] < 0 < iv["upper"] and iv["level"] == 0.95
    assert d["best"]["label"] == sel.LABEL_OPTIMISTIC and d["text"] == "the validation-selected configuration"
    assert d["best"]["positive_g"] == 6 and d["best"]["min_per_g_mean_gain"] == pytest.approx(20 / N)
    assert d["runner_up"]["code"] == "F100" and d["runner_up"]["mean_gain"] == pytest.approx(20 / N)


# ---------------------------------------------------------------------------------------------
# g* and the band


def _band_screen(per_g_gain, lowest=None):
    gains = {"F011-FB": {g: per_g_gain[g] for g in G}}
    return make_screen(syn.screen_spec(gains=gains, lowest=lowest))


def test_g_star_default():
    # negative at 100,000 and 200,000, positive above: g* = 300,000
    pg = {100_000: (-30, -30, -30), 200_000: (-10, -10, -10), 300_000: (5, 5, 5), 350_000: (5, 5, 5),
          400_000: (6, 6, 6), 450_000: (6, 6, 6)}
    scr = _band_screen(pg)
    ds = sel.d_screen(scr)
    gs = sel.g_star(scr, ds, "F011-FB", "default")
    assert gs["status"] == "ok" and gs["g_star"] == 300_000 and gs["screen_band"] == [300_000, 350_000, 400_000, 450_000]
    assert gs["per_g"][100_000]["qualifies"] is False


def test_g_star_needs_every_higher_g():
    # positive at 100,000 but the mean at 350,000 is exactly 0 (not > 0): g* = 400,000
    pg = {100_000: (5, 5, 5), 200_000: (5, 5, 5), 300_000: (5, 5, 5), 350_000: (5, 5, -10),
          400_000: (6, 6, 6), 450_000: (6, 6, 6)}
    # keep envelopes monotone: seed 3's value at 350,000 is below its 300,000 value, so set it from CH-8 directly
    spec = syn.screen_spec(gains={"F011-FB": pg})
    spec["F011-FB"][3][350_000] = spec["CH-8"][3][350_000] - 10
    for g in (100_000, 200_000, 300_000):
        spec["F011-FB"][3][g] = spec["CH-8"][3][g] - 10
    scr = make_screen(spec)
    ds = sel.d_screen(scr)
    gs = sel.g_star(scr, ds, "F011-FB", "default")
    assert gs["per_g"][350_000]["mean"] == 0.0 and gs["g_star"] == 400_000


def test_g_star_infeasible_seed_blocks():
    pg = {g: (5, 5, 5) for g in G}
    spec = syn.screen_spec(gains={"F011-FB": pg})
    for g in G:
        spec["F011-FB"][2][g] = None if g <= 400_000 else spec["F011-FB"][2][g]
    scr = make_screen(spec)
    gs = sel.g_star(scr, sel.d_screen(scr), "F011-FB", "default")
    assert gs["g_star"] == 450_000


def test_g_star_none_is_incomplete_no_band():
    scr = _band_screen({g: (-1, -1, -1) for g in G})
    ds = sel.d_screen(scr)
    gs = sel.g_star(scr, ds, "F011-FB", "default")
    assert gs["status"] == sel.INCOMPLETE_NO_BAND and gs["screen_band"] == []
    assert sel.d_band(gs, ds).status == sel.INCOMPLETE_NO_BAND


def test_g_star_strict_sd():
    # 300,000 has mean 10 and sd 10 (gains 0, 10, 20): passes default, fails strict (mean > sd)
    pg = {100_000: (-5, -5, -5), 200_000: (-5, -5, -5), 300_000: (0, 10, 20), 350_000: (30, 30, 31),
          400_000: (30, 30, 31), 450_000: (30, 30, 31)}
    scr = _band_screen(pg)
    ds = sel.d_screen(scr)
    assert sel.g_star(scr, ds, "F011-FB", "default")["g_star"] == 300_000
    assert sel.g_star(scr, ds, "F011-FB", "strict_sd")["g_star"] == 350_000


def test_g_star_interval_rule_and_ties():
    # qualifying runs: [100k, 200k] and [400k, 450k] (equal length): the larger allowances win
    pg = {100_000: (5, 5, 5), 200_000: (5, 5, 5), 300_000: (5, 5, 5), 350_000: (5, 5, 5),
          400_000: (9, 9, 9), 450_000: (9, 9, 9)}
    spec = syn.screen_spec(gains={"F011-FB": pg})
    spec["CH-8"] = {s: {g: v + (20 if g in (300_000, 350_000) else 0) for g, v in env.items()} for s, env in spec["CH-8"].items()}
    # CH-8 must stay monotone: raise it at 400,000 and 450,000 by 20 as well and the arm's by 20
    for s in (1, 2, 3):
        for g in (400_000, 450_000):
            spec["CH-8"][s][g] += 20
            spec["F011-FB"][s][g] += 20
    scr = make_screen(spec)
    ds = sel.d_screen(scr)
    gs = sel.g_star(scr, ds, "F011-FB", "interval")
    assert gs["g_lo"] == 400_000 and gs["g_hi"] == 450_000 and gs["screen_band"] == [400_000, 450_000]
    conf = sel.Domain("ok", (200_000, 300_000, 350_000, 400_000, 450_000))
    assert sel.d_band(gs, conf).allowances == (400_000, 450_000)


def test_d_band_intersects_d_conf():
    gs = {"status": "ok", "rule": "default", "g_star": 350_000}
    assert sel.d_band(gs, sel.Domain("ok", (300_000, 350_000, 400_000))).allowances == (350_000, 400_000)
    assert sel.d_band(gs, sel.Domain("ok", (100_000, 200_000))).status == sel.INCOMPLETE_NO_BAND
    assert sel.d_band(gs, sel.Domain(sel.INCOMPLETE, ())).status == sel.INCOMPLETE


def test_d_c2_options():
    conf = sel.Domain("ok", (300_000, 400_000))
    a = sel.claim_domain_config({"base": "a", "c3r": False})
    assert sel.d_c2(a, conf, None) == conf
    b = sel.claim_domain_config({"base": "b", "band_rule": "default", "c3r": False})
    with pytest.raises(sel.UnsetDecision):
        sel.d_c2(b, conf, None)
    assert sel.d_c2(b, conf, {"status": "ok", "rule": "default", "g_star": 400_000}).allowances == (400_000,)


# ---------------------------------------------------------------------------------------------
# decision inputs fail closed


@pytest.mark.parametrize("bad", [None, {}, {"base": "c", "c3r": False}, {"base": "a"}, {"base": "b", "c3r": False},
                                 {"base": "b", "band_rule": "wider", "c3r": False}, {"base": "a", "c3r": True},
                                 {"base": "a", "c3r": "no"}, {"base": "a", "band_rule": "default", "c3r": False},
                                 {"base": "a", "c3r": False, "extra": 1}])
def test_claim_domain_fails_closed(bad):
    with pytest.raises(sel.UnsetDecision):
        sel.claim_domain_config(bad)


def test_s4_designs():
    a = sel.claim_domain_config({"base": "b", "band_rule": "default", "c3r": False})
    assert (sel.s4_design("A", a).n, sel.s4_design("A", a).c2_level, sel.s4_design("A", a).label_contrasts) == (5, 0.975, 7)
    assert sel.s4_design("A", a).seeds == (4, 5, 6, 7, 8) and sel.s4_design("B", a).seeds == tuple(range(4, 12))
    assert sel.s4_design("B", a).pid and sel.s4_design("B", a).c3_level == 0.975
    c = sel.s4_design("C", a)
    assert (c.n, c.pid, c.c3r, c.c2_level, c.label_contrasts) == (8, False, False, 0.95, 6)
    cb = sel.claim_domain_config({"base": "b", "band_rule": "default", "c3r": True, "c3r_level": "bonferroni"})
    d = sel.s4_design("C", cb)
    assert d.c3r and d.c2_level == 0.975 and d.c3r_level == 0.975
    assert not sel.s4_design("A", cb).c3r  # (c) has no effect under A and B
    cn = sel.claim_domain_config({"base": "a", "c3r": True, "c3r_level": "nested"})
    with pytest.raises(sel.UnsetDecision):
        sel.s4_design("C", cn)
    assert sel.s4_design("C", cn, contains_350k=True).c2_level == 0.95
    assert sel.s4_design("C", cn, contains_350k=True).c3r_level == 0.95
    assert sel.s4_design("C", cn, contains_350k=False).c2_level == 0.975
    for bad in (None, "D", "a"):
        with pytest.raises(sel.UnsetDecision):
            sel.s4_design(bad, a)


def test_headline_fails_closed():
    for bad in (None, "NR", "h1", ""):
        with pytest.raises(sel.UnsetDecision):
            sel.check_headline(bad)
    with pytest.raises(sel.UnsetDecision):
        sel.Screen([], None)


# ---------------------------------------------------------------------------------------------
# PID and control selections


def test_pid_selection_never_over_budget():
    env = {100_000: None, 200_000: None, 300_000: None, 350_000: None, 400_000: 40_000, 450_000: 40_100}
    cfg, s = syn.run_records("CH-8", 1, env, stage="S3", schedule="pid", arm_id="PID-CH")
    r = sel.run_from_records(cfg, s)
    ps = sel.pid_selection(r, "H1")
    assert ps["status"] == sel.COST_INFEASIBLE and ps["point"] is None
    env2 = {100_000: None, 200_000: None, 300_000: 39_000, 350_000: 39_500, 400_000: 41_000, 450_000: 41_000}
    cfg, s = syn.run_records("CH-8", 1, env2, stage="S3", schedule="pid", arm_id="PID-CH")
    r = sel.run_from_records(cfg, s)
    ps = sel.pid_selection(r, "H1")
    assert ps["status"] == "selected" and ps["point"]["val_correct"] == 39_500 and ps["slack"] == 1_000
    assert sel.check_pid_summary(r, "H1")["point"]["val_correct"] == 39_500


def test_pid_summary_mismatch_refused():
    env = {100_000: None, 200_000: None, 300_000: 39_000, 350_000: 39_500, 400_000: 41_000, 450_000: 41_000}
    cfg, s = syn.run_records("CH-8", 1, env, stage="S3", schedule="pid", arm_id="PID-CH")
    s["pid_selection"]["ckpt_sha256"] = s["front"]["H1"][0]["ckpt_sha256"]
    r = sel.run_from_records(cfg, s)
    with pytest.raises(sel.SelectError, match="differs"):
        sel.check_pid_summary(r, "H1")


def test_pid_summary_null_matches_no_selection():
    cfg, s = syn.run_records("CH-8", 1, {g: None for g in G}, stage="S3", schedule="pid", arm_id="PID-CH")
    assert s["pid_selection"] is None
    r = sel.run_from_records(cfg, s)
    assert sel.check_pid_summary(r, "H1")["status"] == sel.COST_INFEASIBLE
    env = {100_000: None, 200_000: None, 300_000: 39_000, 350_000: 39_500, 400_000: 41_000, 450_000: 41_000}
    cfg, s = syn.run_records("CH-8", 1, env, stage="S3", schedule="pid", arm_id="PID-CH")
    s["pid_selection"] = None  # the rule selects a checkpoint, the summary says none
    with pytest.raises(sel.SelectError, match="differs"):
        sel.check_pid_summary(sel.run_from_records(cfg, s), "H1")


def test_pid_filter_outcome_reported():
    cfg, s = syn.run_records("CH-8", 1, {g: None for g in G}, stage="S3", schedule="pid", arm_id="PID-CH")
    s["filter_outcome"]["H1"]["350000"] = {"outcome": sel.NO_ACCEPTED, "diagnostic_top1": 0.41, "cost": 340_000}
    r = sel.run_from_records(cfg, s)
    ps = sel.pid_selection(r, "H1")
    assert ps["status"] == sel.NO_ACCEPTED and ps["diagnostic"]["cost"] == 340_000


def test_control_selection_ties_to_earlier_epoch():
    ep = syn.epochs_trace([(0, 0.40, 900_000), (1, 0.66, 900_000), (2, 0.66, 900_000), (3, 0.65, 900_000)])
    out = sel.control_selection(ep)
    assert out["epoch"] == 1


def test_context_checkpoint_8_10():
    r = sel.run_from_records(*syn.run_records("CH-8", 1, syn.flat_env(40_000), released_front=[
        {"epoch": 10, "train_count": 360_000.0, "log_val_accuracy": 0.70, "path": "a"},
        {"epoch": 20, "train_count": 340_000.0, "log_val_accuracy": 0.64, "path": "b"},
        {"epoch": 30, "train_count": 300_000.0, "log_val_accuracy": 0.64, "path": "c"},
        {"epoch": 40, "train_count": 200_000.0, "log_val_accuracy": 0.60, "path": "d"}]))
    c = sel.context_checkpoint_8_10(r)
    assert c["point"]["path"] == "c"  # 0.70 is above 350,000; the tie at 0.64 goes to the lower count


# ---------------------------------------------------------------------------------------------
# collapse, windows, dwell


def _trace(pre_acc, e350, sq_acc, later):
    rows = [(e, pre_acc, 900_000) for e in range(0, e350)]
    rows += [(e, sq_acc, 340_000) for e in range(e350, e350 + 1000)]
    e = e350 + 1000
    for acc, h in later:
        rows += [(x, acc, h) for x in range(e, e + 500)]
        e += 500
    return syn.epochs_trace(rows)


def test_collapse_none():
    out = sel.collapse(_trace(0.66, 1500, 0.65, []), "H1")
    assert out["label"] == "no collapse" and out["e_g"] == 1500 and out["A_ref"] == pytest.approx(0.66, abs=1e-4)


def test_collapse_sustained_and_recovered():
    # cycles start at multiples of 500 at or after e_350 + 1000 = 2500
    out = sel.collapse(_trace(0.66, 1500, 0.55, [(0.58, 340_000), (0.59, 340_000)]), "H1")
    assert out["label"] == "COLLAPSE" and out["persistence"] == "SUSTAINED"
    assert out["drop"] == pytest.approx(0.11, abs=1e-4)
    out = sel.collapse(_trace(0.66, 1500, 0.55, [(0.58, 340_000), (0.62, 340_000)]), "H1")
    assert out["persistence"] == "RECOVERED at cycle [3000, 3500)"
    out = sel.collapse(_trace(0.66, 1500, 0.55, [(0.62, 360_000)]), "H1")  # above budget: does not recover
    assert out["persistence"] == "SUSTAINED"
    out = sel.collapse(_trace(0.66, 1500, 0.55, []), "H1")
    assert out["persistence"] == "SUSTAINED (no later cycle)"


def test_collapse_threshold_is_strict_and_exact():
    # A_ref - A_sq = 0.05 exactly (3,100 jets of 62,000) is not a collapse
    ep = [{"epoch": 0, "val_correct": 40_000, "val_n": N, "H1": 900_000, "H2": 900_000, "NR": 900_010},
          {"epoch": 1, "val_correct": 36_900, "val_n": N, "H1": 340_000, "H2": 340_000, "NR": 340_010}]
    assert sel.collapse(ep, "H1")["label"] == "no collapse"
    ep[1]["val_correct"] = 36_899
    assert sel.collapse(ep, "H1")["label"] == "COLLAPSE"


def test_collapse_reference_window_and_labels():
    tr = _trace(0.66, 2600, 0.6, [])
    tr[2100]["val_correct"] = syn.acc(0.70)  # epoch 2100 is outside [0, min(2000, e_350))
    out = sel.collapse(tr, "H1")
    assert out["A_ref"] == pytest.approx(0.66, abs=1e-4)
    out = sel.collapse(syn.epochs_trace([(e, 0.6, 300_000) for e in range(5)]), "H1")
    assert out["label"] == "no pre-squeeze reference" and out["e_g"] == 0
    out = sel.collapse(syn.epochs_trace([(e, 0.6, 600_000) for e in range(5)]), "H1")
    assert out["label"] == "not squeezed (cost infeasible)" and out["A_sq"] is None


def test_collapse_pid_takes_ramp_reference():
    ramp = _trace(0.70, 1500, 0.69, [])
    pid = syn.epochs_trace([(e, 0.60, 340_000) for e in range(0, 1200)])
    out = sel.collapse(pid, "H1", ref_epochs=ramp)
    assert out["A_ref"] == pytest.approx(0.70, abs=1e-4) and out["label"] == "COLLAPSE"


def test_collapse_grid_and_dwell():
    tr = _trace(0.66, 1500, 0.65, [])
    grid = sel.collapse_grid(tr, "H1")
    assert grid[100_000]["label"] == "not squeezed (cost infeasible)" and grid[350_000]["e_g"] == 1500
    d = sel.dwell(tr, "H1")
    assert d["(300000,350000]"] == 1000 and d["(450000,inf)"] == 1500 and sum(d.values()) == 2500
    d2 = sel.dwell(syn.epochs_trace([(0, 0.6, 100_000), (1, 0.6, 100_001), (2, 0.6, 450_000)]), "H1")
    assert d2["<=100000"] == 1 and d2["(100000,200000]"] == 1 and d2["(400000,450000]"] == 1


def test_collapse_table_drop_minus_ch8():
    spec = syn.screen_spec()
    runs = []
    for cfg, sm in syn.build_runs(spec):
        lab = sel.arm_label(cfg["arm_id"], cfg["config_code"], cfg["form"])
        sq = 0.60 if lab == "F011-FB" else 0.64
        runs.append(sel.run_from_records(cfg, sm, epochs=_trace(0.66, 1500, sq, [])))
    scr = sel.Screen(runs, "H1")
    t = sel.collapse_table(scr, labels=[sel.CH8, "F011-FB", "F100"])
    row = t["F011-FB"]["2"]["350000"]
    assert row["drop"] == pytest.approx(0.06, abs=1e-4) and row["drop_minus_ch8"] == pytest.approx(0.04, abs=1e-4)
    assert t["F100"]["1"]["350000"]["drop_minus_ch8"] == 0.0 and t["CH-8"]["3"]["350000"]["drop_minus_ch8"] == 0.0
    assert t["F100"]["1"]["100000"]["drop_minus_ch8"] is None  # not squeezed at 100,000


def test_event_line_and_lambda_segment():
    ep = syn.epochs_trace([(0, 0.6, 900_000), (1, 0.6, 900_000), (2, 0.6, 900_000)])
    ep[1]["binary_sites"] = {"mha.query": {"qg_b": 2}, "ffn.expand": {"qg_b": [1, 0, 3]}}
    ep[2]["binary_sites"] = {"mha.query": {"qg_b": 0}, "ffn.expand": {"qg_b": [0, 0, 3]}}
    side = {"x": {"epoch": 2, "structure_record": {"degenerate": {"K": "kernel_pruned", "Q": None}}}}
    ev = sel.event_line(ep, 1260, side)
    assert ev["first_b0"] == {"ffn.expand": 1, "mha.query": 2}
    assert ev["degenerate_onset"] == {"K": {"epoch": 2, "route": "kernel_pruned"}}
    assert ev["nearest_lr_restart"] == 1500 and ev["lambda_segment"] == "linear"
    assert sel.lambda_segment(2000) == "log" and sel.lambda_segment(7000) == "constant"


# ---------------------------------------------------------------------------------------------
# dominance and factorial


def test_dominance():
    ch = [(100_000, Fraction(60, 100)), (300_000, Fraction(65, 100))]
    assert not sel.dominates(ch, ch)  # identical: no strict improvement
    assert sel.dominates([(100_000, Fraction(61, 100)), (300_000, Fraction(65, 100))], ch)
    assert sel.dominates([(90_000, Fraction(65, 100))], ch)
    assert not sel.dominates([(100_000, Fraction(61, 100))], ch)  # no point covers (300k, 0.65)
    assert not sel.dominates([(100_000, Fraction(61, 100)), (300_000, Fraction(64, 100))], ch)
    assert sel.dominates(ch + [(50_000, Fraction(55, 100))], ch + [(60_000, Fraction(55, 100))])
    assert not sel.dominates([(90_000, Fraction(1))], [])


def _fact_values(main, inter=None, seeds=(1, 2, 3), domain=(300_000, 400_000), seed_eff=(0.0, 0.004, -0.002)):
    vals = {}
    for c in sel.FACTORIAL_CODES:
        b = sel.bits_of(c)
        for si, s in enumerate(seeds):
            for g in domain:
                v = 0.65 + seed_eff[si] + sum(m * (1 if bit else -1) / 2 for m, bit in zip(main, b))
                if inter:
                    v += inter * sel.contrast_sign(c, (1, 1, 0)) / 2
                vals[(c, s, g)] = v
    return vals


def test_factorial_additive_model():
    vals = _fact_values((0.010, -0.004, 0.002))
    f = sel.factorial(vals, (1, 2, 3), (300_000, 400_000))
    assert f["cells_averaged"] == 6
    assert f["effects"]["attention"] == pytest.approx(0.010) and f["effects"]["ffn"] == pytest.approx(-0.004)
    assert f["effects"]["head"] == pytest.approx(0.002)
    for k in ("attention:ffn", "attention:head", "ffn:head", "attention:ffn:head"):
        assert f["effects"][k] == pytest.approx(0.0, abs=1e-15)
    assert f["additive_cell"] == "F101"
    row = f["per_g"][300_000]
    assert row["df"] == 14 and row["sigma_F"] == pytest.approx(0.0, abs=1e-12)


def test_factorial_interaction_and_block_interval():
    import numpy as np
    vals = _fact_values((0.010, 0.0, 0.0), inter=0.006)
    rng = np.random.default_rng(1)
    for k in vals:
        vals[k] += float(rng.normal(0, 0.003))
    f = sel.factorial(vals, (1, 2, 3), (300_000,))
    row = f["per_g"][300_000]
    eff = row["effects"]["attention"]
    y = np.array([[vals[(c, s, 300_000)] for s in (1, 2, 3)] for c in sel.FACTORIAL_CODES])
    res = y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + y.mean()
    sig = float(np.sqrt((res ** 2).sum() / 14))
    assert row["sigma_F"] == pytest.approx(sig, rel=1e-12)
    assert eff["block"]["t"] == pytest.approx(2.1447866879169273, rel=1e-12)
    assert eff["block"]["half_width"] == pytest.approx(2.1447866879169273 * sig * (1 / 6) ** 0.5, rel=1e-12)
    assert eff["per_seed"]["n"] == 3 and eff["per_seed"]["t"] == pytest.approx(4.302652729749464, rel=1e-12)
    assert f["effects"]["attention:ffn"] == pytest.approx(0.006, abs=0.004)


def test_factorial_identical_cells_zero_effects_and_intervals():
    vals = {(c, s, g): Fraction(40_300, N) for c in sel.FACTORIAL_CODES for s in (1, 2, 3) for g in (300_000,)}
    f = sel.factorial(vals, (1, 2, 3), (300_000,))
    assert all(v == 0 for v in f["effects"].values())
    row = f["per_g"][300_000]
    for e in row["effects"].values():
        assert e["block"]["lower"] == 0.0 == e["block"]["upper"]
        assert e["per_seed"]["lower"] == 0.0 == e["per_seed"]["upper"]
    assert f["additive_cell"] == "F000"


def test_factorial_infeasible_cells_and_feasibility_effects():
    vals = _fact_values((0.01, 0.0, 0.0))
    vals[("F111", 2, 400_000)] = None
    f = sel.factorial(vals, (1, 2, 3), (300_000, 400_000))
    assert f["cells_averaged"] == 5 and f["per_g"][400_000] is None
    assert f["feasibility_effects"]["attention"] == {"binary": 23, "hgq": 24, "difference": -1}


def test_control_slot_rule():
    assert sel.control_slot(True)["first_slot"] == "DS-HGQ"
    assert sel.control_slot(False)["admission_order"] == ["FP32", "DS-HGQ", "W8A8", "GR", "BE"]


def test_m16_imports_no_backend():
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); import cbl.select, cbl.stats; "
            "bad = [m for m in ('keras', 'jax', 'jaxlib', 'tensorflow', 'torch', 'hgq', 'hls4ml', 'da4ml', "
            "'cbl.env', 'cbl.wandb_log') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)") % str(FORK)
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
