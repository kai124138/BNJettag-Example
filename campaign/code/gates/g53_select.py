"""Gate G53 (PREFLIGHT-PLAN section 3 row G53, section 4 row B3b; STUDY v0.9 sections 8.2-8.8, 13
selection and evaluation row, 19.2 items 11 and 12).

Domains, form rule, coverage-first BEST, NA and ties, g* and D_band, empty-domain INCOMPLETE,
screen tables without verdict labels, factorial effects and dominance, PID selection and collapse
windows, on constructed data (tests/fixtures/m16_synth.py; never real data). Checks, on the Mac CPU
with no backend:
  - repository: HEAD equals --expect-sha12 and `git status --porcelain` is empty;
  - the STUDY v0.9 snapshot hashes to its recorded sha256 (61e68527...89df4);
  - import isolation of the four M16 modules (static and in a fresh -I interpreter);
  - each rule on a constructed case whose answer is known by construction, one line per rule;
  - the decision inputs that have no default (headline, claim domain with every item-12 option,
    S4 option with every item-11 option, the F000 pooled-row flag) fail closed;
  - the unit tests `tests/test_select.py` and `tests/test_s2.py`, as a sub-step.

Run as

    python -I -B gates/g53_select.py --expect-sha12 SHA12 [--study PATH]

Prints one line per check and ends with "G53 PASS" (exit 0) or "G53 FAIL" (exit 1). Writes
nothing. Every number printed is an engineering reading, not a quotable result.
"""

import sys
from pathlib import Path
from fractions import Fraction

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I does not add the script directory
import m16gate as mg  # noqa: E402

sys.path.insert(0, str(mg.FORK))
sys.path.insert(0, str(mg.FORK / "tests"))
from cbl import s2  # noqa: E402
from cbl import select as sel  # noqa: E402
from cbl import stats as st  # noqa: E402
from fixtures import m16_synth as syn  # noqa: E402

G = sel.GRID
N = syn.VAL_N


def screen(spec, headline="H1"):
    return sel.Screen([sel.run_from_records(c, s) for c, s in syn.build_runs(spec)], headline)


def panel(scr):
    ds = sel.d_screen(scr)
    pan = sel.Panel.from_screen(scr, ds)
    return ds, pan, pan.values(scr)


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def main():
    args = mg.parse_args(__doc__)
    gate = mg.start("G53", args)
    mg.study_text(gate, args.study)
    mg.check_imports(gate)
    ck = gate.check

    # ---- records, NA and ties (section 8.1)
    a = sel.Point(10, "a" * 64, 40_000, N, 90_000)
    b = sel.Point(20, "b" * 64, 40_000, N, 80_000)
    c = sel.Point(5, "c" * 64, 40_000, N, 80_000)
    d = sel.Point(5, "0" * 64, 40_000, N, 80_000)
    ck(sel.step_envelope([a, b], 100_000) == b and sel.step_envelope([a, b, c], 100_000) == c
       and sel.step_envelope([a, b, c, d], 100_000) == d and sel.step_envelope([a], 50_000) is None,
       "step envelope ties: equal top-1 to lower cost, then earlier epoch, then hash; nothing at cost <= g gives NA")
    cfg, sm = syn.run_records("F100", 1, syn.flat_env(40_000, lowest=300_000))
    sm["envelope"]["H1"]["100000"] = 0
    ck(raises(lambda: sel.run_from_records(cfg, sm), sel.SelectError), "an envelope entry of 0 is refused: NA, never zero")
    cfg, sm = syn.run_records("F100", 1, syn.flat_env(30_000))
    ck(raises(lambda: sel.run_from_records(cfg, sm), sel.SelectError), "a front point failing Chang's filter (top-1 <= 0.5) is refused")

    # ---- domains (section 8.2)
    spec = syn.screen_spec(lowest={"CH-8": 300_000})
    spec["CH-8"][2][300_000] = None
    ds = sel.d_screen(screen(spec))
    ck(ds.status == "ok" and ds.allowances == (350_000, 400_000, 450_000),
       "D_screen from CH-8 alone: feasible in each of seeds 1-3 (%s)" % list(ds.allowances))
    spec = syn.screen_spec()
    for g in G:
        spec["CH-8"][3][g] = None
    scr = screen(spec)
    rec = s2.s2_record(scr, sel.claim_domain_config({"base": "b", "band_rule": "default", "c3r": False}), False, nsim=100)
    ck(sel.d_screen(scr).status == sel.INCOMPLETE and rec["status"] == sel.INCOMPLETE and "best" not in rec
       and "form_choice" not in rec, "empty D_screen: the screen outputs are INCOMPLETE, no form, no BEST")
    conf = {s: sel.run_from_records(*syn.run_records("CH-8", s, syn.flat_env(40_300, lowest=300_000 if s == 6 else 100_000),
                                                     stage="S4", arm_id="CH-8-C")) for s in range(4, 9)}
    dsc = sel.d_screen(screen(syn.screen_spec(lowest={"CH-8": 200_000})))
    dc = sel.d_conf(dsc, conf, "H1", range(4, 9))
    ck(dc.allowances == (300_000, 350_000, 400_000, 450_000), "D_conf = D_screen and CH-8 feasible in every confirmation seed")
    empty = {s: sel.run_from_records(*syn.run_records("CH-8", s, {g: None for g in G}, stage="S4", arm_id="CH-8-C"))
             for s in range(4, 9)}
    ck(sel.d_conf(dsc, empty, "H1", range(4, 9)).status == sel.INCOMPLETE, "empty D_conf: C2 INCOMPLETE")

    # ---- form rule (section 8.3)
    _, pan, vals = panel(screen(syn.screen_spec(lowest={"F011-BF": 200_000, "F111-BF": 200_000},
                                                gains={"F001-BF": {g: (500, 500, 500) for g in G}})))
    fc = sel.form_choice(pan, vals)
    ck(fc["step"] == 1 and fc["form"] == "FB" and fc["counts"] == {"FB": 6, "BF": 4} and fc["delta_form"] < 0,
       "form step 1: more configurations feasible everywhere wins (FB 6 against BF 4), Delta_form logged")
    _, pan, vals = panel(screen(syn.screen_spec(gains={"F010-BF": {g: (30, 30, 30) for g in G}})))
    fc = sel.form_choice(pan, vals)
    ck(fc["step"] == 2 and fc["form"] == "BF" and fc["delta_form"] == Fraction(-30 * 18, N * 108),
       "form step 2: equal counts, Delta_form < 0 chooses BF (exact %s)" % fc["delta_form"])
    _, pan, vals = panel(screen(syn.screen_spec(gains={"F001-FB": {g: (7, 0, 0) for g in G}, "F110-FB": {g: (0, 0, -7) for g in G}})))
    fc = sel.form_choice(pan, vals)
    ck(fc["delta_form"] == 0 and fc["step"] == 3 and fc["form"] == "FB", "form step 3: Delta_form exactly 0 (Fractions) chooses FB")
    spec = syn.screen_spec()
    for cc in syn.FUSED:
        for f in ("FB", "BF"):
            for s in (1, 2, 3):
                spec["%s-%s" % (cc, f)][s] = {g: None for g in G}
    _, pan, vals = panel(screen(spec))
    fc = sel.form_choice(pan, vals)
    ck(fc["common_cells"] == 0 and fc["step"] == 3 and fc["form"] == "FB", "form step 3: no common cell chooses FB")

    # ---- BEST (section 8.4)
    spec = syn.screen_spec(gains={"F111-FB": {g: (900, 900, 900) for g in G}, "F001-FB": {g: (10, 10, 10) for g in G}})
    spec["F111-FB"][1][100_000] = None
    _, pan, vals = panel(screen(spec))
    bc = sel.best_choice(pan, vals, "FB")
    ck(bc["best"] == "F001" and bc["ranking"][-1]["code"] == "F111" and bc["ranking"][-1]["coverage"] == 17,
       "BEST coverage first: a missed cell is a deficit (F111, coverage 17, ranks last despite the largest gain)")
    spec = syn.screen_spec(gains={"F100": {g: (20, 20, 20) for g in G}, "F010-FB": {g: (20, 20, 20) for g in G}})
    _, pan, vals = panel(screen(spec))
    tie_id = sel.best_choice(pan, vals, "FB")["best"]
    runs = []
    for cfg, sm in syn.build_runs(spec):
        if cfg["config_code"] == "F100":
            cfg, sm = syn.run_records("F100", cfg["seed"], spec["F100"][cfg["seed"]], cost_offset=5_000)
        runs.append(sel.run_from_records(cfg, sm))
    _, pan2, vals2 = panel(sel.Screen(runs, "H1"))
    tie_h = sel.best_choice(pan2, vals2, "FB")["best"]
    ck(tie_id == "F010" and tie_h == "F100", "BEST ties: equal mean gain to lower mean actual H, then configuration id ascending")
    gains = {"F000": {g: (999, 999, 999) for g in G}}
    gains.update({lab: {g: (-50, -50, -50) for g in G} for lab in ["F100"] + ["%s-FB" % x for x in syn.FUSED]})
    gains["F101-FB"] = {g: (-10, -10, -10) for g in G}
    _, pan, vals = panel(screen(syn.screen_spec(gains=gains)))
    bc = sel.best_choice(pan, vals, "FB")
    ck(bc["best"] == "F101" and bc["ranking"][0]["mean_gain"] < 0 and all(e["code"] != "F000" for e in bc["ranking"]),
       "BEST excludes F000 and may score negative")

    # ---- g* and D_band (section 8.2, section 19.3)
    def band_screen(pg):
        return screen(syn.screen_spec(gains={"F011-FB": pg}))
    pg = {100_000: (-30, -30, -30), 200_000: (-10, -10, -10), 300_000: (5, 5, 5), 350_000: (5, 5, 5),
          400_000: (6, 6, 6), 450_000: (6, 6, 6)}
    scr = band_screen(pg)
    gs = sel.g_star(scr, sel.d_screen(scr), "F011-FB", "default")
    ck(gs["g_star"] == 300_000 and gs["screen_band"] == [300_000, 350_000, 400_000, 450_000], "g* default rule: 300,000")
    gb = sel.d_band(gs, sel.Domain("ok", (100_000, 350_000, 450_000)))
    ck(gb.allowances == (350_000, 450_000), "D_band = {g in D_conf : g >= g*}")
    scr = band_screen({g: (-1, -1, -1) for g in G})
    gs0 = sel.g_star(scr, sel.d_screen(scr), "F011-FB", "default")
    ck(gs0["status"] == sel.INCOMPLETE_NO_BAND and sel.d_band(gs0, sel.Domain("ok", G)).status == sel.INCOMPLETE_NO_BAND,
       "no qualifying g: D_band empty, C2 INCOMPLETE (no band)")
    pg = {100_000: (-5, -5, -5), 200_000: (-5, -5, -5), 300_000: (0, 10, 20), 350_000: (30, 30, 31),
          400_000: (30, 30, 31), 450_000: (30, 30, 31)}
    scr = band_screen(pg)
    ck(sel.g_star(scr, sel.d_screen(scr), "F011-FB", "strict_sd")["g_star"] == 350_000,
       "g* alternative (i), mean above one s_d: 300,000 (mean = sd exactly) excluded, g* = 350,000")
    spec = syn.screen_spec(gains={"F011-FB": {g: ((5, 5, 5) if g < 400_000 else (9, 9, 9)) for g in G}})
    for s in (1, 2, 3):
        for g in (300_000, 350_000, 400_000, 450_000):
            spec["CH-8"][s][g] += 20
        for g in (400_000, 450_000):
            spec["F011-FB"][s][g] += 20
    scr = screen(spec)
    gi = sel.g_star(scr, sel.d_screen(scr), "F011-FB", "interval")
    ck(gi["g_lo"] == 400_000 and gi["g_hi"] == 450_000, "g* alternative (ii), interval band: equal-length runs go to the larger allowances")
    ck([round(st.normal_sign_probability(x), 2) for x in (0.0, 0.5, 1.0)] == [0.5, 0.81, 0.96],
       "band entry probability Phi(delta sqrt(3)) at 0, 0.5, 1.0 sigma_d: 0.50, 0.81, 0.96 (STUDY v0.9 :906)")

    # ---- screen outputs carry no verdict label (section 8.6)
    spec = syn.screen_spec(gains={"F011-FB": {g: (12, 9, 15) for g in G}, "F100": {g: (3, 2, 1) for g in G}})
    spec["F011-FB"][2][450_000] += 4
    scr = screen(spec)
    rec = s2.s2_record(scr, sel.claim_domain_config({"base": "b", "band_rule": "default", "c3r": False}), False, nsim=2_000)
    ck(rec["status"] == "ok" and s2.verdict_tokens(rec) == [], "S2 validation record holds no ABOVE, BELOW, CROSSING or LOCAL token")
    ck(rec["best"]["best"] == "F011" and rec["best_descriptives"]["best"]["label"] == sel.LABEL_OPTIMISTIC,
       "BEST's validation gains carry the label '%s'" % sel.LABEL_OPTIMISTIC)
    ho = {}
    for (lab, s), r in scr.runs.items():
        for cc in sel.COUNTS:
            for g in G:
                p = r.point(cc, g)
                if p is not None:
                    ho[p.ckpt_sha256] = {"correct": int(round(float(p.top1) * 260_000)), "n": 260_000}
    for cc in sel.COUNTS:
        t = s2.screen_heldout_tables(scr, rec["form_choice"]["form"], rec["best"]["best"], ho, False, count=cc)
        ck(s2.verdict_tokens(t) == [] and t["label"] == sel.LABEL_SCREEN_HELDOUT,
           "screen held-out tables under %s hold no verdict token and carry the screen label" % cc)
    ck(all(1 <= v <= len(rec["d_screen"]["allowances"]) for v in rec["plug_in"]["D_screen"]["distinct_pairs_per_seed"].values()),
       "S2 prints the distinct (BEST, CH-8) checkpoint pairs per seed: %s" % rec["plug_in"]["D_screen"]["distinct_pairs_per_seed"])

    # ---- factorial effects and dominance (section 8.6)
    vals = {}
    seed_eff = (0.0, 0.004, -0.002)
    for code in sel.FACTORIAL_CODES:
        bits = sel.bits_of(code)
        for si, s in enumerate((1, 2, 3)):
            for g in (300_000, 400_000):
                vals[(code, s, g)] = 0.65 + seed_eff[si] + sum(m * (1 if bt else -1) / 2 for m, bt in zip((0.010, -0.004, 0.002), bits))
    f = sel.factorial(vals, (1, 2, 3), (300_000, 400_000))
    ck(abs(f["effects"]["attention"] - 0.010) < 1e-12 and abs(f["effects"]["ffn"] + 0.004) < 1e-12
       and abs(f["effects"]["head"] - 0.002) < 1e-12 and all(abs(f["effects"][k]) < 1e-12 for k in
                                                              ("attention:ffn", "attention:head", "ffn:head", "attention:ffn:head")),
       "factorial on an additive model: main effects 0.010, -0.004, 0.002 and zero interactions")
    row = f["per_g"][300_000]
    ck(row["df"] == 14 and abs(row["effects"]["attention"]["block"]["t"] - 2.1447866879169273) < 1e-12
       and abs(st.t_crit(15, 0.95) * (1 / 6) ** 0.5 - 0.8756) < 1e-4,
       "randomized-block interval: df = 14, t = 2.145, half-width t sqrt(1/6) = 0.876 sigma_F (STUDY v0.9 :1005)")
    ck(abs(st.t_crit(3, 0.95) * (2 / 3) ** 0.5 - 3.513) < 5e-4, "three-seed paired interval half-width 4.303 sqrt(2/3) = 3.513 sigma")
    ck(f["additive_cell"] == "F101", "additive-model cell: each group binary where its main effect is positive (F101)")
    ch = [(100_000, Fraction(60, 100)), (300_000, Fraction(65, 100))]
    ck(not sel.dominates(ch, ch) and sel.dominates([(100_000, Fraction(61, 100)), (300_000, Fraction(65, 100))], ch)
       and not sel.dominates([(100_000, Fraction(61, 100))], ch),
       "per-seed dominance: identical fronts do not dominate; covering every CH-8 point with one strict improvement does")

    # ---- PID selection, control selection and collapse windows (sections 8.6, 8.8, 8.9)
    env = {100_000: None, 200_000: None, 300_000: 39_000, 350_000: 39_500, 400_000: 41_000, 450_000: 41_000}
    r = sel.run_from_records(*syn.run_records("CH-8", 1, env, stage="S3", schedule="pid", arm_id="PID-CH"))
    ps = sel.pid_selection(r, "H1")
    env2 = {100_000: None, 200_000: None, 300_000: None, 350_000: None, 400_000: 40_000, 450_000: 40_100}
    r2 = sel.run_from_records(*syn.run_records("CH-8", 1, env2, stage="S3", schedule="pid", arm_id="PID-CH"))
    ck(ps["point"]["val_correct"] == 39_500 and ps["slack"] == 1_000 and sel.pid_selection(r2, "H1")["status"] == sel.COST_INFEASIBLE,
       "PID selection: highest top-1 at H <= 350,000 with slack; never the closest over-budget point")

    def trace(pre, e350, sq, later):
        rows = [(e, pre, 900_000) for e in range(e350)] + [(e, sq, 340_000) for e in range(e350, e350 + 1000)]
        e = e350 + 1000
        for acc, h in later:
            rows += [(x, acc, h) for x in range(e, e + 500)]
            e += 500
        return syn.epochs_trace(rows)
    o1 = sel.collapse(trace(0.66, 1500, 0.55, [(0.58, 340_000), (0.59, 340_000)]), "H1")
    o2 = sel.collapse(trace(0.66, 1500, 0.55, [(0.58, 340_000), (0.62, 340_000)]), "H1")
    o3 = sel.collapse(syn.epochs_trace([(e, 0.6, 300_000) for e in range(5)]), "H1")
    o4 = sel.collapse(syn.epochs_trace([(e, 0.6, 600_000) for e in range(5)]), "H1")
    ck(o1["label"] == "COLLAPSE" and o1["persistence"] == "SUSTAINED" and o2["persistence"] == "RECOVERED at cycle [3000, 3500)"
       and o3["label"] == "no pre-squeeze reference" and o4["label"] == "not squeezed (cost infeasible)",
       "collapse: reference [0, min(2000, e_350)), squeeze window [e_350, e_350 + 1000), later LR cycles, both NA labels")
    ep = [{"epoch": 0, "val_correct": 40_000, "val_n": N, "H1": 900_000, "H2": 900_000, "NR": 900_010},
          {"epoch": 1, "val_correct": 36_900, "val_n": N, "H1": 340_000, "H2": 340_000, "NR": 340_010}]
    ok_strict = sel.collapse(ep, "H1")["label"] == "no collapse"
    ep[1]["val_correct"] = 36_899
    ck(ok_strict and sel.collapse(ep, "H1")["label"] == "COLLAPSE", "collapse threshold: A_ref - A_sq > 0.05, strict and exact")
    runs = []
    for cfg, sm in syn.build_runs(syn.screen_spec()):
        lab = sel.arm_label(cfg["arm_id"], cfg["config_code"], cfg["form"])
        runs.append(sel.run_from_records(cfg, sm, epochs=trace(0.66, 1500, 0.60 if lab == "F011-FB" else 0.64, [])))
    ct = sel.collapse_table(sel.Screen(runs, "H1"), labels=[sel.CH8, "F011-FB"])
    ck(abs(ct["F011-FB"]["1"]["350000"]["drop_minus_ch8"] - 0.04) < 1e-4,
       "the paired A_ref - A_sq drop: arm minus CH-8 on the same seed (0.04)")
    ck(sel.control_selection(syn.epochs_trace([(0, 0.40, 900_000), (1, 0.66, 900_000), (2, 0.66, 900_000)]))["epoch"] == 1,
       "FP32 and W8A8 control selection: highest top-1 over the run, ties to the earlier epoch")

    # ---- per-class background rejection at eps_S = 0.5 (section 8, constructive K6)
    import numpy as np
    labels = np.array([0] * 4 + [1] * 4)
    scores = np.zeros((8, 5))
    scores[:, 0] = [0.9, 0.8, 0.3, 0.2, 0.85, 0.1, 0.05, 0.01]  # signal: two of four above 0.8; background: one of four
    rj = st.background_rejection(scores, labels, 0)
    ck(rj["eps_b"] == 0.25 and rj["rejection"] == 4.0, "background rejection at eps_S = 0.5 on a constructed case: eps_B = 0.25, 1/eps_B = 4")

    # ---- decision inputs fail closed (section 19.2 items 11 and 12)
    bad_claims = [None, {}, {"base": "c", "c3r": False}, {"base": "b", "c3r": False}, {"base": "a", "c3r": True},
                  {"base": "a", "c3r": False, "x": 1}]
    ck(all(raises(lambda b=b: sel.claim_domain_config(b), sel.UnsetDecision) for b in bad_claims),
       "claim_domain unset or malformed raises (base, band_rule under b, c3r, c3r_level under c3r)")
    good = [{"base": "a", "c3r": False}, {"base": "a", "c3r": True, "c3r_level": "bonferroni"},
            {"base": "a", "c3r": True, "c3r_level": "nested"}] + [
        {"base": "b", "band_rule": r, "c3r": c3, **({"c3r_level": "nested"} if c3 else {})} for r in sel.BAND_RULES for c3 in (False, True)]
    ck(all(sel.claim_domain_config(gd).as_dict() for gd in good), "every item-12 option is accepted: (a), (b) with each band rule, (c) Bonferroni or nested")
    cl = sel.claim_domain_config({"base": "b", "band_rule": "default", "c3r": False})
    ck(all(raises(lambda o=o: sel.s4_design(o, cl), sel.UnsetDecision) for o in (None, "", "D")) and
       [sel.s4_design(o, cl).option for o in ("A", "B", "C")] == ["A", "B", "C"],
       "s4_option unset or unknown raises; A, B and C are each accepted")
    ck(raises(lambda: sel.check_headline(None), sel.UnsetDecision) and raises(lambda: sel.Screen([], None), sel.UnsetDecision),
       "headline unset raises")
    ck(raises(lambda: s2.pooled_variance(scr, "FB", 300_000, None), sel.UnsetDecision),
       "the F000 pooled-row flag (section 13 determinism record) unset raises")

    mg.run_pytest(gate, ["tests/test_select.py", "tests/test_s2.py"])
    return mg.finish(gate)


if __name__ == "__main__":
    sys.exit(main())
