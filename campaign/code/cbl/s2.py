"""M16 screen outputs: the S2 validation record and the screen's descriptive held-out tables
(PREFLIGHT-PLAN section 2 row M16; STUDY v0.8 sections 6, 8.2-8.4, 8.6, 8.9).

STUDY means `review/snapshots/v0.8/STUDY.md` (sha256 290022fa...98ca8); line numbers refer to it.
STUDY v0.9 (`review/snapshots/v0.9/STUDY.md`, sha256 61e68527...89df4) is the version of record: the
power-table entries and plug-in powers this module prints come from `cbl.stats`, closed form at
rho = 0 and the stated simulation at rho = 0.9 (v0.9 :749). The null selection reference stays a
simulation at both rho, since selection has no closed form.

The screen carries no verdict label (:957): nothing in this module produces ABOVE, BELOW, CROSSING
or LOCAL, and `verdict_tokens` lets a gate check it. Every validation gain and plug-in power carries
the label "optimistic: selected on these seeds' validation" (:978). The held-out tables exist only
after the section 8.5 freeze; they take the sealed reader's output as an argument and never open a
file themselves.

Every number on PREFLIGHT data is an engineering reading, not a quotable result. Owner decisions
B3b-16 to B3b-20 are recorded in `preflight/gates/G53/<sha12>/PREFLIGHT-rows.md`.

Imports: the standard library, numpy and scipy (through cbl.stats) only.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from cbl import select as sel
from cbl import stats as st

VERDICT_RE = re.compile(r"\b(ABOVE|BELOW|CROSSING|LOCAL)\b")
NULL_REF_CHUNK = 10_000  # owner decision B3b-16: draws in chunks of 10,000 confirmations
RHOS = (0.0, 0.9)  # :964, :966, :978
VAL_JETS, HELDOUT_JETS = 62_000, 260_000  # :979


class S2Error(ValueError):
    pass


def verdict_tokens(obj) -> List[str]:
    """Every string in `obj` (keys and values) that contains a verdict token; empty for a screen
    output (:957)."""
    out = []
    if isinstance(obj, str):
        if VERDICT_RE.search(obj):
            out.append(obj)
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            out.extend(verdict_tokens(str(k)))
            out.extend(verdict_tokens(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(verdict_tokens(v))
    return out


def _f(x):
    return float(x) if isinstance(x, Fraction) else x


# ---------------------------------------------------------------------------------------------
# pooled seed variance (section 8.6 :971-976)


def pooled_variance(screen: sel.Screen, form: str, g: int, f000_bitwise_equal_ch8) -> dict:
    """At allowance g: the two-way table of validation top-1 with one row per arm (CH-8 and the
    chosen-form cells F000-F111) and one column per seed, keeping the arms feasible in all three
    seeds; sigma-hat_pool^2 is its residual mean square with df = 2 (rows - 1), sigma-hat_d =
    sqrt(2) sigma-hat_pool, with the 95 % chi-square band of sigma_d / sigma-hat_d, each arm's own
    paired s_d against CH-8 and the seed mean square over the residual mean square.

    F000's row is left out when the section 13 determinism record's F000 run equals its CH-8 runs
    bitwise (`f000_bitwise_equal_ch8` True), and enters otherwise; the flag has no default (:976)."""
    if not isinstance(f000_bitwise_equal_ch8, bool):
        raise sel.UnsetDecision("f000_bitwise_equal_ch8 must be true or false: it comes from the section 13 "
                                "determinism record and is fixed before S1")
    rows, names, own = [], [], {}
    labels = [sel.CH8] + [sel.label_in_form(c, form) for c in sel.FACTORIAL_CODES
                          if not (c == sel.F000 and f000_bitwise_equal_ch8)]
    for lab in labels:
        pts = [screen.point(lab, s, g) for s in screen.seeds]
        if any(p is None for p in pts):
            continue
        rows.append([float(p.top1) for p in pts])
        names.append(lab)
        if lab != sel.CH8:
            ch = [screen.point(sel.CH8, s, g) for s in screen.seeds]
            own[lab] = st.paired_interval([float(p.top1 - q.top1) for p, q in zip(pts, ch)]).sd
    out = {"g": g, "rows": names, "f000_row": not f000_bitwise_equal_ch8, "own_paired_sd": own}
    if len(rows) < 2:
        out.update({"status": "NA", "reason": "fewer than two arms feasible in all seeds"})
        return out
    tw = st.two_way(rows)
    lo, hi = st.chi2_band(tw.df_residual)
    sig_pool = tw.sigma
    out.update({"status": "ok", "df": tw.df_residual, "sigma_pool": sig_pool, "sigma_d": math.sqrt(2.0) * sig_pool,
                "band_sigma_d_over_hat": [lo, hi], "sigma_d_upper": math.sqrt(2.0) * sig_pool * hi,
                "ms_seed_over_ms_residual": tw.as_dict()["ms_columns_over_ms_residual"],
                "two_way": tw.as_dict()})
    return out


# ---------------------------------------------------------------------------------------------
# null selection reference (section 8.6 :978; v0.8 constructive B2)


def null_selection_reference(panel: sel.Panel, sigma_pool: Mapping[int, float], rho: float,
                             nsim: int = st.NSIM, seed: int = st.SEED, chunk: int = NULL_REF_CHUNK) -> dict:
    """The median and 95th percentile of the selected candidate's mean paired validation gain when
    every candidate's true gain is 0, simulated with the exact section 8.3-8.4 rules (`form_choice`
    and `best_choice` on the panel, whose feasibility and actual H are the observed screen pattern).

    Simulated unit: each arm's top-1 at (seed, g), CH-8 included, with residual sd sigma_pool[g];
    equicorrelation rho between the allowances of one (arm, seed); the two forms of a configuration
    drawn with correlation rho_f = rho; arms and seeds otherwise independent (owner decision B3b-17:
    the separable composition, corr(FB at g, BF at g') = rho_f * rho for g != g'). Layout (owner
    decision B3b-16): one generator default_rng(seed); per chunk of `chunk` confirmations, w of
    shape (chunk, V, S) then e of shape (chunk, V, S, D), V = 2 + 2 x (fused configurations in the
    panel); vector 0 is CH-8, 1 is F100, 2..7 the FB draws, 8..13 the independent parts of BF."""
    labels = panel.labels
    if not (set(labels) <= {sel.CH8, "F100"} | {sel.label_in_form(c, f) for c in sel.FUSED_BN_CODES for f in sel.FORMS}):
        raise S2Error("the null reference panel holds CH-8, F100 and fused-BN configurations only")
    if not 0.0 <= rho < 1.0:
        raise S2Error("rho must lie in [0, 1)")
    S, D = len(panel.seeds), len(panel.domain)
    sig = np.array([float(sigma_pool[g]) for g in panel.domain])
    if np.any(~np.isfinite(sig)) or np.any(sig < 0):
        raise S2Error("sigma_pool must be finite and >= 0 at every g")
    fused = panel.fused_present
    V = 2 + 2 * len(fused)
    vec = {sel.CH8: (0, None), "F100": (1, None)}
    for k, c in enumerate(fused):
        vec[sel.label_in_form(c, "FB")] = (2 + k, None)
        vec[sel.label_in_form(c, "BF")] = (2 + k, 2 + len(fused) + k)
    rho_f = rho
    order = sorted(panel.index, key=panel.index.get)
    rng = np.random.default_rng(seed)
    out_vals = np.empty(nsim)
    done = 0
    sr, s1r = math.sqrt(rho), math.sqrt(1.0 - rho)
    while done < nsim:
        m = min(chunk, nsim - done)
        w = rng.standard_normal((m, V, S))
        e = rng.standard_normal((m, V, S, D))
        z = sr * w[:, :, :, None] + s1r * e  # (m, V, S, D)
        cells = np.empty((m, len(order)))
        for j, (lab, s, g) in enumerate(order):
            a, b = vec[lab]
            si, gi = panel.seeds.index(s), panel.domain.index(g)
            x = z[:, a, si, gi] if b is None else rho_f * z[:, a, si, gi] + math.sqrt(1.0 - rho_f ** 2) * z[:, b, si, gi]
            cells[:, j] = sig[gi] * x
        for i, row in enumerate(cells.tolist()):
            fc = sel.form_choice(panel, row)
            bc = sel.best_choice(panel, row, fc["form"])
            mg = bc["ranking"][0]["mean_gain"]
            out_vals[done + i] = mg if mg is not None else math.nan
        done += m
    finite = out_vals[np.isfinite(out_vals)]
    if finite.size == 0:
        return {"status": "NA", "reason": "the selected candidate has no feasible cell", "rho": rho, "nsim": nsim}
    return {"status": "ok", "rho": rho, "rho_f": rho_f, "nsim": nsim, "seed": seed, "median": float(np.median(finite)),
            "p95": float(np.quantile(finite, 0.95)), "mean": float(finite.mean()), "sd": float(finite.std(ddof=1)),
            "n_finite": int(finite.size)}


# ---------------------------------------------------------------------------------------------
# the S2 record


def _distinct_pairs(screen: sel.Screen, best_label: str, domain: Sequence[int], count: Optional[str] = None) -> Dict[str, int]:
    out = {}
    for s in screen.seeds:
        pairs = set()
        for g in domain:
            p, q = screen.point(best_label, s, g, count), screen.point(sel.CH8, s, g, count)
            pairs.add((p.ckpt_sha256 if p else None, q.ckpt_sha256 if q else None))
        out[str(s)] = len(pairs)
    return out


def _plugin_block(pg: Mapping[int, dict], domain: Sequence[int], sds: Mapping[int, float], factor: float) -> dict:
    """Plug-in powers for every S2 option and rho over `domain`, with sigma at g = factor * sds[g]."""
    out = {}
    infeasible = [g for g in domain if not pg[g]["all_feasible"]]
    for name, n, level in sel.S2_OPTIONS:
        out[name] = {}
        for rho in RHOS:
            if not domain:
                out[name][str(rho)] = {"status": "NA", "reason": "empty domain"}
                continue
            if infeasible:
                feas = [g for g in domain if g not in infeasible]
                beside = (st.plugin_power([float(pg[g]["mean"]) for g in feas], [factor * sds[g] for g in feas], n, level, rho).as_dict()
                          if feas else None)
                out[name][str(rho)] = {"status": "zero", "joint": 0.0, "smallest_single": 0.0,
                                       "reason": "BEST infeasible in some screen seed at %s; the criterion needs feasibility "
                                                 "at every allowance (owner decision B3b-18)" % infeasible,
                                       "over_feasible_allowances_only": beside}
                continue
            out[name][str(rho)] = st.plugin_power([float(pg[g]["mean"]) for g in domain],
                                                  [factor * sds[g] for g in domain], n, level, rho).as_dict()
    return out


def s2_record(screen: sel.Screen, claim: sel.ClaimDomain, f000_bitwise_equal_ch8, run_null_reference: bool = True,
              nsim: int = st.NSIM, best_degenerate_attention: Optional[bool] = None) -> dict:
    """The S2 validation outputs in the section 8.6 order (:959-982): D_screen; the form and its
    inputs; BEST and its descriptives; per g the paired s_d and, per confirmation option, the
    threshold and the true gain for 80 % at the screen s_d and at its upper bound; plug-in powers at
    BEST's observed gains over D_screen and, under (b), over the screen band, at s_d and at 6.28 s_d,
    with the smallest single-allowance power and the distinct checkpoint pairs per seed; the pooled
    sigma-hat_pool with its band and the same quantities at sigma-hat_d and its upper bound; g* and
    the screen band under (b); labels, the runner-up and the null selection reference; coverage of
    every configuration; per-seed dominance and the factorial; the held control slot."""
    claim = sel.claim_domain_config(claim.as_dict() if isinstance(claim, sel.ClaimDomain) else claim)
    if not isinstance(f000_bitwise_equal_ch8, bool):
        raise sel.UnsetDecision("f000_bitwise_equal_ch8 must be true or false (section 13 determinism record)")
    rec = {"stage": "S2", "split": "validation", "n_jets": VAL_JETS, "seeds": list(screen.seeds),
           "headline": screen.headline, "label": sel.LABEL_OPTIMISTIC, "claim_domain": claim.as_dict()}
    ds = sel.d_screen(screen)
    rec["d_screen"] = ds.as_dict()
    if ds.status != "ok":
        rec["status"] = sel.INCOMPLETE
        rec["reason"] = ds.reason
        return rec
    dom = ds.allowances
    pan = sel.Panel.from_screen(screen, ds)
    vals = pan.values(screen)
    form = sel.form_choice(pan, vals)
    rec["form_choice"] = {k: _f(v) for k, v in form.items()}
    best = sel.best_choice(pan, vals, form["form"])
    rec["best"] = {"best": best["best"], "best_label": best["best_label"], "best_has_no_feasible_cell": best["best_has_no_feasible_cell"],
                   "ranking": [{k: _f(v) for k, v in e.items()} for e in best["ranking"]]}
    rec["best_descriptives"] = sel.best_descriptives(screen, pan, vals, form["form"], best)
    bl = best["best_label"]
    pg = sel.per_g_gains(screen, bl, dom)
    up = st.screen_sd_upper_factor()
    per_g = {}
    for g in dom:
        v = pg[g]
        row = {"gains": [_f(x) for x in v["gains"]], "all_feasible": v["all_feasible"], "mean": _f(v["mean"]),
               "s_d": v["sd"], "options": {}}
        for name, n, level in sel.S2_OPTIONS:
            thr = st.threshold(n, level)
            o = {"threshold_in_s_d": thr, "threshold_top1": thr * v["sd"] if v["sd"] is not None else None, "gain_for_80pct": {}}
            for rho in RHOS:
                e = st.power_table_entry(n, level, rho, len(dom))
                o["gain_for_80pct"][str(rho)] = {"in_sigma_d": e, "at_s_d": e * v["sd"] if v["sd"] is not None else None,
                                                 "at_upper_s_d": e * up * v["sd"] if v["sd"] is not None else None}
            row["options"][name] = o
        per_g[str(g)] = row
    rec["per_g"] = per_g
    rec["screen_sd_upper_factor"] = up
    feas_dom = [g for g in dom if pg[g]["all_feasible"]]
    sds = {g: pg[g]["sd"] for g in feas_dom}
    gs = sel.g_star(screen, ds, bl, claim.band_rule) if claim.base == "b" else None
    rec["g_star"] = gs and {k: (v if k != "per_g" else {str(g): r for g, r in v.items()}) for k, v in gs.items()}
    domains = {"D_screen": list(dom)}
    if gs is not None:
        domains["screen_band"] = list(gs["screen_band"])
    plug = {}
    for dname, dlist in domains.items():
        sd_any = {g: (pg[g]["sd"] if pg[g]["sd"] is not None else math.nan) for g in dlist}
        plug[dname] = {"at_s_d": _plugin_block(pg, dlist, sd_any, 1.0), "at_upper_s_d": _plugin_block(pg, dlist, sd_any, up),
                       "distinct_pairs_per_seed": _distinct_pairs(screen, bl, dlist), "label": sel.LABEL_OPTIMISTIC}
    rec["plug_in"] = plug
    rec["g_nonpositive_or_infeasible"] = sorted(g for g in dom if not pg[g]["all_feasible"] or pg[g]["mean"] <= 0)
    pooled = {str(g): pooled_variance(screen, form["form"], g, f000_bitwise_equal_ch8) for g in dom}
    rec["pooled"] = pooled
    at_pool = {}
    if all(pooled[str(g)]["status"] == "ok" for g in dom) and all(pg[g]["all_feasible"] for g in dom):
        sdp = {g: pooled[str(g)]["sigma_d"] for g in dom}
        sdu = {g: pooled[str(g)]["sigma_d_upper"] for g in dom}
        for g in dom:
            sd_best, hi = pg[g]["sd"], sdu[g]
            pooled[str(g)]["best_own_sd_above_band"] = bool(sd_best is not None and sd_best > hi)
        if any(sdp[g] == 0.0 for g in dom):
            at_pool = {"status": "NA", "reason": "sigma-hat_d is 0 at an allowance (owner decision B3b-2)"}
        else:
            at_pool = {"thresholds": {name: {str(g): st.threshold(n, level) * sdp[g] for g in dom} for name, n, level in sel.S2_OPTIONS},
                       "thresholds_upper": {name: {str(g): st.threshold(n, level) * sdu[g] for g in dom} for name, n, level in sel.S2_OPTIONS},
                       "plug_in": {dname: _plugin_block(pg, dl, sdp, 1.0) for dname, dl in domains.items()},
                       "plug_in_upper": {dname: _plugin_block(pg, dl, sdu, 1.0) for dname, dl in domains.items()},
                       "label": sel.LABEL_OPTIMISTIC}
    else:
        at_pool = {"status": "NA", "reason": "sigma-hat_pool or BEST's feasibility missing at some g of D_screen"}
    rec["at_pooled_sigma"] = at_pool
    rec["validation_vs_heldout_se_ratio"] = math.sqrt(HELDOUT_JETS / VAL_JETS)
    if run_null_reference and all(pooled[str(g)]["status"] == "ok" for g in dom):
        null_labels = [l for l in pan.labels if l != sel.F000]
        npan = sel.Panel(null_labels, pan.seeds, pan.domain, pan.feasible, pan.cost)
        sp = {g: pooled[str(g)]["sigma_pool"] for g in dom}
        rec["null_selection_reference"] = {str(rho): null_selection_reference(npan, sp, rho, nsim=nsim) for rho in RHOS}
    else:
        rec["null_selection_reference"] = {"status": "not run"}
    rec["coverage"] = {lab: pan.coverage(lab) for lab in pan.labels if lab != sel.CH8}
    rec["dominance"] = dominance_table(screen, form["form"])
    fv = sel.factorial_values(screen, form["form"], dom)
    fac = sel.factorial(fv, screen.seeds, dom)
    rec["factorial"] = _factorial_json(fac)
    rec["additive_cell_vs_best"] = {"additive_cell": fac["additive_cell"], "best": best["best"],
                                    "differ": fac["additive_cell"] is not None and fac["additive_cell"] != best["best"]}
    if best_degenerate_attention is not None:
        rec["control_slot"] = sel.control_slot(bool(best_degenerate_attention))
    rec["status"] = "ok"
    return rec


def _factorial_json(fac: dict) -> dict:
    out = dict(fac)
    out["per_g"] = {str(g): r for g, r in fac["per_g"].items()}
    return out


def dominance_table(screen: sel.Screen, form: str, heldout: Optional[Mapping[str, Fraction]] = None,
                    count: Optional[str] = None) -> dict:
    """Per seed and candidate (the 13): whether the candidate's frozen front points dominate CH-8's
    (validation, or held-out at the same frozen points when `heldout` maps ckpt hashes to top-1)."""
    c = count or screen.headline
    out = {}
    cands = ["F100"] + [sel.label_in_form(k, f) for k in sel.FUSED_BN_CODES for f in sel.FORMS]
    for lab in cands:
        out[lab] = {}
        for s in screen.seeds:
            def pts(label):
                return [(p.cost, p.top1 if heldout is None else heldout[p.ckpt_sha256]) for p in sel.frozen_points(screen.run(label, s), c)]
            out[lab][str(s)] = sel.dominates(pts(lab), pts(sel.CH8))
    return out


# ---------------------------------------------------------------------------------------------
# held-out tables of the screen (section 8.6 :984-992), after the freeze


def screen_heldout_tables(screen: sel.Screen, form: str, best: str, heldout: Mapping[str, Mapping],
                          f000_bitwise_equal_ch8, count: Optional[str] = None, jets: Optional[Mapping] = None,
                          manifest: Optional[Mapping] = None) -> dict:
    """Descriptive held-out tables for the 13 candidates at each g in D_screen (and cost-reach records
    at g outside it), on one count (the headline, the other count or NR).

    `heldout` maps ckpt_sha256 -> {"correct": int, "n": int}, the sealed reader's output at the
    frozen pre-trace checkpoints; `jets` optionally holds per-jet arrays {"labels": (n,), ckpt_sha256:
    {"correct": bool (n,), "scores": (n, 5)}} for the paired bootstrap and the background rejection.
    `manifest` is the section 8.5 freeze manifest; when given, every checkpoint used must be one of
    its frozen points."""
    if not isinstance(f000_bitwise_equal_ch8, bool):
        raise sel.UnsetDecision("f000_bitwise_equal_ch8 must be true or false (section 13 determinism record)")
    c = count or screen.headline
    ds = sel.d_screen(screen)
    if ds.status != "ok":
        return {"status": sel.INCOMPLETE, "reason": ds.reason}
    frozen = None
    if manifest is not None:
        frozen = set()
        for rows in manifest["grid_checkpoints"][c].values():
            for row in rows.values():
                frozen.update(v for v in row.values() if v != "NA")

    def top1(p):
        if frozen is not None and p.ckpt_sha256 not in frozen:
            raise sel.FreezeRefused("checkpoint %s is not a frozen point" % p.ckpt_sha256[:12])
        h = heldout[p.ckpt_sha256]
        return Fraction(int(h["correct"]), int(h["n"]))

    cands = ["F100"] + [sel.label_in_form(k, f) for k in sel.FUSED_BN_CODES for f in sel.FORMS]
    tables = {"label": sel.LABEL_SCREEN_HELDOUT, "split": "held-out", "count": c, "seeds": list(screen.seeds),
              "candidates": {}, "cost_reach": {}}
    for lab in cands + [sel.F000]:
        rows = {}
        for g in ds.allowances:
            rows[str(g)] = _gap_row(screen, lab, sel.CH8, g, c, top1, jets)
        tables["candidates"][lab] = rows
        reach = [g for g in sel.GRID if g not in ds.allowances and all(screen.point(lab, s, g, c) is not None for s in screen.seeds)]
        if reach:
            tables["cost_reach"][lab] = {"allowances": reach, "label": "cost-reach advantage (descriptive; outside D)"}
    bl = sel.label_in_form(best, form)
    tables["best_minus_f000"] = {str(g): _gap_row(screen, bl, sel.F000, g, c, top1, jets) for g in ds.allowances}
    tables["f000_minus_ch8_reading"] = (
        "the determinism record's runs are bitwise identical: F000 - CH-8 is expected to be 0, and any difference is "
        "an implementation finding" if f000_bitwise_equal_ch8 else
        "the determinism record's runs differ: F000 - CH-8 measures run-to-run divergence and is reported as the "
        "null-arm paired s_d")
    tables["dominance"] = dominance_table(screen, form, heldout={h: Fraction(int(v["correct"]), int(v["n"])) for h, v in heldout.items()},
                                          count=c)
    fv = sel.factorial_values(screen, form, ds.allowances, count=c,
                              heldout={h: Fraction(int(v["correct"]), int(v["n"])) for h, v in heldout.items()})
    tables["factorial"] = _factorial_json(sel.factorial(fv, screen.seeds, ds.allowances))
    return tables


def _gap_row(screen, lab, ref, g, count, top1, jets) -> dict:
    pts = [screen.point(lab, s, g, count) for s in screen.seeds]
    refs = [screen.point(ref, s, g, count) for s in screen.seeds]
    row = {"per_seed": [], "val_to_heldout_drop": []}
    for p, q in zip(pts, refs):
        if p is None or q is None:
            row["per_seed"].append(None)
            row["val_to_heldout_drop"].append(None if p is None else float(p.top1 - top1(p)))
            continue
        row["per_seed"].append(float(top1(p) - top1(q)))
        row["val_to_heldout_drop"].append(float(p.top1 - top1(p)))
    if any(x is None for x in row["per_seed"]):
        row["status"] = sel.COST_INFEASIBLE if any(p is None for p in pts) else "reference infeasible"
        return row
    row["status"] = "ok"
    row["interval"] = st.paired_interval(row["per_seed"], 0.95).as_dict()
    if jets is not None:
        a = np.stack([np.asarray(jets[p.ckpt_sha256]["correct"], dtype=bool) for p in pts])
        b = np.stack([np.asarray(jets[q.ckpt_sha256]["correct"], dtype=bool) for q in refs])
        row["bootstrap"] = st.paired_bootstrap(a, b)
        labels = np.asarray(jets["labels"])
        rej = {}
        for k in range(np.asarray(jets[pts[0].ckpt_sha256]["scores"]).shape[1]):
            ra = [st.background_rejection(jets[p.ckpt_sha256]["scores"], labels, k)["rejection"] for p in pts]
            rb = [st.background_rejection(jets[q.ckpt_sha256]["scores"], labels, k)["rejection"] for q in refs]
            d = [x - y for x, y in zip(ra, rb)]
            rej[str(k)] = {"arm": ra, "reference": rb,
                           "paired": st.paired_interval(d, 0.95).as_dict() if all(math.isfinite(x) for x in d) else {"status": "NA (infinite rejection)", "values": d}}
        row["background_rejection_eps_s_0.5"] = rej
    return row


CONTROL_REFERENCE = {"DS-HGQ": "CH-8", "GR": "F111", "BE": "F111"}  # section 8.9 (:1118)


def control_heldout_rows(screen: sel.Screen, form: str, controls: Mapping[str, Mapping[int, sel.Run]],
                         heldout: Mapping[str, Mapping], count: Optional[str] = None) -> dict:
    """The controls' descriptive held-out rows (section 8.6 :984, section 8.9). DS-HGQ against CH-8,
    GR and BE against the chosen-form F111, each on the section 8.1 front at each g of D_screen;
    FP32 and W8A8 at their whole-run selections (control_selection on the per-epoch record, which
    must name the saved checkpoint), each seed's held-out top-1 with W8A8's H and NR, outside every
    comparison (owner decision B3b-20)."""
    c = count or screen.headline
    ds = sel.d_screen(screen)
    out = {"label": sel.LABEL_SCREEN_HELDOUT + "; controls stay outside every verdict", "count": c, "controls": {}}
    if ds.status != "ok":
        out["status"] = sel.INCOMPLETE
        return out

    def t1(sha):
        h = heldout[sha]
        return Fraction(int(h["correct"]), int(h["n"]))

    for name, runs in controls.items():
        if name in CONTROL_REFERENCE:
            ref = CONTROL_REFERENCE[name]
            ref_lab = sel.CH8 if ref == sel.CH8 else sel.label_in_form(ref, form)
            rows = {}
            for g in ds.allowances:
                per, drop = [], []
                for s in screen.seeds:
                    p, q = runs[s].point(c, g), screen.point(ref_lab, s, g, c)
                    per.append(None if (p is None or q is None) else float(t1(p.ckpt_sha256) - t1(q.ckpt_sha256)))
                    drop.append(None if p is None else float(p.top1 - t1(p.ckpt_sha256)))
                row = {"reference": ref_lab, "per_seed": per, "val_to_heldout_drop": drop}
                if any(x is None for x in per):
                    row["status"] = sel.COST_INFEASIBLE
                else:
                    row["status"] = "ok"
                    row["interval"] = st.paired_interval(per, 0.95).as_dict()
                rows[str(g)] = row
            out["controls"][name] = rows
        elif name in ("FP32", "W8A8"):
            per = {}
            for s in screen.seeds:
                cs = sel.control_selection(runs[s].epochs)
                sha = cs.get("ckpt_sha256")
                if not sha:
                    raise S2Error("%s seed %d: the selected epoch names no saved checkpoint" % (name, s))
                row = {"epoch": cs["epoch"], "ckpt_sha256": sha, "heldout_top1": float(t1(sha)),
                       "val_top1": float(Fraction(cs["val_correct"], cs["val_n"]))}
                if name == "W8A8":
                    row.update({"H1": cs.get("H1"), "H2": cs.get("H2"), "NR": cs.get("NR")})
                per[str(s)] = row
            vals = [r["heldout_top1"] for r in per.values()]
            out["controls"][name] = {"per_seed": per, "mean": sum(vals) / len(vals),
                                     "sd": st.paired_interval(vals).sd if len(vals) >= 2 else None,
                                     "note": "selection over the whole run without Chang's filter; contextual only"}
        else:
            raise S2Error("unknown control %r" % name)
    out["status"] = "ok"
    return out


def pid_screen_comparison(pid_screen: sel.Screen, best_label: str, heldout: Optional[Mapping[str, Mapping]] = None) -> dict:
    """PID-BEST against PID-CH at their PID selections (section 8.6 :996), on validation, or on the
    held-out set when `heldout` is given; the section 8.8 filter outcomes named."""
    h = pid_screen.headline
    per_seed, outcomes = [], {}
    for s in pid_screen.seeds:
        a = sel.check_pid_summary(pid_screen.run(best_label, s), h)
        b = sel.check_pid_summary(pid_screen.run(sel.CH8, s), h)
        outcomes[str(s)] = {"PID-BEST": a["status"], "PID-CH": b["status"]}
        if a["status"] != "selected" or b["status"] != "selected":
            per_seed.append(None)
            continue
        if heldout is None:
            va = Fraction(a["point"]["val_correct"], a["point"]["val_n"])
            vb = Fraction(b["point"]["val_correct"], b["point"]["val_n"])
        else:
            ha, hb = heldout[a["point"]["ckpt_sha256"]], heldout[b["point"]["ckpt_sha256"]]
            va, vb = Fraction(ha["correct"], ha["n"]), Fraction(hb["correct"], hb["n"])
        per_seed.append(float(va - vb))
    out = {"split": "validation" if heldout is None else "held-out", "per_seed": per_seed, "outcomes": outcomes,
           "label": sel.LABEL_OPTIMISTIC if heldout is None else sel.LABEL_SCREEN_HELDOUT}
    if all(x is not None for x in per_seed) and len(per_seed) >= 2:
        out["interval"] = st.paired_interval(per_seed, 0.95).as_dict()
    return out
