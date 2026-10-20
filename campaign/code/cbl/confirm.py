"""M16 confirmation: C2, its strong form and labels, C3, C3r, C4, C6, the degenerate-layer labels
and the section 8.10 pull (PREFLIGHT-PLAN section 2 row M16; STUDY v0.8 sections 8.2, 8.5, 8.7,
8.8 "Attention use", 8.10; section 19.2 items 11 and 12).

STUDY means `review/snapshots/v0.8/STUDY.md` (sha256 290022fa...98ca8); line numbers refer to it
unless marked v0.9. STUDY v0.9 (`review/snapshots/v0.9/STUDY.md`, sha256 61e68527...89df4) is the
version of record; for this module it adds only the definitions of the descriptive labels
(v0.9 :1039-1042), which `descriptive_labels` implements as written there.

All contrasts are on the held-out test set at the frozen pre-trace checkpoints (:950, :1018). The
held-out values come in as an argument (the sealed reader's output after the section 8.5 freeze);
this module opens no file. The S4 seeds take no part in selection, so no candidate correction
enters (:777). Every number on PREFLIGHT data is an engineering reading, not a quotable result.
Owner decisions B3b-21 to B3b-26 are recorded in `preflight/gates/G53/<sha12>/PREFLIGHT-rows.md`.

Imports: the standard library, numpy and scipy (through cbl.stats) only.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Sequence

from cbl import select as sel
from cbl import stats as st

ABOVE_ON_GRID = "ABOVE ON GRID"
BELOW_ON_GRID = "BELOW ON GRID"
FLAT = "FLAT/UNRESOLVED"
ABOVE, BELOW = "ABOVE", "BELOW"
CROSSING, LOCAL_ADVANTAGE, LOCAL_DEFICIT = "CROSSING", "LOCAL ADVANTAGE", "LOCAL DEFICIT"
NO_ACCEPTED_LABEL = "NO ACCEPTED CHECKPOINT / ACCURACY FILTER FAILURE"
COMPLETE_ENDS = ("completed", "stopif")  # owner decision B3b-21
REFERENCE_8_10 = Fraction(663, 1000)  # 66.3 %, paper Table 1 row 1 (:1121)
LOG8 = math.log(8.0)  # 2.0794 (:1092)


class ConfirmError(ValueError):
    pass


def _heldout_top1(heldout: Mapping[str, Mapping], sha: str) -> Fraction:
    try:
        h = heldout[sha]
    except KeyError:
        raise ConfirmError("no held-out value for frozen checkpoint %s" % sha[:12]) from None
    return Fraction(int(h["correct"]), int(h["n"]))


def _runs_complete(runs: Mapping[int, sel.Run], seeds: Sequence[int]) -> List[str]:
    probs = []
    for s in seeds:
        if s not in runs:
            probs.append("seed %d missing" % s)
        elif runs[s].end not in COMPLETE_ENDS:
            probs.append("seed %d ended %s" % (s, runs[s].end))
    return probs


def contrast(best_runs: Mapping[int, sel.Run], ch8_runs: Mapping[int, sel.Run], heldout: Mapping[str, Mapping],
             count: str, g: int, seeds: Sequence[int]) -> dict:
    """d_s(g) = held-out top-1(BEST, s, g) - top-1(CH-8, s, g) at the step-envelope checkpoints (:1018)."""
    per, pairs, best_inf, ch_inf = [], [], [], []
    for s in seeds:
        p, q = best_runs[s].point(count, g), ch8_runs[s].point(count, g)
        if p is None:
            best_inf.append(s)
        if q is None:
            ch_inf.append(s)
        pairs.append((p.ckpt_sha256 if p else None, q.ckpt_sha256 if q else None))
        per.append(None if (p is None or q is None) else float(_heldout_top1(heldout, p.ckpt_sha256) - _heldout_top1(heldout, q.ckpt_sha256)))
    out = {"g": g, "count": count, "per_seed": per, "pairs": pairs}
    if best_inf:
        out["status"] = sel.COST_INFEASIBLE
        out["best_infeasible_seeds"] = best_inf
    elif ch_inf:
        out["status"] = sel.INCOMPLETE
        out["reason"] = "CH-8 infeasible in seeds %s" % ch_inf
    else:
        out["status"] = "ok"
    return out


def _with_interval(c: dict, level: float) -> dict:
    c = dict(c)
    if c["status"] == "ok":
        c["interval"] = st.paired_interval(c["per_seed"], level).as_dict()
    return c


def _iut(rows: Sequence[dict]) -> str:
    """Intersection-union verdict over a domain (:1022-1025)."""
    if any(r["status"] == sel.INCOMPLETE for r in rows):
        return sel.INCOMPLETE
    if rows and all(r["status"] == "ok" and r["interval"]["lower"] > 0 for r in rows):
        return ABOVE_ON_GRID
    if rows and all(r["status"] == sel.COST_INFEASIBLE or (r["status"] == "ok" and r["interval"]["upper"] < 0) for r in rows):
        return BELOW_ON_GRID
    return FLAT


def degenerate_labels(runs: Mapping[int, sel.Run], points: Mapping[int, Optional[str]]) -> List[str]:
    """'with degenerate <layer> (<route>)' for every degenerate layer among Q, K, V, O and the FFN
    contraction in the structure record of the checkpoint behind a verdict, in any seed (:1032)."""
    labs = set()
    for s, sha in points.items():
        if sha is None:
            continue
        side = runs[s].sidecars.get(sha)
        if side is None:
            raise ConfirmError("no front sidecar for checkpoint %s of seed %d" % (sha[:12], s))
        for layer, route in sorted(((side.get("structure_record") or {}).get("degenerate") or {}).items()):
            if route and layer in ("Q", "K", "V", "O", "FFN"):
                labs.add("with degenerate %s (%s)" % (layer, route))
    return sorted(labs)


def c2(design: sel.S4Design, claim: sel.ClaimDomain, dconf: sel.Domain, dc2: sel.Domain,
       best_runs: Mapping[int, sel.Run], ch8_runs: Mapping[int, sel.Run], heldout: Mapping[str, Mapping],
       headline: str, gate_failures: Sequence[str] = (), with_labels: bool = True) -> dict:
    """C2, the front, as an intersection-union test over D_C2 at level L (:1022-1028), with every
    g of D_conf reported, the allowances of D_conf below the band labelled outside the claim, the
    strong form over all of D_conf under (b), |D_C2| and its naming, the distinct (BEST, CH-8) pairs
    per seed, the cost-reach records at g outside D_conf, the degenerate-layer labels, and the same
    verdict under the non-headline count (accounting-dependent when they differ)."""
    h = sel.check_headline(headline)
    seeds, L = design.seeds, design.c2_level
    out = {"claim": "C2", "option": design.option, "level": L, "count": h, "seeds": list(seeds),
           "claim_domain": claim.as_dict(), "d_conf": dconf.as_dict(), "d_c2": dc2.as_dict()}
    probs = list(gate_failures) + ["BEST " + p for p in _runs_complete(best_runs, seeds)] + \
        ["CH-8 " + p for p in _runs_complete(ch8_runs, seeds)]
    if probs:
        out.update({"verdict": sel.INCOMPLETE, "reason": "; ".join(probs)})
        return out
    if dconf.status != "ok":
        out.update({"verdict": sel.INCOMPLETE, "reason": dconf.reason})
        return out
    if dc2.status != "ok":
        out.update({"verdict": dc2.status if dc2.status == sel.INCOMPLETE_NO_BAND else sel.INCOMPLETE, "reason": dc2.reason})
        return out
    per_g = {g: _with_interval(contrast(best_runs, ch8_runs, heldout, h, g, seeds), L) for g in dconf.allowances}
    claim_rows = [per_g[g] for g in dc2.allowances]
    verdict = _iut(claim_rows)
    out["verdict"] = verdict
    out["size"] = len(dc2.allowances)
    out["naming"] = "front result" if len(dc2.allowances) >= 2 else "single-allowance comparison"
    out["per_g"] = {str(g): dict(r, in_claim=g in dc2.allowances,
                                 note=None if g in dc2.allowances else "outside the claim") for g, r in per_g.items()}
    out["distinct_pairs_per_seed"] = {str(s): len({per_g[g]["pairs"][i] for g in dc2.allowances}) for i, s in enumerate(seeds)}
    if claim.base == "b":
        strong = _iut([per_g[g] for g in dconf.allowances])
        out["strong_form"] = "ABOVE ON ALL OF D_conf" if strong == ABOVE_ON_GRID else "not ABOVE ON ALL OF D_conf"
    reach = [g for g in sel.GRID if g not in dconf.allowances and all(best_runs[s].point(h, g) is not None for s in seeds)]
    out["cost_reach"] = {"allowances": reach, "label": "cost-reach advantage (descriptive; outside D)"} if reach else None
    if with_labels:
        best_pts = {s: {g: (best_runs[s].point(h, g).ckpt_sha256 if best_runs[s].point(h, g) else None) for g in dc2.allowances}
                    for s in seeds}
        ch_pts = {s: {g: (ch8_runs[s].point(h, g).ckpt_sha256 if ch8_runs[s].point(h, g) else None) for g in dc2.allowances}
                  for s in seeds}
        out["degenerate_labels"] = {str(g): degenerate_labels(best_runs, {s: best_pts[s][g] for s in seeds}) for g in dc2.allowances}
        out["ch8_degenerate_beside"] = {str(g): degenerate_labels(ch8_runs, {s: ch_pts[s][g] for s in seeds}) for g in dc2.allowances}
        out["verdict_labels"] = sorted({l for v in out["degenerate_labels"].values() for l in v})
        other = sel.other_count(h)
        alt = c2(design, claim, _domain_under(dconf, ch8_runs, other, seeds), _dc2_under(claim, dc2, ch8_runs, other, seeds),
                 best_runs, ch8_runs, heldout, other, gate_failures, with_labels=False)
        out["non_headline"] = {"count": other, "verdict": alt["verdict"]}
        out["accounting_dependent"] = alt["verdict"] != verdict
    return out


def _domain_under(dconf: sel.Domain, ch8_runs, count, seeds) -> sel.Domain:
    """D_conf is fixed under the headline; under the other count the same allowances are used where
    CH-8 is feasible in every seed (owner decision B3b-23)."""
    if dconf.status != "ok":
        return dconf
    dom = tuple(g for g in dconf.allowances if all(ch8_runs[s].point(count, g) is not None for s in seeds))
    return sel.Domain("ok", dom) if dom else sel.Domain(sel.INCOMPLETE, (), "D_conf empty under %s" % count)


def _dc2_under(claim, dc2, ch8_runs, count, seeds) -> sel.Domain:
    return _domain_under(dc2, ch8_runs, count, seeds) if dc2.status == "ok" else dc2


# ---------------------------------------------------------------------------------------------
# descriptive labels (STUDY v0.9 section 8.7 :1038-1042; owner decision B3b-22)


def _label(rows: Mapping[str, dict], whole_front: Optional[str]):
    pos = [k for k, r in rows.items() if r["status"] == "ok" and r["interval"]["lower"] > 0]
    neg = [k for k, r in rows.items() if r["status"] == sel.COST_INFEASIBLE or (r["status"] == "ok" and r["interval"]["upper"] < 0)]
    if pos and neg:
        label = CROSSING
    elif pos and whole_front != ABOVE_ON_GRID:
        label = LOCAL_ADVANTAGE
    elif neg and whole_front != BELOW_ON_GRID:
        label = LOCAL_DEFICIT
    else:
        label = None
    return pos, neg, label


def descriptive_labels(design: sel.S4Design, c2_record: dict, dconf: sel.Domain, best_runs, ch8_runs, heldout,
                       headline: str, pid: Optional[dict] = None) -> dict:
    """CROSSING, LOCAL ADVANTAGE and LOCAL DEFICIT (STUDY v0.9 :1038-1042). The intervals are the
    simultaneous Bonferroni intervals over 7 contrasts (six grid points plus the PID; options A and
    B) or 6 (option C); the labels are read on them at the grid points g of D_conf, where a COST
    INFEASIBLE g counts as a deficit:
      - CROSSING: at least one significantly positive and one significantly negative g;
      - LOCAL ADVANTAGE: some lower bounds > 0 and no upper bound < 0, but C2's ABOVE ON GRID fails;
      - LOCAL DEFICIT: some upper bounds < 0 and no lower bound > 0, but C2's BELOW ON GRID fails.
    A g is significantly positive when BEST is feasible in every seed and its lower bound is > 0.
    The PID contrast belongs to the Bonferroni family under A and B but is not read for the labels;
    its interval is printed beside. The same labels are computed under the non-headline count, on
    the allowances of D_conf where CH-8 is feasible under it (B3b-23), against C2's verdict under
    that count; a difference is labelled accounting-dependent."""
    h = sel.check_headline(headline)
    m = design.label_contrasts
    level = st.bonferroni_level(0.05, m)
    seeds = design.seeds
    out = {"contrasts": m, "level": level, "threshold_in_s_d": st.t_crit(len(seeds), level) / math.sqrt(len(seeds)),
           "read_at": "g in D_conf", "note": "descriptive, not a claim"}
    verdict = c2_record.get("verdict")
    if dconf.status != "ok":
        out.update({"label": None, "reason": "D_conf is %s" % dconf.status})
        return out
    if verdict == sel.INCOMPLETE:
        out.update({"label": None, "reason": "C2 is INCOMPLETE: %s" % c2_record.get("reason")})
        return out
    rows = {str(g): _with_interval(contrast(best_runs, ch8_runs, heldout, h, g, seeds), level) for g in dconf.allowances}
    pos, neg, label = _label(rows, verdict)
    out.update({"rows": rows, "positive": pos, "negative": neg, "label": label, "whole_front": verdict})
    if design.pid and pid is not None:
        row = {"status": pid["status"]}
        if pid["status"] == "ok":
            row["interval"] = st.paired_interval(pid["per_seed"], level).as_dict()
        out["pid_row"] = dict(row, note="in the Bonferroni family; not read for the labels (v0.9 :1039)")
    other = sel.other_count(h)
    dom_o = _domain_under(dconf, ch8_runs, other, seeds)
    v_o = (c2_record.get("non_headline") or {}).get("verdict", verdict)
    if dom_o.status == "ok":
        rows_o = {str(g): _with_interval(contrast(best_runs, ch8_runs, heldout, other, g, seeds), level) for g in dom_o.allowances}
        label_o = _label(rows_o, v_o)[2]
    else:
        label_o = None
    out["non_headline"] = {"count": other, "label": label_o, "whole_front": v_o}
    out["accounting_dependent"] = label_o != label
    return out


# ---------------------------------------------------------------------------------------------
# C3 and C3r


def pid_contrast(pid_best: Mapping[int, sel.Run], pid_ch: Mapping[int, sel.Run], heldout, headline: str,
                 seeds: Sequence[int]) -> dict:
    """d_s(PID) at the PID selections (:1018). PID-BEST's filter outcome in any seed names the
    result (COST INFEASIBLE first, then NO ACCEPTED CHECKPOINT / ACCURACY FILTER FAILURE); a missing
    PID-CH selection makes it INCOMPLETE (owner decision B3b-24)."""
    per, best_out, ch_out, pts = [], {}, {}, {}
    for s in seeds:
        a = sel.check_pid_summary(pid_best[s], headline)
        b = sel.check_pid_summary(pid_ch[s], headline)
        best_out[s], ch_out[s] = a["status"], b["status"]
        pts[s] = a["point"]["ckpt_sha256"] if a["point"] else None
        if a["point"] and b["point"]:
            per.append(float(_heldout_top1(heldout, a["point"]["ckpt_sha256"]) - _heldout_top1(heldout, b["point"]["ckpt_sha256"])))
        else:
            per.append(None)
    if any(v == sel.COST_INFEASIBLE for v in best_out.values()):
        status = sel.COST_INFEASIBLE
    elif any(v == sel.NO_ACCEPTED for v in best_out.values()):
        status = NO_ACCEPTED_LABEL
    elif any(v != "selected" for v in ch_out.values()):
        status = sel.INCOMPLETE
    else:
        status = "ok"
    return {"status": status, "per_seed": per, "pid_best": {str(k): v for k, v in best_out.items()},
            "pid_ch": {str(k): v for k, v in ch_out.items()}, "best_points": pts}


def c3(design: sel.S4Design, pid_best, pid_ch, heldout, headline: str, gate_failures: Sequence[str] = ()) -> dict:
    """C3, the PID at 350,000 H, options A and B only (:1029): ABOVE or BELOW from the paired
    interval at level L; the section 8.8 outcomes otherwise."""
    if not design.pid:
        return {"claim": "C3", "tested": False, "note": "not tested under option C; the screen's PID pair stays descriptive"}
    seeds = design.seeds
    probs = list(gate_failures) + ["PID-BEST " + p for p in _runs_complete(pid_best, seeds)] + \
        ["PID-CH " + p for p in _runs_complete(pid_ch, seeds)]
    if probs:
        return {"claim": "C3", "tested": True, "verdict": sel.INCOMPLETE, "reason": "; ".join(probs)}
    pc = pid_contrast(pid_best, pid_ch, heldout, headline, seeds)
    out = {"claim": "C3", "tested": True, "level": design.c3_level, "contrast": pc}
    if pc["status"] != "ok":
        out["verdict"] = pc["status"]
        return out
    iv = st.paired_interval(pc["per_seed"], design.c3_level)
    out["interval"] = iv.as_dict()
    out["verdict"] = ABOVE if iv.lower > 0 else (BELOW if iv.upper < 0 else FLAT)
    out["degenerate_labels"] = degenerate_labels(pid_best, pc["best_points"])
    return out


def c3r(design: sel.S4Design, best_runs, ch8_runs, heldout, headline: str, gate_failures: Sequence[str] = ()) -> dict:
    """C3r, the ramp front at 350,000 H, option C with option (c) only (:1030): ABOVE or BELOW from
    the paired interval at its level at g = 350,000, whether or not 350,000 is in D_C2; COST
    INFEASIBLE if BEST is infeasible there in any seed, INCOMPLETE if CH-8 is."""
    if not design.c3r:
        return {"claim": "C3r", "tested": False, "note": "tested only under option C with option (c)"}
    seeds = design.seeds
    probs = list(gate_failures) + ["BEST " + p for p in _runs_complete(best_runs, seeds)] + \
        ["CH-8 " + p for p in _runs_complete(ch8_runs, seeds)]
    if probs:
        return {"claim": "C3r", "tested": True, "verdict": sel.INCOMPLETE, "reason": "; ".join(probs)}
    c = contrast(best_runs, ch8_runs, heldout, headline, sel.PID_TARGET, seeds)
    out = {"claim": "C3r", "tested": True, "level": design.c3r_level, "contrast": c, "level_note": design.level_note}
    if c["status"] != "ok":
        out["verdict"] = c["status"]
        return out
    iv = st.paired_interval(c["per_seed"], design.c3r_level)
    out["interval"] = iv.as_dict()
    out["verdict"] = ABOVE if iv.lower > 0 else (BELOW if iv.upper < 0 else FLAT)
    out["degenerate_labels"] = degenerate_labels(best_runs, {s: c["pairs"][i][0] for i, s in enumerate(seeds)})
    return out


# ---------------------------------------------------------------------------------------------
# C4, firmware dominance (:1036-1050)

LUT_ADVANTAGE, NOT_LUT_ADVANTAGE, LUT_UNRESOLVED, HW_INCOMPLETE = (
    "LUT ADVANTAGE", "NOT A LUT ADVANTAGE", "LUT UNRESOLVED", "HARDWARE INCOMPLETE")


def c4_point(pairs: Sequence[Mapping], level: float, n: int) -> dict:
    """One allowance (or the PID point): the n deciding pairs' LUT_BEST - LUT_CH-8 at level L, with
    the paired post-trace held-out top-1 differences beside. Each pair: {lut_best, lut_ch8, part,
    stage, settings, status ("ok", "missing" or "capacity_failed"), allowance_exceeded_after_trace,
    post_trace_top1_best, post_trace_top1_ch8, function_changed}."""
    reasons = []
    if len(pairs) != n:
        reasons.append("%d of %d pairs" % (len(pairs), n))
    if any(p.get("status") != "ok" for p in pairs):
        reasons.append("a missing or capacity-failed pair")
    if len({(p.get("part"), p.get("stage"), str(p.get("settings"))) for p in pairs}) > 1:
        reasons.append("pairs on mixed parts, stages or settings")
    if any(p.get("allowance_exceeded_after_trace") for p in pairs):
        reasons.append("allowance exceeded after trace")
    per_pair = [("not a LUT advantage" if p.get("lut_best") is not None and p.get("lut_ch8") is not None
                 and p["lut_best"] >= p["lut_ch8"] else None) for p in pairs]
    acc = [p.get("post_trace_top1_best") - p.get("post_trace_top1_ch8") for p in pairs
           if p.get("post_trace_top1_best") is not None and p.get("post_trace_top1_ch8") is not None]
    out = {"per_pair_labels": per_pair, "function_changed": any(bool(p.get("function_changed")) for p in pairs)}
    if len(acc) == len(pairs) and len(acc) >= 2:
        out["post_trace_accuracy"] = st.paired_interval(acc, level).as_dict()
    if reasons:
        out.update({"label": HW_INCOMPLETE, "reasons": reasons})
        return out
    d = [float(p["lut_best"] - p["lut_ch8"]) for p in pairs]
    iv = st.paired_interval(d, level)
    out["lut"] = iv.as_dict()
    out["label"] = LUT_ADVANTAGE if iv.upper < 0 else (NOT_LUT_ADVANTAGE if iv.lower >= 0 else LUT_UNRESOLVED)
    return out


def c4(design: sel.S4Design, c2_record: dict, deciding: Mapping[str, Sequence[Mapping]], c3_record: Optional[dict] = None,
       c3r_record: Optional[dict] = None) -> dict:
    """C4, firmware dominance, evaluated only where C2 is ABOVE ON GRID (and at the PID point where
    C3 is ABOVE, or at 350,000 where C3r is ABOVE). SUPPORTED: every g of D_C2 a LUT ADVANTAGE and
    the post-trace accuracy lower bound > 0 at every g. BLOCKED: any g NOT A LUT ADVANTAGE, any
    post-trace accuracy upper bound < 0, or a changed exported function. Otherwise HARDWARE
    INCOMPLETE if any g is, else UNRESOLVED (owner decision B3b-25 for the order)."""
    L, n = design.c2_level, design.n
    out = {"claim": "C4", "level": L, "note": "stock export route; savings of a sign-then-scale export are not credited"}
    if c2_record.get("verdict") != ABOVE_ON_GRID:
        out["verdict"] = "NOT EVALUATED (C2 is not ABOVE ON GRID)"
    else:
        pts = {}
        for g in c2_record["d_c2"]["allowances"]:
            pts[str(g)] = c4_point(deciding.get(str(g), ()), L, n)
        out["per_g"] = pts
        blocked = any(p["label"] == NOT_LUT_ADVANTAGE or p["function_changed"]
                      or ("post_trace_accuracy" in p and p["post_trace_accuracy"]["upper"] < 0) for p in pts.values())
        supported = all(p["label"] == LUT_ADVANTAGE and "post_trace_accuracy" in p and p["post_trace_accuracy"]["lower"] > 0
                        for p in pts.values())
        if blocked:
            out["verdict"] = "BLOCKED"
        elif supported:
            out["verdict"] = "SUPPORTED"
        elif any(p["label"] == HW_INCOMPLETE for p in pts.values()):
            out["verdict"] = HW_INCOMPLETE
        else:
            out["verdict"] = "UNRESOLVED"
    pid_ok = (c3_record or {}).get("verdict") == ABOVE or (c3r_record or {}).get("verdict") == ABOVE
    if pid_ok:
        key = "PID" if (c3_record or {}).get("verdict") == ABOVE else str(sel.PID_TARGET)
        out["pid_point"] = dict(c4_point(deciding.get(key, ()), (c3_record or c3r_record).get("level", L), n), point=key)
    return out


# ---------------------------------------------------------------------------------------------
# C6, attention use (:1084-1100)


def c6_point(sidecars: Sequence[Mapping]) -> dict:
    """"Used" at one g from BEST's frozen checkpoints in every confirmation seed: the Delta_attn,
    Delta_static and Delta_unif two-sided 95 % intervals above zero, H_attn < log 8 in every seed,
    and no seed with a degenerate K, V or O (a Q equal to 0 everywhere counts as a degenerate K).
    A degenerate Q gives 'used, with input-independent queries', or 'used, as query-independent
    attention pooling' where Q_t is identical across t. A Delta with a nonzero downstream WRAP
    overflow count carries 'overflow-affected'."""
    n = len(sidecars)
    if n < 2:
        raise ConfirmError("C6 needs at least two seeds")
    diag = [s.get("attention_diagnostics") or {} for s in sidecars]
    struct = [s.get("structure_record") or {} for s in sidecars]
    out = {"intervals": {}, "overflow_affected": [], "conditions": {}}
    ok = True
    for key in ("delta_attn_mean", "delta_static", "delta_unif"):
        vals = [d.get(key) for d in diag]
        if any(v is None for v in vals):
            raise ConfirmError("attention_diagnostics.%s missing in a seed" % key)
        iv = st.paired_interval([float(v) for v in vals], 0.95)
        out["intervals"][key] = iv.as_dict()
        out["conditions"][key + " interval above 0"] = iv.lower > 0
        ok = ok and iv.lower > 0
        if any((d.get("overflow") or {}).get(key, 0) for d in diag):
            out["overflow_affected"].append(key)
    h = [d.get("H_attn") for d in diag]
    out["conditions"]["H_attn < log 8 in every seed"] = all(v is not None and v < LOG8 for v in h)
    ok = ok and out["conditions"]["H_attn < log 8 in every seed"]
    deg_kvo, deg_q, pooling = [], False, False
    for srec in struct:
        dg = srec.get("degenerate") or {}
        for layer in ("K", "V", "O"):
            if dg.get(layer):
                deg_kvo.append(layer)
        q_zero = srec.get("q_all_zero")
        q_zero = any(q_zero) if isinstance(q_zero, (list, tuple)) else bool(q_zero)
        if dg.get("Q") and q_zero:
            deg_kvo.append("K (Q equal to 0 everywhere)")
        elif dg.get("Q"):
            deg_q = True
            qi = srec.get("q_identical_across_t")
            if qi and (all(qi) if isinstance(qi, (list, tuple)) else bool(qi)):
                pooling = True
    out["conditions"]["no degenerate K, V or O"] = not deg_kvo
    ok = ok and not deg_kvo
    if not ok:
        out["label"] = "not demonstrated"
    elif pooling:
        out["label"] = "used, as query-independent attention pooling"
    elif deg_q:
        out["label"] = "used, with input-independent queries"
    else:
        out["label"] = "used"
    if out["overflow_affected"]:
        out["label"] += " (overflow-affected)"
    out["degenerate_kvo"] = sorted(set(deg_kvo))
    return out


def c6(per_g_sidecars: Mapping[int, Sequence[Mapping]], dc2: sel.Domain) -> dict:
    """C6 per g of D_conf and overall at every g of D_C2; a secondary claim outside the alpha of C2,
    C3 and C3r (:1097-1100). The overall qualifier is the strongest one among the g of D_C2
    (owner decision B3b-26)."""
    rows = {str(g): c6_point(sc) for g, sc in per_g_sidecars.items()}
    out = {"claim": "C6", "level": 0.95, "per_g": rows}
    if dc2.status != "ok":
        out["overall"] = "not demonstrated"
        return out
    labs = [rows[str(g)]["label"] for g in dc2.allowances if str(g) in rows]
    if len(labs) != len(dc2.allowances) or any(l.startswith("not demonstrated") for l in labs):
        out["overall"] = "not demonstrated"
    elif any("pooling" in l for l in labs):
        out["overall"] = "used, as query-independent attention pooling"
    elif any("input-independent" in l for l in labs):
        out["overall"] = "used, with input-independent queries"
    else:
        out["overall"] = "used"
    if out["overall"] != "not demonstrated" and any("overflow-affected" in l for l in labs):
        out["overall"] += " (overflow-affected)"
    return out


# ---------------------------------------------------------------------------------------------
# section 8.10 pull


def pull_8_10(ch8_runs: Mapping[int, sel.Run], heldout_by_path: Mapping[str, Mapping], stage: str,
              pid_ch_heldout: Optional[Sequence[float]] = None) -> dict:
    """The seed-mean held-out top-1 of CH-8 at its section 8.10 context checkpoint against 66.3 %:
    the difference, the CH-8 seed sd, the pull (difference / sd) and the trigger |difference| >
    max(2 sd, 1 percentage point) (:1120-1127). The pull is NA when the sd is 0."""
    vals, pts = [], {}
    for s, r in sorted(ch8_runs.items()):
        c = sel.context_checkpoint_8_10(r)
        if c["point"] is None:
            raise ConfirmError("seed %d has no released-callback point at training-count native <= 350,000" % s)
        h = heldout_by_path[c["point"]["path"]]
        vals.append(Fraction(int(h["correct"]), int(h["n"])))
        pts[str(s)] = c["point"]
    mean = sum(vals, Fraction(0)) / len(vals)
    sd = st.paired_interval([float(v) for v in vals]).sd if len(vals) >= 2 else None
    diff = mean - REFERENCE_8_10
    trig_bar = max(2 * sd, 0.01) if sd is not None else 0.01
    out = {"stage": stage, "label": sel.LABEL_CONTEXT_8_10, "reference": float(REFERENCE_8_10),
           "per_seed": [float(v) for v in vals], "seed_mean": float(mean), "sd": sd, "difference": float(diff),
           "pull": (float(diff) / sd) if sd else None, "trigger_bar": trig_bar,
           "triggered": abs(float(diff)) > trig_bar, "points": pts}
    if out["triggered"]:
        out["consequence"] = "VERIFY must give a quantitative cause before any 'beats Chang' wording"
    if pid_ch_heldout is not None:
        out["pid_ch_seed_mean"] = float(sum(pid_ch_heldout) / len(pid_ch_heldout))
    return out
