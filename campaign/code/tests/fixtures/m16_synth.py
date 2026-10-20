"""Synthetic sidecars for the M16 tests and gates G53-G56 (constructed cases, never real data).

Builds run configs (interfaces section 5), run summaries (section 3), per-epoch frozen records
(section 1) and front sidecars (section 2) from small specifications. Every value is invented to
exercise a rule; nothing here is a reading of any trained model.

The validation set size is the STUDY's 62,000 jets (section 4); top-1 is given as a count of
correct jets so that every tie is exact.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

GRID = (100_000, 200_000, 300_000, 350_000, 400_000, 450_000)
COUNTS = ("H1", "H2", "NR")
VAL_N = 62_000
FUSED = ("F001", "F010", "F011", "F101", "F110", "F111")


def sha(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def acc(top1: float) -> int:
    """A top-1 value as a count of correct jets out of 62,000."""
    return int(round(top1 * VAL_N))


def label_parts(label: str) -> Tuple[str, Optional[str], Optional[str]]:
    """(arm_id, config_code, form) of a panel label."""
    if label == "CH-8":
        return "CH-8", None, None
    if "-" in label:
        code, form = label.split("-")
        return code, code, form
    return label, label, None


def config(label: str, seed: int, stage: str = "S1", schedule: str = "ramp", arm_id: Optional[str] = None,
           control: Optional[str] = None) -> dict:
    a, code, form = label_parts(label)
    return {"schema_version": 1, "campaign": "chang-bitlinear-20261016", "arm_id": arm_id or a,
            "builder": "released" if a == "CH-8" else "lab", "config_code": code, "form": form, "control": control,
            "seed": seed, "stage": stage, "schedule": schedule, "headline": None, "claim_domain": None}


def front_from_envelope(run_id: str, env: Mapping[int, Optional[int]], cost_offset: int = 1_000,
                        costs: Optional[Mapping[int, int]] = None) -> list:
    """Front points whose step envelope at each grid g has `env[g]` correct jets (None: NA). The
    envelope must be non-decreasing in g where defined. The point that first reaches a value sits at
    cost g - cost_offset (or costs[g]) and an epoch increasing with cost."""
    pts, last = [], None
    for g in GRID:
        c = env.get(g)
        if c is None:
            if last is not None:
                raise ValueError("NA above a feasible allowance: an envelope is monotone in g")
            continue
        if last is not None and c < last:
            raise ValueError("envelope must be non-decreasing in g")
        if last is None or c > last:
            cost = costs[g] if costs and g in costs else g - cost_offset
            epoch = 7_000 - cost // 100
            pts.append({"epoch": epoch, "ckpt_sha256": sha(run_id, "ckpt", g, c), "val_correct": c, "val_n": VAL_N,
                        "cost": cost})
            last = c
    return pts


def envelope_of(front: Sequence[Mapping]) -> dict:
    out = {}
    for g in GRID:
        feas = [p for p in front if p["cost"] <= g]
        if not feas:
            out[str(g)] = "NA"
            continue
        p = min(feas, key=lambda q: (-q["val_correct"] / q["val_n"], q["cost"], q["epoch"], q["ckpt_sha256"]))
        out[str(g)] = dict(p, slack=g - p["cost"])
    return out


def summary(run_id: str, fronts: Mapping[str, Sequence[Mapping]], end: str = "completed",
            released_front: Sequence[Mapping] = (), pid_selection: Optional[Mapping] = None) -> dict:
    env = {c: envelope_of(fronts[c]) for c in COUNTS}
    fo = {c: {str(g): ("feasible" if env[c][str(g)] != "NA" else "COST INFEASIBLE") for g in GRID} for c in COUNTS}
    return {"run_id": run_id, "config_sha256": sha(run_id, "config"), "code_sha": "0" * 40, "end": end,
            "epochs_completed": 7_000, "front": {c: list(fronts[c]) for c in COUNTS}, "envelope": env,
            "filter_outcome": fo, "diagnostic_ckpts": {}, "released_front": list(released_front), "dwell": {},
            "pid_selection": pid_selection}


def run_records(label: str, seed: int, env: Mapping[int, Optional[int]], stage: str = "S1", schedule: str = "ramp",
                arm_id: Optional[str] = None, other_env: Optional[Mapping[str, Mapping[int, Optional[int]]]] = None,
                cost_offset: int = 1_000, released_front: Sequence[Mapping] = (), end: str = "completed"):
    """(config, summary) of one run whose headline envelope has `env` (g -> correct or None). The
    other counts copy it unless `other_env` gives their own envelopes."""
    cfg = config(label, seed, stage, schedule, arm_id=arm_id)
    rid = "%s_%s_s%d_%s_%s" % (cfg["arm_id"], cfg["form"] or "nf", seed, stage, schedule)
    fronts = {}
    for c in COUNTS:
        e = (other_env or {}).get(c, env)
        fronts[c] = front_from_envelope(rid + "/" + c, e, cost_offset=cost_offset)
    pid_sel = None
    s = summary(rid, fronts, end=end, released_front=released_front)
    if schedule == "pid":
        p = s["envelope"]["H1"][str(350_000)]
        pid_sel = None if p == "NA" else {"ckpt_sha256": p["ckpt_sha256"], "cost": p["cost"], "slack": p["slack"]}
        s["pid_selection"] = pid_sel
    return cfg, s


def flat_env(correct: int, lowest: int = 100_000) -> dict:
    """Feasible at every g >= lowest with the same top-1, NA below."""
    return {g: (correct if g >= lowest else None) for g in GRID}


def screen_spec(base: int = 40_300, gains: Optional[Mapping[str, Mapping[int, Sequence[int]]]] = None,
                lowest: Optional[Mapping[str, int]] = None, seed_shift: Sequence[int] = (0, 37, -21)) -> dict:
    """A full S1 screen spec: label -> seed -> {g: correct or None}. CH-8 has `base` + seed shift at
    every g; another arm has CH-8's value plus gains[label][g][seed index] (default 0); `lowest`
    makes an arm infeasible below that g."""
    labels = ["CH-8", "F000", "F100"] + ["%s-%s" % (c, f) for c in FUSED for f in ("FB", "BF")]
    gains = gains or {}
    lowest = lowest or {}
    out = {}
    for lab in labels:
        out[lab] = {}
        for si, s in enumerate((1, 2, 3)):
            env = {}
            for g in GRID:
                lo = lowest.get(lab, 100_000)
                if g < lo:
                    env[g] = None
                    continue
                gi = (gains.get(lab) or {}).get(g)
                env[g] = base + seed_shift[si] + (gi[si] if gi is not None else 0)
            # keep the envelope monotone: a later g never below an earlier one
            run_max = None
            for g in GRID:
                if env[g] is not None:
                    run_max = env[g] if run_max is None else max(run_max, env[g])
                    env[g] = run_max
            out[lab][s] = env
    return out


def build_runs(spec: Mapping[str, Mapping[int, Mapping[int, Optional[int]]]], stage: str = "S1", schedule: str = "ramp",
               arm_ids: Optional[Mapping[str, str]] = None):
    """[(config, summary)] for every (label, seed) of a spec."""
    out = []
    for lab, seeds in spec.items():
        for s, env in seeds.items():
            out.append(run_records(lab, s, env, stage=stage, schedule=schedule, arm_id=(arm_ids or {}).get(lab)))
    return out


def epochs_trace(values: Sequence[Tuple[int, float, int]]) -> list:
    """Per-epoch frozen records from (epoch, top1, H) triples; H1 = H2 = H, NR = H + 10."""
    return [{"run_id": "synthetic", "epoch": e, "step": e * 200, "val_correct": acc(a), "val_n": VAL_N,
             "val_top1": acc(a) / VAL_N, "H1": h, "H2": h, "NR": h + 10} for e, a, h in values]


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, indent=1) + "\n"


def freeze_example(sel, claim=None, s4=None):
    """A complete section 8.5 manifest built from a constructed screen and PID pair (`sel` is cbl.select)."""
    spec = screen_spec(gains={"F011-FB": {g: (12, 9, 15) for g in GRID}})
    scr = sel.Screen([sel.run_from_records(c, s) for c, s in build_runs(spec)], "H1")
    ds = sel.d_screen(scr)
    pan = sel.Panel.from_screen(scr, ds)
    vals = pan.values(scr)
    form = sel.form_choice(pan, vals)
    best = sel.best_choice(pan, vals, form["form"])
    claim = claim or sel.claim_domain_config({"base": "b", "band_rule": "default", "c3r": False})
    gs = sel.g_star(scr, ds, best["best_label"], claim.band_rule) if claim.base == "b" else None
    pid_spec = {"CH-8": spec["CH-8"], best["best_label"]: spec[best["best_label"]]}
    pid_runs = [sel.run_from_records(c, s) for c, s in build_runs(pid_spec, stage="S3", schedule="pid",
                                                                      arm_ids={"CH-8": "PID-CH", best["best_label"]: "PID-BEST"})]
    pid_screen = sel.Screen(pid_runs, "H1", schedule="pid")
    frozen = set()
    for c in sel.COUNTS:
        for lab in sel.screen_labels():
            for s in (1, 2, 3):
                for g in GRID:
                    p = scr.point(lab, s, g, c)
                    if p is not None:
                        frozen.add(p.ckpt_sha256)
    for r in pid_runs:
        p = sel.pid_selection(r, "H1")["point"]
        if p:
            frozen.add(p["ckpt_sha256"])
    structure = {h: {"degenerate": {"Q": None, "K": None, "V": None, "O": None, "FFN": None}} for h in sorted(frozen)}
    m = sel.build_freeze_manifest(study_sha256="290022fa41a726b8ae66ff302f7910a69c4c148ec27b15c2448cab83fb898ca8",
                                  code_sha="0" * 40, screen=scr, dscreen=ds, form=form, best=best, claim=claim, gstar=gs,
                                  pid_screen=pid_screen, controls_admitted=["FP32"],
                                  control_selections={"FP32": {"1": {"epoch": 10}}}, structure_records=structure,
                                  s4=s4 or {"funded": False})
    return m
