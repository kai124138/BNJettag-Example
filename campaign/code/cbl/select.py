"""M16 selection: records, domains, form rule, BEST, the claim band, PID and control selections,
collapse diagnostic, dominance and the factorial, and the section 8.5 freeze manifest (PREFLIGHT-PLAN
section 2 row M16; STUDY v0.8 sections 8.1-8.5, 8.6 (PID selection, dominance, factorial), 8.8,
8.9, 8.10 and 19.2 items 11-12).

STUDY means `review/snapshots/v0.8/STUDY.md` (sha256 290022fa...98ca8); line numbers refer to it.
STUDY v0.9 (`review/snapshots/v0.9/STUDY.md`, sha256 61e68527...89df4), the version of record,
changes nothing this module implements (v0.9 change log: section 6's power table and section 8.7's
label definitions only).
Inputs are the run summary (interfaces section 3), the run config (section 5), the per-epoch
frozen record (section 1) and the front sidecars (section 2) of `docs/interfaces.md`. Nothing here
reads held-out data; selection is on validation only (STUDY section 8, :862).

Decision quantities (feasibility, means compared with 0, ties) are exact: top-1 is the Fraction
`val_correct / val_n` (interfaces section 0), and means of Fractions stay Fractions. The rule
functions `form_choice` and `best_choice` accept any numbers that support +, -, / and comparison,
so the null selection reference (s2.py) calls the same functions on simulated floats.

Every number on PREFLIGHT data is an engineering reading, not a quotable result. Owner decisions
B3b-4 to B3b-7 and B3b-10 to B3b-16 are recorded in
`preflight/gates/G53/<sha12>/PREFLIGHT-rows.md`.

Imports: the standard library, numpy and scipy (through cbl.stats) only.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from cbl import stats as st

# ---------------------------------------------------------------------------------------------
# constants (interfaces section 4; STUDY sections 5, 8.1, 8.8)

GRID: Tuple[int, ...] = (100_000, 200_000, 300_000, 350_000, 400_000, 450_000)  # :886
PID_TARGET = 350_000  # :809, :994
FILTER_TOP1 = Fraction(1, 2)  # Chang's filter, val top-1 > 0.5 (:877)
FILTER_COST = 500_000  # and cost < 500,000 (:877)
COLLAPSE_DELTA = Fraction(1, 20)  # delta = 0.05 top-1 [OD-14] (:1065)
RAMP_LINEAR_END = 2_000  # run_train.py:90, the linear segment of the lambda ramp (:1060)
RAMP_LOG_END = 7_000
LR_CYCLE = 500  # run_train.py:93, cosine restarts every 500 epochs (:1063)
SQUEEZE_WINDOW = 1_000  # two LR cycles (:1063)
COUNTS = ("H1", "H2", "NR")
SCREEN_SEEDS = (1, 2, 3)
FORMS = ("FB", "BF")
FUSED_BN_CODES = ("F001", "F010", "F011", "F101", "F110", "F111")  # the six with a fused-BN binary site
CANDIDATE_CODES = ("F100",) + FUSED_BN_CODES  # the seven nonzero configurations (:920)
FACTORIAL_CODES = ("F000", "F001", "F010", "F011", "F100", "F101", "F110", "F111")  # 2^3, bits attention, FFN, head
GROUPS = ("attention", "ffn", "head")
CH8 = "CH-8"
F000 = "F000"
DETERMINISTIC_STAGES = ("S1", "S3", "S4")

LABEL_OPTIMISTIC = "optimistic: selected on these seeds' validation"  # :978
LABEL_SCREEN_HELDOUT = ("screen, descriptive; selected on the same seeds' validation, so the selected "
                        "configuration's gap is optimistic")  # :986
LABEL_CONTEXT_8_10 = "context: Chang's paper, his split and protocol (one head, attention floor, PID, XCU250)"  # :1122
TEXT_SELECTED = "the validation-selected configuration"  # :932

COST_INFEASIBLE = "COST INFEASIBLE"
NO_ACCEPTED = "NO ACCEPTED CHECKPOINT"  # with ACCURACY FILTER FAILURE (:1055)
INCOMPLETE = "INCOMPLETE"
INCOMPLETE_NO_BAND = "INCOMPLETE (no band)"


class SelectError(ValueError):
    """A record that does not meet the interfaces schema, or a rule input that is missing."""


class UnsetDecision(SelectError):
    """A decision input with no default (headline, claim domain, S4 option, the F000 pooled-row flag)
    is unset or not one of its allowed values: fail closed (interfaces section 5)."""


class FreezeRefused(PermissionError):
    """The held-out read is refused: no complete section 8.5 freeze manifest."""


def arm_label(arm_id: str, config_code: Optional[str], form: Optional[str]) -> str:
    """Panel label of an arm: CH-8, F000, F100, or <code>-<form> for a fused-BN configuration."""
    if arm_id in ("CH-8", "CH-8-C", "PID-CH", "PID-CH-C"):
        return CH8
    if config_code is None:
        return arm_id
    if config_code in FUSED_BN_CODES:
        if form not in FORMS:
            raise SelectError("configuration %s needs a form, got %r" % (config_code, form))
        return "%s-%s" % (config_code, form)
    if form is not None:
        raise SelectError("configuration %s has no fused-BN site; form must be null" % config_code)
    return config_code


def code_of(label: str) -> str:
    return label.split("-")[0] if label != CH8 else CH8


def bits_of(code: str) -> Tuple[int, int, int]:
    """(attention, FFN, head) bits of F<abc>; 1 is binary (STUDY section 5 "Codes", :656)."""
    if len(code) != 4 or code[0] != "F" or any(c not in "01" for c in code[1:]):
        raise SelectError("not a configuration code: %r" % code)
    return int(code[1]), int(code[2]), int(code[3])


# ---------------------------------------------------------------------------------------------
# records


@dataclass(frozen=True)
class Point:
    """A front checkpoint as the run summary lists it (interfaces section 3 `front`)."""

    epoch: int
    ckpt_sha256: str
    correct: int
    n: int
    cost: int

    @property
    def top1(self) -> Fraction:
        return Fraction(self.correct, self.n)

    def best_key(self):
        """Highest top-1, then lower cost, then earlier epoch, then checkpoint hash (STUDY :879)."""
        return (-self.top1, self.cost, self.epoch, self.ckpt_sha256)

    def passes_filter(self) -> bool:
        return self.top1 > FILTER_TOP1 and self.cost < FILTER_COST

    def as_dict(self) -> dict:
        return {"epoch": self.epoch, "ckpt_sha256": self.ckpt_sha256, "val_correct": self.correct,
                "val_n": self.n, "cost": self.cost}


def _int(x, what) -> int:
    if isinstance(x, bool) or not isinstance(x, int):
        raise SelectError("%s must be an integer, got %r" % (what, x))
    return x


def _point(d: Mapping, what: str) -> Point:
    for k in ("epoch", "ckpt_sha256", "val_correct", "val_n", "cost"):
        if k not in d:
            raise SelectError("%s lacks %r" % (what, k))
    p = Point(_int(d["epoch"], what + ".epoch"), str(d["ckpt_sha256"]), _int(d["val_correct"], what + ".val_correct"),
              _int(d["val_n"], what + ".val_n"), _int(d["cost"], what + ".cost"))
    if p.n <= 0 or not 0 <= p.correct <= p.n or p.cost < 0 or p.epoch < 0:
        raise SelectError("%s has an impossible value: %r" % (what, d))
    if len(p.ckpt_sha256) != 64 or any(c not in "0123456789abcdef" for c in p.ckpt_sha256):
        raise SelectError("%s.ckpt_sha256 is not a lowercase sha256 hex string" % what)
    return p


def step_envelope(points: Iterable[Point], g: int) -> Optional[Point]:
    """The front checkpoint with the highest validation top-1 at cost <= g under the section 8.1 ties,
    or None (NA, never zero) (STUDY :886-889)."""
    feas = [p for p in points if p.cost <= g]
    return min(feas, key=Point.best_key) if feas else None


@dataclass(frozen=True)
class Run:
    """One run: identity from its config, fronts and envelopes from its summary."""

    run_id: str
    arm_id: str
    config_code: Optional[str]
    form: Optional[str]
    seed: int
    stage: str
    schedule: str
    control: Optional[str]
    end: str
    front: Mapping[str, Tuple[Point, ...]]
    envelope: Mapping[str, Mapping[int, Optional[Point]]]
    filter_outcome: Mapping[str, Mapping[int, dict]]
    released_front: Tuple[dict, ...]
    pid_selection: Optional[dict]
    epochs: Tuple[dict, ...] = ()
    sidecars: Mapping[str, dict] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return arm_label(self.arm_id, self.config_code, self.form)

    def point(self, count: str, g: int) -> Optional[Point]:
        return self.envelope[count][g]


RUN_ENDS = ("completed", "stopif", "integrity_stop", "budget_stop", "failed")


def run_from_records(config: Mapping, summary: Mapping, epochs: Optional[Sequence[Mapping]] = None,
                     sidecars: Optional[Mapping[str, Mapping]] = None) -> Run:
    """Build a Run from the config (interfaces section 5) and the run summary (section 3), with the
    per-epoch records (section 1) and the front sidecars (section 2) when given.

    Checks: the summary names the config's sha256 if the config carries one; every count has a
    front and an envelope over the whole grid; every front point passes Chang's filter; every
    envelope entry is "NA" or a point, and equals the step envelope recomputed from the front
    (owner decision B3b-4), with slack = g - cost."""
    for k in ("arm_id", "seed", "stage", "schedule"):
        if k not in config:
            raise SelectError("config lacks %r" % k)
    for k in ("run_id", "end", "front", "envelope"):
        if k not in summary:
            raise SelectError("run summary lacks %r" % k)
    if summary["end"] not in RUN_ENDS:
        raise SelectError("run summary end %r is not one of %s" % (summary["end"], RUN_ENDS))
    rid = str(summary["run_id"])
    front, env, fo = {}, {}, {}
    for c in COUNTS:
        if c not in summary["front"] or c not in summary["envelope"]:
            raise SelectError("%s: front or envelope lacks count %s" % (rid, c))
        pts = tuple(_point(p, "%s.front.%s[%d]" % (rid, c, i)) for i, p in enumerate(summary["front"][c]))
        for p in pts:
            if not p.passes_filter():
                raise SelectError("%s: front point at epoch %d fails Chang's filter (top-1 > 0.5, cost < 500,000)"
                                  % (rid, p.epoch))
        front[c] = pts
        env[c] = {}
        raw = summary["envelope"][c]
        for g in GRID:
            e = raw.get(str(g), raw.get(g)) if isinstance(raw, Mapping) else None
            if e is None:
                raise SelectError("%s: envelope %s lacks g = %d" % (rid, c, g))
            want = step_envelope(pts, g)
            if e == "NA":
                if want is not None:
                    raise SelectError("%s: envelope %s at %d is NA but the front has a point at cost <= g" % (rid, c, g))
                env[c][g] = None
                continue
            if e == 0 or not isinstance(e, Mapping):
                raise SelectError("%s: envelope %s at %d must be a point or the string NA, never %r" % (rid, c, g, e))
            p = _point(e, "%s.envelope.%s.%d" % (rid, c, g))
            if want is None or p != want:
                raise SelectError("%s: envelope %s at %d differs from the step envelope of the front" % (rid, c, g))
            if "slack" in e and e["slack"] != g - p.cost:
                raise SelectError("%s: envelope %s at %d has slack %r, not g - cost" % (rid, c, g, e["slack"]))
            env[c][g] = p
        fo_raw = (summary.get("filter_outcome") or {}).get(c) or {}
        fo[c] = {}
        for g in GRID:
            v = fo_raw.get(str(g), fo_raw.get(g)) if isinstance(fo_raw, Mapping) else None
            fo[c][g] = dict(v) if isinstance(v, Mapping) else ({"outcome": v} if v is not None else {})
    run = Run(run_id=rid, arm_id=str(config["arm_id"]), config_code=config.get("config_code"), form=config.get("form"),
              seed=_int(config["seed"], "config.seed"), stage=str(config["stage"]), schedule=str(config["schedule"]),
              control=config.get("control"), end=str(summary["end"]), front=front, envelope=env, filter_outcome=fo,
              released_front=tuple(summary.get("released_front") or ()), pid_selection=summary.get("pid_selection"),
              epochs=tuple(epochs or ()), sidecars=dict(sidecars or {}))
    run.label  # validates code and form
    return run


def check_pid_summary(run: Run, headline: str) -> dict:
    """Recompute the PID selection from the front (section 8.6) and require the run summary's
    `pid_selection` (writer M9, interfaces section 3) to name the same checkpoint: null, or an
    object without a checkpoint, only when the rule selects none. Returns the recomputed selection;
    every M16 consumer of a PID selection goes through here."""
    mine = pid_selection(run, headline)
    theirs = run.pid_selection
    want = mine["point"]["ckpt_sha256"] if mine["status"] == "selected" else None
    got = theirs.get("ckpt_sha256") if isinstance(theirs, Mapping) else None
    if theirs is not None and not isinstance(theirs, Mapping):
        raise SelectError("%s: pid_selection in the summary is neither null nor an object" % run.run_id)
    if got != want:
        raise SelectError("%s: pid_selection in the summary (%r) differs from the section 8.6 rule (%r)"
                          % (run.run_id, got, want))
    return mine


def check_headline(headline) -> str:
    """The headline count, H1 or H2, with no default (interfaces section 5)."""
    if headline not in ("H1", "H2"):
        raise UnsetDecision("headline must be H1 or H2, got %r (fixed by section 3.4 before any training result)" % (headline,))
    return headline


def other_count(headline: str) -> str:
    return "H2" if check_headline(headline) == "H1" else "H1"


# ---------------------------------------------------------------------------------------------
# decision inputs that fail closed (STUDY section 19.2 items 11 and 12)


BAND_RULES = ("default", "strict_sd", "interval")  # section 8.2 (:897-906); section 19.3 (i) and (ii)
C3R_LEVELS = ("bonferroni", "nested")  # section 8.7 (:1021); section 19.2 item 12 (c)


@dataclass(frozen=True)
class ClaimDomain:
    base: str  # "a": D_C2 = D_conf; "b": D_C2 = D_band
    band_rule: Optional[str]  # under "b": default, strict_sd or interval
    c3r: bool  # option (c); acts only under S4 option C
    c3r_level: Optional[str]  # under c3r: bonferroni (pre-registered) or nested

    def as_dict(self) -> dict:
        return {"base": self.base, "band_rule": self.band_rule, "c3r": self.c3r, "c3r_level": self.c3r_level}


def claim_domain_config(cfg) -> ClaimDomain:
    """Kai's claim-domain choice (section 19.2 item 12), as the config field `claim_domain`
    (interfaces section 5, extended in section 10): {base: "a" | "b", band_rule (required under b):
    "default" | "strict_sd" | "interval", c3r: bool, c3r_level (required when c3r):
    "bonferroni" | "nested"}. Every field has no default; an unset or unknown value raises."""
    if cfg is None:
        raise UnsetDecision("claim_domain is unset: Kai decides it before any launch (section 19.2 item 12)")
    if not isinstance(cfg, Mapping):
        raise UnsetDecision("claim_domain must be an object, got %r" % (cfg,))
    base = cfg.get("base")
    if base not in ("a", "b"):
        raise UnsetDecision("claim_domain.base must be 'a' or 'b', got %r" % (base,))
    c3r = cfg.get("c3r")
    if not isinstance(c3r, bool):
        raise UnsetDecision("claim_domain.c3r must be true or false, got %r" % (c3r,))
    band_rule = cfg.get("band_rule")
    if base == "b":
        if band_rule not in BAND_RULES:
            raise UnsetDecision("claim_domain.band_rule must be one of %s under base b, got %r" % (BAND_RULES, band_rule))
    elif band_rule is not None:
        raise UnsetDecision("claim_domain.band_rule applies only under base b")
    c3r_level = cfg.get("c3r_level")
    if c3r:
        if c3r_level not in C3R_LEVELS:
            raise UnsetDecision("claim_domain.c3r_level must be one of %s when c3r is true, got %r" % (C3R_LEVELS, c3r_level))
    elif c3r_level is not None:
        raise UnsetDecision("claim_domain.c3r_level applies only when c3r is true")
    extra = set(cfg) - {"base", "band_rule", "c3r", "c3r_level"}
    if extra:
        raise UnsetDecision("claim_domain has unknown fields %s" % sorted(extra))
    return ClaimDomain(base, band_rule, c3r, c3r_level)


@dataclass(frozen=True)
class S4Design:
    """One S4 option with the claim domain (STUDY section 8.7 table, :1007-1011)."""

    option: str
    seeds: Tuple[int, ...]
    n: int
    pid: bool  # C3 tested (A, B)
    c3r: bool  # C3r tested (C with option (c))
    c2_level: float  # L
    c3_level: Optional[float]
    c3r_level: Optional[float]
    label_contrasts: int  # 7 (A, B) or 6 (C)
    level_note: str

    def as_dict(self) -> dict:
        return dict(self.__dict__, seeds=list(self.seeds))


def s4_design(option, claim: ClaimDomain, contains_350k: Optional[bool] = None) -> S4Design:
    """The levels of one S4 option under the claim domain. The nested C3r variant needs to know
    whether 350,000 is in D_C2 (fixed on validation before the held-out read, :1021); with
    `contains_350k` None it raises."""
    if option not in ("A", "B", "C"):
        raise UnsetDecision("s4_option must be A, B or C, got %r (Kai decides after S2, section 19.2 item 11)" % (option,))
    if not isinstance(claim, ClaimDomain):
        raise UnsetDecision("a ClaimDomain from claim_domain_config is required")
    if option == "A":
        return S4Design("A", tuple(range(4, 9)), 5, True, False, 0.975, 0.975, None, 7,
                        "C2 and C3 share alpha = 0.05 (Bonferroni over two hypotheses)")
    if option == "B":
        return S4Design("B", tuple(range(4, 12)), 8, True, False, 0.975, 0.975, None, 7,
                        "C2 and C3 share alpha = 0.05 (Bonferroni over two hypotheses)")
    seeds = tuple(range(4, 12))
    if not claim.c3r:
        return S4Design("C", seeds, 8, False, False, 0.95, None, None, 6, "C2 alone at alpha = 0.05")
    if claim.c3r_level == "bonferroni":
        return S4Design("C", seeds, 8, False, True, 0.975, None, 0.975, 6,
                        "C2 and C3r share alpha = 0.05 (Bonferroni, pre-registered default)")
    if contains_350k is None:
        raise UnsetDecision("the nested C3r variant needs whether 350,000 is in D_C2")
    if contains_350k:
        return S4Design("C", seeds, 8, False, True, 0.95, None, 0.95, 6,
                        "nested: 350,000 in D_C2, so C3r's null implies C2's; both at C2's own level")
    return S4Design("C", seeds, 8, False, True, 0.975, None, 0.975, 6,
                    "nested variant chosen, but 350,000 is not in D_C2: the Bonferroni split returns")


# options for which S2 prints thresholds and powers (STUDY :963: A, B, C alone, C with (c))
S2_OPTIONS = (
    ("A", 5, 0.975),
    ("B", 8, 0.975),
    ("C alone", 8, 0.95),
    ("C with (c)", 8, 0.975),
)


# ---------------------------------------------------------------------------------------------
# the screen


class Screen:
    """The runs of one stage of the screen, indexed by (panel label, seed), with the headline."""

    def __init__(self, runs: Iterable[Run], headline: str, seeds: Sequence[int] = SCREEN_SEEDS,
                 schedule: str = "ramp"):
        self.headline = check_headline(headline)
        self.seeds = tuple(seeds)
        self.runs: Dict[Tuple[str, int], Run] = {}
        for r in runs:
            if r.schedule != schedule or r.control is not None:
                continue
            key = (r.label, r.seed)
            if key in self.runs:
                raise SelectError("two %s runs for %s seed %d" % (schedule, r.label, r.seed))
            self.runs[key] = r

    def run(self, label: str, seed: int) -> Run:
        try:
            return self.runs[(label, seed)]
        except KeyError:
            raise SelectError("no run for %s seed %d (failed runs stay in the ledger; a missing run is an input error)"
                              % (label, seed)) from None

    def point(self, label: str, seed: int, g: int, count: Optional[str] = None) -> Optional[Point]:
        return self.run(label, seed).point(count or self.headline, g)

    def require(self, labels: Iterable[str]):
        missing = [(l, s) for l in labels for s in self.seeds if (l, s) not in self.runs]
        if missing:
            raise SelectError("screen runs missing: %s" % missing)


def screen_labels(form: Optional[str] = None) -> Tuple[str, ...]:
    """The 45 S1 ramp arms (CH-8, F000, F100, six configurations in both forms), or with `form` the
    chosen-form factorial cells F000-F111 and CH-8."""
    if form is None:
        return (CH8, F000, "F100") + tuple("%s-%s" % (c, f) for c in FUSED_BN_CODES for f in FORMS)
    return (CH8,) + tuple(c if c not in FUSED_BN_CODES else "%s-%s" % (c, form) for c in FACTORIAL_CODES)


def label_in_form(code: str, form: str) -> str:
    return "%s-%s" % (code, form) if code in FUSED_BN_CODES else code


# ---------------------------------------------------------------------------------------------
# domains (section 8.2)


@dataclass(frozen=True)
class Domain:
    status: str  # "ok" or INCOMPLETE
    allowances: Tuple[int, ...]
    reason: Optional[str] = None

    def as_dict(self) -> dict:
        return {"status": self.status, "allowances": list(self.allowances), "reason": self.reason}


def d_screen(screen: Screen) -> Domain:
    """D_screen = {g in G : CH-8 has a feasible front checkpoint at g in each of seeds 1-3}, on
    validation from CH-8 alone (:893-894). Empty: the screen outputs are INCOMPLETE (:896)."""
    screen.require([CH8])
    dom = tuple(g for g in GRID if all(screen.point(CH8, s, g) is not None for s in screen.seeds))
    if not dom:
        return Domain(INCOMPLETE, (), "D_screen is empty: no form, no BEST; Kai decides before any S3 or S4 launch")
    return Domain("ok", dom)


def d_conf(dscreen: Domain, ch8_conf_runs: Mapping[int, Run], headline: str, seeds: Sequence[int]) -> Domain:
    """D_conf = D_screen intersected with {g : CH-8 feasible in every confirmation seed} (:895).
    Empty: C2 is INCOMPLETE (:896)."""
    check_headline(headline)
    if dscreen.status != "ok":
        return Domain(INCOMPLETE, (), "D_screen is empty")
    missing = [s for s in seeds if s not in ch8_conf_runs]
    if missing:
        return Domain(INCOMPLETE, (), "CH-8 confirmation runs missing for seeds %s" % missing)
    dom = tuple(g for g in dscreen.allowances if all(ch8_conf_runs[s].point(headline, g) is not None for s in seeds))
    if not dom:
        return Domain(INCOMPLETE, (), "D_conf is empty: C2 is INCOMPLETE")
    return Domain("ok", dom)


# ---------------------------------------------------------------------------------------------
# the panel: fixed feasibility and cost, values passed separately


class Panel:
    """Structure of the screen on D_screen: which (label, seed, g) cells are feasible and their
    actual cost. Values are passed as a sequence indexed by `index`, so that the same rule functions
    run on exact validation values and on simulated values (null selection reference)."""

    def __init__(self, labels: Sequence[str], seeds: Sequence[int], domain: Sequence[int],
                 feasible: Mapping[Tuple[str, int, int], bool], cost: Mapping[Tuple[str, int, int], Optional[int]]):
        self.labels, self.seeds, self.domain = tuple(labels), tuple(seeds), tuple(domain)
        if CH8 not in self.labels:
            raise SelectError("a panel needs CH-8")
        self.feasible = {(l, s, g): bool(feasible[(l, s, g)]) for l in self.labels for s in self.seeds for g in self.domain}
        for s in self.seeds:
            for g in self.domain:
                if not self.feasible[(CH8, s, g)]:
                    raise SelectError("CH-8 is infeasible at (%d, %d) inside D_screen" % (s, g))
        self.cost = {k: (cost[k] if self.feasible[k] else None) for k in self.feasible}
        self.index = {}
        for k in sorted(self.feasible, key=lambda k: (self.labels.index(k[0]), self.seeds.index(k[1]), self.domain.index(k[2]))):
            self.index[k] = len(self.index)
        self.cells = {l: tuple((s, g) for s in self.seeds for g in self.domain if self.feasible[(l, s, g)]) for l in self.labels}
        self.forms_present = tuple(f for f in FORMS if any(l.endswith("-" + f) for l in self.labels))
        self.fused_present = tuple(c for c in FUSED_BN_CODES if any(l.startswith(c + "-") for l in self.labels))

    @classmethod
    def from_screen(cls, screen: Screen, domain: Domain, labels: Optional[Sequence[str]] = None,
                    count: Optional[str] = None) -> "Panel":
        if domain.status != "ok":
            raise SelectError("no panel on an INCOMPLETE domain")
        labels = tuple(labels or screen_labels())
        screen.require(labels)
        c = count or screen.headline
        feas, cost = {}, {}
        for l in labels:
            for s in screen.seeds:
                for g in domain.allowances:
                    p = screen.point(l, s, g, c)
                    feas[(l, s, g)] = p is not None
                    cost[(l, s, g)] = p.cost if p is not None else None
        return cls(labels, screen.seeds, domain.allowances, feas, cost)

    def values(self, screen: Screen, count: Optional[str] = None) -> List[Optional[Fraction]]:
        """Exact validation top-1 at every feasible cell (None elsewhere), in `index` order."""
        c = count or screen.headline
        vals: List[Optional[Fraction]] = [None] * len(self.index)
        for (l, s, g), i in self.index.items():
            p = screen.point(l, s, g, c)
            vals[i] = p.top1 if p is not None else None
        return vals

    def coverage(self, label: str, seed: Optional[int] = None) -> int:
        return sum(1 for (s, g) in self.cells[label] if seed is None or s == seed)

    def feasible_everywhere(self, label: str) -> bool:
        return len(self.cells[label]) == len(self.seeds) * len(self.domain)


def form_choice(panel: Panel, vals: Sequence) -> dict:
    """Section 8.3 (:911-916). 1: more configurations feasible in all three seeds at every g in
    D_screen wins. 2: on equal counts (including zero), Delta_form = mean over cells (configuration,
    seed, g) feasible in both forms of top-1(FB) - top-1(BF); > 0 chooses FB, < 0 BF. 3: Delta_form
    = 0 or no common cell chooses FB. Delta_form is computed and logged whatever step decides."""
    counts = {f: sum(1 for c in panel.fused_present if label_in_form(c, f) in panel.cells
                     and panel.feasible_everywhere(label_in_form(c, f))) for f in FORMS}
    common = []
    for c in panel.fused_present:
        fb, bf = label_in_form(c, "FB"), label_in_form(c, "BF")
        if fb not in panel.cells or bf not in panel.cells:
            continue
        for s in panel.seeds:
            for g in panel.domain:
                if panel.feasible[(fb, s, g)] and panel.feasible[(bf, s, g)]:
                    common.append(vals[panel.index[(fb, s, g)]] - vals[panel.index[(bf, s, g)]])
    delta = (sum(common[1:], common[0]) / len(common)) if common else None
    if counts["FB"] != counts["BF"]:
        form, step = ("FB" if counts["FB"] > counts["BF"] else "BF"), 1
    elif delta is None:
        form, step = "FB", 3
    elif delta > 0:
        form, step = "FB", 2
    elif delta < 0:
        form, step = "BF", 2
    else:
        form, step = "FB", 3
    return {"form": form, "step": step, "counts": counts, "delta_form": delta, "common_cells": len(common)}


def _rank_key(entry):
    cov, mean, mean_h, code = entry["coverage"], entry["mean_gain"], entry["mean_H"], entry["code"]
    # coverage descending; mean gain descending (None last); mean actual H ascending (None last); code ascending
    return (-cov, (0, -mean) if mean is not None else (1, 0), (0, mean_h) if mean_h is not None else (1, 0), code)


def candidate_entries(panel: Panel, vals: Sequence, form: str, seed: Optional[int] = None) -> List[dict]:
    """Per candidate in `form`: coverage, unweighted mean of top-1 minus the paired CH-8 value over
    its feasible cells, and mean actual H over them (section 8.4, :920-924), restricted to `seed`
    when given."""
    out = []
    for code in CANDIDATE_CODES:
        lab = label_in_form(code, form)
        if lab not in panel.cells:
            continue
        cells = [(s, g) for (s, g) in panel.cells[lab] if seed is None or s == seed]
        gains = [vals[panel.index[(lab, s, g)]] - vals[panel.index[(CH8, s, g)]] for (s, g) in cells]
        mean = (sum(gains[1:], gains[0]) / len(gains)) if gains else None
        mean_h = Fraction(sum(panel.cost[(lab, s, g)] for (s, g) in cells), len(cells)) if cells else None
        out.append({"code": code, "label": lab, "coverage": len(cells), "mean_gain": mean, "mean_H": mean_h})
    return out


def best_choice(panel: Panel, vals: Sequence, form: str) -> dict:
    """BEST among the seven nonzero configurations in the chosen form (F100 plus six): coverage
    first, then the mean gain over CH-8, then lower mean actual H, then configuration id ascending
    (owner decision B3b-5). F000 is excluded; BEST may score negative (:918-924)."""
    if form not in FORMS:
        raise SelectError("form must be FB or BF")
    entries = candidate_entries(panel, vals, form)
    if not entries:
        raise SelectError("no candidate configuration in the panel")
    ranked = sorted(entries, key=_rank_key)
    return {"best": ranked[0]["code"], "best_label": ranked[0]["label"], "ranking": ranked,
            "best_has_no_feasible_cell": ranked[0]["coverage"] == 0}


def per_g_gains(screen: Screen, label: str, domain: Sequence[int], count: Optional[str] = None) -> Dict[int, dict]:
    """At each g: the paired validation gains of `label` over CH-8 in each seed (None where
    infeasible), feasibility in all seeds, the exact mean over feasible seeds and the sd (ddof = 1)
    when all seeds are feasible."""
    c = count or screen.headline
    out = {}
    for g in domain:
        gains = []
        for s in screen.seeds:
            p, q = screen.point(label, s, g, c), screen.point(CH8, s, g, c)
            gains.append(None if (p is None or q is None) else p.top1 - q.top1)
        feas = [x for x in gains if x is not None]
        all_feas = len(feas) == len(gains)
        mean = st.exact_mean(feas) if feas else None
        var = None
        if all_feas and len(feas) >= 2:
            var = sum(((x - mean) ** 2 for x in feas), Fraction(0)) / (len(feas) - 1)
        out[g] = {"gains": gains, "all_feasible": all_feas, "mean": mean, "var": var,
                  "sd": math.sqrt(var) if var is not None else None}
    return out


def best_descriptives(screen: Screen, panel: Panel, vals: Sequence, form: str, best: dict) -> dict:
    """The descriptive outputs of section 8.4 (:926-930): each seed's rank of every configuration,
    the number of seeds in which BEST ranks first, the BEST - runner-up paired gap over per-seed
    means with its ordinary 95 % t interval, BEST's mean gain, its minimum per-g mean gain and its
    count of positive g (owner decision B3b-6 for the per-seed means and the minimum)."""
    ranking = best["ranking"]
    best_code = ranking[0]["code"]
    runner = ranking[1] if len(ranking) > 1 else None
    per_seed_rank = {}
    firsts = 0
    for s in panel.seeds:
        r = sorted(candidate_entries(panel, vals, form, seed=s), key=_rank_key)
        per_seed_rank[s] = [e["code"] for e in r]
        firsts += int(r[0]["code"] == best_code)
    gap = None
    if runner is not None:
        diffs = []
        for s in panel.seeds:
            common = [g for g in panel.domain if panel.feasible[(ranking[0]["label"], s, g)]
                      and panel.feasible[(runner["label"], s, g)]]
            if common:
                d = [vals[panel.index[(ranking[0]["label"], s, g)]] - vals[panel.index[(runner["label"], s, g)]] for g in common]
                diffs.append(sum(d[1:], d[0]) / len(d))
            else:
                diffs.append(None)
        have = [float(x) for x in diffs if x is not None]
        gap = {"per_seed": [float(x) if x is not None else None for x in diffs],
               "interval": st.paired_interval(have, 0.95).as_dict() if len(have) >= 2 else None,
               "seeds_without_common_cell": sum(1 for x in diffs if x is None)}

    def summary(entry):
        if entry is None:
            return None
        pg = per_g_gains(screen, entry["label"], panel.domain)
        full = {g: v for g, v in pg.items() if v["all_feasible"]}
        mins = min((v["mean"] for v in full.values()), default=None)
        return {"code": entry["code"], "mean_gain": float(entry["mean_gain"]) if entry["mean_gain"] is not None else None,
                "min_per_g_mean_gain": float(mins) if mins is not None else None,
                "positive_g": sum(1 for v in full.values() if v["mean"] > 0),
                "g_infeasible_in_some_seed": sorted(g for g, v in pg.items() if not v["all_feasible"]),
                "label": LABEL_OPTIMISTIC}

    return {"per_seed_rank": per_seed_rank, "seeds_best_first": firsts, "best_vs_runner_up": gap,
            "best": summary(ranking[0]), "runner_up": summary(runner), "text": TEXT_SELECTED}


# ---------------------------------------------------------------------------------------------
# g* and the band (section 8.2; section 19.3 alternatives (i) and (ii))


def g_star(screen: Screen, dscreen: Domain, best_label: str, rule: str) -> dict:
    """g* and the screen band {g in D_screen : g >= g*} by `rule` (owner decision B3b-7 for the
    order of D_screen used by the interval rule):
      - "default" (section 8.2, :898): the smallest g in D_screen such that at every g' >= g in
        D_screen BEST is feasible in all three seeds and its mean paired validation gain is > 0;
      - "strict_sd" (section 19.3 (i), :1584): the same with the mean > the validation s_d at g';
      - "interval" (section 19.3 (ii)): the longest run of consecutive allowances of D_screen at
        each of which BEST is feasible in all seeds with mean > 0; ties to the larger allowances.
    Every per-g mean and feasibility behind it is returned for the S2 log."""
    if rule not in BAND_RULES:
        raise UnsetDecision("band rule must be one of %s, got %r" % (BAND_RULES, rule))
    if dscreen.status != "ok":
        return {"status": INCOMPLETE, "rule": rule, "reason": dscreen.reason, "g_star": None, "screen_band": []}
    pg = per_g_gains(screen, best_label, dscreen.allowances)

    def ok(g):
        v = pg[g]
        if not v["all_feasible"] or v["mean"] is None:
            return False
        if rule == "strict_sd":  # mean > sd, exactly: mean > 0 and mean^2 > var
            return v["mean"] > 0 and v["var"] is not None and v["mean"] ** 2 > v["var"]
        return v["mean"] > 0

    dom = list(dscreen.allowances)
    qual = {g: ok(g) for g in dom}
    detail = {g: {"gains": [float(x) if x is not None else None for x in pg[g]["gains"]], "all_feasible": pg[g]["all_feasible"],
                  "mean": float(pg[g]["mean"]) if pg[g]["mean"] is not None else None, "sd": pg[g]["sd"],
                  "qualifies": qual[g]} for g in dom}
    if rule in ("default", "strict_sd"):
        gs = None
        for i, g in enumerate(dom):
            if all(qual[h] for h in dom[i:]):
                gs = g
                break
        if gs is None:
            return {"status": INCOMPLETE_NO_BAND, "rule": rule, "g_star": None, "screen_band": [], "per_g": detail}
        return {"status": "ok", "rule": rule, "g_star": gs, "screen_band": [g for g in dom if g >= gs], "per_g": detail}
    runs, cur = [], []
    for g in dom:
        if qual[g]:
            cur.append(g)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    if not runs:
        return {"status": INCOMPLETE_NO_BAND, "rule": rule, "g_lo": None, "g_hi": None, "screen_band": [], "per_g": detail}
    longest = max(runs, key=lambda r: (len(r), r[-1]))
    return {"status": "ok", "rule": rule, "g_lo": longest[0], "g_hi": longest[-1], "g_star": longest[0],
            "screen_band": list(longest), "per_g": detail}


def d_band(gstar: dict, dconf: Domain) -> Domain:
    """D_band = {g in D_conf : g >= g*} (default and strict rules) or {g in D_conf : g_lo <= g <= g_hi}
    (interval rule); empty gives INCOMPLETE (no band) (:900-901)."""
    if dconf.status != "ok":
        return Domain(INCOMPLETE, (), dconf.reason)
    if gstar.get("status") != "ok":
        return Domain(INCOMPLETE_NO_BAND, (), "no g satisfies the band rule")
    if gstar["rule"] == "interval":
        dom = tuple(g for g in dconf.allowances if gstar["g_lo"] <= g <= gstar["g_hi"])
    else:
        dom = tuple(g for g in dconf.allowances if g >= gstar["g_star"])
    if not dom:
        return Domain(INCOMPLETE_NO_BAND, (), "D_band is empty")
    return Domain("ok", dom)


def d_c2(claim: ClaimDomain, dconf: Domain, gstar: Optional[dict]) -> Domain:
    """The claim domain: D_conf under (a), D_band under (b) (:897)."""
    if claim.base == "a":
        return dconf
    if gstar is None:
        raise UnsetDecision("option (b) needs the S2 g* record")
    return d_band(gstar, dconf)


# ---------------------------------------------------------------------------------------------
# PID and control selections (sections 8.6, 8.9, 8.10)


def pid_selection(run: Run, headline: str) -> dict:
    """The checkpoint with the highest validation top-1 among those with frozen H <= 350,000 that
    pass Chang's filter, ties of section 8.1, with actual cost and slack; never the closest
    over-budget point; otherwise the section 8.8 filter outcome (:994). The front is the
    nondominated filtered set, so this is its step envelope at 350,000."""
    h = check_headline(headline)
    p = step_envelope(run.front[h], PID_TARGET)
    if p is not None:
        return {"status": "selected", "point": p.as_dict(), "cost": p.cost, "slack": PID_TARGET - p.cost}
    fo = run.filter_outcome.get(h, {}).get(PID_TARGET, {})
    outcome = fo.get("outcome") or COST_INFEASIBLE
    if outcome not in (COST_INFEASIBLE, NO_ACCEPTED):
        outcome = COST_INFEASIBLE if not fo else outcome
    return {"status": outcome, "point": None, "diagnostic": fo or None}


def control_selection(epochs: Sequence[Mapping]) -> dict:
    """FP32 and W8A8: the highest validation top-1 over the whole run, with neither Chang filter
    condition, ties to the earlier epoch, then the hash (section 8.9, :1112)."""
    best = None
    for e in epochs:
        k = (-Fraction(_int(e["val_correct"], "val_correct"), _int(e["val_n"], "val_n")), _int(e["epoch"], "epoch"),
             str(e.get("ckpt_sha256", "")))
        if best is None or k < best[0]:
            best = (k, e)
    if best is None:
        return {"status": "no epochs", "epoch": None}
    e = best[1]
    return {"status": "selected", "epoch": e["epoch"], "val_correct": e["val_correct"], "val_n": e["val_n"],
            "ckpt_sha256": e.get("ckpt_sha256"), "H1": e.get("H1"), "H2": e.get("H2"), "NR": e.get("NR")}


def context_checkpoint_8_10(run: Run) -> dict:
    """Each seed's section 8.10 CH-8 context point: the released-callback front point with the
    highest validation top-1 at training-count native <= 350,000 (:936-944, :1121). Ties: lower
    training count, then earlier epoch, then path (owner decision B3b-14)."""
    pts = []
    for p in run.released_front:
        tc = p.get("train_count")
        if tc is None or float(tc) > PID_TARGET:
            continue
        pts.append(p)
    if not pts:
        return {"status": "none at training-count native <= 350,000", "point": None}
    p = min(pts, key=lambda q: (-float(q["log_val_accuracy"]), float(q["train_count"]), int(q["epoch"]), str(q.get("path", ""))))
    return {"status": "selected", "point": dict(p)}


# ---------------------------------------------------------------------------------------------
# collapse diagnostic, windows and dwell (section 8.8)


def _epoch_rows(epochs: Sequence[Mapping], count: str):
    rows = []
    for e in epochs:
        rows.append((_int(e["epoch"], "epoch"), Fraction(_int(e["val_correct"], "val_correct"), _int(e["val_n"], "val_n")),
                     _int(e[count], count)))
    rows.sort()
    for a, b in zip(rows, rows[1:]):
        if a[0] == b[0]:
            raise SelectError("epoch %d recorded twice" % a[0])
    return rows


def collapse(epochs: Sequence[Mapping], count: str, g: int = PID_TARGET,
             ref_epochs: Optional[Sequence[Mapping]] = None) -> dict:
    """The collapse diagnostic at allowance g (STUDY :1058-1068; the headline uses g = 350,000).

    e_g: the first epoch with H <= g. A_ref: the maximum validation top-1 over [0, min(2000, e_g)),
    from `ref_epochs` when given (a PID run takes A_ref from its ramp run, :1061). A_sq: the maximum
    over epochs in [e_g, e_g + 1000) with H <= g. COLLAPSE if A_ref - A_sq > 0.05. Later windows:
    the LR cycles [500m, 500(m+1)) that start at or after e_g + 1000; SUSTAINED if every later
    cycle's best top-1 at H <= g stays below A_ref - 0.05, else RECOVERED at the first that does not
    (owner decision B3b-10: a cycle without an epoch at H <= g counts as not recovered; with no
    later cycle the collapse is SUSTAINED, labelled so)."""
    rows = _epoch_rows(epochs, count)
    out = {"g": g, "count": count, "e_g": None, "A_ref": None, "A_sq": None, "drop": None, "label": None,
           "persistence": None, "later_cycles": []}
    if not rows:
        out["label"] = "no epochs"
        return out
    e_g = next((ep for ep, _, h in rows if h <= g), None)
    out["e_g"] = e_g
    if e_g is None:
        out["label"] = "not squeezed (cost infeasible)"
    ref_rows = _epoch_rows(ref_epochs, count) if ref_epochs is not None else rows
    ref_e = next((ep for ep, _, h in ref_rows if h <= g), None) if ref_epochs is not None else e_g
    ref_end = min(RAMP_LINEAR_END, ref_e) if ref_e is not None else RAMP_LINEAR_END
    ref_vals = [a for ep, a, _ in ref_rows if ep < ref_end]
    if ref_e == 0 or not ref_vals:
        out["label"] = out["label"] or "no pre-squeeze reference"
        out["A_ref"] = None
    else:
        out["A_ref"] = max(ref_vals)
    if e_g is None:
        return _collapse_floats(out)
    sq = [a for ep, a, h in rows if e_g <= ep < e_g + SQUEEZE_WINDOW and h <= g]
    out["A_sq"] = max(sq) if sq else None
    if out["A_ref"] is None or out["A_sq"] is None:
        out["label"] = out["label"] or "NA"
        return _collapse_floats(out)
    out["drop"] = out["A_ref"] - out["A_sq"]
    if out["drop"] > COLLAPSE_DELTA:
        out["label"] = "COLLAPSE"
        bar = out["A_ref"] - COLLAPSE_DELTA
        last = rows[-1][0]
        m0 = -(-(e_g + SQUEEZE_WINDOW) // LR_CYCLE)
        persistence = "SUSTAINED (no later cycle)"
        any_cycle = False
        for m in range(m0, last // LR_CYCLE + 1):
            lo, hi = LR_CYCLE * m, LR_CYCLE * (m + 1)
            vals = [a for ep, a, h in rows if lo <= ep < hi and h <= g]
            best = max(vals) if vals else None
            any_cycle = True
            out["later_cycles"].append({"cycle": [lo, hi], "best": float(best) if best is not None else None})
            if best is not None and not best < bar:
                persistence = "RECOVERED at cycle [%d, %d)" % (lo, hi)
                break
        else:
            if any_cycle:
                persistence = "SUSTAINED"
        out["persistence"] = persistence
    else:
        out["label"] = "no collapse"
    return _collapse_floats(out)


def _collapse_floats(out):
    for k in ("A_ref", "A_sq", "drop"):
        if isinstance(out[k], Fraction):
            out[k + "_exact"] = "%d/%d" % (out[k].numerator, out[k].denominator)
            out[k] = float(out[k])
    return out


def collapse_grid(epochs: Sequence[Mapping], count: str, ref_epochs: Optional[Sequence[Mapping]] = None) -> dict:
    """The grid table: the same quantities at each g in G (:1068)."""
    return {g: collapse(epochs, count, g, ref_epochs) for g in GRID}


def collapse_table(screen: Screen, labels: Optional[Sequence[str]] = None) -> dict:
    """Per arm, seed and g: the collapse record and, beside the label, A_ref - A_sq minus CH-8's
    value on the same seed (constructive K8, :1066, :1078), from the runs' per-epoch records on
    the headline count."""
    labels = tuple(labels or screen_labels())
    out = {}
    ch = {s: collapse_grid(screen.run(CH8, s).epochs, screen.headline) for s in screen.seeds}
    for lab in labels:
        out[lab] = {}
        for s in screen.seeds:
            grid = ch[s] if lab == CH8 else collapse_grid(screen.run(lab, s).epochs, screen.headline)
            rows = {}
            for g in GRID:
                r = dict(grid[g])
                c = ch[s][g]["drop"]
                r["drop_minus_ch8"] = (r["drop"] - c) if (r["drop"] is not None and c is not None) else None
                rows[str(g)] = r
            out[lab][str(s)] = rows
    return out


DWELL_BINS = (("<=100000", None, 100_000),) + tuple(
    ("(%d,%d]" % (a, b), a, b) for a, b in zip(GRID, GRID[1:])) + (("(450000,inf)", 450_000, None),)


def dwell(epochs: Sequence[Mapping], count: str) -> dict:
    """Epochs spent in each grid interval of H: (-inf, 100000], (100000, 200000], ...,
    (400000, 450000], (450000, inf) (:1069; owner decision B3b-11)."""
    rows = _epoch_rows(epochs, count)
    out = {name: 0 for name, _, _ in DWELL_BINS}
    for _, _, h in rows:
        for name, lo, hi in DWELL_BINS:
            if (lo is None or h > lo) and (hi is None or h <= hi):
                out[name] += 1
                break
    return out


def lambda_segment(epoch: int) -> str:
    """The segment of Chang's lambda schedule at `epoch`: linear [0, 2000), log [2000, 7000), constant
    from 7000 (run_train.py:90)."""
    return "linear" if epoch < RAMP_LINEAR_END else ("log" if epoch < RAMP_LOG_END else "constant")


def event_line(epochs: Sequence[Mapping], e_g: Optional[int], sidecars: Mapping[str, Mapping] = None) -> dict:
    """The section 8.8 event line (:1083): the first epoch with b = 0 at each binary scale (any column
    at a per-column site; owner decision B3b-12), the first saved checkpoint at which each layer is
    degenerate, the LR restart nearest e_g, and the lambda segment at e_g."""
    first_b0 = {}
    for e in sorted(epochs, key=lambda e: e["epoch"]):
        for site, rec in (e.get("binary_sites") or {}).items():
            b = rec.get("qg_b")
            vals = b if isinstance(b, (list, tuple)) else [b]
            if site not in first_b0 and any(v == 0 for v in vals if v is not None):
                first_b0[site] = e["epoch"]
    onset = {}
    for sha, sc in sorted((sidecars or {}).items(), key=lambda kv: kv[1].get("epoch", 0)):
        for layer, route in ((sc.get("structure_record") or {}).get("degenerate") or {}).items():
            if route and layer not in onset:
                onset[layer] = {"epoch": sc.get("epoch"), "route": route}
    restart = None if e_g is None else int(LR_CYCLE * round(e_g / LR_CYCLE))
    return {"first_b0": first_b0, "degenerate_onset": onset, "nearest_lr_restart": restart,
            "lambda_segment": lambda_segment(e_g) if e_g is not None else None}


def accepted_then_collapsed(run: Run, coll: dict, count: str) -> bool:
    """'accepted, then collapsed': a front checkpoint saved before e_g and a SUSTAINED collapse
    (:1081; owner decision B3b-13)."""
    if coll.get("label") != "COLLAPSE" or not str(coll.get("persistence", "")).startswith("SUSTAINED"):
        return False
    return any(p.epoch < coll["e_g"] for p in run.front[count])


# ---------------------------------------------------------------------------------------------
# dominance and the factorial (section 8.6, :998-1001)


def frozen_points(run: Run, count: str) -> List[Point]:
    """The distinct step-envelope checkpoints of a run over G (owner decision B3b-15)."""
    seen, out = set(), []
    for g in GRID:
        p = run.point(count, g)
        if p is not None and p.ckpt_sha256 not in seen:
            seen.add(p.ckpt_sha256)
            out.append(p)
    return out


def dominates(cand: Sequence[Tuple[float, object]], ref: Sequence[Tuple[float, object]]) -> bool:
    """Per-seed empirical Pareto dominance (STUDY :999): every reference point (cost, top-1) has a
    candidate point at no greater cost and no lower top-1, with at least one strict improvement over
    the whole front, i.e. at least one reference point has a covering candidate point that is lower
    in cost or higher in top-1 (owner decision B3b-15: the set-level reading; identical fronts do not
    dominate). An empty reference set is not dominated."""
    if not ref:
        return False
    strict = False
    for (rc, ra) in ref:
        cover = [(cc, ca) for (cc, ca) in cand if cc <= rc and ca >= ra]
        if not cover:
            return False
        strict = strict or any(cc < rc or ca > ra for (cc, ca) in cover)
    return strict


CONTRASTS = (
    ("attention", (1, 0, 0)), ("ffn", (0, 1, 0)), ("head", (0, 0, 1)),
    ("attention:ffn", (1, 1, 0)), ("attention:head", (1, 0, 1)), ("ffn:head", (0, 1, 1)),
    ("attention:ffn:head", (1, 1, 1)),
)


def contrast_sign(code: str, mask: Tuple[int, int, int]) -> int:
    """+1 or -1: the product over the masked groups of (+1 if binary, -1 if HGQ)."""
    s = 1
    for b, m in zip(bits_of(code), mask):
        if m:
            s *= 1 if b else -1
    return s


def factorial(values: Mapping[Tuple[str, int, int], Optional[object]], seeds: Sequence[int], domain: Sequence[int],
              level: float = 0.95) -> dict:
    """2^3 factorial over F000-F111 in the chosen form (values keyed by (code, seed, g), None where
    infeasible). Effects: at each (seed, g) where all eight cells are feasible, mean over the four +
    cells minus mean over the four - cells, averaged over those (seed, g) cells. Feasibility effects:
    feasible cells with the group binary minus with it HGQ. At each g where all eight cells are
    feasible in all seeds, the randomized-block interval t(0.975, df = 14) sigma_F sqrt(1/6) and,
    beside it, the interval of the per-seed effects (df = 2). The additive-model cell: each group
    binary where its main effect is positive (:999-1001)."""
    full = [(s, g) for s in seeds for g in domain if all(values.get((c, s, g)) is not None for c in FACTORIAL_CODES)]

    def eff(s, g, mask):
        plus = [values[(c, s, g)] for c in FACTORIAL_CODES if contrast_sign(c, mask) > 0]
        minus = [values[(c, s, g)] for c in FACTORIAL_CODES if contrast_sign(c, mask) < 0]
        return sum(plus[1:], plus[0]) / 4 - sum(minus[1:], minus[0]) / 4

    effects = {}
    for name, mask in CONTRASTS:
        vals = [eff(s, g, mask) for (s, g) in full]
        effects[name] = float(sum(vals[1:], vals[0]) / len(vals)) if vals else None
    feas_eff = {}
    for gi, grp in enumerate(GROUPS):
        on = sum(1 for c in FACTORIAL_CODES if bits_of(c)[gi] for s in seeds for g in domain if values.get((c, s, g)) is not None)
        off = sum(1 for c in FACTORIAL_CODES if not bits_of(c)[gi] for s in seeds for g in domain if values.get((c, s, g)) is not None)
        feas_eff[grp] = {"binary": on, "hgq": off, "difference": on - off}
    per_g = {}
    for g in domain:
        if not all(values.get((c, s, g)) is not None for c in FACTORIAL_CODES for s in seeds):
            per_g[g] = None
            continue
        y = [[float(values[(c, s, g)]) for s in seeds] for c in FACTORIAL_CODES]
        tw = st.two_way(y)
        row = {"sigma_F": tw.sigma, "df": tw.df_residual, "effects": {}}
        t14 = st.t_crit(tw.df_residual + 1, level)  # t_crit takes n and uses df = n - 1
        for name, mask in CONTRASTS:
            per_seed = [float(eff(s, g, mask)) for s in seeds]
            m = sum(per_seed) / len(per_seed)
            ssum = len(FACTORIAL_CODES) / 16.0  # sum of the squared +-1/4 coefficients = 1/2
            half = t14 * tw.sigma * math.sqrt(ssum / len(seeds))  # sqrt(1/6) at three seeds
            row["effects"][name] = {"estimate": m, "block": {"t": t14, "half_width": half, "lower": m - half, "upper": m + half},
                                    "per_seed": st.paired_interval(per_seed, level).as_dict()}
        per_g[g] = row
    additive = None
    if all(effects[n] is not None for n in ("attention", "ffn", "head")):
        additive = "F" + "".join("1" if effects[n] > 0 else "0" for n in ("attention", "ffn", "head"))
    return {"cells_averaged": len(full), "effects": effects, "feasibility_effects": feas_eff, "per_g": per_g,
            "additive_cell": additive}


def factorial_values(screen: Screen, form: str, domain: Sequence[int], count: Optional[str] = None,
                     heldout: Optional[Mapping[str, Fraction]] = None) -> Dict[Tuple[str, int, int], Optional[Fraction]]:
    """Factorial cell values keyed by (code, seed, g): validation top-1, or with `heldout`
    (ckpt_sha256 -> held-out top-1) the held-out top-1 at the same frozen points."""
    out = {}
    for code in FACTORIAL_CODES:
        lab = label_in_form(code, form)
        for s in screen.seeds:
            for g in domain:
                p = screen.point(lab, s, g, count)
                if p is None:
                    out[(code, s, g)] = None
                elif heldout is None:
                    out[(code, s, g)] = p.top1
                else:
                    out[(code, s, g)] = heldout[p.ckpt_sha256]
    return out


def control_slot(best_degenerate_attention: bool) -> dict:
    """The held control slot of section 19.2 item 9 and section 10 admission (:1257): DS-HGQ if BEST
    has a degenerate attention layer (Q, K, V or O, binary or HGQ) at any frozen point of any screen
    seed, otherwise FP32; further slots go to the other of the two, then W8A8, GR and BE."""
    first = "DS-HGQ" if best_degenerate_attention else "FP32"
    other = "FP32" if first == "DS-HGQ" else "DS-HGQ"
    return {"first_slot": first, "admission_order": [first, other, "W8A8", "GR", "BE"]}


def degenerate_attention(sidecar: Mapping) -> List[Tuple[str, str]]:
    """(layer, route) of every degenerate layer among Q, K, V and O in a front sidecar's structure record."""
    rec = (sidecar.get("structure_record") or {}).get("degenerate") or {}
    return [(l, r) for l, r in sorted(rec.items()) if r and l in ("Q", "K", "V", "O")]


# ---------------------------------------------------------------------------------------------
# the section 8.5 freeze manifest


FREEZE_SCHEMA_VERSION = 1
FREEZE_REQUIRED = (
    ("study_sha256", "the STUDY of record"),
    ("code_sha", "the fork commit that computed the selections"),
    ("headline", "the headline count H1 or H2"),
    ("d_screen", "D_screen (section 8.2)"),
    ("form_choice", "the form choice and its inputs (section 8.3)"),
    ("best", "BEST and its ranking (section 8.4)"),
    ("claim_domain", "Kai's claim-domain choice (section 19.2 item 12)"),
    ("grid_checkpoints", "every candidate's grid checkpoint hashes in both forms under H1, H2 and NR"),
    ("controls_admitted", "which controls were admitted after S2 (section 10)"),
    ("control_selections", "the control selections (section 8.9)"),
    ("pid_selections", "the PID selections (section 8.6)"),
    ("context_8_10", "each seed's section 8.10 CH-8 context checkpoint"),
    ("structure_records", "the structure record of every frozen checkpoint (section 8.1)"),
    ("s4", "Kai's S4 decision and option; if funded, D_conf and the S4 selections"),
)
_HELDOUT_TOKENS = ("test", "heldout", "holdout")


def _no_heldout_keys(obj, path="") -> List[str]:
    """Keys that name test or held-out data (the M15 rule, interfaces section 8): the manifest is
    written before the held-out read and holds no held-out value."""
    bad = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            kk = str(k).lower()
            squashed = "".join(ch for ch in kk if ch not in "-_./ ")
            if "test" in kk or "heldout" in squashed or "holdout" in squashed:
                bad.append(path + "/" + str(k))
            bad.extend(_no_heldout_keys(v, path + "/" + str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad.extend(_no_heldout_keys(v, "%s[%d]" % (path, i)))
    return bad


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def freeze_manifest_problems(manifest) -> List[str]:
    """Every way `manifest` falls short of section 8.5 (:936-944). Empty means complete."""
    p = []
    if not isinstance(manifest, Mapping) or not manifest:
        return ["the manifest is not a non-empty JSON object"]
    if manifest.get("schema_version") != FREEZE_SCHEMA_VERSION:
        p.append("schema_version must be %d" % FREEZE_SCHEMA_VERSION)
    for k, what in FREEZE_REQUIRED:
        if k not in manifest or manifest[k] is None:
            p.append("missing %s: %s" % (k, what))
    if p:
        return p
    if manifest["headline"] not in ("H1", "H2"):
        p.append("headline must be H1 or H2")
    ds = manifest["d_screen"]
    if not isinstance(ds, Mapping) or ds.get("status") != "ok" or not ds.get("allowances"):
        p.append("d_screen must be an ok domain with allowances (an INCOMPLETE screen has no held-out read)")
    if not isinstance(manifest["form_choice"], Mapping) or manifest["form_choice"].get("form") not in FORMS:
        p.append("form_choice.form must be FB or BF")
    if not isinstance(manifest["best"], Mapping) or manifest["best"].get("best") not in CANDIDATE_CODES:
        p.append("best.best must be one of the seven candidates")
    try:
        cd = claim_domain_config(manifest["claim_domain"])
        if cd.base == "b":
            gs = manifest.get("g_star")
            if not isinstance(gs, Mapping) or gs.get("rule") != cd.band_rule:
                p.append("under option (b) the manifest needs g_star with the band rule %s" % cd.band_rule)
    except UnsetDecision as e:
        p.append("claim_domain: %s" % e)
    gc = manifest["grid_checkpoints"]
    want_labels = screen_labels()
    seeds = [str(s) for s in SCREEN_SEEDS]
    if not isinstance(gc, Mapping):
        p.append("grid_checkpoints must be an object")
    else:
        for c in COUNTS:
            if c not in gc:
                p.append("grid_checkpoints lacks count %s" % c)
                continue
            for lab in want_labels:
                for s in seeds:
                    row = (gc[c].get(lab) or {}).get(s) if isinstance(gc[c], Mapping) else None
                    if not isinstance(row, Mapping):
                        p.append("grid_checkpoints %s %s seed %s missing" % (c, lab, s))
                        continue
                    for g in GRID:
                        v = row.get(str(g))
                        if v != "NA" and not (isinstance(v, str) and len(v) == 64):
                            p.append("grid_checkpoints %s %s seed %s g %d must be a sha256 or NA" % (c, lab, s, g))
    if not isinstance(manifest["controls_admitted"], list):
        p.append("controls_admitted must be a list (possibly empty)")
    if not isinstance(manifest["pid_selections"], Mapping):
        p.append("pid_selections must be an object")
    else:
        for arm in ("PID-CH", "PID-BEST"):
            for s in seeds:
                if s not in (manifest["pid_selections"].get(arm) or {}):
                    p.append("pid_selections %s seed %s missing" % (arm, s))
    if not isinstance(manifest["context_8_10"], Mapping) or any(s not in manifest["context_8_10"] for s in seeds):
        p.append("context_8_10 needs each screen seed")
    if not isinstance(manifest["structure_records"], Mapping):
        p.append("structure_records must be an object keyed by checkpoint hash")
    else:
        frozen = set()
        if isinstance(gc, Mapping):
            for c in COUNTS:
                for lab, rows in (gc.get(c) or {}).items():
                    for s, row in (rows or {}).items():
                        frozen.update(v for v in (row or {}).values() if isinstance(v, str) and len(v) == 64)
        for arm, rows in (manifest["pid_selections"] or {}).items() if isinstance(manifest["pid_selections"], Mapping) else []:
            for s, sel in (rows or {}).items():
                if isinstance(sel, Mapping) and isinstance(sel.get("ckpt_sha256"), str):
                    frozen.add(sel["ckpt_sha256"])
        missing = sorted(h for h in frozen if h not in manifest["structure_records"])
        if missing:
            p.append("structure_records lacks %d frozen checkpoint(s), first %s" % (len(missing), missing[0][:12]))
    s4 = manifest["s4"]
    if not isinstance(s4, Mapping) or not isinstance(s4.get("funded"), bool):
        p.append("s4.funded must be true or false (Kai's S4 decision)")
    elif s4["funded"]:
        if s4.get("option") not in ("A", "B", "C"):
            p.append("s4.option must be A, B or C when funded")
        dc = s4.get("d_conf")
        if not isinstance(dc, Mapping) or "status" not in dc:
            p.append("s4.d_conf is required when funded")
        if not isinstance(s4.get("selections"), Mapping):
            p.append("s4.selections is required when funded")
        cd = manifest.get("claim_domain")
        if isinstance(cd, Mapping) and cd.get("base") == "b" and not isinstance(s4.get("d_band"), Mapping):
            p.append("s4.d_band is required when funded under option (b)")
    bad = _no_heldout_keys({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if bad:
        p.append("held-out key(s) in the manifest: %s" % bad[:3])
    if "manifest_sha256" in manifest:
        body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if hashlib.sha256(_canonical(body)).hexdigest() != manifest["manifest_sha256"]:
            p.append("manifest_sha256 does not match the canonical content")
    else:
        p.append("missing manifest_sha256")
    return p


def write_freeze_manifest(path, manifest: Mapping, overwrite: bool = False) -> str:
    """Write the section 8.5 freeze manifest with its content hash, refusing an incomplete one and an
    existing file (a freeze is written once). Returns the content sha256."""
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    body["schema_version"] = FREEZE_SCHEMA_VERSION
    body["manifest_sha256"] = hashlib.sha256(_canonical({k: v for k, v in body.items() if k != "manifest_sha256"})).hexdigest()
    problems = freeze_manifest_problems(body)
    if problems:
        raise SelectError("freeze manifest incomplete: %s" % "; ".join(problems))
    p = Path(path)
    if p.exists() and not overwrite:
        raise SelectError("freeze manifest %s exists; a freeze is written once" % p)
    p.write_text(json.dumps(body, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    return body["manifest_sha256"]


def require_freeze_manifest(path) -> dict:
    """The check the sealed held-out reader calls before its one batch read (section 8.5): the
    manifest must exist, parse, and be complete. Raises FreezeRefused otherwise."""
    if path is None:
        raise FreezeRefused("no section 8.5 freeze manifest was given: the held-out read is refused")
    p = Path(path)
    if not p.is_file():
        raise FreezeRefused("section 8.5 freeze manifest %s does not exist: the held-out read is refused" % p)
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        raise FreezeRefused("freeze manifest %s is not JSON: the held-out read is refused" % p) from None
    problems = freeze_manifest_problems(m)
    if problems:
        raise FreezeRefused("freeze manifest %s is incomplete (%d problem(s): %s): the held-out read is refused"
                            % (p, len(problems), "; ".join(problems[:3])))
    return m


class SealedReaderStub:
    """Stand-in for M3's sealed held-out reader (B2, lane P) in the Mac part of G56: it opens
    nothing and refuses without a complete freeze manifest. The real reader calls
    `require_freeze_manifest` before reading (G56 PCC part, after the merge)."""

    def __init__(self, manifest_path):
        self.manifest = require_freeze_manifest(manifest_path)
        self.manifest_sha256 = self.manifest["manifest_sha256"]

    def read(self, ckpt_sha256: str) -> dict:
        frozen = set()
        for c in COUNTS:
            for rows in self.manifest["grid_checkpoints"][c].values():
                for row in rows.values():
                    frozen.update(v for v in row.values() if v != "NA")
        if ckpt_sha256 not in frozen:
            raise FreezeRefused("checkpoint %s is not a frozen point of the manifest" % ckpt_sha256[:12])
        return {"ckpt_sha256": ckpt_sha256, "manifest_sha256": self.manifest_sha256, "stub": True}


def _jsonable(x):
    if isinstance(x, Fraction):
        return {"num": x.numerator, "den": x.denominator, "float": float(x)}
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def build_freeze_manifest(*, study_sha256: str, code_sha: str, screen: Screen, dscreen: Domain, form: Mapping,
                          best: Mapping, claim: ClaimDomain, gstar: Optional[Mapping], pid_screen: Screen,
                          controls_admitted: Sequence[str], control_selections: Mapping, structure_records: Mapping,
                          s4: Mapping) -> dict:
    """Assemble the section 8.5 manifest from the S2 record and the S3 runs (:936-944). The PID
    selections and the section 8.10 context points are recomputed here from the runs; every other
    item is passed in as frozen. The result still has to pass `write_freeze_manifest`."""
    claim_domain_config(claim.as_dict())
    pid = {}
    for arm, lab in (("PID-CH", CH8), ("PID-BEST", label_in_form(best["best"], form["form"]))):
        pid[arm] = {}
        for s in screen.seeds:
            sel_ = check_pid_summary(pid_screen.run(lab, s), screen.headline)
            pid[arm][str(s)] = {"status": sel_["status"],
                                "ckpt_sha256": sel_["point"]["ckpt_sha256"] if sel_["point"] else None,
                                "cost": sel_.get("cost"), "slack": sel_.get("slack")}
    ctx = {}
    for s in screen.seeds:
        c = context_checkpoint_8_10(screen.run(CH8, s))
        ctx[str(s)] = c
    m = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "study_sha256": study_sha256,
        "code_sha": code_sha,
        "headline": screen.headline,
        "d_screen": dscreen.as_dict(),
        "form_choice": _jsonable(form),
        "best": _jsonable({"best": best["best"], "best_label": best["best_label"], "ranking": best["ranking"]}),
        "claim_domain": claim.as_dict(),
        "grid_checkpoints": grid_checkpoint_table(screen),
        "controls_admitted": list(controls_admitted),
        "control_selections": _jsonable(control_selections),
        "pid_selections": pid,
        "context_8_10": _jsonable(ctx),
        "structure_records": _jsonable(structure_records),
        "s4": _jsonable(s4),
    }
    if claim.base == "b":
        m["g_star"] = _jsonable(gstar)
    return m


def grid_checkpoint_table(screen: Screen) -> dict:
    """grid_checkpoints for the manifest: count -> label -> seed -> g -> sha256 or NA."""
    out = {}
    for c in COUNTS:
        out[c] = {}
        for lab in screen_labels():
            out[c][lab] = {}
            for s in screen.seeds:
                out[c][lab][str(s)] = {str(g): (screen.point(lab, s, g, c).ckpt_sha256 if screen.point(lab, s, g, c) else "NA")
                                       for g in GRID}
    return out
