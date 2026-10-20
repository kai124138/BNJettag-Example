"""M9: the primary front under H, the descriptive fronts under the other count and NR, the two diagnostic
checkpoints, exact-metric sidecars, the step envelope with NA, and the run summary (STUDY section 8.1).

STUDY means `review/snapshots/v0.10/STUDY.md` (sha256 7180c24f6b55807e...); "plan" means PREFLIGHT-PLAN.md
(sha256 2f1be9f3...30ff8), section 2 row M9 and section 3 row G37; "interfaces" means `docs/interfaces.md`
sections 1-3, 11 and 17. Owner decisions B8-n are in `preflight/gates/G37/<sha12>/PREFLIGHT-rows.md`.
Nothing here is quotable.

Input: the per-epoch frozen record of interfaces section 1 (writer M8): `epoch`, `val_correct`, `val_n`, `H1`,
`H2`, `NR` are required (integers); `run_id`, `step`, `train_count`, `subtotals`, `pruning_channel`,
`quantizer_structure`, `binary_sites`, `log_val_accuracy`, `lr`, `beta` are copied when present. Top-1 is the
exact fraction val_correct / val_n, never Keras's logged accuracy.

The rules are M16's, imported from `cbl.select`, so that M16's reader (`run_from_records`,
`check_pid_summary`) recomputes exactly what M9 writes:
- Fronts. One per count (H1, H2, NR), each with Chang's filter in its own units: top-1 > 1/2 and cost <
  500,000 (`select.Point.passes_filter`; run_train.py:83-89). Membership (owner decision B8-1): an epoch joins
  a front unless a member has top-1 >= its top-1 and cost <= its cost (equal on both: the earlier epoch
  stays); members it weakly dominates leave. Members therefore have distinct top-1 and distinct cost, and the
  last tie-break of STUDY 8.1, the checkpoint hash, cannot fire within one run.
- A checkpoint is saved when its epoch joins any of the three fronts or becomes a diagnostic checkpoint of any
  count; the sidecar's `joined` lists the fronts, `diagnostic` the diagnostic roles.
- Diagnostic checkpoints per count (B8-5), never in a front or BEST: `best_at_350000`, the highest top-1 at
  cost <= 350,000 ignoring the filter (`select.Point.best_key` ties); `min_cost`, the lowest cost, then the
  higher top-1, then the earlier epoch.
- Step envelope: `select.step_envelope` of the front at each g of the grid; "NA", never 0, when no member has
  cost <= g; with cost and slack = g - cost.
- Filter outcome per count and g (STUDY 8.8; B8-4): `feasible` when the envelope has a point; otherwise
  `COST INFEASIBLE` when no epoch had cost <= g, else `NO ACCEPTED CHECKPOINT` with the diagnostic point (the
  best epoch at cost <= g, filter ignored, same ties).
- Dwell: `select.dwell` over the epochs seen. PID runs: `pid_selection` is the envelope of the headline
  front at 350,000 (`select.pid_selection`'s rule).

Checkpoints and files (B8-2, B8-3). A checkpoint is the Keras `.keras` file `front/ckpt/epoch=<e>.keras`;
`ckpt_sha256` is the sha256 of its bytes (interfaces section 2). Keras writes a save time into the file
(metadata.json `date_saved` and the weights entry's zip time), so equal states saved twice hash
differently; `state_sha256` (cbl.rng.variables_sha256 over the roles) is the deterministic identity that
resume compares. With `keep_superseded=False` a file is deleted once its epoch has left every front and is no
diagnostic checkpoint any more (as Chang's ParetoFront deletes, pareto.py:62-65); its sidecar and the events
ledger stay. Directory `front/`: `ckpt/`, `sidecars/epoch=<e>.json` (written once), `attention/epoch=<e>.json`
(attention diagnostics, written once, in-loop or by the run-end audit), `events.jsonl` (one line per epoch, the
ledger from which `replay` rebuilds the state), `summary.json` (interfaces section 3).

The callback (B8-14). `FrontSaver.on_frozen_record(epoch, record)` is the entry point; it must run after the
frozen evaluation of the same epoch has produced the record, inside the stream guard, and it writes no log key.
`chain_evaluator(evaluator, saver)` wraps M8's evaluator so that the saver runs inside the frozen-evaluation
step (the callback order stays the release plus FrozenEvalHook); `keras_callback(saver)` is the alternative
eighth callback. `register(ctx, ...)` binds a saver to a runner Context (cbl/runner.py). The saver never draws
from `random` or NumPy's global stream, never iterates a Keras Dataset, and calls layers only in inference
mode (cbl/structure.py, cbl/ablate.py); G37 checks the stream and variable hashes around it.

Imports: the standard library, NumPy and `cbl.select` (NumPy and SciPy) at import time; Keras only inside
`register`, `keras_callback` and the binding helpers.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from cbl import select as sel

SCHEMA_SIDECAR = 'cbl1016-front-sidecar/1'
SCHEMA_SUMMARY = 'cbl1016-run-summary/1'
SCHEMA_EVENT = 'cbl1016-front-event/1'
COUNTS = sel.COUNTS
GRID = sel.GRID
PID_TARGET = sel.PID_TARGET
DIAG_ROLES = ('best_at_350000', 'min_cost')
REQUIRED = ('epoch', 'val_correct', 'val_n', 'H1', 'H2', 'NR')
COPIED = ('run_id', 'step', 'train_count', 'subtotals', 'pruning_channel', 'quantizer_structure', 'binary_sites',
          'log_loss', 'log_accuracy', 'log_val_loss', 'log_val_accuracy', 'lr', 'beta', 'pid',
          'quantizers', 'bn', 'activations', 'train_frozen_top1_gap', 'dwell')
ABLATION_MODES = ('every_save', 'run_end')
RUN_ENDS = sel.RUN_ENDS
N_CLASSES = 5


class FrontError(ValueError):
    """A frozen record that is not interfaces section 1, or a saver used out of order."""


# ----------------------------------------------------------------------------------------------
# records and points


@dataclass(frozen=True)
class Rec:
    epoch: int
    correct: int
    n: int
    cost: tuple            # (H1, H2, NR)

    def c(self, count: str) -> int:
        return self.cost[COUNTS.index(count)]

    @property
    def top1(self) -> Fraction:
        return Fraction(self.correct, self.n)


def _int(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise FrontError(f'{what} must be an integer, not {v!r} ({type(v).__name__})')
    return int(v)


def rec_from(epoch, record) -> Rec:
    """The required fields of an interfaces section 1 record as a Rec; refuses missing, non-integer or impossible
    values and an epoch that differs from the callback's."""
    missing = [k for k in REQUIRED if k not in record]
    if missing:
        raise FrontError(f'frozen record lacks {missing} (interfaces section 1)')
    r = Rec(epoch=_int(record['epoch'], 'epoch'), correct=_int(record['val_correct'], 'val_correct'),
            n=_int(record['val_n'], 'val_n'), cost=tuple(_int(record[c], c) for c in COUNTS))
    if epoch is not None and r.epoch != int(epoch):
        raise FrontError(f'record epoch {r.epoch} differs from the callback epoch {epoch}')
    if r.n <= 0 or not 0 <= r.correct <= r.n or min(r.cost) < 0 or r.epoch < 0:
        raise FrontError(f'impossible frozen record {record!r}')
    return r


def best_key(r: Rec, count: str):
    """STUDY 8.1 ties for 'highest top-1 at cost <= g': higher top-1, lower cost, earlier epoch (the hash
    cannot tie within a run). Equal to select.Point.best_key without the hash."""
    return (-r.top1, r.c(count), r.epoch)


def min_cost_key(r: Rec, count: str):
    return (r.c(count), -r.top1, r.epoch)


def passes_filter(r: Rec, count: str) -> bool:
    return r.top1 > sel.FILTER_TOP1 and r.c(count) < sel.FILTER_COST


# ----------------------------------------------------------------------------------------------
# the pure state


class FrontState:
    """Fronts, diagnostic checkpoints, per-g best epochs and dwell over the epochs offered so far."""

    def __init__(self):
        self.front = {c: [] for c in COUNTS}
        self.diag = {c: {role: None for role in DIAG_ROLES} for c in COUNTS}
        self.best_at = {c: {g: None for g in GRID} for c in COUNTS}
        self.epochs = []                 # Recs in offer order

    def offer(self, r: Rec) -> dict:
        """Add one epoch; return what changed: joined fronts, members that left, diagnostic roles taken."""
        if self.epochs and r.epoch <= self.epochs[-1].epoch:
            raise FrontError(f'epoch {r.epoch} offered after epoch {self.epochs[-1].epoch}')
        self.epochs.append(r)
        joined, left, diag_new = [], {}, []
        for c in COUNTS:
            if passes_filter(r, c):
                fr = self.front[c]
                if not any(m.top1 >= r.top1 and m.c(c) <= r.c(c) for m in fr):
                    gone = [m for m in fr if r.top1 >= m.top1 and r.c(c) <= m.c(c)]
                    self.front[c] = [m for m in fr if m not in gone] + [r]
                    joined.append(c)
                    if gone:
                        left[c] = [m.epoch for m in gone]
            cur = self.diag[c]['best_at_350000']
            if r.c(c) <= PID_TARGET and (cur is None or best_key(r, c) < best_key(cur, c)):
                self.diag[c]['best_at_350000'] = r
                diag_new.append(f'{c}:best_at_350000')
            cur = self.diag[c]['min_cost']
            if cur is None or min_cost_key(r, c) < min_cost_key(cur, c):
                self.diag[c]['min_cost'] = r
                diag_new.append(f'{c}:min_cost')
            for g in GRID:
                cur = self.best_at[c][g]
                if r.c(c) <= g and (cur is None or best_key(r, c) < best_key(cur, c)):
                    self.best_at[c][g] = r
        return {'epoch': r.epoch, 'joined': joined, 'left': left, 'diagnostic': diag_new}

    def dwell(self, current=None):
        """M9 cumulative dwell including current once; restored epochs come from M14's snapshot."""
        rows = [{'epoch': r.epoch, 'val_correct': r.correct, 'val_n': r.n,
                 **{c: r.c(c) for c in COUNTS}} for r in self.epochs]
        if current is not None and (not rows or rows[-1]['epoch'] < current['epoch']):
            rows.append(current)
        return {c: sel.dwell(rows, c) for c in COUNTS}

    def live_epochs(self) -> set:
        """Epochs whose checkpoint is still needed: front members and current diagnostic checkpoints."""
        out = {m.epoch for c in COUNTS for m in self.front[c]}
        out |= {r.epoch for c in COUNTS for r in self.diag[c].values() if r is not None}
        return out

    def roles_of(self, epoch) -> dict:
        return {'fronts': [c for c in COUNTS if any(m.epoch == epoch for m in self.front[c])],
                'diagnostic': [f'{c}:{role}' for c in COUNTS for role in DIAG_ROLES
                               if self.diag[c][role] is not None and self.diag[c][role].epoch == epoch]}

    # JSON state for resume (M14, B10)
    def state(self) -> dict:
        rec = lambda r: None if r is None else [r.epoch, r.correct, r.n, list(r.cost)]   # noqa: E731
        return {'epochs': [rec(r) for r in self.epochs],
                'front': {c: [m.epoch for m in self.front[c]] for c in COUNTS},
                'diag': {c: {role: (None if r is None else r.epoch) for role, r in self.diag[c].items()} for c in COUNTS},
                'best_at': {c: {str(g): (None if r is None else r.epoch) for g, r in self.best_at[c].items()} for c in COUNTS}}

    @classmethod
    def from_state(cls, st) -> 'FrontState':
        s = cls()
        by_epoch = {}
        for e in st['epochs']:
            r = Rec(epoch=e[0], correct=e[1], n=e[2], cost=tuple(e[3]))
            s.epochs.append(r)
            by_epoch[r.epoch] = r
        s.front = {c: [by_epoch[e] for e in st['front'][c]] for c in COUNTS}
        s.diag = {c: {role: (None if e is None else by_epoch[e]) for role, e in st['diag'][c].items()} for c in COUNTS}
        s.best_at = {c: {g: (None if st['best_at'][c][str(g)] is None else by_epoch[st['best_at'][c][str(g)]])
                         for g in GRID} for c in COUNTS}
        return s


def replay(recs) -> FrontState:
    """A FrontState rebuilt by offering every record again, in order (the batch recompute G37 compares with the
    incremental state)."""
    s = FrontState()
    for r in recs:
        s.offer(r)
    return s


def batch_front(recs, count: str) -> list:
    """The filtered nondominated set by brute force (for the gates): members are the filtered epochs that no
    other filtered epoch weakly dominates, an equal pair keeping the earlier epoch."""
    f = [r for r in recs if passes_filter(r, count)]
    out = []
    for r in f:
        dominated = any((o.top1 >= r.top1 and o.c(count) <= r.c(count)) and
                        ((o.top1, -o.c(count)) != (r.top1, -r.c(count)) or o.epoch < r.epoch) for o in f if o is not r)
        if not dominated:
            out.append(r)
    return out


# ----------------------------------------------------------------------------------------------
# metrics


def softmax64(logits):
    z = np.asarray(logits, np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def auc_one_vs_rest(scores, y, k):
    """Exact one-vs-rest AUC of class k from scores (n,) with average ranks for ties (Mann-Whitney); None when a
    class is absent. Computed with integer doubled ranks, so the value is a correctly rounded fraction."""
    s = np.asarray(scores, np.float64)
    pos = np.asarray(y).ravel() == k
    n1 = int(pos.sum())
    n0 = len(s) - n1
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    ends = np.r_[starts[1:], len(ss)]
    twice = np.repeat(starts + 1 + ends, ends - starts).astype(np.int64)   # doubled average rank of each tie group
    r2 = np.empty(len(ss), np.int64)
    r2[order] = twice
    num_ = int(r2[pos].sum()) - n1 * (n1 + 1)
    return float(Fraction(num_, 2 * n1 * n0))


def val_metrics(logits, y) -> dict:
    """Validation top-1 (exact count) and macro one-vs-rest AUC over the five softmax outputs (STUDY 8), from
    the M11 baseline logits (B8-16)."""
    lg = np.asarray(logits)
    y = np.asarray(y).ravel()
    p = softmax64(lg)
    per = [auc_one_vs_rest(p[:, k], y, k) for k in range(lg.shape[1])]
    good = [a for a in per if a is not None]
    return {'correct': int(np.sum(np.argmax(lg, axis=1) == y)), 'n': int(len(y)),
            'val_auc_per_class': per, 'val_auc_macro': (math.fsum(good) / len(good)) if len(good) == len(per) else None,
            'logits_sha256': hashlib.sha256(np.ascontiguousarray(lg).tobytes()).hexdigest()}


# ----------------------------------------------------------------------------------------------
# the run summary (interfaces section 3)


def _point(r: Rec, count: str, sha: str) -> dict:
    return {'epoch': r.epoch, 'ckpt_sha256': sha, 'val_correct': r.correct, 'val_n': r.n, 'cost': r.c(count)}


def _diag_point(r: Rec, count: str) -> dict:
    return {'epoch': r.epoch, 'val_correct': r.correct, 'val_n': r.n, 'val_top1': r.correct / r.n, 'cost': r.c(count)}


def summary_of(state: FrontState, sha_of, *, run_id, config_sha256, code_sha, end, epochs_completed, headline,
               schedule, released_front=(), extra=None) -> dict:
    """The run summary: fronts, envelopes, filter outcomes, diagnostic checkpoints, released front, dwell and the
    PID selection. `sha_of(epoch)` gives the checkpoint hash of a saved epoch."""
    if end not in RUN_ENDS:
        raise FrontError(f'end {end!r} is not one of {RUN_ENDS}')
    front, env, fo, diag, dwell = {}, {}, {}, {}, {}
    for c in COUNTS:
        mem = sorted(state.front[c], key=lambda r: r.epoch)
        front[c] = [_point(r, c, sha_of(r.epoch)) for r in mem]
        pts = [sel.Point(r.epoch, sha_of(r.epoch), r.correct, r.n, r.c(c)) for r in mem]
        env[c], fo[c] = {}, {}
        for g in GRID:
            p = sel.step_envelope(pts, g)
            if p is not None:
                env[c][str(g)] = dict(p.as_dict(), slack=g - p.cost)
                fo[c][str(g)] = {'outcome': 'feasible'}
                continue
            env[c][str(g)] = 'NA'
            b = state.best_at[c][g]
            if b is None:
                fo[c][str(g)] = {'outcome': sel.COST_INFEASIBLE}
            else:
                fo[c][str(g)] = {'outcome': sel.NO_ACCEPTED, 'diagnostic': _diag_point(b, c),
                                 'label': 'ACCURACY FILTER FAILURE'}
        diag[c] = {}
        for role in DIAG_ROLES:
            r = state.diag[c][role]
            diag[c][role] = None if r is None else dict(_point(r, c, sha_of(r.epoch)), val_top1=r.correct / r.n)
        rows = [{'epoch': r.epoch, 'val_correct': r.correct, 'val_n': r.n, c: r.c(c)} for r in state.epochs]
        dwell[c] = sel.dwell(rows, c) if rows else {name: 0 for name, _, _ in sel.DWELL_BINS}
    pid = None
    if schedule == 'pid':
        if headline not in COUNTS[:2]:
            raise FrontError(f'a PID run needs the headline H1 or H2, not {headline!r}')
        p = env[headline][str(PID_TARGET)]
        if p == 'NA':
            pid = {'count': headline, 'status': fo[headline][str(PID_TARGET)]['outcome'], 'ckpt_sha256': None,
                   'point': None, 'diagnostic': fo[headline][str(PID_TARGET)].get('diagnostic')}
        else:
            pid = {'count': headline, 'status': 'selected', 'ckpt_sha256': p['ckpt_sha256'], 'point': p,
                   'cost': p['cost'], 'slack': PID_TARGET - p['cost']}
    out = {'schema': SCHEMA_SUMMARY, 'run_id': run_id, 'config_sha256': config_sha256, 'code_sha': code_sha,
           'end': end, 'epochs_completed': int(epochs_completed), 'epochs_offered': len(state.epochs),
           'headline': headline, 'front': front, 'envelope': env, 'filter_outcome': fo, 'diagnostic_ckpts': diag,
           'released_front': list(released_front), 'dwell': dwell, 'pid_selection': pid}
    out.update(extra or {})
    return out


def released_front_of(pareto, run_dir) -> list:
    """Chang's callback front as released (pareto.py: `paths`, `record` = [val_accuracy, ebops]), by training
    count: {epoch, train_count, log_val_accuracy, path}, the epoch read from his file name `epoch={epoch}-...`
    (run_train.py:87)."""
    out = []
    for p, rec in zip(pareto.paths, pareto.record):
        m = re.match(r'epoch=(\d+)-', Path(p).name)
        out.append({'epoch': int(m.group(1)) if m else None, 'train_count': float(rec[1]),
                    'log_val_accuracy': float(rec[0]), 'path': os.path.relpath(str(p), str(run_dir))})
    return out


# ----------------------------------------------------------------------------------------------
# the saver


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _write_json_once(path: Path, obj):
    if path.exists():
        raise FrontError(f'{path} exists; sidecars are written once')
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_jsonable) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Fraction):
        return {'num': o.numerator, 'den': o.denominator, 'float': float(o)}
    raise TypeError(f'not JSON: {type(o).__name__}')


class FrontSaver:
    """Saves front and diagnostic checkpoints with their exact-metric sidecars (module docstring).

    Bound functions (no Keras here; `register` supplies the Keras ones, the Mac gate supplies stand-ins):
      save_fn(path)            writes the checkpoint file of the current state;
      state_fn()               the deterministic state hash (cbl.rng.variables_sha256 of the roles);
      inloop_fn()              {'metrics': val_metrics, 'structure_record': ..., 'attention_diagnostics' or
                               None, 'seconds': ...} at the current state (cbl.structure, cbl.ablate);
      audit_fn(ckpt_path, record) -> attention diagnostics of a saved checkpoint (run_end mode).
    `ablations` and `keep_superseded` have no default (owner decisions B8-17 and B8-3)."""

    def __init__(self, out_dir, *, run_id, config_sha256, code_sha, headline, schedule, save_fn, state_fn,
                 inloop_fn, audit_fn, ablations, keep_superseded):
        if ablations not in ABLATION_MODES:
            raise FrontError(f'ablations must be one of {ABLATION_MODES}, not {ablations!r} (no default)')
        if not isinstance(keep_superseded, bool):
            raise FrontError(f'keep_superseded must be true or false, not {keep_superseded!r} (no default)')
        if headline not in (None, 'H1', 'H2'):
            raise FrontError(f'headline must be H1, H2 or null, not {headline!r}')
        if schedule not in ('ramp', 'pid'):
            raise FrontError(f'schedule must be ramp or pid, not {schedule!r}')
        if schedule == 'pid' and headline is None:
            raise FrontError('a PID run needs the headline (its sensor reads it, STUDY 7.2)')
        self.dir = Path(out_dir)
        self.run_id, self.config_sha256, self.code_sha = run_id, config_sha256, code_sha
        self.headline, self.schedule = headline, schedule
        self.save_fn, self.state_fn, self.inloop_fn, self.audit_fn = save_fn, state_fn, inloop_fn, audit_fn
        self.ablations, self.keep_superseded = ablations, keep_superseded
        self.state = FrontState()
        self.saved = {}                  # epoch -> {'path', 'sha256', 'state_sha256', 'bytes', 'deleted_at'}
        self.on_disk = set()             # saved epochs whose file has not been deleted
        self.records = {}                # epoch -> {'val_correct', 'val_n'} of saved epochs (the run-end audit)
        self.finalized = None
        for d in ('ckpt', 'sidecars', 'attention'):
            (self.dir / d).mkdir(parents=True, exist_ok=True)
        self.events_path = self.dir / 'events.jsonl'

    # paths, relative to the run directory (the parent of front/)
    def rel(self, p: Path) -> str:
        return os.path.relpath(str(p), str(self.dir.parent))

    def ckpt_path(self, epoch) -> Path:
        return self.dir / 'ckpt' / f'epoch={epoch}.keras'

    def sidecar_path(self, epoch) -> Path:
        return self.dir / 'sidecars' / f'epoch={epoch}.json'

    def attention_path(self, epoch) -> Path:
        return self.dir / 'attention' / f'epoch={epoch}.json'

    def sha_of(self, epoch) -> str:
        return self.saved[epoch]['sha256']

    def on_frozen_record(self, epoch, record) -> dict:
        """One epoch's frozen record (interfaces section 1). Returns the event written to events.jsonl."""
        if self.finalized is not None:
            raise FrontError('the saver is finalized')
        t0 = time.time()
        r = rec_from(epoch, record)
        change = self.state.offer(r)
        need = bool(change['joined'] or change['diagnostic'])
        ev = {'schema': SCHEMA_EVENT, 'epoch': r.epoch, 'val_correct': r.correct, 'val_n': r.n,
              **{c: r.c(c) for c in COUNTS}, 'joined': change['joined'], 'left': change['left'],
              'diagnostic': change['diagnostic'], 'saved': need, 'deleted': []}
        if need:
            ev.update(self._save(r, record, change))
        if not self.keep_superseded:
            live = self.state.live_epochs()
            for e in sorted(self.on_disk - live):
                s = self.saved[e]
                os.remove(self.dir.parent / s['path'])
                s['deleted_at'] = r.epoch
                self.on_disk.discard(e)
                ev['deleted'].append(e)
        ev['seconds'] = time.time() - t0
        with open(self.events_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(ev, sort_keys=True, default=_jsonable) + '\n')
        return ev

    def _save(self, r: Rec, record, change) -> dict:
        t0 = time.time()
        path = self.ckpt_path(r.epoch)
        if path.exists():
            raise FrontError(f'{path} exists')
        self.save_fn(path)
        sha = sha256_file(path)
        size = path.stat().st_size
        t_save = time.time() - t0
        state_sha = self.state_fn()
        inloop = self.inloop_fn()
        met = inloop['metrics']
        copied = {k: record[k] for k in COPIED if k in record}
        side = {'schema': SCHEMA_SIDECAR, 'label': 'validation only', 'run_id': self.run_id or record.get('run_id'),
                'epoch': r.epoch, 'ckpt_path': self.rel(path), 'ckpt_sha256': sha, 'ckpt_bytes': size,
                'state_sha256': state_sha, 'val_correct': r.correct, 'val_n': r.n, 'val_top1': r.correct / r.n,
                'H1': r.c('H1'), 'H2': r.c('H2'), 'NR': r.c('NR'), 'joined': list(change['joined']),
                'diagnostic': list(change['diagnostic']),
                'chang_filter': {c: passes_filter(r, c) for c in COUNTS},
                'val_auc_macro': met['val_auc_macro'], 'val_auc_per_class': met['val_auc_per_class'],
                'val_logits_sha256': met['logits_sha256'], 'm11_correct': met['correct'], 'm11_n': met['n'],
                'm11_equals_record': bool(met['correct'] == r.correct and met['n'] == r.n),
                'structure_record': inloop['structure_record'],
                'attention_diagnostics_file': None, 'attention_mode': self.ablations,
                'seconds': {'save': t_save, **(inloop.get('seconds') or {})}, 'record': copied}
        if inloop.get('attention_diagnostics') is not None:
            ap = self.attention_path(r.epoch)
            _write_json_once(ap, dict(inloop['attention_diagnostics'], source='in loop', epoch=r.epoch, ckpt_sha256=sha))
            side['attention_diagnostics_file'] = self.rel(ap)
        _write_json_once(self.sidecar_path(r.epoch), side)
        self.saved[r.epoch] = {'path': self.rel(path), 'sha256': sha, 'state_sha256': state_sha, 'bytes': size,
                               'deleted_at': None}
        self.on_disk.add(r.epoch)
        self.records[r.epoch] = {'val_correct': r.correct, 'val_n': r.n}
        return {'ckpt_sha256': sha, 'state_sha256': state_sha, 'ckpt_bytes': size}

    def finalize(self, *, end, epochs_completed, released_front=()) -> dict:
        """Run end: attention diagnostics of the checkpoints still saved (run_end mode, by the audit path), then
        the run summary, checked by M16's reader. Returns the summary, also written to front/summary.json."""
        if self.finalized is not None:
            return self.finalized
        t0 = time.time()
        audited = []
        if self.ablations == 'run_end':
            for e in sorted(self.state.live_epochs()):
                ap = self.attention_path(e)
                if ap.exists():
                    continue
                s = self.saved[e]
                diag = self.audit_fn(self.dir.parent / s['path'], self.records.get(e, {}))
                _write_json_once(ap, dict(diag, source='checkpoint audit', epoch=e, ckpt_sha256=s['sha256']))
                audited.append(e)
        extra = {'checkpoints': {str(e): dict(s) for e, s in sorted(self.saved.items())},
                 'live_epochs': sorted(self.state.live_epochs()), 'attention_mode': self.ablations,
                 'attention_audited': audited, 'keep_superseded': self.keep_superseded,
                 'finalize_seconds': time.time() - t0, 'label': 'validation only'}
        summ = summary_of(self.state, self.sha_of, run_id=self.run_id, config_sha256=self.config_sha256,
                          code_sha=self.code_sha, end=end, epochs_completed=epochs_completed, headline=self.headline,
                          schedule=self.schedule, released_front=released_front, extra=extra)
        summ['m16_reader'] = m16_check(summ, self.headline, self.schedule)
        _write_json_once(self.dir / 'summary.json', summ)
        self.finalized = summ
        return summ

    # resume support (M14, batch B10)
    def snapshot(self) -> dict:
        return {'state': self.state.state(), 'saved': {str(e): dict(s) for e, s in self.saved.items()},
                'records': {str(e): dict(v) for e, v in self.records.items()},
                'ablations': self.ablations, 'keep_superseded': self.keep_superseded}

    def restore(self, snap):
        if snap['ablations'] != self.ablations or snap['keep_superseded'] != self.keep_superseded:
            raise FrontError('a snapshot from a saver with other modes')
        self.state = FrontState.from_state(snap['state'])
        self.saved = {int(e): dict(s) for e, s in snap['saved'].items()}
        self.on_disk = {e for e, s in self.saved.items() if s['deleted_at'] is None}
        self.records = {int(e): dict(v) for e, v in snap['records'].items()}


def m16_check(summary, headline, schedule) -> dict:
    """M16's reader on the summary (cbl.select.run_from_records; check_pid_summary for PID runs)."""
    cfg = {'arm_id': 'M9-self-check', 'seed': 0, 'stage': 'gate', 'schedule': schedule}
    try:
        run = sel.run_from_records(cfg, summary)
        if schedule == 'pid':
            sel.check_pid_summary(run, headline)
        return {'ok': True}
    except Exception as e:  # noqa: BLE001 - recorded in the summary, never hidden
        return {'ok': False, 'error': f'{type(e).__name__}: {e}'}


def load_sidecars(front_dir) -> dict:
    """{ckpt_sha256: sidecar} with `attention_diagnostics` merged from attention/ where it exists: the shape
    M16 reads (interfaces section 2)."""
    d = Path(front_dir)
    out = {}
    for p in sorted((d / 'sidecars').glob('epoch=*.json')):
        s = json.loads(p.read_text(encoding='utf-8'))
        ap = d / 'attention' / p.name
        s['attention_diagnostics'] = json.loads(ap.read_text(encoding='utf-8')) if ap.exists() else None
        out[s['ckpt_sha256']] = s
    return out


def read_events(front_dir) -> list:
    return [json.loads(line) for line in (Path(front_dir) / 'events.jsonl').read_text(encoding='utf-8').splitlines() if line]


def recs_from_events(events) -> list:
    return [Rec(epoch=e['epoch'], correct=e['val_correct'], n=e['val_n'], cost=tuple(e[c] for c in COUNTS)) for e in events]


# ----------------------------------------------------------------------------------------------
# Keras binding (imports Keras; used in the training process and the PC gates)


def record_from_logs(epoch, logs) -> dict:
    """The interfaces section 1 record as M8 leaves it in the epoch logs (its keys, unprefixed)."""
    out = {k: logs[k] for k in REQUIRED + COPIED if k in logs}
    out.setdefault('epoch', int(epoch))
    return out


def bind(model, roles, X_val, y_val, *, ablations, jit=True, batch=None):
    """save_fn, state_fn, inloop_fn and audit_fn for a Keras model. X_val (float32) and y_val are the
    validation arrays as NumPy (never a Keras Dataset). One cbl.ablate.Exec serves every save (one
    compilation per pass kind)."""
    from cbl import ablate, rng, structure

    X = np.asarray(X_val)
    y = np.asarray(y_val).ravel()
    if X.dtype != np.float32:
        raise FrontError(f'validation inputs must be float32, not {X.dtype}')
    ex = ablate.Exec(ablate.Graph(model), jit=jit, batch=ablate.VAL_BATCH if batch is None else batch)

    def save_fn(path):
        model.save(str(path))

    def state_fn():
        return rng.variables_sha256(roles)

    def inloop_fn():
        t0 = time.time()
        attn = None
        if ablations == 'every_save':
            base = ablate.BaselinePass(ex, X, y)
            zv, logits = base.zv, base.logits
        else:
            zv, logits = structure.zero_variance_pass(ex, X)
        t1 = time.time()
        srec = structure.record_from(structure.quantizer_state(model, ex.g), zv, batch=ex.batch, seconds=t1 - t0)
        met = val_metrics(logits, y)
        if ablations == 'every_save':
            attn = ablate.diagnostics_from(ex.g, base, ablate.ablations(ex, X, y, base))
        return {'metrics': met, 'structure_record': srec, 'attention_diagnostics': attn,
                'seconds': {'pass_and_structure': t1 - t0, 'attention': time.time() - t1}}

    def audit_fn(ckpt_path, record):
        return ablate.audit_checkpoint(ckpt_path, X, y, record_correct=record.get('val_correct'), jit=jit,
                                       batch=ex.batch)

    return save_fn, state_fn, inloop_fn, audit_fn


def register(ctx, *, ablations, keep_superseded, out_dir=None, jit=True, batch=None) -> FrontSaver:
    """A FrontSaver bound to a runner Context (cbl/runner.py `prepare`): ctx.model, ctx.roles, the validation
    arrays ctx.flow.tv.X_val (float16 from the reader, widened exactly to float32; float32 as is; any other dtype
    is refused by `bind`) and y_val, ctx.cfg (headline, schedule, code_sha), ctx.record['config_sha256'] and the
    W&B run name as run_id; front/ under ctx.out_dir."""
    cfg = ctx.cfg
    X = np.asarray(ctx.flow.tv.X_val)
    if X.dtype == np.float16:
        # cbl.data.read_train_val returns float16 as chang/data.py:33 does; the model's float32 input conversion
        # widens it exactly, so the saver's passes see the values the model sees
        X = X.astype(np.float32)
    save_fn, state_fn, inloop_fn, audit_fn = bind(ctx.model, ctx.roles, X, ctx.flow.tv.y_val,
                                                  ablations=ablations, jit=jit, batch=batch)
    run_id = getattr(ctx.wandb_run, 'name', None) if ctx.wandb_run is not None else None
    return FrontSaver(Path(out_dir) if out_dir else Path(ctx.out_dir) / 'front', run_id=run_id,
                      config_sha256=ctx.record.get('config_sha256'), code_sha=cfg.get('code_sha'),
                      headline=cfg.get('headline'), schedule=cfg['schedule'], save_fn=save_fn, state_fn=state_fn,
                      inloop_fn=inloop_fn, audit_fn=audit_fn, ablations=ablations, keep_superseded=keep_superseded)


def chain_evaluator(evaluator, saver: FrontSaver):
    """M8's evaluator(epoch, logs) followed by the saver on the record it produced: the record M8 returns if it
    returns a mapping, else the section 1 keys it left in the logs. Use as `frozen_eval_hook(evaluator=...)`."""
    def run(epoch, logs):
        out = evaluator(epoch, logs) if evaluator is not None else None
        record = out if isinstance(out, dict) else record_from_logs(epoch, logs)
        saver.on_frozen_record(epoch, record)
        return out
    return run


def keras_callback(saver: FrontSaver, record_of=record_from_logs):
    """The alternative wiring: an eighth callback after FrozenEvalHook that reads the record from the logs."""
    import keras

    class FrontSaverCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                raise FrontError('FrontSaverCallback needs the epoch logs')
            saver.on_frozen_record(epoch, record_of(epoch, logs))

    return FrontSaverCallback()


__all__ = ['FrontSaver', 'FrontState', 'Rec', 'rec_from', 'replay', 'batch_front', 'summary_of', 'val_metrics',
           'auc_one_vs_rest', 'released_front_of', 'load_sidecars', 'read_events', 'recs_from_events', 'm16_check',
           'register', 'bind', 'chain_evaluator', 'keras_callback', 'record_from_logs', 'ABLATION_MODES']
