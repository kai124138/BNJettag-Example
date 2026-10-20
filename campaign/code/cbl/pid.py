"""M12: the lab PID, hgq2's BetaPID with the frozen-H sensor and its own log keys (STUDY section 7.2).

STUDY means `review/snapshots/v0.10/STUDY.md` (sha256 7180c24f6b55807e...): section 7.2 whole (the PID stage),
7.1 (callback order and semantics), 3.3 (the frozen evaluation is the PID sensor), 12 (the PID's logged values) and
the section 13 PID row; "plan" means PREFLIGHT-PLAN.md (sha256 2f1be9f3...30ff8), section 2 row M12 and section 3
rows G43-G47 (G46 with M14, batch B10). Interface notes: docs/interfaces.md sections 1 (the record's `pid` block), 5
(the config's `pid` field) and 18 (B9). Owner decisions B9-n are in PREFLIGHT.md (B9 section). Nothing here is
quotable.

Kept from the release (plan row M12, reuse "controller law and defaults"). The class subclasses hgq2 0.1.9's
`hgq.utils.sugar.beta_pid.BetaPID` (installed beta_pid.py, sha256 c25dac9185b5..., identical to the snapshot's
reference-code/HGQ2/src/hgq/utils/sugar/beta_pid.py; STUDY 7.3) and keeps its law: `on_epoch_begin` is the stock
method, run through `super()` on the stored reading `_ebops` (beta_pid.py:131-160): warmup epochs hold init_beta
(:134-137); at epoch == warmup the integral is set so that the controller starts at the current beta (:140-148);
then beta = 10 ** PID(0, log10(sensed / target)) with log true (:150-151), damping below target (:155-156), the clamp
to [min_beta, max_beta] (:158), and every layer's `_beta` is assigned (:159; BaseBetaPID.set_beta, :49-53). Its PID
object is the stock `PID` (:8-32), subclassed only to record the output of its last call (`super().__call__` does
the arithmetic).

Changed (STUDY 7.2: "the stock BetaPID is modified in its sensor and its log key, and runs in H units"):
1. Sensor (chang C1). The stock on_epoch_end reads the top-level public aggregate (get_ebops, :41-47, :165). This one
   computes frozen H(e) in its own on_epoch_end by the shared non-mutating frozen evaluation (`sensor_reader`:
   `FrozenEvaluator.headline_value(epoch, headline)`, cbl/frozen_eval.py, which caches the epoch's evaluation; the
   appended callback reuses it) inside the stream guard's watch, and stores it as the reading that the stock
   on_epoch_begin(e + 1) acts on (stock timing, :139, :167).
2. Log keys. logs['ebops'] is never written (the stock writes it, :166), so ParetoFront and StopIf read FreeEBOPs's
   training count in PID runs as in ramp runs. logs['beta'] is written as the stock does (:164); PBar's template reads
   it (run_train.py:79-81). The sensed value goes under its own key `pid/sensed_H`; the stock aggregate, computed by
   the stock get_ebops, is logged beside it as `pid/stock_aggregate`; the rest of the block of STUDY 12 follows
   (`BLOCK_KEYS`). Scalars only: a field without a value (the error, the reading acted on and the unclamped beta
   during warmup) is left out, never written as null (cbl/frozen_eval.py `pid_block` reads the keys back).
3. Train begin. The stock on_train_begin runs (:125-129: init_beta on every layer, beta = init_beta, a stock
   reading); its reading is kept as `stock_at_train_begin` and the stored reading is cleared, because the frozen
   sensor has no reading before epoch 0. warmup >= 1 is required, so the stock law never acts without a frozen
   reading; on_epoch_begin refuses if it would.
4. Persisted state (STUDY 7.2 "On resume, the integral, previous error, lambda, epoch and sensed cost are
   persisted"; M14 in B10 stores it): `state()` and `restore(state)`; a restored controller's on_train_begin
   re-applies the restored beta and does not reinitialize, and its next on_epoch_begin must be the epoch after the
   stored one.

Values (the config's `pid` field must equal PID_RECORD; interfaces section 5): target_H 350000 (the OD-12 reading of
Q16's "350,000 EBOPs" in headline units, adopted under Kai's 2026-10-17 delegation and not confirmed, STUDY 19.2 item
4); init_beta 2e-8 (lambda_0 > 0: Chang's beta0 = 0 (model.py:295) with init_beta None makes the stock controller
take log10(0) at epoch == warmup, beta_pid.py:128, :148; STUDY 7.2); p 1, i 0.002, d 0, warmup 10, log true, max_beta
infinite, min_beta 0, damping 0, the released defaults of BetaPID.__init__ (beta_pid.py:95-106; STUDY 7.2). The
configs write max_beta as the JSON string "inf" (gen/configs.py PID; batch B12), which `parse_max_beta` reads.

Units (OD-12) are chosen in one place: `SENSOR` and `sensor_reader`. 'headline' reads the frozen headline count H
(H1 or H2, the config's `headline`). A native-unit PID would change `sensor_reader` (to frozen NR, which equals native
on CH-8 and F000, STUDY 3.1(a'), or to the stock aggregate) and the config key `target_H` (gen/configs.py PID and
PID_RECORD here), and restate gate G43; nothing else reads the units.

Imports: the standard library at import time (cbl.runner.validate_config uses PID_RECORD and check_record on the
Mac); Keras and hgq only inside `pid_class()`, after cbl.env.setup().
"""
from __future__ import annotations

import contextlib
import math

PID_RECORD = {'target_H': 350000, 'init_beta': 2e-08, 'p': 1.0, 'i': 0.002, 'd': 0.0, 'warmup': 10, 'log': True,
              'max_beta': 'inf', 'min_beta': 0.0, 'damping': 0.0}
SOURCES = {
    'target_H': 'STUDY 7.2 [OD-12]: Q16 "a PID run at 350,000 EBOPs" read as 350,000 H (delegated, flagged)',
    'init_beta': 'STUDY 7.2: lambda_0 = 2e-8 > 0 (model.py:295 beta0=0; beta_pid.py:125-148)',
    'p': 'beta_pid.py:99 default', 'i': 'beta_pid.py:100 default', 'd': 'beta_pid.py:101 default',
    'warmup': 'beta_pid.py:102 default', 'log': 'beta_pid.py:103 default', 'max_beta': 'beta_pid.py:104 default',
    'min_beta': 'beta_pid.py:105 default', 'damping': 'beta_pid.py:106 default (damp_beta_on_target)',
}
KWARG_OF = {'target_H': 'target_ebops', 'init_beta': 'init_beta', 'p': 'p', 'i': 'i', 'd': 'd', 'warmup': 'warmup',
            'log': 'log', 'max_beta': 'max_beta', 'min_beta': 'min_beta', 'damping': 'damp_beta_on_target'}
CALLBACK_NAME = 'LabBetaPID'          # the class name in the runner's callback order
CONFIG_NAME = 'lab:BetaPID'           # its name in a config's `callbacks` list (gen/configs.py)
SENSOR = 'headline'                   # OD-12: the PID senses the frozen headline count H (module docstring)
LOG_PREFIX = 'pid/'                   # = cbl.frozen_eval.PID_LOG_PREFIX
SENSED_KEY = LOG_PREFIX + 'sensed_H'
STOCK_KEY = LOG_PREFIX + 'stock_aggregate'
BLOCK_KEYS = ('sensed_H', 'stock_aggregate', 'beta', 'error', 'integral', 'clamp_state', 'slack', 'acted_on_H',
              'beta_unclamped')
CLAMP_STATES = ('warmup', 'none', 'max', 'min')
STATE_KIND = 'cbl1016-pid-state'
STATE_SCHEMA = 1
KEEP_BLOCKS = 2


class PIDConfigError(ValueError):
    """A `pid` config field that is not the record this campaign runs (fail closed)."""


class PIDError(RuntimeError):
    """The controller was used out of order, without its sensor, or from a state it cannot resume."""


# ----------------------------------------------------------------------------------------------
# the config's pid field (pure)


def parse_max_beta(v) -> float:
    """The config's max_beta: the JSON string "inf" (infinite, the released default) or a finite positive number."""
    if v == 'inf':
        return math.inf
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise PIDConfigError(f'max_beta must be the string "inf" or a finite positive number, not {v!r}')
    return float(v)


def controller_kwargs(rec: dict) -> dict:
    """BetaPID's constructor arguments from a `pid` record, after the checks the stock class does not make."""
    if not isinstance(rec, dict) or set(rec) != set(PID_RECORD):
        raise PIDConfigError(f'the pid field must have exactly the keys {sorted(PID_RECORD)}, not '
                             f'{sorted(rec) if isinstance(rec, dict) else rec!r}')
    num = lambda k: isinstance(rec[k], (int, float)) and not isinstance(rec[k], bool) and math.isfinite(rec[k])  # noqa: E731
    for k in ('target_H', 'init_beta', 'p', 'i', 'd', 'min_beta', 'damping'):
        if not num(k):
            raise PIDConfigError(f'pid.{k} must be a finite number, not {rec[k]!r}')
    if not isinstance(rec['target_H'], int) or rec['target_H'] <= 0:
        raise PIDConfigError(f'pid.target_H must be a positive int (H units, OD-12), not {rec["target_H"]!r}')
    if not rec['init_beta'] > 0:
        raise PIDConfigError(f'pid.init_beta must be > 0 (lambda_0; STUDY 7.2: beta0=0 reaches log10(0)), not '
                             f'{rec["init_beta"]!r}')
    if not isinstance(rec['warmup'], int) or isinstance(rec['warmup'], bool) or rec['warmup'] < 1:
        raise PIDConfigError(f'pid.warmup must be an int >= 1 (the frozen sensor has no reading before epoch 0), '
                             f'not {rec["warmup"]!r}')
    if rec['log'] is not True and rec['log'] is not False:
        raise PIDConfigError(f'pid.log must be true or false, not {rec["log"]!r}')
    if rec['i'] == 0:
        raise PIDConfigError('pid.i must be nonzero: the stock integral start at epoch == warmup divides by i '
                             '(beta_pid.py:145, :148)')
    mx = parse_max_beta(rec['max_beta'])
    if not 0 <= rec['min_beta'] < mx:
        raise PIDConfigError(f'pid.min_beta {rec["min_beta"]!r} must be >= 0 and below max_beta {rec["max_beta"]!r}')
    if not 0 <= rec['damping'] < 1:
        raise PIDConfigError(f'pid.damping must be in [0, 1), not {rec["damping"]!r}')
    out = {KWARG_OF[k]: rec[k] for k in PID_RECORD}
    out['max_beta'] = mx
    return out


def check_record(rec) -> dict:
    """The config's `pid` field must equal PID_RECORD in value and JSON type; returns controller_kwargs."""
    kw = controller_kwargs(rec)
    bad = sorted(k for k, want in PID_RECORD.items() if type(rec[k]) is not type(want) or rec[k] != want)
    if bad:
        raise PIDConfigError('the pid field differs from the record this campaign runs (STUDY 7.2): '
                             + '; '.join(f'{k} {rec[k]!r} != {PID_RECORD[k]!r}' for k in bad))
    return kw


def sensor_reader(evaluator, headline):
    """OD-12 in one place: the reading the controller acts on is the frozen headline count H of the epoch, from the
    shared cached evaluation (cbl.frozen_eval.FrozenEvaluator.headline_value). Returns read(epoch) -> int."""
    if SENSOR != 'headline':
        raise PIDConfigError(f'sensor {SENSOR!r} is not built')
    if headline not in ('H1', 'H2'):
        raise PIDConfigError(f'a PID run needs the headline H1 or H2 for its sensor (no default), not {headline!r}')

    def read(epoch):
        return evaluator.headline_value(epoch, headline)
    read.units = headline
    return read


def _hex(x):
    return None if x is None else float(x).hex()


# ----------------------------------------------------------------------------------------------
# the callback (Keras and hgq imported here)

_CLASS = None


def pid_class():
    """The LabBetaPID class, built once after cbl.env.setup() (it subclasses the installed hgq2 BetaPID)."""
    global _CLASS
    if _CLASS is not None:
        return _CLASS
    from hgq.utils.sugar.beta_pid import PID, BetaPID

    class TracedPID(PID):
        """The stock PID (beta_pid.py:8-32); records its last call, the arithmetic is the stock __call__."""

        def __init__(self, p, i, d, neg):
            super().__init__(p, i, d, neg)
            self.last = None

        def __call__(self, sv, pv):
            out = super().__call__(sv, pv)
            self.last = {'sv': sv, 'pv': pv, 'out': out}
            return out

    class LabBetaPID(BetaPID):
        """BetaPID with the frozen-H sensor and its own log keys (module docstring)."""

        def __init__(self, record, *, headline):
            kw = check_record(record)
            super().__init__(**kw)
            stock = self.pid
            self.pid = TracedPID(stock.p, stock.i, stock.d, stock.neg)
            self.record, self.kwargs, self.headline = dict(record), dict(kw), headline
            if headline not in ('H1', 'H2'):
                raise PIDConfigError(f'a PID run needs the headline H1 or H2, not {headline!r}')
            self.sensor = self.guard = None
            self._ebops = None
            self.stock_at_train_begin = None
            self.restored = None
            self._expect_epoch = None
            self.last_epoch = None
            self.model_seen = None
            self._begin = {}
            self.blocks = {}

        def bind(self, *, sensor, guard=None):
            """The sensor (sensor_reader(...)) and the stream guard, bound once the frozen evaluator exists."""
            if getattr(sensor, 'units', None) != self.headline:
                raise PIDError(f'the sensor reads {getattr(sensor, "units", None)!r}, the controller expects '
                               f'{self.headline!r}')
            self.sensor, self.guard = sensor, guard

        def signature(self) -> dict:
            """What the runner records and the gates compare: the record, the constructor arguments, the sensor."""
            kw = dict(self.kwargs)
            kw['max_beta'] = 'inf' if math.isinf(kw['max_beta']) else kw['max_beta']
            return {'class': type(self).__name__, 'base': f'{BetaPID.__module__}.{BetaPID.__qualname__}',
                    'record': dict(self.record), 'kwargs': kw, 'sensor': SENSOR, 'headline': self.headline,
                    'pid_class': type(self.pid).__name__, 'gains': {'p': self.pid.p, 'i': self.pid.i, 'd': self.pid.d,
                                                                    'neg': self.pid.neg}}

        # ---- the three hooks

        def on_train_begin(self, logs=None):
            if self.sensor is None:
                raise PIDError('LabBetaPID has no sensor bound (cbl.pid.sensor_reader through bind)')
            self.model_seen = {'id': id(self.model), 'class': f'{type(self.model).__module__}.{type(self.model).__qualname__}',
                               'n_top_layers': len(self.model.layers)}
            if self.restored is not None:
                self.set_beta(self.beta)                       # the restored lambda, on every layer
                return
            super().on_train_begin(logs)                       # beta_pid.py:125-129
            self.stock_at_train_begin = self._ebops
            self._ebops = None                                 # no frozen reading before epoch 0

        def on_epoch_begin(self, epoch, logs=None):
            if self._expect_epoch is not None:
                if epoch != self._expect_epoch:
                    raise PIDError(f'restored at epoch {self._expect_epoch - 1}; on_epoch_begin({epoch}) is not the next '
                                   'epoch')
                self._expect_epoch = None
            acting = epoch >= self.warmup
            if acting and self._ebops is None:
                raise PIDError(f'on_epoch_begin({epoch}): no frozen-H reading to act on')
            acted = self._ebops if acting else None
            self.pid.last = None
            super().on_epoch_begin(epoch, logs)                # the stock law, beta_pid.py:131-160
            if not acting:
                entry = {'acted_on_H': None, 'error': None, 'beta_unclamped': None, 'clamp_state': 'warmup'}
            else:
                out = self.pid.last['out']
                pre = 10.0 ** out if self.log else out         # restated from :150-156 to record the clamp
                if acted < self.target_ebops:
                    pre *= 1 - self.damp_beta_on_target
                if max(min(pre, self.max_beta), self.min_beta) != self.beta:
                    raise PIDError(f'epoch {epoch}: the restated clamp {pre!r} does not give the stock beta {self.beta!r}')
                state = 'max' if pre > self.max_beta else ('min' if pre < self.min_beta else 'none')
                entry = {'acted_on_H': acted, 'error': self.pid.prev_error, 'beta_unclamped': pre, 'clamp_state': state}
            entry.update(beta=self.beta, integral=self.pid.integral)
            self._begin[epoch] = entry

        def on_epoch_end(self, epoch, logs=None):
            if not isinstance(logs, dict):
                raise PIDError('LabBetaPID needs the epoch logs')
            begin = self._begin.pop(epoch, None)
            if begin is None:
                raise PIDError(f'on_epoch_end({epoch}) without on_epoch_begin({epoch})')
            clash = sorted(k for k in logs if k.startswith(LOG_PREFIX))
            if clash:
                raise PIDError(f'another callback wrote {clash}')
            logs['beta'] = self.beta                           # beta_pid.py:164
            watch = self.guard.watch('PID frozen-H sensor') if self.guard is not None else contextlib.nullcontext()
            with watch:
                sensed = self.sensor(epoch)                    # frozen H(e), the shared cached evaluation
            if isinstance(sensed, bool) or not isinstance(sensed, int):
                raise PIDError(f'the sensor returned {sensed!r}, not an int count')
            stock = self.get_ebops()                           # the stock sensor, beta_pid.py:41-47, :165
            self._ebops = sensed                               # acted on at on_epoch_begin(epoch + 1), :139, :167
            block = dict(begin, sensed_H=sensed, stock_aggregate=stock, slack=self.target_ebops - sensed)
            for k in BLOCK_KEYS:
                if block[k] is not None:
                    logs[LOG_PREFIX + k] = block[k]
            self.blocks[epoch] = block
            for e in sorted(self.blocks)[:-KEEP_BLOCKS]:
                del self.blocks[e]
            self.last_epoch = epoch

        # ---- persisted state (M14, B10)

        def state(self) -> dict:
            if self.last_epoch is None:
                raise PIDError('no epoch has ended; nothing to persist')
            st = {'kind': STATE_KIND, 'schema': STATE_SCHEMA, 'record': dict(self.record), 'headline': self.headline,
                  'sensor': SENSOR, 'epoch': self.last_epoch, 'integral': float(self.pid.integral),
                  'prev_error': float(self.pid.prev_error), 'beta': float(self.beta), 'sensed_H': self._ebops,
                  'stock_at_train_begin': self.stock_at_train_begin}
            st['hex'] = {k: _hex(st[k]) for k in ('integral', 'prev_error', 'beta', 'stock_at_train_begin')}
            return st

        def restore(self, st: dict):
            if self.last_epoch is not None or self.restored is not None:
                raise PIDError('restore into a fresh controller only')
            want = {'kind': STATE_KIND, 'schema': STATE_SCHEMA, 'record': self.record, 'headline': self.headline,
                    'sensor': SENSOR}
            bad = sorted(k for k, v in want.items() if st.get(k) != v)
            if bad:
                raise PIDError(f'the state is not this controller\'s: {bad}')
            for k in ('integral', 'prev_error', 'beta', 'stock_at_train_begin'):
                if _hex(st[k]) != st['hex'][k]:
                    raise PIDError(f'state field {k} does not match its hex record')
            if isinstance(st['sensed_H'], bool) or not isinstance(st['sensed_H'], int) or not isinstance(st['epoch'], int):
                raise PIDError('state sensed_H and epoch must be ints')
            self.pid.integral, self.pid.prev_error = float(st['integral']), float(st['prev_error'])
            self.beta, self._ebops = float(st['beta']), st['sensed_H']
            self.stock_at_train_begin = st['stock_at_train_begin']
            self.last_epoch, self._expect_epoch = st['epoch'], st['epoch'] + 1
            self.restored = dict(st)

    _CLASS = LabBetaPID
    return _CLASS


def make_pid(record, *, headline):
    """A LabBetaPID for the config's `pid` record and headline; bind its sensor before fit."""
    return pid_class()(record, headline=headline)


__all__ = ['PID_RECORD', 'SOURCES', 'CALLBACK_NAME', 'CONFIG_NAME', 'SENSOR', 'SENSED_KEY', 'STOCK_KEY', 'BLOCK_KEYS',
           'CLAMP_STATES', 'PIDConfigError', 'PIDError', 'parse_max_beta', 'controller_kwargs', 'check_record',
           'sensor_reader', 'pid_class', 'make_pid']
