"""M8: the frozen evaluation, its per-epoch record, the callback appended after StopIf, and the checkpoint
and export audit entry points.

STUDY means `review/snapshots/v0.10/STUDY.md` (sha256 7180c24f6b55807e...): section 3.3 (the frozen
evaluation: moving moments, current weights and quantizer parameters, H1, H2 and NR, at epoch end after
validation, mutating nothing, no layer call in training or tracing mode; the checkpoint audit must equal it
exactly; the export audit must equal the frozen evaluation of the same post-trace state), 7.1 (the appended
callback), 7.2 (the PID senses the same cached value), 8.1 (per-epoch record), 12, 13 (cost row). "plan"
means PREFLIGHT-PLAN.md (sha256 2f1be9f3...30ff8), section 2 row M8 and section 3 rows G12, G34, G35, G36.
Interface notes: docs/interfaces.md sections 1 (the per-epoch record), 10 (recorder and guard), 13 (the
runner's appended callback) and 16 (B7).

The evaluator (`FrozenEvaluator`)
- validation top-1: `model.stateless_call(trainable, non_trainable, x, training=False)` under one
  `jax.jit`, on the validation tensors the runner's validation Dataset holds (`ds_val.x`, `ds_val.y`),
  sliced in fixed 2790-row chunks in order (owner decision B7-1); the jitted function also returns whether
  every non-trainable variable it returned equals the one it was given, which must hold. The argmax of the
  logits against the labels gives an exact count; no Keras data adapter, epoch hook or metric is used.
- counts: `cbl.cost.frozen_counts` (M7) on the layers' current state: H1, H2, NR per layer and group,
  with native, penalty (b) and counter per layer;
- binary sites: per site from `cbl.binary.SCALE.record` (q_g's b and i, q_g(g), zero columns by route,
  saturated columns, sign occupancy, sign-flip rate against the previous evaluation, single-sign columns);
- a variable-hash guard: `cbl.rng.variables_sha256` over every variable by role before and after; any
  difference raises `FrozenStateChanged`. A snap that changes a constant or a width above its quantizer's
  bits raises from cbl.cost_pure; nothing is caught.
No layer is called in training or tracing mode: the only model call is the stateless inference call; the
counts read quantizer properties, `qkernel` (inference) and `_compute_ebops` (bits and shapes). Nothing
draws from the module-level `random` stream or NumPy's global state; the runner runs the evaluation inside
`StreamGuard.watch('frozen evaluation')`.

One cached value (STUDY 7.2 "Sensor timing"). `evaluate(epoch)` computes once per epoch and returns the
same object to a second caller in the same epoch after checking the variable hash, so the PID (M12, B9),
whose `on_epoch_end` runs before the appended callback's, and the callback read one evaluation. The cache
keeps the last two epochs. [B9] `cached(epoch)` returns that epoch's core after the same hash check and
refuses when the epoch has not been evaluated yet; it is how a reader that must run after the evaluation
(the front saver, M9, inside the callback's frozen-evaluation step) gets the record without evaluating.

The record (interfaces.md section 1; owner decision B7-5 for what B7 fills): the core above, and from the
callback: run_id, step, the epoch's two batch-order hashes, train_count (the int FreeEBOPs writes),
Keras's logged loss and accuracy, lr and beta. M13's B18P enrich_row adds M8's state/validation
diagnostics, M10's structure/pruning fractions, M9's cumulative dwell and M20's allocation reading
before persistence and M15 logging. Core-only audits leave these diagnostics absent. The PID block
copies the scalar pid/ log keys, preserving warmup's absent error. M15 uses strict="produced",
with only native_share_removed unresolved (NOT_PRODUCED). Released log keys are never changed.
section1_record builds the cached-core row for the front saver; M13 applies the same enrichment there.


Audits. `checkpoint_audit` reloads a saved `.keras` checkpoint in this process and runs the same
evaluation; its AUDIT fields must equal the frozen record of the epoch it was saved at (G34).
`export_audit` is the entry point the export driver (M18, B14, mulder) calls: the evaluation before and
after the trace it is given (Chang's trace on train then validation, `no_growth=False`), their difference,
and, when given the exported constants and formats per site, the counts recomputed from those, compared
with the post-trace evaluation (G35, not run in B7).

Keras and JAX are imported inside functions (cbl.env.setup() first). Nothing here is quotable.
"""
from __future__ import annotations

import contextlib
import json
import time

import numpy as np

from cbl.runner import BINARY_ENRICHMENT_FIELDS

VAL_BATCH = 2790                     # run_train.py:74, the validation Dataset's batch size
CACHE_EPOCHS = 2
LOG_KEYS = ('H1', 'H2', 'NR', 'val_correct', 'val_n', 'val_top1')
AUDIT_KEYS = ('val_correct', 'val_n', 'val_top1', 'H1', 'H2', 'NR', 'snap_changed', 'subtotals', 'vars_sha256_after')
# Enrichment is absent from core-only reloads; sign flips depend on the previous evaluation in this process.
BINARY_AUDIT_EXCLUDE = BINARY_ENRICHMENT_FIELDS | {'sign_flip_rate'}
# Core-only audits omit diagnostics. M13 enriches runner epoch rows before M9 and M15.
DEFERRED = {'quantizer_structure': 'M13 epoch enrichment (B18P)',
            'pruning_channel': 'M13 epoch enrichment (B18P)',
            'train_frozen_top1_gap': 'M13 epoch enrichment, fused-BN binary only (B18P)'}
PID_LOG_PREFIX = 'pid/'              # [B9] the lab PID's own log keys (cbl/pid.py); the record's `pid` block
HASH_KEYS = ('batch_index_seq_sha256', 'index_map_sha256')
RECORD_SCHEMA = 1


class FrozenEvalError(RuntimeError):
    """The frozen evaluation could not be made as STUDY section 3.3 states it."""


class FrozenStateChanged(FrozenEvalError):
    """A variable changed during the frozen evaluation, or between two readers of one cached value."""


def _num(x):
    import keras

    return np.asarray(keras.ops.convert_to_numpy(x))


def _logval(v):
    return v if isinstance(v, str) else float(v).hex()


class FrozenEvaluator:
    """The non-mutating evaluation of STUDY section 3.3 (module docstring)."""

    def __init__(self, model, *, roles, top_names, x_val, y_val, batch_size=VAL_BATCH):
        import jax
        import jax.numpy as jnp

        from cbl import cost

        self.model, self.roles = model, roles
        self.entries = cost.layer_entries(model, top_names)
        self.binary_entries = [e for e in self.entries if e.kind == 'binary']
        n = int(x_val.shape[0])
        if int(y_val.shape[0]) != n or n == 0:
            raise FrozenEvalError(f'validation tensors of {n} and {int(y_val.shape[0])} rows')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise FrozenEvalError(f'batch_size {batch_size!r}')
        self.x_val, self.y_val, self.batch_size, self.n_val = x_val, y_val, batch_size, n
        self.bounds = [(lo, min(lo + batch_size, n)) for lo in range(0, n, batch_size)]
        m = model

        def infer(tv, ntv, x):
            y, new_ntv = m.stateless_call(tv, ntv, x, training=False)
            same = jnp.array(True)
            for a, b in zip(ntv, new_ntv):
                same = jnp.logical_and(same, jnp.array_equal(a, b, equal_nan=True))
            return y, same

        self._infer = jax.jit(infer)
        self.cache = {}
        self._prev_signs = {}
        self.n_evaluations = 0

    # ------------------------------------------------------------------------------------------

    def _val_top1(self):
        m = self.model
        tv = [v.value for v in m.trainable_variables]
        ntv = [v.value for v in m.non_trainable_variables]
        correct, n, unchanged = 0, 0, True
        for lo, hi in self.bounds:
            y, same = self._infer(tv, ntv, self.x_val[lo:hi])
            logits = _num(y)
            lab = _num(self.y_val[lo:hi])
            if lab.ndim == 2:
                lab = np.argmax(lab, axis=-1)
            if logits.ndim != 2 or lab.shape != (hi - lo,):
                raise FrozenEvalError(f'logits {logits.shape} and labels {lab.shape} for rows {lo}:{hi}')
            correct += int(np.sum(np.argmax(logits, axis=-1) == lab))
            n += hi - lo
            unchanged = unchanged and bool(_num(same))
        if not unchanged:
            raise FrozenStateChanged('the inference call returned changed non-trainable variables')
        return correct, n

    def _binary_sites(self, records):
        from cbl import cost

        out = {}
        for e in self.binary_entries:
            rec = records[e.name]
            _, n_in, _ = cost.contraction(e.layer, e.shapes[0])
            K = np.asarray(rec['K']).reshape(n_in, -1)
            S = np.asarray(rec['S']).reshape(n_in, -1)
            n_out = K.shape[1]
            cols = lambda x: np.broadcast_to(np.asarray(x), (n_out,))          # noqa: E731
            b, q, sat = cols(rec['b']), cols(rec['q']), cols(rec['saturated'])
            zero = q == 0
            nz = K != 0
            pos = np.any(K > 0, axis=0)
            neg = np.any(K < 0, axis=0)
            prev = self._prev_signs.get(e.name)
            flip = None if prev is None or prev.shape != S.shape else float(np.mean(prev != S))
            self._prev_signs[e.name] = S.copy()
            out[e.name] = {
                'per_column': bool(rec['per_column']), 'form': rec['form'], 'penalty': rec['penalty'], 'gr': bool(rec['gr']),
                'qg_b': np.asarray(rec['b']).tolist(), 'qg_i': np.asarray(rec['i']).tolist(),
                'qg_value': np.asarray(rec['q']).tolist(),
                'zero_cols': {'pruned': int(np.sum(zero & (b == 0))), 'rounded_to_zero': int(np.sum(zero & (b > 0)))},
                'saturated_cols': int(np.sum(sat)),
                'sign_occupancy': float(np.mean(S > 0)),
                'sign_flip_rate': flip,
                'single_sign_cols': int(np.sum(np.any(nz, axis=0) & (pos != neg))),
            }
        return out

    def evaluate(self, epoch=None) -> dict:
        """The frozen evaluation of the model's current state (module docstring). With an epoch, a second
        call in the same epoch returns the cached value after checking the variable hash."""
        from cbl import binary, cost, rng

        if epoch is not None and epoch in self.cache:
            return self.cached(epoch)
        h0 = rng.variables_sha256(self.roles)
        t0 = time.time()
        correct, n = self._val_top1()
        records = {e.name: binary.SCALE.record(e.layer) for e in self.binary_entries}
        counts = cost.frozen_counts(self.entries, binary_records=records)
        sites = self._binary_sites(records)
        h1 = rng.variables_sha256(self.roles)
        if h1 != h0:
            raise FrozenStateChanged(f'the frozen evaluation changed the variables ({h0[:12]} then {h1[:12]})')
        core = {'epoch': epoch, 'val_correct': correct, 'val_n': n, 'val_top1': correct / n,
                'H1': counts['H1'], 'H2': counts['H2'], 'NR': counts['NR'], 'snap_changed': counts['snap_changed'],
                'subtotals': {'layer': counts['layers'], 'group': counts['groups']},
                'binary_sites': sites or None, 'vars_sha256_before': h0, 'vars_sha256_after': h1,
                'val_batch_size': self.batch_size, 'n_val_batches': len(self.bounds),
                'frozen_seconds': time.time() - t0}
        self.n_evaluations += 1
        if epoch is not None:
            self.cache[epoch] = core
            for k in sorted(self.cache)[:-CACHE_EPOCHS]:
                del self.cache[k]
        return core

    def cached(self, epoch) -> dict:
        """[B9] The core of this epoch's evaluation, already made, after checking that no variable changed since;
        FrozenEvalError if the epoch has not been evaluated (a reader that must follow the evaluation)."""
        from cbl import rng

        if epoch is None or epoch not in self.cache:
            raise FrozenEvalError(f'epoch {epoch} has no frozen evaluation yet (cached epochs {sorted(self.cache)})')
        core = self.cache[epoch]
        h0 = rng.variables_sha256(self.roles)
        if core['vars_sha256_after'] != h0:
            raise FrozenStateChanged(f'epoch {epoch}: the variables changed between two readers of the cached '
                                     f'evaluation ({core["vars_sha256_after"][:12]} then {h0[:12]})')
        return core

    def sign_state(self) -> dict:
        """[B10] The per-site signs of the last evaluation, against which the next `sign_flip_rate` is measured
        ({site: S array}); M14 persists it so that the first record after a resume equals the uninterrupted run's."""
        return {k: np.array(v, copy=True) for k, v in sorted(self._prev_signs.items())}

    def restore_sign_state(self, signs: dict):
        """[B10] Restore `sign_state()` into this evaluator before its first evaluation (M14)."""
        names = {e.name for e in self.binary_entries}
        if self.n_evaluations or self._prev_signs:
            raise FrozenEvalError('restore_sign_state before the first evaluation only')
        if not set(signs) <= names:
            raise FrozenEvalError(f'signs for unknown sites {sorted(set(signs) - names)}')
        self._prev_signs = {k: np.array(v, copy=True) for k, v in signs.items()}

    def headline_value(self, epoch, headline) -> int:
        """The headline count H of the epoch's (cached) evaluation; the PID's sensor (STUDY 7.2)."""
        if headline not in ('H1', 'H2'):
            raise FrozenEvalError(f'headline {headline!r}: H1 or H2, no default (interfaces.md section 5)')
        return self.evaluate(epoch)[headline]


# ----------------------------------------------------------------------------------------------
# the record and the appended callback


def pid_block(logs):
    """[B9] The lab PID's block of this epoch: its `pid/` log keys without the prefix, or None (ramp runs)."""
    out = {k[len(PID_LOG_PREFIX):]: v for k, v in sorted(logs.items()) if k.startswith(PID_LOG_PREFIX)}
    return out or None


def step_of(model):
    """The optimizer step count, or None when it cannot be read (recorded as null, never guessed)."""
    import keras

    try:
        return int(keras.ops.convert_to_numpy(model.optimizer.iterations))
    except Exception:  # noqa: BLE001
        return None


def section1_record(epoch, logs, *, evaluator, model, run_id) -> dict:
    """[B9] The interfaces.md section 1 record of this epoch, as the appended callback writes it but without the
    epoch's wall time, from the evaluation already made this epoch (`FrozenEvaluator.cached`) and the logs. For a
    reader inside the frozen-evaluation step (the front saver, cbl/runner.py); it evaluates nothing."""
    core = evaluator.cached(epoch)
    hashes = {k: logs[k] for k in HASH_KEYS}
    return epoch_row(epoch=epoch, logs=logs, core=core, hashes=hashes, step=step_of(model), run_id=run_id,
                     epoch_seconds=None)


def epoch_row(*, epoch, logs, core, hashes, step, run_id, epoch_seconds) -> dict:
    """The interfaces.md section 1 record of one epoch (module docstring)."""
    row = {'schema': RECORD_SCHEMA, 'run_id': run_id, 'epoch': int(epoch), 'step': step, **hashes,
           'log_loss': float(logs['loss']), 'log_accuracy': float(logs['accuracy']),
           'log_val_loss': float(logs['val_loss']), 'log_val_accuracy': float(logs['val_accuracy']),
           'lr': float(logs['learning_rate']), 'beta': float(logs['beta']) if 'beta' in logs else None,
           'train_count': int(logs['ebops']) if 'ebops' in logs else None,
           'sys/epoch_seconds': epoch_seconds}
    for k in DEFERRED:
        row[k] = None
    row['pid'] = pid_block(logs)
    if core is not None:
        row.update({k: core[k] for k in ('val_correct', 'val_n', 'val_top1', 'H1', 'H2', 'NR', 'snap_changed',
                                         'subtotals', 'binary_sites', 'vars_sha256_before', 'vars_sha256_after',
                                         'val_batch_size', 'n_val_batches')})
        row['sys/frozen_seconds'] = core['frozen_seconds']
    return row


COMPACT_KEYS = ('epoch', 'step', 'batch_index_seq_sha256', 'index_map_sha256', 'log_loss', 'log_accuracy', 'log_val_loss',
                'log_val_accuracy', 'lr', 'beta', 'train_count', 'val_correct', 'val_n', 'val_top1', 'H1', 'H2', 'NR',
                'vars_sha256_after', 'sys/epoch_seconds')


def make_hook(recorder, *, evaluator=None, probe=None, guard=None, wandb_run=None, epochs_path=None, run_id=None,
              train_begin=None, after_epoch=None, enrich_row=None):
    """The callback appended after StopIf (STUDY 7.1; interfaces.md section 13). It marks each Keras epoch
    in M4's batch-index recorder, writes the epoch's two batch-order hashes and LOG_KEYS as new log keys,
    runs the evaluator and then `probe(epoch, logs)`, the read-only step after it, inside
    `guard.watch('frozen evaluation')` (in the runner since B9: the front saver on `section1_record`, then a gate's
    probe), refuses if a released key changed, appends the row to `epochs_path` and logs it through M15.
    [B10] `train_begin(hook)` runs in its on_train_begin, after every released callback's and after Keras's pre-fit
    draws (only Keras's own History follows it; M14:
    the first checkpoint of a fresh run, or the restore of a resumed one); `after_epoch(epoch, logs, row)` runs as
    the last step of its on_epoch_end, after every callback of the epoch (M14: the checkpoint when one is due)."""
    import keras

    class FrozenEvalHook(keras.callbacks.Callback):
        """M8's appended callback (the class name the runner and G13 check in the callback order)."""

        def __init__(self):
            super().__init__()
            self.records = []
            self._t0 = None

        def on_train_begin(self, logs=None):
            if train_begin is not None:
                train_begin(self)

        def on_epoch_begin(self, epoch, logs=None):
            recorder.mark_keras_epoch(epoch)
            self._t0 = time.time()

        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                raise RuntimeError('FrozenEvalHook needs the epoch logs')
            ep = recorder.epoch(epoch)
            before = {k: _logval(v) for k, v in logs.items()}
            new = {'batch_index_seq_sha256': ep['batch_index_seq_sha256'], 'index_map_sha256': ep['index_map_sha256']}
            clash = sorted(set(new) & set(logs))
            if clash:
                raise RuntimeError(f'released callbacks already wrote {clash}')
            logs.update(new)
            core = None
            watch = guard.watch('frozen evaluation') if guard is not None else contextlib.nullcontext()
            with watch:
                if evaluator is not None:
                    core = evaluator.evaluate(epoch)
                if probe is not None:
                    probe(epoch, logs)
            if core is not None:
                clash = sorted(set(LOG_KEYS) & set(before))
                if clash:
                    raise RuntimeError(f'released callbacks already wrote {clash}')
                logs.update({k: core[k] for k in LOG_KEYS})
            changed = sorted(k for k, v in before.items() if k not in logs or _logval(logs[k]) != v)
            if changed:
                raise RuntimeError(f'the frozen evaluation changed released log keys {changed} (STUDY section 7.1)')
            step = step_of(self.model)
            row = epoch_row(epoch=epoch, logs=logs, core=core, hashes=new, step=step, run_id=run_id,
                            epoch_seconds=time.time() - self._t0 if self._t0 else None)
            if enrich_row is not None:
                row = enrich_row(row, logs)
            from cbl import wandb_log
            row['sys/host_rss_mb'] = wandb_log.host_peak_rss_mb()
            self.records.append({k: row.get(k) for k in COMPACT_KEYS})
            if epochs_path is not None:
                with open(epochs_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({**row, 'logs_hex': before}, sort_keys=True) + '\n')
            if wandb_run is not None:
                from cbl import wandb_log
                wrow = {k: v for k, v in row.items() if v is not None and not k.startswith('sys/') and k != 'schema'}
                wrow['sys'] = {'epoch_seconds': row['sys/epoch_seconds'], 'frozen_seconds': row.get('sys/frozen_seconds'),
                               'host_rss_mb': row['sys/host_rss_mb'], 'alloc_hours': row.get('sys/alloc_hours')}
                wandb_run.log_epoch(wrow)
            if after_epoch is not None:
                after_epoch(epoch, logs, row)

    return FrozenEvalHook()


# ----------------------------------------------------------------------------------------------
# audits


def audit_fields(core) -> dict:
    """The fields of a frozen evaluation that the checkpoint audit must reproduce exactly."""
    out = {k: core[k] for k in AUDIT_KEYS}
    bs = core.get('binary_sites')
    out['binary_sites'] = None if bs is None else {s: {k: v for k, v in r.items() if k not in BINARY_AUDIT_EXCLUDE}
                                                   for s, r in bs.items()}
    return out


def audit_mismatches(expect, got) -> list:
    """Keys of audit_fields that differ (exact comparison; JSON round-trip of both sides)."""
    a = json.loads(json.dumps(audit_fields(expect) if 'subtotals' in expect else expect, sort_keys=True))
    b = json.loads(json.dumps(audit_fields(got), sort_keys=True))
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def checkpoint_audit(ckpt_path, *, role_spec, x_val, y_val, batch_size=VAL_BATCH, expect=None) -> dict:
    """Reload a saved .keras checkpoint and run the frozen evaluation on it (STUDY 3.3 checkpoint audit).
    `expect` is the frozen record (or its audit_fields) of the epoch the checkpoint was saved at."""
    import keras

    from cbl import binary  # noqa: F401  registers the cbl1016 classes
    from cbl import rng

    model = keras.models.load_model(ckpt_path)
    roles = rng.roles_of(model, role_spec)
    ev = FrozenEvaluator(model, roles=roles, top_names=role_spec.top_names(model), x_val=x_val, y_val=y_val,
                         batch_size=batch_size)
    core = ev.evaluate()
    out = {'checkpoint': str(ckpt_path), 'checkpoint_sha256': rng.sha256_file(ckpt_path), 'audit': audit_fields(core),
           'core': core}
    if expect is not None:
        out['mismatches'] = audit_mismatches(expect, core)
        out['equal'] = not out['mismatches']
    return out


def export_audit(model, *, role_spec, trace, x_val, y_val, batch_size=VAL_BATCH, exported=None) -> dict:
    """The export audit entry point (STUDY 3.3; G35 in B14 on mulder). `trace(model)` is Chang's trace as the
    export driver runs it (train then validation, no_growth=False); `exported(model)`, when given, returns
    (sites, native_layers) for cbl.cost_pure.aggregate built from the exported constants and formats. Returns
    the evaluation before and after the trace, their differing audit fields, and the exported counts against
    the post-trace evaluation per layer."""
    from cbl import cost_pure, rng

    def run():
        roles = rng.roles_of(model, role_spec)
        return FrozenEvaluator(model, roles=roles, top_names=role_spec.top_names(model), x_val=x_val, y_val=y_val,
                               batch_size=batch_size).evaluate()

    pre = run()
    trace(model)
    post = run()
    out = {'pre': audit_fields(pre), 'post': audit_fields(post), 'trace_changed': audit_mismatches(pre, post)}
    if exported is not None:
        sites, natives = exported(model)
        agg = cost_pure.aggregate(sites, natives)
        layers = post['subtotals']['layer']
        diff = sorted(n for n, v in agg['layers'].items()
                      if n not in layers or any(v[c] != layers[n][c] for c in ('H1', 'H2', 'NR')))
        out['exported'] = {'H1': agg['H1'], 'H2': agg['H2'], 'NR': agg['NR'], 'differing_layers': diff,
                           'missing_layers': sorted(set(layers) - set(agg['layers']))}
        out['exported_equal'] = (not diff and not out['exported']['missing_layers']
                                 and all(agg[c] == post[c] for c in ('H1', 'H2', 'NR')))
    return out


__all__ = ['FrozenEvaluator', 'FrozenEvalError', 'FrozenStateChanged', 'make_hook', 'epoch_row', 'checkpoint_audit',
           'export_audit', 'audit_fields', 'audit_mismatches', 'VAL_BATCH', 'LOG_KEYS', 'AUDIT_KEYS', 'DEFERRED',
           'pid_block', 'step_of', 'section1_record', 'PID_LOG_PREFIX', 'HASH_KEYS']
