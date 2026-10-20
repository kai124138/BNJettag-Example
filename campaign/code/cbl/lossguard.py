"""B10-F1: guard against penalty terms that Keras drops from a traced training step.

Keras 3.14 (installed keras/src/layers/layer.py:1251-1332) remembers the losses a layer adds inside a
StatelessScope only by Python id (`Layer._loss_ids.add(id(x))`). The set lives on the layer across calls and
traces; it is emptied only by `Layer._clear_losses()`, which every top-level layer runs when its `__call__` opens
a call context (layer.py:1680-1687) and which first removes from the scope's loss list every live loss whose id
is in that set. After a trace its tracers die and CPython reuses their addresses, so a penalty that an earlier
layer added in the current trace can carry an id that is still in a later layer's stale set; the later layer's
`_clear_losses()` then deletes it, and the traced program lacks that layer's beta * EBOPs term (in the loss and in
its gradient) for every step that program runs. Whether it happens depends on the heap of the process, so two
fresh processes of one config can train different objectives (finding B10-F1; investigation ticket
preflight/investigations/B10-F1.md). The same holds on the GPU: tracing is host Python.

`guard(model)` wraps the model instance's `stateless_call` (the trainer's forward,
keras/src/backend/jax/trainer.py:56) so that, before each call, the `_loss_ids` set of the model and of every
nested layer is emptied. Within one trace every collected loss stays alive in the scope until the trace ends, so
ids in the sets are then unique and current; a trace without a stale-id collision is unchanged, bitwise. The
wrapper also counts the terms a training call returns against the model's structure (one penalty per layer that
holds a beta variable, one regularization term per regularized trainable weight) and raises LossTermsMismatch
when they differ, so that a dropped or duplicated term stops the run instead of changing its objective.

Imported after `cbl.env.setup()` (it imports nothing at module level). Nothing here is quotable.
"""

GUARD_ATTR = '_cbl_lossguard'


class LossTermsMismatch(RuntimeError):
    """A training-mode stateless call returned a number of loss terms other than the structure implies."""


def expected_terms(model, training: bool) -> int:
    """Penalty terms (one per layer with a beta variable, added by hgq's call wrapper when training) plus the
    weight-regularization terms Keras appends (Layer.losses, layer.py:1308-1317)."""
    n_reg = sum(1 for v in model.trainable_weights if getattr(v, 'regularizer', None) is not None)
    if not training:
        return n_reg
    n_pen = sum(1 for ly in model._flatten_layers(include_self=False, recursive=True)
                if getattr(ly, 'enable_ebops', False) and getattr(ly, '_beta', None) is not None)
    return n_pen + n_reg


def guard(model, *, census: bool = True) -> dict:
    """Install the guard on `model` (idempotent). Returns a record for the run record."""
    if getattr(model, GUARD_ATTR, None) is not None:
        return dict(getattr(model, GUARD_ATTR)['record'])
    orig = model.stateless_call
    state = {'calls': 0, 'cleared_ids': 0, 'checked': 0}

    def stateless_call(*args, **kwargs):
        for ly in model._flatten_layers(include_self=True, recursive=True):
            ids = getattr(ly, '_loss_ids', None)
            if ids:
                state['cleared_ids'] += len(ids)
                ids.clear()
        state['calls'] += 1
        out = orig(*args, **kwargs)
        if census and kwargs.get('return_losses') and kwargs.get('training') is True:
            got, want = len(out[2] or []), expected_terms(model, True)
            state['checked'] += 1
            if got != want:
                raise LossTermsMismatch(f'a training call returned {got} loss terms, the model structure implies {want}')
        return out

    record = {'module': 'cbl/lossguard.py', 'finding': 'B10-F1', 'census': census,
              'expected_training_terms': expected_terms(model, True)}
    object.__setattr__(model, 'stateless_call', stateless_call)
    object.__setattr__(model, GUARD_ATTR, {'record': record, 'state': state})
    return dict(record)


def counters(model) -> dict | None:
    """The guard's counters (calls, ids cleared, training calls checked), or None without a guard. They depend on
    the process (a resumed process makes other calls), so they are not part of the run record."""
    g = getattr(model, GUARD_ATTR, None)
    return None if g is None else dict(g['state'])
