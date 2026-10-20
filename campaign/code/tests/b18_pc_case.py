"""Real-stack child for test_b18_pc; no data files, network or credentials."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbl import env
env.setup(mem_fraction='0.5', jax_platforms='cpu')
import jax
import keras
import numpy as np
from cbl import rng, model as lab, cost, frozen_eval, runner, wandb_log, lossguard
from cbl.epoch_diagnostics import EpochDiagnostics
import model as chang_model

arm, form = sys.argv[1:]
form = None if form == 'none' else form
keras.utils.set_random_seed(1)
if arm == 'CH-8':
    model = chang_model.get_model(*rng.CH8_BUILDER_ARGS)
    lossguard.guard(model)
    spec = rng.CH8_ROLE_SPEC
else:
    a = lab.arm_spec(arm, groups={g: 'B' if arm == 'F111' else 'Q' for g in ('attention', 'ffn', 'head')},
                     form=form, headline='H1')
    model = lab.builder(a)()
    lossguard.guard(model)
    spec = lab.role_spec(a)
roles = rng.roles_of(model, spec)
gen = np.random.default_rng(5)
x = gen.normal(size=(40, 8, 3)).astype('float32')
# The release compiles SparseCategoricalCrossentropy: labels are class indices.
y = (np.arange(40) % 5).astype('int32')
runner.compile_release(model)
entries = cost.layer_entries(model, spec.top_names(model))
diag = EpochDiagnostics(model, entries, spec.top_names(model), x[:19], y[:19], 8)
diag.install_train_observer()

class Capture(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.logs = dict(logs)
cap = Capture()
# Five batches exercises steps_per_execution=4 and its short tail; no Dataset shuffle draws.
model.fit(x, y, batch_size=8, epochs=1, shuffle=False, callbacks=[cap], verbose=0)
for e in diag.fb:
    assert 'b18_fb/' + e.name in cap.logs, cap.logs
    assert 0 <= cap.logs['b18_fb/' + e.name] <= 1
h0, p0, n0 = rng.variables_sha256(roles), rng.py_state_sha256(), rng.np_state_sha256()
ev = frozen_eval.FrozenEvaluator(model, roles=roles, top_names=spec.top_names(model), x_val=x[:19], y_val=y[:19], batch_size=8)
core = ev.evaluate(0)
row = diag.evaluate(core['val_top1'], cap.logs)
assert rng.variables_sha256(roles) == h0
assert rng.py_state_sha256() == p0 and rng.np_state_sha256() == n0
assert row['activations'] and row['bn'] and row['quantizers']
flat = wandb_log.flatten(row)
patterns = [k for k in wandb_log.KEYS if k.row == 'epoch' and k.pattern.startswith(
    ('quantizers/', 'bn/', 'activations/', 'quantizer_structure/', 'pruning_channel/'))]
for k in patterns:
    if k.pattern in wandb_log.NOT_PRODUCED or ('binary' in k.when and not diag.binary):
        continue
    assert any(wandb_log.matches(n, k.pattern) for n in flat), k.pattern
for e in diag.binary:
    assert set(('moment_mismatch_mean', 'moment_mismatch_var')) <= row['binary_sites'][e.name].keys()
for e in diag.fused:
    assert 0 <= row['binary_sites'][e.name]['dead_channel_frac'] <= 1
if diag.fused:
    assert -1 <= row['train_frozen_top1_gap'] <= 1
# Diagnostic training replay returns no state to the optimizer: compare exact real-step outputs.
model.optimizer.build(model.trainable_variables)
state = ([v.value for v in model.trainable_variables], [v.value for v in model.non_trainable_variables],
         [v.value for v in model.optimizer.variables], [v.value for v in model.metrics_variables])
base = type(model).train_step(model, state, (x[:8], y[:8]))
observed = model.train_step(state, (x[:8], y[:8]))
for a, b in zip(jax.tree_util.tree_leaves(base[1]), jax.tree_util.tree_leaves(observed[1])):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
print('PC producer checks passed', arm, form)
