"""Child process of gates G10 (F000 part), G11 (across arms), G13, G14 and G15 (modules M13
cbl/runner.py and M5 cbl/model.py), and since B7 of the runner parts of G12, G34 and G36 (module M8
cbl/frozen_eval.py). Gate code, not campaign runner code.

Each invocation is one fresh process, as in a pod, so Keras's name counters and the global random
streams start from nothing. It observes and writes one JSON record; the parent gate compares.

    draw   CH-8's initial-tensor file of a seed on this device (cbl.runner.draw, decision U1)
    run    the runner (prepare, fit, finish) for one arm with an engineering config built here; M8's
           frozen evaluation runs in the appended callback (since B7), followed by a read-only probe
           (`Probe`, below; --no-probe leaves it out). B7 options: --save-ckpt saves a .keras checkpoint
           at every epoch end after the evaluation (G34); --spy-frozen records the `training` argument of
           every hgq Quantizer call made while M8 evaluates (G34); --decoy-frozen makes M8's evaluation
           draw from the module-level random stream first (G12 control); the full per-epoch records
           (`epochs.jsonl`) are returned as `frozen_rows`
    readings  the runner's prepare (no fit), then M8's evaluation of the initial state (G36)
    step   the runner's prepare (data, seeds, Datasets, model, initial tensors, compile), then the
           probe at the initial state, one `train_on_batch` on a fixed batch, and the probe again;
           no fit (G14 "one optimizer step")
    ref    Chang's `run_train.py` executed by runpy as `__main__`, byte-unchanged, with this harness
           (G13 reference side): the smoke-only device remap of preflight/env/README.md on the CPU
           (`hgq.utils.sugar.Dataset` rebound to a factory that passes 'cpu:0' for 'gpu:0'; on the GPU
           the factory passes the device through), M4's batch-index recorder on the training Dataset,
           and a wrapper of `keras.Model.fit` that sets the gate's epochs in place of 7000, appends
           the runner's frozen-evaluation callback with the same probe but without M8's evaluator (the
           release does not evaluate, so equality with the runner, which does, also shows that the
           evaluation changes nothing in training), and (unless --no-reseed) applies decision U5's
           pre-fit reseed just before the real `fit`, as the runner does
    mha    G15 in one process: CH-8, F000 and CH-8 again; nested MHA projections by role; the
           single wrap of the HGQ call and build; negative and positive controls of the wrap check;
           since B6 also F100, F111-FB and BE-BF (binary MHA, binary projections by role)
    refuse G10 (B6): a binary arm built by M5 after section 6 steps 1-3 refuses CH-8's file under a
           RoleSpec that declares empty unmatched and unused sets, assigning nothing, then loads it
           under its own RoleSpec

The probe is non-mutating: at each epoch end it hashes every variable by role, the optimizer
variables, the gradient on a fixed training batch of the release's loss (sparse categorical
cross-entropy from logits, `run_train.py:99`, plus the model's own losses: the EBOPs penalty and the
regularizers) through `model.stateless_call(..., training=True, return_losses=True)` under
`jax.value_and_grad`, the updated non-trainable state that call would write (BN moving statistics,
per-layer EBOPs, WRAP integer bits), the training-mode logits, the inference logits on the whole
validation set, the per-layer EBOPs counters and the random states; it checks that the variables
and both streams are unchanged afterwards. It does not use the compiled loss or metrics, so it
builds nothing the first `fit` would otherwise build (which would change Keras's pre-fit draws).
Every number is engineering, not quotable.
"""

import argparse
import json
import os
import random
import re
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))

PROBE_BATCH = 2790          # the first 2790 rows of the training array, a fixed batch
WRAP_FILE = 'hgq/layers/core/base.py'


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n', encoding='utf-8')


def logval(v):
    return v if isinstance(v, str) else float(v).hex()


def ser(obj):
    """A JSON-ready, address-free form of a layer or quantizer configuration."""
    if isinstance(obj, dict):
        return {str(k): ser(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [ser(x) for x in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, 'get_config'):
        try:
            return {'class': type(obj).__name__, 'config': ser(obj.get_config())}
        except NotImplementedError:
            pass
    return re.sub(r' at 0x[0-9a-f]+', '', repr(obj))


BINARY_GATE_ARMS = ('F100', 'F001', 'F010', 'F011', 'F101', 'F110', 'F111', 'GR', 'BE')


def lab_fields(arm, form):
    """groups, embedding, control and config_code of a lab arm (interfaces.md section 5)."""
    if arm == 'F000':
        return {'attention': 'Q', 'ffn': 'Q', 'head': 'Q'}, 'Q', None, 'F000'
    if arm in ('GR', 'BE'):
        return {'attention': 'B', 'ffn': 'B', 'head': 'B'}, ('B' if arm == 'BE' else 'Q'), arm, 'F111'
    groups = {g: ('B' if c == '1' else 'Q') for g, c in zip(('attention', 'ffn', 'head'), arm[1:])}
    return groups, 'Q', None, arm


def engineering_config(*, arm, seed, epochs, code_sha, data_manifest_sha256, init_manifest_path, init_manifest_sha256,
                       wandb_mode='disabled', stage='gate', form=None, headline=None):
    """An interfaces.md section 5 config for an engineering run of CH-8, F000 or a binary arm (gate code;
    the config generator is M17, batch B12). A binary arm carries its form and the headline that selects
    its penalty (b) branch; CH-8's and F000's configs are as in B5."""
    from cbl import rng, runner
    from cbl import wandb_log

    lab = arm != 'CH-8'
    cfg = {'schema_version': 1, 'campaign': runner.CAMPAIGN, 'arm_id': arm, 'builder': 'lab' if lab else 'released',
           'config_code': 'F000' if lab else None, 'form': None, 'control': None, 'seed': seed, 'stage': stage,
           'tier': None, 'schedule': 'ramp', 'ramp': runner.RAMP_RECORD, 'epochs': epochs, 'cost_callbacks': True,
           'headline': None, 'claim_domain': None, 's4_option': None, 'matmul_precision': 'float32',
           'init_tensors': {'manifest_path': str(init_manifest_path), 'manifest_sha256': init_manifest_sha256},
           'prefit_reseed': rng.prefit_reseed(seed), 'data_manifest_sha256': data_manifest_sha256, 'code_sha': code_sha,
           'param_count': None, 'attempt': 1,
           'wandb': {'project': wandb_log.PROJECT, 'group': wandb_log.GROUP, 'run_name': None, 'mode': wandb_mode}}
    if lab:
        groups, emb, control, code = lab_fields(arm, form)
        cfg.update(groups=groups, embedding=emb, control=control, config_code=code)
    if arm in BINARY_GATE_ARMS:
        cfg.update(form=form, headline=headline, form_source='fixed' if form else None)
    return cfg


# ----------------------------------------------------------------------------------------------
# the probe


class Probe:
    """Read-only snapshot of a model's state, its training-step gradient and its outputs."""

    def __init__(self, model, roles, ds_train, ds_val):
        import jax
        import keras

        from cbl import lossguard

        # Also covers the released fit interception; runner models are already guarded.
        lossguard.guard(model)
        self.model, self.roles = model, roles
        self.role_of = {id(v): r for r, v in roles.items()}
        self.xb, self.yb = ds_train.x[:PROBE_BATCH], ds_train.y[:PROBE_BATCH]
        self.xv, self.yv = ds_val.x, ds_val.y
        self.rows = []
        m = model
        loss_obj = keras.losses.SparseCategoricalCrossentropy(from_logits=True)       # run_train.py:99

        def lossfn(tv, ntv, x, y):
            y_pred, new_ntv, losses = m.stateless_call(tv, ntv, x, training=True, return_losses=True)
            data_loss = loss_obj(y, y_pred)
            total = data_loss + sum(losses) if losses else data_loss
            return total, (data_loss, y_pred, new_ntv, len(losses))

        def grads(tv, ntv, x, y):
            (total, aux), g = jax.value_and_grad(lossfn, has_aux=True)(tv, ntv, x, y)
            return total, aux, g

        self._grad = jax.jit(grads)
        self._infer = jax.jit(lambda tv, ntv, x: m.stateless_call(tv, ntv, x, training=False)[0])

    def snapshot(self):
        import keras
        import numpy as np
        from cbl import rng

        m = self.model
        sha = rng.tensor_sha256
        before = rng.variables_sha256(self.roles)
        py0, np0 = rng.py_state_sha256(), rng.np_state_sha256()
        tv = [v.value for v in m.trainable_variables]
        ntv = [v.value for v in m.non_trainable_variables]
        loss, aux, g = self._grad(tv, ntv, self.xb, self.yb)
        data_loss, y_pred, new_ntv, n_losses = aux
        yv = np.asarray(keras.ops.convert_to_numpy(self._infer(tv, ntv, self.xv)), dtype=np.float32)
        arrays = rng.variable_arrays(self.roles)
        out = {
            'variables': {r: sha(a) for r, a in arrays.items()},
            'variables_sha256': before,
            'ebops': {r: int(a) for r, a in arrays.items() if r.endswith('/ebops')},
            'bn_moving': {r: sha(a) for r, a in arrays.items() if r.endswith('/moving_mean') or r.endswith('/moving_var')},
            'optimizer': [sha(np.asarray(keras.ops.convert_to_numpy(v.value))) for v in m.optimizer.variables] if m.optimizer else [],
            'optimizer_iterations': int(keras.ops.convert_to_numpy(m.optimizer.iterations)) if m.optimizer else None,
            'loss': float(np.asarray(loss)).hex(), 'data_loss': float(np.asarray(data_loss)).hex(),
            'n_model_losses': int(n_losses),
            'train_logits_sha256': sha(np.asarray(keras.ops.convert_to_numpy(y_pred), dtype=np.float32)),
            'grads': {self.role_of[id(v)]: sha(np.asarray(gi)) for v, gi in zip(m.trainable_variables, g)},
            'step_non_trainable': {self.role_of[id(v)]: sha(np.asarray(x)) for v, x in zip(m.non_trainable_variables, new_ntv)},
            'step_ebops': {self.role_of[id(v)]: int(np.asarray(x)) for v, x in zip(m.non_trainable_variables, new_ntv)
                           if self.role_of[id(v)].endswith('/ebops')},
            'infer_logits_sha256': sha(yv),
            'infer_correct': int(np.sum(np.argmax(yv, axis=-1) == np.asarray(keras.ops.convert_to_numpy(self.yv)))),
            'infer_n': int(yv.shape[0]),
        }
        out['free_ebops'] = free_ebops(m)
        out['unchanged'] = rng.variables_sha256(self.roles) == before
        out['random_unchanged'] = rng.py_state_sha256() == py0 and rng.np_state_sha256() == np0
        out['py_state_sha256'] = py0
        if not (out['unchanged'] and out['random_unchanged']):
            raise RuntimeError(f'the probe changed state: variables {out["unchanged"]}, streams {out["random_unchanged"]}')
        return out

    def __call__(self, epoch, logs):
        row = {'epoch': int(epoch), 'logs': {k: logval(v) for k, v in logs.items()}}
        row.update(self.snapshot())
        self.rows.append(row)


def free_ebops(model):
    """FreeEBOPs's sum (hgq/utils/sugar/ebops.py:8-15) over the flattened layers, as an int."""
    import keras
    from hgq.layers import QLayerBase

    return int(sum(int(keras.ops.convert_to_numpy(ly._ebops)) for ly in model._flatten_layers()
                   if isinstance(ly, QLayerBase) and ly.enable_ebops))


def array_sha(x):
    import keras
    import numpy as np
    from cbl import rng

    a = np.asarray(keras.ops.convert_to_numpy(x))
    return {'sha256': rng.tensor_sha256(a), 'dtype': str(a.dtype), 'shape': list(a.shape)}


def structure(model, roles):
    """Layer classes and configurations (address-free; names removed) in `_flatten_layers` order,
    and each role's shape, dtype and trainable flag."""
    import keras

    layers = []
    for ly in model._flatten_layers(include_self=False):
        try:
            cfg = ly.get_config()
            cfg.pop('name', None)
            c = ser(cfg)
        except NotImplementedError:
            c = None
        layers.append({'class': type(ly).__name__, 'config': c})
    trainable = {id(v) for v in model.trainable_variables}
    vars_ = {r: {'shape': list(v.shape), 'dtype': keras.backend.standardize_dtype(v.dtype), 'trainable': id(v) in trainable}
             for r, v in roles.items()}
    return {'layers': layers, 'top_classes': [type(ly).__name__ for ly in model.layers], 'variables': vars_,
            'n_layers': len(layers), 'n_without_config': sum(1 for x in layers if x['config'] is None)}


# ----------------------------------------------------------------------------------------------
# commands


def cmd_draw(a, rec):
    from cbl import runner

    mpath, msha, man, flow = runner.draw(a.seed, train_archive=a.archive, archive_kind='smoke', device=a.device,
                                         out_dir=a.out_dir, place=a.place)
    rec.update(manifest=str(mpath), manifest_sha256=msha, file=man['file'], identity=man['identity'],
               probe_forward=man['probe_forward'], n_roles=man['n_roles'], flow=flow.record(),
               index_map_at_construction=list(flow.ds_train._index_map))


def _cfg(a):
    return engineering_config(arm=a.arm, seed=a.seed, epochs=a.epochs, code_sha=a.code_sha,
                              data_manifest_sha256=a.data_manifest_sha, init_manifest_path=a.manifest,
                              init_manifest_sha256=a.manifest_sha, wandb_mode=a.wandb_mode, form=a.form,
                              headline=a.headline)


def _context_record(ctx):
    import keras  # noqa: F401
    fl = ctx.flow
    return {'data_arrays': {k: array_sha(x) for k, x in (('X_train', fl.ds_train.x), ('y_train', fl.ds_train.y),
                                                         ('X_val', fl.ds_val.x), ('y_val', fl.ds_val.y))},
            'datasets': {'train_init_shift': fl.ds_train._init_shift, 'val_init_shift': fl.ds_val._init_shift,
                         'n_batches': len(fl.ds_train), 'val_n_batches': len(fl.ds_val),
                         'dataset_device': fl.ds_train.device},
            'structure': structure(ctx.model, ctx.roles)}


class FrozenSpy:
    """While M8's FrozenEvaluator.evaluate runs, record the `training` argument of every hgq Quantizer call
    (trace-time calls under jit included) and nothing else; results unchanged."""

    def __init__(self):
        self.calls, self.active, self.evaluations = [], False, 0

    def install(self):
        from hgq.quantizer import Quantizer

        from cbl import frozen_eval

        spy = self
        q_call = Quantizer.call
        ev = frozen_eval.FrozenEvaluator.evaluate

        def call(qself, inputs, training=None):
            if spy.active:
                spy.calls.append(repr(training))
            return q_call(qself, inputs, training=training)

        def evaluate(eself, *args, **kw):
            spy.active = True
            spy.evaluations += 1
            try:
                return ev(eself, *args, **kw)
            finally:
                spy.active = False
        Quantizer.call = call
        frozen_eval.FrozenEvaluator.evaluate = evaluate

    def record(self):
        from collections import Counter
        return {'evaluations': self.evaluations, 'quantizer_calls': len(self.calls), 'training_values': dict(Counter(self.calls))}


def install_decoy_frozen():
    """G12 control: M8's evaluation draws from the module-level random stream before it runs."""
    from cbl import frozen_eval

    ev = frozen_eval.FrozenEvaluator.evaluate

    def evaluate(eself, *args, **kw):
        random.random()
        return ev(eself, *args, **kw)
    frozen_eval.FrozenEvaluator.evaluate = evaluate


class CkptSaver:
    """A read-only probe that saves the model as a .keras checkpoint at every epoch end (G34)."""

    def __init__(self, model, out_dir):
        self.model, self.dir, self.saved = model, Path(out_dir), []
        self.dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, epoch, logs):
        from cbl import rng
        p = self.dir / f'epoch-{int(epoch)}.keras'
        self.model.save(p)
        self.saved.append({'epoch': int(epoch), 'path': str(p), 'sha256': rng.sha256_file(p)})


def cmd_run(a, rec):
    from cbl import runner

    cfg = _cfg(a)
    holder = {}
    spy = FrozenSpy() if a.spy_frozen else None

    def factory(ctx):
        # called by prepare after cbl.env.setup() and compile, before the reseed and fit: hgq may be imported here
        if spy is not None:
            spy.install()
        if a.decoy_frozen:
            install_decoy_frozen()
        holder['ctx_record'] = _context_record(ctx)
        probes = []
        if not a.no_probe:
            holder['probe'] = Probe(ctx.model, ctx.roles, ctx.flow.ds_train, ctx.flow.ds_val)
            holder['initial'] = holder['probe'].snapshot()
            probes.append(holder['probe'])
        if a.save_ckpt:
            holder['saver'] = CkptSaver(ctx.model, a.save_ckpt)
            probes.append(holder['saver'])
        if not probes:
            return None

        def probe(epoch, logs):
            for p in probes:
                p(epoch, logs)
        return probe

    code, ctx = runner.run(cfg, train_archive=a.archive, archive_kind='smoke', device=a.device, out_dir=a.out_dir,
                           source_bundle=a.source_bundle, evaluator_factory=factory, keep_full=None)
    r = ctx.record
    ep = Path(a.out_dir) / 'epochs.jsonl'
    rec['frozen_rows'] = [json.loads(x) for x in ep.read_text(encoding='utf-8').splitlines()] if ep.exists() else None
    rec['frozen_evaluation'] = r.get('frozen_evaluation')
    rec['frozen_spy'] = spy.record() if spy is not None else None
    rec['checkpoints'] = holder['saver'].saved if 'saver' in holder else None
    rec['runner'] = {k: r.get(k) for k in ('end', 'exit', 'refused', 'error', 'traceback', 'violation', 'config_sha256',
                                           'summary', 'code', 'release', 'archive', 'load', 'compile', 'callback_order',
                                           'callback_signature', 'reseed', 'index_map_at_reseed', 'epochs_completed',
                                           'epoch_records', 'matmul_precision_before_fit', 'matmul_precision_after_fit',
                                           'guard', 'recorder', 'flow', 'final_variables_sha256', 'keras_version',
                                           'wandb', 'dataset_device', 'device')}
    rec['config'] = cfg
    rec['initial'] = holder.get('initial')
    rec['context'] = holder.get('ctx_record')
    rec['probe_rows'] = holder['probe'].rows if 'probe' in holder else None
    rec['out_files'] = sorted(str(p.relative_to(a.out_dir)) for p in Path(a.out_dir).rglob('*') if p.is_file())[:200]
    return code


def cmd_step(a, rec):
    from cbl import runner

    cfg = _cfg(a)
    ctx = runner.Context()
    runner.prepare(cfg, ctx=ctx, train_archive=a.archive, archive_kind='smoke', device=a.device, out_dir=a.out_dir,
                   source_bundle=a.source_bundle, keep_full=None)
    import keras                                       # after prepare: cbl.env.setup() sets the backend first
    import numpy as np
    probe = Probe(ctx.model, ctx.roles, ctx.flow.ds_train, ctx.flow.ds_val)
    rec['load'] = ctx.record['load']
    rec['context'] = _context_record(ctx)
    rec['callback_order'] = ctx.record['callback_order']
    rec['compile'] = ctx.record['compile']
    rec['init'] = probe.snapshot()
    xb, yb = ctx.flow.ds_train.x[:PROBE_BATCH], ctx.flow.ds_train.y[:PROBE_BATCH]
    logs = ctx.model.train_on_batch(xb, yb, return_dict=True)
    rec['step_logs'] = {k: logval(np.asarray(keras.ops.convert_to_numpy(v))) for k, v in logs.items()}
    rec['after_step'] = probe.snapshot()
    ctx.wandb_run.finish(exit_code=0)
    return 0


def cmd_readings(a, rec):
    """G36: the runner's prepare (no fit), then M8's frozen evaluation of the initial state."""
    from cbl import runner

    cfg = _cfg(a)
    ctx = runner.Context()
    runner.prepare(cfg, ctx=ctx, train_archive=a.archive, archive_kind='smoke', device=a.device, out_dir=a.out_dir,
                   source_bundle=a.source_bundle, keep_full=None)
    core = ctx.evaluator.evaluate()
    rec['config'] = cfg
    rec['load'] = {k: ctx.record['load'][k] for k in ('manifest_sha256', 'n_loaded', 'unmatched', 'unused', 'loaded_sha256')}
    rec['core'] = {k: core[k] for k in ('H1', 'H2', 'NR', 'snap_changed', 'val_correct', 'val_n', 'vars_sha256_before',
                                        'vars_sha256_after')}
    rec['groups'] = core['subtotals']['group']
    rec['layers'] = {n: {k: v[k] for k in ('H1', 'H2', 'NR', 'native', 'kind', 'group')} for n, v in core['subtotals']['layer'].items()}
    rec['binary_sites'] = core['binary_sites']
    rec['frozen_seconds'] = core['frozen_seconds']
    ctx.wandb_run.finish(exit_code=0)
    return 0


def cmd_ref(a, rec):
    """Chang's run_train.py under runpy (module docstring)."""
    from cbl import env

    st = env.setup(mem_fraction='0.5', jax_platforms='cpu' if a.device == 'cpu' else 'cuda')
    rec['setup_precision'] = st['matmul_precision']
    import runpy

    import hgq.utils.sugar as hgq_sugar
    import keras
    from cbl import rng, runner

    rt = FORK / 'chang' / 'run_train.py'
    rec['run_train_sha256'] = runner.sha256_file(rt)
    if rec['run_train_sha256'] != runner.code_base_sha256('chang/run_train.py'):
        raise RuntimeError('run_train.py differs from code-base.txt')
    real_dataset = hgq_sugar.Dataset
    remap = a.device == 'cpu'
    made = []
    state = {}

    def dataset_factory(x_set, y_set=None, batch_size=None, device='cpu:0', *args, **kw):
        used = 'cpu:0' if (remap and device == 'gpu:0') else device
        ds = real_dataset(x_set, y_set, batch_size, used, *args, **kw)
        entry = {'device_asked': device, 'device_used': used, 'shuffle': bool(ds.shuffle), 'batch_size': batch_size,
                 'init_shift': ds._init_shift, 'index_map': None if ds._index_map is None else list(ds._index_map),
                 'random_state_sha256_after': rng.py_state_sha256()}
        if ds.shuffle:
            state['recorder'] = rng.instrument(ds)
        made.append(entry)
        return ds

    hgq_sugar.Dataset = dataset_factory
    orig_fit = keras.Model.fit

    def fit(self, x=None, *args, **kw):
        state['release_epochs'] = kw.get('epochs')
        cbs = list(kw.get('callbacks') or [])
        state['release_order'] = [type(c).__name__ for c in cbs]
        state['callback_signature'] = runner.callback_signature(cbs)
        roles = rng.roles_of(self, rng.CH8_ROLE_SPEC)
        state['initial_variables'] = {r: rng.tensor_sha256(v) for r, v in rng.variable_arrays(roles).items()}
        state['random_state_sha256_after_build_and_compile'] = rng.py_state_sha256()
        ds_val = kw.get('validation_data')
        probe = Probe(self, roles, x, ds_val)
        state['probe'] = probe
        state['context'] = {'data_arrays': {k: array_sha(t) for k, t in (('X_train', x.x), ('y_train', x.y),
                                                                          ('X_val', ds_val.x), ('y_val', ds_val.y))},
                            'datasets': {'train_init_shift': x._init_shift, 'val_init_shift': ds_val._init_shift,
                                         'n_batches': len(x), 'val_n_batches': len(ds_val), 'dataset_device': x.device},
                            'structure': structure(self, roles)}
        state['initial'] = probe.snapshot()
        hook = runner.frozen_eval_hook(state['recorder'], probe=probe, epochs_path=Path(a.out_dir) / 'epochs.jsonl')
        state['hook'] = hook
        state['compile'] = {'optimizer': type(self.optimizer).__name__, 'optimizer_config': self.optimizer.get_config(),
                            'loss': self.loss.get_config(), 'steps_per_execution': self.steps_per_execution}
        kw['callbacks'] = cbs + [hook]
        kw['epochs'] = a.epochs
        state['fit_order'] = [type(c).__name__ for c in kw['callbacks']]
        if a.no_reseed:
            state['reseed'] = None
        else:
            state['reseed'] = rng.prefit_reseed(a.seed)
            random.seed(state['reseed'])                                     # decision U5, as the runner does
        state['index_map_at_fit'] = list(x._index_map)
        state['py_state_sha256_at_fit'] = rng.py_state_sha256()
        return orig_fit(self, x, *args, **kw)

    keras.Model.fit = fit
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    sys.argv = [str(rt), '-i', a.ref_data_dir, '-o', str(out / 'chang_output'), '-n', '8', '-f', '3', '-m', 'xfm']
    rec['run_train_argv'] = list(sys.argv)
    g = runpy.run_path(str(rt), run_name='__main__')
    hgq_sugar.Dataset = real_dataset
    keras.Model.fit = orig_fit
    rec['datasets_made'] = made
    rec['release_epochs'] = state['release_epochs']
    rec['release_order'] = state['release_order']
    rec['fit_order'] = state['fit_order']
    rec['callback_signature'] = state['callback_signature']
    rec['compile'] = state['compile']
    rec['initial_variables'] = state['initial_variables']
    rec['initial'] = state['initial']
    rec['context'] = state['context']
    rec['reseed'] = state['reseed']
    rec['index_map_at_fit'] = state['index_map_at_fit']
    rec['py_state_sha256_at_fit'] = state['py_state_sha256_at_fit']
    rec['random_state_sha256_after_build_and_compile'] = state['random_state_sha256_after_build_and_compile']
    rec['probe_rows'] = state['probe'].rows
    rec['epoch_records'] = state['hook'].records
    rec['recorder'] = state['recorder'].summary()
    rec['matmul_precision_after_fit'] = env.matmul_precision()
    rec['stop_training'] = bool(g['model'].stop_training)
    rec['args'] = vars(g['args'])
    rec['len_dataset_train'] = len(g['dataset_train'])
    rec['final_variables_sha256'] = rng.variables_sha256(rng.roles_of(g['model'], rng.CH8_ROLE_SPEC))
    rec['keras_version'] = keras.__version__
    return 0


def wrap_depth(fn, code):
    d = 0
    while fn is not None and getattr(fn, '__code__', None) is code:
        d += 1
        fn = getattr(fn, '__wrapped__', None)
    return d, fn


def wrap_record(classes):
    """For each QLayerMeta class: how many QLayerMeta wrappers its call and build carry, and how
    many they should carry (base.py's own condition, applied to the innermost function)."""
    from hgq.layers.core.base import QLayerBase, QLayerMeta
    from keras.layers import Layer

    consts = {c.co_name: c for c in QLayerMeta.__call__.__code__.co_consts if hasattr(c, 'co_name')}
    call_code, build_code = consts['call'], consts['build']
    out = {}
    for cls in sorted(classes, key=lambda c: c.__qualname__):
        dc, inner_call = wrap_depth(cls.call, call_code)
        db, _ = wrap_depth(cls.build, build_code)
        eligible = (inner_call is not Layer.call and cls._compute_ebops is not QLayerBase._compute_ebops
                    and not getattr(cls, '__no_wrap_call__', False))
        out[cls.__qualname__] = {'call_wraps': dc, 'call_wraps_expected': 1 if eligible else 0, 'build_wraps': db,
                                 'build_wraps_expected': 1, 'in_wrapped_cls': cls in QLayerMeta._wrapped_cls,
                                 'module': cls.__module__,
                                 'call_code_file': call_code.co_filename.endswith(WRAP_FILE)}
    return out


def cmd_mha(a, rec):
    from cbl import env

    env.setup(mem_fraction='0.5', jax_platforms='cpu')
    import keras
    import model as chang_model
    from hgq.layers import QEinsumDense
    from hgq.layers.core.base import QLayerBase, QLayerMeta
    from cbl import model as lab
    from cbl import rng

    PROJ = (('_query_dense', 'query'), ('_key_dense', 'key'), ('_value_dense', 'value'), ('_output_dense', 'attention_output'))
    models = {}
    for name, build, spec in (('CH-8', lambda: chang_model.get_model(*rng.CH8_BUILDER_ARGS), rng.CH8_ROLE_SPEC),
                              ('F000', lab.builder(lab.F000_SPEC), lab.role_spec(lab.F000_SPEC)),
                              ('CH-8 again', lambda: chang_model.get_model(*rng.CH8_BUILDER_ARGS), rng.CH8_ROLE_SPEC)):
        random.seed(1)
        keras.utils.set_random_seed(1)
        m = build()
        roles = rng.roles_of(m, spec)
        mha = m.layers[3]
        proj = {}
        for attr, seg in PROJ:
            sub = getattr(mha, attr)
            kr = f'mha/{seg}/kernel'
            br = f'mha/{seg}/bias'
            proj[seg] = {'attr': attr, 'layer_name': sub.name, 'class': type(sub).__name__,
                         'kernel_role_is_attr_kernel': roles.get(kr) is sub.kernel,
                         'kernel_shape': list(sub.kernel.shape), 'has_bias_role': br in roles,
                         'bias_role_is_attr_bias': (roles.get(br) is sub.bias) if br in roles else None,
                         'roles': sorted(r for r in roles if r.startswith(f'mha/{seg}/')),
                         'kq_config_is_mha_conf': sub.kq.config is mha._qkvo_kq_conf,
                         'kq_config': ser(sub.kq.config.get_config()),
                         'iq_config_is_mha_conf': (sub.iq.config is mha._qkvo_iq_conf) if sub.enable_iq else None,
                         'bq_config_is_mha_conf': (sub.bq.config is mha._qkvo_bq_conf) if getattr(sub, 'bq', None) is not None else None,
                         'oq_config_is_mha_conf': (sub.oq.config is mha._qkvo_oq_conf) if sub.enable_oq else None}
        mha_roles = sorted(r for r in roles if r.startswith('mha/'))
        x = keras.random.normal((4, 8, 3), seed=0)
        tv = [v.value for v in m.trainable_variables]
        ntv = [v.value for v in m.non_trainable_variables]
        _, _, losses = m.stateless_call(tv, ntv, x, training=True, return_losses=True)
        classes = {type(ly) for ly in m._flatten_layers() if isinstance(ly, QLayerBase)}
        models[name] = {'mha_class': type(mha).__name__, 'projections': proj, 'mha_roles': mha_roles,
                        'n_roles': len(roles), 'wraps': wrap_record(classes), 'n_losses_training_call': len(losses),
                        'mha_qkvo_kq_conf': ser(mha._qkvo_kq_conf.get_config()),
                        'top_layer_names': [ly.name for ly in m.layers]}
    rec['models'] = models
    # B6: binary attention (decision U3) in three binary arms
    from cbl import binary as B
    bmodels = {}
    for label, arm, form, hl in (('F100', 'F100', None, 'H1'), ('F111-FB', 'F111', 'FB', 'H1'), ('BE-BF', 'BE', 'BF', 'H2')):
        groups, emb, control, _ = lab_fields(arm, form)
        spec = lab.arm_spec(arm, groups=groups, embedding=emb, form=form, control=control, headline=hl)
        random.seed(1)
        keras.utils.set_random_seed(1)
        m = lab.build(spec)
        roles = rng.roles_of(m, lab.role_spec(spec))
        mha = m.layers[3]
        proj = {}
        for attr, seg in PROJ:
            sub = getattr(mha, attr)
            kr, br = f'mha/{seg}/kernel', f'mha/{seg}/bias'
            proj[seg] = {'attr': attr, 'class': type(sub).__name__, 'kernel_role_is_attr_kernel': roles.get(kr) is sub.kernel,
                         'kernel_shape': list(sub.kernel.shape), 'has_bias_role': br in roles,
                         'bias_role_is_attr_bias': (roles.get(br) is sub.bias) if br in roles else None,
                         'roles': sorted(r for r in roles if r.startswith(f'mha/{seg}/')),
                         'has_kq_attr': '_kq' in sub.__dict__,
                         'qg_config_is_mha_conf': sub.qg.config is mha._qkvo_kq_conf,
                         'qg_config': ser(sub.qg.config.get_config()),
                         'iq_config_is_mha_conf': (sub.iq.config is mha._qkvo_iq_conf) if sub.enable_iq else None,
                         'bq_config_is_mha_conf': (sub.bq.config is mha._qkvo_bq_conf) if getattr(sub, 'bq', None) is not None else None,
                         'oq_config_is_mha_conf': (sub.oq.config is mha._qkvo_oq_conf) if sub.enable_oq else None,
                         'penalty': sub.penalty, 'gr': sub.gr}
        x = keras.random.normal((4, 8, 3), seed=0)
        tv = [v.value for v in m.trainable_variables]
        ntv = [v.value for v in m.non_trainable_variables]
        _, _, losses = m.stateless_call(tv, ntv, x, training=True, return_losses=True)
        classes = {type(ly) for ly in m._flatten_layers() if isinstance(ly, QLayerBase)}
        bmodels[label] = {'mha_class': type(mha).__name__, 'is_binary_mha': isinstance(mha, B.BinaryMultiHeadAttention),
                          'projections': proj, 'mha_roles': sorted(r for r in roles if r.startswith('mha/')),
                          'n_roles': len(roles), 'wraps': wrap_record(classes), 'n_losses_training_call': len(losses),
                          'mha_qkvo_kq_conf': ser(mha._qkvo_kq_conf.get_config()),
                          'binary_classes': sorted(c.__name__ for c in classes if c in B.BINARY_CLASSES)}
    rec['binary_models'] = bmodels
    # controls of the wrap check, made after QEinsumDense has been instantiated (so its call and build
    # are already wrapped): a subclass that inherits both (each wrapped twice at its first
    # instantiation), one that defines only its own call (build wrapped twice), and one that defines
    # its own call and build from the unwrapped originals (each wrapped once)
    consts = {c.co_name: c for c in QLayerMeta.__call__.__code__.co_consts if hasattr(c, 'co_name')}
    _, original_call = wrap_depth(QEinsumDense.call, consts['call'])
    _, original_build = wrap_depth(QEinsumDense.build, consts['build'])
    rec['qeinsumdense_wrapped_before_controls'] = QEinsumDense in QLayerMeta._wrapped_cls

    import functools

    class InheritsWrapped(QEinsumDense):
        pass

    class DefinesOwnCall(QEinsumDense):
        @functools.wraps(original_call)
        def call(self, *args, **kwargs):
            return original_call(self, *args, **kwargs)

    class DefinesOwnCallAndBuild(QEinsumDense):
        @functools.wraps(original_call)
        def call(self, *args, **kwargs):
            return original_call(self, *args, **kwargs)

        @functools.wraps(original_build)
        def build(self, *args, **kwargs):
            return original_build(self, *args, **kwargs)

    calls = {}
    for cls in (QEinsumDense, InheritsWrapped, DefinesOwnCall, DefinesOwnCallAndBuild):
        layer = cls('...c,cC->...C', 4)
        try:
            y = layer(keras.ops.ones((2, 3)), training=True)
            calls[cls.__qualname__] = {'ok': True, 'shape': list(y.shape), 'n_losses': len(layer.losses),
                                       'ebops': int(keras.ops.convert_to_numpy(layer._ebops))}
        except Exception as e:  # noqa: BLE001 - recorded: a double wrap may break the layer
            calls[cls.__qualname__] = {'ok': False, 'error': f'{type(e).__name__}: {str(e)[:200]}'}
    rec['controls'] = wrap_record({InheritsWrapped, DefinesOwnCall, DefinesOwnCallAndBuild})
    rec['control_calls'] = calls
    return 0


def cmd_refuse(a, rec):
    """G10 (B6): section 6 steps 1-3 for a binary arm on the CPU, then a load of CH-8's file under a RoleSpec
    that declares empty unmatched and unused sets (refused before any assignment), then under the arm's own."""
    from cbl import env

    env.setup(mem_fraction='0.5', jax_platforms='cpu')
    import model as chang_model  # noqa: F401  (env.setup imports Chang's module first)
    from cbl import model as lab
    from cbl import rng, runner

    groups, emb, control, _ = lab_fields(a.arm, a.form)
    spec = lab.arm_spec(a.arm, groups=groups, embedding=emb, form=a.form, control=control, headline=a.headline)
    good = lab.role_spec(spec)
    wrong = rng.RoleSpec(arm_id=spec.arm_id, top_names=good.top_names, n_variables=rng.CH8_N_VARIABLES)
    flow = rng.SeedFlow(a.seed, dataset_device='cpu:0')
    flow.load_data(a.archive, expect_sha256=a.archive_sha, data_manifest_sha256=a.data_manifest_sha)
    flow.seed_streams()
    flow.build_datasets(batch_size=runner.TRAIN_BATCH, val_batch_size=runner.VAL_BATCH)
    flow.build_model(lab.builder(spec), arm_id=spec.arm_id)
    roles = rng.roles_of(flow.model, good)
    before = rng.variables_sha256(roles)
    try:
        flow.load_initial_tensors(a.manifest, expect_manifest_sha256=a.manifest_sha, role_spec=wrong)
        rec['wrong_spec'] = {'refused': False}
    except rng.InitTensorRefused as e:
        rec['wrong_spec'] = {'refused': True, 'message': str(e)[:400]}
    rec['wrong_spec']['variables_unchanged'] = rng.variables_sha256(roles) == before
    ld = flow.load_initial_tensors(a.manifest, expect_manifest_sha256=a.manifest_sha, role_spec=good)
    rec['good_spec'] = {k: ld[k] for k in ('n_loaded', 'unmatched', 'unused', 'random_unchanged_by_load',
                                           'numpy_unchanged_by_load')}
    return 0


def run_child(name, cmd, *, device, seed, archive, archive_sha, data_manifest_sha, work, children_out, extra=(),
              watches=None, timeout=3600):
    """Run one child in a fresh process of this interpreter (with an open-file audit when watches
    are given); returns (exit code, record, argv, audit)."""
    import shutil
    import subprocess

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    out = work / f'{name}.json'
    argv = [sys.executable, '-I', '-B', str(FORK / 'gates' / 'runner_child.py'), cmd, '--device', device, '--seed', str(seed),
            '--archive', str(archive), '--archive-sha', archive_sha, '--data-manifest-sha', data_manifest_sha,
            '--out', str(out)] + [str(x) for x in extra]
    env = {k: v for k, v in os.environ.items() if k not in ('KERAS_BACKEND', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_MEM_FRACTION')}
    audit = None
    if watches:
        import fileaudit
        r = fileaudit.run_audited(argv, watches=watches, env=env, cwd=str(work), timeout=timeout)
        (work / f'{name}.stdout.txt').write_text(r['stdout'])
        (work / f'{name}.stderr.txt').write_text(r['stderr'])
        rc = r['returncode']
        audit = {k: r[k] for k in ('events', 'events_before_start', 'fd_paths')}
    else:
        with open(work / f'{name}.stdout.txt', 'w') as so, open(work / f'{name}.stderr.txt', 'w') as se:
            rc = subprocess.run(argv, stdout=so, stderr=se, cwd=str(work), env=env, timeout=timeout).returncode
    rec = json.loads(out.read_text()) if out.exists() else None
    if children_out:
        co = Path(children_out)
        co.mkdir(parents=True, exist_ok=True)
        for suffix in ('.json', '.stdout.txt', '.stderr.txt'):
            src = work / f'{name}{suffix}'
            if src.exists():
                shutil.copy2(src, co / src.name)
        (co / f'{name}.argv.txt').write_text(' '.join(argv) + '\n')
        if audit is not None:
            (co / f'{name}.audit.json').write_text(json.dumps(audit, indent=1, sort_keys=True) + '\n')
    return rc, rec, argv, audit


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('cmd', choices=['draw', 'run', 'step', 'ref', 'mha', 'refuse', 'readings'])
    ap.add_argument('--device', choices=['cpu', 'gpu'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--archive-sha', required=True)
    ap.add_argument('--data-manifest-sha', required=True)
    ap.add_argument('--out', required=True, help='JSON record path')
    ap.add_argument('--out-dir', help='draw, run, step, ref: output directory')
    ap.add_argument('--place', default='engineering')
    ap.add_argument('--arm', choices=['CH-8', 'F000'] + list(BINARY_GATE_ARMS), default='CH-8')
    ap.add_argument('--form', choices=['FB', 'BF'], default=None)
    ap.add_argument('--headline', choices=['H1', 'H2'], default=None, help='binary arms: the penalty (b) branch')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--manifest')
    ap.add_argument('--manifest-sha')
    ap.add_argument('--code-sha')
    ap.add_argument('--source-bundle')
    ap.add_argument('--wandb-mode', choices=['disabled', 'offline'], default='disabled')
    ap.add_argument('--ref-data-dir', help='ref: directory with 150c-train.h5 and a placeholder 150c-test.h5')
    ap.add_argument('--no-reseed', action='store_true', help='ref: control without the pre-fit reseed')
    ap.add_argument('--no-probe', action='store_true', help='run: no read-only probe after M8')
    ap.add_argument('--save-ckpt', default=None, help='run: save a .keras checkpoint per epoch in this directory (G34)')
    ap.add_argument('--spy-frozen', action='store_true', help='run: record quantizer training flags during M8 (G34)')
    ap.add_argument('--decoy-frozen', action='store_true', help='run: M8 draws from random first (G12 control)')
    a = ap.parse_args()
    rec = {'cmd': a.cmd, 'args': vars(a)}
    code = 0
    try:
        fn = {'draw': cmd_draw, 'run': cmd_run, 'step': cmd_step, 'ref': cmd_ref, 'mha': cmd_mha, 'refuse': cmd_refuse,
              'readings': cmd_readings}[a.cmd]
        code = fn(a, rec) or 0
    except Exception as e:  # noqa: BLE001 - recorded for the parent, then exit 1
        rec['error'] = f'{type(e).__name__}: {e}'
        rec['traceback'] = traceback.format_exc()
        code = 1
    rec['exit'] = code
    dump(a.out, rec)
    return code


if __name__ == '__main__':
    sys.exit(main())
