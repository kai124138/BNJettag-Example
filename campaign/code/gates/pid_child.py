"""Child process of gates G43-G47 (PREFLIGHT-PLAN section 3 rows G43-G47; STUDY v0.10 section 7.2; module M12
cbl/pid.py through M13 cbl/runner.py). Gate code, not campaign code.

Each invocation is one fresh process, as in a pod. It observes and writes one JSON record; the parent compares.

    run      the runner (prepare, fit, finish) for one PID arm with an engineering config built here (stage `gate`,
             the config's `pid` field equal to cbl.pid.PID_RECORD, the generator's `recipe`, no `ramp`, a headline),
             with read-only spies installed by the runner's gate hook (`evaluator_factory`): every call of the frozen
             evaluator (`evaluate`, `cached`) in order with whether it computed; every released callback's
             on_epoch_end with the type and value of logs['ebops'] before and after it and the keys it added; what
             StopIf's condition and ParetoFront's filter read from logs['ebops']; and, after M8's evaluation inside
             the appended callback, the model's distinct `_beta` values. After fit: the epoch records
             (`epochs.jsonl`, with the PID block), the runner record, the PID's signature, state and the model it
             received, the evaluator's evaluation count, Chang's ParetoFront record, and two controls on the trained
             model: the stock aggregate of the model itself and of a functional wrapper around it
    replay   G46: a LabBetaPID driven through its three hooks on a freshly built CH-8 model with a sensor that returns
             a recorded sequence of frozen H values (`--rows`, the epoch records of a run child): `full` (epochs
             0..E-1), `first` (epochs 0..K-1, then the persisted state written as JSON), `resume` (a fresh controller
             restored from that JSON, epochs K..E-1), `tamper` (as resume with the integral set to 0, a control) and
             `fresh` (epochs K..E-1 without a restore, a control that must be refused)
    units    G45 and G47: the installed BetaPID's constructor defaults (inspect.signature) and the snapshot's
             (reference-code, by syntax tree), both files hashed; cbl.pid's record, its constructor arguments and
             refusals; the lambda_0 control (Chang's CH-8 has every beta 0, model.py:295; the stock BetaPID with
             init_beta None reaches log10(0) at epoch == warmup); LabBetaPID's initial beta on every layer; and the
             log-key control (the stock BetaPID overwrites logs['ebops'] with its float aggregate, LabBetaPID leaves
             FreeEBOPs's int)

Every number is engineering, not quotable.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

SNAPSHOT_BETA_PID = 'reference-code/HGQ2/src/hgq/utils/sugar/beta_pid.py'


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n', encoding='utf-8')


def typed(v):
    """A log value as (type name, exact token): ints as ints, floats as float.hex, strings as they are."""
    if v is None:
        return None
    if isinstance(v, bool):
        return ['bool', v]
    if isinstance(v, int):
        return ['int', v]
    if isinstance(v, float):
        return ['float', v.hex()]
    if isinstance(v, str):
        return ['str', v]
    try:
        return [type(v).__name__, float(v).hex()]
    except Exception:  # noqa: BLE001
        return [type(v).__name__, repr(v)[:80]]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def pid_config(a):
    """An interfaces.md section 5 config for an engineering PID run (gate code; the configs of record are M17's)."""
    import runner_child
    from cbl import pid

    from gen import configs as GC
    from gen import release as GR

    base = 'CH-8' if a.arm == 'PID-CH' else 'F111'
    cfg = runner_child.engineering_config(arm=base, seed=a.seed, epochs=a.epochs, code_sha=a.code_sha,
                                          data_manifest_sha256=a.data_manifest_sha, init_manifest_path=a.manifest,
                                          init_manifest_sha256=a.manifest_sha, wandb_mode=a.wandb_mode, form=a.form,
                                          headline=a.headline)
    cfg['arm_id'] = a.arm
    cfg['schedule'] = 'pid'
    cfg.pop('ramp')
    cfg['pid'] = dict(pid.PID_RECORD)
    cfg['recipe'] = GC._recipe(GR.release_constants())
    cfg['headline'] = a.headline
    cfg['callbacks'] = ['FreeEBOPs', 'LearningRateScheduler', pid.CONFIG_NAME, 'PBar', 'ParetoFront', 'StopIf',
                        'lab:frozen_evaluation']
    return cfg


# ----------------------------------------------------------------------------------------------
# run


def cmd_run(a, rec):
    from cbl import runner

    cfg = pid_config(a)
    H = {'seq': [], 'betas': [], 'stop_reads': [], 'pareto_reads': [], 'recording': False}

    def factory(ctx):
        # called by prepare after the evaluator exists and the PID is bound, before the saver, hook and guard wraps
        seq = H['seq']
        ev = ctx.evaluator
        H['model'] = {'id': id(ctx.model), 'class': f'{type(ctx.model).__module__}.{type(ctx.model).__qualname__}',
                      'is_flow_model': ctx.model is ctx.flow.model, 'n_top_layers': len(ctx.model.layers),
                      'top_classes': [type(ly).__name__ for ly in ctx.model.layers]}
        orig_eval, orig_cached = ev.evaluate, ev.cached

        depth = [0]

        def evaluate(epoch=None):
            seq.append({'event': 'evaluate', 'epoch': epoch, 'computes': epoch is None or epoch not in ev.cache})
            depth[0] += 1
            try:
                return orig_eval(epoch)
            finally:
                depth[0] -= 1

        def cached(epoch):
            seq.append({'event': 'cached', 'epoch': epoch, 'inside_evaluate': depth[0] > 0})
            return orig_cached(epoch)
        ev.evaluate, ev.cached = evaluate, cached
        for cb in ctx.released:
            name = type(cb).__name__
            orig = cb.on_epoch_end

            def spy(epoch, logs=None, _orig=orig, _name=name):
                keys0 = set(logs)
                entry = {'event': 'on_epoch_end', 'cb': _name, 'epoch': int(epoch), 'ebops_before': typed(logs.get('ebops'))}
                seq.append(entry)           # events inside the callback (the PID's evaluation) follow it in seq
                out = _orig(epoch, logs)
                entry.update(ebops_after=typed(logs.get('ebops')), added=sorted(set(logs) - keys0))
                return out
            cb.on_epoch_end = spy
        stop, pareto = ctx.released[5], ctx.released[4]
        cond, enable = stop.stop_condition, pareto.enable_if

        def stop_condition(logs, model):
            if 'val_loss' in logs:
                H['stop_reads'].append({'epoch': len(H['stop_reads']), 'ebops': typed(logs['ebops'])})
            return cond(logs, model)

        def enable_if(x):
            if 'val_loss' in x:
                H['pareto_reads'].append({'epoch': x.get('epoch'), 'ebops': typed(x['ebops'])})
            return enable(x)
        stop.stop_condition, pareto.enable_if = stop_condition, enable_if
        model = ctx.model

        def probe(epoch, logs):
            import keras
            vals = sorted({float(keras.ops.convert_to_numpy(ly._beta)).hex() for ly in model._flatten_layers()
                           if getattr(ly, '_beta', None) is not None})
            H['betas'].append({'epoch': int(epoch), 'beta_values': vals, 'ebops_seen_by_hook': typed(logs.get('ebops'))})
            seq.append({'event': 'hook_step', 'epoch': int(epoch)})
        return probe

    code, ctx = runner.run(cfg, train_archive=a.archive, archive_kind='smoke', device=a.device, out_dir=a.out_dir,
                           source_bundle=a.source_bundle, evaluator_factory=factory, keep_full=None)
    r = ctx.record
    ep = Path(a.out_dir) / 'epochs.jsonl'
    rows = [json.loads(x) for x in ep.read_text(encoding='utf-8').splitlines()] if ep.exists() else None
    rec['config'] = cfg
    rec['frozen_rows'] = [{k: x.get(k) for k in ('epoch', 'H1', 'H2', 'NR', 'val_correct', 'val_n', 'train_count', 'beta',
                                                 'lr', 'pid', 'step')} for x in rows] if rows else None
    rec['runner'] = {k: r.get(k) for k in ('end', 'exit', 'refused', 'error', 'traceback', 'violation', 'summary',
                                           'callback_order', 'callback_signature', 'pid', 'epochs_completed', 'guard',
                                           'front', 'front_error', 'load', 'wandb', 'code')}
    fs = r.get('front_summary') or {}
    rec['front_summary'] = {k: fs.get(k) for k in ('end', 'headline', 'pid_selection', 'm16_reader', 'epochs_offered')}
    rec['seq'] = H['seq']
    rec['betas'] = H['betas']
    rec['stop_reads'] = H['stop_reads']
    rec['pareto_reads'] = H['pareto_reads']
    rec['model'] = H.get('model')
    if ctx.pid is not None and ctx.model is not None:
        import keras
        from types import SimpleNamespace

        from hgq.utils.sugar.beta_pid import BaseBetaPID
        from cbl import cost
        p = ctx.pid
        rec['pid_object'] = {'class': type(p).__name__, 'mro': [c.__name__ for c in type(p).__mro__[:4]],
                             'model_is_runner_model': p.model is ctx.model, 'model_seen': p.model_seen,
                             'runner_model_id': id(ctx.model), 'stock_at_train_begin': p.stock_at_train_begin,
                             'state': p.state() if p.last_epoch is not None else None, 'signature': p.signature(),
                             'pid_class': type(p.pid).__name__, 'pid_mro': [c.__name__ for c in type(p.pid).__mro__[:3]]}
        rec['n_evaluations'] = ctx.evaluator.n_evaluations
        pf = ctx.released[4]
        rec['pareto'] = {'record': [[typed(float(v)) for v in rr] for rr in pf.record],
                         'paths': [Path(x).name for x in pf.paths]}
        # controls on the trained model, after fit and after run.json (nothing here enters the run's records)
        rec['controls'] = {'stock_aggregate_model': cost.public_aggregate(ctx.model),
                           'private_total_model': cost.private_total(ctx.model)}
        try:
            inp = keras.Input(shape=tuple(ctx.model.input_shape[1:]))
            wrapper = keras.Model(inp, ctx.model(inp))
            rec['controls'].update(stock_aggregate_wrapper=float(BaseBetaPID.get_ebops(SimpleNamespace(model=wrapper))),
                                   wrapper_top_classes=[type(ly).__name__ for ly in wrapper.layers])
        except Exception as e:  # noqa: BLE001 - recorded; the parent's control check then fails
            rec['controls']['wrapper_error'] = f'{type(e).__name__}: {e}'
    return code


# ----------------------------------------------------------------------------------------------
# replay (G46)


def _model():
    from cbl import env

    env.setup(mem_fraction='0.5', jax_platforms='cpu')
    import keras
    import model as chang_model
    from cbl import rng

    random.seed(1)
    keras.utils.set_random_seed(1)
    return chang_model.get_model(*rng.CH8_BUILDER_ARGS)


def _betas(model):
    import keras
    return sorted({float(keras.ops.convert_to_numpy(ly._beta)).hex() for ly in model._flatten_layers()
                   if getattr(ly, '_beta', None) is not None})


def _drive(p, model, epochs, sensed):
    out = []
    for e in epochs:
        p.on_epoch_begin(e, {})
        betas_after_begin = _betas(model)
        logs = {'ebops': 0}
        p.on_epoch_end(e, logs)
        out.append({'epoch': e, 'logs': {k: typed(v) for k, v in sorted(logs.items())}, 'betas_after_begin': betas_after_begin,
                    'integral': typed(p.pid.integral), 'prev_error': typed(p.pid.prev_error), 'beta': typed(p.beta)})
    return out


def cmd_replay(a, rec):
    model = _model()
    from cbl import pid

    rows = json.loads(Path(a.rows).read_text())['frozen_rows']
    sensed = {r['epoch']: r['pid']['sensed_H'] for r in rows}
    E = len(rows)

    def read(epoch):
        return sensed[epoch]
    read.units = a.headline
    p = pid.make_pid(dict(pid.PID_RECORD), headline=a.headline)
    p.bind(sensor=read)
    p.set_model(model)
    rec['mode'], rec['epochs_total'] = a.mode, E
    if a.mode == 'full':
        p.on_train_begin()
        rec['epochs'] = _drive(p, model, range(E), sensed)
        rec['state'] = p.state()
    elif a.mode == 'first':
        p.on_train_begin()
        rec['epochs'] = _drive(p, model, range(a.stop), sensed)
        st = p.state()
        Path(a.state).write_text(json.dumps(st, indent=1, sort_keys=True) + '\n')
        rec['state'] = st
        rec['state_file_sha256'] = sha256_file(a.state)
    elif a.mode in ('resume', 'tamper'):
        st = json.loads(Path(a.state).read_text())
        rec['state_file_sha256'] = sha256_file(a.state)
        rec['state_read'] = st
        if a.mode == 'tamper':
            st = dict(st, integral=0.0)
            st['hex'] = dict(st['hex'], integral=(0.0).hex())
        p.restore(st)
        p.on_train_begin()
        rec['betas_after_train_begin'] = _betas(model)
        rec['epochs'] = _drive(p, model, range(st['epoch'] + 1, E), sensed)
        rec['state'] = p.state()
    elif a.mode == 'fresh':
        p.on_train_begin()
        try:
            rec['epochs'] = _drive(p, model, range(a.stop, E), sensed)
            rec['refused'] = None
        except pid.PIDError as e:
            rec['refused'] = f'{type(e).__name__}: {e}'
    else:
        raise ValueError(a.mode)
    return 0


# ----------------------------------------------------------------------------------------------
# units (G45, G47 controls)


def _snapshot_defaults(path):
    """BetaPID.__init__'s defaults in the snapshot's source, read from its syntax tree (float('inf') as 'inf')."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'BetaPID':
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == '__init__':
                    args = f.args.args[1:]
                    defs = f.args.defaults
                    out = {}
                    for arg, d in zip(args[len(args) - len(defs):], defs):
                        if isinstance(d, ast.Call) and getattr(d.func, 'id', None) == 'float':
                            out[arg.arg] = d.args[0].value
                        else:
                            out[arg.arg] = ast.literal_eval(d)
                    return {'defaults': out, 'lines': [f.lineno, f.end_lineno],
                            'required': [x.arg for x in args[:len(args) - len(defs)]]}
    raise RuntimeError('BetaPID.__init__ not found in the snapshot')


def cmd_units(a, rec):
    model = _model()
    import inspect

    import hgq.utils.sugar.beta_pid as bp
    import keras  # noqa: F401
    from hgq.utils.sugar.beta_pid import BetaPID
    from cbl import pid

    sig = inspect.signature(BetaPID.__init__)
    inst = {k: ('inf' if isinstance(v.default, float) and math.isinf(v.default) else v.default)
            for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
    snap = Path(a.lab_root) / SNAPSHOT_BETA_PID
    rec['installed'] = {'file': bp.__file__, 'sha256': sha256_file(bp.__file__), 'defaults': inst,
                        'required': [k for k, v in sig.parameters.items() if v.default is inspect.Parameter.empty and k != 'self'],
                        'init_lines': [BetaPID.__init__.__code__.co_firstlineno]}
    rec['snapshot'] = dict(_snapshot_defaults(snap), file=str(snap), sha256=sha256_file(snap))
    rec['record'] = dict(pid.PID_RECORD)
    kw = pid.check_record(dict(pid.PID_RECORD))
    rec['kwargs'] = {k: ('inf' if isinstance(v, float) and math.isinf(v) else v) for k, v in kw.items()}
    rec['max_beta_parsed'] = typed(pid.parse_max_beta('inf'))
    refusals = {}
    for name, over in (('init_beta 0', {'init_beta': 0.0}), ('init_beta negative', {'init_beta': -2e-8}),
                       ('warmup 0', {'warmup': 0}), ('max_beta Infinity', {'max_beta': 'Infinity'}),
                       ('max_beta float inf', {'max_beta': math.inf}), ('max_beta null', {'max_beta': None}),
                       ('i 0', {'i': 0.0}), ('target float', {'target_H': 350000.0}), ('p int', {'p': 1}),
                       ('target 400000', {'target_H': 400000}), ('extra key', {'extra': 1}), ('log 1', {'log': 1})):
        try:
            pid.check_record(dict(pid.PID_RECORD, **over))
            refusals[name] = None
        except pid.PIDConfigError as e:
            refusals[name] = str(e)[:200]
    rec['refusals'] = refusals
    # lambda_0 control: Chang's model has every beta 0 (LayerConfigScope(beta0=0), model.py:295)
    rec['chang_betas'] = _betas(model)
    stock = BetaPID(350000, init_beta=None)
    stock.set_model(model)
    stock.on_train_begin()
    rec['stock_none'] = {'beta_after_train_begin': typed(stock.beta)}
    for e in range(stock.warmup):
        stock.on_epoch_begin(e, {})
    stock._ebops = 1.0e6
    try:
        stock.on_epoch_begin(stock.warmup, {})
        rec['stock_none']['at_warmup'] = {'raised': None, 'beta': typed(stock.beta)}
    except Exception as e:  # noqa: BLE001 - the expected log10(0)
        rec['stock_none']['at_warmup'] = {'raised': f'{type(e).__name__}: {e}', 'frame': traceback.extract_tb(e.__traceback__)[-1].line}
    # LabBetaPID's initial beta on every layer and through warmup
    p = pid.make_pid(dict(pid.PID_RECORD), headline='H1')

    def read(epoch):
        return 400000
    read.units = 'H1'
    p.bind(sensor=read)
    p.set_model(model)
    p.on_train_begin()
    rec['lab_initial'] = {'betas_after_train_begin': _betas(model), 'beta': typed(p.beta),
                          'stock_at_train_begin': typed(p.stock_at_train_begin)}
    warm = []
    for e in range(p.warmup):
        p.on_epoch_begin(e, {})
        logs = {'ebops': 123456}
        p.on_epoch_end(e, logs)
        warm.append({'epoch': e, 'betas': _betas(model), 'logs_beta': typed(logs['beta']),
                     'ebops_after': typed(logs['ebops']), 'pid_keys': sorted(k for k in logs if k.startswith('pid/'))})
    rec['lab_warmup'] = warm
    # log-key control: the stock BetaPID's on_epoch_end against LabBetaPID's on the same logs
    s2 = BetaPID(350000, init_beta=2e-8)
    s2.set_model(model)
    s2.on_train_begin()
    s2.on_epoch_begin(0, {})
    logs_s = {'ebops': 123456}
    s2.on_epoch_end(0, logs_s)
    rec['stock_log_keys'] = {'ebops_before': typed(123456), 'ebops_after': typed(logs_s['ebops']), 'keys': sorted(logs_s)}
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('cmd', choices=['run', 'replay', 'units'])
    ap.add_argument('--out', required=True, help='JSON record path')
    ap.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--archive')
    ap.add_argument('--archive-sha')
    ap.add_argument('--data-manifest-sha')
    ap.add_argument('--out-dir')
    ap.add_argument('--arm', choices=['PID-CH', 'PID-BEST'], default='PID-CH')
    ap.add_argument('--form', choices=['FB', 'BF'], default=None)
    ap.add_argument('--headline', choices=['H1', 'H2'], default='H1')
    ap.add_argument('--epochs', type=int, default=14)
    ap.add_argument('--manifest')
    ap.add_argument('--manifest-sha')
    ap.add_argument('--code-sha')
    ap.add_argument('--source-bundle')
    ap.add_argument('--wandb-mode', choices=['disabled', 'offline'], default='disabled')
    ap.add_argument('--rows', help='replay: a run child record with frozen_rows')
    ap.add_argument('--mode', choices=['full', 'first', 'resume', 'tamper', 'fresh'])
    ap.add_argument('--stop', type=int, help='replay first/fresh: K')
    ap.add_argument('--state', help='replay: the persisted-state JSON file')
    ap.add_argument('--lab-root', default=str(FORK.parents[2]), help='units: the lab root holding reference-code/')
    a = ap.parse_args()
    rec = {'cmd': a.cmd, 'args': vars(a)}
    code = 0
    try:
        code = {'run': cmd_run, 'replay': cmd_replay, 'units': cmd_units}[a.cmd](a, rec) or 0
    except Exception as e:  # noqa: BLE001 - recorded for the parent, then exit 1
        rec['error'] = f'{type(e).__name__}: {e}'
        rec['traceback'] = traceback.format_exc()
        code = 1
    rec['exit'] = code
    dump(a.out, rec)
    return code


if __name__ == '__main__':
    sys.exit(main())
