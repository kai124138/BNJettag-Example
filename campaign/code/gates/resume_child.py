"""Child process of gates G46 (whole-run part), G48 and G50 (PREFLIGHT-PLAN section 3 rows G46, G48, G50; STUDY
v0.10 section 6 step 8, 7.2, 13 resume row and "further" rows; module M14 cbl/resume.py through M13 cbl/runner.py).
Gate code, not campaign code.

Each invocation is one fresh process, as in a pod after a restart.

    run    the runner (prepare, fit, finish) for one arm with an engineering config built here (stage `gate`): CH-8,
           a binary arm, or PID-CH; with --resume it resumes --out-dir from its latest checkpoint. Kill seams, all a
           real SIGKILL of this process (no cleanup runs): --kill boundary:K (right after the checkpoint of K done
           epochs is named in latest.json), --kill stage:STAGE:K (after that step of the checkpoint write of K epochs:
           tmp_written or renamed), --kill mid:E:J (in epoch E, right after the J-th batch fetch of the training
           Dataset). Control --no-restore-streams makes M14's stream restore a no-op (the comparison must then fail).
           After the run: the per-epoch records, the run record, the final states of every persisted piece (variables
           and optimizer hashes, both global streams, StopIf, ParetoFront, PID, front saver, sidecars, events, summary),
           every layer's configuration and the beta0 values in them (finding B7-F1), M4's recorder, the resume record.
    tree   a digest of every file under --out-dir (path, size, sha256), for the refusal controls
    reload G50: in a fresh process, every listed .keras file loaded with keras.models.load_model and every listed
           resume variables.npz assigned by role into a model built as the runner builds the arm; each one's state
           hash (cbl.rng.variables_sha256 over the arm's roles), file sha256 and the beta0 values of the configuration;
           with --mutate a copy of the first .keras file with one kernel entry changed, saved and hashed (control)
    guard  G51: one runner run with a probe after epoch 0, inside the training process: the installed held-out guard
           asked about --heldout, --hardlink, a path under --raw-dir and the training archive (no open), then open
           attempts on the placeholders in --placeholders only (builtins.open, os.open, numpy.load, h5py.File, a
           symbolic link, a raw held-out file name), each expected to raise HeldOutAccessRefused
    callbacks  the persisted released callbacks on constructed epochs (G48): Chang's ParetoFront and StopIf, built
           with the runner's restated arguments (run_train.py:83-91) on a CH-8 model, fed a fixed sequence of logs
           that passes Chang's filter, joins and leaves the front and meets StopIf's condition. --phase full runs
           epochs 0..11; --phase first runs 0..split+2 and writes the state after epoch split-1 (cbl.resume's
           ParetoFront and StopIf state, the file list and links of the live files) as a killed run would leave it;
           --phase resume (a fresh process) rewinds the directory to that state with cbl.resume.rewind, restores both
           callbacks and runs split..11. Records per epoch StopIf's counters and stop flag, and at the end the
           ParetoFront record and its file names

The configs are gate copies of interfaces section 5 configs (runner_child.engineering_config, pid_child.pid_config).
Every reading is engineering, not quotable.
"""

import argparse
import hashlib
import json
import os
import signal
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n', encoding='utf-8')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(root):
    root = Path(root)
    out = {}
    for p in sorted(root.rglob('*')):
        if p.is_file():
            out[str(p.relative_to(root))] = [p.stat().st_size, sha256_file(p)]
    return out


def config_for(a):
    import runner_child

    if a.arm == 'PID-CH':
        import pid_child
        ns = SimpleNamespace(arm='PID-CH', seed=a.seed, epochs=a.epochs, code_sha=a.code_sha,
                             data_manifest_sha=a.data_manifest_sha, manifest=a.manifest, manifest_sha=a.manifest_sha,
                             wandb_mode=a.wandb_mode, form=None, headline=a.headline)
        return pid_child.pid_config(ns)
    return runner_child.engineering_config(arm=a.arm, seed=a.seed, epochs=a.epochs, code_sha=a.code_sha,
                                           data_manifest_sha256=a.data_manifest_sha, init_manifest_path=a.manifest,
                                           init_manifest_sha256=a.manifest_sha, wandb_mode=a.wandb_mode, form=a.form,
                                           headline=a.headline)


def kill_now():
    sys.stdout.flush()
    sys.stderr.flush()
    os.kill(os.getpid(), signal.SIGKILL)


def install_kill(ctx, spec):
    """The kill seams (module docstring)."""
    if not spec:
        return None
    parts = spec.split(':')
    if parts[0] == 'boundary':
        k = int(parts[1])
        ctx.resume.on_stage = lambda stage, done: kill_now() if (stage == 'latest_written' and done == k) else None
    elif parts[0] == 'stage':
        st, k = parts[1], int(parts[2])
        ctx.resume.on_stage = lambda stage, done: kill_now() if (stage == st and done == k) else None
    elif parts[0] == 'mid':
        e, j = int(parts[1]), int(parts[2])
        rec = ctx.flow.recorder
        orig = rec._fetch

        def fetch(ds, idx, lookups):
            orig(ds, idx, lookups)
            seg = rec._segment
            if seg is not None and seg['keras_epoch'] == e and len(seg['raw']) >= j:
                kill_now()
        rec._fetch = fetch
    else:
        raise ValueError(f'unknown kill spec {spec!r}')
    return spec


def layer_configs(model):
    """Every layer's get_config (names left out: Keras numbers them per process) and the beta0 values in them."""
    import runner_child

    confs, beta0 = [], []
    for ly in model._flatten_layers(include_self=False):
        try:
            c = ly.get_config()
        except Exception:  # noqa: BLE001
            c = {'unserializable': type(ly).__name__}
        c = dict(c)
        c.pop('name', None)
        s = runner_child.ser(c)
        confs.append([type(ly).__name__, s])
        if isinstance(c, dict) and 'beta0' in c:
            beta0.append([type(ly).__name__, repr(c['beta0'])])
    blob = json.dumps(confs, sort_keys=True, default=str).encode()
    return {'n_layers': len(confs), 'configs_sha256': hashlib.sha256(blob).hexdigest(), 'beta0': beta0}


def norm(obj, drop=('ckpt_sha256', 'sha256', 'ckpt_bytes', 'bytes', 'seconds', 'finalize_seconds', 'frozen_seconds',
                     'created_utc', 'utc')):
    """A record without the fields that differ between two saves of one state or between two processes by design:
    file-byte hashes and sizes of .keras files (B8-2: Keras writes a save time) and wall times."""
    if isinstance(obj, dict):
        return {k: norm(v, drop) for k, v in obj.items() if k not in drop}
    if isinstance(obj, list):
        return [norm(x, drop) for x in obj]
    return obj


def cmd_run(a, rec):
    from cbl import resume as resume_mod
    from cbl import runner

    cfg = config_for(a)
    if a.no_restore_streams:
        resume_mod.restore_streams = lambda ctx, st: None          # control: the comparison must then fail
    ctx = runner.Context()
    code = None
    try:
        runner.prepare(cfg, ctx=ctx, train_archive=a.archive, archive_kind=a.archive_kind, device=a.device,
                       out_dir=a.out_dir, source_bundle=a.source_bundle, keep_full=None, resume=a.resume,
                       resume_every=a.every)
        rec['kill'] = install_kill(ctx, a.kill)
        code = runner.fit(ctx)
    except runner.ConfigRefused as e:
        ctx.record.update(end='refused', refused=str(e))
        code = runner.EXIT_REFUSED
    except Exception as e:  # noqa: BLE001
        ctx.record.update(end='failed', error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc())
        code = runner.EXIT_FAILED
    code = runner.finish(ctx, code)
    r = ctx.record
    out = Path(a.out_dir)
    ep = out / 'epochs.jsonl'
    rows = [json.loads(x) for x in ep.read_text(encoding='utf-8').splitlines() if x] if ep.exists() else []
    rec['config'] = cfg
    rec['rows'] = [{k: v for k, v in x.items() if not k.startswith('sys/')} for x in rows]
    rec['runner'] = {k: r.get(k) for k in ('end', 'exit', 'refused', 'error', 'traceback', 'violation', 'epochs_completed',
                                           'initial_epoch', 'resume', 'resume_from', 'resume_every', 'guard', 'recorder',
                                           'final_variables_sha256', 'wandb', 'flow', 'heldout_guard', 'launch_check',
                                           'reseed', 'callback_order', 'front_error', 'config_sha256', 'code')}
    rec['epoch_records'] = [{k: v for k, v in x.items() if not k.startswith('sys/')} for x in (r.get('epoch_records') or [])]
    if ctx.model is not None and r.get('end') in ('completed', 'stopif'):
        import random

        from cbl import rng
        rec['final'] = {
            'variables_sha256': rng.variables_sha256(ctx.roles),
            'optimizer_sha256': resume_mod.optimizer_sha256(ctx.model.optimizer),
            'optimizer_iterations': int(ctx.model.optimizer.iterations.numpy()),
            'py_state_sha256': rng.py_state_sha256(random.getstate()), 'np_state_sha256': rng.np_state_sha256(),
            'index_map_sha256': resume_mod.int_list_sha256(ctx.flow.ds_train._index_map),
            'stopif': resume_mod.stopif_state(ctx.released[5]),
            'pareto': resume_mod.pareto_state(ctx.released[4]),
            'pid': ctx.pid.state() if (ctx.pid is not None and ctx.pid.last_epoch is not None) else None,
            'front_snapshot': norm(ctx.front.snapshot()) if ctx.front is not None else None,
            'signs': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in ctx.evaluator.sign_state().items()},
            'layers': layer_configs(ctx.model),
            'stop_training': bool(ctx.model.stop_training),
        }
        from cbl import front
        fd = out / 'front'
        rec['front'] = {
            'events': [norm(e) for e in front.read_events(fd)] if (fd / 'events.jsonl').exists() else [],
            'sidecars': {str(s['epoch']): norm(s) for s in front.load_sidecars(fd).values()},
            'summary': norm(json.loads((fd / 'summary.json').read_text())) if (fd / 'summary.json').exists() else None,
            'files': sorted(str(p.relative_to(out)) for p in fd.rglob('*') if p.is_file()),
        }
        rec['pareto_files'] = sorted(p.name for p in (out / 'pareto').iterdir()) if (out / 'pareto').is_dir() else []
    rec['resume_dir'] = sorted(str(p.relative_to(out)) for p in (out / 'resume').rglob('*')) if (out / 'resume').is_dir() else []
    return code


def arm_build(a):
    """(builder, RoleSpec) as cbl.runner.prepare makes them for the arm."""
    import dataclasses

    import model as chang_model
    from cbl import model as lab
    from cbl import rng

    if a.arm in ('CH-8', 'PID-CH'):
        spec = rng.CH8_ROLE_SPEC if a.arm == 'CH-8' else dataclasses.replace(rng.CH8_ROLE_SPEC, arm_id=a.arm)
        return (lambda: chang_model.get_model(*rng.CH8_BUILDER_ARGS)), spec
    import runner_child
    groups, emb, control, _ = runner_child.lab_fields(a.arm, a.form)
    sp = lab.arm_spec(a.arm, groups=groups, embedding=emb, form=a.form, control=control, headline=a.headline)
    return lab.builder(sp), lab.role_spec(sp)


def cmd_reload(a, rec):
    from cbl import env

    env.setup(mem_fraction='0.5', jax_platforms='cpu')
    import keras
    import numpy as np

    from cbl import binary  # noqa: F401  registers the cbl1016 classes
    from cbl import rng
    build, spec = arm_build(a)
    files = json.loads(Path(a.files).read_text())
    out = {'keras': {}, 'npz': {}}
    for p in files.get('keras', []):
        m = keras.models.load_model(p)
        roles = rng.roles_of(m, spec)
        out['keras'][p] = {'state_sha256': rng.variables_sha256(roles), 'file_sha256': sha256_file(p),
                           'beta0': layer_configs(m)['beta0']}
    for p in files.get('npz', []):
        m = build()
        roles = rng.roles_of(m, spec)
        arrs = rng.read_tensor_file(p)
        if set(arrs) != set(roles):
            raise RuntimeError(f'{p}: roles differ from the model\'s')
        for r, v in roles.items():
            v.assign(arrs[r])
        out['npz'][p] = {'state_sha256': rng.variables_sha256(roles), 'file_sha256': sha256_file(p)}
    if a.mutate and files.get('keras'):
        src = files['keras'][0]
        m = keras.models.load_model(src)
        roles = rng.roles_of(m, spec)
        r0 = next(r for r in sorted(roles) if r.endswith('/kernel'))
        x = np.array(keras.ops.convert_to_numpy(roles[r0].value))
        x.flat[0] = x.flat[0] + np.float32(0.5)
        roles[r0].assign(x)
        dst = Path(a.mutate)
        m.save(str(dst))
        m2 = keras.models.load_model(str(dst))
        out['mutated'] = {'source': src, 'role': r0, 'state_sha256': rng.variables_sha256(rng.roles_of(m2, spec))}
    rec['reload'] = out
    return 0


CB_LOGS = [(0.55, 480000), (0.58, 470000), (0.60, 490000), (0.45, 100000), (0.62, 300000), (0.59, 250000),
           (0.65, 200000), (0.60, 9000), (0.66, 199000), (0.70, 9500), (0.71, 20000), (0.72, 5000)]


def cmd_callbacks(a, rec):
    from cbl import env

    env.setup(mem_fraction='0.5', jax_platforms='cpu')
    import random

    import keras
    import model as chang_model
    from hgq.utils.sugar import ParetoFront, StopIf

    from cbl import resume as R
    from cbl import rng, runner

    random.seed(1)
    keras.utils.set_random_seed(1)
    model = chang_model.get_model(*rng.CH8_BUILDER_ARGS)
    run_dir = Path(a.out_dir)
    pf = ParetoFront(str(run_dir / 'pareto'), runner.PARETO_METRICS, runner.PARETO_SIDES, fname_format=runner.PARETO_FNAME,
                     enable_if=runner.pareto_enable_if)
    stop = StopIf(runner.stop_condition)
    for cb in (pf, stop):
        cb.set_model(model)
    k = a.split
    start, end = 0, len(CB_LOGS)
    if a.phase == 'first':
        end = k + 3
    if a.phase == 'resume':
        st = json.loads(Path(a.state).read_text())
        rec['rewind'] = R.rewind(run_dir, st, run_dir / 'ck', run_dir / 'discarded')
        R.restore_pareto(pf, st['pareto'])
        R.restore_stopif(stop, st['stopif'])
        start = k
    else:
        pf.on_train_begin()
    per = []
    for e in range(start, end):
        va, eb = CB_LOGS[e]
        logs = {'val_accuracy': va, 'accuracy': va, 'ebops': eb, 'val_loss': 1.0 - va, 'loss': 1.0 - va}
        model.stop_training = False
        pf.on_epoch_end(e, logs)
        stop.on_epoch_end(e, logs)
        per.append({'epoch': e, 'stopif': R.stopif_state(stop), 'stop_training': bool(model.stop_training),
                    'front': [p.name for p in pf.paths]})
        if a.phase == 'first' and e == k - 1:
            ck = run_dir / 'ck'
            live = {os.path.relpath(str(p), str(run_dir)): {} for p in pf.paths}
            links = {}
            for rel in live:
                R.link_or_copy(run_dir / rel, ck / 'links' / rel)
                links[rel] = {'sha256': sha256_file(ck / 'links' / rel), 'method': 'link'}
            st = {'pareto': R.pareto_state(pf), 'stopif': R.stopif_state(stop), 'append_only': {}, 'front_files': [],
                  'pareto_files': R.files_under(run_dir / 'pareto', run_dir), 'live': {'front': {}, 'pareto': live},
                  'links': links}
            Path(a.state).write_text(json.dumps(st, indent=1, sort_keys=True))
    rec['per_epoch'] = per
    rec['pareto'] = R.pareto_state(pf)
    rec['pareto_files'] = sorted(p.name for p in (run_dir / 'pareto').iterdir())
    rec['stopif'] = R.stopif_state(stop)
    return 0


def cmd_guard(a, rec):
    from cbl import runner

    cfg = config_for(a)
    ph = Path(a.placeholders)
    H = {}

    def factory(ctx):
        def probe(epoch, logs):
            if epoch != 0 or H:
                return
            import h5py
            import numpy as np

            from cbl import data
            g = data.heldout_guard()
            H['queries'] = {'heldout': g.offends(a.heldout), 'hardlink': g.offends(a.hardlink),
                            'raw_dir': g.offends(os.path.join(a.raw_dir, 'any-file.h5')), 'train': g.offends(a.archive)}
            H['refused_before'] = len(g.refused)
            tries = {'open': lambda: open(ph / '150c-test.h5', 'rb'),
                     'os.open': lambda: os.open(str(ph / '150c-test.h5'), os.O_RDONLY),
                     'np.load': lambda: np.load(str(ph / '150c-test.h5')),
                     'h5py.File': lambda: h5py.File(str(ph / '150c-test.h5'), 'r'),
                     'open via symlink': lambda: open(ph / 'innocent.h5', 'rb'),
                     'h5py.File raw name': lambda: h5py.File(str(ph / a.raw_name), 'r')}
            out = {}
            for k, fn in tries.items():
                try:
                    obj = fn()
                    out[k] = {'refused': False}
                    try:
                        obj.close() if hasattr(obj, 'close') else os.close(obj)
                    except Exception:  # noqa: BLE001
                        pass
                except Exception as e:  # noqa: BLE001
                    out[k] = {'refused': isinstance(e, data.HeldOutAccessRefused), 'type': type(e).__name__,
                              'message': str(e)[:200]}
            H['attempts'] = out
            H['refused_after'] = list(g.refused[H['refused_before']:])
        return probe

    code, ctx = runner.run(cfg, train_archive=a.archive, archive_kind=a.archive_kind, device=a.device, out_dir=a.out_dir,
                           source_bundle=a.source_bundle, evaluator_factory=factory, keep_full=None)
    r = ctx.record
    guard = dict(r.get('heldout_guard') or {})
    from cbl import data
    g = data.heldout_guard()
    if g is not None and 'refused_before' in H:
        guard['refused'] = list(g.refused[:H['refused_before']])
    rec['runner'] = {k: r.get(k) for k in ('end', 'exit', 'refused', 'error', 'traceback', 'epochs_completed', 'launch_check')}
    rec['runner']['heldout_guard'] = guard
    rec['queries'] = H.get('queries')
    rec['attempts'] = H.get('attempts')
    rec['guard_refused_after'] = H.get('refused_after')
    return code


def cmd_tree(a, rec):
    rec['tree'] = tree_digest(a.out_dir)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('cmd', choices=['run', 'tree', 'callbacks', 'reload', 'guard'])
    ap.add_argument('--out', required=True, help='JSON record path')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--archive')
    ap.add_argument('--archive-kind', choices=['smoke', 'record'], default='smoke')
    ap.add_argument('--archive-sha')
    ap.add_argument('--data-manifest-sha')
    ap.add_argument('--arm', default='CH-8')
    ap.add_argument('--form', choices=['FB', 'BF'], default=None)
    ap.add_argument('--headline', choices=['H1', 'H2'], default=None)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--manifest')
    ap.add_argument('--manifest-sha')
    ap.add_argument('--code-sha')
    ap.add_argument('--source-bundle')
    ap.add_argument('--wandb-mode', choices=['disabled', 'offline'], default='disabled')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--kill', default=None)
    ap.add_argument('--no-restore-streams', action='store_true')
    ap.add_argument('--phase', choices=['full', 'first', 'resume'])
    ap.add_argument('--split', type=int, default=8)
    ap.add_argument('--state')
    ap.add_argument('--files', help='reload: JSON {"keras": [...], "npz": [...]}')
    ap.add_argument('--placeholders')
    ap.add_argument('--raw-name')
    ap.add_argument('--heldout')
    ap.add_argument('--hardlink')
    ap.add_argument('--raw-dir')
    ap.add_argument('--mutate', help='reload: path of the mutated copy (control)')
    a = ap.parse_args()
    rec = {'cmd': a.cmd, 'args': vars(a)}
    code = 0
    try:
        code = {'run': cmd_run, 'tree': cmd_tree, 'callbacks': cmd_callbacks, 'reload': cmd_reload,
                'guard': cmd_guard}[a.cmd](a, rec) or 0
    except Exception as e:  # noqa: BLE001 - recorded for the parent, then exit 1
        rec['error'] = f'{type(e).__name__}: {e}'
        rec['traceback'] = traceback.format_exc()
        code = 1
    rec['exit'] = code
    dump(a.out, rec)
    return code


if __name__ == '__main__':
    sys.exit(main())
