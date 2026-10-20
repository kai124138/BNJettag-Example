"""Child process of gates G10-G12 (module M4 cbl/rng.py). Gate code, not campaign runner code.

Each invocation is one fresh process, so that Keras's process-global name counters and the global
random streams start from nothing, as in a pod. It observes and writes a JSON record; the parent
gate does the comparisons and the reconstructions.

    draw      section 6 steps 1-3 with CH-8's released builder, then the initial-tensor file and
              manifest of decision U1 (cbl.rng.draw_initial_tensors)
    run       steps 1-5 (the file loaded by role), then, unless --no-fit, Chang's compile and
              callbacks (run_train.py:79-100) plus lab callbacks, all wrapped by the stream guard,
              and `fit` for --epochs epochs; records the batch-index recorder, the guard, the probe
              callback's in-run state checks and the final module-level random state
    refusals  the loader's and the flow's refusals on tampered inputs (G10 negative controls), and
              the positive load

The fit recipe copies run_train.py:79-100 with `epochs` set by the gate and the ParetoFront output
directory in the work directory; the training archive is the one the gate names (the smoke archive
in B4). The lab callbacks are a probe that stands where the frozen-evaluation callback (M8) will
stand, after StopIf: it marks each Keras epoch in the recorder, writes the epoch's two hashes into
the logs under new keys, and compares the live random state with the reconstruction; and, when
asked, one decoy that draws from a global stream (G12 controls).

    <venv>/bin/python -I -B gates/rng_child.py draw|run|refusals --device cpu|gpu --seed S \\
        --archive A --archive-sha X --data-manifest-sha M --out PATH [...]
"""

import argparse
import json
import random
import shutil
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))

EXIT_VIOLATION = 3


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n', encoding='utf-8')


def setup(device):
    from cbl import env

    st = env.setup(mem_fraction='0.5', jax_platforms='cpu' if device == 'cpu' else 'cuda')
    prov = env.provenance()
    return st, {k: prov[k] for k in ('jax_backend', 'jax_devices', 'device_kind', 'packages', 'executable')}


def flow_to_model(a, rng):
    """Steps 1-3 of section 6 for CH-8 (decision U2: the Dataset device is the gate's)."""
    chang_model = sys.modules['model']
    flow = rng.SeedFlow(a.seed, dataset_device='cpu:0' if a.device == 'cpu' else 'gpu:0')
    flow.load_data(a.archive, expect_sha256=a.archive_sha, data_manifest_sha256=a.data_manifest_sha)
    flow.seed_streams()
    flow.build_datasets()
    from cbl import lossguard

    model = flow.build_model(lambda: chang_model.get_model(*rng.CH8_BUILDER_ARGS), arm_id='CH-8')
    lossguard.guard(model)
    return flow


def cmd_draw(a, rec):
    from cbl import rng

    flow = flow_to_model(a, rng)
    mpath, msha, man = rng.draw_initial_tensors(flow, a.out_dir, place=a.place, code_sha=a.code_sha)
    rec.update(flow=flow.record(), manifest=str(mpath), manifest_sha256=msha, file=man['file'], n_roles=man['n_roles'],
               role_order=man['role_order'], probe_forward=man['probe_forward'],
               stochastic_quantizers=man['stochastic_quantizers'], identity=man['identity'],
               index_map_at_construction=list(flow.ds_train._index_map))


def _refused(fn):
    from cbl import rng

    try:
        fn()
    except (rng.InitTensorRefused, rng.FlowError) as e:
        return {'refused': True, 'type': type(e).__name__, 'message': str(e)}
    except Exception as e:  # noqa: BLE001 - any other exception is recorded and fails the case
        return {'refused': False, 'type': type(e).__name__, 'message': str(e)}
    return {'refused': False, 'type': None, 'message': 'no exception'}


def cmd_refusals(a, rec):
    from cbl import rng

    flow = flow_to_model(a, rng)
    model = flow.model
    roles = rng.roles_of(model, rng.CH8_ROLE_SPEC)
    v0 = rng.variables_sha256(roles)
    ident = flow.run_identity()
    src_m = Path(a.manifest)
    man = json.loads(src_m.read_bytes())
    work = Path(a.work)
    cases = {}

    def load(mpath, sha, spec=rng.CH8_ROLE_SPEC):
        return lambda: rng.load_initial_tensors(model, mpath, expect_manifest_sha256=sha, role_spec=spec,
                                                run_identity=ident)

    def tampered(name, edit=None, flip_file=False):
        d = work / f'tamper-{name}'
        d.mkdir(parents=True)
        shutil.copy2(src_m.parent / man['file']['name'], d / man['file']['name'])
        m2 = json.loads(src_m.read_bytes())
        if edit:
            edit(m2)
        b = rng.manifest_bytes(m2) if edit else src_m.read_bytes()
        (d / src_m.name).write_bytes(b)
        if flip_file:
            p = d / man['file']['name']
            raw = bytearray(p.read_bytes())
            raw[len(raw) // 2] ^= 0x01     # one bit in the middle of the file
            p.write_bytes(bytes(raw))
        return d / src_m.name, rng.sha256_bytes(b)

    other_platform = 'gpu' if ident['device']['jax_platform'] == 'cpu' else 'cpu'
    first_role = man['role_order'][0]
    specs = {
        'a_no_expected_sha': (src_m, None, None),
        'b_wrong_expected_sha': (src_m, '0' * 64, None),
        'c_file_byte_flipped': tampered('c', flip_file=True) + (None,),
        'd_seed_differs': tampered('d', lambda m: m['identity'].__setitem__('seed', a.seed + 1)) + (None,),
        'e_platform_differs': tampered('e', lambda m: m['identity']['device'].__setitem__('jax_platform', other_platform)) + (None,),
        'f_role_sha_differs': tampered('f', lambda m: m['roles'][first_role].__setitem__('sha256', '0' * 64)) + (None,),
        'g_data_differs': tampered('g', lambda m: m['identity']['data'].__setitem__('n_train', m['identity']['data']['n_train'] + 1)) + (None,),
        'h_datasets_draw_differs': tampered('h', lambda m: m['identity']['datasets'].__setitem__(
            'index_map_sha256_at_construction', '0' * 64)) + (None,),
        'i_draw_code_differs': tampered('i', lambda m: m['identity']['draw_code'].__setitem__('cbl/rng.py', '0' * 64)) + (None,),
        'j_stack_differs': tampered('j', lambda m: m['identity']['stack'].__setitem__('keras', '3.15.0')) + (None,),
        'k_role_set_differs': (src_m, a.manifest_sha, rng.RoleSpec(
            arm_id='CH-8', top_names=lambda mm: [n if n != 'ffn2' else 'ffn2x' for n in rng.ch8_top_names(mm)])),
    }
    for name, (mp, sha, spec) in specs.items():
        r = _refused(load(mp, sha, spec or rng.CH8_ROLE_SPEC))
        r['variables_unchanged'] = rng.variables_sha256(roles) == v0
        cases[name] = r
    # flow order and arguments
    f2 = rng.SeedFlow(a.seed, dataset_device=flow.dataset_device)
    cases['l_datasets_before_seed'] = _refused(lambda: f2.build_datasets())
    cases['m_seed_before_data'] = _refused(lambda: f2.seed_streams())
    cases['n_bad_device'] = _refused(lambda: rng.SeedFlow(a.seed, dataset_device='gpu'))
    cases['o_reseed_before_load'] = _refused(lambda: flow.reseed_before_fit())
    # the positive load, then the reseed's config check
    rec['positive_load'] = flow.load_initial_tensors(a.manifest, expect_manifest_sha256=a.manifest_sha,
                                                     role_spec=rng.CH8_ROLE_SPEC)
    cases['p_config_reseed_differs'] = _refused(lambda: flow.reseed_before_fit(expect_prefit_reseed=rng.prefit_reseed(a.seed) + 1))
    cases['p_config_reseed_differs']['steps_after'] = list(flow.done)
    r = flow.reseed_before_fit(expect_prefit_reseed=rng.prefit_reseed(a.seed))
    rec.update(cases=cases, reseed=r, flow=flow.record())


def cmd_run(a, rec):
    import keras
    import numpy as np
    from cbl import rng
    from hgq.utils.sugar import BetaScheduler, Dataset, FreeEBOPs, ParetoFront, PBar, PieceWiseSchedule, StopIf
    from keras.callbacks import LearningRateScheduler

    import run_train  # Chang's module, for its schedule (run_train.py:21-37); its import seeds np and random
                      # with 42 (run_train.py:17-18), before step 1 reseeds both

    flow = flow_to_model(a, rng)
    model = flow.model
    load = flow.load_initial_tensors(a.manifest, expect_manifest_sha256=a.manifest_sha, role_spec=rng.CH8_ROLE_SPEC,
                                     check_own_draw=a.check_own_draw)
    rec['load'] = load
    rec['probe_forward'] = rng.probe_forward_sha256(model)
    rec['n_batches'] = len(flow.ds_train)
    if a.no_fit:
        r = flow.reseed_before_fit()
        rec.update(flow=flow.record(), reseed=r, py_state_after_reseed=list(random.getstate()[1][:4]))
        return 0

    # run_train.py:79-100, epochs from the gate
    out = Path(a.work) / 'pareto'
    pbar = PBar('loss: {loss:.2f}/{val_loss:.2f} - acc: {accuracy:.4f}/{val_accuracy:.4f} - lr: {learning_rate:.2e} - beta: {beta:.1e}')
    ebops = FreeEBOPs()
    pareto = ParetoFront(str(out), ['val_accuracy', 'ebops'], [1, -1],
                         fname_format='epoch={epoch}-val_acc={val_accuracy:.3f}-train_acc={accuracy:.3f}-ebops={ebops}-val_loss={val_loss:.3f}.keras',
                         enable_if=lambda x: x['val_accuracy'] > 0.5 and x['ebops'] < 5e5)
    beta_sched = BetaScheduler(PieceWiseSchedule([(0, 2e-8, 'linear'), (2000, 3e-7, 'log'), (7000, 3.0e-6, 'constant')]))
    stop = StopIf(lambda x: x['ebops'] < 1e4)
    lr_sched = LearningRateScheduler(run_train.cosine_decay_restarts_schedule(3e-3, 500, t_mul=1.0, m_mul=1.0, alpha=1e-6, alpha_steps=10))
    released = [ebops, lr_sched, beta_sched, pbar, pareto, stop]

    guard = rng.StreamGuard(raise_on_violation=True)
    recorder = flow.recorder
    state = {'expected': None}
    probe_log = []

    class LabProbe(keras.callbacks.Callback):
        """Stands where the frozen-evaluation callback will stand (after StopIf)."""

        def on_epoch_begin(self, epoch, logs=None):
            recorder.mark_keras_epoch(epoch)
            exp = state['expected']
            if exp is not None and epoch < len(exp['epochs']):
                probe_log.append({'epoch': epoch, 'at': 'on_epoch_begin',
                                  'state_equals_reconstruction': rng.py_state_sha256() == exp['epochs'][epoch]['state_sha256_at_begin']})

        def on_epoch_end(self, epoch, logs=None):
            ep = recorder.epoch(epoch)
            if logs is not None:
                logs['batch_index_seq_sha256'] = ep['batch_index_seq_sha256']   # new keys only (STUDY 7.1)
                logs['index_map_sha256'] = ep['index_map_sha256']
            if a.decoy == 'watch':
                with guard.watch('frozen-eval stand-in'):
                    random.random()
            else:
                with guard.watch('frozen-eval stand-in'):
                    pass
            exp = state['expected']
            if exp is not None and epoch < len(exp['epochs']):
                probe_log.append({'epoch': epoch, 'at': 'on_epoch_end',
                                  'state_equals_reconstruction': rng.py_state_sha256() == exp['epochs'][epoch]['state_sha256_at_end'],
                                  'batch_index_seq_sha256': ep['batch_index_seq_sha256']})

    class DecoyDraw(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            random.random()

    class DecoyDataset(keras.callbacks.Callback):
        """Constructs an hgq Dataset during training, which draws (dataset.py:70,77)."""

        def on_epoch_end(self, epoch, logs=None):
            Dataset(np.zeros((5, 8, 3), np.float32), np.zeros(5, np.int32), 2, 'cpu:0', shuffle=True)

    class DecoyNumpy(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            np.random.rand()

    lab = [LabProbe()]
    decoys = {'draw': DecoyDraw, 'dataset': DecoyDataset, 'numpy': DecoyNumpy}
    if a.decoy in decoys:
        lab.append(decoys[a.decoy]())
    for cb in released:
        guard.wrap(cb, owner='released')
    for cb in lab:
        guard.wrap(cb, owner='lab')
    callbacks = released + lab
    rec['callback_order'] = [type(c).__name__ for c in callbacks]

    opt = keras.optimizers.Adam()
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'], steps_per_execution=4)

    for _ in range(a.extra_draws):          # stand-in for an arm whose build drew more often
        random.random()
    if a.no_reseed:
        guard.arm('no reseed (control)')
        rec['reseed'] = None
    else:
        rec['reseed'] = flow.reseed_before_fit(guard=guard)
    st = random.getstate()
    rec['state_at_fit'] = [st[0], list(st[1]), st[2]]
    rec['index_map_at_fit'] = list(flow.ds_train._index_map)
    state['expected'] = rng.reconstruct_fit_stream(
        reseed=rec['reseed'], state=None if rec['reseed'] is not None else st,
        index_map=rec['index_map_at_fit'], n_batches=len(flow.ds_train), epochs=a.epochs)

    exit_code = 0
    try:
        model.fit(flow.ds_train, epochs=a.epochs, validation_data=flow.ds_val, callbacks=callbacks, verbose=0)
        rec['violation'] = None
    except rng.StreamViolation as e:
        rec['violation'] = str(e)
        exit_code = EXIT_VIOLATION
    guard.disarm()
    fin = random.getstate()
    rec.update(flow=flow.record(), recorder=recorder.summary(), guard=guard.record(), probe_log=probe_log,
               final_state=[fin[0], list(fin[1]), fin[2]], final_state_sha256=rng.py_state_sha256(fin),
               final_np_state_sha256=rng.np_state_sha256(), epochs_requested=a.epochs,
               keras_version=keras.__version__)
    return exit_code


# ----------------------------------------------------------------------------------------------
# parent side: used by the gate scripts, standard library only


def smoke_inputs(data_manifest_path, archive):
    """The smoke archive's expected sha256 (manifest entry `smoke`) and the manifest's own sha256."""
    import hashlib

    raw = Path(data_manifest_path).read_bytes()
    man = json.loads(raw)
    return {'archive': str(archive), 'archive_sha256': man['smoke']['archive']['sha256'],
            'archive_rows': man['smoke']['archive']['rows'], 'record_archive_sha256': man['train']['archive']['sha256'],
            'heldout_archive_sha256': man['heldout']['archive']['sha256'],
            'data_manifest_sha256': hashlib.sha256(raw).hexdigest(), 'data_manifest': str(data_manifest_path)}


def run_child(name, cmd, *, device, seed, inputs, work, children_out, extra=(), timeout=1800):
    """Run one child in a fresh process of this interpreter; returns (exit code, record, argv).
    Its JSON record, stdout and stderr are copied to children_out/<name>.*"""
    import os
    import subprocess

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    out = work / f'{name}.json'
    argv = [sys.executable, '-I', '-B', str(FORK / 'gates' / 'rng_child.py'), cmd, '--device', device, '--seed', str(seed),
            '--archive', inputs['archive'], '--archive-sha', inputs['archive_sha256'],
            '--data-manifest-sha', inputs['data_manifest_sha256'], '--out', str(out)] + [str(x) for x in extra]
    env = {k: v for k, v in os.environ.items() if k not in ('KERAS_BACKEND', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_MEM_FRACTION')}
    with open(work / f'{name}.stdout.txt', 'w') as so, open(work / f'{name}.stderr.txt', 'w') as se:
        r = subprocess.run(argv, stdout=so, stderr=se, cwd=str(work), env=env, timeout=timeout)
    rec = json.loads(out.read_text()) if out.exists() else None
    if children_out:
        co = Path(children_out)
        co.mkdir(parents=True, exist_ok=True)
        for suffix in ('.json', '.stdout.txt', '.stderr.txt'):
            src = work / f'{name}{suffix}'
            if src.exists():
                shutil.copy2(src, co / src.name)
        (co / f'{name}.argv.txt').write_text(' '.join(argv) + '\n')
    return r.returncode, rec, argv


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('cmd', choices=['draw', 'run', 'refusals'])
    ap.add_argument('--device', choices=['cpu', 'gpu'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--archive-sha', required=True)
    ap.add_argument('--data-manifest-sha', required=True)
    ap.add_argument('--out', required=True, help='JSON record path')
    ap.add_argument('--out-dir', help='draw: directory for the file and manifest')
    ap.add_argument('--place', default='engineering')
    ap.add_argument('--code-sha', default=None)
    ap.add_argument('--manifest')
    ap.add_argument('--manifest-sha')
    ap.add_argument('--work')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--extra-draws', type=int, default=0)
    ap.add_argument('--no-reseed', action='store_true')
    ap.add_argument('--no-fit', action='store_true')
    ap.add_argument('--check-own-draw', action='store_true')
    ap.add_argument('--decoy', choices=['none', 'draw', 'dataset', 'numpy', 'watch'], default='none')
    a = ap.parse_args()
    rec = {'cmd': a.cmd, 'args': vars(a)}
    code = 0
    try:
        st, prov = setup(a.device)
        rec.update(setup_precision=st['matmul_precision'], provenance=prov)
        if a.cmd == 'draw':
            cmd_draw(a, rec)
        elif a.cmd == 'refusals':
            cmd_refusals(a, rec)
        else:
            code = cmd_run(a, rec) or 0
    except Exception as e:  # noqa: BLE001 - recorded for the parent, then exit 1
        rec['error'] = f'{type(e).__name__}: {e}'
        rec['traceback'] = traceback.format_exc()
        code = 1
    rec['exit'] = code
    dump(a.out, rec)
    return code


if __name__ == '__main__':
    sys.exit(main())
