"""Gate G12 (PREFLIGHT-PLAN section 3 row G12; STUDY v0.8 section 6 step 6; module M4 cbl/rng.py,
`StreamGuard`). PC CPU, venv of record, 2-file smoke archive of B2.

Each run is CH-8 for seed 1 in a fresh process (gates/rng_child.py): the initial tensors drawn on
the CPU and loaded by role, the pre-fit reseed, Chang's compile and callbacks (run_train.py:79-100)
wrapped by the guard as released code, and the lab probe after StopIf (it stands where the
frozen-evaluation callback will stand, marks epochs and writes the two batch-order hashes as new
log keys, and runs a stand-in block under `guard.watch`) wrapped as lab code. The guard compares
the module-level random state and NumPy's legacy state before and after every call of every
overridden hook, from the reseed to the end of fit, and raises on a change made by lab code.
Checks:
1. Clean run (--epochs epochs): no violation by any lab or released callback; the wrapped hooks are
   the ones each class overrides, and every wrapped epoch hook ran once per epoch; the guard was
   armed in the state Random(prefit_reseed(1)) and disarmed in the run's final state, which equals
   this gate's reconstruction of the two consumers' draws (no other draw happened anywhere); the
   probe's in-run comparison with the reconstruction holds at every epoch boundary.
2. Controls, each a run with one decoy lab callback appended after the probe, or a decoy step under
   `guard.watch`, each expected to stop at its first call with StreamViolation (exit 3) naming
   itself: a `random.random()` draw; the construction of an hgq Dataset during training, which
   draws (dataset.py:70,77; STUDY section 6 step 6); a NumPy global draw; a draw inside the watched
   stand-in for the frozen-evaluation step. No released callback is named in any of them.
3. The frozen evaluation (M8, since B7), through the runner (gates/runner_child.py run --no-probe):
   CH-8 and F111-BF (headline H2), seed 1, the same epochs, the file drawn on the CPU loaded by role. M8's
   evaluation runs inside `guard.watch('frozen evaluation')` in the appended callback, wrapped as lab
   code: no violation by any lab or released callback; the watched block ran once per epoch with no change;
   the appended callback is wrapped as lab with its overridden hooks; the stream at the end of training
   equals the reconstruction of the Dataset's and the adapter's draws alone. Control: the same run with
   M8's evaluation made to draw from the module-level stream first (--decoy-frozen) stops at its first
   evaluation with StreamViolation (exit 3) naming the frozen-evaluation block, and no released callback.
The children of part 3 run under the open-file audit of the held-out paths.

    <venv>/bin/python -I -B gates/g12_stream_guard.py --expect-sha12 SHA12 --work DIR --archive SMOKE_ARCHIVE \\
        [--epochs 3] [--data-manifest PATH] [--json PATH] [--children-out DIR]
"""

import argparse
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import gatelib  # noqa: E402
import rng_child  # noqa: E402
import runnergate as rg  # noqa: E402
from cbl import rng  # noqa: E402

GATE = 'G12'
RELEASED = ['FreeEBOPs', 'LearningRateScheduler', 'BetaScheduler', 'PBar', 'ParetoFront', 'StopIf']
DECOYS = {  # decoy -> (who, hook, python_random, numpy_global)
    'draw': ('DecoyDraw', 'on_epoch_end', True, False),
    'dataset': ('DecoyDataset', 'on_epoch_end', True, False),
    'numpy': ('DecoyNumpy', 'on_epoch_end', False, True),
    'watch': ('frozen-eval stand-in', 'watch', True, False),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--expect-sha12', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--data-manifest', default=str(FORK / 'manifests' / 'data-prep-150c.json'))
    ap.add_argument('--json', default=None)
    ap.add_argument('--children-out', default=None)
    a = ap.parse_args()
    g = gatelib.Gate(GATE)
    rec = {'header': g.header(a.expect_sha12), 'epochs': a.epochs}
    gatelib.check_venv_of_record(g)
    work = Path(a.work)
    rg.fileaudit.require_unsynced(work)
    g.check(not work.exists(), f'work directory {work} does not exist yet')
    work.mkdir(parents=True)
    inputs = rng_child.smoke_inputs(a.data_manifest, a.archive)
    rec['inputs'] = inputs
    got = rng.sha256_file(a.archive)
    g.check(got == inputs['archive_sha256'] and got not in (inputs['record_archive_sha256'], inputs['heldout_archive_sha256']),
            f'the archive is the smoke archive {got[:12]} (not the archive of record, not the held-out archive)')

    def child(name, cmd, extra):
        code, r, argv = rng_child.run_child(name, cmd, device='cpu', seed=1, inputs=inputs, work=work / name,
                                            children_out=a.children_out, extra=extra)
        g.info(f'{name} argv', ' '.join(argv[3:]))
        return code, r

    _, d = child('draw-s1', 'draw', ['--out-dir', work / 'draw-s1' / 'init', '--place', 'PCC gate G12', '--code-sha', rec['header']['HEAD']])
    if not g.check(d is not None and d.get('exit') == 0 and 'error' not in d, f'draw-s1: exit 0 ({d.get("error") if d else "no record"})'):
        g.check_clean('at end')
        return g.finish(rec, a.json)
    base = ['--manifest', d['manifest'], '--manifest-sha', d['manifest_sha256']]

    # 1. clean run
    code, r = child('run-clean', 'run', base + ['--work', work / 'run-clean', '--epochs', a.epochs])
    if g.check(code == 0 and r is not None and r.get('exit') == 0 and r.get('violation') is None and 'error' not in r,
               f'run-clean: exit 0, no StreamViolation ({(r or {}).get("error") or (r or {}).get("violation")})'):
        gd = r['guard']
        g.check(gd['lab_violations'] == [] and gd['released_violations'] == [],
                f'run-clean: no lab and no released callback changed either stream ({len(gd["violations"])} violations)')
        g.check(r['callback_order'] == RELEASED + ['LabProbe'],
                f'run-clean: callback order {r["callback_order"]} (released order, then the lab probe)')
        wrapped = {w['who']: (w['owner'], w['hooks']) for w in gd['wrapped']}
        g.check(all(wrapped.get(n, ('', []))[0] == 'released' and wrapped[n][1] for n in RELEASED)
                and wrapped.get('LabProbe', ('', []))[0] == 'lab',
                f'run-clean: every callback wrapped with its overridden hooks {wrapped}')
        calls = gd['calls']
        epoch_hooks = {k: v for k, v in calls.items() if k.endswith('.on_epoch_end') or k.endswith('.on_epoch_begin')}
        g.check(epoch_hooks and all(v['calls'] == a.epochs and v['py_changed'] == 0 and v['np_changed'] == 0 for v in epoch_hooks.values()),
                f'run-clean: each wrapped epoch hook ran {a.epochs} times with no change ({len(epoch_hooks)} hooks)')
        w = calls.get('frozen-eval stand-in.watch', {})
        g.check(w.get('calls') == a.epochs and w.get('py_changed') == 0 and w.get('np_changed') == 0,
                f'run-clean: the watched stand-in block ran {w.get("calls")} times with no change')
        batch = {k: v for k, v in calls.items() if 'batch' in k}
        g.check(all(v['py_changed'] == 0 and v['np_changed'] == 0 for v in batch.values()),
                f'run-clean: batch hooks {sorted(batch)} ran {[v["calls"] for v in batch.values()]} times with no change')
        u5 = rng.prefit_reseed(1)
        g.check(gd['armed_at']['py_state_sha256'] == rng.py_state_sha256(random.Random(u5).getstate()),
                'run-clean: the guard was armed right after the reseed, in the state Random(prefit_reseed(1))')
        exp = rng.reconstruct_fit_stream(reseed=r['reseed'], index_map=r['index_map_at_fit'], n_batches=r['n_batches'], epochs=a.epochs)
        g.check(gd['disarmed_at']['py_state_sha256'] == r['final_state_sha256'] == exp['end_state_sha256'],
                'run-clean: the stream at the end of training equals the reconstruction of the Dataset\'s and the adapter\'s '
                'draws alone (no other draw anywhere between the reseed and the end of fit)')
        pl = r['probe_log']
        g.check(len(pl) == 2 * a.epochs and all(x['state_equals_reconstruction'] for x in pl),
                f'run-clean: in-run stream state equals the reconstruction at every epoch begin and end ({len(pl)} points)')

    # 2. decoys
    for decoy, (who, hook, py, npg) in DECOYS.items():
        code, r = child(f'run-decoy-{decoy}', 'run', base + ['--work', work / f'run-decoy-{decoy}', '--epochs', a.epochs, '--decoy', decoy])
        if not g.check(code == rng_child.EXIT_VIOLATION and r is not None and r.get('violation'),
                       f'decoy {decoy}: the run stopped with StreamViolation, exit {code} ({(r or {}).get("violation") or (r or {}).get("error")})'):
            continue
        lab = r['guard']['lab_violations']
        g.check(len(lab) >= 1 and (lab[0]['who'], lab[0]['hook'], lab[0]['python_random'], lab[0]['numpy_global']) == (who, hook, py, npg)
                and f'{who}.{hook}' in r['violation'],
                f'decoy {decoy}: first violation {lab[0] if lab else None} names {who}.{hook}')
        g.check(r['guard']['released_violations'] == [], f'decoy {decoy}: no released callback named')
        g.check(len(r['recorder']['epochs']) == 1, f'decoy {decoy}: stopped in epoch 0 ({len(r["recorder"]["epochs"])} closed epochs)')
    # 3. the frozen evaluation (M8) through the runner
    inp = rg.inputs(a.data_manifest, a.archive)
    ph = work / 'audit-placeholder'
    ph.mkdir()
    (ph / '150c-test.h5').write_bytes(b'')
    ch = rg.Children(g, device='cpu', inp=inp, work=work, children_out=a.children_out, audit_watches=rg.watches(inp, ph))
    head = rec['header']['HEAD']
    dr = {'manifest': d['manifest'], 'manifest_sha256': d['manifest_sha256']}
    for arm, form, headline in (('CH-8', None, None), ('F111', 'BF', 'H2')):
        lab = arm if form is None else f'{arm}-{form}'
        extra = rg.draw_args(dr, head, inp['bundle_sha256'], epochs=a.epochs, arm=arm) + ['--no-probe']
        extra += (['--form', form] if form else []) + (['--headline', headline] if headline else [])
        R, _ = ch(f'm8-{lab}', 'run', 1, extra + ['--out-dir', work / f'm8-{lab}' / 'out'])
        if R is None:
            continue
        rr = R['runner']
        gd = rr['guard']
        g.check(gd['lab_violations'] == [] and gd['released_violations'] == [],
                f'm8-{lab}: no lab and no released callback changed either stream ({len(gd["violations"])} violations)')
        w = gd['calls'].get('frozen evaluation.watch', {})
        g.check(w.get('calls') == a.epochs and w.get('py_changed') == 0 and w.get('np_changed') == 0,
                f'm8-{lab}: M8\'s frozen evaluation ran {w.get("calls")} times inside the watched block with no change')
        wrapped = {x['who']: (x['owner'], x['hooks']) for x in gd['wrapped']}
        g.check(wrapped.get('FrozenEvalHook', ('', []))[0] == 'lab' and wrapped['FrozenEvalHook'][1]
                and rr['callback_order'] == RELEASED + ['FrozenEvalHook'],
                f'm8-{lab}: callback order {rr["callback_order"]}, the appended callback wrapped as lab code with hooks '
                f'{wrapped.get("FrozenEvalHook", ("", []))[1]}')
        rows = R.get('frozen_rows') or []
        g.check(len(rows) == a.epochs and all(x['H1'] is not None and x['vars_sha256_before'] == x['vars_sha256_after']
                                              for x in rows),
                f'm8-{lab}: {len(rows)} frozen records (H1 {[x["H1"] for x in rows]}), variables unchanged by each evaluation')
        exp = rng.reconstruct_fit_stream(reseed=rr['reseed'], index_map=rr['index_map_at_reseed'],
                                         n_batches=rr['flow']['datasets']['n_batches'], epochs=a.epochs)
        g.check(gd['disarmed_at']['py_state_sha256'] == exp['end_state_sha256'],
                f'm8-{lab}: the stream at the end of training equals the reconstruction of the Dataset\'s and the adapter\'s '
                f'draws alone (no draw by the frozen evaluation)')
    code, R, argv, audit = rg.runner_child.run_child(
        'm8-decoy', 'run', device='cpu', seed=1, archive=a.archive, archive_sha=inp['archive_sha256'],
        data_manifest_sha=inp['data_manifest_sha256'], work=work / 'm8-decoy', children_out=a.children_out,
        extra=rg.draw_args(dr, head, inp['bundle_sha256'], epochs=a.epochs, arm='CH-8')
        + ['--no-probe', '--decoy-frozen', '--out-dir', work / 'm8-decoy' / 'out'], watches=rg.watches(inp, ph))
    g.info('m8-decoy argv', ' '.join(argv[3:]))
    rr = (R or {}).get('runner') or {}
    lab_v = (rr.get('guard') or {}).get('lab_violations') or []
    g.check(R is not None and rr.get('end') == 'integrity_stop' and rr.get('exit') == rng_child.EXIT_VIOLATION
            and lab_v and lab_v[0]['who'] == 'frozen evaluation' and lab_v[0]['hook'] == 'watch'
            and not rr['guard']['released_violations'],
            f'decoy m8: M8\'s evaluation drawing from random stops the run with StreamViolation, exit {rr.get("exit")}, '
            f'first violation {lab_v[0] if lab_v else None}')
    g.check_clean('at end')
    return g.finish(rec, a.json)


if __name__ == '__main__':
    sys.exit(main())
