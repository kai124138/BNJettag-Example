"""Gate G11 (PREFLIGHT-PLAN section 3 row G11; STUDY v0.8 section 6 step 7, section 7.1 last
bullet; decision U5; module M4 cbl/rng.py). Run once with --device cpu (PCC) and once with
--device gpu (PCG, the RTX 4060 Ti), in the venv of record, on the 2-file smoke archive of B2.

Per device, each in a fresh process (gates/rng_child.py): CH-8's initial tensors for seed 1 drawn
on this device (decision U1), then six CH-8 runs of --epochs epochs that load them by role, with
Chang's compile and callbacks (run_train.py:79-100), the recording training Dataset and the lab
probe after StopIf:
  A, B  the section-6 flow with the pre-fit reseed (two independent runs of the same arm and seed);
  C     as A, with three extra draws from the module-level stream between the load and the
        reseed, standing in for an arm whose model build draws more initializer seeds;
  D     no reseed (the stream continues from the build, as in the release); control;
  E     as D with one extra draw (two 32-bit words); recorded, not a pass condition (below);
  F     as D with 100 extra draws (200 words); control.
Checks:
1. Every run: one worker, no multiprocessing, drop_last False, shuffle True; no recorder anomaly;
   one pre-fit Dataset epoch (Keras's symbolic build) with two fetches; per Keras epoch n fetches,
   the mapped indices a permutation and equal to `_index_map[raw index]`, the map unchanged during
   the epoch, `on_epoch_begin` drawing nothing (drop_last False, critical C34); the lab probe's
   in-run comparison of the live stream with the reconstruction true at every epoch begin and end.
2. Reconstruction (this gate, from the reseed value, or for D and E from the stream state at
   `fit`, and the training map at that point): pre-fit order and fetches, per epoch the batch
   order, the mapped indices, the map before and after its reshuffle, the stream state at the
   Dataset's epoch begin, after the order shuffle and after the map's reshuffle, and the stream's
   final state, all equal to the run's record. The first divergence is printed.
3. A, B and C have identical per-epoch `batch_index_seq_sha256` and `index_map_sha256`, and A's map
   at the reseed is the construction map of the draw. Controls: D differs from A; F differs from D
   in every epoch; A's record differs from the reconstruction under prefit_reseed(2). So the
   comparison can fail, and without the reseed a different draw history changes the batch order.
   Changed during development, before any run of record: the first draft required E to differ
   from D. In a development run E's training epochs equalled D's: `random.shuffle` draws through
   `Random._randbelow`, which rejects and redraws, so the number of 32-bit words a shuffle consumes
   varies, and a two-word offset was absorbed within the pre-fit shuffles (E's pre-fit fetches
   differed from D's, its training epochs did not). An offset can re-synchronize only if the
   shorter stream rejects that many more times, which 200 words over these few shuffles cannot,
   so F carries the control; E stays as a record that, without the reseed, whether two draw
   histories give the same batch order is a matter of chance.
4. Across arms (added in B5): the runner (M13, gates/runner_child.py run) trains CH-8 (Chang's
   get_model) and F000 (M5) for --epochs epochs, each loading the same draw by role, under the
   default XLA flags; the per-epoch `batch_index_seq_sha256` and `index_map_sha256` of the two equal
   each other, the reconstruction from prefit_reseed(1) and run A's (the child's copy of the recipe).
5. Binary arms (added in B6): the runner also trains F111-FB (headline H1) and F111-BF (headline H2),
   one binary arm of each form, from the same draw; their per-epoch hashes equal CH-8's and F000's.

    <venv>/bin/python -I -B gates/g11_batch_order.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
        --archive SMOKE_ARCHIVE [--epochs 3] [--data-manifest PATH] [--json PATH] [--children-out DIR]
"""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import gatelib  # noqa: E402
import rng_child  # noqa: E402
import runner_child  # noqa: E402
import runnergate  # noqa: E402
from cbl import rng  # noqa: E402

GATE = 'G11'
RUNS = {'A': {'extra': 0, 'reseed': True}, 'B': {'extra': 0, 'reseed': True}, 'C': {'extra': 3, 'reseed': True},
        'D': {'extra': 0, 'reseed': False}, 'E': {'extra': 1, 'reseed': False}, 'F': {'extra': 100, 'reseed': False}}


def first_divergence(rec, exp, epochs):
    """None if the run's record equals the reconstruction, else a description of the first difference."""
    pre = rec['recorder']['prefit']
    if len(pre) != 1:
        return f'{len(pre)} pre-fit Dataset epochs, expected 1'
    p, q = pre[0], exp['prefit']
    for key, a, b in (('pre-fit order', p['raw'], q['order']), ('pre-fit fetches', p['mapped'], q['mapped']),
                      ('pre-fit map', p['index_map'], q['index_map']), ('pre-fit map after reset', p['index_map_after_end'], q['index_map_after']),
                      ('pre-fit state at begin', p['state_sha256_at_begin'], q['state_sha256_at_begin']),
                      ('pre-fit state after order', p['state_sha256_before_end'], q['state_sha256_after_order']),
                      ('pre-fit state after reset', p['state_sha256_after_end'], q['state_sha256_at_end'])):
        if a != b:
            return f'{key}: run {a} != reconstruction {b}'
    eps = rec['recorder']['epochs']
    if len(eps) != epochs:
        return f'{len(eps)} recorded epochs, expected {epochs}'
    for e, (r, x) in enumerate(zip(eps, exp['epochs'])):
        for key, a, b in (('batch order', r['raw'], x['order']), ('mapped fetches', r['mapped'], x['mapped']),
                          ('index map', r['index_map'], x['index_map']), ('index map after reshuffle', r['index_map_after_end'], x['index_map_after']),
                          ('state at epoch begin', r['state_sha256_at_begin'], x['state_sha256_at_begin']),
                          ('state after the order shuffle', r['state_sha256_before_end'], x['state_sha256_after_order']),
                          ('state after the reshuffle', r['state_sha256_after_end'], x['state_sha256_at_end']),
                          ('batch_index_seq_sha256', r['batch_index_seq_sha256'], x['batch_index_seq_sha256']),
                          ('index_map_sha256', r['index_map_sha256'], x['index_map_sha256'])):
            if a != b:
                return f'epoch {e} {key}: run {a} != reconstruction {b}'
    if rec['final_state_sha256'] != exp['end_state_sha256']:
        return f'final stream state {rec["final_state_sha256"][:12]} != reconstruction {exp["end_state_sha256"][:12]}'
    return None


def hashes(rec):
    return [(e['batch_index_seq_sha256'], e['index_map_sha256']) for e in rec['recorder']['epochs']]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--device', choices=['cpu', 'gpu'], required=True)
    ap.add_argument('--expect-sha12', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--data-manifest', default=str(FORK / 'manifests' / 'data-prep-150c.json'))
    ap.add_argument('--json', default=None)
    ap.add_argument('--children-out', default=None)
    a = ap.parse_args()
    g = gatelib.Gate(GATE)
    rec = {'header': g.header(a.expect_sha12), 'device': a.device, 'epochs': a.epochs}
    gatelib.check_venv_of_record(g)
    work = Path(a.work)
    g.check(not work.exists(), f'work directory {work} does not exist yet')
    work.mkdir(parents=True)
    inputs = rng_child.smoke_inputs(a.data_manifest, a.archive)
    rec['inputs'] = inputs
    got = rng.sha256_file(a.archive)
    g.check(got == inputs['archive_sha256'] and got not in (inputs['record_archive_sha256'], inputs['heldout_archive_sha256']),
            f'the archive is the smoke archive {got[:12]} (not the archive of record, not the held-out archive)')

    def child(name, cmd, extra):
        code, r, argv = rng_child.run_child(name, cmd, device=a.device, seed=1, inputs=inputs, work=work / name,
                                            children_out=a.children_out, extra=extra)
        g.info(f'{name} argv', ' '.join(argv[3:]))
        return code, r

    _, d = child('draw-s1', 'draw', ['--out-dir', work / 'draw-s1' / 'init', '--place', f'PC{a.device[0].upper()} gate G11',
                                     '--code-sha', rec['header']['HEAD']])
    if not g.check(d is not None and d.get('exit') == 0 and 'error' not in d, f'draw-s1: exit 0 ({d.get("error") if d else "no record"})'):
        g.check_clean('at end')
        return g.finish(rec, a.json)
    dev = d['identity']['device']
    if a.device == 'gpu':
        g.check(dev['jax_platform'] == 'gpu' and all('4060 Ti' in k for k in dev['device_kind']) and dev['dataset_device'] == 'gpu:0',
                f'draw on the RTX 4060 Ti: {dev}')
    else:
        g.check(dev['jax_platform'] == 'cpu' and dev['dataset_device'] == 'cpu:0', f'draw on the CPU: {dev}')
    rec['draw'] = {'manifest_sha256': d['manifest_sha256'], 'file_sha256': d['file']['sha256'], 'device': dev}

    runs = {}
    for name, spec in RUNS.items():
        extra = ['--manifest', d['manifest'], '--manifest-sha', d['manifest_sha256'], '--work', work / f'run-{name}',
                 '--epochs', a.epochs, '--extra-draws', spec['extra']]
        if not spec['reseed']:
            extra.append('--no-reseed')
        code, r = child(f'run-{name}', 'run', extra)
        runs[name] = r
        ok = g.check(code == 0 and r is not None and r.get('exit') == 0 and r.get('violation') is None and 'error' not in r,
                     f'run {name}: exit 0, no stream violation ({(r or {}).get("error") or (r or {}).get("violation")})')
        if not ok:
            continue
        rd = r['recorder']
        di = rd['dataset']
        g.check(di['workers'] == 1 and not di['use_multiprocessing'] and di['drop_last'] is False and di['shuffle'] is True
                and di['class'].endswith('CblRecordingDataset') and di['base'] == 'hgq.utils.dataset.Dataset',
                f'run {name}: training Dataset {di}')
        g.check(r['keras_version'] == rng.KERAS_MODELLED, f'run {name}: Keras {r["keras_version"]} is the modelled {rng.KERAS_MODELLED}')
        g.check(rd['anomalies'] == [] and not rd['open_segment'], f'run {name}: no recorder anomaly {rd["anomalies"][:3]}')
        g.check(r['load']['n_loaded'] == rng.CH8_N_VARIABLES and r['load']['unmatched'] == [] and r['load']['unused'] == [],
                f'run {name}: 244 tensors loaded by role from the draw, unmatched and unused empty')
        n = r['n_batches']
        g.check(len(rd['prefit']) == 1 and rd['prefit'][0]['n_fetches'] == min(2, n),
                f'run {name}: one pre-fit Dataset epoch with {rd["prefit"][0]["n_fetches"] if rd["prefit"] else None} fetches (Keras symbolic build)')
        eps = rd['epochs']
        g.check(len(eps) == a.epochs and all(e['n_fetches'] == n and e['mapped_is_permutation'] and e['mapped_equals_map_of_raw']
                                             and e['index_map_unchanged_during_epoch'] and not e['on_epoch_begin_drew'] for e in eps),
                f'run {name}: {len(eps)} epochs of {n} fetches each, mapped = map[raw], a permutation, map fixed within the epoch, '
                'on_epoch_begin draws nothing')
        pl = r['probe_log']
        g.check(len(pl) == 2 * a.epochs and all(x['state_equals_reconstruction'] for x in pl),
                f'run {name}: in-run stream state equals the reconstruction at every epoch begin and end ({len(pl)} points)')
        st = r['state_at_fit']
        exp = rng.reconstruct_fit_stream(reseed=r['reseed'], state=None if spec['reseed'] else (st[0], tuple(st[1]), st[2]),
                                         index_map=r['index_map_at_fit'], n_batches=n, epochs=a.epochs)
        div = first_divergence(r, exp, a.epochs)
        g.check(div is None, f'run {name}: record equals this gate\'s reconstruction from the '
                + ('reseed' if spec['reseed'] else 'stream state at fit') + ('' if div is None else f'; first divergence: {div}'))
        if spec['reseed']:
            g.check(r['reseed'] == rng.prefit_reseed(1), f'run {name}: reseed is prefit_reseed(1)')
            g.check(r['index_map_at_fit'] == d['index_map_at_construction'],
                    f'run {name}: the map at the reseed is the construction map of the draw')
        g.info(f'run {name} per-epoch (batch_index_seq_sha256, index_map_sha256), engineering', hashes(r))

    if all(runs.get(k) and runs[k].get('exit') == 0 and 'recorder' in runs[k] for k in RUNS):
        hA, hB, hC, hD, hE, hF = (hashes(runs[k]) for k in 'ABCDEF')
        g.check(hA == hB, 'A and B (same arm, same seed, separate processes): identical per-epoch hashes')
        g.check(hA == hC, 'A and C (C drew three more times before the reseed): identical per-epoch hashes')
        g.check(hD != hA, 'control: D (no reseed) differs from A')
        g.check(all(x != y for x, y in zip(hF, hD)) and len(hF) == len(hD) == a.epochs,
                'control: F (no reseed, 100 extra draws) differs from D in every epoch, as arms with different draw '
                'histories can without the reseed (STUDY section 6 step 5)')
        e_same = [x == y for x, y in zip(hE, hD)]
        rec['E_vs_D'] = {'training_epochs_equal': e_same,
                         'prefit_fetches_equal': runs['E']['recorder']['prefit'][0]['mapped'] == runs['D']['recorder']['prefit'][0]['mapped'],
                         'prefit_order_equal': runs['E']['recorder']['prefit'][0]['raw'] == runs['D']['recorder']['prefit'][0]['raw']}
        g.info('E (no reseed, one extra draw) against D, recorded only', rec['E_vs_D'])
        wrong = rng.reconstruct_fit_stream(reseed=rng.prefit_reseed(2), index_map=runs['A']['index_map_at_fit'],
                                           n_batches=runs['A']['n_batches'], epochs=a.epochs)
        g.check(first_divergence(runs['A'], wrong, a.epochs) is not None,
                'control: A\'s record differs from the reconstruction under prefit_reseed(2)')
        rec['per_epoch_hashes'] = {k: hashes(runs[k]) for k in RUNS}
        rec['prefit'] = {k: runs[k]['recorder']['prefit'][0]['mapped'] for k in RUNS}
    else:
        g.check(False, 'all six runs completed; the pairing checks need them')
    # 4: across arms, through the runner (B5)
    bundle = runnergate.sha256_file(runnergate.BUNDLE)
    arm_hashes = {}
    for arm, form, hl in (('CH-8', None, None), ('F000', None, None), ('F111', 'FB', 'H1'), ('F111', 'BF', 'H2')):
        label = arm + (f'-{form}' if form else '')
        name = f'runner-{label}'
        code, r, argv, _ = runner_child.run_child(
            name, 'run', device=a.device, seed=1, archive=inputs['archive'], archive_sha=inputs['archive_sha256'],
            data_manifest_sha=inputs['data_manifest_sha256'], work=work / name, children_out=a.children_out,
            extra=['--arm', arm, '--epochs', a.epochs, '--manifest', d['manifest'], '--manifest-sha', d['manifest_sha256'],
                   '--code-sha', rec['header']['HEAD'], '--source-bundle', bundle, '--out-dir', work / name / 'out']
            + (['--form', form, '--headline', hl] if form else []))
        g.info(f'{name} argv', ' '.join(argv[3:]))
        ok = g.check(code == 0 and r is not None and 'error' not in r and r['runner'].get('end') == 'completed',
                     f'{name}: the runner completed {a.epochs} epochs ({(r or {}).get("error") or ((r or {}).get("runner") or {}).get("error")})')
        if ok:
            R = r['runner']
            arm_hashes[label] = [(e['batch_index_seq_sha256'], e['index_map_sha256']) for e in R['epoch_records']]
            n_unm = 27 if form else 0
            g.check(len(R['load']['unmatched']) == n_unm and len(R['load']['unused']) == n_unm
                    and R['reseed'] == rng.prefit_reseed(1) and R['index_map_at_reseed'] == d['index_map_at_construction'],
                    f'{name}: loaded the draw by role ({n_unm} unmatched, {n_unm} unused), reseeded with prefit_reseed(1), '
                    f'map at the reseed is the construction map')
            rd = R['recorder']
            g.check(rd['anomalies'] == [] and len(rd['prefit']) == 1 and all(e['mapped_is_permutation'] and e['mapped_equals_map_of_raw']
                                                                             and e['index_map_unchanged_during_epoch'] for e in rd['epochs']),
                    f'{name}: recorder without anomaly; one pre-fit record; every epoch a permutation equal to map[raw]')
    if len(arm_hashes) == 4:
        exp = rng.reconstruct_fit_stream(reseed=rng.prefit_reseed(1), index_map=d['index_map_at_construction'],
                                         n_batches=runs['A']['n_batches'] if runs.get('A') else 0, epochs=a.epochs)
        he = [(e['batch_index_seq_sha256'], e['index_map_sha256']) for e in exp['epochs']]
        hA = hashes(runs['A']) if runs.get('A') and 'recorder' in runs['A'] else None
        g.check(arm_hashes['CH-8'] == arm_hashes['F000'] == he and (hA is None or hA == he),
                'across arms: CH-8 and F000 through the runner fetch the same batches in every epoch, equal to the '
                'reconstruction from prefit_reseed(1) and to run A')
        g.check(arm_hashes['F111-FB'] == arm_hashes['F111-BF'] == arm_hashes['CH-8'],
                'across arms (B6): F111-FB and F111-BF, one binary arm of each form, fetch the same batches in every '
                'epoch as CH-8 and F000')
        rec['across_arms'] = arm_hashes
    else:
        g.check(False, f'all four runner arms completed ({sorted(arm_hashes)})')
    g.check_clean('at end')
    return g.finish(rec, a.json)


if __name__ == '__main__':
    sys.exit(main())
