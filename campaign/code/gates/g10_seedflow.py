"""Gate G10 (PREFLIGHT-PLAN section 3 row G10; STUDY v0.8 section 6 "RNG flow" steps 1-5 and the
section 13 RNG row; decisions U1 and U5; module M4 cbl/rng.py). PC CPU, venv of record.

Checks, each child in a fresh process (gates/rng_child.py), on the 2-file smoke archive of B2:
1. Seed flow in the section-6 order. Three draws of CH-8's initial tensors (seed 1 twice, seed 2):
   the flow ran np.random.seed(42) and M3's reader (step 1), random.seed(s) and
   keras.utils.set_random_seed(s) (step 2: random equals Random(s), NumPy equals RandomState(s)),
   the two Datasets (step 3: drop_last False, shuffle as released) and then the model. This gate
   reconstructs the Datasets' construction draws from random.Random(s) itself (randint(0,
   remainder), the map's shuffle, the validation randint; dataset.py:66-79) and compares them and
   the stream's state after them with the child's record.
2. The draw (decision U1): 244 roles, unique, first segments CH-8's semantic layer names; no
   quantizer with a seed generator (so hgq's NumPy draws at quantizer construction,
   quantizer/internal/base.py:83, affect nothing); two draws of seed 1 give the same file bytes,
   the same tensors and the same forward probe; seed 2 differs from seed 1 in exactly the 11
   Glorot kernels (embedding, Q, K, V, O, the two FFN contractions, three head layers, logits),
   every other tensor being a constant initializer; the manifest's identity names seed, device,
   Dataset device, the smoke archive and the data manifest.
3. Loading by role (section 6 step 4): a CH-8 run for seed 1 repeats steps 1-3, and before loading
   its own draw equals the file in all 244 roles; after loading every variable equals the file;
   the unmatched set (arm roles not in the file) and the unused set (file roles not in the arm)
   are empty on CH-8; loading draws from neither stream; the forward probe equals the drawing
   process's; the stream after the build equals the draw's (same draws in the same order); the
   pre-fit reseed is prefit_reseed(1) of decision U5 and leaves random in Random(that) state.
4. Refusals: no expected manifest sha256; a wrong one; one flipped bit in the tensor file; the
   manifest's seed, device platform, data identity, construction draw, draw-code hash or Keras
   version edited (with the expected sha256 of the edited manifest); a role's sha256 edited; a
   role map that renames one layer (unmatched and unused not as declared). Each is refused before
   any variable is assigned. Steps out of order and a bad Dataset device are refused by the flow,
   and a config prefit_reseed that differs from U5 is refused.
5. F000 (added in B5): the runner (M13, gates/runner_child.py step) builds F000 with M5 for seed 1
   on the CPU and loads the draw-s1 file by role: 244 tensors, the unmatched and unused sets empty
   as F000's RoleSpec declares, F000's own draw equal to CH-8's file in all 244 roles before the
   load, and the load drawing from neither stream.
6. Binary arms (added in B6): the runner builds F100, F111-FB, F111-BF, GR-BF and BE-FB with M5 and M6
   for seed 1 on the CPU and loads the draw-s1 file by role: 244 - 3n tensors at n binary sites, the
   unmatched set exactly the scale quantizers' `<site>/qg/FixedPointQuantizerKBI[0]/{k,b,i}` and the
   unused set exactly Chang's kq `<site>/kq/FixedPointQuantizerKBI[0]/{k,b,i}` at the sites this gate
   lists from STUDY section 2.2 "Sites" (not from M5), the arm's own draw equal to CH-8's file in every
   loaded role, the unmatched tensors at their constant initializers (k = 1, b = 4, i = 0; scalar at
   non-BN sites, one per column at fused-BN sites), and no draw from either stream. STUDY section 6 step 4
   names only b and i: q_g's uint8 k is the finding interfaces.md section 10 anticipated. A refusal
   child shows F111-FB's file load refused, with no variable changed, under a RoleSpec that declares
   empty sets, and accepted under its own.
DS-HGQ, FP32 and W8A8 are stubs in M5, so their RoleSpecs come with their builders; G10 reruns then.

    <venv>/bin/python -I -B gates/g10_seedflow.py --expect-sha12 SHA12 --work DIR --archive SMOKE_ARCHIVE \\
        [--data-manifest PATH] [--json PATH] [--children-out DIR]
"""

import argparse
import json
import random
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

GATE = 'G10'
KERNEL_ROLES = sorted(['embed/kernel', 'mha/query/kernel', 'mha/key/kernel', 'mha/value/kernel',
                       'mha/attention_output/kernel', 'ffn1/kernel', 'ffn2/kernel', 'head1/kernel', 'head2/kernel',
                       'head3/kernel', 'logits/kernel'])
REFUSALS = {
    'a_no_expected_sha': ('InitTensorRefused', 'no expected manifest sha256'),
    'b_wrong_expected_sha': ('InitTensorRefused', 'differs from the expected'),
    'c_file_byte_flipped': ('InitTensorRefused', 'differs from the manifest'),
    'd_seed_differs': ('InitTensorRefused', 'seed:'),
    'e_platform_differs': ('InitTensorRefused', 'device.jax_platform'),
    'f_role_sha_differs': ('InitTensorRefused', 'differs from its manifest entry'),
    'g_data_differs': ('InitTensorRefused', 'data.n_train'),
    'h_datasets_draw_differs': ('InitTensorRefused', 'datasets.index_map_sha256_at_construction'),
    'i_draw_code_differs': ('InitTensorRefused', 'draw_code.cbl/rng.py'),
    'j_stack_differs': ('InitTensorRefused', 'stack.keras'),
    'k_role_set_differs': ('InitTensorRefused', 'unmatched'),
    'l_datasets_before_seed': ('FlowError', 'seed_streams must be the last step'),
    'm_seed_before_data': ('FlowError', 'load_data must be the last step'),
    'n_bad_device': ('FlowError', 'dataset_device'),
    'o_reseed_before_load': ('FlowError', 'load_initial_tensors must be the last step'),
    'p_config_reseed_differs': ('FlowError', 'prefit_reseed'),
}


def event(rec, step):
    return next(e for e in rec['flow']['events'] if e['step'] == step)


def check_draw(g, name, rec, seed, inputs):
    ok = g.check(rec is not None and rec.get('exit') == 0 and 'error' not in rec,
                 f'{name}: draw child exit 0 ({rec.get("error") if rec else "no record"})')
    if not ok:
        return False
    fl = rec['flow']
    g.check(fl['steps'] == list(rng.SeedFlow.ORDER[:4]), f'{name}: steps in the section-6 order {fl["steps"]}')
    ev = event(rec, 'seed_streams')
    g.check(ev['random_equals_seed'] and ev['numpy_equals_seed'],
            f'{name}: after step 2 random equals Random({seed}) and NumPy equals RandomState({seed})')
    ds = event(rec, 'build_datasets')
    g.check(all(ds['checks'].values()), f'{name}: the flow\'s own construction checks {ds["checks"]}')
    d = fl['data']
    exp = rng.reconstruct_datasets(seed, d['n_train'], d['n_val'])
    g.check(ds['train_init_shift'] == exp['train']['init_shift'] and ds['val_init_shift'] == exp['val']['init_shift']
            and rec['index_map_at_construction'] == exp['train']['index_map']
            and ds['py_state_sha256'] == exp['state_sha256'] and ds['n_batches'] == exp['train']['n_batches'],
            f'{name}: this gate\'s reconstruction from Random({seed}) equals the drawn shifts '
            f'({ds["train_init_shift"]}, {ds["val_init_shift"]}), training map {rec["index_map_at_construction"]} '
            f'and the stream state after the Datasets')
    g.check(d['train_archive_sha256'] == inputs['archive_sha256'] and d['data_manifest_sha256'] == inputs['data_manifest_sha256'],
            f'{name}: step 1 read the smoke archive {d["train_archive_sha256"][:12]} under data manifest '
            f'{d["data_manifest_sha256"][:12]}')
    g.check(rec['n_roles'] == rng.CH8_N_VARIABLES and len(set(rec['role_order'])) == rng.CH8_N_VARIABLES,
            f'{name}: {rec["n_roles"]} roles, unique')
    tops = {r.split('/')[0] for r in rec['role_order']}
    g.check(tops <= {n for _, n in rng.CH8_TOP}, f'{name}: role first segments are CH-8 layer names {sorted(tops)}')
    g.check(rec['stochastic_quantizers'] == [], f'{name}: no quantizer has a seed generator (stochastic rounding)')
    idn = rec['identity']
    g.check(idn['seed'] == seed and idn['device']['jax_platform'] == 'cpu' and idn['device']['dataset_device'] == 'cpu:0',
            f'{name}: manifest identity seed {idn["seed"]}, platform {idn["device"]["jax_platform"]}, '
            f'Dataset device {idn["device"]["dataset_device"]}')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--expect-sha12', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--data-manifest', default=str(FORK / 'manifests' / 'data-prep-150c.json'))
    ap.add_argument('--json', default=None)
    ap.add_argument('--children-out', default=None)
    a = ap.parse_args()
    g = gatelib.Gate(GATE)
    rec = {'header': g.header(a.expect_sha12)}
    gatelib.check_venv_of_record(g)
    work = Path(a.work)
    g.check(not work.exists(), f'work directory {work} does not exist yet')
    work.mkdir(parents=True)
    inputs = rng_child.smoke_inputs(a.data_manifest, a.archive)
    rec['inputs'] = inputs
    g.info('inputs', inputs)
    got = rng.sha256_file(a.archive)
    g.check(got == inputs['archive_sha256'] and got not in (inputs['record_archive_sha256'], inputs['heldout_archive_sha256']),
            f'the archive is the smoke archive {got[:12]} (not the archive of record, not the held-out archive)')
    head = rec['header']['HEAD']
    co = a.children_out

    def child(name, cmd, seed, extra=()):
        code, r, argv = rng_child.run_child(name, cmd, device='cpu', seed=seed, inputs=inputs, work=work / name,
                                            children_out=co, extra=extra)
        g.info(f'{name} argv', ' '.join(argv[3:]))
        return code, r

    # 1-2: draws
    draws = {}
    for name, seed in (('draw-s1', 1), ('draw-s1-again', 1), ('draw-s2', 2)):
        _, r = child(name, 'draw', seed, ['--out-dir', work / name / 'init', '--place', 'PCC gate G10', '--code-sha', head])
        draws[name] = r
        check_draw(g, name, r, seed, inputs)
    if not all(r and r.get('exit') == 0 for r in draws.values()):
        g.check(False, 'all three draws completed; later checks need them')
        g.check_clean('at end')
        return g.finish(rec, a.json)
    m1 = json.loads(Path(draws['draw-s1']['manifest']).read_bytes())
    m1b = json.loads(Path(draws['draw-s1-again']['manifest']).read_bytes())
    m2 = json.loads(Path(draws['draw-s2']['manifest']).read_bytes())
    g.check(m1['file']['sha256'] == m1b['file']['sha256'] and m1['roles'] == m1b['roles']
            and m1['probe_forward']['sha256'] == m1b['probe_forward']['sha256'],
            f'seed 1 drawn twice: same file bytes {m1["file"]["sha256"][:12]}, same 244 tensors, same forward probe')
    diff = sorted(r for r in m1['roles'] if m1['roles'][r]['sha256'] != m2['roles'][r]['sha256'])
    g.check(diff == KERNEL_ROLES, f'seed 2 against seed 1: exactly the 11 Glorot kernels differ {diff}')
    g.check(m1['identity']['datasets']['index_map_sha256_at_construction'] != m2['identity']['datasets']['index_map_sha256_at_construction']
            or m1['identity']['datasets']['train_init_shift'] != m2['identity']['datasets']['train_init_shift'],
            'seed 2 against seed 1: the Datasets\' construction draws differ')
    rec['draws'] = {k: {'manifest_sha256': v['manifest_sha256'], 'file_sha256': v['file']['sha256'], 'file_bytes': v['file']['bytes'],
                        'probe_forward_sha256': v['probe_forward']['sha256']} for k, v in draws.items()}
    g.info('draws (engineering, not quotable)', rec['draws'])

    # 3: load by role, own draw, reseed
    _, r = child('run-s1-load', 'run', 1, ['--manifest', draws['draw-s1']['manifest'], '--manifest-sha', draws['draw-s1']['manifest_sha256'],
                                          '--work', work / 'run-s1-load', '--no-fit', '--check-own-draw'])
    if g.check(r is not None and r.get('exit') == 0 and 'error' not in r, f'run-s1-load: child exit 0 ({r.get("error") if r else "no record"})'):
        ld = r['load']
        g.check(r['flow']['steps'] == list(rng.SeedFlow.ORDER), f'run-s1-load: all six steps in order {r["flow"]["steps"]}')
        own = ld['own_draw_before_load']
        g.check(own['equal_roles'] == rng.CH8_N_VARIABLES and own['differing_roles'] == [],
                f'run-s1-load: before loading, the run\'s own draw equals the file in {own["equal_roles"]} roles')
        g.check(ld['n_loaded'] == rng.CH8_N_VARIABLES and ld['n_roles_file'] == ld['n_roles_arm'] == rng.CH8_N_VARIABLES,
                f'run-s1-load: {ld["n_loaded"]} variables loaded by role and equal to the file after assignment')
        g.check(ld['unmatched'] == [] and ld['unused'] == [], 'run-s1-load: on CH-8 the unmatched and unused sets are empty')
        g.check(ld['manifest_sha256'] == draws['draw-s1']['manifest_sha256'] and ld['file_sha256'] == m1['file']['sha256'],
                'run-s1-load: loaded the manifest and file of draw-s1')
        g.check(ld['random_unchanged_by_load'] and ld['numpy_unchanged_by_load'], 'run-s1-load: loading drew from neither stream')
        g.check(r['probe_forward']['sha256'] == m1['probe_forward']['sha256'], 'run-s1-load: the forward probe equals the drawing process\'s')
        g.check(event(r, 'build_model')['py_state_sha256'] == m1['random_state_sha256_after_build'],
                'run-s1-load: the stream after the model build equals the draw\'s')
        u5 = rng.prefit_reseed(1)
        ev = event(r, 'reseed_before_fit')
        g.check(r['reseed'] == u5 and ev['prefit_reseed'] == u5 and ev['py_state_sha256'] == rng.py_state_sha256(random.Random(u5).getstate()),
                f'run-s1-load: pre-fit reseed is prefit_reseed(1) = {u5} and random is in Random({u5}) state')
        g.check(r['flow']['index_map_at_reseed'] == draws['draw-s1']['index_map_at_construction'],
                'run-s1-load: the training map at the reseed is the construction map of the draw')
        times = [e['t'] for e in r['flow']['events']]
        g.check(times == sorted(times), 'run-s1-load: step timestamps increase in the section-6 order')

    # 4: refusals
    _, r = child('refusals-s1', 'refusals', 1, ['--manifest', draws['draw-s1']['manifest'], '--manifest-sha',
                                                draws['draw-s1']['manifest_sha256'], '--work', work / 'refusals-s1'])
    if g.check(r is not None and r.get('exit') == 0 and 'error' not in r, f'refusals-s1: child exit 0 ({r.get("error") if r else "no record"})'):
        cases = r['cases']
        g.check(set(cases) == set(REFUSALS), f'refusals-s1: {len(cases)} cases run')
        for name, (typ, needle) in REFUSALS.items():
            c = cases.get(name, {})
            ok = c.get('refused') and c.get('type') == typ and needle in (c.get('message') or '')
            if 'variables_unchanged' in c:
                ok = ok and c['variables_unchanged']
            g.check(ok, f'refusals-s1 {name}: refused ({c.get("type")}: {(c.get("message") or "")[:150]})'
                    + ('; no variable changed' if c.get('variables_unchanged') else ''))
        g.check('reseed_before_fit' not in cases['p_config_reseed_differs'].get('steps_after', ['reseed_before_fit']),
                'refusals-s1: the refused reseed left the flow before step 5')
        pl = r['positive_load']
        g.check(pl['n_loaded'] == rng.CH8_N_VARIABLES and pl['unmatched'] == [] and pl['unused'] == [],
                'refusals-s1: the untampered manifest with its sha256 loads (positive control)')
        g.check(r['reseed'] == rng.prefit_reseed(1), 'refusals-s1: the reseed with the matching config value succeeds')

    # 5: F000 through the runner (B5)
    bundle = runnergate.sha256_file(runnergate.BUNDLE)
    code, r, argv, _ = runner_child.run_child(
        'step-F000-s1', 'step', device='cpu', seed=1, archive=inputs['archive'], archive_sha=inputs['archive_sha256'],
        data_manifest_sha=inputs['data_manifest_sha256'], work=work / 'step-F000-s1', children_out=co,
        extra=['--arm', 'F000', '--manifest', draws['draw-s1']['manifest'], '--manifest-sha', draws['draw-s1']['manifest_sha256'],
               '--code-sha', head, '--source-bundle', bundle, '--out-dir', work / 'step-F000-s1' / 'out'])
    g.info('step-F000-s1 argv', ' '.join(argv[3:]))
    if g.check(code == 0 and r is not None and 'error' not in r, f'step-F000-s1: child exit 0 ({(r or {}).get("error")})'):
        ld = r['load']
        g.check(ld['arm_id'] == 'F000' and ld['manifest_sha256'] == draws['draw-s1']['manifest_sha256'],
                'F000: loaded the manifest of draw-s1 under its own RoleSpec')
        g.check(ld['n_loaded'] == rng.CH8_N_VARIABLES and ld['n_roles_arm'] == ld['n_roles_file'] == rng.CH8_N_VARIABLES,
                f'F000: {ld["n_loaded"]} tensors loaded by role')
        g.check(ld['unmatched'] == [] and ld['unused'] == [], 'F000: the unmatched and unused sets are empty, as its RoleSpec declares')
        own = ld['own_draw_before_load']
        g.check(own['equal_roles'] == rng.CH8_N_VARIABLES and own['differing_roles'] == [],
                'F000: its own draw equals CH-8\'s file in all 244 roles before the load')
        g.check(ld['random_unchanged_by_load'] and ld['numpy_unchanged_by_load'], 'F000: loading drew from neither stream')
    # 6: binary arms through the runner (B6)
    import numpy as np
    MHA = ['mha/query', 'mha/key', 'mha/value', 'mha/attention_output']
    TAIL = ['ffn1', 'ffn2', 'head1', 'head2', 'head3']
    BN_COLS = {'embed': 24, 'ffn1': 32, 'head1': 32, 'head2': 32, 'head3': 32}
    arms = (('F100', None, 'H1', MHA), ('F111', 'FB', 'H1', MHA + TAIL), ('F111', 'BF', 'H2', MHA + TAIL),
            ('GR', 'BF', 'H1', MHA + TAIL), ('BE', 'FB', 'H2', ['embed'] + MHA + TAIL))
    for arm, form, hl, sites in arms:
        label = arm + (f'-{form}' if form else '')
        name = f'step-{label}-s1'
        code, r, argv, _ = runner_child.run_child(
            name, 'step', device='cpu', seed=1, archive=inputs['archive'], archive_sha=inputs['archive_sha256'],
            data_manifest_sha=inputs['data_manifest_sha256'], work=work / name, children_out=co,
            extra=['--arm', arm, '--headline', hl, '--manifest', draws['draw-s1']['manifest'],
                   '--manifest-sha', draws['draw-s1']['manifest_sha256'], '--code-sha', head, '--source-bundle', bundle,
                   '--out-dir', work / name / 'out'] + (['--form', form] if form else []))
        g.info(f'{name} argv', ' '.join(argv[3:]))
        if not g.check(code == 0 and r is not None and 'error' not in r, f'{name}: child exit 0 ({(r or {}).get("error")})'):
            continue
        ld = r['load']
        want_unm = sorted(f'{s}/qg/FixedPointQuantizerKBI[0]/{v}' for s in sites for v in ('k', 'b', 'i'))
        want_unu = sorted(f'{s}/kq/FixedPointQuantizerKBI[0]/{v}' for s in sites for v in ('k', 'b', 'i'))
        n = rng.CH8_N_VARIABLES - 3 * len(sites)
        g.check(ld['arm_id'] == arm and ld['manifest_sha256'] == draws['draw-s1']['manifest_sha256']
                and ld['n_loaded'] == n and ld['n_roles_arm'] == ld['n_roles_file'] == rng.CH8_N_VARIABLES,
                f'{label}: loaded draw-s1 under its own RoleSpec, {ld["n_loaded"]} tensors by role ({len(sites)} binary sites)')
        g.check(ld['unmatched'] == want_unm and ld['unused'] == want_unu,
                f'{label}: unmatched = q_g\'s k, b, i and unused = kq\'s k, b, i at exactly the {len(sites)} binary sites')
        own = ld['own_draw_before_load']
        g.check(own['equal_roles'] == n and own['differing_roles'] == [],
                f'{label}: its own draw equals CH-8\'s file in all {n} loaded roles before the load')
        const = {}
        for s in sites:
            shp = (BN_COLS[s],) if s in BN_COLS else ()
            for v, val, dt in (('k', 1, np.uint8), ('b', 4.0, np.float32), ('i', 0.0, np.float32)):
                const[f'{s}/qg/FixedPointQuantizerKBI[0]/{v}'] = rng.tensor_sha256(np.full(shp, val, dt))
        g.check(ld['unmatched_values_sha256'] == const,
                f'{label}: the unmatched tensors hold their constant initializers (k 1 uint8, b 4, i 0; scalar at non-BN '
                f'sites, per column at fused-BN sites)')
        g.check(ld['random_unchanged_by_load'] and ld['numpy_unchanged_by_load'], f'{label}: loading drew from neither stream')
    g.info('finding (B6)', 'q_g carries the KBI quantizer\'s uint8 k beside b and i (fixed_point_quantizer.py:218-224), '
           'so the unmatched set is k, b and i; STUDY section 6 step 4 names b and i (wording, not a design change)')
    code, r, argv, _ = runner_child.run_child(
        'refuse-F111-FB', 'refuse', device='cpu', seed=1, archive=inputs['archive'], archive_sha=inputs['archive_sha256'],
        data_manifest_sha=inputs['data_manifest_sha256'], work=work / 'refuse-F111-FB', children_out=co,
        extra=['--arm', 'F111', '--form', 'FB', '--headline', 'H1', '--manifest', draws['draw-s1']['manifest'],
               '--manifest-sha', draws['draw-s1']['manifest_sha256']])
    g.info('refuse-F111-FB argv', ' '.join(argv[3:]))
    if g.check(code == 0 and r is not None and 'error' not in r, f'refuse-F111-FB: child exit 0 ({(r or {}).get("error")})'):
        w = r['wrong_spec']
        g.check(w['refused'] and 'unmatched' in w.get('message', '') and w['variables_unchanged'],
                f'refuse-F111-FB: a RoleSpec declaring empty unmatched and unused sets is refused before any assignment '
                f'({w.get("message", "")[:120]})')
        gs = r['good_spec']
        g.check(gs['n_loaded'] == rng.CH8_N_VARIABLES - 27 and len(gs['unmatched']) == 27 and len(gs['unused']) == 27,
                'refuse-F111-FB: the same flow then loads under F111\'s own RoleSpec (217 tensors, 27 unmatched, 27 unused)')

    rec['u5'] = {s: rng.prefit_reseed(s) for s in range(1, 12)}
    g.check(len(set(rec['u5'].values())) == 11, 'U5 values for seeds 1-11 are distinct')
    g.info('U5 prefit_reseed(s), s = 1..11', rec['u5'])
    g.check_clean('at end')
    return g.finish(rec, a.json)


if __name__ == '__main__':
    sys.exit(main())
