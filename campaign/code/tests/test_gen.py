"""Unit tests of M17 (`gen/`), the config generator. Mac CPU, no backend; synthetic decision inputs
only (a constructed S2 record and parameter-count record, never real results).

Run:  python -I -B -m pytest -q tests/test_gen.py   (from the fork root, with -p no:cacheprovider)
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))

from gen import arms as A  # noqa: E402
from gen import configs as C  # noqa: E402
from gen import inputs as I  # noqa: E402
from gen import protocol as P  # noqa: E402
from gen import release as R  # noqa: E402

INPUTS = json.loads((FORK / 'gen' / 'inputs.json').read_text())


def write_inputs(tmp_path, **changes):
    rec = copy.deepcopy(INPUTS)
    rec.update(changes)
    p = tmp_path / 'inputs.json'
    p.write_text(json.dumps(rec, indent=1))
    return p


@pytest.fixture(scope='module')
def base():
    inp = I.load()
    return inp, C.build_all(inp), C.generate(inp)


def s2_file(tmp_path, form='FB', best='F101', status=None):
    rec = {'stage': 'S2', 'split': 'validation', 'form_choice': {'form': form}, 'best': {'best': best}}
    if status:
        rec = {'stage': 'S2', 'status': status}
    p = tmp_path / f's2-{form}-{best}-{status}.json'
    p.write_text(json.dumps(rec))
    return p


def complete_inputs(tmp_path, base_cfgs, **extra):
    keys = sorted({C.model_key(c) for c in base_cfgs if C.model_key(c)} | {
        'lab|-|F101|FB|Q', 'lab|-|F100|-|Q', 'lab|GR|F111|FB|Q', 'lab|BE|F111|FB|B', 'lab|-|F101|BF|Q',
        'lab|GR|F111|BF|Q', 'lab|BE|F111|BF|B'})
    pc = tmp_path / 'param-counts.json'
    pc.write_text(json.dumps({'kind': 'cbl1016-param-counts', 'counts': {k: 1000 + i for i, k in enumerate(keys)}}))
    fields = dict(headline='H1', claim_domain={'base': 'b', 'band_rule': 'default', 'c3r': False, 'c3r_level': None},
                  s4_option='C', s2_record=str(s2_file(tmp_path)), init_tensors={'root': '/data/cbl1016/init', 'platform': 'gpu'},
                  smoke_init_tensors={'root': '/data/cbl1016/init-smoke', 'platform': 'gpu'}, bench_epochs=50,
                  param_counts=str(pc))
    fields.update(extra)
    return I.load(write_inputs(tmp_path, **fields))


# ------------------------------------------------------------------ release constants (STUDY 7.1)

def test_release_constants_equal_study_7_1():
    r = R.release_constants()
    assert r['source']['sha256'] == 'bfe36347c12614c6d0aa0f0f1100c5ea4e985179d86628e1e9351a4d643d169a'
    assert r['batch_size'] == 2790 and r['val_batch_size'] == 2790
    assert r['steps_per_execution'] == 4 and r['epochs'] == 7000
    assert r['learning_rate'] == 3e-3
    assert r['lr_schedule'] == {'function': 'cosine_decay_restarts_schedule', 'first_decay_steps': 500,
                                't_mul': 1.0, 'm_mul': 1.0, 'alpha': 1e-6, 'alpha_steps': 10}
    assert r['beta_schedule'] == [[0, 2e-8, 'linear'], [2000, 3e-7, 'log'], [7000, 3e-6, 'constant']]
    assert r['pareto_filter'] == [['val_accuracy', '>', 0.5], ['ebops', '<', 5e5]]
    assert r['stopif'] == [['ebops', '<', 1e4]]
    assert r['callbacks'] == ['FreeEBOPs', 'LearningRateScheduler', 'BetaScheduler', 'PBar', 'ParetoFront', 'StopIf']
    assert r['dataset_shuffle'] is True and r['get_model_literals'] == [7, 7, 1e-8]
    assert r['loss'] == {'class': 'SparseCategoricalCrossentropy', 'from_logits': True}


def test_release_refuses_a_changed_file(tmp_path):
    fork = tmp_path / 'fork'
    (fork / 'chang').mkdir(parents=True)
    (fork / 'code-base.txt').write_text((FORK / 'code-base.txt').read_text())
    (fork / 'chang' / 'run_train.py').write_bytes((FORK / 'chang' / 'run_train.py').read_bytes() + b'\n')
    with pytest.raises(R.ReleaseError):
        R.release_constants(fork)


def test_no_backend_import():
    code = ('import sys; sys.path.insert(0, %r)\n'
            'from gen import configs, inputs\n'
            'configs.generate(inputs.load())\n'
            'bad = sorted(m for m in sys.modules if m.split(".")[0] in ("keras", "jax", "jaxlib", "hgq", "tensorflow", "torch"))\n'
            'print(bad); sys.exit(1 if bad else 0)\n' % str(FORK))
    r = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ----------------------------------------------------------------------------- arm identities

def expected_screen_labels():
    out = ['CH-8', 'F000', 'F100'] + [f'{c}-{f}' for c in A.FUSED_CODES for f in ('FB', 'BF')]
    return out + ['PID-CH', 'PID-BEST', 'FP32', 'W8A8', 'DS-HGQ', 'GR', 'BE']


def test_counts_and_names(base):
    _, cfgs, _ = base
    by = {}
    for c in cfgs:
        k = f'S4/{c["s4_family"]}' if c['stage'] == 'S4' else c['stage']
        by.setdefault(k, []).append(c)
    assert {k: len(v) for k, v in by.items()} == {'S1': 45, 'S3': 21, 'S4/A': 20, 'S4/B': 32, 'S4/C': 16,
                                                   'smoke': 15, 'bench': 15, 'det': 3}
    screen = [c for c in cfgs if c['stage'] in ('S1', 'S3')]
    assert [f'{c["label"]}-s{c["seed"]}' for c in screen] == [f'{l}-s{s}' for l in expected_screen_labels() for s in (1, 2, 3)]
    assert len({C.config_path(c) for c in cfgs}) == len(cfgs)
    assert {c['arm_id'] for c in by['S4/C']} == {'CH-8-C', 'BEST-C'}
    assert [c['seed'] for c in by['S4/A'] if c['arm_id'] == 'CH-8-C'] == [4, 5, 6, 7, 8]
    assert [c['seed'] for c in by['S4/B'] if c['arm_id'] == 'PID-BEST-C'] == list(range(4, 12))
    det = by['det']
    assert [(c['arm_id'], c['seed'], c['replica'], c['epochs']) for c in det] == [('CH-8', 1, 1, 20), ('CH-8', 1, 2, 20), ('F000', 1, None, 20)]
    assert all(c['replica'] is None for c in cfgs if c['stage'] != 'det')
    assert [C.config_id(c) for c in det] == ['CH-8-s1-r1', 'CH-8-s1-r2', 'F000-s1']
    assert all(c['epochs'] == 3 and c['data']['view'] == 'smoke' for c in by['smoke'])
    assert all(c['epochs'] is None and c['data']['view'] == 'train' for c in by['bench'])
    assert len({c['arm_class'] for c in by['smoke']}) == 15


def test_s4_seeds_match_m16():
    from cbl import select as sel
    claim = sel.claim_domain_config({'base': 'a', 'band_rule': None, 'c3r': False, 'c3r_level': None})
    for fam in ('A', 'B', 'C'):
        assert tuple(sel.s4_design(fam, claim).seeds) == A.S4_SEEDS[fam]


def test_tiers_stages_and_item9(tmp_path, base):
    _, cfgs, _ = base
    tiers = {c['label']: (c['stage'], c['tier']) for c in cfgs if c['stage'] in ('S1', 'S3')}
    assert tiers['CH-8'] == ('S1', 'T0') and tiers['F111-BF'] == ('S1', 'T1')
    assert tiers['PID-BEST'] == ('S3', 'T1') and tiers['FP32'] == ('S3', 'T2') and tiers['BE'] == ('S3', 'T6')
    alt = C.build_all(I.load(write_inputs(tmp_path, item9='cap475')))
    moved = {c['label'] for c, d in zip(cfgs, alt) if c['stage'] != d['stage']}
    assert moved == {'FP32', 'W8A8', 'DS-HGQ'}
    for c, d in zip(cfgs, alt):
        diff = {k for k in c if c[k] != d[k]} - {'generator', 'stage', 'admission', 'wandb'}
        assert diff == set(), (c['label'], diff)


def test_fields_from_cbl(base):
    from cbl import rng
    from cbl import wandb_log as wl
    _, cfgs, _ = base
    for c in cfgs:
        assert c['prefit_reseed'] == rng.prefit_reseed(c['seed'])
        if c['wandb']['run_name'] is not None:
            assert c['wandb']['run_name'] == wl.run_name_stem(c['arm_id'], c['form'], c['seed'], c['stage'], c['schedule'], 1,
                                                              c['replica'])
        assert c['data_manifest_sha256'] == '686b15db996010cc5eadb7556397488c92ce3b48c2bccdb2a0d947f1011a7f13'
        assert c['cost_callbacks'] == (c['control'] != 'FP32')
        assert (c['schedule'] == 'pid') == c['arm_id'].startswith('PID-')
        assert ('ramp' in c) == (c['schedule'] == 'ramp') and ('pid' in c) == (c['schedule'] == 'pid')
        assert not any('test' in k.lower() or 'heldout' in k.lower() for k in c)
    # one launchable set is the screen, the engineering runs and one S4 family
    for fam in ('A', 'B', 'C'):
        stems = [c['wandb']['run_name'] for c in cfgs if c['wandb']['run_name'] and c.get('s4_family') in (None, fam)]
        dup = {s for s in stems if stems.count(s) > 1}
        assert dup == set()   # owner decision B12-5: the det pair carries _r1 and _r2
    det = [c['wandb']['run_name'] for c in cfgs if c['stage'] == 'det']
    assert det == ['CH-8_nf_s1_det_ramp_r1_a1', 'CH-8_nf_s1_det_ramp_r2_a1', 'F000_nf_s1_det_ramp_a1']
    assert all(c['launch_blocks'] == [] for c in cfgs)


def test_templates_null_until_s2(base):
    _, cfgs, _ = base
    tmpl = [c for c in cfgs if c['template'] is not None and c['stage'] not in ('smoke', 'bench')]
    assert {c['arm_id'] for c in tmpl} == {'PID-BEST', 'GR', 'BE', 'BEST-C', 'PID-BEST-C'}
    for c in tmpl:
        assert c['config_code'] is None and c['form'] is None and c['form_source'] is None
        assert c['best_source'] is None and c['s2_choice_sha256'] is None and c['wandb']['run_name'] is None
        assert 's2_record' in c['unset_inputs']


# ------------------------------------------------------------------------------ fail closed

def test_every_config_fails_closed_now(base):
    _, cfgs, _ = base
    for c in cfgs:
        assert c['headline'] is None and c['claim_domain'] == INPUTS['claim_domain'] and c['code_sha'] is None
        assert c['init_tensors'] == {'manifest_path': None, 'manifest_sha256': None}
        assert 'claim_domain' not in c['unset_inputs']
        assert ('headline' in c['unset_inputs']) == (not C.headline_independent(c))
        assert c['launch_confirmations'] == {'od13': True, 'od15': True, 'od12': True}
        # B10: G58's record is an input; a config without a G58 count (templates before S2, the controls) stays unset
        assert ('param_counts' in c['unset_inputs']) == (c['param_count'] is None)
        if c['stage'] == 'S4':
            assert c['s4_option'] is None and 's4_option' in c['unset_inputs']
        with pytest.raises(C.UnsetInput):
            C.check_launchable(c, code_sha='a' * 40, init_manifest_sha256='b' * 64)


def test_complete_inputs_launch_only_what_they_should(tmp_path, base):
    inp = complete_inputs(tmp_path, base[1])
    cfgs = C.build_all(inp)
    ok, refused = [], []
    for c in cfgs:
        try:
            C.check_launchable(c, code_sha='a' * 40, init_manifest_sha256='b' * 64)
            ok.append(c)
        except C.UnsetInput:
            refused.append(c)
    refused_ids = sorted({(c['stage'], c.get('s4_family'), c['label'], c['replica']) for c in refused})
    assert refused_ids == sorted({('S4', f, c['label'], None) for c in cfgs if c['stage'] == 'S4' and (f := c['s4_family']) != 'C'})
    assert any(c['stage'] == 'det' and c['replica'] == 2 for c in ok)    # B12-5: replica 2 launches
    for c in ok:
        with pytest.raises(C.UnsetInput):
            C.check_launchable(c)                       # no handoff fields
        assert c['init_tensors']['manifest_path'] == (
            f'/data/cbl1016/init{"-smoke" if c["stage"] == "smoke" else ""}/init-s{c["seed"]}-gpu.manifest.json')
    pid_best = [c for c in cfgs if c['label'] == 'PID-BEST']
    assert all(c['config_code'] == 'F101' and c['form'] == 'FB' and c['best_source'].startswith('s2_choice:')
               and c['arm_class'] == 'pid-binary-FB' for c in pid_best)
    gr = [c for c in cfgs if c['label'] == 'GR']
    assert all(c['config_code'] == 'F111' and c['form'] == 'FB' and c['best_source'] is None for c in gr)


def test_best_f100_has_no_form(tmp_path, base):
    inp = complete_inputs(tmp_path, base[1], s2_record=str(s2_file(tmp_path, form='BF', best='F100')))
    cfgs = C.build_all(inp)
    best = [c for c in cfgs if c['template'] == A.T_BEST and c['stage'] in ('S3', 'S4')]
    assert all(c['config_code'] == 'F100' and c['form'] is None and c['form_source'] is None for c in best)
    assert {c['arm_class'] for c in best} == {'pid-binary-nonbn', 'binary-nonbn'}
    be = [c for c in cfgs if c['label'] == 'BE']
    assert all(c['form'] == 'BF' and c['embedding'] == 'B' for c in be)


def test_protocol_unchanged_by_decisions(tmp_path, base):
    inp0, cfgs0, files0 = base
    inp = complete_inputs(tmp_path, cfgs0)
    files = C.generate(inp)
    for name in ('screen', 'claim-A', 'claim-B', 'claim-C'):
        p = f'configs/protocol/{name}.json'
        assert files[p] == files0[p], name


@pytest.mark.parametrize('change, message', [
    (dict(headline='H3'), 'headline'),
    (dict(s4_option='D'), 's4_option'),
    (dict(item9=None), 'item9'),
    (dict(claim_domain={'base': 'b', 'c3r': False}), 'claim_domain'),
    (dict(bench_epochs=0), 'bench_epochs'),
    (dict(init_tensors={'root': 'relative/dir', 'platform': 'gpu'}), 'init_tensors'),
    (dict(init_tensors={'root': '/data', 'platform': None}), 'init_tensors'),
    (dict(data_manifest={'path': 'manifests/data-prep-150c.json', 'sha256': '0' * 64}), 'differs'),
])
def test_inputs_refused(tmp_path, change, message):
    with pytest.raises(I.InputError, match=message):
        I.load(write_inputs(tmp_path, **change))


def test_inputs_need_every_key(tmp_path):
    rec = copy.deepcopy(INPUTS)
    del rec['headline']
    p = tmp_path / 'inputs.json'
    p.write_text(json.dumps(rec))
    with pytest.raises(I.InputError, match='missing'):
        I.load(p)


def test_incomplete_s2_refused(tmp_path):
    with pytest.raises(I.InputError, match='INCOMPLETE'):
        I.load(write_inputs(tmp_path, s2_record=str(s2_file(tmp_path, status='INCOMPLETE'))))


# ------------------------------------------------------------------------ determinism, diff

def test_generation_is_deterministic(base):
    inp, _, files = base
    assert C.generate(inp) == files
    for p, b in files.items():
        assert b.endswith(b'\n') and b.decode('ascii')


def test_diff_detects_a_hand_edit(base):
    _, _, files = base
    edited = dict(files)
    p = 'configs/S1/F101-BF-s3.json'
    edited[p] = edited[p].replace(b'"epochs": 7000', b'"epochs": 6999')
    edited['configs/S1/extra.json'] = b'{}\n'
    del edited['configs/smoke/CH-8-s1.json']
    d = C.diff_trees(files, edited)
    assert d == {'missing': ['configs/smoke/CH-8-s1.json'], 'extra': ['configs/S1/extra.json'], 'different': [p]}


def test_manifest_lists_every_config(base):
    _, cfgs, files = base
    man = json.loads(files['configs/MANIFEST.json'])
    assert man['n_configs'] == len(cfgs) == 167
    import hashlib
    for e in man['configs']:
        assert hashlib.sha256(files[e['path']]).hexdigest() == e['sha256']
    assert set(files) == {e['path'] for e in man['configs']} | set(man['protocol']) | {'configs/MANIFEST.json'}


# --------------------------------------------------------------------------------- protocol

def test_derived_arms(base):
    _, _, files = base
    s = json.loads(files['configs/protocol/screen.json'])
    by = {a['name']: a for a in s['arms']}
    assert len(s['arms']) == 66
    assert by['CH-8-s1']['changed_factors'] == [] and by['PID-CH-s2']['changed_factors'] == ['schedule.cost_control']
    assert by['FP32-s3']['changed_factors'] == ['quantization', 'schedule.cost_control']
    assert by['FP32-s3']['schedule']['cost_control'] == 'none'
    assert by['PID-BEST-s1']['schedule']['cost_control'] == 'lab BetaPID, target 350000 H (OD-12 reading), init 2e-8, frozen-H sensor'
    assert by['F110-BF-s2']['quantization'].startswith('binary attention+FFN; binarize-then-fold')
    assert s['dataset']['identity_sha256'] == '686b15db996010cc5eadb7556397488c92ce3b48c2bccdb2a0d947f1011a7f13'
    assert [m['n'] for m in s['metrics']] == [260000, 260000, 62000, 260000]
    assert [len(json.loads(files[f'configs/protocol/claim-{f}.json'])['arms']) for f in 'ABC'] == [20, 32, 16]


def test_derived_protocol_tracks_a_config_change(base):
    inp, cfgs, files = base
    cfgs2 = copy.deepcopy(cfgs)
    victim = next(c for c in cfgs2 if c['label'] == 'F011-BF' and c['seed'] == 2)
    victim['epochs'] = 6999
    d = P.derive_all(cfgs2, inp['data_manifest']['json'])
    a0 = json.loads(files['configs/protocol/screen.json'])['arms']
    a1 = d['screen']['arms']
    changed = [(x['name'], y['changed_factors']) for x, y in zip(a0, a1) if x != y]
    assert changed == [('F011-BF-s2', ['quantization', 'schedule.epochs'])]


def test_short_float():
    assert P.short_float(2e-8) == '2e-8' and P.short_float(3e-3) == '3e-3' and P.short_float(0.002) == '2e-3'


# ------------------------------------------------------------- the runner's acceptance (M13)

NOT_BUILT = ('is not built yet', 'schedule pid needs the lab PID', 'admitted in engineering stages only')
# what the runner on main admits after B10: CH-8, F000, F100, F001-F111 in both forms in every stage; GR and BE in
# engineering stages only; PID-CH (S3), PID-CH-C (S4) and CH-8-C (S4, B10) in every stage, PID-BEST engineering only
BINARY_ENGINEERING = ('F100', 'F111-FB', 'F111-BF', 'GR-FB', 'GR-BF', 'BE-FB', 'BE-BF')
PID_ENGINEERING = ('PID-CH', 'PID-BEST-F111-FB', 'PID-BEST-F111-BF')
PASS_LABELS = {'S1': None,                                   # all 45
               'S3': ('PID-CH',), 'S4': ('PID-CH-C', 'CH-8-C'),
               'smoke': ('CH-8', 'F000') + BINARY_ENGINEERING + PID_ENGINEERING,
               'bench': ('CH-8', 'F000') + BINARY_ENGINEERING + PID_ENGINEERING,
               'det': ('CH-8', 'F000')}


def expected_pass(c):
    if c['stage'] not in PASS_LABELS:
        return False
    labels = PASS_LABELS[c['stage']]
    return labels is None or c['label'] in labels


def runner_verdicts(cfgs, **handoff):
    from cbl import runner
    out = {}
    for c in cfgs:
        cc = C.with_handoff(c, **handoff) if handoff else c
        try:
            runner.validate_config(cc, data_manifest_path=FORK / 'manifests' / 'data-prep-150c.json')
            out[C.config_path(c)] = None
        except runner.ConfigRefused as e:
            out[C.config_path(c)] = str(e)
    return out


def test_ramp_and_mode_are_the_runners(base):
    from cbl import runner
    _, cfgs, _ = base
    rel = R.release_constants()
    rr = runner.RAMP_RECORD
    assert rr == {'lambda': rel['beta_schedule'], 'lr': rel['learning_rate'],
                  'lr_first_decay_steps': rel['lr_schedule']['first_decay_steps'], 't_mul': rel['lr_schedule']['t_mul'],
                  'm_mul': rel['lr_schedule']['m_mul'], 'lr_floor': rel['lr_schedule']['alpha'],
                  'alpha_steps': rel['lr_schedule']['alpha_steps'], 'batch': rel['batch_size'],
                  'steps_per_execution': rel['steps_per_execution']}
    assert all(c['ramp'] == rr for c in cfgs if c['schedule'] == 'ramp')
    assert all(c['wandb']['mode'] in runner.WANDB_MODES_ALLOWED for c in cfgs)


def test_runner_accepts_built_arms_only_once_inputs_and_handoff_exist(tmp_path, base):
    _, cfgs, _ = base
    hand = dict(code_sha='a' * 40, init_manifest_sha256='b' * 64)
    now = runner_verdicts(cfgs, **hand)
    assert all(v is not None for v in now.values())          # every config refused now
    for c in cfgs:
        r = now[C.config_path(c)]
        if expected_pass(c):
            assert not any(x in r for x in NOT_BUILT), (C.config_path(c), r)
            if c['label'] in BINARY_ENGINEERING:
                assert 'headline' in r                         # B5-4: a binary arm needs the headline in every stage
    full = C.build_all(complete_inputs(tmp_path, cfgs))
    v = runner_verdicts(full, **hand)
    ok = sorted(p for p, r in v.items() if r is None)
    assert ok == sorted(C.config_path(c) for c in full if expected_pass(c))
    assert len(ok) == 109                                     # 88 at B9, plus CH-8-C's 21 S4 configs (B10)
    assert all(any(x in r for x in NOT_BUILT) for r in v.values() if r is not None)
    from cbl import runner
    with pytest.raises(runner.ConfigRefused):
        runner.validate_config(next(c for c in full if c['arm_id'] == 'CH-8'))   # no handoff: refused


def test_with_handoff_changes_only_the_two_fields(base):
    _, cfgs, _ = base
    c = cfgs[0]
    h = C.with_handoff(c, code_sha='c' * 40, init_manifest_sha256='d' * 64)
    assert h['code_sha'] == 'c' * 40 and h['init_tensors']['manifest_sha256'] == 'd' * 64
    assert c['code_sha'] is None and c['init_tensors']['manifest_sha256'] is None
    assert {k for k in c if c[k] != h[k]} == {'code_sha', 'init_tensors'}
    for bad in (dict(code_sha='x', init_manifest_sha256='d' * 64), dict(code_sha='c' * 40, init_manifest_sha256=None)):
        with pytest.raises(C.UnsetInput):
            C.with_handoff(c, **bad)


@pytest.mark.parametrize('level', ['bonferroni', 'nested'])
def test_no_c3r_refuses_a_level(tmp_path, level):
    cd = dict(INPUTS['claim_domain'], c3r_level=level)
    with pytest.raises(I.InputError, match='c3r_level applies only'):
        I.load(write_inputs(tmp_path, claim_domain=cd))


@pytest.mark.parametrize('decision', ['od13', 'od15', 'od12'])
def test_unconfirmed_decision_blocks_launch(tmp_path, base, decision):
    inp = complete_inputs(tmp_path, base[1], **{decision: None})
    cfg = next(c for c in C.build_all(inp) if c['arm_id'] == 'CH-8')
    # Recompute from fields even if the denormalized unset list was removed.
    cfg['unset_inputs'] = []
    with pytest.raises(C.UnsetInput, match=decision):
        C.check_launchable(cfg, code_sha='a' * 40, init_manifest_sha256='b' * 64)
    with pytest.raises(I.InputError, match=decision):
        I.load(write_inputs(tmp_path, **{decision: False}))


def test_v04_headline_hold_on_launch_path(tmp_path, base):
    from cbl import runner
    cfgs = C.build_all(complete_inputs(tmp_path, base[1], headline=None))
    seen = set()
    for c in cfgs:
        launch = C.with_handoff(c, code_sha='a' * 40, init_manifest_sha256='b' * 64)
        if c['arm_id'] in ('CH-8', 'F000'):
            C.check_launchable(launch)
            runner.validate_config(launch)
            seen.add((c['arm_id'], c['stage']))
        else:
            with pytest.raises(C.UnsetInput, match='headline'):
                C.check_launchable(launch)
            if c['arm_id'] == 'F100' or c['config_code'] == 'F111' or c['schedule'] == 'pid':
                with pytest.raises(runner.ConfigRefused):
                    runner.validate_config(launch)
    assert seen == {(arm, stage) for arm in ('CH-8', 'F000') for stage in ('S1', 'smoke', 'bench', 'det')}
