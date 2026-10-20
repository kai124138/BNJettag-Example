import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nrp.common import PRODUCTS, Refused, encoded, init_root, write_once
from nrp.spec import resolve, check_execution, wire_plan, expand_plan
from nrp.jobs import build_job, build_brief
from nrp import run_pack as R
from lab_root import lab_root


def test_launch_copy_changes_only_handoff_fields(plan, generated):
    for arm in plan['arms']:
        old = json.loads(generated[arm['config_path']])
        new = copy.deepcopy(arm['config'])
        new['code_sha'] = None
        new['init_tensors']['manifest_sha256'] = None
        assert new == old
        assert arm['out'] == '/data/cbl1016/runs/smoke/' + arm['name']


@pytest.mark.parametrize('field,value', [('headline', None), ('claim_domain', None), ('param_count', None),
                                        ('epochs', None), ('launch_blocks', ['blocked'])])
def test_every_launch_refusal(spec, generated, field, value):
    if field == 'headline':
        spec['arms'][0]['config_path'] = 'configs/smoke/F100-s1.json'
    path = spec['arms'][0]['config_path']
    cfg = json.loads(generated[path]); cfg[field] = value
    generated[path] = encoded(cfg)
    with pytest.raises(ValueError):
        resolve(spec, generated, 'b' * 40, 'c' * 64)


def test_current_generated_configs_are_refused(spec, generated):
    path = spec['arms'][0]['config_path']
    generated[path] = Path(path).read_bytes()
    with pytest.raises(ValueError, match='refused'):
        resolve(spec, generated, 'b' * 40, 'c' * 64)


@pytest.mark.parametrize('product', PRODUCTS)
def test_pvc_only_job_for_each_product(spec, plan, product):
    plan['gpu_product'] = product
    job = build_job(spec, plan)
    pod = job['spec']['template']['spec']
    assert set(v['name'] for v in pod['volumes']) == {'data', 'scratch'}
    assert pod['volumes'][0]['persistentVolumeClaim']['claimName'] == 'kai-data'
    c = pod['containers'][0]
    assert c['resources']['requests'] == c['resources']['limits']
    assert c['resources']['requests']['nvidia.com/gpu'] == '1'
    assert all('configMap' not in v and 'secret' not in v and 'hostPath' not in v for v in pod['volumes'])
    assert '/cmcode' not in [v['mountPath'] for v in c['volumeMounts']]
    assert job['spec']['activeDeadlineSeconds'] > 0 and job['spec']['ttlSecondsAfterFinished'] > 0
    assert job['spec']['podReplacementPolicy'] == 'Failed'
    assert pod['automountServiceAccountToken'] is False
    assert c['command'][-1] == '/cmcode/nrp/run_pack.py'


@pytest.mark.parametrize('mode', ['draw', 'bench', 'det', 'pack'])
def test_all_job_modes(spec, generated, mode):
    spec['mode'] = mode
    spec['stage'] = 'S1' if mode == 'pack' else mode
    spec['archive_kind'] = 'record'
    if mode == 'draw':
        spec['seeds'] = [1, 2, 3]
    elif mode == 'det':
        spec['mem_fraction'] = 0.25
        spec['arms'] = [{'config_path': p, 'init_manifest_sha256': 'a' * 64}
                        for p in sorted(generated) if p.startswith('configs/det/')]
    else:
        spec['arms'] = [{'config_path': f'configs/{spec["stage"]}/{n}-s1.json', 'init_manifest_sha256': 'a' * 64}
                        for n in ('CH-8', 'F000')]
    # v0.4: CH-8/F000 modes work without selecting a headline.
    for arm in spec.get('arms', []) if mode != 'draw' else []:
        path = arm['config_path']
        cfg = json.loads(generated[path])
        cfg['headline'] = None
        generated[path] = encoded(cfg)
    plan = resolve(spec, generated, 'b' * 40, 'c' * 64)
    assert plan['k'] == (1 if mode == 'draw' else (3 if mode == 'det' else 2))
    assert build_brief(spec, plan)['scientific_gate']['status'] == 'pending'
    assert all(x.startswith('/data/cbl1016/') for x in build_brief(spec, plan)['outputs'])


def test_unbuilt_control_and_wrong_product_refused(spec, generated):
    spec['arms'][0]['config_path'] = 'configs/smoke/FP32-s1.json'
    with pytest.raises(ValueError):
        resolve(spec, generated, 'b' * 40, 'c' * 64)
    spec['gpu_product'] = 'NVIDIA-A100'
    with pytest.raises(ValueError, match='unsupported'):
        resolve(spec, generated, 'b' * 40, 'c' * 64)


def test_pack_revalidates_generated_bytes_and_path(tmp_path, plan, generated):
    for rel, data in generated.items():
        p = tmp_path / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    check_execution(plan, tmp_path)
    assert expand_plan(wire_plan(plan), tmp_path) == plan
    plan['arms'][0]['config']['epochs'] = 2
    with pytest.raises(ValueError, match='launch copy'):
        check_execution(plan, tmp_path)


def test_resume_cli_and_identity(tmp_path, plan):
    arm = plan['arms'][0]
    arm['out'] = str(tmp_path / 'run')
    first = R.command(arm, plan, tmp_path / 'config.json')
    assert '--resume' not in first
    from nrp.restart import launch_record
    from nrp_test_helpers import checkpoint
    write_once(Path(arm['out']) / 'launch.json', launch_record(arm, plan))
    checkpoint(Path(arm['out']), arm, plan)
    second = R.command(arm, plan, tmp_path / 'config.json')
    assert second == first + ['--resume']
    assert second[second.index('--pack-k') + 1] == '2'
    assert second[second.index('--resume-every') + 1] == '1'
    plan['job_attempt'] += 1
    assert R.command(arm, plan, tmp_path / 'config.json') == second


def test_failed_terminal_sibling_never_skipped(tmp_path, plan):
    arm = plan['arms'][0]; arm['out'] = str(tmp_path)
    (tmp_path / 'run.json').write_text('{"exit":3,"end":"integrity_stop"}')
    with pytest.raises(Refused, match='terminal'):
        R.completed(arm, plan)


def test_atomic_record_replay_mismatch(tmp_path):
    write_once(tmp_path / 'x.json', {'v': 1})
    write_once(tmp_path / 'x.json', {'v': 1})
    with pytest.raises(ValueError):
        write_once(tmp_path / 'x.json', {'v': 2})


@pytest.mark.parametrize('exit_code,expected', [(0, 0), (1, 1), (3, 3), (2, 2)])
def test_real_process_supervision_offline(tmp_path, monkeypatch, plan, exit_code, expected):
    # Real separate OS processes, synthetic runner contract; no ML/GPU/backend is loaded.
    monkeypatch.setattr(R, 'check_execution', lambda *a: None)
    monkeypatch.setattr(R, 'completed', lambda a, p: (Path(a['out']) / 'run.json').exists())
    worker = tmp_path / 'worker.py'
    worker.write_text('''import pathlib,sys,time,json,os
p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True)
(p/'pid').write_text(str(os.getpid()))
time.sleep(0.05)
(p/'epochs.jsonl').write_text('{"epoch":0}\\n')
(p/'run.json').write_text('{}')
sys.exit(int(sys.argv[2]))
''')
    for i, a in enumerate(plan['arms']):
        a['out'] = str(tmp_path / ('run' + str(i)))
    monkeypatch.setattr(R, 'command', lambda a, p, c, root: [sys.executable, str(worker), a['out'], str(exit_code)])
    sampler = lambda: {'product': plan['gpu_product'], 'process_mib': {}, 'utilization_pct': 40,
                       'used_mib': 100, 'total_mib': 1000}
    rc = R.execute(plan, output_root=str(tmp_path), sample=sampler, poll_s=0.01)
    assert rc == expected
    records = [json.loads(p.read_text()) for p in tmp_path.glob('packs/**/pack.json')]
    assert len(records) == 1 and records[0]['exit'] == expected
    rows = [json.loads(p.read_text()) for p in tmp_path.glob('run*/telemetry/*.json')]
    assert len(rows) == 2
    assert len({(Path(a['out']) / 'pid').read_text() for a in plan['arms']}) == 2
    assert sum(x['allocated_gpu_seconds'] for x in rows) == pytest.approx(records[0]['wall_seconds'])


def test_imports_never_load_backend():
    script = 'import sys; import nrp.jobs, nrp.spec, nrp.run_pack, tools.campaign, tools.operations, tools.bench_rates; assert not any(x in sys.modules for x in ("keras","jax","tensorflow"))'
    r = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_draw_partial_is_never_overwritten(tmp_path, plan):
    arm = {'seed': 1, 'out': str(tmp_path)}
    p = tmp_path / 'init-s1-gpu.npz'; p.write_bytes(b'partial synthetic file')
    with pytest.raises(Refused, match='partial'):
        R.drawn(arm, plan, Path.cwd())
    assert p.read_bytes() == b'partial synthetic file'


def test_draw_verified_reuse(tmp_path, plan):
    from nrp.common import digest
    from cbl import rng
    import numpy as np
    arm = {'seed': 1, 'out': str(tmp_path)}
    tensor = tmp_path / 'init-s1-gpu.npz'
    values = np.array([1], dtype='float32')
    rng.write_tensor_file(tensor, {'synthetic': values})
    dman = json.loads(Path('manifests/data-prep-150c.json').read_text())
    rec = {'kind': rng.INIT_KIND, 'schema': rng.INIT_SCHEMA, 'arm_drawn': 'CH-8',
        'roles': {'synthetic': {'dtype': 'float32', 'shape': [1], 'sha256': rng.tensor_sha256(values)}},
        'identity': {'seed': 1, 'device': {'jax_platform': 'gpu', 'n_devices': 1,
        'device_kind': [plan['gpu_product'].replace('-', ' ')]}, 'draw_code': rng.draw_code_sha256(),
        'stack': rng.stack_versions(), 'matmul_precision': 'float32',
        'data': {'train_archive_sha256': dman['smoke']['archive']['sha256'],
                 'data_manifest_sha256': digest(Path('manifests/data-prep-150c.json').read_bytes())}},
        'file': {'name': tensor.name, 'sha256': digest(tensor.read_bytes())}}
    # Device labels use hyphens for spaces; internal RTX model hyphens are absent in this product.
    write_once(tmp_path / 'init-s1-gpu.manifest.json', rec)
    assert R.drawn(arm, plan, Path.cwd())
    tensor.write_bytes(b'tampered')
    with pytest.raises(Refused, match='hash'):
        R.drawn(arm, plan, Path.cwd())


def test_completion_without_record_is_integrity_failure(tmp_path, monkeypatch, plan):
    monkeypatch.setattr(R, 'check_execution', lambda *a: None)
    monkeypatch.setattr(R, 'completed', lambda *a: False)
    monkeypatch.setattr(R, 'command', lambda *a: [sys.executable, '-c', 'pass'])
    for i, a in enumerate(plan['arms']):
        a['out'] = str(tmp_path / str(i))
    assert R.execute(plan, output_root=str(tmp_path), sample=lambda: {'product': plan['gpu_product']}, poll_s=.01) == 76


def test_stale_staged_bundle_is_refused_before_artifacts(tmp_path, spec, monkeypatch):
    from nrp.jobs import prepare
    lab = lab_root('run_handoff.py')
    camp = lab / 'campaigns/2026-10-16-chang-bitlinear'
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    with pytest.raises(Refused, match='lacks B17'):
        prepare(spec, bundle=camp/'preflight/nrp/bundle/cbl1016-main-f8ca91d7dce4.bundle',
                bundle_path='/data/cbl1016/code/cbl1016-main-f8ca91d7dce4.bundle',
                commit='f8ca91d7dce442b581d02678aebe6dff64f1a236', data_info=camp/'preflight/nrp/data_info.json',
                out=tmp_path/'out', launcher=lab/'tools/run_handoff.py')
    assert not (tmp_path/'out').exists()
