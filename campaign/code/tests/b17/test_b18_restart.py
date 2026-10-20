import copy
from pathlib import Path
import sys
import json
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nrp_test_helpers import checkpoint
from nrp.common import write_once, Refused, ROOT, PRODUCTS
from nrp.restart import launch_record, recorded, prepare_run, bind_plan
from nrp.jobs import build_job
from nrp.run_pack import command
from nrp.accounting import publish, hours, prior_seconds


def local_arm(plan, tmp_path):
    arm = plan['arms'][0]
    arm['out'] = str(tmp_path / 'run')
    return arm


def test_pretrain_kill_preserves_every_byte_and_original_interval(tmp_path, plan):
    arm = local_arm(plan, tmp_path)
    plan['resume_every'] = 7
    root = Path(arm['out'])
    write_once(root / 'launch.json', launch_record(arm, plan))
    (root / 'some-partial-log').write_bytes(b'evidence\x00')
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    plan['resume_every'] = 1
    cmd = command(arm, plan, 'config.json')
    assert '--resume' not in cmd and cmd[cmd.index('--resume-every') + 1] == '7'
    assert not root.exists()
    saved, = tmp_path.glob('run.pretrain-*Z-*')
    assert {p.relative_to(saved): p.read_bytes() for p in saved.rglob('*') if p.is_file()} == before


def test_resume_uses_run_interval_not_new_spec(tmp_path, plan):
    arm = local_arm(plan, tmp_path)
    plan['resume_every'] = 9
    write_once(Path(arm['out']) / 'launch.json', launch_record(arm, plan))
    checkpoint(arm['out'], arm, plan)
    plan['resume_every'] = 1
    cmd = command(arm, plan, 'config.json')
    assert '--resume' in cmd and cmd[cmd.index('--resume-every') + 1] == '9'


def test_legacy_checkpoint_interval_and_product_are_read(tmp_path, plan):
    arm = local_arm(plan, tmp_path)
    checkpoint(arm['out'], arm, plan, every=6)
    assert recorded(arm, plan)['resume_every'] == 6
    assert '--resume' in command(arm, plan, 'c.json')


@pytest.mark.parametrize('what', ['product', 'hash', 'interval', 'terminal', 'missing_metadata'])
def test_invalid_existing_run_never_archived_or_overwritten(tmp_path, plan, what):
    arm = local_arm(plan, tmp_path)
    root = Path(arm['out'])
    write_once(root / 'launch.json', launch_record(arm, plan))
    cp = checkpoint(root, arm, plan)
    if what == 'product':
        plan['gpu_product'] = PRODUCTS[1]
    elif what == 'hash':
        cp.write_text('{}')
    elif what == 'interval':
        d = json.loads((root / 'launch.json').read_text()); d['resume_every'] = 2
        (root / 'launch.json').write_text(json.dumps(d))
    elif what == 'terminal':
        (root / 'run.json').write_text('{}')
    else:
        (root / 'launch.json').unlink(); (root / 'resume/latest.json').unlink()
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):
        command(arm, plan, 'c.json')
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert not list(tmp_path.glob('run.pretrain-*'))


def test_generated_retry_affinity_comes_from_run_records(tmp_path, plan, spec):
    spec['run_state_root'] = str(tmp_path)
    for arm in plan['arms']:
        dest = tmp_path / Path(arm['out']).relative_to(ROOT)
        write_once(dest / 'launch.json', {**launch_record(arm, plan), 'resume_every': 5})
        checkpoint(dest, arm, plan, every=5)
    plan['job_attempt'] = 2
    plan['resume_every'] = 100
    job = build_job(spec, plan)
    pod = job['spec']['template']['spec']
    terms = pod['affinity']['nodeAffinity']['requiredDuringSchedulingIgnoredDuringExecution']['nodeSelectorTerms']
    assert terms[0]['matchExpressions'][0] == {'key': 'nvidia.com/gpu.product', 'operator': 'In',
                                             'values': [plan['gpu_product']]}
    env = {r['name']: r.get('value') for r in pod['containers'][0]['env']}
    wire = json.loads(env['CBL_PLAN'])
    assert [a['resume_every'] for a in wire['arms']] == [5, 5]
    plan['gpu_product'] = PRODUCTS[1]
    with pytest.raises(Refused, match='selected GPU'):
        build_job(spec, plan)


def test_retry_without_run_inventory_refuses(tmp_path, plan):
    plan['job_attempt'] = 2
    with pytest.raises(Refused, match='run_state_root'):
        bind_plan(plan, run_state_root=tmp_path / 'absent')


def test_allocation_cumulative_across_resume_and_live_share_changes(tmp_path):
    p = tmp_path / 'live.json'
    publish(p, seconds=1800, prior=3600, live=2, now=100)
    assert hours(p, now=160) == pytest.approx(5430 / 3600)
    publish(p, seconds=1830, prior=3600, live=1, now=160)
    assert hours(p, now=220) == pytest.approx(5490 / 3600)
    write_once(tmp_path / 'allocations/one.json', {'seconds': 3600, 'prior_seconds': 0})
    write_once(tmp_path / 'allocations/two.json', {'seconds': 1890, 'prior_seconds': 3600})
    assert prior_seconds(tmp_path) == 5490


def test_allocation_archive_baseline_survives_later_resumes(tmp_path):
    # The first pre-train invocation was archived; its known allocation remains in the new baseline.
    write_once(tmp_path / 'allocations/fresh.json', {'seconds': 120, 'prior_seconds': 30})
    assert prior_seconds(tmp_path) == 150
    write_once(tmp_path / 'allocations/resumed.json', {'seconds': 90, 'prior_seconds': 150})
    assert prior_seconds(tmp_path) == 240
