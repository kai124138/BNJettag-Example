import copy
import json
from pathlib import Path

import pytest
from nrp.common import Refused, write_once, load_tool
from nrp.jobs import build_job, build_brief
from tools.configmaps import check
from tools.campaign import budget, admit
from tools.bench_rates import rates, project
from tools.operations import evaluate
from tools.draft_brief import draft
from tools.retries import POLICY

from lab_root import lab_root


def inventory(names=()):
    return {'kind': 'ConfigMapList', 'items': [{'metadata': {'name': n}} for n in names]}


def jobs(names=(), terminal=False):
    return {'kind': 'JobList', 'items': [{'metadata': {'name': 'kai-x'},
        'status': {'conditions': [{'type': 'Complete', 'status': 'True'}]} if terminal else {},
        'spec': {'template': {'spec': {'containers': [{'name': 'x', 'volumeMounts':
                    [{'name': str(i)} for i, _ in enumerate(names)]}],
            'volumes': [{'name': str(i), 'configMap': {'name': n}} for i, n in enumerate(names)]}}}}]}


def test_bundle_g80_zero_and_pending():
    result = check(inventory(), jobs())
    assert result['passed'] and result['configmap_count'] == result['mounted_count'] == 0
    assert result['unfinished_jobs'] == ['kai-x']


def test_g80_set_identity_not_just_counts():
    with pytest.raises(Refused, match='mismatch'):
        check(inventory(['kai-a']), jobs(['kai-b']), bundle_only=False)


def test_g80_unique_mounts_legacy_and_bundle_refusal():
    assert check(inventory(['kai-a']), jobs(['kai-a', 'kai-a']), bundle_only=False)['mounted_count'] == 1
    with pytest.raises(Refused, match='both counts zero'):
        check(inventory(['kai-a']), jobs(['kai-a']))


def test_g80_finished_and_unused_mounts():
    with pytest.raises(Refused):
        check(inventory(['kai-a']), jobs(['kai-a'], terminal=True))
    j = jobs(['kai-a']); j['items'][0]['spec']['template']['spec']['containers'][0]['volumeMounts'] = []
    with pytest.raises(Refused):
        check(inventory(['kai-a']), j)


def test_g80_incomplete_inventory():
    c = inventory(); c['metadata'] = {'continue': 'page2'}
    with pytest.raises(Refused, match='incomplete'):
        check(c, jobs())


def test_reserve_and_total_budget():
    candidate = {'job': 'new', 'spec': {'mode': 'smoke', 'job_attempt': 1}}
    state = {'resources': {'jobs': {'old': {'tracked': True, 'final': True, 'measured': 49}}}}
    attempts = {'old': {'spec': {'mode': 'bench', 'job_attempt': 1}}}
    assert budget(state, attempts, candidate, 1)['reserve'] == 50
    with pytest.raises(Refused, match='exhausted'):
        budget(state, attempts, candidate, 1.01)
    candidate['spec'] = {'mode': 'pack', 'job_attempt': 1}
    assert budget(state, attempts, candidate, 350) == {'training': 350, 'reserve': 49}
    with pytest.raises(Refused):
        budget(state, attempts, candidate, 351)


def test_pending_and_ambiguous_jobs_count():
    entries = {str(i): {'tracked': True, 'final': False, 'bound': 1, 'submission': 'refused-or-ambiguous'} for i in range(4)}
    attempts = {name: {'spec': {'mode': 'smoke', 'job_attempt': 1}} for name in entries}
    with pytest.raises(Refused, match='four unfinished'):
        budget({'resources': {'jobs': entries}}, attempts,
               {'job': 'new', 'spec': {'mode': 'pack', 'job_attempt': 1}}, 1)


def test_failed_first_attempt_and_retries_debit_reserve():
    state = {'resources': {'jobs': {'x': {'tracked': True, 'final': True, 'measured': 10}}},
             'observations': {'x#0': {'status': 'failed'}}}
    attempts = {'x': {'spec': {'mode': 'pack', 'job_attempt': 1}}}
    result = budget(state, attempts, {'job': 'y', 'spec': {'mode': 'pack', 'job_attempt': 2}}, 1)
    assert result == {'training': 0, 'reserve': 11}


def test_authority_bound_run_identity(spec, plan):
    attempt = {'job': build_job(spec, plan)['metadata']['name'], 'spec': spec, 'plan': plan}
    record = {'source': 'git-bundle', 'pvc_root': '/data/cbl1016',
              'bundle': {'commit': plan['code_sha'], 'sha256': plan['source_bundle']},
              'job': build_job(spec, plan), 'brief': build_brief(spec, plan)}
    policy = {'admitted_configs': [a['config_path'] for a in plan['arms']]}
    admit(record, attempt, policy)
    with pytest.raises(Refused, match='outside authority'):
        admit(record, attempt, {'admitted_configs': []})
    record['job']['spec']['template']['spec']['containers'][0]['command'] = ['unsafe-change']
    with pytest.raises(Refused, match='generated'):
        admit(record, attempt, policy)


def test_draft_validates_with_real_harness(tmp_path):
    LAB = lab_root('harness.py')
    (tmp_path / 'BRIEF.md').write_text(draft(Path.cwd()))
    h = load_tool(LAB / 'tools/harness.py', 'b17_test_harness')
    assert h.validate_brief(tmp_path) == []
    assert h.current_approval(tmp_path) is None
    b = h.load_brief(tmp_path)
    assert b['gpu_h_max'] == 400 and b['reserve_gpu_h'] == {'reserve': 50}
    assert b['max_concurrent_gpu_jobs'] == 4
    assert b['repairs_total_max'] == 0 and b['infra_relaunches_per_key'] == b['infra_relaunches_max'] == 0
    assert b['scientific_settings']['b17_retry_policy'] == POLICY


def test_real_launcher_contract_and_hard_bound(spec, plan):
    LAB = lab_root('run_handoff.py', 'harness.py')
    rh = load_tool(LAB / 'tools/run_handoff.py', 'b17_test_launcher')
    h = load_tool(LAB / 'tools/harness.py', 'b17_test_bound')
    job = build_job(spec, plan)
    assert rh.check_bundle_job(job) == 'data'
    bound, _ = h.gpu_bound(job, 'hard')
    assert bound == pytest.approx((1200 + 60 + 300) / 3600, abs=1e-6)


def test_evaluate_lifecycle_only(tmp_path, spec, plan):
    job = 'kai-cbl1016-test'
    write_once(tmp_path / f'attempts/{job}.json', {'job': job, 'spec': spec, 'plan': plan, 'commit': plan['code_sha']})
    state = {'observations': {job + '#0': {'job': job, 'status': 'pending'}}}
    (tmp_path / 'STATE.json').write_text(json.dumps(state))
    assert evaluate(tmp_path, job) == 'PENDING'
    state['observations'][job + '#0']['status'] = 'succeeded'
    (tmp_path / 'STATE.json').write_text(json.dumps(state))
    result = json.loads(evaluate(tmp_path, job).read_text())
    assert result['metric'] == 'engineering/job_succeeded' and result['status'] == 'exploratory'


def benchmark(tmp_path, long=False):
    wall = 10860 if long else 20
    pack = {'exit': 0, 'skipped_complete': [], 'job': 'j', 'pod': 'p', 'wall_seconds': wall,
            'k': 2, 'gpu_product': 'GPU', 'code_sha': 'b'*40, 'source_bundle': 'c'*64,
            'telemetry': ['a.json', 'b.json']}
    write_once(tmp_path / 'pack.json', pack)
    for name, cls, n in [('a', 'fast', 10), ('b', 'slow', 5)]:
        write_once(tmp_path / (name + '.json'), {'exit': 0, 'resume_requested': False,
            'source_bundle': 'c'*64, 'gpu_product': 'GPU', 'arm_class': cls,
            'allocated_gpu_seconds': wall/2, 'epochs_completed_this_process': n,
            'gpu_peak_mib_sampled': 400, 'host_peak_mib_sampled': 100})
    samples = [{'elapsed_s': t, 'product': 'GPU', 'utilization_pct': 50, 'used_mib': 800, 'total_mib': 1000}
               for t in range(0, wall + 1, 60 if long else 10)]
    (tmp_path / 'gpu.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in samples))
    return {'job': 'j', 'pod': 'p', 'source': 'synthetic lifecycle', 'complete': True, 'gpu_seconds': wall + 10}


def test_all_inclusive_rates_no_ten_percent_factor(tmp_path):
    life = benchmark(tmp_path)
    r = rates(tmp_path, life)
    assert r['classes']['fast']['gpu_seconds_per_run_epoch'] == 1.5
    assert r['classes']['slow']['gpu_seconds_per_run_epoch'] == 3
    assert not r['utilization_pass'] and not r['production_certified']
    p = project([{'epochs': 7000, 'arm_class': 'fast'}]*51, r)
    assert p['training_gpu_h'] == pytest.approx(51*7000*1.5/3600)


def test_rolling_utilization_is_three_hours(tmp_path):
    r = rates(tmp_path, benchmark(tmp_path, long=True))
    assert r['rolling_three_hour_min_utilization_pct'] == 50
    assert r['utilization_pass'] and not r['production_certified']


def test_missing_lifecycle_and_memory_refuse(tmp_path):
    life = benchmark(tmp_path)
    life['complete'] = False
    with pytest.raises(Refused):
        rates(tmp_path, life)
    life['complete'] = True
    row = json.loads((tmp_path / 'a.json').read_text()); row['gpu_peak_mib_sampled'] = None
    (tmp_path / 'a.json').write_text(json.dumps(row))
    with pytest.raises(Refused):
        rates(tmp_path, life)


def test_repair_command_is_fail_closed(tmp_path):
    from tools.operations import main
    assert main(['repair-check', '--campaign-dir', str(tmp_path), '--job', 'kai-x']) == 2


def test_determinism_record_does_not_turn_difference_into_failure(tmp_path):
    from tools.determinism import compare
    dirs = []
    for arm, replica in [('CH-8', 1), ('CH-8', 2), ('F000', 0)]:
        d = tmp_path / f'{arm}-{replica}'; d.mkdir(); dirs.append(d)
        write_once(d / 'run.json', {'exit': 0, 'end': 'completed', 'config': {'stage': 'det', 'seed': 1,
            'epochs': 20, 'arm_id': arm, 'replica': replica or None, 'init_tensors': {'same': True}},
            'code': {'head': 'a'*40}, 'wandb': {'provenance': {'gpu_product': 'same'}}, 'final_variables_sha256': 'b'*64})
        (d / 'epochs.jsonl').write_text(''.join(json.dumps({'epoch': e, 'vars_sha256_after': 'c'*64,
            'vars_sha256_before': 'c'*64, 'run_id': str(d), 'sys/epoch_seconds': replica + .1,
            'H1': 100 if arm == 'CH-8' else 101})+'\n' for e in range(20)))
    result = compare(dirs)
    assert result['comparisons'][0]['bitwise_equal']
    assert not result['f000_bitwise_equal_ch8']
    assert result['comparisons'][1]['differing_epoch_fields'] == ['H1']


@pytest.mark.parametrize('previous', [1, 3, 7, 1000000])
def test_infrastructure_relaunch_preserves_scientific_identity(tmp_path, monkeypatch, spec, plan, previous):
    from types import SimpleNamespace
    from tools import operations as O
    job = build_job(spec, plan)['metadata']['name']
    spec['job_attempt'] = previous
    rec = {'job': job, 'spec': spec, 'plan': plan, 'bundle': 'frozen.bundle', 'bundle_path': '/data/cbl1016/code/x.bundle',
           'commit': plan['code_sha'], 'data_info': 'data_info.json', 'launcher': 'launcher.py'}
    write_once(tmp_path / f'attempts/{job}.json', rec)
    write_once(tmp_path / 'STATE.json', {'stopped': False, 'observations': {job: {'job': job, 'status': 'failed', 'class': 'infra',
        'job_reason': 'NodeLost', 'attempts': []}}})
    monkeypatch.setattr(O, 'load_tool', lambda *a: SimpleNamespace(load_brief=lambda _: retry_brief()))
    got = []
    monkeypatch.setattr(O, 'prepare', lambda new, **kw: got.append((new, kw)) or tmp_path/'new-handoff')
    assert O.relaunch(tmp_path, job, 27, 'relaunch') == tmp_path/'new-handoff'
    new, kwargs = got[0]
    assert new['job_attempt'] == previous + 1 and spec['job_attempt'] == previous
    new['job_attempt'] = previous
    assert new == spec and kwargs['commit'] == rec['commit'] and kwargs['bundle'] == rec['bundle']
    with pytest.raises(Refused, match='only infrastructure'):
        O.relaunch(tmp_path, job, 28, 'deterministic')


@pytest.mark.parametrize('job_attempt', [1, 8, 1000000])
def test_guarded_harness_submission_checks_g80_every_time(tmp_path, monkeypatch, spec, plan, job_attempt):
    LAB = lab_root('harness.py')
    from types import SimpleNamespace
    from nrp.common import digest
    from tools import campaign as C
    h = load_tool(LAB / 'tools/harness.py', 'b17_guard_test')
    spec['run_state_root'] = str(tmp_path)  # complete empty synthetic PVC capture
    spec['job_attempt'] = plan['job_attempt'] = job_attempt
    order = []
    job = build_job(spec, plan)['metadata']['name']
    rec = {'job': job, 'spec': spec, 'plan': plan}
    write_once(tmp_path / f'attempts/{job}.json', rec)
    policy = {'admitted_configs': [a['config_path'] for a in plan['arms']]}
    write_once(tmp_path/'ADMISSION.json', policy)
    brief = {'campaign': 'chang-bitlinear-20261016', 'gpu_h_max': 400, 'reserve_gpu_h': {'reserve': 50},
             'max_concurrent_gpu_jobs': 4, 'kubectl': 'never-called',
             'frozen_files': {'ADMISSION.json': digest((tmp_path/'ADMISSION.json').read_bytes())}}
    record = {'source': 'git-bundle', 'pvc_root': '/data/cbl1016',
              'bundle': {'commit': plan['code_sha'], 'sha256': plan['source_bundle']},
              'job': build_job(spec, plan), 'brief': build_brief(spec, plan)}
    brief.update(retry_brief())
    h.load_brief = lambda _: brief
    h.require_approval = lambda _: order.append('approval')
    h.watch_once = lambda _: order.append('refresh')
    write_once(tmp_path/'STATE.json', copy.deepcopy(h.EMPTY_STATE))
    h.record = lambda *args: order.append('record')
    h.submit = lambda *args: order.append('submit') or {'status': 'submitted'}
    monkeypatch.setattr(C, 'load_tool', lambda *a: SimpleNamespace(validate_dir=lambda _: record))
    monkeypatch.setattr(C, 'live', lambda _: order.append('G80') or {'passed': True})
    monkeypatch.setattr(C, 'notify_relaunch', lambda *args: None)
    handoff = tmp_path/'handoff'
    write_once(handoff/'job.json', record['job'])
    C.install_guards(h, tmp_path, LAB)
    assert h.submit(tmp_path, handoff)['status'] == 'submitted'
    assert h.submit(tmp_path, handoff)['status'] == 'submitted'
    assert order == ['approval', 'refresh', 'G80', 'record', 'submit']*2
    def fail(_):
        raise Refused('G80 mismatch')
    monkeypatch.setattr(C, 'live', fail)
    with pytest.raises(h.GateError, match='G80'):
        h.submit(tmp_path, handoff)
    assert order[-2:] == ['refresh', 'record']
    assert spec['pack'] in h.load_state(tmp_path)['b17_stopped_lineages']
    with pytest.raises(h.GateError, match='lineage stopped'):
        h.submit(tmp_path, handoff)
    assert order.count('submit') == 2



def test_allocation_includes_init_and_queue_retry_is_evidence_bound(tmp_path):
    LAB = lab_root('harness.py')
    from tools.campaign import install_guards
    h = load_tool(LAB/'tools/harness.py', 'b17_lifetime_test')
    install_guards(h, tmp_path, LAB)
    pod = {'metadata': {'name': 'p'}, 'spec': {'initContainers': [{'name': 'init'}],
            'containers': [{'name': 'pack', 'resources': {'limits': {'nvidia.com/gpu': '1'}}}]},
           'status': {'phase': 'Succeeded', 'startTime': '2026-10-19T00:00:00Z',
            'initContainerStatuses': [{'name': 'init', 'state': {'terminated': {'startedAt': '2026-10-19T00:00:00Z',
                                                    'finishedAt': '2026-10-19T00:01:00Z', 'exitCode': 0}}}],
            'containerStatuses': [{'name': 'pack', 'state': {'terminated': {'startedAt': '2026-10-19T00:01:00Z',
                                                    'finishedAt': '2026-10-19T00:03:00Z', 'exitCode': 0}}}]}}
    rec = h._attempt(pod, h.parse_iso('2026-10-19T00:03:00Z'))
    assert rec['gpu_h'] == .05 and rec['gpu_h_kind'] == 'measured'
    queue = {'job_terminal': 'Failed', 'job_reason': 'DeadlineExceeded', 'attempts': [
        {'phase': 'Pending', 'started': None, 'exit_code': None, 'allocation_started': False}]}
    assert h.classify(queue)[0] == 'infra'
    # Missing pod records do not prove that a Job never allocated a GPU.
    queue['attempts'] = []
    assert h.classify(queue)[0] == 'unknown'


@pytest.mark.parametrize('terminal,allocations,expected', [
    (True, [False], 0.0),
    (True, [False, False], 0.0),
    (True, [], 10.0),
    (True, [True], 0.25),
    (False, [False], 10.0),
    (False, [True], 10.0),
    (True, [False, True], 10.0),
    (True, [None], 10.0),
    (True, [False, None], 10.0),
])
def test_allocation_budget_requires_terminal_never_started_evidence(tmp_path, terminal, allocations, expected):
    LAB = lab_root('harness.py')
    from tools.campaign import install_guards
    h = load_tool(LAB/'tools/harness.py', 'b17_allocation_budget')
    (tmp_path/'BRIEF.md').write_text(draft(Path.cwd()))
    brief = h.load_brief(tmp_path)
    install_guards(h, tmp_path, LAB)
    state = copy.deepcopy(h.EMPTY_STATE)
    state['resources']['jobs']['j1'] = {'tracked': True, 'bound': 10, 'reserved': 10}
    observations = {}
    for i in range(max(1, len(allocations))):
        obs = failure()
        obs.update(index=i, job_terminal='Failed' if terminal else None,
                   status='failed' if terminal else 'pending', job_reason='DeadlineExceeded')
        pod = obs['attempts'][0]
        pod.update(pod=f'j1-pod-{i}', phase='Pending', gpu_h=None, gpu_h_kind='unknown')
        if not allocations:
            obs.update(attempts=[], pod_records=0)
        elif allocations[i] is not None:
            pod['allocation_started'] = allocations[i]
            if allocations[i]:
                pod.update(phase='Failed' if terminal else 'Running', gpu_h=0.25,
                           gpu_h_kind='measured' if terminal else 'estimated')
        observations[f'j1#{i}'] = obs
    _, state = h.decide(brief, state, observations)
    assert state['resources']['jobs']['j1']['never_started'] is (expected == 0.0)
    attempts = {'j1': {'spec': {'mode': 'pack', 'job_attempt': 2}}}
    candidate = {'job': 'new', 'spec': {'mode': 'pack', 'job_attempt': 2}}
    assert budget(state, attempts, candidate, 1) == {'training': 0.0, 'reserve': expected + 1}
    if expected == 0.0:
        assert budget(state, attempts, candidate, 50)['reserve'] == 50
    else:
        with pytest.raises(Refused, match='budget/reserve exhausted'):
            budget(state, attempts, candidate, 50)


def test_production_needs_measured_certification(spec, generated):
    from nrp.spec import resolve
    spec.update(mode='pack', stage='S1', archive_kind='record')
    for arm in spec['arms']:
        arm['config_path'] = arm['config_path'].replace('/smoke/', '/S1/')
    plan = resolve(spec, generated, 'b'*40, 'c'*64)
    job = build_job(spec, plan)
    attempt = {'job': job['metadata']['name'], 'spec': spec, 'plan': plan}
    record = {'source': 'git-bundle', 'pvc_root': '/data/cbl1016', 'job': job, 'brief': build_brief(spec, plan),
              'bundle': {'commit': plan['code_sha'], 'sha256': plan['source_bundle']}}
    policy = {'admitted_configs': [a['config_path'] for a in plan['arms']], 'production_certification': None}
    with pytest.raises(Refused, match='certification pending'):
        admit(record, attempt, policy)


def retry_brief():
    return {'scientific_settings': {'b17_retry_policy': dict(POLICY)},
            'infra_relaunches_max': 0, 'infra_relaunches_per_key': 0,
            'repairs_total_max': 0, 'repair_attempts_per_class': 0}


def failure(job='j1', reason='Evicted', code=None, pack='pack'):
    return {'job': job, 'index': None, 'status': 'failed', 'job_terminal': 'Failed',
            'job_reason': None, 'arm': pack, 'role': 'train', 'stage': 'smoke', 'pod_records': 1,
            'attempts': [{'pod': job + '-pod', 'phase': 'Failed', 'reason': reason,
                          'exit_code': code, 'disruption': False, 'gpu_h': 0.01,
                          'gpu_h_kind': 'measured', 'started': None, 'finished': None}]}


@pytest.mark.parametrize('reason', ['Evicted', 'Preempted', 'NodeLost', 'NodeNotReady',
    'ErrImagePull', 'ImagePullBackOff', 'FailedMount', 'FailedAttachVolume',
    'FailedScheduling', 'SchedulingTimeout', 'PendingTimeout'])
def test_infrastructure_failure_classification(reason):
    from tools.retries import classify
    assert classify(failure(reason=reason))[0] == 'infra'


@pytest.mark.parametrize('code', [1, 2, 3, 76, 124, 42])
def test_nonzero_code_overrides_disruption(code):
    from tools.retries import classify
    obs = failure(code=code)
    obs['attempts'][0]['disruption'] = True
    assert classify(obs)[0] == 'deterministic'
    obs['attempts'].append(failure()['attempts'][0])
    assert classify(obs)[0] == 'deterministic'  # earlier evidence also wins


@pytest.mark.parametrize('reason', ['ConfigRefused', 'ResumeRefused', 'HashMismatch',
                                  'ProvenanceMismatch', 'GateFailure', 'IntegrityFailure'])
def test_integrity_reason_overrides_infrastructure(reason):
    from tools.retries import classify
    obs = failure()
    obs['job_reason'] = reason
    assert classify(obs)[0] == 'deterministic'


def test_pending_deadline_and_signal_evidence():
    from tools.retries import classify
    obs = failure(code=None)
    obs['job_reason'] = 'DeadlineExceeded'
    obs['attempts'][0].update(phase='Pending', reason=None, allocation_started=True)
    assert classify(obs)[0] == 'infra'  # scheduled/image-pulling can still be pending
    obs['attempts'][0]['started'] = '2026-10-19T00:00:00Z'
    assert classify(obs)[0] != 'infra'
    assert classify(failure(reason=None, code=143))[0] != 'infra'
    assert classify(failure(reason='Preempted', code=143))[0] == 'infra'


def test_adapter_decisions_unlimited_and_terminal_lineage(tmp_path):
    LAB = lab_root('harness.py')
    from tools.campaign import install_guards
    h = load_tool(LAB/'tools/harness.py', 'b17_retry_decisions')
    (tmp_path/'BRIEF.md').write_text(draft(Path.cwd()))
    brief = h.load_brief(tmp_path)
    install_guards(h, tmp_path, LAB)
    state = copy.deepcopy(h.EMPTY_STATE)
    state['counters']['infra_relaunches']['pack'] = 1000000
    for i in range(12):
        obs = failure(job=f'j{i}')
        actions, state = h.decide(brief, state, {obs['job']: obs})
        assert any(a.get('task') == 'relaunch' for a in actions)
        # Refreshing an already handled event must retain infrastructure evidence.
        _, state = h.decide(brief, state, {obs['job']: obs})
        assert state['observations'][obs['job']]['class'] == 'infra'
        assert state['observations'][obs['job']]['attempts'][0]['reason'] == 'Evicted'
        missing = dict(obs, attempts=[])
        _, state = h.decide(brief, state, {obs['job']: missing})
        assert state['observations'][obs['job']]['class'] == 'infra'
    assert state['counters']['infra_relaunches']['pack'] == 1000012
    bad = failure(job='bad', code=3)
    actions, state = h.decide(brief, state, {'bad': bad})
    assert any(a['kind'] == 'stop_branch' for a in actions)
    assert 'pack' in state['b17_stopped_lineages']
    obs = failure(job='later')
    actions, state = h.decide(brief, state, {'later': obs})
    assert not any(a.get('task') == 'relaunch' for a in actions)
    assert not any(q['task'] == 'relaunch' and q['status'] == 'pending' for q in state['queue'])


def test_notification_rolling_window_and_idempotence(tmp_path):
    LAB = lab_root('harness.py')
    from tools.retries import notify_relaunch
    from datetime import timedelta
    h = load_tool(LAB/'tools/harness.py', 'b17_retry_notify')
    now = h.parse_iso('2026-10-19T12:00:00Z')
    h.now_utc = lambda: now
    brief = dict(retry_brief(), notify_channels=['file'])
    (tmp_path/'STATE.json').write_text(json.dumps(h.EMPTY_STATE))
    def submitted(job, pack='p'):
        notify_relaunch(h, tmp_path, brief, {'job': job, 'spec': {'job_attempt': 100, 'pack': pack}})
    for i in range(6):
        submitted(str(i))
    assert not (tmp_path/h.NOTIFY).exists()
    submitted('other', 'other-pack')
    submitted('six')
    message = (tmp_path/h.NOTIFY).read_text()
    assert '7 infrastructure relaunches' in message and 'continuing' in message
    submitted('six')
    assert (tmp_path/h.NOTIFY).read_text() == message
    now += timedelta(hours=24)
    submitted('next-day')
    assert (tmp_path/h.NOTIFY).read_text() == message
    assert not h.load_state(tmp_path)['stopped']


def test_raw_init_and_waiting_failure_evidence(tmp_path):
    LAB = lab_root('harness.py')
    from tools.campaign import install_guards
    h = load_tool(LAB/'tools/harness.py', 'b17_pod_evidence')
    install_guards(h, tmp_path, LAB)
    pod = {'metadata': {'name': 'pod'}, 'spec': {}, 'status': {'phase': 'Failed', 'reason': 'Evicted',
        'initContainerStatuses': [{'name': 'bundle', 'state': {'terminated': {'exitCode': 1}}}],
        'containerStatuses': [{'name': 'pack', 'state': {'waiting': {'reason': 'ImagePullBackOff'}}}]}}
    a = h._attempt(pod, h.now_utc())
    assert 'ImagePullBackOff' in a['reasons'] and 1 in a['exit_codes']
    assert h.classify({'attempts': [a]})[0] == 'deterministic'
    pod['status']['initContainerStatuses'] = []
    assert h.classify({'attempts': [h._attempt(pod, h.now_utc())]})[0] == 'infra'


def test_preparation_refusal_stops_lineage_and_preserves_evidence(tmp_path, monkeypatch, spec, plan):
    LAB = lab_root('harness.py')
    from tools import operations as O
    h = load_tool(LAB/'tools/harness.py', 'b17_prepare_refusal')
    h.load_brief = lambda _: retry_brief()
    monkeypatch.setattr(O, 'load_tool', lambda *args: h)
    job = 'failed-job'
    rec = {'job': job, 'spec': spec, 'plan': plan, 'bundle': 'bundle', 'bundle_path': 'bundle-path',
           'commit': plan['code_sha'], 'data_info': 'data', 'launcher': 'launcher'}
    write_once(tmp_path/f'attempts/{job}.json', rec)
    state = copy.deepcopy(h.EMPTY_STATE)
    state['observations'][job] = failure(job=job)
    write_once(tmp_path/'STATE.json', state)
    def refuse(*args, **kwargs):
        raise Refused('hash mismatch')
    monkeypatch.setattr(O, 'prepare', refuse)
    with pytest.raises(Refused, match='hash mismatch'):
        O.relaunch(tmp_path, job, 1, 'relaunch')
    assert h.load_state(tmp_path)['observations'][job] == state['observations'][job]
    assert spec['pack'] in h.load_state(tmp_path)['b17_stopped_lineages']
    assert json.loads((tmp_path/f'attempts/{job}.json').read_text()) == rec
    with pytest.raises(Refused, match='lineage stopped'):
        O.relaunch(tmp_path, job, 2, 'relaunch')


def test_preparation_rechecks_raw_integrity_over_stale_class(tmp_path, spec):
    from tools.operations import relaunch
    obs = failure(code=3)
    obs['class'] = 'infra'
    write_once(tmp_path/'attempts/j1.json', {'job': 'j1', 'spec': spec})
    write_once(tmp_path/'STATE.json', {'stopped': False, 'observations': {'j1': obs}})
    with pytest.raises(Refused, match='infrastructure classification'):
        relaunch(tmp_path, 'j1', 1, 'relaunch')


@pytest.mark.parametrize('previous', [8, 1000000])
def test_unlimited_retries_still_need_reserve(previous):
    candidate = {'job': 'new', 'spec': {'mode': 'pack', 'job_attempt': previous}}
    state = {'resources': {'jobs': {'old': {'tracked': True, 'final': True, 'measured': 49.5}}}}
    attempts = {'old': {'spec': {'mode': 'pack', 'job_attempt': 2}}}
    assert budget(state, attempts, candidate, .5)['reserve'] == 50
    with pytest.raises(Refused, match='budget/reserve exhausted'):
        budget(state, attempts, candidate, .51)
