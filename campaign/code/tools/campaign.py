#!/usr/bin/env python3
"""Campaign harness adapter: submit/tick through unchanged lab harness and launcher.

Use this entry point for every campaign submit and tick. Direct lab-harness submit
does not implement the campaign's G80 and split reserve gates. No admission policy.
is claimed to be Kubernetes RBAC. Importing this module makes no cluster call.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import sys

FORK = Path(__file__).resolve().parents[1]
if str(FORK) not in sys.path:
    sys.path.insert(0, str(FORK))
from nrp.common import CAMPAIGN, ROOT, need, read, load_tool, digest, write_once, positive, Refused
from tools.configmaps import live
from nrp.jobs import build_job, build_brief
from nrp.spec import wire_plan
from tools.retries import classify, policy as retry_policy, lineage_stopped, notify_relaunch


def budget(state, attempts, candidate, bound):
    """Reserve is a distinct 50 GPU-h bucket; training is at most 350, total at most 400.

    Unknown lifecycle costs retain full harness bounds; ambiguous submissions count.
    Failed/retried Jobs are retained and every new Job reserves its entire hard bound.
    """
    spend = {'training': 0.0, 'reserve': 0.0}
    unfinished = 0
    for name, entry in state['resources']['jobs'].items():
        if name == candidate['job']:
            continue
        need(name in attempts, 'resource Job missing campaign attempt record')
        need(entry.get('tracked'), 'untracked campaign GPU Job: reconcile before any launch')
        if not entry.get('final'):
            unfinished += 1
        if entry.get('never_started'):
            value = 0.0
        else:
            value = (entry.get('measured') if entry.get('final') else
                     max(entry.get('bound') or 0, entry.get('reserved') or 0,
                         entry.get('estimated') or 0, entry.get('measured') or 0, entry.get('lifetime_upper') or 0))
        need(type(value) in (int, float) and value >= 0, 'missing accounting evidence')
        spec = attempts[name]['spec']
        failed = any(o.get('status') == 'failed' for k, o in state.get('observations', {}).items()
                     if o.get('job', k.split('#')[0]) == name)
        kind = 'reserve' if spec['mode'] != 'pack' or spec['job_attempt'] > 1 or failed else 'training'
        spend[kind] += value
    spec = candidate['spec']
    kind = 'reserve' if spec['mode'] != 'pack' or spec['job_attempt'] > 1 else 'training'
    positive(bound, 'candidate GPU bound')
    need(unfinished < 4, 'four unfinished GPU Jobs; defer wave')
    spend[kind] += bound
    need(spend['training'] <= 350 and spend['reserve'] <= 50 and sum(spend.values()) <= 400,
         'campaign budget/reserve exhausted: ' + json.dumps(spend))
    return spend


def admit(record, attempt, policy):
    need(record['source'] == 'git-bundle' and record['pvc_root'] == ROOT, 'only campaign PVC bundle mode')
    plan = attempt['plan']
    need(record['bundle']['commit'] == plan['code_sha'] and record['bundle']['sha256'] == plan['source_bundle'],
         'attempt/bundle identity mismatch')
    need(record['job']['metadata']['name'] == attempt['job'], 'attempt/Job mismatch')
    embedded = next(e['value'] for e in record['job']['spec']['template']['spec']['containers'][0]['env']
                    if e['name'] == 'CBL_PLAN')
    need(json.loads(embedded) == wire_plan(plan), 'attempt execution plan changed')
    need(record['job'] == build_job(attempt['spec'], plan) and record['brief'] == build_brief(attempt['spec'], plan),
         'Job/brief differs from generated campaign contract')
    need(plan['campaign'] == CAMPAIGN, 'wrong campaign')
    if plan['mode'] == 'draw':
        need(all(a['seed'] in policy['draw_seeds'] for a in plan['arms']), 'unapproved draw seed')
    else:
        for arm in plan['arms']:
            need(arm['config_path'] in policy['admitted_configs'], 'run identity outside authority')
    if plan['mode'] == 'pack':
        cert = policy['production_certification']
        need(isinstance(cert, dict) and cert.get('status') == 'passed', 'B19 production certification pending')
        need(cert['code_sha'] == plan['code_sha'] and cert['gpu_product'] == plan['gpu_product'] and
             cert['k'] == plan['k'], 'production certification does not cover code/product/K')
        need(all(cert.get(k) is True for k in ('utilization_pass', 'device_memory_pass', 'host_projection_pass',
                                              'fingerprint_pass', 'cost_pass')), 'production gate not passed')
        positive(cert['required_screen_gpu_h'], 'required screen projection')
        need(cert['required_screen_runs'] == 51 and cert['required_screen_gpu_h'] <= 350,
             'required 51-run screen exceeds cap; Kai must decide scope/cap')
        need(cert.get('evidence'), 'certification evidence missing')


def install_guards(h, cdir, lab):
    original_submit = h.submit
    original_decide = h.decide
    original_attempt = h._attempt
    def allocated_attempt(pod, now):
        attempt = original_attempt(pod, now)
        status = pod.get('status', {})
        start = status.get('startTime')
        attempt['allocation_started'] = bool(start)
        spec = pod.get('spec', {})
        expected = {c['name'] for c in spec.get('containers', []) + spec.get('initContainers', [])}
        states = status.get('containerStatuses', []) + status.get('initContainerStatuses', [])
        attempt['exit_codes'] = [c.get('state', {}).get('terminated', {}).get('exitCode') for c in states]
        attempt['reasons'] = [c.get('state', {}).get(kind, {}).get('reason')
                              for c in states for kind in ('waiting', 'terminated')]
        attempt['reasons'] += [c.get('reason') for c in status.get('conditions', [])]
        terminated = {c['name']: c.get('state', {}).get('terminated', {}).get('finishedAt') for c in states}
        if start:
            final = bool(expected) and expected <= set(terminated) and all(terminated[n] for n in expected)
            end = max(h.parse_iso(terminated[n]) for n in expected) if final else now
            seconds = max(0.0, (end - h.parse_iso(start)).total_seconds())
            attempt['gpu_h'] = round(h._gpus(spec.get('containers', [])) * seconds / 3600, 6)
            attempt['gpu_h_kind'] = 'measured' if final else 'estimated'
        else:
            attempt['gpu_h'] = None
            attempt['gpu_h_kind'] = 'unknown'
        return attempt
    h._attempt = allocated_attempt  # include init allocation; never patch the lab-root file
    h.classify = classify

    def decide(brief, state, observations):
        retry_policy(brief)
        observations = copy.deepcopy(observations)
        for key, obs in observations.items():
            previous = state.get('observations', {}).get(key, {})
            if obs.get('status') == previous.get('status') == 'failed':
                # TTL/partial refreshes must not erase a known code failure or infra cause.
                pods = {a['pod']: a for a in copy.deepcopy(previous.get('attempts', []))}
                for current in obs.get('attempts', []):
                    saved = pods.setdefault(current['pod'], {})
                    codes = set(saved.get('exit_codes', []) + current.get('exit_codes', []) +
                                [saved.get('exit_code'), current.get('exit_code')])
                    reasons = set(saved.get('reasons', []) + current.get('reasons', []) +
                                  [saved.get('reason'), current.get('reason')])
                    saved.update({k: v for k, v in current.items() if v is not None})
                    saved['exit_codes'] = sorted(c for c in codes if c is not None)
                    saved['reasons'] = sorted(r for r in reasons if r)
                obs['attempts'] = list(pods.values())
                obs['job_reasons'] = sorted(r for r in {previous.get('job_reason'), obs.get('job_reason'),
                    *previous.get('job_reasons', [])} if r)
                obs['job_reason'] = obs.get('job_reason') or previous.get('job_reason')
        # Only this in-memory comparison is unbounded. On-disk legacy caps are zero,
        # so using the lab harness directly cannot silently apply a finite retry policy.
        actions, updated = original_decide(dict(brief, infra_relaunches_per_key=float('inf')),
                                           state, observations)
        for name, entry in updated['resources']['jobs'].items():
            job_obs = [o for o in observations.values() if o['job'] == name]
            pods = [a for o in job_obs for a in o.get('attempts', [])]
            entry['never_started'] = (bool(job_obs) and all(o.get('job_terminal') for o in job_obs)
                                      and bool(pods) and all(a.get('allocation_started') is False for a in pods))
        stopped = updated.setdefault('b17_stopped_lineages', {})
        for key, obs in observations.items():
            if obs.get('status') == 'failed':
                cls, why = classify(obs)
                # Retain raw failure evidence, including after handled-event refreshes.
                updated['observations'][key].update(obs, **{'class': cls})
                if cls != 'infra':
                    pack = (obs.get('arm') or key).split()[0]
                    stopped.setdefault(pack, {'job': obs['job'], 'why': why})
        # A later infrastructure event cannot revive a lineage stopped by code/integrity.
        def blocked(key):
            obs = updated['observations'].get(key, {})
            return (obs.get('arm') or key).split()[0] in stopped
        actions = [a for a in actions if not (a.get('task') == 'relaunch' and blocked(a['key']))]
        for task in updated['queue']:
            if task['task'] == 'relaunch' and blocked(task['key']) and task['status'] == 'pending':
                task['status'] = 'failed'
                task.setdefault('detail', {})['stop_reason'] = 'lineage stopped; new commit and handoff required'
        return actions, updated
    h.decide = decide
    # The lab stage_order mechanism reserves for later stages only, whereas this campaign's
    # reserve funds engineering before, during and after training. Enforce both buckets here.
    h.stage_cap = lambda brief, stage: brief['gpu_h_max']

    def checked_submit(campaign_dir, handoff_dir, run_handoff=None, stage=None):
        root = Path(campaign_dir)
        with (root / '.b17-wave.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            brief = h.load_brief(root)
            h.require_approval(root)
            retry_policy(brief)
            need(brief['campaign'] == CAMPAIGN, 'wrong campaign authority')
            need(brief['gpu_h_max'] == 400 and brief['reserve_gpu_h'] == {'reserve': 50} and
                 brief['max_concurrent_gpu_jobs'] == 4, 'adapter bound to STUDY 10 cap/reserve/concurrency')
            rh = load_tool(lab / 'tools/run_handoff.py', 'cbl_supported_launcher')
            record = rh.validate_dir(handoff_dir)
            name = record['job']['metadata']['name']
            attempts = {p.stem: read(p) for p in (root / 'attempts').glob('*.json')}
            need(name in attempts, 'no generated attempt record')
            candidate = attempts[name]
            policy_path = root / 'ADMISSION.json'
            need(brief['frozen_files'].get('ADMISSION.json') == digest(policy_path.read_bytes()),
                 'admission policy not bound to authority')
            admit(record, candidate, read(policy_path))
            # Refresh lifecycle before the cap/count check; pending Jobs count as unfinished.
            h.watch_once(root)
            state = h.load_state(root)
            bound, _ = h.gpu_bound(record['job'], 'hard')
            lineage_stopped(state, candidate['spec']['pack'])
            budget(state, attempts, candidate, bound)
            result = live(brief['kubectl'])  # immediately before EVERY wave/retry; no automatic deletion
            h.record(root, {'type': 'G80', 'job': name, **result})
            result = original_submit(root, handoff_dir, run_handoff, stage)
            if result.get('status') in ('submitted', 'already-submitted'):
                notify_relaunch(h, root, brief, candidate)
            return result

    def guarded_submit(campaign_dir, handoff_dir, run_handoff=None, stage=None):
        try:
            return checked_submit(campaign_dir, handoff_dir, run_handoff, stage)
        except (Refused, h.GateError) as e:
            if str(e).startswith(('four unfinished', 'campaign budget/reserve exhausted')):
                return {'status': 'deferred', 'why': str(e)}
            # Admission/identity/G80 gate failures stop automatic work for this lineage.
            # Read only the local generated identity; never repair or overwrite the evidence.
            root = Path(campaign_dir)
            job_file = Path(handoff_dir) / 'job.json'
            if job_file.is_file():
                name = read(job_file).get('metadata', {}).get('name')
                candidates = {p.stem: read(p) for p in (root / 'attempts').glob('*.json')}
                if name in candidates:
                    pack = candidates[name]['spec']['pack']
                    with h.locked_state(root) as state:
                        state.setdefault('b17_stopped_lineages', {}).setdefault(
                            pack, {'job': name, 'why': str(e)})
                        for task in state['queue']:
                            parent = candidates.get(task['key'].split('#')[0], {})
                            if (task['task'] == 'relaunch' and task['status'] in ('pending', 'claimed')
                                    and parent.get('spec', {}).get('pack') == pack):
                                task['status'] = 'failed'
                                task.setdefault('detail', {})['stop_reason'] = str(e)
                    h.record(root, {'type': 'b17_lineage_stop', 'pack': pack, 'job': name, 'why': str(e)})
            raise h.GateError(str(e)) from e
    h.submit = guarded_submit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab-root', required=True)
    parser.add_argument('command', choices=('submit', 'tick', 'validate'))
    parser.add_argument('campaign_dir')
    parser.add_argument('handoff', nargs='?')
    parser.add_argument('--holder', default='cbl1016-cron')
    a = parser.parse_args(argv)
    lab = Path(a.lab_root).resolve()
    h = load_tool(lab / 'tools/harness.py', 'cbl_harness')
    install_guards(h, a.campaign_dir, lab)
    try:
        if a.command == 'submit':
            need(a.handoff, 'handoff required')
            result = h.submit(a.campaign_dir, a.handoff)
        elif a.command == 'tick':
            result = h.tick(a.campaign_dir, a.holder)
        else:
            result = h.validate_brief(a.campaign_dir)
            need(not result, str(result))
            result = 'VALID (no approval written)'
        print(json.dumps(result))
    except (ValueError, h.GateError) as e:
        print('REFUSED:', e, file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
