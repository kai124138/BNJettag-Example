"""Campaign-only infrastructure policy; no cluster access or retry count ceiling."""
from nrp.common import need

POLICY = {'infra_relaunches_max': None, 'infra_relaunch_notify_after_24h': 6}
INFRA_REASONS = {
    'Evicted', 'Preempted', 'PreemptionByScheduler', 'NodeLost', 'NodeNotReady',
    'NodeShutdown', 'Shutdown', 'UnexpectedAdmissionError',
    'ErrImagePull', 'ImagePullBackOff', 'InvalidImageName', 'FailedMount',
    'FailedAttachVolume', 'FailedScheduling', 'Unschedulable',
    'SchedulingTimeout', 'PendingTimeout',
}
INTEGRITY_REASONS = {
    'IntegrityFailure', 'ConfigRefused', 'ResumeRefused', 'HashMismatch',
    'ProvenanceMismatch', 'GateFailure',
}
SIGNAL_EXITS = {137, 143, -9, -15}


def policy(brief):
    # The unchanged lab schema accepts protected arbitrary settings only in this map.
    value = brief.get('scientific_settings', {}).get('b17_retry_policy')
    need(isinstance(value, dict) and value.get('infra_relaunches_max', 'missing') is None,
         'BRIEF must authorize unlimited infrastructure relaunches')
    n = value.get('infra_relaunch_notify_after_24h')
    need(type(n) is int and n >= 0, 'BRIEF needs a nonnegative 24h notification threshold')
    need(brief.get('infra_relaunches_max') == brief.get('infra_relaunches_per_key') == 0 and
         brief.get('repairs_total_max') == brief.get('repair_attempts_per_class') == 0,
         'legacy retries and automatic repairs must remain disabled outside the adapter')
    return value


def classify(obs):
    attempts = obs.get('attempts', [])
    reasons = {obs.get('job_reason'), *obs.get('job_reasons', [])}
    for a in attempts:
        reasons.update([a.get('reason'), *a.get('reasons', [])])
        # Bundle init failures and runner refusals beat a simultaneous eviction/preemption.
        codes = [a.get('exit_code'), *a.get('exit_codes', [])]
        if any(c not in (None, 0) and c not in SIGNAL_EXITS for c in codes):
            return 'deterministic', 'nonzero code/integrity exit; preserve evidence, new commit and handoff required'
    if reasons & INTEGRITY_REASONS:
        return 'deterministic', 'code/integrity refusal; preserve evidence, new commit and handoff required'
    if obs.get('job_reason') == 'DeadlineExceeded':
        if (attempts and all(a.get('phase') == 'Pending' and
                            a.get('started') is None and a.get('exit_code') is None for a in attempts)):
            return 'infra', 'queue-only deadline: observed pending pods, no training container start'
        return 'unknown', 'deadline without proof that all pods remained pending'
    if reasons & INFRA_REASONS or any(a.get('disruption') for a in attempts):
        return 'infra', 'observed infrastructure failure: ' + ', '.join(sorted(r for r in reasons if r))
    return 'unknown', 'no established infrastructure cause; preserve evidence for review'


def lineage_stopped(state, pack):
    need(pack not in state.get('b17_stopped_lineages', {}),
         'lineage stopped: new committed fix, checks and handoff required (STUDY 11)')


def notify_relaunch(h, cdir, brief, candidate):
    """Count distinct successful submissions in (now - 24h, now], never queued intents."""
    if candidate['spec']['job_attempt'] <= 1:
        return
    limit = policy(brief)['infra_relaunch_notify_after_24h']
    job, pack = candidate['job'], candidate['spec']['pack']
    now = h.now_utc()
    message = None
    with h.locked_state(cdir) as state:
        history = state.setdefault('b17_infra_relaunches', {})
        if job in history:
            return
        at = state['resources']['jobs'].get(job, {}).get('submitted_at') or h.iso(now)
        history[job] = {'pack': pack, 'at': at}
        count = sum(e['pack'] == pack and 0 <= (now - h.parse_iso(e['at'])).total_seconds() < 86400
                    for e in history.values())
        if count > limit:
            message = (f'Kai: lineage {pack} has {count} infrastructure relaunches in 24 h '
                       f'(threshold > {limit}); continuing subject to 350/50 GPU-h budgets and G80.')
    if message:
        h.notify(cdir, brief, message)
