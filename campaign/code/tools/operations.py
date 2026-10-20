"""M21 harness command contracts. Preparation is offline; no command here submits."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from nrp.common import read, need, write_once, load_tool, token
from nrp.jobs import prepare
from tools.retries import classify, policy, lineage_stopped


def attempt(cdir, job):
    token(job)
    rec = read(Path(cdir) / 'attempts' / (job + '.json'))
    need(rec['job'] == job, 'attempt record identity mismatch')
    return rec


def relaunch(cdir, job, harness_attempt, cls):
    need(cls == 'relaunch', 'only infrastructure relaunches; code fixes need new commit, PREFLIGHT and handoff')
    rec = attempt(cdir, job)
    state = read(Path(cdir) / 'STATE.json')
    obs = [o for key, o in state['observations'].items() if o.get('job', key.split('#')[0]) == job]
    need(obs and all(o['status'] == 'failed' and classify(o)[0] == 'infra' for o in obs),
         'relaunch needs the harness infrastructure classification')
    need(not state['stopped'], 'campaign stopped')
    spec = copy.deepcopy(rec['spec'])
    spec['job_attempt'] += 1
    lineage_stopped(state, spec['pack'])
    lab = Path(cdir).resolve().parents[1]
    h = load_tool(lab / 'tools/harness.py', 'cbl_harness_policy')
    policy(h.load_brief(cdir))  # no count ceiling; admission rechecks budgets and G80
    try:
        handoff = prepare(spec, bundle=rec['bundle'], bundle_path=rec['bundle_path'], commit=rec['commit'],
                          data_info=rec['data_info'], out=cdir, launcher=rec['launcher'])
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as e:
        with h.locked_state(cdir) as current:
            current.setdefault('b17_stopped_lineages', {}).setdefault(
                spec['pack'], {'job': job, 'why': str(e)})
        h.record(cdir, {'type': 'b17_lineage_stop', 'pack': spec['pack'], 'job': job, 'why': str(e)})
        raise
    # Same source bundle and scientific configs, new Job identity; M14 owns rewind/resume.
    write_once(Path(cdir) / 'relaunches' / (job + '.json'), {'parent_job': job, 'handoff': str(handoff),
               'harness_attempt': harness_attempt, 'scientific_identity': 'unchanged', 'class': cls})
    return handoff


def evaluate(cdir, job):
    rec = attempt(cdir, job)
    state = read(Path(cdir) / 'STATE.json')
    obs = [o for k, o in state['observations'].items() if o.get('job', k.split('#')[0]) == job]
    if not obs or any(o.get('status') not in ('succeeded', 'failed') for o in obs):
        return 'PENDING'
    if any(o['status'] != 'succeeded' for o in obs):
        return 'INVALID Job did not succeed; no tagging score inferred'
    return write_once(Path(cdir) / 'evidence/evaluations' / (job + '.json'), {
        'metric': 'engineering/job_succeeded', 'value': 1, 'split': 'none; lifecycle only', 'n': 1,
        'status': 'exploratory', 'job': job, 'stage': rec['plan']['stage'], 'code_sha': rec['commit'],
        'source': 'harness STATE observations; not a smoke gate or a scientific evaluation',
        'runs': [a['name'] for a in rec['plan']['arms']]})


def workspace(cdir, job):
    rec = attempt(cdir, job)
    root = Path(cdir) / 'code'
    head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True)
    status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True, text=True, check=True)
    need(head.stdout.strip() == rec['commit'] and not status.stdout, 'workspace must be clean at failed run commit; no reset performed')
    return 'WORKSPACE verified; automatic code repair disabled'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'relaunch', 'evaluate', 'workspace', 'repair-check'))
    p.add_argument('--campaign-dir', required=True)
    p.add_argument('--job'); p.add_argument('--attempt', type=int); p.add_argument('--cls')
    p.add_argument('--spec'); p.add_argument('--bundle'); p.add_argument('--bundle-path')
    p.add_argument('--commit'); p.add_argument('--data-info'); p.add_argument('--launcher')
    a = p.parse_args(argv)
    try:
        if a.command == 'prepare':
            print(prepare(read(a.spec), bundle=a.bundle, bundle_path=a.bundle_path, commit=a.commit,
                          data_info=a.data_info, out=a.campaign_dir, launcher=a.launcher))
        elif a.command == 'relaunch':
            print(relaunch(a.campaign_dir, a.job, a.attempt, a.cls))
        elif a.command == 'evaluate':
            print(evaluate(a.campaign_dir, a.job))
        elif a.command == 'workspace':
            print(workspace(a.campaign_dir, a.job))
        else:
            need(False, 'automatic repair refused: preserve evidence; new commit, re-verification and new handoff required (STUDY 11)')
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as e:
        print('REFUSED:', e, file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
