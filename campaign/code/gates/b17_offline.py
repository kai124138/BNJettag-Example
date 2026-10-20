#!/usr/bin/env python3
"""G76 and G78 transport-only evidence; never submit or contact a cluster.

The existing staged bundle predates B17 and its smoke config has unset inputs. This
diagnostic smoke handoff exercises the real bundle transport validator with those
unchanged bytes. It only checks/refuses that config; it cannot train. It is not a
substitute for the B18-frozen training smoke handoff or for G79/G82/G83.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
from nrp.common import ROOT, PYTHON, digest, read, write_once, atom
from nrp.jobs import build_job
from tools.draft_brief import draft


def record_command(out, label, argv, env):
    p = subprocess.run(argv, cwd=FORK, env=env, capture_output=True, text=True)
    interpreter = subprocess.run([argv[0], '--version'], env=env, capture_output=True, text=True)
    text = f'command: {shlex.join(argv)}\ninterpreter: {interpreter.stdout.strip()}\nexit_code: {p.returncode}\n'
    text += p.stdout + p.stderr
    (out / (label + '.txt')).write_text(text)
    if p.returncode:
        raise RuntimeError(f'{label} exit {p.returncode}; see evidence')
    return p.stdout.strip()


def smoke_probe():
    """Generated diagnostic Job. Intentionally no launch copy or invented U1 hash."""
    spec = {'resources': {'cpu': 2, 'memory_gi': 4, 'deadline_s': 600, 'ttl_s': 86400, 'cache_gi': 2},
            'single_arm_justification': 'G78 offline transport diagnostic only; no training', 'excluded_nodes': []}
    plan = {'pack': 'g78-smoke-transport', 'job_attempt': 1, 'stage': 'smoke', 'k': 1,
            'gpu_product': 'NVIDIA-GeForce-RTX-4090', 'deterministic_ops': False,
            'diagnostic_only': True, 'arms': []}
    job = build_job(spec, plan)
    out = ROOT + '/preflight/G78-smoke-transport'
    script = ('import json,sys\nfrom pathlib import Path\n'
              'sys.path.insert(0,"/cmcode")\nfrom gen.configs import check_launchable\n'
              f'out=Path({out!r});out.mkdir(parents=True,exist_ok=True)\n'
              'cfg=json.loads(Path("/cmcode/configs/smoke/CH-8-s1.json").read_text())\n'
              'try:\n check_launchable(cfg)\n'
              'except Exception as e:\n (out/"refused.txt").write_text(str(e)+"\\n");sys.exit(2)\n'
              'sys.exit("Transport diagnostic never trains, even if the config is launchable")\n')
    container = job['spec']['template']['spec']['containers'][0]
    container['command'] = [PYTHON, '-B', '-c', script]
    container['env'] = [e for e in container['env'] if e['name'] != 'CBL_PLAN']
    brief = {'purpose': 'G78 offline diagnostic smoke transport validation only; no training or scientific admission',
             'changes': 'B17 generated Job shape over the existing staged bundle; unchanged unresolved smoke config. '
                        'The command only checks and refuses the config and cannot invoke training.',
             'approval_ref': 'UNAPPROVED B17 offline transport probe',
             'scientific_gate': {'status': 'pending', 'reference': 'B18 bundle and scientific inputs required before training smoke'},
             'runs': [{'name': 'g78-smoke-contract-probe', 'config_path': 'code/configs/smoke/CH-8-s1.json'}],
             'expected_metrics': [{'name': 'config refusal', 'split': 'none', 'expectation': 'no training, no data read'}],
             'stop_rules': [{'condition': 'config unresolved or diagnostic complete', 'action': 'existing-runner-guard',
                             'approval_ref': 'B17 offline transport test only'}], 'outputs': [out]}
    return job, brief


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for f in ('evidence', 'lab-root', 'bundle', 'commit', 'data-info'):
        p.add_argument('--' + f, required=True)
    a = p.parse_args()
    out, lab = Path(a.evidence).resolve(), Path(a.lab_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    temp = out / 'tmp'; temp.mkdir(exist_ok=True)
    env = dict(os.environ, TMPDIR=str(temp), PYTHONDONTWRITEBYTECODE='1')
    text = draft(FORK)
    (out / 'BRIEF.draft.md').write_text(text)
    write_once(out / 'ADMISSION.draft.json', read(FORK / 'tools/admission.json'))
    validation = out / 'G76-draft'; validation.mkdir(exist_ok=True)
    (validation / 'BRIEF.md').write_text(text)
    record_command(out, 'G76', ['python3', '-B', str(lab / 'tools/harness.py'), 'validate', str(validation)], env)
    job, brief = smoke_probe()
    probe = out / 'G78-smoke-transport' / digest(json.dumps(job, sort_keys=True).encode())[:12]
    probe.mkdir(parents=True, exist_ok=True)
    write_once(probe / 'job-original.json', job); write_once(probe / 'brief.json', brief)
    handoff = record_command(out, 'G78-prepare', ['python3', '-B', str(lab / 'tools/run_handoff.py'), 'prepare',
        '--job', str(probe / 'job-original.json'), '--brief', str(probe / 'brief.json'),
        '--data-info', a.data_info, '--data-path', ROOT + '/data/data_info.json', '--out', str(probe / 'handoffs'),
        '--bundle', a.bundle, '--bundle-path', ROOT + '/code/' + Path(a.bundle).name,
        '--commit', a.commit, '--pvc-root', ROOT], env).splitlines()[-1]
    record_command(out, 'G78-validate', ['python3', '-B', str(lab / 'tools/run_handoff.py'), 'validate', handoff], env)
    record_command(out, 'G78-offline-launch', ['python3', '-B', str(lab / 'tools/run_handoff.py'), 'launch', handoff], env)
    atom(out / 'gate-scope.json', {'G76': 'PASS draft schema; no approval',
        'G78': 'PARTIAL: transport-only diagnostic smoke prepare/validate/offline launch passed; training smoke remains blocked',
        'training_smoke_blockers': ['staged bundle predates B17', 'generated configs have unset decisions and U1 manifest hashes',
                                    'B18 commit/freeze and B19 draws/gates required'],
        'G77': 'NOT RUN: cluster-aware nrp_doctor lint', 'G79': 'NOT RUN: in-pod bundle identity and mismatch checks',
        'G80': 'NOT RUN: fresh namespace ConfigMap/Job inventory; offline logic tested',
        'sources': {str(p): digest(p.read_bytes()) for p in [lab/'tools/harness.py', lab/'tools/run_handoff.py', Path(a.bundle)]}})
    print(json.dumps(read(out / 'gate-scope.json'), indent=1))


if __name__ == '__main__':
    main()
