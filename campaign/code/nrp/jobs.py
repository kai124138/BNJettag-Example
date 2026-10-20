"""Generate JSON Jobs and prepare bundle-mode handoffs; never submits."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from nrp.common import (ROOT, CAMPAIGN, IMAGE, PYTHON, need, pvc, digest, encoded, read,
                        write_once, load_tool, Refused)
from nrp.spec import resolve, wire_plan


def build_job(spec, plan):
    from nrp.restart import bind_plan
    plan = bind_plan(plan, run_state_root=spec.get('run_state_root'))
    r = spec['resources']
    name = f'kai-cbl1016-{plan["pack"]}-a{plan["job_attempt"]}'
    need(len(name) <= 63, 'Job name too long')
    labels = {'user': 'kai', 'campaign': CAMPAIGN, 'bnjettag.io/stage': plan['stage'],
              'bnjettag.io/arm': plan['pack']}
    annotations = {'bnjettag.io/arms-per-pod': str(plan['k']),
                   'bnjettag.io/arm': plan['pack'],
                   'bnjettag.io/gpu-product': plan['gpu_product'],
                   'bnjettag.io/pack': plan['pack']}
    if plan['k'] == 1:
        annotations['bnjettag.io/single-arm-justified'] = spec['single_arm_justification']
    resources = {'cpu': str(r['cpu']), 'memory': f'{r["memory_gi"]}Gi', 'nvidia.com/gpu': '1',
                 'ephemeral-storage': f'{r["cache_gi"]}Gi'}
    env = {'CBL_PLAN': json.dumps(wire_plan(plan), sort_keys=True, separators=(',', ':')),
           'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
           'GIT_OPTIONAL_LOCKS': '0', 'GIT_CONFIG_PARAMETERS': "'safe.directory=/cmcode'",
           'HOME': '/scratch', 'TMPDIR': '/scratch', 'XDG_CACHE_HOME': '/scratch/cache',
           'WANDB_MODE': 'offline', 'WANDB_PROJECT': 'bnjettag-chang-bitlinear',
           'WANDB_RUN_GROUP': CAMPAIGN, 'WANDB_CACHE_DIR': '/scratch/wandb-cache',
           'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
           'XLA_FLAGS': '--xla_gpu_deterministic_ops=true' if plan['deterministic_ops'] else ''}
    expressions = [{'key': 'nvidia.com/gpu.product', 'operator': 'In', 'values': [plan['gpu_product']]}]
    if spec['excluded_nodes']:
        expressions.append({'key': 'kubernetes.io/hostname', 'operator': 'NotIn', 'values': spec['excluded_nodes']})
    return {'apiVersion': 'batch/v1', 'kind': 'Job', 'metadata': {'name': name, 'namespace': 'cms-ml',
            'labels': labels, 'annotations': annotations}, 'spec': {
        'parallelism': 1, 'completions': 1, 'backoffLimit': 0, 'podReplacementPolicy': 'Failed',
        'activeDeadlineSeconds': r['deadline_s'], 'ttlSecondsAfterFinished': r['ttl_s'],
        'podFailurePolicy': {'rules': [{'action': 'FailJob', 'onExitCodes': {
            'containerName': 'pack', 'operator': 'In', 'values': [1, 2, 3, 76, 124]}}]},
        'template': {'metadata': {'labels': labels, 'annotations': annotations}, 'spec': {
            'automountServiceAccountToken': False, 'restartPolicy': 'Never', 'terminationGracePeriodSeconds': 60,
            'nodeSelector': {'kubernetes.io/arch': 'amd64'},
            'affinity': {'nodeAffinity': {'requiredDuringSchedulingIgnoredDuringExecution': {
                'nodeSelectorTerms': [{'matchExpressions': expressions}]}}},
            'containers': [{'name': 'pack', 'image': IMAGE, 'workingDir': '/cmcode',
                'command': [PYTHON, '-B', '/cmcode/nrp/run_pack.py'],
                'env': [{'name': k, 'value': v} for k, v in env.items()] + [
                    {'name': 'CBL_JOB', 'value': name},
                    {'name': 'CBL_POD', 'valueFrom': {'fieldRef': {'fieldPath': 'metadata.name'}}},
                    {'name': 'CBL_NODE', 'valueFrom': {'fieldRef': {'fieldPath': 'spec.nodeName'}}}],
                'resources': {'requests': resources, 'limits': dict(resources)},
                'volumeMounts': [{'name': 'data', 'mountPath': '/data'}, {'name': 'scratch', 'mountPath': '/scratch'}]}],
            'volumes': [{'name': 'data', 'persistentVolumeClaim': {'claimName': 'kai-data'}},
                        {'name': 'scratch', 'emptyDir': {'sizeLimit': f'{r["cache_gi"]}Gi'}}]}}}}


def build_brief(spec, plan):
    return {'purpose': f'{CAMPAIGN} {plan["mode"]} {plan["pack"]}; engineering until VERIFY',
            'changes': f'Bundle {plan["source_bundle"]}, commit {plan["code_sha"]}; immutable per-arm launch copies; '
                       'infrastructure Job attempt changes no scientific run identity.',
            'approval_ref': spec['approval_ref'],
            'scientific_gate': {'status': spec['scientific_gate'], 'reference': spec['gate_reference']},
            'runs': [{'name': a['name'], 'config_path': 'code/' + a['config_path']} for a in plan['arms']],
            'expected_metrics': [{'name': 'engineering time, memory, utilization and integrity',
                                  'split': 'training/validation only; no held-out read',
                                  'expectation': 'Measured, not predicted; no physics conclusion from PREFLIGHT.'}],
            'stop_rules': [{'condition': 'runner refusal, integrity stop or pack deadline',
                            'action': 'existing-runner-guard',
                            'approval_ref': 'STUDY v0.10 sections 11, 13; interfaces sections 13, 19'}],
            'outputs': sorted({a['out'] for a in plan['arms']} | {f'{ROOT}/packs/{plan["pack"]}'}),
            'launch_copies': [{'generated_sha256': a['generated_sha256'], 'config': a['config']}
                              for a in plan['arms'] if 'config' in a]}


def prepare(spec, *, bundle, bundle_path, commit, data_info, out, launcher):
    """All admission runs before artifacts are written. Caller supplies final committed bundle."""
    rh = load_tool(launcher, 'cbl_run_handoff')
    raw = Path(bundle).read_bytes()
    _, _, files = rh.bundle_source(raw, commit)
    files = {p.removeprefix('code/'): b for p, b in files.items()}
    need('nrp/run_pack.py' in files, 'bundle lacks B17 runner; B18 must commit and freeze B17 first')
    # Refuse preparing a newer or older execution contract with the local generator.
    fork = Path(__file__).parents[1]
    contract = [str(p.relative_to(fork)) for d in ('nrp', 'gen', 'cbl') for p in (fork / d).glob('*.py')]
    for rel in contract:
        need(files.get(rel) == (Path(__file__).parents[1] / rel).read_bytes(), f'bundle/local tooling mismatch: {rel}')
    plan = resolve(spec, files, commit, digest(raw))
    from nrp.restart import bind_plan
    plan = bind_plan(plan, run_state_root=spec.get('run_state_root'))
    job, brief = build_job(spec, plan), build_brief(spec, plan)
    directory = Path(out) / 'manifests' / job['metadata']['name']
    for n, obj in (('spec.json', spec), ('plan.json', plan), ('job.json', job), ('brief.json', brief)):
        write_once(directory / n, obj)
    # Use the supported launcher as the source of truth. Its own temporary clone is offline.
    argv = [sys.executable, '-B', str(launcher), 'prepare', '--job', str(directory / 'job.json'),
            '--brief', str(directory / 'brief.json'), '--data-info', str(data_info),
            '--data-path', ROOT + '/data/data_info.json', '--out', str(Path(out) / 'handoffs'),
            '--bundle', str(bundle), '--bundle-path', pvc(bundle_path), '--commit', commit, '--pvc-root', ROOT]
    result = subprocess.run(argv, capture_output=True, text=True)
    need(result.returncode == 0, 'run_handoff prepare failed: ' + result.stderr[-1000:])
    handoff = Path(result.stdout.strip().splitlines()[-1])
    rh.validate_dir(handoff)
    write_once(Path(out) / 'attempts' / (job['metadata']['name'] + '.json'), {
        'job': job['metadata']['name'], 'spec': spec, 'plan': plan, 'handoff': str(handoff.resolve()),
        'bundle': str(Path(bundle).resolve()), 'bundle_path': bundle_path, 'commit': commit,
        'data_info': str(Path(data_info).resolve()), 'launcher': str(Path(launcher).resolve()),
        'status': 'prepared_not_submitted'})
    return handoff


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('spec', 'bundle', 'bundle-path', 'commit', 'data-info', 'out', 'launcher'):
        p.add_argument('--' + flag, required=True)
    a = p.parse_args(argv)
    try:
        print(prepare(read(a.spec), bundle=a.bundle, bundle_path=a.bundle_path, commit=a.commit,
                      data_info=a.data_info, out=a.out, launcher=a.launcher))
    except (ValueError, KeyError, OSError) as e:
        print('REFUSED:', e, file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
