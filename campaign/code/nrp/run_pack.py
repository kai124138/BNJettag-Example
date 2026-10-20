#!/usr/bin/env python3
"""One process per arm on one GPU. Only the child imports the ML stack.

Exit 0: all completed/StopIf; 1: code failure; 2: refused; 3: integrity;
76: invalid completion evidence; 124: deadline; 143: interrupted. No blind child retries.
GPU-seconds are allocated GPU wall seconds, shared equally among live children, not
kernel-active seconds. The benchmark reader adds measured pod startup/shutdown overhead.
"""
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

FORK = Path(__file__).resolve().parents[1]
if str(FORK) not in sys.path:
    sys.path.insert(0, str(FORK))
from nrp.common import ROOT, need, read, digest, encoded, write_once, atom, archive, Refused
from nrp.spec import check_execution, expand_plan
from nrp import accounting
from nrp.restart import launch_record


def completed(arm, plan):
    """A valid finished sibling is retained; failed or corrupt terminal state is refused."""
    path = Path(arm['out']) / 'run.json'
    if not path.exists():
        return False
    rec = read(path)
    need(rec.get('exit') == 0 and rec.get('end') in ('completed', 'stopif'), 'terminal failed/integrity run requires repair')
    need(rec.get('config') == arm['config'] and rec.get('code', {}).get('head') == plan['code_sha'],
         'completed sibling has different code/config')
    need(rec.get('wandb', {}).get('run_name') == arm['name'], 'completed sibling identity mismatch')
    need((Path(arm['out']) / 'front/summary.json').is_file(), 'completed sibling lacks front summary')
    return True


def epochs(path):
    p = Path(path) / 'epochs.jsonl'
    if not p.exists():
        return 0
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    es = [r['epoch'] for r in rows]
    need(es == list(range(len(es))), 'epoch records must be contiguous and unique')
    return len(es)


def drawn(arm, plan, code_root):
    root = Path(arm['out'])
    stem = f'init-s{arm["seed"]}-gpu'
    manifest = root / (stem + '.manifest.json')
    tensor = root / (stem + '.npz')
    if not manifest.exists():
        need(not tensor.exists(), 'partial initial tensor draw exists; inspect, never overwrite')
        return False
    rec = read(manifest)
    from cbl import rng  # NumPy/file hashes only; no backend initialization
    need(rec.get('kind') == rng.INIT_KIND and rec.get('schema') == rng.INIT_SCHEMA, 'invalid draw manifest')
    need(rec['arm_drawn'] == 'CH-8' and rec['identity']['seed'] == arm['seed'], 'draw seed/builder mismatch')
    dev = rec['identity']['device']
    need(dev['jax_platform'] == 'gpu' and dev['n_devices'] == 1 and
         [x.replace(' ', '-') for x in dev['device_kind']] == [plan['gpu_product']], 'draw device mismatch')
    need(rec['file']['name'] == tensor.name and digest(tensor.read_bytes()) == rec['file']['sha256'], 'draw tensor hash mismatch')
    need(rec['identity']['draw_code'] == {rel: digest((Path(code_root) / rel).read_bytes()) for rel in rng.DRAW_CODE},
         'draw code set/hash changed')
    need(rec['identity']['stack'] == rng.stack_versions() and rec['identity']['matmul_precision'] == 'float32',
         'draw runtime stack/precision changed')
    try:
        rng.read_checked_tensors(manifest, rec)
    except rng.InitTensorRefused as e:
        raise Refused(str(e)) from e
    for rel, sha in rec['identity']['draw_code'].items():
        need(digest((Path(code_root) / rel).read_bytes()) == sha, 'draw code changed')
    dman = read(Path(code_root) / 'manifests/data-prep-150c.json')
    key = 'smoke' if plan['archive_kind'] == 'smoke' else 'train'
    need(rec['identity']['data']['train_archive_sha256'] == dman[key]['archive']['sha256'], 'draw archive mismatch')
    need(rec['identity']['data']['data_manifest_sha256'] == digest((Path(code_root) / 'manifests/data-prep-150c.json').read_bytes()),
         'draw preparation manifest mismatch')
    return True


def command(arm, plan, config_path, code_root=FORK, python=sys.executable):
    base = [str(python), '-B', str(Path(code_root) / 'cbl/runner.py')]
    shared = ['--train-archive', archive(plan['archive_kind']), '--archive-kind', plan['archive_kind'],
              '--device', 'gpu', '--out', arm['out'], '--mem-fraction', str(plan['mem_fraction'])]
    if plan['mode'] == 'draw':
        return base + ['draw', '--seed', str(arm['seed']), '--place', 'NRP/' + plan['gpu_product']] + shared
    from nrp.restart import prepare_run
    every = prepare_run(arm, plan)
    cmd = base + ['run', '--config', str(config_path), '--source-bundle', plan['source_bundle'],
                  '--resume-every', str(every), '--pack-k', str(plan['k'])] + shared
    if Path(arm['out']).exists():
        cmd.append('--resume')  # prepare_run has checked the committed checkpoint
    return cmd


def sample_gpu():
    """No environment/config capture. Missing telemetry stays null, never zero."""
    def query(args):
        p = subprocess.run(['nvidia-smi'] + args + ['--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=10)
        need(p.returncode == 0, 'nvidia-smi unavailable')
        return list(csv.reader(p.stdout.splitlines()))
    try:
        rows = query(['--query-gpu=name,uuid,driver_version,utilization.gpu,memory.used,memory.total'])
        need(len(rows) == 1, 'exactly one visible GPU required (U1 n_devices)')
        name, uid, driver, util, used, total = [x.strip() for x in rows[0]]
        processes = query(['--query-compute-apps=pid,used_gpu_memory'])
        return {'product': name.replace(' ', '-'), 'uuid': uid, 'driver': driver,
                'utilization_pct': float(util), 'used_mib': float(used), 'total_mib': float(total),
                'process_mib': {int(p): float(m) for p, m in processes}}
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return {'unavailable': type(e).__name__}


def rss_mib(pid):
    try:
        fields = Path(f'/proc/{pid}/status').read_text().splitlines()
        return max(float(x.split()[1]) / 1024 for x in fields if x.startswith(('VmRSS:', 'VmHWM:')))
    except (OSError, ValueError):
        return None


def terminate(children):
    live = [e for e in children if e['proc'].poll() is None]
    for e in live:
        try:
            os.killpg(e['proc'].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    end = time.monotonic() + 10
    while any(e['proc'].poll() is None for e in live) and time.monotonic() < end:
        time.sleep(0.1)
    for e in live:
        if e['proc'].poll() is None:
            os.killpg(e['proc'].pid, signal.SIGKILL)
        e['proc'].wait()


def execute(plan, *, code_root=FORK, output_root=ROOT, sample=sample_gpu, poll_s=1.0):
    """output_root is only the evidence root; production arm paths are checked independently."""
    check_execution(plan, code_root)
    sample0 = sample()
    need(sample0.get('product') == plan['gpu_product'], 'GPU identity unavailable or different from selected product')
    job = os.environ.get('CBL_JOB', plan['pack'])
    invocation = uuid.uuid4().hex
    evidence = Path(output_root) / 'packs' / plan['pack'] / job / invocation
    evidence.mkdir(parents=True, exist_ok=False)
    write_once(evidence / 'plan.json', plan)
    locks, children, skipped = [], [], []
    stop = [None]
    old_handlers = {}
    started = time.monotonic()
    previous = started
    last_sample = -1e9
    samples = []
    code = 0
    try:
        # Hold all run locks before starting any process; no overlapping attempts on the PVC.
        for arm in plan['arms']:
            lockdir = Path(output_root) / 'run-locks'
            lockdir.mkdir(exist_ok=True)
            key = digest((arm['out'] + '/' + str(arm['seed'])).encode())
            handle = (lockdir / key).open('a+')
            locks.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, lambda signum, frame: stop.__setitem__(0, signum))

        pending = list(plan['arms'])
        while pending or any(not e.get('finished') for e in children):
            live = [e for e in children if not e.get('finished')]
            now = time.monotonic()
            # Charge elapsed allocation to processes live during that interval, including their exits.
            for e in live:
                e['allocated_gpu_seconds'] += (now - previous) / len(live)
            previous = now
            for e in live:
                rec = accounting.publish(e['allocation_path'], seconds=e['allocated_gpu_seconds'],
                                         prior=e['prior_seconds'], live=len(live), now=now)
                if Path(e['arm']['out']).exists():
                    atom(Path(e['arm']['out']) / 'allocations' / (invocation + '-' + e['arm']['name'] + '.json'), rec)
            if now - last_sample >= 60 or not samples:
                s = sample() if samples else sample0
                s['host_rss_mib'] = {e['arm']['name']: rss_mib(e['proc'].pid) for e in live}
                s['checkpoint_epochs'] = {}
                for e in live:
                    try:
                        s['checkpoint_epochs'][e['arm']['name']] = read(Path(e['arm']['out']) / 'resume/latest.json')['epochs_done']
                    except (OSError, ValueError, KeyError):
                        s['checkpoint_epochs'][e['arm']['name']] = None
                samples.append({'elapsed_s': now - started, **s})
                with (evidence / 'gpu.jsonl').open('a') as f:
                    f.write(json.dumps(samples[-1]) + '\n')
                last_sample = now
                for e in live:
                    gm = s.get('process_mib', {}).get(e['proc'].pid)
                    hm = rss_mib(e['proc'].pid)
                    for key, value in (('gpu_peak_mib_sampled', gm), ('host_peak_mib_sampled', hm)):
                        if value is not None:
                            e[key] = max(e.get(key) or 0, value)
                if 'product' in s and s['product'] != plan['gpu_product']:
                    code = 3
                    stop[0] = signal.SIGTERM
            for e in live:
                hm = rss_mib(e['proc'].pid)
                if hm is not None:
                    e['host_peak_mib_sampled'] = max(e.get('host_peak_mib_sampled') or 0, hm)
                rc = e['proc'].poll()
                if rc is None:
                    continue
                e.update(finished=True, exit=rc, wall_seconds=now - e['started'])
                if rc == 0 and plan['mode'] != 'draw':
                    try:
                        need(completed(e['arm'], plan), 'child exited 0 without completion record')
                    except (ValueError, OSError):
                        e['exit'] = 76
                elif rc == 0:
                    try:
                        need(drawn(e['arm'], plan, code_root), 'draw exited 0 without manifest')
                    except (ValueError, OSError, KeyError):
                        e['exit'] = 76
                if e['exit']:
                    code = 3 if e['exit'] == 3 else (code or (e['exit'] if e['exit'] > 0 else 143))
                    if e['exit'] in (2, 3, 76):
                        stop[0] = signal.SIGTERM
            if now - started >= plan['deadline_s'] - 30:
                code, stop[0] = 124, signal.SIGTERM
            if stop[0]:
                code = code or 143
                break
            # Draws are sequential; packs start every arm together.
            while pending and (plan['mode'] != 'draw' or not any(e['proc'].poll() is None for e in children)):
                arm = pending.pop(0)
                if plan['mode'] != 'draw' and completed(arm, plan):
                    skipped.append(arm['name'])
                    continue
                if plan['mode'] == 'draw':
                    if drawn(arm, plan, code_root):
                        skipped.append(arm['name'])
                        continue
                before = 0
                if plan['mode'] != 'draw' and Path(arm['out']).exists():
                    latest = Path(arm['out']) / 'resume/latest.json'
                    before = read(latest)['epochs_done'] if latest.exists() else 0
                config = evidence / (arm['name'] + '.config.json')
                if 'config' in arm:
                    write_once(config, arm['config'])
                prior = accounting.prior_seconds(arm['out'])
                cmd = command(arm, plan, config, code_root)
                log = (evidence / (arm['name'] + '.log')).open('xb')
                env = dict(os.environ, WANDB_DIR=arm['out'], PYTHONDONTWRITEBYTECODE='1',
                           CBL_LAUNCH_RECORD=json.dumps({**launch_record(arm, plan),
                               'resume_every': int(cmd[cmd.index('--resume-every') + 1])})
                           if '--resume-every' in cmd else '',
                           CBL_ALLOCATION_FILE=str(evidence / (arm['name'] + '.allocation.json')))
                allocation_path = evidence / (arm['name'] + '.allocation.json')
                accounting.publish(allocation_path, seconds=0, prior=prior, live=sum(e['proc'].poll() is None for e in children) + 1)
                try:
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=code_root,
                                            env=env, start_new_session=True)
                except Exception:
                    log.close()
                    raise
                children.append({'arm': arm, 'proc': proc, 'log': log, 'command': cmd,
                                 'started': time.monotonic(), 'before': before, 'allocated_gpu_seconds': 0.0,
                                 'allocation_path': allocation_path, 'prior_seconds': prior,
                                 'gpu_peak_mib_sampled': None, 'host_peak_mib_sampled': None})
                last_sample = -1e9  # obtain an early sample even for a short smoke
            time.sleep(poll_s)
    except Exception:
        code = code or 2
        raise
    finally:
        terminate(children)
        wall = time.monotonic() - started
        # Include supervisor startup, sampling and shutdown; distribute otherwise unassigned time.
        unassigned = max(0.0, wall - sum(e['allocated_gpu_seconds'] for e in children))
        for e in children:
            e['log'].close()
            e['allocated_gpu_seconds'] += unassigned / len(children)
            end_epochs = 0
            try:
                end_epochs = epochs(e['arm']['out']) if plan['mode'] != 'draw' else 0
            except (OSError, ValueError, KeyError):
                code = code or 76
            done = max(0, end_epochs - e['before'])
            row = {'schema': 1, 'job': job, 'invocation': invocation, 'name': e['arm']['name'],
                   'arm_class': e['arm']['arm_class'], 'gpu_product': plan['gpu_product'], 'k': plan['k'],
                   'code_sha': plan['code_sha'], 'source_bundle': plan['source_bundle'],
                   'exit': e.get('exit', e['proc'].returncode), 'epochs_completed_this_process': done,
                   'wall_seconds': e.get('wall_seconds', time.monotonic() - e['started']),
                   'allocated_gpu_seconds': e['allocated_gpu_seconds'],
                   'gpu_seconds_per_run_epoch': e['allocated_gpu_seconds'] / done if done else None,
                   'gpu_peak_mib_sampled': e['gpu_peak_mib_sampled'], 'host_peak_mib_sampled': e['host_peak_mib_sampled'],
                   'gpu_seconds_basis': 'equal share among live processes; includes supervisor overhead; excludes pod init',
                   'resume_requested': '--resume' in e['command'], 'command': e['command']}
            write_once(evidence / (e['arm']['name'] + '.telemetry.json'), row)
            run_dir = Path(e['arm']['out'])
            if run_dir.exists():
                atom(run_dir / 'allocations' / (invocation + '-' + e['arm']['name'] + '.json'),
                     {'seconds': e['allocated_gpu_seconds'], 'prior_seconds': e['prior_seconds'], 'live': 1,
                      'monotonic': time.monotonic()})
                stem = invocation + '-' + e['arm']['name']
                write_once(run_dir / 'telemetry' / (stem + '.json'), row)
                # Preserve the complete child log under the durable run root too.
                target = run_dir / 'telemetry' / (stem + '.log')
                with target.open('xb') as f:
                    f.write((evidence / (e['arm']['name'] + '.log')).read_bytes())
        write_once(evidence / 'pack.json', {'schema': 1, 'job': job, 'invocation': invocation,
                   'wall_seconds': wall, 'exit': code, 'skipped_complete': skipped,
                   'telemetry': [e['arm']['name'] + '.telemetry.json' for e in children],
                   'gpu_product': plan['gpu_product'], 'k': plan['k'], 'code_sha': plan['code_sha'],
                   'source_bundle': plan['source_bundle'], 'node': os.environ.get('CBL_NODE'),
                   'pod': os.environ.get('CBL_POD')})
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        for handle in locks:
            handle.close()
    return code


def main():
    try:
        plan = expand_plan(json.loads(os.environ['CBL_PLAN']), FORK)
        return execute(plan)
    except (ValueError, KeyError, OSError) as e:
        print('PACK REFUSED:', e, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
