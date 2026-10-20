"""Shared parent of gates G37-G42 (PREFLIGHT-PLAN section 3 rows G37-G42; STUDY v0.10 sections 2.2, 8.1, 8.8 and
the section 13 structure-record and ablation row; modules M9 cbl/front.py, M10 cbl/structure.py, M11
cbl/ablate.py). Standard library only, plus gatelib, fileaudit, binarygate and runnergate.

A gate's Keras part runs in one fresh child process (`gates/b8_child.py <cmd>`) on one device under the kernel
open-file audit of B2, watching the held-out archive, its directory and the raw held-out files: these gates use
constructed models and constructed jets only, so the audit must show no open of any of them. On the GPU the
child runs with XLA_FLAGS=--xla_gpu_deterministic_ops=true (decision B5-1); on the CPU with no XLA_FLAGS. The
parent prints the child's checks one by one, checks the child's backend, devices, matmul precision and
XLA_FLAGS, and keeps the child's readings in result.json. Every reading is engineering, not quotable.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import binarygate
import fileaudit
import gatelib
import runnergate

FORK = Path(__file__).resolve().parents[1]


def run_child(g, cmd, device, work, data_manifest, children_out=None, timeout=5400, extra_args=()):
    """Run `b8_child.py cmd` under the audit; print its checks into gate g. Returns the child's record."""
    env = {k: v for k, v in os.environ.items()
           if k not in ('KERAS_BACKEND', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_FLAGS')}
    if device == 'gpu':
        env['XLA_FLAGS'] = runnergate.DETERMINISTIC_GPU_FLAG
    out = work / f'{cmd}.json'
    argv = [sys.executable, '-I', '-B', str(FORK / 'gates' / 'b8_child.py'), cmd, '--device', device,
            '--out', str(out), '--work', str(work / 'child-work')] + list(extra_args)
    g.info('child argv', ' '.join(argv[3:]) + (f'  [XLA_FLAGS={env["XLA_FLAGS"]}]' if 'XLA_FLAGS' in env else ''))
    r = fileaudit.run_audited(argv, watches=binarygate.heldout_watches(data_manifest), env=env, cwd=str(work),
                              timeout=timeout)
    (work / f'{cmd}.stdout.txt').write_text(r['stdout'])
    (work / f'{cmd}.stderr.txt').write_text(r['stderr'])
    audit = {k: r[k] for k in ('events', 'events_before_start', 'fd_paths')}
    if children_out:
        co = Path(children_out)
        co.mkdir(parents=True, exist_ok=True)
        for src in (out, work / f'{cmd}.stdout.txt', work / f'{cmd}.stderr.txt'):
            if src.exists():
                shutil.copy2(src, co / src.name)
        (co / f'{cmd}.argv.txt').write_text(' '.join(argv) + '\n')
        (co / f'{cmd}.audit.json').write_text(json.dumps(audit, indent=1, sort_keys=True) + '\n')
    child = json.loads(out.read_text()) if out.exists() else None
    ok = r['returncode'] == 0 and child is not None and 'error' not in child
    g.check(ok, f'{cmd} child exit 0' + ('' if ok else f' ({(child or {}).get("error")})'))
    if child is not None and 'traceback' in child:
        g.info('child traceback', child['traceback'][-1500:])
    st = (child or {}).get('setup', {})
    g.check(st.get('jax_backend') == device, f'child JAX backend {st.get("jax_backend")} ({st.get("devices")})')
    g.check(st.get('matmul_precision') == st.get('matmul_precision_after') == 'float32',
            f'child matmul precision {st.get("matmul_precision")} before and {st.get("matmul_precision_after")} after its '
            f'checks (STUDY 7.3)')
    want_flags = runnergate.DETERMINISTIC_GPU_FLAG if device == 'gpu' else None
    g.check(st.get('XLA_FLAGS') == want_flags, f'child XLA_FLAGS {st.get("XLA_FLAGS")!r} (decision B5-1 on the GPU)')
    fe = runnergate.forbidden_events(audit)
    g.check(not fe, f'open-file audit: no open of the held-out archive or any raw held-out file ({len(fe)} forbidden '
                    f'events, {len(audit["events"])} events in the watched paths)')
    checks = (child or {}).get('checks', [])
    g.check(len(checks) > 0, f'the child ran {len(checks)} checks')
    for c in checks:
        g.check(c['ok'], c['what'])
    return child


def parser(doc, devices=('cpu', 'gpu')):
    ap = argparse.ArgumentParser(description=doc.splitlines()[0])
    ap.add_argument('--device', choices=list(devices), required=True)
    ap.add_argument('--expect-sha12', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--data-manifest', default=str(FORK / 'manifests' / 'data-prep-150c.json'))
    ap.add_argument('--json', default=None)
    ap.add_argument('--children-out', default=None)
    return ap


def main(gate, cmd, doc, timeout=5400, devices=('cpu', 'gpu')):
    a = parser(doc, devices).parse_args()
    g = gatelib.Gate(gate)
    rec = {'header': g.header(a.expect_sha12)}
    gatelib.check_venv_of_record(g)
    print('label       engineering, not quotable')
    work = Path(a.work)
    g.check(not work.exists(), f'work directory {work} does not exist yet')
    work.mkdir(parents=True)
    child = run_child(g, cmd, a.device, work, a.data_manifest, a.children_out, timeout)
    rec['child'] = {'setup': (child or {}).get('setup'), 'info': (child or {}).get('info'),
                    'n_checks': len((child or {}).get('checks', []))}
    g.check_clean('at end')
    return g.finish(rec, a.json)
