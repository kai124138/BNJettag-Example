"""Shared part of gates G26-G33 (PREFLIGHT-PLAN section 3 rows G26-G33; STUDY v0.10 sections 3.1-3.2 and 13
cost row; module M7 cbl/cost.py with cbl/cost_pure.py). Standard library only, plus gatelib, fileaudit and
runnergate; the B6 pattern of gates/binarygate.py with the child gates/cost_child.py.

A gate runs one fresh child process (`gates/cost_child.py <cmd>`) on one device under the kernel open-file
audit of B2, watching the held-out archive, its directory and the raw held-out files: these gates use
constructed tensors only, so the audit must show no open of any of them. On the GPU the child runs with
XLA_FLAGS=--xla_gpu_deterministic_ops=true (decision B5-1); on the CPU with no XLA_FLAGS. The parent prints
the child's checks one by one, checks the child's backend, devices, matmul precision and XLA_FLAGS, and keeps
the child's readings in result.json. Every reading is engineering, not quotable.
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
DETERMINISTIC_GPU_FLAG = runnergate.DETERMINISTIC_GPU_FLAG
CHILD = 'cost_child.py'


def run_child(g, cmd, *, device, work, data_manifest, children_out, extra=(), timeout=7200, name=None):
    """One audited child; returns (record or None, audit). Prints nothing but the argv line."""
    name = name or cmd
    env = {k: v for k, v in os.environ.items()
           if k not in ('KERAS_BACKEND', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_FLAGS')}
    if device == 'gpu':
        env['XLA_FLAGS'] = DETERMINISTIC_GPU_FLAG
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    out = work / f'{name}.json'
    argv = [sys.executable, '-I', '-B', str(FORK / 'gates' / CHILD), cmd, '--device', device, '--out', str(out),
            '--work', str(work / f'{name}-work')] + [str(x) for x in extra]
    g.info(f'{name} argv', ' '.join(argv[3:]) + (f'  [XLA_FLAGS={env["XLA_FLAGS"]}]' if 'XLA_FLAGS' in env else ''))
    r = fileaudit.run_audited(argv, watches=binarygate.heldout_watches(data_manifest), env=env, cwd=str(work), timeout=timeout)
    (work / f'{name}.stdout.txt').write_text(r['stdout'])
    (work / f'{name}.stderr.txt').write_text(r['stderr'])
    audit = {k: r[k] for k in ('events', 'events_before_start', 'fd_paths')}
    if children_out:
        co = Path(children_out)
        co.mkdir(parents=True, exist_ok=True)
        for src in (out, work / f'{name}.stdout.txt', work / f'{name}.stderr.txt'):
            if src.exists():
                shutil.copy2(src, co / src.name)
        (co / f'{name}.argv.txt').write_text(' '.join(argv) + '\n')
        (co / f'{name}.audit.json').write_text(json.dumps(audit, indent=1, sort_keys=True) + '\n')
    child = json.loads(out.read_text()) if out.exists() else None
    if child is not None:
        child['returncode'] = r['returncode']
    return child, audit


def check_child(g, name, child, audit, device):
    """The parent's checks of one child: exit, backend, precision, XLA_FLAGS, audit; then its own checks."""
    ok = child is not None and child.get('returncode') == 0 and 'error' not in child
    g.check(ok, f'{name} child exit 0' + ('' if ok else f' ({(child or {}).get("error")})'))
    if child is not None and 'traceback' in child:
        g.info('child traceback', child['traceback'][-1500:])
    st = (child or {}).get('setup', {})
    want_backend = 'cpu' if device == 'cpu' else 'gpu'
    g.check(st.get('jax_backend') == want_backend, f'{name}: child JAX backend {st.get("jax_backend")} ({st.get("devices")})')
    g.check(st.get('matmul_precision') == st.get('matmul_precision_after') == 'float32',
            f'{name}: child matmul precision {st.get("matmul_precision")} before and {st.get("matmul_precision_after")} '
            f'after its checks (STUDY 7.3)')
    want_flags = DETERMINISTIC_GPU_FLAG if device == 'gpu' else None
    g.check(st.get('XLA_FLAGS') == want_flags, f'{name}: child XLA_FLAGS {st.get("XLA_FLAGS")!r} (decision B5-1 on the GPU)')
    fe = runnergate.forbidden_events(audit)
    g.check(not fe, f'{name}: open-file audit, no open of the held-out archive or any raw held-out file ({len(fe)} '
                    f'forbidden events, {len(audit["events"])} events in the watched paths)')
    checks = (child or {}).get('checks', [])
    g.check(len(checks) > 0, f'{name}: the child ran {len(checks)} checks')
    for c in checks:
        g.check(c['ok'], c['what'])
    return {'setup': st, 'info': (child or {}).get('info'), 'n_checks': len(checks)}


def main(gate, cmd, doc, timeout=7200):
    ap = argparse.ArgumentParser(description=doc.splitlines()[0])
    ap.add_argument('--device', choices=['cpu', 'gpu'], required=True)
    ap.add_argument('--expect-sha12', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--data-manifest', default=str(FORK / 'manifests' / 'data-prep-150c.json'))
    ap.add_argument('--json', default=None)
    ap.add_argument('--children-out', default=None)
    a = ap.parse_args()
    g = gatelib.Gate(gate)
    rec = {'header': g.header(a.expect_sha12)}
    gatelib.check_venv_of_record(g)
    print('label       engineering, not quotable')
    work = Path(a.work).resolve()
    g.check(not work.exists(), f'work directory {work} does not exist yet')
    work.mkdir(parents=True)
    child, audit = run_child(g, cmd, device=a.device, work=work, data_manifest=a.data_manifest,
                             children_out=a.children_out, timeout=timeout)
    rec['child'] = check_child(g, cmd, child, audit, a.device)
    g.check_clean('at end')
    return g.finish(rec, a.json)
