"""Shared helpers for the gate scripts from G02 on (PREFLIGHT-PLAN section 2 row M22).

Standard library only. A gate prints a header (interpreter, host, git, HEAD), one line per check
("PASS  ..." or "FAIL  ..."), and ends with "<gate> PASS" (exit 0) or "<gate> FAIL" (exit 1).
`g01_base.py` predates this file and keeps its own copy of the git and header code.
"""

import importlib.metadata as md
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

FORK = Path(__file__).resolve().parents[1]
LAB_ROOT_DEFAULT = Path(__file__).resolve().parents[4]
VENV_OF_RECORD = 'cbl1016-jax2'  # B1 owner decision, PREFLIGHT.md
FREEZE = FORK / 'env' / 'training' / 'freeze.txt'


def git(*args):
    env = dict(os.environ, LC_ALL='C', GIT_PAGER='cat', GIT_OPTIONAL_LOCKS='0')
    r = subprocess.run(['git', '-C', str(FORK)] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env=env, check=False)
    return r.returncode, r.stdout.decode('utf-8', 'replace'), r.stderr.decode('utf-8', 'replace')


class Gate:
    def __init__(self, name):
        self.name = name
        self.failures = []
        self.checks = []

    def check(self, ok, what):
        ok = bool(ok)
        print('%s  %s' % ('PASS' if ok else 'FAIL', what), flush=True)
        self.checks.append({'ok': ok, 'what': what})
        if not ok:
            self.failures.append(what)
        return ok

    def info(self, key, value):
        print('      %s: %s' % (key, value if isinstance(value, str) else json.dumps(value, default=str)), flush=True)

    def header(self, expect_sha12):
        _, git_version, _ = git('--version')
        rc, head, err = git('rev-parse', 'HEAD')
        head = head.strip()
        rec = {
            'gate': self.name, 'interpreter': sys.executable, 'prefix': sys.prefix,
            'python': sys.version.replace('\n', ' '), 'platform': platform.platform(), 'host': platform.node(),
            'git': git_version.strip(), 'fork': str(FORK), 'HEAD': head if rc == 0 else 'ERROR ' + err.strip(),
        }
        for k in ('gate', 'interpreter', 'prefix', 'python', 'platform', 'host', 'git', 'fork', 'HEAD'):
            print('%-11s %s' % (k, rec[k]))
        print('', flush=True)
        self.check(rc == 0 and len(head) == 40, 'git rev-parse HEAD succeeds')
        if expect_sha12 is not None:
            self.check(head[:12] == expect_sha12, 'HEAD[:12] %s equals --expect-sha12 %s' % (head[:12], expect_sha12))
        self.check_clean('at start')
        return rec

    def check_clean(self, when):
        rc, out, _ = git('status', '--porcelain')
        for line in out.splitlines():
            print('      | %s' % line)
        return self.check(rc == 0 and out == '', 'git status --porcelain is empty (%s)' % when)

    def finish(self, record=None, json_path=None):
        if json_path:
            rec = dict(record or {}, gate=self.name, checks=self.checks, failures=self.failures,
                       outcome='FAIL' if self.failures else 'PASS')
            Path(json_path).write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + '\n', encoding='utf-8')
        print('')
        if self.failures:
            print('%s FAIL (%d failed check(s))' % (self.name, len(self.failures)), flush=True)
            return 1
        print('%s PASS' % self.name, flush=True)
        return 0


def _norm(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def installed_distributions():
    """{normalized name: version} of every distribution on sys.path."""
    out = {}
    for d in md.distributions():
        name = d.metadata['Name']
        if name:
            out[_norm(name)] = d.version
    return out


def check_venv_of_record(gate, freeze_path=FREEZE):
    """The running interpreter is the venv of record and its distributions equal freeze.txt."""
    gate.check(Path(sys.prefix).name == VENV_OF_RECORD, 'sys.prefix %s is the venv of record %s' % (sys.prefix, VENV_OF_RECORD))
    want = {}
    for line in Path(freeze_path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            name, _, ver = line.partition('==')
            want[_norm(name)] = ver
    have = installed_distributions()
    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    wrong = sorted(n for n in set(want) & set(have) if want[n] != have[n])
    gate.info('distributions', {'freeze_lines': len(want), 'installed': len(have), 'missing': missing, 'extra': extra, 'version_differs': wrong})
    gate.check(not (missing or extra or wrong), 'installed distributions equal env/training/freeze.txt (%d packages)' % len(want))


def site_packages_of(top='hgq'):
    """The directory that holds the installed package `top`, found without importing it."""
    import importlib.util

    spec = importlib.util.find_spec(top)
    if spec is None or spec.origin is None:
        raise RuntimeError('%s is not installed in %s' % (top, sys.prefix))
    return Path(spec.origin).resolve().parents[1]


def subprocess_python(code, extra_env=None, timeout=600):
    """Run `code` in a fresh process of this interpreter (-I -B), JAX on the CPU. Returns
    (returncode, stdout, stderr)."""
    env = {k: v for k, v in os.environ.items() if k not in ('KERAS_BACKEND', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_MEM_FRACTION')}
    env['JAX_PLATFORMS'] = 'cpu'
    env.update(extra_env or {})
    r = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True, env=env,
                       cwd=str(FORK), timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def result_line(stdout):
    rows = [json.loads(line[len('RESULT '):]) for line in stdout.splitlines() if line.startswith('RESULT ')]
    return rows[-1] if rows else None
