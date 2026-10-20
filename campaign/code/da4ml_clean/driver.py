#!/usr/bin/env python3
"""Clean-rerun driver for the PREFLIGHT da4ml test (module M19).

Implements STUDY v0.8 (`review/snapshots/v0.8/STUDY.md`) §3.4 Addendum A steps 2-7, §11 (mulder),
the §13 da4ml row and the interruption mechanics of §19.2 item 5. It calls the frozen
`run_case.sh` unchanged, by path, once per line of `cases_bc.txt`, and never edits a frozen file.
It applies no part of the headline rule; `table.py` builds the per-line table and gate G74 applies
the rule later.

What one attempt does, in order:
  start checks  the nine frozen scripts and env.sh hash to the expected table; env.sh, sourced in
                a bash subshell at the fixed path run_case.sh:5 reads, gives SCR = $HOME/scratch/
                cbl1016-da4ml-test; the da4ml version under PY is the route's; on a clone route the
                clone's pip freeze equals its lock.
  step 3        exclusivity lock under $SCR and a process scan; m3-env's pip freeze hashed;
                $SCR/proj/main and $SCR/logs renamed into $SCR/archive/<run>-aNN/ and listed with
                sha256 (nothing deleted); empty proj/main and logs created, tmp created if absent,
                numba_cache kept and listed; on a clone route env.sh renamed byte-identical into
                the archive and replaced by a copy that differs only in its PY line, whose sha256
                is recorded before the first line.
  step 4        the four hgq4 context lines first; after the first, build_<line>.log and
                proj/main/<line> must exist. A failed per-line assertion here, or the driver check,
                is a fault (exit 3); a context line that did not complete a stage halts (exit 6)
                until it is rerun or reviewed. Neither is a void.
  rule lines    the 40 stock/fact lines in file order, one attempt each. Per-line assertions before
                each line (no proj/main/<line>; env.sh hash of the branch; da4ml version under PY;
                frozen scripts' hashes; run_case.sh stdout recorded to a new file) and after it
                (no "skip" printed; stdout recorded; frozen scripts' and env.sh hashes again).
                A failure applies --on-assertion: void-restart voids the attempt and restarts from
                step 3; fail-line fails that line and continues.
  end           env.sh restored and rechecked (clone route), m3-env's pip freeze hashed again and
                diffed, lock moved into the archive, table written.

Each line runs in its own session, so its own process group (sid = pgid = pid of run_case.sh).
GNU timeout in run_case.sh:12 and synth_one.sh:8,12 moves the build and the tools into new
process groups inside that session, so the driver stops the whole session, not only the leader's
group. On SIGTERM, SIGINT, SIGHUP or a driver error the exit path stops and waits for that session
(TERM to every member, then KILL after --grace-seconds, until none is left), then restores and
rechecks env.sh, then writes `interrupted` with the stage the line was in into its sidecar. A
process that calls setsid leaves the session and is not covered. SIGKILL cannot be trapped: a
restarted driver finds the in-flight line in run.json, stops its session, restores and rechecks
env.sh, finalises the sidecar, and then applies --on-interrupt:
fail-line (option (i)) resumes from the next unstarted line, re-establishing exclusivity and the
branch's env.sh, without retrying the stopped line; void-restart (option (ii)) voids the attempt
and restarts from step 3. An interruption before any rule line started is a step-4 fault.

Kai's two item-5 answers have no default. --mode real refuses to start, before writing anything,
unless --on-assertion, --on-interrupt, --decision (the recorded answer), --equivalence-record
(G71) and the route's inputs are given; it refuses --expected-hashes and --home. --mode dry-run
needs --home (a temporary root, not the account's home) and refuses stand-ins that are the frozen
scripts themselves.

Exit codes: 0 attempt complete; 2 refused; 3 fault before any rule line started (step 4) or a
failed start check; 4 attempt void and not restarted (--max-voids reached); 5 interrupted;
6 context halt.

Standard library only; written for Python 3.6 or later; run with any python3, never with $PY.

    python3 driver.py --mode real --evidence DIR --route m3-env --da4ml-version 0.5.2 \\
        --on-assertion {void-restart,fail-line} --on-interrupt {fail-line,void-restart} \\
        --decision FILE --equivalence-record FILE [--max-voids N] [--grace-seconds S]
    python3 driver.py ... --route clone --clone-python PATH --clone-lock FILE ...
"""

import argparse
import difflib
import hashlib
import json
import os
import platform
import pwd
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import table as T  # noqa: E402

SCHEMA_RUN = 'cbl1016-da4ml-clean-run/1'
SCHEMA_SIDECAR = 'cbl1016-da4ml-clean-sidecar/1'
FIXED_SCR = os.path.join('scratch', 'cbl1016-da4ml-test')    # run_case.sh:5, synth_one.sh:5, env.sh:2
ENV_VARS = ('SCR', 'PY', 'TMPDIR', 'NUMBA_CACHE_DIR', 'KERAS_BACKEND')
VERSION_DISTS = ('hls4ml', 'da4ml', 'keras', 'hgq2', 'tensorflow', 'numpy')
LOCK_NAME = 'cbl1016-clean-rerun.lock'
EXIT_OK, EXIT_REFUSED, EXIT_FAULT, EXIT_VOID, EXIT_INTERRUPTED, EXIT_CONTEXT_HALT = 0, 2, 3, 4, 5, 6
POLICIES = ('void-restart', 'fail-line')
# Project files copied into the evidence after each line (same relative layout as the project).
ARTIFACTS = ('case_info.json', 'STATUS', 'synth_result.json', 'csim_io.npz', 'build_prj.tcl', 'build_opt.tcl',
             'csynth_stdout.log', 'myproject_prj/solution1/syn/report/csynth.xml',
             'vivado/post_synth_util.rpt', 'vivado/post_synth_timing.rpt', 'vivado/post_opt_util.rpt',
             'vivado/post_opt_timing.rpt', 'vivado/vivado.log', 'vivado/vivado_stdout.log')
VERSION_PROBE = r'''
import json, sys
out = {"python": sys.version.split()[0], "executable": sys.executable, "prefix": sys.prefix}
try:
    import importlib.metadata as md
except Exception:
    md = None
for d in %r:
    try:
        out[d] = md.version(d) if md else None
    except Exception:
        out[d] = None
print(json.dumps(out))
''' % (VERSION_DISTS,)


# ------------------------------------------------------------------------------------ basics

class Refused(Exception):
    pass


class Interrupted(Exception):
    pass


class Stop(object):
    signal_name = None


STOP = Stop()


def _on_signal(signum, _frame):
    if STOP.signal_name is None:
        try:
            STOP.signal_name = signal.Signals(signum).name
        except Exception:  # noqa: BLE001
            STOP.signal_name = 'signal %d' % signum


def now():
    t = time.time()
    return time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(t)) + '.%03dZ' % int((t % 1) * 1000)


def account_home():
    return pwd.getpwuid(os.getuid()).pw_dir


def real(p):
    return os.path.realpath(p)


def is_within(path, root):
    path, root = real(path), real(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def run_capture(argv, env=None, timeout=300, cwd=None):
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                         env=env, cwd=cwd)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        return -9, out, err
    return p.returncode, out, err


def listing(root):
    """[(sha256 or 'symlink' or 'dir', size, relpath)] of every entry under root, sorted."""
    rows = []
    for dp, dns, fns in os.walk(root, followlinks=False):
        dns.sort()
        for n in sorted(fns) + [d for d in dns if os.path.islink(os.path.join(dp, d))]:
            p = os.path.join(dp, n)
            rel = os.path.relpath(p, root)
            if os.path.islink(p):
                rows.append(('symlink', '-', '%s -> %s' % (rel, os.readlink(p))))
            elif os.path.isfile(p):
                rows.append((T.sha256_file(p), str(os.path.getsize(p)), rel))
        if not fns and not dns and dp != root:
            rows.append(('dir', '0', os.path.relpath(dp, root) + '/'))
    return sorted(rows, key=lambda r: r[2])


def write_listing(rows, path):
    with open(path, 'w') as f:
        f.write('# sha256\tsize\tpath\n')
        for r in rows:
            f.write('\t'.join(r) + '\n')
    return T.sha256_file(path)


def session_members(sid):
    """[(pid, pgid)] of the live (non-zombie) processes whose session id is `sid`. A line runs in
    its own session; GNU timeout in run_case.sh:12 and synth_one.sh:8,12 moves the build and the
    tools into new process groups inside it, so the session, not the leader's group, is the unit
    that holds every process of the line. Linux: /proc/<pid>/stat; elsewhere: ps and getsid."""
    res = []
    if os.path.exists('/proc/self/stat'):
        for n in os.listdir('/proc'):
            if not n.isdigit():
                continue
            try:
                with open('/proc/%s/stat' % n) as f:
                    rest = f.read().rsplit(')', 1)[1].split()
            except (IOError, OSError, IndexError):
                continue
            if len(rest) > 3 and rest[0] != 'Z' and int(rest[3]) == sid:
                res.append((int(n), int(rest[2])))
        return res
    rc, out, _ = run_capture(['ps', '-A', '-o', 'pid=', '-o', 'pgid=', '-o', 'stat='], timeout=60)
    for line in out.decode('utf-8', 'replace').splitlines():
        f = line.split()
        if len(f) < 3 or f[2].startswith('Z'):
            continue
        try:
            if os.getsid(int(f[0])) == sid:
                res.append((int(f[0]), int(f[1])))
        except (ProcessLookupError, PermissionError, OSError, ValueError):
            continue
    return res


def _signal_members(sid, sig, members):
    try:
        os.killpg(sid, sig)
    except (ProcessLookupError, PermissionError):
        pass
    for pid, _ in members:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def stop_session(sid, proc, grace):
    """Stop and wait for every process of the line's session `sid` (= the pgid and pid of its
    run_case.sh): SIGTERM to every member, SIGKILL after `grace` seconds to every member still
    there (repeated for members that appear meanwhile), until none is left. Returns the record:
    each round's signal, pids and process groups, and whether the session is empty."""
    rec = OrderedDict([('sid', sid), ('pgid', sid), ('start', now()), ('rounds', []), ('signals', []),
                       ('gone', None), ('waited_s', None)])
    t0 = time.time()

    def alive():
        if proc is not None:
            proc.poll()
        return session_members(sid)

    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 60.0)):
        m = alive()
        if not m:
            break
        rec['rounds'].append(OrderedDict([('signal', sig.name), ('time', now()), ('pids', sorted(p for p, _ in m)),
                                          ('pgids', sorted(set(g for _, g in m)))]))
        rec['signals'].append(sig.name)
        _signal_members(sid, sig, m)
        t1 = time.time()
        while time.time() - t1 < wait:
            m = alive()
            if not m:
                break
            if sig == signal.SIGKILL:
                _signal_members(sid, sig, m)
            time.sleep(0.1)
    if proc is not None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    rec['gone'] = not alive()
    rec['waited_s'] = round(time.time() - t0, 3)
    rec['end'] = now()
    return rec


# -------------------------------------------------------------------------------- arguments

def parse_args(argv):
    ap = argparse.ArgumentParser(description='Clean-rerun driver (M19); see the module docstring.')
    ap.add_argument('--mode', choices=('real', 'dry-run'), required=True)
    ap.add_argument('--evidence', help='evidence root for this route version (run.json, sidecars, reports)')
    ap.add_argument('--route', choices=('m3-env', 'clone'), required=True)
    ap.add_argument('--da4ml-version', help='the route version; asserted under PY before every line')
    ap.add_argument('--on-assertion', choices=POLICIES, default=None,
                    help="Kai's item-5 answer for a failed per-line assertion or a skip; no default")
    ap.add_argument('--on-interrupt', choices=('fail-line', 'void-restart'), default=None,
                    help="Kai's item-5 answer for an interruption, (i) fail-line or (ii) void-restart; no default")
    ap.add_argument('--decision', help='path to the recorded item-5 decision (required in real mode)')
    ap.add_argument('--equivalence-record', help='path to the G71 record (required in real mode)')
    ap.add_argument('--clone-python', help='clone route: the clone venv interpreter')
    ap.add_argument('--clone-lock', help="clone route: the clone's locked package list")
    ap.add_argument('--max-voids', type=int, default=2,
                    help='stop instead of restarting once the run record holds this many void attempts')
    ap.add_argument('--grace-seconds', type=float, default=30.0, help='SIGTERM to SIGKILL delay for a line group')
    ap.add_argument('--context-reviewed', help='note file: a halted context stage was reviewed as a stage outcome')
    ap.add_argument('--home', help='dry run only: the temporary root that stands in for $HOME')
    ap.add_argument('--expected-hashes', help='dry run only: JSON of the stand-ins\' sha256')
    return ap.parse_args(argv)


def _nonempty_file(p):
    return bool(p) and os.path.isfile(p) and os.path.getsize(p) > 0


def preconditions(a):
    """Argument-level refusals. Reads only; writes nothing. Returns the list of reasons."""
    why = []
    if a.on_assertion is None:
        why.append("--on-assertion is required: Kai's item-5 answer has no default")
    if a.on_interrupt is None:
        why.append("--on-interrupt is required: Kai's item-5 answer has no default")
    if not a.da4ml_version:
        why.append('--da4ml-version is required')
    if not a.evidence:
        why.append('--evidence is required')
    if a.route == 'clone':
        if not a.clone_python or not os.path.isfile(a.clone_python or ''):
            why.append('--route clone needs --clone-python, an existing interpreter')
        elif not re.match(r'^[A-Za-z0-9_./+-]+$', a.clone_python):
            why.append('--clone-python must be a plain path (it is written into the env.sh copy)')
        if not _nonempty_file(a.clone_lock):
            why.append('--route clone needs --clone-lock, a non-empty file')
    elif a.clone_python or a.clone_lock:
        why.append('--clone-python and --clone-lock belong to --route clone only')
    if a.max_voids < 1:
        why.append('--max-voids must be at least 1')
    if a.grace_seconds <= 0:
        why.append('--grace-seconds must be positive')
    if a.context_reviewed and not _nonempty_file(a.context_reviewed):
        why.append('--context-reviewed must be a non-empty file')
    if a.mode == 'real':
        if not a.decision:
            why.append('real mode needs --decision, the path to the recorded item-5 decision')
        elif not _nonempty_file(a.decision):
            why.append('--decision %s is not a non-empty file' % a.decision)
        if not _nonempty_file(a.equivalence_record):
            why.append('real mode needs --equivalence-record, the G71 record, as a non-empty file')
        if a.expected_hashes:
            why.append('--expected-hashes is for a dry run only; real mode uses the embedded STUDY table')
        if a.home:
            why.append('--home is for a dry run only; real mode uses $HOME, as run_case.sh does')
        elif os.environ.get('HOME') and real(os.environ['HOME']) != real(account_home()):
            why.append('real mode: $HOME is not the account home directory')
    else:
        if not a.home:
            why.append('a dry run needs --home, a temporary root')
        elif real(a.home) == real(account_home()) or is_within(account_home(), a.home):
            why.append('a dry run never uses the account home directory as --home')
        if a.decision and not _nonempty_file(a.decision):
            why.append('--decision %s is not a non-empty file' % a.decision)
    return why


# ------------------------------------------------------------------------------------ driver

class Driver(object):
    def __init__(self, a):
        self.a = a
        self.dry = a.mode == 'dry-run'
        self.home = real(a.home) if self.dry else real(os.environ.get('HOME') or account_home())
        self.scripts = os.path.join(self.home, FIXED_SCR, 'scripts')
        self.envsh = os.path.join(self.scripts, 'env.sh')
        self.evidence = os.path.abspath(a.evidence)
        self.run_path = os.path.join(self.evidence, 'run.json')
        self.expected, self.expected_source = T.load_expected(a.expected_hashes if self.dry else None)
        self.run = None
        self.att = None
        self.inflight = None       # {'case', 'sc', 'sc_path', 'proc', 'pgid', 'stdout'}
        self.host = socket.gethostname()
        self.me = os.path.abspath(__file__)
        self.me_sha = T.sha256_file(self.me)
        self.table_sha = T.sha256_file(os.path.join(HERE, 'table.py'))
        self.cases = None
        self.child_env = dict(os.environ)
        self.child_env['HOME'] = self.home
        self.envsh_expected = None   # the branch's expected env.sh sha256 for the current attempt
        self.py = None
        self.scr = os.path.join(self.home, FIXED_SCR)   # env.sh's SCR is asserted equal to this path

    # ---------------------------------------------------------------------------- records

    def log(self, msg):
        line = '%s %s\n' % (now(), msg)
        with open(os.path.join(self.evidence, 'driver.log'), 'a') as f:
            f.write(line)

    def save(self):
        T.dump_json(self.run, self.run_path)

    def adir(self, att=None):
        return os.path.join(self.evidence, 'attempts', 'a%02d' % (att or self.att)['id'])

    # ---------------------------------------------------------------------- environment

    def source_env(self):
        """SCR, PY, TMPDIR, NUMBA_CACHE_DIR, KERAS_BACKEND as env.sh sets them, sourced at the
        fixed path in a bash subshell with this HOME."""
        code = ('set -u; source "$1" >/dev/null 2>&1 || exit 97; '
                'for v in %s; do printf "%%s=%%s\\0" "$v" "${!v-}"; done' % ' '.join(ENV_VARS))
        env = {'HOME': self.home, 'PATH': os.environ.get('PATH', '/usr/bin:/bin')}
        rc, out, err = run_capture(['bash', '-c', code, 'source-env', self.envsh], env=env, timeout=60)
        if rc != 0:
            raise Refused('sourcing %s failed (rc=%s)' % (self.envsh, rc))
        vals = OrderedDict()
        for item in out.split(b'\0'):
            if item:
                k, _, v = item.decode('utf-8', 'replace').partition('=')
                vals[k] = v
        return vals

    def probe_versions(self, py):
        rc, out, err = run_capture([py, '-I', '-B', '-c', VERSION_PROBE], env=self._py_env(), timeout=120)
        try:
            return json.loads(out.decode('utf-8', 'replace').strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            return {'error': 'version probe rc=%s: %s' % (rc, err.decode('utf-8', 'replace')[-300:])}

    def _py_env(self):
        env = dict(self.child_env)
        env.update({'PYTHONDONTWRITEBYTECODE': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1', 'PIP_NO_INPUT': '1'})
        return env

    def pip_freeze(self, py):
        """(sha256 of the freeze output, path of the stored list) for interpreter `py`."""
        rc, out, err = run_capture([py, '-I', '-B', '-m', 'pip', 'freeze', '--disable-pip-version-check'],
                                   env=self._py_env(), timeout=300)
        if rc != 0:
            return None, 'pip freeze rc=%s: %s' % (rc, err.decode('utf-8', 'replace')[-200:])
        sha = hashlib.sha256(out).hexdigest()
        fdir = os.path.join(self.evidence, 'freeze')
        if not os.path.isdir(fdir):
            os.makedirs(fdir)
        p = os.path.join(fdir, sha + '.txt')
        if not os.path.exists(p):
            with open(p, 'wb') as f:
                f.write(out)
        return sha, os.path.relpath(p, self.evidence)

    def hash_scripts(self):
        return OrderedDict((f, T.sha256_or_none(os.path.join(self.scripts, f))) for f in T.HASHED_FILES)

    def frozen_mismatch(self, hashes):
        return [f for f in T.FROZEN_SHA256 if hashes.get(f) != self.expected[f]]

    def scan_foreign(self, exclude_pgids=()):
        """Processes, other than this driver and the listed groups, whose command line names
        $SCR or the test's scratch name (ps -A). Recorded; the start of step 3 refuses on a hit."""
        rc, out, err = run_capture(['ps', '-A', '-o', 'pid=', '-o', 'pgid=', '-o', 'command='], timeout=60)
        if rc != 0:
            return {'available': False, 'error': err.decode('utf-8', 'replace')[-200:]}
        needles = [n for n in (self.scr, os.path.join(self.home, FIXED_SCR)) if n]
        hits = []
        for line in out.decode('utf-8', 'replace').splitlines():
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            pid, pgid, cmd = int(parts[0]), int(parts[1]), parts[2]
            if pid == os.getpid() or pgid in exclude_pgids or 'ps -A -o' in cmd:
                continue
            if any(n in cmd for n in needles):
                hits.append({'pid': pid, 'pgid': pgid, 'command': cmd[:240]})
        return {'available': True, 'time': now(), 'hits': hits}

    # ------------------------------------------------------------------------- run record

    def open_record(self):
        exists = os.path.isfile(self.run_path)
        if not exists:
            return None
        run = T.load_json(self.run_path)
        if not isinstance(run, dict) or run.get('schema') != SCHEMA_RUN:
            raise Refused('%s is not a run record of this driver' % self.run_path)
        want = OrderedDict([('dry_run', self.dry), ('route', self.a.route), ('da4ml_version', self.a.da4ml_version),
                            ('policies', {'on_assertion': self.a.on_assertion, 'on_interrupt': self.a.on_interrupt}),
                            ('home', self.home)])
        for k, v in want.items():
            if run.get(k) != v:
                raise Refused('run record %s holds %s=%r; this invocation has %r' % (self.run_path, k, run.get(k), v))
        if self.a.decision and (run.get('decision') or {}).get('sha256') != T.sha256_file(self.a.decision):
            raise Refused('the decision file differs from the one recorded at the start of this run')
        if self.a.route == 'clone' and (run.get('clone') or {}).get('python') != self.a.clone_python:
            raise Refused('--clone-python differs from the run record')
        return run

    def new_record(self):
        os.makedirs(self.evidence)
        run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-%d' % os.getpid()
        rec = OrderedDict([
            ('schema', SCHEMA_RUN), ('run_id', run_id), ('dry_run', self.dry), ('created', now()),
            ('home', self.home), ('scripts', self.scripts), ('route', self.a.route),
            ('da4ml_version', self.a.da4ml_version),
            ('policies', {'on_assertion': self.a.on_assertion, 'on_interrupt': self.a.on_interrupt}),
            ('decision', {'path': self.a.decision, 'sha256': T.sha256_file(self.a.decision)} if self.a.decision else None),
            ('equivalence_record', {'path': self.a.equivalence_record, 'sha256': T.sha256_file(self.a.equivalence_record)}
             if self.a.equivalence_record else None),
            ('clone', {'python': self.a.clone_python, 'lock': self.a.clone_lock,
                       'lock_sha256': T.sha256_file(self.a.clone_lock)} if self.a.route == 'clone' else None),
            ('expected_hashes', self.expected), ('expected_hash_source', self.expected_source),
            ('m3env_python', None), ('invocations', []), ('attempts', []),
        ])
        self.run = rec
        sdir = os.path.join(self.evidence, 'scripts')
        os.makedirs(sdir)
        for f in T.HASHED_FILES:
            with open(os.path.join(self.scripts, f), 'rb') as src, open(os.path.join(sdir, f), 'wb') as dst:
                dst.write(src.read())
        self.save()
        self.log('run record created: %s (dry_run=%s, route=%s, da4ml=%s, policies=%s)'
                 % (run_id, self.dry, self.a.route, self.a.da4ml_version, json.dumps(rec['policies'])))

    # ----------------------------------------------------------------------------- entry

    def main(self):
        run = self.open_record()
        last = (run or {}).get('attempts', [])[-1:] or [None]
        last = last[0]
        if last is not None and last.get('status') == 'complete':
            raise Refused('the run record already holds a complete attempt (%s): one clean rerun per route '
                          'version; a new version or route needs a new evidence root' % last['id'])
        if run is None:
            self.start_checks(fresh=True)        # refuses before any write
            self.new_record()
        else:
            self.run = run
        inv = OrderedDict([('n', len(self.run['invocations']) + 1), ('start', now()), ('pid', os.getpid()),
                           ('host', self.host), ('python', sys.version.split()[0]), ('argv', sys.argv),
                           ('driver_sha256', self.me_sha), ('table_sha256', self.table_sha),
                           ('max_voids', self.a.max_voids), ('grace_seconds', self.a.grace_seconds),
                           ('child_env_names', sorted(self.child_env)), ('exit', None)])
        self.run['invocations'].append(inv)
        self.save()
        self.log('invocation %d start: pid %d host %s driver %s' % (inv['n'], os.getpid(), self.host, self.me_sha))
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, _on_signal)
        code = None
        try:
            self.recover()
            code = self.loop()
        except Refused as e:
            self.log('refused: %s' % e)
            print('refused: %s' % e, file=sys.stderr)
            self.abort('refused: %s' % e)
            code = EXIT_REFUSED
        except Interrupted as e:
            code = self.emergency('interrupted by %s' % e)
        except BaseException as e:  # noqa: BLE001
            tb = traceback.format_exc()
            self.log('driver error: %s' % tb)
            code = self.emergency('driver error: %s: %s' % (type(e).__name__, e))
        inv['exit'] = code
        inv['end'] = now()
        self.save()
        self.log('invocation %d end: exit %s' % (inv['n'], code))
        return code

    def check_stop(self):
        if STOP.signal_name:
            raise Interrupted(STOP.signal_name)

    # ------------------------------------------------------------------------ start checks

    def start_checks(self, fresh):
        """Before step 3 (and before the evidence exists on a fresh run): frozen hashes, env.sh,
        SCR, PY's da4ml version, the clone lock. Raises Refused on a fresh run, or returns a list
        of failed checks for an attempt that is already recorded."""
        problems = []
        if not os.path.isdir(self.scripts):
            raise Refused('no scripts directory at %s' % self.scripts)
        h = self.hash_scripts()
        if self.dry:
            real_hits = [f for f in ('run_case.sh', 'synth_one.sh', 'stage_b_build.py')
                         if h.get(f) == T.FROZEN_SHA256[f]]
            if real_hits:
                raise Refused('dry run: %s in %s are the frozen scripts, not stand-ins' % (real_hits, self.scripts))
        mism = self.frozen_mismatch(h)
        if mism:
            problems.append('frozen script sha256 mismatch: %s' % ', '.join(mism))
        if h['env.sh'] != self.expected['env.sh']:
            problems.append('env.sh sha256 %s is not the original %s' % (h['env.sh'], self.expected['env.sh']))
        if not problems:
            env = self.source_env()
            scr_want = os.path.join(self.home, FIXED_SCR)
            if real(env.get('SCR', '')) != real(scr_want):
                problems.append('env.sh SCR=%r is not %s' % (env.get('SCR'), scr_want))
            self.scr = env.get('SCR')
            m3 = env.get('PY')
            if not m3 or not os.path.isfile(m3):
                problems.append('env.sh PY=%r is not a file' % m3)
            else:
                py = self.a.clone_python if self.a.route == 'clone' else m3
                v = self.probe_versions(py)
                if self.a.route == 'clone':
                    # venv interpreters are symlinks to one base interpreter: compare the venvs' prefixes
                    vm = self.probe_versions(m3)
                    if not v.get('prefix') or real(v.get('prefix')) == real(vm.get('prefix') or ''):
                        problems.append('the clone interpreter runs in the m3-env prefix %s' % vm.get('prefix'))
                if v.get('da4ml') != self.a.da4ml_version:
                    problems.append('da4ml under %s is %r, not %s' % (py, v.get('da4ml'), self.a.da4ml_version))
                if self.a.route == 'clone' and not problems:
                    rc, out, err = run_capture([py, '-I', '-B', '-m', 'pip', 'freeze', '--disable-pip-version-check'],
                                               env=self._py_env(), timeout=300)
                    want = sorted(l.strip() for l in open(self.a.clone_lock) if l.strip() and not l.startswith('#'))
                    have = sorted(l.strip() for l in out.decode('utf-8', 'replace').splitlines() if l.strip())
                    if rc != 0 or want != have:
                        problems.append("the clone's pip freeze differs from --clone-lock (%d lines against %d)"
                                        % (len(have), len(want)))
        lock = os.path.join(self.home, FIXED_SCR, LOCK_NAME)
        if os.path.exists(lock):
            other = T.load_json(lock) or {}
            if not self.run or other.get('run_id') != self.run.get('run_id'):
                problems.append('an exclusivity lock of another job exists: %s' % lock)
        if fresh and problems:
            raise Refused('; '.join(problems))
        return problems

    # ------------------------------------------------------------------------- recovery

    def recover(self):
        """After a driver that ended without its exit path (SIGKILL, power): stop the in-flight
        line's group, then restore and recheck env.sh, then finalise the line's sidecar."""
        for att in self.run['attempts']:
            swap = att.get('env_swap') or {}
            if att.get('status') == 'running' or swap.get('active'):
                self.att = att
                inflight = att.get('in_flight')
                stop = None
                if inflight:
                    pg = inflight.get('pgid')
                    if pg:
                        stop = stop_session(pg, None, self.a.grace_seconds)
                    else:
                        stop = self.stop_by_name(inflight['name'])
                    self.log('restart: stopped the session of in-flight line %s: %s' % (inflight['name'], json.dumps(stop)))
                restore = self.restore_envsh(att, 'restart')
                if inflight:
                    case = self.case_by_name(inflight['name'])
                    sc_path = os.path.join(self.adir(att), 'sidecars', case['name'] + '.provenance.json')
                    sc = T.load_json(sc_path) or self.sidecar_skeleton(att, case)
                    self.finalise_interrupted(att, case, sc, sc_path, 'driver ended without its exit path '
                                              '(detected at restart)', stop, 'restart', restore)
                if att.get('status') == 'running':
                    att['status'] = 'interrupted'
                    att['cause'] = 'driver ended without its exit path (detected at restart)'
                    att['in_flight'] = None
                    att['end'] = now()
                    self.release_lock(att, 'stale lock after a driver that ended without its exit path')
                    self.freeze_after(att)
                    self.save()
                    self.safe_table(att)

    def stop_by_name(self, name):
        """In-flight line without a recorded session id: stop the sessions of processes whose
        command line names its project."""
        res = self.scan_foreign()
        sids = set()
        for h in res.get('hits', []):
            if name in h['command']:
                try:
                    sids.add(os.getsid(h['pid']))
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        return {'by_name': name, 'sessions': [stop_session(x, None, self.a.grace_seconds) for x in sorted(sids)]}

    def case_by_name(self, name):
        for c in self.load_cases():
            if c['name'] == name:
                return c
        raise Refused('line %s is not in cases_bc.txt' % name)

    def load_cases(self):
        if self.cases is None:
            p = os.path.join(self.evidence, 'scripts', 'cases_bc.txt')
            if T.sha256_or_none(p) != self.expected['cases_bc.txt']:
                raise Refused('evidence copy of cases_bc.txt does not have the expected sha256')
            self.cases = T.parse_cases(T.read_text(p))
        return self.cases

    # ----------------------------------------------------------------------------- loop

    def voids(self):
        return sum(1 for x in self.run['attempts'] if x.get('status') == 'void')

    def loop(self):
        while True:
            self.check_stop()
            att = self.run['attempts'][-1] if self.run['attempts'] else None
            st = att.get('status') if att else None
            if st in ('fault', 'interrupted') and att.get('phase') == 'step3':
                att['status'] = 'abandoned'
                att['abandoned'] = now()
                self.save()
                self.log('attempt %d abandoned: stopped inside step 3 (%s); a new attempt starts from step 3'
                         % (att['id'], att.get('cause')))
                continue
            if att is None or st in ('void', 'abandoned'):
                if st == 'void' and self.voids() >= self.a.max_voids:
                    self.log('void attempts %d reached --max-voids %d: not restarting' % (self.voids(), self.a.max_voids))
                    return EXIT_VOID
                att = self.new_attempt()
                r = self.drive(att, 'step3')
            elif st == 'complete':
                raise Refused('attempt %d is complete' % att['id'])
            elif st == 'interrupted' and att.get('phase') == 'rule':
                if self.a.on_interrupt == 'void-restart':
                    self.void(att, 'interrupted after a rule line had started (option (ii)): %s' % att.get('cause'))
                    continue
                r = self.drive(att, 'resume')
            elif st == 'fault' and att.get('phase') == 'start':
                r = self.drive(att, 'step3')
            elif st == 'context-halt' and self.a.context_reviewed:
                att.setdefault('context_reviews', []).append(
                    {'time': now(), 'path': self.a.context_reviewed, 'sha256': T.sha256_file(self.a.context_reviewed)})
                self.save()
                self.log('context halt reviewed (%s): proceeding to the rule lines' % self.a.context_reviewed)
                r = self.drive(att, 'rule')
            elif st in ('fault', 'context-halt', 'interrupted'):
                r = self.drive(att, 'context-rerun')
            else:
                raise Refused('attempt %d has status %r; nothing to do' % (att['id'], st))
            if r == 'void':
                continue
            return r

    def new_attempt(self):
        n = len(self.run['attempts']) + 1
        att = OrderedDict([('id', n), ('status', 'running'), ('phase', 'start'), ('cause', None), ('start', now()),
                           ('end', None), ('archive_dir', None), ('started', []), ('in_flight', None),
                           ('context_runs', 0), ('resumes', []), ('env_swap', None), ('exclusivity', []),
                           ('driver_sha256_at_rule_start', None), ('envsh_expected', None)])
        self.run['attempts'].append(att)
        self.att = att
        for sub in ('sidecars', 'stdout', 'reports', 'results'):
            os.makedirs(os.path.join(self.adir(att), sub))
        self.save()
        self.log('attempt %d created' % n)
        return att

    def void(self, att, cause):
        att['status'] = 'void'
        att['cause'] = cause
        att['voided'] = now()
        self.save()
        self.log('attempt %d VOID: %s' % (att['id'], cause))
        self.safe_table(att)

    # ---------------------------------------------------------------------------- attempt

    def drive(self, att, start):
        self.att = att
        problems = self.start_checks(fresh=False)
        if problems:
            msg = 'start checks failed: ' + '; '.join(problems)
            if start == 'step3':
                att['phase'] = 'start'
                return self.end_attempt(att, 'fault', msg, EXIT_FAULT)
            att.setdefault('blocked', []).append({'time': now(), 'start': start, 'problems': problems})
            self.save()
            self.log('attempt %d not continued (%s): %s' % (att['id'], start, msg))
            return EXIT_FAULT
        att['status'] = 'running'
        att['cause'] = None
        self.save()
        if start == 'step3':
            self.step3(att)
        else:
            self.reestablish(att, start)
        if start in ('step3', 'context-rerun'):
            r = self.context_lines(att)
            if r is not None:
                return r
        att['phase'] = 'rule'
        if att['driver_sha256_at_rule_start'] is None:
            att['driver_sha256_at_rule_start'] = self.me_sha
            att['table_sha256_at_rule_start'] = self.table_sha
        elif att['driver_sha256_at_rule_start'] != self.me_sha:
            return self.end_attempt(att, 'interrupted', 'driver changed after a rule line had started (%s, recorded %s)'
                                    % (self.me_sha, att['driver_sha256_at_rule_start']), EXIT_REFUSED)
        self.save()
        for case in self.load_cases():
            if case['role'] != 'rule' or case['name'] in att['started']:
                continue
            res = self.run_line(att, case, rule=True)
            if res['failed']:
                if self.a.on_assertion == 'void-restart':
                    self.end_attempt(att, 'void', 'per-line assertion failed at %s: %s'
                                     % (case['name'], ', '.join(res['failed'])), None)
                    att['voided'] = now()
                    self.save()
                    self.log('attempt %d VOID at %s' % (att['id'], case['name']))
                    return 'void'
                self.log('line %s failed (fail-line remedy): %s' % (case['name'], ', '.join(res['failed'])))
        return self.end_attempt(att, 'complete', None, EXIT_OK)

    def step3(self, att):
        """Archive by rename, recreate, keep numba_cache, swap env.sh on a clone route."""
        att['phase'] = 'step3'
        self.take_lock(att, 'step 3')
        env = self.source_env()
        self.scr = env['SCR']
        self.run['m3env_python'] = self.run.get('m3env_python') or env['PY']
        sha, where = self.pip_freeze(self.run['m3env_python'])
        att['m3env_freeze_before'] = {'sha256': sha, 'list': where, 'time': now()}
        scr = self.scr
        arch = os.path.join(scr, 'archive', '%s-a%02d' % (self.run['run_id'], att['id']))
        os.makedirs(arch)
        att['archive_dir'] = arch
        moved = []
        for src, dst in ((os.path.join(scr, 'proj', 'main'), 'proj_main'), (os.path.join(scr, 'logs'), 'logs')):
            if os.path.lexists(src):
                ino = os.lstat(src).st_ino
                os.rename(src, os.path.join(arch, dst))
                moved.append({'from': src, 'to': os.path.join(arch, dst), 'inode': ino,
                              'inode_after': os.lstat(os.path.join(arch, dst)).st_ino})
        rows = listing(arch)
        lsha = write_listing(rows, os.path.join(self.adir(att), 'archive-listing.tsv'))
        made = []
        for d in (os.path.join(scr, 'proj', 'main'), os.path.join(scr, 'logs')):
            os.makedirs(d)
            made.append(d)
        tmp_existed = os.path.isdir(os.path.join(scr, 'tmp'))
        if not tmp_existed:
            os.makedirs(os.path.join(scr, 'tmp'))
        nc = os.path.join(scr, 'numba_cache')
        nrows = listing(nc) if os.path.isdir(nc) else []
        nsha = write_listing(nrows, os.path.join(self.adir(att), 'numba-cache-listing.tsv'))
        att['step3'] = OrderedDict([('time', now()), ('moved', moved), ('archive_listing', {
            'path': 'archive-listing.tsv', 'sha256': lsha, 'entries': len(rows)}), ('created', made),
            ('tmp_existed', tmp_existed), ('tmp_created', not tmp_existed),
            ('numba_cache', {'present': os.path.isdir(nc), 'listing': 'numba-cache-listing.tsv', 'sha256': nsha,
                             'entries': len(nrows)})])
        self.save()
        self.log('step 3: archived %s into %s (%d entries listed), recreated proj/main and logs, tmp %s'
                 % ([m['from'] for m in moved], arch, len(rows), 'kept' if tmp_existed else 'created'))
        self.establish_envsh(att)

    def reestablish(self, att, why):
        """Resume (option (i)), a step-4 context rerun or a reviewed context halt: exclusivity and
        the branch's env.sh again; the directories of step 3 are not archived again."""
        self.take_lock(att, why)
        env = self.source_env()
        self.scr = env['SCR']
        sha, where = self.pip_freeze(self.run['m3env_python'])
        att.setdefault('m3env_freeze_reestablished', []).append({'sha256': sha, 'list': where, 'time': now()})
        for d in (os.path.join(self.scr, 'proj', 'main'), os.path.join(self.scr, 'logs'), os.path.join(self.scr, 'tmp')):
            if not os.path.isdir(d):
                os.makedirs(d)
                att.setdefault('recreated_on_reestablish', []).append(d)
        self.establish_envsh(att)
        if why == 'resume':
            stopped = [x for x in att['started'] if self.sidecar_outcome(att, x) == 'interrupted']
            nxt = [c['name'] for c in self.load_cases() if c['role'] == 'rule' and c['name'] not in att['started']]
            att['resumes'].append(OrderedDict([('time', now()), ('option', '(i) fail-line'), ('not_retried', stopped),
                                               ('exempt_from_dir_absent', stopped),
                                               ('next_line', nxt[0] if nxt else None)]))
            self.log('resume under option (i): stopped line(s) %s not retried, next line %s'
                     % (stopped, nxt[0] if nxt else None))
        self.save()

    def sidecar_outcome(self, att, name):
        sc = T.load_json(os.path.join(self.adir(att), 'sidecars', name + '.provenance.json')) or {}
        return sc.get('outcome')

    def establish_envsh(self, att):
        if self.a.route != 'clone':
            att['envsh_expected'] = self.expected['env.sh']
            self.envsh_expected = att['envsh_expected']
            env = self.source_env()
            self.py = env['PY']
            self.save()
            return
        orig = self.envsh
        data = open(orig, 'rb').read()
        if hashlib.sha256(data).hexdigest() != self.expected['env.sh']:
            raise Refused('env.sh is not the original before the swap')
        lines = data.split(b'\n')
        idx = [i for i, l in enumerate(lines) if l.startswith(b'export PY=')]
        if len(idx) != 1:
            raise Refused('env.sh has %d PY lines, expected 1' % len(idx))
        new = list(lines)
        new[idx[0]] = b'export PY=' + self.a.clone_python.encode()
        copy = b'\n'.join(new)
        diff = [i for i in range(len(lines)) if lines[i] != new[i]]
        if len(new) != len(lines) or diff != idx:
            raise Refused('the env.sh copy would differ in more than the PY line')
        arch_orig = os.path.join(att['archive_dir'], 'env.sh.original')
        swap = att.get('env_swap') or OrderedDict([('swaps', [])])
        swap['active'] = True
        swap['original_sha256'] = self.expected['env.sh']
        swap['original_inode'] = os.lstat(orig).st_ino
        swap['archive_path'] = arch_orig
        att['env_swap'] = swap
        self.save()                                   # record before the first rename
        os.rename(orig, arch_orig)
        mode = os.lstat(arch_orig).st_mode & 0o777
        fd = os.open(orig, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(fd, 'wb') as f:
            f.write(copy)
        csha = T.sha256_file(orig)
        if swap.get('copy_sha256') and swap['copy_sha256'] != csha:
            raise Refused('the env.sh copy hashes to %s, not the recorded %s' % (csha, swap['copy_sha256']))
        swap['copy_sha256'] = csha
        swap['py_line'] = idx[0] + 1
        swap['diff'] = list(difflib.unified_diff([x.decode() for x in lines], [x.decode() for x in new],
                                                 'env.sh', 'env.sh (copy)', lineterm='', n=0))
        swap['swaps'].append({'time': now(), 'copy_sha256': csha, 'original_moved_to': arch_orig})
        att['envsh_expected'] = csha
        self.envsh_expected = csha
        self.save()
        env = self.source_env()
        self.py = env['PY']
        self.log('env.sh swapped for the clone route: original (inode %s) moved to %s; copy sha256 %s, PY line %d'
                 % (swap['original_inode'], arch_orig, csha, swap['py_line']))

    def restore_envsh(self, att, when):
        swap = att.get('env_swap') or {}
        if not swap.get('active'):
            h = T.sha256_or_none(self.envsh)
            rec = {'time': now(), 'when': when, 'swap': False, 'env_sh_sha256': h,
                   'ok': h == (att.get('envsh_expected') or self.expected['env.sh'])}
            att.setdefault('env_rechecks', []).append(rec)
            return rec
        orig, arch_orig = self.envsh, swap['archive_path']
        moved_copy = None
        if os.path.exists(arch_orig):
            if os.path.lexists(orig):
                moved_copy = os.path.join(att['archive_dir'], 'env.sh.clone-copy-%s' % re.sub(r'[^0-9A-Za-z]', '', now()))
                os.rename(orig, moved_copy)
            os.rename(arch_orig, orig)
        h = T.sha256_or_none(orig)
        ok = h == swap['original_sha256']
        ino = os.lstat(orig).st_ino if os.path.lexists(orig) else None
        swap['active'] = False
        rec = OrderedDict([('time', now()), ('when', when), ('swap', True), ('copy_moved_to', moved_copy),
                           ('env_sh_sha256', h), ('ok', ok), ('inode', ino),
                           ('inode_equals_original', ino == swap.get('original_inode'))])
        swap.setdefault('restores', []).append(rec)
        att.setdefault('env_rechecks', []).append(rec)
        self.save()
        self.log('env.sh restored (%s): sha256 %s %s' % (when, h, 'OK' if ok else 'MISMATCH'))
        return rec

    def take_lock(self, att, why):
        scr = os.path.join(self.home, FIXED_SCR)
        p = os.path.join(scr, LOCK_NAME)
        scan = self.scan_foreign()
        if scan.get('hits'):
            raise Refused('exclusivity: other processes name $SCR: %s' % json.dumps(scan['hits'])[:600])
        body = {'run_id': self.run['run_id'], 'attempt': att['id'], 'host': self.host, 'pid': os.getpid(),
                'start': now(), 'why': why}
        takeover = None
        if os.path.exists(p):
            other = T.load_json(p) or {}
            if other.get('run_id') != self.run['run_id']:
                raise Refused('exclusivity lock of another job at %s' % p)
            takeover = other
            os.rename(p, os.path.join(att['archive_dir'] or scr, 'stale-%s' % LOCK_NAME))
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, 'w') as f:
            json.dump(body, f)
        att['exclusivity'].append(OrderedDict([('start', body['start']), ('why', why), ('lock', p),
                                               ('scan_at_start', scan), ('stale_lock_taken_over', takeover),
                                               ('end', None)]))
        self.save()

    def release_lock(self, att, why):
        p = os.path.join(self.home, FIXED_SCR, LOCK_NAME)
        ex = att['exclusivity'][-1] if att.get('exclusivity') else None
        if ex is None or ex.get('end'):
            return
        ex['scan_at_end'] = self.scan_foreign()
        if os.path.exists(p):
            dst = os.path.join(att['archive_dir'] or os.path.dirname(p), 'exclusivity-%d-%s.lock'
                               % (len(att['exclusivity']), re.sub(r'[^0-9A-Za-z]', '', now())))
            os.rename(p, dst)
            ex['lock_moved_to'] = dst
        ex['end'] = now()
        ex['end_why'] = why
        self.save()

    def freeze_after(self, att):
        if not self.run.get('m3env_python') or not att.get('m3env_freeze_before'):
            return
        sha, where = self.pip_freeze(self.run['m3env_python'])
        before = att['m3env_freeze_before']
        diff = []
        if sha and before.get('sha256') and sha != before['sha256']:
            a_ = open(os.path.join(self.evidence, before['list'])).read().splitlines()
            b_ = open(os.path.join(self.evidence, where)).read().splitlines()
            diff = list(difflib.unified_diff(a_, b_, 'before', 'after', lineterm='', n=0))
        att.setdefault('m3env_freeze_after', []).append(
            {'sha256': sha, 'list': where, 'time': now(), 'equal_to_before': sha == before.get('sha256'), 'diff': diff})
        self.save()

    def end_attempt(self, att, status, cause, code):
        rec = self.restore_envsh(att, 'end of attempt (%s)' % status)
        self.freeze_after(att)
        self.release_lock(att, status)
        att['status'] = status
        att['cause'] = cause
        att['end'] = now()
        att['in_flight'] = None
        self.save()
        self.log('attempt %d %s%s; env.sh recheck %s' % (att['id'], status, (': ' + cause) if cause else '',
                                                        'OK' if rec.get('ok') else 'MISMATCH'))
        self.safe_table(att)
        return code

    def safe_table(self, att):
        try:
            T.write_table(self.evidence, att['id'], self.expected, self.expected_source)
        except Exception as e:  # noqa: BLE001
            self.log('table for attempt %d not written: %s' % (att['id'], e))

    # ------------------------------------------------------------------------ context lines

    def context_lines(self, att):
        att['phase'] = 'context'
        cases = [c for c in self.load_cases() if c['role'] == 'context']
        if att['context_runs'] > 0 or any(c['name'] in att['started'] for c in cases):
            self.archive_context(att, cases)
        att['context_runs'] += 1
        self.save()
        for i, case in enumerate(cases):
            res = self.run_line(att, case, rule=False)
            if res['failed']:
                return self.end_attempt(att, 'fault', 'step 4: per-line assertion failed at context line %s: %s'
                                        % (case['name'], ', '.join(res['failed'])), EXIT_FAULT)
            if i == 0:
                chk = OrderedDict([('time', now()), ('line', case['name']),
                                   ('build_log', os.path.isfile(os.path.join(self.scr, 'logs', 'build_%s.log' % case['name']))),
                                   ('proj_dir', os.path.isdir(os.path.join(self.scr, 'proj', 'main', case['name'])))])
                chk['ok'] = chk['build_log'] and chk['proj_dir']
                att.setdefault('driver_checks', []).append(chk)
                self.save()
                if not chk['ok']:
                    return self.end_attempt(att, 'fault', 'step 4 driver check after %s: build log %s, project directory %s'
                                            % (case['name'], 'present' if chk['build_log'] else 'MISSING',
                                               'present' if chk['proj_dir'] else 'MISSING'), EXIT_FAULT)
        try:
            T.write_table(self.evidence, att['id'], self.expected, self.expected_source)
        except Exception as e:  # noqa: BLE001
            return self.end_attempt(att, 'context-halt', 'the table of the context lines could not be built: %s: %s'
                                    % (type(e).__name__, e), EXIT_CONTEXT_HALT)
        tab = T.load_json(os.path.join(self.adir(att), 'results', 'table.json'))
        incomplete = [r['name'] for r in tab['rows'] if r['role'] == 'context' and r['failed_at']]
        att.setdefault('context_completeness', []).append({'time': now(), 'incomplete': incomplete})
        self.save()
        if incomplete and not self.a.context_reviewed:
            return self.end_attempt(att, 'context-halt', 'context line(s) without a result at some stage: %s; '
                                    'rerun after a fix, or pass --context-reviewed if it is a stage outcome'
                                    % ', '.join(incomplete), EXIT_CONTEXT_HALT)
        if incomplete:
            att.setdefault('context_reviews', []).append(
                {'time': now(), 'path': self.a.context_reviewed, 'sha256': T.sha256_file(self.a.context_reviewed),
                 'incomplete': incomplete})
            self.save()
        return None

    def archive_context(self, att, cases):
        """Step 4 rerun: move the context lines' project directories, build logs and parse.log
        into the archive, and their evidence aside, before the four lines run again."""
        if any(c['name'] in att['started'] for c in self.load_cases() if c['role'] == 'rule'):
            raise Refused('a rule line has started; the context lines cannot be rerun in this attempt')
        k = att['context_runs']
        dst = os.path.join(att['archive_dir'], 'context-run-%d' % k)
        os.makedirs(os.path.join(dst, 'proj_main'))
        os.makedirs(os.path.join(dst, 'logs'))
        moved = []
        for c in cases:
            for src, to in ((os.path.join(self.scr, 'proj', 'main', c['name']), os.path.join(dst, 'proj_main', c['name'])),
                            (os.path.join(self.scr, 'logs', 'build_%s.log' % c['name']),
                             os.path.join(dst, 'logs', 'build_%s.log' % c['name']))):
                if os.path.lexists(src):
                    os.rename(src, to)
                    moved.append(src)
        pl = os.path.join(self.scr, 'logs', 'parse.log')
        if os.path.lexists(pl):
            os.rename(pl, os.path.join(dst, 'logs', 'parse.log'))
            moved.append(pl)
        lsha = write_listing(listing(dst), os.path.join(self.adir(att), 'context-run-%d-archive-listing.tsv' % k))
        ev = os.path.join(self.adir(att), 'context-run-%d' % k)
        for sub in ('sidecars', 'stdout', 'reports'):
            os.makedirs(os.path.join(ev, sub))
            for c in cases:
                for n in os.listdir(os.path.join(self.adir(att), sub)):
                    if n == c['name'] or n.startswith(c['name'] + '.'):
                        os.rename(os.path.join(self.adir(att), sub, n), os.path.join(ev, sub, n))
        att['started'] = [x for x in att['started'] if x not in [c['name'] for c in cases]]
        att.setdefault('context_archives', []).append({'time': now(), 'run': k, 'archive': dst, 'moved': moved,
                                                       'listing_sha256': lsha, 'evidence_moved_to': ev})
        self.save()
        self.log('step 4 rerun: context run %d archived into %s (%d paths)' % (k, dst, len(moved)))

    # ----------------------------------------------------------------------------- lines

    def sidecar_skeleton(self, att, case):
        return OrderedDict([
            ('schema', SCHEMA_SIDECAR), ('dry_run', self.dry), ('run_id', self.run['run_id']), ('attempt', att['id']),
            ('line_no', case['line_no']), ('name', case['name']), ('role', case['role']), ('pair', case['pair']),
            ('case', OrderedDict((k, case[k]) for k in ('shape', 'g', 'inp', 'path', 'bias_field'))),
            ('host', self.host), ('driver_sha256', self.me_sha), ('route', self.a.route), ('start', now()),
            ('end', None), ('outcome', 'started'), ('interrupted', None), ('assertions', []),
            ('assertion_failures', []), ('policy_applied', None)])

    def write_sidecar(self, sc, path):
        T.dump_json(sc, path)

    def run_line(self, att, case, rule):
        self.check_stop()
        name = case['name']
        proj = os.path.join(self.scr, 'proj', 'main', name)
        sc_path = os.path.join(self.adir(att), 'sidecars', name + '.provenance.json')
        out_path = os.path.join(self.adir(att), 'stdout', name + '.stdout')
        err_path = os.path.join(self.adir(att), 'stdout', name + '.stderr')
        sc = self.sidecar_skeleton(att, case)
        att['started'].append(name)                    # one attempt per line, recorded first
        att['in_flight'] = {'name': name, 'pgid': None, 'since': sc['start'], 'state': 'pre-checks'}
        self.save()
        self.inflight = {'case': case, 'sc': sc, 'sc_path': sc_path, 'proc': None, 'pgid': None}
        self.write_sidecar(sc, sc_path)

        def asrt(key, ok, detail):
            sc['assertions'].append(OrderedDict([('name', key), ('ok', bool(ok)), ('detail', detail), ('time', now())]))

        hb = self.hash_scripts()
        sc['hashes_before'] = hb
        asrt('proj_dir_absent', not os.path.lexists(proj), proj)
        asrt('env_sh_sha256', hb['env.sh'] == self.envsh_expected, {'have': hb['env.sh'], 'want': self.envsh_expected})
        mism = self.frozen_mismatch(hb)
        asrt('frozen_scripts_sha256_before', not mism, mism)
        env = self.source_env() if hb['env.sh'] == self.envsh_expected else None
        sc['env'] = env
        py = (env or {}).get('PY') or self.py
        vers = self.probe_versions(py)
        sc['versions'] = vers
        sc['KERAS_BACKEND'] = (env or {}).get('KERAS_BACKEND')
        asrt('da4ml_version', vers.get('da4ml') == self.a.da4ml_version, {'have': vers.get('da4ml'),
                                                                          'want': self.a.da4ml_version, 'PY': py})
        fsha, flist = self.pip_freeze(py)
        sc['pip_freeze_sha256'] = fsha
        sc['pip_freeze_list'] = flist
        sc['foreign_processes_before'] = self.scan_foreign()
        try:
            fo = open(out_path, 'xb')
            fe = open(err_path, 'xb')
            asrt('stdout_recorded_file_created', True, os.path.relpath(out_path, self.evidence))
        except (IOError, OSError) as e:
            fo = fe = None
            asrt('stdout_recorded_file_created', False, str(e))
        sc['provenance_checks'] = OrderedDict([('scripts_before_ok', not mism),
                                               ('env_sh_before_ok', hb['env.sh'] == self.envsh_expected)])
        failed = [x['name'] for x in sc['assertions'] if not x['ok']]
        if failed:
            for fh in (fo, fe):
                if fh:
                    fh.close()
            sc['outcome'] = 'not_run_assertion_failed'
            sc['assertion_failures'] = failed
            sc['policy_applied'] = self.policy_for(rule)
            sc['end'] = now()
            self.write_sidecar(sc, sc_path)
            att['in_flight'] = None
            self.inflight = None
            self.save()
            self.log('line %s not run: assertion(s) failed before the line: %s' % (name, failed))
            return {'failed': failed}
        self.check_stop()
        argv = ['bash', os.path.join(self.scripts, 'run_case.sh'), case['shape'], case['g'], case['inp'],
                case['path'], case['bias_field']]
        sc['run_case_argv'] = argv
        att['in_flight']['state'] = 'spawning'
        self.save()
        cwd = os.path.join(self.scr, 'tmp')
        proc = subprocess.Popen(argv, stdout=fo, stderr=fe, stdin=subprocess.DEVNULL, cwd=cwd, env=self.child_env,
                                start_new_session=True)
        fo.close()
        fe.close()
        pgid = proc.pid
        self.inflight.update({'proc': proc, 'pgid': pgid})
        att['in_flight'].update({'pgid': pgid, 'state': 'running', 'spawned': now()})
        sc['pgid'] = pgid
        sc['spawned'] = now()
        sc['cwd'] = cwd
        sc['outcome'] = 'running'
        self.save()
        self.write_sidecar(sc, sc_path)
        while proc.poll() is None:
            self.check_stop()
            time.sleep(0.1)
        sc['run_case_rc'] = proc.returncode
        sc['returned'] = now()
        proc.poll()
        sweep = stop_session(pgid, proc, self.a.grace_seconds) if session_members(pgid) else None
        sc['survivors_after_return'] = sweep
        if sweep:
            self.log('line %s: session %d still had members after run_case.sh returned: %s'
                     % (name, pgid, json.dumps(sweep)))
        att['in_flight'] = None
        self.inflight = None
        self.save()
        ha = self.hash_scripts()
        sc['hashes_after'] = ha
        stdout_text = T.read_text(out_path) or ''
        status_text = T.read_text(os.path.join(proj, 'STATUS')) or ''
        rc = T.rc_reports(name, stdout_text, status_text)
        sc['reports'] = rc
        mism_a = self.frozen_mismatch(ha)
        asrt('no_skip', not rc['skip'], 'skip printed' if rc['skip'] else None)
        asrt('stdout_recorded', rc['skip'] or rc['build_rc'] is not None,
             {'bytes': len(stdout_text.encode()), 'sha256': T.sha256_or_none(out_path)})
        asrt('frozen_scripts_sha256_after', not mism_a, mism_a)
        asrt('env_sh_sha256_after', ha['env.sh'] == self.envsh_expected, {'have': ha['env.sh'], 'want': self.envsh_expected})
        sc['provenance_checks'].update([('scripts_after_ok', not mism_a), ('env_sh_after_ok', ha['env.sh'] == self.envsh_expected)])
        sc['artifacts'] = self.copy_artifacts(att, name, proj)
        sc['foreign_processes_after'] = self.scan_foreign()
        sc['end'] = now()
        sc['outcome'] = 'returned'
        failed = [x['name'] for x in sc['assertions'] if not x['ok']]
        if failed:
            sc['assertion_failures'] = failed
            sc['policy_applied'] = self.policy_for(rule)
        self.write_sidecar(sc, sc_path)
        self.log('line %s returned rc=%s build=%s csynth=%s vivado=%s%s' % (
            name, proc.returncode, rc['build_rc'], rc['csynth_rc'], rc['vivado_rc'],
            ('; assertion(s) failed: %s' % failed) if failed else ''))
        return {'failed': failed}

    def policy_for(self, rule):
        if not rule:
            return 'step-4 fault'
        return 'void' if self.a.on_assertion == 'void-restart' else 'fail-line'

    def copy_artifacts(self, att, name, proj):
        dst = os.path.join(self.adir(att), 'reports', name)
        got = OrderedDict()
        for rel in ARTIFACTS:
            src = os.path.join(proj, rel)
            if os.path.isfile(src):
                to = os.path.join(dst, rel)
                if not os.path.isdir(os.path.dirname(to)):
                    os.makedirs(os.path.dirname(to))
                with open(src, 'rb') as a_, open(to, 'wb') as b_:
                    b_.write(a_.read())
                got[rel] = T.sha256_file(to)
        bl = os.path.join(self.scr, 'logs', 'build_%s.log' % name)
        if os.path.isfile(bl):
            to = os.path.join(dst, 'build.log')
            if not os.path.isdir(dst):
                os.makedirs(dst)
            with open(bl, 'rb') as a_, open(to, 'wb') as b_:
                b_.write(a_.read())
            got['build.log'] = T.sha256_file(to)
        return got

    # ----------------------------------------------------------------------- interruption

    def finalise_interrupted(self, att, case, sc, sc_path, cause, stop, detected_by, restore):
        name = case['name']
        proj = os.path.join(self.scr or os.path.join(self.home, FIXED_SCR), 'proj', 'main', name)
        out_path = os.path.join(self.adir(att), 'stdout', name + '.stdout')
        stdout_text = T.read_text(out_path) or ''
        status_text = T.read_text(os.path.join(proj, 'STATUS')) or ''
        rc = T.rc_reports(name, stdout_text, status_text)
        stage = T.stage_in_progress(rc)
        sc['reports'] = rc
        sc['artifacts'] = self.copy_artifacts(att, name, proj)
        sc['interrupted'] = OrderedDict([('stage', stage), ('cause', cause), ('time', now()),
                                         ('detected_by', detected_by), ('group_stop', stop),
                                         ('env_sh_restore', restore)])
        sc['outcome'] = 'interrupted'
        sc['end'] = now()
        try:
            sc['hashes_after'] = self.hash_scripts()
        except Exception:  # noqa: BLE001
            pass
        self.write_sidecar(sc, sc_path)
        self.log('line %s INTERRUPTED in %s (%s)' % (name, stage, cause))

    def abort(self, cause):
        """A refusal raised while an attempt was running (never while a line is in flight):
        restore and recheck env.sh, release the lock, and record the attempt as interrupted in the
        rule phase or as a fault before it."""
        att = self.att
        if att is None or att.get('status') != 'running':
            return
        if self.inflight is not None:
            self.emergency(cause)
            return
        st = 'interrupted' if att.get('phase') == 'rule' else 'fault'
        self.end_attempt(att, st, cause, None)

    def emergency(self, cause):
        """The exit path: stop and wait for the line's group, restore and recheck env.sh, write
        `interrupted` with the stage into the line's sidecar, record the attempt."""
        att = self.att
        fl = self.inflight
        stop = None
        if fl is not None and fl.get('pgid'):
            stop = stop_session(fl['pgid'], fl.get('proc'), self.a.grace_seconds)
            self.log('exit path: session %s stopped: %s' % (fl['pgid'], json.dumps(stop)))
        restore = None
        if att is not None:
            try:
                restore = self.restore_envsh(att, 'exit path')
            except Exception as e:  # noqa: BLE001
                self.log('exit path: env.sh restore failed: %s' % e)
        if fl is not None and att is not None:
            self.finalise_interrupted(att, fl['case'], fl['sc'], fl['sc_path'], cause, stop, 'exit trap', restore)
        if att is not None and att.get('status') == 'running':
            att['status'] = 'interrupted'
            att['cause'] = cause
            att['interrupted_line'] = fl['case']['name'] if fl else None
            att['in_flight'] = None
            att['end'] = now()
            self.release_lock(att, 'interrupted')
            self.freeze_after(att)
            self.save()
            self.safe_table(att)
        self.inflight = None
        self.log('exit path complete: %s' % cause)
        return EXIT_INTERRUPTED


def main(argv=None):
    a = parse_args(sys.argv[1:] if argv is None else argv)
    why = preconditions(a)
    if why:
        for w in why:
            print('refused: %s' % w, file=sys.stderr)
        return EXIT_REFUSED
    try:
        d = Driver(a)
    except (ValueError, IOError, OSError) as e:
        print('refused: %s' % e, file=sys.stderr)
        return EXIT_REFUSED
    try:
        return d.main()
    except Refused as e:
        print('refused: %s' % e, file=sys.stderr)
        return EXIT_REFUSED


if __name__ == '__main__':
    sys.exit(main())
