"""Gate G72 (PREFLIGHT-PLAN section 3 row G72; STUDY v0.8 §3.4 Addendum A, §11, §13 da4ml row,
§19.2 item 5; module M19): dry run of the clean-rerun driver and table script on the Mac.

Every run here is against stand-in scripts (`tests/da4ml_clean_stubs/`) in temporary roots, with
`HOME` set to the root so the stand-ins keep the frozen scripts' fixed path
`$HOME/scratch/cbl1016-da4ml-test/scripts/env.sh`. The stand-ins call `timeout` where the frozen
scripts do (`run_case.sh:12`, `synth_one.sh:8,12`); the stand-in `timeout` moves itself and its
command into a new process group, as GNU timeout does, so a line's session holds several groups. The frozen scripts are never executed: the
gate only hashes them and reads `parse_reports.py` as a syntax tree. Nothing runs on mulder.

Checks:
1. static: the driver's embedded hash table equals STUDY v0.8 lines 60-72 (and the snapshot and
   the headline-rule block hash to their recorded digests); the frozen scripts on disk hash to
   that table (read only); the frozen `parse_reports.py` defines csynth(p), util(f), timing(f),
   the functions the table loads (AST only); the driver and table import the standard library
   only and parse under the Python 3.7 grammar;
2. unit tests of the table logic (`tests/test_da4ml_clean_table.py`);
3. refusals: real mode without either item-5 answer, without the decision record, without the
   G71 record, with dry-run options; dry run without either answer, against the account home,
   or against copies of the frozen scripts; each refused with exit 2 and nothing written;
4. scenarios (fresh root each): a clean run on each route; every per-line assertion, under both
   remedies; SIGTERM mid-line with a child that ignores SIGTERM, under all four combinations of
   the two answers and both routes; SIGKILL of the driver and recovery on restart, under both
   interruption options; the step-4 faults (driver check, an assertion before any rule line, a
   context halt and its review, an interruption during a context line); the void cap; a frozen
   hash mismatch at start. Each checks the run record, the sidecars, the archive and the 44-row
   table.

Every number written by a stand-in is synthetic; every reading is engineering, not quotable.

    <python> -I -B gates/g72_da4ml_clean_dryrun.py [--expect-sha12 SHA12] [--keep DIR] [--only NAME] [--jobs N]
        [--keep-temp]

`--keep DIR` copies every scenario root except its two stand-in venvs to DIR.

Prints one line per check and ends with "G72 PASS" (exit 0) or "G72 FAIL" (exit 1).
"""
import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gatelib  # noqa: E402

GATE = 'G72'
FORK = gatelib.FORK
DRIVER = FORK / 'da4ml_clean' / 'driver.py'
TABLE = FORK / 'da4ml_clean' / 'table.py'
STUBS = FORK / 'tests' / 'da4ml_clean_stubs'
UNIT = FORK / 'tests' / 'test_da4ml_clean_table.py'
CAMPAIGN = Path('/Users/kaiyamaguchi/lab/bnjettag/campaigns/2026-10-16-chang-bitlinear')
STUDY = CAMPAIGN / 'review' / 'snapshots' / 'v0.8' / 'STUDY.md'
STUDY_SHA = '290022fa41a726b8ae66ff302f7910a69c4c148ec27b15c2448cab83fb898ca8'
RULE_BLOCK_SHA = 'd9f2e386d04710e1a424ad5f4511ea830f14b17c1e86afb15fa8959da7168dbb'
FROZEN_DIR = CAMPAIGN / 'preflight' / 'da4ml-test' / 'scripts'
FIXED = Path('scratch') / 'cbl1016-da4ml-test'

sys.path.insert(0, str(FORK / 'da4ml_clean'))
import table as T  # noqa: E402

CASES = T.parse_cases((STUBS / 'scripts' / 'cases_bc.txt').read_text())
NAME = {c['line_no']: c['name'] for c in CASES}
CONTEXT = [c['name'] for c in CASES if c['role'] == 'context']
RULE = [c['name'] for c in CASES if c['role'] == 'rule']
LINE_NO = {c['name']: c['line_no'] for c in CASES}
HEX64 = re.compile(r'^[0-9a-f]{64}$')
DISTS = [('hls4ml', '1.3.0'), ('keras', '3.15.0'), ('hgq2', '0.1.9'), ('tensorflow', '0.0.0+standin'),
         ('numpy', '0.0.0+standin')]


# ---------------------------------------------------------------------------- utilities

class Result:
    def __init__(self, name):
        self.name = name
        self.rows = []
        self.info = {}

    def check(self, ok, what):
        self.rows.append((bool(ok), '%s: %s' % (self.name, what)))
        return bool(ok)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(root):
    """{relpath: sha256 or 'dir'} of a tree, to show that a refusal wrote nothing."""
    out = {}
    root = Path(root)
    if not root.exists():
        return out
    for p in sorted(root.rglob('*')):
        rel = str(p.relative_to(root))
        if p.is_symlink():
            out[rel] = 'link:' + os.readlink(p)
        elif p.is_dir():
            out[rel] = 'dir'
        else:
            out[rel] = sha(p)
    return out


def tree_listing(root):
    """Same rows as the driver's archive listing (sha256, size, relpath), for files only."""
    rows = []
    for p in sorted(Path(root).rglob('*')):
        if p.is_file() and not p.is_symlink():
            rows.append((sha(p), str(p.stat().st_size), str(p.relative_to(root))))
    return sorted(rows, key=lambda r: r[2])


def read_listing(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        a, b, c = line.split('\t')
        if a != 'dir':
            rows.append((a, b, c))
    return rows


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie still answers kill(0); ask ps for its state
    r = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], stdout=subprocess.PIPE, text=True)
    return r.returncode == 0 and r.stdout.strip() != '' and not r.stdout.strip().startswith('Z')


def session_members(sid):
    """[(pid, pgid)] of the live (non-zombie) processes in session `sid`. The stand-in `timeout`
    moves the build and the tools into their own process groups, as GNU timeout does, so a line
    is gone only when its whole session is empty."""
    out = subprocess.run(['ps', '-A', '-o', 'pid=,pgid=,stat='], stdout=subprocess.PIPE, text=True).stdout
    res = []
    for line in out.splitlines():
        f = line.split()
        if len(f) < 3 or f[2].startswith('Z'):
            continue
        try:
            if os.getsid(int(f[0])) == sid:
                res.append((int(f[0]), int(f[1])))
        except (ProcessLookupError, PermissionError, OSError):
            continue
    return res


def session_alive(sid):
    return bool(session_members(sid))


def wait_for(path, timeout, proc=None):
    """True once `path` exists; False on timeout or if `proc` exits first."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if Path(path).exists():
            return True
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(0.05)
    return False


def keys_in(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from keys_in(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from keys_in(v)


def make_venv(path, da4ml_version):
    subprocess.run([sys.executable, '-m', 'venv', str(path)], check=True, stdout=subprocess.DEVNULL)
    py = path / 'bin' / 'python'
    pure = Path(subprocess.run([str(py), '-I', '-c', 'import sysconfig; print(sysconfig.get_paths()["purelib"])'],
                               check=True, stdout=subprocess.PIPE, text=True).stdout.strip())
    for name, ver in DISTS + [('da4ml', da4ml_version)]:
        d = pure / ('%s-%s.dist-info' % (name, ver))
        d.mkdir()
        (d / 'METADATA').write_text('Metadata-Version: 2.1\nName: %s\nVersion: %s\nSummary: G72 stand-in\n' % (name, ver))
        (d / 'INSTALLER').write_text('g72\n')
        (d / 'RECORD').write_text('')
    (pure / 'da4ml').mkdir()
    (pure / 'da4ml' / '__init__.py').write_text('__version__ = %r  # G72 stand-in\n' % da4ml_version)
    return py


# --------------------------------------------------------------------------------- roots

class Root:
    """One temporary root: HOME with the stand-in scratch tree, an old run to archive, a
    numba cache, two stand-in venvs (m3-env with da4ml 0.5.2, a clone with 0.6.0)."""

    def __init__(self, base, label):
        self.label = label
        self.root = Path(base) / label
        self.home = self.root / 'home'
        self.scr = self.home / FIXED
        self.scripts = self.scr / 'scripts'
        self.evidence = self.root / 'evidence'
        self.expected = self.root / 'expected-hashes.json'
        self.decision = self.root / 'decision-stand-in.txt'
        self.clone_lock = self.root / 'clone.lock'

    def setup(self):
        self.root.mkdir(parents=True)
        (self.root / 'tmp').mkdir()
        shutil.copytree(STUBS / 'scripts', self.scripts)
        shutil.copytree(STUBS / 'tools', self.home / 'stub-tools')
        shutil.copytree(STUBS / 'xilinx', self.home / 'stub-xilinx')
        (self.home / 'stub-faults').mkdir()
        (self.home / 'stub-markers').mkdir()
        self.m3 = make_venv(self.home / 'venvs' / 'm3-env', '0.5.2')
        self.clone = make_venv(self.home / 'venvs' / 'clone', '0.6.0')
        freeze = subprocess.run([str(self.clone), '-I', '-B', '-m', 'pip', 'freeze', '--disable-pip-version-check'],
                                check=True, stdout=subprocess.PIPE, text=True).stdout
        self.clone_lock.write_text('# G72 stand-in clone lock\n' + freeze)
        exp = {f: sha(self.scripts / f) for f in T.HASHED_FILES}
        self.expected.write_text(json.dumps(exp, indent=1))
        self.exp = exp
        self.decision.write_text('G72 dry run: stand-in for the recorded item-5 decision; not a decision.\n')
        # an old run to archive: 42 project directories, logs, a numba cache, no tmp
        for c in CASES[:42]:
            d = self.scr / 'proj' / 'main' / c['name']
            d.mkdir(parents=True)
            (d / 'synth_result.json').write_text('{"stand_in": "old run, %s"}\n' % c['name'])
        (self.scr / 'logs').mkdir()
        for c in CASES[:42]:
            (self.scr / 'logs' / ('build_%s.log' % c['name'])).write_text('old run stand-in log\n')
        (self.scr / 'logs' / 'parse.log').write_text('old run stand-in parse log\n')
        (self.scr / 'numba_cache').mkdir()
        (self.scr / 'numba_cache' / 'index.nbi').write_bytes(b'stand-in numba index\n')
        (self.scr / 'numba_cache' / 'f.nbc').write_bytes(b'stand-in numba cache\n')
        self.pre = {
            'proj_main_inode': os.lstat(self.scr / 'proj' / 'main').st_ino,
            'logs_inode': os.lstat(self.scr / 'logs').st_ino,
            'env_sh_inode': os.lstat(self.scripts / 'env.sh').st_ino,
            'old_listing': [(a, b, 'proj_main/' + c) for a, b, c in tree_listing(self.scr / 'proj' / 'main')]
            + [(a, b, 'logs/' + c) for a, b, c in tree_listing(self.scr / 'logs')],
            'numba': tree_listing(self.scr / 'numba_cache'),
            'tmp_existed': (self.scr / 'tmp').exists(),
        }
        return self

    def fault(self, name, *lines):
        (self.home / 'stub-faults' / name).write_text(''.join(l + '\n' for l in lines))

    def env(self):
        return {'PATH': '%s:/usr/bin:/bin:/usr/sbin:/sbin' % (self.home / 'stub-tools'), 'HOME': str(self.home),
                'LANG': 'C', 'LC_ALL': 'C',
                'TMPDIR': str(self.root / 'tmp')}

    def argv(self, A, I, route='m3-env', extra=(), mode='dry-run'):
        a = [sys.executable, '-I', '-B', str(DRIVER), '--mode', mode, '--home', str(self.home),
             '--evidence', str(self.evidence), '--route', route,
             '--da4ml-version', '0.6.0' if route == 'clone' else '0.5.2',
             '--expected-hashes', str(self.expected), '--grace-seconds', '1', '--decision', str(self.decision)]
        if A:
            a += ['--on-assertion', A]
        if I:
            a += ['--on-interrupt', I]
        if route == 'clone':
            a += ['--clone-python', str(self.clone), '--clone-lock', str(self.clone_lock)]
        return a + list(extra)

    def run(self, A, I, route='m3-env', extra=(), timeout=900):
        p = subprocess.run(self.argv(A, I, route, extra), env=self.env(), stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr

    def popen(self, A, I, route='m3-env', extra=()):
        return subprocess.Popen(self.argv(A, I, route, extra), env=self.env(), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)

    def record(self):
        return json.loads((self.evidence / 'run.json').read_text())

    def adir(self, n):
        return self.evidence / 'attempts' / ('a%02d' % n)

    def sidecar(self, n, name):
        p = self.adir(n) / 'sidecars' / (name + '.provenance.json')
        return json.loads(p.read_text()) if p.exists() else None

    def table(self, n):
        return json.loads((self.adir(n) / 'results' / 'table.json').read_text())

    def marker(self, name, what):
        return self.home / 'stub-markers' / ('%s.%s' % (name, what))

    def restore_script(self, f):
        shutil.copyfile(STUBS / 'scripts' / f, self.scripts / f)

    def restore_da4ml(self, py):
        pure = Path(subprocess.run([str(py), '-I', '-c', 'import sysconfig; print(sysconfig.get_paths()["purelib"])'],
                                   check=True, stdout=subprocess.PIPE, text=True).stdout.strip())
        for m in pure.glob('da4ml-*.dist-info/METADATA'):
            ver = m.parent.name[len('da4ml-'):-len('.dist-info')]
            m.write_text('Metadata-Version: 2.1\nName: da4ml\nVersion: %s\nSummary: G72 stand-in\n' % ver)

    def kill_leftovers(self):
        """Safety net at the end: kill every process of any session a sidecar names."""
        for p in self.evidence.glob('attempts/*/sidecars/*.json'):
            try:
                pg = json.loads(p.read_text()).get('pgid')
            except ValueError:
                continue
            for pid, _ in (session_members(pg) if pg else []):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for p in (self.home / 'stub-markers').glob('*.survivor.pid'):
            try:
                os.kill(int(p.read_text()), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass


# ----------------------------------------------------------------------- shared checks

def check_table(r, root, n, expect_failed, expect_status=None, label=None):
    """44 rows in file order, 4 context and 20 pairs, the expected failed stage per line, stages
    before it completed and from it on failed, failed stages without values, no headline key,
    and the table's sha256 file."""
    tab = root.table(n)
    rows = tab['rows']
    h = tab['header']
    lab = label or 'a%02d table' % n
    r.check(len(rows) == 44 and [x['name'] for x in rows] == [c['name'] for c in CASES],
            '%s has 44 rows in cases_bc.txt order' % lab)
    r.check(sum(x['role'] == 'context' for x in rows) == 4
            and [x['name'] for x in rows if x['role'] == 'context'] == CONTEXT,
            '%s labels the four hgq4 lines context' % lab)
    r.check(len(set(x['pair'] for x in rows if x['role'] == 'rule')) == 20
            and all(x['pair'] is None for x in rows if x['role'] == 'context'), '%s has 20 pairs' % lab)
    bad = []
    for x in rows:
        want = expect_failed.get(x['name'])
        got = x['failed_at']
        stages = x['stages']
        cut = T.STAGES.index(want) if want else len(T.STAGES)
        shape_ok = all(stages[s] == ('completed' if i < cut else 'failed') for i, s in enumerate(T.STAGES))
        vals = x['values']
        vals_ok = True
        if stages['csim'] == 'failed' and vals['csim_mismatch'] is not None:
            vals_ok = False
        if stages['csynth'] == 'failed' and vals['II_max'] is not None:
            vals_ok = False
        if stages['vivado'] == 'failed' and (vals['post_opt_LUT'] is not None or vals['post_opt_WNS_ns'] is not None):
            vals_ok = False
        if stages['vivado'] == 'completed' and (vals['post_opt_LUT'] is None or vals['post_opt_WNS_ns'] is None):
            vals_ok = False
        if got != want or not shape_ok or not vals_ok:
            bad.append((x['line_no'], x['name'], want, got, x['fail_cause']))
    r.check(not bad, '%s: failed stages as expected (%d lines expected failed) %s'
            % (lab, len(expect_failed), json.dumps(bad[:4]) if bad else ''))
    forb = sorted(set(k for k in keys_in(tab) if k in T.FORBIDDEN_KEYS))
    r.check(not forb, '%s holds no headline, condition or term key %s' % (lab, forb if forb else ''))
    sums = (root.adir(n) / 'results' / 'table.sha256').read_text().split('\n')
    ok = True
    for line in sums:
        if line.strip():
            d, f = line.split('  ')
            ok = ok and sha(root.adir(n) / 'results' / f) == d
    r.check(ok and len([l for l in sums if l.strip()]) == 2, '%s: table.sha256 matches table.json and table.md' % lab)
    r.check(h['dry_run'] is True and 'DRY RUN' in h['note'], '%s header marks the dry run' % lab)
    if expect_status:
        r.check(h['attempt_status'] == expect_status, '%s header status %s' % (lab, h['attempt_status']))
    return tab


def check_sidecars(r, root, n, route='m3-env', copy_sha=None):
    """Every started line of attempt n has one sidecar with the step-5 fields."""
    rec = root.record()
    att = rec['attempts'][n - 1]
    want_env = copy_sha if route == 'clone' else root.exp['env.sh']
    r.check(len(set(att['started'])) == len(att['started']), 'a%02d: no line started twice (one attempt per line)' % n)
    probs = []
    for name in att['started']:
        sc = root.sidecar(n, name)
        if sc is None:
            probs.append((name, 'no sidecar'))
            continue
        for k in ('host', 'start', 'end', 'hashes_before', 'versions', 'pip_freeze_sha256', 'assertions'):
            if sc.get(k) in (None, '', []):
                probs.append((name, 'missing ' + k))
        hb = sc.get('hashes_before') or {}
        if any(f not in hb for f in T.SIDECAR_FILES):
            probs.append((name, 'hashes_before lacks a step-5 file'))
        if sc.get('outcome') == 'returned':
            for k in ('hashes_after', 'run_case_rc', 'reports', 'KERAS_BACKEND'):
                if sc.get(k) is None:
                    probs.append((name, 'missing ' + k))
            if sc['reports'].get('build_rc') is None and not sc['reports'].get('skip') and 'stdout_recorded' not in sc['assertion_failures']:
                probs.append((name, 'no build rc'))
        if sc.get('pip_freeze_sha256') and not HEX64.match(sc['pip_freeze_sha256']):
            probs.append((name, 'freeze hash'))
        if sc.get('dry_run') is not True:
            probs.append((name, 'dry_run flag'))
    r.check(not probs, 'a%02d: %d sidecars with hashes before/after of the step-5 files, versions, KERAS_BACKEND, '
                       'pip freeze sha256, host, times and return codes %s' % (n, len(att['started']),
                                                                              json.dumps(probs[:3]) if probs else ''))
    if att['started']:
        sc = root.sidecar(n, att['started'][0])
        r.check(sc['versions'].get('da4ml') == ('0.6.0' if route == 'clone' else '0.5.2')
                and sc['KERAS_BACKEND'] == 'tensorflow', 'a%02d: sidecars record da4ml %s and KERAS_BACKEND %s'
                % (n, sc['versions'].get('da4ml'), sc.get('KERAS_BACKEND')))
    clean = [x for x in att['started'] if (root.sidecar(n, x) or {}).get('outcome') == 'returned'
             and not (root.sidecar(n, x) or {}).get('assertion_failures')]
    envok = all((root.sidecar(n, x)['hashes_before']['env.sh'] == want_env and
                 root.sidecar(n, x)['hashes_after']['env.sh'] == want_env and
                 all(root.sidecar(n, x)['hashes_before'][f] == root.exp[f] for f in T.FROZEN_SHA256))
                for x in clean)
    r.check(envok, 'a%02d: frozen stand-ins equal the expected table and env.sh equals the branch hash before and '
                   'after each of %d clean lines' % (n, len(clean)))


def check_context_first(r, root, n):
    att = root.record()['attempts'][n - 1]
    first = att['started'][:4]
    starts = {x: root.sidecar(n, x)['start'] for x in att['started']}
    ctx_last = max(starts[x] for x in CONTEXT if x in starts)
    rule_first = min([starts[x] for x in RULE if x in starts] or ['9'])
    r.check(first == CONTEXT and ctx_last < rule_first,
            'a%02d: the four hgq4 lines ran first, in file order, before any rule line' % n)
    chk = (att.get('driver_checks') or [{}])[-1]
    r.check(chk.get('ok') is True and chk.get('line') == CONTEXT[0],
            'a%02d: driver check after %s found its build log and project directory' % (n, CONTEXT[0]))


def check_step3(r, root, n, prior=None):
    """Archive by rename with listing, recreated directories, tmp, numba_cache kept."""
    rec = root.record()
    att = rec['attempts'][n - 1]
    s3 = att['step3']
    arch = Path(att['archive_dir'])
    moved = {Path(m['from']).name: m for m in s3['moved']}
    if prior is None:
        r.check(moved['main']['inode'] == root.pre['proj_main_inode'] == os.lstat(arch / 'proj_main').st_ino
                and moved['logs']['inode'] == root.pre['logs_inode'] == os.lstat(arch / 'logs').st_ino,
                'a%02d: proj/main and logs moved into %s by rename (same inodes)' % (n, arch.name))
        lst = read_listing(root.adir(n) / 'archive-listing.tsv')
        r.check(lst == sorted(root.pre['old_listing'], key=lambda x: x[2]) and len(lst) == 85,
                'a%02d: archive listing with sha256 equals the old tree before the run (%d files)' % (n, len(lst)))
        r.check(tree_listing(arch / 'proj_main') == tree_listing(arch / 'proj_main')
                and all(sha_ == sha(arch / rel) for sha_, _, rel in lst), 'a%02d: archived files unchanged, nothing deleted' % n)
        r.check(s3['tmp_created'] is True and not root.pre['tmp_existed'] and (root.scr / 'tmp').is_dir(),
                'a%02d: tmp was absent and was created' % n)
    else:
        r.check('main' in moved and 'logs' in moved and (arch / 'proj_main').is_dir(),
                'a%02d: the earlier attempt\'s proj/main and logs moved into %s' % (n, arch.name))
    nlst = read_listing(root.adir(n) / 'numba-cache-listing.tsv')
    r.check(nlst == root.pre['numba'] == tree_listing(root.scr / 'numba_cache'),
            'a%02d: numba_cache kept in place and listed (%d files)' % (n, len(nlst)))
    old = [p for p in (root.scr / 'proj' / 'main').glob('*/synth_result.json') if 'old run' in p.read_text()]
    r.check(not old, 'a%02d: recreated proj/main holds no file of the archived run' % n)


def check_exclusivity(r, root, n):
    att = root.record()['attempts'][n - 1]
    ex = att['exclusivity']
    ok = bool(ex) and all(e.get('start') and e.get('end') and e.get('lock_moved_to') for e in ex)
    ok = ok and not (root.scr / 'cbl1016-clean-rerun.lock').exists()
    ok = ok and all(not e['scan_at_start'].get('hits') for e in ex)
    r.check(ok, 'a%02d: exclusivity recorded (%d window(s), lock, process scans, lock moved into the archive)'
            % (n, len(ex)))


def expected_normal_failures():
    return {}


# --------------------------------------------------------------------------- scenarios

def sc_base_m3env(base):
    r = Result('base-m3env')
    root = Root(base, 'base-m3env').setup()
    f = {3: 'build_fail', 6: 'csim_mismatch', 7: 'csynth_fail', 10: 'vivado_fail', 13: 'vivado_noreport',
         16: 'survivor_after_return'}
    for ln, fl in f.items():
        root.fault(NAME[ln], fl)
    rc, out, err = root.run('void-restart', 'fail-line')
    r.check(rc == 0, 'driver exit 0 (got %s) %s' % (rc, err[-300:] if rc else ''))
    rec = root.record()
    r.check(len(rec['attempts']) == 1 and rec['attempts'][0]['status'] == 'complete', 'one attempt, complete')
    check_step3(r, root, 1)
    check_context_first(r, root, 1)
    check_sidecars(r, root, 1)
    check_exclusivity(r, root, 1)
    tab = check_table(r, root, 1, {NAME[3]: 'build', NAME[7]: 'csynth', NAME[10]: 'vivado', NAME[13]: 'vivado'},
                      'complete')
    row6 = [x for x in tab['rows'] if x['line_no'] == 6][0]
    r.check(row6['failed_at'] is None and row6['values']['csim_mismatch'] == 7,
            'a C-simulation mismatch is a completed stage with its count (7), not a failed one')
    sc16 = root.sidecar(1, NAME[16])
    sv = sc16.get('survivors_after_return') or {}
    spid = int(root.marker(NAME[16], 'survivor.pid').read_text())
    r.check(sv.get('signals') == ['SIGTERM', 'SIGKILL'] and sv.get('gone') is True and not pid_alive(spid),
            'a child left behind after run_case.sh returned was stopped (SIGTERM ignored, then SIGKILL) and is gone')
    att = rec['attempts'][0]
    r.check(sha(root.scripts / 'env.sh') == root.exp['env.sh'] and att['env_swap'] is None
            and all(x['ok'] for x in att['env_rechecks']), 'm3-env route: env.sh untouched and rechecked')
    r.check(att['m3env_freeze_after'][-1]['equal_to_before'] is True, "m3-env's pip freeze hash equal before and after")
    sc3 = root.sidecar(1, NAME[3])
    r.check(sc3['run_case_rc'] == 1 and sc3['reports']['build_rc'] == 1, 'build failure: run_case.sh rc 1, build rc 1 recorded')
    # the table script alone reproduces the driver's table byte for byte
    before = (root.adir(1) / 'results' / 'table.json').read_bytes()
    p = subprocess.run([sys.executable, '-I', '-B', str(TABLE), '--evidence', str(root.evidence), '--attempt', '1',
                        '--expected-hashes', str(root.expected)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    r.check(p.returncode == 0 and (root.adir(1) / 'results' / 'table.json').read_bytes() == before,
            'table.py run alone reproduces the table byte for byte')
    p = subprocess.run([sys.executable, '-I', '-B', str(TABLE), '--evidence', str(root.evidence), '--attempt', '1'],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    r.check(p.returncode != 0, 'table.py without the stand-in hashes refuses the stand-in parser (embedded table)')
    # one clean rerun per route version: a complete attempt is never rerun
    rj = (root.evidence / 'run.json').read_bytes()
    snap = snapshot(root.root)
    rc2, _, err2 = root.run('void-restart', 'fail-line')
    r.check(rc2 == 2 and 'complete' in err2 and (root.evidence / 'run.json').read_bytes() == rj
            and snapshot(root.root) == snap, 'a second invocation on a complete run is refused and writes nothing')
    root.kill_leftovers()
    return r


def sc_base_clone(base):
    r = Result('base-clone')
    root = Root(base, 'base-clone').setup()
    root.fault(NAME[2], 'csim_mismatch')
    rc, out, err = root.run('fail-line', 'void-restart', route='clone')
    r.check(rc == 0, 'driver exit 0 (got %s) %s' % (rc, err[-300:] if rc else ''))
    rec = root.record()
    att = rec['attempts'][0]
    r.check(len(rec['attempts']) == 1 and att['status'] == 'complete', 'one attempt, complete')
    sw = att['env_swap']
    copy_sha = sw['copy_sha256']
    first = min(root.sidecar(1, x)['start'] for x in att['started'])
    r.check(bool(HEX64.match(copy_sha)) and copy_sha != root.exp['env.sh'] and sw['swaps'][0]['time'] < first,
            "the env.sh copy's sha256 was recorded before the first line started")
    diff = [l for l in sw['diff'] if l[:1] in '+-' and not l.startswith(('+++', '---'))]
    r.check(len(diff) == 2 and diff[0] == '-export PY=$HOME/venvs/m3-env/bin/python'
            and diff[1] == '+export PY=%s' % root.clone and sw['py_line'] == 3,
            'the copy differs from the original only in its PY line, which points at the clone')
    arch = Path(att['archive_dir'])
    copies = sorted(arch.glob('env.sh.clone-copy-*'))
    orig_lines = (STUBS / 'scripts' / 'env.sh').read_text().split('\n')
    cp_lines = copies[0].read_text().split('\n') if copies else []
    r.check(len(copies) == 1 and sha(copies[0]) == copy_sha and len(cp_lines) == len(orig_lines)
            and [i for i in range(len(orig_lines)) if orig_lines[i] != cp_lines[i]] == [2],
            'the copy is kept in the archive and differs in line 3 only')
    r.check(sha(root.scripts / 'env.sh') == root.exp['env.sh']
            and os.lstat(root.scripts / 'env.sh').st_ino == root.pre['env_sh_inode']
            and sw['active'] is False and all(x['ok'] and x['inode_equals_original'] for x in sw['restores']),
            'the original env.sh was moved into the archive and restored by rename (same inode, same sha256) and rechecked')
    check_sidecars(r, root, 1, route='clone', copy_sha=copy_sha)
    sc = root.sidecar(1, RULE[0])
    r.check(sc['env']['PY'] == str(root.clone), 'lines ran with PY pointing at the clone')
    check_step3(r, root, 1)
    check_context_first(r, root, 1)
    check_exclusivity(r, root, 1)
    check_table(r, root, 1, {}, 'complete')
    r.check(att['m3env_freeze_after'][-1]['equal_to_before'] is True and rec['m3env_python'].endswith('m3-env/bin/python'),
            "m3-env's pip freeze hash recorded before and after, equal, diff empty")
    r.check(rec['clone']['lock_sha256'] == sha(root.clone_lock), "the clone's lock hash is recorded")
    root.kill_leftovers()
    return r


# (fault line, fault lines, assertion, line where it is detected, persistent file or None)
KINDS = {
    'dir': (5, ['foreign_dir ' + NAME[8]], 'proj_dir_absent', 8, None),
    'skip': (5, ['skip'], 'no_skip', 5, None),
    'stdout': (5, ['no_stdout'], 'stdout_recorded', 5, None),
    'script': (5, ['tamper_common'], 'frozen_scripts_sha256_after', 5, 'common.py'),
    'envsh': (5, ['tamper_envsh'], 'env_sh_sha256_after', 5, 'env.sh'),
    'da4ml': (5, ['da4ml_change'], 'da4ml_version', 6, 'da4ml'),
}
LATER = {'script': 'frozen_scripts_sha256_before', 'envsh': 'env_sh_sha256', 'da4ml': 'da4ml_version'}


def sc_assert(base, kind, A, I):
    label = 'assert-%s-%s' % (kind, A)
    r = Result(label)
    root = Root(base, label).setup()
    fl, lines, aname, det, persist = KINDS[kind]
    root.fault(NAME[fl], *lines)
    rc, out, err = root.run(A, I)
    rec = root.record()
    a1 = rec['attempts'][0]
    sc = root.sidecar(1, NAME[det])
    r.check(aname in (sc or {}).get('assertion_failures', []),
            'the planted fault fires assertion %s at line %d' % (aname, det))
    if kind == 'skip':
        r.check(sc['reports']['skip'] is True, 'the skip printed by run_case.sh is visible in its recorded stdout')
    if A == 'fail-line':
        r.check(rc == 0 and len(rec['attempts']) == 1 and a1['status'] == 'complete',
                'fail-line: the attempt continues and completes (exit %s)' % rc)
        failing = {NAME[det]}
        if persist:
            failing |= {x for x in RULE if LINE_NO[x] > det}
        bad = []
        for x in failing:
            s = root.sidecar(1, x)
            want = aname if x == NAME[det] else LATER[kind]
            if s['policy_applied'] != 'fail-line' or want not in s['assertion_failures']:
                bad.append(x)
        r.check(not bad, 'fail-line: %d line(s) failed by the remedy with the expected assertion %s'
                % (len(failing), bad[:3] if bad else ''))
        if kind == 'dir':
            r.check(sc['outcome'] == 'not_run_assertion_failed'
                    and not root.marker(NAME[det], 'run_case.started').exists(),
                    'the line whose project directory already existed was not run')
        tab = check_table(r, root, 1, {x: 'build' for x in failing}, 'complete')
        if kind == 'skip':
            row = [x for x in tab['rows'] if x['name'] == NAME[det]][0]
            r.check(row['skip_printed'] and row['synth_result_json']['present'],
                    'the skipped line is failed although a foreign synth_result.json exists')
        if kind == 'envsh':
            r.check(a1['env_rechecks'][-1]['ok'] is False, 'the end-of-attempt env.sh recheck records the mismatch')
    else:
        r.check(a1['status'] == 'void' and NAME[det] in a1['cause'] and aname in a1['cause'],
                'void-restart: attempt 1 voided with its cause (%s)' % a1.get('cause'))
        r.check((root.sidecar(1, NAME[det]) or {}).get('policy_applied') == 'void'
                and len(a1['started']) == 4 + sum(1 for x in RULE if LINE_NO[x] <= det),
                'the void attempt stopped at the failing line and stays on record')
        if persist:
            a2 = rec['attempts'][1]
            r.check(rc == 3 and a2['status'] == 'fault' and a2['phase'] == 'start',
                    'the restart from step 3 refuses while the fault persists (exit %s, attempt 2 %s)' % (rc, a2['status']))
            if persist == 'da4ml':
                root.restore_da4ml(root.m3)
            else:
                root.restore_script(persist)
            rc, out, err = root.run(A, I)
            rec = root.record()
        r.check(rc == 0 and len(rec['attempts']) == 2 and rec['attempts'][1]['status'] == 'complete',
                'attempt 2 restarted from step 3 and completed (exit %s)' % rc)
        a2 = rec['attempts'][1]
        r.check((Path(a2['archive_dir']) / 'proj_main' / NAME[fl]).is_dir(),
                "the void attempt's project directories were archived by attempt 2's step 3")
        check_step3(r, root, 2, prior=1)
        check_context_first(r, root, 2)
        check_table(r, root, 1, dict({x: 'build' for x in RULE if LINE_NO[x] > det},
                                     **({NAME[det]: 'build'} if kind in ('dir', 'skip', 'stdout', 'da4ml') else {})),
                    'void', label='void attempt a01 table')
        check_table(r, root, 2, {}, 'complete')
        tab1 = root.table(1)
        r.check(tab1['header']['label'].startswith('context'), 'the void attempt is labelled context')
    root.kill_leftovers()
    return r


def interrupt_and_check(r, root, A, I, route, kill):
    """Run with line 9 hanging in C-synthesis (reports already written, a child ignoring
    SIGTERM); stop the driver with SIGTERM or SIGKILL; check; restart; check the policy."""
    L = NAME[9]
    root.fault(L, 'hang_csynth 300', 'reports_before_hang', 'survivor')
    p = root.popen(A, I, route)
    ok = wait_for(root.marker(L, 'csynth.hanging'), 300, p)
    r.check(ok, 'line 9 reached its C-synthesis stage (marker)')
    if not ok:
        p.kill()
        out, err = p.communicate()
        r.check(False, 'driver ended before line 9 hung: exit %s %s' % (p.returncode, err[-400:]))
        return
    time.sleep(0.3)
    spid = int(root.marker(L, 'survivor.pid').read_text())
    tpid = int(root.marker(L, 'csynth.hanging').read_text())
    env_during = sha(root.scripts / 'env.sh')
    rec0 = root.record()
    pg = rec0['attempts'][0]['in_flight']['pgid']
    mem = session_members(pg)
    r.check(len(set(g for _, g in mem)) >= 2 and (tpid in [x for x, _ in mem]) and os.getpgid(tpid) != pg,
            "before the signal the line's session holds %d process groups; the hung tool is outside the leader's group "
            "(the stand-in timeout's new group)" % len(set(g for _, g in mem)))
    p.send_signal(signal.SIGKILL if kill else signal.SIGTERM)
    out, err = p.communicate(timeout=240)
    if route == 'clone':
        r.check(env_during == rec0['attempts'][0]['env_swap']['copy_sha256'], 'during the line env.sh was the copy')
    if kill:
        r.check(p.returncode == -9, 'driver killed with SIGKILL')
        r.check(session_alive(pg) and pid_alive(spid) and (route != 'clone' or sha(root.scripts / 'env.sh') == env_during),
                "after SIGKILL no exit path ran: the line's session is still alive and env.sh is still the copy")
        rc, out, err = root.run(A, I, route)
        detected = 'restart'
    else:
        r.check(p.returncode == 5, 'driver exit 5 after SIGTERM (got %s)' % p.returncode)
        rc = None
        detected = 'exit trap'
    rec = root.record()
    a1 = rec['attempts'][0]
    sc = root.sidecar(1, L)
    it = sc['interrupted'] or {}
    gs = it.get('group_stop') or {}
    rs = it.get('env_sh_restore') or {}
    r.check(sc['outcome'] == 'interrupted' and it.get('stage') == 'csynth' and it.get('detected_by') == detected,
            'sidecar of line 9: interrupted in csynth, detected by the %s' % detected)
    r.check(gs.get('signals') == ['SIGTERM', 'SIGKILL'] and gs.get('gone') is True and not session_alive(pg)
            and not pid_alive(spid) and not pid_alive(tpid),
            "the line's whole session was stopped and waited for: the hung tool and the child that ignored SIGTERM are "
            "gone, and the record says gone")
    seen = set()
    for x in gs.get('rounds') or []:
        seen |= set(x.get('pgids') or [])
    r.check(len(seen) >= 2, 'the stop record lists the %d process groups of the session it signalled' % len(seen))
    r.check(rs.get('ok') is True and gs.get('end', '') <= rs.get('time', '') <= it.get('time', '')
            and sha(root.scripts / 'env.sh') == root.exp['env.sh'],
            'env.sh restored and rechecked after the group was stopped and before the sidecar was written (%s)'
            % ('swap' if route == 'clone' else 'recheck only'))
    rep = root.adir(1) / 'reports' / L
    r.check((rep / 'myproject_prj/solution1/syn/report/csynth.xml').is_file()
            and (rep / 'vivado' / 'post_opt_util.rpt').is_file(),
            'the interrupted line left C-synthesis and Vivado reports in its evidence')
    if not kill:
        r.check(a1['status'] == 'interrupted' and a1['phase'] == 'rule'
                and not (root.scr / 'cbl1016-clean-rerun.lock').exists(),
                'attempt recorded as interrupted in the rule phase; lock released')
        stdout_sha = sha(root.adir(1) / 'stdout' / (L + '.stdout'))
        start = sc['start']
        rc, out, err = root.run(A, I, route)
        rec = root.record()
        a1 = rec['attempts'][0]
    else:
        stdout_sha = sha(root.adir(1) / 'stdout' / (L + '.stdout'))
        start = sc['start']
    r.check(rc == 0, 'the restarted driver completes (exit %s) %s' % (rc, err[-300:] if rc else ''))
    if I == 'fail-line':
        r.check(len(rec['attempts']) == 1 and a1['status'] == 'complete', 'option (i): same attempt, completed')
        rsm = a1['resumes'][0]
        r.check(rsm['not_retried'] == [L] and rsm['exempt_from_dir_absent'] == [L] and rsm['next_line'] == NAME[10],
                'option (i): line 9 not retried, exempt from the directory assertion, resume at line 10')
        sc9 = root.sidecar(1, L)
        r.check(sc9['start'] == start and sha(root.adir(1) / 'stdout' / (L + '.stdout')) == stdout_sha,
                'line 9 has one sidecar and one stdout, both from its single attempt')
        sc10 = root.sidecar(1, NAME[10])
        r.check(sc10['start'] > it['time'] and all(x['ok'] for x in sc10['assertions']),
                'line 10 ran after the restart, with every per-line assertion')
        tab = check_table(r, root, 1, {L: 'csynth'}, 'complete')
        row = [x for x in tab['rows'] if x['name'] == L][0]
        r.check(row['artifacts_present']['csynth_report'] and row['artifacts_present']['post_opt_util']
                and row['stages']['csynth'] == 'failed', 'the table fails line 9 at csynth whatever artifacts exist')
        if route == 'clone':
            sw = a1['env_swap']
            r.check(len(sw['swaps']) == 2 and sw['swaps'][0]['copy_sha256'] == sw['swaps'][1]['copy_sha256']
                    and len(a1['exclusivity']) == 2, 'option (i): exclusivity and the env.sh copy re-established')
        check_sidecars(r, root, 1, route, (a1.get('env_swap') or {}).get('copy_sha256'))
    else:
        r.check(len(rec['attempts']) == 2 and a1['status'] == 'void' and 'option (ii)' in a1['cause']
                and rec['attempts'][1]['status'] == 'complete', 'option (ii): attempt 1 voided, attempt 2 complete')
        a2 = rec['attempts'][1]
        r.check((Path(a2['archive_dir']) / 'proj_main' / L).is_dir(), "attempt 1's directories archived by attempt 2")
        check_table(r, root, 1, dict({x: 'build' for x in RULE if LINE_NO[x] > 9}, **{L: 'csynth'}), 'void',
                    label='void attempt a01 table')
        check_table(r, root, 2, {}, 'complete')
        check_step3(r, root, 2, prior=1)
        check_context_first(r, root, 2)
        check_sidecars(r, root, 2, route, (a2.get('env_swap') or {}).get('copy_sha256'))
    r.check(sha(root.scripts / 'env.sh') == root.exp['env.sh'], 'env.sh is the original at the end')


def sc_sigterm(base, A, I, route):
    label = 'sigterm-%s-%s-%s' % (A, I, route)
    r = Result(label)
    root = Root(base, label).setup()
    interrupt_and_check(r, root, A, I, route, kill=False)
    root.kill_leftovers()
    return r


def sc_sigkill(base, A, I):
    label = 'sigkill-%s-%s-clone' % (A, I)
    r = Result(label)
    root = Root(base, label).setup()
    interrupt_and_check(r, root, A, I, 'clone', kill=True)
    root.kill_leftovers()
    return r


def sc_step4_driver_check(base):
    r = Result('step4-driver-check')
    root = Root(base, 'step4-driver-check').setup()
    root.fault(CONTEXT[0], 'nolog')
    rc, out, err = root.run('void-restart', 'fail-line')
    a1 = root.record()['attempts'][0]
    chk = a1['driver_checks'][0]
    r.check(rc == 3 and a1['status'] == 'fault' and chk['ok'] is False and chk['build_log'] is False
            and a1['started'] == [CONTEXT[0]], 'a missing logs directory stops the run after the first context line '
                                              '(driver check fails, exit %s), before any rule line' % rc)
    rc, out, err = root.run('void-restart', 'fail-line')
    rec = root.record()
    a1 = rec['attempts'][0]
    r.check(rc == 0 and len(rec['attempts']) == 1 and a1['status'] == 'complete' and a1['context_runs'] == 2,
            'after the fix the four context lines run again in the same attempt, no void (exit %s)' % rc)
    r.check((root.adir(1) / 'context-run-1' / 'sidecars' / (CONTEXT[0] + '.provenance.json')).is_file()
            and a1['context_archives'][0]['run'] == 1, 'the first context run is kept in the evidence and the archive')
    check_context_first(r, root, 1)
    check_table(r, root, 1, {}, 'complete')
    return r


def sc_step4_assertion(base):
    r = Result('step4-assertion')
    root = Root(base, 'step4-assertion').setup()
    root.fault(CONTEXT[1], 'tamper_common')
    rc, out, err = root.run('fail-line', 'void-restart')
    a1 = root.record()['attempts'][0]
    r.check(rc == 3 and a1['status'] == 'fault' and 'frozen_scripts_sha256_after' in a1['cause']
            and not any(x in a1['started'] for x in RULE),
            'an assertion failing at a context line is a step-4 fault, not a void or a failed line (exit %s)' % rc)
    rc, out, err = root.run('fail-line', 'void-restart')
    r.check(rc == 3, 'with the tampered script still in place the driver does not continue (exit %s)' % rc)
    root.restore_script('common.py')
    rc, out, err = root.run('fail-line', 'void-restart')
    rec = root.record()
    a1 = rec['attempts'][0]
    arch = Path(a1['context_archives'][0]['archive'])
    r.check(rc == 0 and len(rec['attempts']) == 1 and a1['status'] == 'complete' and a1['context_runs'] == 2,
            'after the fix: context rerun in the same attempt, completed (exit %s)' % rc)
    r.check((arch / 'proj_main' / CONTEXT[0]).is_dir() and (arch / 'proj_main' / CONTEXT[1]).is_dir()
            and (arch / 'logs' / 'parse.log').is_file() and (arch / 'logs' / ('build_%s.log' % CONTEXT[0])).is_file(),
            "the first context run's project directories, build logs and parse.log were moved into the archive")
    check_table(r, root, 1, {}, 'complete')
    return r


def sc_context_halt(base):
    r = Result('context-halt')
    root = Root(base, 'context-halt').setup()
    root.fault(CONTEXT[2], 'build_fail')
    rc, out, err = root.run('void-restart', 'void-restart')
    a1 = root.record()['attempts'][0]
    r.check(rc == 6 and a1['status'] == 'context-halt' and CONTEXT[2] in a1['cause']
            and not any(x in a1['started'] for x in RULE),
            'a context line without a stage result halts before any rule line (exit %s)' % rc)
    note = root.root / 'context-review.txt'
    note.write_text('G72 stand-in review: the build failure of %s is a planted stage outcome.\n' % CONTEXT[2])
    rc, out, err = root.run('void-restart', 'void-restart', extra=['--context-reviewed', str(note)])
    rec = root.record()
    a1 = rec['attempts'][0]
    r.check(rc == 0 and a1['status'] == 'complete' and a1['context_runs'] == 1
            and a1['context_reviews'][0]['sha256'] == sha(note),
            'with --context-reviewed the run proceeds to the rule lines without rerunning the context lines')
    check_table(r, root, 1, {CONTEXT[2]: 'build'}, 'complete')
    return r


def sc_context_interrupt(base):
    r = Result('context-interrupt')
    root = Root(base, 'context-interrupt').setup()
    root.fault(CONTEXT[1], 'hang_build 300')
    p = root.popen('fail-line', 'void-restart')
    ok = wait_for(root.marker(CONTEXT[1], 'build.hanging'), 300, p)
    time.sleep(0.2)
    bpid = int(root.marker(CONTEXT[1], 'build.hanging').read_text()) if ok else -1
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=240)
    a1 = root.record()['attempts'][0]
    sc = root.sidecar(1, CONTEXT[1])
    r.check(ok and not pid_alive(bpid) and not session_alive(sc['pgid']),
            'the hung build, which ran in the stand-in timeout\'s own process group, is gone with the whole session')
    r.check(ok and p.returncode == 5 and a1['status'] == 'interrupted' and a1['phase'] == 'context'
            and sc['interrupted']['stage'] == 'build', 'SIGTERM during a context line: interrupted in build (exit %s)'
            % p.returncode)
    rc, out, err = root.run('fail-line', 'void-restart')
    rec = root.record()
    r.check(rc == 0 and len(rec['attempts']) == 1 and rec['attempts'][0]['status'] == 'complete'
            and rec['attempts'][0]['context_runs'] == 2,
            'before any rule line an interruption is a step-4 fault even under option (ii): no void, context rerun')
    check_table(r, root, 1, {}, 'complete')
    return r


def sc_void_cap(base):
    r = Result('void-cap')
    root = Root(base, 'void-cap').setup()
    root.fault(NAME[5], 'skip')
    rc, out, err = root.run('void-restart', 'fail-line', extra=['--max-voids', '1'])
    rec = root.record()
    r.check(rc == 4 and len(rec['attempts']) == 1 and rec['attempts'][0]['status'] == 'void',
            'with --max-voids 1 the first void is not restarted (exit %s)' % rc)
    rc, out, err = root.run('void-restart', 'fail-line', extra=['--max-voids', '2'])
    rec = root.record()
    r.check(rc == 0 and len(rec['attempts']) == 2 and rec['attempts'][1]['status'] == 'complete',
            'a new invocation with --max-voids 2 restarts from step 3 and completes')
    return r


def sc_hash_mismatch(base):
    r = Result('hash-mismatch')
    root = Root(base, 'hash-mismatch').setup()
    exp = json.loads(root.expected.read_text())
    exp['synth_one.sh'] = '0' * 64
    root.expected.write_text(json.dumps(exp))
    snap = snapshot(root.root)
    rc, out, err = root.run('void-restart', 'fail-line')
    r.check(rc == 2 and 'synth_one.sh' in err and not root.evidence.exists() and snapshot(root.root) == snap,
            'a frozen-script hash mismatch at start is refused before anything is written (exit %s)' % rc)
    return r


def sc_refusals(base):
    r = Result('refusals')
    root = Root(base, 'refusals').setup()
    eq = root.root / 'g71-stand-in.txt'
    eq.write_text('G72 stand-in for the G71 record; not a record.\n')
    full = ['--mode', 'real', '--evidence', str(root.evidence), '--route', 'm3-env', '--da4ml-version', '0.5.2',
            '--on-assertion', 'void-restart', '--on-interrupt', 'fail-line', '--decision', str(root.decision),
            '--equivalence-record', str(eq)]

    def drop(argv, opt):
        i = argv.index(opt)
        return argv[:i] + argv[i + 2:]

    dry = ['--mode', 'dry-run', '--home', str(root.home), '--evidence', str(root.evidence), '--route', 'm3-env',
           '--da4ml-version', '0.5.2', '--on-assertion', 'void-restart', '--on-interrupt', 'fail-line',
           '--expected-hashes', str(root.expected)]
    cases = [
        ('real mode without --on-assertion', drop(full, '--on-assertion'), '--on-assertion is required'),
        ('real mode without --on-interrupt', drop(full, '--on-interrupt'), '--on-interrupt is required'),
        ('real mode without --decision', drop(full, '--decision'), 'real mode needs --decision'),
        ('real mode with a decision path that does not exist',
         drop(full, '--decision') + ['--decision', str(root.root / 'absent.txt')], 'is not a non-empty file'),
        ('real mode without the G71 record', drop(full, '--equivalence-record'), 'needs --equivalence-record'),
        ('real mode with --expected-hashes', full + ['--expected-hashes', str(root.expected)], 'for a dry run only'),
        ('real mode with --home', full + ['--home', str(root.home)], '--home is for a dry run only'),
        ('real mode under a HOME that is not the account home', full, '$HOME is not the account home'),
        ('dry run without --on-assertion', drop(dry, '--on-assertion'), '--on-assertion is required'),
        ('dry run without --on-interrupt', drop(dry, '--on-interrupt'), '--on-interrupt is required'),
        ('dry run with the account home as --home', drop(dry, '--home') + ['--home', os.path.expanduser('~')],
         'never uses the account home'),
        ('clone route without --clone-lock', dry[:dry.index('--route') + 1] + ['clone'] + dry[dry.index('--route') + 2:]
         + ['--clone-python', str(root.clone)], 'needs --clone-lock'),
    ]
    for what, argv, msg in cases:
        snap = snapshot(root.root)
        p = subprocess.run([sys.executable, '-I', '-B', str(DRIVER)] + argv, env=root.env(), stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, timeout=120)
        r.check(p.returncode == 2 and msg in p.stderr and not root.evidence.exists() and snapshot(root.root) == snap,
                'refused, nothing written: %s (exit %s)' % (what, p.returncode))
    # stand-ins that are copies of the frozen scripts: refused before anything runs (copied, never executed)
    fz = Root(base, 'refusals-frozen-copies').setup()
    for f in ('run_case.sh', 'synth_one.sh', 'stage_b_build.py'):
        shutil.copyfile(FROZEN_DIR / f, fz.scripts / f)
    exp = json.loads(fz.expected.read_text())
    exp.update({f: sha(fz.scripts / f) for f in ('run_case.sh', 'synth_one.sh', 'stage_b_build.py')})
    fz.expected.write_text(json.dumps(exp))
    snap = snapshot(fz.root)
    rc, out, err = fz.run('void-restart', 'fail-line')
    r.check(rc == 2 and 'are the frozen scripts' in err and not fz.evidence.exists() and snapshot(fz.root) == snap,
            'a dry run against copies of the frozen scripts is refused before anything runs (exit %s)' % rc)
    return r


SCENARIOS = [('base-m3env', sc_base_m3env), ('base-clone', sc_base_clone)]
for _k in KINDS:
    for _i, _A in enumerate(('fail-line', 'void-restart')):
        _I = ('fail-line', 'void-restart')[(_i + list(KINDS).index(_k)) % 2]
        SCENARIOS.append(('assert-%s-%s' % (_k, _A), (lambda k, A, I: lambda b: sc_assert(b, k, A, I))(_k, _A, _I)))
for _A, _I, _route in (('void-restart', 'fail-line', 'm3-env'), ('void-restart', 'void-restart', 'clone'),
                       ('fail-line', 'fail-line', 'clone'), ('fail-line', 'void-restart', 'm3-env')):
    SCENARIOS.append(('sigterm-%s-%s-%s' % (_A, _I, _route),
                      (lambda A, I, ro: lambda b: sc_sigterm(b, A, I, ro))(_A, _I, _route)))
for _A, _I in (('void-restart', 'fail-line'), ('fail-line', 'void-restart')):
    SCENARIOS.append(('sigkill-%s-%s-clone' % (_A, _I), (lambda A, I: lambda b: sc_sigkill(b, A, I))(_A, _I)))
SCENARIOS += [('step4-driver-check', sc_step4_driver_check), ('step4-assertion', sc_step4_assertion),
              ('context-halt', sc_context_halt), ('context-interrupt', sc_context_interrupt),
              ('void-cap', sc_void_cap), ('hash-mismatch', sc_hash_mismatch), ('refusals', sc_refusals)]


# --------------------------------------------------------------------------- static

def static_checks(g):
    data = STUDY.read_bytes()
    g.check(hashlib.sha256(data).hexdigest() == STUDY_SHA, 'STUDY v0.8 snapshot sha256 %s' % STUDY_SHA[:16])
    lines = data.decode('utf-8').split('\n')
    i = lines.index('<!-- headline-rule v0.3 begin -->')
    j = lines.index('<!-- headline-rule v0.3 end -->')
    block = ('\n'.join(lines[i:j + 1]) + '\n').encode('utf-8')
    g.check(hashlib.sha256(block).hexdigest() == RULE_BLOCK_SHA,
            'headline-rule block (lines %d-%d) hashes to the recorded d9f2e386... (read for its hash only)' % (i + 1, j + 1))
    tab = {}
    for line in lines[59:72]:
        m = re.match(r'^\s*\| `([^`]+)` \| `([0-9a-f]{64})` \|\s*$', line)
        if m:
            tab[m.group(1)] = m.group(2)
    m = re.search(r'`env\.sh`.*?sha256 `([0-9a-f]{64})`', lines[71])
    g.check(list(tab.items()) == list(T.FROZEN_SHA256.items()),
            'embedded hash table equals STUDY v0.8 lines 60-70, nine files, same order')
    g.check(m is not None and m.group(1) == T.ENV_SH_SHA256, 'embedded env.sh hash equals STUDY v0.8 line 72')
    g.check(all(sha(FROZEN_DIR / f) == T.FROZEN_SHA256[f] for f in T.FROZEN_SHA256)
            and sha(FROZEN_DIR / 'env.sh') == T.ENV_SH_SHA256,
            'the frozen scripts on disk hash to the embedded table (hashed only, never executed)')
    tree = ast.parse((FROZEN_DIR / 'parse_reports.py').read_text())
    fns = {n.name: [a.arg for a in n.args.args] for n in tree.body if isinstance(n, ast.FunctionDef)}
    g.check(len(fns.get('csynth', [])) == 1 and len(fns.get('util', [])) == 1 and len(fns.get('timing', [])) == 1,
            'frozen parse_reports.py defines csynth(p), util(f), timing(f), the functions the table loads (AST only)')
    stub = ast.parse((STUBS / 'scripts' / 'parse_reports.py').read_text())
    sfns = {n.name: [a.arg for a in n.args.args] for n in stub.body if isinstance(n, ast.FunctionDef)}
    g.check(all(sfns.get(k) == fns.get(k) for k in ('csynth', 'util', 'timing', 'main')),
            'the stand-in parser has the same function names and arguments')
    std = set(sys.stdlib_module_names) | {'table'}
    for f in (DRIVER, TABLE):
        src = f.read_text()
        mods = set()
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Import):
                mods |= {a.name.split('.')[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.level == 0:
                mods.add(n.module.split('.')[0])
        g.check(mods <= std, '%s imports the standard library only %s' % (f.name, sorted(mods - std) or ''))
        try:
            ast.parse(src, feature_version=(3, 7))
            g.check(True, '%s parses under the Python 3.7 grammar (written for 3.6+; library calls not checked on 3.6)' % f.name)
        except SyntaxError as e:
            g.check(False, '%s does not parse under the 3.7 grammar: %s' % (f.name, e))
    g.check(sha(STUBS / 'scripts' / 'cases_bc.txt') == T.FROZEN_SHA256['cases_bc.txt'],
            'the stand-in cases_bc.txt is byte-identical to the frozen case list')
    g.check(all(sha(STUBS / 'scripts' / f) != T.FROZEN_SHA256[f] for f in T.FROZEN_SHA256 if f != 'cases_bc.txt')
            and sha(STUBS / 'scripts' / 'env.sh') != T.ENV_SH_SHA256, 'every other stand-in differs from its frozen file')


def unit_tests(g):
    p = subprocess.run([sys.executable, '-I', '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', str(UNIT)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(FORK),
                       env={'PATH': '/usr/bin:/bin', 'HOME': tempfile.mkdtemp(prefix='g72-unit-'), 'LANG': 'C'})
    tail = [l for l in p.stdout.splitlines() if l.strip()][-1:]
    g.info('pytest', tail)
    g.check(p.returncode == 0, 'unit tests of the table logic pass (%s)' % (tail[0] if tail else 'no output'))


# ------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--expect-sha12')
    ap.add_argument('--keep')
    ap.add_argument('--only', action='append')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--keep-temp', action='store_true', help='leave the temporary base (with its venvs) in place')
    a = ap.parse_args()
    g = gatelib.Gate(GATE)
    g.header(a.expect_sha12)
    static_checks(g)
    unit_tests(g)
    base = Path(tempfile.mkdtemp(prefix='g72-'))
    g.info('temporary base', str(base))
    todo = [(n, f) for n, f in SCENARIOS if not a.only or n in a.only]
    results = {}

    def one(item):
        n, f = item
        t0 = time.time()
        try:
            res = f(base)
        except Exception as e:  # noqa: BLE001
            import traceback
            res = Result(n)
            res.check(False, 'scenario raised %s' % traceback.format_exc()[-1500:])
        res.info['seconds'] = round(time.time() - t0, 1)
        return n, res

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for n, res in ex.map(one, todo):
            results[n] = res
    for n, _ in todo:
        res = results[n]
        print('--- scenario %s (%.1f s)' % (n, res.info['seconds']))
        for ok, what in res.rows:
            g.check(ok, what)
    if a.keep:
        dst = Path(a.keep)
        for n, _ in todo:
            src = base / n
            if src.exists():
                shutil.copytree(src, dst / n, symlinks=True, ignore=shutil.ignore_patterns('venvs'))
    for n, _ in todo:
        if (base / n).exists():
            Root(base, n).kill_leftovers()
    if not a.keep_temp:
        shutil.rmtree(base, ignore_errors=True)
    g.check_clean('at end')
    return g.finish()


if __name__ == '__main__':
    sys.exit(main())
