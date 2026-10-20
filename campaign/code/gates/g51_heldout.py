"""Gate G51 (PREFLIGHT-PLAN section 3 row G51; STUDY v0.10 section 13 "Further required gates": "validation-only
tracing and selection", section 4 "Held-out test set", section 8.5; modules M3 cbl/data.py, M13 cbl/runner.py and
M16 cbl/select.py, stats.py, s2.py, confirm.py). Two places: --place mac (the static audit, Mac CPU, no backend) and
--place pcc (the static audit again and the runtime guard, PC CPU, venv of record, 2-file smoke archive).

Static audit, over every file `git ls-files` lists in the fork:
- text: every file that names a held-out identifier (150c-test, heldout, held-out, held_out, holdout, read_heldout,
  the held-out archive's sha256 prefix, the raw held-out directory) is in the allowlist below, with its reason;
- production code (cbl/, gen/; syntax trees, docstrings excluded):
  R1 `read_heldout` is called nowhere (the sealed reader is called only after the freeze, by code not yet written);
  R2 only cbl/data.py imports h5py, and only cbl/data.py holds a string constant naming the held-out archive;
  R3 Chang's `get_data`, which opens 150c-test.h5 (chang/data.py:13), is called nowhere outside chang/;
  R4 the selection and statistics modules (cbl/select.py, stats.py, s2.py, confirm.py) import neither h5py nor
     cbl.data and load no array file: they read only records handed to them;
  R5 only cbl/data.py reads the preparation manifest's `heldout` block, apart from gen/protocol.py, which reads the
     held-out row count for the derived protocol's metric n (a count, no data);
  R6 in cbl/runner.py `prepare` installs the held-out guard before the training data are read;
  R7 no `trace_minmax` call: the export trace (M18, B14) is not in the fork; STUDY 9 has it run on train, then
     validation.
Runtime guard (pcc): one runner run (CH-8, 1 epoch) in a fresh process under the kernel open-file audit, with a gate
probe that, inside the training process after epoch 0, asks the installed guard about the held-out archive, a hard
link to it made by this gate (a link opens nothing), a path under the raw held-out directory and the training archive
(names only, no open), and then tries real opens of placeholders only (an empty `150c-test.h5`, an empty file with a
raw held-out file's name, and a symbolic link with an innocent name to the first), through builtins.open, os.open,
numpy.load and h5py.File: each must raise HeldOutAccessRefused, and the kernel audit must see no open of a placeholder
and none of a held-out file. The real held-out archive is never opened, even if the guard failed. The run completes,
so the guard lets the training archive through. Control: an unguarded process that opens the placeholder is seen by
the audit. Every reading is engineering, not quotable.

    python -I -B gates/g51_heldout.py --place mac --expect-sha12 SHA12 [--json PATH]
    <venv>/bin/python -I -B gates/g51_heldout.py --place pcc --expect-sha12 SHA12 --work DIR --archive SMOKE \\
        [--json PATH] [--children-out DIR]
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import gatelib  # noqa: E402
from syncroot import require_unsynced  # noqa: E402

GATE = 'G51'
PATTERN = re.compile(r'150c-test|heldout|held-out|held_out|holdout|read_heldout|9748d1aa2af060d2|bnjettag-data/val|val_dir',
                     re.IGNORECASE)
ALLOW = (
    (r'^cbl/data\.py$', 'M3: the sealed reader, the held-out counts and hashes of the preparation manifest, the guard'),
    (r'^cbl/wandb_log\.py$', 'M15: the held-out key refusal before the freeze'),
    (r'^cbl/(select|stats|s2|confirm)\.py$', 'M16: freeze manifest and held-out tables from records handed in (R4)'),
    (r'^cbl/runner\.py$', 'M13: docstring and the guard installation (R6)'),
    (r'^cbl/rng\.py$', 'M4: docstring (the held-out lines removed from the reader)'),
    (r'^chang/', 'Chang\'s release, byte-identical; get_data is never called by the fork (R3)'),
    (r'^configs/protocol/', 'derived protocol prose (held-out evaluation described, M17)'),
    (r'^gen/(protocol\.py|protocol_text\.json|inputs\.json)$', 'protocol prose and provenance text (M17)'),
    (r'^manifests/data-prep-150c\.json$', 'B2 preparation manifest: hashes and counts only'),
    (r'^NOTES-B17\.md$', 'B17 documentation: evaluation does not read held-out data'),
    (r'^NOTES-B18\.md$', 'B18 documentation: S4 option remains unset until after S2 and before held-out reading'),
    (r'^nrp/jobs\.py$', 'launch brief prose: training/validation only; no held-out read'),
    (r'^tools/README\.md$', 'harness documentation: lifecycle evaluation implies no held-out read'),
    (r'^tools/draft_brief\.py$', 'draft brief prose: evaluation does not open the held-out archive'),
    (r'^docs/', 'interface notes'),
    (r'^gates/', 'gate code: audits, placeholders, the sealed reader\'s gates'),
    (r'^tests/', 'unit tests'),
)
PRODUCTION = ('cbl/', 'gen/', 'nrp/', 'tools/', 'da4ml_clean/')
M16 = ('cbl/select.py', 'cbl/stats.py', 'cbl/s2.py', 'cbl/confirm.py')


def git_files():
    rc, out, _ = gatelib.git('ls-files')
    return [x for x in out.splitlines() if x]


def _docstring_nodes(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            b = node.body[0]
            if isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and isinstance(b.value.value, str):
                out.add(id(b.value))
    return out


def static_audit(g):
    files = git_files()
    hits = {}
    for f in files:
        p = FORK / f
        try:
            text = p.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            continue
        n = len(PATTERN.findall(text))
        if n:
            hits[f] = n
    unexplained = {f: n for f, n in hits.items() if not any(re.search(rx, f) for rx, _ in ALLOW)}
    g.info('files naming a held-out identifier', {'files': len(hits), 'by_reason': {why: sorted(f for f in hits if re.search(rx, f))
                                                                                     for rx, why in ALLOW if any(re.search(rx, f) for f in hits)}})
    g.check(not unexplained, f'text: every one of the {len(hits)} files naming a held-out identifier is in the allowlist '
                             f'with its reason' + (f' (unexplained: {unexplained})' if unexplained else ''))
    prod = [f for f in files if f.endswith('.py') and f.startswith(PRODUCTION)]
    r1, r2i, r2s, r3, r4, r5, r7 = [], [], [], [], [], [], []
    runner_order = None
    for f in prod:
        src = (FORK / f).read_text(encoding='utf-8')
        tree = ast.parse(src)
        docs = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
                if name == 'read_heldout':
                    r1.append(f'{f}:{node.lineno}')
                if name == 'get_data':
                    r3.append(f'{f}:{node.lineno}')
                if name == 'trace_minmax':
                    r7.append(f'{f}:{node.lineno}')
                if f in M16 and name == 'load' and isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id in ('np', 'numpy'):
                    r4.append(f'{f}:{node.lineno} np.load')
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module or '']
                names = [x.name for x in node.names]
                if any(m == 'h5py' or m.startswith('h5py.') for m in mods) and f != 'cbl/data.py':
                    r2i.append(f'{f}:{node.lineno}')
                if f in M16 and (any(m in ('h5py', 'cbl.data') for m in mods)
                                 or (isinstance(node, ast.ImportFrom) and node.module == 'cbl' and 'data' in names)):
                    r4.append(f'{f}:{node.lineno} import')
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                if ('150c-test' in node.value or node.value.endswith('test.h5')) and f != 'cbl/data.py':
                    r2s.append(f'{f}:{node.lineno}')
            if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == 'heldout'
                    and f not in ('cbl/data.py', 'gen/protocol.py')):
                r5.append(f'{f}:{node.lineno}')
        if f == 'cbl/runner.py':
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'prepare':
                    calls = [(n.lineno, n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, 'id', None))
                             for n in ast.walk(node) if isinstance(n, ast.Call)]
                    g_ = [ln for ln, nm in calls if nm == 'install_heldout_guard']
                    d_ = [ln for ln, nm in calls if nm == 'load_data']
                    runner_order = (g_, d_)
    g.check(not r1, f'R1: read_heldout is called nowhere in production code ({len(prod)} files)' + (f' ({r1})' if r1 else ''))
    g.check(not r2i and not r2s, 'R2: only cbl/data.py imports h5py and holds a string constant naming the held-out archive'
                                 + (f' (imports {r2i}, constants {r2s})' if (r2i or r2s) else ''))
    g.check(not r3, 'R3: Chang\'s get_data is called nowhere outside chang/' + (f' ({r3})' if r3 else ''))
    g.check(not r4 and all((FORK / m).exists() for m in M16),
            'R4: cbl/select.py, stats.py, s2.py and confirm.py import neither h5py nor cbl.data and load no array file'
            + (f' ({r4})' if r4 else ''))
    g.check(not r5, 'R5: only cbl/data.py reads the preparation manifest\'s heldout block (gen/protocol.py its row count)'
                    + (f' ({r5})' if r5 else ''))
    g.check(runner_order is not None and len(runner_order[0]) == 1 and runner_order[1] and runner_order[0][0] < min(runner_order[1]),
            f'R6: cbl/runner.py prepare installs the held-out guard (line {runner_order and runner_order[0]}) before it '
            f'reads the training data (line {runner_order and runner_order[1]})')
    g.check(not r7, 'R7: no trace_minmax call in the fork outside chang/ (the export trace is M18\'s, B14)'
                    + (f' ({r7})' if r7 else ''))
    return {'hits': hits, 'production_files': len(prod)}


def check_work_paths(work, *, hardlink_dir=None, temp_dir=None):
    """Refuse unsafe harness placement before creating any probe files."""
    import tempfile

    require_unsynced(work, hardlink_dir if hardlink_dir is not None else work)
    require_unsynced(temp_dir if temp_dir is not None else tempfile.gettempdir())


def runtime(g, a, rec):
    import shutil

    import fileaudit
    import runnergate as rg

    work = Path(a.work).resolve()
    check_work_paths(work, hardlink_dir=(work / 'hardlink-to-heldout.h5').parent)
    g.check(not work.exists(), f'work directory {work} does not exist yet')
    work.mkdir(parents=True)
    inp = rg.inputs(a.data_manifest, a.archive)
    man = json.loads(Path(a.data_manifest).read_text())
    got = rg.sha256_file(a.archive)
    g.check(got == inp['archive_sha256'], f'the archive is the smoke archive {got[:12]}')
    ph = work / 'placeholders'
    ph.mkdir()
    (ph / '150c-test.h5').write_bytes(b'')
    raw_name = man['heldout']['files'][0]['name']
    (ph / raw_name).write_bytes(b'')
    (ph / 'innocent.h5').symlink_to(ph / '150c-test.h5')
    held = Path(inp['heldout'])
    hl = work / 'hardlink-to-heldout.h5'
    os.link(held, hl)                                   # link(2) opens nothing; removed below
    watches = rg.watches(inp, ph) + [(ph / raw_name, 'placeholder')]
    kids = rg.Children(g, device='cpu', inp=inp, work=work, children_out=a.children_out, audit_watches=watches)
    d, _ = kids('draw-s1', 'draw', 1, ['--out-dir', work / 'draw-s1' / 'init', '--place', 'PCC gate G51'])
    if d is None:
        hl.unlink()
        return
    out = work / 'guard.json'
    argv = [sys.executable, '-I', '-B', str(FORK / 'gates' / 'resume_child.py'), 'guard', '--out', str(out),
            '--out-dir', str(work / 'run'), '--device', 'cpu', '--seed', '1', '--archive', inp['archive'],
            '--archive-kind', 'smoke', '--archive-sha', inp['archive_sha256'], '--data-manifest-sha',
            inp['data_manifest_sha256'], '--manifest', d['manifest'], '--manifest-sha', d['manifest_sha256'],
            '--code-sha', rec['header']['HEAD'], '--source-bundle', inp['bundle_sha256'], '--arm', 'CH-8', '--epochs', '1',
            '--placeholders', str(ph), '--raw-name', raw_name, '--heldout', str(held), '--hardlink', str(hl),
            '--raw-dir', inp['heldout_raw_dir']]
    env = {k: v for k, v in os.environ.items() if k not in ('KERAS_BACKEND', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
                                                           'XLA_FLAGS')}
    r = fileaudit.run_audited(argv, watches=watches, env=env, cwd=str(work), timeout=3600)
    hl.unlink()
    (work / 'guard.stdout.txt').write_text(r['stdout'])
    (work / 'guard.stderr.txt').write_text(r['stderr'])
    audit = {k: r[k] for k in ('events', 'events_before_start', 'fd_paths')}
    R = json.loads(out.read_text()) if out.exists() else None
    if a.children_out:
        co = Path(a.children_out)
        co.mkdir(parents=True, exist_ok=True)
        for f in (out, work / 'guard.stdout.txt', work / 'guard.stderr.txt'):
            if f.exists():
                shutil.copy2(f, co / f.name)
        (co / 'guard.audit.json').write_text(json.dumps(audit, indent=1, sort_keys=True) + '\n')
        (co / 'guard.argv.txt').write_text(' '.join(argv) + '\n')
    g.info('guard child argv', ' '.join(argv[3:]))
    ok = r['returncode'] == 0 and R is not None and 'error' not in R and (R.get('runner') or {}).get('end') == 'completed'
    g.check(ok, f'the guarded runner run completed with exit 0, the training archive read through the guard '
                f'({(R or {}).get("error") or ((R or {}).get("runner") or {}).get("error")})')
    if R is None:
        return
    gr = (R.get('runner') or {}).get('heldout_guard') or {}
    g.check(gr.get('installed') and gr.get('h5py_patched') and set(gr.get('names') or []) >= {'150c-test.h5', raw_name}
            and str(held) in (gr.get('known_inodes') or []),
            f'the runner installed the guard: {len(gr.get("names") or [])} names, raw directory {gr.get("dirs")}, '
            f'{len(gr.get("known_inodes") or [])} known inodes including the held-out archive, h5py patched')
    q = R.get('queries') or {}
    g.check(q.get('heldout') == 'held-out file name' and q.get('hardlink') == 'the inode of a held-out file'
            and q.get('raw_dir') == 'under the raw held-out directory' and q.get('train') is None,
            f'inside the training process the guard names the held-out archive, a hard link to it (by inode) and a path '
            f'under the raw held-out directory, and passes the training archive (no open made): {q}')
    at = R.get('attempts') or {}
    want = ('open', 'os.open', 'np.load', 'h5py.File', 'open via symlink', 'h5py.File raw name')
    g.check(set(at) == set(want) and all(v.get('refused') and v.get('type') == 'HeldOutAccessRefused' for v in at.values()),
            f'inside the training process every open attempt on the placeholders is refused with HeldOutAccessRefused: '
            f'{ {k: v.get("type") for k, v in at.items()} }')
    ph_events = [e for e in audit['events'] if e['label'] == 'placeholder']
    fe = rg.forbidden_events(audit)
    g.check(not ph_events and not fe, f'the kernel audit saw no open of a placeholder ({len(ph_events)}) and no open of the '
                                      f'held-out archive or a raw held-out file ({len(fe)}): every refusal came before the open')
    g.check(len(gr.get('refused') or []) == 0 and len(R.get('guard_refused_after') or []) == len(want),
            f'the guard\'s own record lists the {len(R.get("guard_refused_after") or [])} refused attempts of the probe '
            f'and none during training')
    # control: an unguarded process opening the placeholder is seen by the audit
    rc = fileaudit.run_audited([sys.executable, '-I', '-B', '-c', f'open({str(ph / "150c-test.h5")!r}, "rb").close()'],
                               watches=watches, env=env, cwd=str(work), timeout=120)
    g.check(rc['returncode'] == 0 and any(e['label'] == 'placeholder' for e in rc['events']),
            'control: an unguarded process that opens the placeholder is seen by the kernel audit')
    rec['runtime'] = {'queries': q, 'attempts': at, 'guard': gr}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--place', choices=['mac', 'pcc'], required=True)
    ap.add_argument('--expect-sha12', required=True)
    ap.add_argument('--work')
    ap.add_argument('--archive')
    ap.add_argument('--data-manifest', default=str(FORK / 'manifests' / 'data-prep-150c.json'))
    ap.add_argument('--json', default=None)
    ap.add_argument('--children-out', default=None)
    a = ap.parse_args()
    if a.place == 'pcc':
        check_work_paths(a.work)
    g = gatelib.Gate(GATE)
    rec = {'header': g.header(a.expect_sha12), 'place': a.place}
    print('label       engineering, not quotable')
    rec['static'] = static_audit(g)
    if a.place == 'pcc':
        gatelib.check_venv_of_record(g)
        runtime(g, a, rec)
    g.check_clean('at end')
    return g.finish(rec, a.json)


if __name__ == '__main__':
    sys.exit(main())
