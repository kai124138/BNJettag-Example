"""M2: reference identity, the HGQ2 snapshot against the installed hgq2 0.1.9.

STUDY v0.8 section 7.3 "Reference identity (chang C17)"; `review/chang-v0.3.md` C17 and C18;
PREFLIGHT-PLAN section 2 row M2 and section 3 row G04. Standard library only: no Keras, JAX or
hgq import, so it runs on any interpreter. The installed tree it reads is the one passed in (for
gate G04, the `hgq` package of `~/lab/venvs/cbl1016-jax2`).

What it does:
1. Re-hashes both trees. The snapshot `reference-code/HGQ2/src` by the tree-hash method of
   `code-base.txt` (compared with the tree line there), and its `hgq/` part alone. The installed
   `hgq/` and `qkeras/` files as a sorted listing in the format of
   `env/training/hgq2-0.1.9-installed-files.sha256` (compared byte for byte with that record), and
   each file against the wheel's `RECORD`.
2. Compares the two trees file by file and splits every differing file into hunks
   (difflib.SequenceMatcher on lines, installed 0.1.9 as the old side, the snapshot as the new
   side). A hunk id is the first 16 hex digits of the sha256 of its file, kind and text on both
   sides, so it does not depend on line numbers.
3. Checks every module section 7.3 cites against its expectation: the "differ" list, the
   "identical" list, and the files the "Two shims" paragraph cites. A cited "identical" module
   that differs, or a cited "differ" module that is identical, fails.
4. Requires every hunk of a cited file to be classified in the registry
   (`cbl/refdiff_known.json`: behavioral, addition or cosmetic, with the mechanism and a note); an
   unclassified hunk or a registry entry that matches no hunk fails, so a changed tree cannot pass
   silently.
5. Checks the hgq line citations of section 7.3 against the cited tree and line.
6. Writes `report.json` (machine-readable), `refdiff.md` (the short human list of each behavioral
   difference in a cited mechanism) and `diffs/<file>.diff` into the output directory.

Run:  python -I -B cbl/refdiff.py --snapshot <lab>/reference-code/HGQ2/src --site-packages <dir> --out <dir>
Exit 0 when nothing fails (findings are listed either way), 1 otherwise.
"""

import argparse
import ast
import base64
import difflib
import hashlib
import json
import sys
from pathlib import Path

FORK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWN = FORK_ROOT / 'cbl' / 'refdiff_known.json'
DEFAULT_RECORD = FORK_ROOT / 'env' / 'training' / 'hgq2-0.1.9-installed-files.sha256'
DEFAULT_CODE_BASE = FORK_ROOT / 'code-base.txt'
SNAPSHOT_TREE_SOURCE = 'reference-code/HGQ2/src'  # the tree line of code-base.txt
INSTALLED_TOPS = ('hgq', 'qkeras')
INSTALLED_DIST = 'hgq2-0.1.9.dist-info'
INSTALLED_VERSION = '0.1.9'

# STUDY v0.8 section 7.3, "Reference identity": the chang review's C17 lists, with each short
# name resolved to a path under the package root. `dir` entries cover every file below them.
CITED = [
    # the "differ" list
    {'cite': 'layers/core/einsum_dense.py', 'expect': 'differ', 'paths': ['hgq/layers/core/einsum_dense.py'],
     'study': '0.1.9 has no parallelization factor (section 3.1a)'},
    {'cite': 'layers/core/base.py', 'expect': 'differ', 'paths': ['hgq/layers/core/base.py'],
     'study': "0.1.9 wraps every QLayer's build and evaluates its q<var> properties right after it (check_collapsed_weight, installed 0.1.9 base.py:118-140)"},
    {'cite': 'layers/attn/mha.py', 'expect': 'differ', 'paths': ['hgq/layers/attn/mha.py'],
     'study': 'the snapshot only adds QMultiHeadAttentionT'},
    {'cite': 'quantizer/quantizer.py', 'expect': 'differ', 'paths': ['hgq/quantizer/quantizer.py'],
     'study': 'serialization differs'},
    {'cite': 'minmax_trace.py', 'expect': 'differ', 'paths': ['hgq/utils/minmax_trace.py'],
     'study': 'a verbose print'},
    {'cite': 'config/layer.py', 'expect': 'differ', 'paths': ['hgq/config/layer.py'],
     'study': 'the snapshot adds enable_sq'},
    {'cite': 'utils/sugar', 'expect': 'differ', 'dir': 'hgq/utils/sugar',
     'study': 'StopIf'},
    # the "identical" list
    {'cite': 'einsum_dense_batchnorm.py', 'expect': 'identical', 'paths': ['hgq/layers/einsum_dense_batchnorm.py']},
    {'cite': 'activation.py', 'expect': 'identical', 'paths': ['hgq/layers/activation.py'],
     'resolution': "three files are named activation.py; the C17 list covers the files the xfm model uses, which is "
                   "hgq/layers/activation.py (QAffinedUnaryFunctionLUT, model.py:8); _dais_tracer/layers/activation.py is "
                   "covered by the _dais_tracer/ entry; _alkaid_keras_plugin/activation.py is not cited"},
    {'cite': 'softmax.py', 'expect': 'identical', 'paths': ['hgq/layers/softmax.py']},
    {'cite': 'ops/merge.py', 'expect': 'identical', 'paths': ['hgq/layers/ops/merge.py']},
    {'cite': 'beta_pid.py', 'expect': 'identical', 'paths': ['hgq/utils/sugar/beta_pid.py']},
    {'cite': 'beta_scheduler.py', 'expect': 'identical', 'paths': ['hgq/utils/sugar/beta_scheduler.py']},
    {'cite': 'ebops.py', 'expect': 'identical', 'paths': ['hgq/utils/sugar/ebops.py']},
    {'cite': 'pareto.py', 'expect': 'identical', 'paths': ['hgq/utils/sugar/pareto.py']},
    {'cite': 'dataset.py', 'expect': 'identical', 'paths': ['hgq/utils/dataset.py']},
    {'cite': 'quantizer/config.py', 'expect': 'identical', 'paths': ['hgq/quantizer/config.py']},
    {'cite': 'internal/fixed_point_quantizer.py', 'expect': 'identical', 'paths': ['hgq/quantizer/internal/fixed_point_quantizer.py']},
    {'cite': '_dais_tracer/', 'expect': 'identical', 'dir': 'hgq/_dais_tracer'},
    # the "Two shims" paragraph
    {'cite': 'hgq/layers/__init__.py', 'expect': 'differ', 'paths': ['hgq/layers/__init__.py'],
     'study': 'exports QLinformerAttention only in 0.1.9 (shim 2)'},
    {'cite': 'layers/attn/linformer.py:183', 'expect': 'differ', 'paths': ['hgq/layers/attn/linformer.py'],
     'study': 'QLinformerAttentionT exists in the snapshot only (shim 2)'},
    {'cite': 'layers/attn/__init__.py', 'expect': 'differ', 'paths': ['hgq/layers/attn/__init__.py'],
     'study': 'the re-export path of QLinformerAttentionT into hgq.layers (shim 2; not named in section 7.3, '
              'included because hgq/layers/__init__.py imports the name from it)'},
]

# The hgq line citations of STUDY v0.8 section 7.3, checked in the tree they name.
CITATIONS = [
    {'tree': 'installed', 'file': 'hgq/layers/core/base.py', 'lines': [118, 140], 'expect': 'check_collapsed_weight',
     'study': 'section 7.3 reference identity: installed 0.1.9 base.py:118-140'},
    {'tree': 'snapshot', 'file': 'hgq/layers/attn/linformer.py', 'lines': [183, 183], 'expect': 'class QLinformerAttentionT',
     'study': 'section 7.3 shim 2: snapshot linformer.py:183'},
    {'tree': 'installed', 'file': 'hgq/layers/einsum_dense_batchnorm.py', 'lines': [103, 104], 'expect': 'kernel_shape, _norm_bias_cast_shape, output_shape',
     'study': 'section 7.3 fallback: the fused-BN einsum-shape unpacking, 0.1.9 einsum_dense_batchnorm.py:103-104'},
    {'tree': 'installed', 'file': 'hgq/layers/core/einsum_dense.py', 'lines': [75, 75], 'expect': 'full_output_shape',
     'study': 'section 7.3 fallback: full_output_shape, 0.1.9 core/einsum_dense.py:75'},
    {'tree': 'installed', 'file': 'hgq/layers/attn/mha.py', 'lines': [387, 389], 'expect': 'full_output_shape',
     'study': 'section 7.3 fallback: full_output_shape, 0.1.9 attn/mha.py:387-389'},
]
# section 7.3 shim 1: the snapshot stop_if.py "imports only Keras"
STOPIF_FILE = 'hgq/utils/sugar/stop_if.py'

KINDS = ('behavioral', 'addition', 'cosmetic')


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def walk(root, tops=None):
    """{relpath: sha256} of regular files under root (or under root/<top> for each top), skipping
    symbolic links, __pycache__/ and *.pyc, relpaths with '/' separators relative to root."""
    root = Path(root)
    out = {}
    bases = [root / t for t in tops] if tops else [root]
    for base in bases:
        if not base.is_dir():
            continue
        for p in base.rglob('*'):
            rel = p.relative_to(root)
            if '__pycache__' in rel.parts or p.name.endswith('.pyc'):
                continue
            if p.is_symlink() or not p.is_file():
                continue
            out[rel.as_posix()] = sha256_bytes(p.read_bytes())
    return out


def listing(files):
    """The listing format of code-base.txt and of the installed-files record: one line
    '<sha256>  <relpath>' per file, sorted by relpath as UTF-8 bytes."""
    return ''.join('%s  %s\n' % (files[r], r) for r in sorted(files, key=lambda s: s.encode('utf-8')))


def tree_hash(files):
    return sha256_bytes(listing(files).encode('utf-8'))


def parse_code_base_tree(path):
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        parts = line.split('  ')
        if parts and parts[0] == 'tree' and len(parts) == 4:
            return {'hash': parts[1], 'source': parts[2], 'files': int(parts[3].split('=', 1)[1])}
    raise ValueError(f'no tree line in {path}')


def check_record(site_packages, files):
    """Each walked file against the wheel's RECORD (urlsafe base64 sha256), and the two path sets."""
    record = Path(site_packages) / INSTALLED_DIST / 'RECORD'
    rows = {}
    for line in record.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        path, digest, _size = line.rsplit(',', 2)
        if path.split('/', 1)[0] in INSTALLED_TOPS and not path.endswith('.pyc') and '__pycache__' not in path:
            rows[path] = digest
    mismatched = []
    for path, digest in rows.items():
        if path not in files:
            continue
        algo, _, b64 = digest.partition('=')
        want = base64.urlsafe_b64decode(b64 + '=' * (-len(b64) % 4)).hex() if algo == 'sha256' else None
        if want != files[path]:
            mismatched.append(path)
    return {
        'record_file': str(record),
        'record_entries': len(rows),
        'only_in_record': sorted(set(rows) - set(files)),
        'only_on_disk': sorted(set(files) - set(rows)),
        'hash_mismatch': sorted(mismatched),
        'ok': not mismatched and set(rows) == set(files),
    }


def symbol_spans(text):
    """[(start, end, qualname)] of every class and function in a Python source, 1-based lines."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    spans = []

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                spans.append((start, child.end_lineno, name))
                visit(child, name + '.')
            else:
                visit(child, prefix)

    visit(tree, '')
    return spans


def enclosing(spans, line):
    best = None
    for start, end, name in spans:
        if start <= line <= end and (best is None or start >= best[0]):
            best = (start, end, name)
    return best[2] if best else '<module>'


def hunks_for(relpath, old_text, new_text):
    """Hunks from installed (old) to snapshot (new). One-sided files give one hunk."""
    a = old_text.splitlines(keepends=True) if old_text is not None else []
    b = new_text.splitlines(keepends=True) if new_text is not None else []
    sa = symbol_spans(old_text) if old_text is not None and relpath.endswith('.py') else []
    sb = symbol_spans(new_text) if new_text is not None and relpath.endswith('.py') else []
    out, seen = [], {}
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        ta, tb = ''.join(a[i1:i2]), ''.join(b[j1:j2])
        base = sha256_bytes(('%s\0%s\0%s\0%s' % (relpath, tag, ta, tb)).encode('utf-8'))[:16]
        k = seen.get(base, 0)
        seen[base] = k + 1
        hid = base if k == 0 else '%s-%d' % (base, k)
        out.append({
            'id': hid,
            'tag': tag,
            'installed_lines': [i1 + 1, i2] if i2 > i1 else [i1, i1],
            'snapshot_lines': [j1 + 1, j2] if j2 > j1 else [j1, j1],
            'installed_symbol': enclosing(sa, i1 + 1 if i2 > i1 else max(i1, 1)) if sa else '<module>',
            'snapshot_symbol': enclosing(sb, j1 + 1 if j2 > j1 else max(j1, 1)) if sb else '<module>',
            'installed_text': ta,
            'snapshot_text': tb,
        })
    return out


def compare(snapshot_src, site_packages):
    snap = walk(snapshot_src, INSTALLED_TOPS)
    inst = walk(site_packages, INSTALLED_TOPS)
    files = {}
    for rel in sorted(set(snap) | set(inst), key=lambda s: s.encode('utf-8')):
        if rel in snap and rel in inst:
            status = 'identical' if snap[rel] == inst[rel] else 'differ'
        else:
            status = 'snapshot_only' if rel in snap else 'installed_only'
        entry = {'status': status, 'snapshot_sha256': snap.get(rel), 'installed_sha256': inst.get(rel)}
        if status != 'identical':
            old = (Path(site_packages) / rel).read_text(encoding='utf-8') if rel in inst else None
            new = (Path(snapshot_src) / rel).read_text(encoding='utf-8') if rel in snap else None
            entry['hunks'] = hunks_for(rel, old, new)
            entry['unified_diff'] = ''.join(difflib.unified_diff(
                old.splitlines(keepends=True) if old is not None else [],
                new.splitlines(keepends=True) if new is not None else [],
                fromfile='installed-0.1.9/' + rel, tofile='snapshot/' + rel))
        files[rel] = entry
    return snap, inst, files


def cited_paths(item, files):
    if 'dir' in item:
        prefix = item['dir'].rstrip('/') + '/'
        return sorted(r for r in files if r.startswith(prefix))
    return list(item['paths'])


def short_name_candidates(item, files):
    """Every path whose tail equals the cited short name, so the report shows how it was resolved."""
    if 'dir' in item:
        return [item['dir']]
    short = item['cite'].split(':', 1)[0]
    return sorted(r for r in files if r == short or r.endswith('/' + short))


def check_citations(snapshot_src, site_packages):
    roots = {'snapshot': Path(snapshot_src), 'installed': Path(site_packages)}
    out = []
    for c in CITATIONS:
        lines = (roots[c['tree']] / c['file']).read_text(encoding='utf-8').splitlines()
        lo, hi = c['lines']
        cited = '\n'.join(lines[lo - 1:hi])
        found = [n + 1 for n, s in enumerate(lines) if c['expect'] in s]
        other = 'installed' if c['tree'] == 'snapshot' else 'snapshot'
        olines = (roots[other] / c['file']).read_text(encoding='utf-8').splitlines()
        ofound = [n + 1 for n, s in enumerate(olines) if c['expect'] in s]
        out.append(dict(c, ok=c['expect'] in cited, cited_text=cited, found_at=found, found_in_other_tree=ofound,
                        other_tree=other))
    # shim 1: stop_if.py imports only Keras (besides the standard library)
    tree = ast.parse((Path(snapshot_src) / STOPIF_FILE).read_text(encoding='utf-8'))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split('.')[0])
    third = sorted(m for m in mods if m not in sys.stdlib_module_names)
    out.append({'tree': 'snapshot', 'file': STOPIF_FILE, 'lines': None, 'expect': "only 'keras' outside the standard library",
                'study': 'section 7.3 shim 1: stop_if.py imports only Keras', 'ok': third == ['keras'],
                'imports': sorted(mods), 'non_stdlib_imports': third})
    return out


def load_known(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    entries = data['hunks']
    for e in entries:
        if e.get('kind') not in KINDS:
            raise ValueError(f'registry entry {e.get("file")} {e.get("id")}: kind {e.get("kind")!r} not in {KINDS}')
        for key in ('file', 'id', 'mechanism', 'note', 'in_study'):
            if key not in e:
                raise ValueError(f'registry entry {e.get("file")} {e.get("id")}: missing {key}')
    return data


def evaluate(snapshot_src, site_packages, record_path=DEFAULT_RECORD, code_base_path=DEFAULT_CODE_BASE,
             known_path=DEFAULT_KNOWN):
    snapshot_src, site_packages = Path(snapshot_src).resolve(), Path(site_packages).resolve()
    failures, findings = [], []

    # 1. re-hash both trees
    snap_all = walk(snapshot_src)
    tree_rec = parse_code_base_tree(code_base_path)
    snap_hash = {
        'tree_hash': tree_hash(snap_all), 'files': len(snap_all),
        'recorded_hash': tree_rec['hash'], 'recorded_files': tree_rec['files'], 'recorded_source': tree_rec['source'],
        'hgq_tree_hash': tree_hash(walk(snapshot_src, ['hgq'])),
    }
    snap_hash['ok'] = snap_hash['tree_hash'] == tree_rec['hash'] and snap_hash['files'] == tree_rec['files']
    if not snap_hash['ok']:
        failures.append('snapshot tree hash %s (%d files) != code-base.txt %s (%d files)' % (
            snap_hash['tree_hash'], snap_hash['files'], tree_rec['hash'], tree_rec['files']))

    snap, inst, files = compare(snapshot_src, site_packages)
    inst_listing = listing(inst)
    record_text = Path(record_path).read_text(encoding='utf-8')
    inst_hash = {
        'listing_sha256': sha256_bytes(inst_listing.encode('utf-8')), 'files': len(inst),
        'record_path': str(record_path), 'record_sha256': sha256_bytes(record_text.encode('utf-8')),
        'record_lines': len(record_text.splitlines()), 'listing_equals_record': inst_listing == record_text,
    }
    inst_hash['wheel_record'] = check_record(site_packages, inst)
    if not inst_hash['listing_equals_record']:
        failures.append('installed hgq/qkeras listing differs from %s' % record_path)
    if not inst_hash['wheel_record']['ok']:
        failures.append('installed files differ from the wheel RECORD: %s' % inst_hash['wheel_record'])
    version_file = site_packages / 'hgq' / '_version.py'
    inst_hash['hgq_version_py'] = version_file.read_text(encoding='utf-8').strip() if version_file.is_file() else None
    dist = site_packages / INSTALLED_DIST
    inst_hash['dist_info_present'] = dist.is_dir()
    if not dist.is_dir():
        failures.append('%s not found under %s' % (INSTALLED_DIST, site_packages))

    # 2-4. cited modules and the registry
    known = load_known(known_path)
    reg = {(e['file'], e['id']): e for e in known['hunks']}
    used = set()
    cited_report, cited_files = [], set()
    for item in CITED:
        paths = cited_paths(item, files)
        rows = []
        for p in paths:
            f = files.get(p)
            if f is None:
                failures.append('cited %s: %s is in neither tree' % (item['cite'], p))
                rows.append({'path': p, 'status': 'missing'})
                continue
            cited_files.add(p)
            row = {'path': p, 'status': f['status'], 'hunks': []}
            for h in f.get('hunks', []):
                e = reg.get((p, h['id']))
                if e is None:
                    failures.append('cited %s: unclassified hunk %s in %s (installed %s, snapshot %s)' % (
                        item['cite'], h['id'], p, h['installed_lines'], h['snapshot_lines']))
                else:
                    used.add((p, h['id']))
                row['hunks'].append({k: h[k] for k in ('id', 'tag', 'installed_lines', 'snapshot_lines',
                                                       'installed_symbol', 'snapshot_symbol')}
                                    | ({'kind': e['kind'], 'mechanism': e['mechanism'], 'note': e['note'],
                                        'in_study': e['in_study']} if e else {'kind': None}))
            rows.append(row)
        statuses = {r['status'] for r in rows}
        if not paths:
            failures.append('cited %s: no file found' % item['cite'])
            outcome = 'missing'
        elif item['expect'] == 'identical':
            outcome = 'identical' if statuses == {'identical'} else 'differ'
            if outcome != 'identical':
                failures.append('cited %s is in the section 7.3 "identical" list but differs: %s' % (
                    item['cite'], [r['path'] for r in rows if r['status'] != 'identical']))
        else:
            outcome = 'differ' if statuses - {'identical'} else 'identical'
            if outcome != 'differ':
                failures.append('cited %s is expected to differ but is identical' % item['cite'])
        cited_report.append({k: item[k] for k in item if k != 'paths'} | {
            'paths': paths, 'candidates': short_name_candidates(item, files), 'outcome': outcome, 'files': rows})
    stale = sorted(set(reg) - used)
    for f, h in stale:
        failures.append('registry entry %s %s matches no hunk of a cited file' % (f, h))

    for c in cited_report:
        for r in c['files']:
            for h in r.get('hunks', []):
                if h.get('kind') == 'behavioral':
                    findings.append('behavioral difference in %s (%s), %s: %s [%s]' % (
                        r['path'], h['mechanism'], 'named in STUDY section 7.3' if h['in_study'] else 'NOT named in STUDY section 7.3',
                        h['note'], h['id']))

    # 5. line citations
    citations = check_citations(snapshot_src, site_packages)
    for c in citations:
        if not c['ok']:
            findings.append('citation %s: %s %s lines %s do not contain %r (found in that tree at %s; in the %s tree at %s)' % (
                c['study'], c['tree'], c['file'], c['lines'], c['expect'], c.get('found_at'), c.get('other_tree'),
                c.get('found_in_other_tree')))

    uncited = [{'path': r, 'status': f['status'], 'hunks': len(f.get('hunks', []))}
               for r, f in files.items() if f['status'] != 'identical' and r not in cited_files]

    summary = {
        'files_compared': len(files),
        'identical': sum(1 for f in files.values() if f['status'] == 'identical'),
        'differ': sum(1 for f in files.values() if f['status'] == 'differ'),
        'snapshot_only': sorted(r for r, f in files.items() if f['status'] == 'snapshot_only'),
        'installed_only': sorted(r for r, f in files.items() if f['status'] == 'installed_only'),
        'cited_hunks': sum(len(r.get('hunks', [])) for c in cited_report for r in c['files']),
        'registry_entries': len(reg),
    }
    return {
        'module': 'cbl/refdiff.py',
        'snapshot_src': str(snapshot_src),
        'site_packages': str(site_packages),
        'known_path': str(known_path),
        'known_sha256': sha256_bytes(Path(known_path).read_bytes()),
        'snapshot': snap_hash,
        'installed': inst_hash,
        'summary': summary,
        'cited': cited_report,
        'citations': citations,
        'uncited_differing': uncited,
        'failures': failures,
        'findings': findings,
        'files': files,
    }


def render_md(rep):
    s, i = rep['snapshot'], rep['installed']
    L = ['# Reference identity: HGQ2 snapshot against installed hgq2 0.1.9', '',
         'Written by `cbl/refdiff.py` (M2). Old side: installed 0.1.9; new side: the snapshot. Engineering record, not quotable.', '',
         '## Hashes', '',
         '- snapshot `%s`: tree hash `%s`, %d files; code-base.txt records `%s`, %d files: %s' % (
             s['recorded_source'], s['tree_hash'], s['files'], s['recorded_hash'], s['recorded_files'], 'equal' if s['ok'] else 'DIFFERENT'),
         '- snapshot `hgq/` alone: tree hash `%s`' % s['hgq_tree_hash'],
         '- installed `hgq/` + `qkeras/`: listing sha256 `%s`, %d files; record `%s` (%d lines): %s' % (
             i['listing_sha256'], i['files'], i['record_sha256'], i['record_lines'], 'byte-identical' if i['listing_equals_record'] else 'DIFFERENT'),
         '- wheel RECORD: %d entries, hash mismatches %d, path sets %s' % (
             i['wheel_record']['record_entries'], len(i['wheel_record']['hash_mismatch']),
             'equal' if not (i['wheel_record']['only_in_record'] or i['wheel_record']['only_on_disk']) else 'DIFFERENT'),
         '', '## Cited modules (STUDY v0.8 section 7.3)', '']
    for c in rep['cited']:
        res = ' Resolution: %s.' % c['resolution'] if c.get('resolution') else ''
        L.append('- `%s`: expected %s, found %s.%s' % (c['cite'], c['expect'], c['outcome'], res))
        for r in c['files']:
            if r['status'] == 'identical':
                continue
            L.append('  - `%s` (%s)' % (r['path'], r['status']))
            for h in r['hunks']:
                if h.get('kind') in ('behavioral', 'addition'):
                    L.append('    - %s, %s: %s (installed %s, snapshot %s; %s)%s' % (
                        h['kind'], h['mechanism'], h['note'], h['installed_lines'], h['snapshot_lines'], h['id'],
                        '' if h['in_study'] else ' [not named in section 7.3]'))
            n_cos = sum(1 for h in r['hunks'] if h.get('kind') == 'cosmetic')
            if n_cos:
                L.append('    - %d cosmetic hunk(s)' % n_cos)
    L += ['', '## Behavioral differences in cited mechanisms', '']
    beh = [f for f in rep['findings'] if f.startswith('behavioral')]
    L += ['- ' + f for f in beh] or ['- none']
    L += ['', '## Line citations of section 7.3', '']
    for c in rep['citations']:
        L.append('- %s: %s (%s `%s`%s)' % (c['study'], 'ok' if c['ok'] else 'MISMATCH', c['tree'], c['file'],
                                         ' lines %s' % c['lines'] if c['lines'] else ''))
        if not c['ok'] and c.get('found_at') is not None:
            L.append('  - %r is at %s in the %s tree and at %s in the %s tree' % (
                c['expect'], c['found_at'], c['tree'], c['found_in_other_tree'], c['other_tree']))
    L += ['', '## Files that differ and are not cited', '']
    L += ['- `%s` (%s, %d hunk(s))' % (u['path'], u['status'], u['hunks']) for u in rep['uncited_differing']] or ['- none']
    L += ['', '## Result', '']
    L += ['- FAIL: ' + f for f in rep['failures']] or ['- no failure']
    L.append('- findings: %d' % len(rep['findings']))
    return '\n'.join(L) + '\n'


def write_outputs(rep, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(rep, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    (out / 'refdiff.md').write_text(render_md(rep), encoding='utf-8')
    for rel, f in rep['files'].items():
        if f.get('unified_diff'):
            p = out / 'diffs' / (rel + '.diff')
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f['unified_diff'], encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--snapshot', required=True, help='the snapshot source root, <lab>/reference-code/HGQ2/src')
    ap.add_argument('--site-packages', required=True, help='the directory holding the installed hgq/ and hgq2 dist-info')
    ap.add_argument('--out', required=True)
    ap.add_argument('--record', default=str(DEFAULT_RECORD))
    ap.add_argument('--code-base', default=str(DEFAULT_CODE_BASE))
    ap.add_argument('--known', default=str(DEFAULT_KNOWN))
    ap.add_argument('--dump-hunks', action='store_true', help='print every hunk of every differing file and exit')
    a = ap.parse_args(argv)
    if a.dump_hunks:
        _, _, files = compare(a.snapshot, a.site_packages)
        for rel, f in files.items():
            for h in f.get('hunks', []):
                print(json.dumps({'file': rel, 'status': f['status']} | {k: h[k] for k in h if not k.endswith('_text')}))
        return 0
    rep = evaluate(a.snapshot, a.site_packages, a.record, a.code_base, a.known)
    write_outputs(rep, a.out)
    print(render_md(rep))
    return 1 if rep['failures'] else 0


if __name__ == '__main__':
    sys.exit(main())
