"""Tests of M2 `cbl/refdiff.py` on small constructed trees. Standard library only."""

import base64
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
from cbl import refdiff  # noqa: E402


def write(root, rel, text):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')
    return p


def load_g01():
    spec = importlib.util.spec_from_file_location('g01_base', FORK / 'gates' / 'g01_base.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tree_hash_matches_the_code_base_method_of_g01(tmp_path):
    write(tmp_path, 'b/z.py', 'z\n')
    write(tmp_path, 'a.py', 'a\n')
    write(tmp_path, 'B.py', 'B\n')
    write(tmp_path, 'pkg/__pycache__/a.cpython-312.pyc', 'x')
    write(tmp_path, 'pkg/m.pyc', 'x')
    os.symlink(tmp_path / 'a.py', tmp_path / 'link.py')
    files = refdiff.walk(tmp_path)
    assert sorted(files) == ['B.py', 'a.py', 'b/z.py']
    assert (refdiff.tree_hash(files), len(files)) == load_g01().tree_hash(tmp_path)
    # sorted by path as UTF-8 bytes: 'B.py' < 'a.py' < 'b/z.py'
    assert [line.split('  ')[1] for line in refdiff.listing(files).splitlines()] == ['B.py', 'a.py', 'b/z.py']


def test_hunks_lines_symbols_and_stable_ids():
    old = 'import a\n\n\nclass K:\n    def f(self):\n        return 1\n'
    new = 'import a\nimport b\n\n\nclass K:\n    def f(self):\n        return 2\n'
    hs = refdiff.hunks_for('m.py', old, new)
    assert [h['tag'] for h in hs] == ['insert', 'replace']
    assert hs[0]['installed_lines'] == [1, 1] and hs[0]['snapshot_lines'] == [2, 2]
    assert hs[1]['installed_lines'] == [6, 6] and hs[1]['snapshot_lines'] == [7, 7]
    assert hs[1]['installed_symbol'] == 'K.f' and hs[1]['snapshot_symbol'] == 'K.f'
    # the id depends on content, not on position
    shifted = refdiff.hunks_for('m.py', '# c\n' + old, '# c\n' + new)
    assert [h['id'] for h in shifted] == [h['id'] for h in hs]
    assert refdiff.hunks_for('m.py', old, old) == []
    one_sided = refdiff.hunks_for('m.py', None, new)
    assert len(one_sided) == 1 and one_sided[0]['tag'] == 'insert' and one_sided[0]['snapshot_lines'] == [1, 7]


def test_duplicate_hunks_get_distinct_ids():
    old = 'a\nx\nb\nx\nc\n'
    new = 'a\ny\nb\ny\nc\n'
    ids = [h['id'] for h in refdiff.hunks_for('d.txt', old, new)]
    assert len(ids) == 2 and len(set(ids)) == 2 and ids[1] == ids[0] + '-1'


def b64(text):
    return base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).decode().rstrip('=')


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A snapshot src/, a site-packages with RECORD, a code-base file, a listing record and a registry."""
    snap, site = tmp_path / 'src', tmp_path / 'site'
    same = 'def f():\n    return 1\n'
    files_s = {
        'hgq/same.py': same,
        'hgq/changed.py': 'def g():\n    return 2\n',
        'hgq/cite.py': 'x = 1\ny = 2\n',
        'hgq/stop_if.py': 'import keras\nfrom typing import Any\n',
        'qkeras/__init__.py': '',
    }
    files_i = {
        'hgq/same.py': same,
        'hgq/changed.py': 'def g():\n    return 1\n',
        'hgq/cite.py': 'x = 1\ny = 2\n',
        'qkeras/__init__.py': '',
    }
    for r, t in files_s.items():
        write(snap, r, t)
    for r, t in files_i.items():
        write(site, r, t)
    write(snap, 'README', 'not under hgq/ but in the tree hash\n')
    dist = site / refdiff.INSTALLED_DIST
    dist.mkdir()
    (dist / 'RECORD').write_text(''.join('%s,sha256=%s,%d\n' % (r, b64(t), len(t)) for r, t in files_i.items())
                                 + '%s/RECORD,,\n' % refdiff.INSTALLED_DIST)
    snap_all = refdiff.walk(snap)
    code_base = tmp_path / 'code-base.txt'
    code_base.write_text('# c\ntree  %s  reference-code/HGQ2/src  files=%d\n' % (refdiff.tree_hash(snap_all), len(snap_all)))
    record = tmp_path / 'installed.sha256'
    record.write_text(refdiff.listing(refdiff.walk(site, refdiff.INSTALLED_TOPS)))
    changed = refdiff.hunks_for('hgq/changed.py', files_i['hgq/changed.py'], files_s['hgq/changed.py'])
    stop = refdiff.hunks_for('hgq/stop_if.py', None, files_s['hgq/stop_if.py'])
    known = tmp_path / 'known.json'
    entries = [
        {'file': 'hgq/changed.py', 'id': changed[0]['id'], 'kind': 'behavioral', 'mechanism': 'g', 'note': 'returns 2', 'in_study': True},
        {'file': 'hgq/stop_if.py', 'id': stop[0]['id'], 'kind': 'addition', 'mechanism': 'StopIf', 'note': 'snapshot only', 'in_study': True},
    ]
    known.write_text(json.dumps({'schema': 1, 'hunks': entries}))
    monkeypatch.setattr(refdiff, 'CITED', [
        {'cite': 'changed.py', 'expect': 'differ', 'paths': ['hgq/changed.py']},
        {'cite': 'same.py', 'expect': 'identical', 'paths': ['hgq/same.py']},
        {'cite': 'stop_if.py', 'expect': 'differ', 'paths': ['hgq/stop_if.py']},
    ])
    monkeypatch.setattr(refdiff, 'CITATIONS', [
        {'tree': 'installed', 'file': 'hgq/cite.py', 'lines': [2, 2], 'expect': 'y = 2', 'study': 'ok one'},
        {'tree': 'snapshot', 'file': 'hgq/cite.py', 'lines': [1, 1], 'expect': 'y = 2', 'study': 'wrong line'},
    ])
    monkeypatch.setattr(refdiff, 'STOPIF_FILE', 'hgq/stop_if.py')
    return {'snap': snap, 'site': site, 'record': record, 'code_base': code_base, 'known': known,
            'entries': entries, 'tmp': tmp_path}


def run(world, **over):
    w = dict(world, **over)
    return refdiff.evaluate(w['snap'], w['site'], w['record'], w['code_base'], w['known'])


def test_consistent_world_passes_and_lists_findings(world):
    rep = run(world)
    assert rep['failures'] == []
    assert rep['snapshot']['ok'] and rep['installed']['listing_equals_record'] and rep['installed']['wheel_record']['ok']
    beh = [f for f in rep['findings'] if f.startswith('behavioral')]
    assert len(beh) == 1 and 'hgq/changed.py' in beh[0]
    cit = [f for f in rep['findings'] if f.startswith('citation')]
    assert len(cit) == 1 and 'wrong line' in cit[0]
    assert rep['summary']['snapshot_only'] == ['hgq/stop_if.py']
    refdiff.write_outputs(rep, world['tmp'] / 'out')
    assert (world['tmp'] / 'out' / 'diffs' / 'hgq' / 'changed.py.diff').is_file()
    assert 'returns 2' in (world['tmp'] / 'out' / 'refdiff.md').read_text()


def test_identical_list_module_that_differs_fails(world, monkeypatch):
    write(world['site'], 'hgq/same.py', 'def f():\n    return 3\n')
    # keep the installed records consistent, so only the cited-module rule can fail
    world['record'].write_text(refdiff.listing(refdiff.walk(world['site'], refdiff.INSTALLED_TOPS)))
    rec = world['site'] / refdiff.INSTALLED_DIST / 'RECORD'
    rec.write_text(rec.read_text().replace(b64('def f():\n    return 1\n'), b64('def f():\n    return 3\n')))
    rep = run(world)
    assert any('"identical" list but differs' in f for f in rep['failures'])
    assert any('unclassified hunk' in f and 'hgq/same.py' in f for f in rep['failures'])


def test_unclassified_and_stale_registry_entries_fail(world):
    entries = [dict(world['entries'][0], id='0' * 16), world['entries'][1]]
    world['known'].write_text(json.dumps({'schema': 1, 'hunks': entries}))
    rep = run(world)
    assert any('unclassified hunk' in f and 'hgq/changed.py' in f for f in rep['failures'])
    assert any('matches no hunk' in f for f in rep['failures'])


def test_bad_registry_kind_is_refused(world):
    world['known'].write_text(json.dumps({'schema': 1, 'hunks': [dict(world['entries'][0], kind='minor')]}))
    with pytest.raises(ValueError):
        run(world)


def test_expected_difference_that_is_absent_fails(world):
    write(world['site'], 'hgq/changed.py', 'def g():\n    return 2\n')
    world['record'].write_text(refdiff.listing(refdiff.walk(world['site'], refdiff.INSTALLED_TOPS)))
    rec = world['site'] / refdiff.INSTALLED_DIST / 'RECORD'
    rec.write_text(rec.read_text().replace(b64('def g():\n    return 1\n'), b64('def g():\n    return 2\n')))
    rep = run(world)
    assert any('expected to differ but is identical' in f for f in rep['failures'])
    assert any('matches no hunk' in f for f in rep['failures'])


def test_installed_tree_changes_are_caught_by_listing_and_record(world):
    write(world['site'], 'hgq/same.py', 'def f():\n    return 1  # edited\n')
    rep = run(world)
    assert any('listing differs' in f for f in rep['failures'])
    assert any('wheel RECORD' in f for f in rep['failures'])


def test_snapshot_tree_change_is_caught(world):
    write(world['snap'], 'README', 'edited\n')
    rep = run(world)
    assert any('snapshot tree hash' in f for f in rep['failures'])


def test_stop_if_import_check(world):
    write(world['snap'], 'hgq/stop_if.py', 'import keras\nimport numpy\n')
    snap_all = refdiff.walk(world['snap'])
    world['code_base'].write_text('tree  %s  reference-code/HGQ2/src  files=%d\n' % (refdiff.tree_hash(snap_all), len(snap_all)))
    rep = run(world)
    c = [c for c in rep['citations'] if c['file'] == 'hgq/stop_if.py'][0]
    assert not c['ok'] and c['non_stdlib_imports'] == ['keras', 'numpy']
