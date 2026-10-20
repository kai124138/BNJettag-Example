"""Tests of M3 `cbl/data.py` and of `gates/fileaudit.py`, on synthetic files only.

No test opens the campaign's data: every raw file and archive here is written by the test, in the
HLS4ML layout that DATASET.md:21-32 describes (`jets` (n, 59) with the one-hot labels in columns
-6:-1, `jetConstituentList` (n, 150, 16)). numpy and h5py only; the inotify tests run on Linux.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip('h5py')

FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))
sys.path.insert(0, str(FORK / 'tests'))

from cbl import data  # noqa: E402


def complete_freeze_manifest(path):
    """A complete section-8.5 freeze manifest built from constructed screen data (M16's fixture),
    written with its content hash; the sealed reader needs M15's rule and M16's completeness."""
    from cbl import select as sel
    from fixtures import m16_synth as syn
    return sel.write_freeze_manifest(path, syn.freeze_example(sel))


def make_raw(path, n, seed, *, two_hot_row=None):
    rng = np.random.default_rng(seed)
    jets = rng.standard_normal((n, 59)).astype(np.float32)
    lab = rng.integers(0, 5, n)
    jets[:, -6:] = 0
    jets[np.arange(n), 53 + lab] = 1
    if two_hot_row is not None:
        jets[two_hot_row, 53:58] = 0
        jets[two_hot_row, 53:55] = 1
    cl = rng.standard_normal((n, 150, 16)).astype(np.float32)
    cl[..., 5] = rng.exponential(5.0, (n, 150)).astype(np.float32)   # pt, some below the 2 GeV gate
    with h5py.File(path, 'w') as f:
        f.create_dataset('jets', data=jets)
        f.create_dataset('jetConstituentList', data=cl)
    return lab


def make_raw_dir(d, sizes, seed0):
    d.mkdir(parents=True)
    for k, n in enumerate(sizes):
        make_raw(d / f'jetImage_{k}_150p_{k * 1000}_{k * 1000 + n}.h5', n, seed0 + k)
    return d


def chang_data():
    spec = importlib.util.spec_from_file_location('chang_data_ref', FORK / 'chang' / 'data.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp('prep')
    tr = make_raw_dir(root / 'train', [40, 25, 35], 10)
    va = make_raw_dir(root / 'val', [30, 20], 100)
    out = root / 'out'
    run_tr = data.run_chang_prepare(tr, out / data.TRAIN_ARCHIVE, jobs=2)
    run_va = data.run_chang_prepare(va, out / data.HELDOUT_ARCHIVE, jobs=2)
    return {'root': root, 'train': tr, 'val': va, 'out': out, 'run_tr': run_tr, 'run_va': run_va}


def test_glob_order_is_path_glob(prepared):
    assert data.glob_order(prepared['train']) == [p.name for p in Path(prepared['train']).glob('*.h5')]
    assert prepared['run_tr']['glob_order'] == data.glob_order(prepared['train'])
    assert prepared['run_tr']['glob_order_after_equal'] is True


def test_chang_unchanged():
    got = data.check_chang_unchanged()
    assert got['chang/tools/prepare_data.py'] == data.code_base_sha256('chang/tools/prepare_data.py')


def test_blocks_verify_in_glob_order(prepared):
    files = [data.raw_file_record(prepared['train'] / n)[0] for n in prepared['run_tr']['glob_order']]
    rec = data.verify_blocks(prepared['out'] / data.TRAIN_ARCHIVE, files, read_features=True)
    assert rec['ok'] and rec['rows_files'] == 100 and rec['distinct_label_blocks']
    vfiles = [data.raw_file_record(prepared['val'] / n, feature_block=False)[0] for n in prepared['run_va']['glob_order']]
    vrec = data.verify_blocks(prepared['out'] / data.HELDOUT_ARCHIVE, vfiles, read_features=False)
    assert vrec['ok'] and vrec['read_features'] is False
    # identity: archive row -> (glob index, file, row)
    off, entry = 0, {'files': []}
    for k, f in enumerate(files):
        entry['files'].append({'index': k, 'name': f['name'], 'offset': off, 'rows': f['rows']})
        off += f['rows']
    k, name, r = data.jet_identity(entry, files[0]['rows'] + 3)
    assert (k, name, r) == (1, files[1]['name'], 3)


def test_blocks_detect_a_different_order(prepared, tmp_path):
    files = [data.raw_file_record(prepared['train'] / n)[0] for n in prepared['run_tr']['glob_order']]
    rec = data.verify_blocks(prepared['out'] / data.TRAIN_ARCHIVE, files[::-1], read_features=True)
    assert not rec['ok'] and rec['blocks_failed']


def test_archive_record_held_out_reads_no_features(prepared):
    rec = data.archive_record(prepared['out'] / data.HELDOUT_ARCHIVE, read_features=False)
    assert 'feature_sha256' not in rec and rec['rows'] == 50 and sum(rec['label_counts']) == 50
    assert rec['datasets']['feature']['dtype'] == 'float16'


def test_prepare_refuses_overwrite(prepared):
    with pytest.raises(data.DataIdentityError):
        data.run_chang_prepare(prepared['train'], prepared['out'] / data.TRAIN_ARCHIVE, jobs=1)


def test_prepare_fails_on_two_hot(tmp_path):
    d = tmp_path / 'bad'
    d.mkdir()
    make_raw(d / 'a.h5', 10, 1, two_hot_row=3)
    with pytest.raises(data.DataIdentityError, match='exited'):
        data.run_chang_prepare(d, tmp_path / 'o' / data.TRAIN_ARCHIVE, jobs=1)


@pytest.mark.parametrize('ptetaphi', [True, False])
def test_reader_equals_chang_get_data(prepared, ptetaphi):
    chang = chang_data()
    np.random.seed(42)
    (xt, yt), (xv, yv), (xs, ys) = chang.get_data(prepared['out'], 8, ptetaphi)
    np.random.seed(42)
    tv = data.read_train_val(prepared['out'] / data.TRAIN_ARCHIVE, 8, ptetaphi)
    for a, b in ((xt, tv.X_train), (yt, tv.y_train), (xv, tv.X_val), (yv, tv.y_val)):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert tv.record['n_train'] == 90 and tv.record['n_val'] == 10
    # the sealed reader with a dummy manifest equals Chang's test path
    man = prepared['root'] / f'freeze-{ptetaphi}.json'
    content_sha = complete_freeze_manifest(man)
    sha = data.file_sha256(prepared['out'] / data.HELDOUT_ARCHIVE)
    X, y, rec = data.read_heldout(prepared['out'] / data.HELDOUT_ARCHIVE, freeze_manifest=man,
                                  expect_archive_sha256=sha, shift=tv.shift, scale=tv.scale,
                                  n_constituents=8, ptetaphi=ptetaphi)
    assert X.dtype == xs.dtype and X.shape == xs.shape and X.tobytes() == xs.tobytes()
    assert y.dtype == ys.dtype and y.tobytes() == ys.tobytes()
    assert rec['freeze_manifest_sha256'] == data.file_sha256(man)
    assert rec['freeze_manifest_content_sha256'] == content_sha


def test_reader_leaves_the_generator_as_chang_does(prepared):
    chang = chang_data()
    np.random.seed(42)
    chang.get_data(prepared['out'], 8, True)
    after_chang = np.random.get_state()[1].copy()
    np.random.seed(42)
    data.read_train_val(prepared['out'] / data.TRAIN_ARCHIVE, 8, True)
    assert np.array_equal(np.random.get_state()[1], after_chang)


def test_reader_refusals(prepared, tmp_path):
    arch = prepared['out'] / data.TRAIN_ARCHIVE
    with pytest.raises(data.HeldOutSealed):
        data.read_train_val(prepared['out'] / data.HELDOUT_ARCHIVE, 8, True)
    np.random.seed(42)
    with pytest.raises(data.DataIdentityError, match='differs from the manifest'):
        data.read_train_val(arch, 8, True, expect_sha256='0' * 64)
    np.random.seed(42)
    np.random.rand()
    with pytest.raises(data.DataIdentityError, match='draw happened'):
        data.read_train_val(arch, 8, True)
    np.random.seed(7)
    data.read_train_val(arch, 8, True, expect_seed=None)        # guard off: Chang's own behaviour
    # a zero std is an integrity failure
    z = tmp_path / 'z'
    z.mkdir()
    with h5py.File(z / data.TRAIN_ARCHIVE, 'w') as f:
        x = np.ones((20, 150, 16), dtype=np.float16) * 3
        f.create_dataset('feature', data=x)
        f.create_dataset('label', data=np.arange(20) % 5)
    np.random.seed(42)
    with pytest.raises(data.DataIdentityError, match='std'):
        data.read_train_val(z / data.TRAIN_ARCHIVE, 8, True)


def test_heldout_refusals_come_before_the_archive(prepared, tmp_path):
    np.random.seed(42)
    tv = data.read_train_val(prepared['out'] / data.TRAIN_ARCHIVE, 8, True)
    missing_archive = tmp_path / 'does-not-exist' / data.HELDOUT_ARCHIVE
    kw = dict(expect_archive_sha256='a' * 64, shift=tv.shift, scale=tv.scale, n_constituents=8, ptetaphi=True)
    bad_json = tmp_path / 'bad.json'
    bad_json.write_text('{not json')
    empty = tmp_path / 'empty.json'
    empty.write_text('{}')
    lst = tmp_path / 'list.json'
    lst.write_text('[1, 2]')
    m15_only = tmp_path / 'm15-only.json'                  # passes M15's rule, fails M16's completeness
    m15_only.write_text(json.dumps({'dummy': True}))
    good = tmp_path / 'good.json'
    complete_freeze_manifest(good)
    tampered = tmp_path / 'tampered.json'
    t = json.loads(good.read_text())
    t['best']['best'] = 'F111' if t['best']['best'] != 'F111' else 'F110'
    tampered.write_text(json.dumps(t))
    for m in (None, tmp_path / 'absent.json', bad_json, empty, lst, m15_only, tampered):
        # the archive path does not exist: a refusal, not FileNotFoundError, shows no access happened
        with pytest.raises(data.HeldOutSealed):
            data.read_heldout(missing_archive, freeze_manifest=m, **kw)
    for bad in (None, '', 'A' * 64, 'a' * 63, 123):
        with pytest.raises(data.HeldOutSealed):
            data.read_heldout(missing_archive, freeze_manifest=good, **dict(kw, expect_archive_sha256=bad))
    with pytest.raises(data.HeldOutSealed, match='differs'):
        data.read_heldout(prepared['out'] / data.HELDOUT_ARCHIVE, freeze_manifest=good, **kw)
    with pytest.raises(data.DataIdentityError):
        data.read_heldout(prepared['out'] / data.HELDOUT_ARCHIVE, freeze_manifest=good,
                          **dict(kw, shift=tv.shift.astype(np.float64)))


def test_tarball_record_and_members(prepared, tmp_path):
    tgz = tmp_path / 't.tar.gz'
    names = data.glob_order(prepared['val'])
    with tarfile.open(tgz, 'w:gz') as tf:
        tf.add(prepared['val'], arcname='val', recursive=False)
        tf.add(prepared['val'] / names[0], arcname='val/' + names[0])
        tf.add(prepared['train'] / data.glob_order(prepared['train'])[0], arcname='val/' + names[1])  # wrong content
        tf.add(prepared['train'] / data.glob_order(prepared['train'])[1], arcname='val/extra.h5')
    import hashlib
    rec = data.tarball_record(tgz, expect_md5=hashlib.md5(tgz.read_bytes()).hexdigest())
    assert rec['md5_equal'] is True and len(rec['members']) == 3 and rec['non_file_members']
    files = [data.raw_file_record(prepared['val'] / n, feature_block=False)[0] for n in names]
    cmp = data.compare_members(rec, files)
    assert cmp['match'] == 1 and cmp['mismatch'] == [names[1]] and cmp['tarball_only'] == ['val/extra.h5']
    assert data.tarball_record(tgz, expect_md5='0' * 32)['md5_equal'] is False


def test_jet_digests_find_a_shared_jet():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((50, 150, 16)).astype(np.float32)
    b = rng.standard_normal((40, 150, 16)).astype(np.float32)
    assert data.digest_intersection(data.jet_digests(a), data.jet_digests(b)) == 0
    b[7] = a[11]
    assert data.digest_intersection(data.jet_digests(a), data.jet_digests(b)) == 1


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='inotify is Linux only')
def test_fileaudit_sees_h5py_opens_through_links(tmp_path):
    import fileaudit
    watched = tmp_path / 'w'
    watched.mkdir()
    target = watched / data.HELDOUT_ARCHIVE
    with h5py.File(target, 'w') as f:
        f.create_dataset('label', data=np.arange(3))
    other = tmp_path / 'o'
    other.mkdir()
    link = other / 'elsewhere.h5'
    link.symlink_to(target)
    code = 'import h5py, sys; h5py.File(sys.argv[1], "r").close()'
    r = fileaudit.run_audited([sys.executable, '-I', '-B', '-c', code, str(link)],
                              watches=[(target, 'file'), (watched, 'dir')], cwd=str(tmp_path))
    assert r['returncode'] == 0
    assert fileaudit.events_with(r['events'], {'file'}) and fileaudit.events_with(r['events'], {'dir'}, data.HELDOUT_ARCHIVE)
    r2 = fileaudit.run_audited([sys.executable, '-I', '-B', '-c', 'print(1)'], watches=[(target, 'file'), (watched, 'dir')], cwd=str(tmp_path))
    assert r2['events'] == [] and r2['events_before_start'] == []
