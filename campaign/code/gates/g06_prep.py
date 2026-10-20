"""Gate G06 (PREFLIGHT-PLAN section 3 row G06; STUDY v0.8 section 4; module M3 cbl/data.py).

Runs on the PC CPU in the venv of record, at a committed sha, against the committed preparation
manifest `manifests/data-prep-150c.json`, and checks that
1. the manifest is M3's kind and schema, was prepared at an ancestor of HEAD, and records Chang's
   `prepare_data.py` and `data.py` at their `code-base.txt` hashes, which they still have
   (Chang's script unchanged); numpy, h5py, HDF5, tqdm and Python equal the manifest's;
2. the raw training directory lists 62 files and the raw held-out directory 26, in the glob order
   the manifest records (the call of `prepare_data.py:21` on the same directory), each with the
   recorded sha256, size, rows, label block hash and per-class counts, and (training files) the
   float16 feature block hash; the one-hot assertion of `prepare_data.py:13` holds in every file;
   620,000 and 260,000 jets;
3. the archives of record hash to the manifest's sha256; their datasets have the recorded shape
   and dtype; block k of `150c-train.h5` equals training file k in glob order (feature and label
   hashes), block k of `150c-test.h5` equals held-out file k (label hashes; the held-out feature
   dataset is never read), so jet identity (file in glob order, row) holds;
4. the split of record: `read_train_val(150c-train.h5, 8, True)` after `np.random.seed(42)` gives
   558,000 training and 62,000 validation jets with the manifest's permutation, split, statistics
   and array hashes, and every per-feature std is finite and nonzero;
5. the 2-file smoke archive hashes as recorded and is block-equal to the first two training files;
6. preparation reproduced: Chang's script, run again unchanged into --work on the two raw
   directories and on a fresh 2-file link directory, writes archives byte-identical to those of
   record (the copies are deleted after hashing);
7. provenance: each Zenodo tarball is present, its md5 equals the Zenodo record's, its sha256
   the manifest's, and every on-disk file equals the tarball member of the same name, with no
   member missing on disk.

Every reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g06_prep.py --expect-sha12 SHA12 --work DIR [--json PATH]
        [--manifest PATH] [--zenodo-dir DIR --zenodo-api-json FILE] [--jobs 4]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import numpy as np  # noqa: E402

import gatelib  # noqa: E402
from cbl import data  # noqa: E402

GATE = 'G06'
RAW_KEYS = ('sha256', 'bytes', 'rows', 'one_hot', 'label_counts', 'label_block_sha256')


def check_raw(g, kind, directory, entry, expect_files, expect_rows, feature_block):
    order = data.glob_order(directory)
    g.check(order == entry['glob_order'], '%s: glob order of %s equals the manifest (%d files)' % (kind, directory, len(order)))
    g.check(len(order) == expect_files, '%s: %d files (expected %d)' % (kind, len(order), expect_files))
    bad, rows, one_hot = [], 0, True
    recs = []
    for fe in entry['files']:
        rec, _ = data.raw_file_record(Path(directory) / fe['name'], feature_block=feature_block)
        keys = RAW_KEYS + (('feature_block_sha256',) if feature_block else ())
        diff = [k for k in keys if rec.get(k) != fe.get(k)]
        if diff or fe['offset'] != rows:
            bad.append((fe['name'], diff))
        rows += rec['rows']
        one_hot = one_hot and rec['one_hot']
        recs.append(rec)
    g.info('%s files differing from the manifest' % kind, bad)
    g.check(not bad, '%s: every file has the recorded sha256, size, rows, offsets, label block and class counts%s'
            % (kind, ', float16 feature block' if feature_block else ''))
    g.check(one_hot, '%s: one-hot assertion of prepare_data.py:13 holds in every file' % kind)
    g.check(rows == expect_rows, '%s: %d jets (expected %d)' % (kind, rows, expect_rows))
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--expect-sha12', default=None)
    ap.add_argument('--work', required=True)
    ap.add_argument('--json', default=None)
    ap.add_argument('--manifest', default=str(data.MANIFEST_PATH))
    ap.add_argument('--zenodo-dir', default=None)
    ap.add_argument('--zenodo-api-json', default=None)
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    g = gatelib.Gate(GATE)
    head = g.header(a.expect_sha12)
    gatelib.check_venv_of_record(g)
    print('label       engineering, not quotable; held-out: hashes and counts only')
    out = {'header': head}

    mpath = Path(a.manifest)
    man = json.loads(mpath.read_text(encoding='utf-8'))
    out['manifest_sha256'] = data.file_sha256(mpath)
    g.info('manifest', {'path': str(mpath), 'sha256': out['manifest_sha256']})
    g.check(man.get('kind') == data.MANIFEST_KIND and man.get('schema') == data.MANIFEST_SCHEMA,
            'manifest kind %s schema %s' % (man.get('kind'), man.get('schema')))
    pa = man.get('prepared_at_code_sha') or ''
    rc, _, _ = gatelib.git('merge-base', '--is-ancestor', pa, 'HEAD')
    g.check(len(pa) == 40 and rc == 0, 'manifest prepared at %s, an ancestor of HEAD' % pa[:12])
    try:
        chang = data.check_chang_unchanged()
    except data.DataIdentityError as e:
        chang = {'error': str(e)}
    g.check(chang == man.get('chang_sha256'), "Chang's prepare_data.py and data.py equal code-base.txt and the manifest: %s" % chang)
    v = data.versions()
    vd = {k: (v[k], man['versions'].get(k)) for k in ('python', 'numpy', 'h5py', 'hdf5', 'tqdm')}
    g.info('versions (now, manifest)', vd)
    g.check(all(x == y for x, y in vd.values()), 'python, numpy, h5py, HDF5 and tqdm equal the manifest')

    P = man['paths']
    E = data.EXPECTED
    tr_recs = check_raw(g, 'train raw', P['train_dir'], man['train'], E['train_files'], E['train_jets'], True)
    va_recs = check_raw(g, 'held-out raw', P['val_dir'], man['heldout'], E['heldout_files'], E['heldout_jets'], False)

    # archives of record
    arch = Path(P['out_dir'])
    a_tr = data.archive_record(arch / data.TRAIN_ARCHIVE, read_features=True)
    g.check(a_tr == man['train']['archive'], 'train archive %s: sha256 %s, layout, label and feature hashes equal the manifest'
            % (data.TRAIN_ARCHIVE, a_tr['sha256']))
    g.check(a_tr['datasets']['feature']['shape'] == [E['train_jets'], 150, 16] and a_tr['datasets']['feature']['dtype'] == 'float16'
            and a_tr['datasets']['label']['shape'] == [E['train_jets']] and a_tr['datasets']['label']['dtype'] == 'int64',
            'train archive: feature (620000, 150, 16) float16, label (620000,) int64')
    g.info('train archive label counts', a_tr['label_counts'])
    b_tr = data.verify_blocks(arch / data.TRAIN_ARCHIVE, tr_recs, read_features=True)
    g.check(b_tr['ok'] and b_tr == man['train']['blocks'], 'train archive block k equals training file k in glob order (feature and label), %d distinct label blocks' % len(tr_recs))
    a_va = data.archive_record(arch / data.HELDOUT_ARCHIVE, read_features=False)
    g.check(a_va == man['heldout']['archive'], 'held-out archive %s: sha256 %s, layout and label hash equal the manifest (feature not read)'
            % (data.HELDOUT_ARCHIVE, a_va['sha256']))
    b_va = data.verify_blocks(arch / data.HELDOUT_ARCHIVE, va_recs, read_features=False)
    g.check(b_va['ok'] and b_va == man['heldout']['blocks'], 'held-out archive block k equals held-out file k in glob order (labels only), %d distinct label blocks' % len(va_recs))
    out['archives'] = {'train': a_tr['sha256'], 'heldout': a_va['sha256']}

    # split of record
    np.random.seed(data.SPLIT_SEED)
    tv = data.read_train_val(arch / data.TRAIN_ARCHIVE, 8, True, expect_sha256=man['train']['archive']['sha256'])
    keys = ('n_constituents', 'ptetaphi', 'val_size', 'n_jets', 'n_train', 'n_val', 'order_dtype', 'order_sha256',
            'split_sha256', 'stats_sha256', 'stats_dtype', 'stats_shape', 'std_finite', 'std_nonzero', 'arrays')
    diff = [k for k in keys if tv.record[k] != man['split'][k]]
    g.check(not diff, 'split: read_train_val(150c-train.h5, 8, True) after np.random.seed(42) equals the manifest (differs: %s)' % diff)
    g.check(tv.record['n_train'] == E['n_train'] and tv.record['n_val'] == E['n_val'],
            'split: %d training and %d validation jets (expected 558000 and 62000)' % (tv.record['n_train'], tv.record['n_val']))
    g.check(tv.record['std_finite'] and tv.record['std_nonzero'] and bool(np.all(np.isfinite(tv.scale))) and bool(np.all(tv.scale > 0)),
            'split: every per-feature std (%d) finite and nonzero' % tv.scale.size)
    lc = {'train': np.bincount(tv.y_train, minlength=5).tolist(), 'val': np.bincount(tv.y_val, minlength=5).tolist()}
    g.info('split label counts', lc)
    g.check(lc == man['split']['label_counts'], 'split label counts equal the manifest')
    out['split'] = {k: tv.record[k] for k in ('order_sha256', 'split_sha256', 'stats_sha256')}
    del tv

    # smoke archive
    sm = man['smoke']
    g.check([f['name'] for f in sm['files']] == man['train']['glob_order'][:2] and len(sm['files']) == 2,
            'smoke: built from the first two training files in glob order')
    g.check(data.glob_order(P['smoke_src_dir']) == sm['glob_order'], 'smoke: glob order of its link directory equals the manifest')
    a_sm = data.archive_record(Path(P['smoke_dir']) / data.TRAIN_ARCHIVE, read_features=True)
    g.check(a_sm == sm['archive'], 'smoke archive sha256 %s and content hashes equal the manifest' % a_sm['sha256'])
    sm_recs = [next(r for r in tr_recs if r['name'] == f['name']) for f in sm['files']]
    b_sm = data.verify_blocks(Path(P['smoke_dir']) / data.TRAIN_ARCHIVE, sm_recs, read_features=True)
    g.check(b_sm['ok'] and a_sm['rows'] == sum(r['rows'] for r in sm_recs), 'smoke archive blocks equal its two files (%d jets)' % a_sm['rows'])

    # preparation reproduced
    work = Path(a.work)
    from syncroot import require_unsynced
    require_unsynced(work)
    g.check(not work.exists(), '--work %s does not exist yet' % work)
    work.mkdir(parents=True)
    repro = {}
    try:
        for key, src, name, rec in (('train', P['train_dir'], data.TRAIN_ARCHIVE, a_tr),
                                    ('heldout', P['val_dir'], data.HELDOUT_ARCHIVE, a_va)):
            r = data.run_chang_prepare(src, work / key / name, jobs=a.jobs)
            sha = data.file_sha256(work / key / name)
            repro[key] = {'sha256': sha, 'seconds': r['seconds'], 'glob_order_equal': r['glob_order'] == man[key]['glob_order']}
            g.check(sha == rec['sha256'] and repro[key]['glob_order_equal'],
                    'reproduced: Chang\'s script on %s again writes %s byte-identical to the archive of record (%.0f s)' % (src, name, r['seconds']))
            (work / key / name).unlink()
        src = work / 'smoke-src'
        src.mkdir()
        for n in man['train']['glob_order'][:2]:
            (src / n).symlink_to(Path(P['train_dir']) / n)
        r = data.run_chang_prepare(src, work / 'smoke' / data.TRAIN_ARCHIVE, jobs=a.jobs)
        sha = data.file_sha256(work / 'smoke' / data.TRAIN_ARCHIVE)
        repro['smoke'] = {'sha256': sha, 'seconds': r['seconds']}
        g.check(sha == sm['archive']['sha256'], 'reproduced: the 2-file smoke archive again byte-identical (%.0f s)' % r['seconds'])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    g.check(not work.exists(), 'reproduction copies deleted')
    out['reproduced'] = repro

    # provenance
    prov = man.get('provenance') or {}
    g.check(bool(prov), 'manifest holds the Zenodo provenance record')
    if prov and a.zenodo_dir:
        exp = data.zenodo_expected(a.zenodo_api_json)
        g.check(data.file_sha256(a.zenodo_api_json) == prov['api_json_sha256'], 'Zenodo API record file equals the manifest (sha256)')
        g.check(exp['record_id'] == 3602260, 'Zenodo record id %s' % exp['record_id'])
        for key, recs in (('hls4ml_LHCjet_150p_train.tar.gz', tr_recs), ('hls4ml_LHCjet_150p_val.tar.gz', va_recs)):
            p = Path(a.zenodo_dir) / key
            if not g.check(p.exists() and prov['tarballs'][key]['present'], '%s present' % key):
                continue
            t = data.tarball_record(p, expect_md5=exp['files'][key]['md5'])
            cmp = data.compare_members(t, recs)
            g.info(key, {'bytes': t['bytes'], 'size_zenodo': exp['files'][key]['size'], 'md5': t['md5'],
                         'md5_zenodo': exp['files'][key]['md5'], 'sha256': t['sha256'], 'members': len(t['members']),
                         'match': cmp['match'], 'mismatch': cmp['mismatch'], 'not_in_tarball': cmp['not_in_tarball'],
                         'tarball_only': cmp['tarball_only']})
            g.check(t['md5_equal'] and t['bytes'] == exp['files'][key]['size'], '%s md5 and size equal the Zenodo record' % key)
            g.check(t['sha256'] == prov['tarballs'][key]['sha256'], '%s sha256 equals the manifest' % key)
            g.check(cmp['match'] == len(recs) and not cmp['mismatch'] and not cmp['not_in_tarball'] and not cmp['tarball_only'],
                    '%s: all %d on-disk files equal their tarball members byte for byte; no member missing on disk' % (key, len(recs)))
            g.check(cmp['member_order'] == prov['tarballs'][key]['members_vs_disk']['member_order'], '%s member order equals the manifest' % key)
    elif prov:
        g.check(False, '--zenodo-dir not given: provenance not rechecked')

    g.check_clean('at end')
    return g.finish(out, a.json)


if __name__ == '__main__':
    sys.exit(main())
