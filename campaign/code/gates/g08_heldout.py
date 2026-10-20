"""Gate G08 (PREFLIGHT-PLAN section 3 row G08; STUDY v0.8 sections 4 and 8.5; module M3 cbl/data.py).

Runs on the PC CPU in the venv of record, at a committed sha. Held-out data are touched for
hashes and counts only: file sha256, row counts, per-class counts from the label dataset, label
block hashes and per-jet digests. The held-out archive's `feature` dataset is never read, and no
model output, metric or other statistic of held-out data is computed. Checks:
1. the raw held-out directory holds 26 files in the recorded glob order, each with the recorded
   sha256, size, rows, label block hash and per-class counts; 260,000 jets in all;
2. the held-out archive `150c-test.h5` hashes to the manifest's sha256, holds `feature`
   (260000, 150, 16) float16 and `label` (260000,) int64 by its metadata, has the recorded
   per-class counts, and its label block k equals held-out file k in glob order (identity);
3. identities are disjoint from training: no file name and no file sha256 is shared between the
   62 training and 26 held-out files, and no per-jet digest (sha256 of the raw
   `jetConstituentList` row bytes) is shared; the values equal the manifest's;
4. the sealed reader `cbl.data.read_heldout`, in a process of its own under the open-file audit
   (gates/fileaudit.py), refuses the held-out archive of record with no freeze manifest, a
   missing one, invalid JSON, an empty object, a JSON list and an object that passes M15's rule
   but is not a complete section-8.5 manifest, and with a complete synthetic manifest but no valid
   expected sha256; each refusal comes before any open of the archive (no inotify
   event on it, its name in its directory, the raw held-out directory or any raw held-out file
   during the window). It is never called on the archive of record with a valid manifest;
5. on a dummy pair, never the held-out archive: `150c-train.h5` a link to the 2-file smoke archive
   of record and `150c-test.h5` an archive that Chang's script writes in --work from training
   files 3 and 4 of the glob order, the sealed reader refuses a wrong expected sha256 and, with a
   complete synthetic manifest, returns X_test and y_test bitwise equal to Chang's `get_data` on the
   same directory, as do X_train, y_train, X_val and y_val from `read_train_val`; the dummy
   archive's open is seen (positive control of the audit).

Since batch B5 the reader applies M15's rule (a path, an existing file, a non-empty JSON object)
and then M16's completeness check (`cbl.select.require_freeze_manifest`, G56); the "complete
synthetic manifest" is built by M16's test fixture from constructed screen data
(`tests/fixtures/m16_synth.py`) and is not the campaign record. Every reading is engineering, not
quotable. Run:

    <venv>/bin/python -I -B gates/g08_heldout.py --expect-sha12 SHA12 --work DIR [--json PATH] [--manifest PATH] [--jobs 4]
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))
sys.path.insert(0, str(FORK / 'tests'))

import numpy as np  # noqa: E402

import fileaudit  # noqa: E402
import gatelib  # noqa: E402
from cbl import data  # noqa: E402

GATE = 'G08'
SIX = ('X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test')

SEALED = r'''
import json, sys
import numpy as np
sys.dont_write_bytecode = True
spec = json.loads(sys.argv[1])
sys.path.insert(0, spec['fork'])
from cbl import data
res = {}
np.random.seed(42)
tv = data.read_train_val(spec['dummy_train'], 8, True)
kw = dict(shift=tv.shift, scale=tv.scale, n_constituents=8, ptetaphi=True)
def attempt(label, archive, manifest, expect):
    try:
        data.read_heldout(archive, freeze_manifest=manifest, expect_archive_sha256=expect, **kw)
        res[label] = 'RETURNED DATA'
    except data.HeldOutSealed as e:
        res[label] = 'refused: ' + str(e)
    except Exception as e:
        res[label] = 'other %s: %s' % (type(e).__name__, str(e))
real, rsha = spec['heldout'], spec['heldout_sha256']
attempt('real/no manifest', real, None, rsha)
attempt('real/missing manifest', real, spec['missing'], rsha)
attempt('real/invalid json', real, spec['invalid'], rsha)
attempt('real/empty object', real, spec['empty'], rsha)
attempt('real/json list', real, spec['list'], rsha)
attempt('real/M15-only manifest', real, spec['m15_only'], rsha)
attempt('real/valid dummy manifest, no expected sha', real, spec['dummy_manifest'], None)
attempt('real/valid dummy manifest, malformed expected sha', real, spec['dummy_manifest'], rsha.upper())
attempt('dummy/no manifest', spec['dummy_test'], None, spec['dummy_test_sha256'])
attempt('dummy/valid dummy manifest, wrong expected sha', spec['dummy_test'], spec['dummy_manifest'], spec['smoke_sha256'])
X, y, rec = data.read_heldout(spec['dummy_test'], freeze_manifest=spec['dummy_manifest'],
                              expect_archive_sha256=spec['dummy_test_sha256'], **kw)
res['dummy/valid dummy manifest, right sha'] = 'returned %d jets' % len(y)
for k, v in (('X_train', tv.X_train), ('y_train', tv.y_train), ('X_val', tv.X_val), ('y_val', tv.y_val),
             ('X_test', X), ('y_test', y)):
    np.save(spec['out'] + '/' + k + '.npy', v)
print('RESULT ' + json.dumps({'cases': res, 'record': rec}))
'''

REF = r'''
import importlib.util, json, sys
import numpy as np
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('data', sys.argv[1])
chang = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chang)
np.random.seed(42)
(xt, yt), (xv, yv), (xs, ys) = chang.get_data(sys.argv[2], 8, True)
for k, v in (('X_train', xt), ('y_train', yt), ('X_val', xv), ('y_val', yv), ('X_test', xs), ('y_test', ys)):
    np.save(sys.argv[3] + '/' + k + '.npy', v)
print('RESULT {}')
'''


def child_env():
    env = {k: v for k, v in os.environ.items() if k not in ('KERAS_BACKEND', 'JAX_PLATFORMS')}
    env['JAX_PLATFORMS'] = 'cpu'
    return env


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--expect-sha12', default=None)
    ap.add_argument('--work', required=True)
    ap.add_argument('--json', default=None)
    ap.add_argument('--manifest', default=str(data.MANIFEST_PATH))
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    g = gatelib.Gate(GATE)
    head = g.header(a.expect_sha12)
    gatelib.check_venv_of_record(g)
    print('label       engineering, not quotable; held-out: hashes and counts only, feature never read')
    out = {'header': head}

    mpath = Path(a.manifest)
    man = json.loads(mpath.read_text(encoding='utf-8'))
    out['manifest_sha256'] = data.file_sha256(mpath)
    g.info('manifest', {'path': str(mpath), 'sha256': out['manifest_sha256']})
    g.check(man.get('kind') == data.MANIFEST_KIND, 'manifest kind %s' % man.get('kind'))
    P, E = man['paths'], data.EXPECTED
    val_dir, tr_dir, arch_dir = Path(P['val_dir']), Path(P['train_dir']), Path(P['out_dir'])
    heldout = arch_dir / data.HELDOUT_ARCHIVE

    # 1. raw held-out files
    order = data.glob_order(val_dir)
    g.check(order == man['heldout']['glob_order'] and len(order) == E['heldout_files'],
            'raw held-out directory: %d files in the recorded glob order (expected 26)' % len(order))
    va_recs, va_dig, bad = [], [], []
    for fe in man['heldout']['files']:
        rec, dig = data.raw_file_record(val_dir / fe['name'], feature_block=False, digests=True)
        if [k for k in ('sha256', 'bytes', 'rows', 'one_hot', 'label_counts', 'label_block_sha256') if rec[k] != fe[k]]:
            bad.append(fe['name'])
        va_recs.append(rec)
        va_dig.append(dig)
    g.check(not bad, 'every raw held-out file has the recorded sha256, size, rows, label block and class counts (differ: %s)' % bad)
    n_raw = sum(r['rows'] for r in va_recs)
    g.check(n_raw == E['heldout_jets'], 'raw held-out files: %d jets (expected 260000)' % n_raw)

    # 2. the held-out archive: hashes, metadata and labels only
    a_va = data.archive_record(heldout, read_features=False)
    g.info('held-out archive', {'sha256': a_va['sha256'], 'bytes': a_va['bytes'], 'datasets': a_va['datasets'],
                                'label_counts': a_va['label_counts']})
    g.check(a_va == man['heldout']['archive'], 'held-out archive sha256, layout, label hash and class counts equal the manifest')
    d = a_va['datasets']
    g.check(d['feature']['shape'] == [E['heldout_jets'], 150, 16] and d['feature']['dtype'] == 'float16'
            and d['label']['shape'] == [E['heldout_jets']] and d['label']['dtype'] == 'int64',
            'held-out archive metadata: feature (260000, 150, 16) float16, label (260000,) int64')
    g.check(sum(a_va['label_counts']) == E['heldout_jets'] and len(a_va['label_counts']) == data.N_CLASSES,
            'held-out per-class counts over 5 classes sum to 260000: %s' % a_va['label_counts'])
    b_va = data.verify_blocks(heldout, va_recs, read_features=False)
    g.check(b_va['ok'], 'held-out archive label block k equals held-out file k in glob order (%d distinct label blocks)' % len(va_recs))
    out['heldout'] = {'archive_sha256': a_va['sha256'], 'label_counts': a_va['label_counts']}

    # 3. disjoint identities
    tr_recs, tr_dig = [], []
    for n in man['train']['glob_order']:
        rec, dig = data.raw_file_record(tr_dir / n, feature_block=False, digests=True)
        tr_recs.append(rec)
        tr_dig.append(dig)
    names = sorted(set(man['train']['glob_order']) & set(order))
    shas = len({r['sha256'] for r in tr_recs} & {r['sha256'] for r in va_recs})
    shared = data.digest_intersection(np.concatenate(tr_dig), np.concatenate(va_dig))
    dj = {'file_names_shared': names, 'file_sha256_shared': shas, 'jet_digests_shared': shared}
    g.info('disjointness', dj)
    g.check(not names and shas == 0, 'no file name or file sha256 shared between the 62 training and 26 held-out files')
    g.check(shared == 0, 'no per-jet digest shared between %d training and %d held-out jets'
            % (sum(len(x) for x in tr_dig), sum(len(x) for x in va_dig)))
    g.check(all(man['disjoint'][k] == v for k, v in dj.items()), 'disjointness equals the manifest')
    del tr_dig, va_dig

    # 4-5. the sealed reader
    work = Path(a.work)
    fileaudit.require_unsynced(work)
    g.check(not work.exists(), '--work %s does not exist yet' % work)
    pair, src, sealed_out, ref_out = work / 'pair', work / 'dummy-src', work / 'sealed', work / 'ref'
    for x in (pair, src, sealed_out, ref_out):
        x.mkdir(parents=True)
    try:
        smoke = Path(P['smoke_dir']) / data.TRAIN_ARCHIVE
        g.check(data.file_sha256(smoke) == man['smoke']['archive']['sha256'], 'smoke archive of record hashes to the manifest')
        (pair / data.TRAIN_ARCHIVE).symlink_to(smoke)
        picks = man['train']['glob_order'][2:4]
        for n in picks:
            (src / n).symlink_to(tr_dir / n)
        r = data.run_chang_prepare(src, pair / data.HELDOUT_ARCHIVE, jobs=a.jobs)
        dummy_sha = data.file_sha256(pair / data.HELDOUT_ARCHIVE)
        g.info('dummy held-out archive', {'from training files': picks, 'sha256': dummy_sha, 'seconds': r['seconds']})
        g.check(dummy_sha != a_va['sha256'], 'the dummy archive is not the held-out archive')
        dm = work / 'freeze-dummy.json'
        from cbl import select as sel
        from fixtures import m16_synth as syn
        sel.write_freeze_manifest(dm, syn.freeze_example(sel))       # complete, synthetic; not the campaign record
        (work / 'm15-only.json').write_text(json.dumps({'G08': 'passes M15\'s rule only; not a section-8.5 manifest'}))
        (work / 'invalid.json').write_text('{not json')
        (work / 'empty.json').write_text('{}')
        (work / 'list.json').write_text('[1, 2, 3]')
        spec = {'fork': str(FORK), 'heldout': str(heldout), 'heldout_sha256': man['heldout']['archive']['sha256'],
                'missing': str(work / 'absent.json'), 'invalid': str(work / 'invalid.json'),
                'empty': str(work / 'empty.json'), 'list': str(work / 'list.json'), 'dummy_manifest': str(dm),
                'm15_only': str(work / 'm15-only.json'),
                'dummy_train': str(pair / data.TRAIN_ARCHIVE), 'dummy_test': str(pair / data.HELDOUT_ARCHIVE),
                'dummy_test_sha256': dummy_sha, 'smoke_sha256': man['smoke']['archive']['sha256'], 'out': str(sealed_out)}
        watches = [(heldout, 'heldout'), (arch_dir, 'data-dir'), (val_dir, 'heldout-raw-dir')]
        watches += [(val_dir / n, 'heldout-raw') for n in order]
        watches += [(pair / data.HELDOUT_ARCHIVE, 'dummy-heldout')]
        py = [sys.executable, '-I', '-B', '-c']
        r_s = fileaudit.run_audited(py + [SEALED, json.dumps(spec)], watches=watches, env=child_env(), cwd=str(work))
        g.info('sealed-reader process', {'returncode': r_s['returncode'], 'stderr_tail': r_s['stderr'][-800:],
                                         'events': r_s['events']})
        g.check(r_s['returncode'] == 0, 'sealed-reader process ran')
        res = json.loads([ln for ln in r_s['stdout'].splitlines() if ln.startswith('RESULT ')][-1][7:]) if r_s['returncode'] == 0 else {'cases': {}}
        for label, outcome in res['cases'].items():
            g.info(label, outcome[:160])
        refusals = [k for k in res['cases'] if not k.endswith('right sha')]
        g.check(len(refusals) == 10 and all(res['cases'][k].startswith('refused: ') for k in refusals),
                'sealed reader refuses all 10 cases: 8 on the held-out archive of record (no, missing, invalid, empty, list, '
                'M15-only manifest; complete synthetic manifest without or with a malformed expected sha256) and 2 on the '
                'dummy (no manifest; wrong sha256)')
        g.check('incomplete' in res['cases'].get('real/M15-only manifest', ''),
                'the M15-only manifest is refused by M16\'s completeness check')
        g.check(res['cases'].get('dummy/valid dummy manifest, right sha', '').startswith('returned '),
                'sealed reader returns data with a valid dummy manifest on the dummy archive')
        fe = (fileaudit.events_with(r_s['events'], {'heldout', 'heldout-raw', 'heldout-raw-dir'})
              + fileaudit.events_with(r_s['events'], {'data-dir'}, data.HELDOUT_ARCHIVE)
              + [e for e in r_s['events'] if e['overflow']])
        g.check(not r_s['events_before_start'] and not fe,
                'sealed-reader window: no open of the held-out archive, its name in the data directory, the raw held-out directory or any raw held-out file (%d forbidden events)' % len(fe))
        g.check(bool(fileaudit.events_with(r_s['events'], {'dummy-heldout'})), 'positive control: the sealed reader\'s open of the dummy archive is seen')
        fd_bad = [p for p in r_s['fd_paths'] if os.path.realpath(p) == os.path.realpath(heldout)
                  or os.path.realpath(p).startswith(os.path.realpath(val_dir) + os.sep)]
        g.check(not fd_bad, 'sealed-reader fd inventory holds no held-out path')

        r_ref = fileaudit.run_audited(py + [REF, str(FORK / 'chang' / 'data.py'), str(pair), str(ref_out)],
                                      watches=watches, env=child_env(), cwd=str(work))
        g.check(r_ref['returncode'] == 0, "Chang's get_data ran on the dummy pair")
        fe = (fileaudit.events_with(r_ref['events'], {'heldout', 'heldout-raw', 'heldout-raw-dir'})
              + fileaudit.events_with(r_ref['events'], {'data-dir'}, data.HELDOUT_ARCHIVE))
        g.check(not fe and bool(fileaudit.events_with(r_ref['events'], {'dummy-heldout'})),
                "reference window: Chang's get_data opened the dummy test archive and nothing held-out")
        if r_s['returncode'] == 0 and r_ref['returncode'] == 0:
            for k in SIX:
                x, y = np.load(sealed_out / (k + '.npy')), np.load(ref_out / (k + '.npy'))
                g.check(same(x, y), "dummy pair %s: lab readers equal Chang's get_data in dtype, shape and bytes (%s %s)"
                        % (k, x.dtype, list(x.shape)))
        out['sealed'] = {'cases': res['cases'], 'record': res.get('record'), 'events': r_s['events']}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    g.check(not work.exists(), 'work directory deleted')
    g.check_clean('at end')
    return g.finish(out, a.json)


if __name__ == '__main__':
    sys.exit(main())
