"""Gate G07 (PREFLIGHT-PLAN section 3 row G07; STUDY v0.8 section 4; module M3 cbl/data.py).

Runs on the PC CPU in the venv of record, at a committed sha, and checks that
1. the lab's train/validation reader `cbl.data.read_train_val` returns, on the archive of record,
   arrays bitwise equal to Chang's `get_data` (`chang/data.py`, loaded by path, unchanged): X_train,
   y_train, X_val and y_val equal in dtype, shape and bytes. Each side runs in its own process,
   after `np.random.seed(42)` (run_train.py:41), with CH-8's arguments (8 constituents, pt, etarel,
   phirel; run_train.py:71). Chang's `get_data` reads `150c-test.h5` eagerly (data.py:13-15), so it
   is called on a placeholder directory: `150c-train.h5` there is a symbolic link to the archive of
   record and `150c-test.h5` a 16-jet file of zeros written by this gate, which is not the held-out
   archive and does not enter the train or validation arrays (data.py:28-29 use the training
   archive only). A one-bit change in a copy is shown to break the comparison (negative control);
2. both readers reproduce the manifest's split record (558,000 and 62,000 jets; permutation, split,
   statistics and array hashes);
3. open-file audit (gates/fileaudit.py): the reader's process, which stands in for the training
   process until the runner exists (M13, B5; G51 audits the runner), opens no held-out file. Kernel
   inotify watches cover, from before the process starts until after it ends, the held-out archive,
   the directory that holds it (by entry name), the raw held-out directory and each raw held-out
   file. Any open of these by any process during the window fails the gate. Positive controls: the
   reader's own open of the training archive is seen; Chang's `get_data` is seen opening the
   placeholder `150c-test.h5` through h5py's C open; a third process that opens a watched decoy
   through a symbolic link elsewhere is seen. The /proc fd inventory of the reader's process holds
   the training archive and nothing named `150c-test.h5` and no held-out path.

Every reading is engineering, not quotable. No held-out byte is read. Run:

    <venv>/bin/python -I -B gates/g07_reader.py --expect-sha12 SHA12 --work DIR [--json PATH] [--manifest PATH]
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

import numpy as np  # noqa: E402

import fileaudit  # noqa: E402
import gatelib  # noqa: E402
from cbl import data  # noqa: E402

GATE = 'G07'
ARRAYS = ('X_train', 'y_train', 'X_val', 'y_val')

REF = r'''
import importlib.util, json, sys
import numpy as np
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('data', sys.argv[1])
chang = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chang)
np.random.seed(42)                                   # run_train.py:41
(xt, yt), (xv, yv), (xs, ys) = chang.get_data(sys.argv[2], 8, True)
for k, v in (('X_train', xt), ('y_train', yt), ('X_val', xv), ('y_val', yv)):
    np.save(sys.argv[3] + '/' + k + '.npy', v)
print('RESULT ' + json.dumps({'placeholder_test_rows': int(len(ys))}))
'''

LAB = r'''
import json, sys
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
from cbl import data
np.random.seed(42)                                   # as run_train.py:41
tv = data.read_train_val(sys.argv[2], 8, True, expect_sha256=sys.argv[3])
for k in ('X_train', 'y_train', 'X_val', 'y_val'):
    np.save(sys.argv[4] + '/' + k + '.npy', getattr(tv, k))
print('RESULT ' + json.dumps(tv.record))
'''

DECOY = r'''
import sys, h5py
h5py.File(sys.argv[1], 'r').close()
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
    a = ap.parse_args()
    g = gatelib.Gate(GATE)
    head = g.header(a.expect_sha12)
    gatelib.check_venv_of_record(g)
    print('label       engineering, not quotable; no held-out byte is read')
    out = {'header': head}

    mpath = Path(a.manifest)
    man = json.loads(mpath.read_text(encoding='utf-8'))
    out['manifest_sha256'] = data.file_sha256(mpath)
    g.info('manifest', {'path': str(mpath), 'sha256': out['manifest_sha256']})
    g.check(man.get('kind') == data.MANIFEST_KIND, 'manifest kind %s' % man.get('kind'))
    try:
        chang = data.check_chang_unchanged()
    except data.DataIdentityError as e:
        chang = {'error': str(e)}
    g.check(chang == man.get('chang_sha256'), "Chang's data.py equals code-base.txt and the manifest")

    P = man['paths']
    arch_dir = Path(P['out_dir'])
    train = arch_dir / data.TRAIN_ARCHIVE
    heldout = arch_dir / data.HELDOUT_ARCHIVE
    val_dir = Path(P['val_dir'])
    train_sha = man['train']['archive']['sha256']
    g.check(data.file_sha256(train) == train_sha, 'training archive of record hashes to the manifest (%s)' % train_sha[:16])
    g.check(heldout.exists() and len(data.glob_order(val_dir)) == data.EXPECTED['heldout_files'],
            'held-out archive and raw held-out directory exist (watched; never opened by this gate)')

    work = Path(a.work)
    fileaudit.require_unsynced(work)
    g.check(not work.exists(), '--work %s does not exist yet' % work)
    ph = work / 'placeholder'
    ref_out, lab_out, decoy_dir, link_dir = work / 'ref', work / 'lab', work / 'decoy', work / 'link'
    for d in (ph, ref_out, lab_out, decoy_dir, link_dir):
        d.mkdir(parents=True)
    (ph / data.TRAIN_ARCHIVE).symlink_to(train)
    import h5py
    with h5py.File(ph / data.HELDOUT_ARCHIVE, 'w') as f:
        f.create_dataset('feature', data=np.zeros((16, 150, 16), dtype=np.float16))
        f.create_dataset('label', data=np.arange(16) % 5)
    decoy = decoy_dir / data.HELDOUT_ARCHIVE
    with h5py.File(decoy, 'w') as f:
        f.create_dataset('label', data=np.arange(3))
    (link_dir / 'elsewhere.h5').symlink_to(decoy)

    forbidden = [(heldout, 'heldout'), (arch_dir, 'data-dir'), (val_dir, 'heldout-raw-dir')]
    forbidden += [(val_dir / n, 'heldout-raw') for n in data.glob_order(val_dir)]
    watches = forbidden + [(train, 'train'), (ph / data.HELDOUT_ARCHIVE, 'placeholder'), (decoy, 'decoy')]
    forbidden_real = {os.path.realpath(p) for p, _ in forbidden if p.is_file()} | {os.path.realpath(val_dir)}

    def forbidden_events(r):
        return (fileaudit.events_with(r['events'], {'heldout', 'heldout-raw', 'heldout-raw-dir'})
                + fileaudit.events_with(r['events'], {'data-dir'}, data.HELDOUT_ARCHIVE)
                + [e for e in r['events'] if e['overflow']])

    py = [sys.executable, '-I', '-B', '-c']
    try:
        # Chang's get_data on the placeholder directory
        r_ref = fileaudit.run_audited(py + [REF, str(FORK / 'chang' / 'data.py'), str(ph), str(ref_out)],
                                      watches=watches, env=child_env(), cwd=str(work))
        g.info('reference process', {'returncode': r_ref['returncode'], 'stderr_tail': r_ref['stderr'][-600:],
                                     'events': r_ref['events']})
        g.check(r_ref['returncode'] == 0, "Chang's get_data ran on the placeholder directory")
        # the lab reader, in its own process
        r_lab = fileaudit.run_audited(py + [LAB, str(FORK), str(train), train_sha, str(lab_out)],
                                      watches=watches, env=child_env(), cwd=str(work))
        g.info('reader process', {'returncode': r_lab['returncode'], 'stderr_tail': r_lab['stderr'][-600:],
                                  'events': r_lab['events'], 'fd_paths': r_lab['fd_paths']})
        g.check(r_lab['returncode'] == 0, 'cbl.data.read_train_val ran in its own process')
        # decoy through a link
        r_dec = fileaudit.run_audited(py + [DECOY, str(link_dir / 'elsewhere.h5')], watches=watches,
                                      env=child_env(), cwd=str(work))
        g.info('decoy process', {'returncode': r_dec['returncode'], 'events': r_dec['events']})

        # bitwise equality
        ref = {k: np.load(ref_out / (k + '.npy')) for k in ARRAYS}
        lab = {k: np.load(lab_out / (k + '.npy')) for k in ARRAYS}
        for k in ARRAYS:
            g.info(k, {'dtype': str(lab[k].dtype), 'shape': list(lab[k].shape), 'sha256': data.array_sha256(lab[k])})
            g.check(same(ref[k], lab[k]), "%s: lab reader equals Chang's get_data in dtype, shape and bytes" % k)
        flipped = np.ascontiguousarray(lab['X_train']).copy()
        flipped.reshape(-1).view(np.uint8)[12345] ^= 1
        g.check(not same(ref['X_train'], flipped), 'negative control: one flipped bit in a copy of X_train breaks the comparison')

        # the split record
        rec = json.loads([ln for ln in r_lab['stdout'].splitlines() if ln.startswith('RESULT ')][-1][7:])
        keys = ('n_train', 'n_val', 'order_sha256', 'split_sha256', 'stats_sha256', 'arrays', 'std_finite', 'std_nonzero')
        diff = [k for k in keys if rec.get(k) != man['split'][k]]
        g.check(not diff and rec['n_train'] == data.EXPECTED['n_train'] and rec['n_val'] == data.EXPECTED['n_val'],
                'reader record equals the manifest split (558000 and 62000 jets, permutation, split, statistics, arrays; differs: %s)' % diff)
        refh = {k: {'sha256': data.array_sha256(ref[k]), 'dtype': str(ref[k].dtype), 'shape': list(ref[k].shape)} for k in ARRAYS}
        g.check(refh == man['split']['arrays'], "Chang's get_data arrays hash to the manifest split arrays")
        out['arrays'] = refh

        # open-file audit
        for lbl, r in (('reference', r_ref), ('reader', r_lab), ('decoy', r_dec)):
            g.check(not r['events_before_start'], '%s: no watched open before the process started' % lbl)
            fe = forbidden_events(r)
            g.check(not fe, '%s process window: no open of the held-out archive, its name in the data directory, the raw held-out directory or any raw held-out file (%d forbidden events)' % (lbl, len(fe)))
        g.check(bool(fileaudit.events_with(r_lab['events'], {'train'})), 'positive control: the reader\'s h5py open of the training archive is seen')
        g.check(bool(fileaudit.events_with(r_ref['events'], {'placeholder'})), "positive control: Chang's get_data is seen opening the placeholder 150c-test.h5 (h5py's C open)")
        g.check(r_dec['returncode'] == 0 and bool(fileaudit.events_with(r_dec['events'], {'decoy'})),
                'positive control: an open of a watched decoy through a symbolic link elsewhere is seen')
        g.check(not fileaudit.events_with(r_lab['events'], {'placeholder', 'decoy'}), 'reader process: no open of the placeholder or the decoy')
        fd_real = {os.path.realpath(p) for p in r_lab['fd_paths'] if p.startswith('/')}
        g.check(os.path.realpath(train) in fd_real, 'reader fd inventory holds the training archive')
        bad_fd = sorted(p for p in fd_real if p in forbidden_real or os.path.basename(p) == data.HELDOUT_ARCHIVE
                        or p.startswith(os.path.realpath(val_dir) + os.sep))
        g.check(not bad_fd, 'reader fd inventory (%d paths) holds no held-out path and nothing named 150c-test.h5' % len(fd_real))
        out['audit'] = {k: {'events': r['events'], 'fd_paths': r['fd_paths']} for k, r in
                        (('reference', r_ref), ('reader', r_lab), ('decoy', r_dec))}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    g.check(not work.exists(), 'work directory deleted')
    g.check_clean('at end')
    return g.finish(out, a.json)


if __name__ == '__main__':
    sys.exit(main())
