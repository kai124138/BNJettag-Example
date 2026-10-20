"""Data of chang-bitlinear-20261016 (module M3; STUDY v0.8 sections 4 and 8.5; gates G06-G08).

STUDY means `review/snapshots/v0.8/STUDY.md` (sha256 290022fa41a726b8...); "plan" means
PREFLIGHT-PLAN.md (sha256 2f1be9f3...30ff8), section 2 row M3 and section 9 items 12-13.

What this module does
- Preparation with Chang's script unchanged. `run_chang_prepare` runs `chang/tools/prepare_data.py`
  by path, in a child process of this interpreter, exactly as `prepare_data.sh:21-22` does
  (`-i <dir>/ -o <archive> -j 4`, working directory the output directory). The script is checked
  against its `code-base.txt` hash first; nothing in it is patched or wrapped.
- Glob order. The script lists its input with `Path(inp).glob('*.h5')` (`prepare_data.py:21`),
  which follows `os.scandir` order and is unsorted (STUDY section 4 "Split identity").
  `glob_order` makes the same call on the same directory immediately before and after the run;
  the two lists must agree. That is the secondary record. The primary evidence is content:
  `verify_blocks` shows that block k of the archive (rows offset_k to offset_k + rows_k) equals
  file k of that order, by sha256 of the float16 feature block and of the label block for the
  training archive, and of the label block only for the held-out archive, whose features this
  module never reads back.
- Jet identity is (file in glob order, row in that file) (plan section 9 item 12). The manifest
  lists every file in glob order with its offset, so archive row r is file k with
  offset_k <= r < offset_k + rows_k, row r - offset_k (`jet_identity`). The split is the
  permutation `order` of `data.py:35-36`; train jets are archive rows order[:n_train] and
  validation jets order[n_train:]. `order_sha256` and `split_sha256` identify it.
- Train/validation reader. `read_train_val` is `chang/data.py:7-43` with every line that touches
  `150c-test.h5` removed (STUDY section 4 "Held-out test set": the I/O wrapper is split so that
  training touches only train and validation; a holdout-isolation departure, not a preprocessing
  change). The remaining statements are copied in their order and dtype, including the in-place
  `*=` of the pT gate, the float32 `np.std` and `np.mean` over axes (0, 1) with keepdims, the
  float16 cast after standardization and the shuffle on the global legacy NumPy generator. Two
  guards are added; neither changes an output: the generator must be in the state that
  `np.random.seed(42)` leaves (`run_train.py:41`, then no draw before `data.py:36`), and a zero
  or nonfinite std raises (STUDY section 4 "A zero or nonfinite std is an integrity failure").
  It takes the training archive's path, never a directory, so it cannot build the test path,
  and it refuses a file named `150c-test.h5`.
- Sealed held-out reader. `read_heldout` refuses unless a section-8.5 freeze manifest is given.
  `check_freeze_manifest` applies M15's rule (`cbl.wandb_log.freeze_manifest_sha256`: a path is
  given, the file exists and is a non-empty JSON object) and then M16's completeness check
  (`cbl.select.require_freeze_manifest`, gate G56; joined in batch B5), both called, not copied,
  so one rule serves every reader (interfaces.md sections 9 and 11). Both are checked before the
  archive is touched. Then the archive's sha256 must equal the value the caller takes from the
  committed preparation manifest. Only then does it read, with Chang's test-path
  statements (`data.py:13-15,19-20,24,26,31,33`) and the training archive's shift and scale.
- Provenance. `tarball_record` hashes a downloaded tarball (md5 for Zenodo's record, sha256) and
  every member, in one streaming pass that never extracts anything.
- Disjointness. `jet_digests` gives a per-jet sha256 of the raw `jetConstituentList` row bytes;
  only the size of the intersection between training and held-out digests is recorded.

Held-out discipline: for the held-out archive and the raw held-out files this module computes
only file hashes, row counts, per-class counts, label-block hashes and per-jet digests; it never
returns or records a held-out feature value outside `read_heldout`.

numpy and h5py only (h5py imported inside the functions that need it); no Keras, JAX or
TensorFlow import. Every reading is engineering, not quotable.

CLI (the preparation of record, on the PC, at a committed sha):

    <venv>/bin/python -I -B cbl/data.py prepare --train-dir D --val-dir D --out-dir D \\
        --smoke-src-dir D --smoke-dir D --manifest-out F [--zenodo-dir D --zenodo-api-json F] [--jobs 4]
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

FORK = Path(__file__).resolve().parents[1]
CODE_BASE = FORK / 'code-base.txt'
CHANG_PREPARE = FORK / 'chang' / 'tools' / 'prepare_data.py'
CHANG_DATA = FORK / 'chang' / 'data.py'
TRAIN_ARCHIVE = '150c-train.h5'      # prepare_data.sh:21; data.py:10
HELDOUT_ARCHIVE = '150c-test.h5'     # prepare_data.sh:22; data.py:13
N_CLASSES = 5                        # jets[:, -6:-1], prepare_data.py:12
SPLIT_SEED = 42                      # run_train.py:41
VAL_SIZE = 0.1                       # data.py:7
MANIFEST_KIND = 'cbl1016-data-prep-manifest'
MANIFEST_PATH = FORK / 'manifests' / 'data-prep-150c.json'   # committed after the preparation of record
MANIFEST_SCHEMA = 1
# STUDY section 4 (derived there from 620,000 x 0.9 / 0.1) and DATASET.md:9-19 (26 files, 260,000 jets).
EXPECTED = {'train_files': 62, 'train_jets': 620000, 'n_train': 558000, 'n_val': 62000,
            'heldout_files': 26, 'heldout_jets': 260000}
BUF = 1 << 23


class DataIdentityError(ValueError):
    """The data differ from their declared identity, or fail an integrity rule of STUDY section 4."""


class HeldOutSealed(PermissionError):
    """The held-out test set was asked for without a valid section-8.5 freeze manifest."""


# ----------------------------------------------------------------------------------------------
# hashes


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(BUF)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def array_sha256(a):
    """sha256 of the C-ordered bytes (the derived_view.py:52-54 pattern); dtype and shape are
    recorded beside it, not hashed."""
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def code_base_sha256(fork_path):
    """The sha256 that `code-base.txt` records for a fork path such as 'chang/tools/prepare_data.py'."""
    for line in CODE_BASE.read_text(encoding='utf-8').splitlines():
        parts = line.split('  ')
        if len(parts) == 4 and parts[0] == 'file' and parts[2] == fork_path:
            return parts[1]
    raise DataIdentityError(f'code-base.txt has no entry for {fork_path}')


def check_chang_unchanged():
    """Chang's prepare_data.py and data.py equal their code-base.txt hashes; returns both."""
    out = {}
    for rel, p in (('chang/tools/prepare_data.py', CHANG_PREPARE), ('chang/data.py', CHANG_DATA)):
        want, got = code_base_sha256(rel), file_sha256(p)
        if got != want:
            raise DataIdentityError(f'{rel} sha256 {got} differs from code-base.txt {want}')
        out[rel] = got
    return out


def versions():
    import h5py
    rec = {'python': sys.version.split()[0], 'numpy': np.__version__, 'h5py': h5py.version.version,
           'hdf5': h5py.version.hdf5_version, 'interpreter': sys.executable, 'prefix': sys.prefix,
           'platform': platform.platform(), 'host': platform.node()}
    try:
        import importlib.metadata as md
        rec['tqdm'] = md.version('tqdm')
    except Exception:  # noqa: BLE001 - recorded, not required here
        rec['tqdm'] = None
    return rec


# ----------------------------------------------------------------------------------------------
# glob order and Chang's preparation


def glob_order(directory):
    """File names in the order `prepare_data.py:20-21` sees them: list(Path(inp).glob('*.h5'))."""
    return [p.name for p in Path(directory).glob('*.h5')]


def run_chang_prepare(inp_dir, out_path, *, jobs=4, python=None, log_dir=None):
    """Run Chang's prepare_data.py unchanged on `inp_dir`, writing `out_path`.

    The command is that of prepare_data.sh:21-22 (`python3 prepare_data.py -i <dir>/ -o <out>
    -j 4`, run from the output directory) with this interpreter under -I -B. The glob order is
    taken with the same call just before and just after; a difference raises. Returns the run
    record; raises DataIdentityError on a nonzero exit (the one-hot assertion of
    prepare_data.py:13 fails that way)."""
    check_chang_unchanged()
    inp_dir, out_path = Path(inp_dir), Path(out_path)
    if out_path.exists():
        raise DataIdentityError(f'refusing to overwrite {out_path}')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    python = python or sys.executable
    cmd = [python, '-I', '-B', str(CHANG_PREPARE), '-i', str(inp_dir) + '/', '-o', str(out_path), '-j', str(jobs)]
    before = glob_order(inp_dir)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(out_path.parent), capture_output=True, text=True)
    seconds = time.time() - t0
    after = glob_order(inp_dir)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stem = out_path.parent.name + '.' + out_path.stem   # distinct for the archive of record and the smoke archive
        (log_dir / f'{stem}.stdout.txt').write_text(r.stdout)
        (log_dir / f'{stem}.stderr.txt').write_text(r.stderr)
    rec = {'command': cmd, 'cwd': str(out_path.parent), 'returncode': r.returncode,
           'seconds': round(seconds, 1), 'glob_order': before, 'glob_order_after_equal': before == after,
           'stderr_tail': r.stderr[-400:]}
    if r.returncode != 0:
        raise DataIdentityError(f'prepare_data.py exited {r.returncode}: {r.stderr[-2000:]}')
    if before != after:
        raise DataIdentityError(f'glob order of {inp_dir} changed during the run')
    return rec


# ----------------------------------------------------------------------------------------------
# raw files, archives, block verification


def raw_file_record(path, *, feature_block=True, digests=False):
    """Hashes and counts of one raw HLS4ML file.

    Labels are `argmax(jets[:, -6:-1])` with the one-hot test of prepare_data.py:12-14.
    feature_block: also the sha256 of `jetConstituentList` cast to float16 (prepare_data.py:15),
    which is block k of the archive when this file is file k. digests: also return the per-jet
    sha256 of the raw `jetConstituentList` rows (not recorded, used for disjointness)."""
    import h5py
    path = Path(path)
    rec = {'name': path.name, 'bytes': path.stat().st_size, 'sha256': file_sha256(path)}
    with h5py.File(path, 'r') as f:
        jets_shape = tuple(f['jets'].shape)
        cl_shape = tuple(f['jetConstituentList'].shape)
        _label = f['jets'][:, -6:-1]
        one_hot = bool(np.all(np.sum(_label, axis=1) == 1))
        label = np.argmax(_label, axis=1)
        rec.update(rows=int(jets_shape[0]), jets_shape=list(jets_shape), constituents_shape=list(cl_shape),
                   constituents_dtype=str(f['jetConstituentList'].dtype),
                   one_hot=one_hot, label_counts=np.bincount(label, minlength=N_CLASSES).tolist(),
                   label_dtype=str(label.dtype), label_block_sha256=array_sha256(label))
        if cl_shape[0] != jets_shape[0]:
            raise DataIdentityError(f'{path.name}: jets {jets_shape} and jetConstituentList {cl_shape} disagree')
        dig = None
        if feature_block or digests:
            raw = np.array(f['jetConstituentList'])
            if feature_block:
                rec['feature_block_sha256'] = array_sha256(raw.astype(np.float16))
            if digests:
                dig = jet_digests(raw)
            del raw
    return rec, dig


def jet_digests(raw):
    """Per-jet sha256 (first 16 bytes) of the C-ordered bytes of each row of a raw array."""
    raw = np.ascontiguousarray(raw)
    out = np.empty((raw.shape[0], 16), dtype=np.uint8)
    for i in range(raw.shape[0]):
        out[i] = np.frombuffer(hashlib.sha256(raw[i].view(np.uint8)).digest()[:16], dtype=np.uint8)
    return out


def digest_intersection(a, b):
    """Number of distinct per-jet digests present in both arrays (n, 16) uint8."""
    sa = {bytes(r) for r in a}
    return sum(1 for r in {bytes(r) for r in b} if r in sa)


def archive_record(path, *, read_features):
    """sha256, dataset layout and per-class counts of a prepared archive. read_features is False
    for the held-out archive: its `feature` dataset is then described from metadata only."""
    import h5py
    path = Path(path)
    rec = {'name': path.name, 'bytes': path.stat().st_size, 'sha256': file_sha256(path)}
    with h5py.File(path, 'r') as f:
        rec['datasets'] = {k: {'shape': list(f[k].shape), 'dtype': str(f[k].dtype),
                               'compression': f[k].compression, 'chunks': list(f[k].chunks or [])}
                           for k in ('feature', 'label')}
        label = np.array(f['label'])
        rec['rows'] = int(label.shape[0])
        rec['label_counts'] = np.bincount(label, minlength=N_CLASSES).tolist()
        rec['label_sha256'] = array_sha256(label)
        if read_features:
            h = hashlib.sha256()
            n = f['feature'].shape[0]
            for a in range(0, n, 20000):
                h.update(np.ascontiguousarray(f['feature'][a:a + 20000]).view(np.uint8))
            rec['feature_sha256'] = h.hexdigest()
    return rec


def verify_blocks(archive, files, *, read_features):
    """Block k of the archive equals file k (glob order). `files` are raw_file_record dicts in
    glob order. Returns {ok, blocks: [...], distinct_label_blocks, ...}; label blocks must be
    pairwise distinct for the match to identify the order."""
    import h5py
    blocks, off, ok = [], 0, True
    with h5py.File(archive, 'r') as f:
        n_rows = int(f['label'].shape[0])
        for k, fr in enumerate(files):
            rows = fr['rows']
            lab = np.array(f['label'][off:off + rows])
            b = {'index': k, 'name': fr['name'], 'offset': off, 'rows': rows,
                 'label_equal': array_sha256(lab) == fr['label_block_sha256']}
            if read_features:
                feat = np.array(f['feature'][off:off + rows])
                b['feature_equal'] = array_sha256(feat) == fr['feature_block_sha256']
            ok = ok and b['label_equal'] and b.get('feature_equal', True)
            blocks.append(b)
            off += rows
    distinct = len({fr['label_block_sha256'] for fr in files}) == len(files)
    rec = {'ok': bool(ok and off == n_rows and distinct), 'rows_files': off, 'rows_archive': n_rows,
           'distinct_label_blocks': distinct, 'read_features': read_features,
           'blocks_failed': [b['index'] for b in blocks if not (b['label_equal'] and b.get('feature_equal', True))]}
    return rec


def jet_identity(manifest_archive, row):
    """(glob index, file name, row in file) of archive row `row`, from a manifest archive entry."""
    for fe in manifest_archive['files']:
        if fe['offset'] <= row < fe['offset'] + fe['rows']:
            return fe['index'], fe['name'], row - fe['offset']
    raise IndexError(row)


# ----------------------------------------------------------------------------------------------
# readers


class TrainVal(NamedTuple):
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    shift: np.ndarray
    scale: np.ndarray
    record: dict


def _rng_state_equals_seed(seed):
    s, r = np.random.get_state(), np.random.RandomState(seed).get_state()
    return s[0] == r[0] and np.array_equal(s[1], r[1]) and s[2:] == r[2:]


def read_train_val(train_archive, n_constituents, ptetaphi, val_size=VAL_SIZE, *,
                   expect_sha256=None, expect_seed=SPLIT_SEED):
    """`chang/data.py:7-43` without the held-out lines. Returns TrainVal.

    The caller seeds the global legacy generator as run_train.py:41 does; with expect_seed set
    (the default, 42) the reader refuses unless the generator is exactly in that state, so no
    draw happened in between. expect_sha256: the archive's sha256 from the preparation manifest
    (the file is hashed before it is read)."""
    import h5py
    path = Path(train_archive)
    if path.name == HELDOUT_ARCHIVE:
        raise HeldOutSealed(f'read_train_val refuses {path}: that is the held-out archive name')
    rec = {'archive': str(path), 'n_constituents': int(n_constituents), 'ptetaphi': bool(ptetaphi),
           'val_size': val_size}
    if expect_sha256 is not None:
        got = file_sha256(path)
        if got != expect_sha256:
            raise DataIdentityError(f'{path} sha256 {got} differs from the manifest {expect_sha256}')
        rec['archive_sha256'] = got
    if expect_seed is not None:
        if not _rng_state_equals_seed(expect_seed):
            raise DataIdentityError(f'the global NumPy generator is not in the state np.random.seed({expect_seed}) '
                                    'leaves (run_train.py:41); a draw happened before the split')
        rec['rng_state_at_entry'] = f'np.random.seed({expect_seed})'

    # data.py:10-12
    with h5py.File(path) as f:
        X_train_val = np.array(f['feature'][:, :n_constituents]).astype(np.float16)  # type: ignore
        y_train_val = np.array(f['label'])
    # data.py:17-18
    X_train_val = X_train_val.astype(np.float32)
    y_train_val = y_train_val.astype(np.int32)
    # data.py:22-23,25
    if ptetaphi:
        X_train_val = X_train_val[..., [5, 8, 11]]
        X_train_val *= X_train_val[..., :1] >= 2
    # data.py:28-30
    scale = np.std(X_train_val, axis=(0, 1), keepdims=True)
    shift = np.mean(X_train_val, axis=(0, 1), keepdims=True)
    X_train_val = (X_train_val - shift) / scale  # type: ignore
    # data.py:33 (train part)
    X_train_val = X_train_val.astype(np.float16)
    # data.py:35-42
    order = np.arange(len(X_train_val))
    np.random.shuffle(order)
    X_train_val, y_train_val = X_train_val[order], y_train_val[order]
    n_train = int((1 - val_size) * len(X_train_val))
    X_train, X_val = X_train_val[:n_train], X_train_val[n_train:]
    y_train, y_val = y_train_val[:n_train], y_train_val[n_train:]

    finite = bool(np.all(np.isfinite(scale)))
    nonzero = bool(np.all(scale != 0))
    rec.update(
        n_jets=int(len(order)), n_train=int(n_train), n_val=int(len(order) - n_train),
        order_dtype=str(order.dtype), order_sha256=array_sha256(order),
        split_sha256=hashlib.sha256(np.ascontiguousarray(np.sort(order[:n_train])).view(np.uint8)).hexdigest(),
        stats_sha256=hashlib.sha256(np.ascontiguousarray(shift).view(np.uint8).tobytes()
                                    + np.ascontiguousarray(scale).view(np.uint8).tobytes()).hexdigest(),
        stats_dtype=str(scale.dtype), stats_shape=list(scale.shape),
        std_finite=finite, std_nonzero=nonzero,
        arrays={k: {'sha256': array_sha256(v), 'dtype': str(v.dtype), 'shape': list(v.shape)}
                for k, v in (('X_train', X_train), ('y_train', y_train), ('X_val', X_val), ('y_val', y_val))},
    )
    if not (finite and nonzero):
        raise DataIdentityError(f'per-feature std is zero or nonfinite (finite {finite}, nonzero {nonzero}); '
                                'STUDY section 4: integrity failure')
    return TrainVal(X_train, y_train, X_val, y_val, shift, scale, rec)


def check_freeze_manifest(path):
    """The section-8.5 freeze manifest check of the sealed reader, in two steps, both called, not
    copied, and both before the archive is touched:

    1. M15's rule (`cbl.wandb_log.freeze_manifest_sha256`): a path is given, the file exists and
       holds a non-empty JSON object. It returns the sha256 of the file's bytes.
    2. M16's completeness check (`cbl.select.require_freeze_manifest`, gate G56): the manifest
       parses, has every section-8.5 field and its own `manifest_sha256` (the sha256 of its
       canonical content) matches. Joined in batch B5 (interfaces.md sections 9 and 11).

    Returns {'file_sha256', 'manifest_sha256'}; raises HeldOutSealed. `cbl.select` is imported
    here, not at module level, because it imports scipy (through cbl.stats)."""
    try:
        from cbl.wandb_log import HeldOutKeyRefused, freeze_manifest_sha256
    except ImportError:  # run as a file: the fork is not on sys.path yet
        sys.path.insert(0, str(FORK))
        from cbl.wandb_log import HeldOutKeyRefused, freeze_manifest_sha256
    try:
        file_sha = freeze_manifest_sha256(path)
    except HeldOutKeyRefused as e:
        raise HeldOutSealed(f'held-out read refused: {e}') from None
    from cbl.select import FreezeRefused, require_freeze_manifest
    try:
        manifest = require_freeze_manifest(path)
    except FreezeRefused as e:
        raise HeldOutSealed(f'held-out read refused: {e}') from None
    if file_sha != freeze_manifest_sha256(path):
        raise HeldOutSealed('held-out read refused: the freeze manifest changed while it was checked')
    return {'file_sha256': file_sha, 'manifest_sha256': manifest['manifest_sha256']}


def read_heldout(archive, *, freeze_manifest, expect_archive_sha256, shift, scale, n_constituents, ptetaphi):
    """The held-out test set, sealed (STUDY sections 4 and 8.5).

    Order of checks: the freeze manifest, M15's rule then M16's completeness (no access to the
    archive before both pass; check_freeze_manifest); the
    expected archive sha256 is a 64-hex string; the archive's sha256 equals it; shift and scale
    are the float32 statistics of the training archive (read_train_val). Then the statements of
    chang/data.py:13-15,19-20,24,26,31,33 for X_test and y_test. Returns (X_test, y_test, record)."""
    freeze = check_freeze_manifest(freeze_manifest)
    if not (isinstance(expect_archive_sha256, str) and len(expect_archive_sha256) == 64
            and all(c in '0123456789abcdef' for c in expect_archive_sha256)):
        raise HeldOutSealed('held-out read refused: no valid expected archive sha256 (take it from the '
                            'committed preparation manifest)')
    shift, scale = np.asarray(shift), np.asarray(scale)
    if shift.dtype != np.float32 or scale.dtype != np.float32 or shift.ndim != 3 or scale.shape != shift.shape:
        raise DataIdentityError('shift and scale must be the float32 (1, 1, F) statistics of read_train_val')
    path = Path(archive)
    got = file_sha256(path)
    if got != expect_archive_sha256:
        raise HeldOutSealed(f'held-out read refused: {path} sha256 differs from the expected archive sha256')
    import h5py
    # data.py:13-15
    with h5py.File(path) as f:
        X_test = np.array(f['feature'][:, :n_constituents]).astype(np.float16)  # type: ignore
        y_test = np.array(f['label'])
    # data.py:19-20
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.int32)
    # data.py:22,24,26
    if ptetaphi:
        X_test = X_test[..., [5, 8, 11]]
        X_test *= X_test[..., :1] >= 2
    # data.py:31,33 (test part)
    X_test = (X_test - shift) / scale  # type: ignore
    X_test = X_test.astype(np.float16)
    rec = {'archive': str(path), 'archive_sha256': got, 'freeze_manifest_sha256': freeze['file_sha256'],
           'freeze_manifest_content_sha256': freeze['manifest_sha256'], 'n': int(len(y_test)), 'n_constituents': int(n_constituents), 'ptetaphi': bool(ptetaphi)}
    return X_test, y_test, rec


# ----------------------------------------------------------------------------------------------
# the runtime held-out guard of a training process (batch B10, gate G51)


class HeldOutAccessRefused(HeldOutSealed):
    """A training process tried to open a held-out file (STUDY section 13 "validation-only tracing and selection")."""


_GUARD = None


class HeldOutGuard:
    """Refuses, in the process that installs it, every open of a held-out file. Installed by the runner (M13) before
    it reads any data; it cannot be removed (an audit hook lasts as long as the process).

    Which files: every path whose name, or whose resolved name, is the held-out archive's (`150c-test.h5`) or one of
    the 26 raw held-out files' (the preparation manifest's `heldout` block); every path under the raw held-out
    directory of record (`paths.val_dir`); and every path that resolves to the inode of one of those files on this
    machine (a hard link or a copy under another name is caught by inode only if it is a hard link). Which opens: the
    `open` audit event (builtins.open, io.open, os.open and so every Python-level reader, NumPy's included), and
    h5py's `File`, which opens in C below the audit hook, through `h5py._hl.files.make_fid`. Both checks run before
    the file is opened, so the kernel open-file audit (gates/fileaudit.py) sees no open of a refused file. Checking a
    name, resolving a path and `stat` open nothing."""

    def __init__(self, manifest, *, extra_paths=()):
        hb = manifest['heldout']
        self.names = frozenset({HELDOUT_ARCHIVE, hb['archive']['name']} | {f['name'] for f in hb['files']})
        paths = manifest.get('paths') or {}
        self.dirs = tuple(os.path.realpath(d) for d in [paths.get('val_dir')] if d)
        known = []
        if paths.get('out_dir'):
            known.append(Path(paths['out_dir']) / hb['archive']['name'])
        if paths.get('val_dir'):
            known += [Path(paths['val_dir']) / f['name'] for f in hb['files']]
        known += [Path(p) for p in extra_paths]
        self.inodes = {}
        for p in known:
            try:
                st = os.stat(p)
            except OSError:
                continue
            self.inodes[(st.st_dev, st.st_ino)] = str(p)
        self.refused = []
        self.h5py_patched = False

    def offends(self, path):
        """The reason `path` names a held-out file, or None."""
        if isinstance(path, int):
            return None
        try:
            p = os.fsdecode(os.fspath(path))
        except TypeError:
            return None
        if os.path.basename(p) in self.names:
            return 'held-out file name'
        real = os.path.realpath(p)
        if os.path.basename(real) in self.names:
            return 'resolves to a held-out file name'
        for d in self.dirs:
            if real == d or real.startswith(d + os.sep):
                return 'under the raw held-out directory'
        if self.inodes:
            try:
                st = os.stat(real)
            except OSError:
                return None
            if (st.st_dev, st.st_ino) in self.inodes:
                return 'the inode of a held-out file'
        return None

    def check(self, path, via):
        why = self.offends(path)
        if why is not None:
            self.refused.append({'path': os.fsdecode(os.fspath(path)), 'via': via, 'why': why})
            raise HeldOutAccessRefused(f'held-out access refused in a training process ({via}: {why}): {path}')

    def audit(self, event, args):
        if event == 'open' and args:
            self.check(args[0], 'open')

    def patch_h5py(self):
        import h5py._hl.files as files

        orig = files.make_fid
        guard = self

        def make_fid(name, *a, **kw):
            guard.check(name, 'h5py.File')
            return orig(name, *a, **kw)
        make_fid.__wrapped__ = orig
        files.make_fid = make_fid
        self.h5py_patched = True

    def record(self) -> dict:
        return {'installed': True, 'names': sorted(self.names), 'dirs': list(self.dirs),
                'known_inodes': sorted(self.inodes.values()), 'h5py_patched': self.h5py_patched,
                'refused': list(self.refused)}


def install_heldout_guard(manifest, *, extra_paths=()) -> dict:
    """Install the held-out guard in this process once (HeldOutGuard); returns its record. A second call returns the
    installed guard's record (the names must agree)."""
    global _GUARD
    if _GUARD is not None:
        g = HeldOutGuard(manifest, extra_paths=extra_paths)
        if g.names != _GUARD.names:
            raise HeldOutSealed('a held-out guard with other names is already installed')
        return _GUARD.record()
    g = HeldOutGuard(manifest, extra_paths=extra_paths)
    g.patch_h5py()
    sys.addaudithook(g.audit)
    _GUARD = g
    return g.record()


def heldout_guard():
    """The installed guard, or None."""
    return _GUARD


# ----------------------------------------------------------------------------------------------
# provenance


def tarball_record(path, *, expect_md5=None):
    """md5 and sha256 of a tarball, and name, size and sha256 of every regular member, in one
    streaming pass (tarfile mode 'r|gz'); nothing is extracted."""
    path = Path(path)
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(BUF)
            if not b:
                break
            md5.update(b)
            sha.update(b)
    rec = {'name': path.name, 'bytes': path.stat().st_size, 'md5': md5.hexdigest(), 'sha256': sha.hexdigest(),
           'md5_expected': expect_md5, 'md5_equal': (md5.hexdigest() == expect_md5) if expect_md5 else None}
    members, other = [], []
    with tarfile.open(path, mode='r|gz') as tf:
        for m in tf:
            if m.isfile():
                h = hashlib.sha256()
                src = tf.extractfile(m)
                while True:
                    b = src.read(BUF)
                    if not b:
                        break
                    h.update(b)
                members.append({'path': m.name, 'size': m.size, 'sha256': h.hexdigest()})
            else:
                other.append({'path': m.name, 'type': m.type.decode('ascii', 'replace') if isinstance(m.type, bytes) else str(m.type)})
    rec['members'] = members
    rec['non_file_members'] = other
    return rec


def compare_members(tar_rec, files):
    """Each on-disk file (raw_file_record) against the tar member with the same base name."""
    by_base = {}
    for m in tar_rec['members']:
        by_base.setdefault(os.path.basename(m['path']), []).append(m)
    match, mismatch, absent = [], [], []
    for fr in files:
        ms = by_base.get(fr['name'], [])
        if not ms:
            absent.append(fr['name'])
        elif len(ms) == 1 and ms[0]['sha256'] == fr['sha256'] and ms[0]['size'] == fr['bytes']:
            match.append(fr['name'])
        else:
            mismatch.append(fr['name'])
    names = {fr['name'] for fr in files}
    tar_only = sorted(m['path'] for m in tar_rec['members'] if os.path.basename(m['path']) not in names)
    return {'match': len(match), 'mismatch': mismatch, 'not_in_tarball': absent, 'tarball_only': tar_only,
            'member_order': [m['path'] for m in tar_rec['members']]}


def zenodo_expected(api_json):
    """{file key: {'size', 'md5'}} from the saved Zenodo API record."""
    d = json.loads(Path(api_json).read_text(encoding='utf-8'))
    out = {}
    for f in d.get('files', []):
        ck = f.get('checksum', '')
        out[f['key']] = {'size': f.get('size'), 'md5': ck.split(':', 1)[1] if ck.startswith('md5:') else None}
    return {'record_id': d.get('id'), 'doi': d.get('doi'), 'title': d.get('metadata', {}).get('title'),
            'updated': d.get('updated'), 'files': out}


# ----------------------------------------------------------------------------------------------
# the preparation of record


def _archive_entry(rec_run, files, arch, blocks):
    off, fl = 0, []
    for k, fr in enumerate(files):
        e = {kk: fr[kk] for kk in ('name', 'bytes', 'sha256', 'rows', 'one_hot', 'label_counts', 'label_block_sha256')}
        if 'feature_block_sha256' in fr:
            e['feature_block_sha256'] = fr['feature_block_sha256']
        e.update(index=k, offset=off)
        off += fr['rows']
        fl.append(e)
    return {'glob_order': rec_run['glob_order'], 'glob_order_after_equal': rec_run['glob_order_after_equal'],
            'prepare_command': rec_run['command'], 'prepare_cwd': rec_run['cwd'],
            'prepare_returncode': rec_run['returncode'], 'prepare_seconds': rec_run['seconds'],
            'files': fl, 'n_files': len(fl), 'rows': off, 'archive': arch, 'blocks': blocks}


def prepare_record(*, train_dir, val_dir, out_dir, smoke_src_dir, smoke_dir, jobs=4,
                   zenodo_dir=None, zenodo_api_json=None, log_dir=None, code_sha=None):
    """Prepare both archives and the 2-file smoke archive with Chang's script and return the
    manifest (hashes and counts only)."""
    import h5py  # noqa: F401 - fail early if absent
    train_dir, val_dir, out_dir = Path(train_dir), Path(val_dir), Path(out_dir)
    smoke_src_dir, smoke_dir = Path(smoke_src_dir), Path(smoke_dir)
    man = {'kind': MANIFEST_KIND, 'schema': MANIFEST_SCHEMA, 'label': 'engineering, not quotable',
           'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'prepared_at_code_sha': code_sha,
           'chang_sha256': check_chang_unchanged(), 'versions': versions(), 'expected': dict(EXPECTED),
           'paths': {'train_dir': str(train_dir), 'val_dir': str(val_dir), 'out_dir': str(out_dir),
                     'smoke_src_dir': str(smoke_src_dir), 'smoke_dir': str(smoke_dir)},
           'identity': 'jet = (file index in glob order, row in that file); archive row r -> files[k] with '
                       'offset <= r < offset + rows; train jets = archive rows order[:n_train]'}

    # raw files, in the glob order the script will see
    tr_order, va_order = glob_order(train_dir), glob_order(val_dir)
    tr_files, tr_dig = [], []
    for n in tr_order:
        fr, dig = raw_file_record(train_dir / n, feature_block=True, digests=True)
        tr_files.append(fr)
        tr_dig.append(dig)
    va_files, va_dig = [], []
    for n in va_order:
        fr, dig = raw_file_record(val_dir / n, feature_block=False, digests=True)
        va_files.append(fr)
        va_dig.append(dig)

    # Chang's script, unchanged
    run_tr = run_chang_prepare(train_dir, out_dir / TRAIN_ARCHIVE, jobs=jobs, log_dir=log_dir)
    run_va = run_chang_prepare(val_dir, out_dir / HELDOUT_ARCHIVE, jobs=jobs, log_dir=log_dir)
    if run_tr['glob_order'] != tr_order or run_va['glob_order'] != va_order:
        raise DataIdentityError('glob order at preparation differs from the order the raw files were hashed in')
    arch_tr = archive_record(out_dir / TRAIN_ARCHIVE, read_features=True)
    arch_va = archive_record(out_dir / HELDOUT_ARCHIVE, read_features=False)
    blk_tr = verify_blocks(out_dir / TRAIN_ARCHIVE, tr_files, read_features=True)
    blk_va = verify_blocks(out_dir / HELDOUT_ARCHIVE, va_files, read_features=False)
    man['train'] = _archive_entry(run_tr, tr_files, arch_tr, blk_tr)
    man['heldout'] = _archive_entry(run_va, va_files, arch_va, blk_va)

    # split of record: CH-8's configuration (get_model('xfm', ..., 8, True); run_train.py:71,77)
    np.random.seed(SPLIT_SEED)
    tv = read_train_val(out_dir / TRAIN_ARCHIVE, 8, True, expect_sha256=arch_tr['sha256'])
    man['split'] = {k: tv.record[k] for k in ('n_constituents', 'ptetaphi', 'val_size', 'n_jets', 'n_train', 'n_val',
                                              'order_dtype', 'order_sha256', 'split_sha256', 'stats_sha256',
                                              'stats_dtype', 'stats_shape', 'std_finite', 'std_nonzero', 'arrays')}
    man['split']['seed'] = SPLIT_SEED
    man['split']['label_counts'] = {'train': np.bincount(tv.y_train, minlength=N_CLASSES).tolist(),
                                    'val': np.bincount(tv.y_val, minlength=N_CLASSES).tolist()}
    stats_values = {'shift': tv.shift.ravel().tolist(), 'scale': tv.scale.ravel().tolist()}
    del tv

    # disjointness of training and held-out jets
    tr_all, va_all = np.concatenate(tr_dig), np.concatenate(va_dig)
    man['disjoint'] = {
        'file_names_shared': sorted(set(tr_order) & set(va_order)),
        'file_sha256_shared': len({f['sha256'] for f in tr_files} & {f['sha256'] for f in va_files}),
        'jet_digest': 'sha256[:16] of the bytes of each raw jetConstituentList row as stored (dtype %s)'
                      % ', '.join(sorted({f['constituents_dtype'] for f in tr_files + va_files})),
        'jet_digests_shared': digest_intersection(tr_all, va_all),
    }
    del tr_all, va_all, tr_dig, va_dig

    # 2-file smoke archive: the first two training files in glob order, by symbolic link
    smoke_src_dir.mkdir(parents=True, exist_ok=False)
    for n in tr_order[:2]:
        (smoke_src_dir / n).symlink_to(train_dir / n)
    sm_order = glob_order(smoke_src_dir)
    sm_files = [next(f for f in tr_files if f['name'] == n) for n in sm_order]
    run_sm = run_chang_prepare(smoke_src_dir, smoke_dir / TRAIN_ARCHIVE, jobs=jobs, log_dir=log_dir)
    arch_sm = archive_record(smoke_dir / TRAIN_ARCHIVE, read_features=True)
    blk_sm = verify_blocks(smoke_dir / TRAIN_ARCHIVE, sm_files, read_features=True)
    man['smoke'] = _archive_entry(run_sm, sm_files, arch_sm, blk_sm)
    man['smoke']['note'] = ('engineering input only (plan section 9 item 10): a different archive, so its split '
                            'and statistics differ from the archive of record')

    # provenance
    if zenodo_dir is not None:
        exp = zenodo_expected(zenodo_api_json) if zenodo_api_json else {'files': {}}
        prov = {'record': {k: exp.get(k) for k in ('record_id', 'doi', 'title', 'updated')},
                'api_json_sha256': file_sha256(zenodo_api_json) if zenodo_api_json else None, 'tarballs': {}}
        for key, files in (('hls4ml_LHCjet_150p_train.tar.gz', tr_files), ('hls4ml_LHCjet_150p_val.tar.gz', va_files)):
            p = Path(zenodo_dir) / key
            if not p.exists():
                prov['tarballs'][key] = {'present': False}
                continue
            tr = tarball_record(p, expect_md5=exp['files'].get(key, {}).get('md5'))
            cmp = compare_members(tr, files)
            prov['tarballs'][key] = {'present': True, 'bytes': tr['bytes'], 'size_expected': exp['files'].get(key, {}).get('size'),
                                     'md5': tr['md5'], 'md5_expected': tr['md5_expected'], 'md5_equal': tr['md5_equal'],
                                     'sha256': tr['sha256'], 'n_members': len(tr['members']),
                                     'non_file_members': tr['non_file_members'], 'members_vs_disk': cmp}
        man['provenance'] = prov
    return man, stats_values


def dumps(obj):
    return json.dumps(obj, indent=1, sort_keys=True) + '\n'


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='M3 preparation of record (Chang\'s script unchanged)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    for k in ('--train-dir', '--val-dir', '--out-dir', '--smoke-src-dir', '--smoke-dir', '--manifest-out'):
        p.add_argument(k, required=True)
    p.add_argument('--zenodo-dir', default=None)
    p.add_argument('--zenodo-api-json', default=None)
    p.add_argument('--log-dir', default=None)
    p.add_argument('--stats-out', default=None, help='shift and scale values (evidence, not committed)')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--code-sha', default=None)
    a = ap.parse_args(argv)
    out = Path(a.manifest_out)
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    man, stats = prepare_record(train_dir=a.train_dir, val_dir=a.val_dir, out_dir=a.out_dir,
                                smoke_src_dir=a.smoke_src_dir, smoke_dir=a.smoke_dir, jobs=a.jobs,
                                zenodo_dir=a.zenodo_dir, zenodo_api_json=a.zenodo_api_json,
                                log_dir=a.log_dir, code_sha=a.code_sha)
    out.write_text(dumps(man), encoding='utf-8')
    if a.stats_out:
        Path(a.stats_out).write_text(dumps({'label': 'engineering, not quotable; training-archive statistics',
                                            **stats}), encoding='utf-8')
    print(f'manifest {out} sha256 {file_sha256(out)}')
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(FORK))
    sys.exit(main())
