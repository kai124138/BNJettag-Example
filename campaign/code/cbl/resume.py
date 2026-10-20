"""M14: resume of a runner run from its own run directory, exactly where the killed process left it.

STUDY means `review/snapshots/v0.10/STUDY.md` (sha256 7180c24f6b55807e...): section 6 step 8 ("On resume, the
persisted module-level `random` state restores both consumers"), section 7.2 ("On resume, the integral, previous
error, lambda, epoch and sensed cost are persisted"), section 12 (the W&B run id) and the section 13 resume row
(persist the module-level `random` state, the ParetoFront record, the StopIf counters and the PID state; uninterrupted
and resumed runs identical on the home PC). "plan" means PREFLIGHT-PLAN.md (sha256 2f1be9f3...30ff8), section 2 row M14
and section 3 rows G46, G48 and G50. Interface notes: docs/interfaces.md sections 10 (the B4 note for M14), 16, 17
(`FrontSaver.snapshot`), 18 (`LabBetaPID.state`) and 19 (this module, B10). Owner decisions B10-n are in PREFLIGHT.md
(B10 section). Nothing here is quotable.

Pattern (read-only; that campaign is paused): `campaigns/2026-10-12-mha-bitnet/code/bnhgq2/mhab_resume_fit.py`, sha256
0e4abde32fea21576e2a73239a03c56ac9048d8d3a20f62b9a8aaca5ebd01643 (fork main 6a145873): a checkpoint directory per
generation with the code sha checked before anything is loaded, `latest.json` written last, two generations kept,
files written by the killed process after the checkpoint kept on disk, front files a later step deleted restored from
the checkpoint, and a resume ledger. Not taken: its `.keras` reload of the model (here the model is rebuilt by the
runner and every variable is assigned by role, below), its ParetoFront rebuilt from file names (here the record is
persisted exactly), and its own data order (here the released hgq Dataset and Keras's adapter keep drawing from the
module-level stream, which is persisted).

What a resume restores, and how
- The model is never reloaded from a `.keras` file. The resumed process runs the runner's `prepare` exactly as a fresh
  one does (data, seeds, the two Datasets, the same builder, the seed's initial-tensor file, callbacks, compile), so
  every layer is constructed with the arguments the fresh run used. Then every variable is assigned by role from the
  checkpoint's `variables.npz` and the state hash (`cbl.rng.variables_sha256` over the roles) must equal the
  checkpoint's. Finding B7-F1 (after `load_model` the MHA's softmax and its two tables come back with beta0 1e-5
  instead of Chang's 0) therefore cannot arise: no configuration is deserialized, and every `_beta` variable is
  restored by value with the rest. Gate G48 compares every layer's configuration between the uninterrupted and the
  resumed process.
- The optimizer's variables (iteration, learning rate, Adam's moments) by index and path, after Keras has built the
  optimizer in `fit`'s symbolic build.
- The module-level `random` state, NumPy's legacy state and the training Dataset's `_index_map` (Python state of its
  own, interfaces section 10). A fresh `fit(initial_epoch=k)` repeats the pre-fit draws (the symbolic build's batch
  order shuffle and fetches, then the reset's reshuffle of the map) that an uninterrupted run made once, so these
  three are restored in the appended callback's `on_train_begin`, which Keras calls after the symbolic build and the
  reset (keras/src/backend/jax/trainer.py:418-419, then :437) and after every released callback's `on_train_begin`
  (the appended callback is last). This restore is STUDY section 6 step 8's write; the stream guard is armed right
  after it (in a fresh run it is armed at the pre-fit reseed).
- StopIf's two counters (vendor/stop_if.py:54-55), Chang's ParetoFront record (`paths` and `record`,
  hgq/utils/sugar/pareto.py:28-29, values as float.hex), the lab PID's state (`LabBetaPID.state()`, restored into the
  fresh controller before `fit`), the front saver's state (`FrontSaver.snapshot()`), the frozen evaluator's
  per-site signs (so that the first record after a resume carries the same `sign_flip_rate` as the uninterrupted run)
  and the per-epoch compact records of the appended callback (rebuilt from `epochs.jsonl`).
- The W&B run id: `cbl.wandb_log.Run.open(expect_run_id=)` refuses a run whose name gives another id.

When a checkpoint is written. In a fresh run once at train begin (no epoch done yet: the state after the pre-fit
draws), and at the end of every epoch e with (e + 1) % every == 0, at the last epoch and when StopIf stopped the run,
as the last step of the appended callback's `on_epoch_end`, after every callback of that epoch (StopIf, ParetoFront,
the PID, M8, the front saver, `epochs.jsonl`, W&B) has run. At that point the module-level stream is in the state the
next epoch starts from (the Dataset's map reshuffle runs before the callbacks; the next epoch's batch-order shuffle
runs at its first fetch), which the checkpoint checks against M4's recorder. A checkpoint is a directory
`resume/ckpt-<epochs done, 5 digits>/` written under a temporary name, renamed, then named in `resume/latest.json`
(written by replace); the two newest are kept. It holds `state.json`, `variables.npz`, `optimizer.npz`,
`prev_signs.npz` and `links/`: hard links (copies where the filesystem refuses links) of the front checkpoint files
and ParetoFront files that are live at that point, because the killed process may delete a file that the restored
state still needs (`keep_superseded` false, B8-3; ParetoFront removes dominated files, pareto.py:62-65).

What a resume refuses (`ResumeRefused`, the runner's exit 2), before any file changes: no checkpoint; a finished run
(`run.json` exists); another code sha than the checkpoint's (and the fork HEAD); another config (sha256 of the
canonical config); another W&B run name or id; another run identity (seed, device, stack, matmul precision,
draw-code hashes, data and Dataset identity, `cbl.rng.SeedFlow.run_identity`); another checkpoint interval; a file
whose sha256 differs from the checkpoint's record; an append-only file shorter than recorded or with another prefix.

What a resume changes on disk (`rewind`). `epochs.jsonl` and `front/events.jsonl` are cut back to the bytes they had
at the checkpoint (the prefix's sha256 is checked first); every front file and ParetoFront file written after the
checkpoint, `front/summary.json`, temporary checkpoint directories and checkpoint directories newer than
`latest.json` are moved, never deleted, under `resume/discarded/<n>/` with their relative paths; live files the
killed process deleted are linked back from the checkpoint. `resume/resumes/<n>.json` records each resume. W&B rows
the killed process logged after the checkpoint stay where they are: offline, the resumed process writes a second
offline directory with the same run id (cbl/wandb_log.py docstring); W&B is an index only (STUDY section 12).

Imports: the standard library and NumPy at import time; Keras inside the functions that need it, after
`cbl.env.setup()`. Nothing here is quotable.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np

KIND = 'cbl1016-resume-state'
SCHEMA = 1
DIR = 'resume'
LATEST = 'latest.json'
RESUMES = 'resumes'
DISCARDED = 'discarded'
KEEP = 2
STAGES = ('tmp_written', 'renamed', 'latest_written')
CKPT_PREFIX = 'ckpt-'
TMP_PREFIX = '.tmp-ckpt-'


class ResumeError(RuntimeError):
    """A checkpoint could not be written or restored exactly."""


class ResumeRefused(ResumeError):
    """The run directory cannot be resumed exactly; nothing on disk was changed."""


# ----------------------------------------------------------------------------------------------
# pure helpers: hashing, state encodings, files


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')


def _hex(x):
    return None if x is None else float(x).hex()


def py_state_to_json(st) -> list:
    """`random.getstate()` as JSON: [version, the 625 words, gauss_next as float.hex or null]."""
    version, words, gauss = st
    return [int(version), [int(w) for w in words], _hex(gauss)]


def py_state_from_json(o) -> tuple:
    return (int(o[0]), tuple(int(w) for w in o[1]), None if o[2] is None else float.fromhex(o[2]))


def py_state_sha256(st) -> str:
    """The hash cbl.rng.py_state_sha256 gives (version, words, gauss_next as JSON)."""
    version, words, gauss = st
    return sha256_bytes(json.dumps([version, list(words), gauss]).encode())


def np_state_to_json(st) -> list:
    """NumPy's legacy global state as JSON: [name, key (624 uint32), pos, has_gauss, cached_gaussian as float.hex]."""
    name, key, pos, has_gauss, cached = st
    return [str(name), np.asarray(key, dtype='<u4').tolist(), int(pos), int(has_gauss), float(cached).hex()]


def np_state_from_json(o) -> tuple:
    return (o[0], np.asarray(o[1], dtype=np.uint32), int(o[2]), int(o[3]), float.fromhex(o[4]))


def int_list_sha256(seq) -> str:
    """As cbl.rng.int_seq_sha256: the little-endian int64 bytes of the sequence."""
    return sha256_bytes(np.asarray(list(seq), dtype='<i8').tobytes())


def prefix_record(path) -> dict:
    """{'bytes', 'sha256'} of a file's whole content now (an absent file: 0 bytes, the hash of b'')."""
    p = Path(path)
    data = p.read_bytes() if p.exists() else b''
    return {'bytes': len(data), 'sha256': sha256_bytes(data), 'exists': p.exists()}


def check_prefix(path, rec) -> dict:
    """The file still starts with the recorded bytes; returns {'bytes_now', 'tail_bytes'}; ResumeRefused otherwise."""
    p = Path(path)
    data = p.read_bytes() if p.exists() else b''
    n = rec['bytes']
    if len(data) < n:
        raise ResumeRefused(f'{p} has {len(data)} bytes, fewer than the {n} recorded at the checkpoint')
    if sha256_bytes(data[:n]) != rec['sha256']:
        raise ResumeRefused(f'{p}: its first {n} bytes differ from the checkpoint record')
    return {'bytes_now': len(data), 'tail_bytes': len(data) - n}


def cut_to_prefix(path, rec, discard_dir, rel) -> dict:
    """Move the bytes after the recorded prefix to discard_dir/rel + '.tail', then truncate. check_prefix first."""
    p = Path(path)
    info = check_prefix(p, rec)
    if info['tail_bytes']:
        data = p.read_bytes()
        out = Path(discard_dir) / (rel + '.tail')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data[rec['bytes']:])
        with open(p, 'r+b') as f:
            f.truncate(rec['bytes'])
            f.flush()
            os.fsync(f.fileno())
    return info


def files_under(root, base) -> list:
    """Sorted paths, relative to base, of the regular files under root (none when root is absent)."""
    root, base = Path(root), Path(base)
    if not root.is_dir():
        return []
    return sorted(str(p.relative_to(base)) for p in root.rglob('*') if p.is_file())


def link_or_copy(src, dst) -> str:
    """Hard-link src to dst (the files are written once and never modified); copy where links are refused."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return 'link'
    except OSError:
        shutil.copy2(src, dst)
        return 'copy'


def move_aside(src, discard_dir, rel) -> str:
    out = Path(discard_dir) / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, out)
    return str(out)


def _fsync_dir(d):
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_bytes_sync(path, data: bytes):
    with open(path, 'xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def json_bytes(obj, *, compact=False) -> bytes:
    if compact:
        return (json.dumps(obj, sort_keys=True, separators=(',', ':'), default=_jsonable) + '\n').encode('utf-8')
    return (json.dumps(obj, indent=1, sort_keys=True, default=_jsonable) + '\n').encode('utf-8')


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f'not JSON: {type(o).__name__}')


# ----------------------------------------------------------------------------------------------
# the released callbacks' state (pure on their attributes)


def stopif_state(cb) -> dict:
    """StopIf's counters (vendor/stop_if.py:54-55)."""
    return {'counter_epochs': int(cb.counter_epochs), 'counter_steps': int(cb.counter_steps)}


def restore_stopif(cb, st):
    cb.counter_epochs, cb.counter_steps = int(st['counter_epochs']), int(st['counter_steps'])


def pareto_state(pf) -> dict:
    """Chang's ParetoFront record (pareto.py:28-29): file names under its directory and the metric arrays, exactly."""
    names = []
    for p in pf.paths:
        p = Path(p)
        if p.parent != Path(pf.path):
            raise ResumeError(f'ParetoFront path {p} is not under {pf.path}')
        names.append(p.name)
    rec = [{'dtype': str(np.asarray(r).dtype), 'shape': list(np.asarray(r).shape),
            'hex': [float(x).hex() for x in np.asarray(r, dtype=np.float64).ravel()]} for r in pf.record]
    for r, e in zip(pf.record, rec):
        back = np.asarray([float.fromhex(h) for h in e['hex']], dtype=np.float64).astype(e['dtype']).reshape(e['shape'])
        if not np.array_equal(back, np.asarray(r)) or back.dtype != np.asarray(r).dtype:
            raise ResumeError('a ParetoFront record entry does not round-trip exactly through float.hex')
    return {'names': names, 'record': rec}


def restore_pareto(pf, st):
    pf.paths = [Path(pf.path) / n for n in st['names']]
    pf.record = [np.asarray([float.fromhex(h) for h in e['hex']], dtype=np.float64).astype(e['dtype']).reshape(e['shape'])
                 for e in st['record']]


def pareto_digest(pf) -> str:
    return sha256_bytes(canonical(pareto_state(pf)))


# ----------------------------------------------------------------------------------------------
# variables and the optimizer (Keras imported here)


def optimizer_arrays(optimizer) -> tuple:
    """({'<index>': array}, [path per index]) of every optimizer variable, in the optimizer's order."""
    import keras

    arrays, paths = {}, []
    for i, v in enumerate(optimizer.variables):
        arrays[f'{i:04d}'] = np.asarray(keras.ops.convert_to_numpy(v.value))
        paths.append(v.path)
    return arrays, paths


def arrays_sha256(arrays: dict, names=None) -> str:
    """One hash over (key, name, dtype, shape, bytes) of every array, in key order."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.asarray(arrays[k])
        h.update(json.dumps([k, None if names is None else names[int(k)], str(a.dtype), list(a.shape)]).encode())
        h.update(np.asarray(a, order='C').tobytes())
    return h.hexdigest()


def optimizer_sha256(optimizer) -> str:
    arrays, paths = optimizer_arrays(optimizer)
    return arrays_sha256(arrays, paths)


def _assign_checked(var, arr, what):
    import keras

    if tuple(var.shape) != tuple(arr.shape) or keras.backend.standardize_dtype(var.dtype) != str(arr.dtype):
        raise ResumeError(f'{what}: variable {tuple(var.shape)} {var.dtype} against checkpoint {arr.shape} {arr.dtype}')
    var.assign(arr)


# ----------------------------------------------------------------------------------------------
# reading a run directory's checkpoint


def resume_dir(run_dir) -> Path:
    return Path(run_dir) / DIR


def read_latest(run_dir) -> tuple:
    """(state, checkpoint directory) named by resume/latest.json, every listed file hash-checked; ResumeRefused if
    there is none or anything differs. Changes nothing."""
    rd = resume_dir(run_dir)
    lp = rd / LATEST
    if not lp.is_file():
        raise ResumeRefused(f'{run_dir} has no {DIR}/{LATEST}: nothing to resume from')
    latest = json.loads(lp.read_text(encoding='utf-8'))
    ck = rd / latest['checkpoint']
    sp = ck / 'state.json'
    if not sp.is_file():
        raise ResumeRefused(f'{sp} does not exist')
    raw = sp.read_bytes()
    if sha256_bytes(raw) != latest['state_sha256']:
        raise ResumeRefused(f'{sp} sha256 differs from {LATEST}')
    st = json.loads(raw)
    if st.get('kind') != KIND or st.get('schema') != SCHEMA:
        raise ResumeRefused(f'{sp} is not a {KIND} schema {SCHEMA} record')
    if st['epochs_done'] != latest['epochs_done'] or ck.name != checkpoint_name(st['epochs_done']):
        raise ResumeRefused(f'{LATEST} and {sp} disagree on the epochs done')
    for rel, want in st['files'].items():
        p = ck / rel
        if not p.is_file() or sha256_file(p) != want:
            raise ResumeRefused(f'checkpoint file {p} is missing or its sha256 differs from state.json')
    for rel, e in st['links'].items():
        p = ck / 'links' / rel
        if not p.is_file() or sha256_file(p) != e['sha256']:
            raise ResumeRefused(f'checkpoint link {p} is missing or its sha256 differs from state.json')
    return st, ck


def checkpoint_name(epochs_done) -> str:
    return f'{CKPT_PREFIX}{int(epochs_done):05d}'


def resume_count(run_dir) -> int:
    d = resume_dir(run_dir) / RESUMES
    return len(list(d.glob('*.json'))) if d.is_dir() else 0


def check_static(st, *, cfg, config_sha256, head, run_dir, every, run_name=None):
    """The checks a resume makes before the runner builds anything: code sha, config, interval, W&B run name (when
    given; it carries the source bundle, so the run id follows), an unfinished run."""
    bad = []
    if run_name is not None and st['run']['run_name'] != run_name:
        bad.append(f'W&B run name: checkpoint {st["run"]["run_name"]}, this run {run_name} (another source bundle, '
                   'attempt or identity)')
    if (Path(run_dir) / 'run.json').exists():
        bad.append('run.json exists: the run finished, nothing to resume')
    if st['code_sha'] != head or cfg.get('code_sha') != head:
        bad.append(f'code sha: checkpoint {st["code_sha"]}, config {cfg.get("code_sha")}, fork HEAD {head} '
                   '(RULES section 5: a fix is a new run, never a resume)')
    if st['config_sha256'] != config_sha256:
        bad.append(f'config sha256: checkpoint {st["config_sha256"]}, this config {config_sha256}')
    if st['every'] != every:
        bad.append(f'checkpoint interval: checkpoint {st["every"]}, this run {every}')
    if bad:
        raise ResumeRefused('; '.join(bad))


# ----------------------------------------------------------------------------------------------
# the run's resume object


class Resume:
    """Writes the run's checkpoints and, in a resumed process, restores the state of the checkpoint it was given.

    `ctx` is the runner's Context (cfg, flow, model, roles, released, pid, front, evaluator, hook, guard, wandb_run,
    record, out_dir). Built at the end of `prepare`, after every callback exists and before `fit`. In a resumed
    process (`state` given) the constructor checks the run identity, rewinds the run directory, restores the PID and
    the front saver; `train_begin` (the appended callback's `on_train_begin`) restores the model, the optimizer, the
    streams, the map, StopIf and ParetoFront and arms the guard. `on_stage(stage, epochs_done)`, None in runs, is a
    gate seam called after each step of a checkpoint write (STAGES)."""

    def __init__(self, ctx, *, every, state=None, checkpoint_dir=None, on_stage=None):
        if not isinstance(every, int) or isinstance(every, bool) or every < 1:
            raise ResumeError(f'the checkpoint interval must be an int >= 1, not {every!r}')
        self.ctx, self.every, self.on_stage = ctx, every, on_stage
        self.run_dir = Path(ctx.out_dir)
        self.dir = resume_dir(self.run_dir)
        self.state, self.ckpt_dir = state, checkpoint_dir
        self.written = []
        self.restored = None
        self.record = {'every': every, 'resumed': state is not None}
        if state is not None:
            self._prepare_resumed()

    @property
    def resumed(self) -> bool:
        return self.state is not None

    @property
    def epochs_done(self) -> int:
        return 0 if self.state is None else int(self.state['epochs_done'])

    # ------------------------------------------------------------------------------------------ resumed: before fit

    def _prepare_resumed(self):
        from cbl import frozen_eval

        ctx, st = self.ctx, self.state
        ident = json.loads(json.dumps(ctx.flow.run_identity(), sort_keys=True, default=_jsonable))
        bad = []
        if ident != st['identity']:
            a, b = _flatten(st['identity']), _flatten(ident)
            diff = sorted(k for k in set(a) | set(b) if a.get(k, '<absent>') != b.get(k, '<absent>'))
            bad.append('run identity differs: ' + '; '.join(f'{k}: {a.get(k, "<absent>")!r} != {b.get(k, "<absent>")!r}'
                                                            for k in diff[:6]))
        run = {'run_name': ctx.wandb_run.name, 'run_id': ctx.wandb_run.run_id}
        if {k: st['run'][k] for k in run} != run:
            bad.append(f'W&B run: checkpoint {st["run"]}, this run {run}')
        roles = sorted(ctx.roles)
        if roles != st['variables']['roles']:
            bad.append(f'roles: {len(roles)} in this model against {len(st["variables"]["roles"])} in the checkpoint')
        if ctx.pid is None and st['pid'] is not None:
            bad.append('the checkpoint holds a PID state and this run has no PID')
        if ctx.pid is not None and st['pid'] is None and st['epochs_done'] > 0:
            bad.append('this PID run\'s checkpoint after an epoch holds no PID state')
        if bad:
            raise ResumeRefused('; '.join(bad))
        n = resume_count(self.run_dir) + 1
        discard = self.dir / DISCARDED / f'{n:03d}'
        rw = rewind(self.run_dir, st, self.ckpt_dir, discard)
        if ctx.pid is not None and st['pid'] is not None:
            ctx.pid.restore(st['pid'])
        if ctx.front is not None:
            ctx.front.restore(st['front'])
        rows = [json.loads(x) for x in (self.run_dir / 'epochs.jsonl').read_text(encoding='utf-8').splitlines() if x] \
            if (self.run_dir / 'epochs.jsonl').exists() else []
        if [r['epoch'] for r in rows] != list(range(st['epochs_done'])):
            raise ResumeError(f'epochs.jsonl after the rewind holds epochs {[r["epoch"] for r in rows][:5]}..., not '
                              f'0..{st["epochs_done"] - 1}')
        ctx.hook.records[:] = [{k: r.get(k) for k in frozen_eval.COMPACT_KEYS} for r in rows]
        entry = {'n': n, 'from_checkpoint': self.ckpt_dir.name, 'epochs_done': st['epochs_done'],
                 'code_sha': st['code_sha'], 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 'host': os.uname().nodename, 'rewind': rw}
        (self.dir / RESUMES).mkdir(parents=True, exist_ok=True)
        _write_bytes_sync(self.dir / RESUMES / f'{n:03d}.json', json_bytes(entry))
        self.record.update(n=n, from_checkpoint=self.ckpt_dir.name, epochs_done=st['epochs_done'], rewind=rw)

    # ------------------------------------------------------------------------------------------ train begin

    def train_begin(self, hook):
        """The appended callback's on_train_begin: a fresh run writes its first checkpoint; a resumed run restores."""
        if self.resumed:
            self._restore()
        else:
            self._write(0, label='train_begin')

    def _restore(self):
        from cbl import rng

        ctx, st = self.ctx, self.state
        ck = self.ckpt_dir
        t0 = time.time()
        # the model, by role
        arrays = rng.read_tensor_file(ck / st['variables']['file'])
        if set(arrays) != set(ctx.roles):
            raise ResumeError('checkpoint variables and the model roles differ')
        for r, v in ctx.roles.items():
            _assign_checked(v, arrays[r], r)
        got = rng.variables_sha256(ctx.roles)
        if got != st['variables']['state_sha256']:
            raise ResumeError(f'after the restore the state hash is {got[:12]}, the checkpoint\'s {st["variables"]["state_sha256"][:12]}')
        # the optimizer, built by fit's symbolic build
        opt = ctx.model.optimizer
        if not opt.built:
            raise ResumeError('the optimizer is not built at train begin')
        oarr = rng.read_tensor_file(ck / st['optimizer']['file'])
        if [v.path for v in opt.variables] != st['optimizer']['paths'] or set(oarr) != {f'{i:04d}' for i in range(len(opt.variables))}:
            raise ResumeError('the optimizer variables differ from the checkpoint\'s by path or count')
        for i, v in enumerate(opt.variables):
            _assign_checked(v, oarr[f'{i:04d}'], f'optimizer variable {i} {v.path}')
        if optimizer_sha256(opt) != st['optimizer']['sha256']:
            raise ResumeError('after the restore the optimizer hash differs from the checkpoint\'s')
        # StopIf, ParetoFront (released[5], released[4]: run_train.py:95)
        restore_stopif(ctx.released[5], st['stopif'])
        restore_pareto(ctx.released[4], st['pareto'])
        if pareto_digest(ctx.released[4]) != st['pareto_sha256']:
            raise ResumeError('the restored ParetoFront record differs from the checkpoint\'s')
        # the evaluator's per-site signs
        if st['prev_signs'] is not None:
            ctx.evaluator.restore_sign_state(rng.read_tensor_file(ck / st['prev_signs']['file']))
        # the streams and the map: STUDY 6 step 8, after the pre-fit draws of this fit
        restore_streams(ctx, st)
        self.restored = {'epochs_done': st['epochs_done'], 'state_sha256': got, 'seconds': time.time() - t0,
                         'py_state_sha256': rng.py_state_sha256(), 'np_state_sha256': rng.np_state_sha256(),
                         'index_map_sha256': int_list_sha256(ctx.flow.ds_train._index_map)}
        self.record['restored'] = self.restored
        ctx.guard.arm(label='after the resume restore (STUDY section 6 step 8)')

    # ------------------------------------------------------------------------------------------ epoch end

    def after_epoch(self, epoch, logs, row):
        """The last step of the appended callback's on_epoch_end: a checkpoint when one is due."""
        done = int(epoch) + 1
        last = done >= int(self.ctx.cfg['epochs'])
        if done % self.every == 0 or last or bool(self.ctx.model.stop_training):
            self._write(done, label='epoch_end', row=row)

    # ------------------------------------------------------------------------------------------ writing

    def capture(self, epochs_done, *, row=None) -> tuple:
        """(state without file hashes, {name: arrays}, {rel: source path}) of the run now."""
        from cbl import frozen_eval, rng

        ctx = self.ctx
        t0 = time.time()
        state_sha = rng.variables_sha256(ctx.roles)
        if row is not None and row.get('vars_sha256_after') not in (None, state_sha):
            raise ResumeError(f'epoch {epochs_done - 1}: the state hash {state_sha[:12]} differs from the frozen '
                              f'evaluation\'s {row["vars_sha256_after"][:12]}')
        py = random.getstate()
        rec_check = None
        if epochs_done > 0:
            ep = ctx.flow.recorder.epoch(epochs_done - 1)
            rec_check = ep['state_sha256_after_end']
            if rec_check != rng.py_state_sha256(py):
                raise ResumeError(f'epoch {epochs_done - 1}: the module-level stream moved after the Dataset\'s epoch end')
        opt = ctx.model.optimizer
        oarr, opaths = optimizer_arrays(opt)
        varr = rng.variable_arrays(ctx.roles)
        signs = ctx.evaluator.sign_state()
        pf = ctx.released[4]
        imap = list(ctx.flow.ds_train._index_map)
        live_front = {}
        if ctx.front is not None:
            for e, s in sorted(ctx.front.saved.items()):
                if s['deleted_at'] is None:
                    live_front[s['path']] = {'sha256': s['sha256']}
        live_pareto = {os.path.relpath(str(p), str(self.run_dir)): {} for p in pf.paths}
        st = {
            'kind': KIND, 'schema': SCHEMA, 'label': 'run state; engineering when written by a gate',
            'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'epochs_done': int(epochs_done), 'next_epoch': int(epochs_done), 'every': self.every,
            'code_sha': ctx.cfg['code_sha'], 'config_sha256': ctx.record['config_sha256'],
            'run': {'run_name': ctx.wandb_run.name, 'run_id': ctx.wandb_run.run_id, 'mode': ctx.wandb_run.mode},
            'identity': json.loads(json.dumps(ctx.flow.run_identity(), sort_keys=True, default=_jsonable)),
            'variables': {'file': 'variables.npz', 'roles': sorted(varr), 'state_sha256': state_sha},
            'optimizer': {'file': 'optimizer.npz', 'paths': opaths, 'sha256': arrays_sha256(oarr, opaths),
                          'iterations': frozen_eval.step_of(ctx.model)},
            'random': {'state': py_state_to_json(py), 'sha256': rng.py_state_sha256(py)},
            'numpy': {'state': np_state_to_json(np.random.get_state(legacy=True)), 'sha256': rng.np_state_sha256()},
            'index_map': {'map': imap, 'sha256': int_list_sha256(imap)},
            'recorder_state_after_epoch': rec_check,
            'stopif': stopif_state(ctx.released[5]),
            'pareto': pareto_state(pf), 'pareto_sha256': pareto_digest(pf),
            'pid': ctx.pid.state() if (ctx.pid is not None and epochs_done > 0) else None,
            'front': ctx.front.snapshot() if ctx.front is not None else None,
            'prev_signs': None if not signs else {'file': 'prev_signs.npz', 'sites': sorted(signs)},
            'stop_training': bool(ctx.model.stop_training),
            'append_only': {'epochs.jsonl': prefix_record(self.run_dir / 'epochs.jsonl'),
                            'front/events.jsonl': prefix_record(self.run_dir / 'front' / 'events.jsonl')},
            'front_files': [f for f in files_under(self.run_dir / 'front', self.run_dir) if f != os.path.join('front', 'events.jsonl')],
            'pareto_files': files_under(self.run_dir / 'pareto', self.run_dir),
            'live': {'front': live_front, 'pareto': live_pareto},
            'epoch_record': None if row is None else {'epoch': row.get('epoch'), 'vars_sha256_after': row.get('vars_sha256_after'),
                                                      'step': row.get('step')},
            'capture_seconds': time.time() - t0,
        }
        arrays = {'variables.npz': varr, 'optimizer.npz': oarr}
        if signs:
            arrays['prev_signs.npz'] = signs
        sources = {rel: self.run_dir / rel for rel in list(live_front) + list(live_pareto)}
        return st, arrays, sources

    def _write(self, epochs_done, *, label, row=None):
        from cbl import rng

        t0 = time.time()
        st, arrays, sources = self.capture(epochs_done, row=row)
        st['written_at'] = label
        self.dir.mkdir(parents=True, exist_ok=True)
        name = checkpoint_name(epochs_done)
        tmp = self.dir / f'{TMP_PREFIX}{int(epochs_done):05d}'
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        files = {}
        for fname, arrs in arrays.items():
            files[fname] = rng.write_tensor_file(tmp / fname, arrs)
        links = {}
        for rel, src in sources.items():
            method = link_or_copy(src, tmp / 'links' / rel)
            links[rel] = {'sha256': sha256_file(tmp / 'links' / rel), 'method': method}
            want = st['live']['front'].get(rel, {}).get('sha256')
            if want is not None and links[rel]['sha256'] != want:
                raise ResumeError(f'live front checkpoint {rel} sha256 differs from the saver\'s record')
        st['files'], st['links'] = files, links
        raw = json_bytes(st, compact=True)
        _write_bytes_sync(tmp / 'state.json', raw)
        _fsync_dir(tmp)
        self._stage('tmp_written', epochs_done)
        final = self.dir / name
        if final.exists():
            # a checkpoint of the same epoch exists only after a resume that rewound past it; keep it aside
            n = resume_count(self.run_dir)
            move_aside(final, self.dir / DISCARDED / f'{n:03d}-superseded', name)
        os.replace(tmp, final)
        _fsync_dir(self.dir)
        self._stage('renamed', epochs_done)
        latest = {'checkpoint': name, 'epochs_done': int(epochs_done), 'state_sha256': sha256_bytes(raw),
                  'utc': st['created_utc']}
        lt = self.dir / (LATEST + '.tmp')
        if lt.exists():
            lt.unlink()
        _write_bytes_sync(lt, json_bytes(latest))
        os.replace(lt, self.dir / LATEST)
        _fsync_dir(self.dir)
        self._stage('latest_written', epochs_done)
        for old in sorted(d for d in self.dir.glob(CKPT_PREFIX + '*') if d.is_dir())[:-KEEP]:
            if old.name != name:
                shutil.rmtree(old)
        self.written.append({'epochs_done': int(epochs_done), 'label': label, 'state_sha256': st['variables']['state_sha256'],
                             'state_json_sha256': sha256_bytes(raw), 'seconds': time.time() - t0})

    def _stage(self, stage, epochs_done):
        if self.on_stage is not None:
            self.on_stage(stage, int(epochs_done))

    def summary(self) -> dict:
        out = dict(self.record)
        out['written'] = list(self.written)
        out['n_written'] = len(self.written)
        return out


def restore_streams(ctx, st):
    """The module-level stream, NumPy's legacy state and the training Dataset's _index_map, from the checkpoint; the
    map is assigned in place, so M4's recording list keeps recording."""
    from cbl import rng

    py = py_state_from_json(st['random']['state'])
    random.setstate(py)
    np.random.set_state(np_state_from_json(st['numpy']['state']))
    ds = ctx.flow.ds_train
    ds._index_map[:] = list(st['index_map']['map'])
    checks = {'random': rng.py_state_sha256() == st['random']['sha256'],
              'numpy': rng.np_state_sha256() == st['numpy']['sha256'],
              'index_map': int_list_sha256(ds._index_map) == st['index_map']['sha256']}
    if not all(checks.values()):
        raise ResumeError(f'the restored streams differ from the checkpoint: {checks}')


def _flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(_flatten(v, key + '.'))
        else:
            out[key] = v
    return out


# ----------------------------------------------------------------------------------------------
# rewinding the run directory (no Keras)


def rewind(run_dir, st, ckpt_dir, discard) -> dict:
    """Bring the run directory back to the checkpoint (module docstring, "What a resume changes on disk"). Every
    check runs before anything moves; returns what moved and what was linked back."""
    run_dir, ckpt_dir, discard = Path(run_dir), Path(ckpt_dir), Path(discard)
    rd = resume_dir(run_dir)
    # checks first
    prefixes = {rel: check_prefix(run_dir / rel, rec) for rel, rec in st['append_only'].items()}
    front_now = [f for f in files_under(run_dir / 'front', run_dir) if f != os.path.join('front', 'events.jsonl')]
    pareto_now = files_under(run_dir / 'pareto', run_dir)
    keep_front, keep_pareto = set(st['front_files']), set(st['pareto_files'])
    live = dict(st['live']['front'])
    live.update(st['live']['pareto'])
    missing = sorted((keep_front - set(front_now)) | (keep_pareto - set(pareto_now)))
    not_restorable = [m for m in missing if m not in live]
    if not_restorable:
        raise ResumeRefused(f'files recorded at the checkpoint are gone and the checkpoint holds no copy: {not_restorable[:5]}')
    for rel in live:
        p = run_dir / rel
        want = st['links'][rel]['sha256']
        if p.exists() and sha256_file(p) != want:
            raise ResumeRefused(f'{p} differs from the checkpoint\'s copy (files are written once)')
    out = {'append_only': {}, 'moved': [], 'linked_back': [], 'discard': str(discard.relative_to(run_dir))}
    # then the changes
    for rel, rec in st['append_only'].items():
        out['append_only'][rel] = dict(prefixes[rel], cut=prefixes[rel]['tail_bytes'] > 0)
        if prefixes[rel]['tail_bytes']:
            cut_to_prefix(run_dir / rel, rec, discard, rel)
    for rel in sorted(set(front_now) - keep_front) + sorted(set(pareto_now) - keep_pareto):
        move_aside(run_dir / rel, discard, rel)
        out['moved'].append(rel)
    for rel in missing:
        link_or_copy(ckpt_dir / 'links' / rel, run_dir / rel)
        out['linked_back'].append(rel)
    if rd.is_dir():
        name = ckpt_dir.name
        for d in sorted(rd.iterdir()):
            if d.name.startswith(TMP_PREFIX) or (d.name.startswith(CKPT_PREFIX) and d.is_dir() and d.name > name):
                move_aside(d, discard, os.path.join(DIR, d.name))
                out['moved'].append(os.path.join(DIR, d.name))
        lt = rd / (LATEST + '.tmp')
        if lt.exists():
            move_aside(lt, discard, os.path.join(DIR, lt.name))
            out['moved'].append(os.path.join(DIR, lt.name))
    return out


__all__ = ['Resume', 'ResumeError', 'ResumeRefused', 'read_latest', 'check_static', 'rewind', 'restore_streams',
           'stopif_state', 'restore_stopif', 'pareto_state', 'restore_pareto', 'pareto_digest', 'optimizer_arrays',
           'optimizer_sha256', 'arrays_sha256', 'py_state_to_json', 'py_state_from_json', 'np_state_to_json',
           'np_state_from_json', 'prefix_record', 'check_prefix', 'cut_to_prefix', 'files_under', 'link_or_copy',
           'KIND', 'SCHEMA', 'DIR', 'LATEST', 'KEEP', 'STAGES', 'checkpoint_name', 'resume_count']


def create_run_directory(out_dir, launch=None):
    """M13/M20 publish a fresh run and its pre-train launch identity together.

    A kill while preparing the temporary sibling leaves the target absent. A kill
    after rename leaves launch.json available even before backend initialization.
    Never replace an existing run. The pack holds the exclusive run lock.
    """
    import tempfile
    out = Path(out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    if launch is None:
        out.mkdir(exist_ok=False)
        return
    if out.exists():
        raise FileExistsError(out)
    temp = Path(tempfile.mkdtemp(prefix='.' + out.name + '.starting-', dir=out.parent))
    _write_bytes_sync(temp / 'launch.json', json_bytes(launch))
    _fsync_dir(temp)
    # Recheck under M20's advisory lock. Do not replace an already created directory.
    if out.exists():
        raise FileExistsError(out)
    temp.rename(out)
    _fsync_dir(out.parent)
