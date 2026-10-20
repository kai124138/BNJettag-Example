"""Shared parts of gates G13 and G14 (and the runner parts of G10 and G11): inputs, the placeholder
reference directory, audited children and record comparisons. Standard library only, plus gatelib,
rng_child and runner_child. Every reading is engineering, not quotable.
"""

import hashlib
import json
import os
from pathlib import Path

import fileaudit  # noqa: F401  (Linux; imported by runner_child.run_child when watches are given)
import rng_child
import runner_child

FORK = Path(__file__).resolve().parents[1]
BUNDLE = FORK.parent / 'preflight' / 'pc-setup' / 'code-main.bundle'
DETERMINISTIC_GPU_FLAG = '--xla_gpu_deterministic_ops=true'
PROBE_FIELDS = ('variables', 'variables_sha256', 'ebops', 'bn_moving', 'optimizer', 'optimizer_iterations', 'loss',
                'data_loss', 'n_model_losses', 'train_logits_sha256', 'grads', 'step_non_trainable', 'step_ebops',
                'infer_logits_sha256', 'infer_correct', 'infer_n', 'free_ebops', 'py_state_sha256')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def inputs(data_manifest, archive):
    inp = rng_child.smoke_inputs(data_manifest, archive)
    man = json.loads(Path(data_manifest).read_text(encoding='utf-8'))
    inp['heldout'] = str(Path(man['paths']['out_dir']) / '150c-test.h5')
    inp['heldout_raw_dir'] = man['paths']['val_dir']
    inp['bundle_sha256'] = sha256_file(BUNDLE)
    return inp


def placeholder_dir(work, archive):
    """`150c-train.h5` a link to the smoke archive and `150c-test.h5` a 16-jet file of zeros (G07's
    pattern): Chang's get_data reads the test file eagerly (data.py:13-15), and the held-out archive is
    never given to it."""
    fileaudit.require_unsynced(work, Path(work) / 'refdata')
    import h5py
    import numpy as np

    d = Path(work) / 'refdata'
    d.mkdir(parents=True)
    (d / '150c-train.h5').symlink_to(archive)
    with h5py.File(d / '150c-test.h5', 'w') as f:
        f.create_dataset('feature', data=np.zeros((16, 150, 16), dtype=np.float16))
        f.create_dataset('label', data=np.arange(16) % 5)
    return d


def watches(inp, ph_dir):
    held = Path(inp['heldout'])
    raw = Path(inp['heldout_raw_dir'])
    w = [(held, 'heldout'), (held.parent, 'data-dir'), (raw, 'heldout-raw-dir')]
    w += [(raw / n, 'heldout-raw') for n in sorted(os.listdir(raw))]
    w.append((Path(ph_dir) / '150c-test.h5', 'placeholder'))
    return w


def forbidden_events(audit):
    ev = audit['events'] if audit else []
    out = [e for e in ev if e['label'] in ('heldout', 'heldout-raw', 'heldout-raw-dir') or e.get('overflow')]
    out += [e for e in ev if e['label'] == 'data-dir' and e['name'] == '150c-test.h5']
    return out + list((audit or {}).get('events_before_start') or [])


def placeholder_opened(audit):
    return any(e['label'] == 'placeholder' for e in (audit or {}).get('events', []))


class Children:
    """Runs runner_child children of one gate with a fixed device, inputs and environment."""

    def __init__(self, g, *, device, inp, work, children_out, audit_watches=None, env_extra=None):
        self.g, self.device, self.inp, self.work, self.co = g, device, inp, Path(work), children_out
        self.watches, self.env_extra = audit_watches, dict(env_extra or {})

    def __call__(self, name, cmd, seed, extra=(), env_extra=None):
        env = dict(self.env_extra if env_extra is None else env_extra)
        saved = {k: os.environ.get(k) for k in ('XLA_FLAGS',)}
        try:
            if 'XLA_FLAGS' in env:
                os.environ['XLA_FLAGS'] = env['XLA_FLAGS']
            else:
                os.environ.pop('XLA_FLAGS', None)
            code, rec, argv, audit = runner_child.run_child(
                name, cmd, device=self.device, seed=seed, archive=self.inp['archive'], archive_sha=self.inp['archive_sha256'],
                data_manifest_sha=self.inp['data_manifest_sha256'], work=self.work / name, children_out=self.co,
                extra=extra, watches=self.watches)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.g.info(f'{name} argv', ' '.join(argv[3:]) + (f'  [XLA_FLAGS={env["XLA_FLAGS"]}]' if 'XLA_FLAGS' in env else ''))
        ok = code == 0 and rec is not None and 'error' not in rec
        if rec is not None and cmd == 'run':
            ok = ok and rec['runner'].get('end') == 'completed'
        detail = (rec or {}).get('error') or ((rec or {}).get('runner') or {}).get('error') or \
            ((rec or {}).get('runner') or {}).get('refused')
        self.g.check(ok, f'{name}: child exit 0' + (', run completed' if cmd == 'run' else '') + (f' ({detail})' if detail else ''))
        if self.watches is not None:
            fe = forbidden_events(audit)
            self.g.check(not fe, f'{name}: open-file audit, no open of the held-out archive or any raw held-out file '
                                 f'({len(fe)} forbidden events)')
        return rec if ok else None, audit


def draw_args(rec_draw, code_sha, bundle_sha, epochs=None, arm=None, wandb_mode=None):
    out = ['--manifest', rec_draw['manifest'], '--manifest-sha', rec_draw['manifest_sha256'], '--code-sha', code_sha,
           '--source-bundle', bundle_sha]
    if epochs is not None:
        out += ['--epochs', epochs]
    if arm is not None:
        out += ['--arm', arm]
    if wandb_mode is not None:
        out += ['--wandb-mode', wandb_mode]
    return out


def diff_snapshot(a, b, fields=PROBE_FIELDS):
    """Fields of two probe snapshots that differ, with a count for the per-role maps."""
    out = {}
    for f in fields:
        x, y = a.get(f), b.get(f)
        if x != y:
            if isinstance(x, dict) and isinstance(y, dict):
                keys = set(x) | set(y)
                out[f] = sum(1 for k in keys if x.get(k) != y.get(k))
            else:
                out[f] = 'differs'
    return out


def diff_rows(ra, rb, fields=PROBE_FIELDS):
    """Per epoch: the differing probe fields and the differing log keys."""
    if len(ra) != len(rb):
        return [{'epochs': f'{len(ra)} against {len(rb)} rows'}]
    out = []
    for x, y in zip(ra, rb):
        d = diff_snapshot(x, y, fields)
        lk = sorted(k for k in set(x['logs']) | set(y['logs']) if x['logs'].get(k) != y['logs'].get(k))
        if lk:
            d['logs'] = lk
        out.append(d)
    return out


def untimed(epoch_records):
    """The appended callback's compact per-epoch records without their wall-clock field."""
    return [{k: v for k, v in e.items() if k != 'sys/epoch_seconds'} for e in epoch_records or []]


def all_equal(diffs):
    return all(not d for d in diffs)


def first_diff(diffs):
    for e, d in enumerate(diffs):
        if d:
            return f'epoch {e}: {d}'
    return None
