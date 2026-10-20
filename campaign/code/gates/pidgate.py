"""Shared parts of gates G43-G47 (module M12 cbl/pid.py through M13 cbl/runner.py): the prologue, the draw, audited
gates/pid_child.py children and the per-epoch views the gates compare. Standard library only, plus gatelib,
runnergate and fileaudit. Every reading is engineering, not quotable.

The PID runs are seed 1, stage `gate`, on the 2-file smoke archive of B2, for EPOCHS = 14 epochs: warmup is 10
(STUDY 7.2), so epochs 0-9 hold lambda_0, epoch 10 starts the integral (beta_pid.py:140-148) and epochs 10-13 run the
law on the frozen readings of epochs 9-12. Three arms: PID-CH with headline H1 (the released builder), PID-BEST as
F111-FB with H1 and as F111-BF with H2 (the lab builder, both fused-BN forms and both headline branches), as G34 did
for the ramp.
"""

import json
import math
import os
import sys
from pathlib import Path

import fileaudit  # noqa: F401  (Linux)
import gatelib
import runnergate as rg

FORK = Path(__file__).resolve().parents[1]
SEED = 1
EPOCHS = 14
WARMUP = 10
TARGET_H = 350000
INIT_BETA = 2e-08
ARMS = (('PID-CH', None, 'H1'), ('PID-BEST', 'FB', 'H1'), ('PID-BEST', 'BF', 'H2'))


def label(arm, form, headline):
    return f'{arm}-F111-{form}-{headline}' if form else f'{arm}-{headline}'


class Gate:
    """The common prologue of G43-G47 and their children (pid_child.py under the open-file audit)."""

    def __init__(self, name, a, *, device='cpu'):
        self.g = gatelib.Gate(name)
        self.a = a
        self.rec = {'header': self.g.header(a.expect_sha12), 'epochs': EPOCHS, 'seed': SEED, 'device': device}
        gatelib.check_venv_of_record(self.g)
        print('label       engineering, not quotable')
        self.work = Path(a.work).resolve()
        fileaudit.require_unsynced(self.work)
        self.g.check(not self.work.exists(), f'work directory {self.work} does not exist yet')
        self.work.mkdir(parents=True)
        self.head = self.rec['header']['HEAD']
        self.inp = rg.inputs(a.data_manifest, a.archive)
        got = rg.sha256_file(a.archive)
        self.g.check(got == self.inp['archive_sha256'] and got not in (self.inp['record_archive_sha256'],
                                                                      self.inp['heldout_archive_sha256']),
                     f'the archive is the smoke archive {got[:12]} (not the archive of record, not the held-out archive)')
        ph = self.work / 'audit-placeholder'
        ph.mkdir()
        (ph / '150c-test.h5').write_bytes(b'')
        self.watches = rg.watches(self.inp, ph)
        self.device = device
        self.runner_kids = rg.Children(self.g, device=device, inp=self.inp, work=self.work, children_out=a.children_out,
                                       audit_watches=self.watches)
        self.draw = None

    def draw_s1(self):
        d, _ = self.runner_kids('draw-s1', 'draw', SEED, ['--out-dir', self.work / 'draw-s1' / 'init', '--place',
                                                          f'PCC gate {self.g.name}'])
        self.draw = d
        return d

    def child(self, name, cmd, extra=()):
        """One pid_child.py child in a fresh process of this interpreter under the open-file audit."""
        import shutil

        w = self.work / name
        w.mkdir(parents=True, exist_ok=True)
        out = w / f'{name}.json'
        argv = [sys.executable, '-I', '-B', str(FORK / 'gates' / 'pid_child.py'), cmd, '--out', str(out)]
        argv += [str(x) for x in extra]
        env = {k: v for k, v in os.environ.items() if k not in ('KERAS_BACKEND', 'JAX_PLATFORMS',
                                                               'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_FLAGS')}
        r = fileaudit.run_audited(argv, watches=self.watches, env=env, cwd=str(w), timeout=3600)
        (w / f'{name}.stdout.txt').write_text(r['stdout'])
        (w / f'{name}.stderr.txt').write_text(r['stderr'])
        audit = {k: r[k] for k in ('events', 'events_before_start', 'fd_paths')}
        rec = json.loads(out.read_text()) if out.exists() else None
        if self.a.children_out:
            co = Path(self.a.children_out)
            co.mkdir(parents=True, exist_ok=True)
            for suffix in ('.json', '.stdout.txt', '.stderr.txt'):
                src = w / f'{name}{suffix}'
                if src.exists():
                    shutil.copy2(src, co / src.name)
            (co / f'{name}.argv.txt').write_text(' '.join(argv) + '\n')
            (co / f'{name}.audit.json').write_text(json.dumps(audit, indent=1, sort_keys=True) + '\n')
        ok = r['returncode'] == 0 and rec is not None and 'error' not in rec
        if rec is not None and cmd == 'run':
            ok = ok and (rec.get('runner') or {}).get('end') == 'completed' and (rec.get('runner') or {}).get('exit') == 0
        detail = (rec or {}).get('error') or ((rec or {}).get('runner') or {}).get('error') or \
            ((rec or {}).get('runner') or {}).get('refused') or ((rec or {}).get('runner') or {}).get('front_error')
        self.g.info(f'{name} argv', ' '.join(argv[3:]))
        self.g.check(ok, f'{name}: child exit 0' + (', run completed with exit 0' if cmd == 'run' else '')
                     + (f' ({str(detail)[:300]})' if detail else ''))
        fe = rg.forbidden_events(audit)
        self.g.check(not fe, f'{name}: open-file audit, no open of the held-out archive or any raw held-out file '
                             f'({len(fe)} forbidden events)')
        return rec if ok else None

    def run_args(self, arm, form, headline, epochs=EPOCHS):
        d = self.draw
        out = ['--device', self.device, '--seed', SEED, '--archive', self.inp['archive'], '--archive-sha',
               self.inp['archive_sha256'], '--data-manifest-sha', self.inp['data_manifest_sha256'], '--manifest',
               d['manifest'], '--manifest-sha', d['manifest_sha256'], '--code-sha', self.head, '--source-bundle',
               self.inp['bundle_sha256'], '--arm', arm, '--headline', headline, '--epochs', epochs]
        if form:
            out += ['--form', form]
        return out

    def run(self, arm, form, headline, epochs=EPOCHS):
        lab = label(arm, form, headline)
        return lab, self.child(f'run-{lab}', 'run', self.run_args(arm, form, headline, epochs)
                               + ['--out-dir', self.work / f'run-{lab}' / 'out'])

    def finish(self):
        self.g.check_clean('at end')
        return self.g.finish(self.rec, self.a.json)


def standard_args(ap):
    ap.add_argument('--device', choices=['cpu'], default='cpu')
    ap.add_argument('--expect-sha12', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--data-manifest', default=str(FORK / 'manifests' / 'data-prep-150c.json'))
    ap.add_argument('--json', default=None)
    ap.add_argument('--children-out', default=None)


# ----------------------------------------------------------------------------------------------
# views of a run child's record


def blocks(R):
    """{epoch: the record's pid block} of a run child."""
    return {r['epoch']: r['pid'] for r in R['frozen_rows']}


def rows(R):
    return {r['epoch']: r for r in R['frozen_rows']}


def law(sensed, *, target=TARGET_H, init_beta=INIT_BETA, p=1.0, i=0.002, d=0.0, warmup=WARMUP, epochs=EPOCHS):
    """The stock BetaPID law (beta_pid.py:8-32, :131-160 with log true, damping 0, clamp [0, inf)) restated here
    independently, on the frozen readings: sensed[e] is read at on_epoch_end(e) and acted on at on_epoch_begin(e+1).
    Returns per epoch the beta in force, the error and the integral after the epoch's begin (None in warmup)."""
    beta, integral, prev = init_beta, 0.0, 0.0
    out = {}
    for e in range(epochs):
        if e < warmup:
            out[e] = {'beta': beta, 'error': None, 'integral': integral, 'acted_on_H': None}
            continue
        ebops = sensed[e - 1]
        if e == warmup:
            err0 = math.log10(ebops / target + 1e-9)
            integral = (math.log10(beta) - p * err0) / i - err0
        err = math.log10(ebops / target) - 0
        integral += err
        der = err - prev
        prev = err
        beta = 10.0 ** (p * err + i * integral + d * der)
        beta = max(min(beta, math.inf), 0.0)
        out[e] = {'beta': beta, 'error': err, 'integral': integral, 'acted_on_H': ebops}
    return out


def hexes(x):
    return None if x is None else float(x).hex()
