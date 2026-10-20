"""Shared parts of gates G46 (whole-run part), G48 and G50 (module M14 cbl/resume.py through M13 cbl/runner.py): the
prologue, the seed's draw, audited gates/resume_child.py children (a SIGKILLed child is expected where a kill seam is
set), the kill-and-resume chains, and the comparison of an uninterrupted run with a resumed one. Standard library
only, plus gatelib, runnergate and fileaudit. Every reading is engineering, not quotable.

The runs are seed 1, stage `gate`, on the 2-file smoke archive of B2. On the GPU (PCG) every child runs with
XLA_FLAGS=--xla_gpu_deterministic_ops=true (owner decision B5-1).
"""

import json
import os
import shutil
import sys
from pathlib import Path

import fileaudit  # noqa: F401  (Linux)
import gatelib
import runnergate as rg

FORK = Path(__file__).resolve().parents[1]
SEED = 1
SIGKILL_RC = -9


class Gate:
    """The common prologue of the resume gates and their children (resume_child.py under the open-file audit)."""

    def __init__(self, name, a):
        self.g = gatelib.Gate(name)
        self.a = a
        self.device = a.device
        self.rec = {'header': self.g.header(a.expect_sha12), 'seed': SEED, 'device': a.device, 'children': []}
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
        self.env = {'XLA_FLAGS': rg.DETERMINISTIC_GPU_FLAG} if a.device == 'gpu' else {}
        if a.device == 'gpu':
            self.g.info('XLA_FLAGS of every child', rg.DETERMINISTIC_GPU_FLAG)
        self.runner_kids = rg.Children(self.g, device=a.device, inp=self.inp, work=self.work, children_out=a.children_out,
                                       audit_watches=self.watches, env_extra=self.env)
        self.draw = None
        self.kids = None

    def draw_s1(self):
        d, _ = self.runner_kids('draw-s1', 'draw', SEED, ['--out-dir', self.work / 'draw-s1' / 'init', '--place',
                                                          f'{self.a.device.upper()} gate {self.g.name}'])
        self.draw = d
        self.kids = Kids(self.g, self.a, work=self.work, watches=self.watches, inp=self.inp, draw=d, head=self.head,
                         device=self.device, env=self.env, rec=self.rec)
        return d

    def finish(self):
        self.g.check_clean('at end')
        return self.g.finish(self.rec, self.a.json)


class Kids:
    """resume_child.py children of one gate (G46's whole-run part builds one on pidgate's prologue)."""

    def __init__(self, g, a, *, work, watches, inp, draw, head, device, env, rec):
        self.g, self.a, self.work, self.watches, self.inp = g, a, Path(work), watches, inp
        self.draw, self.head, self.device, self.env, self.rec = draw, head, device, dict(env), rec
        self.rec.setdefault('children', [])

    def child(self, name, cmd, extra=(), *, expect_kill=False):
        """One resume_child.py child in a fresh process under the open-file audit. With expect_kill the child must end
        by SIGKILL (its record is then None)."""
        w = self.work / 'children' / name
        w.mkdir(parents=True, exist_ok=True)
        out = w / f'{name}.json'
        argv = [sys.executable, '-I', '-B', str(FORK / 'gates' / 'resume_child.py'), cmd, '--out', str(out)]
        argv += [str(x) for x in extra]
        env = {k: v for k, v in os.environ.items() if k not in ('KERAS_BACKEND', 'JAX_PLATFORMS',
                                                               'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_FLAGS')}
        env.update(self.env)
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
        self.g.info(f'{name} argv', ' '.join(argv[3:]))
        self.rec['children'].append({'name': name, 'returncode': r['returncode']})
        if expect_kill:
            ok = r['returncode'] == SIGKILL_RC and rec is None
            self.g.check(ok, f'{name}: the child was killed by SIGKILL at its seam (return code {r["returncode"]})')
        else:
            ok = r['returncode'] == 0 and rec is not None and 'error' not in rec
            detail = (rec or {}).get('error') or ((rec or {}).get('runner') or {}).get('error') or \
                ((rec or {}).get('runner') or {}).get('refused')
            if cmd == 'run' and rec is not None:
                ok = ok and rec['runner'].get('end') == 'completed' and rec['runner'].get('exit') == 0
            self.g.check(ok, f'{name}: child exit 0' + (', run completed with exit 0' if cmd == 'run' else '')
                         + (f' ({str(detail)[:300]})' if detail else ''))
        fe = rg.forbidden_events(audit)
        self.g.check(not fe, f'{name}: open-file audit, no open of the held-out archive or any raw held-out file '
                             f'({len(fe)} forbidden events)')
        return rec if ok else None

    def raw_child(self, name, cmd, extra=()):
        """A child whose outcome the caller judges (refusal controls): (return code, record)."""
        w = self.work / 'children' / name
        w.mkdir(parents=True, exist_ok=True)
        out = w / f'{name}.json'
        argv = [sys.executable, '-I', '-B', str(FORK / 'gates' / 'resume_child.py'), cmd, '--out', str(out)]
        argv += [str(x) for x in extra]
        env = {k: v for k, v in os.environ.items() if k not in ('KERAS_BACKEND', 'JAX_PLATFORMS',
                                                               'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_FLAGS')}
        env.update(self.env)
        r = fileaudit.run_audited(argv, watches=self.watches, env=env, cwd=str(w), timeout=3600)
        (w / f'{name}.stdout.txt').write_text(r['stdout'])
        (w / f'{name}.stderr.txt').write_text(r['stderr'])
        rec = json.loads(out.read_text()) if out.exists() else None
        if self.a.children_out:
            co = Path(self.a.children_out)
            co.mkdir(parents=True, exist_ok=True)
            if out.exists():
                shutil.copy2(out, co / out.name)
            (co / f'{name}.argv.txt').write_text(' '.join(argv) + '\n')
        self.g.info(f'{name} argv', ' '.join(argv[3:]))
        fe = rg.forbidden_events({k: r[k] for k in ('events', 'events_before_start', 'fd_paths')})
        self.g.check(not fe, f'{name}: open-file audit, no forbidden event ({len(fe)})')
        return r['returncode'], rec

    def run_args(self, arm, *, epochs, every, form=None, headline=None, source_bundle=None):
        d = self.draw
        out = ['--device', self.device, '--seed', SEED, '--archive', self.inp['archive'], '--archive-kind', 'smoke',
               '--archive-sha', self.inp['archive_sha256'], '--data-manifest-sha', self.inp['data_manifest_sha256'],
               '--manifest', d['manifest'], '--manifest-sha', d['manifest_sha256'], '--code-sha', self.head,
               '--source-bundle', source_bundle or self.inp['bundle_sha256'], '--arm', arm, '--epochs', epochs,
               '--every', every]
        if form:
            out += ['--form', form]
        if headline:
            out += ['--headline', headline]
        return out


def standard_args(ap, devices=('cpu', 'gpu')):
    ap.add_argument('--device', choices=list(devices), default='cpu')
    ap.add_argument('--expect-sha12', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--data-manifest', default=str(FORK / 'manifests' / 'data-prep-150c.json'))
    ap.add_argument('--json', default=None)
    ap.add_argument('--children-out', default=None)


def latest(run_dir):
    p = Path(run_dir) / 'resume' / 'latest.json'
    return json.loads(p.read_text()) if p.exists() else None


def n_rows(run_dir):
    p = Path(run_dir) / 'epochs.jsonl'
    return len([x for x in p.read_text().splitlines() if x]) if p.exists() else 0


# ----------------------------------------------------------------------------------------------
# comparing an uninterrupted run with a resumed one


def _recorder_epochs(R):
    rec = (R.get('runner') or {}).get('recorder') or {}
    return {e['keras_epoch']: e for e in rec.get('epochs') or []}


def _scientific_rows(rows):
    """Exclude process telemetry even from records loaded outside the child writer."""
    if rows is None:
        return None
    return [{k: v for k, v in row.items() if not k.startswith('sys/')} for row in rows]


def diff_runs(U, R, *, resumed_from):
    """{aspect: description} of everything that differs between the uninterrupted record U and the resumed record R
    (resume_child.py run records); empty when they are identical. Wall times and .keras file-byte hashes are left
    out by the child (B8-2); process telemetry is also excluded here for loaded records.
    `resumed_from` is the first epoch the resumed process trained."""
    out = {}
    ru, rr = _scientific_rows(U.get('rows') or []), _scientific_rows(R.get('rows') or [])
    if len(ru) != len(rr):
        out['rows'] = f'{len(ru)} against {len(rr)} epoch records'
    else:
        bad = []
        for a, b in zip(ru, rr):
            d = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            if d:
                bad.append((a.get('epoch'), d[:6]))
        if bad:
            out['rows'] = f'{len(bad)} epochs differ, first {bad[0]}'
    if _scientific_rows(U.get('epoch_records')) != _scientific_rows(R.get('epoch_records')):
        out['epoch_records'] = 'the appended callback\'s compact records differ'
    fu, fr = U.get('final') or {}, R.get('final') or {}
    for k in sorted(set(fu) | set(fr)):
        if fu.get(k) != fr.get(k):
            out[f'final.{k}'] = 'differs'
    for k in ('events', 'sidecars', 'summary', 'files'):
        if (U.get('front') or {}).get(k) != (R.get('front') or {}).get(k):
            out[f'front.{k}'] = 'differs'
    if U.get('pareto_files') != R.get('pareto_files'):
        out['pareto_files'] = 'differs'
    uu, rr_ = U['runner'], R['runner']
    if uu.get('final_variables_sha256') != rr_.get('final_variables_sha256'):
        out['final_variables_sha256'] = 'differs'
    if (uu.get('wandb') or {}).get('run_id') != (rr_.get('wandb') or {}).get('run_id'):
        out['wandb.run_id'] = 'differs'
    gu, gr = (uu.get('guard') or {}).get('disarmed_at') or {}, (rr_.get('guard') or {}).get('disarmed_at') or {}
    for k in ('py_state_sha256', 'np_state_sha256'):
        if gu.get(k) != gr.get(k):
            out[f'stream at the end of fit ({k})'] = 'differs'
    eu, er = _recorder_epochs(U), _recorder_epochs(R)
    keys = ('raw', 'mapped', 'index_map', 'index_map_after_end', 'batch_index_seq_sha256', 'index_map_sha256',
            'state_sha256_at_begin', 'state_sha256_before_end', 'state_sha256_after_end', 'mapped_is_permutation',
            'mapped_equals_map_of_raw', 'index_map_unchanged_during_epoch', 'n_fetches')
    bad = [e for e in sorted(er) if e >= resumed_from and {k: eu.get(e, {}).get(k) for k in keys} != {k: er[e].get(k) for k in keys}]
    if bad or sorted(e for e in er) != [e for e in sorted(eu) if e >= resumed_from]:
        out['recorder'] = f'per-epoch fetch records differ from epoch {resumed_from} on: {bad[:5]}'
    return out


def checks_of_diff(g, label, d):
    g.check(not d, f'{label}: identical to the uninterrupted run in every epoch record (variables, optimizer step, '
                   f'batch order, frozen counts, PID block), the final variables, optimizer, both streams, the map, '
                   f'StopIf, ParetoFront, PID, front saver, sidecars, events, summary, layer configurations, the fetch '
                   f'records and the W&B run id' + (f' (differs: {sorted(d)[:8]})' if d else ''))


def chain(kids, label, arm, *, epochs, every, kills, form=None, headline=None, wandb_mode=None):
    """A fresh run killed at kills[0], resumed and killed at kills[1], ..., then resumed to its end. Returns (final
    record, [epochs done named in latest.json after each kill], run directory)."""
    out_dir = kids.work / 'runs' / label
    base = kids.run_args(arm, epochs=epochs, every=every, form=form, headline=headline) + ['--out-dir', out_dir]
    if wandb_mode:
        base += ['--wandb-mode', wandb_mode]
    points = []
    for i, k in enumerate(kills):
        kids.child(f'{label}-k{i}', 'run', base + (['--resume'] if i else []) + ['--kill', k], expect_kill=True)
        lt = latest(out_dir)
        points.append(None if lt is None else lt['epochs_done'])
    R = kids.child(f'{label}-r', 'run', base + ['--resume'])
    return R, points, out_dir


def uninterrupted(kids, label, arm, *, epochs, every=1, form=None, headline=None):
    out_dir = kids.work / 'runs' / label
    return kids.child(label, 'run', kids.run_args(arm, epochs=epochs, every=every, form=form, headline=headline)
                      + ['--out-dir', out_dir]), out_dir


def pre_resume_agree(U, R, point):
    """Whether the resumed run's records before the last resume point (written by the earlier, killed process or
    processes) equal the uninterrupted run's: the two fresh processes agreed up to the state the resume restored."""
    if U is None or R is None or point is None:
        return False
    a = _scientific_rows((U.get('rows') or [])[:point])
    b = _scientific_rows((R.get('rows') or [])[:point])
    return len(a) == point and a == b


def chain_checked(kids, U, label, arm, *, epochs, every, kills, form=None, headline=None, wandb_mode=None, attempts=2):
    """chain() with the B10-14 rule: if the fresh processes disagree with the uninterrupted run before the resume point
    (run-to-run variation of the fresh runs, finding B10-F1, which says nothing about the resume), the case is run
    again in a new directory, at most `attempts` times. Returns (record, points, run dir, attempt log); every attempt
    is recorded and reported."""
    log = []
    R = pts = d = None
    for i in range(1, attempts + 1):
        lab = label if i == 1 else f'{label}-a{i}'
        R, pts, d = chain(kids, lab, arm, epochs=epochs, every=every, kills=kills, form=form, headline=headline,
                          wandb_mode=wandb_mode)
        point = pts[-1] if pts else None
        agree = pre_resume_agree(U, R, point)
        log.append({'attempt': i, 'label': lab, 'points': pts, 'pre_resume_agree': agree})
        if agree or R is None:
            break
    return R, pts, d, log
