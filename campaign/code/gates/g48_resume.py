"""Gate G48 (PREFLIGHT-PLAN section 3 row G48; STUDY v0.10 section 6 step 8, section 7.2 "On resume ...", section 12
(the W&B run id), the section 13 resume row and "equivalence between uninterrupted and resumed runs"; module M14
cbl/resume.py through M13 cbl/runner.py). Run with --device cpu (PCC) and --device gpu (PCG, the RTX 4060 Ti, every
child under XLA_FLAGS=--xla_gpu_deterministic_ops=true), in the venv of record, on the 2-file smoke archive, seed 1.

Uninterrupted and resumed runs are identical. Every run is a fresh process of gates/resume_child.py through the
runner, under the kernel open-file audit of the held-out paths, and every kill is a real SIGKILL of the training
process at a seam (no cleanup, no finally block runs). Arms: CH-8 (released builder, ramp, 6 epochs), F111-FB with
headline H1 (lab builder, binary sites, 6 epochs) and PID-CH with H1 (the lab PID, 14 epochs: warmup 10).

Kill-and-resume cases (checkpoint interval in brackets):
- CH-8: at an epoch boundary right after the checkpoint of 2 epochs [1]; mid-epoch, after the 3rd batch fetch of
  epoch 4, so that the resume rewinds epoch 3's records [3]; inside the checkpoint write of 4 epochs, after the
  temporary directory is complete and before the rename [1]; after the rename and before latest.json [1] (a newer
  checkpoint directory than latest.json must be set aside); twice in one run, at the boundary of 1 epoch and then
  mid-epoch 3 in the resumed process [1]; mid-epoch 0, resumed from the train-begin checkpoint [1]; at the boundary
  of 2 epochs with W&B offline, where the resumed process must open the same run id (a second offline directory);
  and the boundary case with a live front checkpoint file deleted after the checkpoint, as the killed process's
  saver would have deleted it, which the resume must link back from the checkpoint.
- F111-FB H1: at the boundary of 3 epochs [1]; mid-epoch 4 [3].
- PID-CH H1: at the boundary of 5 (warmup) and of 11 (acting) epochs [1]; mid-epoch 11 [4], resumed from 8.

Checks per case: the killed child ended by SIGKILL with latest.json naming the expected checkpoint; the resumed run
completed; and it is identical to the uninterrupted run of its arm in every per-epoch record (`epochs.jsonl` without
its two wall-time fields: variable hashes, the optimizer step, the fetched batch-index and _index_map hashes, the
frozen counts and top-1, the Keras logs as float.hex, the PID block), the appended callback's compact records, the
final model variables, optimizer variables and iteration, both global streams at the end of fit and the training
Dataset's map, StopIf's counters, ParetoFront's record, the PID state, the front saver's snapshot, sidecars, events
and run summary (file-byte hashes of .keras files left out, B8-2: one state saved twice hashes differently; the
state hash stays), every layer's configuration including beta0 (finding B7-F1 cannot arise: nothing is
deserialized), M4's per-epoch fetch records of the resumed epochs, and the W&B run id.

Constructed part: Chang's ParetoFront and StopIf (resume_child.py callbacks) on twelve constructed epochs that join
and leave the front and meet StopIf's condition, interrupted after epoch 7 with the process running on to epoch 10
(deleting a file live at the interruption): the resumed process's record, file names and counters equal the
uninterrupted sequence's. The smoke runs never pass Chang's filter (B9-F2), so their ParetoFront record is empty.

Fresh-run agreement (owner decision B10-14, finding B10-F1). Every chain also checks that the records the killed
process wrote before the resume point equal the uninterrupted run's: the two fresh processes reached the same state,
so the resumed continuation must equal the uninterrupted one. On the PC CPU a fresh F111-FB run was seen to differ from
another fresh run of the same config in rare cases (in two variables of the key projection, from epoch 0, with no
resume involved); a chain whose fresh processes disagree before the resume point says nothing about the resume, so it
is run once more in a new directory (at most two attempts), and every attempt and every disagreement is reported.

Controls (the gate can fail): a resume whose stream restore is disabled differs from the uninterrupted run; a resume
is refused (exit 2) with nothing in the run directory changed for another code sha in the checkpoint, another config,
another checkpoint interval, another source bundle (W&B run name), a finished run (run.json) and a checkpoint file
whose bytes changed. Every reading is engineering, not quotable.

    <venv>/bin/python -I -B gates/g48_resume.py --device cpu|gpu --expect-sha12 SHA12 --work DIR --archive SMOKE \\
        [--json PATH] [--children-out DIR]
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

import resumegate as RG  # noqa: E402

GATE = 'G48'


def copy_run(src, dst):
    shutil.copytree(src, dst, symlinks=True)
    return dst


def tree(kids, name, d):
    rc, rec = kids.raw_child(name, 'tree', ['--out-dir', d])
    return (rec or {}).get('tree')


def strip_attempts(t):
    return {k: v for k, v in (t or {}).items() if not k.startswith('resume/attempts/')}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    RG.standard_args(ap)
    a = ap.parse_args()
    G = RG.Gate(GATE, a)
    g = G.g
    if G.draw_s1() is None:
        return G.finish()
    K = G.kids
    results = {}

    def compare(label, U, R, resumed_from):
        if U is None or R is None:
            g.check(False, f'{label}: both records exist')
            return None
        d = RG.diff_runs(U, R, resumed_from=resumed_from)
        RG.checks_of_diff(g, label, d)
        results[label] = {'resumed_from': resumed_from, 'differs': sorted(d)}
        return d

    def expect_point(label, got, want):
        g.check(got == want, f'{label}: after the kill latest.json names the checkpoint of {want} epochs done (got {got})')

    attempts_log = {}

    def run_case(label, arm, U, *, epochs, every, kills, want, form=None, headline=None, wandb_mode=None):
        R, pts, d, log = RG.chain_checked(K, U, label, arm, epochs=epochs, every=every, kills=kills, form=form,
                                          headline=headline, wandb_mode=wandb_mode)
        attempts_log[label] = log
        if len(log) > 1:
            g.info(f'{label}: attempts (B10-14)', log)
        expect_point(label, pts, want)
        g.check(log[-1]['pre_resume_agree'], f'{label}: the records written before the resume point equal the '
                                            f'uninterrupted run\'s (fresh-run agreement; attempts {len(log)})')
        return R, pts, d

    # ---- CH-8
    U, u_dir = RG.uninterrupted(K, 'CH-8-U', 'CH-8', epochs=6)
    if U is not None:
        g.check(U['runner']['resume']['n_written'] == 7 and not U['runner']['resume']['resumed'],
                f'CH-8 uninterrupted: 7 checkpoints written (train begin and 6 epoch ends), no resume '
                f'({U["runner"]["resume"].get("n_written")})')
        lb = (U.get('final') or {}).get('layers') or {}
        g.info('CH-8 layer configurations', {'n_layers': lb.get('n_layers'), 'beta0 entries': len(lb.get('beta0') or []),
                                            'beta0 values': sorted({b for _, b in lb.get('beta0') or []})})
    cases = [('CH-8-boundary2', ['boundary:2'], 1, [2]), ('CH-8-mid4', ['mid:4:3'], 3, [3]),
             ('CH-8-tmp4', ['stage:tmp_written:4'], 1, [3]), ('CH-8-renamed3', ['stage:renamed:3'], 1, [2]),
             ('CH-8-twice', ['boundary:1', 'mid:3:2'], 1, [1, 3]), ('CH-8-mid0', ['mid:0:2'], 1, [0])]
    for label, kills, every, want in cases:
        R, pts, d = run_case(label, 'CH-8', U, epochs=6, every=every, kills=kills, want=want)
        compare(label, U, R, want[-1])
        if R is not None:
            rw = (R['runner'].get('resume') or {}).get('rewind') or {}
            if label == 'CH-8-mid4':
                g.check(rw.get('append_only', {}).get('epochs.jsonl', {}).get('cut') is True,
                        f'{label}: the resume cut epochs.jsonl back to the checkpoint (epoch 3\'s record of the killed '
                        f'process set aside)')
            if label == 'CH-8-tmp4':
                g.check('resume/.tmp-ckpt-00004' in rw.get('moved', []),
                        f'{label}: the incomplete checkpoint directory was set aside ({rw.get("moved")})')
            if label == 'CH-8-renamed3':
                g.check('resume/ckpt-00003' in rw.get('moved', []),
                        f'{label}: the checkpoint directory newer than latest.json was set aside ({rw.get("moved")})')
    # the W&B offline case: the resumed process opens the same run id
    R, pts, d = run_case('CH-8-offline2', 'CH-8', U, epochs=6, every=1, kills=['boundary:2'], want=[2], wandb_mode='offline')
    if R is not None and U is not None:
        cdiff = {k: (U['config'].get(k), R['config'].get(k)) for k in set(U['config']) | set(R['config'])
                 if U['config'].get(k) != R['config'].get(k)}
        g.check(list(cdiff) == ['wandb'] and {k for k in set(cdiff['wandb'][0]) | set(cdiff['wandb'][1])
                                              if cdiff['wandb'][0].get(k) != cdiff['wandb'][1].get(k)} == {'mode'},
                f'CH-8-offline2: its config differs from the uninterrupted run\'s in wandb.mode only, so the run summary\'s '
                f'config_sha256 differs by design and is left out of the comparison ({sorted(cdiff)})')
        import copy as _copy
        R2 = _copy.deepcopy(R)
        if (R2.get('front') or {}).get('summary') and (U.get('front') or {}).get('summary'):
            R2['front']['summary']['config_sha256'] = U['front']['summary']['config_sha256']
        compare('CH-8-offline2', U, R2, 2)
    if R is not None:
        rid = R['runner']['wandb']['run_id']
        offl = sorted(p.name for p in (d / 'wandb' / 'wandb').glob('offline-run-*')) if (d / 'wandb' / 'wandb').is_dir() else \
            sorted(p.name for p in (d / 'wandb').rglob('offline-run-*') if p.is_dir())
        g.check(len(offl) == 2 and all(n.endswith('-' + rid) for n in offl),
                f'CH-8-offline2: two offline W&B directories, both with run id {rid} (the killed and the resumed '
                f'process; offline runs ignore resume, M15) ({offl})')
    # a live front checkpoint deleted after the checkpoint: linked back
    d0 = K.work / 'runs' / 'CH-8-linkback'
    base = K.run_args('CH-8', epochs=6, every=1) + ['--out-dir', d0]
    K.child('CH-8-linkback-k0', 'run', base + ['--kill', 'boundary:2'], expect_kill=True)
    lt = RG.latest(d0)
    st = json.loads((d0 / 'resume' / lt['checkpoint'] / 'state.json').read_text()) if lt else {}
    live = sorted((st.get('live') or {}).get('front') or {})
    g.check(bool(live), f'CH-8-linkback: the checkpoint names live front checkpoint files ({live})')
    if live:
        (d0 / live[0]).unlink()
    R = K.child('CH-8-linkback-r', 'run', base + ['--resume'])
    if R is not None and live:
        lb = ((R['runner'].get('resume') or {}).get('rewind') or {}).get('linked_back')
        g.check(lb == [live[0]], f'CH-8-linkback: the deleted live file {live[0]} was linked back from the checkpoint ({lb})')
    compare('CH-8-linkback', U, R, 2)

    # ---- controls on copies of a run killed at the boundary of 2 epochs
    kd = K.work / 'runs' / 'ctl-killed'
    kb = K.run_args('CH-8', epochs=6, every=1) + ['--out-dir', kd]
    K.child('ctl-killed-k0', 'run', kb + ['--kill', 'boundary:2'], expect_kill=True)
    nd = copy_run(kd, K.work / 'runs' / 'ctl-norestore')
    Rn = K.child('ctl-norestore-r', 'run', K.run_args('CH-8', epochs=6, every=1) + ['--out-dir', nd, '--resume',
                                                                                    '--no-restore-streams'])
    if U is not None and Rn is not None:
        dn = RG.diff_runs(U, Rn, resumed_from=2)
        g.check('rows' in dn and 'recorder' in dn,
                f'control: with the stream restore disabled the resumed run differs from the uninterrupted one '
                f'({sorted(dn)[:6]}; the comparison can fail)')
        e2 = next((r for r in Rn['rows'] if r['epoch'] == 2), {})
        u2 = next((r for r in U['rows'] if r['epoch'] == 2), {})
        g.check(e2.get('batch_index_seq_sha256') != u2.get('batch_index_seq_sha256'),
                'control: its epoch 2 fetched another batch order (the repeated pre-fit draws of a fresh fit)')

    def refusal(label, prep, extra, needle):
        cd = copy_run(kd, K.work / 'runs' / f'ctl-{label}')
        prep(cd)
        t0 = tree(K, f'ctl-{label}-tree0', cd)
        args = K.run_args('CH-8', epochs=extra.get('epochs', 6), every=extra.get('every', 1),
                          source_bundle=extra.get('source_bundle')) + ['--out-dir', cd, '--resume']
        rc, rec = K.raw_child(f'ctl-{label}-r', 'run', args)
        msg = ((rec or {}).get('runner') or {}).get('refused') or ''
        t1 = tree(K, f'ctl-{label}-tree1', cd)
        g.check(rc == 2 and needle in msg and (rec or {}).get('runner', {}).get('end') == 'refused',
                f'control: a resume with {label} is refused (exit {rc}; "{msg[:160]}")')
        g.check(t0 is not None and strip_attempts(t0) == strip_attempts(t1),
                f'control: the refused resume ({label}) changed no file of the run directory '
                f'({len(t0 or {})} files before, {len(t1 or {})} after, the attempt record aside)')

    def other_code_sha(cd):
        lt_ = RG.latest(cd)
        sp = cd / 'resume' / lt_['checkpoint'] / 'state.json'
        stj = json.loads(sp.read_text())
        stj['code_sha'] = 'f' * 40
        raw = (json.dumps(stj, sort_keys=True, separators=(',', ':')) + '\n').encode()
        sp.write_bytes(raw)
        import hashlib
        lt_['state_sha256'] = hashlib.sha256(raw).hexdigest()
        (cd / 'resume' / 'latest.json').write_text(json.dumps(lt_, indent=1, sort_keys=True) + '\n')

    def changed_file(cd):
        lt_ = RG.latest(cd)
        p = cd / 'resume' / lt_['checkpoint'] / 'optimizer.npz'
        b = bytearray(p.read_bytes())
        b[-200] ^= 1
        p.write_bytes(bytes(b))

    refusal('another code sha in the checkpoint', other_code_sha, {}, 'code sha')
    refusal('another config (epochs 7)', lambda cd: None, {'epochs': 7}, 'config sha256')
    refusal('another checkpoint interval', lambda cd: None, {'every': 2}, 'interval')
    refusal('another source bundle', lambda cd: None, {'source_bundle': 'e' * 64}, 'run name')
    refusal('a finished run', lambda cd: (cd / 'run.json').write_text('{}\n'), {}, 'finished')
    refusal('a changed checkpoint file', changed_file, {}, 'sha256 differs')

    # ---- F111-FB H1
    Uf, _ = RG.uninterrupted(K, 'F111-FB-U', 'F111', epochs=6, form='FB', headline='H1')
    if Uf is not None:
        signs = (Uf.get('final') or {}).get('signs') or {}
        g.check(len(signs) > 0, f'F111-FB: the evaluator holds per-site signs ({len(signs)} sites), persisted by M14')
        rows = Uf.get('rows') or []
        g.check(rows and all(all(s.get('sign_flip_rate') is not None for s in (r.get('binary_sites') or {}).values())
                             for r in rows[1:]),
                'F111-FB: every record after epoch 0 carries a sign-flip rate at every binary site')
    for label, kills, every, want in (('F111-FB-boundary3', ['boundary:3'], 1, [3]), ('F111-FB-mid4', ['mid:4:3'], 3, [3])):
        R, pts, _ = run_case(label, 'F111', Uf, epochs=6, every=every, kills=kills, want=want, form='FB', headline='H1')
        compare(label, Uf, R, want[-1])

    # ---- PID-CH H1
    Up, _ = RG.uninterrupted(K, 'PID-CH-U', 'PID-CH', epochs=14, headline='H1')
    for label, kills, every, want in (('PID-CH-boundary5', ['boundary:5'], 1, [5]),
                                      ('PID-CH-boundary11', ['boundary:11'], 1, [11]),
                                      ('PID-CH-mid11', ['mid:11:3'], 4, [8])):
        R, pts, _ = run_case(label, 'PID-CH', Up, epochs=14, every=every, kills=kills, want=want, headline='H1')
        compare(label, Up, R, want[-1])
        if R is not None and Up is not None:
            g.check([r['pid'] for r in R['rows']] == [r['pid'] for r in Up['rows']]
                    and (R.get('final') or {}).get('pid') == (Up.get('final') or {}).get('pid'),
                    f'{label}: the PID block of every epoch (sensed H, beta, error, integral, clamp state, slack) and the '
                    f'final PID state equal the uninterrupted run\'s')

    # ---- constructed: ParetoFront and StopIf
    cbd = K.work / 'callbacks'
    full = K.child('cb-full', 'callbacks', ['--out-dir', cbd / 'full', '--phase', 'full'])
    st = K.work / 'callbacks-state.json'
    K.child('cb-first', 'callbacks', ['--out-dir', cbd / 'resumed', '--phase', 'first', '--split', 8, '--state', st])
    res = K.child('cb-resume', 'callbacks', ['--out-dir', cbd / 'resumed', '--phase', 'resume', '--split', 8, '--state', st])
    if full is not None and res is not None:
        g.info('constructed ParetoFront', {'final front files': full['pareto_files'], 'stopif': full['stopif'],
                                           'rewind': res.get('rewind')})
        rw = res.get('rewind') or {}
        g.check(bool(rw.get('linked_back')) and bool(rw.get('moved')),
                f'constructed: the rewind set aside the files the interrupted process wrote after the checkpoint '
                f'({rw.get("moved")}) and linked back the live file it deleted ({rw.get("linked_back")})')
        g.check(res['pareto'] == full['pareto'] and res['pareto_files'] == full['pareto_files'] and res['stopif'] == full['stopif']
                and res['per_epoch'] == full['per_epoch'][8:],
                'constructed: after the resume, ParetoFront\'s record (float.hex) and file names, StopIf\'s counters and '
                'stop flags of epochs 8-11 equal the uninterrupted sequence\'s')
        g.check(any(e['stopif']['counter_epochs'] for e in full['per_epoch'][:8]) and any(e['stop_training'] for e in full['per_epoch']),
                'constructed: the sequence meets StopIf\'s condition (counters and stop flag exercised)')
    n_dis = sum(1 for log in attempts_log.values() for x in log if not x['pre_resume_agree'])
    g.info('fresh-run disagreements before the resume point (B10-F1)', {'attempts': sum(len(v) for v in attempts_log.values()),
                                                                        'disagreements': n_dis})
    G.rec['results'] = results
    G.rec['attempts'] = attempts_log
    return G.finish()


if __name__ == '__main__':
    sys.exit(main())
