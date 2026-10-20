"""Gate G34 (PREFLIGHT-PLAN section 3 row G34; STUDY v0.10 section 3.3: the frozen evaluation mutates nothing,
makes no layer call in training or tracing mode, and the checkpoint audit must equal it exactly; module M8
cbl/frozen_eval.py through M13 cbl/runner.py). PC CPU and PC GPU (RTX 4060 Ti), venv of record, 2-file smoke
archive of B2.

Children, each a fresh process of the venv of record under the open-file audit (no open of the held-out
archive or any raw held-out file); on the GPU every child runs with XLA_FLAGS=--xla_gpu_deterministic_ops=true
(decision B5-1):
  draw     CH-8's initial-tensor file for seed 1 on this device (cbl.runner.draw, decision U1);
  run-*    the runner for CH-8, F111-FB (headline H1) and F111-BF (headline H2), seed 1, stage `gate`,
           --epochs epochs, W&B disabled, the file loaded by role; M8's evaluation in the appended callback
           with a spy that records the `training` argument of every hgq Quantizer call made while it
           evaluates, and a probe after it that saves a .keras checkpoint at every epoch end
           (gates/runner_child.py run --no-probe --spy-frozen --save-ckpt);
  audit-*  cbl.frozen_eval.checkpoint_audit of every saved checkpoint in a fresh process, against the frozen
           record of its epoch (gates/cost_child.py audit), and one control per arm with one quantizer bit of
           the reloaded model moved (--mutate).

Checks, per run: the run completes; every epoch has one record with the interfaces.md section 1 keys B7 fills;
the variable hash before and after each evaluation is equal; snap_changed is 0; H2 <= H1 <= NR; val_n is the
validation size; the stream guard saw no change of the global streams, and its 'frozen evaluation' block ran
once per epoch with no change; the spy saw M8 evaluate once per epoch and every quantizer call during the
evaluations had training None or False (no training- or tracing-mode call); one checkpoint per epoch. Per
checkpoint: the audit equals the frozen record exactly (val_correct, val_n, val_top1, H1, H2, NR, snap_changed,
every per-layer and per-group subtotal, the binary-site record without the sign-flip rate, the variable hash).
Controls: the mutated reload differs. Recorded beside (not a pass condition): the frozen validation top-1
against Keras's logged val_accuracy, and the evaluation time per epoch. Every reading is engineering, not
quotable.

    <venv>/bin/python -I -B gates/g34_frozen_eval.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
        --archive SMOKE_ARCHIVE [--epochs 3] [--data-manifest PATH] [--json PATH] [--children-out DIR]
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import costgate  # noqa: E402
import gatelib  # noqa: E402
import runnergate as rg  # noqa: E402

GATE = 'G34'
SEED = 1
ARMS = (('CH-8', None, None), ('F111', 'FB', 'H1'), ('F111', 'BF', 'H2'))
ROW_KEYS = ('schema', 'run_id', 'epoch', 'step', 'batch_index_seq_sha256', 'index_map_sha256', 'val_correct', 'val_n',
            'val_top1', 'H1', 'H2', 'NR', 'subtotals', 'train_count', 'log_loss', 'log_accuracy', 'log_val_loss',
            'log_val_accuracy', 'lr', 'beta', 'vars_sha256_before', 'vars_sha256_after', 'snap_changed', 'binary_sites',
            'quantizer_structure', 'pruning_channel', 'pid')
ALLOWED_TRAINING = {'None', 'False'}


def label_of(arm, form):
    return arm if form is None else f'{arm}-{form}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--device', choices=['cpu', 'gpu'], required=True)
    ap.add_argument('--expect-sha12', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--data-manifest', default=str(FORK / 'manifests' / 'data-prep-150c.json'))
    ap.add_argument('--json', default=None)
    ap.add_argument('--children-out', default=None)
    a = ap.parse_args()
    g = gatelib.Gate(GATE)
    rec = {'header': g.header(a.expect_sha12), 'epochs': a.epochs, 'device': a.device}
    gatelib.check_venv_of_record(g)
    print('label       engineering, not quotable')
    work = Path(a.work).resolve()
    rg.fileaudit.require_unsynced(work)
    g.check(not work.exists(), f'work directory {work} does not exist yet')
    work.mkdir(parents=True)
    head = rec['header']['HEAD']
    inp = rg.inputs(a.data_manifest, a.archive)
    got = rg.sha256_file(a.archive)
    g.check(got == inp['archive_sha256'] and got not in (inp['record_archive_sha256'], inp['heldout_archive_sha256']),
            f'the archive is the smoke archive {got[:12]} (not the archive of record, not the held-out archive)')
    ph = work / 'audit-placeholder'
    ph.mkdir()
    (ph / '150c-test.h5').write_bytes(b'')
    watches = rg.watches(inp, ph)
    env_extra = {'XLA_FLAGS': rg.DETERMINISTIC_GPU_FLAG} if a.device == 'gpu' else {}
    ch = rg.Children(g, device=a.device, inp=inp, work=work, children_out=a.children_out, audit_watches=watches,
                     env_extra=env_extra)
    d, _ = ch('draw-s1', 'draw', SEED, ['--out-dir', work / 'draw-s1' / 'init', '--place', f'{a.device.upper()} gate G34'])
    if d is None:
        g.check_clean('at end')
        return g.finish(rec, a.json)
    rec['runs'] = {}
    for arm, form, headline in ARMS:
        lab = label_of(arm, form)
        extra = rg.draw_args(d, head, inp['bundle_sha256'], epochs=a.epochs, arm=arm) + \
            ['--no-probe', '--spy-frozen', '--save-ckpt', work / f'ckpt-{lab}']
        if form:
            extra += ['--form', form]
        if headline:
            extra += ['--headline', headline]
        R, _ = ch(f'run-{lab}', 'run', SEED, extra + ['--out-dir', work / f'run-{lab}' / 'out'])
        if R is None:
            continue
        rr = R['runner']
        rows = R.get('frozen_rows') or []
        n_val = R['frozen_evaluation']['n_val'] if R.get('frozen_evaluation') else None
        g.check(len(rows) == a.epochs and all(set(ROW_KEYS) <= set(x) for x in rows),
                f'{lab}: {len(rows)} frozen records, one per epoch, each with the interfaces.md section 1 keys B7 fills')
        g.check(all(x['vars_sha256_before'] == x['vars_sha256_after'] for x in rows),
                f'{lab}: the variable hash is unchanged by every frozen evaluation')
        g.check(all(x['snap_changed'] == 0 and x['H2'] <= x['H1'] <= x['NR'] and x['val_n'] == n_val for x in rows),
                f'{lab}: snap_changed 0, H2 <= H1 <= NR and val_n = {n_val} in every record '
                f'({[(x["H1"], x["H2"], x["NR"]) for x in rows]})')
        gd = rr['guard']
        w = gd['calls'].get('frozen evaluation.watch', {})
        g.check(gd['lab_violations'] == [] and gd['released_violations'] == [] and w.get('calls') == a.epochs
                and w.get('py_changed') == 0 and w.get('np_changed') == 0,
                f'{lab}: the stream guard saw no change of the global streams; its frozen-evaluation block ran {w.get("calls")} '
                f'times with no change')
        sp = R['frozen_spy'] or {}
        vals = set(sp.get('training_values', {}))
        g.check(sp.get('evaluations') == a.epochs and sp.get('quantizer_calls', 0) > 0 and vals <= ALLOWED_TRAINING,
                f'{lab}: M8 evaluated {sp.get("evaluations")} times; the {sp.get("quantizer_calls")} hgq Quantizer calls made '
                f'during the evaluations had training {sorted(vals)} only (no training- or tracing-mode call)')
        ck = R.get('checkpoints') or []
        g.check(len(ck) == a.epochs, f'{lab}: {len(ck)} checkpoints saved, one per epoch')
        gap = [{'epoch': x['epoch'], 'frozen_val_top1': x['val_top1'], 'keras_val_accuracy': x['log_val_accuracy'],
                'frozen_seconds': x.get('sys/frozen_seconds')} for x in rows]
        g.info(f'{lab} recorded beside, not a pass condition', gap)
        rec['runs'][lab] = {'rows': [{k: x.get(k) for k in ('epoch', 'H1', 'H2', 'NR', 'val_correct', 'val_n', 'train_count',
                                                             'sys/frozen_seconds')} for x in rows],
                            'spy': sp, 'checkpoints': ck, 'watch': w}
        by_epoch = {x['epoch']: x for x in rows}
        for c in ck:
            ex = work / f'expect-{lab}-{c["epoch"]}.json'
            ex.write_text(json.dumps(by_epoch[c['epoch']], sort_keys=True))
            args = ['--ckpt', c['path'], '--expect', ex, '--archive', a.archive, '--archive-sha', inp['archive_sha256'],
                    '--arm', arm] + (['--form', form] if form else []) + (['--headline', headline] if headline else [])
            child, audit = costgate.run_child(g, 'audit', device=a.device, work=work / 'audits', data_manifest=a.data_manifest,
                                              children_out=a.children_out, extra=args, name=f'audit-{lab}-e{c["epoch"]}')
            costgate.check_child(g, f'audit-{lab}-e{c["epoch"]}', child, audit, a.device)
        if ck:
            c = ck[-1]
            ex = work / f'expect-{lab}-{c["epoch"]}.json'
            args = ['--ckpt', c['path'], '--expect', ex, '--archive', a.archive, '--archive-sha', inp['archive_sha256'],
                    '--arm', arm, '--mutate'] + (['--form', form] if form else []) + (['--headline', headline] if headline else [])
            child, audit = costgate.run_child(g, 'audit', device=a.device, work=work / 'audits', data_manifest=a.data_manifest,
                                              children_out=a.children_out, extra=args, name=f'audit-{lab}-mutated')
            costgate.check_child(g, f'audit-{lab}-mutated', child, audit, a.device)
    g.check(len(rec['runs']) == len(ARMS), f'{len(rec["runs"])} of {len(ARMS)} runs checked')
    g.check_clean('at end')
    return g.finish(rec, a.json)


if __name__ == '__main__':
    os.environ.pop('XLA_FLAGS', None)
    sys.exit(main())
