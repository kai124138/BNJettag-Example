"""Gate G49, the PC CPU part (PREFLIGHT-PLAN section 3 row G49; STUDY v0.10 section 13 "metric reload agreement
within 1e-7 with TF32 off"; docs/conventions/quantization-and-cost.md:43-45; docs/methodology/03-phases.md phase 2
"every arm's checkpoint reloads within 1e-7 with TF32 off"; modules M8, M13, M17). PC CPU, venv of record, the
training archive of record (`~/lab/scratch/cbl1016/data/150c-train.h5`; never the held-out archive). The GPU path is
B13's.

For every config cbl.runner.validate_config admits under G57's constructed complete input record (109 since B10), one
fresh child (gates/reload_child.py train) runs the runner for one epoch on the archive of record and saves, after the
epoch's frozen evaluation, the model as a .keras file and the validation logits over the full validation set (62,000
jets); a second fresh child (reload) loads the file and checks, against the first:
- M8's frozen evaluation of the reloaded model equals epoch 0's record exactly (validation correct count and top-1,
  H1, H2, NR, per-layer and per-group subtotals, snap count, variable hash, binary-site record);
- the validation logits agree within 1e-7 in absolute value (max |difference| reported; equality recorded);
- Keras's evaluate on the validation Dataset gives the loss and accuracy Keras logged at epoch 0 within 1e-7;
- the matmul precision is float32 in both processes (TF32 off; the CPU has no TF32, recorded anyway).
The config each child receives is a gate copy of the generated one, with only these fields changed and recorded:
code_sha and init_tensors (this HEAD and the seed's draw on the archive of record, as gen.configs.with_handoff makes a
launch copy, with the local manifest path), wandb.mode disabled and run_name null, epochs 1, and stage gate for S1,
S3 and S4 configs. Each seed's CH-8 initial-tensor file is drawn once on the archive of record. Children run under the
kernel open-file audit of the held-out paths. Every reading is engineering, not quotable.

    <venv>/bin/python -I -B gates/g49_reload.py --expect-sha12 SHA12 --work DIR --archive ARCHIVE_OF_RECORD \\
        [--json PATH] [--children-out DIR] [--limit N]
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import fileaudit  # noqa: E402
import g57_configs as G57  # noqa: E402
import gatelib  # noqa: E402
import runnergate as rg  # noqa: E402
from gen import configs as C  # noqa: E402
from gen import inputs as I  # noqa: E402

GATE = 'G49'
TOL = 1e-7


def gate_copy(cfg, *, head, manifest, manifest_sha):
    out = C.with_handoff(cfg, code_sha=head, init_manifest_sha256=manifest_sha)
    out['init_tensors']['manifest_path'] = str(manifest)
    out['wandb'] = dict(out['wandb'], mode='disabled', run_name=None)
    changed = ['code_sha', 'init_tensors.manifest_path', 'init_tensors.manifest_sha256', 'wandb.mode', 'wandb.run_name',
               f'epochs {out["epochs"]} -> 1']
    out['epochs'] = 1
    if out['stage'] in ('S1', 'S3', 'S4'):
        changed.append(f'stage {out["stage"]} -> gate')
        out['stage'] = 'gate'
    return out, changed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--expect-sha12', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--data-manifest', default=str(FORK / 'manifests' / 'data-prep-150c.json'))
    ap.add_argument('--json', default=None)
    ap.add_argument('--children-out', default=None)
    ap.add_argument('--limit', type=int, default=None, help='development only: the first N configs')
    a = ap.parse_args()
    g = gatelib.Gate(GATE)
    rec = {'header': g.header(a.expect_sha12)}
    gatelib.check_venv_of_record(g)
    print('label       engineering, not quotable')
    work = Path(a.work).resolve()
    fileaudit.require_unsynced(work)
    g.check(not work.exists(), f'work directory {work} does not exist yet')
    work.mkdir(parents=True)
    head = rec['header']['HEAD']
    man = json.loads(Path(a.data_manifest).read_text())
    got = rg.sha256_file(a.archive)
    g.check(got == man['train']['archive']['sha256'] and got != man['heldout']['archive']['sha256'],
            f'the archive is the training archive of record {got[:12]} (not the smoke archive, not the held-out archive)')
    inp = rg.inputs(a.data_manifest, a.archive)
    inp['archive_sha256'] = got
    ph = work / 'audit-placeholder'
    ph.mkdir()
    (ph / '150c-test.h5').write_bytes(b'')
    watches = rg.watches(inp, ph)
    env = {k: v for k, v in os.environ.items() if k not in ('KERAS_BACKEND', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
                                                           'XLA_FLAGS')}

    def child(name, cmd, extra):
        w = work / 'children' / name
        w.mkdir(parents=True, exist_ok=True)
        out = w / f'{name}.json'
        argv = [sys.executable, '-I', '-B', str(FORK / 'gates' / 'reload_child.py'), cmd, '--out', str(out), '--archive',
                a.archive] + [str(x) for x in extra]
        r = fileaudit.run_audited(argv, watches=watches, env=env, cwd=str(w), timeout=3600)
        (w / f'{name}.stdout.txt').write_text(r['stdout'])
        (w / f'{name}.stderr.txt').write_text(r['stderr'])
        R = json.loads(out.read_text()) if out.exists() else None
        if a.children_out:
            co = Path(a.children_out)
            co.mkdir(parents=True, exist_ok=True)
            for f in (out, w / f'{name}.stderr.txt'):
                if f.exists():
                    shutil.copy2(f, co / f.name)
            (co / f'{name}.argv.txt').write_text(' '.join(argv) + '\n')
        fe = rg.forbidden_events({k: r[k] for k in ('events', 'events_before_start', 'fd_paths')})
        ok = r['returncode'] == 0 and R is not None and 'error' not in R
        return (R if ok else None), fe, (R or {}).get('error')

    # the configs the runner admits under the constructed complete record
    cfgs = C.build_all(I.load())
    with tempfile.TemporaryDirectory(dir=str(work)) as td:
        full = C.build_all(G57.complete_inputs(Path(td), cfgs))
    admitted = [c for c in full if G57.expected_pass(c)]
    g.check(len(admitted) == G57.N_PASS == 109, f'{len(admitted)} admitted configs under the constructed complete record')
    if a.limit:
        admitted = admitted[:a.limit]
        g.info('development limit', a.limit)
    seeds = sorted({c['seed'] for c in admitted})
    draws = {}
    for s in seeds:
        R, fe, err = child(f'draw-s{s}', 'draw', ['--seed', s, '--out-dir', work / 'init'])
        g.check(R is not None and not fe, f'draw seed {s} on the archive of record ({(R or {}).get("manifest_sha256", err)})')
        draws[s] = R
    rows = []
    worst = 0.0
    for c in admitted:
        name = C.config_path(c).replace('configs/', '').replace('/', '_').replace('.json', '')
        d = draws.get(c['seed'])
        if d is None:
            g.check(False, f'{name}: no draw for seed {c["seed"]}')
            continue
        gc, changed = gate_copy(c, head=head, manifest=d['manifest'], manifest_sha=d['manifest_sha256'])
        cp = work / 'configs' / f'{name}.json'
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(gc, indent=1, sort_keys=True))
        save = work / 'ckpt' / f'{name}.keras'
        save.parent.mkdir(parents=True, exist_ok=True)
        lg = work / 'logits' / f'{name}.npy'
        lg.parent.mkdir(parents=True, exist_ok=True)
        T, fe1, e1 = child(f'train-{name}', 'train', ['--config', cp, '--out-dir', work / 'runs' / name, '--save', save,
                                                       '--logits', lg, '--source-bundle', inp['bundle_sha256']])
        trained = T is not None and T['runner'].get('end') == 'completed' and T.get('row0') is not None and save.exists()
        L, fe2 = None, []
        if trained:
            tr = work / 'children' / f'train-{name}' / f'train-{name}.json'
            L, fe2, e2 = child(f'reload-{name}', 'reload', ['--config', cp, '--ckpt', save, '--logits', lg, '--train-record', tr])
        p = (T or {}).get('probe') or {}
        logs = p.get('logs') or {}
        ev = (L or {}).get('evaluate') or {}
        dl = abs(ev.get('loss', float('nan')) - logs.get('val_loss', float('nan'))) if L else None
        da = abs(ev.get('accuracy', float('nan')) - logs.get('val_accuracy', float('nan'))) if L else None
        mx = ((L or {}).get('logits') or {}).get('max_abs_diff')
        ok = (trained and L is not None and not fe1 and not fe2 and L['audit_mismatches'] == []
              and mx is not None and mx <= TOL and dl is not None and dl <= TOL and da is not None and da <= TOL
              and p.get('matmul_precision') == 'float32' and L.get('matmul_precision') == 'float32')
        if mx is not None:
            worst = max(worst, mx, dl or 0.0, da or 0.0)
        g.check(ok, f'{name}: reload agreement (audit mismatches {(L or {}).get("audit_mismatches")}, logits max |diff| '
                    f'{mx}, val loss |diff| {dl}, val accuracy |diff| {da}, precision {p.get("matmul_precision")}/'
                    f'{(L or {}).get("matmul_precision")}, forbidden events {len(fe1) + len(fe2)})'
                    + ('' if trained else f' (training child: {e1 or (T or {}).get("runner")})'))
        rows.append({'config': C.config_path(c), 'label': c['label'], 'stage': c['stage'], 'seed': c['seed'], 'changed': changed,
                     'trained': trained, 'audit_mismatches': (L or {}).get('audit_mismatches'), 'logits_max_abs_diff': mx,
                     'logits_equal': ((L or {}).get('logits') or {}).get('equal'), 'val_loss_abs_diff': dl,
                     'val_accuracy_abs_diff': da, 'core': (L or {}).get('core'), 'ckpt_sha256': (T or {}).get('ckpt_sha256')})
        shutil.rmtree(work / 'runs' / name, ignore_errors=True)
    n_equal = sum(1 for r in rows if r['logits_equal'])
    g.info('summary', {'configs': len(rows), 'logits bitwise equal': n_equal, 'largest |difference| (logits, loss, accuracy)': worst})
    g.check(len(rows) == len(admitted) and all(r['trained'] for r in rows),
            f'all {len(rows)} admitted configs trained one epoch on the archive of record and reloaded in a fresh process')
    rec['rows'] = rows
    rec['tolerance'] = TOL
    g.check_clean('at end')
    return g.finish(rec, a.json)


if __name__ == '__main__':
    sys.exit(main())
