"""Gate G14 (PREFLIGHT-PLAN section 3 row G14; STUDY v0.10 section 13 "CH-8 runner, F000" row: "F000
equals CH-8 in tensors, forward, backward, one optimizer step and native cost on identical batches";
section 5 F000 "our reconstruction of Chang"; section 6 step 4; modules M5 cbl/model.py and M13
cbl/runner.py). Run once with --device cpu (PCC) and once with --device gpu (PCG, the RTX 4060 Ti), in
the venv of record, on the 2-file smoke archive of B2, at a committed sha.

Children, each a fresh process of the venv of record (gates/runner_child.py) under the open-file
audit: CH-8's initial-tensor file for seed 1 on this device (decision U1); for CH-8 (Chang's
get_model) and F000 (M5) each, `step` (the runner's prepare, the probe at the initial state, one
`train_on_batch` on the fixed batch of the first 2790 training rows, the probe again) and `run` (the
runner, --epochs epochs, W&B disabled). On the CPU one more F000 run logs through M15 in W&B offline
mode. On the GPU every comparison runs with XLA_FLAGS=--xla_gpu_deterministic_ops=true on both sides
(owner decision B5-1), and the two runs repeated under the default flags are recorded as the
determinism control (a finding, not a pass condition).

Checks (F000 against CH-8, bitwise):
1. structure: the same layer classes and configurations in `_flatten_layers` order, the same 244 roles
   with the same shapes, dtypes and trainable flags (F000's layers carry no names, so a fresh process
   gives them CH-8's automatic names);
2. tensors: F000's own draw equals CH-8's file in all 244 roles before the load (same classes,
   same construction order, same initializer draws); after the load equal; F000's unmatched and
   unused sets are empty (STUDY section 6 step 4; the G10 part for F000);
3. forward and backward at the initial state: inference logits on the validation set, training-mode
   logits on the fixed batch, the loss, the gradient by role, the non-trainable state the training
   forward would write;
4. native cost: the per-layer EBOPs that training forward writes, layer by layer, and their sum;
5. one optimizer step (`train_on_batch` on the fixed batch): the returned loss and accuracy, every
   variable by role, the optimizer variables (iterations 1), the per-layer EBOPs counters and
   FreeEBOPs's sum, which equals the step-1 sum of check 4;
6. a short fit of --epochs epochs on identical batches: per epoch the logs, the batch-index and
   _index_map hashes (the G11 part across arms), every variable, the optimizer variables, the
   gradient, the step state, the inference logits;
7. CPU only: the F000 run with W&B offline equals the disabled one in every record, and its offline
   run directory exists (M15 through the runner).

Every reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g14_f000.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
        --archive SMOKE_ARCHIVE [--epochs 3] [--data-manifest PATH] [--json PATH] [--children-out DIR]
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import gatelib  # noqa: E402
import runnergate as rg  # noqa: E402
from cbl import rng  # noqa: E402

GATE = 'G14'
SEED = 1


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
    rec = {'header': g.header(a.expect_sha12), 'device': a.device, 'epochs': a.epochs, 'seed': SEED}
    gatelib.check_venv_of_record(g)
    print('label       engineering, not quotable')
    head = rec['header']['HEAD']
    work = Path(a.work)
    g.check(not work.exists(), f'work directory {work} does not exist yet')
    work.mkdir(parents=True)
    inp = rg.inputs(a.data_manifest, a.archive)
    rec['inputs'] = inp
    got = rg.sha256_file(a.archive)
    g.check(got == inp['archive_sha256'] and got not in (inp['record_archive_sha256'], inp['heldout_archive_sha256']),
            f'the archive is the smoke archive {got[:12]} (not the archive of record, not the held-out archive)')
    ph = rg.placeholder_dir(work, a.archive)          # watched only: no child here may open a 150c-test.h5
    main_env = {'XLA_FLAGS': rg.DETERMINISTIC_GPU_FLAG} if a.device == 'gpu' else {}
    rec['main_env'] = main_env
    kids = rg.Children(g, device=a.device, inp=inp, work=work, children_out=a.children_out,
                       audit_watches=rg.watches(inp, ph), env_extra=main_env)

    d, _ = kids('draw', 'draw', SEED, ['--out-dir', work / 'draw' / 'init', '--place', f'PC{a.device[0].upper()} gate G14'])
    if d is None:
        g.check_clean('at end')
        return g.finish(rec, a.json)
    man = json.loads(Path(d['manifest']).read_text())
    file_roles = {r: e['sha256'] for r, e in man['roles'].items()}
    rec['draw'] = {'manifest_sha256': d['manifest_sha256'], 'file_sha256': d['file']['sha256'], 'device': d['identity']['device']}

    def args(arm, mode='disabled'):
        return rg.draw_args(d, head, inp['bundle_sha256'], epochs=a.epochs, arm=arm, wandb_mode=mode)

    sc, _ = kids('step-CH-8', 'step', SEED, args('CH-8') + ['--out-dir', work / 'step-CH-8' / 'out'])
    sf, _ = kids('step-F000', 'step', SEED, args('F000') + ['--out-dir', work / 'step-F000' / 'out'])
    rc, _ = kids('run-CH-8', 'run', SEED, args('CH-8') + ['--out-dir', work / 'run-CH-8' / 'out'])
    rf, _ = kids('run-F000', 'run', SEED, args('F000') + ['--out-dir', work / 'run-F000' / 'out'])
    ro = None
    if a.device == 'cpu':
        ro, _ = kids('run-F000-offline', 'run', SEED, args('F000', 'offline') + ['--out-dir', work / 'run-F000-offline' / 'out'])
    dc = df = None
    if a.device == 'gpu':
        dc, _ = kids('run-CH-8-default-flags', 'run', SEED, args('CH-8') + ['--out-dir', work / 'run-CH-8-dflt' / 'out'], env_extra={})
        df, _ = kids('run-F000-default-flags', 'run', SEED, args('F000') + ['--out-dir', work / 'run-F000-dflt' / 'out'], env_extra={})
    if sc is None or sf is None:
        g.check(False, 'both step children completed; the comparisons need them')
        g.check_clean('at end')
        return g.finish(rec, a.json)

    # 1. structure
    S0, S1 = sc['context']['structure'], sf['context']['structure']
    g.check(S0['top_classes'] == S1['top_classes'] and len(S0['top_classes']) == 14,
            f'structure: the same 14 top-level layer classes in model.layers order')
    g.check(S0['layers'] == S1['layers'],
            f'structure: {S0["n_layers"]} flattened layers with equal classes and configurations '
            f'({S0["n_without_config"]} internal quantizers have no get_config; their classes and variables are compared)')
    g.check(S0['variables'] == S1['variables'] and len(S1['variables']) == rng.CH8_N_VARIABLES,
            'structure: the same 244 roles with equal shapes, dtypes and trainable flags')
    g.check(sc['callback_order'] == sf['callback_order'] and sc['compile'] == sf['compile'],
            'the runner gives both the same callbacks and compile')

    # 2. tensors
    lf = sf['load']
    g.check(lf['arm_id'] == 'F000' and lf['n_loaded'] == rng.CH8_N_VARIABLES and lf['unmatched'] == [] and lf['unused'] == [],
            'F000: 244 tensors loaded by role from CH-8\'s file; unmatched and unused sets empty (section 6 step 4)')
    g.check(lf['own_draw_before_load']['equal_roles'] == rng.CH8_N_VARIABLES and lf['own_draw_before_load']['differing_roles'] == [],
            'F000\'s own draw equals CH-8\'s file in all 244 roles before the load')
    g.check(sc['init']['variables'] == sf['init']['variables'] == file_roles, 'after the load both equal the file in all 244 roles')
    g.check(sc['context']['data_arrays'] == sf['context']['data_arrays'] and sc['context']['datasets'] == sf['context']['datasets'],
            'both read the same data and built the same Datasets')

    # 3-4. forward, backward and native cost at the initial state
    di = rg.diff_snapshot(sc['init'], sf['init'])
    g.check(not di, 'initial state: inference logits, training-mode logits, loss, gradients by role and the step\'s '
                    f'non-trainable state equal{"" if not di else f"; differ: {di}"}')
    e0, e1 = sc['init']['step_ebops'], sf['init']['step_ebops']
    g.check(e0 == e1 and len(e0) > 0, f'native cost: per-layer EBOPs written by one training forward equal in {len(e0)} layers')
    g.info('engineering reading, not quotable: per-layer EBOPs of the training forward at initialization (smoke batch)', e0)
    g.check(sc['init']['n_model_losses'] == sf['init']['n_model_losses'],
            f'the same number of model losses ({sc["init"]["n_model_losses"]}: EBOPs penalties and regularizers)')

    # 5. one optimizer step
    g.check(sc['step_logs'] == sf['step_logs'], f'one optimizer step: train_on_batch returns the same loss and accuracy {sc["step_logs"]}')
    ds = rg.diff_snapshot(sc['after_step'], sf['after_step'])
    g.check(not ds, 'after the step: every variable by role, the optimizer variables, the gradient, the step state and the '
                    f'outputs equal{"" if not ds else f"; differ: {ds}"}')
    g.check(sc['after_step']['optimizer_iterations'] == sf['after_step']['optimizer_iterations'] == 1, 'optimizer iterations 1 in both')
    g.check(sc['after_step']['ebops'] == sf['after_step']['ebops'] == {k: v for k, v in sc['init']['step_ebops'].items()},
            'after the step the per-layer EBOPs counters equal the training forward\'s values of check 4')
    g.check(sc['after_step']['free_ebops'] == sf['after_step']['free_ebops'] == sum(e0.values()),
            f'FreeEBOPs\'s sum after the step equals the sum of the per-layer values ({sum(e0.values())}, engineering)')

    # 6. the short fit
    if rc is not None and rf is not None:
        Rc, Rf = rc['runner'], rf['runner']
        hc = [(e['batch_index_seq_sha256'], e['index_map_sha256']) for e in Rc['epoch_records']]
        hf = [(e['batch_index_seq_sha256'], e['index_map_sha256']) for e in Rf['epoch_records']]
        exp = rng.reconstruct_fit_stream(reseed=rng.prefit_reseed(SEED), index_map=Rc['index_map_at_reseed'],
                                         n_batches=Rc['flow']['datasets']['n_batches'], epochs=a.epochs)
        g.check(len(hc) == a.epochs and hc == hf == [(e['batch_index_seq_sha256'], e['index_map_sha256']) for e in exp['epochs']],
                f'G11 across arms: per-epoch fetched batch-index and _index_map hashes of CH-8 and F000 equal each other and '
                f'the reconstruction from prefit_reseed({SEED})')
        rows = rg.diff_rows(rc['probe_rows'], rf['probe_rows'])
        g.check(rg.all_equal(rows) and len(rows) == a.epochs,
                f'fit, per epoch ({a.epochs}): logs, all variables, optimizer variables, gradients, step state and inference '
                f'logits equal{"" if rg.all_equal(rows) else f"; first difference {rg.first_diff(rows)}"}')
        g.check(rg.untimed(Rc['epoch_records']) == rg.untimed(Rf['epoch_records']) and len(Rc['epoch_records']) == a.epochs,
                'the frozen-evaluation records equal (M8\'s counts and validation top-1 included; epoch timing aside)')
        g.check(Rf['guard']['lab_violations'] == [] and Rf['guard']['released_violations'] == [],
                'F000 run: no change of the global streams by lab or released callbacks')
        rec['per_epoch_hashes'] = hc
        rec['diffs'] = {'fit': rows}
    else:
        g.check(False, 'both runs completed')
    # 7. W&B offline
    if a.device == 'cpu':
        if ro is not None and rf is not None:
            ow = ro['runner']['wandb']
            files = ro.get('out_files') or []
            g.check(ow['mode'] == 'offline' and any(f.startswith('wandb/') and f.endswith('.wandb') for f in files),
                    f'W&B offline through M15: run {ow["run_name"]} wrote an offline run file')
            ro_rows = rg.diff_rows(rf['probe_rows'], ro['probe_rows'])
            g.check(rg.all_equal(ro_rows) and rf['initial'] == ro['initial'],
                    'the offline-logging F000 run equals the disabled one in every record')
        else:
            g.check(False, 'the offline F000 run completed')
    if a.device == 'gpu':
        if dc is not None and df is not None:
            rows_d = rg.diff_rows(dc['probe_rows'], df['probe_rows'])
            rec['gpu_default_flags_control'] = {'CH-8_vs_F000': rows_d,
                                                'CH-8_default_vs_deterministic': rg.diff_rows(rc['probe_rows'], dc['probe_rows']) if rc else None}
            g.info('FINDING (recorded, not a pass condition): CH-8 against F000 under the default XLA flags on the RTX 4060 Ti',
                   {'initial': rg.diff_snapshot(dc['initial'], df['initial']) or 'equal', 'epochs': rows_d})
        else:
            g.check(False, 'the two default-flag runs completed')
    g.check_clean('at end')
    return g.finish(rec, a.json)


if __name__ == '__main__':
    sys.exit(main())
