"""Gate G13 (PREFLIGHT-PLAN section 3 row G13; STUDY v0.10 section 13 reference-fidelity row: "CH-8
from our runner bitwise equal to the reference module on the pinned stack in initial weights, training
and inference outputs, per-layer EBOPs, loss, gradients and BN moving statistics, on identical
batches"; section 7.1; decisions U1, U2, U5; module M13 cbl/runner.py). Run once with --device cpu
(PCC) and once with --device gpu (PCG, the RTX 4060 Ti), in the venv of record, on the 2-file smoke
archive of B2, at a committed sha.

Seed 42, a gate-only seed: Chang's run_train.py seeds Python's `random` with 42 before its Datasets
and model (`run_train.py:42,73-76`), and the runner's section-6 flow with s = 42 seeds it with 42 at
the same point (`random.seed(42)`, `keras.utils.set_random_seed(42)`, after the data read, which draws
nothing from `random`), so the runner's initial-tensor file drawn on this device must equal the
release's own draw.

Children, each a fresh process of the venv of record (gates/runner_child.py) under the open-file
audit (no open of the held-out archive or any raw held-out file in any child):
  draw    CH-8's initial-tensor file for seed 42 on this device (cbl.runner.draw, decision U1);
  run-A   the runner: CH-8, seed 42, stage `gate`, --epochs epochs, W&B disabled (through M15), the
          file loaded by role, the read-only probe as the frozen-evaluation stand-in;
  ref-R   Chang's run_train.py by runpy, byte-unchanged, on a placeholder directory (`150c-train.h5` a
          link to the smoke archive, `150c-test.h5` 16 jets of zeros, never the held-out archive), with
          the harness of runner_child (device remap on the CPU only; the appended callback with the same
          probe but without M8's frozen evaluation, which the runner side runs every epoch (B7-8), so
          equality also shows that the evaluation changes nothing in training;
          the U5 reseed just before the real fit, the one declared departure the release cannot make
          itself; epochs set from 7000 to --epochs);
  run-B   run-A again (same-program control: the comparison is deterministic in this setting);
  ref-N   ref-R without the reseed (control: shows the comparison can fail, and that the reseed is
          what lines the batches up).
On the GPU, run-A, ref-R, run-B and ref-N run with XLA_FLAGS=--xla_gpu_deterministic_ops=true, on
both sides of every comparison (owner decision B5-1, PREFLIGHT.md); two more runs, run-C and run-D,
are run-A under the default flags, and their comparison with each other is recorded as the
determinism control of the pinned GPU stack (a finding, not a pass condition).

Checks (bitwise unless stated):
1. initial weights: the release's own draw at fit entry equals the runner's tensors after the load and
   the draw manifest in all 244 roles; the runner's own draw before the load equals the file;
2. data: X_train, y_train, X_val, y_val (M3's reader against Chang's get_data), the Datasets' shifts,
   maps and batch counts; the model structure (layer classes and configurations, roles);
3. recipe: callback order [FreeEBOPs, LearningRateScheduler, BetaScheduler, PBar, ParetoFront, StopIf]
   as released plus the appended FrozenEvalHook, in both; each released callback's restated arguments
   equal the release's live objects (PBar template, ParetoFront metrics, sides, file-name format and
   filter on fixed inputs, the beta and learning-rate schedules at 16 epochs from 0 to 7001, the StopIf
   condition); optimizer and loss configurations and steps_per_execution;
4. on identical batches, per epoch: the fetched batch-index and _index_map hashes, equal to the
   reconstruction from prefit_reseed(42); the logs (loss, accuracy, val_loss, val_accuracy, ebops,
   learning rate, beta); every variable by role (weights, BN moving statistics, per-layer EBOPs,
   quantizer state) and the optimizer variables; the gradient of the probe's loss by role, the
   training-mode logits and the state that step would write; the inference logits on the validation
   set; and the same probe at the initial state;
5. the runner's run: completed, the stream guard saw no change of the global streams by lab or
   released code, float32 matmul before and after fit, M15's disabled run opened with its run name;
6. controls: run-B equals run-A in every record; ref-N has the same initial weights, but its epoch-0
   batch order and every later record differ from run-A's.

Every reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g13_runner_ref.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
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
from cbl import rng, runner  # noqa: E402

GATE = 'G13'
SEED = 42


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
    g.info('bundle the runs were launched from', inp['bundle_sha256'])
    ph = rg.placeholder_dir(work, a.archive)
    g.check(rg.sha256_file(ph / '150c-test.h5') != inp['heldout_archive_sha256'], 'the placeholder 150c-test.h5 is not the held-out archive')
    main_env = {'XLA_FLAGS': rg.DETERMINISTIC_GPU_FLAG} if a.device == 'gpu' else {}
    rec['main_env'] = main_env
    kids = rg.Children(g, device=a.device, inp=inp, work=work, children_out=a.children_out,
                       audit_watches=rg.watches(inp, ph), env_extra=main_env)

    d, _ = kids('draw', 'draw', SEED, ['--out-dir', work / 'draw' / 'init', '--place', f'PC{a.device[0].upper()} gate G13'])
    if d is None:
        g.check_clean('at end')
        return g.finish(rec, a.json)
    dev = d['identity']['device']
    if a.device == 'gpu':
        g.check(dev['jax_platform'] == 'gpu' and all('4060 Ti' in k for k in dev['device_kind']) and dev['dataset_device'] == 'gpu:0',
                f'draw on the RTX 4060 Ti: {dev}')
    else:
        g.check(dev['jax_platform'] == 'cpu' and dev['dataset_device'] == 'cpu:0', f'draw on the CPU: {dev}')
    man = json.loads(Path(d['manifest']).read_text())
    file_roles = {r: e['sha256'] for r, e in man['roles'].items()}
    rec['draw'] = {'manifest_sha256': d['manifest_sha256'], 'file_sha256': d['file']['sha256'], 'device': dev}
    run_args = rg.draw_args(d, head, inp['bundle_sha256'], epochs=a.epochs, arm='CH-8', wandb_mode='disabled')
    ref_args = ['--epochs', a.epochs, '--ref-data-dir', ph]

    A, auA = kids('run-A', 'run', SEED, run_args + ['--out-dir', work / 'run-A' / 'out'])
    R, auR = kids('ref-R', 'ref', SEED, ref_args + ['--out-dir', work / 'ref-R' / 'out'])
    B, _ = kids('run-B', 'run', SEED, run_args + ['--out-dir', work / 'run-B' / 'out'])
    N, _ = kids('ref-N', 'ref', SEED, ref_args + ['--no-reseed', '--out-dir', work / 'ref-N' / 'out'])
    C = D = None
    if a.device == 'gpu':
        C, _ = kids('run-C', 'run', SEED, run_args + ['--out-dir', work / 'run-C' / 'out'], env_extra={})
        D, _ = kids('run-D', 'run', SEED, run_args + ['--out-dir', work / 'run-D' / 'out'], env_extra={})
    if A is None or R is None:
        g.check(False, 'run-A and ref-R completed; the comparisons need them')
        g.check_clean('at end')
        return g.finish(rec, a.json)
    RA = A['runner']

    # the reference ran the release as released, apart from the harness
    g.check(R['run_train_sha256'] == runner.code_base_sha256('chang/run_train.py'),
            f'ref-R: run_train.py byte-unchanged ({R["run_train_sha256"][:12]}, code-base.txt), run by runpy as __main__')
    g.check(R['release_epochs'] == runner.RELEASE_EPOCHS and R['args']['model'] == 'xfm' and R['args']['n_constituents'] == 8
            and R['args']['n_features'] == 3 and R['args']['batch_size'] == runner.TRAIN_BATCH
            and R['args']['learning_rate'] == runner.LEARNING_RATE,
            f'ref-R: run_train.py called fit with epochs={R["release_epochs"]} (set to {a.epochs} by the harness), '
            f'args {R["args"]}')
    made = R['datasets_made']
    want_used = 'cpu:0' if a.device == 'cpu' else 'gpu:0'
    g.check(len(made) == 2 and all(m['device_asked'] == 'gpu:0' and m['device_used'] == want_used for m in made)
            and [m['shuffle'] for m in made] == [True, False],
            f'ref-R: run_train.py asked for gpu:0 twice (run_train.py:73-74); the harness passed {want_used}'
            + (' (the smoke-only device remap, CPU only)' if a.device == 'cpu' else ' (no remap)'))
    g.check(rg.placeholder_opened(auR), 'ref-R: Chang\'s get_data opened the placeholder 150c-test.h5 (positive control of the audit)')
    g.check(not rg.placeholder_opened(auA), 'run-A: the runner opened no 150c-test.h5 at all (M3\'s reader; STUDY section 4)')
    g.check(RA['dataset_device'] == want_used and A['context']['datasets']['dataset_device'] == want_used,
            f'run-A: the runner passed {want_used} to both Datasets (decision U2)')

    # 1. initial weights
    lo = RA['load']
    g.check(lo['n_loaded'] == rng.CH8_N_VARIABLES and lo['unmatched'] == [] and lo['unused'] == []
            and lo['own_draw_before_load']['equal_roles'] == rng.CH8_N_VARIABLES,
            'run-A: the runner\'s own draw equals the file in all 244 roles; 244 loaded by role; unmatched and unused empty')
    g.check(R['initial_variables'] == file_roles,
            'initial weights: the release\'s own draw at fit entry equals the draw manifest in all 244 roles')
    g.check(A['initial']['variables'] == file_roles and R['initial']['variables'] == file_roles,
            'initial weights: the runner\'s tensors after the load equal the file and the release\'s, role by role')
    dinit = rg.diff_snapshot(A['initial'], R['initial'])
    g.check(not dinit, f'initial state: probe equal (inference and training-mode logits, loss, gradients by role, '
                       f'the step\'s non-trainable state){"" if not dinit else f"; differ: {dinit}"}')

    # 2. data and structure
    g.check(A['context']['data_arrays'] == R['context']['data_arrays'],
            'data: X_train, y_train, X_val and y_val of M3\'s reader equal Chang\'s get_data in dtype, shape and bytes')
    g.check(A['context']['datasets'] == R['context']['datasets'],
            f'Datasets: shifts, batch counts and device equal {A["context"]["datasets"]}')
    g.check(rng.int_seq_sha256(made[0]['index_map']) == RA['flow']['datasets']['index_map_sha256_at_construction'],
            'Datasets: the training map at construction is the same in both')
    g.check(A['context']['structure'] == R['context']['structure'],
            f'model: {A["context"]["structure"]["n_layers"]} layers with equal classes and configurations, 244 roles '
            'with equal shapes, dtypes and trainable flags')

    # 3. recipe
    g.check(RA['callback_order'] == list(runner.RELEASED_ORDER) + [runner.APPENDED] and R['release_order'] == list(runner.RELEASED_ORDER)
            and R['fit_order'] == RA['callback_order'],
            f'callback order as released plus the appended callback: {RA["callback_order"]}')
    sa, sr = RA['callback_signature'], R['callback_signature']
    for name in runner.RELEASED_ORDER:
        if name == 'FreeEBOPs':
            continue
        g.check(sa.get(name) == sr.get(name) and sa.get(name) is not None,
                f'{name}: the runner\'s restated arguments equal the release\'s live object '
                f'({"equal" if sa.get(name) == sr.get(name) else f"{sa.get(name)} != {sr.get(name)}"})'[:400])
    ca, cr = RA['compile'], R['compile']
    g.check(ca['optimizer'] == cr['optimizer'] == 'Adam' and ca['optimizer_config'] == cr['optimizer_config']
            and ca['loss'] == cr['loss'] and ca['steps_per_execution'] == cr['steps_per_execution'] == runner.STEPS_PER_EXECUTION,
            'compile: Adam with the same configuration, the same loss configuration, steps_per_execution 4')

    # 4. identical batches, per epoch
    ha = [(e['batch_index_seq_sha256'], e['index_map_sha256']) for e in RA['epoch_records']]
    hr = [(e['batch_index_seq_sha256'], e['index_map_sha256']) for e in R['epoch_records']]
    exp = rng.reconstruct_fit_stream(reseed=rng.prefit_reseed(SEED), index_map=RA['index_map_at_reseed'],
                                     n_batches=RA['flow']['datasets']['n_batches'], epochs=a.epochs)
    he = [(e['batch_index_seq_sha256'], e['index_map_sha256']) for e in exp['epochs']]
    g.check(len(ha) == a.epochs and ha == hr == he,
            f'batch order: per-epoch fetched batch-index and _index_map hashes equal in both and equal the reconstruction '
            f'from prefit_reseed({SEED}) (epoch 0 {ha[0][0][:12] if ha else None})')
    g.check(RA['reseed'] == R['reseed'] == rng.prefit_reseed(SEED) and RA['index_map_at_reseed'] == R['index_map_at_fit'],
            f'both reseeded with prefit_reseed({SEED}) = {rng.prefit_reseed(SEED)} with the same map')
    rows = rg.diff_rows(A['probe_rows'], R['probe_rows'])
    g.check(len(A['probe_rows']) == a.epochs and rg.all_equal(rows),
            f'per epoch ({a.epochs}): logs, all 244 variables by role (weights, BN moving statistics, per-layer EBOPs, '
            f'quantizer state), optimizer variables, gradients by role, training-mode logits and step state, '
            f'inference logits on the validation set: bitwise equal{"" if rg.all_equal(rows) else f"; first difference {rg.first_diff(rows)}"}')
    last = A['probe_rows'][-1] if A['probe_rows'] else {}
    g.info('engineering readings at the last epoch, not quotable', {
        'logs': {k: last.get('logs', {}).get(k) for k in ('loss', 'accuracy', 'val_loss', 'val_accuracy', 'ebops', 'beta', 'learning_rate')},
        'free_ebops': last.get('free_ebops'), 'infer_correct': last.get('infer_correct'), 'infer_n': last.get('infer_n')})
    g.check(RA['final_variables_sha256'] == R['final_variables_sha256'], 'final variables equal after fit')

    # 5. the runner's run
    gd = RA['guard']
    g.check(gd['lab_violations'] == [] and gd['released_violations'] == [] and gd['armed_at'] and gd['disarmed_at'],
            'run-A: the stream guard was armed at the reseed and disarmed after fit; no lab or released callback changed '
            'the global streams')
    g.check(RA['matmul_precision_before_fit']['jax_default_matmul_precision'] == 'float32'
            and RA['matmul_precision_after_fit']['jax_default_matmul_precision'] == 'float32'
            and R['matmul_precision_after_fit'] == 'float32', 'float32 matmul before and after fit in both')
    g.check(RA['wandb']['mode'] == 'disabled' and RA['wandb']['run_name'].startswith(f'CH-8_nf_s{SEED}_gate_ramp_a1_src-'),
            f'run-A: M15 opened the run in disabled mode as {RA["wandb"]["run_name"]}')
    g.check(RA['epochs_completed'] == a.epochs and len(RA['epoch_records']) == a.epochs,
            f'run-A: {RA["epochs_completed"]} epochs, one frozen-evaluation record each')

    # 6. controls
    if B is not None:
        rb = rg.diff_rows(A['probe_rows'], B['probe_rows'])
        g.check(rg.all_equal(rb) and A['initial'] == B['initial'] and rg.untimed(RA['epoch_records']) == rg.untimed(B['runner']['epoch_records']),
                'control: run-B (the same program in another process) equals run-A in every record'
                + ('' if rg.all_equal(rb) else f'; first difference {rg.first_diff(rb)}'))
    if N is not None:
        hn = [(e['batch_index_seq_sha256'], e['index_map_sha256']) for e in N['epoch_records']]
        rn = rg.diff_rows(A['probe_rows'], N['probe_rows'])
        g.check(N['initial_variables'] == file_roles and N['reseed'] is None and hn and hn[0] != ha[0]
                and all(d.get('variables') for d in rn),
                'control: ref-N (no reseed) starts from the same weights, but its epoch-0 batch order differs and every '
                'epoch\'s variables differ from run-A\'s (the comparison can fail)')
    if a.device == 'gpu':
        if C is not None and D is not None:
            rcd = rg.diff_rows(C['probe_rows'], D['probe_rows'])
            dcd = rg.diff_snapshot(C['initial'], D['initial'])
            rec['gpu_default_flags_control'] = {'initial': dcd, 'epochs': rcd}
            g.info('FINDING (recorded, not a pass condition): run-C against run-D, the same program under the default '
                   'XLA flags on the RTX 4060 Ti', {'initial': dcd or 'equal', 'epochs': rcd})
            g.check(C['runner']['epoch_records'] and [e['batch_index_seq_sha256'] for e in C['runner']['epoch_records']]
                    == [e[0] for e in ha], 'run-C and run-D (default flags): the batch order is the same as run-A\'s')
            rec['gpu_default_vs_deterministic'] = rg.diff_rows(A['probe_rows'], C['probe_rows'])
        else:
            g.check(False, 'run-C and run-D completed')
    rec['per_epoch_hashes'] = ha
    rec['diffs'] = {'A_vs_R': rows}
    g.check_clean('at end')
    return g.finish(rec, a.json)


if __name__ == '__main__':
    sys.exit(main())
