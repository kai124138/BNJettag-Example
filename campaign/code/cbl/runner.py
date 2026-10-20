"""M13: the training runner, `run_train.py`'s `__main__` recipe re-expressed for every arm.

STUDY means `review/snapshots/v0.10/STUDY.md` (sha256 7180c24f6b55807e...), sections 5, 6, 7.1, 7.3
and the section 13 reference-fidelity and CH-8 runner rows; "plan" means PREFLIGHT-PLAN.md (sha256
2f1be9f3...30ff8), section 2 row M13 and section 8 decisions U1, U2 and U5. Gates G13 (the runner
against `run_train.py` run by runpy), G14 (F000 against CH-8), G10 and G11 (the F000 parts).

The recipe, in the release's order (`chang/run_train.py:40-104`), with the declared departures only:
1. `cbl.env.setup()` (M1): the two shims, Chang's `model.py`, the float32 matmul override (STUDY
   section 7.3). Then Chang's `run_train` module is imported, unchanged and hash-checked, for its
   schedule function `cosine_decay_restarts_schedule` (`run_train.py:21-37`); its top level sets
   XLA_PYTHON_CLIENT_MEM_FRACTION to '0.5' and seeds both streams with 42 (`:4`, `:17-18`). The
   runner restores the memory fraction it was given before the JAX backend starts (it refuses if
   the backend has already started), and the seeds are replaced by step 1 below.
2. Data (STUDY section 6 steps 1-3 through M4's `SeedFlow`): `np.random.seed(42)` and M3's reader
   `cbl.data.read_train_val`, bitwise equal to Chang's `get_data` without its held-out lines (gate
   G07). This is the declared I/O departure (STUDY section 4 "Held-out test set"): Chang's `get_data`
   is imported with his module but never called, because it opens `150c-test.h5`. Then
   `random.seed(s)` and `keras.utils.set_random_seed(s)`, the two hgq Datasets as `run_train.py:73-74`
   with the device the runner passes (decision U2: 'gpu:0' on the GPU as released, 'cpu:0' on the
   CPU), and the model (`:76`): Chang's `get_model('xfm', 7, 7, 1e-8, 8, True)` for CH-8, M5's
   builder for the lab arms. [B10-F1] `cbl.lossguard.guard` then wraps the model's `stateless_call`: every
   layer's Keras `_loss_ids` set is emptied before each call, so that no penalty term is dropped from a traced
   training step by a stale loss id, and a training call whose loss-term count differs from the model's
   structure raises (`rec['lossguard']`; the process-dependent counters go to `rec['lossguard_counters']`).
3. Initial tensors (decision U1, section 6 step 4): the seed's CH-8 file is loaded by role; the
   config names the manifest and its sha256 (`init_tensors`), and the loader refuses any mismatch.
4. Callbacks and compile as `run_train.py:79-100`, with Chang's literal arguments restated here
   (they live inside `__main__` and cannot be imported; G13 compares each with the live objects of
   the release): FreeEBOPs, LearningRateScheduler with Chang's schedule, BetaScheduler with
   `PieceWiseSchedule([(0, 2e-8, 'linear'), (2000, 3e-7, 'log'), (7000, 3.0e-6, 'constant')])`, PBar,
   ParetoFront and StopIf, in the order of `:95`, then the appended frozen-evaluation callback
   (STUDY section 7.1; M8, `cbl/frozen_eval.py`, since B7). Adam with defaults, sparse categorical
   cross-entropy from logits, accuracy, `steps_per_execution=4`.
5. Pre-fit reseed (decision U5, section 6 step 5) with the stream guard armed (M4), then `fit` for the
   config's epochs (7000; engineering stages may set fewer) with `validation_data` and `verbose=0`.
The runner prints `len(dataset_train)` as `run_train.py:102` does and does not print
`model.summary()` (`:77`), which changes no state (G13 compares every record with the release).

The appended callback (`frozen_eval_hook`, M8's `cbl.frozen_eval.make_hook` since B7) marks each Keras
epoch in M4's batch-index recorder, writes the epoch's `batch_index_seq_sha256` and `index_map_sha256` and
M8's scalars (`cbl.frozen_eval.LOG_KEYS`) into the logs under new keys only, runs M8's frozen evaluation
(`ctx.evaluator`, a `cbl.frozen_eval.FrozenEvaluator` on the validation Dataset's tensors) and the read-only
step after it, inside `guard.watch('frozen evaluation')`, refuses if anything changed a key the released
callbacks wrote, appends the interfaces.md section 1 record to `epochs.jsonl`, and logs it through M15
(W&B). The PID (M12, B9) will read the same cached evaluation (`ctx.evaluator`).

The front saver (M9, `cbl/front.py`; wired in B9 as interfaces.md section 17 states, owner decision B8-14):
`prepare` binds a `FrontSaver` to the context (`front.register`, ablation mode `FRONT_ABLATIONS` = 'run_end',
B8-17, and `keep_superseded` false, B8-3) and the step after M8's evaluation runs it on the epoch's section 1
record (`front.chain_evaluator` around `cbl.frozen_eval.section1_record`, which reads the evaluation already
made that epoch and evaluates nothing), then a gate's probe. The callback order is unchanged: the saver runs
inside the appended callback's frozen-evaluation step, under the stream guard, and writes no log key. It saves
front and diagnostic checkpoints with their sidecars under `<run dir>/front/`. `finish` calls
`ctx.front.finalize` (the run-end attention diagnostics of the checkpoints still saved, then the run summary,
`front/summary.json`, also in `run.json` as `front_summary`) for the ends completed, stopif, integrity_stop and
failed; a failure there is recorded (`front_error`) and turns exit 0 into 1, and `run.json` is still written.

W&B (M15): opened after step 1 (the provenance needs the split and order hashes) and before step 2's
seeding. PREFLIGHT runs use only `offline` and `disabled`; this runner refuses `online` until the
launcher batch (B17) lifts it. [B10] Rows are logged with `strict="produced"` (owner decision B10-6): every
section-12 key the run's record carries is required in every row; the keys no module produces yet are listed by
name in `cbl.wandb_log.NOT_PRODUCED`, and the PID's warmup epochs and the run's first evaluation are the two stated
row exemptions (`cbl.wandb_log.ROW_EXEMPT`).

[B10] The launch check. In every stage but `gate`, `prepare` calls `gen.configs.check_launchable` on the config it
was given, which must be the launch copy (`gen.configs.with_handoff`), right after `validate_config` and before
anything else runs: an unset decision input, a missing parameter count, a launch block or a missing handoff field is
refused (exit 2) before the environment is set up, so before any training step. Gate copies (stage `gate`) are the
gates' own and skip it.

[B10] The held-out guard (gate G51). Before it reads any data `prepare` installs `cbl.data.install_heldout_guard`
from the preparation manifest: in the training process every open of the held-out archive, of a raw held-out file or
of a path under the raw held-out directory, by name, resolved name or inode, through Python's open or h5py, raises
`HeldOutAccessRefused` before the file is opened.

[B10] Resume (M14, `cbl/resume.py`). Every run writes resume checkpoints into `<run dir>/resume/` (at train begin and
every `resume_every` epochs, from the appended callback). `prepare(..., resume=True)` resumes the run directory from
its latest checkpoint: the static checks (an unfinished run, the code sha, the config, the interval) before anything
is built, then the same steps as a fresh run, then the run identity, the rewind of the directory, the PID and the
front saver; `fit` reseeds without arming the guard and fits from the checkpoint's epoch, and the appended callback's
`on_train_begin` restores the model, the optimizer, the streams and the map, StopIf and ParetoFront, then arms the
guard. A refused resume is exit 2 with nothing changed on disk.

The PID schedule (M12, `cbl/pid.py`, B9; STUDY section 7.2). A config with `schedule` pid (the PID arms only)
replaces BetaScheduler at its position by `cbl.pid.make_pid(cfg['pid'], headline=cfg['headline'])`, Chang's
BetaPID with the frozen-H sensor: after the evaluator exists, `prepare` binds it to `cbl.pid.sensor_reader`
(the epoch's cached frozen headline count) and the stream guard, and wraps it as lab code. The callback order is
then [FreeEBOPs, LearningRateScheduler, LabBetaPID, PBar, ParetoFront, StopIf, FrozenEvalHook] (PID_ORDER).
Everything else is the ramp recipe; the config carries `pid` (equal to `cbl.pid.PID_RECORD`) and no `ramp`, and
its `recipe` must state the released non-lambda values the runner uses (PID_RECIPE). Released-builder arms other
than CH-8 (PID-CH; CH-8-C and PID-CH-C in S4) load CH-8's file under CH-8's role map with their own arm id.

Config (interfaces.md section 5, schema 1): `validate_config` checks every field the runner uses and
refuses the decision inputs without a default (`headline`, `claim_domain`, `s4_option`) when they are
null in S1, S3 and S4; engineering stages (`gate`, `smoke`, `bench`, `det`) may leave them null, and
CH-8 and F000 train on the native count (owner decision B5-4). A binary arm (B6) needs the headline in
every stage, since it selects the arm's penalty (b) branch (the B5-4 consequence; `cbl.model.arm_spec`
refuses a null one).

Exit codes of `main`: 0 completed or stopped by StopIf, 2 refused before training, 3 integrity stop
(a stream violation, interfaces.md section 10), 1 failed. Nothing here is quotable.

    <venv>/bin/python -I -B cbl/runner.py run --config CFG.json --train-archive A --archive-kind record|smoke \\
        --device cpu|gpu --out DIR --source-bundle HEX [--mem-fraction 0.5] [--resume] [--resume-every N]
    <venv>/bin/python -I -B cbl/runner.py draw --seed S --train-archive A --archive-kind record|smoke \\
        --device cpu|gpu --out DIR --place TEXT
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

FORK = Path(__file__).resolve().parents[1]
if str(FORK) not in sys.path:                      # run as a file
    sys.path.insert(0, str(FORK))

CAMPAIGN = 'chang-bitlinear-20261016'
SCHEMA_VERSION = 1
RUN_KIND = 'cbl1016-run-record'

# run_train.py:79-104, restated (they live inside __main__); G13 checks each against the release
PBAR_FORMAT = ('loss: {loss:.2f}/{val_loss:.2f} - acc: {accuracy:.4f}/{val_accuracy:.4f} - lr: {learning_rate:.2e}'
               ' - beta: {beta:.1e}')                                                  # :79-81
PARETO_METRICS = ['val_accuracy', 'ebops']                                             # :85
PARETO_SIDES = [1, -1]                                                                 # :86
PARETO_FNAME = ('epoch={epoch}-val_acc={val_accuracy:.3f}-train_acc={accuracy:.3f}-ebops={ebops}'
                '-val_loss={val_loss:.3f}.keras')                                      # :87
BETA_POINTS = [(0, 2e-8, 'linear'), (2000, 3e-7, 'log'), (7000, 3.0e-6, 'constant')]   # :90
LEARNING_RATE = 3e-3                                                                   # :68 default
LR_FIRST_DECAY_STEPS = 500                                                             # :93
LR_KWARGS = {'t_mul': 1.0, 'm_mul': 1.0, 'alpha': 1e-6, 'alpha_steps': 10}             # :93
STEPS_PER_EXECUTION = 4                                                                # :100
RELEASE_EPOCHS = 7000                                                                  # :104
TRAIN_BATCH = 2790                                                                     # :67 default, :73
VAL_BATCH = 2790                                                                       # :74
RELEASED_ORDER = ('FreeEBOPs', 'LearningRateScheduler', 'BetaScheduler', 'PBar', 'ParetoFront', 'StopIf')  # :95
APPENDED = 'FrozenEvalHook'
FRONT_ABLATIONS = 'run_end'        # M9 ablation mode: owner decision B8-17 (recommended, flagged for Kai; no default)
FRONT_KEEP_SUPERSEDED = False      # M9: delete a checkpoint file once it leaves every front and role (B8-3)
FRONT_ENDS = ('completed', 'stopif', 'integrity_stop', 'failed')    # ends finish() writes a run summary for
PID_CALLBACK = 'LabBetaPID'        # = cbl.pid.CALLBACK_NAME (tests/test_pid.py); STUDY 7.2: at BetaScheduler's position
PID_ORDER = ('FreeEBOPs', 'LearningRateScheduler', PID_CALLBACK, 'PBar', 'ParetoFront', 'StopIf')


def pareto_enable_if(x):
    return x['val_accuracy'] > 0.5 and x['ebops'] < 5e5                                # :88


def stop_condition(x):
    return x['ebops'] < 1e4                                                            # :91


RAMP_RECORD = {                                    # the config's `ramp` field must equal this
    'lambda': [list(p) for p in BETA_POINTS], 'lr': LEARNING_RATE, 'lr_first_decay_steps': LR_FIRST_DECAY_STEPS,
    't_mul': 1.0, 'm_mul': 1.0, 'lr_floor': 1e-6, 'alpha_steps': 10, 'batch': TRAIN_BATCH,
    'steps_per_execution': STEPS_PER_EXECUTION,
}
# a PID config's `recipe` (interfaces.md section 14) must state these released non-lambda values (run_train.py:67-100)
PID_RECIPE = {
    'batch_size': TRAIN_BATCH, 'val_batch_size': VAL_BATCH, 'steps_per_execution': STEPS_PER_EXECUTION,
    'learning_rate': LEARNING_RATE,
    'lr_schedule': {'function': 'cosine_decay_restarts_schedule', 'first_decay_steps': LR_FIRST_DECAY_STEPS,
                    't_mul': LR_KWARGS['t_mul'], 'm_mul': LR_KWARGS['m_mul'], 'alpha': LR_KWARGS['alpha'],
                    'alpha_steps': LR_KWARGS['alpha_steps']},
    'optimizer': 'Adam', 'optimizer_args': {}, 'loss': {'class': 'SparseCategoricalCrossentropy', 'from_logits': True},
}

ARMS = ('CH-8', 'F000', 'F100', 'F001', 'F010', 'F011', 'F101', 'F110', 'F111', 'PID-CH', 'PID-BEST', 'FP32', 'W8A8',
        'DS-HGQ', 'GR', 'BE', 'CH-8-C', 'BEST-C', 'PID-CH-C', 'PID-BEST-C')
RELEASED_BUILDER_ARMS = ('CH-8', 'CH-8-C', 'PID-CH', 'PID-CH-C')
BUILT_IN_B5 = ('CH-8', 'F000')                      # B5's set; kept because branch b12's G57 reads it
# arms validate_config admits (B6): CH-8, F000, the binary configurations in both forms, GR and BE; GR and
# BE only in engineering stages until the S2 choice record is read (owner decision B6-7)
# B9: the PID arms. PID-BEST and PID-BEST-C train BEST in the chosen form, fixed by S2, so like GR and BE they are
# admitted in engineering stages only until the runner reads the S2 record (B6-7). PID-CH and PID-CH-C, admitted in
# engineering stages only until gates G43-G47 passed, are admitted in every stage since (B9; PENDING_GATES_ARMS empty).
# B10: CH-8-C, the S4 confirmation's CH-8 (STUDY section 5: "CH-8 and BEST in the chosen form, ramp", seeds 4-8 or 4-11),
# is CH-8's arm on the confirmation's fresh seeds: the released builder, the ramp, CH-8's role map under its own arm id
# (B9-9); admitted in every stage (owner decision B10-9).
PID_ARMS = ('PID-CH', 'PID-BEST', 'PID-CH-C', 'PID-BEST-C')
BUILT_ARMS = ('CH-8', 'F000', 'F100', 'F001', 'F010', 'F011', 'F101', 'F110', 'F111', 'GR', 'BE', 'CH-8-C') + PID_ARMS
CHOSEN_FORM_ARMS = ('GR', 'BE', 'PID-BEST', 'PID-BEST-C')
PENDING_GATES_ARMS = ()
CAMPAIGN_STAGES = {'S1': range(1, 4), 'S3': range(1, 4), 'S4': range(4, 12)}
ENGINEERING_STAGES = ('gate', 'smoke', 'bench', 'det')
WANDB_MODES_ALLOWED = ('offline', 'disabled')      # PREFLIGHT; B17 decides online (owner decision B5-7)
WANDB_STRICT = 'produced'                           # B18P: only undefined native_share_removed remains (see NOTES-B18P.md)
RESUME_EVERY = 1                                    # M14 checkpoint interval in epochs (owner decision B10-3)
EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_INTEGRITY = 0, 1, 2, 3


class ConfigRefused(ValueError):
    """The config or an input does not meet what this runner accepts (fail closed)."""


# ----------------------------------------------------------------------------------------------
# hashing and identity (pure)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')


def config_sha256(cfg: dict) -> str:
    """sha256 of the config's canonical JSON (sorted keys, no whitespace); interfaces.md section 3."""
    return sha256_bytes(canonical_json(cfg))


def code_base_sha256(rel: str) -> str:
    for line in (FORK / 'code-base.txt').read_text(encoding='utf-8').splitlines():
        parts = line.split('  ')
        if len(parts) == 4 and parts[0] == 'file' and parts[2] == rel:
            return parts[1]
    raise ConfigRefused(f'code-base.txt has no entry for {rel}')


def release_files() -> dict:
    """Chang's three files the runner imports, each equal to its code-base.txt hash."""
    out = {}
    for rel in ('chang/run_train.py', 'chang/model.py', 'chang/data.py'):
        got, want = sha256_file(FORK / rel), code_base_sha256(rel)
        if got != want:
            raise ConfigRefused(f'{rel} sha256 {got} differs from code-base.txt {want}')
        out[rel] = got
    return out


def code_identity() -> dict:
    """The fork's git HEAD and whether the tree is clean (the runner refuses otherwise)."""
    env = dict(os.environ, LC_ALL='C', GIT_PAGER='cat', GIT_OPTIONAL_LOCKS='0')

    def git(*a):
        r = subprocess.run(['git', '-C', str(FORK)] + list(a), capture_output=True, text=True, env=env)
        return r.returncode, r.stdout
    rc, head = git('rev-parse', 'HEAD')
    rc2, status = git('status', '--porcelain')
    files = {rel: sha256_file(FORK / rel) for rel in ('cbl/runner.py', 'cbl/model.py', 'cbl/binary.py', 'cbl/rng.py',
                                                      'cbl/env.py', 'cbl/data.py', 'cbl/wandb_log.py', 'cbl/cost.py',
                                                      'cbl/cost_pure.py', 'cbl/frozen_eval.py', 'cbl/front.py',
                                                      'cbl/structure.py', 'cbl/ablate.py', 'cbl/select.py',
                                                      'cbl/pid.py', 'cbl/resume.py', 'cbl/epoch_diagnostics.py')}
    return {'head': head.strip() if rc == 0 else None, 'clean': rc2 == 0 and status == '',
            'status': status.splitlines()[:20], 'files_sha256': files}


def _is_hex(s, n=64) -> bool:
    return isinstance(s, str) and len(s) == n and all(c in '0123456789abcdef' for c in s)


# ----------------------------------------------------------------------------------------------
# config (interfaces.md section 5, schema 1)


def validate_config(cfg: dict, *, data_manifest_path=None) -> dict:
    """Refuse (ConfigRefused) anything this runner does not run as stated. Returns a summary of
    the fields it acts on. Pure: no Keras or JAX import."""
    from cbl import rng

    def need(cond, msg):
        if not cond:
            raise ConfigRefused(msg)

    need(isinstance(cfg, dict), 'the config is not a JSON object')
    need(cfg.get('schema_version') == SCHEMA_VERSION, f'schema_version must be {SCHEMA_VERSION}')
    need(cfg.get('campaign') == CAMPAIGN, f'campaign must be {CAMPAIGN}')
    arm = cfg.get('arm_id')
    need(arm in ARMS, f'arm_id {arm!r} is not a STUDY section 5 arm')
    stage = cfg.get('stage')
    need(stage in CAMPAIGN_STAGES or stage in ENGINEERING_STAGES, f'stage {stage!r} unknown')
    engineering = stage in ENGINEERING_STAGES
    seed = cfg.get('seed')
    need(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 1, f'seed {seed!r} is not a positive int')
    if not engineering:
        need(seed in CAMPAIGN_STAGES[stage], f'seed {seed} is outside stage {stage} (STUDY sections 5, 6)')
    schedule = cfg.get('schedule')
    need(schedule in ('ramp', 'pid'), 'schedule must be ramp or pid')
    need((schedule == 'pid') == (arm in PID_ARMS),
         f'arm {arm} with schedule {schedule}: the PID arms and only they use the PID schedule (STUDY sections 5, 7.2)')
    need(arm in BUILT_ARMS, f'arm {arm} is not built yet (built: {", ".join(BUILT_ARMS)}; FP32, W8A8, DS-HGQ and BEST-C '
                            'later)')
    if arm in CHOSEN_FORM_ARMS:
        need(engineering, f'{arm} trains in the form chosen in S2; the runner does not read the S2 choice record yet, so '
                          f'{arm} is admitted in engineering stages only (owner decision B6-7)')
    if arm in PENDING_GATES_ARMS:
        need(engineering, f'{arm} is admitted in engineering stages only until gates G43-G47 pass (B9)')
    builder = cfg.get('builder')
    if arm in RELEASED_BUILDER_ARMS:
        need(builder == 'released', f'{arm} uses the released builder')
        need(cfg.get('config_code') is None and cfg.get('form') is None and cfg.get('control') is None,
             f'{arm}: config_code, form and control are null')
    else:
        need(builder == 'lab', f'{arm} uses the lab builder (M5)')
        from cbl import model as m
        try:
            spec = m.arm_spec(arm, groups=cfg.get('groups'), embedding=cfg.get('embedding'), form=cfg.get('form'),
                              control=cfg.get('control'), headline=cfg.get('headline'))
        except m.SpecError as e:
            raise ConfigRefused(str(e)) from None
        need(cfg.get('config_code') == spec.config_code, f'config_code {cfg.get("config_code")} != {spec.config_code}')
    if schedule == 'ramp':
        need(cfg.get('ramp') == RAMP_RECORD, f'ramp must equal the released recipe {RAMP_RECORD}')
        need(cfg.get('pid') is None, 'a ramp config has no pid field')
    else:
        from cbl import pid
        need(cfg.get('ramp') is None, 'a PID config has no ramp: the PID replaces the lambda ramp (STUDY 7.2)')
        try:
            pid.check_record(cfg.get('pid'))
        except pid.PIDConfigError as e:
            raise ConfigRefused(str(e)) from None
        rc = cfg.get('recipe')
        need(isinstance(rc, dict) and {k: rc.get(k) for k in PID_RECIPE} == PID_RECIPE,
             f'a PID config\'s recipe must state the released non-lambda values {PID_RECIPE}')
        need(cfg.get('headline') in ('H1', 'H2'),
             'a PID run needs the headline H1 or H2 in every stage: its sensor reads it (STUDY 7.2; no default)')
    epochs = cfg.get('epochs')
    need(isinstance(epochs, int) and not isinstance(epochs, bool), 'epochs must be an int')
    if engineering:
        need(1 <= epochs <= RELEASE_EPOCHS, f'engineering stage {stage}: epochs in 1..{RELEASE_EPOCHS}')
    else:
        need(epochs == RELEASE_EPOCHS, f'stage {stage}: epochs must be {RELEASE_EPOCHS} (STUDY section 7.1)')
    need(cfg.get('cost_callbacks') is True, 'cost_callbacks is false only for FP32, which is not built')
    need(cfg.get('matmul_precision') == 'float32', 'matmul_precision must be float32 (STUDY section 7.3)')
    for k in ('headline', 'claim_domain'):
        if cfg.get(k) is None:
            from gen.configs import headline_independent
            need(engineering or (k == 'headline' and headline_independent(cfg)),
                 f'{k} is null: refused in stage {stage} (fail closed; interfaces.md section 5)')
    if cfg.get('headline') is not None:
        need(cfg['headline'] in ('H1', 'H2'), 'headline must be H1 or H2')
    if stage == 'S4':
        need(cfg.get('s4_option') in ('A', 'B', 'C'), 's4_option must be A, B or C in S4')
    it = cfg.get('init_tensors')
    need(isinstance(it, dict) and isinstance(it.get('manifest_path'), str) and _is_hex(it.get('manifest_sha256')),
         'init_tensors needs manifest_path and a 64-hex manifest_sha256 (decision U1; null is refused)')
    need(cfg.get('prefit_reseed') == rng.prefit_reseed(seed), f'prefit_reseed must be {rng.prefit_reseed(seed)} (decision U5)')
    need(_is_hex(cfg.get('data_manifest_sha256')), 'data_manifest_sha256 must be a 64-hex sha256')
    if data_manifest_path is not None:
        got = sha256_file(data_manifest_path)
        need(got == cfg['data_manifest_sha256'], f'data manifest {data_manifest_path} sha256 {got} != config')
    need(_is_hex(cfg.get('code_sha'), 40), 'code_sha must be a 40-hex git commit')
    att = cfg.get('attempt')
    need(isinstance(att, int) and not isinstance(att, bool) and att >= 1, 'attempt must be an int >= 1')
    w = cfg.get('wandb')
    need(isinstance(w, dict) and w.get('mode') in WANDB_MODES_ALLOWED,
         f'wandb.mode must be one of {WANDB_MODES_ALLOWED} (PREFLIGHT; online is refused)')
    return {'arm_id': arm, 'stage': stage, 'engineering': engineering, 'seed': seed, 'epochs': epochs,
            'builder': builder, 'schedule': schedule}


def check_launch(cfg: dict, summary: dict) -> dict:
    """[B10] gen.configs.check_launchable on the launch copy in every stage but `gate` (module docstring); a refusal
    is ConfigRefused (exit 2). Pure: the generator imports no Keras, JAX or hgq."""
    if summary['stage'] == 'gate':
        return {'checked': False, 'reason': 'stage gate: a gate copy, not a launch copy'}
    from gen import configs as gen_configs
    try:
        gen_configs.check_launchable(cfg)
    except gen_configs.UnsetInput as e:
        raise ConfigRefused(f'gen.configs.check_launchable: {e}') from None
    return {'checked': True, 'passed': True}


# ----------------------------------------------------------------------------------------------
# release import and callbacks (Keras imported here)


def import_release(mem_fraction) -> dict:
    """After cbl.env.setup(): import Chang's run_train module (hash-checked), then restore the
    memory fraction its top level overwrote, before the JAX backend starts."""
    from jax._src import xla_bridge

    files = release_files()
    want = os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')
    if xla_bridge.backends_are_initialized():
        raise ConfigRefused('the JAX backend started before run_train was imported; the memory fraction would not apply')
    chang = str((FORK / 'chang').resolve())
    if chang not in sys.path:
        sys.path.insert(0, chang)
    import run_train                                  # noqa: F401  Chang's module (run_train.py:1-37)
    got = os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = want
    if Path(run_train.__file__).resolve() != (FORK / 'chang' / 'run_train.py').resolve():
        raise ConfigRefused(f"module 'run_train' comes from {run_train.__file__}")
    if xla_bridge.backends_are_initialized():
        raise ConfigRefused('importing run_train started the JAX backend')
    return {'files_sha256': files, 'mem_fraction_set_by_run_train': got, 'mem_fraction_restored': want,
            'module': run_train}


def released_callbacks(out_dir, run_train, *, schedule='ramp', pid_record=None, headline=None):
    """run_train.py:79-95 in its construction order; returns the list in the order of :95. With schedule pid the
    lab PID (cbl/pid.py) takes BetaScheduler's place and position (STUDY 7.2)."""
    from hgq.utils.sugar import BetaScheduler, FreeEBOPs, ParetoFront, PBar, PieceWiseSchedule, StopIf
    from keras.callbacks import LearningRateScheduler

    pbar = PBar(PBAR_FORMAT)                                                           # :79-81
    ebops = FreeEBOPs()                                                                # :82
    pareto = ParetoFront(str(out_dir), PARETO_METRICS, PARETO_SIDES, fname_format=PARETO_FNAME,
                         enable_if=pareto_enable_if)                                   # :83-89
    if schedule == 'pid':
        from cbl import pid
        beta_sched = pid.make_pid(pid_record, headline=headline)                       # STUDY 7.2, in :90's place
    else:
        beta_sched = BetaScheduler(PieceWiseSchedule(BETA_POINTS))                     # :90
    stop = StopIf(stop_condition)                                                      # :91
    lr_sched = LearningRateScheduler(run_train.cosine_decay_restarts_schedule(
        LEARNING_RATE, LR_FIRST_DECAY_STEPS, **LR_KWARGS))                             # :92-94
    return [ebops, lr_sched, beta_sched, pbar, pareto, stop]                           # :95


def compile_release(model):
    """run_train.py:97-100."""
    import keras

    opt = keras.optimizers.Adam()                                                      # :97
    metrics = ['accuracy']                                                             # :98
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)                # :99
    model.compile(optimizer=opt, loss=loss, metrics=metrics, steps_per_execution=STEPS_PER_EXECUTION)  # :100
    return {'optimizer': type(opt).__name__, 'optimizer_config': opt.get_config(), 'loss': loss.get_config(),
            'metrics': metrics, 'steps_per_execution': STEPS_PER_EXECUTION}


def callback_signature(callbacks, epochs=(0, 1, 2, 9, 10, 11, 499, 500, 501, 1999, 2000, 2001, 4500, 6999, 7000, 7001)):
    """What G13 compares between the runner's callbacks and the release's live objects: types in
    order and, per callback, its literal arguments evaluated on fixed inputs (float.hex)."""
    import numpy as np

    probes = [{'val_accuracy': a, 'ebops': e} for a in (0.4, 0.5, 0.5000001, 0.9) for e in (9999, 10000, 499999, 500000, 1e6)]
    out = {'order': [type(c).__name__ for c in callbacks]}
    for c in callbacks:
        n = type(c).__name__
        if n == 'PBar':
            out['PBar'] = {'template': c.template, 'disable_ebops': c.disable_ebops}
        elif n == 'ParetoFront':
            out['ParetoFront'] = {'metrics': list(c.metrics), 'sides': np.asarray(c.sides).tolist(), 'fname_format': c.fname_format,
                                  'enable_if': [bool(c.enable_if(p)) for p in probes],
                                  'save_weights': getattr(c, '_save_weights', None)}
        elif n == 'BetaScheduler':
            out['BetaScheduler'] = {'beta': [float(c.beta_fn(e)).hex() for e in epochs]}
        elif n == PID_CALLBACK:
            out[PID_CALLBACK] = c.signature()
        elif n == 'LearningRateScheduler':
            out['LearningRateScheduler'] = {'lr': [float(c.schedule(e)).hex() for e in epochs], 'verbose': c.verbose}
        elif n == 'StopIf':
            out['StopIf'] = {'condition': [bool(c.stop_condition(p, None)) for p in probes],
                             'patience_epochs': c.patience_epochs, 'patience_steps': c.patience_steps,
                             'verbose': c.verbose}
    return out


# ----------------------------------------------------------------------------------------------
# the appended frozen-evaluation callback (M8, batch B7)


def _logval(v):
    """A log value as an exact, comparable token: strings as they are, numbers as float.hex."""
    return v if isinstance(v, str) else float(v).hex()


def frozen_eval_hook(recorder, *, evaluator=None, probe=None, guard=None, wandb_run=None, epochs_path=None, run_id=None,
                     train_begin=None, after_epoch=None, enrich_row=None):
    """The callback appended after StopIf (STUDY section 7.1): M8's (`cbl.frozen_eval.make_hook`).
    `evaluator` is a `cbl.frozen_eval.FrozenEvaluator`; `probe` a gate's read-only callable(epoch, logs),
    run after it; `train_begin` and `after_epoch` are M14's (B10). See the module docstring."""
    from cbl import frozen_eval

    return frozen_eval.make_hook(recorder, evaluator=evaluator, probe=probe, guard=guard, wandb_run=wandb_run,
                                 epochs_path=epochs_path, run_id=run_id, train_begin=train_begin,
                                 after_epoch=after_epoch, enrich_row=enrich_row)


# ----------------------------------------------------------------------------------------------
# the run


class Context:
    """Everything `prepare` builds and `fit` uses; gates read it (read-only)."""

    def __init__(self):
        self.record = {}
        self.cfg = self.summary = self.flow = self.model = self.roles = None
        self.callbacks = self.released = self.hook = self.guard = self.run_train = self.wandb_run = None
        self.evaluator = self.probe = None
        self.front = self.frozen_step = None
        self.pid = None
        self.out_dir = None
        self.resume = None


def _dataset_device(device):
    if device not in ('cpu', 'gpu'):
        raise ConfigRefused(f'device must be cpu or gpu, not {device!r}')
    return ('cpu:0', 'cpu') if device == 'cpu' else ('gpu:0', 'cuda')


# Binary-site fields added by prepare's enrich_row, after the frozen core evaluation.
# Reload audits use this same set; enrichment must never overwrite a core field.
BINARY_ENRICHMENT_FIELDS = frozenset({
    'moment_mismatch_mean', 'moment_mismatch_var', 'dead_channel_frac', 'fb_sign_diff_frac',
})


def prepare(cfg, *, train_archive, archive_kind, device, out_dir, source_bundle, mem_fraction='0.5',
            data_manifest_path=None, evaluator_factory=None, keep_full=4, ctx=None, resume=False,
            resume_every=None, pack_k=1) -> Context:
    """Steps 1-4 of the module docstring, up to and including compile. Creates out_dir (refusing an
    existing one) and writes only W&B's offline directory, the ParetoFront directory and the front saver's
    directory (`front/`, M9) into it. [B10] With resume=True, out_dir must hold a resume checkpoint (M14); it is
    resumed in place (module docstring). `resume_every` is the checkpoint interval in epochs (None means
    RESUME_EVERY; a resume must name the interval the run was started with).
    `ctx` is filled as the steps run, so that a refusal keeps what was recorded before it.
    `evaluator_factory(ctx)`, for gates only, returns a read-only probe callable(epoch, logs) that the
    appended callback runs after M8's frozen evaluation."""
    from cbl import env
    from cbl import resume as resume_mod

    ctx = ctx if ctx is not None else Context()
    rec = ctx.record
    rec.update(kind=RUN_KIND, schema=1, label='engineering, not quotable', started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    dataset_device, platforms = _dataset_device(device)
    data_manifest_path = Path(data_manifest_path or FORK / 'manifests' / 'data-prep-150c.json')
    summary = validate_config(cfg, data_manifest_path=data_manifest_path)
    if type(pack_k) is not int or pack_k < 1:
        raise ConfigRefused('pack_k must be a positive integer')
    ctx.cfg, ctx.summary = cfg, summary
    rec.update(config=cfg, config_sha256=config_sha256(cfg), summary=summary, device=device,
               dataset_device=dataset_device)
    rec['launch_check'] = check_launch(cfg, summary)                                   # B10, before anything runs
    ident = code_identity()
    rec['code'] = ident
    if ident['head'] != cfg['code_sha']:
        raise ConfigRefused(f'config code_sha {cfg["code_sha"]} differs from the fork HEAD {ident["head"]}')
    if not ident['clean']:
        raise ConfigRefused(f'the fork tree is not clean: {ident["status"][:5]}')
    resume_every = RESUME_EVERY if resume_every is None else resume_every
    if not isinstance(resume_every, int) or isinstance(resume_every, bool) or resume_every < 1:
        raise ConfigRefused(f'resume_every must be an int >= 1, not {resume_every!r}')
    rec['resume_every'] = resume_every
    state = ckpt = None
    rec['resume_requested'] = bool(resume)
    if resume:                                                                         # M14 static checks (B10)
        if not Path(out_dir).is_dir():
            raise ConfigRefused(f'resume: {out_dir} is not a directory')
        try:
            state, ckpt = resume_mod.read_latest(out_dir)
            from cbl import wandb_log as wl
            resume_mod.check_static(state, cfg=cfg, config_sha256=rec['config_sha256'], head=ident['head'],
                                    run_dir=out_dir, every=resume_every,
                                    run_name=wl.name_from_config(cfg, source_bundle))
        except resume_mod.ResumeRefused as e:
            raise ConfigRefused(f'resume refused: {e}') from None
        rec['resume_from'] = {'checkpoint': ckpt.name, 'epochs_done': state['epochs_done']}
    dman = json.loads(data_manifest_path.read_text(encoding='utf-8'))
    if archive_kind not in ('record', 'smoke'):
        raise ConfigRefused('archive_kind must be record or smoke')
    if archive_kind == 'smoke' and not summary['engineering']:
        raise ConfigRefused(f'the smoke archive is for engineering stages only, not {summary["stage"]}')
    expect_archive = dman['train' if archive_kind == 'record' else 'smoke']['archive']['sha256']
    rec['archive'] = {'path': str(train_archive), 'kind': archive_kind, 'expect_sha256': expect_archive}
    out_dir = Path(out_dir)
    if not resume:
        launch = None
        # M20 writes its immutable launch identity before any backend/data initialization.
        launch_raw = os.environ.get('CBL_LAUNCH_RECORD')
        if launch_raw:
            launch = json.loads(launch_raw)
            from cbl.wandb_log import name_from_config
            if (launch.get('config') != cfg or launch.get('code_sha') != cfg['code_sha'] or
                    launch.get('resume_every') != resume_every or launch.get('source_bundle') != source_bundle or
                    launch.get('run_name') != name_from_config(cfg, source_bundle)):
                raise ConfigRefused('pack launch record differs from runner inputs')
        resume_mod.create_run_directory(out_dir, launch)
    ctx.out_dir = out_dir
    from cbl import data as cbl_data                                                   # G51 runtime guard (B10)
    rec['heldout_guard'] = cbl_data.install_heldout_guard(dman)

    # 1. environment and the release (M1)
    rec['env'] = env.setup(mem_fraction=mem_fraction, jax_platforms=platforms)
    rel = import_release(rec['env']['process_env']['set']['XLA_PYTHON_CLIENT_MEM_FRACTION'])
    ctx.run_train = rel.pop('module')
    rec['release'] = rel
    import keras
    import model as chang_model  # noqa: E402  Chang's module, imported by env.setup()

    from cbl import model as lab_model
    from cbl import rng

    # 2. data, then W&B, then seeds, Datasets and the model (M4)
    flow = rng.SeedFlow(cfg['seed'], dataset_device=dataset_device)
    ctx.flow = flow
    flow.load_data(train_archive, expect_sha256=expect_archive, data_manifest_sha256=cfg['data_manifest_sha256'])
    prov = env.provenance()
    rec['provenance'] = prov
    wprov = {'code_sha': cfg['code_sha'], 'config_sha256': rec['config_sha256'],
             'data_manifest_sha256': cfg['data_manifest_sha256'], 'split_sha256': flow.data_record['split_sha256'],
             'order_sha256': flow.data_record['order_sha256'], 'backend': prov['jax_backend'],
             'matmul_precision': prov['jax_default_matmul_precision'],
             'gpu_product': ','.join(sorted({str(k) for k in prov['device_kind']})) if device == 'gpu' else 'none',
             'pack_k': pack_k}
    from cbl import wandb_log
    ctx.wandb_run = wandb_log.Run.open(cfg, wprov, source_bundle=source_bundle, run_dir=out_dir / 'wandb',
                                       mode=cfg['wandb']['mode'], strict=WANDB_STRICT,
                                       expect_run_id=None if state is None else state['run']['run_id'])
    rec['wandb'] = {'mode': ctx.wandb_run.mode, 'run_name': ctx.wandb_run.name, 'run_id': ctx.wandb_run.run_id,
                    'strict': WANDB_STRICT, 'provenance': wprov}
    flow.seed_streams()
    ds_train, ds_val = flow.build_datasets(batch_size=TRAIN_BATCH, val_batch_size=VAL_BATCH)
    flow.recorder.keep_full = keep_full
    print(len(ds_train), flush=True)                                                   # run_train.py:102
    if summary['builder'] == 'released':
        build = lambda: chang_model.get_model(*rng.CH8_BUILDER_ARGS)                   # noqa: E731  run_train.py:76
        # CH-8's role map; PID-CH, CH-8-C and PID-CH-C carry their own arm id (the loader checks it)
        spec_roles = (rng.CH8_ROLE_SPEC if cfg['arm_id'] == 'CH-8'
                      else dataclasses.replace(rng.CH8_ROLE_SPEC, arm_id=cfg['arm_id']))
    else:
        spec = lab_model.arm_spec(cfg['arm_id'], groups=cfg['groups'], embedding=cfg['embedding'], form=cfg['form'],
                                  control=cfg['control'], headline=cfg.get('headline'))
        build = lab_model.builder(spec)
        spec_roles = lab_model.role_spec(spec)
    model = flow.build_model(build, arm_id=cfg['arm_id'])
    ctx.model = model
    from cbl import lossguard                                                          # B10-F1: no penalty term
    rec['lossguard'] = lossguard.guard(model)                                          # dropped by a stale loss id

    # 3. initial tensors (U1)
    it = cfg['init_tensors']
    rec['load'] = flow.load_initial_tensors(it['manifest_path'], expect_manifest_sha256=it['manifest_sha256'],
                                            role_spec=spec_roles, check_own_draw=True)
    ctx.roles = rng.roles_of(model, spec_roles)

    # 4. callbacks and compile (run_train.py:79-100), the appended callback, the guard
    pid_run = summary['schedule'] == 'pid'
    released = released_callbacks(out_dir / 'pareto', ctx.run_train, schedule=summary['schedule'],
                                  pid_record=cfg.get('pid'), headline=cfg.get('headline'))
    ctx.released = released
    rec['compile'] = compile_release(model)
    guard = rng.StreamGuard(raise_on_violation=True)
    ctx.guard = guard
    from cbl import frozen_eval                                                        # M8
    ctx.evaluator = frozen_eval.FrozenEvaluator(model, roles=ctx.roles, top_names=spec_roles.top_names(model),
                                                x_val=flow.ds_val.x, y_val=flow.ds_val.y, batch_size=VAL_BATCH)
    rec['frozen_evaluation'] = {'module': 'cbl/frozen_eval.py', 'val_batch_size': VAL_BATCH,
                                'n_val_batches': len(ctx.evaluator.bounds), 'n_val': ctx.evaluator.n_val,
                                'counted_layers': len(ctx.evaluator.entries)}
    if pid_run:                                                                        # M12 (B9)
        from cbl import pid
        ctx.pid = released[2]
        ctx.pid.bind(sensor=pid.sensor_reader(ctx.evaluator, cfg['headline']), guard=guard)
        rec['pid'] = ctx.pid.signature()
    ctx.probe = evaluator_factory(ctx) if evaluator_factory is not None else None
    from cbl import front                                                              # M9 (wired in B9)
    ctx.front = front.register(ctx, ablations=FRONT_ABLATIONS, keep_superseded=FRONT_KEEP_SUPERSEDED)
    rec['front'] = {'module': 'cbl/front.py', 'dir': 'front', 'ablations': FRONT_ABLATIONS,
                    'keep_superseded': FRONT_KEEP_SUPERSEDED}
    run_id = ctx.wandb_run.name
    evaluator = ctx.evaluator

    from cbl.epoch_diagnostics import EpochDiagnostics
    diagnostics = EpochDiagnostics(model, evaluator.entries, spec_roles.top_names(model),
                                   flow.ds_val.x, flow.ds_val.y, VAL_BATCH)
    diagnostics.install_train_observer()
    extras = {}
    allocation_started = time.monotonic()
    allocation_path = os.environ.get('CBL_ALLOCATION_FILE')
    if device == 'gpu' and pack_k > 1 and not allocation_path:
        raise ConfigRefused('packed GPU runs require the M20 live allocation record')

    def enrich_row(row, logs):
        epoch = row['epoch']
        if epoch not in extras:
            h0 = rng.variables_sha256(ctx.roles)
            with guard.watch('epoch diagnostics'):
                extra = diagnostics.evaluate(row['val_top1'], logs)
            if rng.variables_sha256(ctx.roles) != h0:
                raise frozen_eval.FrozenStateChanged('epoch diagnostics changed model variables')
            extra['dwell'] = ctx.front.state.dwell(row)
            extras.clear()
            extras[epoch] = extra
        extra = extras[epoch]
        for name, fields in extra['binary_sites'].items():
            assert fields.keys() <= BINARY_ENRICHMENT_FIELDS, (
                f'undeclared binary enrichment fields at {name}: {sorted(fields.keys() - BINARY_ENRICHMENT_FIELDS)}')
        row = {**row, **{k: v for k, v in extra.items() if k != 'binary_sites'}}
        row['binary_sites'] = {n: {**v, **extra['binary_sites'].get(n, {})}
                               for n, v in (row.get('binary_sites') or {}).items()} or None
        if allocation_path:
            from nrp.accounting import hours
            row['sys/alloc_hours'] = hours(allocation_path)
        else:
            # Standalone GPU process: exclusive allocation since prepare; CPU gate has zero GPU hours.
            row['sys/alloc_hours'] = (time.monotonic() - allocation_started) / 3600 if device == 'gpu' else 0.0
        return row

    def section1(epoch, logs):
        row = frozen_eval.section1_record(epoch, logs, evaluator=evaluator, model=model, run_id=run_id)
        return enrich_row(row, logs)
    ctx.frozen_step = after_evaluation(front.chain_evaluator(section1, ctx.front), ctx.probe)

    def train_begin(h):                                                                # M14 (B10)
        ctx.resume.train_begin(h)

    def after_epoch(epoch, logs, row):
        ctx.resume.after_epoch(epoch, logs, row)
    hook = frozen_eval_hook(flow.recorder, evaluator=ctx.evaluator, probe=ctx.frozen_step, guard=guard,
                            wandb_run=ctx.wandb_run, epochs_path=out_dir / 'epochs.jsonl', run_id=run_id,
                            train_begin=train_begin, after_epoch=after_epoch, enrich_row=enrich_row)
    ctx.hook = hook
    for cb in released:
        guard.wrap(cb, owner='lab' if cb is ctx.pid else 'released')
    guard.wrap(hook, owner='lab', name=APPENDED)
    ctx.callbacks = released + [hook]
    rec['callback_order'] = [type(c).__name__ for c in ctx.callbacks]
    order = PID_ORDER if pid_run else RELEASED_ORDER
    if tuple(rec['callback_order'][:6]) != order or rec['callback_order'][6:] != [APPENDED]:
        raise ConfigRefused(f'callback order {rec["callback_order"]}')
    rec['callback_signature'] = callback_signature(ctx.callbacks)
    rec['matmul_precision_before_fit'] = env.assert_matmul_precision('before fit')
    rec['keras_version'] = keras.__version__
    try:                                                                               # M14 (B10), last
        ctx.resume = resume_mod.Resume(ctx, every=resume_every, state=state, checkpoint_dir=ckpt)
    except resume_mod.ResumeRefused as e:
        raise ConfigRefused(f'resume refused: {e}') from None
    rec['resume'] = ctx.resume.record
    return ctx


def fit(ctx: Context, *, epochs=None) -> int:
    """Step 5: the pre-fit reseed with the guard armed, then fit. Returns the exit code."""
    import keras
    from cbl import env, rng

    rec, cfg = ctx.record, ctx.cfg
    epochs = cfg['epochs'] if epochs is None else epochs
    if epochs != cfg['epochs']:
        raise ConfigRefused('fit epochs must be the config epochs')
    resumed = ctx.resume is not None and ctx.resume.resumed
    # a resumed run reseeds as a fresh one does (the flow's record), but the guard is armed only after M14 restored the
    # streams in the appended callback's on_train_begin (STUDY section 6 step 8; cbl/resume.py)
    rec['reseed'] = ctx.flow.reseed_before_fit(expect_prefit_reseed=cfg['prefit_reseed'],
                                               guard=None if resumed else ctx.guard)
    rec['index_map_at_reseed'] = ctx.flow.index_map_at_reseed
    initial_epoch = ctx.resume.epochs_done if resumed else 0
    stopped = resumed and bool(ctx.resume.state['stop_training'])
    rec['initial_epoch'] = initial_epoch
    t0 = time.time()
    code = EXIT_OK
    try:
        # a checkpoint written after StopIf stopped the run ends it: fit then runs no epoch (it still restores the state)
        ctx.model.fit(ctx.flow.ds_train, epochs=initial_epoch if stopped else epochs, validation_data=ctx.flow.ds_val,
                      callbacks=ctx.callbacks, verbose=0, initial_epoch=initial_epoch)  # run_train.py:104
        rec['end'] = 'stopif' if (ctx.model.stop_training or stopped) else 'completed'
    except rng.StreamViolation as e:
        rec['end'], rec['violation'] = 'integrity_stop', str(e)
        code = EXIT_INTEGRITY
    finally:
        ctx.guard.disarm()
        rec['fit_seconds'] = time.time() - t0
    rec['epochs_completed'] = len(ctx.hook.records)
    rec['epochs_requested'] = epochs
    rec['epoch_records'] = ctx.hook.records
    rec['matmul_precision_after_fit'] = env.log_matmul_precision('after fit')
    rec['guard'] = ctx.guard.record()
    rec['recorder'] = ctx.flow.recorder.summary()
    rec['flow'] = ctx.flow.record()
    rec['final_variables_sha256'] = rng.variables_sha256(ctx.roles)
    rec['keras_version'] = keras.__version__
    return code


def after_evaluation(saver_step, probe):
    """The read-only step the appended callback runs after M8's evaluation, inside its frozen-evaluation watch:
    the front saver on the epoch's section 1 record, then a gate's probe (if any)."""
    def step(epoch, logs):
        saver_step(epoch, logs)
        if probe is not None:
            probe(epoch, logs)
    return step


def finish(ctx: Context, code: int) -> int:
    """Writes the run summary through M9 (for the ends in FRONT_ENDS), then run.json; returns the exit code."""
    rec = ctx.record
    if ctx.front is not None and rec.get('end') in FRONT_ENDS:
        from cbl import front
        epochs = rec.get('epochs_completed', len(ctx.hook.records) if ctx.hook is not None else 0)
        try:
            rec['front_summary'] = ctx.front.finalize(
                end=rec['end'], epochs_completed=epochs,
                released_front=front.released_front_of(ctx.released[4], ctx.out_dir))   # [4]: ParetoFront, run_train.py:95
        except Exception as e:  # noqa: BLE001 - recorded; the run then exits 1
            rec['front_summary'] = None
            rec['front_error'] = f'{type(e).__name__}: {e}'
            rec['front_traceback'] = traceback.format_exc()
            if code == EXIT_OK:
                code = EXIT_FAILED
    if ctx.model is not None:                                                          # B10-F1
        from cbl import lossguard
        rec['lossguard_counters'] = lossguard.counters(ctx.model)
    rec['exit'] = code
    rec['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    if ctx.resume is not None:
        rec['resume'] = ctx.resume.summary()
    if ctx.wandb_run is not None:
        ctx.wandb_run.summary({'end': rec.get('end', 'failed'), 'epochs_completed': rec.get('epochs_completed', 0)})
        ctx.wandb_run.finish(exit_code=code)
    if ctx.out_dir is not None:
        text = json.dumps(rec, indent=1, sort_keys=True, default=str) + '\n'
        if rec.get('resume_requested') and 'initial_epoch' not in rec:
            # a resume refused or failed before fit started leaves the run resumable: its record goes beside the
            # checkpoints, never to run.json (which marks a finished run)
            rd = ctx.out_dir / 'resume' / 'attempts'
            rd.mkdir(parents=True, exist_ok=True)
            (rd / f'{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}-{os.getpid()}.json').write_text(text, encoding='utf-8')
        else:
            (ctx.out_dir / 'run.json').write_text(text, encoding='utf-8')
    return code


def run(cfg, **kw) -> tuple:
    """prepare, fit, finish. Returns (exit code, context)."""
    ctx = Context()
    try:
        prepare(cfg, ctx=ctx, **kw)
        code = fit(ctx)
    except ConfigRefused as e:
        ctx.record.update(end='refused', refused=str(e))
        code = EXIT_REFUSED
    except Exception as e:  # noqa: BLE001 - recorded, then failed
        ctx.record.update(end='failed', error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc())
        code = EXIT_FAILED
    return finish(ctx, code), ctx


def draw(seed, *, train_archive, archive_kind, device, out_dir, place, mem_fraction='0.5', data_manifest_path=None):
    """CH-8's initial-tensor file of a seed (decision U1): section 6 steps 1-3 with the released
    builder on this device, then M4's draw_initial_tensors. Returns (manifest path, sha256)."""
    from cbl import env

    dataset_device, platforms = _dataset_device(device)
    data_manifest_path = Path(data_manifest_path or FORK / 'manifests' / 'data-prep-150c.json')
    dman_sha = sha256_file(data_manifest_path)
    dman = json.loads(data_manifest_path.read_text(encoding='utf-8'))
    expect_archive = dman['train' if archive_kind == 'record' else 'smoke']['archive']['sha256']
    ident = code_identity()
    env.setup(mem_fraction=mem_fraction, jax_platforms=platforms)
    import model as chang_model  # noqa: E402

    from cbl import rng
    flow = rng.SeedFlow(seed, dataset_device=dataset_device)
    flow.load_data(train_archive, expect_sha256=expect_archive, data_manifest_sha256=dman_sha)
    flow.seed_streams()
    flow.build_datasets(batch_size=TRAIN_BATCH, val_batch_size=VAL_BATCH)
    flow.build_model(lambda: chang_model.get_model(*rng.CH8_BUILDER_ARGS), arm_id='CH-8')
    mpath, msha, man = rng.draw_initial_tensors(flow, out_dir, place=place, code_sha=ident['head'])
    return mpath, msha, man, flow


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--config', required=True)
    d = sub.add_parser('draw')
    d.add_argument('--seed', type=int, required=True)
    d.add_argument('--place', required=True)
    for p in (r, d):
        p.add_argument('--train-archive', required=True)
        p.add_argument('--archive-kind', choices=['record', 'smoke'], required=True)
        p.add_argument('--device', choices=['cpu', 'gpu'], required=True)
        p.add_argument('--out', required=True)
        p.add_argument('--mem-fraction', default='0.5')
    r.add_argument('--source-bundle', required=True, help='sha256 of the bundle the run was launched from (M15)')
    r.add_argument('--resume', action='store_true', help='resume --out from its latest checkpoint (M14); refused if none')
    r.add_argument('--pack-k', type=int, default=1, help='actual concurrent pack size, provenance only')
    r.add_argument('--resume-every', type=int, default=RESUME_EVERY,
                   help='checkpoint interval in epochs (M14); a resume must name the interval the run started with')
    a = ap.parse_args(argv)
    if a.cmd == 'draw':
        mpath, msha, _, _ = draw(a.seed, train_archive=a.train_archive, archive_kind=a.archive_kind, device=a.device,
                                 out_dir=a.out, place=a.place, mem_fraction=a.mem_fraction)
        print(json.dumps({'manifest': str(mpath), 'manifest_sha256': msha}))
        return EXIT_OK
    cfg = json.loads(Path(a.config).read_text(encoding='utf-8'))
    code, ctx = run(cfg, train_archive=a.train_archive, archive_kind=a.archive_kind, device=a.device, out_dir=a.out,
                    source_bundle=a.source_bundle, mem_fraction=a.mem_fraction, resume=a.resume,
                    resume_every=a.resume_every, pack_k=a.pack_k)
    if code != EXIT_OK:
        print(f'runner exit {code}: {ctx.record.get("refused") or ctx.record.get("error") or ctx.record.get("violation")}',
              file=sys.stderr)
    return code


if __name__ == '__main__':
    sys.exit(main())
