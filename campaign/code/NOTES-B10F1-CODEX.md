# B10-F1 completion (development evidence; not gates of record)

Base: branch `b10f1`, HEAD `1ed37917d8bdd4f704a8f1dae0700f4f1a83b042`.
Implemented and tested; no commits created. B10-F1 census admission passed on all 109 configs.
The reduced G49 overall verdict remains FAIL for 57 B18 binary-site audit-schema comparisons; one deferred
PC pack test remains skipped because its required plan conflicts with the allowed execution scope.

Evidence directory:
`/Users/kaiyamaguchi/lab/bnjettag/campaigns/2026-10-16-chang-bitlinear/preflight/investigations/B10-F1/codex/`.
PC execution is isolated in `/home/kaimoe/lab/scratch/cbl1016/b10f1-codex/`.

## Changes and call-site audit

- `gates/binary_child.py`: guard immediately after both lab and released model construction.
- `gates/cost_child.py`: its `build` already delegates both branches to those binary-child builders; document
  that shared protection. Avoid a redundant second installation.
- `gates/b8_child.py`: guard both branches of its builder (training steps plus later diagnostics).
- `gates/runner_child.py`: guard on entry to `Probe`, before creating the jitted gradient function. This also
  protects the reference release's model when the existing fit interception constructs its Probe. Runner
  models are already guarded; installation is idempotent.
- `gates/rng_child.py`: guard the result of `SeedFlow.build_model` before returning its flow to fit callers.
- `gates/g03_precision.py`: guard after build; fit and subsequent training-gradient lowerings use that model.
- `tests/b18_pc_case.py`: guard both real-stack model branches before fit and direct train-step comparisons.
  Correct the previously unrun fixture from one-hot targets to integer class indices, matching the release
  SparseCategoricalCrossentropy. The initial four PC failures were target-shape errors before training.
- `tests/test_rng.py`: guard the tiny Dense model before the multi-epoch/partial-batch fit.
- `tests/test_lossguard.py`: seven PC regression cases exercise binary, cost and B8 lab/released builders and
  the Probe on real models, injecting deterministic stale loss IDs across two batch shapes. Require the guard
  to be installed, positive expected term count, two successful census checks, and stale IDs actually cleared.

Searched all of `gates/` and `tests/` for fit, train_step, train_on_batch, stateless_call, model construction
and reload sites. Additional sites inspected but unchanged:

- `runner_child.cmd_mha`: one training-mode stateless call per newly constructed model; wrapping negative
  controls deliberately test malformed layer wrappers and are not retrained models.
- `binary_child.train_grads`: standalone-layer output gradients, no collected loss terms in its objective.
- `config_child.build`: inference/save/reload only; prep delegates to the guarded runner.
- `reload_child`, `resume_child`: training delegates to the guarded runner; separate reloaded models perform
  inference/evaluate only. Resume callback replay constructs models but drives callbacks rather than fitting.
- `pid_child`: runner paths already guarded; replay drives PID callbacks, and the wrapper-model control reads
  cost aggregates rather than tracing training.
- Existing unguarded tiny model in `test_lossguard.py` is the intentional failing control for address reuse.

## Decisions

DECISION: Protect concrete gate model instances at construction, and at Probe ownership entry for the
intercepted released fit, using the existing `cbl.lossguard.guard` with its census enabled.
ALTERNATIVES: Global Keras class patch (broader runtime change); guard only the ticket's named files (misses
B8, RNG, precision, and test helper traces); duplicate cost-child guard (unnecessary with shared builders).
CONFIDENCE: HIGH for the guarded child builders: all seven deterministic collision regressions passed on
the real PC stack, as did the three original mechanism tests. The runner's guard implementation is unchanged.
FLAG FOR HUMAN: Existing ticket B10F1-1 still requires the owner STUDY section 7.3 decision/amendment; this task
cannot approve it or turn development checks into gates of record.

DECISION: Run reduced G49 through `gates/g49_reload.py` using an external, recorded development harness.
It verifies the exact bundle plus uncommitted patch in every process, uses the unchanged canonical training
reader once, then supplies the first 65 training and 33 validation rows after the canonical split and
normalization. Batch size 32 forces full and partial training traces. No config limit; one CPU epoch per
admitted config, plus the gate's fresh-process checkpoint reload checks. Fresh processes share a scratch-only
persistent JAX compilation cache to avoid recompiling identical programs; the Python training traces and
loss census still execute in every process, verified by the per-child counters. The harness asserts actual
`validate_config` admission agrees with the gate's predicate for every generated config and records all verdicts.
ALTERNATIVES: Full archive epochs (unnecessarily expensive for census admission); duplicate the G49 loop
(would not exercise the requested script); commit a development snapshot (explicitly forbidden).
CONFIDENCE: HIGH for census admission: all 109 configurations completed one CPU epoch, each checked both
training traces (218 census checks total; 89 expected loss terms per training trace). This is deliberately
not the full G49 validation population or a gate-of-record run.
FLAG FOR HUMAN: The harness explicitly bypasses only clean-tree admission while retaining actual dirty status
and bundle/patch hashes. It labels output `G49-DEV-B10F1`; no clean-tree or gate-of-record PASS is claimed.
The original reader's archive identity and normalization metadata are retained alongside derived subset hashes.

DECISION: Correct the deferred B18 producer fixture to sparse integer labels, matching `runner.compile_release`.
ALTERNATIVES: Change the loss to categorical crossentropy (would stop testing the actual release recipe);
leave all four cases failing before training (would not exercise diagnostics or the new guard).
CONFIDENCE: HIGH; initial failures reported target/output shapes `(8, 5)`, and all four arms passed after
changing only labels from one-hot rows to integer class IDs.
FLAG FOR HUMAN: Fixture repair only; no production objective or runner behavior was changed.

## Validation

- Offline command: `/Users/kaiyamaguchi/lab/venvs/mhab1012-cpu/bin/python -m pytest tests -q -p no:cacheprovider -m "not pc_gpu"`.
  Exit 0: **602 passed, 4 skipped, 26 deselected**, one expected invalid-divide warning from reader refusal test;
  10.49 s pytest time. `offline.txt` and `offline-command.json` contain output and execution metadata.
- Initial PC run: exit 1, **21 passed, 4 failed, 1 skipped, 606 deselected** in 102.28 s. All seven new
  guard cases and 14 of the original 19 cases passed. Four producer cases failed on the pre-existing one-hot
  target fixture; the opt-in pack test skipped. Full output preserved as `pc-tests-initial.txt`.
- Corrected producer fixture retest: exit 0, **4 passed**, 160.24 s (`b18-retest.txt`). Thus **18 of the
  original 19 deferred cases passed** across the initial run and targeted retest; the pack case is skipped.
- Original lossguard mechanism tests on PC: exit 0, **3 passed, 7 deselected**, 1.05 s.
  `lossguard-mechanism.txt` and command JSON contain the output and exact invocation.
- PC interpreter: `/home/kaimoe/lab/venvs/cbl1016-jax2/bin/python`, Python 3.12.13.
  Packages: jax 0.9.2, jaxlib 0.9.2, keras 3.14.0, hgq2 0.1.9, numpy 2.5.4.
  G49 verified all 66 installed packages exactly match `env/training/freeze.txt`.
- Training venv has no pytest. Installed pytest 8.4.2 and its dependencies into scratch
  `pytest-site/` via `uv pip install --target`, with the uv cache also under scratch; did not modify the venv.
- Reduced G49: exit **1**, overall **FAIL (57 failed checks)**, elapsed 2539.22 s. All **167 generated configs**
  were checked against `validate_config`; **109 admitted**, all **109 trained and reloaded**. Every training
  child exited 0 and completed one epoch. **218/218 census checks passed**, exactly two per config, with **89
  expected training terms** in each. All 109 validation-logit arrays were bitwise equal after reload; maximum
  absolute difference in logits, validation loss and validation accuracy was **0.0**. All 109 gate comparison
  lines reported **0 forbidden events**. There were **52 full reload audit passes** and **57 binary-site schema
  mismatches**, described below. `g49-summary.json`, `g49-development.json/.txt`, `g49-admission.jsonl`,
  `g49-child-commands.jsonl` and `g49-children/` retain the evidence. `g49-work/children/` additionally retains
  full stdout/stderr for each child. `g49-evidence.tar.gz` is a copy of those outputs.
- Focused fresh reload diagnostics for `S1_F100-s1` and `S1_F111-FB-s1`: both exit 0 (13.96 s and 13.92 s).
  Across 4 and 9 binary sites respectively, **no common field changed** and **no reload-only field appeared**.
  F100 had only the two moment-mismatch fields in its epoch row; F111-FB also had dead-channel and FB-sign
  diagnostics. See `g49-binary-audit-diagnosis.json` and `diagnosis-*-command.json/.txt`.
- Final `git diff --check`: exit 0. Final changed-file hashes and binary diff equal the final patch tested on
  the PC. HEAD is unchanged. The test temporary directory was removed; bundle, patch and this note remain
  untracked intentionally. No production code or synced PC checkout was changed, and no process was killed.
- Exact deferred list: supplied `preflight/b18p/pc-tests.txt`, 19 nodes. One node, the opt-in real pack test,
  currently requires a production-path plan and supported NRP product; clarification requested because those
  prerequisites conflict with PC-scratch-only execution. No compatible plan was supplied during this task.
  `pc-case-coverage.json` reconciles all 19 exact nodes and all 10 lossguard tests against their output logs.
  No cluster or NRP execution was performed.

## Additional G49 finding (outside the loss-ID guard)

The first binary-arm reloads fail `audit_mismatches == []` with `['binary_sites']`, while their runner exits
are 0, both training traces pass the census, logits/loss/accuracy differences are 0, and no forbidden open
is reported. `runner.enrich_row` adds `moment_mismatch_mean`, `moment_mismatch_var` and, for fused arms,
`dead_channel_frac` / `fb_sign_diff_frac` into the epoch row's binary-site dictionaries. The reload child
compares that enriched row with `FrozenEvaluator`'s core-only result. `BINARY_AUDIT_EXCLUDE` currently excludes
only `sign_flip_rate`, so the extra diagnostic fields cause unequal dictionaries.

DECISION: Preserve the original G49 failure and complete all configurations to assess the B10-F1 census.
ALTERNATIVES: Silently filter diagnostic fields in the development run (would hide a real gate-wiring issue);
expand this patch into frozen-audit semantics (separate scope requiring its own regression coverage).
CONFIDENCE: HIGH: two focused fresh-process reloads confirmed exactly the added diagnostic fields, with
all common fields equal. All 57 G49 failures report only `binary_sites`; every other audit field passes.
FLAG FOR HUMAN: YES. The B18 diagnostic enrichment and G49 core-only reload audit need reconciliation before
a gate-of-record G49 can pass. This is distinct from a missing training penalty; do not claim G49 PASS.

## Intended commits (not created)

1. `fix(gates): guard repeated training traces against stale Keras loss ids` — gate builders and test helpers.
2. `test(lossguard): exercise gate builders under deterministic stale-id collisions` — regression coverage,
   plus the B18 producer fixture correction to the release's sparse-label contract.
3. `docs: record B10-F1 development verification and remaining owner decisions` — this note and evidence pointers.

Transfer artifacts `b10f1-codex.bundle` and `b10f1-codex.patch` are intentionally untracked; do not include them
in source commits. `transfer.json` and `transfer-initial.json` identify both tested patch snapshots.

## Exact summary lines

```text
Offline: 602 passed, 4 skipped, 26 deselected, 1 warning in 10.49s
PC initial: 4 failed, 21 passed, 1 skipped, 606 deselected in 102.28s
PC corrected producer fixture: 4 passed in 160.24s
PC original lossguard mechanism tests: 3 passed, 7 deselected in 1.05s
G49-DEV-B10F1 FAIL (57 failed check(s))
Census aggregation: 109 configurations, 218 checks, 0 training failures, 0 missing guards/checks
```

These are development results, not quotable scientific results or gates of record. The one skipped pack case
and the separate audit-schema failure remain visible; neither is reported as a pass. The requested guard
coverage and census exercise are complete. The original STUDY 7.3 owner amendment remains a human decision.

## G49 enrichment audit wiring fix (2026-10-19)

Base: branch `b10f1`, HEAD `071da57`. Changes left uncommitted.

- `cbl/runner.py`: define `BINARY_ENRICHMENT_FIELDS` once at module scope beside `prepare`, which owns
  `enrich_row`. Before merging binary-site diagnostics, `enrich_row` asserts that every emitted field belongs
  to this set: the two moment-mismatch fields, `dead_channel_frac`, and `fb_sign_diff_frac`.
- `cbl/frozen_eval.py`: derive `BINARY_AUDIT_EXCLUDE` from that shared set plus `sign_flip_rate` (which depends
  on the previous evaluation in the same process). G49's existing reload-child call to `audit_mismatches`
  now compares the same core fields on enriched epoch rows and core-only reload results.
- `tests/test_frozen_audit.py`: eight offline regression cases cover plain and fused-BN enriched rows matching
  their core rows, a changed core quantizer value failing, and an unknown binary-site field failing on either
  the epoch or reload side.

Validation command:
`/Users/kaiyamaguchi/lab/venvs/mhab1012-cpu/bin/python -m pytest tests -q -p no:cacheprovider -m "not pc_gpu"`.
Run with `PYTHONDONTWRITEBYTECODE=1` and `TMPDIR` under this worktree to keep test writes local.
Exit 0: **610 passed, 4 skipped, 26 deselected, 1 warning in 21.73s**. The warning is the existing invalid-divide
warning in `tests/test_data.py::test_reader_refusals`. `git diff --check` passed.
No network, credentials, PC, GPU, or G49 campaign run was used. The historical G49 failure above remains
recorded; these offline regressions do not establish a gate-of-record G49 PASS.

Intended commit (not created):
`fix(g49): share binary enrichment fields with the frozen reload audit`
— include `cbl/runner.py`, `cbl/frozen_eval.py`, `tests/test_frozen_audit.py`, and this note.
