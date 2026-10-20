# B18P — epoch logging and pack restart policy

Date: 2026-10-19. Worktree `/Users/kaiyamaguchi/lab/worktrees/cbl1016-b18p`,
branch `b18p`, HEAD `6cdbd05e65b9c134232f2a2959fb58ff9e2e0962`.
All changes remain unstaged/uncommitted. No network, kubectl, SSH, credential
inspection, submissions, or forbidden review/preflight directories were used.
No changes to `chang/`, `vendor/`, `configs/`, `gen/`, or lab-root `tools/`.
Evidence lives in campaign `preflight/b18p/`; temporary test files stay in the
worktree `.tmp/` or permitted system temporary storage.

## Built, with evidence status

Implemented producers for **26 of B10-F2's 27 missing patterns**. M15 now exempts
only `pruning_channel/*/native_share_removed`; `strict="produced"` stays enabled.
The missing counterfactual needs a STUDY/owner definition, detailed below. Do not
claim that the complete §12 catalogue is satisfied or that strict=True is ready.

M8/M13: quantizer/BN state summaries, stateless validation activation and binary
observations, stateless train-mode validation gap, and last-training-step FB sign
observation. M10: no-data structure projection and pruning fractions. M9: per-epoch
cumulative dwell, derived from existing resumable front state. M20: live cumulative
allocation snapshots and run-directory restart identity. M14: atomic publication
of a fresh directory with launch identity; its existing front snapshot and restore
already cover dwell. M11 checkpoint ablations and M12 PID behavior remain unchanged.

These are implemented producers with offline helper/integration tests. The actual
JAX/Keras producer executions are **not run here**. Four new real-stack tests are
marked `pc_gpu`; all required PC gates and performance checks remain due. No
training result or bitwise resume result is claimed from this Mac build.

## Every B10-F2 pattern, ownership and §12 source

The snapshot is `review/snapshots/v0.12/STUDY.md`. The `/` spellings are M15's
catalogue/interface spellings; STUDY gives the prose requirements listed below.
“Built” in this table does not mean PC execution has passed.

| Pattern | STUDY §12 source | Owner and producer | Status |
| --- | --- | --- | --- |
| `quantizers/*/*/width_mean` | widths and steps, line 1360 | M8 `EpochDiagnostics.state` / `width_summary` | Built |
| `quantizers/*/*/width_max` | same | M8, same | Built |
| `quantizers/*/*/step_min` | same | M8, same | Built |
| `quantizers/*/*/step_max` | same | M8, same | Built |
| `bn/*/gamma_min` | BN scales and moments, line 1360 | M8 `bn_summary` | Built |
| `bn/*/gamma_max` | same | M8, same | Built |
| `bn/*/moving_mean_absmax` | same | M8, same | Built |
| `bn/*/moving_var_min` | same | M8, same | Built |
| `bn/*/moving_var_max` | same | M8, same | Built |
| `activations/*/min` | activation ranges, line 1362 | M8 stateless quantizer output observers | Built |
| `activations/*/max` | same | M8, same | Built |
| `activations/*/zero_width_frac` | zero-width fractions, line 1362 | M8 with M10 inference mask | Built |
| `quantizer_structure/*/kernel_zero_pruned` | per-epoch structure routes, line 1361 | M10 `epoch_structure(quantizer_state(...))` | Built |
| `quantizer_structure/*/kernel_zero_rounded` | same | M10, same | Built |
| `quantizer_structure/*/input_all_zero_width` | same | M10, same | Built |
| `quantizer_structure/*/output_all_zero_width` | same | M10, same | Built |
| `quantizer_structure/softmax_heads/*/zero_width_frac` | per-head softmax fractions, line 1361 | M10, same | Built |
| `pruning_channel/*/zero_width_frac` | HGQ kernel zero widths, line 1363 | M10 `pruning_channels` (also records binary kernel fractions) | Built |
| `pruning_channel/*/native_share_removed` | native cost removed, line 1363 | M10, definition absent | **Unresolved** |
| `pruning_channel/*/input_zero_width_frac` | binary input zero widths, line 1363 | M10 `pruning_channels` | Built |
| `binary_sites/*/moment_mismatch_mean` | x·W versus x·K moments, line 1360 | M8 validation observers, `Moments`, `moment_difference` | Built |
| `binary_sites/*/moment_mismatch_var` | same | M8, same | Built |
| `binary_sites/*/fb_sign_diff_frac` | last-step batch/moving kernel signs, line 1365 | M13 train-step wrapper and M8 `fb_replay` | Built |
| `binary_sites/*/dead_channel_frac` | nonpositive pre-activation on all validation jets, line 1367 | M8 validation observers | Built |
| `train_frozen_top1_gap` | train-mode/frozen validation gap, line 1368 | M8 stateless training diagnostic, wired by M13 | Built |
| `dwell/*/*` | grid interval epochs, line 1369 | M9 `FrontState.dwell`; M14 existing snapshot/restore | Built |
| `sys/alloc_hours` | allocation hours, line 1371 | M20 `nrp.accounting`; M13/M8 row bridge | Built, live supervisor basis |

M13 enriches the row offered to M9 before saving and the row persisted/logged by
M8. Both reuse one diagnostic result per epoch. M9 copies diagnostic fields to
saved sidecars. The M7 frozen counts and their audit fields stay inference-only.
M15's B10 PID-warmup-error and epoch-zero sign-flip exemptions remain unchanged.

## Decisions and assumptions

### B18P-1 — undefined removed-cost share

DECISION: Leave only `pruning_channel/*/native_share_removed` in NOT_PRODUCED.
STUDY §12 and the §8.8 reading table request savings but define neither the
unpruned widths nor the baseline checkpoint. A currently zero-width coefficient
has zero current native cost; that does not identify how much cost was removed.
This cannot be supplied as the requested quantity without an owner definition
recorded in STUDY. Keep strict="produced"; never manufacture zero savings.

ALTERNATIVES: A one-bit floor at pruned entries; initialization widths; last-live
widths; an unpruned reference model. These yield different costs and denominators.

CONFIDENCE: High that the counterfactual is missing; no numerical default chosen.

FLAG FOR HUMAN: YES. Specify baseline widths, denominator, and zero-denominator
behavior, update STUDY, then implement/test the remaining key. Also review the
existing first-epoch/warmup exemptions before calling any mode “as written.”

### B18P-2 — widths, ranges and fractions

DECISION: Width summaries use each built HGQ quantizer's rounded format entries;
steps are `2**(-f)`. Activation min/max are actual quantizer **outputs** over the
complete validation set, not representable grid bounds. Zero-width activation
fractions use M10's inference mask `k+i+f<=0`, distinct from zero native bits.
Nested quantizers and multiple-input quantizer containers are traversed by role.
BN uses gamma (1 if scale=False), moving mean and moving variance.

ALTERNATIVES: Pre-quantizer activation ranges; averaging widths by repeated data
lanes instead of stored format entries; reporting representable bounds as ranges.

CONFIDENCE: Medium pending PC execution, high for the pure arithmetic tests.

FLAG FOR HUMAN: YES, review the range/width aggregation convention and run the PC
producer tests. No change to quantization, training tensors or released code.

### B18P-3 — moment mismatch and dead channels

DECISION: Compare validation population moments of the same quantized input x
contracted with raw latent W versus exported frozen K. Bias is excluded from the
moment comparison, as the requirement says x·W/x·K. Reduce batch and position axes,
retain output channels, combine uneven chunks in float64 using parallel moments,
then report the mean absolute per-channel mean/variance difference. A dead channel
requires pre-activation (x·K plus quantized bias, before activation/output quantizer)
<=0 for every validation jet and position; chunk masks are ANDed. NaNs are not dead.

ALTERNATIVES: Signed differences, maximum difference, batch-average variances,
including bias, or comparing normalized moments. None is silently substituted.

CONFIDENCE: Medium; exact whole-array agreement of accumulation is tested offline.

FLAG FOR HUMAN: YES, aggregation and units are owner conventions because STUDY
names moments but does not give a scalar reduction. PC tensor-path checks are due.

### B18P-4 — train-mode gap and frozen core

DECISION: Train-mode top-1 minus frozen top-1 on the same validation jets, ordered
2790-row chunks and short tail. Training passes use explicit stateless inputs and
discard returned updates. They are separate epoch diagnostics, not M7's frozen
count evaluation. The replayed inference top-1 must equal the frozen evaluator's.
M13 hashes all model variables and uses the RNG guard around diagnostics.

ALTERNATIVES: Calling a live training-mode model (would update BN/counters), using
training-set accuracy, or using a different validation batch size.

CONFIDENCE: Medium pending PC nonmutation tests. These forwards add cost.

FLAG FOR HUMAN: YES. Review the validation-batch convention; run producer tests,
G12 and G48/G46 after integration, and benchmark the extra per-epoch passes in B19.

### B18P-5 — literal last-step FB signs

DECISION: M13 delegates the original JAX train_step, then does a second stateless
forward of that same batch from its pre-update variable arrays. Observe FB signs
under its batch moments and its just-updated moving moments; both use the same
pre-optimizer kernel/gamma. Their fraction of differing sign entries is an extra
scalar log, `b18_fb/<site>`. Keras keeps the final step's logs, including with four
steps per execution and a short tail. M8 consumes those logs at epoch end. No
approximation from the last validation batch or post-optimizer weights is used.
No model/optimizer/metric state is added, so M14 needs no new sign-log checkpoint.

ALTERNATIVES: Altering the released gradient path or model variable schema;
comparing post-update weights to reconstructed batch moments; sampling fewer steps.

CONFIDENCE: Medium-low until the PC tests verify actual last-step propagation and
exact real-step state equality. Pure summaries alone do not validate this path.

FLAG FOR HUMAN: YES. Validate on the pinned PC stack before training and measure
FB replay overhead. This is an additional training forward on every FB step.

### B18P-6 — structure and dwell ownership

DECISION: Reuse M10's existing no-data state and M9's existing offered epochs.
Missing input/output quantizers have no zero-width route (false, not degenerate).
Dwell includes the current epoch exactly once; M14's existing front-state restore
reconstitutes history. All diagnostic fields also copy to M9 sidecars. No changes
to M11's ablations or M12's controller/sensor timing are needed for these 27 keys.

ALTERNATIVES: A second independently checkpointed dwell counter; per-epoch
zero-variance tests for the structure routes (STUDY asks state-only routes here).

CONFIDENCE: High for route projection and constructed resume-dwell tests.

FLAG FOR HUMAN: NO new policy; PC whole-run resume validation remains due.

### B18P-7 — first-attempt identity publication

DECISION: The pack verifies actual GPU product before child launch. M13 validates
the M20 launch record against config/name/source/code/interval, writes launch.json
to a temporary sibling, then publishes the run directory by rename. A kill before
publication leaves the target absent; a kill after publication retains metadata.
The original GPU **product label** and interval are durable before train begin.
The supervisor holds the per-run advisory lock; checkpoints are cross-checked
with launch.json, or supply legacy identity if launch.json predates this batch.

ALTERNATIVES: Writing metadata only at train begin; recording only a Job-level
spec; deriving the original interval from the newest spec default.

CONFIDENCE: High in offline filesystem/identity tests; no real PVC kill was run.

FLAG FOR HUMAN: YES for G83 on the target PVC. Temporary `.starting-*` siblings
from an interrupted publication are preserved; cleanup is not automatic.

### B18P-8 — pre-train kills, terminal state and corruption

DECISION: Under the run lock, a directory with launch metadata and no published
resume/latest.json is renamed to a UTC-dated, uniquely suffixed `.pretrain-*`
sibling. It starts fresh without --resume, retaining its original interval.
Partial checkpoint publication is preserved with the archived directory. A
published but invalid checkpoint or a terminal run is refused, never restarted.
Legacy directories with neither launch metadata nor checkpoint are refused:
there is no evidence from which to recover the original --resume-every.

ALTERNATIVES: Deletion; --resume on any directory; silently choosing the new
interval for a legacy directory; using an uncommitted partial checkpoint.

CONFIDENCE: High for preserved bytes and refusal behavior in offline tests.

FLAG FOR HUMAN: YES for legacy metadata-free directories: recover original
launch identity/interval from first-attempt evidence before migration. This
unrecoverable legacy case is an explicit limit, not a completed automatic restart.

### B18P-9 — Job generation and retry capture

DECISION: Read original records from mounted /data/cbl1016 or spec run_state_root,
a complete local copy with the same relative layout. Retry preparation requires
that root to exist. Product labels must agree across existing pack runs and with
the immutable generated config/U1 product; otherwise refuse. Required node
affinity pins the checked `nvidia.com/gpu.product`. Per-arm intervals are retained
in the wire plan, and the pod rereads them from its authoritative live run tree.
M21 already preserves arbitrary spec fields through relaunch; no network fetch
or lab-root tooling modification is introduced.

ALTERNATIVES: Unpinned retries; trusting only a changed spec; rewriting U1 paths
or launch configs to switch products; adding remote capture to this offline task.

CONFIDENCE: High for generated affinity/refusal tests; capture freshness is an
operator responsibility, and runtime rereads prevent stale intervals being used.

FLAG FOR HUMAN: YES. Supply a fresh complete PVC capture or mounted root to the
retry preparer. Existing B17 setups without it now fail closed. A wrong product
spec must be corrected to the recorded product, not bypassed.

### B18P-10 — allocation semantics and resume

DECISION: sys/alloc_hours is cumulative **supervisor allocation**, sharing elapsed
GPU wall time among currently live children, with monotonic extrapolation between
snapshots. Durable per-invocation cumulative baselines survive resumes and known
pre-train archival costs. No resource usage is rolled back with epochs. CPU gates
log zero GPU hours. Exclusive standalone GPU runs use elapsed prepare-time allocation;
packed GPU runs require M20's snapshot. Final M21 pod lifecycle accounting remains
the source of all-inclusive budget usage.

ALTERNATIVES: GPU kernel utilization; wall time divided permanently by initial K;
claiming pod init/shutdown are already known at an epoch; billing from W&B rows.

CONFIDENCE: High for arithmetic/continuity tests; medium for live attribution.

FLAG FOR HUMAN: YES. Live rows exclude pod init/final unreconciled tails. A total
SIGKILL can lose the final polling interval until lifecycle reconciliation. These
are not final all-inclusive bills. B19 must measure overhead and final lifecycle.

### B18P-11 — strictness and testing placement

DECISION: Strict produced checks now enforce all 26 new patterns when applicable;
the only NOT_PRODUCED exception is the undefined cost share. Existing tests that
actually import JAX/Keras/HGQ move behind pc_gpu, retaining pure checks offline.
New real-stack CH-8, F000, F111-FB and F111-BF producer tests use separate subprocesses.

ALTERNATIVES: Full strict mode with an invented share; leaving all 27 exemptions;
running the wrong Mac ML stack and treating its outcome as the PC gate.

CONFIDENCE: High for test placement/strictness; backend execution remains unverified.

FLAG FOR HUMAN: YES. Run the listed PC tests and applicable gates on the pinned
stack. The four new tests are producer/step checks, not a substitute for G48/G46.

### B18P-12 — G57 in an uncommitted worktree

DECISION: Run the recorded G57 command with this worktree and its actual HEAD.
Do not bypass or alter the clean-tree checks, and do not commit. gen/ is unchanged;
no regeneration write is needed. The Job generator changed, so G57 was run anyway.

ALTERNATIVES: Mislabel dirty-tree failure as pass; change the gate; make an
unauthorized commit; omit the requested gate.

CONFIDENCE: High from recorded output.

FLAG FOR HUMAN: YES. Rerun G57 after the intended commits exist in a clean checkout.

## Verification

Requested offline command (environment: PYTHONDONTWRITEBYTECODE=1,
TMPDIR=$PWD/.tmp; pytest cache disabled):

```sh
/Users/kaiyamaguchi/lab/venvs/mhab1012-cpu/bin/python -m pytest tests -q -p no:cacheprovider -m "not pc_gpu"
```

Final result: **596 passed, 1 skipped, 19 deselected**, exit **0**.
The exact command/interpreter/exit and output are in `preflight/b18p/tests.txt`.
`checks.json` records AST parsing, git diff --check, unchanged HEAD and changed-file
inventory. The one skip is the Linux-only inotify test. The warning is the existing
constructed zero-scale data-reader refusal fixture (`cbl/data.py:367`).
Development runs are retained separately and are not the final result: the first
found three fixture failures (retry inventory setup and a flattened fixture path),
and a targeted run found an incorrectly named dwell-bin assertion; all were fixed.

G57 command, run from this worktree with the recorded PC-CPU interpreter:

```sh
/Users/kaiyamaguchi/lab/venvs/mhab1012-cpu/bin/python -I -B gates/g57_configs.py --expect-sha12 6cdbd05e65b9 --reference /Users/kaiyamaguchi/lab/bnjettag/campaigns/2026-10-16-chang-bitlinear/review/snapshots/v0.8/protocol-draft.json
```

**Exit 1, G57 FAIL: exactly two failures**, git status empty at start and end.
Every other check passed, including generator imports, all config admission
checks, tests/test_gen.py, and byte-for-byte regenerate-and-diff of 172 files
(0 missing, 0 extra, 0 different). `preflight/b18p/G57.txt` is the full output;
`G57-command.json` records command/cwd/environment/exit. No generator/config file
was changed and no clean-tree gate success is claimed.

### PC tests explicitly deferred

`preflight/b18p/pc-tests.txt` contains all exact collected node IDs, including
parameter variants (19 cases). They are:

- `tests/test_b18_pc.py::test_real_epoch_producers_and_stateless_observer`:
  CH-8/none, F000/none, F111/FB, F111/BF. Real five-batch fit with steps_per_execution=4,
  diagnostic key coverage, variable/RNG nonmutation, and exact underlying step state.
- `tests/b17/test_pc_gpu.py::test_real_pack_smoke_on_pc` (existing explicit B17 opt-in).
- `tests/test_env.py`: setup_order_records_and_bindings;
  set_process_env_refuses_after_jax_import; bind_refuses_to_shadow_an_existing_name
  (two cases); chang_import_fails_without_the_shims;
  changed_binding_and_precision_are_detected.
- `tests/test_rng.py`: construction_draws_equal_reconstruction_on_hgq_dataset;
  recorder_records_mapped_index_and_not_shuffle_reads;
  recorder_keep_full_compacts_old_epochs; instrument_existing_dataset;
  guard_wraps_only_overridden_hooks_and_detects;
  fit_stream_matches_reconstruction_multiple_and_not_of_steps_per_execution
  (two cases); flow_refuses_out_of_order.

Use the pinned PC interpreter with `-m pc_gpu` and existing B17 opt-in variables
for its real-pack case. No PC test, G12/G46/G48 rerun, G83, cluster launch, or
throughput measurement was performed here.

## Intended commits (not created)

1. **`logging: produce epoch diagnostics structure and resumable dwell [B18P]`**
   Files: cbl/epoch_diagnostics.py, cbl/structure.py, cbl/front.py,
   cbl/frozen_eval.py, cbl/runner.py (diagnostic/row-enrichment hunks),
   cbl/wandb_log.py, tests/test_b18_diagnostics.py, tests/test_b18_pc.py,
   tests/b18_pc_case.py, tests/test_wandb_log.py.
2. **`nrp: preserve launch product interval and pretrain evidence on retry [B18P]`**
   Files: nrp/restart.py, nrp/jobs.py, nrp/run_pack.py, nrp/accounting.py,
   cbl/resume.py, cbl/runner.py (launch identity/allocation hunks),
   tests/nrp_test_helpers.py, tests/b17/test_b18_restart.py,
   tests/b17/test_nrp.py, tests/b17/test_tools.py, tests/test_resume.py.
   These two commits are an integration pair: run the full suite on the pair.
3. **`preflight: document B18P decisions and defer real-stack tests to PC`**
   Files: pytest.ini, tests/test_env.py, tests/test_rng.py, docs/interfaces.md,
   nrp/README.md, NOTES-B18P.md.

Campaign evidence files under preflight/b18p are separate deliverables, not git
worktree commits. The G57 dirty-tree failure and PC test deferrals remain visible.
