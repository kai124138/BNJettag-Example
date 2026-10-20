# PREFLIGHT B17: launch tooling

Date: 2026-10-19. Worktree: `/Users/kaiyamaguchi/lab/worktrees/cbl1016-b17`.
Original base: `267c6deb053e`. Continuation starts at orchestrator commit `a474b51`,
branch `b17`; only this policy correction remains uncommitted. No network, kubectl, SSH, remote machine or credential file was accessed.
Writes are confined to this worktree and the campaign's `preflight/b17/` directory.
No files under chang/, vendor/, configs/ or the lab-root tools/ were edited.

## Delivered implementation

M20 is in `nrp/`. `spec.py` admits only supported, launchable generated configs,
applies the runner's two-field `with_handoff()` copy, binds source/archive/init
identities and checks the execution plan again in the pod. `jobs.py` generates
bundle-mode handoffs using the unchanged lab `tools/run_handoff.py`. Modes cover
packs, smoke, benchmarks, determinism records and sequential U1 initial draws.
All durable NRP paths are under `/data/cbl1016/`; only kai-data and transient
emptyDir caches are mounted. Ordinary GPU resources use `nvidia.com/gpu`, with
exact RTX 4090, L40S, L40, RTX 3090 or A10 product affinity. Resources, excluded
nodes, deadline and TTL are explicit spec inputs. Manifest edits are not part of
the workflow.

`run_pack.py` runs one process per arm, with separate M15 run directories,
per-run locks, logs, exit status and telemetry. Existing run directories take
M14's resume path; verified completed siblings are skipped. Missing checkpoints,
terminal failures, partial initial draws and identity mismatches are refused.
Infrastructure relaunch preserves scientific attempt, config, source bundle,
run name and run directory. Pack allocation time is divided among live child
processes; benchmark accounting adds measured pod lifecycle overhead. Host/device
memory peaks are sampled lower bounds, not exact allocator maxima. Missing
samples remain null. The runner change only records the actual pack K in
provenance through `--pack-k`; it does not change scientific recipes.

M21 is in `tools/`. `operations.py` supplies prepare, relaunch, evaluate,
workspace and repair-check contracts. `campaign.py` wraps the unchanged lab
harness submit path, including tick's automatic resubmission, with exact run
admission, a frozen external admission policy, simultaneous training/reserve
budgets, unfinished-Job counting and G80 before every wave. `configmaps.py`
compares the complete kai ConfigMap inventory with distinct ConfigMaps mounted
by unfinished kai Jobs; both sets must be empty for bundle operation. It reads
ConfigMap names only. No cleanup is automatic.

`bench_rates.py` reads captured telemetry and measured full pod lifetimes,
reports class-specific GPU-seconds per run-epoch and projects the required
screen without reapplying the planning 1.10 factor. Short measurements do not
certify the rolling three-hour utilization gate. `determinism.py` compares the
two CH-8 and one F000 20-epoch records, reporting differences without inventing
a scientific pass requirement. Evaluation records exploratory lifecycle success;
it does not infer model scores or read held-out data.

`draft_brief.py` writes the schema-valid, unapproved campaign draft. The external
deliverables are `preflight/b17/BRIEF.draft.md` and `ADMISSION.draft.json`.
The committed admission template contains 51 required config identities and
33 engineering identities, still subject to runner support and check_launchable.
Controls and S4 have no production admission. Production certification is null.
The authority freezes the reviewed external ADMISSION.json separately from the
code template so that certification can bind the final commit without a
self-referential commit hash. No approval or cron entry was installed.

`docs/interfaces.md` section 20 and the two module READMEs describe these
contracts. `gates/b17_offline.py` reproduces the offline evidence. All M20/M21
module imports are free of Keras/JAX imports; pure runner admission uses NumPy.
GPU training is isolated in subprocesses and the separately marked PC test.

## Sources and pattern provenance

The current PREFLIGHT plan, STUDY snapshot v0.10, build-todo, flags-for-kai,
staging records, venv handoff, interface contract, generator, runner, launcher,
harness and GPU policy were read before implementation. The normative campaign
numbers come from STUDY v0.10 sections 10–11: 400 GPU-h total, 50 GPU-h reserve,
350 GPU-h training, four unfinished GPU Jobs, 51 required runs, 66 including
controls, and separately approved S4 additions of 20/32/16 runs. Required runs
retain all seeds and 7000 epochs. W&B handling follows section 12: offline on
the PVC, with no credential injection.

`tools/pattern-sources.json` records the exact source path and full SHA-256 for
each reused paused-campaign pattern: BRIEF.md, SETUP.md, code/run_pack.py, and
tools/prepare_pack.py, relaunch_mhab.py, repair_check_mhab.py, evaluate_mhab.py,
bench_rates.py, capture_pods.py, fetch_runs.py and README.md. Reuse is conceptual;
old scientific settings, rates and ConfigMap transport were not copied.

Current launcher SHA-256:
`8190bb8d693833036d8a62cadc90755d33ae3b4165625fb43fd5a3ab6426ab5b`.
Current harness SHA-256:
`6890d29d11bb93bbe8b1822c22e619f904014f6f9dbc41ad5a0aa98cbe6eba8d`.
The staged transport bundle commit is
`f8ca91d7dce442b581d02678aebe6dff64f1a236`, bundle SHA-256
`605dfae1560907d8cb476cca3400c49974d164ec855be04694f15874dbacb125`.
These sources remain unchanged. `preflight/b17/gate-scope.json` records these
hashes and the gate limitations.

## Decisions, assumptions and human flags

### B17-1: unlimited infrastructure retries (supersedes the finite B17 decision)

SUPERSEDED: B17 proposed two automatic relaunches per pack lineage and asked Kai to
accept it. That owner default was incorrect and is withdrawn; there is no retry-count cap.

DECISION/SOURCE: Kai's earlier instruction, "if things fail or queue fails, just loop
until it succeeds", is quoted in campaign `notes/build-todo.md:86-92` (B17 correction):
`/Users/kaiyamaguchi/lab/bnjettag/campaigns/2026-10-16-chang-bitlinear/notes/build-todo.md`.
STUDY v0.10 §11, Resubmission (`review/snapshots/v0.10/STUDY.md:1288`) restricts this to
queue/infrastructure failures and requires code fixes as new commits, re-verified and handed
off anew. Kai's current B17 instruction explicitly includes eviction/preemption, node failure,
image/volume mount failure, scheduling/pending timeout, and activeDeadlineSeconds exceeded
while pending. Missing evidence does not establish an infrastructure cause.

Relaunch without a count limit while the 350 training / 50 reserve GPU-h buckets admit it,
with the ConfigMap inventory/count check before every wave. Each existing run resumes its
latest checkpoint through M14; no epoch-zero restart substitutes for a refused resume.
Code/integrity exits, config/resume refusals, hash/provenance mismatches and gate failures stop
the lineage and preserve evidence for a new committed fix, repeated checks and a new handoff.

More than N = 6 successful infrastructure relaunch submissions of one pack in a rolling 24 h
notify Kai through local NOTIFY.log but keep going. This is the BRIEF field
`scientific_settings.b17_retry_policy.infra_relaunch_notify_after_24h`; the sibling
`infra_relaunches_max: null` denotes no count limit. The map is approval-bound and accepted by
the unchanged lab schema; its contents are operational, with no scientific setting change.
Legacy top-level retry fields are zero to disable direct lab-harness retries. The campaign
adapter alone installs unbounded decisions, preserving budget, approval and G80 gates.

CONFIDENCE: Offline regression evidence below; no live cluster/GPU claim.
No separate owner approval of a finite retry limit remains pending.

### B17-2: code repair and resume

DECISION: Disable automatic agent repairs. Refuse epoch-zero restarts when a
run directory exists but M14 cannot resume it. Code/integrity failures require
a new committed fix, repeated checks and a new handoff, per STUDY section 11.

ALTERNATIVES: Approve a bounded repair scope or a separately identified fresh run.

CONFIDENCE: High; avoids changing scientific identity during an infra retry.

FLAG FOR HUMAN: Code fixes and terminal/partial artifacts require explicit
review; repair-check intentionally refuses automatic repair.

### B17-3: harness scheduling fields

DECISION: Use a 3600-second lease, the schema-minimum unused 600-second score
deadline, zero disabled capabilities and unit factors. The schema's planning
rate of three seconds per epoch comes from STUDY section 10 and never supplies
a Job deadline. Protected operation commands live in authority because the
current harness schema requires that location. A 15-minute PC cron cadence is
documented as a proposal from the hashed paused SETUP pattern, not installed.

ALTERNATIVES: Other scheduling defaults or no cron; these do not set run recipes.

CONFIDENCE: High for schema compatibility; scheduling cadence needs PC review.

FLAG FOR HUMAN: Approve operational defaults and install any cron only on the
home PC after deployment. No Macintosh state-writer is proposed.

### B17-4: reserve interpretation

DECISION: Enforce simultaneous 350 GPU-h training and 50 GPU-h reserve buckets.
Engineering Jobs, failed Jobs and retries debit reserve. Unknown use retains
the full hard bound; every new Job reserves its deadline-based bound. Pending,
ambiguous and incompletely reconciled terminal Jobs retain unfinished slots.

ALTERNATIVES: The lab harness's later-stage-only reserve accounting, which does
not reserve engineering use before/during training; or a revised owner policy.

CONFIDENCE: High that the cap is conservative; allocation among buckets is an
explicit interpretation of STUDY section 10's engineering/failure reserve.

FLAG FOR HUMAN: Confirm bucket classification. Use the campaign adapter for
every submit/tick; direct lab-harness submit bypasses these added checks.
The adapter is a command contract, not Kubernetes access control.

### B17-5: initial tensors and archive separation

DECISION: Draw each seed's CH-8 tensors once per product/archive root, with one
visible GPU. Record and smoke roots are separate under
`/data/cbl1016/initial/<product>/<record|smoke>`. Existing draws require matching
schema, seed, product, stack, code hashes and tensor/archive hashes; partial
draws are refused. Draft authority admits seeds 1–3 only.

ALTERNATIVES: A single archive-independent manifest or silent replacement of
partial draws; neither satisfies the current runner identity contract.

CONFIDENCE: High for the U1 persistence contract; actual GPU reproducibility is
unmeasured here.

FLAG FOR HUMAN: B18 must resolve generator inputs and roots. B19 must execute
draws on the product of record, verify seed-1 against the PC result and supply
the actual manifest hashes. No initial tensors were drawn by B17.

### B17-6: product, K and resources

DECISION: Require explicit product/K, resource limits, deadlines, TTL, excluded
nodes and deterministic-ops selection. K=1 requires justification. Combined JAX
allocation fractions cannot exceed 0.90. Production needs a reviewed certificate
for the exact code/product/K, required-screen cost, utilization, device/host
memory and fingerprint gates. No GPU model implies measured throughput.

ALTERNATIVES: Borrow paused-campaign resources or hard-code a preferred GPU.

CONFIDENCE: High in the admission checks; no product or K is certified here.

FLAG FOR HUMAN: B19 must compare same-workload candidates and queue delays,
capture full lifecycle rates, measure the three-hour utilization window and
project host memory through the full horizon. Review the deterministic-ops
choice; false preserves the current release default. The launcher environment
record limit is 100,000 bytes and may further limit K; oversized packs refuse.

### B17-7: telemetry interpretation

DECISION: Allocate pack GPU wall time equally among currently live children,
then distribute full measured pod overhead proportionally for benchmark rates.
Collect minute/early GPU samples and frequent host samples. Preserve missing
data as null and report sampled peaks as lower bounds.

ALTERNATIVES: Per-process GPU accounting from an external profiler or a denser
sampling schedule; those require separate measured overhead and validation.

CONFIDENCE: High in allocation conservation; per-arm values are accounting
attributions, not direct measurements of an arm's exclusive GPU occupancy.

FLAG FOR HUMAN: Do not certify memory safety from sampled peaks alone or infer
three-hour utilization from a short benchmark. Supply complete lifecycle data.

### B17-8: immutable source and incomplete smoke gate

DECISION: Refuse normal handoff preparation unless the bundle contains B17 and
matches local cbl/gen/nrp Python sources. The currently staged bundle predates
B17; current generated configs also retain unresolved decisions/U1 hashes.
Offline G78 therefore uses a transport-only diagnostic smoke handoff whose
command checks launchability and refuses unresolved inputs. It cannot train
and has no admitted attempt record. It uses actual staged config bytes.

ALTERNATIVES: Invent launch inputs, stage uncommitted code or claim the transport
check establishes a real training-smoke gate; none is supported by evidence.

CONFIDENCE: High in the actual launcher transport validation; training G78 is
incomplete.

FLAG FOR HUMAN: B18 needs final committed code, regenerated configs and a new
complete staged bundle. B19 needs real U1 manifest hashes and runtime gates.
Regenerate the final smoke handoff and repeat G78 before launch.

### B17-9: external admission and scientific scope

DECISION: Store reviewed production certification in campaign ADMISSION.json,
outside the code bundle, and freeze its hash in authority. Admit only explicit
config paths; runner checks still refuse unsupported templates. Exclude control
production and S4 until separate admission. Preserve the 51-run required screen
before adding optional work. Evaluation is lifecycle-only engineering evidence.

ALTERNATIVES: Self-referential certification in the committed bundle, wildcard
admission, or inferring scientific success from Kubernetes completion.

CONFIDENCE: High in identity enforcement and distinction of evidence status.

FLAG FOR HUMAN: Install reviewed ADMISSION/BRIEF files and renew approval after
changing frozen authority. Unsupported controls/S2 and later S4 require their
own implementation and decisions; list membership is not launch permission.

### B17-10: ConfigMap inventory

DECISION: Check complete namespace-wide kai ConfigMap metadata against distinct
mounted references in active/pending kai Jobs, including init/projected mounts.
Bundle mode requires both counts to be zero before each wave or retry. Refuse
pagination, mismatch or legacy kai ConfigMaps rather than deleting anything.

ALTERNATIVES: Check only this campaign or merely require the two counts to match.

CONFIDENCE: High for offline logic; no live inventory was read.

FLAG FOR HUMAN: Cluster owner must reconcile legacy resources and record fresh
G80 evidence. The implementation includes live commands but none ran here.

### B17-11: upstream launcher resources

DECISION: Generate equal CPU/memory requests and limits for the application
container. Leave the lab launcher unchanged. Its injected init container has
requests 100m/128Mi and limits 1 CPU/512Mi, exceeding the policy's 1.2 ratio.

ALTERNATIVES: Fix the supported launcher upstream and regenerate; hand-editing
the manifest would break the generated contract and is excluded.

CONFIDENCE: High from source inspection; cluster lint was not executed.

FLAG FOR HUMAN: Resolve or explicitly review this upstream G77 issue before
training. No claim of successful manifest lint is made.

### B17-12: full allocation accounting

DECISION: Extend the harness's in-memory pod accounting from pod status.startTime
through the last init/application container termination, including init time.
Missing start or incomplete termination evidence retains a conservative bound.
The lab-root harness file remains unchanged.

ALTERNATIVES: Count only the application container or consume a separately
captured scheduler/GPU allocation timeline.

CONFIDENCE: High that the estimate includes init overhead; pod startTime is a
conservative observable proxy rather than a scheduler billing timestamp.

FLAG FOR HUMAN: Retain complete pod lifecycle captures before TTL deletion and
reconcile unknown attempts. B19 benchmark lifecycle inputs must use the same
all-inclusive definition, with their source recorded.

## Original implementation verification and offline gate outputs (historical)

Offline test command:

```sh
TMPDIR="$PWD/.tmp" PYTHONDONTWRITEBYTECODE=1 /Users/kaiyamaguchi/lab/venvs/mhab1012-cpu/bin/python -m pytest tests/b17 tests/test_runner.py tests/test_gen.py tests/test_resume.py -m 'not pc_gpu' -q --basetemp .tmp/pytest-b17
```

Python 3.12.13; exit 0; **139 passed, 1 deselected**, 3.03 seconds.
The original `preflight/b17/tests.txt` has been superseded by the policy-correction
run below; its current command/output is no longer this historical 139-test run. Tests exercise admission/refusal, real local child supervision,
resume commands, immutable identities, product/volume/resource contracts,
retry lineage, conservative accounting, namespace inventory logic, rate
allocation, determinism reporting, draft schema and the real launcher validator.
No test in this offline set executes GPU training. The PC/GPU test remains
explicitly opt-in; real kill/resume equivalence remains a later runtime gate.
Final verification also passed AST parsing for all 17 B17 Python files, all 44
draft frozen-file hashes, all 11 pattern-source hashes and `git diff --check`.
The interpreter and exit status are recorded in `preflight/b17/final-checks.json`.

Gate evidence was regenerated with:

```sh
TMPDIR="$PWD/.tmp" PYTHONDONTWRITEBYTECODE=1 python3 -B gates/b17_offline.py --evidence /Users/kaiyamaguchi/lab/bnjettag/campaigns/2026-10-16-chang-bitlinear/preflight/b17 --lab-root /Users/kaiyamaguchi/lab/bnjettag --bundle /Users/kaiyamaguchi/lab/bnjettag/campaigns/2026-10-16-chang-bitlinear/preflight/nrp/bundle/cbl1016-main-f8ca91d7dce4.bundle --commit f8ca91d7dce442b581d02678aebe6dff64f1a236 --data-info /Users/kaiyamaguchi/lab/bnjettag/campaigns/2026-10-16-chang-bitlinear/preflight/nrp/data_info.json
```

The gate driver and child commands used Python 3.14.7 and exited 0. Exact child
commands, interpreter and exit codes are recorded in the following files under
`/Users/kaiyamaguchi/lab/bnjettag/campaigns/2026-10-16-chang-bitlinear/preflight/b17/`:

| Gate/output | Result |
| --- | --- |
| G76.txt | Real `tools/harness.py validate G76-draft`: VALID, exit 0; no matching approval, submissions refused |
| G78-prepare.txt | Real `tools/run_handoff.py prepare` on generated diagnostic smoke Job: exit 0 |
| G78-validate.txt | Real `tools/run_handoff.py validate` on resulting bundle handoff: VALID, exit 0 |
| G78-offline-launch.txt | Real `tools/run_handoff.py launch` without --submit: OFFLINE_VALIDATED, exit 0, no cluster calls |
| gate-scope.json | G76 pass; G78 partial transport-only; G77/G79/G80 not run |

The validated diagnostic handoff is
`G78-smoke-transport/2753a9a72a89/handoffs/rh-944597d25782b2a6903cf274`.
G76 checks schema rather than deployment approval or completed scientific gates.
G78's result is deliberately limited by B17-8. No GPU runtime, production rate,
W&B artifact or scientific result is claimed.

## Remaining integration and cluster gates

1. B18: resolve owner inputs through gen/, integrate and commit B17, regenerate
   configs, freeze a complete bundle and stage it through the approved process.
   Rebuild handoffs from that bundle; the existing staged bundle is insufficient.
2. Kai/PC owner: review the flagged defaults, install ADMISSION/BRIEF, regenerate
   frozen hashes, validate and issue terminal approval only after required gates.
   Run all campaign state-writing operations through the adapter on the home PC.
3. G77: run the supported cluster-aware nrp_doctor lint with kubectl available,
   including capacity, product selectors, volumes, deadlines, TTL, resource
   ratios and one-GPU constraints. Resolve the injected init-container resource
   ratio in the supported launcher and regenerate handoffs.
4. G78 remainder: validate a real launchable training smoke handoff, then perform
   its authorized runtime portion using the final bundle and real init hashes.
5. G79: on the cluster verify in-pod HEAD/tree and bundle identities, mismatch
   refusal and absence of ConfigMap code transport. Preserve runtime evidence.
6. G80: capture fresh complete kai ConfigMap/Job inventory, verify equality and
   both zero, then verify the check executes before every adapter wave/retry.
7. B19/PC: validate the staged GPU runtime, U1 draws and seed-1 comparison,
   determinism record, real process telemetry, product/K benchmark and all memory,
   utilization/fingerprint/cost gates; run the real M14 kill/resume test. Preserve
   complete pod lifecycle data before TTL cleanup. Fill reviewed production
   certification only from these measurements.

## Original intended commits (now included in orchestrator commit a474b51)

B17 did not write the git index or commit; the orchestrator committed the prior implementation.
The original intended sequence was:

1. **`runner: record actual pack concurrency in provenance`**
   Files: `cbl/runner.py`.
2. **`nrp: add immutable bundle pack jobs and resumable process supervisor`**
   Files: `nrp/__init__.py`, `nrp/common.py`, `nrp/spec.py`, `nrp/jobs.py`,
   `nrp/run_pack.py`, `nrp/README.md`.
3. **`harness: add campaign admission budgets retries and wave inventory checks`**
   Files: `tools/__init__.py`, `tools/campaign.py`, `tools/operations.py`,
   `tools/configmaps.py`, `tools/bench_rates.py`, `tools/determinism.py`,
   `tools/draft_brief.py`, `tools/admission.json`, `tools/pattern-sources.json`,
   `tools/README.md`.
4. **`test: verify B17 offline contracts and record partial preflight gates`**
   Files: `pytest.ini`, `tests/b17/conftest.py`, `tests/b17/test_nrp.py`,
   `tests/b17/test_tools.py`, `tests/b17/test_pc_gpu.py`,
   `gates/b17_offline.py`, `docs/interfaces.md`, `NOTES-B17.md`.
   External draft/evidence files under campaign `preflight/b17/` are separate
   campaign deliverables and are not part of these worktree commits.

## Policy correction verification (2026-10-19, from a474b51)

Only B17-1's retry policy changes. `tools/retries.py` centralizes its BRIEF contract,
infrastructure/code classification and rolling notification count. `tools/campaign.py`
installs unbounded decisions in memory, preserves terminal pod evidence through refresh/TTL,
stops code/integrity/gate-failed lineages, and keeps budget/G80 admission on every submission.
`tools/operations.py` removes its count cap, rechecks raw failure evidence and stops on a
preparation refusal. The draft writer and interface/tool documentation carry the same policy.
No retry-count limit existed in nrp/; its existing M14 latest-checkpoint resume and completed
sibling checks are unchanged and remain covered by tests. No lab-root harness change is needed.

Requested suite, with `PYTHONDONTWRITEBYTECODE=1` and `TMPDIR=$PWD/.tmp`:

```sh
/Users/kaiyamaguchi/lab/venvs/mhab1012-cpu/bin/python -m pytest tests/b17 -q -p no:cacheprovider
```

Exit **0**, **87 passed, 1 skipped**, 3.96 s. The skip is the opt-in PC/GPU test.
`preflight/b17/tests.txt` records command, cwd, environment, interpreter, exit code and output.
Coverage includes retries above 2 and 6 and with a counter of 1,000,000, terminal integrity
precedence over disruption (including init containers), stale observation refusal, retained
failure evidence after pod disappearance, M14 identity/resume commands, preparation/gate
lineage stops, budget exhaustion, G80 on repeated high-attempt submissions, and the >6/24h
notification boundary, pack isolation, deduplication and expiry. No GPU/cluster test ran.

Regeneration (exit **0**) is recorded in `preflight/b17/BRIEF-policy-generate.txt`:

```sh
python3 -B -m tools.draft_brief --code-root /Users/kaiyamaguchi/lab/worktrees/cbl1016-b17 --out /Users/kaiyamaguchi/lab/bnjettag/campaigns/2026-10-16-chang-bitlinear/preflight/b17/BRIEF.draft.md
```

The exact draft bytes were copied to `preflight/b17/G76-draft/BRIEF.md`, then validated:

```sh
python3 /Users/kaiyamaguchi/lab/bnjettag/tools/harness.py validate /Users/kaiyamaguchi/lab/bnjettag/campaigns/2026-10-16-chang-bitlinear/preflight/b17/G76-draft
```

Exit **0**, **VALID**; no matching approval exists, so submissions remain refused.
`preflight/b17/G76.txt` records command, cwd, environment, interpreter, exit code and output.
This supersedes the historical G76 draft validation above; G78 was not rerun for this policy
change, and all earlier cluster/scientific limitations still apply. Current frozen-file hashes,
source hashes and changed-file scope are recorded in `preflight/b17/policy-final-checks.json`.
The previous `final-checks.json` remains historical evidence from the original implementation.
No network, kubectl, SSH, credentials, commits, approvals or launches were used. Writes stay
inside this worktree and preflight/b17; chang/, vendor/, configs/, cbl/ and lab-root tools/ remain
untouched by this continuation.

## Intended commit for this continuation

**`harness: retry infrastructure failures without a count cap within campaign budgets`**

Files: `tools/retries.py`, `tools/operations.py`, `tools/campaign.py`, `tools/draft_brief.py`,
`tools/README.md`, `tests/b17/test_tools.py`, `docs/interfaces.md`, `NOTES-B17.md`.
Leave all changes in the working tree for the orchestrator; no git index or commit is written.
Regenerated BRIEF and validation evidence under campaign preflight/b17 are external deliverables.
