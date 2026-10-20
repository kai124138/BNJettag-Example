# B17 launch tooling

All preparation is offline. Commands run from the campaign code checkout. No file under
configs/ is edited by these tools. No Job is submitted by nrp.jobs or tools.operations.

`python3 -B -m nrp.jobs --spec PACK.json --bundle FINAL.bundle --bundle-path
/data/cbl1016/code/FINAL.bundle --commit FULL_SHA --data-info data_info.json --out CAMPAIGN
--launcher LAB/tools/run_handoff.py` prepares a bundle handoff. The source bundle must contain
the final B17 code and match the local cbl/, gen/ and nrp/ Python modules. The staged f8ca91d7
bundle is insufficient. A new B18 commit needs a new complete bundle file and approved staging.
The supported launcher caps its encoded environment record at 100,000 bytes. Preparation refuses
oversized packs; B19 must check the chosen K against this bound as well as GPU and host memory.
The execution plan reconstructs launch copies from the immutable checkout to avoid a redundant
copy in the environment. The generator does not silently split packs or change measured K.

The pack spec is schema 1. Required fields:

| Field | Meaning |
| --- | --- |
| pack | Lowercase DNS label, at most 26 characters; stable across infrastructure retries |
| mode, stage | pack with S1/S3/S4, or smoke/smoke, bench/bench, det/det, draw/draw |
| job_attempt | Positive Job attempt; does not change the scientific config attempt |
| gpu_product | One exact ordinary-resource product from nrp.common.PRODUCTS |
| archive_kind | smoke only for smoke training; record for bench/det/production; either for draws |
| arms | List of config_path under configs/ and init_manifest_sha256 from the actual U1 draw |
| seeds | Draw mode instead of arms; draws run sequentially, one process per seed |
| mem_fraction | Per-process JAX allocation fraction; K times this value must be at most 0.90 |
| resume_every | M14 checkpoint interval; unchanged on resume |
| deterministic_ops | Explicit boolean; false preserves the release default pending Kai's choice |
| resources | Explicit positive integers cpu, memory_gi, deadline_s, ttl_s, cache_gi |
| excluded_nodes | Explicit current known-bad-node list, reviewed by cluster owner |
| single_arm_justification | Required for K=1 |
| approval_ref, scientific_gate, gate_reference | Factual handoff references; pending until reviewed |

No resource value or K is inferred from a GPU model. Each production product/K needs same-workload
measurements and queue-aware selection under docs/infrastructure/gpu-selection-policy.md. Both
CPU/memory requests and limits are equal in the generated training container. The unchanged lab
launcher injects an init container whose limits exceed 1.2 times requests: G77 must report and
resolve that upstream issue before training; manifests must not be hand-edited to hide it.

U1 roots are `/data/cbl1016/initial/<product>/record` and `.../smoke`. Set gen/inputs.json roots
and platform gpu through the reviewed generator workflow before B18. Draw manifests themselves
are created later and their hashes enter only with_handoff launch copies. Draw Jobs expose one
GPU, as training pods do. A complete verified draw is reused; a partial draw is refused without
overwriting it. Seed-1 production/PC comparison is B19's check. The venv is
`/data/cbl1016/runtime/venv-c8725569731a`, in the staged buildpack-deps digest, still awaiting GPU
validation on the selected product. No installation or network access occurs in a training Job.

Runs are `/data/cbl1016/runs/<stage>/<M15-run-name>/`. M15's source token names the git bundle's
sha256, not the per-Job handoff. Infrastructure retries preserve this token, config, scientific
attempt and run directory. Only the Kubernetes Job/handoff attempt changes. An existing directory
is passed to M14 with --resume. Missing checkpoints and terminal failures are refused; there is
no automatic deletion or epoch-zero restart. Verified completed siblings are skipped. PVC advisory
locks prevent overlapping processes from writing the same run; abrupt process death releases them.

Pack evidence is `/data/cbl1016/packs/<pack>/<job>/<invocation>/`: plan, launch copies, complete
child stdout/stderr, GPU/host samples and telemetry. Each created run directory receives its own
immutable telemetry and log copies. A child refused before creating its run leaves evidence only
in the pack directory, so a failed attempt cannot manufacture an empty resumable run directory.
Minute GPU samples plus early samples record driver/product, utilization, device/process memory,
host RSS and checkpoint epochs. Host peaks are additionally polled each supervisor iteration.
Peaks are sampled lower bounds; missing data is null. SIGTERM/SIGINT/deadline terminates process
groups and preserves M14 checkpoints. No child is blindly relaunched by the pack runner.

Exit codes: 0 all completed/StopIf; 1 code failure; 2 refusal; 3 integrity; 76 missing/invalid
successful completion evidence; 124 deadline; 143 interruption. Unknown positive child exits
propagate. Kubernetes backoff is zero; the approved harness owns infrastructure resubmission.

`python3 -B -m tools.bench_rates PACK_CAPTURE --lifecycle LIFECYCLE.json` computes all-inclusive
GPU-seconds per run-epoch by class. Lifecycle JSON requires job, pod, complete=true, source and
measured gpu_seconds from GPU allocation through termination, including init/shutdown. The
supervisor's equal-live-process allocation splits the measured pod overhead proportionally.
The output includes source hashes and does not apply the planning 1.10 factor. Short utilization
records and absent host projections never certify production. B19 must capture full pod lifecycle,
measure host-memory growth through the full horizon, verify memory/fingerprint/cost gates, compare
products including queue delays, and put reviewed certification in the campaign ADMISSION.json.
No PVC fetch service or transport is created here; run the reader on an authorized capture.

`python3 -B -m tools.determinism CH8_R1 CH8_R2 F000` compares the 20-epoch records and variable
hashes exactly, excluding run identity and wall-time fields. Differences are reported as the
STUDY determinism record, not a failed scientific gate. It needs all three complete runs.

Offline tests:

```
TMPDIR="$PWD/.tmp" PYTHONDONTWRITEBYTECODE=1 /Users/kaiyamaguchi/lab/venvs/mhab1012-cpu/bin/python -m pytest tests/b17 tests/test_runner.py tests/test_gen.py tests/test_resume.py -m 'not pc_gpu' --basetemp .tmp/pytest-b17 -q
```

No B17 module imports Keras or JAX at import time. Pure runner admission uses NumPy only. Real
GPU testing is opt-in with marker pc_gpu, CBL_B17_PC_RUN=1, CBL_B17_PC_PLAN and CBL_B17_PC_PYTHON
on the PC after freeze and local data staging. The real kill/resume equivalence remains G83.

B18P supersedes the existing-directory restart paragraph above. `launch.json`
records the original product and interval before training; a directory without a
published checkpoint is moved to a dated `.pretrain-...` sibling and starts fresh
with that interval. A verified checkpoint uses `--resume`. Corrupt checkpoints
and terminal runs refuse. See interfaces §21 for legacy metadata refusals.

Retry preparation reads the mounted `/data/cbl1016` tree or a complete local
capture named by spec `run_state_root`. Preserve this spec input through M21's
relaunch path. An unavailable capture refuses preparation. Required GPU-product
affinity is cross-checked with the run-directory records and immutable U1 config.
Per-run checkpoint intervals override a later spec default. Live allocation
snapshots feed `sys/alloc_hours`; they exclude pod init and final unreconciled
lifecycle time and do not replace the all-inclusive budget ledger.
