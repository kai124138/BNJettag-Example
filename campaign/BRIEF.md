# BRIEF for approval 1, revision 4 — chang-bitlinear-20261016

BRIEF for approval 1, revision 4; no approval is created. The preceding revision 3
was installed at the campaign root at 18:07Z on 2026-10-20 (notes/LEDGER.md).
Its file SHA256 is fadec31f36b55dfa1ad924b5f5f7b2301f06e93c6d7a5933f810bd75b183de98;
its authority SHA256 is c6fade113530f130c19765d370e103e88bd7fc1aa89d857d82bdefee7b5642b1
(preflight/b18f/validate3.txt). It is preserved as BRIEF.candidate.v3-draws-only.md.
Review this revision before installation; the orchestrator then installs the reviewed text,
keeps ADMISSION.json byte-unchanged, and records the installed identity in PREFLIGHT.
If Kai selects another product, regenerate with prepare_candidate.py --gpu-product NAME,
re-validate and review the resulting authority before installation and presenting approve.
Kai's recorded approval intent does not replace his final harness approval.
Before giving Kai the approve command, the orchestrator must obtain and record Kai's
ruling on the held-out Mutagen incident (notes/flags-for-kai.md item 0) and his
GPU product-of-record choice in decisions.md, before presenting the approve command.
STUDY v0.15 §14 blocks launch for hidden test access; every B19 Job, including a draw,
is a launch. Only after that ruling and the applicable review/launch preconditions does
Kai run lab-root `python3 tools/harness.py approve <campaign>` at a terminal.
No draw precedes that approval (STUDY v0.15 §11).

The question and frozen scientific rules are in review/snapshots/v0.15/STUDY.md. Code arrives
only through `run_handoff.py` bundle mode as a complete, immutable git bundle on kai-data; every durable NRP output is under
/data/cbl1016/. Code bundle/checkout, archives, venv, handoffs, runs, checkpoints, fronts, epoch records,
logs and W&B offline files all stay there. No ConfigMaps, hostPath, other PVC or object storage.
Only kai-data and transient emptyDir caches are mounted (Kai, 2026-10-19,
notes/build-todo.md). No W&B credential is requested: W&B remains offline on the PVC, project
bnjettag-chang-bitlinear, group chang-bitlinear-20261016 (STUDY §12).

First approval scope — U1 initial-tensor draws ONLY:
This approval covers mode `draw`, seeds 1–3 exactly as ADMISSION.draw_seeds, on one GPU
product of record, for the record and smoke archives. It authorizes no non-draw config.
Product-of-record field: NVIDIA-GeForce-RTX-4090 (Kai's choice, decisions.md 2026-10-20 (4)).
Kai is the decider: campaigns/2026-09-29-gpu-benchmark/VERIFY.md:9 says
"The choice of product is Kai's"; docs/infrastructure/gpu-selection-policy.md governs
the selection basis. All VERIFY.md line references below refer to that benchmark.
VERIFY.md:22–24: at each assumed G the pre-registered Chang rule favours a single
product: the A100-SXM4-80GB at G = 1 (one GPU under the quota) and the RTX 4090
at every G from 2 to 16; the A10's projected finish is 1.651 to 1.876 times as long
as the 4090's.
VERIFY.md:31–32: G_p (the GPUs that can actually schedule) was not measured;
the pools differ widely and the 4090's pool is the smallest of the leaders.
The static-ceilings table's row values for schedulable nodes and allocatable GPUs
are quoted exactly below (VERIFY.md:368,370,372–373):

| product | schedulable nodes | allocatable GPUs on them | source |
| --- | --- | --- | --- |
| RTX 4090 | 3 | 16 | VERIFY.md:373 |
| RTX 3090 | 30 | 115 | VERIFY.md:372 |
| A10 | 30 | 237 | VERIFY.md:370 |

These are static upper bounds on G_p assuming empty nodes, not measured G_p
(VERIFY.md:345–346). VERIFY.md:346: the benchmark's own pods waited hours for
one GPU of the 4090, 3090, A6000 and A100.
VERIFY.md:707–710 ("What needs Kai", item 1): the 4090 leads from G = 2 and its
pool is 3 schedulable nodes; both of its benchmark pods waited hours. The 3090's
pool is far larger; the break-even table shows how many 3090s match a given
number of 4090s. Options: the [D4] probe at decision time
(`code/gen_bench.py --probe`), or a choice with the table in hand.
The 4090's class-E K=5 peak of 91.9 % (E enters T at K=4 only, VERIFY.md:18–19)
is a separate memory caution for N=64 class-E pods at K=5
(benchmark STUDY.md:42,105). This prior evidence does not certify this campaign's
pack size or memory use.
This is an explicit departure from STUDY v0.15:1342 and the comparative
same-workload selection gate in gpu-selection-policy.md:32–69: no same-workload
measurement for this campaign exists before record draws on the product. Benchmarks
require the record archive (code/nrp/spec.py:67) and its product-specific init root
(:104–105), while gen/inputs.json holds one init_tensors root per generation.
For approval 1 Kai therefore chooses the product first; B19 under approval 2 measures
on that product and can confirm or reject it. If the measured rule later rejects it,
record draws are redone on the winning product under a new approval; nothing trained
on the rejected product's draws is production.
At c4dc2a80ca2b, gen/inputs.json leaves init_tensors root/platform,
smoke_init_tensors root/platform and bench_epochs null. Every non-draw config therefore
refuses launch: check_launchable rejects missing init_tensors.manifest_path (and epochs
for benchmarks); nrp/spec.py:98 runs the check even for engineering, and :104–105 checks
the product/archive root. See gen/configs.py:389–392 and NOTES-B18.md:36,166–167.
Smoke, kill-and-resume, determinism record, packing benchmark, measured rate and
production certification are outside this approval. The later approval sequence is below.
STUDY v0.15 remains the bound snapshot by design for approval 1, consistent with the
committed configs and MANIFEST. STUDY v0.18 has passed review; its stamp moves only in
the rebind commit and next BRIEF. No binary identity is admitted by this approval.
HOLD: all S1 production submissions, CH-8 and F000 included, are refused on the adapter
path. nrp/spec.py:63–64,94 permits an S1 config only in mode `pack`; campaign.py admit()
(:79–89) refuses pack while ADMISSION.production_certification is null. Any ADMISSION
change breaks its frozen authority hash (campaign.py:192–194; lab harness.py:849–852)
and needs new approval. These are code guards; the scientific_settings scope describes
them. Direct lab-harness submission is outside this BRIEF.

Authority and accounting:
- 400 GPU-h total, with a ring-fenced 50 GPU-h reserve and at most four unfinished GPU Jobs,
  including pending Jobs (STUDY §10, Cap; harness concurrency semantics). Training has 350 GPU-h.
- The required screen has 51 runs: CH-8 3; F000/F100 6; six factorial configurations in two forms
  36; PID pair 6. All 66 including controls do not have automatic admission (STUDY §10 table).
  Required runs keep all seeds and 7000 epochs. Measured per-class all-inclusive rates reserve
  the required set first; there is no 1.10 multiplier on measured rates (STUDY §10 Formula/Admission).
- ADMISSION.json is installed beside BRIEF.md, outside the code bundle. Its bundled
  code/tools/admission.json source remains the 84-config catalogue (51 required and 33
  engineering identities). The installed file is byte-unchanged under Kai's D1 decision:
  exactly 13 identities, CH-8/F000 seeds 1–3, smoke, benchmark and determinism configs
  (two CH-8 seed-1 replicas and one F000 seed-1 run, 20 epochs). These identities name
  what the later rebind will admit; none is launchable under approval 1. Draw admission
  checks draw_seeds only, not admitted_configs (campaign.py:73–78); draws use the data
  preparation manifest, not S1 configs (nrp/spec.py:76–85).
  Its retained source text is historical D1 provenance, not a release of the S1 hold.
  Every draw still needs final Kai approval, runner support and applicable gates.
  No control or S4 launch is admitted. S4 options A/B/C add 20/32/16
  runs and require Kai's separate approval (STUDY v0.15 §10 table and cap).
- Initial tensors follow U1: one CH-8 draw per seed and archive on the product of record, one
  visible GPU, saved under initial/<product>/<record|smoke>. Smoke tensors never seed record runs.
- Every new Job reserves its deadline-based GPU bound. Engineering, failed-Job infrastructure
  retries and restarts debit the reserve; ordinary first production attempts debit training.
  The adapter accounts GPU allocation from pod status.startTime through the last init/application
  container termination, including init time (B17-12). D2 (Kai, 2026-10-20, decisions.md
  "2026-10-20 (2)") is implemented in 84f372fbceac, tools/campaign.py budget() and
  install_guards(): a Job observed terminal with nonempty pod evidence and every pod's
  allocation_started False is charged 0 GPU-h; failed Jobs with final pods charge measured
  time. Live or unknown lifecycles keep the full bound. Failed and pending Jobs remain
  in the ledger; absent pod evidence does not establish zero use. No quota is invented.
- Kai's "if things fail or queue fails, just loop until it succeeds" (notes/build-todo.md,
  B17 correction) authorizes infrastructure relaunches without a count limit: eviction/preemption,
  node failure, image/volume mount failure, scheduling/pending timeout, and activeDeadlineSeconds
  exceeded while observed pending with no training start. Each must pass the 350/50
  GPU-h budget buckets and G80. Existing runs with a valid checkpoint resume through M14.
  B18P preserves a launch-identified directory killed before checkpoint publication in a dated
  .pretrain sibling and starts with its original GPU product and checkpoint interval; no evidence
  is deleted. Corrupt published checkpoints, terminal failures and legacy directories without
  recoverable launch identity are refused. Verified completed siblings remain complete.
  Code/integrity exits, refused configs/resumes, hash/provenance mismatches and gate failures
  stop the lineage and preserve evidence for a new committed fix, checks and handoff (STUDY §11).
  Automatic agent repairs remain disabled (B17-2).
- BRIEF authority scientific_settings.b17_retry_policy sets infra_relaunches_max to null (no
  count limit) and infra_relaunch_notify_after_24h to 6. More than six successful infrastructure
  relaunch submissions of one pack in a rolling 24 h notify Kai through the approved local
  NOTIFY.log file and keep going. The threshold is a notification, never a stop or approval gate.
  The protected map carries this operational policy solely for compatibility with the unchanged
  lab schema. Legacy top-level retry caps are zero: direct lab-harness retry is disabled; the
  campaign adapter implements unlimited retry decisions. No scientific recipe changes.
- Every wave and every resubmission checks namespace-wide kai ConfigMaps against distinct
  ConfigMaps mounted by unfinished kai Jobs. In bundle-only operation both must be zero.
  A mismatch refuses submission; the cluster owner reconciles it (§11 ConfigMaps; G80).

Operations:
Use `code/tools/campaign.py --lab-root <lab> submit <campaign> <handoff>` and the same adapter's
`tick <campaign> --holder cron`. It uses the unchanged lab harness's approval, ledger, lifecycle
and retry logic, adding identity, split-budget and G80 checks to every submit, including tick's
relaunch path. Direct lab-harness submit/tick bypass those added checks and is outside this BRIEF.
The adapter replaces the later-stage-only reserve calculation with the two simultaneous buckets
above; both use 400/50 from §10. It does not install an admission controller.

Initial prepare: from code/, `python3 -B -m tools.operations prepare --campaign-dir <campaign>
--spec <pack.json> --bundle <bundle> --bundle-path /data/cbl1016/code/<name>.bundle --commit <sha>
--data-info <data_info.json> --launcher <lab>/tools/run_handoff.py`.
Relaunch/evaluate/workspace/repair-check argv are protected authority keys below because the
current harness schema does not accept commands in the operations block. They prepare a new
Job, record lifecycle success without physics scores, verify the source checkout, and refuse
automatic code repair, respectively. After approval 2, benchmark rates: `python3 -B -m tools.bench_rates <capture>
--lifecycle <captured-lifetime.json>`. Evaluation does not open the held-out archive.

Operations numbers not specified by STUDY: lease 3600 s (B17-3, local scheduling default),
notification threshold 6 relaunches per 24 h (Kai B17 correction, B17-1). Disabled capabilities use 0; factors use 1. The unused score deadline 600 s is
the harness schema minimum (B17-3). `deadline_seconds_per_epoch` 3 is STUDY §10's planning rate,
not a measured deadline: the Job generator requires explicit resource/deadline inputs and never
uses it. No borrowed paused-campaign timing or resource estimate certifies this workload.

All generated handoffs stay pending until gate references and authority are reviewed.
Every create must pass `nrp_doctor.py` lint through a path where the hook runs; the recorded
kubectl grant does not waive lint, immutable handoffs, the approval, or §14 stops. No cluster
action is performed by this B18f preparation. Production
certification is null in ADMISSION.json: after approval 2, B19 must supply product/K-specific measured rates,
required-screen projection, rolling three-hour utilization, device/host memory and product gates.
Updating the admission file changes its frozen hash and requires a new authority approval.
ADMISSION.json lives outside the code bundle and is installed beside BRIEF.md. Its frozen hash
binds preflight/b18f/ADMISSION.json, the D1-filtered installation candidate; it differs from
code/tools/admission.json's frozen hash. The bundled template stays unchanged. This preparation
produces the review candidate; the orchestrator installs it only after re-review,
preserving the installed ADMISSION bytes and recording the actual installed identity.

B18 decisions and conditional authority (STUDY v0.15 §§10, 11, 14, 19.2; Kai, 2026-10-19):
- OD-13 confirmed: learned-width HGQ binary scale quantization with Chang's weight configuration;
  under BF it acts after the fold. OD-15 confirmed: three-seed screen without verdicts; verdicts
  require fresh-seed S4 confirmation. OD-12 confirmed: PID target 350,000 in headline H units,
  with native beside. Claim domain is (b), band_rule default, without (c): c3r false and
  c3r_level null (a set level is refused). Whole-D_conf is reported beside as the strong form.
- Recorded headline: H1 for every binary arm, F100 included. CH-8/F000 ramp penalties
  remain native and records retain H1/H2/NR. The export-headline recording condition is
  met; binary engineering waits for approval 2 and binary S1 for approval 3, each subject
  to Kai's confirmation of the BF owner record before any binary launch.
  The current code/configs have not yet received the headline commit. No penalty changes
  mid-run. PID needs its headline sensor in that future committed implementation.
- v0.4 checks whole trained F111-FB and F111-BF: normal and factored C-simulations must both
  be bit-exact against the same model on 4,096 validation jets; factored timing WNS >= 0 and
  II = 1; factored post-opt CLB LUTs strictly fewer than normal. Both routes use identical
  INLINE treatment, including CH-8 and F000. Normal timing is reported beside; requiring
  normal timing too remains a Kai item. This is pre-place-and-route evidence. Missing reports
  cannot establish a pass. Failure means a completed failed criterion or an owner-recorded
  impossibility with reason, reported to Kai before binary launch. The recorded outcomes and their sources are bound below.
- Both forms failed, so the split-form rule (Kai, 2026-10-19T16:28:12Z) selects H1
  for every binary arm. The one-form-passes provisions do not arise: no 24 added H2 runs,
  no §19.2 item 5 decision, and no 531.25 GPU-h cap under this outcome.
- Baseline planning: 51 required runs, 357,000 run-epochs, 327.25 training GPU-h plus 50 reserve
  = 377.25 projected GPU-h, within the 350/50 buckets and cap 400. Planning rate is 3 GPU-s per
  run-epoch with factor 1.10; all-inclusive measured rates omit that factor. The required screen
  fits at about 3.2086 planning or 3.529 measured GPU-s/run-epoch. All 66 screen runs project
  to 473.50 GPU-h with reserve and are not automatically funded. No measured rate is claimed.
- item9 stays hold: reserve the required set first; controls wait for S2. The first control slot
  is DS-HGQ if BEST has a degenerate attention layer at any frozen screen-seed point, otherwise
  FP32; further slots go to the other, W8A8, GR, BE. Drop in reverse order when the cap binds.
  No controls are currently
  admitted by ADMISSION.json. Never cut factorial seeds, cells or 7,000 epochs to fit a budget.
  The benchmark can run only after the rebind and approval 2. If its measured required-screen cost exceeds
  the cap, stop before production admission for Kai's cap/scope decision under STUDY v0.15
  §19.2 item 8; this is not a pre-approval benchmark requirement.
- s4_option stays null until after S2 and before any held-out read. A/B/C add 20/32/16 fresh-seed
  runs (A seeds 4–8; B/C seeds 4–11), respectively 128.33/205.33/102.67 planning GPU-h. S4 requires
  separate funding/approval, never leftover-hours funding. No S4 means no confirmatory claims.
- Missing data identity, hidden held-out access, reference/CH-8/F000 mismatch, failed gradient,
  cost, RNG or PID gates, missing initialization readings/review/authority, and unsupported budgets
  remain §14 stops. Nonfinite values, corrupt artifacts, data/hash drift and accounting failures
  stop affected runs; poor accuracy does not. Chang's native-count StopIf < 10,000 remains.
  Preserve prior valid checkpoints; no automatic scientific repair, seed replacement or fresh
  initialization. Empty D_screen blocks S3/S4 until Kai decides. B10-F1 is merged in
  66449e9a94fa; the loss guard applies to every run, CH-8 included (Kai B10F1-1,
  2026-10-19; STUDY v0.15 §7.3). The historical native_share_removed diagnostic and
  B18P deferred-check notes remain in code/NOTES-B18.md; current gate results must be
  read from the final-commit record below.
- Mulder is permitted for the export check only; later deciding-point synthesis needs renewed
  permission. All production training, including CH-8/F000, is held for approval 3.
  Default deciding-point C4 remains on the normal route; optional factored builds are context.
- Per-draw operator check (critical-PREFLIGHT-v2 C4): before EVERY draw handoff,
  verify and record code commit c4dc2a80ca2b8726492f9c49d5de96b3a068f128 and bundle SHA256
  bbc14fe8ff7b6550369211cc729230fbb7d6430e321b378ffd038384a610d212.
  Refuse a mismatched handoff: draw manifests record code hashes that the rebind will bind.
  This is an operator check, not a newly implemented code guard. Keep code frozen until
  draws end; no handoff may substitute the future rebind commit.
- Packing: the approval-1 product-first departure is stated above; approval 2 measures
  product/K against the same-workload rule, >=40% rolling three-hour GPU
  utilization, <=90% device memory and safe full-horizon host-memory projection. Preserve each
  seed's product and the recorded restart interval. B19 supplies certification; no deadline or
  throughput is certified by this candidate. The B17 launcher resource-ratio issue is
  recorded in code/NOTES-B18.md; final/GATES.md records G77 nrp NOT RUN
  (cluster actions excluded). No upstream resolution is asserted.

B19 launch order — three approvals (critical-PREFLIGHT-v1 A1):
1. BRIEF for approval 1, revision 4: Kai selects the U1 GPU product before drawing;
   product-of-record field: NVIDIA-GeForce-RTX-4090 (Kai's choice, decisions.md 2026-10-20 (4)).
   Kai is the decider: campaigns/2026-09-29-gpu-benchmark/VERIFY.md:9 says
   "The choice of product is Kai's"; docs/infrastructure/gpu-selection-policy.md governs
   the selection basis. All VERIFY.md line references below refer to that benchmark.
   VERIFY.md:22–24: at each assumed G the pre-registered Chang rule favours a single
   product: the A100-SXM4-80GB at G = 1 (one GPU under the quota) and the RTX 4090
   at every G from 2 to 16; the A10's projected finish is 1.651 to 1.876 times as long
   as the 4090's.
   VERIFY.md:31–32: G_p (the GPUs that can actually schedule) was not measured;
   the pools differ widely and the 4090's pool is the smallest of the leaders.
   The static-ceilings table's row values for schedulable nodes and allocatable GPUs
   are quoted exactly below (VERIFY.md:368,370,372–373):
   
   | product | schedulable nodes | allocatable GPUs on them | source |
   | --- | --- | --- | --- |
   | RTX 4090 | 3 | 16 | VERIFY.md:373 |
   | RTX 3090 | 30 | 115 | VERIFY.md:372 |
   | A10 | 30 | 237 | VERIFY.md:370 |
   
   These are static upper bounds on G_p assuming empty nodes, not measured G_p
   (VERIFY.md:345–346). VERIFY.md:346: the benchmark's own pods waited hours for
   one GPU of the 4090, 3090, A6000 and A100.
   VERIFY.md:707–710 ("What needs Kai", item 1): the 4090 leads from G = 2 and its
   pool is 3 schedulable nodes; both of its benchmark pods waited hours. The 3090's
   pool is far larger; the break-even table shows how many 3090s match a given
   number of 4090s. Options: the [D4] probe at decision time
   (`code/gen_bench.py --probe`), or a choice with the table in hand.
   The 4090's class-E K=5 peak of 91.9 % (E enters T at K=4 only, VERIFY.md:18–19)
   is a separate memory caution for N=64 class-E pods at K=5
   (benchmark STUDY.md:42,105). This prior evidence does not certify this campaign's
   pack size or memory use.
   No same-workload measurement exists yet: choosing first departs from the
   STUDY/policy comparative gate; B19 measures on it under approval 2.
   If rejected by the measured rule, redo record draws on the winning product under
   a new approval; nothing trained on rejected-product draws is production.
   Mode draw only, seeds 1–3, record and smoke archives. Record Kai's product choice
   alongside the flags item 0 ruling in decisions.md before presenting approve.
   Every B19 Job is a launch.
2. Rebind commit: bind the draw outputs in gen/inputs.json (init_tensors and
   smoke_init_tensors roots/platforms for that product), set bench_epochs, regenerate
   configs, and stamp reviewed STUDY v0.18, sha256
   027ff590825922df3b1090a96eb9bf000dd05e82920d19daf9ae504933cbfc68.
   Include headline H1 for every binary arm per the recorded outcome, both test fixes
   (file-audit c4dc2a80ca2b and loss-guard stand-in ee30820d2ffa), and the G57
   post-outcome branch. The stand-in fix was first committed as ee30820d2ffa,
   then retained in amended 8626e4b (LEDGER 18:01Z), outside this approval's code.
   The amended message withdrew the causal claim but retains the overbroad word
   "deterministic"; remove that residual when finalizing the branch rebind (C9).
   The rebind commit's gate pass must precede BRIEF for approval 2, which binds those new identities.
3. Approval 2 on BRIEF for approval 2: smoke, kill-and-resume, determinism record, packing benchmark
   and measured rate. S1 stays held. No approval-one wording authorizes these activities.
4. Record the resulting product/K production certification in ADMISSION.json; this
   changes its frozen hash and requires another BRIEF authority and approval.
5. Approval 3: production S1, all arms, subject to the applicable gates and certified
   admission; binary arms also require Kai's confirmation of the BF owner record.
   Controls and S4 retain their separate STUDY conditions and funding/approval.

Final identities and frozen-file rule:
- Code is main at c4dc2a80ca2b8726492f9c49d5de96b3a068f128, verified by read-only git. It follows
  29a459ce7ee6 by seven commits. Relative to the v1 candidate's 38f4ebae3874,
  the sole added commit is c4dc2a80ca2b, "tests: the file-audit link test runs its audited
  children from tmp_path, outside any sync root [B18fix]", changing tests/test_data.py only.
  All 55 frozen_files hashes are recomputed and unchanged; the changed test is outside
  the frozen set (the three explicitly frozen test dependencies remain in that set).
  The code/STUDY/MANIFEST/ADMISSION/bundle bindings remain unchanged from v2.
  This re-prepare changes approval scope and evidence wording, not the frozen inputs.
- STUDY of record: review/snapshots/v0.15/STUDY.md; sha256 e4a6b89e0adbe3ab14b508a24335688fba48db3f801c4d1015812cc9dcd73d8e.
  configs/MANIFEST.json at this commit stamps v0.15; its sha256 is 670fcf5fe6d818ae834bfb41f7c546157e8137422eff5b0e20c877fdcf56d85f.
- Bundle: preflight/pc-setup/code-main-c4dc2a80ca2b.bundle; sha256 bbc14fe8ff7b6550369211cc729230fbb7d6430e321b378ffd038384a610d212.
  shasum and git bundle list-heads verify these bytes and main at the final commit.
  staging/STAGING.md records equal in-pod SHA256 at
  /data/cbl1016/code/cbl1016-main-c4dc2a80ca2b.bundle on kai-data;
  staging is cited from that record, not rechecked against the cluster here.
- Retain B18: all direct *.py in cbl, gen, nrp and tools, plus tools/admission.json, configs/MANIFEST.json, gen/inputs.json, gen/protocol_text.json and manifests/data-prep-150c.json; extend with gates/syncroot.py, tests/b17/lab_root.py, tests/test_resumegate.py and tests/test_syncroot.py for the final refusal/comparator dependencies. cbl/lossguard.py is automatically included. Add external ADMISSION.json separately.
  Total: 55 entries, of which 54 are committed code blobs hashed by
  git show c4dc2a80ca2b:<path> piped to shasum -a 256. ADMISSION.json is hashed from the
  prepared installation file, not the bundled template. Individual generated configs are
  covered transitively by the committed MANIFEST; the complete bundle binds the rest.
- ADMISSION.json sha256 160ba907aa7ab254ead671371956f150286c62bd08c834956fde467ec55e1c81; bundled template sha256
  927a2735366fadd1eaa5fa852eb53b410064d5c19b37c6f0acd7b4819f4b7ab1. These hashes intentionally differ (D1).

Gates of record and incidents:
- Full B18f gate run: preflight/b18f/final/GATES.md, 38f4ebae3874,
  updated 2026-10-20 17:00:19Z: FAIL 4, NOT RUN 34, PASS 91, PASS WITH SKIPS 1.
  The four FAIL rows are tests-b1/b2/b5/b10 pcc on the file-audit link test;
  tests-b1 also had the stand-in failure documented below.
- Targeted rerun: preflight/b18f/targeted/GATES.md, c4dc2a80ca2b,
  updated 2026-10-20 17:21:38Z: tests-b2, b5, b10 pcc PASS; tests-b1 pcc FAIL,
  1 failed / 671 passed / 1 skipped. The file-audit test now passes everywhere in
  the four targeted suites. Other rows stand at 38f4ebae3874 by the orchestrator's
  targeted-rerun decision; this is not a new full all-gates pass at c4dc2a80ca2b.
- Targeted-rerun reason, verbatim from the header:
  Only tests-b1, tests-b2, tests-b5 and tests-b10 pcc were rerun, once each, serially. The fix changes only `tests/test_data.py`: two line replacements add `cwd=str(tmp_path)` to audited child calls. All four rows failed that Linux file-audit link test at `38f4ebae3874`, motivating this targeted scope.
- Header correction, verbatim:
  It was not the only failing test in tests-b1. Its prior stdout also records `tests/test_lossguard.py::test_the_stand_in_reproduces_the_drop_without_the_guard` (2 failed, 670 passed, 1 skipped). The other three prior rows each failed only the file-audit link test. Historical evidence and `b18f/final/` are preserved.
- Recorded exception for THIS draw-only approval: The single remaining failure is tests/test_lossguard.py::test_the_stand_in_reproduces_the_drop_without_the_guard, asserting `3 < 3`, a test of the B10-F1 bug STAND-IN (not of the guard; the guard's own tests pass: tests-lossguard pcg 10 passed at 38f4ebae3874); it also failed at 38f4ebae3874 and passed at 29a459ce7ee6; why it passed at 29a459ce7ee6 is not established. The stand-in keys its map by REAL_ID(obj) without retaining objects, so address reuse corrupts its synthetic-ID bookkeeping (instrumented retrace [3,120]: one duplicated and one missing term). This is repeatable under the exact runner but allocator/lifetime-sensitive, not universally deterministic. The guard is not implicated. The branch fix ee30820d2ffa retains (obj, synthetic_id) per scope; first committed as ee30820d2ffa, retained in amended 8626e4b per LEDGER 18:01Z. That message withdrew the causal claim but its residual "deterministic" must be removed when finalizing the branch rebind (C9). It is not in the bound c4dc2a80ca2b commit.
  Source: preflight/b18f/codex/codex-lossguard-test-report.md and the branch fix.
  The rebind gate pass and BRIEF for approval 2 must carry its disposition. This exception does not
  release any non-draw work; approval 1 runs no guarded training.
- Orchestrator rows at c4dc2a80ca2b:
  preflight/gates/tests-offline/c4dc2a80ca2b/mac-orchestrator:
  643 passed, 4 skipped, 26 deselected; and
  preflight/gates/G72/c4dc2a80ca2b/mac-orchestrator: PASS.
  Exit codes and git identities were checked; these results were read, not rerun.
  Four offline skip reasons are unverified in the recorded stdout; skips are not passes.
- Incident source: preflight/investigations/G13-G51/TICKET.md §§1 and 4.1. Gate work
  directories were inside the Mutagen-synced tree. During G51's roughly 30-second hard-link
  exposure, Mutagen read held-out archive bytes; no campaign code path read them. The
  investigation found no remaining copy; whether any transient transfer reached the Mac
  is undetermined. Kai's ruling on STUDY section 14 remains pending. This candidate does
  not resolve or waive that incident. The orchestrator must obtain and record Kai's
  flags item 0 ruling before giving him the approve command (STUDY v0.15 §14,
  hidden test access); even an initial-tensor draw Job is a launch.

Export check outcome of record:
- notes/LEDGER.md, "EXPORT CHECK OUTCOME", 2026-10-20 17:18Z;
  preflight/export/codex/codex-export-eval-report.md; preflight/export/PREFLIGHT-rows.md.
- FB FAIL, criterion 2: factored post-opt WNS −1.090 ns; bit-exact yes on 4,096
  validation jets; II=1; CLB LUT 394,351 < 402,252. These are engineering results,
  pre-place-and-route, not a physical implementation or physics result.
- F111-BF: recorded as a failure by the owner record of 2026-10-20T17:30Z; no C-synthesis result.
  This is the wording of reviewed STUDY v0.18 §3.4; approval 1 still binds v0.15.
  The owner recorded impossibility within the synthesis budget after the 8 h and
  86,400 s attempts (flags item 0b). No BF timing or LUT value is available.
  Owner record made; Kai's choice between the default and a 72 h attempt is pending.
- Both fail ⇒ H1 headline for every binary arm, F100 included. The hold's condition
  "until the export check's headline is recorded" is now met in the record.
  Binary engineering requires approval 2 after the v0.18/H1 rebind and its gate pass;
  binary S1 requires certified ADMISSION and approval 3. Any binary launch also waits
  for Kai's confirmation of the BF owner record. The one-form-passes provisions do not arise
  under the currently recorded outcome.

Number provenance and validation scope:
- Scientific constants and rules above are quoted from STUDY v0.15 §§3.4 (rule v0.4),
  7, 10–14 and 19.2; the split-form decision is decisions.md 2026-10-19T16:28:12Z.
  All otherwise unversioned STUDY citations in this brief mean v0.15. Planning arithmetic
  uses §10's runs × 7,000 × 3 × 1.10 / 3,600; measured break-even drops 1.10.
  Counts and calculations are reproduced in preflight/b18f/checks.json. No rate is measured here.
- Retained catalogue identities and determinism epochs are checked against the committed configs;
  catalogue counts come from code/tools/admission.json. Operational defaults and schema
  bounds retain preflight/b18/BRIEF.candidate.md's B17-1/B17-3 citations and are checked
  against lab tools/harness.py. Dates/commit labels are evidence identifiers, not measurements.
- Harness validate checks authority/operations keys, types and ranges and reports matching
  approval status. It does not validate frozen-file contents or establish gate passage.
  This preparer separately recomputes frozen hashes and validates admission scope; an exact
  validation/BRIEF.md copy is used. Approval is interactive and authority-bound; none is created.
  The draw outputs, rebind commit and its gate pass, approvals 2 and 3, product/K
  certification, and Kai's section 14 ruling remain outstanding. No launch authority is
  granted until Kai approves this candidate after that ruling and applicable preconditions.

```json authority
{
 "agent_invocations_max": 0,
 "approval_ref": "CBL1016-B18F-U1-DRAWS-ONLY: BRIEF for approval 1, revision 4, mode draw, seeds 1-3, record/smoke archives on one product of record; Kai product choice and flags item 0 ruling before approve command; non-draw engineering needs approval 2; S1 needs approval 3",
 "campaign": "chang-bitlinear-20261016",
 "candidate_attempts_max": 0,
 "candidate_check_command": null,
 "candidate_paths": [],
 "candidate_prepare_command": null,
 "candidate_protected": [
  "**"
 ],
 "codex_model": null,
 "dispatch_actions": [],
 "elapsed_behavior": "stop_new_submissions",
 "elapsed_days_max": null,
 "evaluate_command": [
  "python3",
  "-B",
  "-m",
  "tools.operations",
  "evaluate",
  "--campaign-dir",
  "{cdir}",
  "--job",
  "{job}"
 ],
 "frozen_files": {
  "ADMISSION.json": "160ba907aa7ab254ead671371956f150286c62bd08c834956fde467ec55e1c81",
  "code/cbl/__init__.py": "bd653c5655f3fdc04049fb7173fb261da6083ae03578af3eb9cb058d5eb7b2fa",
  "code/cbl/ablate.py": "911185dfb76d703519ab01d14ed879146559551ef6738465c509618110d75ef0",
  "code/cbl/binary.py": "1cb341e97054a4be9f93b8ca9a8b303f45989d2d633a46c37e789129b4fbb722",
  "code/cbl/confirm.py": "675b95eb3bf69e6e79aab9cb44496d185a3f6dff078d31578a6a6cafe4044883",
  "code/cbl/cost.py": "d432071beadac00a780a932d120312b0b707d05f310d0b74f0862288c7940801",
  "code/cbl/cost_pure.py": "ac2d519c2f48dc7ad00446cb4f022a1b4174e68a227cbe79a1a01b519acbb62c",
  "code/cbl/data.py": "fbd308154c8e96714ff865438c016cd72c24f5cd25352e55801b8b00126130d4",
  "code/cbl/env.py": "19ffd18d0fb48317568d199d65f874023bd13e9150295e9c1573e8706b2b2baf",
  "code/cbl/epoch_diagnostics.py": "f0980d925a2610785c908f41e12a07139a6d88743e7c869283c762fc61545f27",
  "code/cbl/front.py": "a175dfaf6e1aff54a7e7dbdb5a54236ddd2fc1bc426bab222f7cc3e190c30e3a",
  "code/cbl/frozen_eval.py": "3d0cc8f2b4d22f0ffe98ca969cc2321a479defe70e1199ebb43c029cca8054f7",
  "code/cbl/lossguard.py": "ed3714577c28b4aa9e632d5658322e265e8def39ae08f83efcb2c1a12cd6511e",
  "code/cbl/model.py": "e80f5b8c90a59222ed781564b3b41b66ac4a54cbcdb33493e854d49ced16a001",
  "code/cbl/pid.py": "1c4c87ac64741e94a12032cfdcd6a5286c77f2b9615718406e518e000be0b28a",
  "code/cbl/refdiff.py": "36ea10f90ae11d40aa0f30c142fbeef64f5c0fe0e493bc948f0a312e56c11aab",
  "code/cbl/resume.py": "c9b6282e6b767520a4570319b686e7f9eeab80065ecde984789afcbbe7c12272",
  "code/cbl/rng.py": "fae57ea950c93fc44a7b621ea95482cdab0c52e162a9a078d62eb03429e0fd78",
  "code/cbl/runner.py": "009f79fb0196ae0de2616ed00f070b7d894fcd83c56f1c22724efd456a9023a8",
  "code/cbl/s2.py": "3dac2010f6e57a58e3a6098b5bc097e687354cbce5d8d10aaf84652a5a9f06b4",
  "code/cbl/select.py": "95e3aadf98f73dabf894b1263abb5a937163fe8a8fbf8aa82a5306215f8c60c4",
  "code/cbl/stats.py": "0dfa2aefd826e23d9495c2cea400ba58195fb82be17b162dc3991c7f9b74d591",
  "code/cbl/structure.py": "edd8b008faf08b9280dfe4a254750e39474b1613f9f4ebcdf2b870883422cb5b",
  "code/cbl/wandb_log.py": "2653119e4dd8eebe8c360f4e2890810d3fe36c669be1244c3f79c5e54822235b",
  "code/configs/MANIFEST.json": "670fcf5fe6d818ae834bfb41f7c546157e8137422eff5b0e20c877fdcf56d85f",
  "code/gates/syncroot.py": "0257f7565cfe1fed8647489d992b716d9e188d3b58a3db6fe8e247ffee9fd4d8",
  "code/gen/__init__.py": "895b4568cabc89da1edf079597d645c164521642fafe67fc354eec2772cd8d66",
  "code/gen/arms.py": "c79f49e9874ef63eba475ebff95cea40a87d62057018dafe26ef0d8dbc8e611d",
  "code/gen/cli.py": "5d537ee3a55561b352ba78f5d1956762eee12146b0f587ce1572c1014150dbed",
  "code/gen/configs.py": "813e73c331aa34cbc242af2a5e44eca61718b1e608d2821cdda6d93c2ba00027",
  "code/gen/inputs.json": "730af938b56413682f0cf97eb6ca8c46cc72049cdbb1ebfd7c2b3a5694cd3a75",
  "code/gen/inputs.py": "f91d0c92a0e68af47cf47e4248d713b34442d5311e962566342b1be8961fdfa8",
  "code/gen/protocol.py": "1547c3f42222dd7c0a5b0468bb02b222a62a7f55473e0f07e6986a115f5b7b7a",
  "code/gen/protocol_text.json": "40da790f30b5c3874f8927d97256eb14e7d38d4d75ed656d33afd021c25909e2",
  "code/gen/release.py": "f5945442cd4e48eea4d7a44faf89834636ed0b988ed675ba965da8c7fb21e103",
  "code/manifests/data-prep-150c.json": "686b15db996010cc5eadb7556397488c92ce3b48c2bccdb2a0d947f1011a7f13",
  "code/nrp/__init__.py": "0d2296b7c8fd93f96ca4df3de3db648f0b80417b0c3b690af75461cf4f70477c",
  "code/nrp/accounting.py": "2daf6b5085258e3656fa54bc551e86789ed9eb8de710719c1fa8a2924412b6ac",
  "code/nrp/common.py": "278afc7648ea475701ba95e54a68e8a1889ee25c776d8d3064bbe6eceb0099d1",
  "code/nrp/jobs.py": "53805efab949d5dce2333465787c945a190ed253cfdb73df8753204a4126d5f3",
  "code/nrp/restart.py": "640dcc7eb0d81922af5b2c1aeae46d693afcc5b57475a23df48d413276df7de7",
  "code/nrp/run_pack.py": "220ce2ad490bd527cb9fe933c5383f40ed668f1662f978be5428d93bb5a9678d",
  "code/nrp/spec.py": "a4bab8cb7664f13ac7557caa74d3f30fd11cd8d5c923745a837da62fc9c28192",
  "code/tests/b17/lab_root.py": "20984c7d42a1f24f4075dbc4d9dc20a3086545b3d15d65a2464baa5cdb1e2784",
  "code/tests/test_resumegate.py": "80cfea1c003a0017f21dea5097ddfd5f0299052c71c8e8239cc4372199c5be4c",
  "code/tests/test_syncroot.py": "69d5a296b23f3eddc86e6e898d8b431b5dccf1d85c1f5852d583108dfd84656c",
  "code/tools/__init__.py": "fd435d3d4e68c78f9d3313d824ca0779b71b9c9e47ad09df1b65418a10bb7322",
  "code/tools/admission.json": "927a2735366fadd1eaa5fa852eb53b410064d5c19b37c6f0acd7b4819f4b7ab1",
  "code/tools/bench_rates.py": "9710f79a8311127d804d74275b6179ef42ba0f56bd1dd9a5af0b129ac00cad21",
  "code/tools/campaign.py": "f4b9e00f3cf7ae27a5726905630b429baf047cc550c7283511850b80c4f6848e",
  "code/tools/configmaps.py": "77dcfa63f7115c008e67bb52063e3f63c54e2f9ee3467250650406c6862e0812",
  "code/tools/determinism.py": "0cccec59b5e7fc00c3ebabeddc2764ab3319e0c1109a4054206bf2ec7d67fccc",
  "code/tools/draft_brief.py": "4f7eac59dd12e6b8d67ce10f3b06d2556fccfd1b3cf513623e25e02569366c40",
  "code/tools/operations.py": "82f7188a24ec59894a9c19aac316432c389161919ae5046860c70d0f23d65f6f",
  "code/tools/retries.py": "dc23fbeaa93cc959307889a6eb2b297f4eb1861c85c7ab0fd4845a5e6911af33"
 },
 "gpu_bound_mode": "hard",
 "gpu_h_max": 400,
 "implement_runner": "codex",
 "infra_relaunches_max": 0,
 "interpret_command": null,
 "interpret_paths": [],
 "max_concurrent_gpu_jobs": 4,
 "namespace": "cms-ml",
 "notify_channels": [
  "file"
 ],
 "prepare_command": [
  "python3",
  "-B",
  "-m",
  "tools.operations",
  "relaunch",
  "--campaign-dir",
  "{cdir}",
  "--job",
  "{job}",
  "--attempt",
  "{attempt}",
  "--cls",
  "{cls}"
 ],
 "repair_paths": [],
 "repair_test_command": [
  "python3",
  "-B",
  "-m",
  "tools.operations",
  "repair-check",
  "--campaign-dir",
  "{cdir}",
  "--job",
  "{job}"
 ],
 "repairs_total_max": 0,
 "reserve_gpu_h": {
  "reserve": 50
 },
 "review_runner": "codex",
 "runner": "codex",
 "runner_command": null,
 "runtime_factor_max": 1,
 "scientific_settings": {
  "admission_catalogue": {
   "config_paths": [
    "configs/S1/CH-8-s1.json",
    "configs/S1/CH-8-s2.json",
    "configs/S1/CH-8-s3.json",
    "configs/S1/F000-s1.json",
    "configs/S1/F000-s2.json",
    "configs/S1/F000-s3.json",
    "configs/bench/CH-8-s1.json",
    "configs/bench/F000-s1.json",
    "configs/det/CH-8-s1-r1.json",
    "configs/det/CH-8-s1-r2.json",
    "configs/det/F000-s1.json",
    "configs/smoke/CH-8-s1.json",
    "configs/smoke/F000-s1.json"
   ],
   "status": "13 unchanged identities for later rebind/admission; none currently launchable; draw admission checks draw_seeds only, not these identities"
  },
  "approval_scope": {
   "approval_2_activities": [
    "smoke",
    "kill-and-resume",
    "determinism record",
    "packing benchmark",
    "measured rate"
   ],
   "approval_2_requires": "Draw-output rebind in gen/inputs.json (record/smoke roots and platforms, bench_epochs); STUDY v0.18 stamp 027ff590825922df3b1090a96eb9bf000dd05e82920d19daf9ae504933cbfc68; H1 for every binary arm; both test fixes (file-audit c4dc2a80ca2b, stand-in ee30820d2ffa retained in amended 8626e4b; C9 message residual due at rebind); G57 post-outcome branch; regenerated configs; rebind gate pass; BRIEF for approval 2; approval 2",
   "approval_number": 1,
   "decision": "critical-PREFLIGHT-v1 A1/B1 and LEDGER correction 2026-10-20 17:58Z",
   "draw_archives": [
    "record",
    "smoke"
   ],
   "draw_seeds": [
    1,
    2,
    3
   ],
   "enforcement": "S1 configs require pack mode (nrp/spec.py:63-64,94). campaign.py admit():79-89 refuses pack while ADMISSION.production_certification is null. Any ADMISSION change breaks its frozen authority hash (campaign.py:192-194; lab harness.py:849-852) and requires new approval. Direct lab-harness submission is outside this BRIEF.",
   "modes": [
    "draw"
   ],
   "non_draw_refusal": "Frozen gen/inputs.json has init_tensors and smoke_init_tensors root/platform null, bench_epochs null; check_launchable and nrp/spec.py:98-105 refuse all non-draw configs (NOTES-B18.md:36,166-167)",
   "pre_approve_condition": "Orchestrator obtains and records Kai's product-of-record choice alongside his flags item 0 Mutagen ruling in decisions.md before giving him the approve command; STUDY v0.15 section 14 hidden test access blocks launch; every B19 Job is a launch",
   "recorded_exception": "The single remaining failure is tests/test_lossguard.py::test_the_stand_in_reproduces_the_drop_without_the_guard, asserting `3 < 3`, a test of the B10-F1 bug STAND-IN (not of the guard; the guard's own tests pass: tests-lossguard pcg 10 passed at 38f4ebae3874); it also failed at 38f4ebae3874 and passed at 29a459ce7ee6; why it passed at 29a459ce7ee6 is not established. The stand-in keys its map by REAL_ID(obj) without retaining objects, so address reuse corrupts its synthetic-ID bookkeeping (instrumented retrace [3,120]: one duplicated and one missing term). This is repeatable under the exact runner but allocator/lifetime-sensitive, not universally deterministic. The guard is not implicated. The branch fix ee30820d2ffa retains (obj, synthetic_id) per scope; first committed as ee30820d2ffa, retained in amended 8626e4b per LEDGER 18:01Z. That message withdrew the causal claim but its residual \"deterministic\" must be removed when finalizing the branch rebind (C9). It is not in the bound c4dc2a80ca2b commit.",
   "release_requires": "After approval 2 engineering evidence, certify product/K in ADMISSION.json, bind its new hash in a new BRIEF authority, then approval 3 for production S1 all arms; binary arms require Kai's confirmation of the BF owner record; controls/S4 retain separate conditions",
   "s1_hold": "ALL S1 production training submissions, CH-8 and F000 included",
   "scope": "U1 initial-tensor draws ONLY",
   "second_brief_requires": "Rebind gate pass including disposition of tests-b1 pcc stand-in failure",
   "study_of_record": "v0.15 by design for approval 1; reviewed v0.18 stamp 027ff590825922df3b1090a96eb9bf000dd05e82920d19daf9ae504933cbfc68 belongs in the rebind and BRIEF for approval 2"
  },
  "b17_retry_policy": {
   "infra_relaunch_notify_after_24h": 6,
   "infra_relaunches_max": null
  },
  "bundle_sha256": "bbc14fe8ff7b6550369211cc729230fbb7d6430e321b378ffd038384a610d212",
  "claim_domain": {
   "band_rule": "default",
   "base": "b",
   "c3r": false,
   "c3r_level": null
  },
  "code_sha": "c4dc2a80ca2b8726492f9c49d5de96b3a068f128",
  "config_manifest_sha256": "670fcf5fe6d818ae834bfb41f7c546157e8137422eff5b0e20c877fdcf56d85f",
  "eligible_now": {
   "activities": [
    "U1 initial-tensor draws on the product of record"
   ],
   "config_paths": [],
   "draw_archives": [
    "record",
    "smoke"
   ],
   "draw_seeds": [
    1,
    2,
    3
   ],
   "gpu_product": "NVIDIA-GeForce-RTX-4090 (Kai's choice, decisions.md 2026-10-20 (4)). Kai is the decider: campaigns/2026-09-29-gpu-benchmark/VERIFY.md:9 says \"The choice of product is Kai's\"; docs/infrastructure/gpu-selection-policy.md governs the selection basis. All VERIFY.md line references below refer to that benchmark. VERIFY.md:22\u201324: at each assumed G the pre-registered Chang rule favours a single product: the A100-SXM4-80GB at G = 1 (one GPU under the quota) and the RTX 4090 at every G from 2 to 16; the A10's projected finish is 1.651 to 1.876 times as long as the 4090's. VERIFY.md:31\u201332: G_p (the GPUs that can actually schedule) was not measured; the pools differ widely and the 4090's pool is the smallest of the leaders. The static-ceilings table's row values for schedulable nodes and allocatable GPUs are quoted exactly below (VERIFY.md:368,370,372\u2013373): | product | schedulable nodes | allocatable GPUs on them | source | | --- | --- | --- | --- | | RTX 4090 | 3 | 16 | VERIFY.md:373 | | RTX 3090 | 30 | 115 | VERIFY.md:372 | | A10 | 30 | 237 | VERIFY.md:370 | These are static upper bounds on G_p assuming empty nodes, not measured G_p (VERIFY.md:345\u2013346). VERIFY.md:346: the benchmark's own pods waited hours for one GPU of the 4090, 3090, A6000 and A100. VERIFY.md:707\u2013710 (\"What needs Kai\", item 1): the 4090 leads from G = 2 and its pool is 3 schedulable nodes; both of its benchmark pods waited hours. The 3090's pool is far larger; the break-even table shows how many 3090s match a given number of 4090s. Options: the [D4] probe at decision time (`code/gen_bench.py --probe`), or a choice with the table in hand. The 4090's class-E K=5 peak of 91.9 % (E enters T at K=4 only, VERIFY.md:18\u201319) is a separate memory caution for N=64 class-E pods at K=5 (benchmark STUDY.md:42,105). This prior evidence does not certify this campaign's pack size or memory use. This is an explicit departure from STUDY v0.15:1342 and the comparative same-workload selection gate in gpu-selection-policy.md:32\u201369: no same-workload measurement for this campaign exists before record draws on the product. Benchmarks require the record archive (code/nrp/spec.py:67) and its product-specific init root (:104\u2013105), while gen/inputs.json holds one init_tensors root per generation. For approval 1 Kai therefore chooses the product first; B19 under approval 2 measures on that product and can confirm or reject it. If the measured rule later rejects it, record draws are redone on the winning product under a new approval; nothing trained on the rejected product's draws is production. The orchestrator fills the chosen product name from Kai's answer using python3 -B set_product.py <name> and re-validates this authority; his choice is recorded in decisions.md 2026-10-20 (4) alongside the Mutagen ruling before approve.",
   "modes": [
    "draw"
   ],
   "production_submissions": [],
   "subject_to": "Orchestrator obtains and records Kai's product-of-record choice alongside his flags item 0 Mutagen ruling in decisions.md before giving him the approve command; final Kai approval and applicable launch gates; no non-draw activity"
  },
  "headline": "H1",
  "headline_hold": {
   "binary_launch_precondition": "Kai's confirmation of the BF owner record (flags item 0b)",
   "both_fail": "H1 for all binary arms",
   "both_pass": "H2 for all binary arms",
   "condition": "until the export check's headline is recorded",
   "condition_met": true,
   "engineering_release_requires": "Draw-output rebind in gen/inputs.json (record/smoke roots and platforms, bench_epochs); STUDY v0.18 stamp 027ff590825922df3b1090a96eb9bf000dd05e82920d19daf9ae504933cbfc68; H1 for every binary arm; both test fixes (file-audit c4dc2a80ca2b, stand-in ee30820d2ffa retained in amended 8626e4b; C9 message residual due at rebind); G57 post-outcome branch; regenerated configs; rebind gate pass; BRIEF for approval 2; approval 2",
   "one_pass": "NOT APPLICABLE: both forms failed",
   "outcomes": {
    "BF": "F111-BF: recorded as a failure by the owner record of 2026-10-20T17:30Z; no C-synthesis result.",
    "FB": "FAIL: criterion 2, factored WNS -1.090 ns"
   },
   "record_sources": [
    "notes/LEDGER.md: EXPORT CHECK OUTCOME, 2026-10-20 17:18Z",
    "preflight/export/codex/codex-export-eval-report.md",
    "preflight/export/PREFLIGHT-rows.md",
    "notes/flags-for-kai.md: item 0b, owner record 2026-10-20T17:30Z",
    "review/snapshots/v0.18/STUDY.md: section 3.4 outcome"
   ],
   "recorded_headline": "H1 for every binary arm, F100 included",
   "release_requires": "After approval 2 engineering evidence, certify product/K in ADMISSION.json, bind its new hash in a new BRIEF authority, then approval 3 for production S1 all arms; binary arms require Kai's confirmation of the BF owner record; controls/S4 retain separate conditions",
   "rule": "STUDY v0.15 section 3.4 headline rule v0.4",
   "scope": "ALL binary arms including F100, smoke and benchmark; PID needs headline sensor",
   "status": "RECORDING_CONDITION_MET; BINARY_ENGINEERING_HELD_FOR_APPROVAL_2; BINARY_S1_HELD_FOR_APPROVAL_3"
  },
  "launch_decisions": {
   "confirmed_by": "Kai",
   "confirmed_on": "2026-10-19",
   "od12": true,
   "od13": true,
   "od15": true
  },
  "one_pass": {
   "additional_h2_runs": 0,
   "reason": "The one-form-passes provisions and 531.25 GPU-h cap do not arise",
   "status": "NOT APPLICABLE: both forms failed"
  },
  "s4_option": null,
  "study": {
   "sha256": "e4a6b89e0adbe3ab14b508a24335688fba48db3f801c4d1015812cc9dcd73d8e",
   "version": "v0.15"
  },
  "transport": {
   "configmap_count_before_every_wave": 0,
   "launcher": "run_handoff.py",
   "lint": "nrp_doctor.py through an active hook path",
   "mode": "bundle",
   "pvc": "kai-data",
   "root": "/data/cbl1016/"
  }
 },
 "selector": "campaign=chang-bitlinear-20261016",
 "stage_order": [
  "S1",
  "S3",
  "S4",
  "reserve"
 ],
 "workspace": "code",
 "workspace_command": [
  "python3",
  "-B",
  "-m",
  "tools.operations",
  "workspace",
  "--campaign-dir",
  "{cdir}",
  "--job",
  "{job}"
 ]
}
```

```json operations
{
 "agent_invocations_warn": null,
 "candidate_runtime_factor": 1,
 "codex_bin": "codex",
 "deadline_seconds_per_epoch": 3,
 "elapsed_days_warn": null,
 "extra_excluded_nodes": [],
 "infra_relaunches_per_key": 0,
 "kubectl": "/home/kaimoe/.local/bin/kubectl",
 "lease_s": 3600,
 "repair_attempts_per_class": 0,
 "runtime_factor_notify": 1,
 "score_deadline_s": 600
}
```
