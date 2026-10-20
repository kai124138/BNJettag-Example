# Items waiting for Kai: chang-bitlinear

One decision at a time, in this order. The first ones unblock the most work. Each line gives what
the item decides, the default that holds until Kai answers, and what waits on it. STUDY item
numbers refer to §19.2 of `STUDY.md` v0.8 (sha256 290022fa…). Kept by the orchestrator, newest
state as of 2026-10-20. This file carries no hardware-test outcome; build agents may read it.

## Decided by Kai on 2026-10-20 (decisions.md 2026-10-20 (1), (2) and (3))

- (3) 18:2xZ: Mutagen read = incident, not a stop; GPU product of record for the draws = NVIDIA-GeForce-RTX-4090; BF owner
  record stands, both F111 forms train under H1.
- (4) 18:3xZ: the 4090 confirmed with the benchmark record's own wording ("wait just go with the 4090") after the
  PREFLIGHT review escalated the misstated basis; the BRIEF for approval 1 (revision 5) names it.

- Trimmed path to training; "i approve all of the arms and the runs and their desgins are good and very intentional"; keep
  going until training is submitted. Formal approval remains `harness.py approve` on the final BRIEF.
- D1 yes: the installed ADMISSION.json admits the 13 eligible-now identities until the export check's headline is recorded;
  binary arms are added then under re-approval. D2 yes: a Job observed never to have started is charged 0 GPU-h; failed
  Jobs charge measured time; live or unknown keep the bound (commit 84f372f).

## Open since 2026-10-20

0b. **DECIDED 2026-10-20 (decisions.md 2026-10-20 (3)): the record stands; both forms stay, under H1.** Export check: F111-BF recorded as impossible to complete within the campaign's synthesis budget (STUDY §3.4 v0.4
   "Outcome and forms" requires the owner's record with the reason and a report to Kai before any binary launch).**
   Facts: BF normal and factored C-synthesis stopped at the 8 h limit and again at the 86,400 s limit (ended 09:29:54 PDT
   2026-10-20) without finishing C-synthesis; Vivado never ran; the trained CH-8 normal route timed out the same way; the FB
   pair completed and fails criterion 2 (factored WNS −1.090 ns). Owner record (orchestrator, 17:30Z UTC): BF is impossible
   to complete within the budget the plan set (B14-7a escalated 8 h → 24 h; no larger budget exists in PLAN.md or STUDY §9);
   B14-7a's consequence line ("no form's outcome is decided by an operator time limit") is therefore superseded by this
   record, with the reason stated. Consequence either way: FB fails on timing, so H1 is the headline for every binary arm
   under Kai's split-form rule; a BF pass after a longer run could only ADD the 24 H2 runs (one-pass case), never change H1.
   Default until Kai answers: the record stands; no BF resynthesis. Alternative Kai may choose: extend BF to a 72 h attempt
   on mulder before the binary arms launch (delays binary arms by up to 3 days; CH-8/F000 unaffected). Reported to Kai in chat
   17:30Z UTC. Waits: the binary arms' admission (second approval).

0. **DECIDED 2026-10-20 (decisions.md 2026-10-20 (3)): incident, not a stop.** Mutagen read of the held-out archive (investigation G13-G51, `preflight/investigations/G13-G51/TICKET.md` §4.1).**
   During gate G51 pcc at 29a459ce7ee6 the gate hard-linked the real held-out archive into the Mutagen-synced work directory
   for about 30 s and the sync agent read its bytes; whether any reached the Mac is undetermined; no copy exists now on either
   machine; no training, selection or evaluation code read them. Fixed by construction in commit 38f4eba (gates refuse
   synced paths) and by the B18f operator procedure (work dirs in scratch). Default until Kai answers: an incident with a
   fix, not a STUDY §14 "hidden test access" stop; preflight continues. Waits: Kai's ruling before `harness.py approve`.

## Decided by Kai on 2026-10-18 (moved here from the list below)

- Plan approved ("okay the plan seems perfect"). Items 1 and 1b follow it: code reaches pods as a git bundle on `kai-data`
  (no ConfigMaps, nothing pushed), runtime as the hashed lock installed into a venv on the PVC (Phase 2, running).
- No mulder before training; training on NRP. Then: mulder permitted for the export check that makes H2 valid. The binary
  arms train under H2 once a whole trained model passes the factored-export check (bit-exact, timing met, fewer LUTs than
  the normal export), under H1 if it fails; CH-8 and F000 launch first. This settles items 2-4 below for now and replaces
  the frozen rule's condition 3 (informally applied to the OPEN-2 table, it gives H1). The INLINE patch is part of both
  export routes, applied identically to every model.
- Launch conditions (Kai, 2026-10-19): OD-13 yes, OD-15 yes, OD-12 yes, claim domain (b) without (c). Items 5-8 below are
  therefore settled.
- Open within it: no separate H1 campaign unless Kai wants one (every run records H1 beside H2).
- Split-form case (Kai, 2026-10-19T16:28:12Z, before the check's outcome): if one form passes and the other fails, every
  binary arm in both forms runs under H1 AND the passing form also runs under H2 (24 more runs, about 154 GPU-h; cap about
  530 GPU-h in that case only). Both pass: H2 for all; both fail: H1 for all.

## Blocking the build or the launch

1. **Launcher route (item 1).** How a Job gets its code now that ConfigMaps may hold no data
   (NRP admin, 2026-10-17). Options in PREFLIGHT-PLAN §4 "Launcher adapter": A1 a private git remote
   with a deploy key (outward), A2 a git bundle on `kai-data`, B ConfigMaps owned by the Job and
   deleted at its end, or the code baked into the image. Default: none; no NRP action happens until
   this is chosen. Waits: B17 launcher part, every NRP smoke, benchmark and run.
1b. **How the JAX runtime reaches the pods (B16; with items 1 and 7).** The plan assumed a Docker image
   built and gated on the PC, but the PC has no Docker. Options: (a) install Docker and the NVIDIA
   container toolkit on the PC (admin change on Kai's machine), build there, gate there, Kai pushes;
   (b) no custom image: a public CUDA base image, and the hashed lock installed once into a venv on
   `kai-data` by one Job, its freeze checked against `cbl1016-jax2`, every pod using that venv (no
   push; G05 becomes a venv gate on NRP); (c) build on NRP (kaniko or the NRP GitLab registry), which
   is a push. Default: none chosen; B16 waits. Recommended by the orchestrator: (b), because it needs no
   image push and the lock is already content-hashed, but it is a launch-route choice and Kai's.
1c. **GPU determinism (B5-1; a STUDY §7.3 stack question).** On the RTX 4060 Ti with the pinned stack and
   default XLA flags, two runs of the same program in separate processes differ (G13: gradients of 35 of
   97 trainable variables; 39-43 of 244 variables after each epoch; engineering readings). With
   `XLA_FLAGS=--xla_gpu_deterministic_ops=true` every comparison is bitwise equal; the CPU is always
   deterministic. Chang's release does not set the flag. Options: (a) production sets the flag, a
   declared departure from the release, with a throughput cost not yet measured (the B19 benchmark
   would measure it); (b) production keeps the default, so the determinism record will likely show
   non-identical runs and F000 − CH-8 then measures run-to-run divergence, as STUDY v0.8 §13 already
   provides for. The PREFLIGHT GPU bitwise gates run under the flag either way. Default: (b), the
   release's behaviour, until Kai decides. Waits: B19 (benchmark and determinism record), the image or
   venv of record.
2. **Mulder beyond the da4ml test (item 6).** Permission for the export environments, option 0,
   the export gates, the clean rerun and synthesis. Default: no mulder session. Waits: B14, B15 on
   mulder, the headline count, every final binary config, and therefore the launch.
3. **Which da4ml run the frozen rule is applied to, and what a fault does (item 5).** Recommended in
   the STUDY: a clean rerun; on a failed assertion, void and restart. New in v0.8: on an interruption
   after a rule line has started, (i) the stopped line fails, or (ii) the rerun is voided and
   restarts; the STUDY makes no recommendation between them. Default: the text as it stands. Must be
   answered and recorded with its time before the clean rerun starts (arbiter P2). The driver built
   in B15 takes both answers as required inputs and refuses to run without them.
4. **INLINE route.** An export-route question raised by an engineering test that blind agents may
   not read; the orchestrator explains it to Kai directly (details in `decisions.md`, results-aware).
   Not defined in the STUDY; a route chosen after a results-aware test is Kai's. Waits: the route
   lock at the end of B14 and the run the rule applies to.
5. **Confirm OD-13 (item 2):** learned-width scale quantizer; BF quantized after the fold. Default:
   as adopted. Waits: B6 (about 10 hours of rework if rejected later). Launch precondition.
6. **Confirm OD-15 (item 3):** a three-seed screen without verdicts. Default: as adopted; B3b is
   being built on it now. Launch precondition.
7. **Confirm OD-12 (item 4):** the PID target 350,000 read in headline-count units. Default: as
   adopted. Waits: B9. Launch precondition.
8. **Claim domain (item 12).** Recommended: (b), the band g ≥ g*. Launch precondition; configs
   carry it and fail closed while it is unset.
9. **JAX/CUDA pins.** jax and jaxlib 0.9.2 with the CUDA 12 plugin, as locked and tested on the PC.
   Default: as locked. A change reruns the GPU gates. Waits: B16 image of record.
10. **Push the JAX runtime image (item 7)**, then **`harness.py approve` (item 10)** after the
    PREFLIGHT gates. Both are Kai's own actions.

## Defaults the build adopted that Kai should review

11. **U1:** each seed's initial tensors are drawn once on the production GPU, saved and loaded by
    every arm, CH-8 included. MEDIUM.
12. **U2:** the runner passes the device (`gpu:0`, or `cpu:0` in CPU gates), as `run_train.py:73-74`;
    no third shim. MEDIUM.
13. **U6:** W8A8 "eight bits, trained steps" as KBI with b fixed at k + b = 8. LOW; it changes Chang's
    datalane quantizer type in that control only.
14. **U7 and the PC lab-root tree:** where launcher and harness commits live. The PC lab root is git
    (remote kai124138/bnjettag-oct9) with one commit, b9e31c0 of 2026-10-10, and 21 modified plus 28
    untracked entries since (the week's Mac edits synced by Mutagen, `tools/harness.py` among them).
    Nothing has been committed there. Waits: B17.
15. **Determinism record (constructive v0.7 B1):** one F000 seed-1 run added to the two CH-8 runs,
    about 0.05 GPU-h from the existing reserve; adopted by the orchestrator instead of the zero-cost
    alternative.
16. **Stack of the two bitwise gates** (reference fidelity, resume): now a PREFLIGHT owner decision,
    recorded before they run (STUDY v0.8 §13 :1348).
17. **B3a-1:** in the H2 bound u_c, the maximum over F runs over the same inputs as the sum. It changes
    H2 only at single-magnitude columns that also hold zero coefficients. MEDIUM; needs Kai or a
    one-line STUDY clarification before configs are final.
18. **B3a-3:** the per-layer combine of the exact kernel term with the float32 bias term (one float32
    addition, then uint32 truncation). Gate G28 in B7 checks it against the native count.
19a. **B2-3:** the check that training and held-out jets are disjoint hashes each raw held-out jet and
    keeps only the overlap count (0); no held-out feature array of the prepared archive is read and no
    model output is computed. MEDIUM. The alternative is a file-level check only (weaker).
19b. **STUDY v0.9 (pre-data amendment, in progress):** the §6 power table's ρ = 0 rows are corrected
    to the exact closed form (five cells were one unit off at printed precision, by at most 0.007 σ_d;
    found by gate G54 and recomputed by the orchestrator); the ρ = 0.9 rows are labelled Monte Carlo
    with their standard error; the undefined labels CROSSING, LOCAL ADVANTAGE and LOCAL DEFICIT are
    restored from v0.2. No threshold, rule or run count changes. Gets a critical review.
19c. **Selection and statistics defaults (B3b, flagged):** B3b-15 dominance as a set-level reading over
    the step-envelope checkpoints; B3b-17 the null reference's correlation structure (ρ within an arm
    and seed, ρ_f = ρ between forms at the same g, ρ_f·ρ across g); B3b-18 plug-in power reported as 0
    where BEST is infeasible in some seed, with the feasible-only power beside; B3b-21 failed,
    integrity-stopped or budget-stopped runs make C2, C3 or C3r INCOMPLETE; B3b-27 item 12's answer must
    also name `band_rule` (default, strict_sd, interval) and `c3r_level` (bonferroni, nested).
    Details: `preflight/gates/G53/1784511c0f34/PREFLIGHT-rows.md`.
19d. **Clean-rerun driver defaults (B15 dry run, flagged; they belong with item 3):** B15m-2 a frozen
    script or `env.sh` whose hash changes during a line is treated like a failed assertion (so it
    follows Kai's item-5 remedy); B15m-3 how the stage of an interrupted line is read (a stop just after
    Vivado reported fails the line at parse, which moves that line toward H1); B15m-6 a rule line counts
    as started from its pre-line checks, and an interruption between two lines fails no line under (i)
    and voids the attempt under (ii). Details: `preflight/gates/G72/d868cad9a07d/PREFLIGHT-rows.md`.
19e. **B4-3 (initial tensors, with U1):** the loader refuses a run if any recorded identity field of the
    seed's initial-tensor file differs, including the number of visible GPUs. So the NRP draw Jobs (B19)
    must see the same GPU count as the pack-runner pods, or `n_devices` must be recorded only, not
    enforced. MEDIUM. Waits: B19.
19f. **B12-3 (smoke and benchmark arms):** the STUDY never lists which arm classes the smoke and the
    packing benchmark run; the generator uses 15 classes (one per arm type and form; list in
    `preflight/gates/G57/e8dce7948801/PREFLIGHT-rows.md`). MEDIUM. Waits: B19.
19g. **B8-17 (when the ablations run):** Δ_attn, Δ_static and Δ_unif run once at run end on every checkpoint still
    saved, not at every save as STUDY §12 reads literally (the outputs are the same for the checkpoints that
    survive; the cost is one pass at the end). MEDIUM. Details: `preflight/gates/G37/e59dfe74a49e/PREFLIGHT-rows.md`.
19h. **B7-9 (initialization readings, arbiter P3):** gate G36 records H1, H2 and NR at initialization for 19
    configurations, seeds 1-3, under the fork sha (identical on CPU and GPU), but it does not reproduce the
    pinned-environment README's earlier table, whose model state differs. Kai decides whether P3's "reproduces
    them" asks for that table to be reproduced too, or whether readings under the fork sha suffice. MEDIUM.
19. **Training lock:** wandb 0.31.0 added for STUDY §12 logging (and its own dependency `cwevals`, a
    W&B package); the 35 earlier pins are unchanged; venv of record `cbl1016-jax2`. For information.

## Later, not blocking now

20. Items 8 (cap or scope if the measured rate is high), 9 (the held control slot), 11 (S4 funding
    and size; recommended option C), 13-16 (constructive proposals; unadopted unless Kai adopts them,
    13-15 before the arm generator and harness admission are built, 16 before the first launch).
21. The `chang` agent definition says HGQ2 is not installed; that line is wrong and is Kai's to approve
    changing.
22. The paused mha-bitnet campaign's own list: `campaigns/2026-10-12-mha-bitnet/notes/flags-for-kai.md`.
