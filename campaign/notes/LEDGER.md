# LEDGER — chang-bitlinear-20261016

Newest on top. A few lines per step or debate round: claim or finding, evidence path, status. Paths relative to
`campaigns/2026-10-16-chang-bitlinear/` unless absolute. Created 2026-10-19 23:20 PDT by orchestrator session 783ef8
(claude job f4eee4ae) on first use, per the master prompt. Entry times from 04:14 PDT on 2026-10-20 were corrected
once from commit timestamps and file mtimes (the orchestrator had estimated them ahead of the clock); later entries are
stamped from `date -u`.

## 2026-10-20 20:44Z UTC — field guide: "Asked before" entry added for the config→training workflow (RULES §6)

- Entry "How do I run a campaign myself, from the config to a training job?" inserted at the top of Asked before in the
  live field-guide artifact (read in full first, local docs/field-guide.html rewritten from the live version + entry) and
  republished to https://claude.ai/artifact/1Qx77jZVjNnS3Ph5oWLxBu.

## 2026-10-20 20:24Z UTC — Kai pauses the campaign ("imma just run everything by myself ... close it and come back to it")

- Kai asked for the full config→training workflow with commands (answered in chat 20:24Z; field-guide entry added) and then
  said he will close this and return later. No action taken on the launcher fix (hold by default); nothing on NRP;
  APPROVAL 1 remains valid; the harness tick line on kaipc is still missing (nothing to monitor yet).
- Handoff `notes/handoff-2026-10-20-1115.md` gains a "PAUSED BY KAI" section 0 with the resume instructions and the
  one question to ask him on return (go/hold on the launcher fix). Feedback memory saved: fewer decision calls; plain
  meaning; defaults where he said so.
- This session stays open per his earlier instruction; it does nothing further on its own.

## 2026-10-20 19:39Z UTC — launcher fix prepared and tested in a copy (lab-only; campaign code and approval unchanged); install awaits Kai

- codex-launcher-fix (report `preflight/b19/launcher/codex-launcher-fix-report.md`; patch `run_handoff.patch`, +26/−3): no
  campaign reader of `resolved_configs` exists (nrp/run_pack.py consumes CBL_PLAN/checkout files; nrp/restart.py launch.json);
  the consumer is the lab launcher's own init script. Bundle-backed configs become {path, sha256, size, source:"bundle"}
  in the record; ConfigMap-mode configs stay inline; references are hashed; the pod side rebuilds from the bundle and verifies.
  Launcher sha256 before 8190bb8d693833036d8a62cadc90755d33ae3b4165625fb43fd5a3ab6426ab5b → after
  d8d1995705708985fc6c3775313b1ae395022f9fca87c4a6fbcb5bf94ce0d2b9. Tests: lab suite 26 passed / 1 skipped; new
  reconstruction/tampering/shell-quoting tests 5 passed; archived-commit B17 test_tools.py + test_nrp.py 96 passed; dry
  prepare: draw record 188,778 → 45,591 B; draw smoke 188,753 → 45,566 B; synthetic S1 K=2 pack 57,832 → 51,638 B;
  prepare/validate/offline-launch/annotation inspect passed. Unverified: in-pod replay, G79, lint, live G80.
- Install steps (after Kai's "go"): cmp live vs before; cp the fixed file over `tools/run_handoff.py`; copy the new test into
  the lab tests; rerun the lab unittest files and B17 on the PC with BNJETTAG_LAB; rerun G76, G78 prepare/validate/offline
  launch, manifest lint, fresh G80; record hashes, diff, tests and sizes in PREFLIGHT and the installed launcher hash in
  the BRIEF operations block (authority block untouched, so APPROVAL stays valid); then the kaipc cron tick line; then
  cluster-ops relaunches the draws with the kept specs.

## 2026-10-20 19:32Z UTC — record-cap analysis: cap is the Linux 128 KiB env limit; fix = bundle references in the record; prepared in a copy, install pending Kai

- codex-record-cap (read-only; its report could not be written because of the sandbox, findings in
  `preflight/b19/draws/codex-record-cap.log` final message): the lab launcher `tools/run_handoff.py` builds
  `resolved_configs` per run (:414) and embeds the record as the RH_RECORD env string (:451) and also writes it to kai-data
  (:71-76); `$C/code/nrp/spec.py:76-85` assigns the same `manifests/data-prep-150c.json` to three seeds (three inline
  copies); MAX_RECORD_ENV = 100,000 sits under the Linux MAX_ARG_STRLEN 128 KiB per-string limit, so raising it is unsafe;
  compact serialization still 143,714 B. Options: (a) raise cap — unsafe; (b) reference bundle-backed configs by
  path+sha256 (recommended; launcher-only if the consumer is lab code; estimated draw record ~46 kB; campaign sha, frozen
  hashes, operations block and approval unchanged in that case); if the consumer is campaign code, a campaign commit and
  re-approval follow. Verification named: lab launcher unit tests, B17 launcher-contract and tampering tests, G76/G78,
  dry prepare sizes for draw and production packs.
- codex-launcher-fix launched 19:32Z in `preflight/b19/launcher/` (live launcher untouched; copy `run_handoff.before.py`,
  sha printed in the ledger's console record): builds the reference form + consumer verification in a copy, runs the
  tests against scratch copies, dry-prepares the two draw packs. Installing over `tools/run_handoff.py` waits for Kai's
  answer (asked 19:32Z): the lab launcher is outside frozen_files but is the reviewed launch path.
- Also to fix before any launch: the kaipc crontab has no `harness.py tick` line for this campaign (cluster-ops noted it);
  the monitoring pass is the lab's rule for approved campaigns (RULES §4).

## 2026-10-20 19:28Z UTC — B19 part 1 BLOCKED before any cluster call: the launcher refuses the draw handoff record (188,778 B > cap 100,000)

- cluster-ops-draws (record `preflight/b19/draws/DRAWS.md`; cluster-inventory top entry): pre-checks passed (approval
  matches; PC and Mac at c4dc2a80ca2b; bundle sha equal on both machines; `kai-` ConfigMaps 0 before and after). Deviation:
  the BRIEF operations block pins kubectl to `/home/kaimoe/.local/bin/kubectl`, so prepare ran on kaipc (same BRIEF, ADMISSION,
  APPROVAL bytes, clean code). Both draw packs (record, smoke) refused by `run_handoff prepare`, exit 2, verbatim:
  "HANDOFF_BLOCKED: handoff record too large for the init container environment; do not split silently". Cap
  MAX_RECORD_ENV = 100,000 (`tools/run_handoff.py:39,438`); draw record 188,778 B; `resolved_configs` 151,959 B because in
  draw mode the run's config embeds `manifests/data-prep-150c.json` (59,825 B). No Job, pod, ConfigMap or kai-data write.
  Built specs (kept): `spec-u1-draw-{record,smoke}.json` (RTX 4090; known-bad nodes excluded; deterministic_ops false;
  mem_fraction 0.5; 4 CPU / 16 Gi / 3 h deadline; scientific_gate "cleared" citing APPROVAL.json, the agent's judgement);
  write-once `manifests/kai-cbl1016-u1-draw-*-a1/` on the kaipc tree only. Also noted: the kaipc crontab has no harness tick
  line for this campaign (nothing would monitor a submitted draw) — to fix before any launch.
- Any fix is a launcher or code change under an approval that froze the code: codex-record-cap (read-only) launched 19:28Z
  to establish the record builder, the cap's purpose, the options (raise the cap; reference the manifest by hash; record as
  a file on kai-data; an existing alternative) and the verification each needs; then the question goes to Kai with a
  recommended default. The approval-1 BRIEF binds code c4dc2a80ca2b; a campaign-code fix moves the sha and needs a new BRIEF
  and approval; a lab-launcher-only fix is outside frozen_files but changes the reviewed launch path and must be recorded,
  tested (tests/b17 launcher contract, G76/G78) and told to Kai.

## 2026-10-20 19:20Z UTC — APPROVAL 1 verified (Kai, 19:18:31Z; brief sha b03b1e53…); cluster-ops launched for the U1 draws

- `$C/APPROVAL.json` present (213 B); `harness.py validate`: approval matches the authority block, VALID. `kai-`
  ConfigMaps 0; no cbl1016/chang/bitlinear Job on cms-ml. decisions.md entry (5).
- cluster-ops-draws launched 19:20Z: U1 initial-tensor draws (seeds 1-3; record and smoke archives) on
  NVIDIA-GeForce-RTX-4090 through the campaign's submit path; pre-launch checks per the BRIEF operations block; evidence
  under `preflight/b19/draws/`; stop on any refusal (never bypass), on a Pending pod beyond 2 h (report), on any hook block
  other than a path fix. The session does not watch the run beyond the agent's bounded wait; the harness cron tick monitors.

## 2026-10-20 19:01Z UTC — Kai: "wait just go with the 4090" (decisions.md (4)); BRIEF for approval 1, revision 5, installed; VALID; approve command given

- Product filled by `preflight/b18f/set_product.py NVIDIA-GeForce-RTX-4090` (run without -I: it imports its sibling
  revise5.py); validate5 exit 0: PASS basis/product-only diff vs revision 4, departure/fallback unchanged, 55 frozen identities
  + ADMISSION bytes + operations unchanged, source line references and static-ceiling rows checked, harness schema VALID.
  Candidate sha 1d54a3885deef0818b9350106e084ec2685078872e71137f4f477d27edfcbd77 installed as `$C/BRIEF.md` (v3 kept as
  `preflight/b18f/BRIEF.installed-v3-20261020T1807Z.md`); `$C/ADMISSION.json` unchanged (160ba907…).
  `python3 tools/harness.py validate campaigns/2026-10-16-chang-bitlinear`: VALID, exit 0, authority sha
  b03b1e530cd2ca5422b8ea3a9ea2cf21bff8d4555b9a22d6209c39f2af2e7e34, "approval: none matching the authority block".
- The v3 review's ESCALATE is closed by its own stated one-step path (VERIFY wording; Kai's answer with that text;
  product set; diff checked), no fourth review. Review record: `review/critical-PREFLIGHT-v{1,2,3}.md`.
- Approve command given to Kai 19:01Z:
  `cd /Users/kaiyamaguchi/lab/bnjettag && python3 tools/harness.py approve campaigns/2026-10-16-chang-bitlinear --by Kai`
  (scope: U1 initial-tensor draws only, seeds 1-3, record and smoke archives, NVIDIA-GeForce-RTX-4090; nothing else).
- After APPROVAL.json exists: `harness.py validate` must show a matching approval; then cluster-ops (never Codex) runs the
  draws through `code/tools/campaign.py submit` / `tools/run_handoff.py` with `kai-` ConfigMaps 0 before; per-draw operator
  check: commit c4dc2a80ca2b, bundle bbc14fe8…; evidence into the tree. Then the rebind commit on top of 2222f5e8f40a.

## 2026-10-20 18:31Z UTC — BRIEF for approval 1, revision 5, ready with the benchmark record's own wording; product placeholder awaits Kai

- codex-b18f-brief5 (report `preflight/b18f/codex/codex-b18f-brief5-report.md`): basis rewritten from VERIFY.md with line
  citations (:9, :22-24, :31-32, :346, :704-708) and the static-ceiling rows (RTX 4090 3 nodes / 16 GPUs; RTX 3090 30 / 115;
  A10 30 / 237); "easiest scheduling default" removed; departure and fallback kept; product field `<PRODUCT> (Kai's choice,
  decisions.md 2026-10-20 (4))` in three places. Candidate sha bc66b5af…; authority sha 8ba989f0…; harness schema VALID with
  the placeholder (custom validator exit 2 only for `<PRODUCT>`); `BRIEF.v4-v5.diff`: 62+/15−, only basis wording and the
  product field changed; revision 4 kept as BRIEF.candidate.v4.md (f6be95ac…). `set_product.py <name>` fills the placeholder
  and reruns the validator (no hand edit).
- Closing steps once Kai answers (no fourth review, per the v3 review's own closing path): set_product.py → validate5 →
  diff check → install `$C/BRIEF.md` (ADMISSION unchanged) → `tools/harness.py validate` → decisions.md (4) → approve command.

## 2026-10-20 18:25Z UTC — PREFLIGHT review iteration 3: ESCALATE (0 A, 1 B, 2 C); B5 = the 4090 basis misstated the benchmark record (orchestrator's error)

- `review/critical-PREFLIGHT-v3.md`: B4 resolved in structure; C4, C9, C10 applied; C8 in part. **B5**: the basis for
  recommending the RTX 4090 misstates the 2026-09-29 gpu-benchmark VERIFY, every time in the 4090's favour: "easiest
  scheduling default" (VERIFY: the 4090's pool is the smallest of the leaders, 3 schedulable nodes / 16 GPUs, against 115
  GPUs for the 3090 and 237 for the A10 on the same ordinary request) and memory as its only caution (VERIFY :31-32, :346,
  :704-708: both of its benchmark pods waited hours; flagged for Kai). The wording originated in the orchestrator's brief
  to codex-b18f-brief4 ("schedules most easily"), repeated to Kai at 18:16Z. Acknowledged as the orchestrator's error.
- Escalation per §6.5: the product choice goes to Kai with the VERIFY's own text (asked 18:25Z); Kai's 18:20Z "NVIDIA GeForce
  RTX 4090" was given on the misstated basis and is to be confirmed or changed. No fourth review: close by (1) VERIFY wording
  in the basis (codex-b18f-brief5, launched 18:25Z, placeholder product + set_product.py), (2) Kai's answer in decisions.md
  (entry (4)), (3) set the product, re-validate, (4) diff against revision 4 (f6be95ac…) to show only the basis and the
  product changed, then install.
- Orchestrator's recommended default to Kai: confirm the 4090 for the draws (the rule's leader on throughput; the draws are
  small jobs; the fallback redoes them on another product if B19's measured rule or queueing rejects the 4090), with the
  3090's far larger pool and the [D4] probe as the alternatives.

## 2026-10-20 18:20Z UTC — Kai decides: Mutagen = incident; RTX 4090 for the draws; BF record stands, both forms under H1 (decisions.md 2026-10-20 (3))

- Verbatim in decisions.md. Both preconditions of the approve command named by the reviews (Mutagen ruling, product choice)
  are now recorded; the binary-admission precondition (BF confirmation) too. Mutagen stays running for evidence sync
  (noted in the decision); pausing it is Kai's call.
- The BRIEF for approval 1, revision 4, already names NVIDIA-GeForce-RTX-4090; at install the product field's "Kai's choice
  must be recorded in decisions.md" is completed with the entry reference through the scripted preparer, then re-validated.
- Waiting: iteration-3 verdict of the PREFLIGHT review. Then install revision 4, validate, give Kai the approve command.

## 2026-10-20 18:20Z UTC — BRIEF for approval 1, revision 4, prepared (VALID); iteration-3 review requested (candidate, not installed)

- codex-b18f-brief4 (report `preflight/b18f/codex/codex-b18f-brief4-report.md`): B4 prose at :32-52, :225-235, :557 (Kai the
  decider; RTX 4090 recommended default with the benchmark/scheduling basis; the same-workload departure; the fallback);
  C4 per-draw operator check (:210-215); C8 header; C9 commit naming; C10 "the approval-2 BRIEF". Bindings unchanged;
  validate4 VALID exit 0, authority sha 226e7e9a…; candidate sha f6be95ac…; v3 kept as BRIEF.candidate.v3-draws-only.md.
  Installed `$C/BRIEF.md` remains v3 until PASS (reviewer's C8 preference).
- Iteration 3 (last solo round) sent to the PREFLIGHT reviewer → `review/critical-PREFLIGHT-v3.md`.
- Still open for Kai (asked 18:16Z): Mutagen ruling (blocking), GPU product (blocking; default 4090), BF record (not blocking).

## 2026-10-20 18:15Z UTC — PREFLIGHT.md: review v2 C8 and C9 applied by the orchestrator (status lines; commit phrase)

- C9: PREFLIGHT.md line 9 read "first committed as 8626e4b5f4e4" because the orchestrator's 18:04Z hash substitution also
  rewrote its own inserted text; now "8626e4b5f4e4 (first committed as ee30820d2ffa; message amended 18:00Z …; same tree
  and parent)". C8: lines 3131 and 3291 now record the v3 installation at 18:07Z (BRIEF fadec31f…, authority c6fade11…,
  validate3 VALID) and that revision 4 is reviewed before it replaces v3. Per the reviewer's preference, revision 4 is
  reviewed as `preflight/b18f/BRIEF.candidate.md` and installed only after PASS.

## 2026-10-20 18:14Z UTC — PREFLIGHT review iteration 2: ITERATE (0 A, 1 B, 5 C); B4 = GPU product of record needs a decider and basis

- `review/critical-PREFLIGHT-v2.md`: A1, B1, B2, B3 of v1 resolved (B3 in substance; residual C8). Recomputed: 55 frozen
  hashes unchanged; authority hash c6fade11… = validate3; no approval. **B4**: the draws' one free parameter, the GPU
  product of record, has no stated decider or basis; the bound STUDY v0.15 :1342 and the GPU policy choose the product from
  same-workload measurements, which cannot precede record draws on that product (spec.py:67,104-105). Fix: BRIEF prose
  naming the decider (Kai), the basis, the departure, the fallback (a rejected product ⇒ new draws + new approval); Kai's
  product choice and his Mutagen ruling go into decisions.md before the approve command. C8-C10 fixable in the same pass;
  C3, C4 (now a per-draw operator check), C6, C7 do not block. Iteration 3 is the last solo round before ESCALATE.
- Recommended default for Kai: NVIDIA-GeForce-RTX-4090 (2026-09-29 gpu-benchmark VERIFY: the pre-registered rule favours the
  RTX 4090 at 2-16 GPUs, the A100 only at one; this campaign plans up to four concurrent GPUs; the policy lists the 4090
  first among ordinary-request products; its K=5 memory caution concerned N=64 class-E pods).
- codex-b18f-brief4 launched 18:14Z (B4 prose; C8-C10; C4 per-draw check; validate4). Then reinstall, validate, iteration 3.

## 2026-10-20 18:10Z UTC — walkthrough page v5 published; handoff-2026-10-20-1115.md written

- `notes/walkthrough-2026-10-20.html` v5 (11:10 PDT): gate results of both b18f records, the export outcome, the
  three-approval sequence, the two open questions for Kai (Mutagen ruling blocking; BF choice not), running-now table.
- `notes/handoff-2026-10-20-1115.md` written (seven sections) at the end of the "B18 closed, first BRIEF installed" step.
- Waiting: PREFLIGHT re-review iteration 2 (Opus agent); Kai's Mutagen ruling (blocks approve) and BF choice.

## 2026-10-20 18:07Z UTC — BRIEF v3 (draws-only scope) installed; harness validate VALID; PREFLIGHT re-review iteration 2 requested

- codex-b18f-brief3 (report `preflight/b18f/codex/codex-b18f-brief3-report.md`): A1 scope = U1 draws only (modes ["draw"],
  seeds 1-3, record and smoke archives, one product of record; every non-draw activity removed with the reason); B1: the
  S1 hold is enforced by nrp/spec.py (S1 only in mode pack) and campaign.py admit() (pack refused while
  production_certification is null) plus the ADMISSION hash; B2: stand-in exception text corrected (29a459ce7ee6 pass
  unexplained; fix ee30820d2ffa = amended 8626e4b5f4e4, identical bytes); C1 header; C2 BRIEF wording cites the BF owner
  record; Mutagen ruling named as a precondition the orchestrator records before presenting approve; the three-approval
  sequence stated. Bindings unchanged (code c4dc2a80ca2b, STUDY v0.15 snapshot, MANIFEST 670fcf5f…, ADMISSION 160ba907…,
  bundle bbc14fe8…; 55 frozen hashes). validate3 VALID; no hand edits. v2 kept as BRIEF.candidate.v2-c4dc2a80ca2b-b19.md.
- Installed at `$C/BRIEF.md` (previous installed v2 kept as `preflight/b18f/BRIEF.installed-v2-20261020T1730Z.md`);
  `$C/ADMISSION.json` unchanged; `tools/harness.py validate` at the lab root: see the line printed with this entry.
- Re-review (iteration 2) sent to the PREFLIGHT critical reviewer naming A1, B1, B2, B3 and the ledger corrections.

## 2026-10-20 18:04Z UTC — PREFLIGHT.md draft3 installed (review B2, B3, approvals sequence, STUDY v0.18 row); commit hash note

- codex-preflight-md3 (report `preflight/b18f/codex/codex-preflight-md3-report.md`): B2 (heap-layout claim withdrawn as
  unsourced), B3 (v2 installed 17:30Z and VALID; v1 review narrowed the scope; v3 in preparation), the three-approval
  sequence, STUDY status v0.18, C2's PREFLIGHT wording. Installed as `$C/PREFLIGHT.md` (3,334 lines; backup
  `preflight/b18f/PREFLIGHT.before-20261020-1805.md`; body from "Fork commits" to the B18f heading verbatim).
- Hash note: the branch-only test-fix commit cited in earlier entries as ee30820d2ffa was message-amended at 18:00Z
  (withdrawing the unsourced sentence) and is now **8626e4b5f4e4**; the branch tip is 2222f5e8f40a. Earlier ledger entries
  keep the old hash as written; PREFLIGHT.md names both.
- Waiting: codex-b18f-brief3 (BRIEF v3, draws-only scope) → reinstall + validate → PREFLIGHT re-review iteration 2;
  Kai's Mutagen ruling (blocking the approve command) and his BF choice (non-blocking).

## 2026-10-20 18:01Z UTC — rebind-ready commit set on branch b18fix: tip 2222f5e8f40a (main unchanged at c4dc2a80ca2b)

- Branch b18fix (not on main): c4dc2a80ca2b → 8626e4b (lossguard stand-in test fix; message amended to withdraw the unsourced
  heap-layout sentence, review B2) → 24f2410 (gen constants STUDY v0.18 027ff590…; gen/inputs.json headline H1 with
  provenance naming "STUDY v0.18 §3.4 outcome"; gates/g57_configs.py and tests/test_gen.py post-outcome branch, snapshot
  path FORK.parent/review/snapshots) → 2222f5e8f40a (configs-only regeneration: stamps, derived hashes, headline null→H1 in
  all 167 configs, unset_inputs minus headline in 154). codex-h1cfg3: test_gen 38 passed; offline 644 passed.
- G57 at 2222f5e8f40a from the worktree: 72 PASS, 2 FAIL = the two snapshot checks, which resolve only when the fork sits
  at <campaign>/code (by design); of record they run in the post-draw pass from `$C/code`
  (`~/.claude/jobs/f4eee4ae/tmp/g57-2222f5e8f40a/stdout.txt`).
- Still missing from the rebind set, by design: the U1 draw outputs (init_tensors roots/manifests, smoke roots, bench_epochs)
  that approval 1 produces. The post-draw rebind commit adds them on top of 2222f5e8f40a.
- Fix round for the PREFLIGHT review running: codex-b18f-brief3 (draws-only scope, B1 wording, B2 text, C1 header, Mutagen
  precondition) and codex-preflight-md3 (B2, B3, approvals sequence, STUDY row). Then reinstall BRIEF (ADMISSION unchanged),
  validate, re-review iteration 2.

## 2026-10-20 17:58Z UTC — PREFLIGHT critical review v1: ITERATE (1 A, 3 B, 7 C); approvals re-sequenced: draws first

- `review/critical-PREFLIGHT-v1.md` (Opus). Bindings verified (55 frozen hashes, ADMISSION 160ba907…, bundle bbc14fe8…
  complete history head c4dc2a80ca2b, clean tree); every FAIL row's cause backed by evidence (G46 recompute: 36/37 fields
  identical, only host-memory telemetry differs).
- **A1**: at c4dc2a80ca2b `check_launchable` refuses every committed config except U1 draws: gen/inputs.json has the
  initial-tensor roots, smoke initial-tensor roots and benchmark epoch count null (NOTES-B18.md:36,166-167 said so). Smoke,
  kill-and-resume, determinism, benchmark, rates and certification each need a rebind commit (the draws' outputs bound into
  the inputs) and a re-approval; certification writes ADMISSION → a further approval. So: approval 1 = U1 draws only;
  approval 2 = after the rebind commit (draw outputs + STUDY v0.18 stamp + headline H1 + the two test fixes + the G57
  post-outcome branch, one gate pass); approval 3 = production after certification. The 17:34Z "main frozen until B19 ends"
  plan is withdrawn: main stays frozen only until the draws end (nothing on the PC needs the synced checkout while draws
  run on NRP); the rebind commit set then lands on main and its pass runs on the PC as usual.
- **B1**: the S1 hold IS enforced in code: an S1 config submits only in `pack` mode, refused while ADMISSION's
  production_certification is null, and changing ADMISSION breaks the authority. The BRIEF's "prose-only" wording and the
  ledger 17:35Z reasoning are corrected: mechanical, via certification null; 13 identities fine.
- **B2**: the ledger 17:31Z / PREFLIGHT.md:3158 / commit ee30820 claim that the stand-in test passed at 29a459ce7ee6
  because earlier failures changed the heap layout is UNSOURCED (the test runs in an isolated subprocess; the Codex report
  does not say it). Withdrawn: why it passed at 29a459ce7ee6 is not established. The branch-only commit message is amended
  after codex-h1cfg3 exits (branch not on main).
- **B3**: PREFLIGHT.md:3110-3115 and :3252 still say ADMISSION is not installed and cite the old validation → fix.
- **Before approve**: flags item 0 + STUDY v0.15 §14 (hidden test access): Kai's Mutagen ruling must be recorded before
  `approve`, since every B19 Job is a launch. Asked Kai as the one blocking question at 17:58Z.
- Fix round: codex-b18f-brief3 (BRIEF scope = U1 draws only; hold wording; approvals sequence) and codex-preflight-md3
  (B2, B3) launched; then iteration 2 of the PREFLIGHT review with the same reviewer.

## 2026-10-20 17:56Z UTC — STUDY v0.18 critical PASS (0 A, 0 B, 3 C); STUDY of record for the second BRIEF = v0.18

- `review/critical-v0.18-amendment.md` (iteration 3): v0.17 B1 (timeline) and B2 (in-place rewrite) resolved; the three
  v0.16 places verified byte-for-byte against the v0.16 snapshot plus annotations; rule blocks, §10, protocol-draft.json
  unchanged; snapshots v0.17/v0.18 read-only. Verdict line in `.claude/memory/review-reports.md`.
- The export outcome is now of record in the STUDY (v0.18, sha 027ff590825922df3b1090a96eb9bf000dd05e82920d19daf9ae504933cbfc68)
  and in PREFLIGHT.md ("Export check outcome (recorded 2026-10-20)"); the BF owner record and the report to Kai are cited;
  Kai's confirmation or his 72 h alternative remains open (flags item 0b) and gates binary admission, not CH-8/F000.
- codex-h1cfg3 launched 17:56Z: constants → v0.18, provenance names "STUDY v0.18 §3.4 outcome", regenerate, tests, G57 dirty.
  Then commit A (gen inputs+constants, gates/g57_configs.py, tests/test_gen.py) and commit B (configs only) on b18fix.

## 2026-10-20 17:54Z UTC — STUDY v0.18 written (B1, B2 of v0.17 resolved); iteration-3 re-review requested

- `$C/STUDY.md` v0.18 sha256 027ff590825922df3b1090a96eb9bf000dd05e82920d19daf9ae504933cbfc68 (2,173 lines); −6/+10 over
  v0.17; v0.16 places restored byte-for-byte with "[Corrected in v0.17: …]" annotations; the outcome block carries the
  17:06/17:10/17:14/17:18/17:30/~17:36Z sequence; rule blocks and §10 unchanged; snapshots v0.17 and v0.18 sealed and
  chmod a-w; notes `review/amendment-v0.18/NOTES.md`. Re-review (iteration 3, last solo round) sent to the same reviewer.
- If PASS: generator constants → v0.18 (sha 027ff590…), regenerate (stamp only), two-step commit on b18fix, G57 clean.

## 2026-10-20 17:52Z UTC — STUDY v0.17 re-review: ITERATE (0 A, 2 B); fixer resumed for v0.18 (iteration 3 must PASS or escalate)

- `review/critical-v0.17-amendment.md`: v0.16 A1 resolved in substance, B1 resolved. New **B1**: "the outcome was read once,
  by the evaluation job (17:14Z)" contradicts LEDGER 17:10Z (orchestrator's anticipatory "once recorded: BF = FAIL" after the
  17:06Z mulder check, before the evaluation); the Kai-report time is cited to the 17:34Z entry while the 17:48Z entry gives
  ~17:36Z. **B2**: v0.17 rewrote the v0.16 change-log entry, front-matter clause and A.1 bullet in place without disclosure;
  restore byte-for-byte and annotate "[Corrected in v0.17 …]". C11: snapshot files writable → chmod a-w. C1, C3-C7 not
  required for PASS. Rule blocks, §10, protocol-draft.json unchanged; snapshot hash = live file.
- Fixer resumed 17:52Z with the exact timeline (17:06 noticed; 17:10 anticipatory reading; 17:14 evaluation confirmed;
  17:18 outcome entry; 17:30 owner record; ~17:36 chat report) and the restore-and-annotate instruction → v0.18.
  §6.5: iteration 3 is the last solo round; an unresolved finding then goes to Kai.
- codex-g57h1b done 17:50Z: G57 snapshot path = FORK.parent/review/snapshots/<version>/STUDY.md with an info line naming it;
  in the worktree it cannot resolve (expected); of record it runs from `$C/code` (Mac) and the synced tree (PC).
- PREFLIGHT critical review (launched 17:37Z) still running.

## 2026-10-20 17:49Z UTC — G57/test_gen post-outcome branch written (codex-g57h1); one portability fix pending (codex-g57h1b)

- codex-g57h1 (report `preflight/b18f/codex/codex-g57h1-report.md`): gates/g57_configs.py +59/−~8, tests/test_gen.py +53:
  hold checks preserved verbatim when inputs.headline is None; when set: every config's headline equals it and is H1/H2;
  claim_domain unchanged; code_sha/init_tensors null; provenance names the stamped STUDY version and "§3.4 outcome" and
  inputs.json's sha matches; every config stamps gen STUDY_VERSION/SHA; the stamped snapshot's sha256 matches and its §3.4
  (authenticated bytes only) contains "Outcome of the v0.4 check"; no H2 unless recorded; headline never in unset_inputs;
  init_tensors and both handoff fields still required; B5-4 engineering admission kept by checking null-headline copies.
  New test_recorded_h1_outcome. Mac: test_gen 38 passed; offline 644 passed; G57 dirty 71 PASS / 3 FAIL (cleanliness only).
- Orchestrator review of the diff: not a weakening (every pre-outcome check survives behind the None branch); one defect:
  the snapshot path is hard-coded to the Mac lab root, which does not exist on the PC (G57 pcc would fail). codex-g57h1b
  launched 17:49Z: derive it from FORK.parent/review/snapshots; nothing else.
- Commit plan once v0.17 PASSES: constants → v0.17, regenerate, commit A (gen inputs+constants, gates/g57_configs.py,
  tests/test_gen.py), commit B (configs only), G57 clean at B. Branch only; main stays at c4dc2a80ca2b.

## 2026-10-20 17:48Z UTC — STUDY v0.17 written by the fixer (A1, B1 resolved); re-review requested (iteration 2)

- `$C/STUDY.md` v0.17 sha256 cb40f57443f91a63008672e4489db1ce9b3683fca3029102d9d805b55c7e8ea6 (2,169 lines); snapshot
  `review/snapshots/v0.17/` SHA256SUMS OK; +11/−6 over v0.16; rule blocks and §10 byte-identical; notes
  `review/amendment-v0.17/NOTES.md`. BF bullet now rests on the v0.4 owner-record clause (17:30Z, results-informed; outcome
  read 17:14Z), supersedes B14-7a's consequence line, says B14-7a is silent on a stopped 24 h copy, drops "no second
  reading", states the default and Kai's open 72 h alternative; READMEs cited (da4ml-test 87709d1d…, perpos-oneblock
  e93d6b84…). Time note: the chat report to Kai went out ~17:36Z; flags item 0b says 17:30Z (written in anticipation);
  the ledger 17:34Z entry attests the report.
- Re-review sent to the same critical reviewer (iteration 2) → `review/critical-v0.17-amendment.md`.
- Branch consequence: the generator constants will move to v0.17 (not v0.16) once it PASSES; the configs regenerate once
  more (stamp only) before the two-step commit.

## 2026-10-20 17:46Z UTC — STUDY v0.16 critical review: ITERATE (1 A, 1 B, 7 C); fixer launched for v0.17

- `review/critical-v0.16-amendment.md` (Opus): every number in the outcome block traced to source (WNS post_opt_timing.rpt:142;
  LUT post_opt_util.rpt:35; II; 0 C-sim mismatches; rc=124 at 28,800 s and 86,400 s; hashes from record.json); diff within
  scope; rule-block hashes unchanged. **A1**: at sealing (17:25Z) BF's failure lacked the v0.4 basis (no completed attempt,
  no owner "impossible to complete" record; v0.4 :576 gives only "cannot give a pass"); the owner record and the Kai report
  (17:30-17:36Z) supersede B14-7a's consequence line results-informed; the amendment misreads B14-7a (its reported stop is
  the 8 h one; silent on a stopped 24 h copy) and keeps a "no second reading" claim; v0.17 must cite both, label the
  supersession with its time, correct the reading, drop that claim, note Kai's open 72 h alternative. H1 for every binary
  arm is unaffected (FB alone excludes the both-pass case); binary admission and the one-pass branch depend on BF.
  **B1**: the da4ml-test and perpos-oneblock READMEs required beside the outcome ("Sequence and record" :580) are not cited.
  C4: the amendment's v0.4 citation is correct; the ledger's 546/549 were v0.3 lines (already corrected at 17:30Z).
- Fixer (lab agent) launched 17:46Z for v0.17 + snapshot `review/snapshots/v0.17/`; then the critical reviewer re-reviews
  (iteration 2). Consequence for the branch: the v0.16-stamped configs stay uncommitted; after v0.17 PASSES the generator
  constants move to v0.17 and the configs are regenerated once (stamp only), then the two-step commit.
- The first approval (B19-only, STUDY v0.15 bound) is unaffected; the PREFLIGHT critical review is still running.

## 2026-10-20 17:45Z UTC — headline configs verified; G57 and test_gen still assert the pre-outcome hold → post-outcome branch (codex-g57h1)

- codex-h1cfg2 (report `preflight/b18f/codex/codex-h1cfg2-report.md`): all 13 CH-8/F000 configs change only `headline`
  null→H1 beyond stamps/hashes; launchability: F111-FB-s1 and CH-8-s1 both report unset_inputs ['init_tensors'] (U1 draws
  fill it); no H2 config. tests/test_gen.py 35 passed / 2 failed; offline 641 passed / 2 failed; G57 (dirty, expect-sha
  ee30820d2ffa) 61 PASS / 7 FAIL = 3 cleanliness + 4 pre-outcome assertions (`g57_configs.py:562-569` "headline, code_sha
  and init_tensors remain null", "v0.4 headline hold remains for binary/PID", and two related).
- Those checks encode the pre-outcome state; the outcome is now recorded. codex-g57h1 launched 17:45Z: two-branch logic (hold
  checks unchanged when inputs.headline is None; when set: every config's headline equals it; provenance names the STUDY
  version/§3.4 outcome and the stamped STUDY snapshot contains "Outcome of the v0.4 check"; no H2 unless set; headline not
  in unset_inputs; init_tensors/code_sha still unset; claim_domain unchanged); same for the two test_gen assertions plus a
  new post-outcome test. Engineering/det null-headline admission checks (B5-4) untouched. The orchestrator reviews the diff
  for weakening before committing (branch only; main unchanged).

## 2026-10-20 17:41Z UTC — headline H1 + STUDY v0.16 stamp regenerated on branch b18fix (uncommitted); global headline accepted

- codex-h1cfg (report `preflight/b18f/codex/codex-h1cfg-report.md`) STOPPED per its brief: gen/inputs.json `headline` is one
  global input (line 4, null → "H1"), so all 167 configs changed `headline` null→H1, including the 13 CH-8/F000 configs the
  brief had reserved; plus stamps v0.15→v0.16, inputs/sources/config hashes, `unset_inputs` [headline, init_tensors] →
  [init_tensors] on 154 configs. No PID-target or launchable-flag change; nothing else.
- Orchestrator ruling: accepted. STUDY v0.16 §3.4 "CH-8 and F000 are read on the same count as the binary arms they are
  compared with (§3.1(c)); their training penalty is native under either (§3.1(b))"; gen/inputs.json's note: CH-8/F000 ramp
  training is count-independent. `headline: H1` on CH-8/F000 configs is the reading count, not a training change. The
  brief's constraint was too strict. codex-h1cfg2 launched 17:41Z: confirm CH-8/F000 diffs are headline-only beyond stamps,
  tests/test_gen.py, offline suite, G57 (dirty), launchability (binary config no longer unset on headline), no H2 configs.
- Then the orchestrator commits in two steps on b18fix (gen inputs + constants; configs-only) and reruns G57 clean. Main
  stays at c4dc2a80ca2b until approval one's B19 work ends (ledger 17:34Z decision).

## 2026-10-20 17:37Z UTC — /review 2026-10-16-chang-bitlinear PREFLIGHT started: validators v1 + critical-reviewer (Opus, solo)

- PREFLIGHT.md draft2 patched (stand-in cause, fix ee30820d2ffa on branch) and installed 17:35Z (3,292 lines; backup
  `preflight/b18f/PREFLIGHT.before-20261020-1736.md`; body from "Fork commits" to the B18f heading verbatim).
- Validators: `review/PREFLIGHT_validators_v1.txt` = nrp_doctor lint on every rendered manifest in the tree (Phase 2 venv
  job, the stage helper pod, the G78 smoke-transport jobs, the one handoff job); no B19 manifest exists before approval.
- Fan-out (RULES §7): 1 agent, critical-reviewer on Opus (PREFLIGHT); the v0.16 amendment critical-reviewer (Opus) is still
  running; no Codex job running. Artifacts under review: `$C/PREFLIGHT.md`, `preflight/b18f/final/GATES.md`,
  `preflight/b18f/targeted/GATES.md`, `$C/BRIEF.md` (= preflight/b18f/BRIEF.candidate.md v2), `$C/ADMISSION.json`.
- Verdict file: `review/critical-PREFLIGHT-v1.md`; verdict line also to `.claude/memory/review-reports.md`.

## 2026-10-20 17:35Z UTC — ADMISSION scope for approval one kept at 13 identities (reasoned) [corrected 17:58Z: the S1 hold is enforced in code via certification null, not prose-only]

- campaign.py `admit()`: mode `draw` checks only `draw_seeds` (L74-75); every other mode checks `config_path` against
  `admitted_configs` (L78). The six S1 identities are therefore not needed for U1 draws; they would matter only if a B19
  step (kill-and-resume, packing benchmark) submits an S1 config in a non-draw mode, which the record does not exclude.
  Decision: keep the 13; the installed BRIEF states (lines 17-19, 46-48) that they identify draws only and do not
  authorize S1 submissions under this approval; the S1 hold is prose and is re-stated in scientific_settings. If the
  critical reviewer asks for the mechanical narrowing, ADMISSION drops to the seven bench/det/smoke identities and the
  candidate is re-prepared (10 min). Recorded so the reviewer sees the reasoning.

## 2026-10-20 17:34Z UTC — decision: the headline pass runs from a second PC checkout outside the sync root; main frozen while approval one is live [corrected 17:58Z: main frozen only until the draws end; the rebind commit then lands on main]

- Hazard (advisor): the PC checkout `/home/kaimoe/lab/bnjettag/campaigns/2026-10-16-chang-bitlinear/code` is two-way synced
  with `$C/code` on the Mac, which the installed BRIEF binds at c4dc2a80ca2b. Resetting that checkout to the headline commit
  set for its pass would push those files into the Mac's `$C/code` working tree while approval one (B19) is live.
- Decision (orchestrator): the headline commit set (gen inputs H1 for binary arms, STUDY v0.16 stamp regen, the two test
  fixes ee30820d2ffa and c4dc2a80ca2b) is built on branch b18fix and its pass runs from a SECOND PC checkout outside the sync
  root (`/home/kaimoe/lab/scratch/cbl1016/code-h1/`, fetched from its own bundle), evidence written into the synced
  evidence tree as usual; `$C/code` main is not fast-forwarded until approval one's B19 work has ended and approval two is
  ready to be installed. Recorded before any headline commit lands on main.
- BF owner record reported to Kai in chat at 17:34Z (this entry supersedes the 17:30Z wording that anticipated it).

## 2026-10-20 17:31Z UTC — lossguard stand-in test: cause found (test bookkeeping, not the guard); fix committed on b18fix (ee30820d2ffa), main unchanged [corrected 17:58Z: the heap-layout explanation is withdrawn, unsourced]

- codex-lossguard-test (report `preflight/b18f/codex/codex-lossguard-test-report.md`): deterministic on the PC (CPU 0/10,
  CUDA 0/3; the pcg row forces CPU at test_lossguard.py:29,115), skipped on the Mac (mhab1012-cpu has no JAX → the Mac
  offline suite never exercised it). `3 < 3`: the retrace returned three entries, one wrong, not three correct penalties.
  Cause: the stand-in keyed its synthetic-id map by REAL_ID(obj) without retaining objects; address reuse after a deletion
  inherited the id and was counted. Passed at 29a459ce7ee6 only because 11 early FileNotFoundError failures changed the
  heap layout. **Not a guard failure**: cbl/lossguard.py:55 clears ids; :61-62 raises on census mismatch. Venvs: PC
  keras 3.14.0 / jax 0.9.2; Mac keras 3.15.0, no jax. Patched: CPU 10/10, CUDA 3/3, PC file 10 passed.
- Orchestrator read the +4/−1 diff and committed ee30820d2ffa on b18fix only. **Main stays at c4dc2a80ca2b** for the B19-only
  approval (the installed BRIEF binds it); the tests-b1 pcc FAIL row is a recorded exception with this cause. The fix
  joins the second commit set (headline H1 + v0.16 stamp + this), which gets its own near-full pass before the second BRIEF.
- Installed at 17:30Z: `$C/BRIEF.md` (sha 27a3d000…, = preflight/b18f/BRIEF.candidate.md v2) and `$C/ADMISSION.json`
  (160ba907…, 13 identities). `python3 tools/harness.py validate campaigns/2026-10-16-chang-bitlinear` (lab root): VALID,
  exit 0, authority sha 99199386d0f6…, "approval: none matching the authority block: submissions refused" (expected until
  Kai runs approve). Note for the reviewer: `$C/STUDY.md` is v0.16 (outcome record, review pending) while this BRIEF binds the
  v0.15 snapshot by design for the B19-only approval.

## 2026-10-20 17:30Z UTC — BF owner record and Kai notification; BRIEF candidate v2 VALID at c4dc2a80ca2b (B19-only scope)

- Correction to the "EXPORT CHECK OUTCOME" entry: the governing text is the v0.4 block's "Outcome and forms" (STUDY v0.16
  line 576: "A failure is a completed attempt that fails any criterion, or an attempt the owner records as impossible to
  complete, with the reason, reported to Kai before any binary launch. A missing or failed build, C-simulation, C-synthesis
  or Vivado result cannot give a pass"); lines 546/549 cited earlier are the superseded v0.3 block's fail-closed provisions,
  which agree. FB: completed attempt failing criterion 2 ⇒ failure. BF: not a completed attempt ⇒ needs the owner's record
  "impossible to complete, with the reason" + report to Kai before any binary launch.
- Owner record made 17:30Z (flags-for-kai.md item 0b; PREFLIGHT.md section via codex-preflight-md2): BF impossible to complete
  within the plan's synthesis budget (8 h, then the B14-7a escalation to 24 h; no larger budget in PLAN.md or STUDY §9); the
  trained CH-8 normal route timed out the same way. B14-7a's consequence line is superseded by this record. Reported to Kai
  in chat 17:30Z with the default (record stands, no resynthesis) and the alternative (a 72 h BF attempt; could only add the
  24 H2 runs since FB fails on timing).
- STUDY v0.16 (experiment-designer; `review/amendment-v0.16/NOTES.md`; snapshot `review/snapshots/v0.16/` sealed, SHA256SUMS
  OK): sha256 e7b321dbc82a6ea3cba5e6ae725ec5c7c0f1db3132a9d8a05ab90649ba0e3290, 2,164 lines; +40/−2 in five hunks (front
  matter, change log, §3.4 outcome block L585-617, A.1); v0.3/v0.4 block hashes and §10 unchanged. Also records: FB worst
  path through a BUFGCE to a softmax invert_table clock-enable, fan-out 236,154 / 238,134; CLB LUT utilization 171 % / 175 %
  (init-CH-8 252 %, WNS −1.172): the designs exceed the xczu7ev part in this OOC flow; context for VERIFY/REPORT.
  Critical review (Opus, solo) launched 17:28Z → `review/critical-v0.16-amendment.md`.
- codex-b18f-brief2: `preflight/b18f/BRIEF.candidate.md` v2 (v1 kept as `BRIEF.candidate.v1-38f4ebae3874.md`): code
  c4dc2a80ca2b, STUDY v0.15, MANIFEST 670fcf5f…, ADMISSION 160ba907… (13), bundle bbc14fe8…; frozen_files 55, no delta
  (tests outside the set); scope B19 only; S1 hold prose-only (the harness has no stage/mode allowlist) repeated in
  scientific_settings and the eligible_now draw/smoke lists; both gate records cited, targeted reason verbatim; validate2
  VALID exit 0; no hand edits.
- codex-preflight-md2 launched 17:29Z (completed gate records, export outcome section, open items) → draft2 for install.

## 2026-10-20 17:24Z UTC — targeted rerun at c4dc2a80ca2b: 3 of 4 suites PASS; one further failing test found (lossguard stand-in)

- `preflight/b18f/targeted/GATES.md` (Updated 17:21:38Z): tests-b2 pcc PASS 15; tests-b5 pcc PASS 86; tests-b10 pcc PASS 243;
  tests-b1 pcc FAIL 1 failed / 671 passed / 1 skipped; tests-offline mac-orchestrator 643 passed; G72 mac-orchestrator PASS.
  PC checkout c4dc2a80ca2b, clean; bundle sha equal both sides. The file-audit link test passes everywhere now.
- **Orchestrator miss, corrected by the targeted job:** the tests-b1 pcc row at 38f4ebae3874 read "2 failed, 670 passed"; I
  read only the first failing test name. The second is tests/test_lossguard.py::test_the_stand_in_reproduces_the_drop_without_the_guard,
  asserting `3 < 3`: the B10-F1 bug stand-in did not reproduce the loss-term drop on the PC CPU in that suite. It passed at
  29a459ce7ee6 (that row's 11 failures were all FileNotFoundError), failed at 38f4ebae3874 and at c4dc2a80ca2b. The GUARD's
  own tests pass (tests-lossguard pcg 10 passed). codex-lossguard-test launched 17:24Z (PID in job tmp): meaning of the
  assertion, 10× PC CPU / 3× in-suite / 3× PC GPU / 10× Mac, venv comparison, cause, minimal fix as an uncommitted diff.
- codex-b18f-brief2 launched 17:24Z: BRIEF rebind to c4dc2a80ca2b, both gate records cited, the tests-b1 row as a recorded
  exception for the B19-only approval (second BRIEF must carry its disposition), export outcome bound, S1 hold expressed.
- PC idle again except the lossguard probe (scratch only).

## 2026-10-20 17:18Z UTC — bundle c4dc2a80ca2b staged on kai-data; STUDY v0.16 drafting; first-approval scope fixed [corrected 17:58Z: approval 1 = U1 draws only; B19 cannot run without a rebind commit]

- cluster-ops-stage-c4dc2a80: `/data/cbl1016/code/cbl1016-main-c4dc2a80ca2b.bundle` (1,297,527 B), in-pod sha256
  bbc14fe8ff7b… = Mac; three bundles now on the PVC (f8ca91d7dce4, 38f4ebae3874, c4dc2a80ca2b), older two re-hash unchanged;
  `kai-` ConfigMaps 0/0; helper pod 17:16:37-17:17:50Z; records in `preflight/b18f/staging/STAGING.md` and cluster-inventory.
- experiment-designer-v016 launched 17:18Z: STUDY v0.16 = outcome record in §3.4 + change log + snapshot `review/snapshots/v0.16/`;
  no rule change. Solo critical review follows.
- First-approval scope (orchestrator): BRIEF binds c4dc2a80ca2b and STUDY v0.15; ADMISSION = the 13 eligible identities;
  authority covers B19 (U1 draws on the product of record, smoke, kill-and-resume, determinism record, packing benchmark,
  measured rate, production certification) and holds every S1 submission (CH-8 and F000 included) until STUDY v0.16 and the
  headline commit (gen inputs H1 for binary arms, configs regenerated) are of record under a second approval. Reason: STUDY
  §3.4 line 549 requires the outcome in the STUDY change log before any production training run, and the headline commit
  needs its own near-full gate pass (~6 h), which runs in parallel with B19 instead of delaying it.

## 2026-10-20 17:18Z UTC — EXPORT CHECK OUTCOME: FB FAIL (criterion 2), BF FAIL (no result); both fail ⇒ H1 for every binary arm

- codex-export-eval (report `preflight/export/codex/codex-export-eval-report.md`; rows `preflight/export/PREFLIGHT-rows.md`;
  gates G61-G70 at 0dc211ecbacd all exit 0 under `preflight/gates/G6x/0dc211ecbacd/mul/`; check exit 1 under
  `preflight/export/mulder/check/0dc211ecbacd/mul/`; evidence copied to `preflight/export/mulder/4f828bb5a8df/`):
  - FB (factored / normal): bit-exact yes/yes on the 4,096 validation jets; II=1 yes/yes; WNS **−1.090 / −1.087 ns**
    (post-opt, OOC synth_design + opt_design as PLAN step 5 prescribes); CLB LUT 394,351 / 402,252 (strictly fewer: yes).
    Criterion 2 (WNS ≥ 0 on the factored route) FAILS ⇒ FB FAIL.
  - BF: C-synthesis rc=124 at both the 8 h and the 86,400 s attempts (ended 09:29:54 PDT); Vivado never ran; no timing or
    LUT result at either cap. STUDY v0.15 §3.4 (line 546): a line that does not complete a stage fails every condition that
    stage enters ⇒ BF FAIL. chk-CH-8-normal (trained) also timed out: context row only, no criterion depends on it.
  - Outcome exists (line 574: FB and BF each have a recorded failure). **Split-form rule (Kai, 2026-10-19T16:28:12Z): both
    fail ⇒ H1 is the headline count for every binary arm, F100 included.** No one-pass case: STUDY §19.2 item 5 and the
    531.25 GPU-h cap do not arise; the 24 H2 runs do not launch.
- D-v0.4 debate NOT held (orchestrator): both readings are mechanical (a negative WNS; a missing stage), so there is no
  defensible "pass" position; the master prompt's debate exists for ambiguous readings. Recorded here as the verdict entry.
- Consequences and sequencing (orchestrator; advisor-reviewed principle "keep binary configs off the first approval's
  critical path"): (1) STUDY §3.4 line 549 requires the outcome, the per-line table and the report paths in PREFLIGHT and in
  the STUDY change log before any production training run ⇒ STUDY v0.16 amendment (experiment-designer) + solo critical
  review; it bumps the study stamp (configs regen, stamp-only commit) and the BRIEF binding. (2) gen inputs headline = H1 for
  binary arms + regeneration ⇒ a code commit whose gate pass is near-full (G49 census and PID/resume families read the
  headline), ~6 h on the PC. (3) Therefore: first approval on c4dc2a80ca2b covers **B19 only** (draws, smoke, resume,
  determinism, packing benchmark, certification), with a stated hold on any S1 submission until v0.16 and the headline
  commit are of record; the binary/headline commit set, its pass, BRIEF v2 and Kai's re-approval run in parallel with B19;
  RUN (CH-8, F000 and binary arms together) after the re-approval. Driver note: the 24 h CH-8/BF C-synthesis timeouts
  mean the trained-model normal route does not synthesize within a day at this size; a finding for VERIFY/REPORT context,
  not a stop.

## 2026-10-20 17:15Z UTC — test fix committed; FINAL commit now c4dc2a80ca2b; targeted rerun launched; bundle re-staging launched

- codex-testfix (report `preflight/b18f/codex/codex-testfix-report.md`): tests/test_data.py lines 262 and 265 gain
  `cwd=str(tmp_path)`; no other call site; Mac: test_data.py 14 passed / 1 skipped (inotify Linux-only); offline 643 passed.
  Orchestrator reviewed (2 lines) and committed on b18fix; `$C/code` main fast-forwarded to
  **c4dc2a80ca2b8726492f9c49d5de96b3a068f128** = 38f4ebae3874 + 1 test-only commit (29a459ce7ee6 + 7).
- New bundle `preflight/pc-setup/code-main-c4dc2a80ca2b.bundle`, sha256 bbc14fe8ff7b6550369211cc729230fbb7d6430e321b378ffd038384a610d212,
  `git bundle verify` ok, main = c4dc2a80ca2b (the 38f4ebae3874 bundle `code-main.bundle` kept).
- Orchestrator rows at c4dc2a80ca2b: tests-offline mac-orchestrator 643 passed / 4 skipped / 26 deselected; G72
  mac-orchestrator PASS (`preflight/gates/{tests-offline,G72}/c4dc2a80ca2b/mac-orchestrator/`).
- **Targeted rerun decision (orchestrator):** the diff between 38f4ebae3874 and c4dc2a80ca2b is tests/test_data.py only
  (2 lines); the full B18f record at 38f4ebae3874 stands for every other row; the four rows that failed on that test
  (tests-b1, b2, b5, b10 pcc) rerun at c4dc2a80ca2b. codex-b18f-targeted launched 17:15Z (PID 34234; brief
  `preflight/b18f/targeted/codex/codex-b18f-targeted-brief.md`; record to `preflight/b18f/targeted/GATES.md`; PC checkout
  reset to c4dc2a80ca2b via the new bundle). Main frozen again until it ends.
- cluster-ops-stage-c4dc2a80 launched (Sonnet) to stage the new bundle beside the two existing ones.
- Pending: codex-export-eval (PID 98542, since 17:10Z).

## 2026-10-20 17:10Z UTC — B18f pass done: FAIL 4 (one test), PASS 91 (+1 with skips), NOT RUN 34; mulder's three 24 h runs timed out

- `preflight/b18f/final/GATES.md` Updated 17:00:19Z (PC completion watcher): **FAIL 4, NOT RUN 34, PASS 91, PASS WITH SKIPS 1**,
  no RUNNING/PENDING; 96 of 96 assigned rows have exit records; PC log ends `BATCH_DONE pc`. Both code trees clean at
  38f4ebae3874. Every previously failing gate PASSES: G08 pcc 36; G13 pcc 51 / pcg 56; G14 pcc/pcg 42; G46 pcc 65;
  G48 pcc/pcg 165; G49 pcc 130 (109/109 reload); G51 pcc 25 / mac 12; G57 mac 68 / pcc 32. tests-b1 pcc 670 passed, 1 skipped.
- The 4 FAIL rows (tests-b1, b2, b5, b10 pcc) are one Linux-only test, tests/test_data.py::test_fileaudit_sees_h5py_opens_through_links
  (skipped on macOS): it calls run_audited without `cwd`; the helper falls back to os.getcwd() = the synced PC checkout and
  the 38f4eba refusal fires (SystemExit; traceback in `gates/tests-b2/38f4ebae3874/pcc/stdout.txt`). Its files are in
  scratch; every gate passes cwd=work. **Cause known; test-only fix** (cwd=tmp_path): codex-testfix launched 17:09Z
  (PID 98349, worktree cbl1016-b18fix). Plan: commit → ff main (the pass is over) → targeted rerun on the PC of the four
  suites + tests-offline mac + G72 mac (orchestrator), recorded as targeted with this reason (diff: tests/test_data.py
  only) → bundle re-created and re-staged → BRIEF rebind to the new sha.
- mulder: no vitis_hls/vivado process remains; the three `prj24` logs end mid-compilation; `*.synth24.stdout` empty →
  consistent with the 86,400 s timeout. STUDY v0.15 §3.4 (line 546): a line that does not complete a stage fails every
  condition that stage enters; outcome exists when FB and BF each have a recorded pass or failure (line 574). So once
  recorded: BF = FAIL (no C-synthesis at either cap); FB = its three criteria from the Vivado results; CH-8 trained normal
  is context only. codex-export-eval launched 17:10Z (PID 98542) to record the facts (G61-G70, PREFLIGHT-rows.md).
- Orchestrator error: the mulder wait loop's pgrep pattern matched its own ssh shell and never fired; mulder's end was
  noticed at 17:06Z. Process checks must exclude the shell (`grep -v "bash -c"`).

## 2026-10-20 12:00Z UTC — PREFLIGHT.md refreshed (codex-preflight-md draft installed by the orchestrator)

- Draft `preflight/b18f/PREFLIGHT.draft.md` (3,188 lines; +411 over the 2,770-line file; only the six stale header lines
  replaced; body from "Fork commits" preserved) reviewed by the orchestrator (header, B18 final pass, B18f sections) and
  installed as `$C/PREFLIGHT.md`; previous file kept as `preflight/b18f/PREFLIGHT.before-20261020.md`; diff
  `preflight/b18f/PREFLIGHT.diff`; report `preflight/b18f/codex/codex-preflight-md-report.md`.
- New sections: record update 2026-10-20; B17; B18P; B18 preparation and B10-F1 integration; B18 final gate pass at
  29a459ce7ee6 (eight FAIL rows with cause and disposition; field-level G46/G48 evidence; G13/G51; held-out incident, Kai's
  ruling PENDING; orchestrator rows); B18f (six commits with verification; D1/D2; gates at 38f4ebae3874 PENDING; limitations:
  mid0 vacuous, attribution, Mutagen flag, logging baseline, B19 evidence due; binary sequencing). Markers to fill after the
  pass: "Gates of record at 38f4ebae3874: PENDING".
- Not written to `$C/code`. Mutagen will sync PREFLIGHT.md to the PC (not read by any gate).

## 2026-10-20 11:55Z UTC — final bundle staged on kai-data (cluster-ops-stage-b18f, Sonnet)

- `/data/cbl1016/code/cbl1016-main-38f4ebae3874.bundle` (1,294,545 B), in-pod sha256 2c0c6ba8fb489222fadbc2d88f8cbc22264f6e580424add5488764d997d5c2f1
  = Mac value; bundle head main = 38f4ebae387413c50eaaa747844a3e2d028759c4. Earlier `cbl1016-main-f8ca91d7dce4.bundle`
  untouched (605dfae1…). `/data/cbl1016/code/` holds exactly the two bundles. PVC 100G, 51G used.
- Helper pod `kai-cbl1016-stage-helper` created from the existing `helper-pod.json` (CPU request, no GPU, no ConfigMap, 14,400 s
  deadline; lint exit 0; one hook block on a relative path, retried absolute), Ready 11:53:13Z, deleted 11:53:59Z; no `kai-`
  pods remain. `kai-` ConfigMaps 0 before and after. Copy by `kubectl exec -i` to a `.part` file, `sha256sum -c`, then move.
- Unverified: `git bundle verify` inside the pod (bytes hashed only). Records: `preflight/b18f/staging/STAGING.md`;
  `.claude/memory/cluster-inventory.md` top entry. Hook note (agent, and reproduced by the orchestrator while writing this
  entry): the pre-kubectl lint hook blocks any Bash command whose text contains the kubectl create/apply flag-and-path
  pattern, even inside a heredoc or a quoted note; write such text with the Write tool or reword it.
- Roadmap step 6 (stage the bundle) done ahead of the review; remaining before the approve command: b18f GATES.md,
  PREFLIGHT.md install, BRIEF rebind + install with ADMISSION.json, harness validate, critical review PASS.

## 2026-10-20 11:54Z UTC — B18f driver exited by design at 11:53Z; PC batch and a PC completion watcher carry the pass

- codex-b18fgates (PID 85407) finished its setup and exited 11:53Z (report `preflight/b18f/codex/codex-b18fgates-report.md`):
  bundle sha equal on both machines; PC checkout reset to 38f4ebae3874 with empty porcelain status
  (`b18f/final/pc-setup-38f4ebae3874.txt`); work/keep/child scratch/TMPDIR outside Mutagen; no WANDB_MODE forced; GPU rows
  with deterministic XLA ops. **Mac rows: 14 of 14 PASS** (incl. G76, offline G78). PC batch PID 2717473 running G49 (5/109
  reload checks passed at exit time), 81 rows queued (`b18f/final/pc-queue.txt`); PC completion watcher PID 2720193 waits for
  `BATCH_DONE` then refreshes the ledger (results.json), audit, coverage, GATES.md and the report. PC logs:
  `~/lab/scratch/cbl1016/b18f-38f4ebae3874.log`, `b18f-finish.log`. Interim GATES.md: PASS 14, NOT RUN 115, PENDING 1 (G49 pcc).
- The caffeinate hold (PID 86110) ended with the driver; nothing on the Mac drives the pass now. Orchestrator loop polls the
  PC every 10 min for `BATCH_DONE` (dies with the session; recovery line in the handoff).

## 2026-10-20 11:52Z UTC — BRIEF candidate refreshed at `preflight/b18f/BRIEF.candidate.md`: VALID (exit 0)

- codex-b18f-brief (report `preflight/b18f/codex/codex-b18f-brief-report.md`): bound code 38f4ebae387413c50eaaa747844a3e2d028759c4,
  STUDY v0.15 e4a6b89e…, MANIFEST 670fcf5f…, installed ADMISSION 160ba907… (13 identities, D1; template unchanged),
  bundle 2c0c6ba8… (`preflight/pc-setup/code-main.bundle`, created by the pass job; `git bundle verify` exit 0, main =
  final commit). frozen_files 55 (B18 rule + gates/syncroot.py, tests/b17/lab_root.py, tests/test_resumegate.py,
  tests/test_syncroot.py; cbl/lossguard.py enters by rule). D2, B10-F1 merged, smoke-only draws until the B19 rebind,
  benchmark timing, the Mutagen incident and the pending b18f gates are reflected. No hand edits. Validation (same isolated
  harness validate as B18): VALID, exit 0, "approval: none matching the authority block: submissions refused" (expected).
- Still to rebind after the pass: the gates-of-record citation (b18f GATES.md). Then install at `$C/BRIEF.md` + `$C/ADMISSION.json`.
- cluster-ops launched to stage the bundle on kai-data (same file, sha 2c0c6ba8…); see next entry.

## 2026-10-20 11:49Z UTC — BRIEF refresh running; caffeinate hold; G57 evidence saved; flags and handoff updated

- codex-b18f-brief launched 11:45Z (Mac PID 85888; workdir `preflight/b18f`, writes only there; brief
  `preflight/b18f/codex/codex-b18f-brief-brief.md`): new BRIEF candidate bound to 38f4ebae3874, STUDY v0.15, MANIFEST,
  installed ADMISSION (13 configs, D1), D2 wording, B10-F1 merged, wording fixes; `harness.py validate` on the candidate.
- Pass job 85407 alive (Mac gates G55, G56 done at +3 min). `caffeinate -i -w 85407` PID 86110 holds off idle sleep.
- G57 PASS 68/68 at 9411e4fa796d copied from the job tmp dir to `preflight/gates/G57/9411e4fa796d/mac-orchestrator/`.
- `notes/flags-for-kai.md`: 2026-10-20 decisions added; new open item 0 (Mutagen held-out read, default incident not a stop).
- `notes/handoff-2026-10-20-0445.md`: caffeinate row; manual GATES.md recovery line.
- Advisor sequencing adopted: main stays frozen until the pass ends; the binary-config commit after the headline gets a
  targeted rerun (G57, configs/statistics family, test suites, admission rows), recorded; CH-8/F000 approval does not wait
  for it. PREFLIGHT.md refresh starts now as a draft under `preflight/b18f/` (moved into place after review).

## 2026-10-20 11:44Z UTC — B18f gate pass launched (Codex PID 85407); handoff-2026-10-20-0445.md written

- First launch (11:41Z) was refused by Codex: "Not inside a trusted directory and --skip-git-repo-check was not specified"
  (`$C/preflight` is not a git repo); log kept as `preflight/b18f/codex/codex-b18fgates.attempt1-refused.log`. Relaunched
  11:42Z with `--skip-git-repo-check`; header shows workdir `$C/preflight`, network on, writable `$C/code`; alive at +10 s.
- Watcher: Claude background shell (dies with the session). Expected end ~17:45Z (previous pass ~6 h).
- `notes/handoff-2026-10-20-0445.md` written (seven sections) at the end of the "close B18" step; next steps 2 (BRIEF refresh)
  and 3 (export evaluation) can proceed while the pass runs, without touching `$C/code`.

## 2026-10-20 11:42Z UTC — sync-root hardening committed; FINAL commit 38f4ebae3874 on main; G72 and offline suite PASS there

- codex-g51fix (report `preflight/b18/codex/codex-g51fix-report.md`): new `gates/syncroot.py` (sync_roots, under_sync_root,
  require_unsynced), fileaudit.run_audited refuses synced watch paths or child cwd (= the gate's work dir; every caller passes
  it), g51_heldout.check_work_paths before any probe/hard link, same check in G06, G07, G08, G12, G34, G36, G49, G56, G57,
  G60 and the shared runnergate/pidgate/resumegate helpers; tests/test_syncroot.py 18 passed; offline 643 passed.
  Orchestrator checks before commit: all run_audited calls pass cwd=work; gates take --work explicitly (no defaults in the
  tree); PC held-out archive `/home/kaimoe/lab/scratch/cbl1016/data/150c-test.h5` and raw dir `/home/kaimoe/lab/bnjettag-data/val`
  are outside the sync roots. Committed 38f4eba on b18fix; `$C/code` main fast-forwarded to
  **38f4ebae387413c50eaaa747844a3e2d028759c4** = 29a459ce7ee6 + 6 commits. Main is now frozen for the B18f pass.
- Orchestrator rows at 38f4ebae3874: tests-offline mac-orchestrator 643 passed / 4 skipped / 26 deselected; G72
  mac-orchestrator PASS (`preflight/gates/{tests-offline,G72}/38f4ebae3874/mac-orchestrator/`).
- B18f gate-pass brief as launched: `preflight/b18f/codex/codex-b18fgates-brief.launched.md` (sha 38f4ebae3874; work dirs
  outside sync roots; record to `preflight/b18f/final/`). First launch attempt 11:41Z: process exited at once; see next entry.

## 2026-10-20 11:32Z UTC (04:3x PDT) — investigator verdicts: G13 pcg and G51 pcc are artifacts of synced work dirs; Mutagen read held-out bytes (to Kai); G72 PASS at 84f372fbceac

- investigator-g13-g51 (Opus; ticket `preflight/investigations/G13-G51/TICKET.md`, 17.8 KB; PC released 11:30Z):
  - G51 pcc: artifact. The one forbidden event (`pcc/children/draw-s1.audit.json`) is an IN_OPEN of the held-out
    archive's inode by Mutagen's agent, after the gate hard-linked that inode into the Mutagen-synced `--work`
    (`g51_heldout.py:187`). The draw child held no held-out fd. At 267c6deb053e, with `--work` outside the sync root, 0 events.
  - G13 pcg: artifact. run-B (same program, seed, args, same gate run) saw 0 events; run-A's fd inventory has no
    placeholder; the draw child (never reads the placeholder) also saw an open in both synced runs and none in scratch runs.
    The opener cannot be reconstructed; today's probe shows the Mutagen agent (PC PID 2708768) opening a new synced file
    within 0.1 s. Neither row shows training, selection or evaluation code reading the held-out set.
  - Origin: operator code `preflight/b18/final/prepare_final_commit.py:35-36` put `--work` inside the synced tree
    (not code of record). Fix: work/keep/tmp outside sync roots (operator); G51 and fileaudit refuse synced paths (code).
  - **For Kai (STUDY §14):** through G51's hard link (about 30 s), Mutagen read the held-out archive's bytes (448,219,074 B);
    whether any reached the Mac is undetermined; no copy exists now on either machine (checked: sizes, ~/.mutagen/staging
    empty, link count 1, no snapshots). No campaign code path read them. Orchestrator's recommended reading: an incident
    with a fix, not a §14 stop (no model, selection or human saw the bytes). Asked in chat 11:32Z.
  - Also: the PC working tree is dirty since 04:17:40 PDT (Mutagen synced the Mac's new code files, not `.git`); the
    investigator's one rerun (11:21-11:22Z) was spoiled by it; the B18f bundle fetch + reset repairs it. Rule for the
    pass: nothing under `$C/code` on the Mac changes while PC gates run (Mutagen would push it mid-gate).
  - regression_log line (ticket §7) appended to `$C/regression_log.md` by the orchestrator.
- codex-g51fix launched 11:31Z (brief `preflight/b18/codex/codex-g51fix-brief.md`): gates/syncroot helper; G51 and
  fileaudit refuse paths under a sync root; tests. Then review, commit, ff main → final sha for the B18f pass.
- Orchestrator-run rows at 84f372fbceac: G72 mac-orchestrator PASS (581 checks); tests-offline mac-orchestrator exit 0
  (`preflight/gates/{G72,tests-offline}/84f372fbceac/mac-orchestrator/`). Both will be rerun if the final sha moves.
- B18f gate briefs updated with the operator fix (work dirs outside sync roots; dirty-tree repair; refusal is correct).

## 2026-10-20 04:28 PDT — D2 committed and cherry-picked; configs split into gen + configs-only commits; G57 PASS 68/68; main at 84f372fbceac

- G57 at the combined gen+configs commit e6db90e395d6 failed one check ("the last commit that touched configs/ touched only
  configs/"); the tip was split (branch local, not on main): 319c52f gen constants, 9411e4fa796d configs-only. G57 at
  9411e4fa796d: PASS, 68/68 (`~/.claude/jobs/f4eee4ae/tmp/g57-9411e4fa796d/stdout.txt`).
- codex-d2 report `preflight/b18/codex/codex-d2-report.md`: tools/campaign.py +14/-3 (never_started only with terminal
  evidence and every observed pod allocation_started False; budget charges 0), tests/b17/test_tools.py +47 (9 cases);
  tests/b17 108 passed; offline 625 passed. Orchestrator read the diff, committed on d2, cherry-picked onto b18fix.
- Correction: the first cherry-pick call failed (bad flag) and main was briefly fast-forwarded to 9411e4fa796d without D2
  (offline suite 616 there); redone: D2 cherry-picked onto b18fix, `$C/code` main fast-forwarded to 84f372fbceac1747fd45ee865f3142d04b3815e0 = 29a459ce7ee6 + 5 commits (G08/tests lab root; resume comparator sys/ drop;
  gen v0.15 constants; configs regen; D2 budget). Offline suite rerun at the tip by the orchestrator after the redo: 625 passed, 4 skipped, 26 deselected.
- This is the final-commit candidate for the B18f gate pass; D1 is an installed-file change at the BRIEF step.
- Waiting: investigator-g13-g51 (deadline ~06:10 PDT); mulder loop.

## 2026-10-20 04:25 PDT — configs regenerated against STUDY v0.15: commit e6db90e395d6 on b18fix

- codex-b18cfg stopped per its rule (derived hashes moved); report `preflight/b18/codex/codex-b18cfg-report.md`. Orchestrator
  checked the diff keys across all 169 files: study_version 336, study_sha256 336, sha256 336, sources_sha256 334, plus the
  gen/__init__.py entry; nothing else. gen/inputs.json unchanged (od13/od15/od12 true; claim_domain b / default / c3r false,
  confirmed 2026-10-19T17:13:14Z). tests/test_gen.py 37 passed. G57 in the dirty worktree: 65 PASS, 3 FAIL (git cleanliness
  only). Committed e6db90e395d6; G57 rerun at that sha recorded in `~/.claude/jobs/f4eee4ae/tmp/g57-e6db90e395d6/stdout.txt` (see next line).
- Walkthrough page updated to version 4 (commits, dispositions, D1/D2 settled).

## 2026-10-20 04:22 PDT — D1/D2 mechanics known; codex-d2 launched (worktree cbl1016-d2, branch d2 from d8dfe039382d)

- Worker admission-mechanics (Sonnet; detail `~/.claude/jobs/f4eee4ae/tmp/admission-mechanics.md`): D1 = installed
  `$C/ADMISSION.json` only (campaign.py `checked_submit` L184-187 reads `root/'ADMISSION.json'` and checks its digest
  against BRIEF frozen_files); `code/tools/admission.json` is only draft_brief.py's template; `eligible_now`/`headline_hold`
  are BRIEF prose; the hold is already enforced in code by gen/configs.py `check_launchable` (UnsetInput 'headline' for
  non-CH-8/F000 arms; called from cbl/runner.py:373-381 and nrp/spec.py). → D1 is done at the BRIEF step, no code change.
  D2 = code: tools/campaign.py `budget()` L26-55 + a test; a Job with no pod records is never final (harness.py
  `_update_resources`); only a pod record with allocation_started False proves never-started (test_tools.py L279-300).
- codex-d2 launched 11:22Z: brief `preflight/b18/codex/codex-d2-brief.md`; sets `never_started` only for terminal Jobs whose
  observed pods all have allocation_started False; budget charges 0 then; unknown/live keep the bound; tests (a)-(d).
- Merge plan: b18cfg configs commit → b18fix; d2 commit → b18fix (no file overlap except tests/b17/test_tools.py, already
  shared); main ff; gate pass at that sha.

## 2026-10-20 04:21 PDT — Kai: "defaults for D1 and D2" (recorded as decisions.md 2026-10-20 (2))

- D1 yes (admission enforces the hold; 13 eligible-now configs until the headline), D2 yes (never-started Job charged 0;
  failed Job charged measured time; full bound only while live/unknown). Both to be implemented before the B18f pass.
- Waiting on worker admission-mechanics for the exact files and the minimal D2 patch; then one Codex job (codex-d1d2) in the
  b18fix worktree after codex-b18cfg exits; orchestrator reviews and commits; main fast-forwards; gate pass at that sha.

## 2026-10-20 04:21 PDT — sequencing before the B18f gate pass: configs → v0.15 first; D1/D2 mechanics checked; gate brief ready

- Gate-pass brief for the fix commit written: `preflight/b18f/codex/codex-b18fgates-brief.md` (derived from the extracted
  B18 brief `preflight/b18/codex/codex-b18gates2-brief.extracted.md`; record goes to `preflight/b18f/final/`; STUDY reads
  pointed at v0.15 e4a6b89e…; corrections: no forced WANDB_MODE, G72/tests-offline by the orchestrator, repoint every path,
  G14 from the clean rerun). The sha in it is replaced with the final commit at launch.
- Sequencing decision (orchestrator): `configs/MANIFEST.json` still stamps STUDY v0.13; regenerating against v0.15 is a
  code-tree commit, so it lands BEFORE the pass to avoid a second ~6 h pass. codex-b18cfg launched 11:20Z in the b18fix
  worktree (brief `preflight/b18/codex/codex-b18cfg-brief.md`; stop rule: if any non-stamp field changes, stop and report).
- D2 (reserve accounting) is a change to `code/tools/campaign.py`; D1 may be installed-file only. Worker admission-mechanics
  (Sonnet) checks what reads ADMISSION.json/admission.json and what D2's minimal patch is. If Kai answers before the pass,
  the changes go into the final commit; if not, the pass runs without them and a later D2 commit gets a targeted rerun of
  the affected gate families and test suites, recorded as such (or a full pass if the critical reviewer asks).
- Gate pass launch conditions: investigator has released the PC; configs commit on main; Kai's D1/D2 answers if available.

## 2026-10-20 04:17 PDT — comparator fix committed; main fast-forwarded to d8dfe039382d (new final-commit candidate)

- codex-b18fix2 exited 11:16Z (3 min; report `preflight/b18/codex/codex-b18fix2-report.md`): `gates/resume_child.py`
  +2/-5 (prefix drop of `sys/` in rows and epoch_records), `gates/resumegate.py` +13/-6 (`_scientific_rows` applied in the
  full-run, epoch_records and pre_resume_agree comparisons; unused ROW_DROP removed), new `tests/test_resumegate.py`
  (6 tests). Verified: 6 passed; offline suite 616 passed / 4 skipped / 26 deselected; replay of the saved G46 children:
  True at points 10 and 14, False with perturbed log_accuracy (`preflight/b18/codex/b18fix2-replay/output.txt`).
  mid0 vacuous agreement confirmed and left unchanged (noted for a later fix).
- Orchestrator read the diff and committed on b18fix; `$C/code` main fast-forwarded from 29a459ce7ee6 to d8dfe039382d59a76216ea31f21b75e1113d576b
  (two commits: f60d3bcd1b69 G08 + tests/b17 lab root; this one). No run uses the code yet.
- Next: gate pass at d8dfe039382d on the PC (Codex, same procedure as `preflight/b18/final/`) after the investigator
  releases the PC; if G13/G51 need a gate fix, that commit comes first so there is one pass.

## 2026-10-20 04:14 PDT — codex-b18fix done; commit f60d3bcd1b69 on branch b18fix (G08 text + tests/b17 lab root)

- Clock correction: the two entries below this one were first headed 01:05 and 01:25 PDT; file mtimes (decisions.md 11:04Z,
  codex-b18fix-brief.md 11:09Z, tests-offline evidence 11:10Z) show 04:05 and 04:12 PDT. Headers fixed; the launches were
  at 04:09-04:10 PDT, so the investigator's 2 h window runs to about 06:10 PDT.
- codex-b18fix exited 11:13Z (report `preflight/b18/codex/codex-b18fix-report.md`): `gates/g08_heldout.py` +3/-3 (full
  exception text kept for the checks, display truncated), new `tests/b17/lab_root.py` (23 lines), `test_tools.py` +9/-1,
  `test_nrp.py` +2/-1. Verified: tests/b17 99 passed; full offline suite 610 passed / 4 skipped / 26 deselected; G08
  attempt() regression with a long-path manifest passes the unchanged predicate; both tools exist on the PC.
- Orchestrator read the diff (no check changed; resolver falls back to the known roots when the worktree is outside the
  lab) and committed f60d3bcd1b69 on b18fix (not yet on main; main stays at 29a459ce7ee6 until the comparator fix lands too).
- codex-b18fix2 (comparator: drop every `sys/` key) launched 11:13Z in the same worktree, Mac PID 12956, log
  `preflight/b18/codex/codex-b18fix2.log`, replay under `preflight/b18/codex/b18fix2-replay/`.
- Investigator-g13-g51 (Opus): launched ~08:20Z with a 2 h cap; no report yet at 11:15Z.

## 2026-10-20 04:12 PDT — G46/G48 cause found: comparator compares process telemetry; tests-offline PASS outside the sandbox

- Worker g46-g48-diff (Sonnet): in every resumed run only `sys/host_rss_mb` (and on GPU `sys/alloc_hours`) differs; loss,
  accuracy (float.hex), beta, step, batch-order and index-map hashes, train_count, subtotals and the PID block are identical.
  `gates/resume_child.py:53` DROP_KEYS drops only `sys/epoch_seconds` and `sys/frozen_seconds`; `cbl/frozen_eval.py:364` and
  `cbl/runner.py:710-713` later added the two telemetry columns. Comparator `pre_resume_agree`, `resumegate.py:264-272`,
  exact equality. Side finding: the `mid0` subcase compares `rows[:0]` (vacuous). Detail `~/.claude/jobs/f4eee4ae/tmp/g46-g48-diff.md`.
- Orchestrator spot-check on the saved records (`G46/29a459ce7ee6/pcc/work/whole-run/children/wr-PID-CH-U` vs
  `wr-PID-CH-boundary10-a2-r`): 14 rows each; 36 fields identical; the only differing key `sys/host_rss_mb` (14/14 rows).
  **Cause known: gate defect, not non-determinism, not a training defect.** The single-thread experiment is not needed.
- Disposition: comparator fix (drop every `sys/` key from the equality comparison; scientific fields unchanged) as part of the
  next commit; G46/G48 rerun in the full gate pass at that commit. Brief written: `preflight/b18/codex/codex-b18fix2-brief.md`;
  launches in the same worktree when codex-b18fix exits (two Codex jobs in one worktree would collide).
- tests-offline mac: rerun by the orchestrator outside the Codex sandbox → `preflight/gates/tests-offline/29a459ce7ee6/
  mac-orchestrator/`: exit 0, 610 passed, 4 skipped, 26 deselected (22.66 s). Cause of the FAIL row confirmed as the sandbox.
- Live FAIL rows now: G08 pcc (fix in progress), G13 pcg and G51 pcc (investigator), G46 pcc / G48 pcc / G48 pcg (cause
  known, fix queued), tests-b1 pcc (fix in progress), tests-offline mac (disposed: PASS outside sandbox).

## 2026-10-20 04:05 PDT — Kai: trimmed path adopted, arms and runs approved as designed; triage launched

- Kai (chat, ~11:00Z = 04:00 PDT): "okay this all seems good, can u jsut keep going, and dont archive this but rather keep going til we run
  the training, i approve all of the arms and the runs and their desgins are good and very intentional". Recorded in
  `.claude/memory/decisions.md` "2026-10-20 (1)". Read as the recommended trimmed path; formal approval remains his
  `harness.py approve` once BRIEF.md exists. This session continues until training is submitted.
- Launched (fan-out: 3 agents, Codex Astra + Opus + Sonnet, plus two orchestrator background loops):
  - `codex-b18fix` (gpt-6-astra high, network on): worktree `~/lab/worktrees/cbl1016-b18fix`, branch b18fix from
    29a459ce7ee6; FIX 1 G08 refusal truncation (`gates/g08_heldout.py` ~L76, check unchanged), FIX 2 tests/b17 lab-root
    resolver (test_tools.py L15, test_nrp.py L219) with skip-with-reason; no commit. Mac PID 12034; brief
    `preflight/b18/codex/codex-b18fix-brief.md`; log `codex-b18fix.log`; report `codex-b18fix-report.md`.
  - `investigator-g13-g51` (lab investigator agent, Opus): one short look, ≤2 h, owns the PC for that window; may rerun
    G13 pcg once under a PID-attributing audit; writes only `preflight/investigations/G13-G51/` and PC scratch
    `~/lab/scratch/cbl1016/inv-g13/`. Verdict per gate: artifact / real read (§14 stop) / undetermined (→ Kai).
  - `g46-g48-diff` (Sonnet, read-only): which fields differ, float-only vs structural, thread settings, and the argv for a
    single-threaded fresh-run pair as the decisive experiment. Output `~/.claude/jobs/f4eee4ae/tmp/g46-g48-diff.md`.
  - Orchestrator: tests-offline rerun outside the sandbox → `preflight/gates/tests-offline/29a459ce7ee6/mac-orchestrator/`.
  - Orchestrator: mulder wait loop (every 10 min, 11 h cap) ends when no `vitis_hls` process remains; then the export
    evaluation job is launched with "runs are done" (no long Mac poller again).
- Sequencing: the gate pass at the fix commit starts only after (a) the fix is committed and (b) the investigator has
  released the PC (fresh-run agreement is load-sensitive). D1 asked in chat; D2 after D1.
- Open: D1, D2; G46/48 disposition after the characterization; export outcome; BRIEF refresh; critical review; bundle;
  approve command.

## 2026-10-20 00:45 PDT — GATES.md regenerated from the finished batch: FAIL 8, PASS 88 (+1 with skips), NOT RUN 33, RUNNING 0

- Worker evidence-sync-check (Sonnet, read-only): Mac and PC trees both hold 100 gate/place dirs for 29a459ce7ee6; all 82
  pairs the batch logs name exist on both; file-level diff of every pc* place: 25,569 vs 25,568 files, the one difference a
  pytest `__pycache__` temp file. summarize.py reads only the Mac tree. Listing: `~/.claude/jobs/f4eee4ae/tmp/evidence-sync-check.md`.
- Orchestrator ran `python3 -B b18/final/audit_evidence.py` (G49 reload 109/109 pass; evidence complete 97, nonzero 8;
  audit issues 0) and `python3 -B b18/final/summarize.py` from `preflight/`. Stale copies kept:
  `b18/final/GATES.stale-20261020T0622Z.md`, `results.stale-20261020T0622Z.json`. New header Updated 2026-10-20T07:36:02Z.
- G14 pcc and pcg now PASS (42 checks). New FAIL rows that the stale file listed as NOT RUN:
  - **G51 pcc**: "draw-s1: open-file audit, no open of the held-out archive or any raw held-out file (1 forbidden events)";
    stdout line 37 of the same run: kernel audit saw 0 placeholder opens and 0 held-out opens, "every refusal came before
    the open". Lead: the guard's self-test (line 35) may be the counted event. Unresolved; same flavour as G13 pcg.
  - **tests-b1 pcc**: 11 failed, 628 passed, 1 skipped; all 11 are FileNotFoundError for the Mac-absolute paths
    `/Users/kaiyamaguchi/lab/bnjettag/tools/harness.py` (9) and `tools/run_handoff.py` (2) in tests/b17/test_tools.py and
    test_nrp.py. Cause known: tests hard-code the Mac lab root; the same files pass on the Mac (tests-offline mac, 609 passed).
    Disposition: test-only path fix in the next commit, or a recorded PC exclusion.
- Live FAIL rows: G08 pcc, G13 pcg, G46 pcc, G48 pcc, G48 pcg, G51 pcc, tests-b1 pcc, tests-offline mac.
- Walkthrough page updated with these counts and republished (same URL).

## 2026-10-20 00:55 PDT — walkthrough page published; peer sessions all offline after the reboot

- `notes/walkthrough-2026-10-20.html` (37 KB) published as https://claude.ai/artifact/JKFpKpHPBjMwraVxLDbnuC
  (private; Kai's own). Content from the three dossiers (timeline, study, prior) plus the live state at 00:30 PDT.
  Time estimates on it name their basis (one full gate pass measured at ~6 h, 18:05-23:55 PDT 10-19).
- ListAgents after the reboot: every peer session shows offline; no second orchestrator exists; the "change nothing"
  hold from 23:20 PDT is moot. This session's id is now 627d46 (was 783ef8; the handoff names 783ef8).
- Open for Kai (asked on the page and in chat): the level of rigour from here, recommended default the trimmed path
  (CH-8 and F000 training first). Nothing relaunched; nothing committed.

## 2026-10-20 00:30 PDT — Mac rebooted 00:15 PDT; PC batch done; G14 rerun PASS; G48 pcg FAIL; both Codex jobs dead

- `pmset -g log`: boot at 2026-10-20 00:15:24 PDT. Both Codex jobs (Mac PIDs 69676 gate pass, 11519 export check) and
  the two detached copiers (28501, 28502) died with it; their last log lines (07:02-07:11Z) are network reconnect errors.
  Not stopped by this session. Final copies of both logs made by hand at 00:30 PDT:
  `preflight/b18/codex/codex-b18gates2.log`, `preflight/export/codex/codex-export.log` (snapshots from 23:37 kept).
- PC batch finished: `~/lab/scratch/cbl1016/b18-29a459ce7ee6.log` ends `BATCH_DONE pc` (tests-pc_gpu pcg exit 0,
  06:55Z). G14 clean rerun ran (`b18-rerun-29a459ce7ee6.log`): `DONE G14 pcc exit 0` → G14 now PASS at 29a459ce7ee6;
  old dir moved to `G14/29a459ce7ee6/pcc-attempt-forced-wandb-disabled`.
- G48 pcg FAIL (same CH-8-boundary2 fresh-run disagreement as pcc). GATES.md (Updated 06:22:59Z, before the batch end):
  FAIL 7 (G08 pcc, G13 pcg, G14 pcc [stale], G46 pcc, G48 pcc, G48 pcg, tests-offline mac), NOT RUN 49, PASS 73,
  RUNNING 1 (G50 pcc, stale). The Codex job died before regenerating GATES.md/results.json from the finished batch:
  **open task**: rerun `preflight/b18/final/summarize.py` (orchestrator-side check first that the Mac's gate dirs hold
  the late PC rows G50-G60 and the tests-* suites).
- mulder: the three prj24 `vitis_hls` runs alive, 14 h 57 m elapsed of 24 h (checked 07:26Z). Export Codex poller is
  dead; a new poller/evaluation needs a launch decision (master prompt: ask Kai before relaunching a stopped job).
- Kai (chat, ~07:00Z): asked for an HTML walkthrough of the campaign (steps done, left, time, why slow). Three Sonnet
  dossier workers wrote `/Users/kaiyamaguchi/.claude/jobs/f4eee4ae/tmp/dossier-{timeline,study,prior}.md`.
  Correction to this session's earlier chat: the campaign is at **N=8**; Chang's comparator is 66.3 % (paper Table 1
  row 1, STUDY §8.10); 77.9 % is his MHA-64 at 350k EBOPs (discovery-350k PROPOSAL.md).

## 2026-10-19 23:40 PDT — session 783ef8 ends at Kai's request; handoff written; no campaign change

- Kai (chat, ~06:33Z): "okay archeive this chat so that it cant edit anythign, and imma start a new chat with the twoer
  loaded". Kai forked this session at ~06:30Z (`bitlinear orchestrator campaign ⑂`, c6f0b226, idle). The question on
  sessions f18b37 / e22c5f stays unanswered; this session never left the "change nothing" gate.
- Done this session: `notes/LEDGER.md` created; `notes/handoff-2026-10-20.md` written (delta over handoff-2026-10-19);
  detached log copiers Mac PIDs 28501 (waits on Codex 69676) and 28502 (waits on Codex 11519), outputs
  `preflight/b18/codex/watch-b18gates.out`, `preflight/export/codex/watch-export.out`; snapshots of both Codex logs
  under the same dirs (`*.snapshot-20261020T0637Z.log`). The two in-session watchers were stopped (the copiers
  replace them, so this session cannot be woken later). Nothing else edited, launched, stopped or committed.
- Status: step 1 (B18 pass finishing) still open; next session starts at handoff-2026-10-20 §6 step 0.

## 2026-10-19 23:20 PDT — session 783ef8 start: running-now check (worker status-check-b18, Sonnet, read-only)

- Handoff `notes/handoff-2026-10-19.md` "Running now" table matches reality at 23:15 PDT. No process touched.
- B18 gate pass Codex (Mac PID 69676) alive, 5 h 09 m; PC batch (kaipc PID 2369146) alive, log 129 lines, no
  `BATCH_DONE`; running G48 pcg (PID boundary 11). G14 clean rerun armed (kaipc PID 2481814 `finish-rerun.sh`);
  `G14/29a459ce7ee6/pcc-attempt-forced-wandb-disabled` not yet created.
- Export Codex (Mac PID 11519) alive, 8 h 38 m; log last entry 22:17 PDT (BF pair and CH-8 still in C-synthesis).
  mulder: three `prj24` csynths at 13 h 45 m elapsed of 86,400 s; FB pair and init-CH-8 `vivado rc=0`;
  the `csynth rc=124` rows in status.sh are the earlier 8 h attempts. mulder disk 70 G free.
- GATES.md `preflight/b18/final/GATES.md` Updated 2026-10-20T05:31:19Z: FAIL 6, NOT RUN 50, PASS 73, RUNNING 1 (G48 pcg).
- NRP: kubectl works; no cbl/chang/bitlinear pod or Job; `kai-` ConfigMaps 0.
- Codex processes on the Mac: exactly the two listed. Logs not copied (processes alive).
- Other sessions (ListAgents): `bitlinear model architecture notes` [f18b37] bg, shell state; `chang-bitlinear-20261016
  b18 brief audit` [e22c5f] bg, idle; `bnjettag-bb` [550d41] interactive. Asked Kai (this session's first reply) to end
  them or confirm they make no further changes; no commit or edit by this session until he answers.
- Current step: handoff §6 step 1 (B18 pass finishing) → step 2 (triage of the FAILs). Status: open.
