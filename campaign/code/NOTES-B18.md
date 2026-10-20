# PREFLIGHT B18: final decision inputs and BRIEF candidate

Prepared 2026-10-19 on branch `b18`, base/unchanged HEAD
`107bd9a172dd23dea5f670a328685cc06ed503c3`, in
`/Users/kaiyamaguchi/lab/worktrees/cbl1016-b18`. The initial worktree was clean.
No commit, staging, network, kubectl, SSH, credential access, approval or launch.
Deliverables are this worktree and campaign `preflight/b18/`; tests use temporary
files. No hand edits to configs/, chang/ or vendor/; no lab-root tool edits.
No prohibited review or preflight/export/ or preflight/da4ml-test/ file was opened.
Only the permitted top decision-log entries through later 4 were read.

## Changed

- `gen/inputs.json`: OD-13, OD-15, OD-12 are explicit `true`, with Kai and
  2026-10-19T17:13:14Z provenance. Claim domain is
  `{base: "b", band_rule: "default", c3r: false, c3r_level: null}`. Headline and
  S4 option remain explicitly null; U1 roots/platforms, benchmark length and S2
  remain unset. No runtime identity or experimental result was invented.
- `gen/inputs.py` validates required confirmation keys as true/null (false or
  malformed values refuse). Generated `launch_confirmations` carry them into
  every config; missing confirmations block launch even if unset_inputs is removed.
  Claim-domain validation also runs on launch copies, rejecting a C3r level when
  C3r is false. The existing generic selector still supports other separately
  specified designs; this campaign's committed choice is b/default without C3r.
- `gen/configs.py` and `cbl/runner.py`: CH-8 and F000 ramp training can have a null
  headline, as v0.4 requires. Binary arms, including F100 and engineering runs,
  remain refused. PID still requires a headline sensor. The exemption changes no
  native training penalty and selects no comparison count; front storage already
  retains H1/H2/NR and supports null headline for ramp runs.
- Generator provenance now names STUDY v0.13, SHA-256
  `3e6d2ab16c970de0a4dd752e445bda5e1a7a267e5ee70e227bca04514f9c3d22`.
  The base protocol reference remains the recorded v0.8 reference; screen arms,
  seeds, schedules, four derived protocol files and run counts are unchanged.
- Regenerated all 172 files using the generator command (167 run configs, four
  protocols, one manifest). No direct config edits. All current generated configs
  still refuse launch because U1/handoff inputs are missing, among other holds.
- G57's schema/expected-decision assertions now reflect confirmed claim domain,
  the three confirmations and the narrow headline exemption. The historical
  assertion that claim_domain must be null would contradict Kai's answer.
  Clean-tree checks were not removed, bypassed or weakened.
- Generator, runner and NRP tests cover the new admission behavior. Updated
  interface sections 5, 13 and 14; section 11's selection contract is unchanged.
- External `preflight/b18/BRIEF.candidate.md` starts from B17's draft, retains its
  operations/schema, and updates decisions, stops, restart rules, counts and
  authority to v0.13. It lists 13 currently eligible config identities (six
  CH-8/F000 screen, seven engineering), plus record/smoke initial draws. Eligibility
  is subject to all remaining gates and approval, not a claim that launch is ready.
  Binary/PID holds and conditional one-pass approval are explicit in authority.
  `code_sha`, `bundle_sha256` and `config_manifest_sha256` remain exactly
  `TO FILL AT B18 FINAL SHA`. Current frozen-file hashes were refreshed from this
  worktree, including gen/inputs.json and all current cbl/nrp Python modules.
  `prepare_candidate.py` in the evidence directory reproduces this derivation.

## Sources and decisions

Sources: the specified sections of STUDY v0.13 (10, 11, 14, 19.2 and headline rule
v0.4); permitted decisions.md entries later 7 through later 4; build-todo and
flags-for-kai; interfaces 5/11/14; NOTES-B17/NOTES-B18P; gen/, tools/, nrp/; B17
BRIEF draft; recorded G57 command. Older open-decision statements in flags and
STUDY are superseded by the 2026-10-19 answers. No export outcome was consulted.

### B18-1: launch decisions

DECISION: Treat Kai's individually explained answers recorded in later 6 as
confirmation of each OD and b/default without C3r. Preserve full dated provenance.
The confirmations are required generator inputs and launch-copy fields, rather
than comments that the launch check cannot enforce.

ALTERNATIVES: Leave them unresolved or keep only free-text approval; either would
misrepresent the provided answers or weaken the explicit fail-closed contract.

CONFIDENCE: High, from the specified decision record and offline refusal tests.

FLAG FOR HUMAN: No new design decision requested. Formal harness approval still
belongs to Kai after the final review; the recorded "i approve" intent is not a
matching APPROVAL.json. No approval file was created here.

### B18-2: count-independent training and the binary hold

DECISION: Keep headline null. Exempt only CH-8/F000 ramp training from requiring
it in both generator and runner admission; keep binary and PID requirements.
This small runner change is necessary: the previous implementation would have
blocked the screen identities the user and v0.4 explicitly permit.

ALTERNATIVES: Pick H1/H2 speculatively, exempt every engineering arm, or write a
BRIEF that the runner cannot carry out. None meets the requested v0.4 rule.

CONFIDENCE: High for offline admission, not real-stack execution. Generator tests
exercise all generated arm classes; NRP resolve covers smoke/bench/det/pack/draw
and specifically refuses F100 with a null headline.

FLAG FOR HUMAN: Both F111 forms need recorded outcomes before setting binary
headline. Both pass -> H2; both fail -> H1; split -> H1 for every binary arm in
both forms plus the passing form's additional H2 runs only under B18-3. The
current single-headline generator does NOT implement the split-form dual-count
extension. That needs the later dated amendment and distinct run identities.
No binary count is fixed by this B18 preparation; no training penalty changes mid-run.

### B18-3: budgets and one-pass admission

DECISION: Current authority remains 350 training + 50 reserve, cap 400 GPU-h.
For exactly one passing form, the candidate records 24 added H2 runs, 75 required
runs, 481.25 planning training GPU-h + the same 50 reserve = 531.25. The enlarged
budget and all added runs remain HOLD until Kai answers their §8 selection/reading
role AND confirms the exact 531.25 cap. Paired seeds 1–3 and §10 tier admission,
drop order and control slot apply unless Kai says otherwise.

ALTERNATIVES: Activate "about 530" now, add another reserve, or infer §8 behavior.
These contradict v0.13's explicit one-pass hold. The base reserve already applies;
it is not itself deferred pending the conditional answer.

CONFIDENCE: High for the specified numbers/hold; all costs are projections, not
measured rates. Baseline required 51 runs project to 327.25 + 50 = 377.25 GPU-h.

FLAG FOR HUMAN: Before the one-pass extension, update/test M21's hard-coded
350/50/400 budget and 51-run production-certificate contract, the generator's
run/count identities and external ADMISSION policy, and add the dated protocol
amendment. The candidate is deliberately not a 75-run launcher implementation.
S4 option remains null until after S2/before held-out reading and needs separate
funding; no S4 claims are supported by the three-seed screen.

### B18-4: infrastructure, retries and identity

DECISION: Preserve B17's unbounded infrastructure-only relaunches inside the
buckets, zero ConfigMaps before every wave/resubmission, run_handoff.py bundle
mode, kai-data /data/cbl1016/ for all durable NRP state, transient emptyDir only.
Stop on code/integrity failures for a new commit, checks and handoff. B18P's
pre-train archive/restart exception is retained; published corrupt checkpoints
never trigger a fresh start. Keep original GPU product and checkpoint interval.

ALTERNATIVES: A finite retry cap, treating unknown failure as infrastructure,
manual manifest edits, changing PVCs or restarting corrupt runs at epoch zero.
None is authorized by the supplied rules.

CONFIDENCE: High for policy transcription and offline tests. No live NRP claim.

FLAG FOR HUMAN: Notification retains the implemented B17 meaning: threshold 6,
notification when the number of successful infrastructure resubmissions of a
pack in a rolling 24 h is greater than 6 (first notice on the seventh), then
continue subject to budgets/G80. This is "after 6", not a stop at six. Active
kubectl permission does not bypass the lint hook or final authority approval.
The B17 init-container resource-ratio flag still needs upstream/G77 resolution.

### B18-5: candidate validation and remaining evidence

DECISION: Keep the candidate outside the campaign root. The unchanged harness
CLI accepts a directory and always reads its BRIEF.md, not an arbitrary filename.
Validate a byte-identical copy at preflight/b18/validation/BRIEF.md. Use real
worktree hashes for frozen_files but placeholders for the three final identities.
The external ADMISSION hash currently binds the unchanged template with null
production certification; it must be refreshed when the reviewed policy is ready.

ALTERNATIVES: Install an unfinished campaign BRIEF, invent a final commit/bundle,
or report schema validity as scientific certification or approval.

CONFIDENCE: High for validator execution and byte identity. Schema validation
checks authority/operations types and ranges, not scientific correctness, final
bundle completeness or frozen-file deployment. The added scientific_settings
holds document authority; existing generator/runner checks enforce the present
null-headline hold. They are not a new general policy interpreter.

FLAG FOR HUMAN: B10-F1's resolution is not established here; follow the final
orchestrator review. B18P leaves native_share_removed undefined pending a baseline,
denominator and zero-denominator rule. Its 19 PC cases and applicable GPU/resume
checks remain due. No export/PC/cluster gate was run or inferred from this Mac
suite. Product/K, initial roots/platform and actual U1 hashes, benchmark length,
all-inclusive rates, full-horizon host memory and rolling three-hour utilization
still belong to B19. Do not describe current generated configs as launch-ready.

## Test and gate outputs

All command stdout/stderr is under campaign `preflight/b18/`. Commands ran from
this worktree with PYTHONDONTWRITEBYTECODE=1 (the generator/G57 also use -I -B).
The Python path is `/Users/kaiyamaguchi/lab/venvs/mhab1012-cpu/bin/python` unless
explicitly written as python3. `commands.json` records cwd, argv, environment and
exit status; `checks.json` records hashes, scope and final artifact checks.

1. `python -I -B gen/cli.py generate`: exit 0, wrote 172 files (`generate.txt`).
   Regenerate-and-diff in G57: 172 files, zero missing/extra/different.
2. `python -m pytest tests/test_gen.py -q -p no:cacheprovider`: exit 0,
   **37 passed** (`test-gen.txt`). G57 repeats this successfully.
3. Requested full command:

   ```sh
   /Users/kaiyamaguchi/lab/venvs/mhab1012-cpu/bin/python -m pytest tests -q -p no:cacheprovider -m "not pc_gpu"
   ```

   Exit 0: **602 passed, 1 skipped, 19 deselected, 1 warning** (`tests.txt`).
   The skip is Linux-only inotify. The existing warning is the intentional
   zero-scale reader-refusal fixture in cbl/data.py:367. The first development
   run had two stale test expectations for CH-8/F000's old headline requirement;
   preserved as tests-development.txt (600 passed, two failures). They were
   corrected to assert the new exemption while retaining the F100 refusal.
4. G57: recorded command retargeted to this worktree and actual HEAD:

   ```sh
   /Users/kaiyamaguchi/lab/venvs/mhab1012-cpu/bin/python -I -B gates/g57_configs.py --expect-sha12 107bd9a172dd --reference /Users/kaiyamaguchi/lab/bnjettag/campaigns/2026-10-16-chang-bitlinear/review/snapshots/v0.8/protocol-draft.json
   ```

   Exit 1, **exactly three failures**, all clean-tree checks: git status empty at
   start; git diff HEAD -- configs empty; git status empty at end. Every other
   check passed, including imports, generation/runner admission, protocol/reference,
   configuration regeneration and the 37 generator tests (`G57.txt`). No working
   tree SHA was invented: --expect-sha12 is the existing base commit, which passed
   the HEAD check; this is NOT certification of the dirty working tree.
5. `PYTHONDONTWRITEBYTECODE=1 python3 /Users/kaiyamaguchi/lab/bnjettag/tools/harness.py validate /Users/kaiyamaguchi/lab/bnjettag/campaigns/2026-10-16-chang-bitlinear/preflight/b18/validation`:
   exit 0, **VALID**, explicitly **no matching approval; submissions refused**
   (`validate.txt`). The checked BRIEF.md is byte-identical to BRIEF.candidate.md.
6. `git diff --check`: exit 0. Study hash verified against the supplied full
   v0.13 hash; HEAD unchanged; no changes in chang/, vendor/ or lab-root tools/.

## Intended commits and post-commit work (orchestrator only)

No commits or index updates were made. Intended integration sequence:

1. `preflight: record Kai launch decisions and preserve v0.4 headline hold [B18]`
   — gen/__init__.py, gen/inputs.json, gen/inputs.py, gen/configs.py,
   cbl/runner.py, gates/g57_configs.py, tests/test_gen.py, tests/test_runner.py,
   tests/b17/test_nrp.py, docs/interfaces.md.
2. `configs: regenerate confirmed B18 decision inputs [B18]`
   — configs/ ONLY. G57 requires the last configs-changing commit to contain
   configs alone; do not squash these two into a mixed config/source commit.
3. `preflight: record B18 candidate decisions and offline gate evidence`
   — NOTES-B18.md. External preflight/b18 artifacts remain campaign deliverables.

On the final committed, clean checkout:

- Run generator `check`, the 37 generator tests, and the requested full offline
  suite; rerun G57 with --expect-sha12 equal to THAT final commit and the same
  reference path. Require all clean-tree checks to pass. Retain the dirty-tree
  evidence here; never relabel it as pass.
- Run B18P's 19 deferred PC cases on the pinned stack and applicable G12/G46/G48,
  reference, RNG, cost/gradient/PID and resume checks on the final integrated code,
  resolving B10-F1 through its designated review. G83 remains real PVC evidence.
- Create and verify a complete final git bundle, fill the three candidate identity
  fields, refresh frozen hashes/external ADMISSION and applicable G76/G78 handoff
  validation. B19 performs the approved draws, smoke, benchmarks, determinism
  record and product/K certification; final cluster G77/G79/G80 remain required.
- Record both v0.4 form outcomes before generating/admitting binary configs.
  Any split extension additionally needs B18-3's explicit decisions and amendment.
  Regenerate and rerun relevant gates whenever generator inputs change.
- Install only the reviewed final BRIEF/ADMISSION at the campaign root; validate
  there and give Kai his harness approve command. No launch is authorized by this
  candidate, and this task neither installs nor requests that approval.
