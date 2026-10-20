# plan.md — STUDY v0.6 → v0.7 (experiment-designer, 2026-10-17)

Owner notebook (03-phases.md, owner protocol step 1). Not a deliverable; STUDY.md is. The v0.5 → v0.6 notebook is kept
in `review/snapshots/v0.6/plan.md` (sha256 aed8a110…3564).

```
Goal:     STUDY v0.6 -> v0.7: constructive-v0.5 K1-K9 as text, P1-P4 as new Kai items, item-5 interruption line, Addendum B
Plan:     snapshot v0.6; this plan; STUDY outside the frozen block and outside Addenda A and C; protocol outputs/metrics;
          in-memory protocol check; hashes; advisor; hand-off
Touches:  campaign dir only: STUDY.md, plan.md, protocol-draft.json, review/snapshots/v0.6/ (no code, cluster, network, mulder)
Cost:     this agent (Opus); no subagents; about one hour
Stops:    when every item is applied or listed with a reason; early if an item needs a rule, run-count, [D] or §10 change
```

## Inputs read
- STUDY v0.6 (d20113dc…40aa), protocol-draft v0.6 (02cda08e…fe7d), the v0.6 plan.
- review/constructive-v0.5.md, whole (239 lines; its line numbers refer to the v0.5 snapshot, re-anchored here).
- Sources the items cite, at those lines only: campaigns/2026-10-05-pilot-program/STUDY.md:118-127 (H1-H5 labels);
  docs/conventions/jet-tagging-metrics.md:33-38 (working point); docs/design/2026-10-11-russell-questions-recommendations.md:358-361;
  .claude/memory/experiment-log.md:1-7 (already cited by the STUDY).

## Blinding (this author, this pass)
- **Partially results-aware on the da4ml direction.** constructive-v0.5.md:20, read in full as briefed, quotes a numeric
  fragment of the da4ml outcome summary that its author displayed by a slip. This author has therefore seen that
  fragment. Consequence, as the constructive reviewer applied to itself: v0.7 makes no change to the headline-rule
  block, Addenda A and C, the H1/H2 choice, the H-to-LUT calibration, C4, deciding-point synthesis or any LUT or firmware
  text. The one item-5 addition (the interruption line) lists options and keeps the text as it stands as the default; it
  makes no new recommendation. The fragment is not reproduced in any file.
- decisions.md: not opened in this pass. The v0.6 pass read :100-145 (first three characters) and :113-121 only.
- Not opened: preflight/da4ml-test/results/, reports/, logs/, perpos-oneblock/, README.md; preflight/env/; both
  evidence-hash files; `PREFLIGHT-PLAN.md`, which appeared in the campaign directory at 17:08 (metadata only: 44,327 B).

## Steps
1. Snapshot v0.6 to review/snapshots/v0.6/ (done 17:11; STUDY, plan, protocol-draft, constructive-v0.5; SHA256SUMS OK;
   read-only).
2. This plan.
3. STUDY v0.7: K1 §8.6, §6, item 11; K2 §8.6 labels; K3 §8.6 factorial; K4 §13 further gates, §8.6 F000 − CH-8; K5 §8.8
   reading table and event lines, §15 pointers; K6 §8 metrics, §8.6 held-out, §18 row; K7 items 8, 9, 11; K8 §8.8; K9
   §8.6, §8.7; §13 selection row; P1-P4 as §19.2 items 13-16; item-5 interruption line; Addendum B; change log; front
   matter; A.1 addendum (v0.7).
4. protocol-draft.json: version strings; outputs (S2 additions, determinism record, diagnosis records), the working-point
   metric in both stages, confirmation outputs for K9.
5. In-memory protocol check and B/C variants; hashes; §10 untouched (diff); block hash.

## Derived numbers (python3 -I, scipy; arithmetic, not results)
- χ² bands of σ/σ̂: df 2 [0.521, 6.285] (ratio 12.07); df 14 [0.732, 1.577]; df 16 [0.745, 1.522] (ratio 2.04).
- t(0.975): df 14 2.145; df 2 4.303. Block half-width 2.145·√(1/6) = 0.876 σ; three-seed paired 4.303·√(2/3) = 3.513 σ.
- Per run 6.416667 GPU-h at r=3. 51 runs 327.25 (377.25 with reserve), r ≤ 3.2086 (3.529 all-inclusive). 39 runs
  250.25 (300.25), r ≤ 4.1958 (4.615). 36 runs 231.00 (281.00), r ≤ 4.5455 (5.000). 54 runs 346.50 (396.50).
  12 and 15 runs: 77.00 and 96.25. At r=4: 51 runs 436.33 (486.33 with reserve); 39 runs 333.67. 3 runs 19.25.
  Unchosen form share of the required screen 115.5/327.25 = 35.3 %.

## Choices (with reasons)
- K8 and K9 are text-level (printed quantities beside existing labels; no rule) and are applied, not listed in §19.3.
- K2's orientation figures are not written into the STUDY (the reviewer's own condition); only the S2 output is.
- P1-P4: one §19.2 line each, numbered 13-16, none adopted. Recommended defaults: P1 adopt if decided before S1
  (rate risk and Q11), MEDIUM; P2 not in this campaign; P3 adopt with m_s = 2 points as a placeholder, LOW; P4 adopt with
  m = 1 point, descriptive, MEDIUM.
- Item-5 interruption line: options (i) the line fails, as the text stands (default until Kai decides); (ii) a void
  trigger like a failed assertion. No recommendation from this author (results-aware).

## Protocol check (v0.7, in memory, nothing written)
- Same method as v0.6 (reproduces the v0.5 digests on the v0.5 snapshot). Field diff v0.6 → v0.7: purpose, selection.rule,
  metrics (working point added), interval_method, outputs (two entries added); confirmation_stage status, purpose,
  selection.rule, metrics (working point), outputs. Arms unchanged (66 and 20); ASCII; indent=1 formatting preserved.
- Screen 68e167ffd573f5385516087a8a157499ebedb16941db30de7a5475f672a5a247; confirmation (A)
  947c247f56c2c5ccd4a1cb447e809d927732eaad3dd02b27db22cfc0a5b4bbee; option B 64f06c5aec3f5d30f45bd7884c322ba312f9a625fb61c6cb4e64421edbeb5b69;
  option C 071898ec05b5f2f096dd264ba8d6bb91a08652c6db32287b8b7afa3bd9e98b03. One block each (dataset.identity_sha256).

## Final (2026-10-17, 17:20 PDT)
- Headline block d9f2e386…8dbb; marker lines once each; the text from the block's begin marker to Addendum B, and
  Addendum C, byte-identical to the v0.6 snapshot (`diff` empty); §10 byte-identical to the v0.6 snapshot.
- Nine frozen scripts and env.sh equal to the change-log hashes (17:19).
- Tables: 19 (the K5 reading table is new), every row with its header's cell count.
- STUDY.md v0.7 sha256 c0ba6b685f50c288e4eef17a864307e72e6b0056e4970be8bfe023d7533404a9 (1,963 lines; after the 17:19 change-log time and the K4 §11 note);
  protocol-draft.json c136a26b1e16445f4765ce81ff4d04bc083bc67d281d78d300db3e4537a360b9.

## v0.7 → v0.8 (fixer, 2026-10-17)

```
Goal:     STUDY v0.7 -> v0.8: constructive-v0.7-delta B1-B4 as specified, plus "For later" items 1, 3, 6 (text only)
Plan:     hashes; this plan; STUDY edits outside the block, Addenda A and C and §10; protocol outputs[11] + version strings; digests; invariants
Touches:  campaign dir only: STUDY.md, protocol-draft.json, plan.md (no code, cluster, network, mulder; no snapshot)
Cost:     this agent (Opus); no subagents; about one hour
Stops:    when B1-B4 are applied or listed with a reason; early if an invariant fails or a blinded outcome is seen
```

### Inputs
- STUDY.md recomputed c0ba6b68…3404a9, equal to review/snapshots/v0.7/STUDY.md; protocol-draft.json c136a26b…60b9 (equal to the brief).
- Read: review/constructive-v0.7-delta.md (whole), review/arbiter-v0.5.md (whole), STUDY v0.7 (whole), RULES.md,
  docs/methodology/03-phases.md:1-80, 06-review.md:1-110; tools/jev/methods.py:1-170 and common.py (digest) for the method;
  literature/bitnet-1bit-ternary/2402.17764_era_of_1bit_llms_1p58.md:1-20 (item 6: :16 is the heading, :17 the absmean ternary line).

### Blinding
- No blinded file opened, listed or searched (decisions.md, review-reports.md, constructive-v0.5.md and its v0.6 copy,
  critical-v0.3.md, PREFLIGHT-PLAN.md, preflight/, v0.5 and v0.6 plan.md). No recursive search. No hardware-test outcome seen;
  Addendum B of the STUDY was read as part of the edited file and reproduces no outcome. Jev down; not used.
- The nine frozen script hashes and env.sh (change log :56-70) cannot be recomputed here (preflight/ is blinded); not claimed.

### Protocol method (reproduced on the v0.7 draft before any edit)
- digest = tools/jev/common.py digest (sha256 of sorted-key compact JSON); check = tools/jev/methods.py check_protocol;
  python3 -I -B. Screen = digest of the whole draft; option A = digest of confirmation_stage; options B and C = confirmation_stage
  with arms rebuilt per group from the seed-4 template (name <group>-s<seed>, seed 4-11; B all four groups, CH-8-C first;
  C CH-8-C and BEST-C only), no other field changed. Script: /tmp/fixer-v08/variants.py (outside the lab tree).
- v0.7 reproduced: 68e167ff…a247, 947c247f…bbee, 64f06c5a…5b69, 071898ec…8b03; one block each (dataset.identity_sha256).

### Orchestrator rulings applied
- B1 steps 1-4 (F000 run added; :974, :987; stack of :1311/:1323). Stack: §11 places resume gates on the home PC; the STUDY
  names no device for the reference-fidelity row and does not imply the production GPU product for either, so the stack is
  listed as a PREFLIGHT owner decision in §13 (fallback of the ruling).
- B2 sentence at :976 and gate at :1325; B3 preamble sentence; B4 steps 1-5 in item 5 and the §13 da4ml row; no recommendation.
- Optional items 1, 3, 6 applied; items 2, 4, 5, 7, 8 not applied (item 8's 1.10 arithmetic covered by B1's recomputation).

### Derived numbers (python3 -I; arithmetic, not results)
- Record cost: 60 run-epochs x 3 s = 180 s = 0.05 GPU-h at the planning rate; x 1.10 = 198 s = 0.055 GPU-h.
  v0.7: 40 x 3 s = 0.0333 GPU-h (0.0367 with 1.10). Added by the F000 run: 20 x 3 s = 0.0167 GPU-h (0.0183 with 1.10).

### Edits (v0.8 lines)
- B1: §13 record :1348 (F000 run, cost, "three runs", stack owner decision); §8.6 :976, :989; §13 rows :1313, :1325; protocol outputs[11].
- B2: §8.6 :978 (sentence as written, "σ̂" -> "σ̂_pool" after item 1); §13 selection row :1327 (gate).
- B3: §19.2 preamble :1474 (sentence as written; "act only if decided before S1" kept, not contradicted).
- B4: §19.2 item 5 :1479 ("Open for Kai (v0.7)" moved to the end, mechanics steps 1-4 after it); §13 da4ml row :1319.
- Items 1 (:971, :978, :1001), 3 (:1076-1081), 6 (:1498). Records :6, :35, :55, §19.4 :1602-1605, A.1 v0.8 addendum :1744-1760.
- Not touched: the script-hash line :58-72 (preflight/ blinded); the v0.7 A.1 addendum's "About 0.03 GPU-h" (kept, superseded).
- §19.4 :1602: the v0.7 parenthetical on local/jev-audit/ scoped to v0.7 ("not inspected in v0.8"), after the advisor check.

### Invariant checks (commands, 2026-10-17)
- Block: `sed -n '/^<!-- headline-rule v0.3 begin -->$/,/^<!-- headline-rule v0.3 end -->$/p' STUDY.md | shasum -a 256`
  -> d9f2e386…8dbb; `grep -c` of each marker line -> 1 and 1.
- Sections, extracted by pattern from both files with `ex() { sed -n "/$2/,/$3/{/$3/!p;}" "$1"; }` and `diff`:
  Addendum A `'^\*\*Addendum A:'` to `'^\*\*Addendum B:'` (46 lines, c8aac5cb…); Addendum C `'^\*\*Addendum C:'` to
  `'^### 3\.5 '` (8 lines, e3ba71d0…); §10 `'^## 10\. Budget'` to `'^## 11\. Operations'` (55 lines, 6d6f08a8…). All diffs empty.
- Tables (python3 -I, unescaped-pipe count per row against its header): 19 tables, 19 separators, no mismatched row.
- `[D]`: no removed line carries one; the two added occurrences are the words "`[D]` label" in the change log and addendum.
- Protocol: raw diff 6 lines (3, 1081, 1130, 1135, 1136, 1473); arms 66 and 20 and stop_rules equal to v0.7; ASCII, indent=1.

### Protocol check (v0.8, in memory, nothing written)
- Screen 3b25814bdba31978566021b6610a8d668521c598465531111ce721636525e46d; option A c16e01003758294cfa3fe981a6a3b4e5f0cf1751ec83d76ad1db1ef7855e0f49;
  option B 70a702a9c86db7b446109dd28be9856680662dbdc31e0a876fc407fef422cb5c; option C cdb34eaa09dca71edd76c76ec0ef70f5f9c1270b61b6c4b3c63256d47247b7a2.
  One block each (dataset.identity_sha256). Field diff v0.7 -> v0.8: purpose, selection.rule, outputs, confirmation_stage
  purpose, selection.rule, status.

### Owner decision listed (for the orchestrator / PREFLIGHT owner)
- DECISION: the product and stack of the bitwise reference-fidelity gate (:1313) and the resume gate (:1325), and whether
  either also runs on the production GPU product. ALTERNATIVES: home PC only (§11 for resume; then the record's "not
  identical" branch is not a gate failure); also on the production GPU product (then it is). CONFIDENCE: LOW.
  FLAG FOR HUMAN: YES (B1 step 4 could not name it from the STUDY).

### Final
- STUDY.md v0.8 sha256 290022fa41a726b8ae66ff302f7910a69c4c148ec27b15c2448cab83fb898ca8 (1,983 lines);
  protocol-draft.json 8f48e716333a4ffb5ca9ae36012480bf6a3484aadc55545ecfdb00fd08d302c8. Nothing snapshotted.

## v0.8 → v0.9 (fixer, 2026-10-18)

```
Goal:     STUDY v0.8 -> v0.9, pre-data amendment: §6 power table at printed precision (G54 finding); labels restored (B3b-22)
Plan:     hashes; recompute closed form and B3b-1 layout in scratch; edit §6, §8.7, records; append here; invariants
Touches:  STUDY.md and this file only (no code, cluster, protocol-draft.json, snapshot)
Cost:     this agent (Opus); no subagents; under one hour
Stops:    when both items are applied; early if an invariant fails or protocol-draft.json would need a change
```

### Inputs and blinding
- STUDY.md recomputed 290022fa…898ca8 before any edit, equal to review/snapshots/v0.8/STUDY.md; protocol-draft.json 8f48e716…08d302c8.
- Read: RULES.md; docs/methodology/03-phases.md, 06-review.md; STUDY v0.8 §6, §8.2, §8.6, §8.7, §13, §18, change log, A.1 v0.8;
  preflight/gates/G53/1784511c0f34/PREFLIGHT-rows.md:92-150 only (G54 section and owner decisions B3b-1 to B3b-4);
  review/snapshots/v0.2/STUDY.md:440-480 (orchestrator's second item); protocol-draft.json, searched for version strings and labels.
- Not opened, listed or searched: decisions.md, review-reports.md, constructive-v0.5.md and its v0.6 copy, critical-v0.3.md,
  PREFLIGHT-PLAN.md, PREFLIGHT.md, the rest of preflight/, v0.5 and v0.6 plan.md. No recursive search. No hardware-test outcome seen. Jev down.

### Derived numbers (scratch, outside the lab tree; arithmetic, not results)
- Closed form, rho = 0 (mhab1012-cpu python -I, scipy nct.sf + brentq): A 2.0500 2.3530 2.5131 2.6204 2.7004 2.7639;
  B 1.3392 1.5233 1.6198 1.6841 1.7320 1.7698; C 1.1560 1.3286 1.4189 1.4790 1.5237 1.5590 (equal to the orchestrator's).
- B3b-1 reimplemented from its text: /tmp/fixer-v09/jointsim.py. Reproduces all seven G54 computed roots and their MC standard
  errors (A,0,3 2.5151; A,0,4 2.6229; A,0.9,6 2.3751; B,0,2 1.5238; B,0,4 1.6846; B,0.9,3 1.4653; C,0,5 1.5250).
  rho = 0.9: A 2.0473 2.1906 2.2640 2.3124 2.3469 2.3751; B 1.3381 1.4219 1.4653 1.4935 1.5135 1.5292;
  C 1.1559 1.2323 1.2719 1.2969 1.3162 1.3306 (|D| = 1 printed from the noncentral t). SE at |D| >= 2: 0.0013-0.0023.
  rho = 0.5, |D| = 6: 2.6661, 1.7033, 1.4938 (STUDY 2.67, 1.70, 1.49).
- /tmp/fixer-v09/other6.py: p(threshold) 0.566/0.536/0.531; joint at six allowances, rho 0/0.5/0.9: A 0.033 0.133 0.363,
  B 0.024 0.135 0.350, C 0.022 0.139 0.351; p(0.5 sigma_d) 0.077/0.143/0.231. All equal to STUDY :763 and :771.

### Edits (v0.9 lines)
- Item 1: §6 :749 (noncentral-t parameters, closed form, B3b-1 layout and root rule with crit_j defined); table :753-758
  (rho = 0 rows to the closed form, labelled "exact (closed form)"; rho = 0.9 rows labelled "Monte Carlo", A |D|=6 2.37 -> 2.38,
  B |D|=3 1.46 -> 1.47, the values the stated layout gives); note :760 (200,000 per cell, SE about 0.0013-0.0023, one unit).
- Gate (item 3 of the brief): §13 and §18 state no gate on reproducing the §6 table (the §13 selection row :1335 gates the S2
  plug-in power and zero differences); none added or restated. G54's restatement is the PREFLIGHT owner's.
- Item 2: §8.7 :1039-1042, definitions restored from v0.2:463-465 (PREFLIGHT B3b-22; flagged for Kai).
- Records: front matter :5-6; change log :35; frozen-rule heading :57; A.1 v0.9 addendum :1770-1783.
- Unchanged and checked: In brief :25; §6 options :784-786; §15 :1392; §17 C2 row :1433; §19.2 item 11 :1494-1496; §19.3 :1548.

### Decisions flagged
- DECISION: the two rho = 0.9 cells print what the stated B3b-1 layout gives (2.38, 1.47), per the brief's condition ("unless the
  stated layout is adopted and reproduces different ones"). ALTERNATIVES: keep 2.37 and 1.46 (then the stated layout does not
  reproduce them at printed precision). CONFIDENCE: HIGH. FLAG FOR HUMAN: NO (within one printed unit and one MC SE of the boundary).
- DECISION: labels read on the 7- or 6-contrast simultaneous intervals over g in D_conf, COST INFEASIBLE as a deficit (§8.2),
  "whole-front criterion" = C2's ABOVE / BELOW ON GRID; comparator CH-8 (v0.2's BE is a control arm, never a comparator).
  ALTERNATIVES: domain D_C2 (then LOCAL DEFICIT could not describe a deficit below g* under option (b)); level-L intervals
  (contradicts v0.8 :1038). CONFIDENCE: MEDIUM. FLAG FOR HUMAN: YES (B3b-22, as instructed).

### Invariant checks (commands, 2026-10-18)
- Block: `sed -n '/^<!-- headline-rule v0.3 begin -->$/,/^<!-- headline-rule v0.3 end -->$/p' STUDY.md | shasum -a 256`
  -> d9f2e386…8dbb; `grep -c` of each marker line -> 1 and 1.
- `ex() { sed -n "/$2/,/$3/{/$3/!p;}" "$1"; }` on v0.8 snapshot and v0.9, then `diff`: Addendum A (46 lines, c8aac5cb),
  Addendum C (8 lines, e3ba71d0), §10 (55 lines, 6d6f08a8); all diffs empty, hashes equal to v0.8's.
- Tables (python3 -I, unescaped-pipe count per row against its header): 19 tables, no mismatched row.
- `[D]`: no removed line carries one; the two added occurrences are the words "`[D]` label" in the change log and addendum.
- Full diff against v0.8: removed lines are :5, :6, :55, :747 and the six table rows only; no threshold line touched.
- protocol-draft.json unchanged (8f48e716…08d302c8); its five "STUDY v0.8" version strings are left as they are.

### Final
- STUDY.md v0.9 sha256 61e68527607b76150db4ab9c1e198c20564b8803646de70dc292d39966f89df4 (2,006 lines). Nothing snapshotted.

## v0.9 → v0.10 (fixer, 2026-10-18)

```
Goal:     STUDY v0.9 -> v0.10: review/critical-v0.9-amendment.md §6 B1 as worded; PREFLIGHT B3b-30 wording in §8.6
Plan:     hashes; read the review's §6; §19.3 entry; §8.6 wording; front matter, change log, A.1 v0.10; invariants
Touches:  STUDY.md and this file only
Cost:     this agent (Opus); no subagents; about 15 minutes
Stops:    when both are applied; early if an invariant fails
```

- Inputs: STUDY.md 61e68527…6f89df4 before any edit, equal to review/snapshots/v0.9/STUDY.md; protocol-draft.json 8f48e716…08d302c8.
  Read review/critical-v0.9-amendment.md:90-116 (§6 findings) only. No other file opened beyond the first brief's; no outcome seen.
- B1: §19.3 entry :1548-1551, the review's words, with its v0.9 line references moved to v0.10 (§8.7 :1041-1044, :1040).
  The v0.9 change log and addendum are kept as written (as v0.8 kept the v0.7 addendum); the mention is in the v0.10 lines.
- B3b-30: §8.6 :972, "the same simulation" -> "at ρ = 0 the closed form, the product of the single-allowance noncentral-t powers,
  and at ρ > 0 the §6 simulation". The parenthetical says what "closed form" means at unequal per-g gains; it also answers C5.
  C1-C4 and C6-C9 not applied (none needed for B1).
- Records: front matter :6; change log :35; frozen-rule heading :59; A.1 v0.10 addendum :1791-1793.
- Invariants: block d9f2e386…8dbb, markers 1 and 1; Addenda A (46), C (8), §10 (55) diff empty against v0.9 and v0.8;
  19 tables, no mismatched row; no removed line carries [D]; diff against v0.9 at :6, :34-35, :59, :972, the §19.3 entry, the addendum.
- Final: STUDY.md v0.10 sha256 7180c24f6b55807e52e45cf0705bb14504df6f231012dc1d68884d661a5f0c09 (2,016 lines); protocol-draft.json unchanged.
