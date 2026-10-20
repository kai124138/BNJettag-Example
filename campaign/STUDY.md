---
id: chang-bitlinear-20261016
campaign: 2026-10-16-chang-bitlinear
date: 2026-10-16
revised: 2026-10-20
status: STUDY PASS (arbiter v0.5, conditional on P1-P6), C items applied in v0.6, constructive-review text items applied in v0.7, constructive v0.7 delta B1-B4 applied in v0.8 (fixer), pre-data amendment v0.9 (fixer; PREFLIGHT gate G54 finding and owner decision B3b-22), critical v0.9-amendment B1 and PREFLIGHT B3b-30 wording applied in v0.10 (fixer), dated amendment v0.11 recording Kai's 2026-10-18 and 2026-10-19 decisions, critical v0.11-amendment A1-A3 and B1-B6 applied in v0.12 (fixer), its fix-verification B7 applied in v0.13 (fixer), dated amendment v0.14 declaring the Keras loss guard (B10F1-1; Kai, 2026-10-19), critical v0.14-amendment B1-B2 applied in v0.15 (fixer), outcome of the §3.4 v0.4 export check recorded in v0.16 (both forms fail; H1 for every binary arm) [Corrected in v0.17: F111-BF's failure rests on the owner record of 2026-10-20T17:30Z, not on the time limit alone; see the v0.17 entry.], critical v0.16-amendment A1 and B1 applied in v0.17 (F111-BF failure rests on the owner record of 2026-10-20T17:30Z, a results-informed reading of B14-7a, with Kai's confirmation pending; earlier engineering READMEs cited), critical v0.17-amendment B1 and B2 applied in v0.18 (v0.16 history restored with annotations; the BF sequence 17:06Z to about 17:36Z stated)
type: training-batch
question: At N=8, which selectively binarized configuration of Chang's released two-head MHA, in the binary form chosen on validation, has the best validation top-1-versus-H front against Chang's own code rerun on the same data (three-seed screen, held-out read descriptive only); and, on five or eight fresh seeds if Kai funds it, does that validation-selected configuration's held-out front lie above Chang's over a claim domain fixed by a pre-registered rule (on screen validation, before any confirmation seed) (confirmation)?
wandb_project: bnjettag-chang-bitlinear
wandb_group: chang-bitlinear-20261016
code_sha: not built
results: none
---

# Chang's MHA with selective binary weights

## In brief

This campaign trains Chang Sun's released jet-tagging transformer: two-head attention over 8 particles with 3 features each, quantized with HGQ2 learned bit widths and trained with his 7,000-epoch recipe. It also trains versions in which the weights of chosen layer groups are binary. In those groups each weight is +1 or −1 times one scale, per layer or per output channel where a batch-normalization scale is folded in, and the scale itself has a learned bit width, like every other constant in Chang's model. Three layer groups can be binarized: the attention projections, the feed-forward block and the classifier head. That gives eight on/off combinations. Two ways of joining binary weights to the folded normalization are trained, and one is kept on validation data.

Every run traces an accuracy-versus-cost front: top-1 accuracy against H, a bit-product count of the exported constants. H is H1 or H2 under §3.4 v0.4, whose criteria are fixed before any training result. Training runs on NRP; CH-8 and F000 do not wait for the count decision. No binary arm, F100 included, launches under either count until the factored-export check on mulder has a recorded outcome for both binary forms; the binary arms then train and are read under H2 if the check passes for both forms and under H1 if it fails for both; if exactly one form passes, every binary arm trains and is read under H1 and the passing form's binary arms also train under H2, 24 more runs (§3.4; Kai, 2026-10-19). The comparator, CH-8, is Chang's own code rerun on the same data split.

**Screen.** The part that fits the 400 GPU-hour cap is a three-seed screen. It chooses the binary form and the best configuration on validation and reports the held-out gaps without a verdict, because three seeds cannot resolve a plausible gain. It needs 51 runs: 327.25 GPU-hours at the planning rate, 377.25 with the reserve.

**Confirmation.** The verdicts would come from a confirmation on fresh seeds of the selected configuration and of CH-8. It needs Kai's budget approval, and Kai also chooses its size (§19.2 item 11). As adopted it is five seeds, ramp and PID, 20 more runs, about 128 GPU-hours at the planning rate; at that size an 80 % chance of a front verdict over six allowances needs a true gain of about 2.4 to 2.8 times the true seed-to-seed sd of the paired gap (derived, §6). Eight seeds of the ramp runs alone, 16 runs and about 103 GPU-hours, would need about 1.3 to 1.6 times that sd, but would leave the PID comparison descriptive. These figures assume the same gain at every allowance. The front verdict needs a gain at every allowance of its domain, so if the true gain is zero or negative at any one of them, its probability is at most 1.25 % when the front claim shares α with a second claim, or 2.5 % when it stands alone (eight seeds, ramp only), whatever the size (§6).

**Claim domain.** The design expects a deficit at tight budgets (below), so a claim over every allowance CH-8 reaches is one the design expects to fail. Kai therefore also decides, before any launch, over which allowances the front claim is made (§19.2 item 12). Recommended: a band of allowances at and above a threshold g*, where g* is set on screen validation by a rule written now (§8.2), with the claim over all allowances reported beside as the strong form.

**What a win would mean.** Every binary model the campaign can export is also a model Chang's per-weight quantizers could export, with the same constants and therefore the same headline cost counts (H1 and H2) and firmware (§1). A confirmed gain would therefore show that training with the binary constraint finds better accuracy-cost points under Chang's recipe, not that binary weights are a cheaper representation.

**Likely outcomes.** The screen ranks the configurations and measures the seed spread. A deficit at tight budgets is the expectation: the cheapest nonzero binary layer costs exactly what a dense 1-bit HGQ kernel costs, and HGQ can go lower by zeroing single weights, which a binary kernel cannot. A gain, if there is one, can be established only by the confirmation. LUT counts from FPGA synthesis exist only at the deciding checkpoints.

## Change log

**v0.18 (2026-10-20, fixer).** This dated amendment applies B1 and B2 of `review/critical-v0.17-amendment.md` and changes the outcome record only. B2: v0.17 had rewritten three places of the v0.16 record in place without disclosure: the front-matter clause (`both forms fail; H1 for every binary arm`), the v0.16 change-log entry ("Both forms fail", "before the outcome" and "of record" removed, "stated in v0.17 below" added) and the v0.16 bullet of A.1. v0.18 restores each byte-for-byte from `review/snapshots/v0.16/STUDY.md` and appends an annotation in square brackets, so the history reads: v0.16 recorded, v0.17 corrected, v0.18 corrects the record of the correction. B1: v0.17's sentence "the outcome was read once, by the evaluation job" was wrong. The outcome block now gives the sequence from the ledger: 17:06Z noticed on mulder; 17:10Z anticipatory reading "once recorded: BF = FAIL" before any evaluation; 17:14Z evaluation confirmed; 17:18Z outcome entry; 17:30Z owner record; about 17:36Z chat report to Kai. The BF classification was anticipated from process state at 17:10Z and confirmed at 17:14Z; the results-informed label stands. The v0.17 change-log entry carries a bracketed correction. No rule, criterion, arm, run count, seed, budget, §10 figure, `[D]` label or decision label changes; the v0.3 and v0.4 blocks, §10 and `protocol-draft.json` are byte-identical to v0.17. Verification is recorded in `review/amendment-v0.18/NOTES.md`; v0.17 is kept in `review/snapshots/v0.17/`.

**v0.17 (2026-10-20, fixer).** This dated amendment applies A1 and B1 of `review/critical-v0.16-amendment.md` and changes the outcome record only. A1: v0.16 stated F111-BF's failure without the basis the v0.4 block requires. At v0.16's sealing (17:25:10Z) BF had no completed attempt and no owner record. The record was made at 17:30Z (`notes/LEDGER.md`, entry "2026-10-20 17:30Z UTC — BF owner record and Kai notification"; `notes/flags-for-kai.md` item 0b), after the outcome was read (the evaluation job's `preflight/export/PREFLIGHT-rows.md` was written at 17:14Z) [Corrected in v0.18: the BF classification was anticipated at 17:10Z, before the evaluation; see the v0.18 entry and the outcome block], and it supersedes the consequence line of owner decision B14-7a. The BF failure classification is therefore results-informed and is labelled so (v0.4 block: "A later change is recorded with its time, as results-informed if made after the outcome is read"). The report to Kai was made in chat (ledger entry 17:34Z); his answer is pending, with the default that the record stands and the alternative of a 72 h BF attempt. FB's failure is a completed attempt and does not depend on this record. B1: the two earlier engineering READMEs that "Sequence and record" requires beside the outcome are cited with their sha256. No rule, criterion, arm, run count, seed, budget, §10 figure, `[D]` label or decision label changes; the v0.3 and v0.4 blocks and §10 are byte-identical to v0.16. Verification is recorded in `review/amendment-v0.17/NOTES.md`; v0.16 is kept in `review/snapshots/v0.16/`.

**v0.16 (2026-10-20, experiment-designer).** This dated amendment records the outcome of the §3.4 v0.4 whole-trained-model factored-export check, which the rule requires in this file before any production training run (v0.3 block, "The outcome, with the per-line table and the report paths, is recorded in PREFLIGHT and in this file's change log"; v0.4 block, "Sequence and record"). Both forms fail. F111-FB fails criterion 2: its factored export is bit-exact, has II = 1 and has fewer CLB LUTs than its normal export (394,351 against 402,252), but its post-opt WNS is −1.090 ns (normal route −1.087 ns). F111-BF has no C-synthesis result on either route: both stopped at the 28,800 s limit and again at the 86,400 s limit of record, and Vivado never ran. By Kai's split-form rule (2026-10-19T16:28:12Z, before the outcome), H1 is the headline count for every binary arm, F100 included; the one-pass provisions (the 24 added H2 runs, the cap note of §19.2 item 8 with its 531.25 GPU-h, and their use in §8 under item 5) do not arise. The trained CH-8 normal export also did not complete C-synthesis within 24 h; it is context, and no criterion depends on it. All readings are engineering pilot readings, not results. The per-line table, the record items and the report paths are in §3.4, "Outcome of the v0.4 check", placed after the v0.4 block. No rule, criterion, arm, run count, seed, budget, §10 figure or `[D]` label changes, and the v0.3 and v0.4 blocks are byte-identical. Verification is recorded in `review/amendment-v0.16/NOTES.md`; v0.15 is kept in `review/snapshots/v0.15/`. [Corrected in v0.17: F111-BF's failure rests on the owner record of 2026-10-20T17:30Z, not on the time limit alone; see the v0.17 entry.]

**v0.15 (2026-10-19, fixer).** This version applies B1 and B2 of `review/critical-v0.14-amendment.md`: §7.3 and the matching v0.14 records cover every gate child that builds a model, and §5 lists the Keras loss guard among CH-8's departures from the release. C items are not applied. Verification is recorded in `review/amendment-v0.14/NOTES.md`, section "v0.15".

**v0.14 (2026-10-19, amendment).** This version records Kai's B10F1-1 decision: §7.3 declares the runner's Keras loss-id guard and loss-term census for every arm, CH-8 included, and every gate child that builds a model (`preflight/investigations/B10-F1.md`, Cause and Recommendation; `decisions.md`, 2026-10-19 (later, 8)). Keras's files are unchanged on disk. The v0.3 block, the v0.4 block, §10 and every other rule are unchanged. Verification is recorded in `review/amendment-v0.14/NOTES.md`.

**v0.13 (2026-10-19, fixer).** This version applies B7 of `review/critical-v0.11-amendment.md`, "Fix verification (v0.12)", under the orchestrator's ruling: in the one-pass case, none of the 24 added H2 runs is admitted until Kai has answered the §8 item and confirmed the exact cap; seeds, admission order, control slot and the `protocol-draft.json` entry are stated (§14, §19.2 item 5). C9–C12 are not applied. The v0.4 block, the v0.3 block, §10 and every other rule are unchanged; v0.12 is kept in `review/snapshots/v0.12/`.

**v0.12 (2026-10-19, fixer).** This version applies the A and B findings of `review/critical-v0.11-amendment.md` (3 A, 6 B, 8 C) under the orchestrator's rulings. A1: no binary arm, F100 included, is admitted to any launch under either count until the v0.4 check has a recorded outcome for both F111 forms, and a failed check is defined; CH-8 and F000 launch independently and train first. A2: the split-form rule in the v0.4 block is Kai's, recorded in `decisions.md` [dec 10-19 both] at 2026-10-19T16:28:12Z before any synthesis, timing or LUT outcome was read: both forms pass, H2 for every binary arm; both fail, H1 for every binary arm; one passes, every binary arm in both forms trains and is read under H1, and the passing form's binary arms (F100, its six fused-BN configurations, three seeds, and the PID runs of its best configuration) also train under H2, 24 more runs, about 154 GPU-h, a 75-run screen of about 481.25 GPU-h (531.25 with the reserve) and a cap of about 530 GPU-h in that case only (a conditional note in §19.2 item 8; §10 unchanged). It supersedes the orchestrator's default recorded at 12:20:19Z ([dec 10-19 split]), which the first v0.12 draft carried. How the one-pass H2 runs enter §8 is a new Kai item (§19.2 item 5). A3: the block is anchored on `preflight/export/PLAN.md` (`:3`, `:35`, `:44-53`, `:67-73`) and on the orchestrator's chat wording to Kai recorded in [dec 10-19 split]; the pilot was trained on the home PC before this amendment, as engineering, not a screen run; the statements that no training run was added and that the checkpoint choice is open are replaced by the specification PLAN.md fixed, labelled `[P]`. B1: the v0.3 recount reads 19 of the 20 case-list pairs, and the 21st pair of the table is named. B2: the v0.4 operational terms are labelled `[P]`, their mapping to the v0.3 conditions is stated, and the timing route of criterion 2 is a Kai item with the normal route's timing recorded beside it (§19.2 item 5, §19.3). B3: export gates are no longer launch preconditions for CH-8 and F000, and binary arms reach them through the check. B4: D10's §0 row records Kai's 2026-10-18 amendment, its label kept, and the v0.11 change log and addendum are corrected in brackets. B5: the v0.4 block hash is recorded below. B6: hardware evidence under H2 at the deciding points is a Kai item with a conservative default. The jet rows (C2) come with A3; no other C item is applied. Addendum B records the v0.12 fixer's exposure. No screen arm, run count, seed, selection rule, statistic, §10 figure or `[D]` label changes; the v0.3 block is byte-identical, and §19.2 items 2–4 and 12 stay open. v0.11 is kept in `review/snapshots/v0.11/`. Appendix A.1 lists the changes at their v0.12 line numbers.

**v0.11 (2026-10-19; Kai decisions of 2026-10-18 and 2026-10-19).** This dated amendment records the decisions quoted in §0. Training runs on NRP without mulder export gates or the clean da4ml rerun as launch preconditions; deciding-point synthesis follows training. The v0.3 headline block is preserved and superseded by v0.4: H2 for binary training and readings if and only if the whole-trained-model factored-export check passes, H1 otherwise. This is a results-informed choice after the OPEN-2 outcome was known (06-review §6.3.3); the v0.4 criteria are fixed before any training result. The factored route is attempted, with the same generated-code INLINE patch in both routes for every model. Route A2 uses a git bundle and a locked venv on `kai-data`, with all NRP campaign state on that PVC and no ConfigMaps. Mulder permission covers the export check only; later deciding-point synthesis is asked again. Infrastructure relaunches have no count limit within the GPU-h cap; code or integrity failures stop for a fix and a new commit and handoff. The orchestrator's unchanged default is no separate H1 campaign, with H1, H2 and NR recorded for every run. §19.2 items 2–4 and 12 remain open launch preconditions. No screen arm, run count, seed, selection rule, statistic, §10 figure or `[D]` label changes. [Corrected in v0.12, critical v0.11 B4: the content of D10's §0 row is amended by Kai's 2026-10-18 decisions, and its label is kept.] Appendix A.1 lists the changes at their v0.11 line numbers.

**v0.10 (2026-10-18, fixer).** This version applies finding B1 of `review/critical-v0.9-amendment.md` §6 (0 A, 1 B, 9 C) as worded there: a §19.3 entry for the B3b-22 label definitions, so that the decision flagged for Kai in §8.7 is in the list of uncertain owner decisions. It also applies the PREFLIGHT owner's B3b-30 wording in §8.6: the S2 plug-in power is computed by the closed form at ρ = 0 and by the §6 simulation at ρ > 0, not "the same simulation"; this also answers the review's C5. The other C items are not applied. No number, threshold, selection or verdict rule, level, run count, `[D]` label or §10 figure changes, and nothing in the headline-rule block or Addenda A and C changes; v0.9 is kept in `review/snapshots/v0.9/`, and the record is the v0.10 addendum to Appendix A.1.

**v0.9 (2026-10-18, fixer).** Pre-data amendment, PREFLIGHT gate G54 finding; no design change. Two items. (1) The §6 power table: the ρ = 0 rows are replaced by the closed form p(δ)^|D| at two decimals and labelled exact (A at |D| = 2, 4, 6: 2.35, 2.62, 2.76, were 2.36, 2.63, 2.77; B at |D| = 2, 4: 1.52, 1.68, were 1.53, 1.69; C unchanged). The ρ = 0.9 rows are labelled Monte Carlo with their standard error, and the simulation layout and root rule are stated as PREFLIGHT owner decision B3b-1 records them, so the table can be reproduced from the STUDY. That layout was fixed in B3b-1 before any comparison and was reproduced independently from its text by the fixer; it gives 2.38 at A, |D| = 6 (was 2.37) and 1.47 at B, |D| = 3 (was 1.46), each within its Monte Carlo standard error (0.0023, 0.0014 σ_d) of the rounding boundary, so those two cells now print what the stated layout gives. Nothing was tuned. No cell moves by more than 0.01 σ_d; the ranges quoted elsewhere ("about 2.4–2.8, 1.5–1.8, 1.3–1.6 σ_d") and every other §6 number are unchanged. The STUDY states no gate on the §6 table, so none is added or restated. (2) PREFLIGHT owner decision B3b-22: `CROSSING`, `LOCAL ADVANTAGE` and `LOCAL DEFICIT`, used in v0.8 but no longer defined, get their definitions back in §8.7, restored from v0.2:463-465, adapted to v0.8's terms and flagged for Kai. No threshold, selection or verdict rule, level, C2 criterion, screen or confirmation run count, `[D]` label or §10 figure changes, and nothing in the headline-rule block or Addenda A and C changes; v0.8 is kept in `review/snapshots/v0.8/`, and the per-item record with locations is the v0.9 addendum to Appendix A.1.

**v0.8 (2026-10-17, fixer).** This version applies the four B findings of `review/constructive-v0.7-delta.md` §4 (arbiter condition P4 (b); 0 A, 4 B) as that file specifies them, under the orchestrator's rulings: B1, the §13 determinism record gains one F000 seed-1 run (60 run-epochs, about 0.05 GPU-h at the planning rate or 0.055 GPU-h with §10's 1.10 factor, from the existing reserve), F000's pooled-variance row and the F000 − CH-8 reading are tied to that record before S1, and the stack of the bitwise reference-fidelity and resume gates is listed as a PREFLIGHT owner decision; B2, the null selection reference's feasibility, simulated unit and form correlation, with a one-candidate gate; B3, an adoption path and lapse points for §19.2 items 13–16; B4, interruption mechanics in §19.2 item 5 and the §13 da4ml row, with that item's "Open for Kai" paragraph moved to its end and the choice between (i) and (ii) left to Kai. B1's added run was adopted by the orchestrator under Kai's 2026-10-17 delegation of checks and validation and is flagged for Kai. Three text clarifications from the review's "For later" list are applied (items 1, 3 and 6). No selection or verdict rule, screen or confirmation run count, `[D]` label or §10 figure changes, and nothing in the headline-rule block or Addenda A and C changes; v0.7 is kept in `review/snapshots/v0.7/`, and the per-finding record with locations is the v0.8 addendum to Appendix A.1.

**v0.7 (2026-10-17, experiment-designer).** This version applies `review/constructive-v0.5.md` (arbiter condition P4 (b); 0 A, 0 B): the text items K1–K9 as S2, held-out and PREFLIGHT outputs, labels and a reading table; the design proposals P1–P4 as new, unadopted Kai items 13–16 of §19.2; a line for Kai under §19.2 item 5 on an interruption after a rule line has started; and Addendum B records of two exposures (the `preflight/env/` reading quoted in the "STUDY v0.5 panel" bullet, and the da4ml outcome fragment in the constructive review, which makes this author partially results-aware). No rule, run count, `[D]` label or §10 figure changes, nothing in the headline-rule block or Addenda A and C changes, and no recommended answer of items 1–12 changes; v0.6 is kept in `review/snapshots/v0.6/` with `SHA256SUMS`, and the per-item record with locations is the v0.7 addendum to Appendix A.1.

**v0.6 (2026-10-17, experiment-designer).** This version applies, without re-review, the 14 merged C items of `review/arbiter-v0.5.md` §6 (critical C39–C44, physics C20–C24 with C21 as text and an option for Kai only, chang C35–C41, and the arbiter's item-5 sequencing) as prose, citations, reporting labels, diagnostic outputs and PREFLIGHT gates; no run count, GPU-hour figure, `[D]` label or pre-registered decision rule changes, no reserved item changes its recommended answer, and the headline-rule block is byte-identical; v0.5 is kept in `review/snapshots/v0.5/` with `SHA256SUMS`, and the per-item record with locations is the v0.6 addendum to Appendix A.1.

**v0.5 (2026-10-17, experiment-designer).** This version applies `review/critical-v0.4.md` (0 A, 2 B, 11 C), `review/physics-v0.4.md` (0 A, 1 B, 5 C) and `review/chang-v0.4.md` (0 A, 3 B, 7 C). v0.4 is kept in `review/snapshots/v0.4/` with `SHA256SUMS`: STUDY `6c07cc3c…` (the version all three reviews read), plan, protocol draft, protocol-check input, OPEN-CHOICES and the three reviews. The per-finding record, with every merge and reconciliation, is in Appendix A.1.

What changed, and why:
- **The clean da4ml rerun can be carried out as written (critical B17, merged with chang B9 and chang B7).** As v0.4 wrote it, the rerun archived the log directory without recreating it, so every line would have failed at build and the headline would have been H1 by procedure. The driver now recreates the directories and runs the four context lines first as a check. On a da4ml 0.6.0 branch it uses a clone of mulder's `m3-env` and leaves the original unchanged. The evidence goes to a new root, and a hashed table script keeps failed lines in the table. Option 0's version rule and one clean rerun per route version are stated (critical C35) (§3.4 Addendum A, §11, §13, §19.2 item 5).
- **One export environment is named (chang B7).** Option 0, the trace, the export audit, every deciding-point export and the cross-backend reload run in it. It is the training pin (Keras 3.14.0, hgq2 0.1.9 with the two shims, quantizers 1.2.2) on JAX on CPU, plus hls4ml 1.3.0, the MHA alias and the da4ml version option 0 fixes. Mulder's `m3-env` cannot build CH-8 or run Chang's trace, and it is never modified (§9, §13).
- **Degenerate layers are defined by their effect, for binary and HGQ layers alike (critical B18, merged with chang B8, physics C17 and critical C38).** A projection or the FFN contraction is degenerate when its output does not vary over the validation jets, by whatever route; the route is recorded. v0.4's attention mechanics do not hold under Chang's per-value quantizers and are replaced (§2.2). C6 gains a static-attention ablation; the uniform-attention ablation and the entropy are redefined (§8.8; critical C31, C32).
- **The primary claim's domain goes to Kai (physics B12).** C2 over every allowance is a claim the design expects to fail at tight budgets. §6 states the bound. S2 reports a plug-in power at BEST's observed gains, labelled optimistic (physics C15). §19.2 item 12 offers three choices: the whole domain; a pre-registered band H ≥ g*, recommended; or, under option C, a ramp point claim at 350,000. §8.2 admits the band rule and nothing else.
- **C4 is renamed "firmware dominance" (critical C37, physics C16).** The rule is unchanged.
- **The control slot is held until S2 (critical C33).** Its control trains in S3, and Kai's answer to item 9 is requested before S1.
- **Thesis sentence and references (physics C18, C19).** The training-procedure caveat is folded into the sentence. BitNet and the prior binary hls4ml study are cited (§1, §2.3, §16).
- **C items** of all three reviews: Appendix A.1. Addendum B records the orchestrator's confirmations for critical C29 and C30.
- **Budget.** No run count changes: item 9 moves one control from S1 to S3, and item 12 and its options add no runs. Every number of §10 is unchanged.
- **The frozen headline-rule block is unchanged.** Its sha256 was recomputed after this revision (below).

**Frozen headline rule (recorded at v0.3; unchanged in v0.4, v0.5, v0.6, v0.7, v0.8, v0.9, v0.10, v0.11, v0.12 and v0.13; superseded by the v0.4 rule below).**
- Rule-block sha256: `d9f2e386d04710e1a424ad5f4511ea830f14b17c1e86afb15fa8959da7168dbb`.
- Recompute with `sed -n '/^<!-- headline-rule v0.3 begin -->$/,/^<!-- headline-rule v0.3 end -->$/p' STUDY.md | shasum -a 256`. The digest covers both marker lines and everything between them.
- Script hashes at the freeze (sha256), reproduced at 07:06, at 08:15 (v0.5), at 16:57 and 17:06 (v0.6, before and after its edits) and at 17:19 (v0.7) on 2026-10-17:

  | file (under `preflight/da4ml-test/scripts/`) | sha256 |
  | --- | --- |
  | `cases_bc.txt` | `1ba9c416b1c4c6007ad07361bd95b1e902058e66039df89e4b5be1a8928a3191` |
  | `stage_b_build.py` | `db49a4041ff1a9a041e9a3d9c87bded997ff3adc96844366941f981028ba83d6` |
  | `stage_b_build_pos.py` | `f214dc3430a0871ee39119a93a719da3448d0fe9868e4cbae70520e6dc9a00d2` |
  | `run_case.sh` | `1253773bd99b58015620e0e99bd12f5e575592d772545a453db43245004169f8` |
  | `synth_one.sh` | `32201339fcab6d42cad096b393fa7d41ca0380e5948f718422f4c07869ee9bb9` |
  | `vivado_ooc.tcl` | `70d269f430e8fe364ae660f9f13fa81bcb8f7c0a9c373f0a1dcecbe43eb14c32` |
  | `clock.xdc` | `e4ec5851c4fff9a4f8c0caf420e487be9e43c2a8f3ce83cb9954b3ddacb4339f` |
  | `common.py` | `cfb50a22bb95a5746e31ff456bd6b8721e9cdad684babd02c0789361e266a1fa` |
  | `parse_reports.py` | `a889ce8ff21f94eae61f9322e332930e99f22d8beb81eadca647c0922707ae38` |

  `env.sh`, which `run_case.sh` and `synth_one.sh` source, is not in this list: sha256 `a1c095ef7ab3b6dc32ee07dcf0b9b3c2e1794b8c48d8c04d970614592f459b78` (05:11), recorded here in v0.4 (critical B14).

**Headline rule v0.4 (recorded in v0.11, amended in v0.12; critical v0.11 B5).**
- Rule-block sha256: `b3f0cc935d0bc2edce691474d7321b7d3983ca17c9f2f28befc75e984db92b23`. The first v0.12 draft, with the orchestrator's 12:20:19Z split-form default, hashed to `bc1db148e68525379c4f85f5600e6926ca5e07c61e00cdf70375dd276a210bef`; Kai's split-form decision of 2026-10-19T16:28:12Z replaced that default before the version was closed.
- Recompute with `sed -n '/^<!-- headline-rule v0.4 begin -->$/,/^<!-- headline-rule v0.4 end -->$/p' STUDY.md | shasum -a 256`. The digest covers both marker lines and everything between them.
- Any change to this block after its hash was recorded goes to Kai.
- The v0.11 block hashed to `99fed7b8682c5e77a62adf6e99d1c443ff663376e669ec354ca1258644da8382` (the v0.11 critical review, and the v0.12 fixer before its edit). The block digest `c1e2bb91…3659` and STUDY digest `b1af6377…5c96` in `review/amendment-v0.11/NOTES.md` predate the orchestrator's correction and are superseded.

**v0.4 (2026-10-17)**: Appendix A.2, with its change-log text kept verbatim. **v0.3 (2026-10-17)**: Appendix A.3. **v0.2 (2026-10-17)** and **v0.1 (2026-10-16)**: Appendix A.4.

## 0. Status, labels and choices

This file is the design only. It authorizes no compute.

All paths are relative to `/Users/kaiyamaguchi/lab/bnjettag`. Earlier line citations refer to the local sources as read on 2026-10-17; v0.11 cites the dated decisions and engineering records read on 2026-10-19. No data file, cluster, mulder session or network was touched.

**Citations of `.claude/memory/decisions.md` (critical C22).** The log grows at the top, and its line numbers shifted twice after v0.3 was written. This file therefore cites an entry by a short tag and a bullet by its opening words, never by line number:
- [dec 10-19 both]: "2026-10-19 (later, 4) — Kai: in the split-form case, do both (2026-10-19T16:28:12Z)" (v0.12);
- [dec 10-19 split]: "2026-10-19 (later, 3) — chang-bitlinear: split-form default for the H2 check, fixed before its outcome (2026-10-19T12:20:19Z)" (v0.12);
- [dec 10-19 PVC]: "2026-10-19 — Kai: everything on NRP stays inside the kai-data PVC";
- [dec 10-18 H2]: "2026-10-18 (later, 2) — Kai: the plan is approved; make H2 valid, test on mulder";
- [dec 10-18 NRP]: "2026-10-18 (later) — Kai: no mulder before training; run the trainings on NRP";
- [dec 10-17 ConfigMaps]: "2026-10-17 — ConfigMaps hold no data and are deleted once their Job ends (NRP admin request)";
- [dec 10-17 delegation]: "2026-10-17 — chang-bitlinear: OPEN-1/2 decided by Kai; overnight automation delegated; Claude models preferred";
- [dec 10-16 later]: "2026-10-16 (later) — Kai re-decides Q1, Q3, Q8-Q11 and authorizes build, verification and GPU queueing";
- [dec 10-16]: "2026-10-16 — Kai: new direction for the binary MHA work (Chang's setup kept; binary weights only where they help; beat Chang)".

Appendix A.2 and A.3 keep earlier versions' line citations verbatim; they refer to the file as it was when those versions were written.

**Decision sources for v0.11.** The quoted words are Kai's, with source line wrapping joined.
- Training and sequencing: [dec 10-18 NRP], "Kai (chat, verbatim)": "WHY ARE U CHEKCING MULDER JOBS? JSUT RUN THE TRAINGINS ON NRP BRO THER IS NO NEED FOR ALL FO THAT". Its "Effect" bullet removes mulder from the path to training, including §14's export gates and Addendum A's clean rerun; deciding-point synthesis still follows training (Q14).
- H2, the factored route and limited mulder permission: [dec 10-18 H2], "Kai (chat, verbatim)": 'okay wait then in that case u can test on mulder and we should work on getting H2 to work since thats the "Edge" that we acutally have if im correct'. Its effect is the conditional whole-model check in §3.4, not a statement that H2 has passed. `campaigns/2026-10-16-chang-bitlinear/notes/flags-for-kai.md`, "Decided by Kai on 2026-10-18", records the INLINE patch as part of both export routes, applied identically to every model.
- Launcher and runtime: [dec 10-18 H2], "Kai on the plan (chat)": "okay the plan seems perfect". The approved plan's Phase 2 is a git bundle on `kai-data`, no ConfigMaps, and `run_handoff.py` bundle mode; [dec 10-18 NRP], "Orchestrator ops defaults", records the hashed lock installed into a venv on the PVC.
- PVC scope: [dec 10-19 PVC], "Kai (chat, verbatim)": "okay and makke sure to keep everyhting in the nrp inside our pvc". All campaign state on NRP is on `kai-data` (§11).
- Resubmission: Kai's earlier instruction, "if things fail or queue fails, just loop until it succeeds", is the instruction already cited in §11 to [dec 10-16 later], the GPU-queueing bullet; it is reproduced in `campaigns/2026-10-16-chang-bitlinear/notes/build-todo.md`, the B17 retry-policy item. §11 states its infrastructure-only scope and the existing fix procedure.
- No separate H1 campaign: [dec 10-18 H2], "Orchestrator default stated to Kai, open to his correction". This is the orchestrator's stated default that Kai did not change, not a separate verbatim Kai decision. Every run records H1 and NR beside H2 (§3.4).


Labels:
- `[D]`: Kai's binding design.
- `[OD-n]`: orchestrator default under Kai's 2026-10-17 delegation (`decisions.md` [dec 10-17 delegation], bullets "Kai (chat, verbatim)", "STUDY v0.2 (experiment-designer, Opus) §19 lists nine" and "STUDY v0.2 reviews"). Each is reviewable by Kai and listed in §19.
- `[P]`: an owner proposal that implements a `[D]`. These are reviewable by Kai and flagged in §19 when confidence is not high.
- `[A]`: a requirement.
- `[L]`: a limitation.

Numbers marked "derived" are arithmetic, never observations.

| choice | status | what it fixes | where |
| --- | --- | --- | --- |
| OPEN-1 binary/BN form | [D9] Kai, 2026-10-17 (`decisions.md` [dec 10-17 delegation], "OPEN-1 (Kai"; `OPEN-CHOICES.md:90-92`) | both forms trained; one kept on validation before any held-out read | §2.3, §5, §8.3 |
| OPEN-2 scale cost | [D10] Kai, 2026-10-17 (`decisions.md` [dec 10-17 delegation], "OPEN-2 (Kai)"; `OPEN-CHOICES.md:93-95`); amended by Kai, 2026-10-18 ([dec 10-18 NRP], [dec 10-18 H2]) and 2026-10-19 ([dec 10-19 both]); dated amendment v0.11 | v0.4 criteria fixed before training results; H2 if the whole-model factored-export check passes for both binary forms, H1 if it fails for both, and if one form passes, H1 for every binary arm plus H2 runs of the passing form's binary arms; the binary count is fixed by the mechanical v0.4 rule before any binary training, while CH-8 and F000 train first by Kai's decision; all counts reported | §3 |
| OPEN-3 seeds | [OD-3], amended by [OD-15] | 3 paired seeds, now a screen | §6 |
| OPEN-4 front rule | [OD-4], amended by [OD-15] and physics B3 | step envelope, domain, BEST, control checkpoint policy; verdicts moved to the confirmation | §8 |
| OPEN-5 PID start | [OD-5] | fresh paired runs, λ₀ = 2e-8, released gains | §7.2 |
| OPEN-6 W8A8 formats | [OD-6] | eight total bits, per value, trained steps | §5 |
| OPEN-7 GR derivative | [OD-7] | scale derivative only | §5 |
| OPEN-8 budget | [OD-8], re-affirmed as resolution (c) | 400 GPU-h including 50 reserve; at most 4 concurrent; drop order. Premise changed with D9 (§10) | §10 |
| OPEN-9 runtime | [OD-9] | JAX, float32 matmul, pinned stack | §7.3 |
| OPEN-10 synthesis | [OD-10] | all deciding points, 7-day window | §9 |
| v0.2 §19 item 4 | [OD-11] (`decisions.md` [dec 10-17 delegation], "STUDY v0.2 … §19", item 4) | pre-stage runs of the chosen form are the factorial cells | §5 |
| v0.2 §19 item 6 | [OD-12], resolution (e) (`decisions.md` [dec 10-17 delegation], "STUDY v0.2 … §19" item 6; "STUDY v0.2 reviews" (e)) | Q16's 350,000 read in H units; dated delegated reading, amendment note under D6. Amends D6, so Kai confirms it individually before any launch (§14; §19.2 item 4) | §7.2 |
| v0.2 §19 item 8 | [OD-13], resolution (a) (`decisions.md` [dec 10-17 delegation], "STUDY v0.2 reviews" (a)) | learned-width scale quantizer; replaces "β exact until readings exist"; amendment note under D3. Amends D3 and fixes BF's quantizer placement inside D9, so Kai confirms it individually before any launch (§14; §19.2 item 2) | §2.2 |
| v0.2 §19 item 9 | [OD-14] (`decisions.md` [dec 10-17 delegation], "STUDY v0.2 … §19" item 9) | collapse δ = 0.05 and its windows only; the restricted domain is removed (§8.2) | §8.8 |
| resolution (b) | [OD-15] (`decisions.md` [dec 10-17 delegation], "STUDY v0.2 reviews" (b)) | three-seed screen without verdicts; verdicts only from a fresh-seed confirmation, whose funding and size are Kai's (§19.2 item 11). Changes D1's claim structure, so Kai confirms it individually before any launch (§14; §19.2 item 3) | §6, §8.6–8.7 |
| resolution (d) | [OD-16] (`decisions.md` [dec 10-17 delegation], "STUDY v0.2 reviews" (d)) | exact-forward surrogate, same Jacobian as Q4 | §2.2 |
| claim domain (physics B12, v0.5) | open: Kai decides before any launch (§19.2 item 12); recommended (b) | C2's domain D_C2: the band D_band = {g ∈ D_conf : g ≥ g*} under (b), or all of D_conf under (a); (c) adds a ramp point claim at 350,000 under option C | §8.2, §8.7, §17 |

**Glossary.** "Q" numbers are Kai's design questions:
- Q1–Q11 are the Russell-plan questions (`docs/design/2026-10-11-russell-questions-recommendations.md`), re-decided on 2026-10-16 (`decisions.md` [dec 10-16 later]).
- Q12–Q16 were opened and decided on 2026-10-16 (`decisions.md` [dec 10-16], bullets "Q12 decided by Kai" to "Q16 decided by Kai").

| label | content |
| --- | --- |
| Q1 | activation steps: Chang's HGQ steps, trained throughout |
| Q2 | fixed A8/A6/A4 activation widths: withdrawn |
| Q3 | normalization: Chang's fused BN |
| Q4 | weight gradient: identity straight-through, with no derivative through α or β |
| Q5–Q7 | pooling, old data, particle count: not used here; N is set by Q13 |
| Q8 | quantizer granularity: per value |
| Q9 | biases and adds: Chang's placement |
| Q10 | firmware: hls4ml with da4ml |
| Q11 | controls: as many as possible |
| Q12 | objective: a better front than CH-8 |
| Q13 | N=8 first, against Chang's MHA-8 hardware row |
| Q14 | axes: top-1 against the cost count, with LUT at the deciding points and macro AUC beside |
| Q15 | full factorial over three groups |
| Q16 | ramp for all configurations, then a PID at 350,000 for the best configuration and for CH-8 |

D1–D10 are the `[D]` labels of this file.

## 1. Question, null, bearing

**[D1, Q12–Q15] Question.** There are two stages.
- **Screen (seeds 1–3, funded).** Which nonzero binary configuration, in the form chosen on validation (§8.3), has the best validation front against CH-8 at matched allowances of the headline cost count H (§3)? The screen reports held-out gaps descriptively.
- **Confirmation (fresh seeds, needs Kai's budget approval).** Does the validation-selected configuration's held-out top-1 front lie above CH-8's at every allowance of the claim domain D_C2 (§8.2)? D_C2 is the band of allowances at and above g* under the recommended option (b) of §19.2 item 12, or every allowance of the evaluation domain under option (a); Kai fixes it before any launch. The seeds are 4–8 as adopted, or 4–11 under the eight-seed options; Kai chooses the size (§8.7, §19.2 item 11).

CH-8 is Chang's released code run on the same data.

**Nulls (confirmation).**
- C2: at some allowance g ∈ D_C2, the mean paired held-out top-1 gap of the selected configuration over CH-8 is ≤ 0.
- C3, when the PID arms are in the confirmation: under the lab PID at 350,000 H, the mean paired held-out top-1 gap of PID-BEST over PID-CH is ≤ 0. Under the ramp-only option (§8.7, option C) C3 is not tested, and the PID comparison stays a screen output (critical C24).
- C3r, only under option C together with option (c) of §19.2 item 12: at g = 350,000 on the ramp front, the mean paired held-out top-1 gap of BEST over CH-8 is ≤ 0.

The screen tests no null.

**Bearing.** A confirmed positive result supports selective binary weights as a useful constraint on this tagger under the declared accounting.
- An advantage in H alone does not establish a LUT advantage. Q14 asks for LUT at the deciding points.
- Binary kernels lose HGQ's per-weight pruning (physics B5). HGQ trains single weights to zero width, so quantization and pruning happen in one loss (`literature/jet-tagging-transformers/2510.24784_submicrosecond_transformers_jet_tagging_fpga.md:31,35`). A binary kernel has no zero weight. It can drop a whole layer or output column only when its scale quantizer reaches zero bits (§2.2), and it can thin inputs only through zero-width activations.
- A deficit at tight budgets may therefore come from the lost pruning channel rather than from the sign map. The design logs both channels (§12) but does not separate them; a ternary {−1, 0, +1} control, which would, is not run (§5).
- **Prior work (physics C18).** The binary weights follow BitNet's BitLinear, which trains 1-bit weights from scratch as a drop-in replacement for a linear layer (H. Wang et al., arXiv:2310.11453; `literature/bitnet-1bit-ternary/2310.11453_bitnet_scaling_1bit_transformers.md:1-20`). Binary and ternary networks have been built with hls4ml before. That paper's abstract reports "similar performance to the higher precision implementation while using drastically fewer FPGA resources", and retaining "full precision on a selected subset of network components" (J. Ngadiuba et al., "Compressing deep neural networks on FPGAs to binary and ternary precision with hls4ml", Mach. Learn.: Sci. Technol. 2, 015001 (2020), arXiv:2003.06308 (the arXiv listing orders the authors alphabetically); abstract and article pages read in `review/arbiter-v0.5.md`; both quoted phrases are verbatim on the arXiv page; a lab literature record is made before any outward use, §16). That is selective binarization in hls4ml, the idea of this thesis, on a different model, including LHC jet identification, and against fixed-precision baselines. What is new here is a learned-width HGQ comparator under the released recipe, a transformer, and an accuracy-cost front rather than one operating point.

**What a confirmed gain would mean (physics B9).** Every binary model this campaign can export is also a model CH-8's quantizers can export, at the same cost.
- Each exported binary constant is ±q, with q on a SAT_SYM fixed-point grid of exact width w(q) ≤ 23 bits (§2.2). Chang's per-value KBI kernel quantizer has the same overflow mode and width bound and an unconstrained integer part (`ic=None`, `reference-code/HGQ2/src/hgq/quantizer/config.py:75-91`; per value, `homogeneous_axis=()`), so it represents ±q exactly at b = w(q).
- H1 and H2 are computed from the exported constants alone, and H2 gives a Chang column with a single magnitude the same treatment as a binary column (§3.2). Identical constants therefore give identical H1 and H2, and the stock export route receives the same constant matrix with the same input formats. Only NR differs, because it uses learned bits.
- The binary configurations therefore search a strict subset of CH-8's exported-model family, at equal headline cost and equal firmware. A confirmed ABOVE cannot mean that binary weights are a cheaper representation. It can only mean that the constrained parameterization, trained under the released recipe and λ schedule, reaches accuracy-cost points that HGQ's per-weight training does not reach. That is an optimization or inductive-bias result under this recipe, not a representational or firmware one.
- The same property fixes the floor. At w = 1 a binary site costs P·n_in·n_out·b̄ under H1 and H2 alike, which is the cost of a dense 1-bit HGQ kernel (§3.5). A 1-bit SAT_SYM HGQ weight takes the values {−2⁻ᶠ, 0, +2⁻ᶠ} with its own exponent per weight, so it is strictly more expressive at that cost. Only zero-width inputs or a zero scale take a binary site lower. A deficit at tight allowances is therefore what the cost definitions predict.
- The control that would separate the sign constraint from the shared width, an HGQ kernel with one learned width per column or per layer, is not run (§5).

Sources: `decisions.md` [dec 10-16], bullets "Q12 decided by Kai" to "Q16 decided by Kai"; `docs/conventions/quantization-and-cost.md:19-25`.

The claim concerns the released two-head MHA at N=8. It does not concern the paper's single-head model, other N, or all-binary superiority (`reference-code/HGQ2-examples/jsc150/model.py:178-210`; `literature/jet-tagging-transformers/2510.24784_submicrosecond_transformers_jet_tagging_fpga.md:21-35`; §16).

**Thesis sentence for review panels (physics B1, C19; v0.6: physics C24, chang C41).** "A transformer jet tagger for the CMS Level-1 trigger, the released HGQ2-examples jsc150 two-head MHA at N=8 (example code accompanying Sun et al., arXiv:2510.24784, at a snapshot that postdates the paper and uses two heads where the paper reports one), in which training BitNet-style binary weights (Wang et al., arXiv:2310.11453) in validation-selected layer groups under the released recipe reaches a higher held-out top-1-versus-H front over an allowance domain fixed by a pre-registered rule (on screen validation, before any confirmation seed) than the same released model rerun on the same split." The sentence is quoted only together with the bullets below.
- In the binary weights, the signs come after global mean centering and the mean-absolute scale has a learned bit width.
- H is a declared bit-product count of the exported constants, H1 or H2.
- The allowance domain is fixed by Kai before any launch (§19.2 item 12). Under the recommended option (b) it is the band D_band, with g* set on screen validation by a rule written now (§8.2); D_band keeps only the allowances of D_conf, so CH-8's validation feasibility in the confirmation seeds also enters it, before any held-out read. The sentence then reads "a higher … front at the grid allowances of D_band, from its lowest member to 450,000 H"; with one allowance, "a higher top-1 at H = ⟨g⟩".
- The three-seed stage selects the combination on validation; a claim rests on fresh confirmation seeds (five or eight, §8.7). LUT statements are limited to the deciding checkpoints.
- Every exportable binary model is also an exportable HGQ model with the same H1, H2 and firmware, so a confirmed advantage is a training-procedure result under this recipe, not a representational one (above).

This is the paragraph to give the physics reviewer of the next panel (physics C13); the v0.2 wording ("identical to a published design", "the best combination is identified") is retired.

## 2. Model

### 2.1 Chang's graph [D2, Q1–Q3, Q8, Q9]

Build `get_model('xfm', 7, 7, 1e-8, 8, True)`, layer by layer:
- input 8×3;
- fused-BN embedding 3→24 with ReLU (`model.py:192`);
- affined tanh LUT (`:194`);
- two-head attention, key and value dimension 16, output width 24 (`:195`);
- residual add of the ReLU embedding and the attention output (`:197`);
- a second affined tanh LUT (`:198`);
- FFN on the tanh output: 24→32 with fused BN and ReLU, then 32→24 without bias (`:199-200`);
- residual add of the pre-tanh stream and the FFN output (`:202`);
- global average pooling;
- three fused-BN/ReLU head layers of width 32;
- five logits without bias (`:204-208`).

Source: `reference-code/HGQ2-examples/jsc150/model.py:178-210,277-327`.

Keep the released quantizer setup unchanged:
- learned widths and trained steps throughout;
- the nested scopes:
  - weights and biases are KBI with SAT_SYM, `b0=4`, `i0=0`;
  - datalane quantizers are KIF with WRAP, `f0=7` and `ic=MinMax(0,12)`, per value and shared over the batch only;
  - the two tanh tables are KBI with WRAP and `bc=Min(4)`, with per-value formats (`reference-code/HGQ2/src/hgq/layers/activation.py:37-44,84-85`). The exception is `QSoftmax`, which forces its exp and inv table output quantizers to SAT, RND_CONV and k0 = 0 (`reference-code/HGQ2/src/hgq/layers/softmax.py:65-75`), with one format per table in the built MHA (`review/chang-v0.4.md`, software check 4). The single format comes from `softmax_allow_heterogeneous_table=False` (`reference-code/HGQ2/src/hgq/layers/attn/mha.py:57`), passed to both tables (`softmax.py:83,94`), which sets `homogeneous_axis=None` and `heterogeneous_axis=()` (`activation.py:42-44`) (chang C40). This is descriptive; CH-8 is the released builder (chang C34);
  - the outer KBI/KIF regularizers;
  - layer `beta0=0`;
- the commented-out line at `model.py:193` stays commented. If uncommented, it would be a datalane KBI scope with `bc=Min(1)`; that it matches the paper's 1-bit attention floor is an inference from the paper's §3, not a fact of the code (chang C20);
- quantized biases and adds as released: Q, V and O have biases and K has none; the FFN contraction and the classifier have no bias.

Sources: `model.py:181-208,277-295`; `reference-code/HGQ2/src/hgq/layers/attn/mha.py:152-202`; `reference-code/HGQ2/src/hgq/quantizer/config.py:75-91,550-611`; `decisions.md` [dec 10-16 later] and [dec 10-16] (Kai's verbatim message and the "Read as" bullet).

Do not add the paper's attention floor, standalone BN, fixed activation ladder, step freeze or BitNet activation normalization. Only the selected kernel quantizers change.

### 2.2 Binary map and scale quantizer [D3, Q4; OD-13; OD-16]

For the tensor T presented to the binary map:
- α = mean(T) over the whole tensor;
- S = Sign(T − α), with Sign(0) = −1;
- β = mean(|T|), uncentered: per layer, or per output column where §2.3 says so;
- g is the exported magnitude: g = β at non-BN sites, and g_c from §2.3 at fused-BN sites;
- the exported kernel is K = q_g(g)·S, where q_g is the scale quantizer below.

**Amendment note under D3, 2026-10-17 [OD-13].** v0.2 exported B = β·S with β an unrounded float32 value. Under the orchestrator's resolution (a), the magnitude now passes through q_g before export (`decisions.md` [dec 10-17 delegation], "STUDY v0.2 reviews" (a)). This was decided under Kai's delegation, not by Kai, and is listed in §19.

**Scale quantizer q_g [OD-13].** q_g is an HGQ2 `Quantizer` built from `QuantizerConfig('default', 'weight')` inside Chang's scopes. It therefore receives exactly the configuration his kernel quantizers get (derived from `model.py:185-188,278-290`, `reference-code/HGQ2/src/hgq/quantizer/config.py:75-91,590-600`). Every scope field is inherited; the resulting configuration is:
- KBI type, `k0=1`, `b0=4`, `i0=0`, so the initial fractional bits are 4;
- RND rounding and SAT_SYM overflow;
- `bc=MinMax(0,23)`;
- `br=MonoL1(1e-8)` and `ir=MonoL1(1e-8)`;
- `i_decay_speed=1e-3`, set by Chang's outer scope (`model.py:284`) and unused under SAT_SYM, because the decay variable is built only for WRAP (`fixed_point_quantizer.py:59-61`) (physics C14);
- b and i trainable. i trains because SAT_SYM is not WRAP (`reference-code/HGQ2/src/hgq/quantizer/internal/fixed_point_quantizer.py:233`).

**When q_g is built (chang C22).** `QuantizerConfigScope` changes the default configurations when the scope is entered (`config.py:613-619`), and `QEinsumDense` builds its kernel-quantizer configuration in `__init__` (`einsum_dense.py:45`). q_g is therefore instantiated inside `get_model`'s scopes, in the layer's `__init__`, never later. Gate (§13): q_g's configuration equals the MHA projection configuration `_qkvo_kq_conf` (`mha.py:64`) field by field, and survives a serialized reload.

Only the granularity differs from a kernel quantizer. There is one (b, i) per layer at a non-BN site (scalar input) and one per output column at a fused-BN site (input shape (n_out,)).

Consequences:
- At initialization the grid step is 2⁻⁴ = 1/16 and the largest magnitude is 15/16 (derived). A larger g saturates until i grows, as Chang's kq(W·s) does under the same scope (`einsum_dense_batchnorm.py:165-168`).
- **Two ways to a zero scale (critical C16, chang C15).** They differ in cost.
  - *Pruned*: b = 0. The SAT_SYM bound 2^i − 2^(i−b) is then 0, so q_g outputs 0 in training and inference alike. `bits_` is 0, so the scale costs 0 in H1, H2, NR and the penalty (b).
  - *Rounded to zero*: b > 0, but RND sends g to 0 (|g| below half the grid step). The exported constant is 0, so H1 and H2 charge 0. `bits_` does not depend on the value (`reference-code/HGQ2/src/hgq/quantizer/quantizer.py:86-88`), so NR and the penalty (b) still charge b times the input widths.
  - The inference mask "k+i+f ≤ 0" (`fixed_point_quantizer.py:148-149`) never fires here, because k = 1 and k+i+f = 1+b ≥ 1.
  - §3.1(b) decides "column is zero" from the rounded bits, so only a pruned column leaves the penalty.
- **Degenerate layers, by effect (physics B10; v0.5: critical B18, chang B8, physics C17, critical C38).** Q, K, V, O or the FFN contraction, binary or HGQ, in any arm, is *degenerate* when its output, after its output quantizer where it has one, has zero variance over the validation jets at every element. The route is recorded:
  - kernel pruned (zero widths: b = 0 for a binary scale, all widths zero for an HGQ kernel);
  - kernel rounded to zero (nonzero widths, every exported constant 0);
  - input quantizer of zero width at every position and channel (inference masks every value with k+i+f ≤ 0, `fixed_point_quantizer.py:148-149`);
  - output quantizer of zero width at every element;
  - other (zero variance by none of these routes).

  Under Chang's per-value datalane scope, the Q, K and V output quantizers have one format per (position, head, dim) (`reference-code/HGQ2/src/hgq/layers/attn/mha.py:162,181,200`), the softmax output quantizer one per (head, query, key) (`mha.py:72,327-341`), and the exp and inv tables one format each (`softmax.py:65-75` for k0, SAT and RND_CONV; the single format from `softmax_allow_heterogeneous_table=False`, `mha.py:57`, passed to both tables, `softmax.py:83,94`, and applied at `activation.py:42-44`, chang C40; `review/chang-v0.4.md` check 4). The consequences differ by layer (chang B8; inference from these sources, not measured):
  - **Degenerate K** by any named route (K has no bias, `mha.py:178`, so K ≡ 0): every score is 0, so the attention weights do not depend on the input. Each (head, query, key) weight is then rounded on its own grid, so the weights are equal only where those formats agree and can be zero for some keys. The attention is input-independent, not exactly uniform.
  - **Degenerate Q** (for example a zero Q kernel; v0.6: critical C42, chang C35): With a zero Q kernel, Q_t = oq_t(b_q) is input-independent. It can differ between query positions only through the per-position Q output-quantizer formats (`mha.py:159,162`), so the attention need not be query-independent; the scores depend on the input only through the keys. Where Q_t is identical across t, every query row of a head has the same attention distribution, which is query-independent attention pooling (inference). A degenerate Q equal to 0 at every element gives zero logits and is reported as for a degenerate K ("input-independent attention weights").
  - **Degenerate V** (for example a zero V kernel): V_j = oq_j(b_v) is fixed per position j and is mixed by input-dependent weights whose rows need not sum to 1. The branch is not constant.
  - **Degenerate O**: the branch is constant.
  - **Without any zero kernel**, a projection whose input or output quantizer has zero width everywhere is equivalent to the above, and per-(head, query, key) softmax output widths can prune or reweight attention entries statically. The paper floors attention widths at one bit "to disable pruning for training stability" and reports N=64 attention collapsing into a Deep Set despite that floor; the released `xfm` leaves the floor commented out (`literature/jet-tagging-transformers/2510.24784_submicrosecond_transformers_jet_tagging_fpga.md:33-35,60-63`; `model.py:193`).

  Degenerate layers with their route, zero columns with theirs, and per head the fraction of zero-width entries of the softmax output quantizer are recorded at every frozen checkpoint for every arm, CH-8 included (§8.1, §12). Verdicts that rest on a degenerate layer are labelled (§8.7). A zero-variance detector is a PREFLIGHT gate (§13).
- The grid is HGQ's ordinary fixed-point grid. There is no β floor and no power-of-two rounding.

**Surrogate [OD-16].**

K = q_g(stop_gradient(g))·S + (T − stop_gradient(T)).

- **Forward.** The forward value is exactly q_g(g)·S in float32, because T − stop_gradient(T) evaluates to exactly 0 and adding 0 is exact. q_g's own output is exact on its grid because `bc = MinMax(0,23)` keeps |q·2^f| below 2^23; HGQ's float32 RND, floor(x·2^f + 0.5)/2^f, is not exact at 24 bits (critical C16; the frozen test script records the same failure, `preflight/da4ml-test/scripts/stage_b_build.py:45-48`). The v0.2 form T + stop_gradient(B − T) is not exact: in a synthetic float32 check it differed from B in 154 of 768 entries (`review/critical-v0.2.md:43-46`).
- **Backward.** ∂K/∂T = I, the identity of Q4. No derivative flows from α, S or β into T, so there is no gradient from β into W.
- **What the scale quantizer learns from.** Its b and i receive the task-loss gradient ∂L/∂q_g = Σ S·∂L/∂K over its reduction, through HGQ's own quantizer surrogate, as at every Chang kernel quantizer. They also receive the cost-penalty gradient through `bits_` (§3.1b) and their MonoL1 regularizers.
- **Status.** This is a numerically exact implementation of Q4 with the same Jacobian, not a change of the decision. Kai is told in one line (§19).
- **Every consumer reads the exact product q_g(g)·S, never an STE sum.** The consumers are the training forward, the penalty (b), frozen H1/H2, `qkernel`, the export and every binary-value check.
- **No kernel quantizer.** No HGQ kernel quantizer (kq) is built or regularized at a binary site. `QEinsumDense.__init__` always creates one (`reference-code/HGQ2/src/hgq/layers/core/einsum_dense.py:45-46`), so the binary layers must not. The only quantizer variables at a binary site are q_g's (critical C14, chang C6).

Sources: `docs/design/2026-10-11-russell-questions-recommendations.md:175-194`; `campaigns/2026-10-12-mha-bitnet/STUDY.md:1052-1054`; `einsum_dense.py:72-83`.

**Non-BN sites.** At Q/K/V/O and the FFN contraction, T = W and g = β per layer, identically in both forms.
- For Q, K and V (kernel 24×2×16) and O (kernel 2×16×24) the layer β pools both heads.
- An output column is one (head, dim) pair for Q, K and V (32 columns) and one output feature for O (24 columns) (chang C10).

### 2.3 The two binary forms at fused-BN sites [D9, OPEN-1]

**Chang's fused layer.** The scale is s_c = γ_c / sqrt(v_c + ε). Here ε is the Keras backend `epsilon()`, not the layer's stored `epsilon=1e-3` (`reference-code/HGQ2/src/hgq/layers/einsum_dense_batchnorm.py:4,29,61,165`).
- In training, (μ_c, v_c) are the batch moments of x·W, where x is the input-quantized activation and W the latent kernel. They are computed only when `training and self.trainable`. Moving averages are updated with momentum 0.99 (`:28,189-202`).
- At inference and export the moving moments are used (`:176-183`).
- The kernel is kq(W·s) (`:165-168`) and the bias is d_c = bq(b_c − μ_c s_c) (`:170-172`).

In both forms the scale quantizer acts on the exported per-column magnitude g_c, at Chang's quantizer site, after the fold, with one (b, i) per output channel C. The forms differ in β (chang C21):
- FB: one β_c per output channel C, reduced over the contracted input axis c of `...c,cC->...C` (critical C4);
- BF: one β per layer, and g_c = β·|s_c| per channel.

**FB: fold then binarize, per-column magnitude.**
- U_ic = W_ic s_c, α = mean_{i,c} U_ic, and S_ic = Sign(U_ic − α).
- g_c = β_c = (1/n_in) Σ_i |U_ic| = |s_c| (1/n_in) Σ_i |W_ic|.
- K = q_g(stop_gradient(g))·S + (U − stop_gradient(U)) replaces kq(U) at `:168`.
- The exported column is K_ic = q_g(g_c) S_ic. A negative s_c is carried by the signs. s_c = 0 gives g_c = 0, and so a zero column.

**BF: binarize then fold, per-layer β, quantized after the fold.**
- α = mean W, S = Sign(W − α) and β = mean|W|.
- The folded column magnitude is g_c = β·|s_c|.
- K_ic = q_g(stop_gradient(β)·|s_c|)·stop_gradient(sign(s_c)·S_ic) + s_c·(W_ic − stop_gradient(W_ic)).
- The exported column is K_ic = q_g(β|s_c|)·sign(s_c)·S_ic. The signs flip where s_c < 0, and s_c = 0 gives a zero column.
- Why the quantizer sits after the fold [OD-13]: s_c is an unrounded float32 BN scale, so quantizing β alone before the fold would leave BF's exported constants at float32 width, about 24 bits, while FB's are learned-width. That would turn the form choice into a comparison of a priced against an unpriced scale.

**Gradients, the direct kernel path.**
- FB: ∂L/∂W_ic = s_c ∂L/∂K_ic, and γ_c receives Σ_i W_ic ∂L/∂K_ic / sqrt(v_c+ε). These are unchanged from v0.2.
- BF: ∂L/∂W_ic = s_c ∂L/∂K_ic. γ_c receives Σ_i β S_ic ∂L/∂K_ic / sqrt(v_c+ε), the binary value through q_g's straight-through input derivative, as in v0.2. The term s_c·(W − stop_gradient(W)) contributes no γ gradient, because its value is exactly 0.
- **Saturated columns (critical C18).** HGQ clips before rounding: `forward` saturates before it rounds unless the mode is WRAP (`quantizers` 1.2.2 `fixed_point/_fixed_point_ops.py:183-185`), and the clip is the `where` at `:15-18` (chang C30). So q_g's input derivative is 0 where β|s_c| exceeds the largest grid value 2^i − 2^−f (15/16 at initialization). In BF, γ_c's direct gradient therefore vanishes on a saturated column, as it does in Chang's kq(W·s). In FB, γ's direct path runs through U − stop_gradient(U) and is not gated. The forms thus differ in this gradient as well, and the difference can enter the form choice. Saturated columns are logged (§12) and are a §13 edge case.

**Gradients, the moment and bias paths (chang B3).** `ops.moments` (`einsum_dense_batchnorm.py:190-196`) has no stop_gradient. W and x therefore also receive gradient through μ_c(W, x) and v_c(W, x), via s_c (`:165`) and the offset −μ_c s_c (`:170`), and γ receives gradient wherever s_c appears. These paths are Chang's and unchanged. The direct-path formulas above are not the full gradient. §13 gates the full training-mode gradient against Chang's own layer.

**Which quantities depend on the moments (physics B6).** In training the batch moments give, and at export the moving moments give:
- FB's signs, because α = mean(W·s) mixes every column's s_c;
- both forms' magnitudes, through s_c.

BF's signs depend only on W and sign(γ_c), because sign(s_c) = sign(γ_c). The training-state and export-state kernels can therefore differ, as Chang's own kq(W·s) can. §3.3 fixes which state each measurement uses.

**Common to both forms.**
- Chang's raw-W moment path, statistics updates, bias quantizer and activation are kept. Moments from x·B (OPEN-1 C) are not adopted.
- α is global over the tensor presented to the binary map. A per-channel α would be a different experiment.
- In both forms the moments normalize a pre-activation, x·W, that the forward never computes, so d_c need not centre the binary output. Dead ReLU channels are logged (§12).

Sources: `advice/physics.md:22-51`; `advice/ml-engineer.md:20-50,118`; `OPEN-CHOICES.md:5-11`.

FB is described as "BitNet-inspired binary weights at Chang's quantizer site with channel scales", not as literal per-layer BitLinear (`advice/physics.md:1,58-62`; BitLinear: Wang et al., arXiv:2310.11453, physics C18).

**Sites.** The binarizable groups are:
- attention: the four nested projections Q/K/V/O (`mha.py:152-223`);
- FFN: the expansion, a fused-BN site (`model.py:199`), and the contraction (`:200`);
- hidden head: three fused-BN layers (`:205-207`).

That is nine kernels, four of them fused-BN. The embedding (`:192`, fused BN) and the classifier (`:208`) stay HGQ, except in BE.

The forms differ only at the four fused-BN sites. F000 and F100 are identical in both forms; F001, F010, F011, F101, F110 and F111 differ.

## 3. Cost accounting [D10, OPEN-2]

### 3.1 Quantities

**(a) Native HGQ2 EBOPs ("native").** This is the count at every HGQ site, meaning every layer with an HGQ kernel quantizer.
- Dense sites contract input widths with the kernel quantizer's `kq.bits_` and add the bias term. The pinned hgq2 0.1.9 applies no parallelization factor: its constructor marks the argument "not used yet" and its `_compute_ebops` returns the contraction plus the bias term (installed 0.1.9 `hgq/layers/core/einsum_dense.py:26,68-77`). The snapshot multiplies by parallelization_factor/n_parallel (`reference-code/HGQ2/src/hgq/layers/core/einsum_dense.py:62-64,85-93`), which is 1 for the released layers because the factor defaults to n_parallel. The counts follow the pinned 0.1.9 (chang C16).
- MHA adds the QK and AV products (`mha.py:386-413`). Its public `ebops` sums the projections, the softmax and its own counter (`mha.py:418-432`; `reference-code/HGQ2/src/hgq/layers/softmax.py:183-192`).
- QAdd uses 0.65× its summed-minus-narrowest widths (`reference-code/HGQ2/src/hgq/layers/ops/merge.py:18-30`).

CH-8's released loss uses native EBOPs unchanged.

**(a′) Native rule at a binary site [OD-13].** This is the same contraction with `kq.bits_` replaced by the scale quantizer's `bits_` broadcast over the input axis: per layer at non-BN sites, per output column at fused-BN sites. The bias term is native. The frozen evaluator computes it. The binary layer's own `_compute_ebops` returns the penalty (b) instead, because HGQ's loss mechanism adds what `_compute_ebops` returns and writes the same value to the training counter (`base.py:100-104`); under H1 the two are equal, under H2 they differ (critical C17).

For a binary arm, the total of native at its HGQ sites and the native rule at its binary sites is the **native-rule count, NR**. For CH-8 and F000, NR is native. NR is never called native HGQ2 EBOPs for a binary arm, because its binary layers are not HGQ2 layers (`docs/conventions/quantization-and-cost.md:23-25`).

**(b) Training penalty.** This is what the λ ramp squeezes.
- CH-8 and F000: native, unchanged.
- Binary arms: native at HGQ sites. At binary sites, the adopted H rule (§3.2) with the exact constant width w(q_g(g)) replaced by the scale quantizer's differentiable `bits_`, and input widths as native. Under H1 this equals the native rule (a′).
- Gradients reach the scale bits through `bits_` and the activation widths through the input `bits_`.
- Only two things are piecewise and detached: whether a column is zero (from the rounded scale bits) and which branch of H2's min is active.
- The penalty uses only layer variables (quantizer bits), never coefficient values. `_compute_ebops(shape)` (`reference-code/HGQ2/src/hgq/layers/core/base.py:94-104`) therefore computes it from layer state inside the trace, and no Python state outlives the trace (chang C7).
- It is added through HGQ's own loss mechanism.

**(c) Headline count H.** H is the front axis and is used for feasibility, selection, PID feedback and confirmation verdicts.
- It is H1 or H2 (§3.2) under §3.4 v0.4. The criteria are fixed before any training result; H2 adoption depends on the trained-model export check.
- It is computed without mutating the model, from moving-moment coefficients and saved quantizer parameters, by identical rules for CH-8 and the binary models.
- The other count and NR are always reported beside it.

**How (b) and (c) differ.** Both are declared differences that apply to every arm alike:
- learned bits against exact widths. This is the same relation as Chang's kq.bits_ against the exact width of his constants, so for the same quantizer state (c) ≤ (b) at every site;
- batch against moving moments, which affects only the exact width of q_g(g_c) and which columns are zero.

The ramp squeezes one cost while selection reads another (`advice/physics.md:97-101`).

### 3.2 Width rules [P]

**Coefficient width.** w(0) = 0. For q ≠ 0, w(q) = ⌊log2|q|⌋ − ν(q) + 1, where ν(q) is the exponent of the lowest set bit of |q|. Every float32 is an exact dyadic number, so ν(q) is defined.
- w(q) is the smallest KBI bit count that holds q exactly under SAT_SYM, sign excluded. That matches HGQ's `bits` under SAT_SYM (`fixed_point_quantizer.py:93-97`).
- **Grid exactness (chang C27).** w(·) assumes every quantizer output lies exactly on its dyadic grid. HGQ computes the step as `2.0**f` through the backend's pow (`quantizers` `_fixed_point_ops.py:142`); an inexact pow on the GPU would push values off the grid, and w would jump to about 24 bits (inference). Each value is therefore snapped to its quantizer's own f before w is taken, and the evaluator asserts that the snap changes nothing. The H1 ≤ NR gate also runs on the GPU path (§13).
- Check (derived): w(1) = 1 and w(0.375) = w(3·2⁻³) = 2.
- For any value on a b-bit SAT_SYM grid, w ≤ b. Hence H1 ≤ NR at every site of every arm, binary sites included.

**H1, the exported-constant count.** H1 is HGQ's contraction with `kq.bits_` replaced by w(K_ic) of each exported constant.
- It applies at every constant-kernel site of either model. Chang's HGQ kernels are read from their `qkernel` (`einsum_dense.py:106-108`; `einsum_dense_batchnorm.py:176-178`).
- At a binary site, w(K_ic) = w(q_g(g_c)) for every coefficient of column c, since S = ±1 does not change the width.
- Input widths, the bias term, the parallelization factor and all other contributions are native.

**H2, the sign-plus-scale count.** A column c of a constant kernel whose nonzero coefficients share one magnitude g_c is charged

min(H1_c, Σ_{t,i: K_ic≠0} b_{t,i} + Σ_t u_c(t)·w(g_c)),

where:
- b_{t,i} is the native input width at position t and input i;
- u_c(t) = ⌈log2 Σ_{i: K_ic≠0} 2^{I_{t,i}}⌉ + max_i F_{t,i} is an upper bound on the width, sign excluded, of the signed ±1 sum. It is computed from the input quantizer's integer bits I and fractional bits F, and ignores the sign pattern and unsigned inputs (critical C6). It is a bound, not the exact accumulator format the §3.4 test uses (`preflight/da4ml-test/scripts/common.py:152-173`);
- the gradient of u_c reaches only max F, because the datalane WRAP integer bits are tracked, not trained (`fixed_point_quantizer.py:381`);
- inputs of zero width are excluded from both terms;
- a zero column costs 0.

Any other column is charged H1_c. A Chang column with a single magnitude gets the same treatment (`advice/physics.md:84`). Everything else is as in H1.

**Per-position widths (chang C5, C19).** All three counts use the per-position input widths b_{t,i}, as HGQ does. The hls4ml DA route builds one constant-matrix multiply per in-place index, an index common to both operands and the output (hls4ml 1.3.0 `utils/einsum_utils.py:153,155`: `_common` and `_inplace = _common & s_out`; chang C29). `...c,cC->...C` and the MHA projections have none (I = 1), so the route builds one multiply at the maximum (k, i, f) over positions and calls it once per position (hls4ml 1.3.0 `backends/vivado/passes/distributed_arithmetic.py:81-91,381-397`). No count prices that widening; it appears only in LUT.

**Parallelization, dtype and rounding (critical C3, chang C8).**
- No parallelization factor enters any count, as in the pinned hgq2 0.1.9 (§3.1a; chang C16). The snapshot's factor is 1 for the released layers, so the two agree on this model; PREFLIGHT's cited-module diff (§7.3) confirms it.
- Every contribution that is native is taken from the layers' own `_compute_ebops` in float32, in the native order, and truncated per layer to uint32 as the native counter is (`base.py:102-103,197-203`). This covers input widths, the bias term, QAdd, softmax and tables.
- Only the kernel term at constant-kernel sites uses w(·) or the scale bits.
- With this rule the gate "reduces to native on F000" (§13) is exact. A float64 computation would differ: for example 0.65 × 180 truncates to 116 in float32 and 117 in float64.

### 3.3 State contract

| measurement | coefficient state | count | when | mutates? | used for |
| --- | --- | --- | --- | --- | --- |
| training penalty | batch moments, current step | (b), from quantizer bits | every step | updates counters as HGQ does | loss |
| logged training count | last step's counters (`reference-code/HGQ2/src/hgq/utils/sugar/ebops.py:7-15`) | native, or (b) | epoch end | no | log; Chang's ParetoFront and StopIf as released, which give the reference front |
| frozen evaluation | moving moments, current weights and quantizer parameters | H1, H2, NR | epoch end, after validation | no layer call in training or tracing mode, because WRAP integer bits move inside the quantizer call in those modes (`fixed_point_quantizer.py:277-300`; `minmax_trace.py:13-21` is the reset) | primary front, domain, feasibility, selection, PID sensor, screen and confirmation evaluation |
| checkpoint audit | reloaded saved checkpoint | same | PREFLIGHT, VERIFY | no | must equal the frozen evaluation exactly |
| export audit | after Chang's trace on train, then validation, `no_growth=False` as released (`reference-code/HGQ2-examples/jsc150/run_test.py:31-33,73`; `test_utils.py:15-34`) | same, from exported constants and formats | export | the trace changes WRAP integer bits | must equal the frozen evaluation of the same post-trace state; the pre/post-trace change is reported |

Three consequences follow:
- **Declared selection departure.** The primary front uses frozen H, whereas Chang's callback front uses the training count. Chang's callback front is kept and reported for every arm (`review/critical-v0.1.md:65`).
- **Equality requirement.** Equality is required only between measurements of the same frozen state. Training-state and moving-state coefficients are never required to agree.
- **State of held-out evaluation.** §8.5 fixes which state the held-out evaluation and the LUT pairing use.

### 3.4 PREFLIGHT da4ml test and the headline rule [D10]

**Sequencing amendment (Kai, 2026-10-18; v0.11).** Training runs on NRP. No mulder export gate, H-to-LUT calibration or clean da4ml rerun is a launch precondition for CH-8 and F000; binary arms reach the export gates through the v0.4 check (§14), and the clean rerun is required of no arm. CH-8 and F000 are independent of the binary penalty choice and train first, subject to the remaining §14 gates. The later permission for a whole-trained-model export check qualifies H2 under v0.4 below; deciding-point synthesis follows training and needs the later permission of §19.2 item 6. The matrix test and Addendum A below are the historical v0.3 path.

**Historical matrix test (mulder; v0.3 required it before training).**
- **Matrices.** The test uses frozen matrices K = S·diag(g), S ∈ {±1}, at the binary site shapes 24→32, 32→32 and 32→24, one position each (`stage_b_build.py`).
- **Scale sets.** They include non-power-of-two values of 3, 5, 8, 12 and 24 significant bits, powers of two, negative values, zeros, uniform per-layer scales and mixed columns (`scripts/common.py:13,26-61`).
- **Inputs.** Homogeneous and heterogeneous per-input (k, i, f) are tested.
- **Paths.**
  - P-stock: stock hls4ml 1.3.0 with the route's da4ml version (0.5.2, or 0.6.0 on a 0.6.0 route, step 2), the versions in mulder's `m3-env` (`campaigns/2026-10-12-mha-bitnet/PREFLIGHT.md:68`) or in its clone (Addendum A; chang C38), given the constant kernel as exported. That environment runs Keras 3.15.0 on TensorFlow (`env.sh:3,9`), which differs from the export environment of §9 in Keras and backend, not in hls4ml or da4ml (Addendum A, step 2).
  - P-fact: a ±1 EinsumDense followed by a per-column diagonal-scale EinsumDense, both through DA. The scale layer's input quantizer is the exact per-channel accumulator format, which never rounds or wraps (`stage_b_build.py:4-8,141-147`); "no intermediate quantizer" in the frozen block means no rounding quantizer between the two layers, and the hashed script governs (critical C21).
- **Stages.** Each line goes through, in order:
  1. C-simulation against exact integer arithmetic, on random grid inputs plus per-column extreme corners;
  2. the da4ml solver cost (`reference-code/hls4ml/hls4ml/backends/vivado/passes/distributed_arithmetic.py:116-150,362-409`);
  3. C-synthesis (Vitis HLS 2023.2, `xcvu13p-flga2577-2-e`, 2.5 ns, `io_parallel`, RF=1);
  4. Vivado `synth_design` plus `opt_design` out of context on xczu7ev.

**Context runs, outside the rule.**
- the four Chang-style 4-bit `hgq4` lines;
- the per-position variant `stage_b_build_pos.py`, with 8 positions, which records the input widths DA actually used when inputs differ per position (chang C5);
- the calibration cases below.

**H-to-LUT calibration (physics B4; reporting requirement, no longer a launch precondition).** PREFLIGHT reports, from the post-opt reports, LUT per unit of H1, H2 and native or native rule for:
- the binary `stock` lines;
- the `hgq4` lines;
- two added calibration cases per shape, both Chang-style matrices on the SAT_SYM grid with i = 0, seeded through `common.rng`, zero width giving a zero weight:
  - uniform: per-element widths drawn uniformly from 0 to 8 bits, so a zero width has probability 1/9;
  - zero-heavy: half the widths zero, the rest drawn uniformly from 1 to 8 bits, because trained HGQ kernels at tight budgets are likely zero-heavy (physics C12).

  Their generators are written and hashed before they run. Both are context; the deciding-point ratios (§9) remain the realistic measurement.

The two ratio sets, binary and Chang-style, are recorded with the export evidence when available; they do not delay training (Kai, 2026-10-18). They are reported beside every H-front comparison (§8.6–8.7). The same ratios are computed at the deciding points (§9).

**Status of the test files when the v0.3 rule was written.** This is from directory listings only; no file under `results/`, `reports/` or `logs/` was opened by the author of v0.3.
- **`results/`:**

  | file | size (bytes) | time |
  | --- | ---: | --- |
  | `smoke_a.jsonl` | 22,263 | 05:17:20 |
  | `stage_a_24x32.jsonl` | 183,709 | 05:17:20 |
  | `stage_a_32x24.jsonl` | 156,644 | 05:17:20 |
  | `stage_a_32x32.jsonl` | 183,749 | 05:17:20 |
  | `stage_a_summary.json` | 37,527 | 05:17:37 |
  | `stage_a_summary.md` | 8,093 | 05:17:37 |
  | `stage_bc_table.json` | 58,307 | 05:58:17 |
  | `stage_bc_table.md` | 5,037 | 05:58:17 |
  | `ebops_compare.json` | 5,703 | 05:58:50 |
  | `ebops_compare.md` | 1,334 | 05:58:50 |
  | `ebops_vs_hgq4.json` | 3,145 | 05:58:50 |
  | `ebops_vs_hgq4.md` | 2,010 | 05:58:50 |

- **`reports/`:** 42 case directories, all at 05:58:17. No directory existed for the pair `32x32 NP2w24 het`.
- **`logs/`:** three `stage_a_*.log` files at 05:17:21.

All times are 2026-10-17, local.

**History of the rule.**
- v0.2 was saved at 05:28:20 with stage-A files present and no hash of its rule.
- `stage_b_build.py` was edited at 05:26, before v0.2.
- The stage-B/C files above appeared after the three reviews and before v0.3.

The pre-result status of this rule therefore rests on the author's attestation. What limits what that attestation must carry: every v0.3 change to the rule makes H2 harder to adopt or is neutral, with one exception:
- the factored full-model export is dropped;
- the case list is pinned to every pair;
- timing is defined per line;
- the C-synthesis fallback is removed;
- the exception: a version provision is added, and a rerun on another da4ml version can move the outcome either way. It was requested by `review/chang-v0.2.md` B4, saved at 05:54, before the stage-B/C files existed at 05:58 (critical C20).

From the moment its sha256 is recorded in the change log, the block below is fixed. The result files' hashes are cited in Addendum B below the block, by file path and list hash, and the later runs are recorded there (v0.4; critical C28).

superseded by v0.4 (Kai, 2026-10-18)

<!-- headline-rule v0.3 begin -->
**Headline rule [P], v0.3.**

Terms:
- *Case list.* The 20 stock/fact pairs in `preflight/da4ml-test/scripts/cases_bc.txt` (sha256 `1ba9c416b1c4c6007ad07361bd95b1e902058e66039df89e4b5be1a8928a3191`). These are its 40 lines whose path field is `stock` or `fact`, including the bias pair. The four `hgq4` lines are context and do not enter this rule.
- *Build.* Each line is built by `stage_b_build.py` (sha256 `db49a4041ff1a9a041e9a3d9c87bded997ff3adc96844366941f981028ba83d6`) through `run_case.sh`, `synth_one.sh`, `vivado_ooc.tcl` and `clock.xdc`.
  - P-stock is one EinsumDense whose constant kernel S·diag(g) is exported as ordinary constants.
  - P-fact is a ±1 EinsumDense with no intermediate quantizer, followed by a per-column diagonal-scale EinsumDense, both through DA.
- *Bit-exact.* The line's compiled hls4ml C-simulation equals the exact integer reference (`common.py`, `exact_out`) on every test input of the line, with zero mismatches.
- *Decisive LUT.* The CLB LUT count in `post_opt_util.rpt`, from Vivado 2023.2 `synth_design -mode out_of_context` followed by `opt_design` on xczu7ev-ffvc1156-2-e.
- *Timing met.* WNS ≥ 0 in `post_opt_timing.rpt`, and II = 1 in the Vitis HLS 2023.2 C-synthesis report (xcvu13p-flga2577-2-e, 2.5 ns, `io_parallel`, RF=1).

H2 is the headline count if and only if all four conditions hold:
1. every `fact` line is bit-exact;
2. every `stock` line is bit-exact;
3. in every pair, the `stock` decisive LUT is no greater than the `fact` decisive LUT;
4. every `stock` line meets timing.

Otherwise H1 is the headline count.

Fail-closed provisions:
- A line that does not complete a stage (build, C-simulation, C-synthesis or Vivado), for any reason, fails every condition that stage enters. No C-synthesis estimate replaces a missing Vivado result. If Vivado is unavailable, the headline is H1.
- The explicit factored full-model export (route 2b of v0.2) is not attempted in this campaign.
- The rule applies to a test run on the hls4ml and da4ml versions of the route that will export the trained models. If that route changes version (§9, H-LUT option 0 needing da4ml 0.6.0), the whole case list is rerun on that version before any production training, and the rule is applied to the rerun. The earlier run is reported beside it.
- The outcome, with the per-line table and the report paths, is recorded in PREFLIGHT and in this file's change log before any production training run starts.
- Any change to this block after its hash was recorded goes to Kai.
<!-- headline-rule v0.3 end -->

<!-- headline-rule v0.4 begin -->
**Headline rule [P], v0.4 (implementing Kai's decision of 2026-10-18; recorded in STUDY v0.11, 2026-10-19; amended in v0.12).** Kai's decision gives the outline: the binary arms train under H2 once a whole trained model passes the factored-export check (bit-exact, timing met, fewer LUTs than the normal export), and under H1 if it fails; this replaces v0.3's condition 3 and its "route 2b not attempted" provision (`decisions.md` [dec 10-18 H2], the "Effect" recorded in the bullet "Kai (chat, verbatim)"). The operational terms below are owner decisions implementing it, labelled `[P]` and flagged for Kai (§19.3).

**Basis and timing.** The orchestrator's application of v0.3 to the existing OPEN-2 table gives H1 (`decisions.md` [dec 10-18 H2], "The orchestrator applied the frozen headline rule informally"). Recounted from `preflight/da4ml-test/results/stage_bc_table.json` (sha256 `70c0de764622c1ca84a5a6c3140a658ce34beee00e37f8ea80a479316935466f`, 47 rows) with the v0.3 conditions as worded above: condition 3 fails in 19 of the 20 case-list pairs, and in the per-position context pair (20 of the 21 pairs in the table); conditions 1, 2 and 4 hold on both scopes. Condition 3 holds only for `32x32 P2 het`, whose two post-opt LUT counts are equal. The 21st pair, `8x24x32 NP2w8 het`, is the 8-position variant built by `stage_b_build_pos.py`, a context run outside the rule (above); the decisions record's "20 of 21" counts it. This is the recorded informal application to that table, not a clean rerun of v0.3's frozen case list. Kai instead chose to secure H2 through a factored whole-model export. The choice is results-informed under `docs/methodology/06-review.md` §6.3.3, made after the OPEN-2 outcome was known.

**Provenance and timeline.** Kai fixed the three criteria in outline on 2026-10-18 (above). The owner's export plan, `campaigns/2026-10-16-chang-bitlinear/preflight/export/PLAN.md`, "Written 2026-10-18 … before any build" (`:3`), fixed the check's specification (`:44-53`, below) and its jet rows (`:35`). The pilot was then trained on the home PC on 2026-10-19, before this amendment was written; its selection files are stamped 01:20:59, 01:30:48 and 01:42:10 (`review/critical-v0.11-amendment.md` A3; [dec 10-19 split]; times not re-read by the v0.12 fixer). By 04:30 that day all five check exports were bit-exact on the 4,096 jets, so criterion 1 was met, and synthesis was running (`PLAN.md:67-73`). The timing and LUT outcomes were pending when v0.11 was written (05:01:56), and the orchestrator's split-form default (12:20:19Z, [dec 10-19 split]) and Kai's split-form rule below that supersedes it (16:28:12Z, [dec 10-19 both]) were recorded before any synthesis, timing or LUT outcome had been read by the orchestrator or Kai. The criteria were therefore fixed in outline and in specification before the pilot was trained and before any screen training result; the v0.11 wording of criterion 1 was written after criterion 1 was known to be met (see "Relation to v0.3").

**Routes and model.** The check runs on mulder on whole models trained briefly on the home PC through the campaign's runner, with checkpoint and frozen export state identified. They are an engineering pilot: not screen runs, they enter no front, and their numbers are not results. `PLAN.md:12` and `:44-48` fixed the pilot before the build as three models: F111-FB and F111-BF, each holding all nine binarizable kernels of its form (§2.3, "Sites"), and CH-8, exported by the normal route for reference. The orchestrator had described the pilot to Kai on 2026-10-18, before his decision, as "Take one model briefly trained on the home PC, export the whole model the factored way, and test it on mulder"; these are the orchestrator's chat words, not Kai's ([dec 10-19 split], bullet "Source for the pilot model"). The check compares two exports of the same model under the same environment and settings of §9:
- normal: each binary kernel exported as the ordinary constant matrix K = S·diag(q_g(g));
- factored: every binary layer exported as a ±1 layer followed by a per-column scale layer, including the fused-BN sites and nested MHA projections present in the model. The intermediate accumulator format is exact, with no rounding or wrapping between sign and scale; biases, nonlinearities and other quantizers retain the model's function;
- both routes: the generated DA kernel wrapper receives `#pragma HLS INLINE`, identically for every model, including CH-8 and F000 (§9). The comparison does not credit the scheduling patch as a gain from factoring.

**Check specification [P]** (`PLAN.md:44-53`, fixed before the build). Runner stage `gate`, seed 1, CH-8 initial tensors drawn on the GPU, XLA deterministic ops, and the H2 penalty branch for both F111 forms. The epoch count was fixed after a 2-epoch timing probe, target 150; the three models were trained for 150 epochs (`PLAN.md:68-69`). The exported state is the front checkpoint with the highest validation top-1, ties to the lowest headline cost, then the earliest. The 4,096 jets are rows 0–4095 of the reader's validation split, never the held-out file (`PLAN.md:35`). This checkpoint rule applies to the pilot only and does not change §8's selection rules.

**Pass criteria [P].** A form passes when all of the following hold for its F111 model (`PLAN.md:51-52`):
1. The compiled hls4ml C-simulation of each export is bit-exact, with zero mismatching output values against the same model's Keras inference forward, on the same 4,096 validation jets (rows 0–4095 above; count as in §9). An isolated layer or an initialized model does not establish this trained whole-model check. The trace and pre/post-trace state record of §9 is retained, and both routes compare the same state.
2. Timing is met: WNS ≥ 0 in `post_opt_timing.rpt` and II = 1 in the Vitis HLS 2023.2 C-synthesis report, for the factored route. The normal route's timing is recorded beside it. The v0.3 terms and §9 settings apply: C-synthesis on `xcvu13p-flga2577-2-e`, 2.5 ns, 27 % clock uncertainty, `io_parallel`, RF=1, fabric multiply binding; Vivado 2023.2 `synth_design -mode out_of_context` followed by `opt_design` on `xczu7ev-ffvc1156-2-e`, with a 2.5 ns clock. This is pre-place-and-route evidence, not timing closure.
3. The factored export has strictly fewer decisive LUTs than the normal export of the same model: CLB LUTs in `post_opt_util.rpt`, at the Vivado post-opt stage and part above. A C-synthesis estimate does not replace this comparison.

**Relation to v0.3 [P].** v0.3 conditions 1 and 2 become criterion 1, now on both routes of the same model; condition 3 is reversed into criterion 3; condition 4 moves from the normal (stock) route to the factored route (criterion 2; `PLAN.md:52`). The timing route is flagged for Kai, with the normal route's timing recorded beside it (§19.2 item 5, §19.3). The extension of criterion 1 to the normal route was written when the five check exports were already known to be bit-exact (`PLAN.md:67-71`), so it changes no outcome.

**Outcome and forms.** The check has an outcome when F111-FB and F111-BF each have a recorded pass or failure. A failure is a completed attempt that fails any criterion, or an attempt the owner records as impossible to complete, with the reason, reported to Kai before any binary launch. A missing or failed build, C-simulation, C-synthesis or Vivado result cannot give a pass, and a pass for one form is not evidence that the other passed.
- **Split-form rule (Kai, 2026-10-19T16:28:12Z, before the outcome).** Both forms pass: H2 is the headline count for every binary arm, F100 included. Both fail: H1 for every binary arm. One form passes: every binary arm in both forms trains and is read under H1, so the FB-BF comparison is on one count, and in addition the passing form's binary arms train under H2: F100, the six fused-BN configurations in that form, three seeds each, and the PID runs of its best configuration, 24 more runs (`decisions.md` [dec 10-19 both]; Kai, chat, verbatim: "cant we do both bro, i dong get why we ca just do btoh"). It was recorded before any synthesis, timing or LUT outcome of the check had been read, and it supersedes the orchestrator's 12:20:19Z default, H1 for every binary arm unless both forms pass ([dec 10-19 split]). It settles the per-form reading of the singular 2026-10-18 wording ("once a whole trained model passes", [dec 10-18 H2], "Effect"). In the one-pass case only, the added runs need about 154 GPU-h at the planning rate (derived: 24 × 6.416667 = 154.00), the screen becomes 75 runs and about 481.25 GPU-h, or 531.25 with the 50 GPU-h reserve (derived from §10's 327.25 and 377.25), and the cap rises from 400 to about 530 GPU-h (the decision's wording); §10's figures are unchanged and stand for the other two cases (§19.2 item 8). The decision does not say how those H2 runs enter §8's selection and reading; §8 is unchanged, and that use goes to Kai before they launch (§19.2 item 5). CH-8 and F000 are read on the same count as the binary arms they are compared with (§3.1(c)); their training penalty is native under either (§3.1(b)). A later change is recorded with its time, as results-informed if made after the outcome is read.
- **Admission.** Until the check has an outcome, no binary arm, F100 included, is admitted to any launch under either count. CH-8 and F000 launch independently.

**Sequence and record.** CH-8 and F000 train on NRP independently of this count choice. The check does not restore mulder or the clean matrix rerun as a precondition for their launch. No binary arm launches before the check has an outcome, and its count is then set by the rule above, on the PC pilot (F111 in each form, with CH-8 exported by the normal route for reference); no training penalty changes mid-run. The outcome record identifies the model and form, checkpoint and export-state hashes, validation-jet identities, converter and INLINE patch, environment and settings, C-simulation mismatches and the two timing and LUT reports. The earlier engineering evidence is cited beside it: `campaigns/2026-10-16-chang-bitlinear/preflight/da4ml-test/README.md` and `campaigns/2026-10-16-chang-bitlinear/preflight/da4ml-test/perpos-oneblock/README.md` ("Result" and "What this does not establish"). Those isolated-layer tests motivate the attempt and do not establish a whole-model pass.

**Other counts.** The orchestrator's stated default, unchanged by Kai, is no separate H1 campaign (`decisions.md` [dec 10-18 H2], "Orchestrator default stated to Kai"). Every run records H1, H2 and NR. A result supported under only one count remains accounting-dependent. Any change to these criteria requires Kai; training outcomes do not choose the criteria.
<!-- headline-rule v0.4 end -->

**Outcome of the v0.4 check (recorded 2026-10-20; STUDY v0.16, basis of the F111-BF failure corrected in v0.17).** This is the outcome record the rule asks for, placed after the v0.4 block, whose text and hash are unchanged. F111-FB fails criterion 2 by a completed attempt, so the both-pass case is excluded and, under the split-form rule above, H1 is the headline count for every binary arm, F100 included, whatever F111-BF's status; CH-8 and F000 are read on H1 with them (§3.1(c)). That much does not depend on F111-BF. Two things do: whether the check has an outcome, which gates binary admission (§14), and whether the one-pass branch is closed. F111-BF's failure rests on the owner record below, made after the outcome was read and so results-informed, and on Kai's confirmation, which is pending (see the F111-BF bullet). While that failure stands, the one-pass provisions do not arise: no H2 runs are added, the conditional cap note of §19.2 item 8 (a 75-run screen, 531.25 GPU-h with the reserve, a cap of about 530 GPU-h) does not apply, and §19.2 item 5 has no one-pass H2 runs to place in §8. Every run still records H1, H2 and NR ("Other counts" above). No rule, criterion, arm, run count, seed, budget or label changes. The sequence, from `campaigns/2026-10-16-chang-bitlinear/notes/LEDGER.md` (UTC, 2026-10-20): at 17:06Z the orchestrator noticed on mulder that no synthesis process remained and the `prj24` logs ended mid-compilation; in the entry "17:10Z" it wrote the anticipatory reading "once recorded: BF = FAIL (no C-synthesis at either cap)", before any evaluation; at 17:14Z the evaluation job (codex-export-eval) read the records and confirmed exit code 124 at both caps and FB's criterion-2 failure (`PREFLIGHT-rows.md` written 17:14Z); the entry "17:18Z UTC — EXPORT CHECK OUTCOME" followed; the owner record is dated 17:30Z; the chat report to Kai went out at about 17:36Z. The F111-BF classification was thus anticipated by the orchestrator at 17:10Z from process state and confirmed by the evaluation at 17:14Z, before the owner record, so it is results-informed and labelled so. FB's failure is read directly from a negative WNS. Every reading below is a pilot reading: engineering, not a result ("Routes and model" above), and not quotable.

- **F111-FB: fails criterion 2.** Criterion 1 holds: both routes are bit-exact, with zero mismatching values on the 4,096 jets. Criterion 2 fails: the factored route's post-opt WNS is −1.090 ns, with II = 1; the normal route, recorded beside it, has WNS −1.087 ns and II = 1. Criterion 3 holds: 394,351 factored against 402,252 normal CLB LUTs. A completed attempt that fails a criterion is a failure ("Outcome and forms" above). On both routes the worst path runs from `input_layer_ap_vld_preg_reg/C` through a BUFGCE to the clock-enable pin of a softmax `invert_table` register, with a maximum fan-out of 236,154 (factored) and 238,134 (normal) (`PREFLIGHT-rows.md:152`).
- **F111-BF: recorded as a failure by the owner record of 2026-10-20T17:30Z; no C-synthesis result.** Both routes are bit-exact, with zero mismatching values on the 4,096 jets. C-synthesis of both routes stopped at its time limit (exit code 124) in the first attempt (28,800 s, ended 2026-10-19 at 12:16 PDT) and in the attempt of record (86,400 s, ended 2026-10-20 at 09:29:54 PDT); Vivado never ran, so neither route has an II, WNS or LUT result. Owner decision B14-7a (`PREFLIGHT-rows.md:78-86`), written before any BF synthesis result was read, says: if a first attempt is stopped by its limit, the 24 h copy is the result of record and the (8 h) stop is reported; it is silent on a 24 h copy that is itself stopped, and its consequence line reads "no form's outcome is decided by an operator time limit". No restart was made (`PREFLIGHT-rows.md:75-86,135,139,148`; `PLAN.md`, "What stops a line"). A missing C-synthesis result cannot give a pass ("Outcome and forms" above) but does not by itself create a failure. At v0.16's sealing (17:25:10Z) BF had neither a completed attempt nor an owner record. The failure is recorded under the second clause of the v0.4 definition, an attempt the owner records as impossible to complete, with the reason, reported to Kai before any binary launch: the owner record of 2026-10-20T17:30Z (made in the orchestrator's session; `notes/LEDGER.md`, entry "2026-10-20 17:30Z UTC — BF owner record and Kai notification"; `notes/flags-for-kai.md` item 0b; the B14 rows are the export lane owner's, ml-engineer's decisions) reads: F111-BF is impossible to complete within the plan's synthesis budget (8 h at first, then the B14-7a escalation to 24 h; no larger budget exists in `PLAN.md` or §9); the trained CH-8 normal route timed out the same way. That record supersedes B14-7a's consequence line. It was made after the outcome was anticipated (17:10Z) and read (17:14Z), so the supersession and the failure classification are results-informed, recorded here at 17:30Z. It was reported to Kai in chat at about 17:36Z (`notes/LEDGER.md`, entry "2026-10-20 17:34Z UTC", written immediately before the report and attesting it; the "17:48Z" entry gives about 17:36Z; `flags-for-kai.md` item 0b is dated 17:30Z because it was written in anticipation of the report), with the default that the record stands and no resynthesis is made, and the alternative that Kai may choose, a 72 h BF attempt on mulder before the binary arms launch. Kai's answer is pending. The v0.3 fail-closed provision (a line that does not complete a stage fails every condition that stage enters) lies in the superseded v0.3 block and reaches the same conclusion directly; it is quoted as consistent, not as governing. Consequence: F111-BF is a recorded failure from 17:30Z. H1 for every binary arm does not depend on it. Binary admission and the one-pass branch (24 added H2 runs) depend on it: they rest on this record and on Kai's confirmation, or on his 72 h alternative, which could only add those 24 runs, since FB fails criterion 2.
- **Trained CH-8, normal route: context only; no criterion depends on it.** It is bit-exact on the 4,096 jets. Its C-synthesis also stopped at both limits, so for this pilot state the trained normal route did not complete C-synthesis within 24 h. This bears on deciding-point synthesis (§9) and is context for VERIFY and REPORT; it is not a stop.

Per-line table (run commit `4f828bb5a8df`, gate commit `0dc211ecbacd`; C-simulation on the 4,096 jets):

| run | form and route | C-sim mismatching values | C-synthesis | II | Vivado | post-opt WNS (ns) | post-opt CLB LUT (utilization) |
| --- | --- | ---: | --- | --- | --- | ---: | ---: |
| `chk-F111-FB-factored` | FB, factored | 0 | completed (rc=0) | 1 | completed (rc=0) | −1.090 | 394,351 (171.16 %) |
| `chk-F111-FB-normal` | FB, normal | 0 | completed (rc=0) | 1 | completed (rc=0) | −1.087 | 402,252 (174.59 %) |
| `chk-F111-BF-factored` | BF, factored | 0 | rc=124 at 28,800 s and at 86,400 s | none | not run | none | none |
| `chk-F111-BF-normal` | BF, normal | 0 | rc=124 at 28,800 s and at 86,400 s | none | not run | none | none |
| `chk-CH-8-normal` (context) | trained CH-8, normal | 0 | rc=124 at 28,800 s and at 86,400 s | none | not run | none | none |
| `init-CH-8-normal` (G70, context) | CH-8 at initialization, normal | 0 | completed (rc=0) | 1 | completed (rc=0) | −1.172 | 580,510 (251.96 %) |

Post-opt means Vivado 2023.2 `synth_design -mode out_of_context` followed by `opt_design` on `xczu7ev-ffvc1156-2-e`, before placement and routing. Utilization is against the 230,400 CLB LUTs of the "Available" column of each `post_opt_util.rpt` (line 35), reported as is under §9's part-capacity order; capacity is not a criterion of the check, and no placement or timing-closure claim is made. The driver log `driver-check.log` ends with `prj24 synth rc=0` for the three stopped lines; that is a status-capture error (`$(date -Is)` overwrites `$?`; `preflight/export/operator/eval-observation.json`), and each line's `STATUS` and `synth_result.json` record rc=124 and no C-synthesis result (`PREFLIGHT-rows.md:137`).

Record items ("Sequence and record" above):
- *Models and checkpoints* (PC pilot, seed 1, 150 epochs, the checkpoint rule above): F111-FB `epoch=38.keras`, sha256 `f732dac0c193034a492746ab863a8da71b57a8fc35e22fad769072504827e49f`; F111-BF `epoch=132.keras`, `c5e2b35236c45de4c37d25b07cd35095955e1dacf713327aafffdb56f3a35cd5`; CH-8 `epoch=100.keras`, `89461fb9db41e2b39d9c213aa3583487b7e727cab9a34599400af6deba7f8a64`.
- *Export states* (post-trace Keras files, sha256): FB factored `542dc9d6…9bc7`, FB normal `5abfc8f7…0c48`, BF factored `4ba23da3…025a`, BF normal `f6fc7c5e…cd61`, trained CH-8 `f4fc5148…799a`. The two routes of a form start from the same checkpoint and give identical logits on the 4,096 jets, in Keras and in C-simulation (FB `9c696760…54e4`, BF `6d0406e1…1ce4`).
- *Validation jets*: rows 0–4095 of the reader's validation split, array sha256 `20c450016fe1ebd3125020656c527805c833921a3d3dd8c218fecb45ff9b6618`, from `trainval.npz`, sha256 `e00eabd5de47b97ec508dce6d1a6eeda1661a02f378b757db9307acfc4d6669c`; never the held-out file.
- *Converter and INLINE patch*: `cbl/export/` at `4f828bb5a8df` (`convert.py` `e1b41a2fd347…`, `factored.py` `2fbc62de9ce5…`, `hlut.py` `485e84bf5bb0…`, the custom H-LUT, used because option 0 failed under da4ml 0.5.2 and 0.6.0, G63). The post-generation step `cbl/export/postgen.py`, sha256 `4b69c437780dde269b647525a7e67585cde8f96f9992702c45d4b88ef96fc559`, is applied to every project: INLINE in 20 of 20 DA kernel wrappers of each factored project and 11 of 11 of each normal project.
- *Environment*: mulder, `venv-da4ml052`, Python 3.12.15, Keras 3.14.0 on JAX 0.9.2 (CPU), hgq2 0.1.9, quantizers 1.2.2, hls4ml 1.3.0, da4ml 0.5.2, numpy 2.5.4; the `m3-env` freeze is identical before and after (`78b9452c…b329`, G61).
- *Settings*: as in criterion 2 above; synthesis by `gates/b14_synth.sh` (sha256 `2377d58a…0c0f`) with byte-identical copies of the frozen `vivado_ooc.tcl` (`70d269f4…`) and `clock.xdc` (`e4ec5851…`) of the v0.3 hash table; time limits of 8 h, then 24 h (B14-7a), for C-synthesis and 12 h for Vivado.
- *Earlier engineering evidence* ("Sequence and record" above), cited beside the outcome; both are isolated-layer tests that motivate the attempt and do not establish a whole-model pass: `campaigns/2026-10-16-chang-bitlinear/preflight/da4ml-test/README.md` (sections "Result", `:8`, and "What this does not establish", `:173`; sha256 `87709d1dd9eb2638a4b9fe365463c9e72a5336d84d332844a686209216649380`) and `campaigns/2026-10-16-chang-bitlinear/preflight/da4ml-test/perpos-oneblock/README.md` (sections "Result", `:9`, and "What this does not establish", `:192`; sha256 `e93d6b846d5e2b32ca698868c28d6e037994917609160c8d2427605ac393f479`). Both files exist at these paths.
- *Gates*: G61–G70 at `0dc211ecbacd` exit 0. The check exits 1: FB fails its WNS test; BF fails its II, WNS and LUT tests, which have no result.

Report paths, under `campaigns/2026-10-16-chang-bitlinear/preflight/`:
- the evaluation report `export/codex/codex-export-eval-report.md`; the PREFLIGHT rows `export/PREFLIGHT-rows.md` ("Owner decisions", B14-1 to B14-8; "Final export evaluation — 2026-10-20");
- the evidence root `export/mulder/4f828bb5a8df/<run>/`: `record.json`, `prj/STATUS`, `prj24/STATUS`, `synth_result.json` and `vitis_hls.log` of each attempt, and for the FB pair and G70 `prj/vivado/post_opt_timing.rpt`, `prj/vivado/post_opt_util.rpt` and `prj/myproject_prj/solution1/syn/report/csynth.xml`;
- the check `export/mulder/check/0dc211ecbacd/mul/` (`argv.json`, `stdout.txt`, `exit-code.txt`, `result.json`), and the gates `gates/G61/0dc211ecbacd/mul/` to `gates/G70/0dc211ecbacd/mul/`;
- the plan `export/PLAN.md`, steps 4 and 5 and "What stops a line".

**Historical addenda to the v0.3 rule (STUDY v0.4, outside the block).** These record the former procedure and its provenance. The v0.11 amendment and the v0.4 headline rule above supersede their launch sequencing and headline-adoption path; the historical text below is retained.

**Addendum A: the run the rule is applied to (critical B14; procedure revised in v0.5 for critical B17, chang B9, chang B7 and critical C35; safeguards added in v0.6 for critical C39 and chang C38) [P; historical v0.3 path].**

**Superseded path (Kai, 2026-10-18; v0.11).** The clean rerun, option-0 sequencing and requirement to record their outcome before production training below no longer govern launch or the headline count. Item 5 is resolved to the v0.4 whole-model check, with Kai's split-form rule and its remaining Kai items (timing route, deciding-point builds, use of the one-pass H2 runs) as stated there (v0.12). Retaining the procedure is not an instruction to run it.
- *Why an addendum is needed.* The block's Build term reads "Each line is built by `stage_b_build.py` (sha256 `db49a404…`)". That file was last changed at 05:26 and builds constant kernels with TRN rounding (`stage_b_build.py:45-48`). The existing run does not show that the term holds line by line:
  - `scripts/run_extra.sh` (06:23) describes a TRN rebuild of `32x32_NP2w8_het_stock`, which had been "built with kq RND", so at least one line was built by an earlier version (as quoted in `review/critical-v0.3.md` B14; this author did not open the file);
  - `run_case.sh` skips any line whose `synth_result.json` exists (`run_case.sh:10`), so re-invoking the case list does not rebuild such lines;
  - `case_info.json` records package versions but no script hash (`stage_b_build.py:192-209`; critical C28);
  - `scripts/cases_rerun.txt` (06:23) reran the pair `32x32 NP2w24 het`, and the block does not say whether a later attempt replaces a failed one (`review/critical-v0.3.md` B14).
- *Reading.* The rule is applied to a clean rerun of all 44 lines of `cases_bc.txt`. There is one clean rerun per route version, and the rule applies to the clean rerun on the da4ml version that exports the trained models (critical C35). No other run counts.
  1. **Order and version.** H-LUT option 0 (§9) runs first, in the export environment (§9), because its outcome fixes the da4ml version of the export route. That version is the lowest one under which option 0 passes, so 0.5.2 if it passes under both. If option 0 passes under neither, the custom H-LUT proceeds on hls4ml's declared range and the version is 0.5.2. The clean rerun then runs on that version, so one run serves both the Build term and the block's version provision. If the route's version changes after a clean rerun, a further clean rerun on the new version follows this same procedure before any production training. An example: the full-model export gates fail on 0.6.0 and the route returns to 0.5.2 with a custom H-LUT. The rule applies to the further rerun, and the earlier clean rerun is reported beside it as context.
  2. **Scripts and environment.** The executed copies on mulder (`$HOME/scratch/cbl1016-da4ml-test/scripts/`) equal the frozen hashes of the change log, and none of the frozen scripts is edited. `run_case.sh:5` and `synth_one.sh:5` source `env.sh` from that fixed path, so the file at that path is the one that counts; a copy elsewhere would never be read.
     - *da4ml 0.5.2 route.* The rerun runs in mulder's `m3-env` (hls4ml 1.3.0, da4ml 0.5.2), with `env.sh` `a1c095ef…` at that path. Neither is changed. A fault that needs a change to `m3-env` or `env.sh` takes the clone route of the 0.6.0 branch below, here with da4ml 0.5.2, with its own lock and hash (critical C39).
     - *da4ml 0.6.0 route.* `m3-env` is not modified. The clone is a new venv built from `m3-env`'s `pip freeze` plus da4ml 0.6.0, never a directory copy, and every install is run as `<clone>/bin/python -m pip`; a venv copied as a directory keeps console-script shebangs that point at `m3-env`'s interpreter, so a `pip` call in it could install into `m3-env` (chang C38, inference). The clone's package list is locked and hashed. The sha256 of `m3-env`'s `pip freeze` is recorded before and after, with the diff of the two freeze lists. `env.sh` `a1c095ef…` is moved byte-identical into the archive. In its place goes a copy that differs from it only in the `PY` line, which points at the clone, and that copy's sha256 is recorded before the first line starts. After the rerun the original is restored and its hash rechecked. The restore runs in an exit trap, and after any interruption, restoring and rechecking the original `env.sh` is the first action (critical C39, chang C38). The Build term does not name `env.sh`, so the block is untouched (critical B17, chang B7).
     - *A declared difference.* In both branches the rerun's Keras 3.15.0 and TensorFlow backend differ from the export environment's Keras 3.14.0 and JAX (§9). The block's version provision names only hls4ml and da4ml, which are equal in the two. The matrix test does not reach the Keras-3.15 failures of §9: `stage_b_build.py:33` imports only `QEinsumDense`, and hls4ml's EinsumDense handler reads the equation, kernel and bias (hls4ml 1.3.0 `converters/keras_v3/einsum_dense.py:66-76`), not `full_output_shape` (chang B7).
  3. **Empty project directories, recreated (critical B17, chang B9).** `run_case.sh` writes each line to `$SCR/proj/main/<line>` (`:9`), its build log to `$SCR/logs/build_<line>.log` (`:12`) and its parse output to `$SCR/logs/parse.log` (`:16`), with `SCR` set by `env.sh:2`. It creates only `$SCR/proj/main` (`:11`). Under `set -uo pipefail` (`:4`), a missing `$SCR/logs` makes the build redirect fail, and the line exits at `:14` before synthesis. Before the rerun:
     - the existing `proj/main` and `logs` directories are renamed into an archive directory, and the archive's file list with sha256 is recorded; nothing is deleted;
     - the driver creates empty `$SCR/proj/main` and `$SCR/logs`, creates `$SCR/tmp` (`TMPDIR`, `env.sh:4`) if it is absent, and keeps `$SCR/numba_cache` (`env.sh:5`) in place with its file list recorded, since a compile cache affects time and not arithmetic;
     - **exclusivity (critical C39, chang C38).** From this step until `env.sh` is restored and rechecked, no other job sources `env.sh` (`env.sh:1`: "sourced by every cbl1016 job on mulder") or writes under `$SCR`. This is recorded. The rerun-environment equivalence check of §13 writes under a separate scratch root, never under `$SCR`, and is completed before this step starts;
     - no skip can occur while exclusivity holds (`run_case.sh:10`), the per-line assertions of step 5 check it, and no script changes.
  4. **The four context lines run first, as a driver check (critical B17, chang B9).** Before any of the 40 rule lines starts, the driver runs the four `hgq4` lines. After the first of them, it checks that `$SCR/logs/build_<line>.log` and `$SCR/proj/main/<line>` exist. A fault that stops a line from running a stage, such as a missing directory, an unset variable or a tool or licence path, is fixed in the driver or the environment while no rule line has started. In the 0.5.2 branch a fix that needs a change to `m3-env` or `env.sh` takes the clone route of step 2 (critical C39). A per-line assertion of step 5 that fails before any rule line has started is such a fault. The new hash or setting is recorded, the context lines' directories are moved into the archive, and the four lines run again. A stage outcome (a C-simulation mismatch, a LUT count, a timing result) is never treated as a fault.
  5. **Per-line provenance.** A new driver, written and hashed in PREFLIGHT before the run, calls `run_case.sh` unchanged for each line and writes a sidecar `provenance.json` for it. The sidecar holds:
     - the sha256 of the executed `stage_b_build.py`, `common.py`, `run_case.sh`, `synth_one.sh`, `vivado_ooc.tcl`, `clock.xdc`, `parse_reports.py` and `env.sh`, read just before the line starts and again after it ends;
     - the versions of hls4ml, da4ml, keras, hgq2, TensorFlow and numpy in the environment `PY` points at, `KERAS_BACKEND`, and the sha256 of that environment's `pip freeze` (chang B7);
     - the host, the start and end times, `run_case.sh`'s return code, and the build, C-synthesis and Vivado return codes it prints and writes to `STATUS` (`run_case.sh:13,15`; `synth_one.sh:9,13`).

     `stage_b_build.py` and `case_info.json` are not modified, because the Build term names the script by its hash.

     **Per-line assertions (critical C39, chang C38).** Before each line the driver asserts four things, and records them in the sidecar:
     - `$SCR/proj/main/<line>` does not exist;
     - `env.sh` has the branch's expected sha256 (`a1c095ef…` on the 0.5.2 route, the recorded copy's on a clone route);
     - the da4ml version under `PY` is the branch's;
     - `run_case.sh`'s stdout is recorded, so a "skip" (`run_case.sh:10`) is visible.

     **Remedy, recommended: void and restart** (critical C39, reconciled with chang C38 in `review/arbiter-v0.5.md` §6 item 1).
     - A mismatch before any rule line has started is a step-4 fault.
     - Once a rule line has started, a failed assertion or a "skip" voids the whole rerun, which restarts from step 3. The void run is reported as context, with its cause.
     - The void is mandatory. It is triggered only by these objective assertions, never by a stage outcome. Every void run stays on record. If no clean rerun can be completed, the block's fail-closed provision applies (H1).
     - Reason: failing a line for a foreign or environment fault would let an operator fault decide the headline toward H1, which is the defect critical B17 removed. Voiding keeps every rule line a function of the frozen scripts alone.
     - **Named alternative (chang C38(2)): fail the line.** A failed assertion after a rule line has started fails that line. It is more conservative and moves toward H1 on an operator fault.
     - The choice between the two is part of §19.2 item 5, which is Kai's alone.
  6. **One attempt per line.** A rule line that does not complete a stage, for any reason, fails every condition that stage enters, as the block says. No rule line is retried, and no line of an earlier run substitutes for it. A voided run (step 5) is discarded whole and reported as context. Its lines are not retried one by one, so "No rule line is retried" applies within the run the rule is applied to. If the clean rerun cannot be carried out at all (no mulder session, no Vivado), the block's fail-closed provision applies and the headline is H1.
  7. **Evidence and table (critical B17).** The rerun's evidence is collected under a new root, `preflight/da4ml-clean/<da4ml version>/`: `reports/`, `results/`, the sidecars, the driver log and the archive listing. The existing `preflight/da4ml-test/` tree is not written to. The per-line table has 44 rows, one per line of `cases_bc.txt`, with the four `hgq4` rows labelled context. It is made by a table script written and hashed before the run. The script reads `cases_bc.txt`, the sidecars and each line's stage artifacts, and marks a line without a stage result as failed at that stage and at every later one. `make_table.py` is not used: it reads only `reports/*/synth_result.json` (`make_table.py:14`), which a line that failed at build never writes, so such a line would drop out of the table.
  8. **Context.** The 05:58 run and the 06:23 runs are reported beside the clean rerun, labelled context.
  9. **Record.** The per-line table, the sidecars, the archive listing and the driver and table-script hashes go into PREFLIGHT and into this file's change log before any production training run starts.
- *Direction.* Relative to the existing run, the clean rerun can complete a line that failed there (toward H2), fail a line that completed (toward H1), and rebuild lines made by an earlier script version (either way). The alternative, the literal reading of the existing run, fails every line not shown to be built by `db49a404`; it is the more conservative reading. Whether any line can be shown to satisfy it depends on evidence outside `case_info.json`, which records no script hash. Because results exist, the choice between the readings is Kai's alone. Recommended: the clean rerun, the only run in which the Build term can be shown for every line.
- *When Kai answers (arbiter v0.5 §4).* Kai answers item 5 before the clean rerun starts, and the answer is recorded in `decisions.md` with its time. If the rerun has run first, his choice is recorded as informed by the results of both runs.
- *Cost and routing.* One repeat of the stage-B/C run on mulder per route version, 44 lines, each within `run_case.sh`'s 30-minute build limit and `synth_one.sh`'s two-hour C-synthesis and Vivado limits. Mulder time is outside the GPU budget. The rerun is the same §3.4 test, which the orchestrator records as covered by Kai's OPEN-2 choice (`decisions.md` [dec 10-17 delegation], "STUDY v0.2 … §19", item 3). If that reading is not accepted, it goes to Kai with §19.2 item 6.

**Addendum B: the blinding record, completed (critical B15; v0.5: critical C28–C30; v0.6: critical C43).**
- *Runs after the v0.3 listing (06:02), from directory metadata only.* At 06:23:40 four files appeared in `scripts/`: `cases_rerun.txt` (sha256 `15dc9b3c388fac478cb03d50733c0ee15126f863d503fb4df41937d2d6efecbf`), `run_extra.sh` (`5666fa869c32d13d489ff319bca32f29515008ac769f6c2e608612460e237c73`), `ebops_compare.py` (`4d3fa2aef14809f7f19a884954461aabc7cc7872fbe1aaccf7dddbac3af9e198`) and `MD5SUMS` (`06ab8eeeb4d8dfb216a7a212b22a5effa99480a9438ad0a5d58be096c76da5c5`). According to `review/critical-v0.3.md` B14 and B15 they ran the missing `32x32 NP2w24 het` pair, a TRN rebuild of one line and three per-position cases. `stat` of the directories themselves gives `reports/` 06:23:30, `logs/` 06:23:31 and `results/` 06:26:26; v0.4 gave 06:23 for all three, which was wrong for `results/` (critical C28). `README.md` (13,490 bytes) is 06:24:28; `perpos-oneblock/` was created at 06:34:54 and last changed at 06:51:40 as listed for v0.5 (it changed again at 10:36:50; see the v0.6 bullet below).
- *The outcome summary.* In `decisions.md` [dec 10-17 delegation], the bullet opening "STUDY v0.3 (experiment-designer", written at 06:33:56 after v0.3 was saved at 06:32:00, summarises the direction and size of the stage-B/C and per-position LUT outcomes (`review/critical-v0.3.md`, header and B15). A later bullet of the same entry, opening "Blinding slip (orchestrator, 2026-10-17)", is the orchestrator's account. The author of v0.4 read neither; only the first 48 characters of each bullet were displayed, in a truncated listing used to locate them and skip them.
- *Evidence hashes.* The orchestrator's evidence-hash list for the test directory is `preflight/da4ml-evidence-hashes-2026-10-17.txt` (114,871 bytes, 06:59), with list hash `88ffcea2b9696eb0f1e4cb786dcd26df96038125cf0976ed68807d53a782199c` as the orchestrator gave it. A sha256 over the list commits to every per-file hash in it, so the result files' hashes are cited here by file path and list hash rather than listed (critical C28).
  - **Orchestrator confirmation, 2026-10-17 (critical C29).** The file on disk hashes to `88ffcea2…199c`, it has been made read-only, and no file under `results/` is newer than it.
  - This author has not verified the confirmation: the file was neither opened nor hashed. `ls -l` shows mode `-r--r--r--`, 114,871 bytes, 06:59 (metadata only). The record of the confirmation in `decisions.md` is the orchestrator's.
- *Activity after v0.5 was saved, and PREFLIGHT activity before STUDY PASS (critical C43; v0.6).* The metadata below are this author's `stat` and `ls -l` readings; the descriptions, the list hash, the read-only status and who has seen what are the orchestrator's records in `decisions.md` [dec 10-17 delegation], bullet opening "STUDY v0.5 panel", and are not verified here.
  - **`preflight/da4ml-test/perpos-oneblock/`** last changed at 10:36:50, after STUDY v0.5 was saved at 10:35:40 (directory metadata). The orchestrator records that it was written by the per-particle test's local copy and is engineering evidence. It is context outside the rule, like the per-position runs of §3.4.
  - **A second evidence-hash file**, `preflight/da4ml-perpos-evidence-hashes-2026-10-17.txt`: 118,319 bytes, created 10:39:20, mode `-r--r--r--` (metadata). The orchestrator records its list hash as `f72106a7…c517c` and the file as read-only. As with the first file, the per-file hashes are cited by path and list hash. This author neither opened nor hashed it.
  - **Who has seen `perpos-oneblock/`.** The orchestrator records the orchestrator and the ml-engineer that ran the test, and no STUDY author or reviewer.
  - **`preflight/env/`**, created 07:44:19 and last changed 07:58:28 (directory metadata). The orchestrator records it as a pinned-stack and CH-8 smoke with initialization readings. It was started before STUDY PASS, against `docs/methodology/03-phases.md:11-12`, as engineering groundwork under Kai's overnight delegation, and no gate is claimed from it. The orchestrator records its readings as seen by the orchestrator and the ml-engineer only. That record's own bullet, "STUDY v0.5 panel", quotes one `preflight/env/` engineering reading, a per-epoch time on the home PC's GPU, so every reader of the bullet has seen that one number; the v0.6 author is one of them (v0.7, recorded at the orchestrator's request). The reading is not used anywhere in this STUDY and is not reproduced here. Under arbiter condition P3, PREFLIGHT either reproduces these readings under its recorded code sha or labels them as taken before STUDY PASS.
  - **The rule's inputs.** The orchestrator records that the directories the frozen headline rule depends on are unchanged. Directory metadata agree: `results/` 06:26:26, `reports/` 06:23:30, `logs/` 06:23:31 and `scripts/` 06:23:41, as in v0.5. The nine frozen scripts and `env.sh` recompute to the change-log hashes (16:57).
- *v0.11 exposure.* The amendment read the dated decisions through [dec 10-18 NRP] and both engineering READMEs cited in v0.4. It is results-aware on OPEN-2 and the per-position follow-up. The v0.3 attestation remains historical; v0.4 is explicitly results-informed, with its criteria fixed before any training result. The orchestrator's correction to the v0.4 block before review (`review/amendment-v0.11/NOTES.md`, "Orchestrator correction") was written after `PLAN.md`'s 04:30 status recorded the pilot trained and criterion 1 met (`review/critical-v0.11-amendment.md` A3, from the file times).
- *v0.12 exposure (export check).* The v0.11 critical reviewer is results-aware on criterion 1, from `preflight/export/PLAN.md:67-71`, and opened none of the check's synthesis outputs (`review/critical-v0.11-amendment.md`, header). The v0.12 fixer read that review, `preflight/export/PLAN.md`, `decisions.md` from its top through [dec 10-18 NRP], `preflight/da4ml-test/scripts/cases_bc.txt`, all of `preflight/da4ml-test/results/stage_bc_table.json` and the name, path, scale-set, INLINE, layout, II and post-opt WNS fields of every row of `preflight/da4ml-test/perpos-oneblock/results/table.json` (no LUT column). It is therefore results-aware on OPEN-2, on the per-position follow-up and on criterion 1 of the check. It opened nothing else under `preflight/export/`, listed no directory there, and does not know the check's timing or LUT outcome.
- *Who was blind.*
  - The v0.3 critical reviewer read the outcome summary and was results-aware (its header). The v0.3 physics and chang reviewers opened none of the result directories (their headers).
  - The author of v0.4 opened no file under `results/`, `reports/`, `logs/` or `perpos-oneblock/`, neither `README.md` nor the evidence-hash file. In `decisions.md` it read lines 1-58, 73-78 and 79-140 as the file stood before its change at 07:33, not lines 59-72, which held the two bullets named above (`review/snapshots/v0.4/plan.md`, "Inputs read"). **Orchestrator confirmation, 2026-10-17 (critical C30):** the bullet that author read at lines 73-77 is the "STUDY v0.3 reviews" bullet and carries no da4ml outcome. The v0.4 statement that its author was blind therefore stands.
  - The v0.4 critical reviewer read `decisions.md` lines 1-58 only and was blind to the outcome; the v0.4 physics and chang reviewers opened none of the result directories (their headers).
  - The author of v0.5 did not open `decisions.md`; every citation of it is carried from v0.4. It opened no file under `results/`, `reports/`, `logs/` or `perpos-oneblock/`, neither `README.md` nor the evidence-hash file, and listed none of those four directories. It took only the parent listings and `stat` metadata of the directories themselves (`review/snapshots/v0.5/plan.md`, "Blinding").
  - The v0.5 critical, physics and chang reviewers and the v0.5 arbiter were blind to the da4ml outcome, and none opened `perpos-oneblock/` or `preflight/env/` (their headers). The orchestrator's "STUDY v0.5 panel" bullet records the three reviews as blind.
  - The author of v0.6 read, in `decisions.md`, only the line numbers and first three characters of lines 100-145, to find the extent of one bullet. It then read that bullet only, lines 113-121 as the file stood at 11:36:09: the bullet opening "STUDY v0.5 panel", which the brief allowed. That bullet quotes one engineering reading from `preflight/env/`, a per-epoch training time on the home PC's GPU. This author has therefore seen that one reading, as has any reader of that bullet. It is not a da4ml outcome. It is not reproduced in this file, and nothing in v0.6 uses it: no run count, rate or GPU-hour figure of §10 changes. Otherwise this author opened no file under `results/`, `reports/`, `logs/`, `perpos-oneblock/` or `preflight/env/`, neither `README.md` nor either evidence-hash file (`review/snapshots/v0.6/plan.md`, "Blinding").
  - The v0.5 constructive reviewer (arbiter condition P4 (b)) read `decisions.md` lines 1-58. In locating that cutoff it displayed the first 24 characters of lines 59-80, one of which is a numeric fragment of the da4ml outcome summary. It declares itself partially results-aware on the direction of the da4ml test, and makes no recommendation touching the headline-rule block, Addenda A to C, the H1/H2 choice, the H-to-LUT calibration, C4, deciding-point synthesis or any LUT or firmware matter (`review/constructive-v0.5.md:17-21`). It read STUDY v0.5 only, not v0.6 (its header).
  - The author of v0.7 read `review/constructive-v0.5.md` in full, as briefed, and with it that fragment, which the review quotes at its line 20. This author is therefore partially results-aware on the direction of the da4ml test, as the constructive reviewer is. Consequence: v0.7 changes nothing in the headline-rule block, in Addenda A and C, in the H1/H2 choice, the calibration, C4, deciding-point synthesis or any LUT or firmware text. Its one addition near the rule, the interruption line under §19.2 item 5, lists options, keeps the text as it stands as the default and makes no recommendation. The fragment is not reproduced in this file. In this pass the author did not open `decisions.md` or any blinded file; `PREFLIGHT-PLAN.md`, which appeared in the campaign directory at 17:08, was not opened (`plan.md`, "Blinding").
- *Prompts.* Until the rule has been applied to the clean rerun and recorded, reviewer and arbiter prompts cite the [dec 10-17 delegation] entry only up to its bullet "STUDY v0.2 reviews", never the two bullets above (critical B15; an orchestrator item). A prompt that hands an agent the "STUDY v0.5 panel" bullet also hands it the one `preflight/env/` reading that bullet quotes (above). A prompt that hands an agent `review/constructive-v0.5.md` hands it the da4ml outcome fragment quoted at its line 20, and that agent is then results-aware on the excluded topics.
- *What this affects.* The rule's pre-result status rests on the v0.3 author's attestation, which is unaffected. A rule-related recommendation from a results-aware reviewer is weighed as such.

**Addendum C: a property of the frozen case list, for Kai (critical C19).** Seven of the twenty pairs carry 24-significant-bit scales, which q_g cannot produce, because `bc = MinMax(0,23)` caps its width at 23 bits: `cases_bc.txt` lines 11-12 (`32x32 NP2w24 het`), 17-18 (`MIX`, whose columns include NP2w24 values, `common.py:44-60`), 21-22 (`UNI24 het`), 26-27 (`NP2w24 hom`), 31-32 (`24x32 NP2w24`), 36-37 (`32x24 NP2w24`) and 41-42 (`32x24 UNI24`). Under the all-pairs conditions these pairs can decide the headline although no trained model exports such constants. The effect is conservative, toward H1. The block and the lines that count are unchanged, because results are known.

**Whatever the outcome:**
- Every selected checkpoint reports H1, H2 and NR. The count not adopted is labelled "not validated".
- A result under only one count is reported as accounting-dependent (`advice/physics.md:97`).
- The binary training penalty (b) uses the adopted rule.
- The grid and the 350,000 PID target are in units of H [OD-12]. They equal Chang's native 350k only if H equals native on his model. Native at CH-8's selections is reported beside them.

### 3.5 Per-site cost examples (derived, schematic)

Assumptions:
- homogeneous 4-bit inputs with I = 1 and F = 3, so u = ⌈log2(n_in·2)⌉ + 3 = 9 at n_in = 24 or 32;
- the kernel contraction term only;
- N = 8 positions for per-position sites, one pooled position for head sites;
- "4-bit learned kernel" is Chang's initial kernel width (b0 = 4).

Formulas, with P positions, n_in inputs, n_out columns and mean input width b̄:
- native = P·n_in·n_out·b̄·b_k;
- H1 = P·n_in·n_out·b̄·w;
- H2 per column = min(P·n_in·b̄·w, P·n_in·b̄ + P·u·w).

| site | native, 4-bit learned kernel | H1, v0.2 exact β (w = 24) | H1, q_g with w = 4 / 2 / 1 | H2, v0.2 exact β (w = 24) | H2, q_g with w = 4 / 2 / 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FFN expansion 24→32 | 98,304 | 589,824 | 98,304 / 49,152 / 24,576 | 79,872 | 33,792 / 29,184 / 24,576 |
| FFN contraction 32→24 | 98,304 | 589,824 | 98,304 / 49,152 / 24,576 | 66,048 | 31,488 / 28,032 / 24,576 |
| head 32→32, pooled | 16,384 | 98,304 | 16,384 / 8,192 / 4,096 | 11,008 | 5,248 / 4,672 / 4,096 |
| Q, K, V and O together | 393,216 | 2,359,296 | 393,216 / 196,608 / 98,304 | 305,664 | 132,864 / 115,584 / 98,304 |

What these numbers mean:
- **Under v0.2's exact β.** The width w(β) was typically 22–24 bits (physics A1: P(w ≥ 22) = 7/8 for random trailing bits), and the λ ramp could not shorten it. One 24→32 site exceeded the 350,000 grid point on its own when b̄ > 2.37 bits (derived: 350,000/(6,144·24)).
- **Under q_g.** The scale starts at w ≤ 4 and is trained by loss and penalty like any Chang kernel width. The same site exceeds 350,000 alone only when b̄ > 14.24 bits (derived: 350,000/(6,144·4)), as Chang's site does at 4-bit weights.
- **Under H1 and H2: the subset property (physics B9).** A binary site costs exactly what a Chang site costs when Chang's kernel holds the same constants, ±q_g(g) per layer or column; CH-8's per-value quantizer can hold them, and H1 and H2 read only the exported constants (§1). The binary arms therefore search a subset of CH-8's exported models at equal cost. A binary point is never unreachable for CH-8: an advantage, at lower cost for equal accuracy or at higher accuracy for equal cost, can only mean that the constrained training reached a point HGQ's per-weight training did not. A disadvantage can come from the lost per-weight zeros (§1).
- **The floor.** At w = 1 the binary site costs P·n_in·n_out·b̄ under both counts (24,576 = 24,576 at 24→32, because H2's min picks H1 there), which is the cost of a dense 1-bit HGQ kernel. A 1-bit SAT_SYM HGQ weight takes {−2⁻ᶠ, 0, +2⁻ᶠ} with its own exponent, so it is strictly more expressive at that cost. Only zero-width inputs or a zero scale take a binary site lower, so a deficit at tight allowances is what the cost definitions predict.
- **Under H2.** The scale term P·u·w stays per column and grows with ⌈log2 n_in⌉ as activations narrow (physics A1). At w = 4 it is 288 of a 1,056-unit column at the 24→32 site (derived).

PREFLIGHT records H1, H2 and NR per configuration at initialization, as engineering readings, before production (§14).

## 4. Data [D4, same-as-Chang]

**Preprocessing.** Chang's preparation script (`reference-code/HGQ2-examples/jsc150/tools/prepare_data.py:10-28`) writes `150c-train.h5` and `150c-test.h5`:
- features are cast to float16 at preparation (`:15`);
- labels are the argmax of `jets[:, -6:-1]`, with a one-hot assertion (`:12-14`).

Then `reference-code/HGQ2-examples/jsc150/data.py:7-43` applies, in order:
1. read file-order slots 0–7 as float16 (`:11`);
2. cast to float32 (`:17`);
3. take columns [5,8,11], which are pt, etarel and phirel (`:23`);
4. zero all three features where pT < 2 (`:25`);
5. compute per-feature mean and std in float32 NumPy over axes (0, 1) of the 8 retained slots of the whole training archive, padding included (`:28-29`);
6. standardize and cast to float16 (`:30,33`);
7. shuffle once with the legacy NumPy generator seeded 42 (`run_train.py:41`; `data.py:35-36`);
8. split 90/10 (`:40-42`).

The expected counts are 558,000 training and 62,000 validation jets (derived from 620,000 × 0.9/0.1), with integer labels for five classes. Sources: `reference-code/HGQ2-examples/jsc150/run_train.py:41-42,71`; `literature/jet-tagging-transformers/2510.24784_submicrosecond_transformers_jet_tagging_fpga.md:41-42`.

**Split identity.** Freeze one split independently of model seeds, together with the actual file order and jet identifiers. The preparation `glob` is unsorted (`reference-code/HGQ2-examples/jsc150/tools/prepare_data.py:19-28`), so seed 42 alone does not identify the split.
- No sorting, pT weighting, augmentation or masking is added.
- Normalizing over training plus validation is deliberate fidelity to the release.
- A zero or nonfinite std is an integrity failure.

**Held-out test set.** This is the separate 26-file, 260,000-jet archive that the dataset publishes as its validation archive. The lab calls it ROC-test; Chang's code calls it `150c-test.h5` (`DATASET.md:9-19`; `reference-code/HGQ2-examples/jsc150/tools/prepare_data.sh:16,21-22`). In this file "validation" always means the 10 % split, and the 260,000-jet archive is always the "held-out test set" (physics C1).
- It receives the pT gate and the training archive's shift and scale (`data.py:26,31`).
- It stays sealed until the final manifest is frozen (§8.5).
- Chang's `get_data` reads it eagerly (`data.py:13-15`). The I/O wrapper is split so that training touches only train and validation. This is a holdout-isolation departure, not a preprocessing change.
- The old N=8 arrays are not reused. They used a pT sort, split seed 1, train-only normalization and float32 (`campaigns/2026-10-12-mha-bitnet/STUDY.md:1233-1275`). Raw identities and integrity checks are reused.
- Manifests, hashes, class map, counts and disjointness are established in PREFLIGHT.

## 5. Arms and stages

**Codes.** Bit order is attention, FFN, hidden head. In the group letters, Q is Chang's learned-width HGQ group and B is the binary map of §2.2–2.3 (physics C1: "H" now means only the headline count). The suffix -FB or -BF names the form. All rows use N=8 and the common data and recipe unless a difference is stated. Tiers are the admission order (§10).

| id | configuration / sole intended difference | seeds | stage, tier | basis |
| --- | --- | --- | --- | --- |
| CH-8 | released builder verbatim, with the declared float32 matmul override and import shims and the Keras loss guard (§7.3), the seeding of §6 and the common I/O wrapper | 1,2,3 | S1, T0 | Q1–3, Q8–9, Q11–14 |
| F000 | Q,Q,Q: our reconstruction of Chang | 1,2,3 | S1, T1 | Q15; implementation control |
| F100 | B,Q,Q; no fused-BN site, forms identical | 1,2,3 | S1, T1 | Q4, Q15 |
| F001, F010, F011, F101, F110, F111, each -FB and -BF | both forms of the six configurations with a fused-BN binary site | 1,2,3; 36 runs | S1, T1 (pre-stage = factorial cells) | D9, Q15 |
| PID-CH | CH-8 under the lab PID at 350,000 H (§7.2) | 1,2,3 | S3, T1 | Q16 |
| PID-BEST | BEST (§8.4) in the chosen form, same PID | 1,2,3 | S3, T1 | Q16 |
| FP32 | Chang graph; quantizers and cost penalties off | 1,2,3 | T2; S3 under the §19.2 item 9 default, in which controls wait for S2 (critical C33); S1 under the 475 GPU-h alternative | Q11 |
| W8A8 | Chang graph; fixed eight-bit weights and activations [OD-6] | 1,2,3 | T3; S3 under the §19.2 item 9 default; S1 under the 475 GPU-h alternative | Q11 |
| DS-HGQ | lab ablation of CH-8 without the attention branch or the first residual; not Chang's Deep Sets | 1,2,3 | T4; S3 under the §19.2 item 9 default (critical C33); S1 under the 475 GPU-h alternative | Q11 |
| GR | F111 in the chosen form; scale derivative [OD-7] | 1,2,3 | S3, T5 | Q4, Q11 |
| BE | F111 in the chosen form plus a binary embedding in the same form | 1,2,3 | S3, T6 | Q11 |
| CH-8-C, BEST-C | confirmation: CH-8 and BEST in the chosen form, ramp | 4–8 (option A, as adopted); 4–11 (options B and C) | S4, requires Kai's budget approval and choice of option (§8.7) | Q12, Q16 |
| PID-CH-C, PID-BEST-C | confirmation: the two PID arms | 4–8 (A); 4–11 (B); not run in C | S4, as above | Q16 |

**Stages.**
- **S1.** Trains every screen row not tied to a choice. Under the §19.2 item 9 default no control trains in S1: the one control slot that fits at r=3 is held until S2 (§10).
- **S2.** Trains nothing. It fixes the domain (§8.2), then the form on validation (§8.3), then BEST (§8.4), then g* and D_band under option (b) of §19.2 item 12 (§8.2), and reports the validation outputs Kai's confirmation decision needs (§8.6). It also fills the held control slot by the item-9 rule, and any further slots the measured rates admit (§10).
- **S3.** Trains the screen rows that need those choices, including the controls admitted after S2 (critical C33).
- **S4.** The confirmation. It trains only if Kai approves the budget, after S2 and before any held-out read (§8.7).

**Pre-stage [D9; reuse is OD-11].** Kai's OPEN-1 decision trains both forms. The pre-stage runs each of the six fused-BN configurations in both forms, at the factorial's seeds (1, 2, 3), initialization and recipe: 36 runs, 231.00 GPU-h at r=3 (derived). The unchosen form accounts for 18 runs and 115.5 GPU-h of that. The form is chosen by the validation-only rule of §8.3, and the runs of the chosen form are the factorial's cells for those configurations; they are not rerun.
- A rerun would repeat the same computation.
- A rerun would add 18 runs, 115.5 GPU-h at r=3 (derived), and push the screen to 69 runs and 492.75 GPU-h with reserve, over the cap.
- The pre-stage is logically prior, since the form is fixed before BEST, but it runs concurrently with CH-8, F000 and F100.
- The unchosen form's runs stay in the ledger and in the screen's descriptive held-out table (§8.6).
- The reuse was adopted under the delegation [OD-11] (`decisions.md` [dec 10-17 delegation], "STUDY v0.2 … §19" item 4) and is flagged for Kai (§19).

**Control definitions.**
- **DS-HGQ.** Removes tanh 1, Q/K/V/O, softmax, A·V and add 1, so the second residual stream is the embedding output. It keeps tanh 2, the FFN, add 2, pooling and the head, and is trained independently.
  - It is a lab ablation, not Chang's Deep Sets (`get_gnn`, `model.py:77-104`; paper rows 9–12) (chang C13).
  - Sources: `campaigns/2026-10-12-mha-bitnet/STUDY.md:1510-1519`; `model.py:192-208`.
- **W8A8 [OD-6].** Eight total bits including sign, at per-value granularity, with trained steps.
  - It keeps Chang's rounding, overflow, bias (also eight-bit) and add placement. Accumulator widths stay derived.
  - Setting `b0=8` alone does not define it (`campaigns/2026-10-12-mha-bitnet/STUDY.md:1077-1088`).
- **GR [OD-7].** The main map with stop_gradient removed from the scale's input: K = q_g(g(T))·stop_gradient(S) + (T − stop_gradient(T)).
  - The forward is unchanged.
  - The added derivative is stop_gradient(S)·∂q_g/∂g·∂g/∂T_j, where ∂β/∂T_j = sign(T_j)/n over β's own reduction (per layer, or per column under FB), zero at T_j = 0, and ∂q_g/∂g is HGQ's straight-through input derivative. Under BF, ∂g_c/∂W_j = |s_c|·sign(W_j)/n.
  - The old GR also changed the centering derivative and is not copied (`campaigns/2026-10-12-mha-bitnet/STUDY.md:1304-1327`).
- **BE.** Binarizes the fused-BN embedding (`model.py:192`) in the chosen form, with its own scale quantizer. The classifier stays HGQ.

**Not applicable.** These earlier arms are excluded:
- LN-A8 (adds LayerNorm, against Q3);
- the A8/A6/A4 ladder and step freeze (against Q1/Q2);
- BK-A8 (redundant, since the scale is kept);
- dynamic activation γ;
- N=64 arms (outside Q13);
- old HGQ-8 (F000 and CH-8 cover it);
- a ternary {−1, 0, +1} control, which would isolate the pruning channel (§1). It is not run, because it is outside Kai's binary scope and the budget (physics B5);
- an HGQ kernel with one learned width per output column or per layer, which would separate the sign constraint from the shared width (§1, physics B9). It is not run, because Q8 fixes per-value granularity for every HGQ quantizer.

Sources: `campaigns/2026-10-12-mha-bitnet/STUDY.md:1305,1521-1539`; `decisions.md` [dec 10-16].

## 6. Seeds, pairing and resolving power [A1, OD-3, OD-15]

**Pairing.** Seeds are paired across all compared arms of a stage: seeds 1–3 in the screen; seeds 4–8 (option A) or 4–11 (options B and C) in the confirmation (§8.7). Pairing covers:
- latent initialization by canonical tensor role and shape, so both forms of a configuration start identically;
- BN initial state;
- the fixed split and the batch order;
- one GPU product and software stack per seed.

Sources: `docs/methodology/03-phases.md:79-82`; `docs/infrastructure/gpu-selection-policy.md:64-66`; `campaigns/2026-10-12-mha-bitnet/STUDY.md:1281-1306`.

**RNG flow (chang B2, B6).** The release seeds NumPy and Python `random` with 42 (`run_train.py:17-18,41-42`). Keras initializer seeds also come from Python `random`. The datasets are built before the model (`run_train.py:73-76`). During training two consumers draw from the module-level Python `random` stream, both every epoch:
- the hgq `Dataset`, which reshuffles its fixed-batch `_index_map` at each epoch end (`reference-code/HGQ2/src/hgq/utils/dataset.py:2-3,111-113`) and also draws when constructed (`:70,77`). Its start-offset draw at `:115-117` happens only with `drop_last=True`; Chang's training `Dataset` keeps the default `drop_last=False` (`run_train.py:73`; `dataset.py:12`), so it draws no offset (critical C34, chang C28);
- Keras's `PyDatasetAdapter`, which permutes the order of the batch indices every epoch with `random.shuffle` (Keras 3.14.0 `keras/src/trainers/data_adapters/py_dataset_adapter.py:4,257-261`). Chang's `fit` passes no `shuffle` (`run_train.py:104`) and Keras defaults to `shuffle=True` (`keras/src/backend/jax/trainer.py:372,413`).

The batch fetched at step j of epoch e is therefore `_index_map[π_e(j)]`: two permutations per epoch from one stream. `shuffle=True` is kept as released. For seed s:
1. `np.random.seed(42)`, then Chang's `get_data` as released. The split and normalization are fixed by 42.
2. Python `random.seed(s)` and `keras.utils.set_random_seed(s)`.
3. The datasets are constructed (`run_train.py:73-74`), then the model is built (`:76`), in the released order. CH-8's initial tensors are drawn here.
4. F000 and every binary arm copy CH-8's drawn initial tensors by role. Tensors without a CH-8 counterpart, which are the scale quantizers' b and i, take their constant initializers, with no random draw. Unmatched tensors are recorded.
5. Just before `fit`, every arm, CH-8 included, reseeds the module-level Python `random` stream, which feeds both consumers, with a seed derived from s.
   - This is a declared CH-8 departure. In the release the stream continues after the initializer draws, so arms with different draw counts would get different batch orders from epoch 1.
6. No lab callback and no step of the frozen evaluation draws from that stream between the reseed and the end of training, in any arm. In particular the frozen evaluation never constructs an hgq `Dataset` during training, because construction draws (`dataset.py:70,77`); it reads the validation arrays in a fixed order, or uses its own `random.Random` instance. A draw that cannot be avoided happens at the same points in every arm.
7. Gate: for every epoch, the sequence of mapped batch indices actually fetched, recorded as the index after the `_index_map` lookup in `Dataset.__getitem__` (`dataset.py:81-89`), is hashed, and the hash is identical across all arms of a seed. The `_index_map` hash is kept as a secondary record; two arms can share it and still differ in batch order (chang B6).
8. On resume, the persisted module-level `random` state (§13) restores both consumers, because both draw from it.

**Reporting.** For each comparison, report each seed's value and the mean, the sample sd (ddof=1), the paired difference interval and the count of seeds with a positive sign (`docs/conventions/jet-tagging-metrics.md:38-46`).

**Thresholds (observed s_d).** These figures are derived from the t distribution with df = n−1; none is measured. With s_d the sample sd of the paired seed differences, the two-sided half-width is t·s_d/√n. An interval clears zero when the observed mean exceeds the half-width, so each entry is a bar for the observed mean in units of the observed s_d. It is not a gain the design can detect; that is the power table below (physics B8).

| n | 95 % two-sided (t quantile 0.975) | 97.5 % two-sided (t quantile 0.9875; C2 and C3 share α) | 6 contrasts (option C labels) | 7 contrasts | 13 contrasts | 91 contrasts |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 2.484 s_d | 3.583 s_d | 6.285 s_d | 6.795 s_d | 9.283 s_d | 24.620 s_d |
| 5 | 1.242 s_d | 1.563 s_d | 2.169 s_d | 2.266 s_d | 2.690 s_d | 4.498 s_d |
| 8 | 0.836 s_d | 1.005 s_d | 1.285 s_d | 1.327 s_d | 1.499 s_d | 2.117 s_d |

**Power (true σ_d) (physics B8; derived).** σ_d is the true sd of the paired seed difference at one allowance. The model: normal paired differences, the same true gain δ at every allowance, and equicorrelation ρ between the allowances of one seed. C2's ABOVE needs every allowance of D_conf to clear its threshold, one-sided at the level of the C2 interval. Single-allowance power p(δ) is from the noncentral t (df = n − 1, noncentrality δ√n/σ_d, critical value t(1 − α, df), with α = 0.0125 at 97.5 % and 0.025 at 95 %) and gives the |D| = 1 column. At ρ = 0 the allowances of a seed are independent, so the joint power is exactly p(δ)^|D| and the ρ = 0 rows are its roots (closed form; scipy `stats.nct.sf`, `optimize.brentq`; v0.9). At ρ = 0.9 joint power is from 200,000 simulated confirmations per cell (NumPy `default_rng(12345)`), with the layout and root rule of PREFLIGHT owner decision B3b-1 (`preflight/gates/G53/1784511c0f34/PREFLIGHT-rows.md:140-143`; v0.9): one generator per seed count n; draws in order w (200,000 × n), then e (200,000 × n × 6); z = √ρ·w + √(1 − ρ)·e; allowance j of a |D|-allowance domain is column j < |D| (nested domains); the same draws for every ρ, level and |D|; at each allowance, from its own n values of z with mean m_j and sd s_j (ddof 1), crit_j = t(1 − α, df)·s_j/√n − m_j is the smallest δ/σ_d at which that allowance clears; the root is the ⌈0.8 × 200,000⌉-th smallest of max_{j<|D|} crit_j over the simulated confirmations. Entries are the true gain δ/σ_d that gives 80 % probability of ABOVE ON GRID with |D_conf| allowances.

| option (§8.7) | n, C2 level | ρ | \|D\| = 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A, as adopted | 5, 97.5 % | 0, exact (closed form) | 2.05 | 2.35 | 2.51 | 2.62 | 2.70 | 2.76 |
| A | 5, 97.5 % | 0.9, Monte Carlo | 2.05 | 2.19 | 2.26 | 2.31 | 2.35 | 2.38 |
| B | 8, 97.5 % | 0, exact (closed form) | 1.34 | 1.52 | 1.62 | 1.68 | 1.73 | 1.77 |
| B | 8, 97.5 % | 0.9, Monte Carlo | 1.34 | 1.42 | 1.47 | 1.49 | 1.51 | 1.53 |
| C | 8, 95 % (C2 alone) | 0, exact (closed form) | 1.16 | 1.33 | 1.42 | 1.48 | 1.52 | 1.56 |
| C | 8, 95 % (C2 alone) | 0.9, Monte Carlo | 1.16 | 1.23 | 1.27 | 1.30 | 1.32 | 1.33 |

The ρ = 0 rows are exact (closed form). The ρ = 0.9 rows are Monte Carlo, 200,000 confirmations per cell, with standard error about 0.0013–0.0023 σ_d at |D| ≥ 2 (their |D| = 1 entry is the noncentral t); the second decimal is uncertain by one unit (v0.9, PREFLIGHT gate G54 finding).

What the table says:
- A true gain equal to the threshold passes at one allowance with probability 0.566 (A), 0.536 (B) or 0.531 (C). At six allowances all pass jointly with probability 0.033, 0.133 or 0.363 under A at ρ = 0, 0.5 or 0.9; 0.024, 0.135, 0.350 under B; 0.022, 0.139, 0.351 under C.
- The σ_d that S2 can offer comes from three seeds, on validation. With df = 2 the 95 % interval of σ_d/s_d is [0.52, 6.28] (χ²), so a screen s_d can understate the gain needed by a factor of up to about six. Under its equal-variance assumption, the pooled σ̂_d of §8.6 narrows that factor to about 1.5 (df = 16; constructive K1, v0.7).
- At ρ = 0.5 the values lie between the two rows shown (for |D| = 6: 2.67 under A, 1.70 under B, 1.49 under C).
- Under option C together with option (c) of §19.2 item 12, C2 shares α with the ramp point claim C3r at 97.5 %. C's rows then equal B's (same n = 8, same level), and its threshold is 1.005 s_d.

**What the table assumes, and the bound (physics B12; derived).** The entries assume the same true gain at every allowance and BEST feasible in every seed, the most favourable case for a fixed mean gain (physics C22). For unequal gains, read the table at the smallest true gain over D_C2. That gives a lower bound on the joint power, and the single-allowance power at that gain gives an upper bound (physics C22).
- ABOVE ON GRID is an intersection, so its probability is at most the smallest single-allowance power.
- If the true gain is ≤ 0 at any allowance of the claim domain D_C2, that probability is at most the one-sided level of the C2 interval: 0.0125 under A and B, 0.025 under C (C alone; C with (c) as B) (central t at δ = 0). This holds whatever the option's size.
- A small positive gain at one allowance barely helps. At δ = 0.5 σ_d the single-allowance power is 0.077 (A), 0.143 (B) or 0.231 (C) (noncentral t).
- The design expects a deficit at tight allowances (§1, §3.5). That is why the claim domain is a decision for Kai before any launch (§19.2 item 12), and why S2 reports a plug-in power at BEST's observed per-allowance gains (§8.6).

**Sign-pattern bound (physics A2, derived).** With n paired seeds there are 2ⁿ sign patterns, so a distribution-free sign test reaches at best a one-sided p of 2⁻ⁿ and a two-sided p of 2¹⁻ⁿ:
- n = 3: 1/8 and 1/4;
- n = 5: 1/32 and 1/16;
- n = 8: 1/256 and 1/128.

At n = 3 no distribution-free check can support anything. At n = 5 the C2 interval's one-sided level of 0.0125 is below 1/32 = 0.031, so the verdicts rest on approximate normality of the seed differences, and this is stated with them. At n = 8 the bound 1/256 = 0.0039 is below both 0.0125 and 0.025, so the count of positive signs at each allowance is reported beside each interval as a distribution-free check: eight of eight gives one-sided p = 1/256, seven of eight gives 9/256 = 0.035. The verdict rule stays the interval.

**[OD-15] Screen, then confirmation.**
- **Screen.** Seeds 1–3 (S1–S3) are a screen. They choose the form and BEST on validation only and carry no verdict label. Their held-out table is descriptive (§8.6), because a whole-front verdict at n=3 would need gains of about 25 s_d (91 contrasts) or 9 s_d (13 contrasts).
- **Confirmation.** Fresh seeds of BEST and CH-8 (S4, §8.7) carry the verdicts. These seeds take no part in selection, so the confirmation needs no candidate correction. Kai chooses the option (§19.2 item 11):
  - A, as adopted: ramp and PID, seeds 4–8, 20 runs, 128.33 GPU-h at r=3. ABOVE needs an observed mean above 1.563 observed s_d at every allowance; 80 % power over six allowances needs a true gain of about 2.4–2.8 σ_d.
  - B: ramp and PID, seeds 4–11, 32 runs, 205.33 GPU-h. Threshold 1.005 s_d; 80 % power at about 1.5–1.8 σ_d.
  - C: ramp only, seeds 4–11, 16 runs, 102.67 GPU-h. C2 alone at α = 0.05, threshold 0.836 s_d; 80 % power at about 1.3–1.6 σ_d. C3 is not tested; the PID comparison stays a screen output. With option (c) of §19.2 item 12, C2 shares α with the ramp point claim C3r: threshold 1.005 s_d and power as under B.
  - Under every option the power holds only if the gain is positive at every allowance of D_C2; otherwise the bound above applies.
- **Timing of the decision.** Funding S4 and its option are Kai's decision. They are made after S2's validation outputs (§8.6) and before any held-out read. Seeds are never added after held-out signs are seen.
- **If S4 is not funded,** the campaign reports the screen and makes no confirmatory claim.
- **AUC** stays descriptive, since a confirmatory AUC gap below 0.005 needs eight seeds (`docs/conventions/jet-tagging-metrics.md:40`). Under options B and C the eight-seed count is met, but AUC is not a pre-registered claim here, so it stays descriptive.

## 7. Training schedule

### 7.1 Ramp [D5, Q1, Q2, Q16]

Copy `run_train.py`:
- Adam with defaults; sparse categorical cross-entropy from logits;
- batch 2,790 for training and validation; `steps_per_execution=4`;
- 7,000 epochs unless StopIf or an integrity or budget stop fires;
- learning rate 3e-3 with cosine restarts every 500 epochs, `t_mul=m_mul=1`, floor 1e-6, `alpha_steps=10`;
- `Dataset(shuffle=True)` reshuffles fixed batches each epoch, and Keras's adapter permutes their order each epoch (`fit` keeps its default `shuffle=True`); both draw from the stream reseeded in §6 (chang B6).

Sources: `run_train.py:67-104`; `reference-code/HGQ2/src/hgq/utils/dataset.py:66-113`.

λ follows `PieceWiseSchedule([(0,2e-8,'linear'),(2000,3e-7,'log'),(7000,3e-6,'constant')])` with the released callback order and semantics (`run_train.py:79-95`; `reference-code/HGQ2/src/hgq/utils/sugar/beta_scheduler.py:34-79`). The training penalty is native for CH-8 and F000 and (b) for binary arms (§3.1).

FP32 disables its cost callbacks and StopIf. W8A8 logs cost, but λ cannot change its fixed widths. Both keep the optimizer, learning rate and 7,000 epochs.

A frozen-evaluation and primary-front callback is appended after StopIf (§8.1). It writes only new log keys, so the released callbacks read exactly the released values. Exact full-precision metrics are recorded; selection never reads rounded filenames.

### 7.2 PID stage [D6, OD-5, OD-12]

**Amendment note under D6, 2026-10-17 [OD-12].** Q16 says "a PID run at 350,000 EBOPs" (`decisions.md` [dec 10-16], "Q16 decided by Kai"). This file reads it as 350,000 in the headline count H. The reading was adopted by the orchestrator under Kai's 2026-10-17 delegation (`decisions.md` [dec 10-17 delegation], "STUDY v0.2 … §19" item 6; "STUDY v0.2 reviews" (e)). It is a dated delegated reading, not a Kai decision, and is flagged for his review. Native at the PID-CH selection is reported beside it.

**Runs.** PID-CH is "CH-8 under the lab PID", not Chang's PID (chang C3):
- the release has no PID (`run_train.py:90`);
- the paper's PID sensed native EBOPs;
- here the stock `BetaPID` is modified in its sensor and its log key, and runs in H units.

After S2, PID-CH and PID-BEST train from the seed's fresh paired initialization for 7,000 epochs. Only the λ ramp is replaced, by `BetaPID(target=350,000 H, init_beta=2e-8)`.
- The sourced defaults are p=1, i=0.002, d=0, warmup 10, log=True, max_beta=∞, min_beta=0 and damping 0 (`reference-code/HGQ2/src/hgq/utils/sugar/beta_pid.py:95-106`).
- A positive λ₀ is needed: Chang's `beta0=0` with `init_beta=None` reaches log10(0) (`model.py:295`; `beta_pid.py:125-148`).
- Gains are not tuned, and warmup is not extended after a poor result.
- On resume, the integral, previous error, λ, epoch and sensed cost are persisted.

**Callback order.** The PID replaces BetaScheduler at its position: [FreeEBOPs, LR scheduler, PID, PBar, ParetoFront (released), StopIf], followed by the frozen-evaluation callback. Stock `BetaPID.on_epoch_end` overwrites `logs['ebops']` (`beta_pid.py:162-167`). Here the PID writes its sensed value under its own key instead, so ParetoFront and StopIf read FreeEBOPs's training count in ramp and PID runs alike.

**Sensor timing (chang C1).** Stock `BetaPID` caches its reading in `on_epoch_end` and acts on it at the next `on_epoch_begin` (`beta_pid.py:139,162-167`). The modified PID computes frozen H(e) itself in its own `on_epoch_end`, by calling the shared non-mutating frozen-evaluation function, and acts on it at `on_epoch_begin(e+1)`. The frozen-evaluation callback reuses the same cached value. Gate: sensed(e) equals frozen H(e) exactly.

**Aggregation.** For the released graph, the top-level public `ebops` read by the stock sensor already includes the nested MHA projections, softmax and table costs (`beta_pid.py:41-47`; `mha.py:418-432`; `softmax.py:183-192`). FreeEBOPs sums flattened private counters (`ebops.py:12-15`), and summing flattened public aggregates would double-count. The stock aggregation semantics are kept.
- PREFLIGHT gates the equality of the top-level public and flattened private totals on CH-8 and on every binary configuration.
- The callbacks receive Chang's functional model object itself, never a wrapper, because `get_ebops` iterates the top-level `model.layers` (`beta_pid.py:44-47`). Gate: the stock aggregate is > 0 after epoch 0 (chang C2).

**Sensor [P].** Both PID arms sense the frozen headline count H (§3.3), the same function selection uses. The stock aggregate is logged beside the sensed value every epoch. v0.2's clause "use the stock sensor if PREFLIGHT finds the two equal" is dropped: H1 < native on CH-8 whenever a learned width exceeds an exact constant width, so the clause would almost never apply (critical C7).

### 7.3 Runtime and software stack [OD-9]

- **Backend.** JAX as released (`run_train.py:3`), with matmul precision set to full float32 for every arm and reload check. The release uses `tensorfloat32` (`model.py:20-23`).
  - `model.py` sets its precision at import, so the override is applied after `import model`.
  - `jax.config.jax_default_matmul_precision` is logged for every run.
  - This one precision departure is declared (`docs/conventions/quantization-and-cost.md:43-45`).
- **Stack pin [P] (chang B1).** Keras 3.14.0 and hgq2 0.1.9 from PyPI under JAX. jax, jaxlib (CUDA build), `quantizers` and CUDA are pinned in PREFLIGHT to versions Keras 3.14.0 supports, and every package is content-hashed.
- **Two shims, declared relative to PyPI 0.1.9 (chang B1, C18).** Both symbols exist in the HGQ2 snapshot and are absent from the installed hgq2 0.1.9 (`~/lab/venvs/mhab1012-k314`): a recursive search finds no `StopIf` and no `QLinformerAttentionT` anywhere in its `hgq` package, `hgq/utils/sugar/__init__.py` exports no `StopIf`, and `hgq/layers/__init__.py` exports `QLinformerAttention` only (checked 2026-10-17; also `review/chang-v0.3.md` C18). Every other name imported by `run_train.py:13` and `model.py:4-18` exists in 0.1.9.
  1. `StopIf` (`run_train.py:13`) is vendored byte-identical from the snapshot's `reference-code/HGQ2/src/hgq/utils/sugar/stop_if.py`, which imports only Keras.
  2. The unused `QLinformerAttentionT` import (`model.py:13`; snapshot `reference-code/HGQ2/src/hgq/layers/attn/linformer.py:183`) is satisfied by a placeholder that raises if called.
  3. Binding: both names are set as attributes on the imported `hgq.utils.sugar` and `hgq.layers` modules before Chang's files are imported. Installed files and Chang's files stay unchanged, and their hashes and the shim file's hash are recorded.
- **Keras loss guard, declared departure.** The runner empties Keras's per-layer loss-id sets before every stateless call of its model and checks the number of loss terms against the model structure (B10-F1; `cbl/lossguard.py`; Kai, 2026-10-19). Keras 3.14 otherwise drops a layer's penalty term when a stale id is reused in a second or later training trace. It applies to every arm, CH-8 included, and to every gate child that builds a model; Keras's files are unchanged on disk.
- **Fallback.** If Keras 3.14.0 cannot be installed with a working JAX GPU stack, the fallback is Keras 3.15 with three further declared shims:
  - the fused-BN einsum-shape unpacking (0.1.9 `einsum_dense_batchnorm.py:103-104`);
  - `full_output_shape` (0.1.9 `core/einsum_dense.py:75`, `attn/mha.py:387-389`);
  - a restore of the Keras-3.14 Glorot fans for Q/K/V/O (`campaigns/2026-10-12-mha-bitnet/review/PREFLIGHT_chang_ch8_env_v1.md` §1, §3(e)).

  That fallback is a PREFLIGHT finding reported to Kai, never adopted silently.
- **Reference identity (chang C17).** The line citations in this file refer to the HGQ2 snapshot under `reference-code/HGQ2/` (about v0.1.9-17) unless marked "installed 0.1.9". PREFLIGHT records the snapshot's content hash and the diff of every cited module against the installed 0.1.9, and re-hashes both. The chang review's diff of the files the xfm model uses (`review/chang-v0.3.md` C17) is the starting point:
  - `layers/core/einsum_dense.py`: 0.1.9 has no parallelization factor (§3.1a);
  - `layers/core/base.py`: 0.1.9 wraps every QLayer's `build` and evaluates its `q<var>` properties right after it (`check_collapsed_weight`, installed 0.1.9 `base.py:118-140`), so a binary layer's `qkernel` must be computable at the end of `build` (a §13 gate);
  - `layers/attn/mha.py`: the snapshot only adds `QMultiHeadAttentionT`;
  - `quantizer/quantizer.py` (serialization), `minmax_trace.py` (a verbose print), `config/layer.py` (the snapshot adds `enable_sq`) and `utils/sugar` (`StopIf`) differ;
  - identical: `einsum_dense_batchnorm.py`, `activation.py`, `softmax.py`, `ops/merge.py`, `beta_pid.py`, `beta_scheduler.py`, `ebops.py`, `pareto.py`, `dataset.py`, `quantizer/config.py`, `internal/fixed_point_quantizer.py` and `_dais_tracer/`.

  A behavioral difference in a cited mechanism is a PREFLIGHT finding.
- **Memory fraction.** The release's XLA memory fraction (`run_train.py:4`) is an operational setting chosen for the measured pack size.
- **Runtime on NRP (Kai, 2026-10-18 and 2026-10-19).** Route A2 installs the hashed JAX lock into a venv on `kai-data`, used by `run_handoff.py` bundle mode (§11). This resolves the runtime-image item (§19.2 item 7); a pushed campaign runtime image is no longer required. The old TensorFlow runner and its timings do not carry over.

## 8. Selection, metrics, screen outputs and confirmation verdicts [D7, OD-4, OD-15]

**Metrics [D7].** Validation selects and the held-out test set evaluates.
- Top-1 is argmax agreement. AUC is macro one-vs-rest over the five softmax outputs, reported with the five per-class AUCs.
- A trigger working point is reported beside them (constructive K6, v0.7): per class, one-vs-rest, the background rejection 1/ε_B at signal efficiency ε_S = 0.5 (`docs/conventions/jet-tagging-metrics.md:35-36`), on the held-out test set from the saved predictions after the freeze (§8.6). It is descriptive and is never read in selection.
- Every value states split, n, N, features, seed count, stage, checkpoint and its H, the other count and NR.
- Q14's top-1 selection deliberately overrides the convention's AUC selection (`docs/conventions/jet-tagging-metrics.md:25-34`; `decisions.md` [dec 10-16], "Q14 decided by Kai").

### 8.1 Primary front and step envelope

**Per-epoch record.** At every epoch end the frozen evaluation records validation top-1, H1, H2 and NR.

**Saved checkpoints.** A checkpoint is saved when it joins the nondominated set (higher accuracy, lower cost) under H. Descriptive secondary fronts track the other count and NR separately. Chang's filter applies in the same units: validation top-1 > 0.5 and cost < 500,000 (`run_train.py:83-89`).
- Ties go to equal accuracy at lower cost, then the earlier epoch, then the checkpoint hash.
- Two diagnostic checkpoints are kept: the best validation top-1 at H ≤ 350,000 ignoring the filter, and the minimum-H checkpoint. They never enter the front or BEST.
- Failed runs and empty fronts stay in the ledger.

**Structure record (physics B10; v0.5: critical B18, C38, chang B8, physics C17).** At every frozen checkpoint of every arm, CH-8 included, the frozen evaluation records, on the validation jets:
- the degenerate layers among Q, K, V, O and the FFN contraction, binary or HGQ, by the effect definition of §2.2, each with its route (kernel pruned, kernel rounded to zero, all-zero-width input quantizer, all-zero-width output quantizer, other);
- the zero columns per site, each with its route (pruned or rounded to zero);
- the fraction of inputs of Q, K, V, O and the FFN contraction that are zero-width at every position;
- per head, the fraction of zero-width entries of the softmax output quantizer;
- for a degenerate Q, per head, whether Q_t is identical across query positions t, and whether Q is 0 at every element (v0.6: critical C42, chang C35; §2.2).

The record is stored in the checkpoint's front sidecar and feeds the labels of §8.7 and §8.8.

**Step envelope.** The grid is G = {100,000; 200,000; 300,000; 350,000; 400,000; 450,000} H. At each g, take the front checkpoint with the highest validation top-1 and H ≤ g, under the same ties.
- This is an attainable step envelope, never an interpolation.
- Actual cost and slack are printed.
- No feasible checkpoint gives NA, never zero.

### 8.2 Evaluation domain [P; physics B3]

D_screen and D_conf are computed on validation from CH-8 alone and frozen before any binary result enters a choice. The claim domain below is the one exception, by a rule fixed now.
- D_screen = {g ∈ G : CH-8 has a feasible front checkpoint at g in each of seeds 1–3}.
- D_conf = D_screen ∩ {g : CH-8 has a feasible front checkpoint at g in every confirmation seed} (physics C8). D_conf therefore never holds an allowance at which BEST was not screened.
- **Empty domains (critical C23, physics C8).** If D_screen is empty, the screen outputs are INCOMPLETE: no form, no BEST, and Kai decides before any S3 or S4 launch. If D_conf is empty, C2 is INCOMPLETE.
- **Claim domain D_C2 (physics B12; Kai decides before any launch, §19.2 item 12).** Under option (a), D_C2 = D_conf. Under option (b), the recommended default, D_C2 = D_band, defined by a rule fixed now and applied in S2 on screen validation only:
  - g* = the smallest g ∈ D_screen such that, at every g′ ∈ D_screen with g′ ≥ g, BEST is feasible in all three screen seeds and BEST's mean paired validation top-1 gain over CH-8 is > 0;
  - D_band = {g ∈ D_conf : g ≥ g*};
  - if no g ∈ D_screen satisfies the condition, D_band is empty and C2 is `INCOMPLETE (no band)`;
  - g*, with the per-g means and feasibility behind it, is logged and frozen in S2 with the other S2 choices, before any S4 seed exists; D_band follows once D_conf is known.
  The threshold is chosen on the screen seeds, and the confirmation seeds take no part in it, exactly as for BEST (§6).
  g* is a three-seed sign filter. An allowance enters the band with probability about 0.50, 0.81 or 0.96 at a true gain of 0, 0.5 or 1.0 σ_d (derived, normal approximation, Φ(δ√3), with σ_d here the true sd of the paired validation gain at that allowance). Every nonempty band contains 450,000, so (b) gives a route only to a gain that extends to the top of the grid (physics C20). Two alternative rules, written in full, are in §19.3 for Kai; the default stays as written.
- |D_C2| is printed with C2. C2 is called a front result only when |D_C2| ≥ 2; with |D_C2| = 1 it is named a single-allowance comparison.
- At a g ∈ D where a candidate is infeasible in any seed, the candidate's contrast at g is COST INFEASIBLE, which counts as a deficit.
- At a g ∉ D where the candidate is feasible in every seed, a separately labelled cost-reach advantage is recorded descriptively.
- No domain is proposed or chosen after binary coverage is known, with one exception: D_band, by the rule above, if Kai chooses option (b) of §19.2 item 12 before any launch. No other rule, and no change to this one, is admitted after a run exists.

### 8.3 Form choice [D9; rule P]

The form is chosen on validation only, over the six fused-BN configurations, seeds 1–3 and g ∈ D_screen:
1. Count, for each form, the configurations feasible in all three seeds at every g ∈ D_screen. More such configurations wins.
2. On equal counts, including zero, compute Δ_form, the mean over cells (configuration, seed, g ∈ D_screen) feasible in both forms of top-1(FB) − top-1(BF). Δ_form > 0 chooses FB, and Δ_form < 0 chooses BF.
3. If Δ_form = 0 or no common cell exists, choose FB, the form at Chang's quantizer site and closest to his fold order (`advice/physics.md:1,58`).

The choice and its inputs are logged and frozen before BEST is computed.

### 8.4 BEST [P]

Among the seven nonzero configurations in the chosen form (F100 plus six):
- Rank first by coverage: the number of (seed, g ∈ D_screen) cells where the configuration is feasible. A missed cell is a deficit, never a removal.
- Then rank by the unweighted mean, over its feasible cells, of validation top-1 minus the paired CH-8 value.
- Break ties by lower mean actual H, then configuration id.
- F000 is excluded. BEST may score negative.

**Descriptive outputs before unblinding (physics B2).**
- each seed's rank of every configuration, by the same rule restricted to that seed;
- the number of seeds in which BEST ranks first;
- the paired validation gap between BEST and the runner-up, using per-seed means over D_screen, with its ordinary 95 % t interval;
- beside BEST's mean gain over CH-8, the minimum over g ∈ D_screen of its per-g mean paired validation gain, and the number of g with a positive mean. C2 needs a gain at every allowance, while the ranking uses the mean; the ranking rule is unchanged (physics C11).

The text says "the validation-selected configuration", never "the best combination".

### 8.5 Freeze, held-out read and the evaluated state [critical B8]

**Freeze before reading.** The following are frozen before one batch read of the held-out test set:
- D_screen and, if S4 ran, D_conf;
- the form choice and BEST;
- Kai's claim-domain choice (§19.2 item 12) and, under option (b), g* and D_band (§8.2);
- every candidate's grid checkpoint hashes in both forms, under H, under the other count and under NR (the descriptive fronts, critical C12);
- which controls were admitted after S2 (§10), the control selections and the PID selections (§8.6);
- each seed's §8.10 CH-8 context checkpoint: the released-callback front point with the highest validation top-1 at training-count native ≤ 350,000 (critical C26);
- the structure record of every frozen checkpoint (§8.1);
- Kai's S4 decision and option and, if funded, D_conf and the S4 selections.

All 13 screen candidates (F100, and six configurations × two forms) are evaluated at their frozen points. No held-out front is rebuilt, and nothing is selected on held-out data. Later plots use the saved predictions.

**Evaluated state.**
- Screen and confirmation metrics use the held-out top-1 of the frozen **pre-trace** checkpoints, which are the state selection used.
- The post-trace exported model's held-out top-1 and H are reported beside them, as the hardware-path model.
- The pre-trace accuracy is never assigned to a changed exported model. This sentence of v0.1 (`review/snapshots/v0.1/STUDY.md:128`) is restored.
- C4 pairs LUT with the post-trace model's accuracy.
- A deciding point whose post-trace H exceeds its allowance g is labelled "allowance exceeded after trace" and is not counted as feasible at g in any hardware claim.

### 8.6 Screen outputs (seeds 1–3) [OD-15]

The screen outputs are descriptive and carry no verdict label: no ABOVE, BELOW, CROSSING or LOCAL label is computed at n=3 (§6).

**Validation outputs, from S2, before Kai's S4 decision:**
- the form choice and its inputs;
- BEST and the descriptive outputs of §8.4, including its minimum per-g gain and its count of positive g;
- the validation paired sd s_d of BEST − CH-8 at each g ∈ D_screen, and beside it, for each confirmation option of §8.7, two different numbers (physics B8):
  - the threshold the observed mean must clear, in observed s_d (1.563 for A, 1.005 for B, 0.836 for C alone; C with (c) as B, 1.005);
  - the true gain that gives 80 % probability of ABOVE ON GRID over |D_screen| allowances, from the §6 power table at ρ = 0 and ρ = 0.9, stated at the screen s_d and at its upper 95 % bound, 6.28 s_d (df = 2);
- **at BEST's observed gains (physics B12)**, for each option:
  - the plug-in joint power: the probability of ABOVE ON GRID under the §6 model (normal paired differences, equicorrelation ρ = 0 and 0.9; at ρ = 0 the closed form, the product of the single-allowance noncentral-t powers, and at ρ > 0 the §6 simulation) with the true gain at each g set to BEST's observed mean validation gain there and σ_d to its observed s_d there, over D_screen and, under option (b) of §19.2 item 12, over the screen band {g ∈ D_screen : g ≥ g*} (D_band itself needs D_conf, which exists only once the confirmation seeds have trained);
  - beside it, the smallest single-allowance power over the same allowances, which bounds the joint power for any ρ;
  - both of these also with σ_d set to 6.28 × the observed s_d at each g, the upper 95 % bound of σ_d/s_d at df = 2 (§6), because at three seeds a small s_d can drive the plug-in power towards 1 (physics C22);
  - with each plug-in power, per seed, the number of distinct (BEST, CH-8) checkpoint pairs behind the allowances of the domain (constructive K9, v0.7). Adjacent allowances whose step envelopes pick the same pair give the same contrast, so the intersection then has fewer distinct terms than |D|, and the ρ = 0.9 rows of §6 are the closer ones;
  - the number of g ∈ D_screen at which BEST's mean validation gain is ≤ 0 or BEST is infeasible in some seed. An observed mean ≤ 0 on three seeds does not show that the true gain is ≤ 0, but the plug-in power over a domain that contains such an allowance is then at most 0.0125 (A, B) or 0.025 (C alone; C with (c) as B), and if the true gain there is ≤ 0 the same bound holds for the confirmation (§6);
- **pooled seed variance (constructive K1, v0.7).** At each g ∈ D_screen, a two-way table of validation top-1 at the step-envelope points, with one row per arm (CH-8 and the chosen-form cells F000 to F111) and one column per seed, keeping the arms feasible in all three seeds at g. The residual mean square after arm and seed effects gives σ̂_pool², with df = 2·(rows − 1), which is 16 when all nine rows are present, and σ̂_d = √2·σ̂_pool. Printed beside it:
  - the 95 % χ² band of σ_d/σ̂_d (0.745 to 1.522 at df = 16, derived);
  - each arm's own paired s_d against CH-8;
  - the seed mean square over the residual mean square, which shows how much the pairing removes.

  The thresholds and plug-in powers above are printed also at σ̂_d and at its upper bound. F000's row is left out (df = 14, band 0.732 to 1.577, derived) if the PREFLIGHT determinism record's (§13) F000 run equals its CH-8 runs bitwise; otherwise it enters (df = 16). This is fixed before S1 and not revisited on screen values (constructive v0.7 B1, v0.8). The unchosen form's rows are not pooled, because both forms of a configuration start from the same tensors and batches (§6), so their residuals are not independent and would overstate df. The pooled value assumes one residual variance for all arms, and binary arms may be noisier than CH-8. Where BEST's own paired s_d lies above the upper end of the pooled band, BEST's own value is the one §19.2 item 11 reads. The pooled value informs item 11 and enters no rule;
- under option (b): g*, the screen band {g ∈ D_screen : g ≥ g*} and the per-g means and feasibility behind them (§8.2);
- **labels and context (physics C15)**: BEST's validation gains and every plug-in power carry the label "optimistic: selected on these seeds' validation", since BEST is the maximum over seven configurations in a form itself chosen on the same seeds. The runner-up's mean gain and minimum per-g gain are printed beside BEST's. The validation set has 62,000 jets against the held-out set's 260,000, so its jet-level standard error is about 2.05 times larger (derived, √(260,000/62,000)). In that one respect the validation s_d tends to overstate the held-out σ_d, while selection makes the gains optimistic. Beside BEST's mean validation gain, the null selection reference is printed (constructive K2, v0.7): the median and 95th percentile of the selected candidate's mean paired validation gain when every candidate's true gain is 0. It is simulated with the exact §8.3–§8.4 rules (form, coverage and ranking), the pooled σ̂_d above, ρ = 0 and 0.9, and the §6 simulation settings. Feasibility is held at the observed screen pattern for every candidate, seed and g, since cost and not a zero accuracy gain decides it. The simulated unit is each arm's top-1 at (seed, g), CH-8 included, with residual sd σ̂_pool, so the shared-comparator correlation of the paired gains is built in. The two forms of a configuration are drawn with correlation ρ_f, set equal to ρ (0 and 0.9) (constructive v0.7 B2, v0.8). A screen gain inside this range is what selection alone produces, and the plug-in powers at BEST's observed gains are read against it;
- coverage of every configuration;
- per-seed dominance and the factorial effects on validation (below).

The PID pair trains in S3, after S2, so it is not among these outputs; its validation comparison is added when S3 ends, before the held-out read.

**Held-out outputs, after the freeze**, for each of the 13 candidates and the controls, at each g:
- each seed's paired gap against CH-8, the mean, the sd, the ordinary 95 % paired t interval and the sign count;
- the label "screen, descriptive; selected on the same seeds' validation, so the selected configuration's gap is optimistic";
- each arm's validation-to-held-out top-1 drop at its frozen points (physics C5);
- a paired bootstrap over held-out jets beside the seed intervals, as finite-test context. The binomial standard error of top-1 at 66.3 % on 260,000 jets is 0.093 percentage points per arm (derived);
- BEST − F000 beside BEST − CH-8, and F000 − CH-8 as the run-to-run floor (physics C6). F000 − CH-8 is a diagnostic only, not a binary effect. Its reading depends on the §13 determinism record (constructive K4, v0.7; v0.8, constructive v0.7 B1): if the record's three runs are bitwise identical, F000 − CH-8 is expected to be 0 and any difference is an implementation finding; if not, it measures run-to-run divergence and is reported as the null-arm paired s_d;
- per class, one-vs-rest, the background rejection 1/ε_B at signal efficiency ε_S = 0.5 (`docs/conventions/jet-tagging-metrics.md:35-36`), from the saved held-out predictions after the freeze, with the same per-seed, paired and sign-count reporting (constructive K6, v0.7). It is descriptive and is never read in selection;
- the same quantities under the non-headline count and under NR;
- per-seed dominance and the factorial effects at the frozen points (below).

**PID selection (critical B13, restored from v0.2).** In every PID run, screen and confirmation, the PID selection is the checkpoint with the highest validation top-1 among those with frozen H ≤ 350,000 that pass Chang's filter, with the ties of §8.1, reporting actual cost and slack. It never takes the closest over-budget point. If no checkpoint qualifies, the outcome is one of §8.8's filter outcomes. Controller trajectories (λ, sensed H, error, integral, clamp state) are reported separately.

**PID screen.** PID-BEST against PID-CH at their PID selections, on validation when S3 ends and on the held-out test set after the freeze, reported the same way, with the outcomes of §8.8 named.

**Dominance and the factorial (critical B13, restored from v0.2).** Both are descriptive at n = 3 and carry no verdict label. Each is computed on validation in S2 and on the held-out test set after the freeze, at the same frozen points.
- *Per-seed empirical Pareto dominance.* For each seed and candidate: every frozen CH-8 front point has a frozen candidate front point at no greater H and no lower top-1, with at least one strict improvement.
- *Factorial main effects and interactions.* In the chosen form the eight configurations F000 to F111 form a 2³ design over the attention, FFN and hidden-head groups, with F000 as the all-HGQ cell. At each (seed, g ∈ D_screen) where all eight cells are feasible, each main effect and each two- and three-way interaction is the mean top-1 over the four cells whose contrast sign is + minus the mean over the four whose sign is −. The effects are averaged over those (seed, g) cells, whose count is printed. The effect of each group on feasibility (feasible cells with the group binary minus with it HGQ) is reported beside. This answers Kai's "if the binaries don't matter, then don't apply it to a certain layer" (`decisions.md` [dec 10-16], Kai's verbatim message) descriptively; it does not change BEST.
  - **Intervals and the additive-model cell (constructive K3, v0.7).** Each of the seven effects is also printed at each g ∈ D_screen where all eight cells are feasible in all three seeds, with a randomized-block interval. The seed is the block and the cell-by-seed interaction is the error, σ̂_F² (df = 14; a symbol distinct from the pooled σ̂_pool above), so the half-width is t(0.975, df = 14)·σ̂_F·√(1/6), with t = 2.145. The interval from the three per-seed effects (df = 2) is printed beside it. Under the block model the first interval is about one quarter of the width of a three-seed paired interval on a single contrast (derived: 2.145·√(1/6) = 0.876 σ against 4.303·√(2/3) = 3.513 σ). The cell that the additive model ranks first, with each group binary where its main effect is positive, is printed beside BEST; if the two differ, the S2 outputs say so. All of this is descriptive, carries no verdict label and leaves BEST unchanged.

### 8.7 Confirmation [OD-15; test P; requires Kai's budget approval and choice of option]

**Options (physics B8).** Kai chooses one after S2's validation outputs and before any held-out read (§19.2 item 11). Thresholds are in observed s_d; the power each option has is in §6. That power assumes a positive gain at every allowance of D_C2: if the true gain is ≤ 0 at any one of them, the chance of `ABOVE ON GRID` is at most 0.0125 (A, B) or 0.025 (C alone; C with (c) as B) for every option (§6, physics B12). The claim domain D_C2 itself is fixed by Kai before any launch (§19.2 item 12, §8.2).

| option | arms | seeds | runs | GPU-h at r=3 (r=6) | C2 interval per g | C3 | descriptive labels |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| A, as adopted | BEST and CH-8, ramp and PID | 4–8 | 20 | 128.33 (256.67) | 97.5 % two-sided, shared α with C3; 1.563 s_d | tested at the same level | 7 contrasts, 2.266 s_d |
| B | BEST and CH-8, ramp and PID | 4–11 | 32 | 205.33 (410.67) | 97.5 %; 1.005 s_d | tested at the same level | 7 contrasts, 1.327 s_d |
| C | BEST and CH-8, ramp only | 4–11 | 16 | 102.67 (205.33) | 95 %, C2 alone; 0.836 s_d. With C3r (§19.2 item 12 option (c)): 97.5 %, shared α; 1.005 s_d | not tested; the screen's PID pair stays descriptive | 6 contrasts, 1.285 s_d |

- All options use the same recipe, data, pairing and RNG flow.
- BEST, the form and the grid are fixed by the screen before these seeds exist.
- The claim-scope protocol file frozen before any S4 launch (§13) is written for the chosen option and the chosen claim domain. `protocol-draft.json` carries option A, as adopted, with the claim domain of §19.2 item 12 written for both (a) and (b).
- Deciding points (§9): at most 70 (A), 112 (B) or 96 (C) (derived).

**Contrasts.** At each g ∈ D_conf, d_s(g) = top-1(BEST, s, g) − top-1(CH-8, s, g), on the held-out test set at the pre-trace frozen checkpoints; in options A and B, d_s(PID) likewise at the PID selections (§8.6). Every g ∈ D_conf is reported; the claim rests on g ∈ D_C2.

**Primary claims.** In options A and B, C2 and C3 share α = 0.05, split by Bonferroni over two hypotheses. In option C, C2 has α = 0.05 alone, unless option (c) of §19.2 item 12 is chosen, in which case C2 and C3r share α the same way. L denotes the C2 interval level: 97.5 % two-sided when C2 shares α (A, B, or C with (c)), 95 % for C2 alone (C).
- **C3r nested in C2 (physics C21; v0.6, text and an option for Kai only).** When 350,000 ∈ D_C2, C3r's null implies C2's null, so C3r could be tested at C2's own level with no added α, as the strong form is nested (physics C21). This variant is offered to Kai under (c), with C2 at 95 % and 0.836 s_d under option C. It applies only if 350,000 ∈ D_C2, which is fixed on validation before the held-out read. The pre-registered default remains the Bonferroni split.
- **C2, the front, as an intersection-union test over D_C2** (D_band under option (b), D_conf under option (a); §8.2).
  - `ABOVE ON GRID`: at every g ∈ D_C2, BEST is feasible in all n seeds, and the lower bound of the two-sided paired t interval at level L (df = n−1) is > 0.
  - Because the claim is an intersection, no correction across g is needed.
  - `BELOW ON GRID`: at every g ∈ D_C2, the upper bound is < 0 or BEST is COST INFEASIBLE.
  - |D_C2| is printed with the verdict, and beside it, per seed, the number of distinct (BEST, CH-8) checkpoint pairs behind its allowances (constructive K9, v0.7; §8.6); an empty D_C2 gives `INCOMPLETE` (§8.2). Under option (b) the allowances of D_conf below g* are printed beside, each as COST INFEASIBLE or with its interval, labelled outside the claim.
  - **Strong form, under option (b).** The same test over all of D_conf is reported beside as `ABOVE ON ALL OF D_conf` or not. It is nested in the band test (its rejection implies the band's), so reporting it adds no α.
- **C3, the PID at 350,000 H (options A and B).** `ABOVE` or `BELOW` from the same interval at level L. `COST INFEASIBLE`, `NO ACCEPTED CHECKPOINT` and `ACCURACY FILTER FAILURE` are as in §8.8.
- **C3r, the ramp front at 350,000 H (option C with option (c) of §19.2 item 12 only).** `ABOVE` or `BELOW` from the paired interval at level L at g = 350,000 on the ramp fronts, whether or not 350,000 is in D_C2. `COST INFEASIBLE` if BEST is infeasible there in any seed; `INCOMPLETE` if CH-8 is.
- **Otherwise.** `FLAT/UNRESOLVED`, which is not equivalence, or `INCOMPLETE` for missing fronts or seeds or a failed gate.

**Degenerate-layer labels (physics B10; v0.5: critical B18, chang B8).** If a BEST checkpoint behind a verdict, at a grid point g or at the PID point, has a degenerate layer, binary or HGQ, among Q, K, V, O and the FFN contraction, by the effect definition of §2.2 and in any seed, that verdict is labelled "with degenerate ⟨layer⟩ (⟨route⟩)", and the thesis wording "binary weights" is not used for it. A C2 verdict carries the label of every g it rests on. CH-8's degenerate layers at the same points are printed beside. A low-budget ABOVE that rests on a degenerate K, V or O is a statement about removing or freezing attention, not about binary weights. DS-HGQ is the control that would show this (screen seeds; descriptive for an S4 verdict; §19.2 item 9; critical C33).

**Descriptive labels, not claims.** `CROSSING`, `LOCAL ADVANTAGE` and `LOCAL DEFICIT` come from simultaneous Bonferroni intervals over 7 contrasts (six grid points plus the PID; options A and B) or 6 (option C). The same verdicts are also computed under the non-headline count; if they disagree, the result is labelled accounting-dependent. At n = 8 the count of positive signs at each g is printed beside each interval (§6).
- Definitions, restored from v0.2:463-465 (PREFLIGHT B3b-22; flagged for Kai), read on those simultaneous intervals at the grid points g ∈ D_conf, where a COST INFEASIBLE g counts as a deficit (§8.2):
  - `CROSSING`: at least one significantly positive and one significantly negative g.
  - `LOCAL ADVANTAGE`: some lower bounds > 0 and no upper bound < 0, but C2's `ABOVE ON GRID` fails.
  - `LOCAL DEFICIT`: the mirror case of LOCAL ADVANTAGE: some upper bounds < 0 and no lower bound > 0, but C2's `BELOW ON GRID` fails.

**C4, firmware dominance (physics B11; renamed in v0.5, critical C37, physics C16).** Written before training. C4 is evaluated only where C2 is `ABOVE ON GRID`, and at the PID point where C3 is `ABOVE` (A, B) or at 350,000 where C3r is `ABOVE` (C with (c)). For each such g:
- take the n paired differences LUT_BEST − LUT_CH-8 of the deciding pairs (§9), all at one part, stage and settings, with the paired post-trace held-out top-1 differences beside them;
- `LUT ADVANTAGE at g`: the upper bound of the two-sided paired t interval at level L is < 0;
- `NOT A LUT ADVANTAGE at g`: its lower bound is ≥ 0;
- `LUT UNRESOLVED at g` otherwise;
- `HARDWARE INCOMPLETE at g`: a missing or capacity-failed pair, a stage on mixed parts (§9), or a deciding point labelled "allowance exceeded after trace" (§8.5).

C4, "firmware dominance: at every g ∈ D_C2, BEST has lower LUT and higher post-trace held-out top-1 than CH-8", is `SUPPORTED` only if every g ∈ D_C2 is `LUT ADVANTAGE` and the post-trace held-out top-1 interval at level L has a lower bound > 0 at every g (an intersection again, so no correction across g). It is `BLOCKED` if any g is `NOT A LUT ADVANTAGE`, if any post-trace accuracy upper bound is < 0, or if the exported function changed (§9). Otherwise it is `UNRESOLVED` or `HARDWARE INCOMPLETE`. The PID point is labelled the same way and reported with C3.
- Every deciding pair, not only one with an H advantage, also carries the per-pair label "not a LUT advantage" when LUT_BEST ≥ LUT_CH-8 (descriptive). Derived: BEST's LUT/H exceeding CH-8's by more than the H gap, H_CH/H_BEST, is equivalent to LUT_BEST > LUT_CH.
- C4 retains the normal constant-matrix export comparison, now with the identical INLINE patch for every model (§9). The factored route is attempted for v0.4 H2 qualification (§3.4); this amendment does not change C4's statistical rule or credit its same-model export check as firmware dominance over CH-8.
- **What the name covers.** The rule tests dominance: a LUT interval wholly below zero and an accuracy interval wholly above zero. By the subset property (§1), BEST and CH-8 at a matched allowance come from one exported-model family, so equal LUT is a plausible case. Higher accuracy at equal LUT then gives `LUT UNRESOLVED` and C4 `UNRESOLVED`; that means dominance was not shown, not that the gain was lost in firmware. A LUT non-inferiority rule would credit that case but needs a margin fixed now, which nothing in the lab grounds; it stays the §19.3 alternative.

**Assumptions.** The intervals describe training-seed variation on the fixed held-out set. They assume independent, approximately normal seed pairs (§6, sign-pattern bound). A jet-level interval never substitutes for seed uncertainty.

**Not adopted.** Pooling the screen and confirmation seeds, 1–8 under option A or 1–11 under options B and C, with the 91-contrast family (critical C11, option two; 2.117 s_d at n = 8) is not adopted, because seeds 1–3 were used for selection (critical C36).

### 8.8 Filter outcomes and the collapse diagnostic [v0.1 review B4; thresholds OD-14]

**Filter outcomes at an allowance.** These are not collapse.
- `COST INFEASIBLE`: no checkpoint has H ≤ the allowance.
- `NO ACCEPTED CHECKPOINT / ACCURACY FILTER FAILURE`: feasible checkpoints exist, but none has validation top-1 > 0.5. Its diagnostic accuracy and cost are reported. The 0.5 threshold is Chang's acceptance threshold, not a collapse detector (`run_train.py:83-89`).

**Collapse diagnostic.** It is computed on validation only from the per-epoch frozen record. It is descriptive and is never a selection or stop rule. For each run:
- **Reference.** A_ref is the maximum validation top-1 over epochs [0, min(2,000, e_350)), the linear segment of the λ ramp before its logarithmic squeeze (`run_train.py:90`).
  - For a PID run, A_ref is taken from the ramp run with the same configuration, form and seed, which shares its initialization.
  - If e_350 = 0 the window is empty, and the run is labelled "no pre-squeeze reference".
- **Squeeze window.** e_350 is the first epoch with H ≤ 350,000. A_sq is the maximum validation top-1 over epochs in [e_350, e_350 + 1,000) with H ≤ 350,000: two LR cycles of 500 epochs (`run_train.py:93`).
  - If H never reaches 350,000, A_sq is NA and the run is labelled "not squeezed (cost infeasible)".
- **Classification.** COLLAPSE holds if A_ref − A_sq > δ, with δ = 0.05 top-1 [OD-14].
  - The later windows are the LR cycles [500m, 500(m+1)) that start at or after e_350 + 1,000.
  - A collapse is SUSTAINED if every later cycle's best accuracy at H ≤ 350,000 stays below A_ref − δ through the end of the run. Otherwise it is RECOVERED at the first cycle that does not.
  - **Scale of δ (constructive K8, v0.7).** δ = 0.05 is large on this model's scale: the paper's N=8 MHA and Deep Sets rows differ by 1.6 points (§16, 66.3 % and 64.7 %), so the label marks only severe failures. A_ref − A_sq minus CH-8's value on the same seed is printed per arm and per g beside the label. δ itself is unchanged [OD-14].
- **Grid table.** The same quantities are tabulated at each g.
- **Dwell time (physics C4).** The number of epochs each run spends in each grid interval of H is reported. One λ schedule meets very different penalty magnitudes, so arms pass the grid at different learning-rate phases.
- **Binary-specific loss.** The binary loss minus CH-8's, and the top-1 drop A_ref − A_sq minus CH-8's (constructive K8), on the same seed are reported. A checkpoint accepted early and followed by a sustained collapse is reported as "accepted, then collapsed".

**Reading a deficit or a collapse (descriptive; written before training; constructive K5, v0.7).** No mechanism is claimed from three seeds. The table fixes which logged quantity is read for which reading, so that the reading is not chosen after the numbers. The pilot-program labels (`campaigns/2026-10-05-pilot-program/STUDY.md:120-127`: H1 controller input, H3 attention starvation, H4 squeeze timing, H5 weight prunability) are labels only; no pilot result is used (the experiment-log entry for that program records its result as pending, `.claude/memory/experiment-log.md:1-7`).

| pattern | reading it points to | quantities read (§8.1, §8.8, §12) | pilot label |
| --- | --- | --- | --- |
| deficit at low allowances in every binary configuration, with CH-8's zero-width kernel share rising as g falls | lost pruning channel (§1) | pruning-channel log: CH-8's zero-width fraction and the native cost it removes, per g | pilot H5 |
| deficit only where a fused-BN site is binary, F100 near F000, or a large Δ_form | the fold (§2.3) | FB sign differences, batch against moving moments; dead channels; train-mode against frozen gap; Δ_form | none |
| a drop that begins at the epoch a non-BN binary scale reaches b = 0 | per-layer scale pruning: one (b, i) removes a whole projection at once, where HGQ's per-value widths remove weights one at a time (inference from §2.2, "Pruned", not measured) | per-epoch q_g b and i (§12); onset and route of the degenerate layer | pilot H3 if the site is Q, K or V |
| deficit or collapse with a degenerate K, V or O, or Δ_static not above zero | attention removal (§15) | structure record; Δ_attn, Δ_static, Δ_unif; DS-HGQ if it ran | pilot H3 |
| drops aligned with LR restarts (multiples of 500 epochs) or with e_350 | squeeze timing | dwell time; e_350; LR phase at e_350 | pilot H4 |
| a PID arm below its ramp arm at 350,000 H with the ramp healthy | the controller | sensed H, stock aggregate, λ, integral, clamp state | pilot H1 |
| deficit at every allowance in F100 and not in F010 or F001 | the sign map in attention | factorial attention main effect, with its block interval (§8.6) | none |

For every run with COLLAPSE, or with A_ref − A_sq larger than CH-8's on the same seed, one event line lists the epochs of: the first b = 0 at each binary scale, the onset of each degenerate layer, the nearest LR restart, and the λ segment (`run_train.py:90`). The table and the event lines are descriptive and enter no rule.

**Attention use, separately (physics C7, B10; revised in v0.5 for chang B8, physics C17 and critical C31, C32; v0.6: chang C35–C37, critical C42).** Computed on validation at BEST's and CH-8's frozen checkpoints at each g ∈ D_conf for claim C6 (S4 only, critical C24), and at the screen's frozen points descriptively. Each ablation changes one tensor of the frozen model and keeps every other layer, quantizer and constant as it is.
- Δ_attn (mean ablation): the validation top-1 of the checkpoint minus its validation top-1 with the attention branch output (`model.py:195-197`) replaced by its validation mean per position and channel.
- Δ_static (static-attention ablation, new in v0.5, chang B8): the validation top-1 minus the validation top-1 with each head's attention probabilities replaced by their validation mean per (head, query, key). The replacement is made after the softmax output quantizer and before the A·V einsum (`reference-code/HGQ2/src/hgq/layers/attn/mha.py:526-536`). The per-(head, query, key) validation mean is accumulated in float64 and cast back, so that the mean of a constant equals it exactly (chang C37). A fixed, input-independent pattern, uniform or not, gives Δ_static = 0.
- Δ_unif (zero-logit ablation, redefined in v0.5, chang B8, critical C31): the validation top-1 minus the validation top-1 with the attention logits set to 0 and passed through the model's own `QSoftmax` with all its quantizers live. Under Chang's per-value quantizers the result need not be exactly 1/8 per key, and its rows need not sum to exactly 1 (§2.2). A model whose K output is identically zero, as it is for a degenerate K by each named route of §2.2 (K has no bias), already has zero logits, so its Δ_unif is exactly 0.
- H_attn: the attention entropy in nats of the post-quantizer probabilities after each row is renormalized to sum to 1, averaged over validation jets, heads and query positions. Rows that are all zero are counted and reported separately and do not enter the average. Uniform attention over 8 positions gives log 8 = 2.0794 (derived).
- **Overflow under WRAP quantizers (chang C36; v0.6).** The ablations are evaluated outside training, where a WRAP datalane quantizer wraps a value outside its range (chang C36, citing `quantizers` 1.2.2 `_fixed_point_ops.py:186-187`), so an ablated tensor can overflow downstream and inflate a Δ. For each ablation, the validation entries where any downstream datalane WRAP quantizer receives a value outside its (k, i, f) range are counted. The downstream quantizers are:
  - O's input quantizer (`mha.py:293`, `enable_iq=True`), for Δ_static and Δ_unif;
  - the second tanh table's input quantizer (`reference-code/HGQ2-examples/jsc150/model.py:198`; `activation.py:106-107`), for Δ_attn.

  The count is reported beside each Δ, labelled "overflow-affected" when nonzero, with the unablated model's count as the baseline. For H_attn, the rows with any negative post-quantizer entry are counted separately. The C6 rule is unchanged, and a C6 statement resting on an overflow-affected Δ carries the label.
- Degenerate Q, K, V and O, with their routes, are read from the structure record (§8.1). A degenerate K is reported as "input-independent attention weights", a degenerate Q as "input-independent queries" (or "query-independent attention pooling" where Q_t is identical across t), a degenerate V as "fixed values", and a degenerate O as "constant branch" (§2.2). A degenerate Q equal to 0 at every element gives zero logits and is reported as for a degenerate K, "input-independent attention weights" (critical C42, chang C35).
- **"Used" (C6)** needs all five, in every case on the confirmation seeds:
  - the Δ_attn paired interval above zero;
  - the Δ_static paired interval above zero;
  - the Δ_unif paired interval above zero;
  - H_attn < log 8 in every seed, a consistency check rather than a test, since any attention that is not exactly uniform passes it;
  - no seed with a degenerate K, V or O (§2.2).

  The intervals are ordinary two-sided 95 % paired t intervals; C6 is a secondary claim outside the α of C2, C3 and C3r. A degenerate Q in any seed does not block "used", but C6 is then reported as "used, with input-independent queries", never as plain "used" (physics C17, critical C32); where the structure record shows Q_t identical across t, it reads "used, as query-independent attention pooling" instead (critical C42, chang C35). A degenerate Q equal to 0 at every element is reported as for a degenerate K. No C6 outcome changes: its logits are 0, so Δ_static is 0 there and "used" already fails. C6 is reported per g ∈ D_conf; "used" overall needs it at every g ∈ D_C2. Otherwise use is "not demonstrated".
- Zero-ablation is reported beside as context, since it would also credit a branch that carries only a constant offset.
- Branch cost, widths, entropy, the softmax zero-width fractions and norms are logged (§12).

**Rescue.** No poor-accuracy early stop exists. Any rescue (a floor, a delayed squeeze, gains, grouping, or a larger budget) is a new design decision taken before held-out access (`.claude/memory/experiment-log.md:1-7`).

### 8.9 Controls [OD-4]

**FP32 and W8A8.** These select the checkpoint with the highest validation top-1 over the whole run, with neither Chang filter condition, breaking ties by earlier epoch, then hash (`OPEN-CHOICES.md:98`).
- They are contextual comparisons under a selection policy different from the binary front.
- They stay outside every verdict, even when a W8A8 checkpoint falls inside the front's cost domain.
- W8A8 reports H and NR at its checkpoint. FP32 has no cost on the quantized front.
- No matched-budget W8A8 comparison is planned; it would need its own validation-only rule fixed before any run.

**DS-HGQ, GR and BE.** These are cost-trained under the same ramp and use §8.1. They are descriptive paired comparisons against CH-8 (DS-HGQ) or against chosen-form F111 (GR, BE).

### 8.10 CH-8 context pull against Chang's 66.3 % [P; critical B11]

This comparison is pre-registered as context, with the label "context: Chang's paper, his split and protocol (one head, attention floor, PID, XCU250)".
- **Quantity.** The seed-mean held-out top-1 of CH-8 at its released-callback-front checkpoint with the highest validation top-1 among those with training-count native ≤ 350,000.
- **Reference.** 66.3 % (paper Table 1 row 1; `literature/jet-tagging-transformers/2510.24784_replication-targets_hgq2-examples.md:61`).
- **Reported.** The difference, the CH-8 seed sd and the pull (difference / sd), per stage.
- **Trigger.** If |difference| > max(2 × CH-8 seed sd, 1 percentage point), VERIFY must give a quantitative cause before any "beats Chang" wording. A list of possible causes is not a cause (`docs/methodology/06-review.md:282-285`).
- **Beside it.** PID-CH's seed-mean is reported as further context, without a trigger.

## 9. Hardware [D8, Q10, Q14; OD-10]

**Sequencing (Kai, 2026-10-18; v0.11).** The export environment, H-LUT, export and capacity gates below remain requirements for valid hardware evidence, but are no longer launch preconditions for CH-8 and F000; binary arms reach them through the v0.4 check (§14). The later mulder permission covers the v0.4 whole-trained-model export check (§3.4). Synthesis of the deciding points follows training (Q14), with permission asked again for that use (§19.2 item 6).

**Factored and INLINE routes (Kai, 2026-10-18; v0.11).** The factored route is attempted in this campaign: each binary layer is exported as a ±1 layer followed by a per-column scale layer, preserving the exact accumulator and the model's function (§3.4 v0.4). Both it and the normal constant-matrix route include the generated-code patch `#pragma HLS INLINE` in the DA kernel wrapper `einsum_dense<N>_da_kernel`. The patch is applied identically to every model, CH-8 and F000 included; this resolves the "INLINE route" Kai item (`notes/flags-for-kai.md`, "Decided by Kai on 2026-10-18", under this campaign). The engineering evidence is `preflight/da4ml-test/README.md` and `preflight/da4ml-test/perpos-oneblock/README.md`, under this campaign, especially "Result", "Stages and settings" and "What this does not establish". It motivates this route and its matched scheduling patch; it is not whole-model evidence. The normal route in the existing C4 rule includes the same INLINE patch; the v0.4 H2 qualification compares factored and normal exports of the same model and does not itself establish C4.

**Route (chang B5).** hls4ml with da4ml on the constant matrix multiplies, as in the paper's §2.2 (`literature/jet-tagging-transformers/2510.24784_submicrosecond_transformers_jet_tagging_fpga.md:39-40`). Departures from Chang's code and the paper, declared and the same for both arms:
1. the export route is hls4ml, not the release's later alkaid default (`reference-code/HGQ2-examples/jsc150/test_utils.py:58-64`);
2. stock hls4ml 1.3.0 with the da4ml version option 0 fixes (0.5.2 unless only 0.6.0 passes), in the export environment below, not the `da4ml-v2` branch of calad0i/hls4ml that HGQ2's docs name (`reference-code/HGQ2/docs/status.md:58`), plus the MHA handler-key compatibility alias (below);
3. DA set per constant-multiply node rather than model-wide as Chang does (`test_utils.py:153`), because the model-wide strategy failed on 1.3.0 (`campaigns/2026-10-12-mha-bitnet/PREFLIGHT.md:68`; `distributed_arithmetic.py:105-108`);
4. the model-level default precision: Chang's code uses `fixed<-1,0>` (`test_utils.py:153`); the frozen §3.4 test uses `fixed<16,6>` (`preflight/da4ml-test/scripts/stage_b_build.py:158`), and production export matches the test. The generated project is checked for any type that falls back to the model default (chang C25);
5. a 2.5 ns clock with the Vitis default clock uncertainty of 27 %, because the frozen test passes none (`stage_b_build.py:164-165`; hls4ml 1.3.0 `backends/vitis/vitis_backend.py:62`), against Chang's 1.0 ns with zero uncertainty (`test_utils.py:142,155`). Production matches the test (chang C25);
6. C-synthesis on `xcvu13p-flga2577-2-e`, the default part of Chang's code (`reference-code/hls4ml/hls4ml/backends/vitis/vitis_backend.py:60`), and Vivado OOC synth plus opt on xczu7ev, against the paper's XCU250 and unstated stage;
7. the lab's common fabric-multiply binding, which Chang's code lacks (`test_utils.py:142-156`; `campaigns/2026-10-12-mha-bitnet/STUDY.md:2104-2135`; `docs/conventions/fpga-synthesis.md:14-29,49-50`).

Paired LUT gaps are unaffected in kind by these departures. Absolute numbers are not comparable with Table 1.
- **Settings.** Vitis HLS 2023.2, `io_parallel`, RF=1, II=1, 2.5 ns.
- **Labelling.** The C-synthesis and Vivado stages are distinct and labelled separately. DSP used and demanded are reported.
- **Remaining parts.** Dynamic QK/AV products and nonlinear tables stay in the design. Binary signs do not imply a DSP-free model.

**Part capacity (physics B7, C9).** The xczu7ev capacity used in every utilization statement is read from the "Available" column of `post_opt_util.rpt`, or from the Vivado part property, recorded in PREFLIGHT; the 230,400 CLB LUTs quoted in `review/physics-v0.2.md:194` (AMD DS891) is not used as a source. The paper's MHA-8 used 246k LUTs on XCU250 (§16), so CH-8 may not fit. The order below is fixed now:
1. **Deciding stage.** It is Vivado OOC `synth_design` plus `opt_design` on xczu7ev. A design above 100 % utilization is reported with its utilization as is, as a logic count, with no placement or timing-closure claim.
2. **If Vivado cannot complete synth plus opt on xczu7ev for capacity reasons** for any pair of a stage, the deciding stage for every pair of that stage moves to xcvu13p-flga2577-2-e OOC synth plus opt (physics C9, critical C25). This applies only if a VU13P Vivado licence probe passes at that time; an earlier probe failed (`docs/conventions/fpga-synthesis.md:49-50`). If the move is not possible, that stage's paired LUT gaps are `HARDWARE INCOMPLETE`; pairs are never compared across parts.
3. **Otherwise** the point is `HARDWARE INCOMPLETE (part capacity)`. Its C-synthesis estimate is reported, labelled as an estimate, and C4 is not supported by it.

All pairs of a stage, and both arms of each pair, use the same part and stage. PREFLIGHT gates the first step on a CH-8-sized model at initialization (§13).

**Export environment (chang B7).** One environment serves H-LUT option 0, Chang's trace and the export audit (§3.3), every deciding-point export, the full-model C-simulation gates and the cross-backend reload. It is built on mulder for the permitted export check, without delaying NRP training, and its package list, with versions and content hashes, is recorded in PREFLIGHT:
- Keras 3.14.0 and hgq2 0.1.9 with the two shims of §7.3, `quantizers` 1.2.2 and numpy as pinned for training;
- hls4ml 1.3.0 with the MHA handler-key compatibility alias (below);
- da4ml at the version option 0 fixes: 0.5.2 first, 0.6.0 only if option 0 fails under 0.5.2 (§3.4 Addendum A, step 1). Each version is its own locked environment, never an in-place upgrade;
- the JAX backend on CPU at the training pin, under which Chang's trace runs unmodified. Chang's own export path sets this backend (`reference-code/HGQ2-examples/jsc150/run_test.py:3`, `KERAS_BACKEND='jax'`; critical C44, chang C39).

If the §7.3 training fallback to Keras 3.15 is taken, the export environment stays at Keras 3.14.0 on CPU, where that fallback's reason (no working JAX GPU stack with Keras 3.14.0) does not arise. The cross-backend reload gate (§13) then carries the difference, and a failure of that gate is a PREFLIGHT finding reported to Kai.

Mulder's `m3-env` (TF 2.21.0, Keras 3.15.0, hgq2 0.1.9, quantizers 1.2.2, hls4ml 1.3.0, da4ml 0.5.2; `campaigns/2026-10-12-mha-bitnet/PREFLIGHT.md:68`) cannot serve, for three reasons (`review/chang-v0.4.md` B7 (a)–(c), with its software checks 1–3):
- under Keras 3.15, hgq2 0.1.9's fused-BN layer fails to unpack the einsum analysis, so CH-8 does not build;
- Keras 3.15 no longer sets `full_output_shape`, which hls4ml's MHA handler reads;
- on TensorFlow, Chang's trace computes the EBOPs einsum `'...c,cC->'`, which TensorFlow rejects.

`m3-env` is never modified (§13). A TensorFlow backend in the export environment would need the explicit-label EBOPs einsum, change (2) of `campaigns/2026-10-12-mha-bitnet/review/PREFLIGHT_chang_ch8_env_v1.md:78`, declared. That is the fallback if conversion fails under JAX, and it is a PREFLIGHT finding reported to Kai. Using `m3-env` for export with three declared Keras-3.15/TensorFlow shims is not adopted (§19.3). The superseded clean-rerun procedure used `m3-env` or its 0.6.0 clone, with hls4ml and da4ml equal to the export environment's (§3.4 Addendum A, step 2); it is retained as history only.

**Export blocker, build item H-LUT [v0.1 review B3; chang B4].**
- **The blocker.** Chang's two affined tanh layers (`model.py:194,198`) default to heterogeneous tables: a learned scalar scale and bias, each of shape (1,) (`reference-code/HGQ2/src/hgq/layers/activation.py:100-102`), with per-value input and output formats (`:76-108`) (chang C23). The inspected hls4ml handler names `QUnaryFunctionLUT`, asserts a homogeneous table, and reads scalar output formats with `.item()` (`reference-code/hls4ml/hls4ml/converters/keras_v3/hgq2/unary_lut.py:18-32,64-75`).
- **Option 0, tested first, in the export environment (chang B4; v0.5: chang B7, C32, C33; critical C35).** hls4ml 1.3.0 converts with `allow_da_fallback=True` by default (`reference-code/hls4ml/hls4ml/converters/__init__.py:178`). An unregistered class then goes to `da4ml.converter.trace_model` as a `DACombinational` node (`reference-code/hls4ml/hls4ml/converters/keras_v3_to_hls.py:176-228`). The hgq2 0.1.9 tracer handles `QAffinedUnaryFunctionLUT`, applying the scalar scale, then the bias, then the function, element by element (`reference-code/HGQ2/src/hgq/_dais_tracer/layers/activation.py:47-61`). The tracer imports `da4ml.converter.plugin` (`reference-code/HGQ2/src/hgq/_dais_tracer/main.py:6`); the lab dossier dates the plugin system to da4ml 0.6.0 (`literature/fpga-inference/da4ml_dossier_job-alpha.md:155-160`). da4ml 0.6.0 lies outside hls4ml 1.3.0's declared range, which exists only for its `da` extra (`da4ml<0.6,>=0.5.2`, installed `hls4ml-1.3.0.dist-info/METADATA:30`); 0.6.0 installs beside hls4ml 1.3.0 with `pip check` clean (`campaigns/2026-10-12-mha-bitnet/PREFLIGHT.md:68`) (chang C33). The test runs steps 1–5 under da4ml 0.5.2, and under 0.6.0 only if a step fails under 0.5.2:
  1. import `da4ml.converter.trace_model` and `da4ml.converter.plugin` (hgq2 0.1.9's metadata requires `da4ml>=0.6` for its test extra and asks da4ml users for at least 0.6: installed `hgq2-0.1.9.dist-info/METADATA:29,78-81`; chang C24);
  2. install the MHA handler-key compatibility alias first, since without it the MHA would also fall back to DA (chang C24);
  3. convert CH-8 at initialization with `allow_da_fallback=True` and `allow_v2_fallback=False` (its default is True, `converters/__init__.py:179`). `allow_da_fallback` is one model-wide flag (`converters/__init__.py:178`; installed hls4ml 1.3.0 `converters/keras_v3_to_hls.py:129-135`), so the restriction to the two `QAffinedUnaryFunctionLUT` layers comes from every other class being registered, the MHA alias included, together with the assertion of exactly two `DACombinational` nodes with everything else through registered handlers (chang C32);
  4. record which tracer `trace_model` dispatched to (hgq2 0.1.9 registers `keras = hgq._dais_tracer.main:HGQ2DAISTracer` under the entry-point group `dais_tracer.plugins`, installed `entry_points.txt`), and assert that the traced nodes reproduce each layer's scalar scale and bias;
  5. run the table-equality and C-simulation gates below.

  Chang's own `hls4ml_da` path also converts with the default `allow_da_fallback=True` (`test_utils.py:149-156`), so option 0 follows his route for these two layers.

  **Outcome.** Option 0 fails at a version on any failure of steps 1–5, and the exception type and message of each step are recorded (chang B7). `fallback_handler` swallows only `KeyError` and `ImportError` (installed hls4ml 1.3.0 `converters/keras_v3_to_hls.py:185-189`), after which the converter raises `ValueError` (`:160`), so the recorded type tells a missing handler or package from a conversion error (chang C32). If option 0 passes, it is the export route for the two tanh layers, at the lowest version under which it passes (0.5.2 if it passes under 0.5.2), before any custom handler (critical C35). If it fails under both, the custom H-LUT proceeds on da4ml 0.5.2. Either outcome is recorded. Using da4ml 0.6.0, outside hls4ml 1.3.0's declared range, is a declared departure, and both routes of the v0.4 whole-model check use that version; Addendum A's former clean-rerun requirement is superseded.
- **Why the old exporter cannot be reused.** The paused campaign's exporter homogenizes the tables (`campaigns/2026-10-12-mha-bitnet/code/bnhgq2/mhab_export.py:16-19,76-99`), which conflicts with Q8.
- **Required.** Exact export of the affined tanh tables, keeping the learned scalar scale and bias and every per-value input and output format, with no homogenization of training quantizers.
- **Gates.** Table equality on every site's full input grid, then a full-model C-simulation on the 4,096 validation jets. These are established on CH-8 and F000 first, then on the binary arms.

**Further export items.**
- **MHA handler (chang C4).**
  - The MHA handler calls `QEinsumDenseHandler()` on Q/K/V/O and reads `qkernel` and `qbias` through `load_weight` (`reference-code/hls4ml/hls4ml/converters/keras_v3/hgq2/multi_head_attention.py:104-113`; `reference-code/hls4ml/hls4ml/converters/keras_v3/hgq2/_base.py:109-114`).
  - Registration is needed for MHA subclasses and for top-level binary classes, not for the projections.
  - The binary projections must duck-type `QEinsumDense`: at least `qkernel`, `qbias`, `_iq`, `_oq`, `_enable_iq`, `_enable_oq` (hls4ml 1.3.0 `converters/keras_v3/hgq2/_base.py:79,87`), `equation` and `full_output_shape`. The full list is established by converting an initialized binary configuration and asserting that no `AttributeError` occurs (chang C26).
  - The compatibility alias for the 1.3.0 handler key `hgq.layers.multi_head_attention.QMultiHeadAttention`, against the 0.1.9 module `hgq.layers.attn.mha`, is a required item.
- **DA strategy.** It is set per constant-multiply EinsumDense node. Variable-input Einsum maps to latency (`reference-code/hls4ml/hls4ml/backends/vivado/vivado_backend.py:832-839,870-880`). The fallback is disabled except for option 0's two nodes, and the generated DA coverage is checked.
- **The kif reshape fault.** This fault forced a shim in the paused campaign (`PREFLIGHT.md:66`; `distributed_arithmetic.py:81-91`). It is retested here with Chang's per-value quantizers.
- **Exported scale type (physics C3; corrected in v0.5, chang C31).** hls4ml 1.3.0 reads no scale quantizer. At a binary site it loads the exported kernel K = q_g(g)·S through `qkernel` (hls4ml 1.3.0 `converters/keras_v3/hgq2/_base.py:109-114`). Its BitExact pass is on by default when `FixedPointQuantizer` nodes exist (installed hls4ml 1.3.0 `model/optimizer/passes/bit_exact.py:895-899`) and sets one `weight_t` per weight array from `minimal_kif` of the data (`:722-740`); every ±q_g(g_c) lies on its quantizer's grid, so that type holds each constant exactly. Under DA the kernel data are consumed directly, and `weight` and `weight_t` are deleted (hls4ml 1.3.0 `backends/vivado/passes/distributed_arithmetic.py:367,408-410`). No default precision therefore rounds the constants on either path. Production export leaves `bit_exact` at hls4ml's default, as the frozen test does (`stage_b_build.py` does not pass it), and PREFLIGHT records whether the pass ran. The bit-exact model-to-C-simulation gate is run explicitly on an initialized binary configuration of each form, not only on CH-8 and F000.

If H-LUT cannot be built by option 0 or a custom handler, the route returns to Kai as a design choice. Q8 is not changed and the old exporter is not substituted.

**Fidelity (chang C14).**
- **Trace.** Trace calibration runs in the export environment and uses train, then validation, as Chang does, with `no_growth=False` as released (`run_test.py:31-44,73`). Chang's `run_test` re-splits the data (`run_test.py:25`); here the frozen split is used, and train-then-validation still covers the whole training archive.
- **Stored values.** Pre- and post-trace cost and metrics are stored.
- **Comparison.** Exported and C-simulated outputs are compared on the same 4,096 validation jets, counting argmax changes, and bit-exact export-to-C is required (`docs/conventions/fpga-synthesis.md:34-41`).
- **Changed models.** An exported model that is not identical to the training model is labelled separately and frozen before the held-out read (§8.5).

**Deciding points [OD-10].** These are the unique ramp checkpoints of BEST and CH-8 at g ∈ D for all seeds of a stage, plus the PID selections. All are fixed before unblinding.
- **Counts (derived).** At most 2×6×3 + 2×3 = 42 in the screen. In the confirmation: 2×6×5 + 2×5 = 70 under option A, 2×6×8 + 2×8 = 112 under B, 2×6×8 = 96 under C. Whether 112 or 96 builds fit the seven-day window is not known; the priority order below decides which pairs are built first, and any pair not built is `HARDWARE INCOMPLETE`.
- **Priority.** Confirmation pairs first if S4 is funded; within a stage, PID 350k, then ramp 350k, then the remaining allowances in ascending order.
- **Window.** The mulder admission window is seven days per stage, with a 24-hour stage timeout and no performance-based retries (`campaigns/2026-10-12-mha-bitnet/STUDY.md:2240-2250`). Missing builds are labelled `HARDWARE INCOMPLETE`.
- **Pairing.** A paired LUT gap needs all pairs of a stage at the same part, stage and settings (Part capacity, above). It is reported with post-trace accuracy, H, FF, DSP, BRAM, latency, II, timing, and the LUT/H1, LUT/H2 and LUT/NR ratios (physics B4). The decision rule for C4 is in §8.7.
- **Cost.** Mulder time is outside the GPU budget.

## 10. Budget and tiers [OD-8; L1]

**No measured rate for this workload exists [L1].**
- The planning rate is r = 3 GPU-s per run-epoch. It is the paused campaign's optimistic packed N=8 estimate, made for its TensorFlow runner (`campaigns/2026-10-12-mha-bitnet/STUDY.md:2307`); this campaign runs JAX (critical B12).
- A 31–38.8 s ceiling was borrowed from N=64 models, where 38.8 s is the A07 model, not E (`:2299-2312`).
- The old smoke pack's 3.053333 GPU-h was a failed pack, not one successful run (`campaigns/2026-10-12-mha-bitnet/PREFLIGHT.md:744-748`).

**Formula.** At the planning rate, H_GPU = runs × 7,000 × r × 1.10 / 3,600, where 1.10 is the prior overhead estimate (`campaigns/2026-10-12-mha-bitnet/STUDY.md:2316-2318`). That is 6.416667 GPU-h per run at r=3 and 12.833333 at r=6 (derived).

Once PREFLIGHT measures all-inclusive rates r_m, which include startup, the per-epoch frozen evaluation, checkpoint saves and evaluation, the factor is dropped: H_GPU = runs × 7,000 × r_m / 3,600 (critical C13). All values below are projections.

| tier, in priority order | added runs | cumulative runs | added GPU-h, r=3 | cumulative incl. 50 reserve, r=3 | added GPU-h, r=6 | cumulative incl. 50 reserve, r=6 | at r=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T0 CH-8 | 3 | 3 | 19.25 | 69.25 | 38.50 | 88.50 | required |
| T1 F000, F100 | 6 | 9 | 38.50 | 107.75 | 77.00 | 165.50 | required |
| T1 pre-stage/factorial, 6 configs × 2 forms | 36 | 45 | 231.00 | 338.75 | 462.00 | 627.50 | required |
| T1 PID pair | 6 | 51 | 38.50 | 377.25 | 77.00 | 704.50 | required |
| T2 FP32 | 3 | 54 | 19.25 | 396.50 | 38.50 | 743.00 | fits |
| T3 W8A8 | 3 | 57 | 19.25 | 415.75 | 38.50 | 781.50 | over cap |
| T4 DS-HGQ | 3 | 60 | 19.25 | 435.00 | 38.50 | 820.00 | over cap |
| T5 GR | 3 | 63 | 19.25 | 454.25 | 38.50 | 858.50 | over cap |
| T6 BE | 3 | 66 | 19.25 | 473.50 | 38.50 | 897.00 | over cap |
| S4 option A, as adopted: ramp and PID, seeds 4–8 (beyond the cap; Kai) | 20 | +20 | 128.33 | 505.58 over the required screen; 601.83 over all 66 | 256.67 | 961.17; 1,153.67 | needs Kai |
| S4 option B: ramp and PID, seeds 4–11 (beyond the cap; Kai) | 32 | +32 | 205.33 | 582.58; 678.83 | 410.67 | 1,115.17; 1,307.67 | needs Kai |
| S4 option C: ramp only, seeds 4–11 (beyond the cap; Kai) | 16 | +16 | 102.67 | 479.92; 576.17 | 205.33 | 909.83; 1,102.33 | needs Kai |

**Cap [OD-8, resolution c].** The cap is 400 GPU-h, including a ring-fenced 50 GPU-h reserve for smoke tests, benchmarks, failures and restarts, with at most four concurrent GPU jobs.
- The required screen is 51 runs: 357,000 run-epochs and 327.25 GPU-h, or 377.25 GPU-h with reserve, at r=3.
- All 66 screen runs would need 462,000 run-epochs and 473.5 GPU-h with reserve.
- At r=3 one control fits: under the item-9 default, DS-HGQ or FP32 by the item-9 rule; under tier order, FP32 (critical C40). The other controls are "not run: budget" unless the measured rates are lower.
- **Premise change.** OD-8 was adopted on the premise "all 48 runs plus reserve project to 358 GPU-hours" (`OPEN-CHOICES.md:61`). Kai's D9 pre-stage raised the total to 66 runs, so keeping all Q11 controls would need a cap of about 475 GPU-h at r=3, which is put to Kai (critical B10, §19).
- **Superseded numbers.** The OPEN-3 and OPEN-8 numbers in `OPEN-CHOICES.md:24-25,61` are v0.1 numbers; this table supersedes them (critical C15).

**Break-even (derived).** Against 350 training GPU-h:
- the required screen fits only if r ≤ 3.2086 s at the planning rate with the 1.10 factor, or r_m ≤ 3.529 s for an all-inclusive measured rate;
- all 66 runs fit at r ≤ 2.4793 s (planning) or 2.727 s (all-inclusive);
- the required screen plus S4 fits the 350 training hours only at r ≤ 2.3047 s under option A, 1.9715 s under B and 2.4423 s under C (planning; S4 is beyond the cap in any case);
- at r=6 the required screen plus reserve is 704.5 GPU-h.

**Admission (critical C33).** Measured per-class rates decide admission. The required set is reserved first, and each admitted control trains all three seeds.
- Under the §19.2 item 9 default, controls wait for S2 and train in S3. The first slot goes to DS-HGQ if BEST has a degenerate attention layer, binary or HGQ (Q, K, V or O, §2.2), at any frozen point of any screen seed, and otherwise to FP32. Any further slots the measured rates admit go to the other of the two, then to W8A8, GR and BE in that order.
- Kai's answer to item 9 is requested before the S1 launch. While it is open the slot is held, which is the default, and nothing in S1 uses it.
- Under the 475 GPU-h alternative all five controls train: FP32, W8A8 and DS-HGQ in S1, GR and BE in S3.
- Run counts and GPU-hours are the same either way; only the stage in which the controls train changes.

**When the cap binds.**
- Controls are dropped first: under the item-9 default, controls are dropped in the reverse of the §10 admission order; under tier order, BE, GR, DS-HGQ, W8A8, FP32 (critical C40).
- The factorial seeds, factorial cells and the 7,000 epochs are never cut.
- If the measured rate puts the required screen over the cap, the campaign stops at PREFLIGHT and Kai decides between a cap change and a scope change (§19).
- S4 is never funded from leftover hours without Kai's approval.

**Packing and wall time.** Pack size K and GPU product are chosen from same-workload measurements: at least 40 % rolling three-hour utilization, at most 90 % device memory, and a safe host-memory projection, with each seed's arms on one product (`docs/infrastructure/gpu-selection-policy.md:11-17,32-69`).
- At r=3, the 327.25 required GPU-h over four GPUs give an idealized active-wall projection of about 82 hours, under the r=3 and 10 % overhead assumptions.
- Stage dependencies (S3 and S4 after S2) and queueing make real wall time longer, and no completion date can be certified.

## 11. Operations

- **CPU gates.** Builds, reloads, cost, gradient, export-on-CPU and resume gates run on the home PC (`decisions.md` [dec 10-17 delegation], "Kai (chat, verbatim)", sub-bullet "Hard stop kept"; `CLAUDE.md:25-26`; the paused campaign's local gates, `campaigns/2026-10-12-mha-bitnet/PREFLIGHT.md:728`).
  - GPU-only paths are checked on the PC's RTX 4060 Ti, in a JAX environment matching the pinned stack, before any NRP smoke. CPU-only gates missed the old `uint8` GPU fault (`PREFLIGHT.md:690-704`).
  - None of these readings is quotable (`CLAUDE.md:23-26`).
- **Mulder.** Access requires Kai's explicit permission and the UCSD-verified connection he establishes (`AGENTS.md:8-12`). Kai removed mulder from the path to NRP training on 2026-10-18, then permitted it for the whole-model factored-export check that qualifies H2 (`decisions.md` [dec 10-18 NRP] and [dec 10-18 H2]; §0 quotes his words). That permission covers the export-check purpose of §19.2 item 6. Later use for deciding-point synthesis is asked again. The old test evidence is preserved and `m3-env` is never modified; Addendum A is a historical procedure, not a launch requirement.
- **NRP.** Training runs on NRP. Route A2 is the route of record: a git bundle on `kai-data`, consumed by `tools/run_handoff.py` bundle mode, with the pinned commit and content hashes recorded. This resolves §19.2 item 1 (`decisions.md` [dec 10-18 H2], "Kai on the plan"; [dec 10-19 PVC]). Launches still require `nrp_doctor.py` lint through a path where the hook runs, an immutable verified handoff, Kai's `python3 tools/harness.py approve`, and the remaining §14 preconditions. The entry "2026-10-19 (later, 2) — chang-bitlinear Phase 2 staged on kai-data; one hook deviation" records the hook-path deviation and the rule that subsequent creates use a path where the hook runs.
- **PVC and ConfigMaps (Kai, 2026-10-19).** All campaign state on NRP stays on `kai-data`, under `/data/cbl1016/`: code bundle and checkout, data archives, venv, handoff records, run outputs, checkpoints, fronts, per-epoch records, logs and W&B offline files. No ConfigMaps, hostPath, other PVC or object storage; emptyDir is only for transient caches (`decisions.md` [dec 10-19 PVC], "Kai (chat, verbatim)"). `kai-` ConfigMaps are counted before every launch wave, including each resubmission; the required count is zero. W&B remains an index copy under §12; durable NRP evidence is on the PVC.
- **Runtime.** The hashed lock is installed into a venv on `kai-data` and gated before any NRP training smoke. This replaces the proposed pushed JAX runtime image and resolves §19.2 item 7. The pins and runtime checks of §7.3 and §13 remain.
- **Resubmission.** Kai's instruction, "if things fail or queue fails, just loop until it succeeds" (`decisions.md` [dec 10-16 later], GPU-queueing bullet, as cited here before v0.11; `notes/build-todo.md`, B17 retry-policy item), means no count limit on infrastructure relaunches within the §10 GPU-h cap. Eviction, preemption, node faults and scheduling timeouts qualify; every attempt stays in the ledger and its allocation time counts against the cap. Code or integrity failures stop for a fix, a new commit, re-verification and a new handoff, as already required here and in RULES §5. This does not authorize scientific retries or relax §14's integrity stops.
- **Backend.** JAX as pinned [OD-9] (§7.3).

## 12. W&B logging

The project is `bnjettag-chang-bitlinear` and the group `chang-bitlinear-20261016`. Run names carry arm, form, seed, stage, ramp or PID, attempt and source bundle. W&B is an index only. The durable evidence is the checkpoints, front sidecars, metric and prediction arrays, cost audits and synthesis reports (`docs/infrastructure/wandb-layout.md:15`; `SYSTEM.md:88`).

Logged per epoch:
- epoch and step;
- train and validation loss and top-1, plus validation macro and per-class AUC for saved candidates;
- learning rate and λ;
- the training count, and frozen H1, H2 and NR, with per-layer and per-group subtotals;
- widths and steps; at binary sites, the scale quantizer's b and i, q_g(g) or q_g(g_c), the zero columns by route (pruned or rounded to zero, §2.2), saturated columns (input at the largest grid value, §2.3), sign occupancy and sign-flip rate; BN scales and moments; and the moment mismatch between x·W and x·K;
- **structure (physics B10; v0.5: critical B18, C38, chang B8):** for every arm, CH-8 included, the kernel and quantizer-width routes to a degenerate layer (zero kernels by route, all-zero-width input or output quantizers of Q, K, V, O and the FFN contraction) at every epoch end, read from quantizer state without a data pass; the per-head fraction of zero-width entries of the softmax output quantizer; and, in every frozen checkpoint's sidecar, the zero-variance test on validation that defines a degenerate layer (§2.2, §8.1);
- activation ranges and zero-width fractions;
- **pruning channel (physics B5):** each HGQ site's fraction of zero-width kernel coefficients and the share of native cost they remove, at every grid checkpoint, for CH-8 and every arm, and the fraction of zero-width inputs at binary sites;
- **fused-BN binary sites (physics B6):**
  - the fraction of FB signs that differ between the last step's batch-moment kernel and the moving-moment kernel;
  - the number of single-sign columns;
  - the fraction of output channels whose pre-activation is ≤ 0 on all validation jets (dead channels);
  - the train-mode versus frozen validation top-1 gap;
- epochs spent per grid interval (§8.8).

PID runs add the sensed H, the stock aggregate, error, integral, clamp state and slack. Every run also logs provenance, split and order hashes, the per-epoch hash of the fetched batch-index sequence and the per-epoch `_index_map` hash (§6), backend and matmul precision, GPU product and K, time, memory and allocation hours.

Validation-only attention diagnostics are logged at saved checkpoints: Q/K/V/O and QK/AV cost and widths, the attention entropy H_attn of the row-renormalized post-quantizer probabilities against log 8 with the count of all-zero rows, the softmax zero-width fractions, degenerate Q/K/V/O by route, branch norm, Δ_attn by mean and zero ablation, Δ_static and Δ_unif (§8.8). No test metric appears before the frozen final evaluation (`docs/methodology/03-phases.md:37-44`).

## 13. PREFLIGHT build list and gates

**v0.11 sequencing.** Export-environment, export/H-LUT, export-audit and part-capacity requirements below apply at the export check and hardware stage; they are no longer launch preconditions for CH-8 and F000, and binary arms reach them through the v0.4 check (§14). The clean da4ml rerun and its rerun-environment equivalence check are the superseded Addendum A path. Their former before-production wording is historical; v0.4's whole-trained-model check governs H2 (§3.4). All non-export training gates remain in force.

Reuse is a starting point, never a claim of a passed gate. The paused campaign stays untouched; its final SHA is `6a145873…` (`campaigns/2026-10-12-mha-bitnet/PREFLIGHT.md:731-740`).

| component | reuse | build and required gate |
| --- | --- | --- |
| runtime venv and stack (critical B12, chang B1; v0.11 PVC route) | none for JAX | venv on `kai-data`, installed from the hashed lock, with JAX/jaxlib-CUDA, Keras 3.14.0, hgq2 0.1.9 plus the two shims, `quantizers`, all pinned and hashed. Gates: GPU device visible; float32 matmul precision active after `import model`; hgq2 and snapshot hashes recorded; no secrets in the runtime. A matching PC GPU environment |
| export environment (chang B7; §9) | none; mulder `m3-env` is not reused for export and is never modified | Keras 3.14.0, hgq2 0.1.9 with the two shims, `quantizers` 1.2.2, numpy as pinned for training, hls4ml 1.3.0 with the MHA alias, da4ml as option 0 fixes (each da4ml version its own locked environment), JAX on CPU; built on mulder; package list with versions and content hashes recorded. Gates: CH-8 builds at initialization; Chang's trace (`trace_minmax`, train then validation) runs unmodified; the cross-backend reload below |
| reference fidelity (chang B1, C17) | snapshot `reference-code/HGQ2/` | the cited-module diff of the snapshot against the installed 0.1.9, both re-hashed (§7.3); CH-8 from our runner bitwise equal to the reference module on the pinned stack in initial weights, training and inference outputs, per-layer EBOPs, loss, gradients and BN moving statistics, on identical batches; the device and product on which this bitwise gate runs are a PREFLIGHT owner decision (determinism record, below; v0.8) |
| data | identity and isolation checks (`code/bnhgq2/derived_view.py:3-22`) | Chang preprocessing, split and order manifest; counts, labels, disjointness; train/validation-only reader |
| RNG (chang B2, B6) | none | the flow of §6; initial tensors copied by role, unmatched tensors listed; per epoch, the hash of the mapped batch indices actually fetched is identical across all arms of a seed (the `_index_map` hash is recorded as secondary); no lab callback or frozen-evaluation step draws from the module-level `random` stream during training |
| CH-8 runner, F000 | fit, resume and exact-sidecar patterns (`code/bnhgq2/mhab_fit.py:1-25,107-122`) | new implementation, reference module unchanged; F000 equals CH-8 in tensors, forward, backward, one optimizer step and native cost on identical batches; nested MHA kernel mapping; single wrap of the HGQ call (`base.py:57-114`; `code/bnhgq2/mhab_model.py:66-71`) |
| binary layers, FB and BF (critical A3, C14; chang B3, C6) | serialization patterns only (old rule differs: `code/bnhgq2/mhab_quant.py:215-280`) | see the binary-layer gates below |
| cost functions | none | H1, H2 and NR per §3.2 in the native dtype; H1 ≤ NR at every site; reduce exactly to native on F000; frozen evaluation equals reload and export audits exactly; zero- and random-input agreement; the per-step penalty equals (b) recomputed independently from the quantizer variables, not from the training kernel tensor; top-level public equals flattened private aggregate; every value snapped to its quantizer's f before w(·), with an assertion that the snap changes nothing; H1 ≤ NR on the GPU path as well as on CPU (chang C27); a binary layer's `_compute_ebops` equals (b), and NR comes from the frozen evaluator (critical C17); initialization readings of H1, H2 and NR per configuration (§14) |
| da4ml test (§3.4), historical v0.3 path | `preflight/da4ml-test/` evidence and frozen scripts, preserved | Addendum A records the former clean-rerun procedure and its safeguards. It is superseded for headline adoption and is not a launch gate (Kai, 2026-10-18). The v0.4 whole-trained-model check, including both export routes and their identical INLINE patch, now governs H2 |
| rerun-environment equivalence (critical C44, chang C39; historical Addendum A path) | `preflight/da4ml-test/scripts/stage_b_build.py`, unchanged | the former equivalence check belongs to Addendum A's clean-rerun path, now superseded; it does not delay NRP training |
| export, H-LUT (§9) | report parsers; DA environment knowledge | in the export environment: option 0 first, da4ml 0.5.2 then 0.6.0 only on failure, with the MHA alias installed, `allow_v2_fallback=False`, the dispatched tracer and every step's exception type recorded; exact affined-tanh export with the scalar scale and bias reproduced; binary projections convert with no `AttributeError`; MHA handler registration and compatibility alias; per-node DA with coverage; the BitExact pass recorded as run; block, then full-model C-simulation; CH-8 and F000 first, then an initialized binary configuration of each form; cross-backend reload, where a checkpoint trained under JAX on the GPU and loaded in the export environment gives identical frozen H and identical argmax on the 4,096 validation jets |
| part capacity (physics B7, C9) | `preflight/da4ml-test/scripts/vivado_ooc.tcl` | a CH-8-sized model at initialization completes OOC synth plus opt on xczu7ev with utilization above 100 % reported, not truncated; the capacity recorded from the report's "Available" column or the part property; the §9 order applies if it does not |
| structure record and attention ablations (physics B10; chang B8; critical B18, C38; v0.6: chang C35–C37, critical C42) | none | the zero-variance detector flags each named route on constructed cases (a zeroed kernel, a kernel rounded to zero, an all-zero-width input quantizer, an all-zero-width output quantizer) for a binary and an HGQ layer, and flags nothing on a constructed non-degenerate case; for a degenerate Q, the per-head record of whether Q_t is identical across t and whether Q is 0 everywhere; softmax zero-width fractions per head; Δ_attn, Δ_static and Δ_unif each give exactly 0 when the ablated tensor is replaced by itself; Δ_unif is exactly 0 on a constructed model with K ≡ 0; Δ_static is exactly 0 on the constructed model with K ≡ 0, with the validation mean accumulated in float64 (chang C37); the downstream WRAP overflow counter counts a constructed overflowing case (chang C36); H_attn equals log 8 on constructed uniform rows |
| PID (chang C1, C2) | controller law and defaults | sensed(e) equals frozen H(e) exactly; Chang's functional model handed to callbacks; stock aggregate > 0 after epoch 0; positive init; interrupted and resumed integral identical; log keys per §7.2 |
| resume (chang C12, B6) | resume patterns | persist the module-level Python `random` state, which both the `Dataset` (`dataset.py:113`; no offset draw with `drop_last=False`, critical C34) and Keras's adapter (`py_dataset_adapter.py:261`) draw from, the ParetoFront record (`reference-code/HGQ2/src/hgq/utils/sugar/pareto.py:28-29`), the StopIf counters (`stop_if.py:54-55`) and the PID state; uninterrupted and resumed runs identical, on the home PC (§11); whether this bitwise gate also runs on the production GPU product is a PREFLIGHT owner decision (determinism record, below; v0.8) |
| arms and callbacks | configuration generator patterns | all arm identities including both forms, PID templates and S4 templates; full factorial; shared initialization; callback order; primary-front saver |
| selection and evaluation | raw prediction patterns | domain, form rule, coverage-first BEST, NA and ties, screen tables without verdict labels, the g* rule and D_band, the S2 plug-in power at observed gains, confirmation intervals (intersection-union over D_C2, the nested whole-D_conf form, and the 7- or 6-contrast labels; critical C36), sealed manifest; identical arms give zero differences and an interval around zero; v0.7 (constructive K1–K3, K6, K8, K9): the pooled σ̂_d with its χ² band, the null selection reference simulated with the exact §8.3–§8.4 rules, the factorial randomized-block intervals and the additive-model cell, the per-class background rejection at ε_S = 0.5, the paired A_ref − A_sq drop, and the count of distinct checkpoint pairs; identical arms give zero effects and intervals around zero, and a pooled residual of 0; v0.8 (constructive v0.7 B2): with one candidate in one form, the simulated median of the null selection reference is 0 |
| harness | freeze, handoff, ledger, pack and telemetry patterns (`campaigns/2026-10-12-mha-bitnet/tools/README.md:7-18,56-57,71-90`) | new identities and authority; measured all-inclusive class rates; GPU training step and kill/resume smoke; ConfigMap count (§11) |

**Binary-layer gates.** These hold on CPU and on the GPU path:
- the training-mode forward kernel, not only `qkernel`, holds exactly the two symmetric values ±q_g(g) per layer at non-BN sites or ±q_g(g_c) per column at fused-BN sites, or is zero where q_g rounds to zero;
- trainable variables equal CH-8's, minus kq's b and i at binary sites, plus q_g's b and i;
- no kq regularization term exists at binary sites;
- training-mode autodiff gradients equal those of Chang's own layer with kq replaced by the stated map at the same site: ∂L/∂W, ∂L/∂γ, ∂L/∂b, ∂L/∂x and ∂L/∂(q_g's b and i), by analytic or autodiff comparison, never finite differences of Sign;
- the penalty gradient reaches q_g's b;
- `qkernel` is computable at the end of `build`, because hgq2 0.1.9 evaluates every `q<var>` property there (installed 0.1.9 `base.py:118-140`; chang C17);
- q_g's configuration equals the MHA projection configuration `_qkvo_kq_conf` (`mha.py:64`) field by field, and survives a serialized reload (chang C22);
- edge cases: Sign(0); β = 0; negative and zero s_c; q_g pruned (b = 0) and q_g rounded to zero (b > 0), each with its costs in H1, H2, NR and the penalty (chang C15); a saturated column, where BF's direct γ gradient is 0 and FB's is not (critical C18); FB columns single-magnitude;
- q_g's float32 output is exact at b = 23, and a forced b = 24 case must fail the exactness check, which shows that the check can fail (critical C16).

**Further required gates:**
- every build and serialized reload;
- metric reload agreement within 1e-7 with TF32 off (`docs/conventions/quantization-and-cost.md:43-45`);
- training, saved and exported checkpoint identity;
- validation-only tracing and selection;
- equivalence between uninterrupted and resumed runs;
- a measured throughput and memory benchmark, then an immutable handoff;
- **a determinism record, not a gate (constructive K4, v0.7; F000 run added in v0.8, constructive v0.7 B1):** two CH-8 seed-1 runs and one F000 seed-1 run on the production GPU product and stack, from the same initialization and batch order, for 20 epochs, compared bitwise in weights and in the per-epoch frozen record, the F000 run against both CH-8 runs. PREFLIGHT runs it from the existing 50 GPU-h reserve, about 0.05 GPU-h at the planning rate, or 0.055 GPU-h with §10's 1.10 factor, plus start-up (derived: 60 run-epochs × 3 s = 180 s; × 1.10 = 198 s). If the three runs are identical, F000 − CH-8 over the full run is expected to be 0, and any difference is an implementation finding. If not, F000 − CH-8 measures run-to-run divergence and is reported as the null-arm paired s_d (§8.6). The record also decides whether F000's row enters the pooled variance of §8.6. It is an NRP GPU run, so like any smoke or benchmark it follows the A2 PVC route and the remaining launch preconditions of §11 and §14. **Stack of the bitwise gates (constructive v0.7 B1, v0.8; a PREFLIGHT owner decision).** §11 places the resume gates, with the build, reload, cost and gradient gates, on the home PC, and GPU-only paths on the PC's RTX 4060 Ti, not on the production GPU product on which this record runs; the STUDY names no device for the bitwise training equality of the reference-fidelity row. On which product and stack each of those two bitwise gates runs, and whether either also runs on the production GPU product and stack, is decided by the PREFLIGHT owner and recorded before the gates run. If either runs there, the record's "not identical" branch is then a failure of that gate.

**Formal protocol check.** Jev's `lab_check_protocol` is run on `protocol-draft.json` before the freeze, and `lab_freeze_protocol` after STUDY passes (`docs/methodology/03-phases.md:59-62`). The draft holds the screen as `scope: "screen"` and the confirmation, option A as adopted, as a separate claim-scope object, which is checked with the same deterministic function. A separate claim-scope protocol file, written for the option and the claim domain Kai chooses, is frozen before any S4 launch.

## 14. Stop rules [A2]

**Fail closed.** Subject to the v0.11 export deferral, the following block launch. §19.2 items 2–4 (OD-13, OD-15, OD-12) and 12 (claim domain) remain open launch preconditions; the plan approval and PVC decision do not resolve them:
- an unresolved implementation choice for training; unresolved export work follows the deferral above and the v0.4 H2 check;
- an F000/CH-8 or CH-8/reference mismatch;
- missing data identity or hidden test access;
- a failed gradient, cost, RNG or PID gate; export gates remain hardware requirements and are no longer launch preconditions for CH-8 and F000; binary arms reach them through the v0.4 check (next item) (Kai, 2026-10-18);
- for any binary arm, F100 included, under either count: no recorded outcome of the §3.4 v0.4 whole-trained-model factored-export check for both F111 forms. A failed check is a completed attempt that fails any criterion, or an attempt the owner records as impossible to complete, with the reason, reported to Kai before any binary launch. The outcome then sets the binary count by the v0.4 split-form rule. CH-8 and F000 do not wait for the headline choice. The clean da4ml rerun of Addendum A is no longer a launch precondition;
- in the one-pass case of §3.4 v0.4, for any of the 24 added H2 runs: no answer from Kai on how they enter §8, or no confirmation by Kai of the exact cap (§19.2 item 5; critical v0.11 B7, v0.13);
- missing initialization readings of H1, H2 and NR per configuration in PREFLIGHT (critical B6, rewritten under resolution a);
- missing review or launch authority (§11);
- OD-12 (amends D6), OD-13 (amends D3 and D9) and OD-15 (amends D1) not each confirmed by Kai individually (§19.2 items 2–4). `harness.py approve` does not substitute for these confirmations, and a batched approval does not count (critical B16);
- no claim domain chosen by Kai (§19.2 item 12; physics B12). Like items 2–4, it is decided on its own, before any launch, and is not revisited after S2;
- an unsupported budget;
- for S4, missing budget approval from Kai or no chosen option (§8.7);
- for S3 and S4, an empty D_screen (§8.2), until Kai decides.

During training, nonfinite values, corrupt artifacts, data or hash drift, and accounting failures stop the affected run and block claims. Prior valid checkpoints are kept. There is no automatic scientific repair, seed replacement, architecture change or fresh initialization (`AGENTS.md:17-33`; `docs/methodology/03-phases.md:33-44`).

**Chang's stop.** Chang's StopIf (training count < 10,000) is kept with its reason. Poor accuracy is never an early stop. Accuracy ≤ 0.5, empty fronts, zero-width activations or an unused attention branch are logged and the schedule finishes (§8.8).

**PID failure.** If BEST fails under PID, the result stands. No second-best configuration is chosen on held-out data.

## 15. Risks

- **Scale precision.** v0.2's exact β carried 22–24 bits with high probability, which no training pressure could shorten. That decided the H1 comparison before training (§3.5; physics A1).
  - Resolution (a) gives the scale a learned width at Chang's initial 4 bits.
  - Residual risks: the per-column scale term under H2 grows with ⌈log2 n_in⌉ as activations narrow, and a saturated scale at initialization (g > 15/16) distorts the fold until i grows.
  - PREFLIGHT reports H1, H2 and NR per configuration at initialization as engineering readings.
- **Lost pruning channel.** A binary kernel has no zero weights. It can drop whole columns or layers only through its scale (§1). A tight-budget deficit may come from this rather than from the sign map. No ternary control isolates it (physics B5; a ternary control is offered to Kai as §19.2 item 14). The quantities read for this and the other readings are fixed in §8.8, "Reading a deficit or a collapse".
- **The fold, not the sign map.** In both forms the moments normalize x·W, a pre-activation the forward never computes. FB's signs also move with the moments between training and export. A fused-BN deficit may come from the fold. Dead channels and sign differences are logged (§12; physics B6).
- **350k may be unattainable or destructive for binary sites.** Learned activation widths may absorb the squeeze. This is detected and reported (§8.8), with no historical mechanism inferred (`.claude/memory/experiment-log.md:1-7`); the reading table of §8.8 fixes which logged quantity is read for which explanation.
- **Release fidelity is not paper fidelity.** The head count, the λ ramp, loader round trips, runtime precision and the firmware settings all differ. The comparator is the rerun CH-8, with a pre-registered context pull (§8.10, §16).
- **The screen cannot resolve; the confirmation may not be funded, or may be underpowered.** At n=3 no verdict is possible (§6). If S4 is not funded, the campaign answers only the screen question. Under option A, 80 % power over six allowances needs a true gain of about 2.4–2.8 σ_d, and the screen's s_d can understate σ_d by up to about six times (§6).
- **A win is a training-procedure result.** Every exportable binary model is an exportable HGQ model of the same cost (§1). A confirmed gain cannot be read as a cheaper representation.
- **A low-budget win may be an attention-removal win.** A degenerate O makes the attention branch constant, a degenerate K makes the attention weights input-independent, and a degenerate V freezes the values; zero-width softmax entries can prune attention statically (§2.2; chang B8). Such a model is closer to a Deep-Sets-like model, which the paper's Table 1 shows to be cheaper at N=8 (§16). Degenerate layers, binary or HGQ, are recorded by effect and label the verdicts (§8.7), and C6 needs a static-attention ablation (§8.8). DS-HGQ, the control that would show it, is over the cap at r=3; under the default it trains in S3 when the item-9 rule picks it (§19.2 item 9).
- **The whole-domain claim is one the design expects to fail (physics B12).** The design expects a deficit at tight allowances, and C2 over every allowance CH-8 reaches would then return ABOVE with probability at most 0.0125 (A, B) or 0.025 (C alone; C with (c) as B) (§6). A gain over the moderate and loose allowances has a confirmatory route only if Kai chooses the band of §19.2 item 12.
- **The form choice may be noise-driven** if the forms are close. Both forms stay in the held-out descriptive table.
- **H need not rank post-opt LUTs.** DA sharing, scale encoding, dynamic attention and stage all matter. Calibration ratios are recorded (§3.4), and missing mulder builds leave a software-only result (`docs/conventions/fpga-synthesis.md:21-29`).
- **The part may be too small for CH-8** (§9). The capacity order is fixed in advance.
- **Compute estimates may be badly wrong.** The rate is a TF-era estimate. The required screen fits only at r ≤ 3.21 s (planning) or 3.53 s (all-inclusive) (§10).
- **The headline choice is results-informed.** The historical v0.3 pre-result status rests on attestation (§3.4). Kai chose v0.4 after the OPEN-2 outcome was known (06-review §6.3.3). Its whole-model criteria are fixed before any training result; the isolated-layer evidence does not establish that either binary form passes.

## 16. Reference table: context, never a gap

These are the paper's Table 1 rows on its stated 260,000 test jets. Seed count and internal split are unstated, and 350,000 EBOPs was a training target, not a measured cost of every row. The rows used XCU250, not the lab's flow (`literature/jet-tagging-transformers/2510.24784_replication-targets_hgq2-examples.md:26-51,59-80`).

| paper row/model | N | top-1 % | latency ns | LUT k | II | DSP | source / status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 MHA | 8 | 66.3 | 104 | 246 | 1 | 0 | `…replication-targets_hgq2-examples.md:61`; context, with the CH-8 pull of §8.10 |
| 5 Linformer | 8 | 66.3 | 110 | 230 | 1 | 0 | `:65`; context |
| 9 Deep Sets (HGQ) | 8 | 64.7 | 49 | 177 | 1 | 0 | `:69`; context; not the DS-HGQ ablation |
| 4 MHA, collapsed attention | 64 | 77.9 | 44 | 47 | 1 | 0 | `:64,86-90`; not an attention success; context |
| 8 Linformer | 64 | 79.8 | 78 | 202 | 1 | 0 | `:68`; cross-N context |
| 12 Deep Sets (HGQ) | 64 | 79.4 | 44 | 191 | 1 | 0 | `:72`; cross-N context |

No lab record exists for this configuration. The round-14 binary N=8 record is a different model, split and schedule (`docs/conventions/jet-tagging-metrics.md:41-42`). CH-8 is the matched comparator.
- Its deviation from 66.3 % is computed as a pull, with the trigger of §8.10. It is never pre-explained and never closed by retuning (critical B11).
- The paper rows differ in head count, recipe, part and test usage, so 66.3 % is not a binding target under `docs/methodology/06-review.md` §6.8. The §8.10 trigger plays that role for the "beats Chang" wording.

**Related work, context only (physics C18).** No number from these works is compared with this campaign.

| work | what it did | relation to this study | source |
| --- | --- | --- | --- |
| BitNet (H. Wang et al., arXiv:2310.11453) | BitLinear, a drop-in replacement for a linear layer that trains 1-bit weights from scratch, in large language models | the binary map of §2.2 follows its sign and mean-absolute-scale rule at Chang's quantizer site; it is not literal per-layer BitLinear (§2.3) | `literature/bitnet-1bit-ternary/2310.11453_bitnet_scaling_1bit_transformers.md:1-20` |
| J. Ngadiuba et al., "Compressing deep neural networks on FPGAs to binary and ternary precision with hls4ml", Mach. Learn.: Sci. Technol. 2, 015001 (2020), arXiv:2003.06308 (the arXiv listing orders the authors alphabetically) | binary and ternary networks in hls4ml, retaining "full precision on a selected subset of network components", with "similar performance to the higher precision implementation while using drastically fewer FPGA resources" (abstract) | the closest prior selective binarization in hls4ml, on a different model against fixed-precision baselines; this study adds a learned-width HGQ comparator, a transformer and a front | abstract and article pages read in `review/arbiter-v0.5.md` (physics C23, corrected there); a lab literature record is made before any outward use |

## 17. Claims and falsifiers

The screen makes no claims; its outputs (§8.6) are descriptive. Each claim below is marked by stage (critical C24): "S4" claims rest on the confirmation only; "both stages" marks implementation checks that must hold in the screen and in the confirmation. L is the C2 interval level of the chosen S4 option (97.5 % two-sided for A and B, and for C with option (c) of §19.2 item 12; 95 % for C alone; §8.7). D_C2 is the claim domain Kai fixes before any launch: D_band under option (b), the recommended default, or D_conf under (a) (§8.2). Thresholds are in observed s_d, power in true σ_d (§6).

| planned claim | stage | required evidence | falsifier / limit |
| --- | --- | --- | --- |
| C1 the factorial changes only the selected weight rules | both stages (implementation check) | F000/CH-8 parity, CH-8/reference parity, mapped tensors, matched data, RNG (fetched batch indices) and recipe | any unmatched graph, preprocessing, update, batch order or cost invalidates attribution |
| C2 BEST's held-out front lies above CH-8's on D_C2 under H | S4 | `ABOVE ON GRID`: at every g ∈ D_C2, BEST feasible in all n seeds and the lower bound of the paired interval at level L > 0, so the observed mean must exceed 1.563 (A), 1.005 (B, or C with (c)) or 0.836 (C alone) observed s_d at each g; 80 % power over six allowances needs a true gain of about 2.4–2.8 (A), 1.5–1.8 (B) or 1.3–1.6 (C alone; C with (c) as B) σ_d, and only if the gain is positive at every g of D_C2: if it is ≤ 0 at any one, P(ABOVE) ≤ 0.0125 (A, B) or 0.025 (C alone; C with (c) as B) (§6); the size of D_C2 printed, and "front" used only when it holds at least two allowances; under (b) the allowances below g* printed beside and the nested whole-D_conf form reported; the non-headline count beside; degenerate-layer labels (§8.7) | a single g ∈ D_C2 with BEST infeasible, or with upper bound < 0, blocks the claim; `BELOW ON GRID` refutes it; a `LOCAL DEFICIT` (7- or 6-contrast level, §8.7) is reported against it; a win under one count only is accounting-dependent; a verdict resting on a degenerate layer, binary or HGQ, is not a "binary weights" result; an empty D_band gives `INCOMPLETE (no band)` |
| C3 BEST beats CH-8 at 350,000 H under the lab PID | S4, options A and B only | n feasible pairs at the PID selections (§8.6), frozen-H feedback, lower bound of the level-L interval > 0 (observed mean above 1.563 or 1.005 observed s_d) | an upper bound < 0 refutes; infeasible or flat cannot support; under option C there is no C3 claim, and the screen's PID pair is descriptive |
| C3r BEST beats CH-8 at 350,000 H on the ramp front | S4, option C with option (c) of §19.2 item 12 only | n feasible pairs at g = 350,000 on the ramp fronts, lower bound of the 97.5 % interval > 0 (observed mean above 1.005 observed s_d), sharing α with C2 | an upper bound < 0 refutes; infeasible or flat cannot support; not tested unless Kai chooses (c) |
| C4 firmware dominance: at every g ∈ D_C2, BEST has lower LUT and higher post-trace held-out top-1 than CH-8 (renamed in v0.5) | S4 | the §8.7 rule: at every g ∈ D_C2 where C2 is `ABOVE ON GRID`, `LUT ADVANTAGE` (upper bound of the level-L paired LUT interval < 0) and a post-trace accuracy lower bound > 0, at one part, stage and settings, on the stock export route | `NOT A LUT ADVANTAGE` at any g, a post-trace accuracy upper bound < 0, "allowance exceeded after trace", a changed-function export or no post-opt evidence blocks it; mixed parts or missing pairs give `HARDWARE INCOMPLETE`; equal LUT with higher accuracy gives `UNRESOLVED`, which means dominance was not shown, not that the gain was lost |
| C5 the selected weights are binary factors | both stages (implementation check) | S ∈ {−1,+1}; one magnitude q_g(g) per layer or q_g(g_c) per column, or a zero column (pruned or rounded to zero, §2.2); exact export and cost audit | requantized multi-level columns, a dropped scale or an unpriced multiply invalidates |
| C6 attention is functionally used | S4 (secondary, ordinary 95 %) | at BEST's frozen points, per g ∈ D_conf and overall at every g ∈ D_C2: the mean-ablation Δ_attn, static-attention Δ_static and zero-logit Δ_unif intervals > 0, H_attn < log 8 in every seed (consistency check), and no degenerate K, V or O in any seed (§2.2, §8.8); a degenerate Q is reported as "used, with input-independent queries", or as "used, as query-independent attention pooling" where Q_t is identical across t, and a Q equal to 0 everywhere as for a degenerate K (v0.6: critical C42, chang C35); downstream WRAP overflow counts beside each Δ, with the label "overflow-affected" when nonzero (chang C36); DS-HGQ context | any condition failing forbids calling the result an attention benefit; a fixed attention pattern (Δ_static interval not above zero) is not attention use |

These are proposed claims, not findings (`CLAUDE.md:20-26`; `docs/methodology/03-phases.md:123-138`).

## 18. Conventions compliance

| convention | status |
| --- | --- |
| metrics: splits | will implement with Chang's 90/10 split (n_val = 62,000, derived), a declared departure from the lab's 80/20 (`jet-tagging-metrics.md:17-20`), required by D4 |
| metrics: headline metric | departure by Q14: top-1 primary; macro AUC and per-class AUCs reported beside (`jet-tagging-metrics.md:27-34`) |
| metrics: seeds and intervals | will implement: the screen (3 seeds) is labelled screen and descriptive; claims need S4's fresh seeds (5 under option A, 8 under B and C) with paired t intervals (§8.7) and stated power (§6); AUC descriptive (the eight-seed rule for gaps under 0.005 AUC is met only under B and C, and AUC is not a pre-registered claim) |
| metrics: working point | will implement, descriptive (`jet-tagging-metrics.md:35-36`): per-class background rejection at ε_S = 0.5 on the held-out test set (§8, §8.6; constructive K6). The Russell-plan Q11 recommendation also lists a fixed-rate working point (`docs/design/2026-10-11-russell-questions-recommendations.md:360`) |
| metrics: labelling, checks 1–6 | will implement; check 3 (reproduce a record) is not applicable, since no record exists. F000/CH-8 parity and the §8.10 pull replace it |
| cost: native eBOPs | will implement for CH-8, F000 and HGQ sites. The headline is a named custom count (H1 or H2) and NR is labelled as the native rule at binary sites; neither is called native (`quantization-and-cost.md:21-25`). Zero- and random-input agreement is required for H and NR |
| cost: selection under a budget | will implement with top-1 per Q14; "no feasible checkpoint" gives NA |
| cost check 1 (two symmetric values) | declared adaptation, flagged for Kai (critical C10; §19). It applies per output column at fused-BN sites, because the BN scale (Q3) gives column magnitudes in both forms (D9), and per layer elsewhere. A column or layer whose scale quantizes to zero, pruned or rounded to zero (§2.2), is recorded with its route and is exempt from "two nonzero values". The regression trigger `06-review.md:239` is read the same way |
| cost checks 2–4 | will implement; the schedule is Chang's 7,000 epochs, labelled |
| FPGA synthesis | will implement: stages labelled, fidelity on 4,096 validation jets, RF and II stated, numbers from report files, synthesized checkpoint equals the selected one, part capacity handled per §9 |
| figures | not applicable at STUDY |

## 19. For Kai: delegated defaults, open decisions and uncertain owner calls

### 19.1 Orchestrator defaults under Kai's 2026-10-17 delegation, for his review

Each is adopted now and is reversible by Kai. The recommended default is the adopted one.
- **[OD-3]** Three paired seeds per screen arm. Default: as adopted.
- **[OD-4]** Step envelope, grid and control checkpoint policy. Default: as adopted.
- **[OD-5]** Fresh PID runs, λ₀ = 2e-8, released gains. Default: as adopted.
- **[OD-6]** W8A8 at eight total bits, per value, with trained steps. Default: as adopted.
- **[OD-7]** GR adds only the scale derivative. Default: as adopted.
- **[OD-9]** JAX, float32 matmul, pinned stack. Default: as adopted.
- **[OD-10]** Synthesize all deciding points within the 7-day window. Default: as adopted.
- **[OD-8] (resolution c)** Cap 400 GPU-h including 50 reserve; under the item-9 default, controls are dropped in the reverse of the §10 admission order; under tier order, BE, GR, DS-HGQ, W8A8, FP32 (critical C40). Default: keep, and see 19.2 item 9 for the alternative and the use of the one control slot.
- **[OD-11]** The chosen-form pre-stage runs are the factorial cells, with no rerun. Default: reuse.
- **[OD-12] (resolution e)** Q16's "350,000 EBOPs" is read as 350,000 in H, a dated delegated reading. Default: H units, with native reported beside. It amends D6, so Kai confirms it individually before any launch (§19.2 item 4).
- **[OD-13] (resolution a)** Each binary scale passes a learned-width HGQ quantizer with Chang's weight configuration. In BF it acts after the fold, on g_c = β|s_c|.
  - Default: as adopted. It amends D3 and fixes BF's placement inside D9, so Kai confirms it individually before any launch (§19.2 item 2).
  - Alternatives, from physics A1: (ii) exact β, with H1 dropped as the fail-closed headline and a per-channel threshold count H3 added at the four fused-BN sites; (iii) exact β, with the question restated as "binary signs with exact float32 scales".
- **[OD-14]** Collapse δ = 0.05 top-1, with a reference window [0, 2,000) and a squeeze window of 1,000 epochs. Default: as adopted.
- **[OD-15] (resolution b)** A three-seed screen with no verdicts; verdicts only from S4. Default: as adopted. It changes D1's claim structure, so Kai confirms it individually before any launch (§19.2 item 3); S4's size is §19.2 item 11.
- **[OD-16] (resolution d)** The surrogate K = q_g(stop_gradient(g))·S + (T − stop_gradient(T)) gives an exact binary forward with the identity Jacobian of Q4, and is not a change of Q4. Default: as adopted.

### 19.2 Decisions only Kai can make

Each is put to Kai one at a time, with the recommended default and what each choice excludes. Items 2–4 and 12 remain open launch preconditions after v0.11 (§14): a batched approval does not count for them, and `harness.py approve` does not substitute (critical B16; Kai decides each design choice himself, one at a time). The numbering of items 1–11 is unchanged from v0.4; item 12 is new in v0.5 (Appendix A.1). Items 13–16, new in v0.7, are the constructive review's design proposals P1–P4 (`review/constructive-v0.5.md` §4). None is adopted, none is a launch precondition, and an unanswered item leaves the current design in place; items 13 and 15 act only if decided before S1. An item of 13–16 adopted after the protocol freeze enters as a dated amendment with its `protocol-draft.json` change, checked by the critical reviewer at the PREFLIGHT review, as for item 12 (c), and the protocol is frozen again after it; unanswered, the frozen protocol stands. Items 13, 14 and 15 lapse if unanswered by the PREFLIGHT build of the arm generator and harness admission, and item 16 at the first launch (constructive v0.7 B3, v0.8).
1. **NRP launcher change — resolved (Kai, 2026-10-18 and 2026-10-19).** Route A2: git bundle on `kai-data`, `run_handoff.py` bundle mode, no ConfigMaps. All NRP campaign state remains inside that PVC (§11). Source: `decisions.md` [dec 10-18 H2], "Kai on the plan", and [dec 10-19 PVC], "Kai (chat, verbatim)"; verbatim words in §0.
2. **Confirm OD-13, which amends D3 and fixes BF's quantizer placement inside D9.** Each binary scale passes a learned-width HGQ quantizer with Chang's weight configuration; under BF it acts after the fold, on g_c = β|s_c| (§2.2–2.3). Recommended: as adopted. It excludes exact float32 scales and the H3 alternative (exact β with a per-channel threshold count at the four fused-BN sites; §19.1).
3. **Confirm OD-15, which changes D1's claim structure.** The funded three-seed stage is a screen with no verdicts; verdicts come only from a fresh-seed confirmation that Kai funds and sizes (item 11). Recommended: as adopted. It excludes verdicts from the funded screen.
4. **Confirm OD-12, which amends D6.** Q16's "a PID run at 350,000 EBOPs" is read as 350,000 in the headline count H, with native reported beside (§7.2). Recommended: as adopted. It excludes a native-unit PID target.
5. **Headline route — superseded by Kai's v0.4 rule (2026-10-18).** The clean-rerun choice and its interruption alternatives no longer decide launch or the headline (§3.4 Addendum A). Source: `decisions.md` [dec 10-18 NRP], "Kai (chat, verbatim)", and [dec 10-18 H2], "Kai (chat, verbatim)" (§0). H2 requires the whole-trained-model factored-export check; H1 otherwise. The check's pilot, specification and checkpoint rule were fixed before the build in `preflight/export/PLAN.md:44-53` and are recorded in §3.4 v0.4; the pilot (F111-FB, F111-BF and CH-8) was trained on the home PC before v0.11. No mid-run penalty change or altered §8 form-selection rule is implied. The items below do not block the count-independent CH-8 and F000 route; no binary arm launches before the check has an outcome (§14). The "INLINE route" item is resolved: both routes receive the identical patch for every model (§9). The split-form and F100 treatment is resolved (Kai, 2026-10-19T16:28:12Z, before the outcome; [dec 10-19 both]; §3.4 v0.4): both forms pass, H2 for every binary arm, F100 included; both fail, H1 for every binary arm; one passes, every binary arm in both forms under H1, plus H2 runs of the passing form's binary arms (F100, its six fused-BN configurations, three seeds, and the PID runs of its best configuration; 24 runs). It supersedes the orchestrator's 12:20:19Z default ([dec 10-19 split]). **Open within that decision, for Kai:**
    - how the one-pass H2 runs enter §8's selection and reading, which the decision does not state; §8 is unchanged until Kai answers, and the answer is needed before those runs launch;
    - one-pass admission (critical v0.11 B7; v0.13): none of the 24 added H2 runs is admitted until Kai has answered the item above and confirmed the exact cap, 531.25 GPU-h with the reserve (derived, §19.2 item 8), against the recorded "about 530". Unless Kai says otherwise, they use the screen's seeds 1–3, paired as in §6, and their admission and drop order and the control slot follow §10's tier order. `protocol-draft.json` gains their entry only if the branch occurs, by a dated amendment then;
    - the timing route of criterion 2 (critical v0.11 B2): required of the factored route only, with the normal route's timing recorded beside it (`PLAN.md:52`). Alternative: timing required on both routes, as v0.3 condition 4 required it of the normal route (§19.3);
    - hardware evidence under H2 at the deciding points (critical v0.11 B6): whether the binary deciding points also get factored builds when H2 is adopted. Default: C4 unchanged on the normal route; factored builds at deciding points not required; any that are made are reported beside C4, labelled context, and decide nothing.
6. **Mulder permission — export check permitted (Kai, 2026-10-18), later synthesis asked again.** The permission in [dec 10-18 H2], "Kai (chat, verbatim)" (§0), covers the whole-model export check that qualifies H2, including its export environment and required hardware checks (§3.4 v0.4, §9). It does not reinstate the clean rerun as a launch precondition for any arm, or the export gates for CH-8 and F000; binary arms reach the export gates through the v0.4 check (§14). Permission for the later deciding-point synthesis window is asked again.
7. **JAX runtime image — resolved by the PVC venv route.** The approved plan installs the hashed lock into a venv on `kai-data`, used by route A2 (§7.3, §11, §13); no pushed campaign runtime image is required. Source: [dec 10-18 H2], "Kai on the plan", [dec 10-18 NRP], "Orchestrator ops defaults", and [dec 10-19 PVC]; verbatim approval and PVC words in §0.
8. **A cap or scope change** if the measured rate exceeds about 3.21 GPU-s per run-epoch at the planning rate, or 3.53 all-inclusive (§10). Recommended: a cap increase sized to the measured required screen, because cutting factorial or pre-stage cells would change the design. Alternative, written now so that it is ready if the measured rate is high (constructive K7, v0.7): choose the form on F011 and F111 only (both forms, 12 runs) and train F001, F010, F101 and F110 in the chosen form only (12 runs). The required screen is then 39 runs (3 CH-8, 3 F000, 3 F100, 12 form-pilot, 12 chosen-form, 6 PID), or 250.25 GPU-h at r=3 (300.25 with the reserve). It fits the 350 training GPU-h up to r ≈ 4.20 s at the planning rate, or 4.62 s all-inclusive, against 3.21 s and 3.53 s now. A variant on F011 alone gives 36 runs, 231.00 GPU-h, and fits up to r ≈ 4.55 s (5.00 s all-inclusive). Either variant changes how D9 is carried out, and OD-11; adds one wave in sequence before S2; and leaves the FB-BF comparison on one or two configurations instead of six (derived from §10's formula, runs × 7,000 × r × 1.10 / 3,600, and without the 1.10 all-inclusive). Adopting it as the default before S1 is item 13.
    - **Conditional cap note (Kai, 2026-10-19T16:28:12Z, before the outcome; [dec 10-19 both]; §3.4 v0.4).** If the v0.4 check passes for exactly one form, the passing form's binary arms also train under H2: 24 more runs, about 154 GPU-h at the planning rate (derived: 24 × 6.416667 = 154.00), so the required screen becomes 75 runs and about 481.25 GPU-h, or 531.25 with the 50 GPU-h reserve (derived from §10's 327.25 and 377.25), and the cap rises from 400 to about 530 GPU-h in that case only (the decision's wording). §10's figures are unchanged and stand when both forms pass or both fail. The thresholds of this item (3.21 and 3.53 GPU-s per run-epoch) are computed for the 51-run screen and are not recomputed here.
9. **Q11 controls under the cap (critical B10, C27, C33; physics B10; v0.5: critical B18).** At r=3 the 400 GPU-h cap admits one control. Keeping W8A8, DS-HGQ, GR and BE as well needs +73.5 GPU-h, a cap of about 475 GPU-h including the reserve. Recommended: keep 400 GPU-h and hold the one slot until S2; its control then trains in S3. The slot goes to DS-HGQ if BEST has a degenerate attention layer, binary or HGQ (Q, K, V or O, §2.2), at any frozen point of any screen seed, and otherwise to FP32. DS-HGQ is the control that decides whether attention removal explains a low-budget result; it trains on the screen seeds, so for an S4 verdict it is descriptive. This excludes W8A8, GR, BE and one of DS-HGQ and FP32 at r=3. Alternative: a cap of 475 GPU-h, which keeps all five controls at r=3, with FP32, W8A8 and DS-HGQ in S1. Kai's answer is requested before the S1 launch; while it is open, the slot is held, which is the default, and nothing in S1 uses it. The tier order is OD-8 and stays Kai's call. Under the form-subset alternative of item 8 at r = 3, all five controls fit: 54 runs and 396.50 GPU-h with the reserve (39 + 15), or 51 runs and 377.25 GPU-h with F011 alone, which equals today's required screen (derived; constructive K7, v0.7).
10. **`harness.py approve`** for this campaign's authority (`decisions.md` [dec 10-17 delegation], sub-bullet "Hard stop kept"). Recommended: run it after STUDY PASS, items 2–4 and 12 and the applicable non-export PREFLIGHT gates. Admission of any binary arm, under either count, also requires a recorded outcome of the §3.4 v0.4 check (§14); the superseded item-5 clean rerun is not required.
11. **Confirmation (S4): funding and size (physics B8; §8.7).** Decided after S2's validation outputs, which include each option's threshold and power at the measured validation s_d and its plug-in power at BEST's observed per-allowance gains, labelled optimistic (§8.6), and the same quantities at the pooled σ̂_d of §8.6, whose 95 % band spans about a factor of two where the three-seed band spans about twelve (derived: 1.522/0.745 and 6.285/0.521; constructive K1, v0.7), with BEST's null selection reference beside (constructive K2), and before any held-out read. Every option's power assumes a positive gain at every allowance of the claim domain D_C2; if the true gain is ≤ 0 at any one of them, P(ABOVE) is at most 0.0125 (A, B) or 0.025 (C alone; C with (c) as B), whatever the size (§6; physics B12). The claim domain itself is item 12.
    - A, as adopted: BEST and CH-8, ramp and PID, seeds 4–8. 20 runs, +128.33 GPU-h at r=3 (+256.67 at r=6). C2 and C3 at 97.5 %. 80 % power over six allowances needs a true gain of about 2.4–2.8 σ_d.
    - B: the same four arms on seeds 4–11. 32 runs, +205.33 (+410.67). C2 and C3 at 97.5 %; about 1.5–1.8 σ_d.
    - C: BEST and CH-8, ramp only, seeds 4–11. 16 runs, +102.67 (+205.33). C2 alone at 95 %; about 1.3–1.6 σ_d. The PID comparison, the second part of Q16, stays a descriptive screen output rather than a claim. Under option (c) of item 12, C2 shares α with a ramp point claim at 350,000 (C3r): threshold 1.005 s_d, power as under B.
    - Recommended: fund it as option C, the cheapest option with the most power for C2, the thesis claim. If Kai wants the PID comparison as a claim, B rather than A, because A's power for C2 is low. Because C changes what Q16's PID run supports, the choice is his.
    - Not funding S4 excludes every claim; the campaign then reports the screen only. Further alternatives: the form-subset alternative of item 8, which recovers 77.0 or 96.25 GPU-h at r=3 (derived: 12 or 15 runs; constructive K7) and changes D9; or a cap change.
12. **Claim domain of C2, decided before any launch (physics B12; §8.2, §8.7).** The design expects a deficit at tight allowances (§1, §3.5). C2 over every allowance CH-8 reaches would then return ABOVE with probability at most 0.0125 (A, B) or 0.025 (C alone; C with (c) as B) under any S4 option (§6), and a gain at moderate and loose allowances would have no confirmatory route.
    - (a) C2 over all of D_conf, the thesis as written. Under the design's expectation the likely S4 outcome is FLAT/UNRESOLVED or BELOW, whatever its size, so the S4 funding recommendation is then conditioned on S2 showing a positive mean validation gain at every g ∈ D_screen.
    - (b) A pre-registered band: D_band = {g ∈ D_conf : g ≥ g*}, where g* is the smallest g ∈ D_screen such that, at every allowance of D_screen at or above it, BEST is feasible in all three screen seeds and its mean paired validation gain over CH-8 is positive (§8.2). The rule is fixed now and applied in S2 on screen validation only, as BEST is; the confirmation seeds take no part. C2 is tested on D_band with the same intersection-union test; the allowances below g* are printed beside it, and the whole-D_conf form is reported beside as the strong form, at no added α. The thesis sentence then reads "a higher … front at the grid allowances of D_band, from its lowest member to 450,000 H"; with one allowance, "a higher top-1 at H = ⟨g⟩" (physics C24, chang C41). §8.2 is amended to admit this one rule. An empty band gives `INCOMPLETE (no band)`. g* is a three-seed sign filter. An allowance enters the band with probability about 0.50, 0.81 or 0.96 at a true gain of 0, 0.5 or 1.0 σ_d (derived, normal approximation). Every nonempty band contains 450,000, so (b) gives a route only to a gain that extends to the top of the grid (physics C20). Two alternative rules, a stricter inclusion criterion and an interval band, are written in full in §19.3.
    - (c) A conditional addition: if option C is chosen under item 11 (after S2), a ramp-front point claim at g = 350,000 (C3r) shares α with C2, both at 97.5 %. It raises the n = 8 threshold from 0.836 to 1.005 s_d and restores a point claim at Q16's budget without the PID arms. It can be combined with (a) or (b) and has no effect under options A and B. When 350,000 ∈ D_C2, C3r's null implies C2's null, so C3r could be tested at C2's own level with no added α, as the strong form is nested (physics C21). This variant is offered to Kai under (c), with C2 at 95 % and 0.836 s_d under option C. It applies only if 350,000 ∈ D_C2, which is fixed on validation before the held-out read. The pre-registered default remains the Bonferroni split. If Kai chooses the nested variant, the designer writes it as a dated amendment before any launch, and the critical reviewer checks it at the PREFLIGHT review against the frozen protocol (arbiter v0.5 §5). That amendment is weighed against the iteration warning of `review/arbiter-v0.5.md` §9.
    - Recommended: (b), with the whole-D_conf result reported beside as the strong form, and without (c). (b) excludes a confirmatory claim over the allowances below g*, which are reported but carry no claim. (c) is left out of the default because under option C and the pre-registered Bonferroni split it lowers C2's power to option B's level; Kai may add it if he wants a claim at Q16's 350,000 without the PID arms. (b) changes the thesis wording, so the choice is Kai's alone. It is fixed before any launch and not revisited after S2.
    - **Revisited in v0.6 in the light of physics C21 (CONFIDENCE: MEDIUM).** The recommendation is unchanged. The nested variant removes (c)'s power cost only when 350,000 ∈ D_C2. In that case an `ABOVE ON GRID` for C2 already covers 350,000, so C3r adds a claim only where C2 fails at another allowance. When 350,000 ∉ D_C2, the split, and so the power cost, returns. If Kai adds (c), the nested variant is recommended over the split, because it keeps C2 at 95 % and 0.836 s_d whenever 350,000 ∈ D_C2.
13. **Form chosen on F011 and F111 (constructive P1, v0.7; useful only if decided before S1).** Should the form be chosen on F011 and F111, the two configurations that hold all four fused-BN sites, with F001, F010, F101 and F110 trained in the chosen form only (item 8's alternative, 39 runs)? It changes how D9 is carried out and OD-11, supersedes item 9's one-slot premise (at r = 3 all five controls fit, 396.50 GPU-h with the reserve), splits S1 into S1a (21 runs) and S1b (12 runs), adding one wave, and restates §8.3 for two configurations; the amendment costs about 1.5 designer-hours and 0.5 reviewer-hours. Default if unanswered: not adopted, and the full pre-stage stands. Recommended: adopt, if decided before S1, because the rate is the largest operational risk ([L1]) and Q11 asks for as many controls as possible, DS-HGQ among them; it excludes the FB-BF comparison on the other four configurations. CONFIDENCE: MEDIUM.
14. **A ternary control at BEST (constructive P2, v0.7).** BEST's binary groups trained with BitNet b1.58's absmean ternary map ({−1, 0, +1} times the same scale quantizer; `literature/bitnet-1bit-ternary/2402.17764_era_of_1bit_llms_1p58.md:17`), seeds 1–3 in S3, 3 runs, 19.25 GPU-h at r = 3 (derived). It is the control that would isolate the pruning channel (§1), but it lies outside Q15's binary factorial, needs a new map, a three-value check and an export gate (about 3–5 ml-engineer hours and 1 designer-hour), and fits beside all five controls only with item 13 on F011 alone (54 runs, 396.50 GPU-h with the reserve, derived). Decided before the PREFLIGHT build. Default if unanswered: not run. Recommended: not in this campaign; for the next one, because it is outside Kai's binary scope. CONFIDENCE: MEDIUM.
15. **Hold the last S1 wave for a CH-8 validation check (constructive P3, v0.7).** CH-8's three seeds and F000 run in the first wave. Before the last S1 wave is admitted, CH-8's seed-mean validation top-1 at its released-callback point with training-count native ≤ 350,000 (the §8.10 quantity, on validation) is read; if it lies more than m_s below 66.3 %, the remaining waves are held and Kai decides, with nothing repaired automatically. It is an operational sentinel across splits, never a result, needs no runs and about 0.5 agent-hours (§10 admission, §14). Decided before S1, with m_s. Default if unanswered: not adopted, and §8.10's pull governs at the held-out read. Recommended: adopt, with m_s = 2 percentage points as a placeholder Kai fixes (not derived; it must also cover the unmeasured offset between validation and the paper's held-out row), because it finds a comparator that does not reproduce before most of the screen's GPU-hours are spent. CONFIDENCE: MEDIUM.
16. **An accuracy non-inferiority reading when C2 is not ABOVE (constructive P4, v0.7).** When C2 is not `ABOVE ON GRID`, the same level-L intervals are read against −m: `WITHIN m OF CH-8 ON D_C2` if every lower bound exceeds −m; descriptive, no α, no claim. m is fixed before any launch; candidates are 0.5 percentage points or the 1-point floor the lab already uses for a reproduction deviation (§8.10); about 0.3 agent-hours. Default if unanswered: not adopted. Recommended: adopt with m = 1 percentage point, because a null confirmation should still state how much top-1 the binary groups cost at matched H, bounded by a margin fixed in advance. CONFIDENCE: MEDIUM.

v0.2's item 9 asked for a restricted front domain if no configuration was eligible. It is removed: the evaluation domain is fixed from CH-8 alone (§8.2). The band of item 12 is different: a claim domain chosen by a rule fixed before any run exists, not a fallback chosen after coverage is known.

### 19.3 Uncertain owner decisions

These use the format of `docs/methodology/03-phases.md:49-57`.
- **Headline rule (§3.4 v0.4), recorded decision and remaining scope.**
  - DECISION: Kai chose the factored whole-trained-model check after the OPEN-2 outcome was known; v0.3's "2b not attempted" provision and Addendum A's adoption path are superseded. The INLINE route is resolved as the same patch in both routes for every model (§9).
  - OPEN: the split-form and F100 treatment is decided by Kai before the outcome ([dec 10-19 both], 2026-10-19T16:28:12Z; it supersedes the [dec 10-19 split] default); open are how the one-pass H2 runs enter §8's selection and reading; the timing route of criterion 2 (below); and hardware evidence under H2 at the deciding points (§19.2 item 5). The pilot, specification and checkpoint rule are fixed (`preflight/export/PLAN.md:44-53`). No new selection rule is adopted.
  - CONFIDENCE: the decision is recorded; its whole-model outcome is unverified. FLAG FOR HUMAN: YES (the unresolved scope only).
- **Timing route of v0.4 criterion 2 (§3.4 v0.4; critical v0.11 B2).**
  - DECISION: timing (WNS ≥ 0 in `post_opt_timing.rpt` and II = 1) is required of the factored route only; the normal route's timing is recorded beside it (`preflight/export/PLAN.md:52`). This moves v0.3 condition 4 from the normal (stock) route to the factored route.
  - ALTERNATIVES: timing required on both routes. The route matters on the existing evidence: in the isolated-layer per-position test with the INLINE patch (`preflight/da4ml-test/perpos-oneblock/results/table.json`, sha256 `93614f4d34398d9d694db210ad9afb0f85c800861d81aa5cf1b99845a20847c7`, field `postopt_WNS`; engineering, one block of 8 positions, not whole-model evidence, not quotable), the five INLINE factored builds have post-opt WNS from 0.411 to 0.752 ns, while the INLINE normal builds have −0.677 ns in three of the four flat scale cases (NP2w8, NP2w24, UNI8; P2 0.95 ns).
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: YES (§19.2 item 5).
- **PID sensor = frozen H (§7.2).**
  - DECISION: as stated.
  - ALTERNATIVES: the stock training-count sensor.
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: YES.
- **Evaluation domain from CH-8 alone, and coverage-first BEST (§8.2, §8.4).**
  - DECISION: as stated. The claim domain within it is Kai's (§19.2 item 12).
  - ALTERNATIVES: v0.2's all-g eligibility, which removes binary configurations that miss low allowances.
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: YES.
- **The g* rule of the claim band (§8.2; §19.2 item 12 (b); v0.6: physics C20).**
  - DECISION: as written in §8.2. g* is the smallest g ∈ D_screen such that, at every g′ ∈ D_screen with g′ ≥ g, BEST is feasible in all three screen seeds and BEST's mean paired validation top-1 gain over CH-8 is > 0. It is a three-seed sign filter: an allowance enters with probability about 0.50, 0.81 or 0.96 at a true gain of 0, 0.5 or 1.0 σ_d (derived, normal approximation). Every nonempty band contains 450,000, because CH-8's and BEST's feasibility are monotone in g, so D_screen and D_conf are upper sets of G.
  - ALTERNATIVES. Each is fixed now and applied in S2 on screen validation only, as the default is; the confirmation seeds take no part.
    - (i) A stricter inclusion criterion. g* is the smallest g ∈ D_screen such that, at every g′ ∈ D_screen with g′ ≥ g, BEST is feasible in all three screen seeds and BEST's mean paired validation top-1 gain over CH-8 at g′ is above one validation s_d at g′ (the sample sd, ddof = 1, of the three paired seed gains there). D_band = {g ∈ D_conf : g ≥ g*}. If no g qualifies, the band is empty and C2 is `INCOMPLETE (no band)`. The thesis wording is the default's.
    - (ii) An interval band. [g_lo, g_hi] is the longest run of consecutive allowances of D_screen, in the order of G, at each of which BEST is feasible in all three screen seeds and BEST's mean paired validation top-1 gain over CH-8 is > 0. Equal-length runs go to the one with the larger allowances. D_band = {g ∈ D_conf : g_lo ≤ g ≤ g_hi}. If no allowance qualifies, the band is empty and C2 is `INCOMPLETE (no band)`. The allowances of D_conf outside the band are printed beside C2, labelled outside the claim. The thesis wording is "for g_lo ≤ H ≤ g_hi". Unlike the default, this band need not reach 450,000, so a gain confined to the middle of the grid has a confirmatory route.
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: YES (§19.2 item 12; the choice is Kai's, and the default stays as written).
- **Form rule (§8.3).**
  - DECISION: coverage on D_screen, then mean paired validation top-1, ties to FB.
  - ALTERNATIVES: a joint (form, configuration) maximum of the BEST score.
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: YES.
- **Confirmation test (§8.7).**
  - DECISION: an intersection-union test for C2 per g at level L (97.5 % shared with C3 under options A and B; 95 % for C2 alone under C); descriptive labels at 7 or 6 contrasts.
  - ALTERNATIVES: all verdicts at 7 contrasts (2.266 s_d at n=5); C2 and C3 each at their own α (1.242 s_d at n=5); pooled seeds 1–8 at 91 contrasts (2.117 s_d).
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: YES.
- **Descriptive label definitions (§8.7 :1041-1044; PREFLIGHT B3b-22; v0.9).**
  - DECISION: `CROSSING`, `LOCAL ADVANTAGE` and `LOCAL DEFICIT` are read on the 7- or 6-contrast simultaneous intervals at g ∈ D_conf. A COST INFEASIBLE g counts as a deficit (§8.2), and C2's `ABOVE ON GRID` or `BELOW ON GRID` (level L, over D_C2) is the whole-front criterion.
  - ALTERNATIVES: (i) read over D_C2, in which case a deficit below g* under option (b) of §19.2 item 12 cannot be labelled; (ii) v0.2's reading, with the whole-front criterion on the same 7- or 6-contrast intervals, in which case `LOCAL ADVANTAGE` can be printed beside a C2 `ABOVE ON GRID`; (iii) level-L intervals for the labels, which contradicts :1040.
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: YES.
- **Confirmation size (§8.7; physics B8).**
  - DECISION: recommend option C (ramp only, seeds 4–11, 16 runs).
  - ALTERNATIVES: A as adopted (20 runs; 80 % power over six allowances at about 2.4–2.8 σ_d); B (32 runs; 1.5–1.8 σ_d; keeps C3).
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: YES (§19.2 item 11; C demotes the PID comparison from a claim).
- **C4 decision rule and name (§8.7; physics B11; v0.5: critical C37, physics C16).**
  - DECISION: per g, the level-L paired t interval on LUT_BEST − LUT_CH-8; C4 needs `LUT ADVANTAGE` and a post-trace accuracy gain at every g of D_C2; named "firmware dominance", because that is what the rule tests.
  - ALTERNATIVES: "all n pairs lower" as a sign rule, which cannot reach one-sided 0.0125 at n = 5 (best p = 1/32) but can at n = 8 (1/256); a non-inferiority margin on LUT, which would credit higher accuracy at equal LUT but needs a margin fixed now that nothing in the lab grounds.
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: YES.
- **C6 rule (§8.8; physics B10; v0.5: chang B8, physics C17, critical C31, C32).**
  - DECISION: "used" needs the mean-ablation, static-attention and zero-logit ablation intervals above zero, H_attn < log 8 in every seed on row-renormalized probabilities (a consistency check), and no degenerate K, V or O by effect; a degenerate Q gives "used, with input-independent queries", or "used, as query-independent attention pooling" where Q_t is identical across t, and a Q equal to 0 everywhere is reported as for a degenerate K (v0.6: critical C42, chang C35; no C6 outcome changes); an overflow-affected Δ carries its label (chang C36); reported per g, "used" overall at every g of D_C2.
  - ALTERNATIVES: v0.4's four conditions, superseded because a static non-uniform pattern passes them; dropping Δ_unif, since Δ_static carries the test of input dependence; v0.3's mean ablation alone; an entropy threshold below log 8, which needs an arbitrary margin.
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: YES.
- **Degenerate layers are labelled, not excluded (§8.7; physics B10; v0.5: critical B18, chang B8).**
  - DECISION: verdicts resting on a degenerate layer, binary or HGQ, defined by effect, carry the label "with degenerate ⟨layer⟩ (⟨route⟩)"; selection is unchanged.
  - ALTERNATIVES: exclude checkpoints with a degenerate attention layer from the binary fronts before any run, which would make attention removal ineligible as a route to a win.
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: YES.
- **BF scale quantizer after the fold (§2.3, part of OD-13).**
  - DECISION: quantize g_c = β|s_c|.
  - ALTERNATIVES: quantize β before the fold, which leaves float32-width constants.
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: YES.
- **Binary-value check with zero columns (§18; critical C10).**
  - DECISION: per column at fused-BN sites, per layer elsewhere, with zero columns recorded as pruned.
  - ALTERNATIVES: treat a zero column as a check failure.
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: YES.
- **CH-8 pull trigger (§8.10).**
  - DECISION: max(2 × CH-8 seed sd, 1 percentage point).
  - ALTERNATIVES: 2 × sd only.
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: YES.
- **Part-capacity order (§9).**
  - DECISION: xczu7ev above 100 % reported as is; VU13P only with a passing licence probe; otherwise HARDWARE INCOMPLETE.
  - ALTERNATIVES: VU13P C-synthesis as the deciding stage, which is an estimate.
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: YES.
- **Export environment (§9; chang B7).**
  - DECISION: the training pin (Keras 3.14.0, hgq2 0.1.9 with the two shims, quantizers 1.2.2) on JAX on CPU, plus hls4ml 1.3.0, the MHA alias and the da4ml version option 0 fixes, built on mulder.
  - ALTERNATIVES: mulder's `m3-env` with three declared Keras-3.15/TensorFlow shims for export (fused-BN unpack, `full_output_shape`, the explicit-label EBOPs einsum); a TensorFlow backend in the export environment with the explicit-label EBOPs einsum declared.
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: YES (conversion under JAX is untested here; it is Chang's own route (`run_test.py:3`); a failure is a PREFLIGHT finding).
- **Stack pin (§7.3).**
  - DECISION: Keras 3.14.0 and hgq2 0.1.9 with two shims.
  - ALTERNATIVES: Keras 3.15 with five shims.
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: NO.
- **H2 accumulator width (§3.2).**
  - DECISION: the sign-pattern-free bound.
  - ALTERNATIVES: the exact format from the sign pattern.
  - CONFIDENCE: MEDIUM. FLAG FOR HUMAN: NO.

### 19.4 Unverified rather than invented

The following list records the earlier design's unverified inputs, with the v0.11 headline-check entries updated. The dated staging record in `decisions.md` does not independently verify the other inputs here:
- source hashes or a software lock;
- a parameter count;
- a data manifest;
- throughput or K;
- a top-1 seed sd;
- a verified whole-trained-model v0.4 export-check outcome; the v0.11 amendment read the OPEN-2 outcome in the decisions and the two engineering READMEs, as recorded in Addendum B. The pilot has been trained and criterion 1 met on the owner's record (`preflight/export/PLAN.md:67-71`, not verified here); the timing and LUT outcomes were not read for v0.12;
- calibration ratios;
- a working H-LUT export, or the export environment of §9;
- a trained front;
- a functioning PID;
- new synthesis;
- the xczu7ev LUT capacity, which PREFLIGHT reads from the post-opt report or the part property (§9);
- Kai's S4 option, his claim-domain choice (§19.2 item 12), and the remaining v0.4 timing-route, deciding-point and one-pass H2-run decisions (§19.2 item 5). The clean da4ml rerun is no longer required for launch or headline adoption.

**A PREFLIGHT directory appeared during this revision.** `preflight/env/` was created at 07:44 on 2026-10-17 and last changed at 07:58 (critical C43; v0.5 gave 08:00). Its parent listing shows `lock/`, `logs/`, `results/`, `scripts/`, `README.md` and `MANIFEST.sha256`; `results/` includes files named `init_readings_cpu.json` and `init_readings_gpu.json`. The author of v0.5 did not open it, so a software lock or initialization readings may exist, but nothing in them is cited or used here: a design revised after reading initialization costs would be design after numbers. Reconciling it with the phase order of `docs/methodology/03-phases.md` is the orchestrator's. The orchestrator's record of what it is, under which authority it ran and who has seen its readings is in §3.4 Addendum B (v0.6).

The Jev MCP server failed to connect in this session too (ENOENT on its runtime interpreter), so `jev_check_methods`, which calls a remote model, was not run. The deterministic protocol check was run in memory with the same function `lab_check_protocol` uses (`tools/jev/methods.py` `check_protocol`, digest from `tools/jev/common.py`; `python3 -I -B`, nothing written; `local/jev-audit/` unchanged at v0.7, newest file 2026-10-09; not inspected in v0.8):
- the v0.8 `protocol-draft.json` (scope screen, 66 arms): protocol sha256 `3b25814bdba31978566021b6610a8d668521c598465531111ce721636525e46d`;
- its `confirmation_stage` (option A, scope claim, 20 arms): `c16e01003758294cfa3fe981a6a3b4e5f0cf1751ec83d76ad1db1ef7855e0f49`;
- in-memory variants for option B (32 arms, CH-8-C first; `70a702a9c86db7b446109dd28be9856680662dbdc31e0a876fc407fef422cb5c`) and option C (16 arms; `cdb34eaa09dca71edd76c76ec0ef70f5f9c1270b61b6c4b3c63256d47247b7a2`), built as in v0.5 (the method reproduces v0.5's `5c96c0aa…` and `4fa2f290…` on the v0.5 draft, and the four v0.7 values on the v0.7 draft, rechecked by the v0.8 fixer before its edit). The v0.6 values were `eafa227d…d852`, `01634bbb…c734`, `b84d2fbf…8aa2` and `11c1fa87…6a8d`; v0.7 adds a working-point metric to both stages and descriptive outputs (constructive K1–K6, K8, K9), and changes no arm. The v0.7 values were `68e167ff…a247`, `947c247f…bbee`, `64f06c5a…5b69` and `071898ec…8b03`; v0.8 adds the F000 run to the determinism record (`outputs[11]`; constructive v0.7 B1) and changes the five "STUDY v0.7" version strings to v0.8, and changes no arm. The v0.8 check was run by the fixer the same way, in memory, with nothing written.

Each returned one blocking finding, the missing dataset identity hash, which awaits the PREFLIGHT data manifest, and no other structural finding (`plan.md`, "Protocol check"). The v0.5 digests were `88ef1c79…036d` (screen) and `69c742a8…abcb` (confirmation); the v0.6 draft differs from v0.5 only in text fields (version strings, `selection.rule`, `interval_method`, `stop_rules`, `outputs`, and in `confirmation_stage` its `status`, `purpose`, `selection.rule`, `interval_method` and `outputs`), and its arms are unchanged. Until `lab_freeze_protocol` can run, the post-fix `protocol-draft.json` sha256 and these digests are the reference (arbiter condition P5). The result is advisory and is not a verdict. Mulder, the dataset and the cluster were not touched.

## Appendix A. Change record by finding

### A.1 v0.5 (2026-10-17)

v0.5 changes nothing between the headline-rule markers; the block's sha256 is still `d9f2e386…8dbb` after this revision. No run count changes. The author of v0.5 was blind to the da4ml outcome and to every PREFLIGHT reading (§3.4 Addendum B; §19.4; `plan.md`, "Blinding").

**Merges, and reconciliations between findings and with constraints the findings did not name:**
- **Critical B17 with chang B9, chang B7 (3)–(4) and critical C35: the clean rerun.** Critical B17 part 1 and chang B9 are the same defect, a renamed `logs/` that is never recreated, and get one fix: the driver recreates the directories, creates `tmp`, keeps `numba_cache` and runs the `hgq4` lines first.
  - **A constraint neither review named.** `run_case.sh:5` and `synth_one.sh:5` source `env.sh` from the absolute path `$HOME/scratch/cbl1016-da4ml-test/scripts/env.sh`. A "copy of `env.sh` that differs only in `PY`" (critical B17) or "a new env.sh pointed at the clone" (chang B7) is therefore read only if it sits at that path.
  - **Chosen.** On a 0.6.0 route the original `a1c095ef…` is archived byte-identical, the copy is placed at that path with its hash recorded, and the original is restored afterwards. `m3-env` is cloned, never modified. This meets chang B7 (3) as written and critical B17's intent: `m3-env` unchanged, the original `env.sh` preserved.
  - **Rejected.** A `HOME` override with a fresh scratch tree would keep every file byte-identical, but Vitis and Vivado read settings and licence paths under the real `HOME`, so it could fail every line.
  - Critical C35's version rule (the lowest passing version; one clean rerun per route version) replaces v0.4's "to nothing else".
- **Chang B7 with critical B17 on where the rerun runs.** Chang B7 moves option 0 and every export out of `m3-env` into a named export environment, while critical B17 keeps the rerun on `env.sh`'s environment. Both hold. The rerun stays in `m3-env` or its clone, because the block's version provision names only hls4ml and da4ml, which equal the export environment's, and the matrix test imports only `QEinsumDense`. Keras 3.15.0 and TensorFlow are declared as a difference.
- **Critical B18 with chang B8, physics C17 and critical C38: degenerate layers.** One definition by effect covers all four findings: the layer's output, after its output quantizer where it has one, has zero variance over the validation jets. It applies to binary and HGQ layers alike (B18), and its routes include all-zero-width input and output quantizers (C38, C17). Chang's five consequence bullets replace v0.4's mechanics.
- **Physics C17 against chang B8 on a zero Q kernel.** Physics C17 asked that C6 report "query-independent attention used". Chang B8 shows, from the per-(position, head, dim) Q output quantizer (`mha.py:162`), that a zero Q kernel gives input-independent but position-dependent queries, so the attention is not query-independent. The source-cited mechanics are adopted and physics' intent is kept: a degenerate Q does not block C6 "used", but C6 then reads "used, with input-independent queries". This also answers critical C32's question whether a zero Q blocks "used".
- **Critical C31 with chang B8 (4): Δ_unif.** Both define it as zero logits through the model's own `QSoftmax`, and "exactly 1/8" is removed. Under that definition a model with K ≡ 0 has Δ_unif = 0 exactly, so the sentence is kept in corrected form rather than dropped. H_attn uses row-renormalized probabilities (both), with all-zero rows counted separately (chang).
- **Physics C17 ("Δ_unif carries the test") with chang B8 (3) (Δ_static).** A static non-uniform pattern passes Δ_unif and the entropy condition; Δ_static closes that gap. All three ablations are required, and the entropy condition is kept as a consistency check (physics C17).
- **Critical C33 with critical B18 at §19.2 item 9.** The slot is held until S2 under the default and its control trains in S3. The trigger reads "a degenerate attention layer, binary or HGQ". Kai's answer is requested before S1; while it is open the hold is the default, so the request blocks nothing.
- **Physics B12's options (a)–(c) as one item.** (c) applies only under option C and can combine with (a) or (b); the C row of the power table then equals B's. Recommended: (b), as the referee recommended. The g* rule is written out exactly, where the referee gave an example. The whole-D_conf form is nested in the band test, so it is reported beside at no added α. Item 12 is appended; items 1–11 keep their numbers, which 36 cross-references use.
- **Physics B12's plug-in power.** It is computed with the §6 simulation model itself, so that S2's number and the table share one model. The smallest single-allowance power is printed beside it as the bound for any ρ.
- **Critical C37 with physics C16: C4.** Renamed. Non-inferiority stays the §19.3 alternative, because its margin cannot be grounded now.
- **Physics C18 against the lab literature.** BitNet is cited through the lab record. arXiv:2003.06308 has no lab record and no network was available, so it is cited by arXiv ID and title as quoted in `review/physics-v0.4.md` C18, with no number used and its author list marked unverified.
- **`preflight/env/` against §19.4.** The directory appeared at 07:44–08:00 and was not opened (§19.4).

**critical-v0.4:**
- **B17** (merged with chang B9): §3.4 Addendum A steps 1–9 (directories recreated, `tmp`, `numba_cache`, the `hgq4` driver check, the 0.6.0 environment with `env.sh` at its fixed path, evidence root `preflight/da4ml-clean/<version>/`, a hashed 44-row table, `make_table.py` not used); §11; §13 da4ml row; §19.2 item 5; §19.3.
- **B18** (merged with chang B8, physics C17, critical C38): §2.2 effect definition; §8.1; §8.7 label "binary or HGQ"; §19.2 item 9 trigger; §12; §13 new row; §15; §17.
- **C28**: Addendum B times (`results/` 06:26:26, `reports/` 06:23:30, `logs/` 06:23:31); §3.4 "cited, by file path and list hash"; `stage_b_build.py:192-209`.
- **C29**: Addendum B, the orchestrator's confirmation recorded; the `decisions.md` record is the orchestrator's.
- **C30**: Addendum B "Who was blind" lists the lines read (v0.4 author: 1-58, 73-78, 79-140; v0.5 author: none), with the orchestrator's confirmation.
- **C31**: §8.8 Δ_unif through `QSoftmax`; H_attn renormalized.
- **C32**: §8.8 per g and overall; a degenerate Q does not block "used" and changes its wording.
- **C33**: §5 tier cells and stages; §10 admission; §8.7 DS-HGQ parenthetical; §19.2 item 9.
- **C34**: §6 RNG flow; §13 resume row.
- **C35**: §3.4 Addendum A reading and step 1; §9 option 0 outcome.
- **C36**: §13 selection row (7- or 6-contrast); §8.7 pooling (seeds 1–8 or 1–11).
- **C37**: §8.7 C4 name and "What the name covers"; §17 C4; §19.3.
- **C38**: merged into B18 (§2.2 routes; §8.1 zero-width input fractions).

**physics-v0.4:**
- **B12**: §6 bound and the (c) note; In brief "Confirmation" and "Claim domain"; §1 question and nulls; §8.2 D_C2, the g* rule and the amended no-post-hoc line; §8.6 plug-in power and counts; §8.7 options paragraph, C2 on D_C2, nested strong form, C3r; §14; §15; §17 C2 and C3r rows; §19.2 items 11 and 12; §0 row; protocol draft.
- **C15**: §8.6 labels, runner-up, jet-sampling direction (√(260,000/62,000) = 2.05, derived).
- **C16**: with critical C37.
- **C17**: with critical B18 and chang B8; C6 consistency-check wording; zero Q reconciled above.
- **C18**: §1 Prior work; §2.3 BitLinear citation; §16 related-work table; thesis sentence.
- **C19**: §1 thesis sentence with the caveat folded in, quoted only together with its bullets.

**chang-v0.4:**
- **B7**: §9 Export environment, which stays at Keras 3.14.0 on CPU if the §7.3 training fallback is taken, with the cross-backend reload gate carrying the difference; option 0 in it (§9, §13); §3.4 Addendum A step 2 (rerun environment, declared difference, sidecar fields); §19.3 entry; `m3-env` never modified (§11, §13).
- **B8**: §2.2 (five consequences, effect definition, routes, softmax zero-width fractions); §8.1; §8.7 labels; §8.8 Δ_static, Δ_unif, H_attn, replacement point; §12; §13 gate row; §15; §17 C6; §19.3.
- **B9**: merged with critical B17.
- **C28**: with critical C34.
- **C29**: §3.2 `einsum_utils.py:153,155`.
- **C30**: §2.3 `_fixed_point_ops.py:183-185` and `:15-18`.
- **C31**: §9 exported scale type rewritten (BitExact `minimal_kif`, DA consumes the kernel, `bit_exact` left at its default and recorded).
- **C32**: §9 option 0 step 3 (`allow_v2_fallback=False`; the model-wide flag; registry plus assertion) and Outcome (exception types).
- **C33**: §9 "outside hls4ml 1.3.0's declared range".
- **C34**: §2.1 table exception for `QSoftmax`.
- **Knowledge-base corrections** to `docs/chang/` are outside this revision's write scope and are for the orchestrator.

**Not applied as written, and why:**
- Critical B17's "a copy of `env.sh` that differs only in `PY`" and chang B7's "point a new env.sh at the clone": applied at the fixed path the frozen scripts read, with the original archived and restored, because a copy elsewhere would never be read (reconciliation above).
- Critical C28's correction of `plan.md:117`: `plan.md` v0.4 is kept verbatim in the snapshot; the v0.5 `plan.md` gives the correct times.
- Critical C29's `decisions.md` record, and its alternative of copying the evidence file into the snapshot: the orchestrator's, who reports the file verified and read-only.
- Physics C18's author attribution "G. Di Guglielmo et al.": the STUDY cites arXiv:2003.06308 by ID and title and marks the author list unverified, because no lab record or network was available.
- Chang knowledge-base corrections: outside the write scope.

**§19.2 numbering (v0.4 → v0.5):** items 1–11 unchanged; item 12 (claim domain) new.

#### A.1 addendum, v0.6 (2026-10-17): the v0.5 C items of `review/arbiter-v0.5.md` §6

v0.6 applies the arbiter's 14 merged C items without re-review (`docs/methodology/06-review.md` §6.1). Nothing between the headline-rule markers changed: the block's sha256 is still `d9f2e386…8dbb`, byte-identical to `review/snapshots/v0.5/STUDY.md`. No run count, GPU-hour figure, `[D]` label or pre-registered decision rule changes. No reserved item of §19.2 changes its recommended answer; item 12's was revisited, as the arbiter's item 3 allows, and kept. The text above in A.1 is v0.5's record and is kept as v0.5 wrote it; where an item below supersedes it, the item says so. The author of v0.6 was blind to the da4ml outcome. It saw one `preflight/env/` reading, through the one `decisions.md` bullet it read (§3.4 Addendum B).

Items, in the arbiter's numbering, with their locations:
1. **Clean-rerun safeguards (critical C39 with chang C38).**
   - Addendum A, header.
   - Step 2: the clone is a new venv from `m3-env`'s `pip freeze`, installs run as `<clone>/bin/python -m pip`, the freeze hash is recorded before and after with the diff, `env.sh` is restored in an exit trap and first after any interruption, and a 0.5.2-branch fault takes the clone route.
   - Step 3: exclusivity, the equivalence check before step 3, and the skip sentence. Step 4: the meaning of "the environment" in the 0.5.2 branch, and a failed assertion before any rule line is a step-4 fault.
   - Step 5: the four per-line assertions, and the remedy, "void and restart", recommended with its reason, with "fail the line" (chang C38(2)) as the named alternative. Step 6: a voided run is discarded whole.
   - Elsewhere: §3.4 P-stock wording; §11 Mulder; §13 da4ml row; §19.2 item 5; §19.3 "The run the rule is applied to".
2. **Tier-order text (critical C40).** §10 "Cap" and "When the cap binds"; §19.1 OD-8; protocol `stop_rules[3]`.
3. **The C-with-(c) case and C3r nesting (critical C41 with physics C21).**
   - "(C alone; C with (c) as B)" at: the §6 bound; the §8.6 threshold bullet, which prints 1.005 s_d for C with (c), and its count bullet; the §8.7 options paragraph; §15; the §17 C2 row (threshold, power and bound); §19.2 item 11; the item 12 introduction.
   - The nesting statement in §8.7 "Primary claims" and in §19.2 item 12 (c), with the arbiter's process for adopting the variant (arbiter v0.5 §5).
   - The item-12 recommendation, revisited: unchanged, CONFIDENCE MEDIUM.
   - Protocol `confirmation_stage.interval_method`. The Bonferroni rule text is unchanged.
4. **Degenerate Q (critical C42 with chang C35).**
   - Locations: the §2.2 degenerate-Q bullet; §8.1 structure record; the §8.8 reporting bullet and C6 paragraph; the §17 C6 row; the §19.3 C6 rule; the protocol C6 method and structure-record outputs.
   - This supersedes v0.5's "so the attention is not query-independent" in the reconciliation above.
   - No C6 outcome changes.
5. **Blinding record (critical C43, STUDY part).**
   - §19.4: "07:58". The A.1 v0.5 text "07:44–08:00" is kept as written.
   - Addendum B: its first bullet notes the 10:36:50 change; a new bullet records the orchestrator's C43 records with this author's metadata; "Who was blind" gains the v0.5 panel and arbiter and the v0.6 author; and "Prompts" gains a sentence.
6. **JAX export route and rerun-environment check (critical C44 with chang C39).** §9 export environment, citing `run_test.py:3`; §19.3 export environment; a new §13 row, "rerun-environment equivalence"; Addendum A step 3.
7. **The g* sign filter (physics C20).** §8.2 after the g* rule; §19.2 item 12 (b); a new §19.3 entry with alternatives (i) and (ii), each written in full. Φ(δ√3) recomputed: 0.5000, 0.8068 and 0.9584 (derived).
8. **Plug-in power at the s_d bound (physics C22).** A new §8.6 sub-bullet; §6 unequal-gains sentences; protocol `outputs[10]` and the screen `interval_method`. 6.2847 recomputed as √(2 / χ² quantile 0.025 at df = 2) (derived).
9. **The single softmax format (chang C40).** The §2.1 and §2.2 citations `mha.py:57`, `softmax.py:83,94` and `activation.py:42-44`, checked in the snapshot.
10. **The prior selective-binarization citation (physics C23, as corrected by the arbiter).** §1 Prior work; the §16 related-work row. The A.1 v0.5 entries on physics C18 are kept as written.
11. **Thesis wording (physics C24 with chang C41).** §1 thesis sentence and its domain bullet; front matter `question:`; §19.2 item 12 (b).
12. **Ablations under WRAP quantizers (chang C36).** A new §8.8 bullet; the §13 structure-row gate; the §17 C6 row; the §19.3 C6 rule; the protocol C6 method and outputs.
13. **Exactness of Δ_static (chang C37).** §8.8 Δ_static; the §13 structure-row gate; the protocol C6 method.
14. **Item-5 sequencing and protocol wording (arbiter §4).** Addendum A, "When Kai answers"; §19.2 item 5; §13 da4ml row; protocol `stop_rules[2]`.

**Departures from the arbiter's wording, with reasons:**
- Front-matter status: the orchestrator's brief gives the wording.
- Item 8: "the most favourable case" now reads "for a fixed mean gain", physics C22's own qualifier. Without it, the new unequal-gains sentence would read against it.
- Item 11: the arbiter's sentence is applied verbatim. The §1 domain bullet adds one clause: D_band keeps only allowances of D_conf, so CH-8's validation feasibility in the confirmation seeds also enters, before any held-out read. The sentence's parenthetical describes where g* is set.
- Item 14: in `stop_rules[2]` the pointer "(STUDY 3.4 Addendum A)" is replaced by the arbiter's "(STUDY 19.2 item 5)", not kept beside it.
- Item 1: the four per-line assertions are also written to the sidecar.
- Propagation beyond the arbiter's list, for consistency only: the §19.3 "run the rule" alternative; the §8.8 heading; the §17 C6 overflow label; the protocol structure-record and ablation outputs.

**Records.**
- The v0.6 protocol digests are in §19.4.
- The nine frozen scripts and `env.sh` recompute to the change-log hashes, at 16:57 before the edits and at 17:06 after them.
- STUDY's own sha256 cannot be written into this file. It is reported to the orchestrator with the protocol-draft sha256, under arbiter condition P1.

#### A.1 addendum, v0.7 (2026-10-17): `review/constructive-v0.5.md` and two hand-off flags

v0.7 applies the constructive review run under arbiter condition P4 (b) (0 A, 0 B). That review read STUDY v0.5, not v0.6; its locations are re-anchored onto v0.6 here. Nothing between the headline-rule markers changed, and the block's sha256 is still `d9f2e386…8dbb`. Addenda A and C are unchanged. No rule, run count, `[D]` label or §10 figure changes, and no recommended answer of §19.2 items 1–12 changes. The author of v0.7 is partially results-aware on the da4ml direction through the review's own declared slip (§3.4 Addendum B), so it made no change to rule, calibration, C4 or firmware text.

Items and locations:
- **K1, pooled seed variance:** §8.6 validation outputs (new bullet); §6 "What the table says"; §19.2 item 11; §13 selection row; protocol screen `interval_method` and `outputs`.
- **K2, null selection reference:** §8.6 "labels and context"; §19.2 item 11; §13 selection row; protocol. The review's orientation figures are not written in; the S2 output computes them.
- **K3, factorial block intervals and the additive-model cell:** §8.6 "Dominance and the factorial"; §13 selection row; protocol. t(0.975, 14) = 2.145, 0.876 σ against 3.513 σ, recomputed (derived).
- **K4, determinism record (not a gate):** §13 "Further required gates"; §8.6 F000 − CH-8 bullet; protocol `outputs`. About 0.03 GPU-h from the existing reserve.
- **K5, reading table and event lines:** §8.8, new paragraph and table after "Binary-specific loss"; §15 "Lost pruning channel" and "350k may be unattainable"; protocol `outputs`. The per-layer scale-pruning row is marked as inference, not measured. The pilot labels were checked at `campaigns/2026-10-05-pilot-program/STUDY.md:120-127`.
- **K6, working point:** §8 "Metrics"; §8.6 held-out outputs; §18 new row; protocol `metrics` in both stages.
- **K7, the form-subset alternative:** §19.2 items 8, 9 and 11. Every number was recomputed (39 runs 250.25 GPU-h, 300.25 with the reserve, r ≤ 4.20 / 4.62 s; 36 runs 231.00, r ≤ 4.55 / 5.00 s; 54 runs 396.50; 12 or 15 runs 77.0 or 96.25). §10 is unchanged.
- **K8, scale of δ:** §8.8 "Classification" (new sub-bullet) and "Binary-specific loss"; protocol `outputs`. δ = 0.05 [OD-14] is unchanged.
- **K9, distinct checkpoint pairs:** §8.6 plug-in sub-bullet; §8.7 "C2"; §13 selection row; protocol, both stages. K8 and K9 are text-level and are applied rather than listed in §19.3.
- **P1–P4:** §19.2 items 13–16, one each, none adopted, with the introduction to §19.2 updated. Defaults if unanswered: not adopted. Recommended: 13 adopt if decided before S1; 14 not in this campaign; 15 adopt with m_s = 2 points as a placeholder; 16 adopt with m = 1 point. All CONFIDENCE MEDIUM. §15 points to item 14.
- **Hand-off flag 2, interruption:** §19.2 item 5, a line with options (i), the text as it stands and the default, and (ii), a void trigger; no recommendation.
- **Hand-off flag 1, the quoted `preflight/env/` reading:** Addendum B, the `preflight/env/` sub-bullet. "Who was blind" gains the constructive reviewer and the v0.7 author, and "Prompts" gains the constructive review as an exposure.
- **Records:** change-log line; front-matter status; §19.4 digests.

#### A.1 addendum, v0.8 (2026-10-17): `review/constructive-v0.7-delta.md` B1–B4 (fixer)

v0.8 applies the four B findings of the P4 (b) delta review of v0.7 (0 A, 4 B), each with the text that review specifies, under the orchestrator's rulings recorded below. Nothing between the headline-rule markers changed, and the block's sha256 is still `d9f2e386…8dbb`. Addenda A and C and §10 are byte-identical to `review/snapshots/v0.7/STUDY.md`. No selection or verdict rule, screen or confirmation run count, `[D]` label or §10 figure changes. The fixer opened no blinded file and saw no da4ml outcome and no home-PC engineering reading (`plan.md`, "v0.7 → v0.8 (fixer)"). The earlier A.1 text is kept as written; the v0.7 addendum's "About 0.03 GPU-h" for K4 is superseded by B1 below. Line numbers are those of v0.8.

- **B1, the determinism record. Orchestrator ruling: fix steps 1–4 adopted under Kai's 2026-10-17 delegation of checks and validation, flagged for Kai, because the zero-cost alternative leaves the §8.6 F000 − CH-8 reading unsupported.**
  - Step 1, §13 determinism record (`:1348`): one F000 seed-1 run added, with the same initialization, batch order and 20 epochs, compared bitwise with both CH-8 runs; "If the two runs" now reads "If the three runs". Cost (derived): 60 run-epochs × 3 s = 180 s, 0.05 GPU-h at the planning rate; × 1.10 = 198 s, 0.055 GPU-h with §10's factor. The F000 run adds 0.0167 GPU-h (0.0183 with the factor). It comes from the existing 50 GPU-h reserve, and §10's text is unchanged.
  - Step 2, §8.6 (`:976`): F000's row is left out (df = 14) if the record's F000 run equals its CH-8 runs bitwise, and otherwise enters (df = 16); this is fixed before S1 and not revisited on screen values.
  - Step 3, §8.6 (`:989`): "if the record's three runs are bitwise identical".
  - Step 4: the STUDY does not imply that the bitwise gates of the reference-fidelity row (`:1313`) and the resume row (`:1325`) run on the production GPU product. §11 places the resume gates on the home PC, and no device is named for the reference-fidelity row. Under the ruling's fallback, the stack is listed as a PREFLIGHT owner decision in §13 (`:1348`, "Stack of the bitwise gates"), and the review's consequence is kept as a condition: if either gate runs on the production GPU product and stack, the record's "not identical" branch is a failure of that gate. Both rows point to it.
  - Protocol `outputs[11]`: the F000 run added.
- **B2, the null selection reference. Ruling: the fix sentence and the gate as written.** §8.6 (`:978`): feasibility held at the observed screen pattern; each arm's top-1 at (seed, g), CH-8 included, simulated with residual sd σ̂_pool; the two forms of a configuration drawn with correlation ρ_f = ρ (0 and 0.9). The review's "σ̂" reads σ̂_pool because of the item-1 clarification below; it is the same quantity. §13 selection row (`:1327`): "with one candidate in one form, the simulated median of the null selection reference is 0", the review's gate with the simulation named.
- **B3, the adoption path of items 13–16. Ruling: the preamble sentence as written.** §19.2 preamble (`:1474`). The existing clause "items 13 and 15 act only if decided before S1" is kept: the new lapse point, the PREFLIGHT build, comes earlier and does not contradict it. Items 13–16 themselves are unchanged.
- **B4, interruption mechanics. Ruling: fix steps 1–5 in §19.2 item 5 and the §13 da4ml row only; Addendum A is not edited; the choice between (i) and (ii) stays Kai's, the text as it stands remains the default until he decides, and no recommendation is made.** §19.2 item 5 (`:1479`): the "Open for Kai (v0.7)" paragraph is moved to the end of the item and followed by "Mechanics of an interruption, under either option": each line in its own process group; the exit trap stops and waits for that group before it restores and rechecks `env.sh`, and writes `interrupted` with the last reported stage into the sidecar; the table script fails an `interrupted` line at that stage and at every later one; under (i) a restarted driver re-establishes step 3's exclusivity and the branch's `env.sh`, does not retry the stopped line, exempts it from the "does not exist" assertion and continues from the next unstarted line; under (ii) the rerun is voided and restarts from step 3. §13 da4ml row (`:1319`): the same mechanics after the void clause.
- **"For later" clarifications, text only, allowed by the orchestrator.** Item 1: σ̂_pool for K1's nine-row table (`:971`, `:978`) and σ̂_F for K3's interaction (`:1001`); σ̂_d is unchanged. Item 3: "pilot" before the H1, H3, H4 and H5 labels of the §8.8 reading table (`:1076-1081`). Item 6: the BitNet b1.58 citation of §19.2 item 14 (`:1498`) points to `:17`, the absmean ternary line, checked in the lab record, where `:16` is a heading. Items 2, 4, 5, 7 and 8 are not applied; item 8's 1.10-factor arithmetic is covered by B1's recomputation, and its remark on what 20 epochs establish is not applied.
- **Records.** Front-matter status (`:6`); change log (`:35`); the frozen-rule heading (`:55`) adds v0.8; §19.4 digests (`:1602-1605`). The nine frozen scripts and `env.sh` (change log `:58-72`) were not recomputed in v0.8, because `preflight/` is outside what the fixer may open; they fall to the orchestrator's check under arbiter condition P1.
- **Protocol.** `protocol-draft.json` changes `outputs[11]` and five version strings (`purpose`, `selection.rule`, `confirmation_stage.status`, `confirmation_stage.purpose`, `confirmation_stage.selection.rule`); it stays ASCII with indent = 1, and the arms (66 and 20) and `stop_rules` are unchanged. The in-memory digests are in §19.4; each check returns only the dataset-identity block.
- STUDY's own sha256 cannot be written into this file. It is reported to the orchestrator with the protocol-draft sha256.

#### A.1 addendum, v0.9 (2026-10-18): PREFLIGHT gate G54 finding and owner decision B3b-22 (fixer)

v0.9 is a pre-data amendment from PREFLIGHT, not a review iteration. Nothing between the headline-rule markers changed, and the block's sha256 is still `d9f2e386…8dbb`. Addenda A and C and §10 are byte-identical to `review/snapshots/v0.8/STUDY.md`. No threshold, selection or verdict rule, level, C2 criterion, run count, `[D]` label or §10 figure changes. Of `preflight/` the fixer opened only `preflight/gates/G53/1784511c0f34/PREFLIGHT-rows.md` (:92-150: its G54 section and owner decisions B3b-1 to B3b-4), and of the earlier snapshots only `review/snapshots/v0.2/STUDY.md:440-480`; it opened no other blinded file and saw no da4ml outcome (`plan.md`, "v0.8 → v0.9 (fixer)"). Line numbers are those of v0.9.

- **Item 1, the §6 power table (PREFLIGHT gate G54 finding).**
  - ρ = 0 rows (`:753`, `:755`, `:757`): the roots of p(δ)^|D| = 0.8, p the one-sided noncentral-t power (scipy `stats.nct.sf`, `optimize.brentq`), recomputed by the fixer: A 2.0500 2.3530 2.5131 2.6204 2.7004 2.7639; B 1.3392 1.5233 1.6198 1.6841 1.7320 1.7698; C 1.1560 1.3286 1.4189 1.4790 1.5237 1.5590 (|D| = 1 to 6). At two decimals A changes at |D| = 2, 4, 6 and B at |D| = 2, 4; C is unchanged. The ρ cell reads "0, exact (closed form)".
  - ρ = 0.9 rows (`:754`, `:756`, `:758`): the ρ cell reads "0.9, Monte Carlo", and the note below the table (`:760`) gives 200,000 confirmations per cell, standard error about 0.0013–0.0023 σ_d at |D| ≥ 2, and the second decimal uncertain by one unit. The fixer reimplemented B3b-1 from its text and reproduced the seven G54 computed roots of the rows file to four decimals, among them 2.3751 (A, |D| = 6) and 1.4653 (B, |D| = 3). Those two cells therefore read 2.38 and 1.47 (were 2.37 and 1.46), the values the stated layout gives; the other 13 simulated ρ = 0.9 entries agree at printed precision. Nothing was tuned: the layout and root rule were fixed in B3b-1 before the first comparison.
  - Layout sentence (`:749`): the noncentral-t parameters, the closed form at ρ = 0, and B3b-1's layout and root rule, with crit_j defined so that the rule can be followed from the STUDY alone.
  - Gate: §13 and §18 state no gate on reproducing the §6 table. The §13 selection row (`:1335`) gates the S2 plug-in power and zero differences for identical arms, not the table. None is added or restated. G54 is a PREFLIGHT-plan gate, and its restatement belongs to the PREFLIGHT owner.
  - Propagation, checked and unchanged at printed precision: In brief (`:25`); §6 options (`:784-786`); §15 (`:1392`); §17 C2 row (`:1433`); §19.2 item 11 (`:1494-1496`); §19.3 (`:1548`). Under A the range runs from 2.38 (ρ = 0.9) to 2.76 (ρ = 0) at |D| = 6, still "about 2.4–2.8"; B (1.53–1.77) and C (1.33–1.56) are unchanged. The ρ = 0.5 values (`:765`: 2.67, 1.70, 1.49), the probabilities at the threshold and at six allowances (`:763`), and the 0.5 σ_d powers (`:771`) were recomputed, with the same layout where simulated, and agree.
- **Item 2, the descriptive labels (PREFLIGHT owner decision B3b-22).** §8.7 (`:1039-1042`): the definitions of `CROSSING`, `LOCAL ADVANTAGE` and `LOCAL DEFICIT`, restored from v0.2:463-465, flagged for Kai. Adaptations, each only as v0.8's terms require: (i) v0.2's 91-contrast intervals become the 7- or 6-contrast simultaneous Bonferroni intervals that v0.8 already names for these labels (`:1038`), not the level-L intervals of C2; (ii) v0.2's "point" becomes g ∈ D_conf, the allowances at which §8.7 defines and reports the contrasts, consistent with a family of six grid points and with §17's LOCAL DEFICIT "reported against" C2; (iii) "the whole-front criterion" becomes C2's `ABOVE ON GRID` for LOCAL ADVANTAGE and `BELOW ON GRID` for LOCAL DEFICIT; (iv) a COST INFEASIBLE g counts as a deficit, by §8.2; (v) LOCAL DEFICIT's "mirror case" is written out beside v0.2's words. The comparator stays CH-8: in v0.2, BE is a control arm, never a comparator. No verdict rule, level or the C2 criterion changes.
  - Flagged for Kai: the domain D_conf rather than D_C2 (they coincide under option (a) of §19.2 item 12; under (b) a deficit below g* can be labelled), and the pairing of the 7- or 6-contrast label intervals with C2's level-L verdicts as the whole-front criterion.
- **Records.** Front matter (`:5-6`); change log (`:35`); the frozen-rule heading (`:57`) adds v0.9; this addendum. `protocol-draft.json` is unchanged (sha256 `8f48e716…08d302c8`), so the §19.4 digests stand; its five version strings still read "STUDY v0.8", and its label sentence already names the 7- or 6-contrast family. The nine frozen scripts and `env.sh` (change log `:60-74`) were not recomputed, because `preflight/da4ml-test/` is outside what the fixer may open.
- STUDY's own sha256 cannot be written into this file. It is reported to the orchestrator.

#### A.1 addendum, v0.10 (2026-10-18): `review/critical-v0.9-amendment.md` B1 and PREFLIGHT B3b-30 (fixer)

- **B1**: §19.3 entry "Descriptive label definitions" (`:1548-1551`), worded as the review's §6 gives it, with its line references moved to v0.10 (§8.7 `:1041-1044`, `:1040`). The v0.9 change log and addendum are kept as written; this line and the v0.10 change log (`:35`) record the entry. **B3b-30**: §8.6 (`:972`) reads "at ρ = 0 the closed form, the product of the single-allowance noncentral-t powers, and at ρ > 0 the §6 simulation" in place of "the same simulation"; this also answers C5. C1–C4 and C6–C9 are not applied. Records: front matter (`:6`), frozen-rule heading (`:59`). The headline-rule block, Addenda A and C, §10 and `protocol-draft.json` are unchanged; line numbers are those of v0.10.

#### A.1 addendum, v0.11 (2026-10-19): Kai decisions of 2026-10-18 and 2026-10-19

This amendment records the decisions and their sources in §0. It is results-aware on the OPEN-2 engineering outcome; the v0.4 criteria are fixed before any training result. The v0.3 block is byte-identical to v0.10. §10, screen arms, run counts, seeds, selection rules, statistics and `[D]` labels are unchanged. [Corrected in v0.12, critical v0.11 B4: the content of D10's §0 row is amended, its label kept.] Line numbers below refer to v0.11.

1. **Training on NRP; export work deferred.** In brief, §3.4 sequencing and Addendum A status, §9, §11, §13 and §14: `:21`, `:439`, `:563`, `:1184`, `:1325`, `:1358`, `:1407`. The clean rerun and export gates leave the launch path; deciding-point synthesis follows training (Q14).
2. **Conditional H2.** §3.1 and §3.4: `:378`, `:507`, `:537`, `:557`. v0.3 is marked superseded outside its unchanged markers. v0.4 records the orchestrator's existing-table H1 reading, the results-informed Kai choice, exact whole-model criteria and H1 fallback, and the open form-coverage case. §15, §19.2 item 5 and §19.3–19.4 propagate that status: `:1445`, `:1532`, `:1560`, `:1643`.
3. **Factored export and INLINE.** §3.4 v0.4, §9 and §19.2 item 5: `:542`, `:1186`, `:1532`. Both routes apply the same generated-code patch to every model, including CH-8 and F000. The engineering READMEs are cited without treating them as a whole-model pass; C4's statistical rule is unchanged (§8.7, `:1098`).
4. **PVC route A2 and runtime.** §0 sources, §7.3, §11, §13 and §19.2 items 1 and 7: `:102`, `:103`, `:907`, `:1326`, `:1327`, `:1328`, `:1364`, `:1528`, `:1534`. Git bundle, data, venv, outputs and logs on `kai-data`; no ConfigMaps; the pushed-runtime-image item is resolved.
5. **Mulder permission.** §9, §11 and §19.2 item 6: `:1184`, `:1325`, `:1533`. Permission covers the export check; later deciding-point synthesis is asked again.
6. **Infrastructure retries.** §11: `:1329`. No relaunch count limit within the existing GPU-h cap; code or integrity failures stop for a fix, a new commit and a new handoff.
7. **No separate H1 campaign.** §0 and §3.4 v0.4: `:105`, `:556`. Recorded as the orchestrator's stated default that Kai did not change; every run records H1, H2 and NR.
8. **Version and remaining preconditions.** Front matter, change log, §0 source record, §14 and §19.2: `:5`, `:6`, `:35`, `:99`, `:1407`, `:1527`. Items 2–4 (OD-13, OD-15, OD-12) and 12 (claim domain) stay open launch preconditions. Checksums, table checks and the §10 comparison are in `review/amendment-v0.11/NOTES.md` under this campaign. [v0.12: the v0.4 block and STUDY checksums recorded there predate the orchestrator's correction and are superseded; see the change log and the v0.12 addendum.]

#### A.1 addendum, v0.12 (2026-10-19): `review/critical-v0.11-amendment.md` A1–A3 and B1–B6 (fixer)

Applied under the orchestrator's rulings. The review's line numbers refer to v0.11; those below refer to v0.12.
1. **A1, binary admission waits for the check's outcome.** In brief `:21`; §3.4 sequencing `:449` ("train first"); v0.4 block "Outcome and forms" `:568` (a failed check defined), "Admission" `:570`, "Sequence and record" `:572`; §14 `:1431`; §19.2 item 10 `:1560`. F100 is named in each; CH-8 and F000 launch independently.
2. **A2, split-form rule.** v0.4 block `:569`. The first v0.12 draft carried the orchestrator's default recorded at 2026-10-19T12:20:19Z ([dec 10-19 split], tag `:98`); it is replaced by Kai's rule, recorded at 2026-10-19T16:28:12Z before any synthesis, timing or LUT outcome was read ([dec 10-19 both], tag `:97`; item 11 below). Propagated to In brief `:21`, the D10 row `:130`, "Provenance and timeline" `:552`, Addendum A status `:581`, §19.2 item 5 `:1551`, §19.3 `:1585` and §19.4 `:1677`.
3. **A3, status and provenance.** v0.4 block "Provenance and timeline" `:552` (PLAN.md `:3`, `:44-53`, `:67-73`; the pilot trained on the home PC before this amendment), "Routes and model" `:554` (three models; the quoted words attributed to the orchestrator's chat wording to Kai, [dec 10-19 split]), "Check specification" `:559` (`PLAN.md:44-53`, jet rows `:35`, which also covers C2), "Outcome and forms" `:568` (the sentence "adds no training run or checkpoint-selection rule" removed); §19.2 item 5 `:1551` ("No new pilot, training run …" removed); Addendum A status `:581`; §19.3 `:1585`; §19.4 `:1670`; Addendum B exposure `:639-640`, including this fixer.
4. **B1, recount.** `:550`: 19 of the 20 case-list pairs; the 21st table pair, `8x24x32 NP2w8 het`, is the per-position context case.
5. **B2, `[P]` and the v0.3 mapping.** Block heading `:548`, "Check specification" `:559`, "Pass criteria" `:561`, "Relation to v0.3" `:566`; §19.2 item 5 `:1553`; §19.3 entry "Timing route of v0.4 criterion 2" `:1587-1590`.
6. **B3, which arms the export gates bind.** `:449`, `:1203`, `:1377`, `:1430`, `:1555`.
7. **B4, D10 amended.** D10 row `:130`; bracketed corrections in the v0.11 change log `:37` and the v0.11 addendum `:1862`; this version's change log `:35`.
8. **B5, v0.4 hash.** Change log `:82-86`, beside v0.3's record; the stale v0.11 checksums in `review/amendment-v0.11/NOTES.md` are marked superseded there and at `:1871`.
9. **B6, hardware evidence under H2.** §19.2 item 5 `:1554`, a Kai item with the conservative default.
10. **Records.** Front matter `:6`; change log `:35`; frozen-rule heading `:63`; §19.4 `:1677`; this addendum. C1 and C3–C8 are not applied. The v0.3 block, §10, the arms, the run counts outside the one-pass case of item 11, seeds, selection and statistics rules, `[D]` labels, §19.2 items 2–4 and 12 and `protocol-draft.json` are unchanged. The checks and STUDY's own sha256 are in `review/amendment-v0.11/NOTES.md`, section "v0.11 → v0.12 (fixer)".
11. **Kai's split-form decision (2026-10-19T16:28:12Z, before the outcome).** v0.4 block `:569`, with the conditional cap note in §19.2 item 8 `:1558`: if exactly one form passes, 24 more runs (about 154 GPU-h at the planning rate, derived), a 75-run screen of about 481.25 GPU-h (531.25 with the reserve) and a cap of about 530 GPU-h in that case only. §10 is byte-identical, so the note is not placed there. How those H2 runs enter §8's selection and reading is not stated by the decision and is a new Kai item (§19.2 item 5 `:1552`). The v0.4 block hash changes with this item and is recorded in the change log `:82-86`.

#### A.1 addendum, v0.13 (2026-10-19): fix-verification B7 (fixer)

- **B7.** §14 `:1434` and §19.2 item 5 `:1556`: in the one-pass case, the 24 added H2 runs wait for Kai's §8 answer and his confirmation of the exact cap; seeds 1–3 paired as in §6, §10 tier order and the control slot unless Kai says otherwise; a dated `protocol-draft.json` amendment if the branch occurs. Records: front matter `:6`, change log `:35`, frozen-rule heading `:65`. Line numbers refer to v0.13; the v0.12 addendum's refer to v0.12. The v0.4 block is untouched (sha256 `b3f0cc93…2b23`).

- **v0.14 (2026-10-19), B10F1-1.** §7.3 declares the Keras loss guard and loss-term census for every arm, CH-8 included, and every gate child that builds a model; records in the front matter and change log. Verification: `review/amendment-v0.14/NOTES.md`.

- **v0.15 (2026-10-19), B1-B2.** Critical v0.14 amendment: §7.3 and the matching v0.14 change-log and A.1 text cover every gate child that builds a model; §5's CH-8 row lists the Keras loss guard (§7.3). C items are not applied. Verification: `review/amendment-v0.14/NOTES.md`, section "v0.15".

- **v0.16 (2026-10-20), export-check outcome.** §3.4, "Outcome of the v0.4 check", after the v0.4 block: both forms fail, so H1 is the headline count for every binary arm; per-line table, record items and report paths. Records in the front matter and change log. The v0.3 and v0.4 blocks, §10 and every rule are unchanged. Verification: `review/amendment-v0.16/NOTES.md`. [Corrected in v0.17: F111-BF's failure rests on the owner record of 2026-10-20T17:30Z, not on the time limit alone; see the v0.17 entry.]

- **v0.17 (2026-10-20), A1, B1.** §3.4 outcome block: the F111-BF failure is derived from the v0.4 owner-record clause and the owner record of 17:30Z (results-informed, B14-7a consequence line superseded, report to Kai with his answer pending); B14-7a read as written; the two earlier da4ml READMEs cited with sha256. Records in the front matter and change log. The v0.3 and v0.4 blocks, §10 and every rule are unchanged. Verification: `review/amendment-v0.17/NOTES.md`.

- **v0.18 (2026-10-20), B1, B2.** The v0.16 front-matter clause, change-log entry and A.1 bullet are restored byte-for-byte from the v0.16 snapshot with bracketed annotations; the outcome block states the BF sequence from the ledger (17:06Z to about 17:36Z). The v0.3 and v0.4 blocks, §10 and every rule are unchanged. Verification: `review/amendment-v0.18/NOTES.md`.

### A.2 v0.4 (2026-10-17)

The change-log text of v0.4 and its per-finding record, verbatim. Their section and item numbers refer to v0.4, and their "Appendix A.1", "A.2" and "A.3" are this file's A.2, A.3 and A.4.

**v0.4 (2026-10-17, experiment-designer).** This version applies `review/critical-v0.3.md` (0 A, 4 B, 12 C), `review/physics-v0.3.md` (0 A, 4 B, 7 C) and `review/chang-v0.3.md` (0 A, 1 B, 13 C). v0.3 is kept in `review/snapshots/v0.3/` with `SHA256SUMS`: STUDY `18f15fbf…` (the version all three reviews read), plan, protocol draft, protocol-check input, OPEN-CHOICES and the three reviews. The per-finding record, with every reconciliation between findings, is in Appendix A.1.

What changed, and why:
- **The PID selection rule and the factorial analysis are restored (critical B13).** v0.3 dropped v0.2's checkpoint rule behind claim C3 and the per-seed dominance and main-effects analysis, without a log entry. Both are back (§8.6, §8.7).
- **The headline rule is applied to one clean rerun (critical B14).** The frozen block's Build term names `stage_b_build.py` `db49a404…`, and at least one line of the existing run was built by an earlier version. PREFLIGHT reruns all 44 lines once, into empty project directories, with per-line script hashes, and applies the rule to that run only. This is a reading of the block, written outside it, and it is Kai's to confirm (§3.4 addendum, §19.2 item 5).
- **The blinding record is completed (critical B15).** The 06:23 runs, the 06:33 outcome summary in `decisions.md`, the orchestrator's evidence-hash file and its list hash, and the fact that the v0.3 panel was not blind are recorded (§3.4 addendum).
- **Kai confirms OD-12, OD-13 and OD-15 one at a time before any launch (critical B16).** They amend his D6, D3/D9 and D1. This is now a §14 launch precondition and §19.2 items 2–4.
- **The confirmation's resolving power is stated as power (physics B8).** §6 has a power table. Kai's S4 decision lists three sizes: as adopted (five seeds, four arms), eight seeds with four arms, and eight seeds of the ramp arms only (§8.7, §10, §19.2 item 11).
- **What a win would mean (physics B9).** Every exportable binary model is also an exportable HGQ model with the same H1, H2 and firmware, so a confirmed gain is a training-procedure result (In brief, §1, §3.5, §5).
- **Degenerate layers (physics B10).** Zero kernels are recorded at every frozen checkpoint for every arm; verdicts that rest on them are labelled; C6 needs non-uniform attention; DS-HGQ is named to Kai as the deciding control (§2.2, §8.1, §8.7, §8.8, §12, §17, §19.2 item 9).
- **C4 has a decision rule (physics B11).** A paired LUT interval per allowance where C2 is ABOVE, at the same level (§8.7, §9, §17).
- **The RNG flow covers Keras's per-epoch shuffle (chang B6).** The gate hashes the batch indices actually fetched (§6, §7.1, §13).
- **C items** of all three reviews: Appendix A.1.
- **Budget.** No screen run count changes. The S4 rows gain the two eight-seed options (§10).
- **The frozen headline-rule block is unchanged.** Its sha256 was recomputed after this revision (below).

**Frozen headline rule (recorded at v0.3; unchanged in v0.4).**
- Rule-block sha256: `d9f2e386d04710e1a424ad5f4511ea830f14b17c1e86afb15fa8959da7168dbb`.
- Recompute with `sed -n '/^<!-- headline-rule v0.3 begin -->$/,/^<!-- headline-rule v0.3 end -->$/p' STUDY.md | shasum -a 256`. The digest covers both marker lines and everything between them.
- Script hashes at the freeze (sha256), reproduced at 07:06 on 2026-10-17:

  | file (under `preflight/da4ml-test/scripts/`) | sha256 |
  | --- | --- |
  | `cases_bc.txt` | `1ba9c416b1c4c6007ad07361bd95b1e902058e66039df89e4b5be1a8928a3191` |
  | `stage_b_build.py` | `db49a4041ff1a9a041e9a3d9c87bded997ff3adc96844366941f981028ba83d6` |
  | `stage_b_build_pos.py` | `f214dc3430a0871ee39119a93a719da3448d0fe9868e4cbae70520e6dc9a00d2` |
  | `run_case.sh` | `1253773bd99b58015620e0e99bd12f5e575592d772545a453db43245004169f8` |
  | `synth_one.sh` | `32201339fcab6d42cad096b393fa7d41ca0380e5948f718422f4c07869ee9bb9` |
  | `vivado_ooc.tcl` | `70d269f430e8fe364ae660f9f13fa81bcb8f7c0a9c373f0a1dcecbe43eb14c32` |
  | `clock.xdc` | `e4ec5851c4fff9a4f8c0caf420e487be9e43c2a8f3ce83cb9954b3ddacb4339f` |
  | `common.py` | `cfb50a22bb95a5746e31ff456bd6b8721e9cdad684babd02c0789361e266a1fa` |
  | `parse_reports.py` | `a889ce8ff21f94eae61f9322e332930e99f22d8beb81eadca647c0922707ae38` |

  `env.sh`, which `run_case.sh` and `synth_one.sh` source, is not in this list: sha256 `a1c095ef7ab3b6dc32ee07dcf0b9b3c2e1794b8c48d8c04d970614592f459b78` (05:11), recorded here in v0.4 (critical B14).

The per-finding record of v0.4, verbatim:

v0.4 changes nothing between the headline-rule markers; the block's sha256 is still `d9f2e386…8dbb` after this revision. No screen run count changes. The author of v0.4 was blind to the da4ml outcome (§3.4 Addendum B; `plan.md`, "Blinding").

**Reconciliations between findings, and with constraints the findings did not name:**
- **One bullet for three findings at §2.2.** Critical C16 (the zero comes from the SAT_SYM bound, not the k+i+f mask), chang C15 ("costs 0 in every count" is false for a scale rounded to zero) and physics B10 (record degenerate layers) all hit the same bullet. It is rewritten once: pruned (b = 0) against rounded to zero (b > 0), the mechanism, and the degenerate-layer record.
- **Critical C17 against §3.1(a′).** `_compute_ebops` can return one value; it returns the penalty (b), and the frozen evaluator computes NR.
- **Chang C16 against §3.1–3.2.** The counts follow the pinned 0.1.9, which applies no parallelization factor; the snapshot's factor is 1 for the released layers, so nothing numeric changes.
- **Physics C8 with critical C23.** D_conf = D_screen ∩ {CH-8 feasible in every confirmation seed}; an empty D_screen makes the screen INCOMPLETE and goes to Kai; an empty D_conf makes C2 INCOMPLETE; "front" only at |D_conf| ≥ 2.
- **Physics C9 with critical C25.** One part per stage: if any pair needs xcvu13p, every pair of the stage moves; otherwise that stage's gaps are HARDWARE INCOMPLETE. The capacity comes from the report.
- **Physics B11 with critical C24.** C4 is evaluated only where C2 is ABOVE ON GRID, so it is an S4 claim; C6 is S4; C1 and C5 are implementation checks in both stages.
- **Physics B8 against `protocol-draft.json`.** The JSON keeps the adopted option A (20 arms); options B and C live in §6, §8.7, §10 and §19.2, were checked structurally in memory, and the claim-scope file frozen before S4 is regenerated for Kai's option.
- **Critical B14's wording against the frozen Build term.** The review asks for script hashes "in each `case_info.json`". `case_info.json` is written by `stage_b_build.py`, whose hash the Build term names, so the script cannot change. A new driver writes a `provenance.json` sidecar per line instead. "Empty project directories, no skip" is met by renaming the existing mulder `proj/main` and `logs` into an archive, because `env.sh` fixes `SCR` and `run_case.sh` skips completed lines; nothing is deleted and no script changes.
- **Critical B14 with the block's version provision.** H-LUT option 0 is tested before the clean rerun, so one clean run on the export route's versions serves both.
- **Critical B13 against the screen's no-verdict status.** v0.2's dominance used held-out top-1. It is restored as descriptive on both splits: validation in S2 (it feeds Kai's S4 decision) and held-out after the freeze, with no verdict label at n = 3. The factorial effects are treated the same way.
- **Physics B10 with critical C27 at §19.2 item 9.** One default: keep 400 GPU-h and fill the one control slot after S2 with DS-HGQ if BEST has a degenerate attention layer, otherwise FP32.
- **Physics B10, extended.** The structure record covers CH-8 too, since an HGQ kernel can also reach all zeros. A zero Q kernel (Q has a bias) is labelled query-independent attention. "Non-uniform attention" is made operational by a uniform-attention ablation beside the entropy condition, because an entropy margin below log 8 would be arbitrary.
- **Chang C25 against §9 departures 4–5.** Production export matches the frozen test (`fixed<16,6>`, Vitis default 27 % clock uncertainty); Chang's values are recorded beside.
- **Critical C22.** `decisions.md` is cited by entry tag and bullet opening words (§0), because the log grew by 20 lines inside the 2026-10-17 entry after v0.3 was written.

**critical-v0.3:**
- **B13**: PID selection rule restored for screen and confirmation (§8.6, used in §8.7 and §8.5); per-seed dominance and the 2³ main effects and interactions restored as descriptive, on validation and held-out (§8.6); protocol draft rule text updated.
- **B14**: §3.4 Addendum A (clean rerun of all 44 lines after option 0, archive by rename, provenance sidecars, one attempt per line, earlier runs as context, direction stated); §13 da4ml row; §14 stop; §19.2 item 5; §19.3 entry; `env.sh` hash recorded in the change log.
- **B15**: §3.4 Addendum B (06:23 runs with hashes of the four scripts, directory times, the 06:33 outcome bullet and the blinding-slip bullet cited by their opening words and not read, the evidence-hash file path and list hash `88ffcea2…`, who was blind, prompt practice). The `decisions.md` record and the prompt practice are the orchestrator's.
- **B16**: §14 fail-closed line; §19.2 items 2–4 (OD-13, OD-15, OD-12), each stating what it excludes, default "as adopted"; §0 table rows and §19.1 lines point to them.
- **C16**: §2.2 Forward bullet (exact because bc ≤ 23; 24-bit failure cited to `stage_b_build.py:45-48`); §2.2 zero mechanism corrected; §13 gate at b = 23 and a forced failing b = 24.
- **C17**: §3.1(a′).
- **C18**: §2.3 saturated-column paragraph; §12 logging; §13 edge case.
- **C19**: §3.4 Addendum C; §19.3 headline-rule entry; §19.2 item 5.
- **C20**: §3.4 History of the rule (the version-provision exception); §15.
- **C21**: §3.4 P-fact description (exact accumulator-format input quantizer).
- **C22**: §0 citation tags; every body citation converted.
- **C23**: §8.2 empty domains; §14 stop for S3 and S4.
- **C24**: §1 C3 null; §17 stage column.
- **C25**: §9 part capacity (every pair of a stage moves); §8.7 C4 stock-route sentence; §19.3.
- **C26**: §8.5 freeze list.
- **C27**: §19.2 item 9 default.

**physics-v0.3:**
- **B8**: §6 threshold table relabelled (observed s_d) and power table added (true σ_d, |D| = 1–6, ρ = 0 and 0.9); §8.6 S2 outputs give threshold and power per option at the screen s_d and at 6.28 s_d; §8.7 options A, B, C; §10 rows; §17 C2 and C3 wording; §19.2 item 11; In brief.
- **B9**: In brief ("What a win would mean", "Likely outcomes"); §1 paragraph and thesis bullet; §3.5 subset property and floor; §5 not-run control (per-column or per-layer HGQ width); §15.
- **B10**: §2.2 degenerate layers; §8.1 structure record; §8.7 labels; §8.8 C6 rule (Δ_unif, H_attn < log 8, no zero K/V/O); §12; §17 C6; §19.2 item 9 (DS-HGQ named as the deciding control); §15.
- **B11**: §8.7 C4 rule (level-L paired LUT interval per g, labels, IUT, every pair labelled); §9 pairing sentence; §17 C4.
- **C8**: §8.2. **C9**: §9 part capacity and capacity source; §13 row; §19.4. **C10**: §6 table header. **C11**: §8.4 and §8.6. **C12**: §3.4 zero-heavy calibration case. **C13**: §1 marks the thesis sentence for the next panel; giving it to the panel is the orchestrator's act. **C14**: §2.2 `i_decay_speed` listed.

**chang-v0.3:**
- **B6**: §6 RNG flow names both consumers, keeps `shuffle=True`, forbids extra draws in callbacks and the frozen evaluation, gates on the hash of the fetched mapped batch indices; §7.1; §12; §13 RNG and resume rows.
- **C15**: §2.2 pruned against rounded to zero; §13 edge case; §18 cost check 1. **C16**: §3.1(a), §3.2. **C17**: §7.3 reference identity list; §13 `qkernel` at end of build. **C18**: §7.3 shims verified here (installed 0.1.9 searched) and the binding method. **C19**: §3.2 per-position widths. **C20**: §2.1. **C21**: §2.3. **C22**: §2.2 build timing; §13 gate. **C23**: §9 blocker and Required bullets. **C24**: §9 option-0 steps. **C25**: §9 departures 2, 4, 5. **C26**: §9 MHA handler list. **C27**: §3.2 grid exactness; §13 cost gates.
- **Knowledge-base corrections** to `docs/chang/` are outside this revision's write scope and are for the orchestrator.

**§19.2 renumbering (v0.3 → v0.4):** old 1 → 1; new 2 (OD-13), 3 (OD-15), 4 (OD-12); new 5 (which run the rule is applied to); old 3 → 6; old 7 → 7; old 6 → 8; old 5 → 9; old 2 → 10; old 4 → 11.

**Not applied as written, and why:**
- Critical B14's "in each `case_info.json`": replaced by a provenance sidecar (reconciliation above).
- Critical B15's "the orchestrator records now … the sha256 of every file": recorded by the orchestrator in the evidence-hash file, which this author cites by path and list hash but did not open or verify, by instruction.
- Physics C13 and critical B15's prompt practice: actions of the orchestrator; the STUDY states them.
- Chang knowledge-base corrections: outside the write scope.

### A.3 v0.3 (2026-10-17)

The change-log text of v0.3, verbatim. Its line citations of `decisions.md` refer to that file as it was at 06:32 on 2026-10-17, and its section and item numbers to v0.3.

**v0.3 (2026-10-17, experiment-designer).** This version applies `review/critical-v0.2.md` (1 A, 8 B, 14 C), `review/physics-v0.2.md` (2 A, 7 B, 7 C) and `review/chang-v0.2.md` (0 A, 5 B, 14 C). It also applies the orchestrator's resolutions (a)–(e) of 2026-10-17 (`.claude/memory/decisions.md:51-58`).

v0.2 is kept in `review/snapshots/v0.2/` with `SHA256SUMS`. It contains STUDY `af8c89de…` (the version all three reviews read), plan, protocol draft, OPEN-CHOICES and the three reviews. The per-finding record is in Appendix A.

What changed, and why:
- **Binary scales are learned-width constants [OD-13, resolution a].** They were exact float32 values of about 24 bits. The exact scale priced the comparison before any training (physics A1, critical B6). Each binary scale now passes through an HGQ quantizer with Chang's weight configuration, and both forms quantize the exported per-column magnitude (§2.2–2.3).
- **The three seeds are a screen [OD-15, resolution b].** Verdicts come only from a fresh-seed confirmation, which needs Kai's budget approval (physics A2, critical B9).
- **The cap stays at 400 GPU-h [OD-8, resolution c].** The alternative of about 475 GPU-h, which would keep the Q11 controls, is put to Kai (critical B10).
- **The binary forward is exact [OD-16, resolution d].** The surrogate is rewritten so that the forward is exactly the binary kernel in float32 (critical A3), with full moment-path gradients (chang B3).
- **Q16's 350,000 is in H units [OD-12, resolution e].** This is recorded as a dated delegated reading, not a Kai decision (critical B5).
- **The headline rule of §3.4 is completed and frozen (critical B7).** The case list is pinned, the timing criterion defined, the C-synthesis fallback removed and the factored full-model export not attempted.
  - Rule-block sha256: `d9f2e386d04710e1a424ad5f4511ea830f14b17c1e86afb15fa8959da7168dbb`.
  - Recompute with `sed -n '/^<!-- headline-rule v0.3 begin -->$/,/^<!-- headline-rule v0.3 end -->$/p' STUDY.md | shasum -a 256`. The digest covers both marker lines and everything between them.
  - Script hashes at the freeze (sha256):

    | file (under `preflight/da4ml-test/scripts/`) | sha256 |
    | --- | --- |
    | `cases_bc.txt` | `1ba9c416b1c4c6007ad07361bd95b1e902058e66039df89e4b5be1a8928a3191` |
    | `stage_b_build.py` | `db49a4041ff1a9a041e9a3d9c87bded997ff3adc96844366941f981028ba83d6` |
    | `stage_b_build_pos.py` | `f214dc3430a0871ee39119a93a719da3448d0fe9868e4cbae70520e6dc9a00d2` |
    | `run_case.sh` | `1253773bd99b58015620e0e99bd12f5e575592d772545a453db43245004169f8` |
    | `synth_one.sh` | `32201339fcab6d42cad096b393fa7d41ca0380e5948f718422f4c07869ee9bb9` |
    | `vivado_ooc.tcl` | `70d269f430e8fe364ae660f9f13fa81bcb8f7c0a9c373f0a1dcecbe43eb14c32` |
    | `clock.xdc` | `e4ec5851c4fff9a4f8c0caf420e487be9e43c2a8f3ce83cb9954b3ddacb4339f` |
    | `common.py` | `cfb50a22bb95a5746e31ff456bd6b8721e9cdad684babd02c0789361e266a1fa` |
    | `parse_reports.py` | `a889ce8ff21f94eae61f9322e332930e99f22d8beb81eadca647c0922707ae38` |
- **Blinding record.** It is stated in full at §3.4. In short:
  - Stage-A da4ml files existed at 05:17 on 2026-10-17. v0.2 was saved at 05:28 without a hash of its rule.
  - Stage-B/C result files appeared at 05:58, before this rule was written.
  - The author of v0.3 opened none of the result files. Its pre-result status therefore rests on attestation, and every v0.3 change to the rule moves toward H1 or is neutral.
  - Hashes of the result files are left to the orchestrator, because this revision was instructed not to open them.

The per-finding record of v0.3, verbatim:


**Orchestrator resolutions (`decisions.md:51-58`):**
- **(a)** → [OD-13]: §2.2 scale quantizer; §2.3 FB and BF on the exported magnitude; §3.1 native rule (a′), NR and penalty (b); §3.2 binary-site widths; §3.5 recomputed examples; §5 GR and BE; §13 gates; §14 stop; §15; §19.
- **(b)** → [OD-15]: §1, §6, §8.6–8.7, §10 S4 row, §17, §19.2 item 4.
- **(c)** → [OD-8]: §0, §10 premise note, §19.1, §19.2 item 5.
- **(d)** → [OD-16]: §2.2 surrogate; §2.3; §13 binary-layer gates.
- **(e)** → [OD-12]: §0, §7.2 amendment note under D6.

**critical-v0.2:**
- **A3**: applied by resolution (d): exact surrogate in every form; every consumer reads the exact product; training-mode kernel gate on CPU and GPU (§2.2, §2.3, §13). The penalty gate now recomputes (b) independently from quantizer variables.
- **B5**: §0 rows [OD-11]–[OD-16]; §5 reuse and §8.8 thresholds relabelled; amendment notes under D6 (§7.2) and D3 (§2.2); item 9 split (the restricted domain is removed, §8.2); §19 rewritten; §11 mulder per `decisions.md:50`.
- **B6**: superseded by resolution (a). The §15 sentence is replaced by the derived ratios and mechanism (§3.5, §15). The §14 stop is rewritten as "initialization readings recorded in PREFLIGHT", because Kai's β-rounding decision was replaced by (a). The OD-13 expiry no longer applies.
- **B7**: §3.4 rule completed. (1) 2b is not attempted. (2) The case list is the 20 stock/fact pairs of `cases_bc.txt` (recounted: 40 of its 44 lines; the review's "22" is a miscount), bit-exactness on every listed pair line including bias, with `hgq4` lines as context. (3) Timing is WNS ≥ 0 post-opt and II = 1 per stock line. (4) No C-synthesis fallback; H1 if Vivado is unavailable. (5) Script hashes and the blinding record in the change log and §3.4; rule-block sha256 in the change log. Not applied: the `decisions.md` entry (outside this revision's write scope) and the result-file hashes (not opened by instruction). Both are for the orchestrator.
- **B8**: §8.5 evaluated state; post-trace reported beside; C4 pairs post-trace accuracy; "allowance exceeded after trace"; the v0.1 sentence is restored and its restoration logged here.
- **B9**: verdict-specific families: intersection-union C2, C3 sharing α, 7-contrast descriptive labels (§8.7); §17 falsifiers with reachable thresholds; §19.3 alternatives. Because of resolution (b) they apply to the confirmation, and the screen has no verdicts.
- **B10**: §0 premise note; §10; §19.2 item 5.
- **B11**: §8.10 pre-registered pull with trigger; §16 sentence replaced.
- **B12**: §13 runtime-image row; §7.3; §10 note that r=3 is TF-era; §19.2 item 7.
- **C2**: decisions.md references re-derived from the current file (shift +14, not +6, after the 51-58 insertion); the resubmission reading is at `:72-76`.
- **C3**: native-dtype rule (§3.2).
- **C4**: axis wording (§2.3).
- **C5**: direct path and moment paths stated, combined with chang B3 (§2.3); gate in §13.
- **C6**: u_c called a bound; gradient to max F only (§3.2).
- **C7**: fallback clause dropped (§7.2).
- **C8**: e_350 = 0, never squeezed, epochs not checkpoints, and later cycles defined (§8.8).
- **C9**: full paths to Chang's tools (§4).
- **C10**: adaptation flagged (§18, §19.3).
- **C11**: confirmation analysis defined (§8.7); option two listed as not adopted.
- **C12**: "native where defined" replaced by NR, defined at every site; descriptive-front checkpoints in the freeze list (§8.5).
- **C13**: planning versus all-inclusive break-even (§10).
- **C14**: no kq built or regularized at binary sites (§2.2, §13).
- **C15**: CH-8 row mentions the overrides and seeding (§5); the plan.md window wording is fixed; the OPEN-CHOICES supersession is noted in §10. The note inside OPEN-CHOICES.md itself is not applied (outside the write scope) and is for the orchestrator.

**physics-v0.2:**
- **A1**: resolution (a), option (i) (§2.2–2.3, §3, §3.5, §15); options (ii) and (iii) listed in §19.1.
- **A2**: resolution (b), framing (a), screen then confirm (§6, §8.6–8.7); sign-pattern bounds in §6; funding routes in §19.2 item 4.
- **B1**: thesis sentence (§1).
- **B2**: descriptive rank outputs; "validation-selected configuration" (§8.4).
- **B3**: domain from CH-8 alone, infeasibility as a deficit, cost-reach advantage, and the post-hoc domain deleted (§8.2, §8.4, §8.7).
- **B4**: calibration ratios (pending) in §3.4; deciding-point ratios and the "not a LUT advantage" label (§8.7, §9).
- **B5**: mechanism in §1 and §15; logging in §12; ternary control listed as not run (§5).
- **B6**: §2.3 moment-dependence corrected; logging in §12; §15.
- **B7**: part-capacity order and PREFLIGHT gate (§9, §13), kept out of the headline rule, whose decisive stage is block-level.
- **C1**: plain summary at the top; Q glossary in §0; F-code letter Q for the HGQ group; "held-out test set" wording; per-finding record moved to this appendix. A short what/when/why change log stays at the top, because `06-review.md:219-222` requires the audit trail there.
- **C2**: applied by resolution (d).
- **C3**: exported scale type and binary-arm C-simulation gate (§9); held-out state (§8.5).
- **C4**: edge cases and dwell time (§8.8, §12).
- **C5**: jet bootstrap and validation-to-held-out drop (§8.6).
- **C6**: BEST − F000 and F000 − CH-8 floor (§8.6).
- **C7**: mean-ablation (§8.8, §17).

**chang-v0.2:**
- **B1**: stack pin, shims, reference identity, CPU gate and TF32 override after import (§7.3, §13).
- **B2**: RNG flow and `_index_map` gate (§6, §13).
- **B3**: full moment-path gradients and autodiff gate (§2.3, §13).
- **B4**: H-LUT option 0 first, and the version provision in the rule (§9, §3.4).
- **B5**: "as in paper §2.2" with seven declared departures (§9).
- **C1**: sensor timing (§7.2).
- **C2**: functional model object; aggregate > 0 (§7.2, §13).
- **C3**: "CH-8 under the lab PID" (§5, §7.2).
- **C4**: MHA export text and compatibility alias (§9).
- **C5**: per-position widths stated; per-position variant as context (§3.2, §3.4).
- **C6**: trainable-variable gate (§13).
- **C7**: penalty from layer variables (§3.1).
- **C8**: float32 native contributions (§3.2).
- **C9**: data details (§4).
- **C10**: wording, columns, the `model.py:193` scope, head pooling, axis names (§2.1–2.3).
- **C11**: citation (§3.3).
- **C12**: resume state (§13).
- **C13**: DS-HGQ label (§5, §16).
- **C14**: `no_growth=False`; Chang's re-split replaced by the frozen split (§3.3, §9).
- **Knowledge-base corrections** to `docs/chang/` are outside this revision's write scope and are for the orchestrator.

### A.4 Earlier versions

**v0.2 (2026-10-17, experiment-designer).** v0.1 is kept in `review/snapshots/v0.1/` with `SHA256SUMS`: STUDY `d6179391…` (the version `review/critical-v0.1.md` read) and OPEN-CHOICES `a2800a55…`. The findings of `review/critical-v0.1.md`:
- **A1**: the PID "integration correction" rationale was withdrawn. The stock aggregation was kept and gated.
- **A2**: the FP32/W8A8 checkpoint policy became [OD-4].
- **B1**: three cost quantities, width rules, a state table and non-mutating frozen evaluation.
- **B2**: resolving power computed; the three-seed stage labelled exploratory.
- **B3**: the H-LUT build item.
- **B4**: the filter outcome was separated from a validation-only collapse diagnostic.
- **C1**: "idealized active-wall projection" wording.

OPEN-1 [D9] and OPEN-2 [D10] were recorded; OPEN-3 to OPEN-10 were adopted at default A. An operations section and a conventions table were added, and the budget was recomputed with the pre-stage.

**v0.1 (2026-10-16).** First draft.
