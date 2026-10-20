# Open choices — chang-bitlinear-20261016

2026-10-16 · draft v0.1 · for Kai Yamaguchi. These recommendations are **not decisions or launch approval**. The matching `[OPEN-n]` marks the working assumption in `STUDY.md`. All source paths are relative to the lab root. Choices already settled by Kai or a usable released-code default are not reopened; in particular the BetaPID gains are available and copied.

## 1. [OPEN-1] Where should binary weights enter Chang's fused BatchNorm?

- **A — Recommended:** binarize the latent kernel before the fold, exporting K_ic=β_w Sign(W_ic−mean(W))·γ_c/sqrt(var_c+ε); keep Chang's raw-W statistics path and fused bias quantizer. This preserves his BN mechanics and a binary factor times a channel scale, but requires a new kernel/cost adapter.
- **B:** replace the quantizer at the existing fold-then-quantize site by binarization of the folded kernel. This is closest to the location of Chang's quantizer, but α/β are then computed on BN-scaled weights and the binary parameterization differs.
- **C:** binarize before BN and also compute BN moments from binary projections. This gives binary-consistent batch statistics but changes the released moment path as well as kernel quantization.

No HGQ requantization after the binary/fold product in A: independent per-value rounding would generally lose the two-valued factor. Neither A nor B is uniquely specified by the sources. Default A is used in STUDY §2; test inference and gradients, including negative/zero channel scales. Source: `reference-code/HGQ2/src/hgq/layers/einsum_dense_batchnorm.py:158-211`; Kai Q3/Q4.

## 2. [OPEN-2] How should β_w and folded BN scales be represented and charged?

- **A — Recommended:** retain β_w=mean(abs(W)) exactly in the training dtype, export the folded constant coefficients as exact dyadic fixed-point values, and price their full effective widths with the HGQ contraction/bias formula. Recompute these discrete binary-site widths each training step and detach them in the cost backward pass; native activation-width and nonbinary-quantizer gradients remain active. The cost adapter must feed the loss, logs and PID identically. This is conservative and may remove the apparent one-bit EBOP advantage; the adapter's total must be labelled separately from stock native EBOPs.
- **B:** export explicit sign-only dot products plus one tensor/channel scale after accumulation; charge scale multiplication and accumulator formats explicitly. Potentially cheaper hardware, but requires a custom converter/cost graph and validated rounding order.
- **C:** round β_w to a power of two, or absorb it into a following quantizer. Powers of two alter Q4; absorption is acceptable only if exported behavior is proven identical and all cost is retained. Neither is a free stock feature to assume.

Default A is used in STUDY §§2,6–9. The adapter must reduce exactly to native cost for CH-8/F000, leaving their builder, learned-width penalty and gradients unchanged; audit emitted coefficient widths diagnostically only. Distinguish the paper's native 350k target from any changed accounting. If Kai requires unchanged native EBOPs as the sole primary axis, this choice must be settled before building or claiming a front: do not silently compare a custom binary total against an unaudited native baseline. Sources: `reference-code/HGQ2/src/hgq/layers/core/einsum_dense.py:85-93`; `docs/conventions/quantization-and-cost.md:21-25`. The sources do not supply a certified binary+BN arbitrary-scale implementation.

## 3. [OPEN-3] Should this stage use three seeds or fund eight-seed confirmation?

- **A — Recommended:** three seeds per configuration/control/PID arm now, top-1 as the primary endpoint; keep AUC descriptive and plan any eight-seed confirmation separately before unblinding.
- **B:** eight seeds for every admitted arm from the start. This strengthens intervals but multiplies the three-seed production count/cost by 8/3: all 16 arm identities would be 128 runs, about 821.33 training GPU-hours at the optimistic 3-second rate, before reserve (derived from STUDY §9).
- **C:** three-seed factorial, then expand only BEST/CH ramp and PID pairs to eight using validation-only decisions before unblinding. This adds 20 runs, about 128.33 GPU-hours at that rate, while nonselected arms remain three-seed exploratory comparisons.

Default A is used in STUDY §§4–5,9. No top-1 variance for this exact design is sourced, so no detectable gap is promised. A confirmatory expected AUC gap below 0.005 requires eight seeds under `docs/methodology/03-phases.md:81-82` and `docs/conventions/jet-tagging-metrics.md:38-46`; A does not waive that rule.

## 4. [OPEN-4] Which precise rule should define the best configuration and a better front?

- **A — Recommended:** STUDY §7's six fixed EBOP allowances; attainable validation-selected step envelope; one configuration maximizing mean paired validation accuracy across all grid points/seeds; complete coverage required; paired t intervals, with a 49-contrast Bonferroni family for the seven fronts and seven potential PID comparisons (only BEST is run; the unrun six remain in the family to protect adaptive selection on reused seeds). Whole-front superiority requires positive simultaneous lower bounds everywhere.
- **B:** piecewise-linear interpolation of each seed's front on a frozen common domain, selecting by integrated validation advantage. Smooth comparisons, but interpolated points are not actual deployable checkpoints and must not be synthesized or described as measured models.
- **C:** make only 350,000 EBOPs confirmatory and the remaining front descriptive. More power and fewer comparisons, but narrows Kai's whole-front objective and needs his explicit choice.

Default A includes the tie/NA rules, negative-score BEST behavior, dominance reading and verdict labels in §7. The six grid values and multiplicity family are proposals, not paper defaults. Source for seed-paired 95% t intervals and flat-if-zero: `docs/conventions/jet-tagging-metrics.md:38-46`; objective: `.claude/memory/decisions.md:49-62`.

## 5. [OPEN-5] Should PID runs start fresh with a positive initial λ or continue ramp checkpoints?

- **A — Recommended:** new paired training runs, 7,000 epochs, λ initialization 2e-8 (Chang's ramp start), HGQ2's ten-epoch PID warmup and all released gains/defaults otherwise. Tests fixed-target training independently of the preceding ramp but costs six full runs.
- **B:** resume each selected ramp checkpoint with fresh PID state and a declared continuation length/initial λ. Cheaper only if shortened, but changes the schedule and conditions the result on ramp training; it requires a further explicit protocol amendment.

Default A is in STUDY §6. Gains p=1,i=0.002,d=0 are already sourced; no gains question remains. A positive initialization is necessary under the default logarithmic controller because Chang `beta0=0` plus `init_beta=None` reaches log10(0). Independently of this choice, the controller must sense the same complete nested-layer aggregate as the front, with no missing/double-counted projections. Sources: `reference-code/HGQ2/src/hgq/utils/sugar/beta_pid.py:41-47,95-106,125-167`; `reference-code/HGQ2/src/hgq/utils/sugar/ebops.py:7-15`; `reference-code/HGQ2-examples/jsc150/model.py:295`; `reference-code/HGQ2-examples/jsc150/run_train.py:90`.

## 6. [OPEN-6] Which formats should define W8A8 beyond its fixed weight/activation widths?

- **A — Recommended:** eight total bits including sign, per-value granularity as Chang, trained step/range allocation throughout, Chang rounding/overflow, biases also eight-bit; preserve derived accumulator/carry widths. This isolates fixed precision against learned widths while retaining his graph.
- **B:** also freeze steps/ranges after calibration. This more closely resembles the old fixed-width campaign, but adds a training change and requires a new calibration/freeze schedule.
- **C:** fix weights/activations at eight bits but leave bias widths learned. Keeps more of Chang's bias behavior but makes “W8A8” less uniformly fixed precision; label it explicitly.

Default A is in STUDY §4. The release has learned KBI/KIF scopes, not a ready fixed-eight control; merely changing its initialization argument does not freeze total signed precision. Sources: `reference-code/HGQ2-examples/jsc150/model.py:185-188,277-295`; `campaigns/2026-10-12-mha-bitnet/STUDY.md:1077-1088`.

## 7. [OPEN-7] Should the GR variant isolate the β_w gradient or reproduce the old combined gradient change?

- **A — Recommended:** keep the main arm's identity STE and add only S·∂β_w/∂W, using β_w=mean(abs(W)), no floor. This directly answers whether stopping the scale gradient matters; STUDY §4 supplies the exact surrogate and zero subgradient.
- **B:** restore the old centered-gradient term as well as the β derivative, adapting to the current β definition. This is closer to old GR-A8, but any difference could come from centering or β and is a combined intervention.

Default A uses F111 as comparator; BE also uses F111, making both fixed before looking at results. GR/BE are outside BEST selection. Source: `campaigns/2026-10-12-mha-bitnet/STUDY.md:1304-1327`; identity definition: `docs/design/2026-10-11-russell-questions-recommendations.md:177-185`. The old implementation is not copied blindly because its gradient/scale choices differ from Q4.

## 8. [OPEN-8] What GPU-hour cap should Kai approve for the new campaign?

- **A — Recommended:** 400 GPU-hours total, with 50 ring-fenced for engineering/restarts and at most four concurrent GPU jobs. Reserve all 33 mandatory runs first; admit complete three-seed controls in the requested order using measured rates. At the optimistic 3-second rate all 48 runs plus reserve project to 358 GPU-hours.
- **B:** fund only the mandatory front/PID stage initially, with a proposed 300 GPU-hour cap including the same reserve. Its optimistic projection is 261.75 GPU-hours; this sacrifices optional controls and still needs throughput certification.
- **C:** approve a larger cap only after PREFLIGHT measures per-class rates and supplies a concrete revised total. At 6 seconds/run-epoch, all controls plus reserve project to 666 GPU-hours, and even mandatory runs plus reserve need 473.5.

Default A is in STUDY §9. None authorizes spending now. Preserve 7,000 epochs and every factorial seed; inability to fit means stop for Kai, not reduce the recipe. The old 3-second rate and 10% overhead were estimates (`campaigns/2026-10-12-mha-bitnet/STUDY.md:2299-2320`); 3.053333 GPU-hours was a failed smoke **pack** (`campaigns/2026-10-12-mha-bitnet/PREFLIGHT.md:744-748`). Actual class rates, K, concurrent commitments and reserve debits must determine admission.

## 9. [OPEN-9] How should released JAX precision be reconciled with the lab's TF32-off rule?

- **A — Recommended:** use JAX as released, with TF32 disabled consistently across all arms and reload checks. Retains the backend but declares one precision departure; requires a newly pinned environment and timing benchmark.
- **B:** use released JAX tensorfloat32 during training, then separate TF32-off verification. Closest training precision, but the policy exception and any numerical change at verification need Kai's decision and explicit evidence.
- **C:** reuse the old TensorFlow route with TF32 off. Saves integration work but changes backend as well as precision; old gates do not establish equivalence for the new graph.

Default A is in STUDY §6. Sources: `reference-code/HGQ2-examples/jsc150/run_train.py:3`; `reference-code/HGQ2-examples/jsc150/model.py:20-23`; `docs/conventions/quantization-and-cost.md:40-45`; old backend departure `campaigns/2026-10-12-mha-bitnet/STUDY.md:1342-1346`.

## 10. [OPEN-10] How much synthesis should support the front claim?

- **A — Recommended:** synthesize all unique validation-selected BEST/CH grid points for all three seeds plus the PID pair, up to 42 checkpoints per stage before deduplication. Prioritize 350k PID, then 350k ramp, then other allowances; copy the old seven-day admission window and 24-hour stage timeout. This can support paired hardware readings across the front, but incomplete builds leave corresponding claims incomplete.
- **B:** synthesize only the 350k ramp/PID pairs, all three seeds (at most 12 checkpoints per stage). Cheaper, but hardware evidence is limited to that allowance, even if the software front is broader.
- **C:** one seed per selected point. Provides conversion and engineering context only; no seed-averaged hardware gap or hardware-front superiority claim.

Default A is in STUDY §8. GPU cap excludes Mulder CPU/tool time. Report C-synthesis and licensed-xczu7ev Vivado post-opt separately; no substitution of C-synthesis LUT for post-opt LUT. Sources: `campaigns/2026-10-12-mha-bitnet/STUDY.md:2104-2135,2240-2250`; `docs/conventions/fpga-synthesis.md:21-41`; hardware objective `.claude/memory/decisions.md:54-55`.

## Kai's answers

- OPEN-1 (2026-10-16): first answered A, then withdrawn by Kai the same day: "i want u to do what chang does". Under discussion: "what Chang does" = fold, then binarize at the place of his per-weight quantizer (einsum_dense_batchnorm.py:158-211); sub-choice per-layer versus per-channel β at the four folded-BN layers.
- OPEN-2 (2026-10-16): Kai: "same as 1, do what chang does, can we talk about this more in depth". Under discussion: whether EBOPs charge a binary weight as its full exported constant (Chang's counting) or as 1 bit plus one scale multiply per channel.

### Final answers (2026-10-17)

- OPEN-1 (Kai, 2026-10-17, after physicist and ML-engineer advice in `advice/`): run both forms (fold-then-binarize with
  per-output-channel β at the fused-BN sites, per-layer β elsewhere; and binarize-then-fold with per-layer β), keep the
  better one on validation only, before any held-out read.
- OPEN-2 (Kai, 2026-10-17): test da4ml first: one frozen binary ±β_c matrix through stock da4ml and an explicitly factored
  path on mulder (exact outputs, solver cost, LUT/timing); then fix the headline cost count before any training result.
  Both counts are reported either way, with identical rules for Chang's model and ours.
- OPEN-3 to OPEN-10: Kai delegated (2026-10-17: "just automate everything from here on out ... imma just leave everything
  to u"). The orchestrator adopts each recommended default (A) as written above, flagged for Kai's review:
  OPEN-3 A (3 seeds per arm), OPEN-4 A plus review A2 (FP32/W8A8 checkpoint selection declared as contextual controls),
  OPEN-5 A (fresh PID runs, λ init 2e-8, released gains), OPEN-6 A, OPEN-7 A, OPEN-8 A (400 GPU-h cap incl. 50 reserve,
  at most 4 concurrent; Kai sees it when approving the authority), OPEN-9 A (JAX as released, TF32 off), OPEN-10 A.

_Note (orchestrator, 2026-10-17): the budget and run numbers above are the v0.1 draft's; STUDY v0.3 §10 supersedes them._
