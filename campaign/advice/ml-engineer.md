# ML engineer’s advice: accuracy versus cost against Chang

2026-10-17. Recommendation, not a measured winner or a campaign amendment. Source paths below are relative to `/Users/kaiyamaguchi/lab/bnjettag`. This review used local files only; no training, conversion, synthesis, network or cluster work was performed.

**OPEN-1: choose binarize-then-fold, initially with the specified per-layer β. OPEN-2: pursue (b), an explicit sign-only dot product followed by one combined scale per output channel; keep (a) as the conservative accounting/export reference.** This is my best engineering bet for preserving accuracy while exploiting binary structure. I would not choose fold-then-binarize with one layerwide β, or choose power-of-two β merely to make the EBOP number smaller. None of these choices guarantees a better front.

There are two different goals here: the clean experiment “change only the weight rule,” and the strongest model we can build. For the clean experiment, keep the specified β = mean|W| and Chang’s moment path. For a later accuracy-oriented extension, my first candidates are channelwise scales at layers without BN and binary-consistent BN moments if diagnostics show a normalization mismatch. These must be named departures, not silently included in the factorial. The current draft specifies the layerwise full-map identity STE and leaves placement/accounting open (`campaigns/2026-10-16-chang-bitlinear/STUDY.md:32-40`; `campaigns/2026-10-16-chang-bitlinear/OPEN-CHOICES.md:5-19`).

**Why preserving the BN scale matters.** Let i index inputs and c index output channels. Write the latent float kernel as W, the signs as S = Sign(W − mean(W)), and the weight scale as β_w. Distinguish β_w from Chang’s cost-penalty coefficient, which I call λ. His callback calls that coefficient `beta` (`reference-code/HGQ2-examples/jsc150/run_train.py:79-95`).

For a frozen BN state, define

\[
s_c=\frac{\gamma_c}{\sqrt{v_c+\epsilon}},\qquad
d_c=Q_b(b_c-\mu_c s_c).
\]

Chang quantizes W_ic s_c with `kq`; his bias is quantized separately as d_c. Training moments come from the float-kernel projection of the already input-quantized x; inference/export uses moving moments (`reference-code/HGQ2/src/hgq/layers/einsum_dense_batchnorm.py:158-208`).

With **binarize-then-fold**, the proposed coefficient is

\[
K_{ic}=S_{ic}\,\beta_w s_c=S_{ic}g_c,\qquad
y_c=g_c\sum_i S_{ic}x_i+d_c.
\]

Thus a single β_w still leaves a different signed magnitude g_c for every channel. A negative BN scale reverses the column’s signs; a zero scale removes its input-dependent contribution. The shifted bias stays. This is an algebraic derivation of the proposed replacement at the fold above, not an implemented result. Do not subsequently round each coefficient with independent HGQ grids: that can destroy the shared column magnitude and the factorization.

With **fold-then-binarize**, first form F_ic = W_ic s_c, then replace it by β_F Sign(F_ic − mean(F)). One global β_F removes the relative channel magnitudes from the kernel. Kai’s reasoning is correct in that sense. It does **not** make BN irrelevant: s still changes the global mean, sign boundaries, global β_F and shifted bias. In fact, global centering means a negative s_c is not necessarily just a column sign reversal. Even a column with s_c = 0 can acquire nonzero binary coefficients when β_F elsewhere is nonzero. That is a poor way to preserve the channel suppression learned by BN.

With **fold-then-binarize and channelwise β**, the scale becomes

\[
\beta_{F,c}=\operatorname{mean}_i|F_{ic}|
=|s_c|\operatorname{mean}_i|W_{ic}|.
\]

That restores channel magnitudes, including zero channels. It is a much more credible competitor than global post-fold β. But changing β granularity does not change the centering rule automatically: global α after folding still couples channels. Pre/post folding become algebraically equivalent only under additional compatible centering/sign conventions, with zero cases handled explicitly; they are not generally equivalent under this campaign’s global α.

The physical analogy is a bank of microphones: signs specify which signals add or subtract, while channel scales are their volume controls. Removing the volume controls can change what reaches the next nonlinearity. Here this matters because fused layers feed ReLU and attention/FFN branches meet residual additions (`reference-code/HGQ2-examples/jsc150/model.py:192-208`). Later learned widths can help accommodate ranges, but are not a proof that these two models have equal expressive power.

**Trainability is the main uncertainty in OPEN-1.** Implement the requested identity surrogate for the entire binary map:

\[
B_{\rm STE}(W)=W+\operatorname{stopgrad}(\beta_w S-W).
\]

Its forward value is β_w S and its backward derivative with respect to W is the identity. This differs from attaching an identity derivative only to Sign and then multiplying by β_w, which scales the weight gradient by β_w. No gradient passes through α or β_w in this full-map surrogate (`campaigns/2026-10-16-chang-bitlinear/STUDY.md:32`).

For pre-fold binarization, apply that surrogate **before** multiplying by s. The direct weight-gradient path then contains s_c; gradients through the original BN moments and shifted bias remain. Large inverse variance can amplify updates, a small γ can weaken the direct path, and sign changes make the forward response discontinuous. Do not wrap the whole folded coefficient in an identity surrogate against raw W: that would change this gradient. For post-fold binarization, apply the identity surrogate against F. Its backward pass can suggest smooth benefits from changing γ even where the actual binary forward kernel cannot express a smooth channel-amplitude change. These are deductions from the surrogate and the differentiable fold/moment operations (`reference-code/HGQ2/src/hgq/layers/einsum_dense_batchnorm.py:165-172,189-208`).

The more serious issue is **which projection BN normalizes**. Under OPEN-1 A, the output before activation is approximately

\[
s_c\{(xB)_c-\mu_c(xW)\}+b_c,
\]

ignoring bias quantization in this explanatory equation. BN estimates the mean/variance of xW, but the forward signal uses xB. There is no guarantee that the latter is centered or has the intended variance. Chang already has a float-statistics/quantized-forward distinction; making the kernel binary can enlarge it. Preserving his path gives a controlled comparison, not automatically well-normalized binary activations (`reference-code/HGQ2/src/hgq/layers/einsum_dense_batchnorm.py:168,189-208`).

If this mismatch is large and accompanies failed learning, **OPEN-1 C—moments from xB—is my preferred rescue hypothesis**. It normalizes the signal actually used. It also changes the experiment, gradients and moving statistics. A post-training BN recalibration would likewise be a declared intervention, not an invisible export fix. Neither should be selected from held-out accuracy.

For accuracy, channelwise absmean β is most attractive **without BN**, where there is no γ already providing a trainable channel scale. With pre-fold BN, γ already supplies that flexibility, so adding another learned channel scale may be redundant and harder to identify during optimization. At non-BN sites, a learned positive channel scale, initialized from channelwise absmean and trained with an explicitly defined surrogate, could outperform the prescribed detached statistic at the same number of deployed scale operations. This is a hypothesis, not evidence of an accuracy gain, and departs from “no gradient through β.” Changing α to a channelwise mean is another independent change and should not be bundled with it.

Binary weights also lose individual connection magnitudes and individual zeros. Under a tight λ ramp, their fixed sign cost leaves fewer weight-width adjustments, so savings may come disproportionately from activation widths or other layers. Full-width arbitrary coefficients in option (a) can intensify that pressure. Track validation accuracy over time, per-layer cost and activation widths, sign-flip rates, β/γ, binary-versus-float moment mismatch, train-versus-inference outputs, and attention branch use. Chang’s acceptance filter is `val_accuracy > 0.5` and `ebops < 5e5`; it is not a definition of collapse (`reference-code/HGQ2-examples/jsc150/run_train.py:83-91`). The historical 350k entry lists several hypotheses and explicitly has no readout; it establishes neither a BN cause nor inevitable failure of this N=8 design (`.claude/memory/experiment-log.md:1-7`).

**Cost: the three representations answer different questions.** Native HGQ EBOPs multiplies input widths by kernel-quantizer widths, adds its bias term and applies its parallelization factor. It does not inspect binary column structure or predict a synthesized adder network (`reference-code/HGQ2/src/hgq/layers/core/einsum_dense.py:85-93`).

| Choice | Accuracy/training implication | Cost and engineering judgment |
| --- | --- | --- |
| (a) Full-width coefficients | Keeps the specified forward coefficients; detached discrete width measurement provides no direct width-learning gradient for the replaced binary kernel. | Simplest reference export, but potentially expensive EBOPs. It requires a new width/cost adapter; it is not stock Chang counting merely because the contraction formula is reused. |
| (b) Signs, then combined channel scale | Same real-arithmetic function as (a); deployed fixed-point equality still needs verification. | Best candidate for exploiting binary structure without changing β. Highest converter/accounting effort; actual LUT advantage remains unmeasured. |
| (c) Power-of-two β | Changes the binary forward map; scale jumps add another discontinuity. | Useful hardware-oriented ablation at non-BN sites. It does not remove an arbitrary BN scale after folding. |

For (a), native `kq.bits_` is a learned format, not a scan for the smallest exact representation of each exported constant. Replacing it with exact coefficient widths is an **extended cost definition**. Finite floating-point coefficients are dyadic, but exact export can require a long significand. “Two possible values” is a one-bit selector description, not a one-bit fixed-point representation of an arbitrary ±g. Also, HGQ’s `SAT_SYM` cost width excludes the extra sign bit whereas `SAT` includes it; storage width and cost width must not be interchanged (`reference-code/HGQ2/src/hgq/layers/core/einsum_dense.py:85-93`; `reference-code/HGQ2/src/hgq/quantizer/internal/fixed_point_quantizer.py:93-105`).

For (b), combine β and BN into **one** g_c. Even a layerwide β requires scaling each channel’s distinct accumulated result. As an explanatory cost model for a fully parallel layer with M inputs, C outputs and P output positions, one might write

\[
E_a\approx P\sum_{i,c}a_i w_{ic},\qquad
E_b\approx P\left[\sum_{i,c}a_i+\sum_c u_c t_c\right],
\]

where a_i is input width, w_ic the full coefficient cost width, u_c the signed accumulator width and t_c the scale cost width. These are schematic estimates, not measured totals or a finished cost specification. Actual per-value widths, sharing/parallelization and bias must follow the graph. The wider accumulator matters: scaling the sum is not priced as scaling a single input. Define treatment of zeros, shifts, negation and rounding; do not add a second adder charge on only the binary side when the baseline contraction convention already models the operation through its bit-product proxy. For an explicit gate/adder model, apply that model to both sides.

With heterogeneous input steps, align operands to a common exact sum grid before adding. Preserve enough signed range, multiply by g_c, add the existing quantized bias, and retain the original activation/output-quantizer boundary. Do not move the scale across ReLU, residual addition or another quantizer on algebra alone. Reassociation can change rounding or overflow. A constant multiply may itself become shifts and additions; “one multiply” describes the graph, not a DSP count or a LUT prediction.

For (c), a fixed power-of-two scale can be a binary-point shift with no general multiplier, but rounding, lost fractional bits, range growth and saturation still need representation. More importantly,

\[
K_{ic}=S_{ic}2^k\frac{\gamma_c}{\sqrt{v_c+\epsilon}}
\]

is generally **not** ± a power of two. One must retain and charge the BN scale, or constrain/round the combined g_c, which changes BN as well as β. Rounding post-fold β per channel can produce shifts, but rounds away some of the very channel magnitudes we want to preserve. Rounding a global post-fold β retains the stronger accuracy objection above. I would test this only after an arbitrary-scale reference works.

**What hls4ml actually verifies about “add first, multiply once.”** The HGQ handler loads `qkernel`/`qbias` when available, including the fused layer class; the EinsumDense handler exports those as ordinary constant kernel/bias arrays. No sign-plus-scale metadata is produced there (`reference-code/hls4ml/hls4ml/converters/keras_v3/hgq2/_base.py:109-127`; `reference-code/hls4ml/hls4ml/converters/keras_v3/einsum_dense.py:66-80`).

The relevant DA code does the following:

- Dense/Conv: constructs a da4ml `FixedVariableArray`, evaluates `out = inp @ kernel`, adds bias, calls `comb_trace`, stores `sol.cost`, and generates HLS code (`reference-code/hls4ml/hls4ml/backends/vivado/passes/distributed_arithmetic.py:116-150`).
- EinsumDense: for each inplace matrix slice, evaluates `out = inp @ kernel[i]`, calls `comb_trace`, accumulates `sol.cost`, and emits the corresponding generated function. The pass does not extract β_c or explicitly build a sign-dot followed by a scale (`reference-code/hls4ml/hls4ml/backends/vivado/passes/distributed_arithmetic.py:362-409`).
- Input formats are passed through K/I/F information; the Einsum path takes maxima over free data positions. This can affect hardware cost relative to the finer per-value training proxy (`reference-code/hls4ml/hls4ml/backends/vivado/passes/distributed_arithmetic.py:28-48,81-91`).

For columns ±β_c, **the mathematical factorization exists, but automatic realization is unverified**. The inspected hls4ml wrapper neither implements that factorization itself nor tells us whether da4ml’s matrix operator/solver discovers it, an equivalent shared shift-add network, or a more expensive network. Do not claim either “DA automatically does it” or “DA cannot do it.” The solver receives the constants and may exploit them; its implementation and generated circuit are the missing evidence. `sol.cost` is a solver result, not a measured FPGA LUT count.

The local checkout declares da4ml as an optional dependency (`reference-code/hls4ml/pyproject.toml:29`). A read-only `python3 -B` import-spec probe returned `None` for da4ml in the default interpreter, consistent with the stated Mac limitation. I did not install anything. I can verify the wrapper and the algebra; I cannot verify da4ml factor extraction, generated C++, bit-exact simulation, LUTs or timing here.

The decisive later check is the same frozen ±g_c matrix through both the stock constant-matrix path and an explicitly factored path, with identical input/output formats. Include non-power-of-two, power-of-two, negative and zero scales and heterogeneous input grids. Compare generated arithmetic, exact fixed-point outputs, solver cost and then synthesized LUT/timing. If stock DA is equally good, keep the simpler converter. If not, implement the factorization explicitly. An explicit decomposition may fit existing da4ml operations; a da4ml core change is not yet proven necessary. Its feasibility still needs that test.

**Implementation effort and risks.** These are relative engineering estimates, not time estimates.

| Change | HGQ2 | hls4ml | da4ml |
| --- | --- | --- | --- |
| Pre-fold binary, layer β | Moderate: custom fused kernel/export path, correct STE, original moments/bias retained, new cost adapter. | Moderate for (a): register custom layer and preserve effective constants. Higher for (b): explicit factorization and exact intermediate formats. | Existing constant-matrix entry point for (a); factorized behavior for (b) unverified. |
| Post-fold, layer β | Smaller forward-code change at `kq`, but substantial semantic/gradient risk; still needs correct cost/export behavior. | Ordinary constants are straightforward in principle. | Same uncertainty about scale extraction. Lower coding effort is not evidence of better accuracy. |
| Channel β, either placement | Moderate additional shape/axis and serialization work; preserve the chosen α and detach rule. | One scale per true output channel, with signs and bias placed correctly. | Similar operation count to layer β after factoring; differing constants can change sharing and LUT cost. |
| Power-of-two β | Reusable exponent ideas, but new behavior relative to the contract. | Easier exact shift representation at non-BN sites; arbitrary folded BN remains. | No arbitrary multiplier for a genuine shift; width/routing/addition costs remain to be measured. |

Channel means must reduce the **contracted input axes**, not blindly all but the last tensor axis. Q/K/V output channels include head and feature indices, whereas O contracts both head and feature dimensions. The released attention builds separate nested projection layers; replacing only top-level dense layers misses them (`reference-code/HGQ2/src/hgq/layers/attn/mha.py:150-223,275-295`). The hls4ml MHA handler invokes its own dense handler for those projections, so adding a new top-level converter registration alone does not establish factorized attention export (`reference-code/hls4ml/hls4ml/converters/keras_v3/hgq2/multi_head_attention.py:77-113`). QK and attention-times-V are still activation products, not constant binary matrix multiplies (`reference-code/HGQ2/src/hgq/layers/attn/mha.py:406-413`).

All options share an existing whole-model export obstacle: Chang’s affined tanh uses heterogeneous table formats, while the inspected unary-LUT handler names the native unary class, rejects heterogeneous tables and reads scalar output formats. Establish exact export on CH-8/F000 first; preserve the learned affine and per-position formats rather than homogenizing training (`reference-code/HGQ2-examples/jsc150/model.py:181-198`; `reference-code/HGQ2/src/hgq/layers/activation.py:76-108`; `reference-code/hls4ml/hls4ml/converters/keras_v3/hgq2/unary_lut.py:18-32,64-75`). Configure DA on intended constant EinsumDense nodes and inspect coverage; the backend accepts DA there but maps ordinary variable-input Einsum to latency (`reference-code/hls4ml/hls4ml/backends/vivado/vivado_backend.py:832-839,870-880`).

**The paused campaign helps as a parts library, not as the new implementation.** In the following bullets, `old/` means `campaigns/2026-10-12-mha-bitnet/code/bnhgq2/`.

- Reuse the pattern for centered signs, Sign(0) = −1, full-map identity STE, serialization and nested projection configuration. Replace its actual targets: `MhabKernelKBI` implements ±1, power-of-two and BK roles, and its β helper has a floor. Its alternative `lab` gradient propagates scale/centering derivatives. None is the new arbitrary-absmean rule unchanged (`old/mhab_quant.py:215-280`; `old/mhab_model.py:93-129`).
- Reuse power-of-two exponent/pinning checks only for a separately named (c) arm. The exponent synchronization/freezing machinery encodes the old protocol; it must not freeze Chang’s learned steps. The old export pins formats and checks exact representability (`old/mhab_quant.py:328-370`; `old/mhab_export.py:106-139`).
- Do not transplant the BN wrapper. It finds standalone `QBatchNormalization` layers and switches them to moving statistics; BK multiplies by a detached β after the BN input quantizer and pairs it with a reciprocal kernel-gradient scale. This is not Chang’s fused float-W moment path (`old/mhab_bn.py:30-49,72-119`; `old/mhab_quant.py:263-265,367-370`; `reference-code/HGQ2/src/hgq/layers/einsum_dense_batchnorm.py:189-208`).
- The old study omits β at its standalone BN sites and has BK-A8 as the β-kept control (`campaigns/2026-10-12-mha-bitnet/STUDY.md:1052-1057,1305`). Algebraically BN can cancel a positive input scale when its moments scale consistently and ε is negligible; finite ε, quantization, moving statistics and surrogate gradients prevent assuming exact training equivalence. With moments from xW rather than xB, that cancellation argument does not justify dropping β here. BK is not a ready hardware solution: its exporter explicitly excludes it (`old/mhab_export.py:47-55,125-127`).
- Reuse export-state copying and single-wrap checks as design patterns. Do not reuse the homogeneous tanh shim or the old HGQ-8 comparator unchanged: they deliberately use homogeneous/per-tensor formats and standalone BN (`old/mhab_export.py:82-99,142-175`; `old/hgq8_model.py:3-20`). HGQ’s metaclass wraps calls for cost and output quantization; the old MHA code warns about wrapping inherited calls twice (`reference-code/HGQ2/src/hgq/layers/core/base.py:57-114`; `old/mhab_model.py:66-71`).
- Avoid the misleadingly named legacy `binarize.py` for this task: it computes mean|W − mean(W)| plus epsilon, uses `np.sign` with a zero state, and contains LayerNorm-specific scale-fold rules. Those differ from the present contract (`old/binarize.py:27-62`).

**A fair front requires one explicit cost contract.** Keep Chang’s released training and native reported front intact. Separately report native/extended coefficient accounting and the common deployed-graph accounting for both Chang and binary candidates; apply equivalent optimizations to both. Do not compare (b)’s smaller structural total against Chang’s native total and call the difference a measured gain. If only unchanged native EBOPs is admissible as the primary axis, neither a new exact-width adapter nor a new structural metric automatically qualifies: first establish a faithful representation in the native quantizer/cost machinery, or explicitly amend the primary metric. Meanwhile report (a)’s conservative extension and (b)’s structural savings/LUT result separately. The lab convention explicitly separates augmented cost from native totals (`docs/conventions/quantization-and-cost.md:21-25`).

Specify the BN state at every cost measurement. My proposed binary penalty uses the current batch-state coefficients; checkpoint feasibility and export use the saved moving-state coefficients. These need not agree. Log both, and make selection depend on a non-mutating evaluation of the saved inference state. This is a declared selection integration change: stock HGQ updates counters on training/tracing calls, and FreeEBOPs reads those stored counters at epoch end; it does not recompute saved-checkpoint inference coefficients (`reference-code/HGQ2/src/hgq/layers/core/base.py:94-104`; `reference-code/HGQ2/src/hgq/utils/sugar/ebops.py:7-15`). Keep the binary penalty, logs and any later controller tied to their declared sensor, and verify no missing/double-counted nested layers. The review already identifies this state-contract gap (`campaigns/2026-10-16-chang-bitlinear/review/critical-v0.1.md:48-65`).

**What would change my mind.** I would favor channelwise post-fold β if paired validation runs showed a better front without damaging channel suppression or export fidelity. I would switch to binary-consistent moments if measured moment mismatch tracked instability and a controlled validation ablation fixed it. I would choose power-of-two scales if synthesis showed a meaningful hardware advantage and paired accuracy comparisons supported the trade. I would retain stock DA export if it produced circuits as good as the explicit factorization. None of those outcomes has been measured here.

If BN-site binary layers hurt the front, selective binarization is a success criterion, not a compromise to conceal. The existing attention-only factorial cell already avoids binary fused-BN sites: projections are ordinary dense layers, while FFN expansion and the hidden head are fused (`reference-code/HGQ2/src/hgq/layers/attn/mha.py:152-223,288-295`; `reference-code/HGQ2-examples/jsc150/model.py:199-208`). A later “binary only where BN folding is absent” arm could add FFN contraction while leaving expansion HGQ; that splits the current FFN group and must be declared separately. Keep the protected classifier HGQ. My priority is a reproducible held-out accuracy gain at comparable cost, confirmed by LUT/timing at deciding checkpoints—not maximizing the number of binary layers.
