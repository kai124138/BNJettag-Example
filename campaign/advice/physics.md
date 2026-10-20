**My recommendation is to preserve Chang's experimental question first.** For OPEN-1, use **fold-then-binarize at Chang's existing kernel-quantizer site, with per-output-channel β at the fused-BN sites**. Keep the stipulated per-layer β at sites without BN. Call this a *channel-scaled binary weight quantizer in Chang's model*. It is a proposed, explicit amendment to the per-layer BitLinear rule, not a decision already made by Kai. If literal adherence to that rule takes priority, use per-layer β after the fold instead, and accept that this also restricts the channel magnitudes that BN can express.

For OPEN-2, **retain the unrounded mean-absolute β, report both full-coefficient and sign-plus-scale accounting, and use synthesis LUTs to decide the hardware claim**. Use a symmetric full-coefficient comparison as the conservative primary EBOP-style result; make sign-plus-scale a separately named result until its implementation is validated. Preserve Chang's native EBOP result alongside these. Never compare binary's cheap representation against Chang's expensive representation without giving both the same accounting and optimization rules.

This advice is based on local source inspection and algebra, not new training, conversion or synthesis. All paths below are relative to `/Users/kaiyamaguchi/lab/bnjettag`. Campaign documents are proposals or historical records where identified. Only this advice file is changed.

**What “the only change is binary weights” can honestly mean.** It can mean that the only experimental intervention is replacing selected weight quantizers, including a declared scale rule and backward surrogate. It cannot mean that training trajectories, learned activation widths, or the cost penalty's weight gradients remain identical after that intervention. Nor can it mean that every deployed coefficient is literally ±1 when channel scales remain.

Chang's release folds BN into the latent kernel and then calls `kq`; it computes training moments from the input projection with the latent kernel, and quantizes the shifted bias separately. Inference uses moving moments. These are concrete operations to preserve, not just the name “BatchNorm.” (`reference-code/HGQ2/src/hgq/layers/einsum_dense_batchnorm.py:158-211`.) The transformer builder keeps the selected fused-BN sites within the FFN and hidden head, and sets the weight/bias scope to `SAT_SYM`, `b0=4`, `f0=4`; its remaining graph and quantizer scopes should remain the reference. (`reference-code/HGQ2-examples/jsc150/model.py:178-210,277-295`.)

Replacing the post-fold quantizer therefore supports the precise statement: “We changed only the selected weight-quantization operators and their necessary cost representation, preserving the released architecture, BN moment path, bias quantization, activation-quantization rules and training schedule.” State β granularity in that sentence's accompanying definition. An unrestricted “only binary weights” is too vague to distinguish the alternatives in OPEN-1.

**The BN reasoning is correct, with an important qualification.** Write the latent kernel as \(W_{ic}\), with input index \(i\) and output channel \(c\). Define

\[
s_c=\frac{\gamma_c}{\sqrt{v_c+\epsilon}},\qquad
U_{ic}=s_cW_{ic},\qquad d_c=b_c-\mu_cs_c.
\]

Chang computes the kernel as \(kq(U)\) and the offset as \(bq(d)\). Here \(b\) denotes the layer's stored offset, and \(\epsilon\) is the backend epsilon used by this implementation. (`reference-code/HGQ2/src/hgq/layers/einsum_dense_batchnorm.py:158-173`.) The following expressions are derived alternatives, not descriptions of an implemented binary layer.

*Binarize then fold, with per-layer β:*

\[
K_{ic}=s_c\,\overline{|W|}\,
\operatorname{Sign}(W_{ic}-\overline W).
\]

The magnitude in channel \(c\) is \(|s_c|\overline{|W|}\). Thus the relative BN scale magnitudes survive; a negative scale flips signs and a zero scale zeros the channel. This is a sound binary factorization. But moving the quantization before the fold changes the operator order relative to Chang. Keeping his raw-kernel moments also does **not** turn this into ordinary BN computed on the binary projection. Computing moments from that projection would be a further change, as OPEN-1 C already recognizes. (`campaigns/2026-10-16-chang-bitlinear/OPEN-CHOICES.md:7-11`; released moment path: `reference-code/HGQ2/src/hgq/layers/einsum_dense_batchnorm.py:185-208`.)

*Fold then binarize, with per-layer β:*

\[
K_{ic}=\overline{|U|}\,
\operatorname{Sign}(U_{ic}-\overline U).
\]

All nondegenerate coefficients now have the same magnitude. The independent channel magnitudes are gone. However, it would be wrong to say “BN has no effect”: its scales still affect the global β, the centering threshold, sign decisions and shifted bias. With global centering and a nonzero global β, even a channel with \(s_c=0\) need not remain zero after this map. That is an algebraic consequence worth testing, not an observed failure.

*Fold then binarize, with per-channel β:*

\[
\alpha=\overline U,\qquad
\beta_c=\frac{1}{n_{\rm in}}\sum_i|U_{ic}|
=|s_c|\frac{1}{n_{\rm in}}\sum_i|W_{ic}|,
\qquad K_{ic}=\beta_c\operatorname{Sign}(U_{ic}-\alpha).
\]

This retains channel-dependent magnitude, including a zero-scale channel, while replacing the quantizer at the same place as Chang. The equality for β is algebraic; it does not establish equal outputs or training dynamics to binarize-then-fold. In particular, global centering couples the channels and the two sign patterns generally differ.

My recommendation changes **β granularity only**, retaining global α over the tensor presented to the binary quantizer. Per-channel α would be another experiment; it must not enter silently. β remains a derived statistic, not an additional freely trained parameter. Use the stipulated full-map identity STE at the chosen quantizer input, with no derivative through α or β; after folding, that means the identity surrogate is with respect to \(U\), while the existing fold and moment paths still carry gradients. This extends the campaign's explicit surrogate definition to the post-fold site. (`campaigns/2026-10-16-chang-bitlinear/STUDY.md:32-34`.)

The scientific tradeoffs are therefore:

| OPEN-1 choice | What it tests and may support | What it cannot establish |
| --- | --- | --- |
| Fold first, per-layer β | The stipulated tensorwise BitLinear weight map applied at Chang's quantizer location | That channel-scale freedom was preserved; poor performance cannot isolate binary signs from the shared-magnitude restriction |
| Fold first, per-channel β at BN sites — recommended | Binary weights as close as possible to Chang's fold order while retaining channel magnitudes | Literal per-layer BitLinear fidelity; any success must include the channel-scale qualification |
| Binary first, per-layer β, then fold | A binary latent kernel with an explicit surviving BN channel scale | An unchanged fold/quantize order, or standard BN on binary projections when raw-kernel moments are retained |
| Binary first, per-channel β, then fold | Channel-scaled latent binarization followed by BN scaling | That the added scale granularity is irrelevant; it changes the stipulated raw-kernel map as well as the order |

No option reproduces all of “BitLinear as published”: this campaign deliberately keeps Chang's activation quantization and BN, whereas the local BitNet note describes its activation quantization and LayerNorm choices. (`literature/bitnet-1bit-ternary/2310.11453_bitnet_scaling_1bit_transformers.md:14-20`; campaign restriction: `campaigns/2026-10-16-chang-bitlinear/STUDY.md:30-32`.) “BitNet-inspired weight binarization” is the accurate description. Per-layer β is closest to the stipulated weight rule; per-channel β answers the different question of preserving Chang's channel flexibility. If both questions matter, predeclare a matched per-layer/per-channel ablation. Do not select the preferred interpretation after seeing held-out accuracy.

**EBOPs and LUTs should answer different questions.** EBOPs is an arithmetic proxy useful for the training pressure and an accuracy-versus-cost comparison. LUTs measure a property of a particular synthesized circuit. A binary sign pattern need not produce a proportional LUT saving: scale arithmetic, accumulation, dynamic attention products, nonlinear functions and the compiler's sharing of constant arithmetic all matter. In particular, Chang's attention cost includes QK and AV activation products, which binary projection weights do not remove. (`reference-code/HGQ2/src/hgq/layers/attn/mha.py:406-432`.)

There is a crucial correction to “Chang's counting.” The dense routine contracts input `bits_` with kernel `bits_`, adds a bias term, and applies its parallelization factor; it does **not** infer the minimum width of each exported coefficient. (`reference-code/HGQ2/src/hgq/layers/core/einsum_dense.py:85-93`.) Furthermore, the quantizer's EBOP `bits` and storage `fbits` differ: under `SAT_SYM`, the former omits the sign contribution that appears in the latter. (`reference-code/HGQ2/src/hgq/quantizer/quantizer.py:71-92`; `reference-code/HGQ2/src/hgq/quantizer/internal/fixed_point_quantizer.py:92-105`.) One binary sign bit is therefore not automatically the same quantity as one unit of the native cost convention.

**OPEN-2 (a): full-coefficient accounting is the conservative comparison, but an extension must be named.** Charge the finite-precision constants actually used by the inference model, including β and any surviving fold scale. Do not add another scale charge when it is already represented inside those constants. Exact finite binary floating-point values can be represented as dyadic fixed-point values; that mathematical fact does not guarantee economical formats or exact equality between different orders of arithmetic.

The draft proposes effective coefficient widths for binary sites while leaving Chang's native learned-width costs unchanged. That preserves his training objective, but using the same multiplication formula is insufficient to prove identical accounting: the two sides obtain their widths differently. (`campaigns/2026-10-16-chang-bitlinear/STUDY.md:38-40`.) My recommendation is to preserve the released native total and also evaluate **both frozen inference graphs** with one declared coefficient-width policy, covering zero values, sign convention, fractional precision, bias cost and parallelization. If the common-policy total differs from native HGQ2, publish both and call the common one *effective-coefficient EBOPs*. This supports a front claim under that named convention, not an unchanged-native-EBOP claim. The lab explicitly requires custom totals to be distinguished from native totals. (`docs/conventions/quantization-and-cost.md:21-25`.)

This accounting can conceal a real implementation advantage by treating a shared scale as if it were repeated in every coefficient. That makes it conservative about factorization, not automatically a faithful predictor of LUTs. Conversely, keeping arbitrary β at its exact training precision can give the binary model finely resolved scales; paying their full cost makes the tradeoff visible rather than proving it is a strict low-precision subset of Chang.

**OPEN-2 (b): sign-plus-scale accounting can be scientifically cleaner for the circuit, provided the circuit exists.** For an inference kernel \(K_{ic}=a_cS_{ic}\), real arithmetic gives

\[
y_{tc}=a_c\sum_i S_{ic}x_{ti}+d_c.
\]

Here \(t\) labels each use of the layer, such as a token. Charge the sign dot products and the scale operation on the **accumulator's precision**, plus bias and all other graph contributions. “One scale per output channel” means one scale constant per channel and a scaling operation for every produced output value; it does not mean one operation per entire jet when the layer is reused across tokens. Shared hardware must be represented through an explicit parallelization/throughput convention applied to both models.

The displayed equation is exact over real numbers; moving multiplication across accumulation can change fixed-point rounding, saturation and overflow. The converter must preserve the specified arithmetic, with full-model fidelity checked. The inspected hls4ml handler reads `qkernel` and `qbias` and registers the existing dense classes; that alone certifies neither a custom binary class nor a sign-plus-scale implementation. (`reference-code/hls4ml/hls4ml/converters/keras_v3/hgq2/_base.py:109-126`.)

Give Chang the same admissible exact factorizations and compiler optimizations. He need not obtain the same savings: a representation genuinely available only to binary weights is a legitimate advantage. The unfair advantage would be applying different rules or granting binary an unimplemented optimization. A general operation-graph cost must specify its accumulation, scale, bias and rounding rules symmetrically; do not add accumulator costs only to one side. Until conversion and accounting are validated, label this an analytical cost estimate, not a measured hardware result.

**OPEN-2 (c): power-of-two β is a separate hardware-oriented model.** Rounding \(\overline{|W|}\) to a power of two changes forward values. It can support “a binary model with power-of-two scales improves this measured front,” but not “the stipulated unrounded BitLinear rule beats Chang.” A constant shift may need no multiplier, yet can change width, truncation and subsequent arithmetic cost. At pre-fold BN sites, a power-of-two β multiplied by an arbitrary BN scale is still generally arbitrary. Thus even the free-shift argument depends on the actual exported graph.

The paused campaign used power-of-two kernels only at specified sites and handled other scales differently; its STUDY is not evidence that every scale here is free. (`campaigns/2026-10-12-mha-bitnet/STUDY.md:1052-1057,1113-1124`.) Its omitted-β BN treatment and BK-A8 arm also do not prove equivalence for this fused construction; BK-A8 was specified as training-only. (`campaigns/2026-10-12-mha-bitnet/STUDY.md:1305`.) Algebraically, for a positive scale \(a\),

\[
\frac{az-a\mu}{\sqrt{a^2v+\epsilon}}
=\frac{z-\mu}{\sqrt{v+\epsilon/a^2}}.
\]

Scale cancellation is exact in the idealized zero-epsilon case with consistently scaled moments. Finite epsilon, moving statistics, quantization and changing scales complicate it. Here Chang obtains moments from the latent projection before kernel quantization; dropping β cannot be justified merely by saying “BN absorbs it.” (`reference-code/HGQ2/src/hgq/layers/einsum_dense_batchnorm.py:165-172,185-208`.)

**Reporting both costs is cleaner than choosing whichever gives the better plot.** Freeze one primary cost and its selection rule before results. For every selected checkpoint, report native cost where valid, effective-coefficient cost, factorized cost with validation status, held-out top-1, and synthesis resources where available. Show separate, clearly labelled axes. Report disagreement between the two cost rankings as a finding about the proxy. A win under only one convention is an accounting-dependent win.

There is an extra complication: cost affects learning. HGQ adds `ebops * beta` to the loss, and Chang schedules that coefficient during training. (`reference-code/HGQ2/src/hgq/layers/core/base.py:94-104`; `reference-code/HGQ2-examples/jsc150/run_train.py:90-104`.) Keep that training coefficient distinct from the weight scale β. Re-evaluating an already trained checkpoint under a second cost is a measurement change; training with the second cost is another experimental condition. Preserve Chang's released training as the reference, declare the binary extension and its surrogate gradient, and do not claim that the same numerical ramp makes two different objectives identical.

Also freeze the state used for cost. Batch-statistics coefficients during training and moving-statistics coefficients at inference are different objects. Native `FreeEBOPs` reads stored counters, while the base layer updates those counters during training or tracing. (`reference-code/HGQ2/src/hgq/utils/sugar/ebops.py:7-15`; `reference-code/HGQ2/src/hgq/layers/core/base.py:94-104`.) Use a non-mutating frozen-inference cost for feasibility, checkpoint selection and export comparison; separately define training-penalty and controller measurements. Preserve the released callback front as a reference if this changes selection. The existing review identifies exactly this unresolved state issue. (`campaigns/2026-10-16-chang-bitlinear/review/critical-v0.1.md:48-65`.)

**What would make “binary beats Chang's front” defensible.** The comparison must use Chang's own released code on the same data, with matching input definition, splits, seeds and training opportunity. Establish that the no-binary reconstruction reproduces the reference before attributing a difference to binarization. Predeclare the cost domain, validation selection, uncertainty analysis and handling of missing points. Keep the held-out sample out of architecture, scale-granularity, accounting and checkpoint selection. The current draft already proposes a validation-selected policy and distinguishes a local advantage from superiority across its tested grid; those distinctions should survive the OPEN-1/2 decision. (`campaigns/2026-10-16-chang-bitlinear/STUDY.md:58-68,103-118`.)

An isolated favorable point supports a local comparison. A grid-wide result supports that tested grid and declared accounting, not every budget or every binary model. Seed uncertainty must accompany the gap; a single favorable training run is insufficient. Unresolved intervals do not demonstrate equivalence. The review warns that resolving power for the proposed whole-front test has not been established. (`campaigns/2026-10-16-chang-bitlinear/review/critical-v0.1.md:67-84`.)

For a hardware claim, synthesize the deciding binary and Chang checkpoints with the same part, tool versions, clock target, throughput/II requirements, reuse policy, arithmetic binding and hls4ml/da4ml route. Compare accuracy and actual LUT together; also report DSP, FF, BRAM, latency and timing so a LUT saving cannot hide resource substitution. Keep C-synthesis estimates separate from later implementation stages. These distinctions are already required by the hardware protocol. (`campaigns/2026-10-16-chang-bitlinear/STUDY.md:122-128`; `docs/conventions/fpga-synthesis.md:21-29,34-41`.) Synthesis of a few points supports those points, not a complete LUT front. An EBOP win without a LUT win remains a proxy-cost result; a LUT win without an EBOP win remains a hardware result.

**The main risks, and what would change my mind.** Per-channel β may preserve useful magnitude freedom yet fail to cure a training failure. Per-layer β may work just as well, in which case its simpler agreement with the stipulated tensorwise map is attractive. A matched validation-only ablation showing no meaningful per-channel benefit, with adequate resolving power and comparable export behavior, would make me prefer per-layer β. That decision must precede held-out evaluation.

A sign-plus-scale converter with verified arithmetic, complete symmetric cost accounting and reproducible LUT savings would make option (b) the stronger hardware interpretation. It would not retroactively turn its axis into native HGQ2 EBOPs. If exact coefficient export is prohibitively costly or factorization cannot preserve the selected model, report that limitation; rounding β would require a separately named model. There is also an existing full-graph export concern: the inspected unary-LUT handler rejects heterogeneous tables, so preserving Chang's quantizers through conversion needs evidence independently of binary scaling. (`reference-code/hls4ml/hls4ml/converters/keras_v3/hgq2/unary_lut.py:18-32`; review: `campaigns/2026-10-16-chang-bitlinear/review/critical-v0.1.md:86-101`.)

Finally, the historical 350k-collapse entry lists competing hypotheses and a pending readout. It does not establish that scale granularity, the controller, attention starvation or binary weights caused the collapse. (`.claude/memory/experiment-log.md:1-7`.) Treat it as motivation to retain accuracy trajectories, channel scales, activation widths and failed runs. It is not evidence for choosing the favorable accounting or for declaring either OPEN-1 option scientifically settled.
