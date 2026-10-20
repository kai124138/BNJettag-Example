# PREFLIGHT build plan: chang-bitlinear-20261016

Plan only. No code has been written, no gate has run, and nothing here authorizes compute. Written
2026-10-17 by the ml-engineer role (opus). Every hour figure below is an estimate in agent-hours,
not a measurement. No number from the Mac or the home PC is quotable (`CLAUDE.md`).

```
Goal:     build and gate the fork, configs, export route and launcher so that S1 can launch
Plan:     batches B0-B20 (section 4), each a commit followed by its gates (section 3)
Touches:  new fork code/; preflight/gates/; preflight/da4ml-clean/; PREFLIGHT.md; PC scratch and venv;
          mulder scratch (after Kai item 6); lab-root run_handoff.py and harness.py (after Kai item 1);
          NRP smoke and benchmark only (after the Kai items of section 5)
Cost:     about 110 agent-hours before fix cycles; at most three agents at once; mulder wall time
          of days; NRP smoke and benchmark inside the 50 GPU-h reserve
Stops:    preflight_final.sh prints PREFLIGHT_ALL_PASS and the critical reviewer passes; early on a
          gate that needs a design change, a refused Kai item, a measured rate over the cap, or no mulder
```

## 0. Inputs, start condition, blinding

**Inputs.**
- `STUDY.md` v0.5, sha256 `8c1f2401…f812b`, passed by `review/arbiter-v0.5.md` with conditions P1-P6.
  Read: §0-§15, §17-§19.
- The v0.6 C-item pass was being written while this plan was drafted. Last sha read: `d20113dc…40aa`
  (mtime 17:06:53). Diffed against `review/snapshots/v0.5/STUDY.md` for §2.2, Addendum A, §6, §8.6,
  §8.8, §9, §11, §13, §14 and §19.2. What changed there:
  - Addendum A and §11: clone safeguards, exclusivity, per-line assertions, a void-and-restart remedy
    with "fail the line" as the named alternative, and Kai answering item 5 before the rerun;
  - §13: a rerun-environment equivalence row, and structure-record and ablation additions (the
    degenerate-Q records, float64 means, overflow counts);
  - §6, §8.6 and §19.2 items 5, 11 and 12: labels for C with (c), plug-in power at 6.28 s_d, the
    C3r nesting note;
  - §9: a citation.

  None of these conflicts with this plan. Gates G39, G41, G42, G55, G71 and G73 implement them as
  the arbiter file (§6 items 1, 4, 6, 8, 12, 13 and 14) states them. The committed v0.6 text must
  be rechecked against this plan.
- `preflight/env/README.md` and the listing and hashes of `preflight/env/scripts/`.
- The paused campaign's `code/` (main at `6a14587`), `tools/README.md`, `PREFLIGHT.md`, `SETUP.md`
  and `BRIEF.md`; `tools/run_handoff.py`; `tools/harness.py`; `docs/infrastructure/run-handoff.md`.

**Start condition.** B0 starts only after the orchestrator has committed STUDY v0.6 and closed
P1, P3 and P4, and has recorded the P5 reference (the post-fix `protocol-draft.json` sha) while
`lab_freeze_protocol` is unavailable (`03-phases.md:11-12`).

**Blinding of this author.**
- Not opened: `preflight/da4ml-test/results/`, `reports/`, `logs/`, `perpos-oneblock/`,
  `preflight/da4ml-test/README.md`, either evidence-hash file, `preflight/env/results/` and
  `preflight/env/logs/`.
- Read in `preflight/da4ml-test/scripts/`: `env.sh`, `run_case.sh` and `synth_one.sh`, plus the
  directory listing.
- `.claude/memory/decisions.md`: lines 1-58 were read in full. A listing that showed only the
  first 30 characters of lines 26-122, made to find the cutoff, also showed the opening words of
  lines 59-72 and 88-121. Those words include a fragment of the da4ml outcome summary (lines
  64-65). This author is therefore not fully blind to that summary. Nothing in this plan depends
  on it: H1 and H2 are built and gated identically, and the frozen block decides between them. The
  orchestrator should record this under P3. This file does not repeat the fragment.
- `preflight/env/README.md` contains initialization readings. They are pre-PASS engineering
  readings, and B7 reproduces them under the fork's sha (P3).

**Jev.** The `jev-lab` MCP server failed to connect in this session (ENOENT on its Python
runtime). `lab_check_protocol` and `jev_check_methods` were not run. Gate G59 uses the same
deterministic function in memory (`tools/jev/methods.py` `inspect_protocol(path, baseline_path)`
under `python3 -I -B`), as the STUDY author did. `jev_check_methods` waits until the server
connects (G86). Its disposition will be quoted as advice, never as a verdict.

## 1. Repository layout

**Where.** A new git repository at `campaigns/2026-10-16-chang-bitlinear/code/`, branch `main`.
- The lab root is not git on the Mac. `campaigns/*/code` forks are git repositories.
- As in the paused campaign (`SETUP.md` §1), development and commits happen on the Mac fork.
  Mutagen ignores `.git`, so the PC receives the history as a `git bundle`, applied with
  `init`, `fetch` and `reset`. `git rev-parse HEAD` on the PC is checked against the Mac before
  any PC gate runs.
- Parallel batches use worktrees under `~/lab/worktrees/cbl1016-<batch>`, made with
  `git -C <fork> worktree add`. The orchestrator merges them. EnterWorktree is never called with
  the shell inside `code/`.

**What it is based on.** It is not based on the mha-bitnet fork, whose lineage carries the
TensorFlow runner and the table-homogenizing exporter (`mhab_export.py:16-19,76-99`, which
conflicts with Q8).
- Base commit: Chang's seven released `jsc150` files, byte-identical to `reference-code/`:
  `data.py` `a02743b6…`, `model.py` `7fcb38b4…`, `run_train.py` `bfe36347…`, `run_test.py`
  `e4456710…`, `test_utils.py` `1c0c4d57…`, `tools/prepare_data.py` `37f37a56…` and
  `tools/prepare_data.sh` `82b20099…`. The commit also holds the vendored `StopIf`
  (`stop_if.py` `ac398006…`, byte-identical to the snapshot) and `code-base.txt`, which records
  every base file's hash against its source and the content hash of the HGQ2 snapshot.
- Second commit: the two shims and the setup order, ported from `preflight/env/scripts/`
  (`cbl_shims.py` `dad29ef0…`, `cbl_common.py` `4c5d9b01…`), with the source hashes recorded.
- Code from the paused fork is copied module by module, each copy its own commit, with its source
  path and its sha at `6a14587` recorded. Only these are copied, as patterns: `derived_view.py`
  (identity and integrity checks, not its N=64 view), `mhab_fit.py` and `mhab_resume_fit.py`
  (fit, resume, exact sidecars), `mhab_wandb.py`, `mhab_integrity.py` (exit codes, markers),
  `mhab_evidence.py` (epoch telemetry). From the campaign `tools/`: `prepare_pack.py`,
  `study_layout.py`, `bench_rates.py`, `capture_pods.py`, `memory_projection.py` and `fetch_runs.py`.

**Tree.**

```
code/
  chang/            the seven released files; never edited
  vendor/stop_if.py
  code-base.txt     provenance
  env/              training lock (from preflight/env/lock), export locks per da4ml version, image Dockerfile
  cbl/              lab package (modules M1-M19, section 2)
  gen/              config generator and derived protocol (M17)
  configs/          generated only; regenerate-and-diff gate
  da4ml_clean/      clean-rerun driver, table script, calibration generators (M19)
  nrp/              pack runner, Job generator, launcher adapter, image (M20)
  tools/            harness commands for BRIEF.md (M21)
  gates/            one script per gate id; run_all.py; preflight_final.sh (M22)
  tests/
```

Evidence stays outside the fork: `preflight/gates/<gate id>/<sha12>/` (command, environment hash,
output), `preflight/da4ml-clean/<da4ml version>/` (Addendum A step 7) and `PREFLIGHT.md`.

**Lab-root tools.** `tools/run_handoff.py`, `tools/harness.py` and `nrp_doctor.py` live in the lab
root, which is not git on the Mac. The PC copy reportedly is git (remote `kai124138/bnjettag-oct9`;
unverified from the Mac). A launcher change is either committed there or vendored into the fork as a
campaign adapter. B0 checks which is possible. "Each change is its own commit" holds only once one of
the two is in place.

## 2. Modules

| id | module | STUDY | what it does | reuse |
| --- | --- | --- | --- | --- |
| M1 | `cbl/env.py` | §7.3 | binds the two shims before Chang's import and refuses to shadow an existing name; sets float32 matmul after `import model` and logs it; sets the backend and memory fraction; the runner chooses the device (decision U2) | `preflight/env/scripts` |
| M2 | `cbl/refdiff.py` | §7.3 reference identity | diffs every cited module of the snapshot against installed 0.1.9, re-hashes both, lists behavioral differences | none |
| M3 | `cbl/data.py` | §4 | preparation manifest: glob order, file hashes, rows, jet identity as (file, row); train/validation reader equal to Chang's `get_data`; sealed held-out reader that refuses without a freeze manifest; 2-file smoke archive | identity patterns of `derived_view.py` |
| M4 | `cbl/rng.py` | §6 | seed flow steps 1-8; per-seed initial-tensor file, drawn once and loaded by every arm (decision U1); role map and unmatched list; pre-fit reseed (decision U5); recorder of mapped batch indices in `Dataset.__getitem__`; guard on the module-level `random` state | none |
| M5 | `cbl/model.py` | §2.1, §5 | Chang's graph rebuilt layer by layer inside his scopes, with a kernel choice per group (Q, or B in FB or BF); F000 is all Q; DS-HGQ, FP32, W8A8, GR and BE variants | none |
| M6 | `cbl/binary.py` | §2.2, §2.3 | q_g built in `__init__` inside the scopes; α, S with Sign(0) = −1, β; the exact surrogate; non-BN binary EinsumDense; FB and BF fused-BN layers with Chang's moment path; binary MHA projections (decision U3); `_compute_ebops` returns penalty (b); `qkernel` computable at the end of `build`; no kq at binary sites; serialization; GR derivative | serialization patterns only |
| M7 | `cbl/cost.py` | §3.1, §3.2, §3.5 | w(q); snap to the quantizer's f, asserting no change; H1; H2 with the u_c bound; NR; native terms from the layers' own `_compute_ebops` in float32, native order, uint32 truncation; penalty (b) for both the H1 and the H2 branch, with the headline as an input; independent penalty recompute; public and private aggregates | none |
| M8 | `cbl/frozen_eval.py` | §3.3, §8.1 | non-mutating evaluator on moving moments with no training- or tracing-mode call and a variable-hash guard; per-epoch record; callback after StopIf that writes only new keys; one cached value shared with the PID; checkpoint and export audit entry points | none |
| M9 | `cbl/front.py` | §8.1 | nondominated set under H with the ties; Chang's filter in H units; secondary fronts under the other count and NR; the two diagnostic checkpoints; exact-metric sidecars; step envelope with NA | sidecar pattern of `mhab_fit.py` |
| M10 | `cbl/structure.py` | §2.2, §8.1 | zero-variance test over the validation jets (inference mode) and its five routes; zero columns with routes; zero-width input fractions; per-head softmax zero-width fractions; whether Q_t is identical across t (arbiter §6 item 4) | none |
| M11 | `cbl/ablate.py` | §8.8 | Δ_attn, Δ_static (per-entry means accumulated in float64, injected after the softmax output quantizer and before A·V), Δ_unif (zero logits through the live `QSoftmax`), H_attn with all-zero and negative rows counted, zero-ablation context, overflow counts at the downstream WRAP quantizers (arbiter §6 item 12); a lab re-expression of the MHA inference forward with injection points (decision U4) | none |
| M12 | `cbl/pid.py` | §7.2 | `BetaPID` subclass: frozen-H sensor read in its own `on_epoch_end`, acted on at `on_epoch_begin(e+1)`; own log key; stock aggregate logged beside; λ₀ = 2e-8; released gains; persisted state | controller law and defaults |
| M13 | `cbl/runner.py` | §5, §6, §7.1 | `run_train.py`'s `__main__` recipe re-expressed, importing Chang's schedule, `get_model` and `get_data`. Declared departures: device, I/O wrapper, initial-tensor load, pre-fit reseed, appended frozen-evaluation callback. Runs every arm | patterns of `mhab_fit.py` |
| M14 | `cbl/resume.py` | §13 resume row | persists the module-level `random` state, the ParetoFront record, the StopIf counters, PID state, the optimizer, front state and the W&B run id; refuses on another code sha | `mhab_resume_fit.py` |
| M15 | `cbl/wandb_log.py` | §12 | project `bnjettag-chang-bitlinear`, group `chang-bitlinear-20261016`; run names; every §12 key; offline and disabled modes; never reads or prints the key; refuses any test or held-out key before the freeze | `mhab_wandb.py` |
| M16 | `cbl/select.py`, `stats.py`, `s2.py`, `confirm.py` | §6, §8.2-§8.10 | D_screen and D_conf; form rule; coverage-first BEST; g* and D_band; factorial and dominance; S2 outputs (thresholds, power, plug-in power at the observed s_d and at 6.28 s_d, labels); descriptive held-out tables; PID selection; collapse diagnostic; the confirmation test over D_C2 with the nested form, C3, C3r, C4 and C6; §8.10 pull; the §8.5 freeze manifest | none |
| M17 | `gen/` | §5, §10, §13 arms row | configs for the 66 screen runs, PID templates, S4 templates A/B/C and smoke and benchmark; the headline count, claim-domain option and S4 option as inputs; parameter counts; derived protocol for the drift check | generator patterns |
| M18 | `cbl/export/` | §9 | export-environment package: MHA handler-key alias; option 0 harness with per-step exception records; custom H-LUT handler only if option 0 fails under both versions; per-node DA with a coverage check; Chang's trace (train then validation, `no_growth=False`); table equality; block and full-model C-simulation; BitExact record; scan for default-precision fallbacks; pre- and post-trace H and metrics; part-capacity build; cross-backend reload | report parsers only |
| M19 | `da4ml_clean/` | §3.4 Addendum A | driver: archive by rename, recreated directories, `env.sh` swap only on a clone route, restored in an exit trap, exclusivity record, `hgq4` lines first, per-line assertions, void rule, sidecars; 44-row table script; calibration generators (uniform, zero-heavy); rerun-environment equivalence check (arbiter §6 item 6) | frozen scripts called unchanged, by hash |
| M20 | `nrp/` | §10, §11 | pack runner (K arms per pod, one process and one run directory each); integrity exit codes; telemetry; lint-clean Job generator; launcher adapter for route A or B (section 4, B17); JAX runtime image from the hashed lock | `run_pack.py`, `prepare_pack.py` patterns |
| M21 | `tools/` | §11, §13 harness row | BRIEF.md draft with authority and operations blocks; prepare, relaunch, evaluate, workspace and repair-check commands; rate reader; ConfigMap count check | paused campaign `tools/` |
| M22 | `gates/` | 03-phases | one script per gate id; `run_all.py`; `preflight_final.sh` prints `PREFLIGHT_ALL_PASS` only when every gate passes at one sha | none |

## 3. Gates

Where: **MAC** is the Mac CPU, for code with no Keras or JAX import (the hashed lock is
`x86_64-manylinux_2_28`, and the Mac venvs are TensorFlow without JAX). **PCC** and **PCG** are the
PC CPU and the RTX 4060 Ti in `~/lab/venvs/cbl1016-jax`. **MUL** is mulder. **NRP** is the cluster.

| id | gate | STUDY | where | module |
| --- | --- | --- | --- | --- |
| G01 | Chang's seven files and `stop_if.py` in the fork hash-equal to `reference-code/`; `git status` clean on the Mac and on the PC after the bundle | §13 reference | MAC, PCC | base |
| G02 | shims bind only absent names, before Chang's import; installed hgq2 tree equal to the lock record; shim hashes recorded | §7.3 | PCC | M1 |
| G03 | matmul precision `float32` after `import model` and after `fit`, logged per run; numerical check on the model's own einsum, not only `jnp.matmul` | §7.3, §13 image | PCC, PCG | M1 |
| G04 | cited-module diff, snapshot against installed 0.1.9, both re-hashed; each behavioral difference in a cited mechanism listed | §7.3 | PCC | M2 |
| G05 | image: GPU visible, precision after import, hgq2 and snapshot hashes, no secret in any layer, `pip freeze` equal to `cbl1016-jax` | §13 image | PCG (docker), NRP | M20 |
| G06 | preparation reproduced with Chang's script unchanged; glob order, file hashes, rows and archive sha recorded; one-hot assertion; 558,000 and 62,000 jets; every std finite and nonzero | §4 | PCC | M3 |
| G07 | train/validation reader equal to Chang's `get_data` bitwise (with a placeholder test path); the training process opens no held-out file (open-file audit) | §4 | PCC | M3 |
| G08 | held-out archive: 26 files, 260,000 jets, class counts and sha only; identities disjoint from training; the sealed reader refuses without a freeze manifest | §4, §8.5 | PCC | M3 |
| G09 | the PVC copy of the training archive hashes to the PC value | §4 | NRP | M3 |
| G10 | seed flow in the §6 order; initial tensors loaded by role from the seed's file; unmatched tensors are q_g's b and i only | §6 | PCC | M4 |
| G11 | per epoch, the hash of the mapped batch indices actually fetched is identical across all arms of a seed and equals its reconstruction from the reseeded stream; `_index_map` hash recorded | §6 step 7 | PCC, PCG | M4 |
| G12 | no lab callback and no frozen-evaluation step changes the module-level `random` state during training | §6 step 6 | PCC | M4, M8 |
| G13 | runner against Chang's `run_train.py` run by `runpy` on the same device (on PCC with the smoke-only device remap of the env README), with the runner's initial-tensor file drawn on that same device: initial weights, training and inference outputs, per-layer EBOPs, loss, gradients and BN moving statistics bitwise equal on identical batches; callback order as released plus the appended callback | §13 reference | PCC, PCG | M13 |
| G14 | F000 equals CH-8 in tensors, forward, backward, one optimizer step and native cost, on identical batches | §13 CH-8 row | PCC, PCG | M5 |
| G15 | nested MHA kernel mapping by role; single wrap of the HGQ call | §13 CH-8 row | PCC | M5 |
| G16 | the training-mode forward kernel holds exactly ±q_g(g) per layer or ±q_g(g_c) per column, or zero | §13 binary | PCC, PCG | M6 |
| G17 | trainable variables equal CH-8's, minus kq's b and i at binary sites, plus q_g's b and i | §13 binary | PCC, PCG | M6 |
| G18 | no kq built and no kq regularization term at binary sites | §2.2, §13 | PCC, PCG | M6 |
| G19 | training-mode autodiff gradients equal Chang's own layer with kq replaced by the map, moment and bias paths included: ∂L/∂W, ∂L/∂γ, ∂L/∂b, ∂L/∂x, ∂L/∂(q_g's b and i); never finite differences of Sign | §2.3, §13 | PCC, PCG | M6 |
| G20 | the penalty gradient reaches q_g's b | §13 | PCC, PCG | M6, M7 |
| G21 | `qkernel` computable at the end of `build` | §7.3, §13 | PCC, PCG | M6 |
| G22 | q_g's configuration equals `_qkvo_kq_conf` field by field (and the scope's weight configuration at non-MHA sites), and survives a serialized reload | §2.2, §13 | PCC, PCG | M6 |
| G23 | edge cases: Sign(0); β = 0; negative and zero s_c; pruned and rounded-to-zero q_g, each with its H1, H2, NR and penalty; a saturated column (BF's direct γ gradient 0, FB's not); FB columns single-magnitude | §13 | PCC, PCG | M6 |
| G24 | q_g's float32 output exact at b = 23; a forced b = 24 fails the check | §13 | PCC, PCG | M6 |
| G25 | w(·) and the H2 column rule on constructed values: w(1) = 1, w(0.375) = 2, and the §3.5 table reproduced from its formulas | §3.2, §3.5 | MAC | M7 |
| G26 | H1, H2 and NR per §3.2 in the native dtype, native terms from `_compute_ebops` in float32, native order, uint32 truncation | §3.2 | PCC | M7 |
| G27 | H1 ≤ NR at every site of every arm, on the CPU and the GPU path | §3.2 | PCC, PCG | M7 |
| G28 | the counts reduce exactly to native on F000 and CH-8 | §3.2 | PCC | M7 |
| G29 | zero- and random-input agreement for H and NR | §13, §18 | PCC | M7 |
| G30 | the per-step penalty equals (b) recomputed from the quantizer variables, never from the training kernel, under both the H1 and the H2 branch | §3.1(b), §13 | PCC | M7 |
| G31 | top-level public aggregate equals the flattened private total on CH-8 and every binary configuration | §7.2 | PCC | M7 |
| G32 | the snap to each quantizer's f changes nothing, on the CPU and the GPU path | §3.2 | PCC, PCG | M7 |
| G33 | a binary layer's `_compute_ebops` equals (b); NR comes only from the evaluator | §3.1(a′) | PCC | M6, M7 |
| G34 | the frozen evaluation leaves every variable hash unchanged, makes no training- or tracing-mode call, and equals the checkpoint audit exactly | §3.3 | PCC, PCG | M8 |
| G35 | the export audit equals the frozen evaluation of the same post-trace state; the pre/post-trace change recorded | §3.3 | MUL | M8, M18 |
| G36 | initialization readings of H1, H2 and NR per configuration, CPU and GPU, recorded as engineering readings under the fork sha | §3.5, §14; P3 | PCC, PCG | M7, M8 |
| G37 | front saver: nondominated under H, ties, Chang's filter in H units, diagnostic checkpoints outside the front, exact sidecars; step envelope gives NA, never zero | §8.1 | MAC, PCC | M9 |
| G38 | the zero-variance detector flags each named route on constructed cases (zeroed kernel, kernel rounded to zero, all-zero-width input quantizer, all-zero-width output quantizer) for a binary and an HGQ layer, and nothing on a non-degenerate case | §2.2, §13 | PCC, PCG | M10 |
| G39 | per-head softmax zero-width fractions; the Q-identical-across-t record | §8.1 | PCC | M10 |
| G40 | the MHA re-expression with injection points equals the layer's own inference forward bitwise | §8.8 | PCC | M11 |
| G41 | Δ_attn, Δ_static and Δ_unif exactly 0 when the ablated tensor is replaced by itself; Δ_unif and Δ_static exactly 0 on a constructed K ≡ 0 model; H_attn = log 8 on uniform rows | §8.8, §13 | PCC | M11 |
| G42 | a constructed overflowing case is counted at the downstream WRAP quantizers; rows with a negative entry counted | arbiter §6 item 12 | PCC | M11 |
| G43 | sensed(e) equals frozen H(e) exactly | §7.2 | PCC | M12 |
| G44 | Chang's functional model object is what the callbacks receive; stock aggregate > 0 after epoch 0 | §7.2 | PCC | M12 |
| G45 | λ₀ = 2e-8 > 0 and the sourced gains | §7.2 | PCC | M12 |
| G46 | interrupted and resumed integral identical | §7.2 | PCC | M12, M14 |
| G47 | the PID writes its own key; `logs['ebops']` is not overwritten; ParetoFront and StopIf read FreeEBOPs's count; callback order [FreeEBOPs, LR, PID, PBar, ParetoFront, StopIf, frozen evaluation] | §7.2 | PCC | M12 |
| G48 | uninterrupted and resumed runs identical (random state, ParetoFront record, StopIf counters, PID state, optimizer, front state) | §13 resume | PCC, PCG | M14 |
| G49 | metric reload agreement within 1e-7 with TF32 off, over the full validation set, for every generated config after one CPU epoch, and on the GPU path for the B13 runs | §13 further, 03-phases | PCC, PCG | M14, M17 |
| G50 | training, saved and exported checkpoint identity (hash chain) | §13 further | PCC, MUL | M14, M18 |
| G51 | validation-only tracing and selection: no held-out path outside the sealed reader (static audit) and a runtime guard | §13 further | MAC, PCC | M3, M16 |
| G52 | offline W&B records hold every §12 key with project and group; the key is never read or printed; a test or held-out key is refused before the freeze | §12 | MAC, PCC | M15 |
| G53 | domains, form rule, coverage-first BEST, NA and ties, g* and D_band, empty-domain INCOMPLETE, screen tables without verdict labels, factorial effects and dominance, PID selection and collapse windows, on constructed data | §8.2-§8.8 | MAC | M16 |
| G54 | the §6 thresholds and power table recomputed by the code (`default_rng(12345)`, 200,000 simulations) equal the STUDY to the printed precision | §6 | MAC | M16 |
| G55 | identical arms give zero differences and an interval around 0; the D_C2 test, the nested whole-D_conf form, the 7- and 6-contrast labels, C3, C3r, C4 and C6 on constructed cases; plug-in power at the observed s_d and at 6.28 s_d | §8.6, §8.7 | MAC | M16 |
| G56 | the §8.5 freeze manifest is complete; the held-out reader refuses without it | §8.5 | MAC, PCC | M16, M3 |
| G57 | the generator emits every arm identity (66 screen runs, PID and S4 templates, smoke and benchmark); regenerate-and-diff is empty; no config edited by hand | §5, 03-phases | MAC | M17 |
| G58 | every config builds on CPU with its parameter count recorded, and reloads after serialization | §13 further | PCC | M17, M5, M6 |
| G59 | the protocol derived from the configs shows no drift against the frozen protocol (in-memory `inspect_protocol` with `baseline_path`); dataset identity hash filled | 03-phases step 6 | MAC | M17 |
| G60 | shared initialization and callback order per arm | §13 arms row | PCC | M17, M13 |
| G61 | export environments built and locked, one per da4ml version, with versions and content hashes; `m3-env`'s `pip freeze` hash unchanged before and after | §9 | MUL | M18 |
| G62 | CH-8 builds at initialization in the export environment; Chang's trace runs unmodified | §9, §13 | MUL | M18 |
| G63 | option 0 steps 1-5: MHA alias installed, `allow_v2_fallback=False`, exactly two `DACombinational` nodes, dispatched tracer recorded, exception type per step; 0.6.0 only on a failure under 0.5.2 | §9 | MUL | M18 |
| G64 | affined-tanh table equality on every site's full input grid, scalar scale and bias reproduced | §9 | MUL | M18 |
| G65 | binary projections and the binary MHA convert with no `AttributeError`; the MHA subclass is registered | §9 | MUL | M18, M6 |
| G66 | per-node DA with coverage; fallback only on option 0's two nodes; the kif reshape fault retested | §9 | MUL | M18 |
| G67 | BitExact pass recorded as run; no type falls back to the model default precision | §9 | MUL | M18 |
| G68 | block, then full-model C-simulation on 4,096 validation jets, bit-exact, for CH-8 and F000, then an initialized binary configuration of each form | §9 | MUL | M18 |
| G69 | cross-backend reload: a checkpoint trained under JAX on the GPU (from B13) gives identical frozen H and identical argmax on the 4,096 jets in the export environment | §9, §13 | PCG then MUL | M18, M8 |
| G70 | part capacity: a CH-8-sized model completes OOC synth plus opt on xczu7ev; utilization above 100 % reported as is; capacity read from the report or the part property | §9 | MUL | M18 |
| G71 | rerun-environment equivalence check: one stock and one fact project generated fresh with `stage_b_build.py` in the export environment and in `m3-env`, no synthesis, generated weights and types diffed, under a separate scratch root, done before Addendum A step 3 | arbiter §6 item 6 | MUL | M19 |
| G72 | driver and table script dry-run against stand-in scripts in a temporary root: archive by rename, recreated directories, assertions, void rule, sidecars, 44 rows with failed stages marked | Addendum A | MAC | M19 |
| G73 | on mulder: executed scripts equal the frozen hashes before and after every line; `env.sh` per branch with exit-trap restore; exclusivity recorded; `hgq4` lines first; one attempt per rule line; evidence under `preflight/da4ml-clean/<version>/` | Addendum A | MUL | M19 |
| G74 | the rule applied to the run Kai's item-5 answer names; the outcome, table, sidecars, archive listing and script hashes recorded in PREFLIGHT and in the STUDY change log before any production run | §3.4 block, Addendum A step 9 | MUL, MAC | M19 |
| G75 | calibration ratios recorded: binary `stock`, `hgq4`, and the uniform and zero-heavy cases, whose generators are hashed before they run | §3.4 | MUL | M19 |
| G76 | BRIEF authority and operations blocks validate (`harness.py validate`); new identities | §13 harness | PCC | M21 |
| G77 | `nrp_doctor.py lint` clean on every manifest; output pasted into PREFLIGHT | 03-phases | PCC | M20 |
| G78 | `run_handoff.py validate` and an offline launch of the smoke handoff | §11 | PCC | M20 |
| G79 | launcher route. Route A: the pod checks HEAD and tree hash against the handoff record and refuses on a mismatch; no ConfigMap is created. Route B: both ConfigMaps carry the Job's ownerReference and are gone after the first harness pass after the Job ends | §11; [dec 10-17 ConfigMaps] | PCC, NRP | M20 |
| G80 | before every wave, the `kai-` ConfigMap count equals the number mounted by active or pending kai Jobs | §11 | NRP | M21 |
| G81 | image gate on the cluster's GPU product; data hash on the PVC (G09) | §13 image | NRP | M20 |
| G82 | 2-file, 3-epoch smoke for each arm class | 03-phases | NRP | M13, M20 |
| G83 | GPU training step plus kill and resume smoke; resumed equals uninterrupted | §13 harness | NRP | M14, M20 |
| G84 | packing benchmark: K per pod at rolling 3-hour utilization ≥ 40 %, device memory ≤ 90 %, a safe host-memory projection; measured all-inclusive class rates r_m; budget recomputed against the cap | §10 | NRP | M20, M21 |
| G85 | `preflight_final.sh` prints `PREFLIGHT_ALL_PASS` at the final sha; mulder gates rerun if any export-path file changed after they passed | 03-phases | PCC, MAC, MUL | M22 |
| G86 | `jev_check_methods` on the PREFLIGHT method text, advisory, once Jev connects | role | MAC | none |

## 4. Order of work

Each batch ends with one or more commits on the fork's `main`, one per module, and then runs its
gates at that commit. Gate outputs go to `preflight/gates/<gate>/<sha12>/`, and PREFLIGHT.md gets
one row per gate with sha, place and outcome. A later commit that touches a gated module reruns
that gate. Every gate reruns at the final sha (B18).

Lanes: **P**, the PC build lane, serial; **M**, Mac code with no backend; **U**, mulder; **L**,
launcher, harness and image.

| batch | lane | scope | gates | estimate (agent-h) | Kai first |
| --- | --- | --- | --- | ---: | --- |
| B0 | P | start condition checked; PC lab-root git and Mutagen `.git` exclusion verified; PC venv lock hash equals `preflight/env/lock`; fork created, base commit, `code-base.txt` | G01 | 1.5 | none |
| B1 | P | M1, M2 | G02-G04 | 3 | JAX 0.9.2 and the CUDA 12 plugin are used as locked; a change reruns the GPU gates |
| B2 | P | M3: preparation on the PC, manifests, readers, sealed held-out hash and counts, smoke archive | G06-G08 | 3 | none |
| B3a | M | M7 pure part: w, snap, u_c, column rule | G25 | 3 | none |
| B3b | M | M16 on synthetic sidecars | G53-G56 | 8 | best after item 3 (OD-15); every item-11 and item-12 option implemented |
| B4 | P | M4 | G10-G12 | 4 | none (decision U1 flagged) |
| B5 | P | M13 and M5 (all-Q path) | G13-G15 | 6 | none |
| B6 | P | M6 | G16-G24 | 11 | best after item 2 (OD-13); a rejection means about 10 h of rework |
| B7 | P | M7 model part and M8; initialization readings | G26-G34, G36 | 6 | none; both penalty branches gated |
| B8 | P | M9, M10, M11 | G37-G42 | 7 | none |
| B9 | P | M12 | G43-G47 | 3 | best after item 4 (OD-12); native units would mean 1-2 h of rework |
| B10 | P | M14 | G48-G51 | 5 | none |
| B11 | M | M15 | G52 | 3 | none |
| B12 | M then P | M17; configs with the headline unset fail closed at launch | G57-G60 | 4 | none for the build; regenerated after B15 |
| B13 | P | GPU consolidation: 2-epoch real-data runs on the 4060 Ti for CH-8, F000, F100, one FB and one BF configuration and a PID run; GPU rows of G16-G34; one GPU-trained checkpoint per form for G69 | GPU rows, G48 | 3 | none |
| B14 | U | rehearsal of option 0 steps 1-4 on the PC CPU in a lock matching the export environment (de-risking only, never the recorded run); then on mulder: M18, export environments, option 0, H-LUT, export gates, cross-backend reload, part capacity; fixes the da4ml version | G35, G50, G61-G70 | 14 | item 6; INLINE route before the route is locked |
| B15 | U | M19 dry-run on the Mac (G72, can start in lane M after B3a); G71; then the clean rerun, the rule and the record; calibration cases after `env.sh` is restored | G71-G75 | 5 | item 5 answered alone and recorded with its time before step 3 (P2); item 6 (clone on a 0.6.0 route); INLINE route |
| B16 | L | JAX runtime image from the hashed lock; PC docker gate; push | G05 | 3 | JAX and CUDA pins before the image of record; item 7 before the push |
| B17 | L | M20, M21: pack runner, Job generator, BRIEF.md draft, harness commands, ConfigMap check; launcher adapter (below); the harness crontab line on the PC, as in the paused campaign's `SETUP.md` §3 (Kai adds it if the permission check refuses) | G76-G80 | 9 | item 1 for the launcher part |
| B18 | P | final code sha; configs regenerated with the adopted headline; `preflight_final.sh` | G85 | 2 | none |
| B19 | L | NRP: training archive copied to `kai-data` by an authorized transfer path chosen by cluster-ops; per-seed initial-tensor draw Jobs on the production product (U1), with the seed-1 draw compared to the PC GPU draw; image and data gates; 2-file/3-epoch smoke; kill and resume smoke; packing benchmark; measured rates. A fix found here is a new commit, so B18 repeats (RULES §5) | G09, G79-G84 | 4 | items 1, 2, 3, 4, 12, 7 and 10 done; headline adopted (§14) |
| B20 | P | PREFLIGHT.md assembled; G86 if Jev is up; critical-reviewer solo review | G86 | 5 | none; item 8 if r_m exceeds 3.53 s |

Total about 113 agent-hours (112.5), an estimate before fix cycles. The serial PC lane is about 60 of them.
Mulder wall time is set by `run_case.sh` and `synth_one.sh` limits: up to 30 minutes of build and
two hours each of C-synthesis and Vivado per line, for 44 lines, plus the part-capacity build.

**Dependencies.**
- B6 needs B5. B7 needs B6 and B3a. B13 needs B10.
- B14's binary export gates need B6, and G69 needs B13.
- The mulder chain is the critical path to final configs: item 6, then B14 (option 0 fixes the da4ml
  version), then item 5, then B15 (headline fixed), then B12's regeneration, then B18. The binary
  penalty (b) uses the adopted rule, so no final binary config exists before B15.
- Without mulder there is no launch. Option 0, the H-LUT export and the export gates all need it,
  and §14 blocks launch on a failed export gate.

**Launcher adapter (B17), both routes, not chosen here.**
- *Route A: git at a pinned commit.* The fork must be reachable from the pod. Either (A1) push it to
  a private remote with a read-only deploy credential held as a cluster Secret, which is outward under
  RULES §3 and needs Kai; or (A2) a `git bundle` on `kai-data`, cloned at the pinned sha by the init
  container. Both need a new source mode in `run_handoff.py`, which now requires one immutable code
  ConfigMap mount and refuses anything else (`run_handoff.py:151,182`). The init container checks
  `git rev-parse HEAD` and the tree hash against the handoff record. The record goes to the run
  directory on the PVC, with its sha in a Job annotation. PREFLIGHT then records commit, tree hash
  and bundle sha in place of a ConfigMap name.
- *Route B: interim, ConfigMaps deleted at Job end.* `run_handoff.py` keeps its payload. After the Job
  is created, both of its ConfigMaps (source and record) get an ownerReference to the Job. The harness
  scheduled pass deletes them at its first pass after the Job reaches a terminal state, and garbage
  collection removes any left when the Job object goes. Each resubmission recreates and later deletes
  its own pair, and the count check (G80) runs before every wave.
- Both routes change lab-root tools. See section 1 for where those commits can live.
- The administrator also named containers as a source. Baking the fork into the image is a third
  route, which this plan does not develop.

## 5. Kai items, in the order the work reaches them

| item | decision | what waits | STUDY's recommended default |
| --- | --- | --- | --- |
| JAX/CUDA pins | jax and jaxlib 0.9.2, CUDA 12 plugin (env README, MEDIUM, flagged); NRP node driver support for CUDA 12.9 checked read-only by cluster-ops | B16; a change reruns the GPU gates of B1-B13 | as locked |
| 2 | confirm OD-13 (learned-width scale; BF quantized after the fold) | B6 should wait; launch precondition | as adopted |
| 3 | confirm OD-15 (screen without verdicts) | B3b should wait; launch precondition | as adopted |
| 4 | confirm OD-12 (PID at 350,000 H) | B9 should wait; launch precondition | as adopted |
| 6 | mulder sessions beyond the da4ml test | B14, B15 | grant before production |
| INLINE route | definition, and whether it replaces or joins the stock route | the end of B14 (route lock) and B15 (the rule applies to the exporting route) | none in the STUDY |
| 5 | which da4ml run the rule applies to, and void-and-restart or fail-the-line | B15 step 3 (P2) | clean rerun; void and restart |
| 1 | launcher route | B17's launcher part; every NRP action | git at a pinned commit |
| 12 | claim domain | before any launch; configs carry it | (b), the band |
| 7 | push the image | B16 push, then B19 | after the PC GPU gate |
| 10 | `harness.py approve` | after B18, before B19 | after items 2-5 and 12 and the gates |
| 8 | cap or scope | after B19 if r_m > 3.53 s per run-epoch | cap increase |
| 9 | the held control slot | before the S1 launch; all control configs are built now | hold the slot |
| 11 | S4 funding and size | after S2; templates built now | option C |

Items 2, 3, 4 and 12 are each asked alone. `harness.py approve` does not stand in for them (§14).
This plan treats the NRP smoke and benchmark as launches for §14, so every precondition there holds
before B19.

## 6. Parallel work and agents

- Up to three agents at once, each an ml-engineer or cluster-ops agent on opus: lane P throughout;
  lane M (B3a, B3b, B11, the generator part of B12, the M19 dry-run); lane U or L as their Kai items
  arrive. Sonnet does lookups only. No fan-out beyond three; if more is needed it is stated first
  (RULES §7).
- Lane M needs only interface notes from lane P (sidecar schema, config schema), fixed in B0.
- Option 0's rehearsal on the PC (B14, first part) can start once B1 has passed.

## 7. Risks and the gates that catch them

| risk | caught by |
| --- | --- |
| the binary forward is not exactly binary (an STE sum leaks into a consumer) | G16, G24, G33 |
| binary-layer gradients differ from Chang's layer, including the moment and bias paths | G19, G20, G23 |
| q_g configured outside Chang's scopes, or wrongly under the MHA's late build | G22, G17 |
| float32/float64 or truncation-order drift in the counts | G28, G26 |
| an inexact GPU pow puts values off the grid, so w(·) jumps to about 24 bits | G32, G27 on the GPU path |
| the frozen evaluation moves WRAP integer bits or draws from `random` | G34, G12 |
| arms of a seed fetch different batches (draw-count differences, threaded fetching) | G11 |
| CPU and GPU initial weights differ (env README finding 2) | G10 via the shared initial-tensor file |
| our runner departs from Chang's release beyond the declared departures | G13, G14 |
| hidden held-out access | G07, G08, G51, G56 |
| the affined tanh tables cannot be exported exactly | G63, G64; if option 0 and a custom handler both fail, the route goes back to Kai (§9) |
| binary projections or the MHA subclass do not convert | G65 |
| CH-8 does not fit xczu7ev | G70, then the §9 capacity order |
| an operator or environment fault decides the headline in the rerun | G71-G73 and the void rule |
| the JAX GPU stack fails on the cluster's driver or product | G05, G81 |
| throughput above the break-even rate | G84, then item 8 |
| ConfigMaps pile up again | G79, G80 |
| resume is not exact | G46, G48, G83 |
| PID sensor and stock aggregate disagree or double-count | G43, G31 |
| the degeneracy detector misses a route | G38 |
| selection or statistics code misapplies a rule, or is circular | G53-G55, G51 |
| configs drift from the frozen protocol | G59, G57 |
| code changes after a gate passed | G85 at the final sha |
| Mac and PC forks diverge | G01; HEAD checked before every PC gate run |

## 8. Owner decisions to flag

```
U1 DECISION: CH-8's initial tensors for seed s are drawn once on the production GPU backend, in the released order (datasets, then get_model), saved with their hash, and loaded by every arm (CH-8 included); CPU and PC gates use files drawn the same way on their own device
   CONSEQUENCE: the draws of record can only run on NRP, in B19, after the final sha. Configs therefore name the initial-tensor manifest by path, not by hash. The manifest sits on kai-data beside the data, its sha goes into PREFLIGHT and the stage handoff, and the runner refuses a mismatch. B19 records whether the cluster draw for seed 1 equals the PC GPU draw
   ALTERNATIVES: draw once on the CPU (a declared difference from the release of at most about 3e-8 in 11 kernels, env README finding 2); draw in every pod (arms of one seed in different pods could then start differently)
   CONFIDENCE: MEDIUM   FLAG FOR HUMAN: YES
U2 DECISION: the runner constructs the Datasets and passes 'gpu:0' on the GPU, exactly as run_train.py:73-74, and 'cpu:0' in CPU gates; no shim touches Chang's files
   ALTERNATIVES: a third, CPU-only shim declared in §7.3
   CONFIDENCE: MEDIUM   FLAG FOR HUMAN: YES
U3 DECISION: binary attention is a QMultiHeadAttention subclass whose build() creates binary projections, configured from the captured _qkvo_kq_conf; the subclass is registered with hls4ml beside the 0.1.9 alias
   ALTERNATIVES: replace _query_dense and the others after the stock build (fragile under 0.1.9's build wrapper, base.py:118-140)
   CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO
U4 DECISION: Δ_static and Δ_unif use a lab re-expression of the MHA inference forward with injection points, gated bitwise against the layer (G40)
   ALTERNATIVES: patch the layer's call at run time
   CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO
U5 DECISION: the pre-fit reseed is random.seed(int from the first 8 bytes of sha256("cbl1016-fit\0" + str(s))), the same for every arm of seed s
   ALTERNATIVES: any fixed derivation; §6 step 5 says only "derived from s"
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO (recorded; a declared CH-8 departure already)
U6 DECISION: W8A8 "eight total bits, trained steps" uses KBI with b fixed so that k + b = 8 and i trainable, for weights and datalanes
   ALTERNATIVES: KIF datalanes with i + f constrained (not a native HGQ2 mode); KIF with b0=8 only (which §5 says does not define it)
   CONFIDENCE: LOW   FLAG FOR HUMAN: YES (it changes Chang's datalane quantizer type in that control; W8A8 trains only if the rates admit it)
U7 DECISION: lab-root tool changes are committed in the PC lab-root repository if B0 confirms it is git; otherwise the launcher change is vendored into the fork
   ALTERNATIVES: either one by policy
   CONFIDENCE: MEDIUM   FLAG FOR HUMAN: YES
```

## 9. What in the STUDY is ambiguous or not implementable as written

1. **INLINE route.** It appears in nothing this author may read: STUDY v0.5, the v0.6 text so far,
   the reviews or the arbiter file. It may relate to the blinded per-position test. It needs a
   definition as a STUDY amendment or an unblinded orchestrator note. Choosing a firmware route after
   a results-aware test is a results-informed choice (06-review §6.3.3), which makes it Kai's.
2. **Route A needs either a push or a PVC bundle.** STUDY §19.2 item 1 says "code from git at a
   pinned commit". A pod can reach git only through a remote, which is outward, or through a bundle
   on `kai-data`. `run_handoff.py` supports neither today.
3. **Lab-root tools are not under git on the Mac** (section 1, U7).
4. **Mulder gates the headline and the launch** (section 4, dependencies). §3.4's fail-closed
   provision covers the headline (H1 without Vivado), but the export gates still block launch.
5. **Initial-tensor backend.** §6 steps 3-4 do not say on which backend CH-8's tensors are drawn,
   and the env README shows that CPU and GPU draws differ. Under U1 the draws of record happen on
   NRP after the final sha, so configs can name them only by path (U1, consequence).
6. **CPU device.** §7.3 declares two shims, but `run_train.py:73-74` requires a GPU. CPU gates need
   a declared mechanism (U2).
7. **Pre-fit reseed.** §6 step 5 says "derived from s" without the formula (U5).
8. **MHA projections are built in `QMultiHeadAttention.build`** (`mha.py:101`, projections at `:152-213`), not in the
   MHA's `__init__`. §2.2's "in the layer's `__init__`, never later" holds for each projection
   layer's own `__init__`. Their configuration comes from `_qkvo_kq_conf`, captured in the MHA's
   `__init__` (`mha.py:64`). G22 checks the result (U3).
9. **W8A8 with trained steps on KIF datalanes** has no direct HGQ2 form (U6).
10. **Smoke definitions.** 03-phases asks for a 2-file, 3-epoch smoke, and §13 asks for a GPU
    training step plus a kill and resume smoke. Both are planned (G82, G83). The 2-file smoke uses
    a separate archive made by Chang's script, so it changes the split and the statistics. It is an
    engineering check only.
11. **When `harness.py approve` comes.** §19.2 item 10 places it after "the PREFLIGHT gates", but §11
    requires it before any NRP smoke or benchmark, which are PREFLIGHT gates too. This plan reads
    "gates" as the Mac, PC and mulder gates.
12. **Jet identifiers.** Chang's prepared archive holds no identifiers. Identity is recorded as
    (file in glob order, row) (M3).
13. **§13's data row** cites the paused fork's `derived_view.py`, whose N=64 derived view does not
    apply here. Only its identity checks are reused.
