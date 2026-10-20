# Interface notes for lane M (DRAFT, batch B0)

**Status: draft.** Written in batch B0 so that lane M (B3a, B3b, B11, the generator part of B12)
can start without the PC lane (PREFLIGHT-PLAN §6). It fixes key names and value types for what
the frozen evaluator (M8), the front (M9) and selection (M16) exchange, and the fields the config
generator (M17) writes. It decides no design question. Where the STUDY leaves a choice open, the
field says "open" and names the batch or Kai item that settles it. Later batches may add fields;
renaming or retyping a field that a module already uses is its own commit and reruns that
module's gates.

**[B2] Extended in batch B2 (`main`)** with section 9, the preparation manifest, the readers and
the sealed held-out reader of M3, and with the marked notes in sections 5 and 6.

**[B4] Extended in batch B4 (`main`)** with section 10, the seed flow, roles, initial-tensor file,
batch-order recorder and stream guard of M4 (`cbl/rng.py`), and with the marked notes in sections
1, 5 and 6. The field `init_tensors`, open until now and used by no module, is given its keys.

**[B3a] Extended in batch B3a (branch `b3a`)** with section 7, the per-site input record of the
counts that `cbl/cost_pure.py` reads, and with the marked notes in sections 0 and 6. Nothing
earlier is renamed or retyped.

**[B5] Extended in batch B5 (`main`)** with section 13, the runner (M13, `cbl/runner.py`) and the lab
builder's all-Q path (M5, `cbl/model.py`), and with the marked notes in sections 5, 9 and 11. Nothing
earlier is renamed or retyped; section 13 fixes the key names of the config's `ramp` field, which
section 5 described by value only.

**[B6] Extended in batch B6 (`main`)** with section 15, the binary layers of M6 (`cbl/binary.py`), the
binary arms' roles and config rules (M5, M13), and what B7 and B8 read from a binary site. Numbered 15
because branch b12 adds its own section 14 before it merges. Nothing earlier is renamed or retyped.

**[B8] Extended in batch B8 (branch `b8`)** with section 17, the front saver (M9 `cbl/front.py`), the structure
record (M10 `cbl/structure.py`) and the ablations with the MHA re-expression (M11 `cbl/ablate.py`): the record
M9 reads, the files it writes, the sidecar, structure-record and attention-diagnostics fields, the run-summary
fields beyond section 3, and the runner wiring. Numbered 17 because batch B7 may add a section 16 on `main`
first. It closes two section 6 items (container format; where the pruning-channel record lives). Nothing
earlier is renamed or retyped.

**[B3b] Extended in batch B3b (branch `b3b`)** with section 10, renumbered 11 at the merge (numbered 10 because batch B2 adds
its section 9 on `main`): what M16 (`cbl/select.py`,
`cbl/stats.py`, `cbl/s2.py`, `cbl/confirm.py`) reads and writes, the two fields it adds to the
config's `claim_domain`, the held-out input record, the section 8.5 freeze manifest, and where
the unadopted STUDY items 13-16 would enter. Marked notes in sections 3, 5 and 6. Nothing
earlier is renamed or retyped.

**[B12] Extended in batch B12 (branch `b12`)** with section 14 (numbered 13 on the branch), the config generator M17
(`gen/`), its input record, the fields it adds to the section 5 schema, the `configs/` tree and the
derived protocol, and with the marked notes in section 5. Nothing earlier is renamed or retyped.

**[B9] Extended in batch B9 (`main`)** with section 18, the lab PID (M12 `cbl/pid.py`), its config field, log keys,
record block and persisted state, the runner's PID schedule and admission, and the front saver's wiring into the runner
(section 17's "Runner wiring", as built); marked notes in sections 1, 5, 13 and 17. Nothing earlier is renamed or
retyped.

**[B10] Extended in batch B10 (`main`)** with section 19, M14's resume checkpoints and resume, the runner's launch check, the
held-out guard, M15's strict level and the parameter-count input; marked notes in sections 13 and 18. Nothing earlier is
renamed or retyped.

**[B7] Extended in batch B7 (`main`)** with section 16, M7's model part (`cbl/cost.py`) and M8 (`cbl/frozen_eval.py`):
the per-epoch record as built, the cached evaluation the PID reads, the audits, and the marked notes in sections 0, 1,
6, 7 and 13. `train_count` is retyped from float64 to int64 (FreeEBOPs logs a Python int; section 1, owner decision
B7-6); nothing else is renamed or retyped.

Every field carries the STUDY section it comes from. STUDY means `review/snapshots/v0.8/STUDY.md`
(sha256 `290022fa41a726b8ae66ff302f7910a69c4c148ec27b15c2448cab83fb898ca8`); "plan" means
`PREFLIGHT-PLAN.md` (sha256 `2f1be9f3…30ff8`), section 2 module rows.

## 0. Conventions

- **Counts.** `H1`, `H2` and `NR` are the frozen counts of §3.1-§3.2; `train_count` is the logged
  training count of §3.3 (native for CH-8 and F000, the penalty (b) for binary arms, §3.1(b)).
  The headline H is `H1` or `H2`, chosen by the config field `headline` (section 5); a reader
  never assumes which. NR equals native for CH-8 and F000 (§3.1(a′)).
- **Cost values.** Per layer, the value is what the §3.2 rule gives with native contributions
  truncated to uint32 per layer as the native counter is (§3.2 "Parallelization, dtype and
  rounding"); totals are the integer sum of per-layer values, stored as int64. Open: the exact
  arithmetic of the kernel term (float32 before truncation or not) is fixed and gated by G25-G34
  (B3a, B7); this note only reserves integer storage.
  **[B3a]** For the pure part this is now fixed in section 7: kernel terms are exact integers,
  and floating point enters only when a kernel term is combined with the layer's native rest
  (one float32 addition, then truncation to uint32). Equality with the native float32 order is
  B7's to gate (G26, G28).
- **Accuracy.** Top-1 is stored as an exact count and total (`*_correct`, `*_n`, int64) with the
  float64 ratio beside it, so that "equal accuracy" in the tie rules (§8.1) is an integer
  comparison. Exact full-precision metrics are recorded; selection never reads rounded filenames
  (§7.1).
- **Splits.** `train`, `val`, `test`. No `test` key exists in any record before the §8.5 freeze
  manifest (§8.5, §12 last paragraph).
- **Grid.** G = {100000, 200000, 300000, 350000, 400000, 450000} in H units (§8.1). The PID
  target is 350000 H (§7.2).
- **Hashes** are lowercase sha256 hex strings.
- **Container.** Open (B7/B8). Proposal: one JSON object per epoch in `epochs.jsonl`, one JSON
  file per saved checkpoint, one run summary; integers as JSON integers, float64 written with
  round-trip precision.
  **[B7]** The per-epoch record is one JSON object per line of `epochs.jsonl` in the run directory (sorted keys),
  as proposed; the checkpoint sidecar and run summary stay B8's (M9).

## 1. Per-epoch frozen record (writer M8; readers M9, M12, M15, M16)

Written by the frozen-evaluation callback appended after StopIf, at every epoch end, after
validation. It writes only new keys, so the released callbacks read the released values (§7.1).
It never calls a layer in training or tracing mode and draws nothing from the module-level
`random` stream (§3.3, §6 step 6).

| key | type | meaning | STUDY |
| --- | --- | --- | --- |
| `run_id` | str | run name: arm, form, seed, stage, ramp or PID, attempt, source bundle | §12 |
| `epoch`, `step` | int64 | epoch index from 0; optimizer step count | §8.1, §12 |
| `val_correct`, `val_n`, `val_top1` | int64, int64, float64 | frozen validation top-1 (argmax agreement), moving moments | §8, §8.1, §3.3 |
| `H1`, `H2`, `NR` | int64 | frozen totals on moving moments and saved quantizer parameters | §3.1(c), §3.2, §3.3, §8.1 |
| `subtotals` | object | per layer and per group (`attention`, `ffn`, `head`, `embedding`, `other`): `{H1, H2, NR, native}`; groups as in §2.2 "Sites" | §12, §2.2 |
| `train_count` | float64 **[B7] int64** | the training count FreeEBOPs logs (`logs['ebops']`), copied, not recomputed. **[B7]** FreeEBOPs writes a Python int (`hgq/utils/sugar/ebops.py:11-15`), so the record holds an int64 (owner decision B7-6) | §3.3, §12 |
| `log_loss`, `log_accuracy`, `log_val_loss`, `log_val_accuracy` | float64 | Keras's logged values, which Chang's ParetoFront and StopIf read | §3.3, §12, §8.10 |
| `lr`, `beta` | float64 | learning rate and λ at the epoch | §12, §7.1 |
| `batch_index_seq_sha256` | str | hash of the mapped batch indices fetched in `Dataset.__getitem__` this epoch. **[B4]** sha256 of the little-endian int64 bytes of the mapped indices in fetch order (`cbl.rng.int_seq_sha256`; section 10) | §6 step 7, §12 |
| `index_map_sha256` | str | hash of `_index_map` (secondary record). **[B4]** the map in force during the epoch, same byte rule | §6 step 7, §12 |
| `vars_sha256_before`, `vars_sha256_after` | str | variable hash around the frozen evaluation; must be equal | §3.3 ("mutates? no"); plan M8 |
| `snap_changed` | int64 | values whose snap to their quantizer's f changed them; must be 0 | §3.2 grid exactness |
| `quantizer_structure` | object | per layer of Q, K, V, O and the FFN contraction: zero kernels by route, all-zero-width input or output quantizer; per head, the zero-width fraction of the softmax output quantizer; read from quantizer state without a data pass | §12 structure bullet |
| `binary_sites` | object or null | per binary site: q_g `b`, `i` (scalar, or per column at fused-BN sites), q_g(g) or q_g(g_c), zero columns by route (`pruned`, `rounded_to_zero`), saturated columns, sign occupancy, sign-flip rate; at fused-BN binary sites also the FB sign-difference fraction (batch against moving moments), single-sign columns, dead channels, train-mode against frozen top-1 gap | §12, §2.2, §2.3 |
| `pruning_channel` | object | per HGQ site: fraction of zero-width kernel coefficients and the share of native cost they remove; fraction of zero-width inputs at binary sites; open: §12 asks for it "at every grid checkpoint", so M9 may copy it into the front sidecar instead | §12 |
| `pid` | object or null | PID runs only: `sensed_H` (int64; equals the headline key of the same epoch exactly), `stock_aggregate`, `beta`, `error`, `integral`, `clamp_state`, `slack`. **[B9]** filled from the PID's `pid/` log keys (section 18): also `acted_on_H` and `beta_unclamped`; `error`, `acted_on_H` and `beta_unclamped` are absent in warmup epochs; null in ramp runs | §7.2, §12 |

**[B7]** Built by M8 as section 16 states: `subtotals` is `{layer: {<name>: {H1, H2, NR, native, counter, penalty_b,
group, kind, class}}, group: {<group>: {H1, H2, NR, native}}}`; `quantizer_structure`, `pruning_channel` and the
data-pass `binary_sites` fields are null until B8, `pid` until B9 (owner decision B7-5).

Readers: M9 builds fronts from `val_*` and the three counts; M12 reads the cached headline value
of the same evaluation (§7.2 "Sensor timing"); M16 reads `val_*`, the counts and `epoch` for the
collapse diagnostic and dwell times (§8.8; §12 last bullet).

## 2. Front-checkpoint sidecar (writer M9, with M10 and M11; readers M16, M14, M18)

One per saved checkpoint. A checkpoint is saved when it joins the nondominated set (higher
accuracy, lower cost) under H, and the descriptive fronts under the other count and NR are
tracked separately (§8.1).

| key | type | meaning | STUDY |
| --- | --- | --- | --- |
| `run_id`, `epoch` | str, int64 | as in section 1 | §8.1 |
| `ckpt_path`, `ckpt_sha256` | str, str | relative path and content hash; the hash is the last tie-break | §8.1 ties |
| `val_correct`, `val_n`, `val_top1`, `H1`, `H2`, `NR` | as section 1 | copied from the epoch record; the checkpoint audit must reproduce them exactly | §3.3 |
| `joined` | list of str | which fronts it joined when saved: subset of `H1`, `H2`, `NR` | §8.1 |
| `chang_filter` | object | per count: `val_top1 > 0.5 and cost < 500000` | §8.1 |
| `val_auc_macro`, `val_auc_per_class` | float64, float64[5] | macro one-vs-rest AUC and the five per-class AUCs, for saved candidates | §8, §12 |
| `structure_record` | object | on the validation jets, inference mode: degenerate layers among Q, K, V, O and the FFN contraction, each with route (`kernel_pruned`, `kernel_rounded_to_zero`, `input_zero_width`, `output_zero_width`, `other`); zero columns per site with route (`pruned`, `rounded_to_zero`); per projection the fraction of inputs zero-width at every position; per head the softmax zero-width fraction; for a degenerate Q, per head `q_identical_across_t` and `q_all_zero` (bool) | §8.1, §2.2 |
| `attention_diagnostics` | object | validation only: Q/K/V/O and QK/AV cost and widths, `H_attn` with `all_zero_rows` and `negative_rows` counts, branch norm, `delta_attn_mean`, `delta_attn_zero`, `delta_static`, `delta_unif`, and per ablation the downstream WRAP overflow count with the unablated baseline | §12 (definitions §8.8) |

## 3. Run summary (writer M9 at run end; reader M16)

| key | type | meaning | STUDY |
| --- | --- | --- | --- |
| `run_id`, `config_sha256`, `code_sha` | str | identity of the run, its config and the fork commit | §12; RULES §5 |
| `end` | str | `completed`, `stopif`, `integrity_stop`, `budget_stop`, `failed`; failed runs and empty fronts stay in the ledger | §7.1, §8.1 |
| `epochs_completed` | int64 | | §7.1 |
| `front` | object | per count (`H1`, `H2`, `NR`): the nondominated set at run end, each point `{epoch, ckpt_sha256, val_correct, val_n, cost}`, ties as in §8.1 | §8.1 |
| `envelope` | object | per count and per g in G: the front point with the highest `val_top1` at cost ≤ g under the §8.1 ties, with `cost` and `slack = g − cost`; the string `"NA"` when none is feasible, never 0 | §8.1 |
| `filter_outcome` | object | per count and g: `feasible`, `COST INFEASIBLE` or `NO ACCEPTED CHECKPOINT` with its diagnostic accuracy and cost | §8.8 |
| `diagnostic_ckpts` | object | per count: best `val_top1` at cost ≤ 350000 ignoring the filter, and the minimum-cost checkpoint; never in the front or BEST | §8.1 |
| `released_front` | list | Chang's callback front as released, by training count: `{epoch, train_count, log_val_accuracy, path}`; source of the §8.10 CH-8 context point | §3.3, §8.10 |
| `dwell` | object | per count: epochs spent in each grid interval | §12, §8.8 |
| `pid_selection` | object or null | PID runs: highest `val_top1` with frozen H ≤ 350000 passing Chang's filter, with cost and slack | §8.6 |

The per-count blocks hold H1, H2 and NR alike, so that M9 and M16 can be built and tested before
the headline is fixed (§3.4). M16 reads the block the config's `headline` names as the primary
front and the other two as descriptive (§3.1(c), §8.1).

M16 computes, and does not expect in the summary: D_screen and D_conf (§8.2), the form choice
(§8.3), BEST (§8.4), g* and D_band (§8.2), the S2 outputs (§8.6), the collapse labels (§8.8) and
the confirmation test (§8.7). Pairing is by `(config_code, form, seed)` against CH-8 at the same
seed (§6 "Pairing"). The held-out prediction record is open: it is the sealed reader's output
(M3, B2) and exists only after the §8.5 freeze manifest.

**[B3b]** M16 rechecks two of these fields rather than trusting them: `cbl.select.run_from_records`
recomputes each `envelope` entry from `front` with the §8.1 ties and refuses a mismatch, a 0 in
place of `"NA"`, or a front point failing Chang's filter; `cbl.select.check_pid_summary` recomputes
`pid_selection` from the H front (its step envelope at 350000) and refuses a different checkpoint.
`end` values `completed` and `stopif` count as complete in the confirmation; `failed`,
`integrity_stop` and `budget_stop` make C2, C3 or C3r INCOMPLETE (section 10, owner decision B3b-21).
The held-out input record M16 expects from the sealed reader is in section 10.

## 4. Constants every module shares

| constant | value | STUDY |
| --- | --- | --- |
| grid G | 100000, 200000, 300000, 350000, 400000, 450000 (H) | §8.1 |
| Chang's filter | `val_top1 > 0.5` and cost `< 500000` | §8.1 |
| collapse δ | 0.05 top-1 [OD-14] | §8.8 |
| classes | 5 softmax outputs | §8 |

## 5. Config schema (writer M17; readers M13, M20, M16)

One JSON object per run. The generator writes every field; nothing is hand-edited (03-phases,
phase 2). **Fail-closed inputs:** `headline`, `claim_domain` and, for S4, `s4_option` have no
default anywhere in the code. While Kai's decision or the §3.4 rule has not fixed them, the
generator writes `null`. **B18 / STUDY v0.13 §3.4 v0.4:** CH-8 and F000 ramp runs
may train before the headline is fixed; every binary arm (including engineering and F100)
and every PID run still requires H1/H2. Claim domain and other launch requirements still apply.
The three `launch_confirmations` (od13, od15, od12) must each be true.

| field | type | allowed values | required | STUDY |
| --- | --- | --- | --- | --- |
| `schema_version` | int | 1 for this draft | always | none |
| `campaign` | str | `chang-bitlinear-20261016` | always | §12 (W&B group) |
| `arm_id` | str | `CH-8`, `F000`, `F100`, `F001`, `F010`, `F011`, `F101`, `F110`, `F111`, `PID-CH`, `PID-BEST`, `FP32`, `W8A8`, `DS-HGQ`, `GR`, `BE`, `CH-8-C`, `BEST-C`, `PID-CH-C`, `PID-BEST-C` | always | §5 table |
| `builder` | str | `released` (CH-8: the released builder verbatim, with the declared departures) or `lab` (M5) | always | §5, §13 reference row |
| `config_code` | str or null | `F` plus three bits in the order attention, FFN, hidden head; null for CH-8, FP32, W8A8 and DS-HGQ | always | §5 "Codes" |
| `groups` | object | `{attention, ffn, head}`, each `Q` (Chang's learned-width HGQ group) or `B` (the binary map) | when `builder = lab` | §5 "Codes", §2.2 "Sites" |
| `embedding` | str | `Q`; `B` only in BE | when `builder = lab` | §2.2 "Sites", §5 BE |
| `form` | str or null | `FB` or `BF` when any fused-BN site is binary (FFN expansion, the three hidden-head layers, or the BE embedding); `null` otherwise (F000 and F100 are identical in both forms). A fused-BN binary site with `form = null` is refused | conditional | §2.3, §5 |
| `form_source` | str | `fixed`, or `s2_choice:<sha256>` for arms in "the chosen form" (PID-BEST, GR, BE, BEST-C, PID-BEST-C); refused while the S2 freeze record does not exist. **[B12]** null while `form` is null, including a template before S2 (section 14) | when `form` is set | §5, §8.3 |
| `best_source` | str or null | `s2_choice:<sha256>` for PID-BEST, BEST-C, PID-BEST-C; refused while the S2 record does not exist | conditional | §5, §8.4 |
| `control` | str or null | `FP32`, `W8A8`, `DS-HGQ`, `GR`, `BE`, or null | always | §5 "Control definitions" |
| `seed` | int | 1-3 in the screen; 4-8 (option A) or 4-11 (options B, C) in S4 | always | §5, §6 |
| `stage` | str | `S1`, `S3`, `S4` (S2 trains nothing). **[B12]** also the engineering stages `smoke`, `bench`, `det` (the STUDY section 13 determinism record), as M15 already admits (section 8) | always | §5 "Stages" |
| `tier` | str or null | `T0` to `T6` as in the §5 table; null in S4 | always | §5 |
| `schedule` | str | `ramp` or `pid` | always | §5, §7.1, §7.2 |
| `ramp` | object | **[B5]** must equal `cbl.runner.RAMP_RECORD` exactly (keys in section 13). Chang's released values, imported from `chang/run_train.py` and recorded for audit: λ `PieceWiseSchedule([(0,2e-8,'linear'),(2000,3e-7,'log'),(7000,3e-6,'constant')])`; lr 3e-3, cosine restarts every 500 epochs, `t_mul = m_mul = 1`, floor 1e-6, `alpha_steps = 10`; batch 2790; `steps_per_execution = 4`. **[B12]** the generator imports `RAMP_RECORD` from `cbl.runner` (an import of standard-library code only) and G57 checks it against the values `gen.release` reads from `run_train.py`'s syntax tree; the same non-λ values, with the optimizer, loss, filter and StopIf, are in `recipe` for every arm (section 14) | `schedule = ramp` | §7.1 |
| `pid` | object | `target_H = 350000`, `init_beta = 2e-8`, p 1, i 0.002, d 0, warmup 10, `log = true`, `max_beta = inf`, `min_beta = 0`, damping 0. **[B9]** must equal `cbl.pid.PID_RECORD` in value and JSON type; `max_beta` is the JSON string `"inf"` (B12), read by `cbl.pid.parse_max_beta`; a PID config has no `ramp` field, and a ramp config no `pid` field (section 18) | `schedule = pid` | §7.2 |
| `epochs` | int | 7000. **[B12]** 3 for `smoke`, 20 for `det`, the `bench_epochs` input for `bench` (null while unset) | always | §7.1 |
| `cost_callbacks` | bool | false only for FP32 (cost callbacks and StopIf off) | always | §7.1 |
| `headline` | str or null | `H1` or `H2`; no default; fixed by the recorded v0.4 export-check outcome for both forms; CH-8/F000 ramp training may precede it. It selects the front axis and, for binary arms, the penalty (b) branch | before launch | §3.1(b), §3.1(c), §3.4 |
| `claim_domain` | object or null | `{base: "a" or "b", c3r: true or false}`; no default; Kai decides before any launch (§19.2 item 12). `c3r` acts only under S4 option C. **[B3b]** two more fields, also without default: `band_rule` (required under base `b`, absent under `a`): `default` (§8.2), `strict_sd` or `interval` (the two §19.3 alternatives); `c3r_level` (required when `c3r` is true, absent otherwise): `bonferroni` (the pre-registered split) or `nested` (§8.7, item 12 (c)). See section 11 | before launch | §8.2, §8.7, §19.2 item 12, §19.3 |
| `s4_option` | str or null | `A`, `B` or `C`; no default; Kai decides after S2 | when `stage = S4` | §8.7, §19.2 item 11; plan M17 |
| `matmul_precision` | str | `float32`, set after `import model` and logged | always | §7.3 |
| `init_tensors` | object | **[B4]** `{manifest_path, manifest_sha256}`: the per-seed initial-tensor manifest of decision U1 (section 10), which names the tensor file and its sha256. The generator writes `manifest_path` (on `kai-data` beside the data for NRP runs) and `manifest_sha256: null`; the stage handoff supplies the sha256 after the draws of record (plan U1 consequence, B19). The runner refuses a null or mismatching sha256 (`cbl.rng.read_manifest`) | always | §6 steps 3-4; plan M4 |
| `prefit_reseed` | int | **[B4]** `cbl.rng.prefit_reseed(seed)` = `int.from_bytes(sha256(b"cbl1016-fit\0" + str(seed).encode()).digest()[:8], "big")` (decision U5); the runner recomputes it and refuses a mismatch (`SeedFlow.reseed_before_fit(expect_prefit_reseed=)`) | always | §6 step 5; plan M4 |
| `data_manifest_sha256` | str | the preparation manifest; open until B2 (M3). **[B2]** sha256 of the bytes of `manifests/data-prep-150c.json` (section 9) | always | §4; plan M3 |
| `code_sha` | str | the fork commit the run uses. **[B12]** null in every committed config: a file cannot name the commit that contains it; the handoff supplies it and `gen.configs.check_launchable` refuses without it (owner decision B12-1) | always | RULES §5 |
| `param_count` | int or null | filled when the config builds on CPU (G57-G60). **[B12]** from gate G58's record (`param_counts` input, section 14), keyed by the model key; null until G58 runs on the PC | after build | 03-phases phase 2 |
| `wandb` | object | `{project: "bnjettag-chang-bitlinear", group: "chang-bitlinear-20261016", run_name, mode}`; the run name carries arm, form, seed, stage, ramp or PID, attempt and source bundle | always | §12 |
| `attempt` | int | 1 for the first submission | always | §12 |

## 6. Open items this draft leaves to later batches

- Container format and file layout of the three records (B7, B8).
- Exact arithmetic of the kernel term before uint32 truncation (B3a G25; B7 G26-G34).
  **[B3a]** Pure part fixed in section 7 and gated by G25; equality with native order, including
  whether the backend fuses the bias term's multiply with the add, stays open for B7 (G26, G28).
- Dtype of the logged `train_count` (B7). **[B7]** int64 (section 1).
- **[B7]** Where the pruning-channel record lives stays B8's; the structure routes per epoch (`quantizer_structure`)
  join B8's M10, which owns the zero-variance detector (section 16).
- Where the pruning-channel record lives, epoch record or front sidecar (B8).
- `init_tensors` and `prefit_reseed` (B4, decisions U1 and U5). **[B4]** Fixed in sections 5 and 10.
- Held-out prediction record and the freeze manifest schema (B2 sealed reader; B3b §8.5).
  **[B2]** The sealed reader exists (section 9); the freeze manifest's schema and completeness
  stay M16's (B3b, G56), and the held-out prediction record is produced through `read_heldout`
  after the freeze.
  **[B3b]** The freeze manifest schema and M16's held-out input record are fixed in section 10;
  the sealed reader's own output format stays B2's, and the G56 PCC part checks that it calls
  `cbl.select.require_freeze_manifest`.

## 7. Per-site input record for the counts [B3a] (writer M7 model part, B7; reader `cbl/cost_pure.py`)

**[B3a]** Added in batch B3a. The frozen evaluator (M8) and the model part of M7 (B7) extract
these plain numbers from the layers at the frozen state (§3.3: moving moments, saved quantizer
parameters, no layer call in training or tracing mode) and pass them to `cbl.cost_pure`, which
returns H1, H2 and NR per layer, per group and in total. The record is the Python dataclass
`SiteInput` (one per constant-kernel site) and `NativeLayer` (one per layer without a constant
kernel); the function is `aggregate(sites, native_layers)`, with `site_terms(site)` per site.

**Canonical form.** Every constant-kernel site is the contraction Σ_{t,i,c} b_{t,i}·k_{ic} over
positions t, inputs i and output columns c (§3.2 "Per-position widths"). P = 8 at per-position
sites and P = 1 at the pooled head sites. Kernels with more than two axes are flattened in C
order: the Q, K and V kernels (24, 2, 16) become (24, 32) with column c = 16·head + dim; the O
kernel (2, 16, 24) becomes (32, 24) with input i = 16·head + dim, and the O input widths
(P, 2, 16) are flattened the same way. B7 states the mapping it uses for each layer.

`SiteInput`:

| field | type | meaning | STUDY |
| --- | --- | --- | --- |
| `name` | str | layer name, unique in the model | §12 subtotals |
| `group` | str | `attention`, `ffn`, `head`, `embedding` or `other` (section 1 `subtotals`) | §2.2 "Sites" |
| `kind` | str | `hgq` or `binary`. The rules are identical; `binary` also requires every column to be zero or ±g_c with no zero coefficient | §2.2, §3.2 |
| `b` | int (P, n_in), ≥ 0 | native input widths b_{t,i}, the input quantizer's `bits_` per position and input; must equal max(0, I + F) (KIF b = relu(i + f), KBI f = b − i) | §3.2 |
| `I`, `F` | int (P, n_in) | the input quantizer's integer and fractional bits, rounded as the quantizer rounds them | §3.2 u_c |
| `K` | float32 (n_in, n_out) | exported constants: `qkernel` at HGQ sites, q_g(g)·S or q_g(g_c)·S at binary sites, on the frozen state; every value must be a float32 | §2.2, §3.2, §3.3 |
| `kernel_f` | int, broadcastable to (n_in, n_out) | f = b − i of each coefficient's quantizer: per value for kq, per layer or per column for q_g | §3.2 grid exactness |
| `kernel_bits` | int, broadcastable to (n_in, n_out), ≥ 0 | NR's kernel widths: kq.bits_ at HGQ sites, q_g's `bits_` broadcast over the input axis at binary sites | §3.1(a), (a′) |
| `rest_f32` | numpy float32 scalar or null | the layer's native contribution other than the kernel term, as the layer computes it in float32 (the bias term `mean(bq.bits_)·size`); null when the layer has no bias | §3.2 dtype |

`NativeLayer`: `name`, `group` as above, and `native` (int, 0 to 2³²−1), the layer's own uint32
counter value. It covers layers without a constant kernel (QAdd, the tanh tables, the softmax,
the MHA's own QK and AV counter, pooling); H1, H2 and NR all take the native value there (§3.2).

Outputs of `aggregate`:

| key | type | meaning |
| --- | --- | --- |
| `H1`, `H2`, `NR` | int | totals, the integer sum of per-layer values (section 1 keys) |
| `layers` | object | per layer: `group`, `kind` (`hgq`, `binary` or `native`), `H1`, `H2`, `NR` (section 1 `subtotals`, without `native`) |
| `groups` | object | per group in the list above: `H1`, `H2`, `NR` |
| `snap_changed` | int | always 0 on return; a snap that changes a value raises `SnapError` instead (section 1 `snap_changed`) |
| `sites` | object | per site a `SiteTerms` record: kernel terms `h1_kernel`, `h2_kernel`, `nr_kernel` and per column `class` (`zero`, `single`, `other`), `g_c`, `w_g`, `H1_c`, `H2_c`, `NR_c`, `branch` (`h1` or `sign_scale` for a single-magnitude column, ties to `h1`), `sum_b`, `sum_u` |

Arithmetic:
- w(·), the snap, u_c, the column rule and all kernel terms are exact integers or rationals.
- Floating point enters only in the per-layer value, trunc_uint32(float32(kernel term) +
  `rest_f32`), which is one IEEE float32 addition (owner decision B3a-3). A kernel term above 2²⁴
  raises `Float32RangeError`. At or below it, every partial sum of the kernel term is an exactly
  representable float32 integer, so the native float32 einsum gives the same integer in any order.
- Each constant is snapped to `kernel_f` and must be unchanged (`SnapError`), and w(K_ic) ≤
  `kernel_bits` must hold for every coefficient (`WidthBoundError`), which is H1 ≤ NR per
  coefficient (§3.2).

Owner decisions B3a-1 (max F over the same inputs as the sum in u_c) and B3a-2 (u_c(t) = 0 at a
position with no included input) are recorded in `preflight/gates/G25/<sha12>/PREFLIGHT-rows.md`.

**[B7]** `NativeLayer.native` is filled by `cbl.cost` with the layer's own `_compute_ebops(*shapes) * ebops_factor`
at the frozen state, cast to uint32 by the backend as the counter is written, not with the counter itself: the
counter holds the last training-mode call's value, computed at that batch's WRAP integer bits before the optimizer
step (owner decision B7-3). The counter is recorded beside it as `counter` (section 16).

Not in this record, and left to B7: the penalty (b), computed inside the trace from quantizer
variables (§3.1(b)); the `native` entry of `subtotals`, which B7 reads from the layer counters,
remembering that a binary layer's counter holds the penalty (b), not native (§3.1(a′)); and the
gate that the per-layer values equal native on F000 and CH-8 (G28).

## 8. W&B run identity and keys (batch B11, module M15)

Numbered 7 on branch b11; renumbered 8 at the merge into main (orchestrator, 2026-10-18), because b3a added its section 7 first. The two references in the `cbl/wandb_log.py` docstring (lines 14 and 18) were corrected to section 8 in batch B2.

Added in B11. The catalogue of record is `cbl.wandb_log.KEYS` (pattern, row, traits, STUDY line);
this section mirrors it. Gate G52 checks it.

**Run identity (§12).**
- Run name `<arm_id>_<form>_s<seed>_<stage>_<schedule>_a<attempt>_src-<12 hex>`; `form` is `nf`
  when the config's form is null. Fields are joined by `_` because arm ids contain `-`.
  **[B12]** An optional replica token `_r<replica>` follows `<schedule>`: empty when the config's
  `replica` is null (every identity with a single run, so every earlier name and run id is
  unchanged), and allowed only in stage `det`, where the STUDY section 13 determinism record runs the
  CH-8 seed-1 identity twice (`CH-8_nf_s1_det_ramp_r1_a1`, `..._r2_a1`). Owner decision B12-5.
- The source bundle is an explicit input to `Run.open` with no default: the content hash
  (12-64 lowercase hex, the name keeps 12) of the bundle a run was launched from. **Open:** which
  bundle (code bundle on `kai-data`, or the immutable handoff) is settled by B17/M20. Because a
  config sits inside that bundle, the config's `wandb.run_name` (section 5) holds the stem
  without the `_src-` token, or null; M15 refuses any other value.
- Run id `cbl` + the first 12 hex of sha256(run name). M14 persists it; `Run.open(expect_run_id=)`
  refuses a mismatch. Offline runs ignore `resume`, so a resumed offline process writes a second
  offline directory with the same id.
- `wandb.mode` is `online`, `offline` or `disabled`; PREFLIGHT uses only `offline` and
  `disabled`. A different non-empty `WANDB_MODE`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_GROUP`
  or `WANDB_RUN_GROUP` in the environment is refused.
- Stages `S1`, `S3`, `S4` (job type `train`), and the engineering stages `smoke`, `bench`, `det`
  (the determinism record) and `gate` (job type `engineering`), all in the one STUDY group.

**Key names.** A W&B key is the `/`-joined path of the record key, so section 1 names are used
unchanged (`val_top1`, `H1`, `batch_index_seq_sha256`, ...). Saved-checkpoint rows carry the
section 2 sidecar under `ckpt/`. The run config holds `identity/*` (written by M15),
`provenance/*` (from the runner) and `config/*` (the run's config). Layer, site, quantizer and
head names are single path segments: no `/`, nested names joined by `.` (`mha.query`).
Lists become `<key>/<i>`, booleans 0/1, strings (hashes) are kept, null is dropped.

**Inner keys fixed here** (section 1 left them open), required when the trait holds:

| key pattern | trait | STUDY §12 item (v0.8 line) |
| --- | --- | --- |
| `subtotals/layer/<layer>/{H1,H2,NR,native}`, `subtotals/group/<group>/{...}` | cost | per-layer and per-group subtotals (1289) |
| `quantizers/<layer>/<quantizer>/{width_mean,width_max,step_min,step_max}` (new) | quantized | widths and steps (1290) |
| `binary_sites/<site>/{qg_b,qg_i,qg_value,zero_cols/pruned,zero_cols/rounded_to_zero,saturated_cols,sign_occupancy,sign_flip_rate,moment_mismatch_mean,moment_mismatch_var}` | binary | binary-site items and the x·W / x·K moment mismatch (1290) |
| `bn/<layer>/{gamma_min,gamma_max,moving_mean_absmax,moving_var_min,moving_var_max}` (new) | all | BN scales and moments (1290) |
| `quantizer_structure/<layer>/{kernel_zero_pruned,kernel_zero_rounded,input_all_zero_width,output_all_zero_width}` | quantized | structure routes (1291) |
| `quantizer_structure/softmax_heads/<head>/zero_width_frac` | quantized, attention | softmax zero-width fraction per head (1291) |
| `activations/<quantizer>/{min,max,zero_width_frac}` (new) | quantized | activation ranges and zero-width fractions (1292) |
| `pruning_channel/<site>/{zero_width_frac,native_share_removed}`; `pruning_channel/<site>/input_zero_width_frac` | quantized; binary | pruning channel (1293) |
| `binary_sites/<site>/fb_sign_diff_frac` | fb | FB sign difference (1295) |
| `binary_sites/<site>/{single_sign_cols,dead_channel_frac}`, top-level `train_frozen_top1_gap` | fused_bn_binary | fused-BN items (1296-1298) |
| `dwell/<count>/<interval>` (new; cumulative epochs per grid interval) | cost | epochs per grid interval (1299) |
| `pid/{sensed_H,stock_aggregate,error,integral,clamp_state,slack}` | pid | PID items (1301) |
| `sys/{epoch_seconds,host_rss_mb,alloc_hours}` (new; from the runner) | all | time, memory, allocation hours (1301) |
| `ckpt/attention_diagnostics/{q,k,v,o,qk,av}/{cost,width_mean}`, `.../softmax_zero_width/<head>`, `.../degenerate/<proj>/<route>` | quantized, attention | attention cost, widths, zero-width, degenerate by route (1303) |
| `ckpt/attention_diagnostics/{H_attn,all_zero_rows,branch_norm,delta_attn_mean,delta_attn_zero,delta_static,delta_unif}` | attention | attention diagnostics (1303) |
| `ckpt/{ckpt_sha256,val_auc_macro,val_auc_per_class/0..4}`, `ckpt/structure_record/*` | all | saved-candidate AUC (1287); zero-variance test (1291) |
| `provenance/{code_sha,config_sha256,data_manifest_sha256,split_sha256,order_sha256,backend,matmul_precision,gpu_product,pack_k}` | all (run config) | provenance, split and order hashes, backend, precision, GPU product and K (1301) |

Change against section 1: the train-mode against frozen top-1 gap is one model-level value per
epoch, so it is the top-level `train_frozen_top1_gap`, not a field of each `binary_sites` entry.

Traits come from the config (section 5): `cost` = `cost_callbacks`; `quantized` = control is not
FP32; `attention` = control is not DS-HGQ; `binary` = a group or the embedding is `B`;
`fused_bn_binary` = `form` is FB or BF; `fb` = `form` is FB; `pid` = `schedule` is `pid`.

**Held-out keys.** Before the §8.5 freeze manifest exists, every history, summary and config key
whose lower-cased text contains `test`, or contains `heldout` or `holdout` with `-_./` and spaces
removed, is refused (fail closed: `latest` is refused too). The manifest is given by path; a
missing path or file, or a file that is not a non-empty JSON object, means refuse. Its schema
and completeness belong to M16 (G56); M15 records its sha256 as `freeze/manifest_sha256`.

## 9. Data: preparation manifest, readers and the sealed held-out reader (batch B2, module M3)

**[B2]** Added in batch B2. Code of record: `cbl/data.py`; gates G06-G08.

**Preparation manifest** `manifests/data-prep-150c.json` (kind `cbl1016-data-prep-manifest`,
schema 1), written by `cbl/data.py prepare` on the PC from the archives of record and committed.
Hashes and counts only: no array, no statistic value. Top-level keys:

| key | content | STUDY |
| --- | --- | --- |
| `train`, `heldout`, `smoke` | `glob_order` (names as `Path(inp).glob('*.h5')` lists them, prepare_data.py:21); `files` in that order, each `{index, name, bytes, sha256, rows, offset, one_hot, label_counts, label_block_sha256}` plus `feature_block_sha256` (float16 cast of `jetConstituentList`) for training files; `archive` `{name, bytes, sha256, datasets, rows, label_counts, label_sha256}` plus `feature_sha256` except for the held-out archive; `blocks` (block k of the archive equals file k); the `prepare_command` | §4 "Split identity", "Manifests, hashes, class map, counts" |
| `split` | `seed` 42, `n_constituents` 8, `ptetaphi` true, `val_size`, `n_jets`, `n_train`, `n_val`, `order_sha256`, `split_sha256`, `stats_sha256`, `std_finite`, `std_nonzero`, `arrays` (sha256, dtype, shape of X_train, y_train, X_val, y_val), `label_counts` of train and val | §4 steps 5-8 |
| `disjoint` | shared file names, shared file sha256 count, shared per-jet digest count (sha256[:16] of each raw `jetConstituentList` row) | §4 last bullet |
| `provenance` | Zenodo record 3602260 (API record sha256), per tarball: bytes, md5 against the record, sha256, members, on-disk files against members | §4; plan B2 |
| `versions`, `chang_sha256`, `paths`, `prepared_at_code_sha` | interpreter and libraries; Chang's two files at their `code-base.txt` hashes; PC paths of record; the fork commit that prepared it | §13 data row |

Definitions, used by section 8's `provenance/*` keys:
- `data_manifest_sha256`: sha256 of the manifest file's bytes.
- `order_sha256`: sha256 of the C-ordered bytes of the int64 permutation `order` after
  `np.random.shuffle` (data.py:35-36); training jets are archive rows `order[:n_train]`.
- `split_sha256`: sha256 of the bytes of `np.sort(order[:n_train])` (which archive rows are
  training jets, independent of their order).
- `stats_sha256`: sha256 of the float32 bytes of `shift` followed by those of `scale`, each of
  shape (1, 1, 3) (data.py:28-29).

**Jet identity** is (file index in glob order, row in that file) (plan section 9 item 12):
archive row r is `files[k]` with `offset <= r < offset + rows`, row `r - offset`
(`cbl.data.jet_identity`).

**Readers.**
- `read_train_val(train_archive, n_constituents, ptetaphi, val_size=0.1, *, expect_sha256=None,
  expect_seed=42)` returns `TrainVal(X_train, y_train, X_val, y_val, shift, scale, record)`, equal
  bitwise to the first two results of Chang's `get_data` (G07). The caller seeds as
  run_train.py:41 does; the reader refuses if the global legacy generator is not in the state
  `np.random.seed(42)` leaves, and refuses a file named `150c-test.h5`.
- `read_heldout(archive, *, freeze_manifest, expect_archive_sha256, shift, scale, n_constituents,
  ptetaphi)` returns `(X_test, y_test, record)`; `record['freeze_manifest_sha256']` names the
  manifest that allowed the read. It refuses (`HeldOutSealed`) unless the freeze manifest passes
  M15's rule (`cbl.wandb_log.freeze_manifest_sha256`, called, not copied), and then unless the
  archive's sha256 equals `expect_archive_sha256`, which the caller takes from this manifest's
  `heldout.archive.sha256`. The rule is checked before the archive is touched. M16 owns the
  freeze manifest's schema and completeness (G56); until G56 joins them, "valid" means only
  M15's rule.
  **[B5]** Joined: `check_freeze_manifest` applies M15's rule and then
  `cbl.select.require_freeze_manifest` (M16's completeness), both before the archive is touched,
  and returns `{file_sha256, manifest_sha256}`; the read record names both,
  `freeze_manifest_sha256` (the file's bytes, M15) and `freeze_manifest_content_sha256` (the
  manifest's canonical content hash, M16). G56's PCC part gates it.

## 10. Seed flow, initial tensors, batch order and the stream guard (batch B4, module M4)

**[B4]** Added in batch B4. Code of record: `cbl/rng.py`; gates G10-G12. STUDY §6 steps 1-8,
§7.1 last bullet, §13 RNG row; plan decisions U1 and U5.

**Seed flow (runner M13).** `SeedFlow(seed, dataset_device='cpu:0' | 'gpu:0')`, then, in this order
and no other (each call refuses unless the previous step was the one before it):
1. `load_data(train_archive, expect_sha256=, data_manifest_sha256=)`: `np.random.seed(42)`, then
   `cbl.data.read_train_val` (section 9) with CH-8's 8 constituents and pt, etarel, phirel;
2. `seed_streams()`: `random.seed(s)`, `keras.utils.set_random_seed(s)` (which also reseeds NumPy);
3. `build_datasets()`: the training Dataset (2790, `shuffle=True`, `drop_last=False`) as the
   recording subclass, and the validation Dataset (2790), on `dataset_device` (decision U2); their
   construction draws must equal the reconstruction from `random.Random(s)`;
   `build_model(builder, arm_id=)`: the arm's model (M5 or Chang's `get_model`);
4. `load_initial_tensors(manifest_path, expect_manifest_sha256=, role_spec=)`;
5. `reseed_before_fit(expect_prefit_reseed=, guard=)`: `random.seed(prefit_reseed(s))`, arms the
   guard, keeps the training map at the reseed (`index_map_at_reseed`).
Compile happens before step 5 (STUDY §6 step 5: "just before `fit`"). `flow.record()` holds every
step with the hashes of both global streams after it (`py_state_sha256`, `np_state_sha256`:
sha256 of the JSON of the state tuple), the data and Dataset identity, the load record and the
reseed.

**Roles (M5, M6 fill them).** A role is `<top>/<segment>/.../<variable name>`: `<top>` the arm's
semantic name of a `model.layers` entry; one segment per tracked sublayer (`Layer._layers`), which is
`<Class>[k]` for a sublayer with Keras's automatic name, the suffix after `<parent name>_` for a
name derived from the parent's (`kq`, `iq`, `bq`, `oq`, `exp_table`, ...), and the name itself
otherwise (`query`, `key`, `value`, `attention_output`); then the variable's own name (`kernel`,
`b`, `i`, `k`, `f`, `beta`, `ebops`, `moving_mean`, ...). Examples: `ffn2/kq/FixedPointQuantizerKBI[0]/b`,
`mha/query/kernel`, `add1/MultipleQuantizers[0]/Quantizer[1]/FixedPointQuantizerKIF[0]/f`. CH-8's
14 top names, in `model.layers` order: `input`, `embed`, `tanh1`, `mha`, `add1`, `tanh2`, `ffn1`,
`ffn2`, `add2`, `pool`, `head1`, `head2`, `head3`, `logits` (`model.py:190-208`); 244 roles.
`RoleSpec(arm_id, top_names, expected_unmatched, expected_unused, n_variables, pending)`:
- F000 (B5) gives CH-8's names to its layers, so that both sets are empty and `n_variables` is 244;
- each binary arm (B6) declares its unmatched roles (the scale quantizers' b and i; §6 step 4 says
  they keep their constant initializers) and its unused roles (kq's variables at binary sites,
  which no binary arm builds, §2.2; kq's uint8 `k` is among them). If q_g carries more variables
  than b and i (a `k`, say), B6 records it as a finding against §6 step 4's wording;
- DS-HGQ, FP32 and W8A8 (B5) declare theirs the same way.

**Initial-tensor file (decision U1).** `draw_initial_tensors(flow, out_dir, place=, code_sha=)`
after steps 1-3 of CH-8 (released builder) writes `init-s<seed>-<platform>.npz` (uncompressed zip,
one `<role>.npy` per role, fixed member metadata, no pickles; equal tensors give equal bytes) and
`init-s<seed>-<platform>.manifest.json` (sorted keys, indent 1). Manifest keys: `kind`
`cbl1016-init-tensors`, `schema` 1, `label`, `place`, `created_utc`, `code_sha`, `arm_drawn`
`CH-8`, `builder`, `flow` (steps run), `identity`, `random_state_sha256_after_build`,
`stochastic_quantizers` (empty on CH-8), `probe_forward` (sha256 of the float32 logits on a fixed
batch from `default_rng(20261018)`, inference mode), `file` `{name, sha256, bytes}`, `n_roles`,
`roles` (role -> `{shape, dtype, sha256, trainable, path_at_draw}`), `role_order`. `identity` =
`{seed, device {jax_platform, device_kind, n_devices, dataset_device}, stack {package versions},
matmul_precision, draw_code {sha256 of cbl/rng.py, cbl/env.py, cbl/data.py, chang/model.py,
chang/data.py}, data {train_archive_sha256, data_manifest_sha256, order_sha256, split_sha256,
stats_sha256, n_train, n_val, n_constituents, ptetaphi}, datasets {batch sizes, n_batches,
remainder, init shifts, index_map_sha256_at_construction, random_state_sha256_after_datasets}}`.
The loader computes the same `identity` for the run and refuses any differing key; then the file's
sha256 and every tensor's sha256, dtype and shape; then the unmatched and unused sets against the
`RoleSpec`; nothing is assigned before all of these pass. The file depends on the archive (the
Datasets draw before the model build, and the draws depend on the archive's length), so a smoke
draw is never the draw of record. The manifest's own sha256 is what PREFLIGHT and the handoff name.

**Batch-order recorder (M8 reads it).** `flow.recorder` (a `BatchIndexRecorder`) is attached to the
training Dataset. The appended frozen-evaluation callback calls `recorder.mark_keras_epoch(epoch)`
in its `on_epoch_begin` (gates use `cbl.rng.epoch_marker`, a callback that does only that) and reads
`recorder.epoch(epoch)` in its `on_epoch_end`, where the epoch's Dataset hooks have all run:
`batch_index_seq_sha256`, `index_map_sha256`, `n_fetches`, `raw` (the adapter's order), `mapped`,
`index_map`, and checks (`mapped_is_permutation`, `mapped_equals_map_of_raw`,
`index_map_unchanged_during_epoch`, `on_epoch_begin_drew`). `recorder.prefit()` holds the two
fetches of Keras's symbolic build before epoch 0, which are not training fetches; `recorder.anomalies`
must stay empty. For a Dataset built by other code, `cbl.rng.instrument(ds)` attaches a recorder.

**Stream guard (M13 and M8).** The runner wraps each released callback with
`guard.wrap(cb, owner='released')` and each lab callback with `owner='lab'` before `fit`, passes
the guard to `reseed_before_fit`, and calls `guard.disarm()` after `fit`. The frozen evaluation (M8)
runs its non-callback steps inside `guard.watch('frozen evaluation')`. A change of the
module-level `random` state or of NumPy's legacy state by lab code raises `StreamViolation`, which
M13 ends as `integrity_stop` (section 3). `guard.record()` goes to the run's evidence.

**Reconstruction.** `reconstruct_fit_stream(reseed=, index_map=, n_batches=, epochs=)` models the
draws of Keras 3.14.0's JAX `fit` (cited in the module docstring): before epoch 0 one batch-order
shuffle, two fetches and one map reshuffle; per epoch one order shuffle, n fetches and one map
reshuffle. Another Keras version reruns G11 before use.

**For M14 (resume, B10).** Two things this batch found that resume must handle (STUDY §6 step 8,
§13 resume row): the Dataset's `_index_map` is Python state of its own, not part of the `random`
state, and must be persisted with it; and a fresh `fit(initial_epoch=k)` repeats the pre-fit draws
(symbolic build and reset) that an uninterrupted run made only once. Restoring both the `random`
state and the map in a callback's `on_train_begin` (which Keras calls after the symbolic build and
the reset, `backend/jax/trainer.py:418-419` then `:437`) would line the resumed stream up with the
uninterrupted one; that restore is §6 step 8's sanctioned write, so the guard is armed after it.
This is a note for B10, not something B4 implements or gates.
## 11. M16 records, decision inputs and entry points [B3b] (`cbl/select.py`, `stats.py`, `s2.py`, `confirm.py`)

Numbered 10 on branch b3b; renumbered 11 at the merge into main (orchestrator, 2026-10-18), because B4 added its section 10 on main first. References to "section 10" in M16 code and in the B3b rows mean this section.

Added in B3b. Gates G53-G56 check what is stated here. Owner decisions B3b-1 to B3b-29 are in
`preflight/gates/G53/<sha12>/PREFLIGHT-rows.md`.

**Inputs M16 reads.**
- The config (section 5) for identity: `arm_id`, `config_code`, `form`, `control`, `seed`,
  `stage`, `schedule`. A panel label is `CH-8`, `F000`, `F100`, or `<code>-<form>` for the six
  fused-BN configurations; CH-8-C, PID-CH and PID-CH-C take the label `CH-8`.
- The run summary (section 3): `run_id`, `end`, `front` and `envelope` for H1, H2 and NR,
  `filter_outcome` (per count and g: a string or an object with `outcome` = `feasible`,
  `COST INFEASIBLE` or `NO ACCEPTED CHECKPOINT`, plus diagnostic fields), `released_front`,
  `pid_selection`. Envelope keys are the grid values as strings or integers.
- The per-epoch record (section 1): `epoch`, `val_correct`, `val_n`, `H1`, `H2`, `NR`; for the
  event line `binary_sites/<site>/qg_b` (a number, or a list per column); for FP32 and W8A8 the
  record of the selected epoch must carry `ckpt_sha256` of its saved checkpoint.
- The front sidecar (section 2), keyed by `ckpt_sha256`. M16 reads `structure_record.degenerate`
  as an object `{Q, K, V, O, FFN}` with a route string or null each, `structure_record.q_all_zero`
  and `structure_record.q_identical_across_t` (a bool, or a list per head), and
  `attention_diagnostics.{delta_attn_mean, delta_static, delta_unif, H_attn}` with
  `attention_diagnostics.overflow` as an object `{<delta key>: count}`. These names are M16's
  expectation of the open structure-record layout (B8, M10 and M11); B8 either writes them or
  adds a mapping here.
- The determinism record's verdict as a bool `f000_bitwise_equal_ch8` (§13, §8.6): no default.

**Held-out input record** (the sealed reader's output, after the freeze): a mapping
`ckpt_sha256 -> {"correct": int, "n": int}` at the frozen pre-trace checkpoints; for the paired
bootstrap and the background rejection optionally `{"labels": int array (n,), <ckpt_sha256>:
{"correct": bool array (n,), "scores": float array (n, 5)}}`, jets in one fixed order for every
checkpoint; for the §8.10 pull, a mapping from the released-front `path` to the same counts.
C4 takes the deciding pairs per allowance (`"<g>"`, or `"PID"`) as a list of
`{lut_best, lut_ch8, part, stage, settings, status ("ok", "missing", "capacity_failed"),
allowance_exceeded_after_trace, post_trace_top1_best, post_trace_top1_ch8, function_changed}`.

**Decision inputs without a default** (`cbl.select.UnsetDecision` on anything else):
`headline` (`H1` or `H2`); `claim_domain` (section 5 with the B3b fields; `claim_domain_config`);
`s4_option` (`A`, `B`, `C`; `s4_design`, which also needs whether 350000 is in D_C2 under the
nested C3r variant); `f000_bitwise_equal_ch8`. Levels: A and B, C2 and C3 at 97.5 %; C alone,
C2 at 95 %; C with (c), C2 and C3r at 97.5 % (Bonferroni) or both at 95 % (nested, only with
350000 in D_C2, otherwise the split returns). Descriptive labels on the simultaneous intervals of 7
contrasts (A, B) or 6 (C), read at the grid points of D_conf only (STUDY v0.9 :1039-1042).
Power-table entries and plug-in powers: the closed form at rho = 0 and the B3b-1 simulation at
rho > 0 (STUDY v0.9 :749); the B3b gates check STUDY v0.9 (sha256 `61e68527…89df4`).

**Outputs** are JSON-ready objects (Fractions become `{num, den, float}` in the manifest):
`s2.s2_record` (the §8.6 validation record, in order), `s2.screen_heldout_tables`,
`s2.control_heldout_rows`, `s2.pid_screen_comparison`, `confirm.c2`, `confirm.descriptive_labels`,
`confirm.c3`, `confirm.c3r`, `confirm.c4`, `confirm.c6`, `confirm.pull_8_10`,
`select.collapse_table`. Screen outputs carry no ABOVE, BELOW, CROSSING or LOCAL token
(`s2.verdict_tokens` is empty), and carry the §8.6 labels.

**Freeze manifest** (`cbl.select.write_freeze_manifest`, `freeze_manifest_problems`,
`require_freeze_manifest`; schema_version 1). One JSON object, written once, with
`manifest_sha256` = sha256 of the canonical JSON (sorted keys, no whitespace) of every other
field. Required: `study_sha256`, `code_sha`, `headline`, `d_screen` (an ok domain), `form_choice`,
`best`, `claim_domain`, `grid_checkpoints` (count -> label -> seed -> g -> sha256 or `"NA"`, for
all 45 screen arms and the three counts), `controls_admitted` (list), `control_selections`,
`pid_selections` (`PID-CH` and `PID-BEST`, seeds 1-3), `context_8_10` (seeds 1-3),
`structure_records` (one per frozen grid and PID checkpoint), `s4` (`funded` true or false; when
funded `option`, `d_conf`, `selections`, and `d_band` under option (b)); `g_star` under option (b).
No key may name test or held-out data (the M15 rule of section 8). `SealedReaderStub` stands in
for M3's reader in the Mac part of G56; M3's reader of record calls `require_freeze_manifest`
before its one batch read (G56 PCC part, after the merge). On `main` at `8a5389b`, B2's
`cbl/data.py` `check_freeze_manifest` applies M15's rule only (a path, an existing file, a
non-empty JSON object); joining the two is a lane-P change after the merge, and the G56 PCC part
runs on it. **[B5]** Joined at `cbf8eb9` (section 9); G56's PCC part ran at `78c5515df8da`.

**Where the unadopted STUDY items 13-16 would enter** (§19.2; not implemented, by the B3b brief):
- Item 13, form chosen on F011 and F111 only: `cbl.select.form_choice` (it counts and averages
  over `Panel.fused_present`, so a panel built with `screen_labels` restricted to the two
  configurations in both forms gives the restated §8.3), with BEST unchanged in
  `cbl.select.best_choice`; the S1a/S1b staging belongs to M17 and M21.
- Item 14, a ternary control at BEST: a new control name in `cbl.s2.control_heldout_rows`
  (`CONTROL_REFERENCE`, against BEST in the chosen form) and in `cbl.select.control_slot`'s
  admission order; the map itself is M6's.
- Item 15, the CH-8 validation sentinel before the last S1 wave: the point is
  `cbl.select.context_checkpoint_8_10`; the check would be a validation counterpart of
  `cbl.confirm.pull_8_10` with margin m_s, called by the harness admission (M21).
- Item 16, the non-inferiority reading within m: `cbl.confirm.c2`, reading the same level-L
  intervals against -m when the verdict is not ABOVE ON GRID.

## 12. da4ml clean rerun: driver, sidecar and table [B15m] (writer M19 `da4ml_clean/`; readers G73, G74)

Numbered 9 on branch b15m; renumbered 12 at the merge into main (orchestrator, 2026-10-18), because B2, B4 and B3b added sections 9-11 first. References to "section 9" in `da4ml_clean/` and in the B15m rows mean this section.

Added in batch B15 (branch `b15m`). Nothing above is renamed or retyped. Sources: STUDY §3.4
Addendum A steps 2-7, §11 (mulder), §13 da4ml row, §19.2 item 5 (including the v0.8 interruption
mechanics). The driver and table are standard-library Python, run with any `python3` (never with
`$PY`); they call the frozen scripts by path and never edit them. Neither applies any part of the
headline rule: G74 does, from the table.

**Inputs with no default.** `--on-assertion {void-restart,fail-line}` and `--on-interrupt
{fail-line,void-restart}` are Kai's item-5 answers; `fail-line` under `--on-interrupt` is option
(i), `void-restart` option (ii). `--mode real` also needs `--decision` (the recorded answer, a
non-empty file whose sha256 goes into the run record) and `--equivalence-record` (G71), and
refuses `--expected-hashes` and `--home`; `--route clone` needs `--clone-python` and
`--clone-lock`. Every refusal happens before anything is written. Re-invocations must repeat the
route, version, policies and decision of the run record. Exit codes: 0 complete, 2 refused,
3 fault before any rule line (or a failed start check), 4 void not restarted (`--max-voids`,
default 2), 5 interrupted, 6 context halt (`--context-reviewed FILE` proceeds).

**Expected hashes.** `table.FROZEN_SHA256` and `table.ENV_SH_SHA256` hold STUDY v0.8 lines 60-72
verbatim (G72 compares them with the snapshot). Real mode uses only these; a dry run may pass a
JSON file of the stand-ins' hashes with the same ten keys.

**Evidence root** (`--evidence`, one per route version; becomes `preflight/da4ml-clean/<version>/`):

| path | content |
| --- | --- |
| `run.json` | run record, schema `cbl1016-da4ml-clean-run/1`: `run_id`, `dry_run`, `home`, `route`, `da4ml_version`, `policies`, `decision`, `equivalence_record`, `clone` (python, lock, lock_sha256), `expected_hashes`, `m3env_python`, `invocations[]` (driver and table sha256, argv, child environment variable names, exit), `attempts[]` |
| `driver.log` | the driver log |
| `scripts/` | byte copies of the nine frozen files and `env.sh`, taken after their hashes were checked |
| `freeze/<sha256>.txt` | every distinct `pip freeze` list, named by its hash |
| `attempts/aNN/sidecars/<line>.provenance.json` | per-line sidecar (below) |
| `attempts/aNN/stdout/<line>.stdout`, `.stderr` | `run_case.sh` output, written directly by the child |
| `attempts/aNN/reports/<line>/` | copies of the project artifacts (`case_info.json`, `STATUS`, `synth_result.json`, `csim_io.npz`, the C-synthesis report, the Vivado reports and log) and `build.log`, same relative layout as `$SCR/proj/main/<line>` |
| `attempts/aNN/archive-listing.tsv`, `numba-cache-listing.tsv` | sha256, size and path of every archived file and of `numba_cache` |
| `attempts/aNN/context-run-K/` | a superseded context run's sidecars, stdout and reports (step-4 rerun) |
| `attempts/aNN/results/table.json`, `table.md`, `table.sha256` | the per-line table and the sha256 of both files |

Archive on mulder: `$SCR/archive/<run_id>-aNN/` holds `proj_main/` and `logs/` renamed by step 3,
`env.sh.original` while a clone copy is in place, the copies afterwards (`env.sh.clone-copy-*`),
the exclusivity lock files and `context-run-K/`.

**Attempt record** (`run.json` `attempts[]`): `id`, `status` (`running`, `complete`, `void`,
`interrupted`, `fault`, `context-halt`, `abandoned`), `phase` (`start`, `step3`, `context`,
`rule`), `cause`, `started[]` (line names in start order; one attempt per line), `in_flight`
(line, pgid, state), `step3` (renames with inodes, listing hash, created directories, tmp,
numba_cache), `env_swap` (original and copy sha256, PY line, diff, swaps, restores with inode
check), `env_rechecks[]`, `exclusivity[]` (lock, process scans at start and end), `driver_checks[]`,
`context_runs`, `context_archives[]`, `context_completeness[]`, `context_reviews[]`, `resumes[]`
(option (i): `not_retried`, `exempt_from_dir_absent`, `next_line`), `m3env_freeze_before`,
`m3env_freeze_after[]` (with diff), `driver_sha256_at_rule_start`.

**Sidecar** (schema `cbl1016-da4ml-clean-sidecar/1`): `line_no`, `name` (as `run_case.sh:7-8`
forms it), `role` (`rule` or `context`), `pair`, `case`, `host`, `start`, `end`, `driver_sha256`,
`pgid` (also the session id: each line runs in its own session), `hashes_before` and `hashes_after` (sha256 of the nine frozen files and `env.sh`), `env` (SCR, PY,
TMPDIR, NUMBA_CACHE_DIR, KERAS_BACKEND sourced from `env.sh`), `KERAS_BACKEND`, `versions` (hls4ml,
da4ml, keras, hgq2, tensorflow, numpy from the distribution metadata under `PY`, plus python and
prefix), `pip_freeze_sha256`, `pip_freeze_list`, `run_case_argv`, `pgid`, `run_case_rc`, `reports`
(`build_rc`, `synth_rc`, `csynth_rc`, `vivado_rc`, `skip`, from stdout and `STATUS`),
`survivors_after_return` (session stop record or null; a stop record holds `sid`, `rounds` with each signal's pids and process groups, `signals`, `gone`), `assertions[]` (name, ok, detail),
`assertion_failures[]`, `policy_applied` (`void`, `fail-line`, `step-4 fault` or null),
`outcome` (`returned`, `interrupted`, `not_run_assertion_failed`), `interrupted` (`stage` in
`build`, `csynth`, `vivado`, `parse`; `cause`, `detected_by` `exit trap` or `restart`,
`group_stop` (the session stop record), `env_sh_restore`), `artifacts` (copied file to sha256), `foreign_processes_before`,
`foreign_processes_after`, `provenance_checks`.

Per-line assertions: before the line `proj_dir_absent`, `env_sh_sha256`,
`frozen_scripts_sha256_before`, `da4ml_version`, `stdout_recorded_file_created`; after it
`no_skip`, `stdout_recorded`, `frozen_scripts_sha256_after`, `env_sh_sha256_after`.

**Table** (schema `cbl1016-da4ml-clean-table/1`): `header` (run, attempt, status, label, route,
version, policies, cases and table-script sha256, expected-hash source, counts, note) and 44
`rows` in `cases_bc.txt` order with `line_no`, `name`, `shape`, `g`, `inp`, `path`, `bias_field`,
`role`, `pair`, `started`, `outcome`, `interrupted_stage`, `assertion_failures`, `policy_applied`,
`rc`, `skip_printed`, `stages` (`build`, `csim`, `csynth`, `vivado`: `completed` or `failed`),
`failed_at`, `fail_cause`, `values` (`csim_mismatch`, `n_values`, `da_kernel_cost_total`,
`II_min`, `II_max`, `est_clock_ns`, `post_opt_LUT`, `post_opt_WNS_ns`; null for a failed stage),
`artifacts_present`, `synth_result_json` (present, keys that differ from the raw reports),
`report_errors` (a report that exists but cannot be read is no stage result), `provenance`. The C-synthesis and Vivado values are read from the copied reports with the frozen
`parse_reports.py` functions `csynth`, `util` and `timing`, loaded by path after its hash check.
The table holds no key for a term, condition or headline of the frozen block.

## 13. Runner and lab builder [B5] (M13 `cbl/runner.py`, M5 `cbl/model.py`)

**[B5]** Added in batch B5. Gates G13 (runner against `run_train.py`), G14 (F000 against CH-8), G15
(nested MHA by role, single wrap), and the F000 parts of G10 and G11. STUDY v0.10 sections 5, 6, 7.1,
7.3, 13; plan decisions U1, U2, U3, U5.

**Entry points.** `prepare(cfg, *, train_archive, archive_kind, device, out_dir, source_bundle,
mem_fraction='0.5', data_manifest_path=None, evaluator_factory=None, keep_full=4, ctx=None)` runs
the recipe up to compile and returns a `Context` (`cfg`, `flow`, `model`, `roles`, `released`, `hook`,
`guard`, `wandb_run`, `record`); `fit(ctx)` reseeds (U5), arms the guard and fits; `finish(ctx, code)`
writes `run.json`; `run(cfg, **kw)` does all three and returns `(exit code, ctx)`. `draw(seed, *,
train_archive, archive_kind, device, out_dir, place)` writes CH-8's initial-tensor file of a seed on
the process's device (U1; the B19 draw Jobs). CLI: `cbl/runner.py run --config F --train-archive A
--archive-kind record|smoke --device cpu|gpu --out D --source-bundle HEX` and `cbl/runner.py draw ...`.
Exit codes: 0 completed or StopIf, 1 failed, 2 refused before training, 3 integrity stop (a stream
violation).

**Config acceptance (`validate_config`).** Schema 1 and campaign as section 5. Arms built in B5: CH-8
(`builder` `released`; `config_code`, `form`, `control` null; `groups` and `embedding` absent or null)
and F000 (`builder` `lab`, `groups` all `Q`, `embedding` `Q`, `config_code` `F000`); every other arm id
is refused as not built. **[B6]** The binary arms, GR and BE are admitted since B6 (section 15). `schedule` must be `ramp` (`pid` waits for M12, B9). **[B9]** `schedule` is `pid` exactly for the PID arms, which are admitted since B9 (section 18). `ramp` must equal
`RAMP_RECORD`:
`{"lambda": [[0, 2e-08, "linear"], [2000, 3e-07, "log"], [7000, 3e-06, "constant"]], "lr": 0.003,
"lr_first_decay_steps": 500, "t_mul": 1.0, "m_mul": 1.0, "lr_floor": 1e-06, "alpha_steps": 10,
"batch": 2790, "steps_per_execution": 4}`; M17 should import it from `cbl.runner` rather than restate
it. `epochs` is 7000 in S1, S3 and S4 and 1 to 7000 in the engineering stages `gate`, `smoke`,
`bench`, `det`. `claim_domain` may be null only in engineering validation; null `headline` additionally
admits count-independent CH-8/F000 ramp screen training (B18, v0.4). This supersedes B5-4
only for that screen exception; `s4_option` is required in S4. `cost_callbacks` true; `matmul_precision` `float32`;
`init_tensors.manifest_sha256` a 64-hex string (null refused); `prefit_reseed` equal to U5's value;
`data_manifest_sha256` equal to the sha256 of `manifests/data-prep-150c.json`; `code_sha` equal to the
fork's git HEAD with a clean tree (owner decision B5-8); `attempt` >= 1; `wandb.mode` `offline` or
`disabled` (`online` refused in PREFLIGHT, owner decision B5-7). The archive kind is an input, not a
config field: `record` (the manifest's `train.archive.sha256`) in any stage, `smoke` (its
`smoke.archive.sha256`) in engineering stages only.

**The appended callback.** `frozen_eval_hook(recorder, *, evaluator=None, guard=None, wandb_run=None,
epochs_path=None)` returns a `FrozenEvalHook` (the class name G13 checks in the callback order), appended after
StopIf. `on_epoch_begin` marks the Keras epoch in M4's recorder; `on_epoch_end` writes
`batch_index_seq_sha256` and `index_map_sha256` into the logs (refusing if a released callback
already wrote them), calls the evaluator inside `guard.watch('frozen evaluation')`, refuses if any
released key changed (compared as `float.hex`), appends a row to `epochs-stub.jsonl` and logs it
through M15. **M8 (B7) plugs in here** as `evaluator_factory(ctx) -> callable(epoch, logs)`, called once
after compile and before the reseed; the callable writes only new keys.
**[B7]** M8 is now built in: `prepare` constructs `ctx.evaluator` (a `cbl.frozen_eval.FrozenEvaluator`) and the
appended callback is `cbl.frozen_eval.make_hook`; `evaluator_factory` remains for gates only and returns a read-only
probe the callback runs after M8's evaluation (section 16).

**Run directory** (`out_dir`, created by `prepare`, refused if it exists): `run.json` (kind
`cbl1016-run-record`, schema 1: `config`, `config_sha256`, `summary`, `code` {head, clean, status,
files_sha256}, `release` {files_sha256, mem fraction set by run_train and restored}, `archive`, `env`,
`provenance`, `wandb` {mode, run_name, run_id, strict, provenance}, `load` (M4's load record),
`compile`, `callback_order`, `callback_signature`, `reseed`, `index_map_at_reseed`, `end`,
`epochs_completed`, `epoch_records`, `guard`, `recorder`, `flow`, `final_variables_sha256`,
`matmul_precision_before_fit`, `matmul_precision_after_fit`, `fit_seconds`, `exit`, timestamps);
`epochs-stub.jsonl` (per epoch: `epoch`, `step`, the two hashes, `log_loss`, `log_accuracy`,
`log_val_loss`, `log_val_accuracy`, `lr`, `beta`, `train_count`, `sys/epoch_seconds`, `logs_hex`)
**[B7]** replaced by `epochs.jsonl`, M8's section 1 record (section 16);
`pareto/` (Chang's ParetoFront); `wandb/` (offline mode). The stub rows are a subset of section 1's
record, which M8 completes.

**Lab builder (M5).** `arm_spec(arm_id, *, groups, embedding='Q', form=None, control=None) -> ArmSpec`
validates section 5's builder fields; `build(spec)` returns the model; `builder(spec)` a zero-argument
callable for `SeedFlow.build_model`; `role_spec(spec)` the arm's `RoleSpec` (F000: CH-8's top names by
class sequence, 244 variables, empty unmatched and unused sets). Site hooks for B6: `_bn_site`
(embedding, `ffn1`, `head1`-`head3`), `_dense_site` (`ffn2`) and `_attention` (the MHA) build Chang's
layer under `Q` and raise `NotBuilt` under `B`. The controls FP32, W8A8, DS-HGQ, GR and BE raise
`NotBuilt`. For B6 (G15): hgq2 0.1.9 wraps a QLayer class's `call` and `build` once at its first
instantiation, and a subclass that inherits an already wrapped method is wrapped again (the EBOPs
penalty added twice, the build check run twice); every lab subclass of an hgq layer, the binary MHA
included (U3), defines its own `call` and `build` from the unwrapped originals, keeping their
signatures (`functools.wraps`), as G15's passing control does.

## 14. Config generator, configs and the derived protocol [B12] (writer M17 `gen/`; readers M13, M20, M21, G57-G60)

Numbered 13 on branch b12; renumbered 14 at the merge of `main` at `46cc25fa9e48` into b12, because B5 added its section 13 first. References to "section 13" of these notes in the B12 commits and rows made before this merge (up to `f78d60c`) mean this section.

Added in batch B12 (branch `b12`). Gates G57 and G59 (Mac) check what is stated here; G58 and G60
need M5, M6 and M13 on the PC. Owner decisions B12-1 to B12-11 are in
`preflight/gates/G57/<sha12>/PREFLIGHT-rows.md`. Sources: STUDY v0.10 sections 5, 7.1, 7.2, 8.7,
10, 13 (arms row, determinism record), 19.2 items 9, 11, 12; plan rows M17, G57-G60, B12.

**Commands** (fork root, any interpreter with NumPy and SciPy; no Keras, JAX, hgq or TensorFlow is
imported): `python -I -B gen/cli.py generate` replaces `configs/`; `check` regenerates in memory and
diffs byte for byte (exit 1 on any missing, extra or different file); `summary` prints counts.
`configs/` is generated only: commit it in its own commit after `generate`.

**Input record `gen/inputs.json`** (`gen.inputs`): every key is required and has no default in the
code; null means unset. `od13`, `od15`, `od12` (true or null), confirmed individually
by Kai on 2026-10-19 (decisions.md, later 6); `headline` (`H1`/`H2`, v0.4 export outcome), `claim_domain` (validated by
`cbl.select.claim_domain_config`, section 11), `s4_option` (`A`/`B`/`C`, Kai after S2),
`s2_record` (path to the `cbl.s2.s2_record` output; the generator reads `form_choice.form`,
`best.best` and the file's sha256, and refuses an INCOMPLETE record), `item9` (`hold`, the STUDY
default while Kai's item-9 answer is open, or `cap475`, which moves only FP32, W8A8 and DS-HGQ to
S1), `data_manifest` (`{path, sha256}`; refused on a mismatch), `init_tensors` and
`smoke_init_tensors` (`{root, platform}`: manifest path `<root>/init-s<seed>-<platform>.manifest.json`,
section 10's file name; the root on kai-data is B17's layout), `bench_epochs`, `param_counts`
(path to G58's record `{kind: cbl1016-param-counts, counts: {<model key>: int}}`), `provenance`
(free text per key). B18 records claim domain `{base: "b", band_rule: "default", c3r: false,
c3r_level: null}`; a non-null level while c3r is false is refused. `headline` and `s4_option`
remain null. Items 13-16 of STUDY section 19.2 have no input.

**Fields added to the section 5 schema** (every config has all of them):

| field | type | meaning |
| --- | --- | --- |
| `label` | str | protocol group and file stem: `CH-8`, `F000`, `F100`, `<code>-<form>` for the six fused-BN codes, the arm id otherwise; engineering labels `F111-FB`, `PID-BEST-F111-FB`, `GR-BF`, ... |
| `arm_class` | str | training-code path, the unit of the smoke and of the measured rates r_m (owner decision B12-3): `released-ramp`, `released-pid`, `lab-hgq`, `binary-nonbn`, `binary-FB`, `binary-BF`, `pid-binary-FB`, `pid-binary-BF`, `fp32`, `w8a8`, `ds-hgq`, `gr-FB`, `gr-BF`, `be-FB`, `be-BF`; before S2 a template has `pid-binary`, `binary-best`, `gr` or `be` |
| `get_model_args` | list | `["xfm", 7, 7, 1e-08, 8, true]` (STUDY 2.1) |
| `template` | str or null | `BEST` (PID-BEST, BEST-C, PID-BEST-C) or `F111-chosen-form` (GR, BE): the model is fixed by S2. Before S2, `config_code`, `groups`, `form`, `form_source`, `best_source` and `wandb.run_name` are null and `s2_record` is in `unset_inputs`; after it, BEST templates take BEST's code, GR and BE F111, the form is the chosen form where a fused-BN site is binary (else null), and `form_source`/`best_source` are `s2_choice:<sha256>` (owner decision B12-2) |
| `s2_choice_sha256` | str or null | sha256 of the S2 record a template was filled from |
| `replica` | int or null | null for every identity with a single run; 1 and 2 for the determinism record's two CH-8 seed-1 runs. M15 puts `_r<replica>` in the run name and run id when it is set, only in stage `det` (section 8; owner decision B12-5) |
| `admission` | str | `required`, `control-slot` (section 10 admission after S2), `all-controls-cap475`, `s4-kai-approval`, `engineering` |
| `recipe` | object | the released non-λ recipe for every arm, PID arms included: batch and validation batch 2790, `steps_per_execution` 4, lr and its cosine schedule, Adam defaults, the loss, `dataset_shuffle`, `fit_shuffle`, `released_epochs` 7000, Chang's ParetoFront metrics, sides and filter, StopIf, and `source` (path, sha256 checked against `code-base.txt`) |
| `callbacks` | list or null | released order plus `lab:frozen_evaluation`; PID arms have `lab:BetaPID` at BetaScheduler's position (STUDY 7.2); null for FP32, whose list is M13's (B5) to fix (STUDY 7.1 says only that its cost callbacks and StopIf are off) |
| `data` | object | `view` (`train`, or `smoke` for the 2-file archive), `archive_name`, `archive_sha256`, `archive_rows`, `n_constituents`, `ptetaphi`, `features`, `split_seed`, `val_size`; for `train` also `n_train`, `n_val`, `order_sha256`, `split_sha256`, `stats_sha256` (from the B2 manifest; no held-out field) |
| `s4_family` | str | S4 only: `A`, `B` or `C`, the template family; launch needs `s4_option` equal to it |
| `generator` | object | `sources_sha256` (sha256 of the sorted `<sha256>  <path>` lines of `gen/*.py` and `gen/protocol_text.json`), `inputs_sha256`, `study_sha256`, `study_version`, `protocol_reference_sha256`, `release_sha256` |
| `launch_confirmations` | object | `od13`, `od15`, `od12`, copied from the inputs; each must be true at launch |
| `unset_inputs` | list | missing required inputs; a null headline is not required by CH-8/F000 ramp training |
| `launch_blocks` | list | reasons a config may not launch even with every input set (an S4 family other than the chosen option); empty in every config at B12 |
| `handoff_fields` | list | `code_sha`, `init_tensors.manifest_sha256`: supplied by the handoff, never by the generator |

`gen.configs.check_launchable(cfg, code_sha=, init_manifest_sha256=)` raises `UnsetInput` for any of
these and for missing confirmations, a required headline, malformed/unset claim domain, S4 option,
initial-tensor path, epochs or parameter count, recomputed from the fields. The runner, pack
runner and launcher call it. The sole headline exemption is CH-8/F000 ramp training; no count
is silently selected. The generated source hashes bind STUDY v0.13; the unchanged base
protocol reference stays at v0.8. A split-form H2 extension needs a later dated amendment.

**Conformance with the runner (section 13), after the merge of `46cc25fa9e48`.**
- `ramp` is `cbl.runner.RAMP_RECORD`, imported (cbl/runner.py imports only the standard library
  at import time); G57 checks it against the values `gen.release` reads from `run_train.py`.
- `wandb.mode` is `offline` in every config (B5-7: PREFLIGHT admits `offline` and `disabled`;
  B17 decides `online`).
- FP32, W8A8 and DS-HGQ carry `groups` all `Q` and `embedding` `Q` (section 5: required under the lab
  builder; no group is binary) and keep `config_code` null (section 5). `validate_config` derives the
  code from `groups` and would require `F000` for them: settled when the controls are built.
- **Launch copy.** `validate_config` requires `code_sha` (the fork HEAD, B5-8) and
  `init_tensors.manifest_sha256` in the config it runs. A committed config holds neither (B12-1, U1),
  so the handoff makes the launch copy with `gen.configs.with_handoff(cfg, code_sha=,
  init_manifest_sha256=)`, which changes only those two fields; `check_launchable` reads them from
  the copy or from its keywords. The handoff record keeps the copy beside the generated config's
  sha256.
- B18: claim domain is confirmed as b/default without C3r; headline remains null.
  `validate_config` and `check_launchable` permit CH-8/F000 ramp training without the headline.
  Missing claim domain is still refused by `check_launchable` in every launch stage; engineering
  validation alone is not launch authority (STUDY section 14 includes smoke, bench and det).
- Under a constructed complete record and handoff fields, `validate_config` accepts the 13 CH-8 and
  F000 configs (S1, smoke, bench, det) and refuses every other config only as not built yet: binary
  arms, GR, BE and the BEST templates (M6, B6), PID arms (M12, B9), FP32, W8A8, DS-HGQ (the controls'
  builder) and CH-8-C (the runner admits only CH-8 and F000 as arm ids). G57 shows this, and with
  every arm admitted in memory, that the binary, GR, BE and CH-8-C ramp configs meet every other rule.
- **After the merge of `b2838835b733` (B6),** G57 reads `cbl.runner.BUILT_ARMS` and
  `CHOSEN_FORM_ARMS`. Under the same constructed record 66 configs pass: all 45 S1 configs, and in
  smoke and bench CH-8, F000, F100, F111-FB, F111-BF, GR-FB, GR-BF, BE-FB, BE-BF (9 each), and the 3
  det configs. Refused, only as not built or not admitted: 38 PID, 15 controls, 21 CH-8-C, 21 BEST-C,
  and the 6 GR and BE configs of S3 (the runner does not read the S2 form, B6-7). With every input
  unset, the 14 binary smoke and bench configs are refused for their null headline (B5-4), as
  expected until the headline is set.
- **[B9] After the PID admission,** under the same constructed record 88 configs pass: the 66 above, PID-CH in S3 (3),
  PID-CH-C in S4 A and B (13), and PID-CH, PID-BEST-F111-FB and PID-BEST-F111-BF in smoke and bench (6). Refused, only as
  not built or not admitted: 15 controls, 21 CH-8-C, 21 BEST-C, and 22 S2-chosen configs admitted in engineering stages
  only (GR and BE in S3 6, PID-BEST in S3 3, PID-BEST-C in S4 13). Of the 38 PID configs 22 pass and 16 are S2-chosen.
  G57, G58, G60 and tests/test_gen.py state this admission.

**Tree** (`configs/`, 172 files at B12): `S1/` (45), `S3/` (21), `S4/A/` (20), `S4/B/` (32),
`S4/C/` (16), `smoke/` (15), `bench/` (15), `det/` (3), named `<label>-s<seed>.json`
(`-r<replica>` where `replica` is set, the CH-8 pair under `det/`); `protocol/screen.json`, `claim-A.json`, `claim-B.json`,
`claim-C.json`; `MANIFEST.json` (every config with its sha256, counts, generator and input hashes).
Configs are sorted-key JSON, indent 1, ASCII, one trailing newline.

**Derived protocol** (`gen.protocol`): arms, inputs, scope and the dataset identity hash are computed
from the configs, metric n from the data manifest, and the prose is carried from
`gen/protocol_text.json`, copied once from the protocol reference (sha256 `8f48e716…`). `screen`
has the reference's layout (with `confirmation_stage` = option A); with the identity hash set back to
the placeholder it is byte-identical to the reference. `claim-B` and `claim-C` carry option A's
prose, as the STUDY author's in-memory variants did; STUDY 8.7 has the claim-scope file rewritten
for the chosen option and claim domain before it is frozen.

**Not done here:** parameter counts and the build and reload of every config (G58), shared
initialization and callback order per arm in the runner (G60), the S1a/S1b staging of item 13.

## 15. Binary layers [B6] (M6 `cbl/binary.py`; M5 hooks, M13 admission)

**[B6]** Added in batch B6. Gates G16-G24, and the binary parts of G10, G11 and G15. STUDY v0.10
sections 2.2, 2.3, 3.1(a')-(b), 5 (GR, BE), 6 step 4, 13; plan decision U3.

**Classes** (registered for Keras serialization as `cbl1016>...`; imported only after `cbl.env.setup()`):
- `BinaryEinsumDense(QEinsumDense)`: non-BN sites (Q, K, V, O, the FFN contraction); one q_g per layer.
- `BinaryEinsumDenseBatchnorm(QEinsumDenseBatchnorm)`: fused-BN sites (FFN expansion, head layers, BE's
  embedding), `form` `FB` or `BF`; one q_g per output column; Chang's `call` (moments, statistics
  updates) runs unwrapped and unchanged, with `get_fused_qkernel_and_qbias` replaced.
- `BinaryMultiHeadAttention(QMultiHeadAttention)`: its build (the installed 0.1.9 build with
  `BinaryEinsumDense` in place of `QEinsumDense`, G22) creates the four binary projections.

Constructor arguments beyond Chang's: `penalty` (`H1` or `H2`, the config's `headline`, no default: a
binary layer refuses None), `gr` (bool, GR's derivative), `form` (fused-BN layers). `kq_conf` is the
configuration Chang's kernel quantizer would get at the site; q_g is built from it (the MHA passes its
`_qkvo_kq_conf` object). Attributes: `qg` (the scale quantizer), `form`, `penalty`, `gr`; `qkernel` and
`qbias` as Chang's (moving moments); `kq` raises AttributeError (no kernel quantizer is built, G18).
`get_config` holds Chang's keys plus `penalty` and `gr` (and `form`), with q_g's configuration under
`kq_conf`; a `.keras` save and `load_model` restore everything bytewise (G22).

**The scale treatment** (OD-13, adopted under Kai's delegation, not confirmed; STUDY section 19.2 item 2)
is the one class `cbl.binary.LearnedWidthScale`, used through `cbl.binary.SCALE`: `create`, `build`
(granularity), `bits`, `kernel_bits`, `magnitude`, `parts` and `kernel` (the three maps, GR included) and
`record`. A different scale treatment replaces that class only.

**Per-site record for B7 (section 7) and B8.** `SCALE.record(layer)` returns plain NumPy values at the
frozen state (moving moments; q_g called in inference mode, which never changes its state):
`per_column`, `form`, `gr`, `penalty`; q_g's `b`, `i`, `f` (rounded, scalar or per column) and `max`
(2^i - 2^-f); `g` (the scale's input: beta, beta_c or beta |s_c|), `q` (q_g(g)), `S` (+-1), `K` (the
exported kernel, as `qkernel`), `sign_s` (BF) and boolean `pruned` (b = 0), `rounded_to_zero` (b > 0,
q = 0) and `saturated` (g above max, where q_g's input derivative is 0). B7's `SiteInput` of a binary
site: `kind` `binary`; `K` = `record['K']` flattened as in section 7; `kernel_f` = `record['f']` and
`kernel_bits` = `record['b']`, broadcast over the input axis (per layer, or per column at fused-BN
sites); `b`, `I`, `F` from the layer's input quantizer as at HGQ sites; `rest_f32` the bias term as
`_compute_ebops` computes it (einsum_dense.py:73-76). G23 builds such records in gate code and runs
`cbl.cost_pure` on them.

**Training counter.** A binary layer's `_compute_ebops` returns the penalty (b) of its branch
(`cbl.binary.penalty_b`), so its `_ebops` counter, FreeEBOPs's sum and the loss term hold (b), not
native (STUDY section 3.1(a')). Under H1, (b) is the native rule; G19 shows it equal to Chang's own
contraction with kq.bits_ replaced by q_g's bits_. Under H2 it is the column rule of section 3.2 with
the scale's bits in place of w(q_g(g)); `cbl.binary.h2_kernel_term` and `u_bound` are its parts. B7's
independent recompute (G30, G33) reads quantizer variables only.

**Roles and initial tensors.** A binary arm keeps CH-8's 14 top-level names (pinned by its class
sequence, `cbl.model.class_sequence`) and 244 variables. At each binary site (`cbl.model.binary_sites`:
`mha/query`, `mha/key`, `mha/value`, `mha/attention_output`, `ffn1`, `ffn2`, `head1`-`head3`, `embed` in
BE) the unmatched set is `<site>/qg/FixedPointQuantizerKBI[0]/{k,b,i}` (constant initializers k = 1,
b = 4, i = 0) and the unused set `<site>/kq/FixedPointQuantizerKBI[0]/{k,b,i}`. STUDY section 6 step 4
names only b and i; q_g's uint8 k is the finding section 10 anticipated (G10).

**Config (M13).** `cbl.runner.BUILT_ARMS` admits CH-8, F000, F100, F001-F111 (form FB or BF where a
fused-BN site is binary), GR and BE; GR and BE only in engineering stages, because their form is the S2
choice and the runner does not yet read the S2 record (owner decision B6-7). A binary arm needs
`headline` H1 or H2 in every stage (it selects the penalty branch; `cbl.model.arm_spec` refuses null).
PID-BEST, BEST-C and PID-BEST-C stay refused (S2 choice; PID is B9). `cbl.runner.BUILT_IN_B5` is kept
unchanged for branch b12's G57, which reads it. `cbl.env.provenance()['env']` records `XLA_FLAGS`.

## 16. Counts and frozen evaluation [B7] (M7 `cbl/cost.py`, M8 `cbl/frozen_eval.py`)

**[B7]** Added in batch B7. Gates G26-G34, G36 and G12's frozen part. STUDY v0.10 sections 3.1-3.3, 3.5, 7.1, 7.2,
8.1, 12, 13 cost row, 14; plan rows M7, M8. Owner decisions B7-1 to B7-12 are in PREFLIGHT.md (B7 section).

**M7, `cbl.cost`** (imported after `cbl.env.setup()`):
- `layer_entries(model, top_names) -> [Entry]`: every `QLayerBase` with `enable_ebops`, walked from `model.layers`
  through `Layer._layers` with `cbl.rng.segment_name`; `Entry(name, role, group, kind, cls, layer, shapes)`. `name` is
  the role prefix joined by `.` (`embed`, `mha.query`, `mha.QSoftmax[0].exp_table`); `kind` is `hgq` (a `QEinsumDense`
  or subclass with a kernel quantizer), `binary` (M6's two kernel layers) or `native`; `shapes` are the shapes
  QLayerMeta's call wrapper passes to `_compute_ebops`, from Keras's recorded build shapes. Groups: section 2.2's
  kernel groups (`attention` = the four projections, `ffn`, `head`, `embedding`), every other layer `other`
  (owner decision B7-4). 20 entries for every configuration of the screen.
- `site_input(entry, record=None)`: section 7's `SiteInput`, canonical (P, n_in) and (n_in, n_out) (contracted axes
  trailing in the input and leading in the kernel, checked); `rest_f32` is the bias term evaluated by the backend with
  the layer's own expression.
- `frozen_counts(entries, *, binary_records=None, with_sites=False)`: `cost_pure.aggregate` plus, per layer, `native`
  (uint32 of the layer's own `_compute_ebops` at the frozen state; null at binary sites), `penalty_b` (binary sites:
  the same call, which returns penalty (b) there), `counter` (the uint32 counter of the last training-mode call) and
  `class`; per group the sum of `native` (null when the group has a binary site).
- `penalty_recompute(layer, shape=None)`: penalty (b) of a binary layer under both branches from quantizer variables
  only (q_g's rounded b, the input quantizer's rounded bits, I and F; `cost_pure.u_bound`), with `value_f32` and
  `counter` of each branch and of the layer's own (`layer.penalty`).
- `public_aggregate(model)` (the stock BetaPID sensor) and `private_total(model)` (the stock FreeEBOPs sum), each by
  calling the installed hgq code; `native_value(layer, shapes)`, `rest_f32(layer)`, `contraction(layer, shape)`.

**M8, `cbl.frozen_eval`:**
- `FrozenEvaluator(model, *, roles, top_names, x_val, y_val, batch_size=2790)`; `evaluate(epoch=None) -> core`:
  validation top-1 by one jitted `stateless_call(..., training=False)` over fixed 2790-row chunks of the validation
  Dataset's tensors (owner decision B7-1), the counts, the binary-site record, the variable hash before and after
  (`FrozenStateChanged` on any difference). With an epoch, a second call in that epoch returns the cached core after
  rechecking the hash; the cache holds the last two epochs. `headline_value(epoch, headline)` is the PID's sensor
  (B9 calls it in its `on_epoch_end`; the appended callback then reuses the same core).
- core keys: `epoch`, `val_correct`, `val_n`, `val_top1`, `H1`, `H2`, `NR`, `snap_changed`, `subtotals`, `binary_sites`
  (per site: `per_column`, `form`, `penalty`, `gr`, `qg_b`, `qg_i`, `qg_value`, `zero_cols/{pruned,rounded_to_zero}`,
  `saturated_cols`, `sign_occupancy`, `sign_flip_rate` (null at a process's first evaluation), `single_sign_cols`),
  `vars_sha256_before`, `vars_sha256_after`, `val_batch_size`, `n_val_batches`, `frozen_seconds`.
- `make_hook(recorder, *, evaluator, probe, guard, wandb_run, epochs_path, run_id)`: the class `FrozenEvalHook`
  appended after StopIf. Keras log keys it adds: the two batch-order hashes and `LOG_KEYS` = `H1`, `H2`, `NR`,
  `val_correct`, `val_n`, `val_top1` (owner decision B7-7). The `epochs.jsonl` row is section 1's record (schema 1)
  with `logs_hex` and `sys/epoch_seconds`, `sys/frozen_seconds`; `FrozenEvalHook.records` keeps a compact row per
  epoch (`COMPACT_KEYS`) for the runner's `run.json`. W&B through M15 with `strict=False` (B5-6 kept).
- Null until later batches (`DEFERRED`): `quantizer_structure`, `pruning_channel` (B8, M10, with the data-pass
  structure record), `train_frozen_top1_gap` and the data-pass binary-site fields (`fb_sign_diff_frac`,
  `dead_channel_frac`, `moment_mismatch_*`; B8), `pid` (B9). M15's `quantizers/*`, `bn/*` and `activations/*` keys are
  not written by B7 (owner decision B7-5).
- `checkpoint_audit(ckpt_path, *, role_spec, x_val, y_val, batch_size=2790, expect=None)`: reload and evaluate; with
  `expect` (an epoch's record) `mismatches` lists the `AUDIT_KEYS` that differ (exact; the binary-site record without
  `sign_flip_rate`). `export_audit(model, *, role_spec, trace, x_val, y_val, batch_size=2790, exported=None)`: the
  evaluation before and after `trace(model)` (Chang's trace, supplied by M18 in B14), the fields the trace changed,
  and with `exported(model) -> (sites, native_layers)` the counts recomputed from exported constants and formats
  against the post-trace evaluation, per layer.

**For B8 and B9.** B8 adds the structure blocks (`quantizer_structure`, `pruning_channel`) and the data-pass fields
by extending `FrozenEvaluator.evaluate`'s core or the hook's row, and owns the checkpoint sidecar (M9), which must
copy the epoch record's counts so that `checkpoint_audit` can be checked against it. B9's PID calls
`ctx.evaluator.headline_value(epoch, cfg['headline'])` in its own `on_epoch_end` and writes the `pid` block; the
evaluator's sign-flip state and cache are per process, which M14 (B10) must know for resume (the first record
after a resume has `sign_flip_rate` null).

## 17. Front saver, structure record and ablations [B8] (M9 `cbl/front.py`, M10 `cbl/structure.py`, M11 `cbl/ablate.py`)

**[B8]** Added in batch B8 (branch `b8`). Gates G37 (MAC, PCC), G38 (PCC, PCG), G39-G42 (PCC). STUDY v0.10
sections 2.2, 8.1, 8.8, 12, 13 (structure-record and ablation row); arbiter v0.5 section 6 items 4, 12 and
13; plan decision U4. Owner decisions B8-1 to B8-17 are in `preflight/gates/G37/<sha12>/PREFLIGHT-rows.md`.
Numbered 17 because B7 may add a section 16 first; renumber at the merge if needed.

**What M9 reads.** The section 1 record of each epoch: `epoch`, `val_correct`, `val_n`, `H1`, `H2`, `NR` are
required and must be integers (a float or a bool is refused); `run_id`, `step`, `train_count`, `subtotals`,
`pruning_channel`, `quantizer_structure`, `binary_sites`, `log_*`, `lr`, `beta`, `pid` are copied into the
sidecar's `record` when present. `front.chain_evaluator` takes the record M8's evaluator returns (a dict) or,
if it returns None, these keys unprefixed from the epoch `logs`. Top-1 is val_correct / val_n.

**Rules** (M16's, from `cbl.select`): Chang's filter per count in that count's units; membership: an epoch
joins unless a member has top-1 >= and cost <= (an equal pair keeps the earlier epoch), and members it weakly
dominates leave, so members have distinct top-1 and cost and the hash tie-break of STUDY 8.1 never fires
within a run. A checkpoint is saved when it joins any front or takes a diagnostic role (`best_at_350000`,
`min_cost`, per count).

**Files** under `<run dir>/front/` (paths in records are relative to the run dir):

| path | content |
| --- | --- |
| `ckpt/epoch=<e>.keras` | `model.save` of the epoch-end state; deleted when it leaves every front and diagnostic role if `keep_superseded` is false |
| `sidecars/epoch=<e>.json` | the sidecar (below), written once |
| `attention/epoch=<e>.json` | the attention diagnostics, written once: in the loop (`every_save`) or by the run-end checkpoint audit (`run_end`) |
| `events.jsonl` | one line per epoch: the record's integers, `joined`, `left` (count -> epochs), `diagnostic`, `saved`, `deleted`, hashes; `front.replay(front.recs_from_events(...))` rebuilds the state |
| `summary.json` | the run summary, written by `finalize` |

**Sidecar** (schema `cbl1016-front-sidecar/1`): `run_id`, `epoch`, `ckpt_path`, `ckpt_sha256` (the file's
bytes; Keras writes a save time, so one state saved twice hashes differently), `ckpt_bytes`, `state_sha256`
(`cbl.rng.variables_sha256` over the roles: the deterministic identity resume compares), `val_correct`,
`val_n`, `val_top1`, `H1`, `H2`, `NR`, `joined`, `diagnostic`, `chang_filter` {count: bool}, `val_auc_macro`,
`val_auc_per_class` (softmax in float64, exact rank AUC), `val_logits_sha256`, `m11_correct`, `m11_n`,
`m11_equals_record` (M11's own count against the record's; not assumed equal), `structure_record`,
`attention_diagnostics_file`, `attention_mode`, `seconds`, `record`. `front.load_sidecars(front_dir)` returns
{ckpt_sha256: sidecar} with `attention_diagnostics` merged from `attention/` (None where absent): the shape
section 11 says M16 reads.

**structure_record** (schema `cbl1016-structure-record/1`; `cbl.structure.record_from`): `n`, `batch`,
`inference_mode`, `degenerate` {Q, K, V, O, FFN: route or null}, route one of `kernel_pruned`,
`kernel_rounded_to_zero`, `input_zero_width`, `output_zero_width`, `other` (the first that holds, in that
order); `routes` {layer: every named route that holds from quantizer state}; `constant`, `constant_zero`,
`max_spread` per layer; `q_identical_across_t` (list per head) and `q_all_zero` (bool), null unless Q is
degenerate; `zero_columns` {site: {kind, n_columns, pruned: [column], rounded_to_zero: [column]}} at the 11
constant-kernel sites `embed`, `Q`, `K`, `V`, `O`, `ffn1`, `FFN`, `head1`-`head3`, `logits` (columns flattened
over the kernel's output axes); `input_zero_width_fraction` and `input_zero_bits_fraction` {layer: fraction};
`softmax_zero_width_fraction` and `softmax_zero_bits_fraction` [per head]; `layers` (per layer kind, kernel,
iq, oq detail); `anomalies` (a route holding while the output varies). `cbl.structure.quantizer_state(model)`
is the same without the data pass, for the per-epoch `quantizer_structure` of section 1 (M8 may call it).

**attention_diagnostics** (schema `cbl1016-attention-diagnostics/1`; `cbl.ablate.diagnostics_from`):
`delta_attn_mean`, `delta_attn_zero`, `delta_static`, `delta_unif` (float64, baseline minus ablated top-1) with
`delta_correct` (integers), `baseline` {correct, n, top1, logits_sha256}, `ablated` {key: {correct, n, top1}},
`baseline_equals_record`; `H_attn` (null if no row is usable), `H_attn_rows` {rows, used, all_zero_rows,
negative_rows, nan_rows}, `all_zero_rows`, `negative_rows`, `log8`; `overflow` {delta key: count at the
STUDY-named quantizer}, `overflow_quantizer` (which), `overflow_downstream` {delta key: {quantizer: counts}},
`overflow_baseline` {quantizer: counts}, counts being {overflow, range_check, masked, live, entries, nonfinite}
at `O.iq`, `add1.iq[1]`, `tanh2.iq`; `branch_norm`, `means_sha256`, `widths`, `qk_av_native`,
`mha_has_output_quantizer`, `n`, `batch`, `seconds`, and `source` (`in loop` or `checkpoint audit`).

**Run summary** beyond section 3: `schema` `cbl1016-run-summary/1`, `epochs_offered`, `headline`,
`filter_outcome` values are objects {outcome, diagnostic, label} (diagnostic: {epoch, val_correct, val_n,
val_top1, cost} of the best epoch at cost <= g ignoring the filter); `diagnostic_ckpts` {count: {best_at_350000,
min_cost}} as front points with `val_top1`; `pid_selection` (PID runs) {count, status, ckpt_sha256, point,
cost, slack}; `checkpoints` {epoch: {path, sha256, state_sha256, bytes, deleted_at}}, `live_epochs`,
`attention_mode`, `attention_audited`, `keep_superseded`, `m16_reader` ({ok} from `cbl.select.run_from_records`
and, for PID runs, `check_pid_summary`), `finalize_seconds`.

**Execution (M10, M11).** `cbl.ablate.Exec(Graph(model), jit=True)` runs the U4 re-expression over chunks of
2790 validation jets (NumPy float32 arrays, never a Keras Dataset), compiled once per pass kind with every
model variable an argument inside `keras.StatelessScope`; `jit=False` is the eager reference. G40 shows both
bytewise equal to `model(x, training=False)` on the gated arms. The reader returns X_val as float16
(chang/data.py:33; `cbl.data.read_train_val`); `front.register` widens it to float32, which is exact and is the
value the model's float32 input conversion gives the layers; `front.bind` and `Exec` take float32 only.
`Graph` accepts only the 14-layer xfm graph with WRAP datalane quantizers at O's input, QAdd's branch input and
the second tanh table's input: CH-8, F000, the binary configurations, GR and BE. DS-HGQ (another graph) and FP32
(quantizers off) are refused, so the controls batch needs a `Graph` route for them or a saver mode without the
structure pass and the ablations; W8A8 keeps the graph (its datalane quantizer type is decision U6's).

**[B9] Wired in B9** (owner decisions B9-1 to B9-3, PREFLIGHT.md): `Context.front` and `Context.frozen_step`; in
`prepare`, after the gate hook and before the appended callback, `front.register(ctx, ablations=FRONT_ABLATIONS,
keep_superseded=FRONT_KEEP_SUPERSEDED)` ('run_end', B8-17; false, B8-3) and the hook's read-only step
`after_evaluation(front.chain_evaluator(section1, ctx.front), ctx.probe)`, where `section1(epoch, logs)` is
`cbl.frozen_eval.section1_record` (the section 1 record from `FrozenEvaluator.cached(epoch)`, the evaluation already
made that epoch, and the logs; it evaluates nothing). Item 2 below, as written for the B5 runner, would hand the saver a
record without H1 (the integers reach the logs only after that step). `finish` finalizes as in item 3; a failure is
recorded as `front_error` and exit 0 becomes 1. `run.json` gains `front` (mode) and `front_summary`.

**Runner wiring** (for the orchestrator after both branches merge; not done on `b8`). In `cbl/runner.py`:
1. `Context.__init__`: `self.front = None`.
2. `prepare`, after `evaluator = evaluator_factory(ctx) if ... else None` and before `hook = frozen_eval_hook(...)`:

       from cbl import front
       ctx.front = front.register(ctx, ablations=<'run_end' or 'every_save'>, keep_superseded=False)
       evaluator = front.chain_evaluator(evaluator, ctx.front)

   The saver then runs inside the frozen-evaluation step, inside `guard.watch('frozen evaluation')`, and the
   callback order stays the release plus `FrozenEvalHook` (G13, G47, G60 unchanged). It writes no log key.
   M8's evaluator must leave the section 1 integers in `logs` or return the record.
3. `finish(ctx, code)`, before `run.json` is written, when `ctx.front` is set and `rec['end']` is one of
   `completed`, `stopif`, `integrity_stop`, `failed`:

       rec['front_summary'] = ctx.front.finalize(end=rec['end'], epochs_completed=rec.get('epochs_completed', 0),
                                                 released_front=front.released_front_of(ctx.released[4], ctx.out_dir))

   (`ctx.released[4]` is Chang's ParetoFront, run_train.py:95 order.)
Alternative wiring: `front.keras_callback(saver)` as an eighth callback after `FrozenEvalHook`; it changes the
callback order that `prepare` refuses and G13, G47 and G60 check, so those three need editing with it.
Resume (M14, B10): `FrontSaver.snapshot()` and `restore(snap)` carry the state; the saver's `records` of saved
epochs are needed by the run-end audit only for `baseline_equals_record`.

**Closed from section 6.** Container format: JSON files as above. The pruning-channel record lives in the
section 1 epoch record (M8); M9 copies it into every sidecar's `record`, which covers "at every grid
checkpoint" because grid checkpoints are saved checkpoints.

## 18. Lab PID, its record and the runner's PID schedule [B9] (M12 `cbl/pid.py`; M13 `cbl/runner.py`; M8 hooks)

**[B9]** Added in batch B9. Gates G43-G47 (PCC; G46 is the controller's part, the whole-run resume is G48 with M14 in
B10). STUDY v0.10 sections 7.1, 7.2 (whole), 3.3, 12, 13 (PID row); plan row M12. Owner decisions B9-n are in
PREFLIGHT.md (B9 section). OD-12 (the target 350,000 in headline units) is built as adopted under Kai's delegation and
is not confirmed (STUDY 19.2 item 4).

**Class.** `cbl.pid.make_pid(record, headline=)` returns a `LabBetaPID` (`cbl.pid.pid_class()`, built after
`cbl.env.setup()`), a subclass of hgq2 0.1.9's `hgq.utils.sugar.beta_pid.BetaPID` whose law is the stock one
(`on_epoch_begin` through `super()`); its PID object is the stock `PID` subclassed to record its last output.
`bind(sensor=, guard=)` attaches the sensor (`cbl.pid.sensor_reader(evaluator, headline)`, the epoch's cached frozen
headline count) and the stream guard; `on_train_begin` refuses without a sensor. `signature()` is what the runner
records (`run.json` `pid`, and `callback_signature['LabBetaPID']`).

**Units in one place (OD-12).** `cbl.pid.SENSOR = 'headline'` and `sensor_reader`; the config key `target_H`. A native
target would change `sensor_reader` and the key (gen/configs.py `PID`, `cbl.pid.PID_RECORD`) and restate G43.

**Timing.** `on_epoch_end(e)`: `logs['beta']` as stock (PBar reads it), then frozen H(e) by the sensor inside
`guard.watch('PID frozen-H sensor')` (the evaluator computes and caches the epoch's evaluation there; the appended
callback reuses it), then the stock aggregate (`get_ebops`), and H(e) is stored as the reading the stock
`on_epoch_begin(e + 1)` acts on. `logs['ebops']` is never written.

**Log keys and the record block.** Scalars written into the epoch logs, `pid/<field>`: `sensed_H` (int, the own key of
STUDY 7.2), `stock_aggregate` (float, the stock sensor, logged beside), `beta` (the lambda in force during the epoch),
`integral` (after the epoch's update), `clamp_state` (`warmup`, `none`, `max`, `min`), `slack` (target_H - sensed_H,
int), and once acting (epoch >= warmup) `error` (log10 of the reading acted on over the target), `acted_on_H` (the
reading acted on at this epoch's begin, sensed_H of the previous epoch) and `beta_unclamped`. A field without a value is
left out, never null (the logs pass through `float.hex` in M8's audit of released keys). M8's `epoch_row` copies every
`pid/` key, without the prefix, into the record's `pid` object (`cbl.frozen_eval.pid_block`); M9 copies the record
into each sidecar; W&B gets `pid/<field>` through M15's flattening (strict mode, B10, must allow the warmup epochs'
missing `error`).

**Persisted state** (for M14, B10): `LabBetaPID.state()` -> `{kind: cbl1016-pid-state, schema: 1, record, headline,
sensor, epoch (the last ended), integral, prev_error, beta, sensed_H, stock_at_train_begin, hex: {integral,
prev_error, beta, stock_at_train_begin}}`; `restore(state)` into a fresh controller (refuses another record,
headline or sensor, or a hex mismatch); the restored controller's `on_train_begin` re-applies the restored beta to every
layer and does not reinitialize, and its next `on_epoch_begin` must be `epoch + 1`. The evaluator's cache and the PID's
last two blocks are per process; M14 stores the state, not the blocks.

**Runner.** `validate_config`: `schedule` pid exactly for `PID_ARMS` (PID-CH, PID-BEST, PID-CH-C, PID-BEST-C); a PID
config has `pid` equal to `cbl.pid.PID_RECORD`, no `ramp`, a `recipe` whose released non-lambda values equal
`cbl.runner.PID_RECIPE`, and a headline (H1 or H2) in every stage. PID-BEST and PID-BEST-C are in `CHOSEN_FORM_ARMS`
(engineering stages only until the runner reads the S2 record, B6-7). `released_callbacks(..., schedule='pid')` puts the
PID at BetaScheduler's index; `prepare` binds it after the evaluator exists, wraps it as lab code, and checks the order
`PID_ORDER` + `FrozenEvalHook`. Released-builder arms other than CH-8 (PID-CH; CH-8-C and PID-CH-C) load CH-8's file
under `dataclasses.replace(CH8_ROLE_SPEC, arm_id=<arm>)`. `Context.pid` is the PID callback. Admission (owner decision
B9-10): PID-CH and PID-CH-C were admitted in engineering stages only (`PENDING_GATES_ARMS`) until G43-G47 passed, and in
every stage since; PID-BEST and PID-BEST-C stay engineering-only with GR and BE; CH-8-C is not admitted yet. **[B10]**
CH-8-C is admitted in every stage (owner decision B10-9; 109 configs pass G57's constructed record). The PID state is
stored in M14's checkpoints (section 19).

## 19. Resume, the launch check, the held-out guard and W&B strict mode [B10] (M14 `cbl/resume.py`; M13, M8, M3, M15)

**[B10]** Added in batch B10. Gates G46 (whole-run part), G48, G49, G50 (PC part), G51 and G57's PC part. STUDY v0.10
sections 6 step 8, 7.2, 12, 13 (resume row and "further" rows); plan rows M14, G46, G48-G51. Owner decisions B10-n are in
PREFLIGHT.md (B10 section). Nothing earlier is renamed or retyped; section 1's `pid` note (strict mode must allow the
warmup epochs' missing `error`) and section 16's note (the first record after a resume has `sign_flip_rate` null) are
settled here.

**Checkpoints (every runner run).** `<run dir>/resume/` holds `ckpt-<epochs done, 5 digits>/` (the two newest),
`latest.json` (`checkpoint`, `epochs_done`, `state_sha256` of the checkpoint's `state.json`, `utc`), `resumes/<n>.json`
(one per resume: source checkpoint, epochs done, code sha, host, the rewind record), `discarded/<n>/` (files a resume set
aside, with their relative paths) and `attempts/` (the run record of a resume refused or failed before `fit`). A checkpoint
is written at train begin in a fresh run (`epochs_done` 0, after Keras's pre-fit draws) and at the end of every epoch e
with (e + 1) % `resume_every` == 0, at the last epoch and when StopIf stopped the run, as the last step of the appended
callback's `on_epoch_end`. Contents: `variables.npz` (every variable by role, `cbl.rng.write_tensor_file`),
`optimizer.npz` (optimizer variables by index, paths in `state.json`), `prev_signs.npz` (the evaluator's per-site signs,
binary arms), `links/` (hard links, copies where refused, of the live front checkpoint files and ParetoFront files) and
`state.json` (kind `cbl1016-resume-state`, schema 1, compact sorted JSON):

| key | content |
| --- | --- |
| `epochs_done`, `next_epoch`, `every`, `written_at` | epochs completed; the next epoch; the interval; `train_begin` or `epoch_end` |
| `code_sha`, `config_sha256`, `run` | the fork commit, the canonical config hash, W&B `{run_name, run_id, mode}` |
| `identity` | `cbl.rng.SeedFlow.run_identity()` (seed, device, stack, matmul precision, draw code, data, Datasets) |
| `variables` | `{file, roles, state_sha256}`; `state_sha256` equals the epoch record's `vars_sha256_after` |
| `optimizer` | `{file, paths, sha256, iterations}` |
| `random`, `numpy` | the module-level `random` state and NumPy's legacy state as JSON, with their hashes |
| `index_map` | the training Dataset's `_index_map` and its hash |
| `recorder_state_after_epoch` | M4's hash of the stream after the epoch's map reshuffle (checked equal to the state saved) |
| `stopif`, `pareto`, `pareto_sha256` | StopIf's two counters; ParetoFront's file names and records (float.hex, dtype, shape) |
| `pid` | `LabBetaPID.state()` (section 18) or null (ramp runs, and `epochs_done` 0) |
| `front` | `FrontSaver.snapshot()` (section 17) |
| `prev_signs` | `{file, sites}` or null |
| `stop_training` | whether StopIf stopped the run at that epoch |
| `append_only` | `{epochs.jsonl, front/events.jsonl}`: `{bytes, sha256, exists}` of each file at the checkpoint |
| `front_files`, `pareto_files`, `live` | files under `front/` (without `events.jsonl`) and `pareto/`; the live ones by path |
| `files`, `links` | sha256 of every checkpoint file and of every link |
| `epoch_record` | `{epoch, vars_sha256_after, step}` of the epoch the checkpoint follows |

**Resume (`cbl/runner.py run --resume`, `prepare(..., resume=True, resume_every=N)`).** Before anything is built:
`latest.json` and every file it names hash-checked; refused (exit 2, nothing changed) for a finished run (`run.json`
exists), another code sha (checkpoint, config or fork HEAD), another config hash, another interval, another W&B run name
(source bundle, attempt). After `prepare`'s usual steps: refused for another run identity or role set; then the rewind
(append-only files cut back after their prefix is checked, later front and ParetoFront files and `front/summary.json` set
aside, live files the killed process deleted linked back, temporary and newer checkpoint directories set aside), the PID
restored into its fresh controller, the front saver restored, and the appended callback's compact records rebuilt from
`epochs.jsonl`. `fit` reseeds (the flow's record) without arming the guard and fits from `epochs_done`
(`initial_epoch`); the appended callback's `on_train_begin` restores the model by role, the optimizer, StopIf,
ParetoFront, the evaluator's signs, then the module-level stream, NumPy's state and the map in place, checks each hash and
arms the guard. A model is never reloaded from `.keras` for a resume, so finding B7-F1 does not arise.

For B17 (pack runner and launcher): a restarted pod reruns the same command with `--resume` and the same
`--resume-every`; a directory with no checkpoint (killed before train begin) is refused, and the pack runner decides. The
run identity includes the device kind and the stack, so a resume on another GPU product is refused (STUDY section 6: one
product and stack per seed). Offline W&B: the resumed process writes a second offline directory with the same run id;
rows of epochs the killed process logged after its last checkpoint are logged again (W&B is an index only).

**Launch check (M13).** `cbl.runner.check_launch(cfg, summary)`: in every stage but `gate`, `gen.configs.check_launchable`
on the config given to the runner, which must be the launch copy (`gen.configs.with_handoff`); a refusal is exit 2 before
the environment is set up. `run.json` `launch_check` is `{checked: true, passed: true}` or
`{checked: false, reason}` for gate copies.

**Held-out guard (M3, M13).** `cbl.data.install_heldout_guard(manifest)` (called by `prepare` before any data read):
`HeldOutGuard` refuses (`HeldOutAccessRefused`, a `HeldOutSealed`) every open, through the `open` audit event or h5py's
`make_fid`, of a path whose name or resolved name is the held-out archive's or a raw held-out file's (the manifest's
`heldout` block), of a path under the raw held-out directory of record, or of a file with the inode of one of those files
on this machine. `run.json` `heldout_guard`: `{installed, names, dirs, known_inodes, h5py_patched, refused}`.

**W&B strict level (M15).** The runner opens its run with `strict="produced"` (`cbl.runner.WANDB_STRICT`): every
catalogue key required for the run's traits, except `cbl.wandb_log.NOT_PRODUCED` (27 patterns no module produces yet,
each with the module that would), must be in every row; `ROW_EXEMPT` allows a missing `pid/error` while
`pid/clamp_state` is `warmup` and a missing `binary_sites/*/sign_flip_rate` at the run's epoch 0. `strict=True` (the
whole catalogue) is unchanged.

**Config generator input (M17).** `gen/inputs.json` `param_counts` is `manifests/param-counts-g58.json`, G58's record at
`07c33ba5fcf0` copied byte for byte (sha256 `b58235ed…960e`); 109 configs carry `param_count`, the S2 templates and the
controls keep it null and unset.

## 20. Launch packs and harness adapter [B17] (M20 `nrp/`, M21 `tools/`)

`nrp.spec.resolve` reads generated configs from exact bundle bytes, calls `with_handoff`,
`check_launchable` and `runner.validate_config`, and records the generated sha256 beside each
launch copy. No scientific field is filled by the launcher. `nrp.jobs.prepare` delegates bundle
preparation and validation to lab-root `tools/run_handoff.py`; it does not submit. Pack spec and
execution layout are in `nrp/README.md`. Every training arm runs in a separate process.

The M15 source token is the git bundle sha256. Job retries change only the Job/handoff attempt;
config attempt, source bundle, run name and run directory are unchanged. M14 resumes existing
directories with the original checkpoint interval; a terminal failed run or a directory without
a checkpoint is refused. Completed siblings require matching config, code, name and front summary.
`runner.prepare(..., pack_k=1)` and CLI `--pack-k` add actual packing to W&B provenance only;
the default, model, callbacks, optimizer and resume identity are unchanged.

Draws use `runner.draw` with one visible GPU and separate product/archive roots on kai-data.
Partial draws are never overwritten. Durable logs and per-arm telemetry live on kai-data under
`/data/cbl1016/`. GPU seconds mean allocated wall seconds shared among live processes; benchmark
rates include measured pod startup/shutdown. Memory peaks are explicitly sampled lower bounds.

The campaign adapter wraps the unchanged lab harness submit path, including tick's retries,
with authority-bound run admission, separate 350/50 training/reserve buckets and G80. Protected
command argv remain in the authority block under the existing harness schema. Approval is not
created by B17. Production certification and unknown scientific inputs remain blocking.

Infrastructure relaunches have no count ceiling (Kai, campaign notes/build-todo.md, B17 correction).
Eviction/preemption, node, image/mount and scheduling/pending failures qualify; a Job deadline
qualifies only with evidence that its pods remained pending without training starting. Code or
integrity exits/refusals, hash/provenance mismatches and gate failures stop the lineage and retain
evidence for a new committed fix and handoff (STUDY §11). Every new submission still passes the
350/50 GPU-h buckets and G80; M14 resumes the latest checkpoint and refuses invalid resume state.
BRIEF authority `scientific_settings.b17_retry_policy` holds `infra_relaunches_max: null` and
`infra_relaunch_notify_after_24h: 6`. This schema-compatible protected map is operational only.
Legacy retry fields are zero; the adapter supplies unlimited decisions without changing the lab
harness. More than six successful relaunch submissions of one pack in 24 h notify Kai through
`NOTIFY.log` while continuing; distinct Job timestamps persist in STATE, excluding deferrals.

## 22. Loss-id guard and loss-term census [B10-F1] (`cbl/lossguard.py`; M13 `cbl/runner.py`)

Numbered 21 on branch b10f1; renumbered 22 at the merge of main (orchestrator, 2026-10-19), because main gained B18P's section 21 first. References to "section 21" in `cbl/lossguard.py` and the B10-F1 ticket mean this section.

Ticket `preflight/investigations/B10-F1.md` (campaign directory). keras 3.14.0 keeps the Python ids of the losses a
layer added inside a stateless call in `Layer._loss_ids` across traces, and a later layer's `_clear_losses()` deletes
any live loss whose id it still holds; after CPython reuses a dead tracer's address, a traced training step can lack
one layer's beta * EBOPs term. Only a second or later training-mode trace in a process is exposed (another batch
shape, a retrace, a second fit).

- `cbl.lossguard.guard(model)`: called by `cbl.runner.prepare` right after the model is built (fresh and resumed
  runs alike). Wraps the instance's `stateless_call`: every nested layer's `_loss_ids` is emptied before each call;
  a call with `return_losses=True, training=True` must return `expected_terms(model, True)` terms (one per layer
  holding a beta variable plus one per regularized trainable weight), else `LossTermsMismatch` (the run ends
  `failed`, exit 1). Idempotent.
- Run record: `rec['lossguard']` = `{module, finding, census, expected_training_terms}` (equal across processes of
  one config); `rec['lossguard_counters']` = `{calls, cleared_ids, checked}` at finish (process-dependent: not for
  comparisons between an uninterrupted and a resumed run).
- Not covered: models that gate children build themselves (`gates/binary_child.py`, `cost_child.py`,
  `runner_child.py` Probe and reference children).
- Stack: STUDY section 7.3 does not yet declare this run-time change of Keras's bookkeeping (owner decision
  B10F1-1 in the ticket, for Kai).

## 21. B18P: epoch producers and restart metadata

M8's `cbl.epoch_diagnostics.EpochDiagnostics` supplies per-quantizer widths/steps,
BN gamma and moving moments, validation activation output ranges/zero-width masks,
and binary-site population moment mismatches and dead-channel fractions. All
forward observations use explicit Keras stateless variable inputs. M13 enriches
both the record offered to M9 and the row written to `epochs.jsonl`/M15, caching
one diagnostic result per epoch. M7/M8's frozen-count core remains inference-only.
The additional train-mode validation pass discards all returned updates; its
accuracy minus frozen accuracy is `train_frozen_top1_gap`. Validation uses the
same ordered 2790-row chunks, including the short tail. M13 checks all model
variables and the RNG guard around diagnostics.

FB runs delegate the original JAX `train_step` unchanged, then replay that batch
from its **pre-update** variable arrays in an isolated stateless scope. The extra
scalar `b18_fb/<site>` logs compare the batch-moment signs with moving-moment signs
at that step (moving variance after that batch's EMA update, before the optimizer
update). Keras's final batch logs, including the last step of a short execution
group, supply `binary_sites/<site>/fb_sign_diff_frac`. These diagnostic logs add
no model, optimizer or metric variables, and need no new M14 checkpoint state.
This is additional forward work; PC exact-state tests and B19 timing remain due.

M10 projects `quantizer_state` to the catalogue's per-layer route keys and
per-head softmax fractions without a data pass. `pruning_channels` records the
kernel zero-width fraction at every kernel site, and binary input zero-width
fractions. M9 supplies cumulative `dwell` from its existing `FrontState` epochs,
including the current epoch exactly once. M14 already persists/restores those
epochs; no separate dwell counter is introduced. M9 sidecars copy these fields.
M11 checkpoint ablations and M12 PID definitions/timing are unchanged.

**One unresolved pattern:** `pruning_channel/*/native_share_removed`. STUDY v0.12
§12 and §8.8 do not define the counterfactual widths or baseline checkpoint for
this cost share. Zero-width coefficients have zero current native cost, which
is not a measurement of savings. M15 `NOT_PRODUCED` now contains only this
pattern; M13 retains `strict="produced"`. No zero or proxy is substituted. The
B10 warmup-error and first-epoch sign-flip exemptions remain as stated in §19.

**M20 launch identity.** Fresh pack runs publish `launch.json` atomically with
the run directory, before backend initialization: schema, run name, config,
code sha, source bundle, GPU product label and `resume_every`. The supervisor
verifies the visible product before launching children. The run lock covers
all restart preparation. Existing checkpoints are hash-checked and cross-checked
against this file; legacy B10 checkpoints can supply product/interval themselves.
A checkpoint-free run with launch identity is moved, byte-for-byte, to a sibling
`<run>.pretrain-<UTC date/time>-<unique suffix>`, then started fresh. The original
interval is passed on every launch, including this fresh restart. A corrupt
checkpoint or terminal run is never treated as a pre-train kill. Legacy directories
with neither checkpoint nor original launch metadata require human recovery;
the original interval cannot be inferred safely.

The Job generator reads these run records before preparing a relaunch. Optional
spec `run_state_root` maps `/data/cbl1016` to a complete locally captured PVC tree;
otherwise it reads that mounted root. A retry without either is refused. Every
recorded product must agree with the generated config's selected product and
with its pack siblings; the required `nvidia.com/gpu.product` affinity uses this
checked product. No remote fetch or generated-config rewrite is performed.

**Allocation.** M20 writes a live snapshot per child outside the run directory,
and durable per-invocation snapshots under `allocations/` once it exists. M13
logs cumulative `sys/alloc_hours` from those snapshots, extrapolating the current
live-process share using the same host monotonic clock. Prior invocation seconds
are never rewound with epochs. These are supervisor allocation hours, excluding
pod init and unreconciled lifecycle tails; final all-inclusive pod billing is
still M21's lifecycle reconciliation. CPU gates log zero GPU allocation hours;
an exclusive standalone GPU run reports elapsed process allocation. Packed GPU
runs require the M20 snapshot. Missing measurements are errors, never fabricated
zeroes. Full SIGKILL recovery can lose the last polling interval, which remains
a limitation until lifecycle reconciliation; these rows are not budget authority.
