# PREFLIGHT: chang-bitlinear-20261016

Status (updated 2026-10-20): B0, B1, B2, B4–B7, B9 and B10, the lane-M batches B3a, B3b,
B11, B12 and B15m, the G52 fix and B8 merge, and B17, B18P and B18 are recorded as done.
B18's pass at `29a459ce7ee6` completed with failures; its regenerated record is
`preflight/b18/final/GATES.md`. The B18f pass at `38f4ebae3874` is complete with four failing test-suite rows;
its targeted rerun at final commit `c4dc2a80ca2b8726492f9c49d5de96b3a068f128` leaves only
the lossguard stand-in test failing in tests-b1 pcc: cause found (a defect in the test's own
bookkeeping, not in the guard), fix committed on branch b18fix as `8626e4b5f4e4 (first committed as ee30820d2ffa; message amended 2026-10-20 18:00Z to withdraw an unsourced sentence; same tree and parent)`, not yet on main
(`preflight/b18f/{final,targeted}/GATES.md`; `notes/LEDGER.md`, 17:24Z).
No launch certification is implied. Every reading here is an engineering record from the Mac
or the home PC, not a quotable result (`CLAUDE.md`). Earlier sections remain historical;
dated supersession notes and the missing batch sections follow the original operator notes.

| item | value |
| --- | --- |
| STUDY of record | Updated 2026-10-20 17:56Z: v0.18, `review/snapshots/v0.18/STUDY.md`, sha256 `027ff590825922df3b1090a96eb9bf000dd05e82920d19daf9ae504933cbfc68`; critical PASS at iteration 3 (`review/critical-v0.16-amendment.md`, `review/critical-v0.17-amendment.md`, `review/critical-v0.18-amendment.md`; `notes/LEDGER.md`, 17:56Z). The BRIEF for approval 1 binds the v0.15 snapshot by design (`review/snapshots/v0.15/STUDY.md`, sha256 `e4a6b89e0adbe3ab14b508a24335688fba48db3f801c4d1015812cc9dcd73d8e`); v0.14/v0.15 declares the Keras loss guard for every arm and model-building gate child. The rebind commit for approval 2 stamps v0.18 (`notes/LEDGER.md`, 17:58Z). |
| STUDY history through B10 (retained 2026-10-20) | `review/snapshots/v0.10/STUDY.md`, sha256 `7180c24f6b55807e52e45cf0705bb14504df6f231012dc1d68884d661a5f0c09` since B5 (v0.9-v0.10: a dated pre-data amendment of the §6 power table, three reporting labels and one §8.6 sentence; critical review PASS, `review/critical-v0.9-amendment.md`). Gates of B0-B4 were written against v0.8 (`290022fa…98ca8`, PASS by `review/arbiter-v0.5.md`, conditions P1-P6; P4 closed by `review/constructive-v0.7-delta.md`) and are repointed at the final sha in B18 |
| build plan | `PREFLIGHT-PLAN.md`, sha256 `2f1be9f360ae0410eecd3e4519fa6ded8c1c3456705b90acaee1d98649730ff8` |
| fork | `campaigns/2026-10-16-chang-bitlinear/code/`, its own git repository, branch `main`, no remote; commits are made on the Mac, the PC receives them as a bundle |
| code sha | Updated 2026-10-20: final commit `c4dc2a80ca2b8726492f9c49d5de96b3a068f128`, main; lineage `29a459ce7ee6` → `f60d3bcd1b69` → `d8dfe039382d` → `319c52f00871` → `9411e4fa796d` → `84f372fbceac` → `38f4ebae3874` → `c4dc2a80ca2b`. Last commit: "tests: the file-audit link test runs its audited children from tmp_path", `tests/test_data.py` only (fork git log; `notes/LEDGER.md`, 17:15Z). Completed full pass and targeted rerun are recorded in B18f below; tests-b1 lossguard stand-in disposition pending. |
| code-sha history through B10 (retained 2026-10-20) | B10 head `267c6deb053e6af4a51e3358a1b4d707d1ebfccc` (B10 gates of record at `267c6deb053e`, section B10); B9 head `07c33ba5fcf095f169eed77e79ea1b61260a9273` (B9 gates of record at `6da2a1edfc0d`, `415c61716290` and `07c33ba5fcf0`, section B9); B7 head `a74cd9d2b9770b9b621371e9e87835a774a6826a` (B7 gates of record at `a74cd9d2b977`; leftovers G52 at `3a0df9894dae`, G58 and G60 at `780b3d908ad6`); B6 head `b2838835b733f052f4fae829563178899993f7a0` (B6 gates of record at `b2838835b733`); B5 head `46cc25fa9e48489f6f44f52938442745e5d72733` (B5 gates at `46cc25fa9e48` and, for the leftovers G08, G52 and G56, at `78c5515df8da`; merged head before B5 `60d5e0df3599`; B4 head `07977dfe…`, B2 head `8a5389b1…`, B1 head `bfc55818…`, B0 head `1973c8b5…`); not a final code sha (that is B18) |
| venv of record | `~/lab/venvs/cbl1016-jax2` on `kaipc` (Python 3.12.13), built from `code/env/training/requirements.lock.txt` (B1 decision below) |
| data of record | PC scratch `~/lab/scratch/cbl1016/data/`: `150c-train.h5` sha256 `c2dc5b4dfe6a6ece71fc065f5a338c05284ec4b9da41ed20fd2e81d96961b1a7`, `150c-test.h5` (held-out) sha256 `9748d1aa2af060d2097a49f09da77a9a856ed5a1ce524709087943a571a00c66`; manifest `code/manifests/data-prep-150c.json` sha256 `686b15db996010cc5eadb7556397488c92ce3b48c2bccdb2a0d947f1011a7f13` (B2) |
| ConfigMap, configs, smoke, benchmark | Updated 2026-10-20: configs generated, regenerated at `9411e4fa796d` against STUDY v0.15 (`code/configs/MANIFEST.json`; `notes/LEDGER.md`, 04:28 PDT). ConfigMap: none, by bundle-only rule. NRP training smoke/packing benchmark: none, B19; earlier PC smoke-archive gates are engineering checks. Sources: `notes/handoff-2026-10-20-0445.md` §§1–3; `preflight/b18f/BRIEF.candidate.md`. Earlier header: none yet. |

## Fork commits

| sha | subject |
| --- | --- |
| `5956858c400c319b4be94e4bbfd8f27060b3665e` | base: Chang's seven jsc150 release files and vendored StopIf, byte-identical |
| `b3f4d29f558831c93c63590fdb3bcc463dc2ae82` | env: training lock copied byte-for-byte from preflight/env/lock |
| `8ba508b859e5bea7fd2d9ec85997fbdacedc41b2` | docs: draft interface notes for lane M (sidecar and config schemas) |
| `1973c8b5174125a02cbfd2dcb7d656a90c45dc3e` | gates: G01, base files byte-identical to reference-code and clean tree |
| `2e9b2dc9654563f2b13baddcd334810e7bec0dc9` | env: add wandb to the training lock |
| `ec1d13a82216f1e17043e9b4e56c147d1522869b` | cbl/env.py: M1, process env, the two shims, Chang's import, float32 matmul |
| `bf237b0a4c066d4af62bafafa978551882a66cad` | cbl/refdiff.py: M2, cited-module diff of the HGQ2 snapshot against 0.1.9 |
| `3508daade2b264dac3af00013ef034ce3a64e341` | gates: gatelib, shared header, checks and venv-of-record test for G02 on |
| `d3ef81be06f6b3f2f19ce5f53f18ccedaffc12ad` | gates: G02, shims bind only absent names before Chang's import |
| `bf6df4c258a6f0d1e1738f7d150db1a9e7633a72` | gates: G03, float32 matmul after import model and after fit, on the model's einsum |
| `bfc558180107e4c46bbb3ddda18e983f150fa9b5` | gates: G04, cited-module diff of the snapshot against installed 0.1.9 |
| `fa6b639…`, `8c29340…` | orchestrator merges of b3a and b11 (section "Lane M merges") |
| `36e0c2df829a030cdd6f74fe5188aa69974b2d33` | cbl/wandb_log.py: docstring points to interfaces.md section 8 after the merge |
| `0c4673a86e323662c9424453fb3e74b9dade8a99` | gates: fileaudit, kernel open-file audit (inotify, /proc fd inventory) for G07 and G08 |
| `6d0056a2ff93738de4f78a0ef10ac483792fa6c1` | cbl/data.py: M3, Chang's preparation unchanged, glob order and block identity, readers |
| `09ed8f9b5a62997bf1841f652639dddb325b8391` | docs: interface notes section 9, data manifest, readers and the sealed held-out reader [B2] |
| `8f0a29abfc2363b2a644b08a6b54d33ed7f5ed26` | gates: G06, preparation reproduced with Chang's script, glob order, hashes, split counts |
| `7069c6f94278d39452f4ffac5664d530f5930aa8` | gates: G07, train/validation reader bitwise equal to Chang's get_data; open-file audit |
| `d7bad87f5e6fc0d98d2c39bdfb512407260f2649` | gates: G08, held-out hashes and class counts, disjoint identities, sealed reader |
| `0021c380af885b8b90ca06f875fcb044622450ab` | cbl/data.py: name Chang's script logs by output directory and archive |
| `5aae150297a9316662a8bde443159250f847790d` | cbl/data.py: the manifest names the raw dtype the per-jet digests hash (float64), not a fixed float32 |
| `8a5389b15e76af7dfea22437969426a4cf5308ac` | manifests: data-prep-150c.json, preparation of record at 5aae150297a9 (hashes and counts only) |
| `8c1a34df953f9f93104463d4314e677ea9190109` | cbl/rng.py: M4, section-6 seed flow, initial tensors by role (U1), pre-fit reseed (U5), batch-order recorder, stream guard |
| `6af9f408cf612b61e1e889f81c13c27e2c726b38` | tests: test_rng.py, pure parts of M4 and its hgq Dataset recorder and guard in subprocesses |
| `b6d6b324d42110ca713efac497ffbd5dc691db9e` | gates: rng_child, fresh-process child of G10-G12 (draw, run with Chang's callbacks, refusals) |
| `015f6327c52e251c71b79325b0fb99728135a63b` | gates: G10, seed flow in the section-6 order, initial tensors by role, refusals |
| `4658d68f4b2f30538147180a42b8ed7699d2e457` | gates: G11, per-epoch mapped batch indices against their reconstruction from the reseed |
| `5a813f2d3c2c6e6efded4fe7e8445ab967e7df79` | gates: G12, no lab callback changes the global random streams during training; decoys |
| `af5c2a20c28d895cf1874afc47c0fd9b37f77dfd` | docs: interface notes section 10, seed flow, roles, initial-tensor manifest, recorder, guard [B4] |
| `126a6052ac3b88e4c4d7551f64d6d0bd85037118` | tests: test_rng.py, Keras fit stream equals its reconstruction with 8 and 7 batches at steps_per_execution 4 |
| `07977dfe1ebe6e34267838410325a36b846175e7` | docs: interface notes section 10, on_train_begin line citation (backend/jax/trainer.py:437) |
| `950aad0…`, `80c7960…`, `60d5e0d…` | orchestrator merges of g52fix, b3b and b15m (section "Lane M merges, second round") |
| `cbf8eb9991d342d9f81d6b95c0267f7eb8ac0983` | cbl/data.py: the sealed reader also calls M16's require_freeze_manifest (G56 join) [B5] |
| `78c5515df8daec18e85047aeebd8f9f48410631c` | gates: G56 PCC part, the sealed reader of record under the open-file audit; G08 uses a complete manifest [B5] |
| `23d4fe8cfa9fc21cc3d899746e27576df9db2916` | cbl/model.py: M5, Chang's xfm graph rebuilt in his scopes with a kernel choice per group; all-Q path (F000) [B5] |
| `6f182e2648f2439bb9c300750fad8435fea41afd` | cbl/runner.py: M13, run_train.py's recipe re-expressed with the declared departures only [B5] |
| `4d231927555fd9c617d90d6c4c821e987fad2082` | gates: runner_child, fresh-process child of G10, G11, G13, G14 and G15 (draw, run, step, ref, mha) |
| `288db489671dfd0c522d79fca02abc75a11a57b3` | gates: runner_child, import Keras after the runner's setup in step mode; wrap controls for call and build |
| `132eba09243da706fac1e222dd7aa6d8b06261ea` | gates: G13, the runner against run_train.py run by runpy on the same device, seed 42, identical batches |
| `f59c236d2281c14dfd65b836e20c1ea82d728df5` | gates: G14, F000 equals CH-8 in structure, tensors, forward, backward, one optimizer step, native cost and a short fit |
| `10b4997a2fa426cd885ee615f016dc944fd31383` | gates: G15, nested MHA kernel mapping by role and the single QLayerMeta wrap of call and build, with controls |
| `2947d06dfacbe3f553ead635e5f24c865f0efabc` | gates: G10 F000 part (loads CH-8's file by role, empty sets) and G11 across arms (CH-8 and F000 through the runner) |
| `546f0d7a13f69dbf66c9e40ba1ea7c6166253d7a` | gates: G13 and G14 compare the frozen-evaluation stub records without their wall-clock field |
| `efc598b5fa17c1605043d62d050246acfadabefe` | cbl/runner.py: docstring, run_train.py:102 printed and :77 model.summary() not, with no state change |
| `483045e02cb3531a961ab208049b2900915a4638` | tests: test_runner.py, restated recipe against run_train.py's syntax tree, config refusals, M5 arm specs |
| `46cc25fa9e48489f6f44f52938442745e5d72733` | docs: interface notes section 13, runner and lab builder; the G56 join in sections 9 and 11; ramp keys [B5] |
| `796d8dbbb35b418d0f4a879c4280841e41e001f8` | cbl/binary.py: M6, binary map with scale quantizer q_g (OD-13 in one class), FB and BF fused-BN layers, binary MHA (U3), penalty (b) [B6] |
| `a9148502dbcf1bc98e0610af1d1136a4a819686b` | cbl/model.py: M5 binary hooks (B in FB or BF, GR, BE) and binary RoleSpecs, q_g unmatched and kq unused by role [B6] |
| `a03819675a83ecd683fc90fef8bb7cc3ceff7171` | cbl/runner.py: admit F100, F001-F111, GR and BE (GR and BE in engineering stages); headline required for binary arms [B6] |
| `9d16a29d5b06a739badc8aaa4a79ee059b81f198` | cbl/env.py: provenance records XLA_FLAGS (B5 finding 1) [B6] |
| `56751eccd0795f9dfbe3f1e18dfbccc6d196af67` | gates: binary_child and binarygate, fresh-process child and audited parent of G16-G24 on constructed tensors [B6] |
| `c903d1a8ab9f0831445e91a8664190093abd8ce5` | gates: G16-G24, the binary-layer gates of STUDY section 13, one script per gate id [B6] |
| `323662884e76d417ab2b98052b3f63b0da132591` | gates: runner_child, binary arms in engineering configs, binary MHA in the mha record, refusal child for G10 [B6] |
| `dd53d6e2764dfca31521d637a406520d886c5bb9` | gates: G10, G11 and G15 binary parts (roles and constant initializers, batch hashes per form, binary MHA by role and wraps) [B6] |
| `5f3a5ef49c75247101679f5a2c6df17dfb4ceb27` | cbl/model.py: arm_spec names H1 or H2 when an all-Q arm's headline is invalid [B6] |
| `99a9520063cc7d0c50c9e247f4c28a6195ca38c0` | gates: binary_child records the active matmul precision before and after its checks [B6] |
| `61f705b3a7ce4cbf20b41c4c4e1fb3580a8f2d30` | tests: test_runner.py, binary arms admitted with a headline, GR and BE engineering only, binary RoleSpecs [B6] |
| `06e56971d401e32ba12a1b53e9b5f91bc7a11089` | docs: interface notes section 15, binary layers, scale treatment, per-site record, roles and admission [B6] |
| `b2838835b733f052f4fae829563178899993f7a0` | gates: binary_child, G20's end-to-end check is stated with its tolerance, not as exact [B6] |
| `3a0df98…` | orchestrator merge of b12 (section "Lane M merge, third round") |
| `779dddf8b6da78ab304301db441a46c113adb14d` | gates: config_child, fresh-process child of G58 and G60 (build and reload; prepare through the runner) [B7] |
| `45ba56b079c48984ee40e7fff7494ee5731cfb59` | gates: G58, every admitted config builds on CPU with its parameter count and reloads after serialization [B7] |
| `bc102acf405a161b9a8861f625203dfab0fba8ab` | gates: G60, shared initialization by role and callback order per admitted config, through the runner [B7] |
| `780b3d908ad604d4c3c6c065d5e6d4079af99ab7` | gates: config_child prep leaves cbl.env.setup() to the runner's prepare [B7] |
| `fe80ace43b64e4a63c2de721c9eac71b94ad06d0` | cbl/cost.py: M7 model part, per-site inputs from the layers, native terms from _compute_ebops, independent penalty (b) recompute, stock aggregates [B7] |
| `e5e4cfd2ea5de47193de77d5e76125411d68eb21` | cbl/frozen_eval.py: M8, non-mutating frozen evaluation with a variable-hash guard, cached per epoch, per-epoch record, appended callback, checkpoint and export audits [B7] |
| `42173092a7a256d0f01cbff2210d58ff1978f909` | cbl/runner.py: M8 replaces the frozen-evaluation stub; the runner builds the evaluator, gates pass probes [B7] |
| `5ca793e0ad24f57ffb0f2fd49df00b52ad6b9f06` | gates: costgate and cost_child, audited parent and fresh-process child of G26-G33 and the G34 audit [B7] |
| `77128f04a32795798d0b7f3f5139e6be5ecc9ba6` | gates: G26-G33, the cost gates of STUDY section 13, one script per gate id [B7] |
| `6f457f134dffbb4afeb23c647db0efd42f6e5532` | gates: runner_child, M8 in run (checkpoint saver, quantizer spy, decoy), readings command, release side without M8's evaluator [B7] |
| `9c2c4ec6f8a3f285ac58015f7ee992c572b678ad` | gates: G34 (frozen evaluation non-mutating, no training-mode call, checkpoint audit) and G36 (initialization readings) [B7] |
| `0f6640e63e7d3eefd8339b9effd078a35abd3df7` | gates: G12 frozen-evaluation part through the runner with a decoy; G13 and G14 wording for M8 [B7] |
| `c0cb5ec1a0222ffe2ce73e9d23a49acb285210da` | docs: interface notes section 16, counts and frozen evaluation; train_count int64 [B7] |
| `a74cd9d2b9770b9b621371e9e87835a774a6826a` | gates: runner_child installs the frozen-evaluation spy and decoy after the runner's setup [B7] |
| `f387d65…` | orchestrator merge of b8 (section "Merge of b8") |
| `44d15dec3b980341ac1073a8471c05a6b353b577` | cbl/frozen_eval.py: cached(epoch) for readers after the evaluation, section1_record for the front saver, the pid block from the PID's pid/ log keys [B9] |
| `6da2a1edfc0d6bdfe77059495d7395a39099da84` | cbl/runner.py: wire the front saver (M9) into the frozen-evaluation step and the run summary into finish; callback order unchanged [B9] |
| `70fdb50edbe70a3820040e80340ef5f710dd0424` | cbl/pid.py: M12, hgq2 BetaPID with the frozen-H sensor, its own pid/ log keys, the stock aggregate beside, lambda_0 2e-8, released gains, persisted state [B9] |
| `85f4e600ddf57c36d358a04304d6979079c9cb6a` | cbl/runner.py: the PID schedule (M12 at BetaScheduler's position, sensor bound after the evaluator, lab-wrapped), PID config acceptance, released-builder arm ids; PID-CH engineering-only until G43-G47 pass [B9] |
| `262023f074b6ce9c222f51ff5d904925c7caa292` | gates: G43-G47, the lab PID through the runner (sensor timing, model and stock aggregate, lambda_0 and gains, resumed integral, log keys and order), pid_child and pidgate [B9] |
| `e998a34abc5e9d5dd523d500ffa7be6460467ca6` | tests: test_pid.py, M12's record, max_beta string, refusals and shared names; test_runner.py, PID refusals and the summary's schedule [B9] |
| `415c617162908108ad3f9bad4c32ff26e61a0e3d` | gates: pid_child records each callback's logs['ebops'] after it on its own entry, not on the last event (the PID's evaluation comes after it) [B9] |
| `528789dfa84a8ab7e104c83d4ecbc8913080d49b` | cbl/runner.py: admit PID-CH and PID-CH-C in every stage after G43-G47 passed at 415c61716290; PID-BEST and PID-BEST-C stay engineering-only (B6-7) [B9] |
| `1694a56fc6559bb9fa3c9b81b34722c5d95212a0` | gates: G57, G58 and G60 state the B9 admission (88 configs, 22 of the 38 PID configs; PID order and S3/S4 gate copies in G60) [B9] |
| `b1a9d652c261af75bd45fb01bd583fd3ee31eba8` | tests: test_gen.py, the runner's acceptance under the B9 admission (88 configs) [B9] |
| `07c33ba5fcf095f169eed77e79ea1b61260a9273` | docs: interface notes section 18, lab PID, its log keys, record block, state and the runner's PID schedule; front saver wiring as built [B9] |

## Gates

| gate | sha12 | place | outcome | evidence |
| --- | --- | --- | --- | --- |
| G01 | `1973c8b51741` | MAC, `/opt/homebrew/bin/python3` (CPython 3.14.7), git 2.50.1 | PASS, exit 0 | `preflight/gates/G01/1973c8b51741/mac/` |
| G01 | `1973c8b51741` | PCC, `~/lab/venvs/cbl1016-jax/bin/python` (CPython 3.12.13), git 2.53.0 | PASS, exit 0 | `preflight/gates/G01/1973c8b51741/pc/` |
| G02 | `bfc558180107` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python` (CPython 3.12.13), JAX on CPU | PASS, exit 0, 34 checks | `preflight/gates/G02/bfc558180107/pcc/` |
| G03 | `bfc558180107` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python`, `JAX_PLATFORMS=cpu` | PASS, exit 0, 66 checks | `preflight/gates/G03/bfc558180107/pcc/` |
| G03 | `bfc558180107` | PCG, `~/lab/venvs/cbl1016-jax2/bin/python`, `JAX_PLATFORMS=cuda`, RTX 4060 Ti | PASS, exit 0, 73 checks | `preflight/gates/G03/bfc558180107/pcg/` |
| G04 | `bfc558180107` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python` | PASS, exit 0, 15 checks | `preflight/gates/G04/bfc558180107/pcc/` |
| G06 | `8a5389b15e76` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python` (CPython 3.12.13), numpy 2.5.4, h5py 3.16.0, HDF5 2.0.0 | PASS, exit 0, 51 checks | `preflight/gates/G06/8a5389b15e76/pcc/` |
| G07 | `8a5389b15e76` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python` | PASS, exit 0, 33 checks | `preflight/gates/G07/8a5389b15e76/pcc/` |
| G08 | `8a5389b15e76` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python` | PASS, exit 0, 35 checks | `preflight/gates/G08/8a5389b15e76/pcc/` |
| G10 | `af5c2a20c28d` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python` (CPython 3.12.13), JAX on CPU, smoke archive | PASS, exit 0, 75 checks; the unmatched and unused sets of F000 and the binary arms pending (B5, B6) | `preflight/gates/G10/af5c2a20c28d/pcc/` |
| G11 | `af5c2a20c28d` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python`, JAX on CPU, smoke archive | PASS, exit 0, 75 checks; the across-arm part pending (B5, B6) | `preflight/gates/G11/af5c2a20c28d/pcc/` |
| G11 | `af5c2a20c28d` | PCG, `~/lab/venvs/cbl1016-jax2/bin/python`, JAX on the RTX 4060 Ti, smoke archive | PASS, exit 0, 75 checks; the across-arm part pending (B5, B6) | `preflight/gates/G11/af5c2a20c28d/pcg/` |
| G12 | `af5c2a20c28d` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python`, JAX on CPU, smoke archive | PASS, exit 0, 35 checks; the frozen-evaluation part pending (B7) | `preflight/gates/G12/af5c2a20c28d/pcc/` |
| G07 | `46cc25fa9e48` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python` | PASS, exit 0, 33 checks; rerun after the G56 join (B5-14) | `preflight/gates/G07/46cc25fa9e48/pcc/` |
| G08 | `78c5515df8da` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python` | PASS, exit 0, 36 checks; positive control with a complete synthetic manifest, M15-only manifest refused | `preflight/gates/G08/78c5515df8da/pcc/` |
| G10 | `46cc25fa9e48` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python`, JAX on CPU, smoke archive | PASS, exit 0, 81 checks; F000 part (B5); binary arms pending (B6) | `preflight/gates/G10/46cc25fa9e48/pcc/` |
| G11 | `46cc25fa9e48` | PCC, same, JAX on CPU, smoke archive | PASS, exit 0, 82 checks; across arms CH-8 and F000 through the runner; binary arms pending (B6) | `preflight/gates/G11/46cc25fa9e48/pcc/` |
| G11 | `46cc25fa9e48` | PCG, same, JAX on the RTX 4060 Ti, smoke archive | PASS, exit 0, 82 checks; as PCC | `preflight/gates/G11/46cc25fa9e48/pcg/` |
| G13 | `46cc25fa9e48` | PCC, same, JAX on CPU, smoke archive, seed 42 | PASS, exit 0, 51 checks | `preflight/gates/G13/46cc25fa9e48/pcc/` |
| G13 | `46cc25fa9e48` | PCG, same, RTX 4060 Ti, `XLA_FLAGS=--xla_gpu_deterministic_ops=true` on both sides (B5-1) | PASS, exit 0, 56 checks; default-flag control recorded as finding 1 | `preflight/gates/G13/46cc25fa9e48/pcg/` |
| G14 | `46cc25fa9e48` | PCC, same, JAX on CPU, smoke archive, seed 1 | PASS, exit 0, 42 checks | `preflight/gates/G14/46cc25fa9e48/pcc/` |
| G14 | `46cc25fa9e48` | PCG, same, RTX 4060 Ti, deterministic GPU ops on both sides (B5-1) | PASS, exit 0, 42 checks; default-flag control recorded as finding 1 | `preflight/gates/G14/46cc25fa9e48/pcg/` |
| G15 | `46cc25fa9e48` | PCC, same, JAX on CPU | PASS, exit 0, 77 checks | `preflight/gates/G15/46cc25fa9e48/pcc/` |
| G25 | `a07bc4cdb47f` | MAC, `~/lab/venvs/mhab1012-cpu/bin/python` (CPython 3.12.13), numpy 2.5.0, branch `b3a` worktree | PASS, exit 0 | `preflight/gates/G25/a07bc4cdb47f/mac/` |
| G52 | `f77dbbb15d37` | MAC, `~/lab/venvs/mhab1012-cpu/bin/python` (CPython 3.12.13), wandb 0.28.0, branch `b11` worktree | PASS, exit 0, 77 checks | `preflight/gates/G52/f77dbbb15d37/mac/` |
| G52 | `36e0c2df829a` | PCC, `cbl1016-jax2` (CPython 3.12.13) with wandb 0.31.0, through an ephemeral `uv run --with pytest==8.4.2` overlay | FAIL, exit 1: 15 checks pass, 3 fail from one cause, the rest not reached (B2 section) | `preflight/gates/G52/36e0c2df829a/pcc/` |
| G52 | `85d6e0f98c7a` | MAC, scratch venv `~/lab/scratch/cbl1016-g52fix/venv-wandb0310` (CPython 3.12.13, wandb 0.31.0), branch `g52fix`; also under `mhab1012-cpu` wandb 0.28.0 (`mac-wandb028/`) | PASS, exit 0, 77 checks each | `preflight/gates/G52/85d6e0f98c7a/mac/` |
| G52 | `78c5515df8da` | PCC, `cbl1016-jax2` (CPython 3.12.13, wandb 0.31.0) through an ephemeral `uv run --with pytest==8.4.2` overlay, `TMPDIR` under scratch; freeze unchanged before and after | PASS, exit 0, 77 checks (the standard-library reader of g52fix) | `preflight/gates/G52/78c5515df8da/pcc/` |
| G53 | `b2290f3555b1` | MAC, `~/lab/venvs/mhab1012-cpu/bin/python`, branch `b3b`, STUDY v0.9 | PASS, exit 0, 53 checks | `preflight/gates/G53/b2290f3555b1/mac/` |
| G54 | `b2290f3555b1` | MAC, same, restated rule (rows file, 2026-10-18T09:11:03Z) | PASS, exit 0, 116 checks | `preflight/gates/G54/b2290f3555b1/mac/` |
| G55 | `b2290f3555b1` | MAC, same | PASS, exit 0, 46 checks | `preflight/gates/G55/b2290f3555b1/mac/` |
| G56 | `b2290f3555b1` | MAC part, same; PCC part pending (B5; done at `78c5515df8da`) | PASS, exit 0, 25 checks | `preflight/gates/G56/b2290f3555b1/mac/` |
| G56 | `78c5515df8da` | PCC part, `~/lab/venvs/cbl1016-jax2/bin/python` (`--pcc`): the Mac part's manifest checks without its pytest sub-step, then M3's sealed reader of record | PASS, exit 0, 43 checks | `preflight/gates/G56/78c5515df8da/pcc/` |
| G72 | `d868cad9a07d` | MAC, `~/lab/venvs/mhab1012-cpu/bin/python`, branch `b15m`, stand-in scripts only | PASS, exit 0, 581 checks | `preflight/gates/G72/d868cad9a07d/mac/` |
| G16 | `b2838835b733` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python`, JAX on CPU, constructed tensors | PASS, exit 0, 82 checks; 17 binary configurations | `preflight/gates/G16/b2838835b733/pcc/` |
| G16 | `b2838835b733` | PCG, same, RTX 4060 Ti, `XLA_FLAGS=--xla_gpu_deterministic_ops=true` (B5-1) | PASS, exit 0, 82 checks | `preflight/gates/G16/b2838835b733/pcg/` |
| G17 | `b2838835b733` | PCC, same | PASS, exit 0, 65 checks | `preflight/gates/G17/b2838835b733/pcc/` |
| G17 | `b2838835b733` | PCG, same, deterministic GPU ops | PASS, exit 0, 65 checks | `preflight/gates/G17/b2838835b733/pcg/` |
| G18 | `b2838835b733` | PCC, same | PASS, exit 0, 48 checks | `preflight/gates/G18/b2838835b733/pcc/` |
| G18 | `b2838835b733` | PCG, same, deterministic GPU ops | PASS, exit 0, 48 checks | `preflight/gates/G18/b2838835b733/pcg/` |
| G19 | `b2838835b733` | PCC, same | PASS, exit 0, 76 checks; 14 cases, every gradient byte-equal to the reference | `preflight/gates/G19/b2838835b733/pcc/` |
| G19 | `b2838835b733` | PCG, same, deterministic GPU ops | PASS, exit 0, 76 checks; as PCC | `preflight/gates/G19/b2838835b733/pcg/` |
| G20 | `b2838835b733` | PCC, same | PASS, exit 0, 243 checks; H1 and H2 | `preflight/gates/G20/b2838835b733/pcc/` |
| G20 | `b2838835b733` | PCG, same, deterministic GPU ops | PASS, exit 0, 243 checks | `preflight/gates/G20/b2838835b733/pcg/` |
| G21 | `b2838835b733` | PCC, same | PASS, exit 0, 22 checks; with controls | `preflight/gates/G21/b2838835b733/pcc/` |
| G21 | `b2838835b733` | PCG, same, deterministic GPU ops | PASS, exit 0, 22 checks | `preflight/gates/G21/b2838835b733/pcg/` |
| G22 | `b2838835b733` | PCC, same | PASS, exit 0, 52 checks; .keras reloads of 17 configurations | `preflight/gates/G22/b2838835b733/pcc/` |
| G22 | `b2838835b733` | PCG, same, deterministic GPU ops | PASS, exit 0, 52 checks | `preflight/gates/G22/b2838835b733/pcg/` |
| G23 | `b2838835b733` | PCC, same | PASS, exit 0, 26 checks | `preflight/gates/G23/b2838835b733/pcc/` |
| G23 | `b2838835b733` | PCG, same, deterministic GPU ops | PASS, exit 0, 26 checks | `preflight/gates/G23/b2838835b733/pcg/` |
| G24 | `b2838835b733` | PCC, same | PASS, exit 0, 18 checks; finding B6-9 recorded | `preflight/gates/G24/b2838835b733/pcc/` |
| G24 | `b2838835b733` | PCG, same, deterministic GPU ops | PASS, exit 0, 18 checks; as PCC | `preflight/gates/G24/b2838835b733/pcg/` |
| G15 | `b2838835b733` | PCC, same, JAX on CPU | PASS, exit 0, 114 checks; binary MHA part (B6) | `preflight/gates/G15/b2838835b733/pcc/` |
| G10 | `b2838835b733` | PCC, same, JAX on CPU, smoke archive | PASS, exit 0, 114 checks; binary arms part (B6) | `preflight/gates/G10/b2838835b733/pcc/` |
| G11 | `b2838835b733` | PCC, same, JAX on CPU, smoke archive | PASS, exit 0, 89 checks; F111-FB and F111-BF across arms (B6) | `preflight/gates/G11/b2838835b733/pcc/` |
| G11 | `b2838835b733` | PCG, same, RTX 4060 Ti, smoke archive, default XLA flags (as B5) | PASS, exit 0, 89 checks; as PCC | `preflight/gates/G11/b2838835b733/pcg/` |
| G02 | `b2838835b733` | PCC, same; rerun, M1 changed (XLA_FLAGS in provenance) | PASS, exit 0, 34 checks | `preflight/gates/G02/b2838835b733/pcc/` |
| G03 | `b2838835b733` | PCC, same, `JAX_PLATFORMS=cpu`; rerun, M1 changed | PASS, exit 0, 66 checks | `preflight/gates/G03/b2838835b733/pcc/` |
| G03 | `b2838835b733` | PCG, same, `JAX_PLATFORMS=cuda`, RTX 4060 Ti; rerun | PASS, exit 0, 73 checks | `preflight/gates/G03/b2838835b733/pcg/` |
| G13 | `b2838835b733` | PCC, same, smoke archive, seed 42; rerun, M13 and M5 changed | PASS, exit 0, 51 checks | `preflight/gates/G13/b2838835b733/pcc/` |
| G13 | `b2838835b733` | PCG, same, deterministic GPU ops on both sides (B5-1); rerun | PASS, exit 0, 56 checks; default-flag control again a finding (B5 finding 1) | `preflight/gates/G13/b2838835b733/pcg/` |
| G14 | `b2838835b733` | PCC, same, smoke archive, seed 1; rerun | PASS, exit 0, 42 checks | `preflight/gates/G14/b2838835b733/pcc/` |
| G14 | `b2838835b733` | PCG, same, deterministic GPU ops (B5-1); rerun | PASS, exit 0, 42 checks; default-flag control again a finding | `preflight/gates/G14/b2838835b733/pcg/` |
| G57 | `4d9f7cc087e0` | MAC, `~/lab/venvs/mhab1012-cpu/bin/python`, branch `b12` (includes main `b2838835b733`) | PASS, exit 0, 62 checks | `preflight/gates/G57/4d9f7cc087e0/mac/` |
| G59 | `4d9f7cc087e0` | MAC, same, empty temporary HOME | PASS, exit 0, 28 checks | `preflight/gates/G59/4d9f7cc087e0/mac/` |
| G52 | `4d9f7cc087e0` | MAC, wandb 0.31.0 (scratch venv) and 0.28.0 (`mhab1012-cpu`); M15 replica token (B12-5); PCC rerun pending | PASS, exit 0, 77 checks each | `preflight/gates/G52/4d9f7cc087e0/` |
| G52 | `3a0df9894dae` | PCC, `cbl1016-jax2` (CPython 3.12.13, wandb 0.31.0) through an ephemeral `uv run --with pytest==8.4.2` overlay, `TMPDIR` under scratch; freeze unchanged before and after; no `--keep` copy (B7 leftover after the b12 merge) | PASS, exit 0, 77 checks | `preflight/gates/G52/3a0df9894dae/pcc/` |
| G58 | `780b3d908ad6` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python`, JAX on CPU, constructed batch | PASS, exit 0, 22 checks; 66 admitted configs built and reloaded, 19 parameter counts; 101 refused by reason; finding B7-F1 disclosed | `preflight/gates/G58/780b3d908ad6/pcc/` |
| G60 | `780b3d908ad6` | PCC, same, JAX on CPU, smoke archive, draws for seeds 1-3 | PASS, exit 0, 217 checks; 66 admitted configs through the runner's prepare | `preflight/gates/G60/780b3d908ad6/pcc/` |
| G26 | `a74cd9d2b977` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python`, JAX on CPU, constructed tensors; quiet window, process scan empty | PASS, exit 0, 109 checks; 19 configurations in 5 states, 1,900 layer readings | `preflight/gates/G26/a74cd9d2b977/pcc/` |
| G27 | `a74cd9d2b977` | PCC, same | PASS, exit 0, 109 checks; with the WidthBoundError control | `preflight/gates/G27/a74cd9d2b977/pcc/` |
| G27 | `a74cd9d2b977` | PCG, same, RTX 4060 Ti, `XLA_FLAGS=--xla_gpu_deterministic_ops=true` (B5-1) | PASS, exit 0, 109 checks | `preflight/gates/G27/a74cd9d2b977/pcg/` |
| G28 | `a74cd9d2b977` | PCC, same | PASS, exit 0, 27 checks; NR equals native in every layer and state; no mismatch (B3a-3) | `preflight/gates/G28/a74cd9d2b977/pcc/` |
| G28 | `a74cd9d2b977` | PCG, same, deterministic GPU ops; added beyond the plan (B7-14) | PASS, exit 0, 27 checks; as PCC | `preflight/gates/G28/a74cd9d2b977/pcg/` |
| G29 | `a74cd9d2b977` | PCC, same | PASS, exit 0, 70 checks; with the training-mode control | `preflight/gates/G29/a74cd9d2b977/pcc/` |
| G30 | `a74cd9d2b977` | PCC, same | PASS, exit 0, 183 checks; 17 binary configurations x H1 and H2, 636 layer checks | `preflight/gates/G30/a74cd9d2b977/pcc/` |
| G31 | `a74cd9d2b977` | PCC, same | PASS, exit 0, 51 checks; with the double-count control | `preflight/gates/G31/a74cd9d2b977/pcc/` |
| G32 | `a74cd9d2b977` | PCC, same | PASS, exit 0, 109 checks; 727,320 constants snapped, SnapError control | `preflight/gates/G32/a74cd9d2b977/pcc/` |
| G32 | `a74cd9d2b977` | PCG, same, deterministic GPU ops | PASS, exit 0, 109 checks; as PCC | `preflight/gates/G32/a74cd9d2b977/pcg/` |
| G33 | `a74cd9d2b977` | PCC, same | PASS, exit 0, 149 checks | `preflight/gates/G33/a74cd9d2b977/pcc/` |
| G34 | `a74cd9d2b977` | PCC, same, smoke archive, 3 epochs, CH-8, F111-FB (H1), F111-BF (H2) | PASS, exit 0, 119 checks; 9 checkpoint audits exact, 3 mutated controls differ | `preflight/gates/G34/a74cd9d2b977/pcc/` |
| G34 | `a74cd9d2b977` | PCG, same, deterministic GPU ops | PASS, exit 0, 119 checks; as PCC | `preflight/gates/G34/a74cd9d2b977/pcg/` |
| G36 | `a74cd9d2b977` | PCC, same, smoke archive, seeds 1-3, 19 configurations | PASS, exit 0, 189 checks; readings in `init-readings.json` | `preflight/gates/G36/a74cd9d2b977/pcc/` |
| G36 | `a74cd9d2b977` | PCG, same, deterministic GPU ops | PASS, exit 0, 189 checks; all 57 readings equal PCC's | `preflight/gates/G36/a74cd9d2b977/pcg/` |
| G12 | `a74cd9d2b977` | PCC, same, smoke archive; frozen-evaluation part (B7) | PASS, exit 0, 50 checks; M8 under the guard, decoy caught | `preflight/gates/G12/a74cd9d2b977/pcc/` |
| G10 | `a74cd9d2b977` | PCC, same, smoke archive; rerun, M13 and runner_child changed | PASS, exit 0, 114 checks | `preflight/gates/G10/a74cd9d2b977/pcc/` |
| G15 | `a74cd9d2b977` | PCC, same; rerun | PASS, exit 0, 114 checks | `preflight/gates/G15/a74cd9d2b977/pcc/` |
| G11 | `a74cd9d2b977` | PCC, same, smoke archive; rerun | PASS, exit 0, 89 checks | `preflight/gates/G11/a74cd9d2b977/pcc/` |
| G11 | `a74cd9d2b977` | PCG, same, RTX 4060 Ti, default XLA flags (as B5, B6); rerun | PASS, exit 0, 89 checks | `preflight/gates/G11/a74cd9d2b977/pcg/` |
| G13 | `a74cd9d2b977` | PCC, same, smoke archive, seed 42; rerun; release side without M8 (B7-8) | PASS, exit 0, 51 checks | `preflight/gates/G13/a74cd9d2b977/pcc/` |
| G13 | `a74cd9d2b977` | PCG, same, deterministic GPU ops on both sides; rerun | PASS, exit 0, 56 checks; default-flag control again a finding (B5 finding 1) | `preflight/gates/G13/a74cd9d2b977/pcg/` |
| G14 | `a74cd9d2b977` | PCC, same, smoke archive, seed 1; rerun | PASS, exit 0, 42 checks | `preflight/gates/G14/a74cd9d2b977/pcc/` |
| G14 | `a74cd9d2b977` | PCG, same, deterministic GPU ops; rerun | PASS, exit 0, 42 checks; default-flag control again a finding | `preflight/gates/G14/a74cd9d2b977/pcg/` |
| G58 | `a74cd9d2b977` | PCC, same; rerun, M13 changed | PASS, exit 0, 22 checks; parameter counts equal the 780b3d908ad6 record | `preflight/gates/G58/a74cd9d2b977/pcc/` |
| G60 | `a74cd9d2b977` | PCC, same, smoke archive; rerun, the appended callback is now M8's | PASS, exit 0, 217 checks | `preflight/gates/G60/a74cd9d2b977/pcc/` |
| G37 | `e59dfe74a49e` | MAC (`mhab1012-cpu`) and PCC (scratch clone `~/lab/scratch/cbl1016/b8-clone`, `cbl1016-jax2`), branch `b8` | PASS, exit 0, 101 (MAC) and 137 (PCC) checks | `preflight/gates/G37/e59dfe74a49e/` |
| G38 | `fa2311498a0d` | PCC and PCG, b8 scratch clone; code unchanged to `e59dfe74a49e` | PASS, exit 0, 95 checks each | `preflight/gates/G38/fa2311498a0d/` |
| G39 | `fa2311498a0d` | PCC, same | PASS, exit 0, 25 checks | `preflight/gates/G39/fa2311498a0d/pcc/` |
| G40 | `fa2311498a0d` | PCC, same; jit and eager bytewise equal to the layer on five arms | PASS, exit 0, 40 checks | `preflight/gates/G40/fa2311498a0d/pcc/` |
| G41 | `e59dfe74a49e` | PCC, b8 scratch clone, quiet window (0 audit events) | PASS, exit 0, 31 checks | `preflight/gates/G41/e59dfe74a49e/pcc/` |
| G42 | `fa2311498a0d` | PCC, same | PASS, exit 0, 19 checks | `preflight/gates/G42/fa2311498a0d/pcc/` |
| G12 | `6da2a1edfc0d` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python`, JAX on CPU, smoke archive; rerun, B9 wiring (M8 and M13 changed); quiet window | PASS, exit 0, 50 checks | `preflight/gates/G12/6da2a1edfc0d/pcc/` |
| G13 | `6da2a1edfc0d` | PCC, same, seed 42; rerun, the front saver in the runner side only | PASS, exit 0, 51 checks | `preflight/gates/G13/6da2a1edfc0d/pcc/` |
| G14 | `6da2a1edfc0d` | PCC, same, seed 1; rerun | PASS, exit 0, 42 checks | `preflight/gates/G14/6da2a1edfc0d/pcc/` |
| G34 | `6da2a1edfc0d` | PCC, same; rerun, M8 changed (cached, section1_record) | PASS, exit 0, 119 checks; one evaluation per epoch with the saver | `preflight/gates/G34/6da2a1edfc0d/pcc/` |
| G37 | `6da2a1edfc0d` | PCC, same; rerun, runner wiring (the saver binding is unchanged) | PASS, exit 0, 137 checks | `preflight/gates/G37/6da2a1edfc0d/pcc/` |
| G60 | `6da2a1edfc0d` | PCC, same, smoke archive; rerun, prepare registers the saver | PASS, exit 0, 217 checks; 66 configs | `preflight/gates/G60/6da2a1edfc0d/pcc/` |
| G13 | `6da2a1edfc0d` | PCG, same, RTX 4060 Ti, deterministic GPU ops on both sides (B5-1); rerun | PASS, exit 0, 56 checks; default-flag control again a finding (B5 finding 1) | `preflight/gates/G13/6da2a1edfc0d/pcg/` |
| G14 | `6da2a1edfc0d` | PCG, same, deterministic GPU ops; rerun | PASS, exit 0, 42 checks; default-flag control again a finding | `preflight/gates/G14/6da2a1edfc0d/pcg/` |
| G34 | `6da2a1edfc0d` | PCG, same, deterministic GPU ops; rerun | PASS, exit 0, 119 checks | `preflight/gates/G34/6da2a1edfc0d/pcg/` |
| G43 | `415c61716290` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python`, JAX on CPU, smoke archive, seed 1, 14 epochs, PID-CH H1 and PID-BEST F111-FB H1 / F111-BF H2; PID arms engineering-only (B9-10) | PASS, exit 0, 40 checks | `preflight/gates/G43/415c61716290/pcc/` |
| G44 | `415c61716290` | PCC, same | PASS, exit 0, 34 checks; wrapper control 0 | `preflight/gates/G44/415c61716290/pcc/` |
| G45 | `415c61716290` | PCC, same (units child and PID-CH H1) | PASS, exit 0, 25 checks; log10(0) control raised | `preflight/gates/G45/415c61716290/pcc/` |
| G46 | `415c61716290` | PCC, same; the controller part (B9-11), K = 5, 10, 12 | PASS, exit 0, 39 checks; tamper and no-restore controls | `preflight/gates/G46/415c61716290/pcc/` |
| G47 | `415c61716290` | PCC, same | PASS, exit 0, 44 checks; ParetoFront saved nothing in 14 smoke epochs (finding B9-F2) | `preflight/gates/G47/415c61716290/pcc/` |
| G57 | `07c33ba5fcf0` | MAC, `~/lab/venvs/mhab1012-cpu/bin/python` (CPython 3.12.13), `main` in `$C/code` (the recorded b12 command repointed); B9 admission | PASS, exit 0, 64 checks; 88 pass, 79 refused by reason; 22 of the 38 PID configs pass | `preflight/gates/G57/07c33ba5fcf0/mac/` |
| G43 | `07c33ba5fcf0` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python`, as at `415c61716290`; B9 head, PID arms admitted in every stage | PASS, exit 0, 40 checks | `preflight/gates/G43/07c33ba5fcf0/pcc/` |
| G44 | `07c33ba5fcf0` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python`, as at `415c61716290` | PASS, exit 0, 34 checks | `preflight/gates/G44/07c33ba5fcf0/pcc/` |
| G45 | `07c33ba5fcf0` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python`, as at `415c61716290` | PASS, exit 0, 25 checks | `preflight/gates/G45/07c33ba5fcf0/pcc/` |
| G46 | `07c33ba5fcf0` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python`, as at `415c61716290` | PASS, exit 0, 39 checks | `preflight/gates/G46/07c33ba5fcf0/pcc/` |
| G47 | `07c33ba5fcf0` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python`, as at `415c61716290` | PASS, exit 0, 44 checks | `preflight/gates/G47/07c33ba5fcf0/pcc/` |
| G58 | `07c33ba5fcf0` | PCC, same, constructed batch; B9 admission | PASS, exit 0, 22 checks; 88 admitted configs built and reloaded, parameter counts equal the 780b3d908ad6 record | `preflight/gates/G58/07c33ba5fcf0/pcc/` |
| G60 | `07c33ba5fcf0` | PCC, same, smoke archive, draws for seeds 1-11; B9 admission (S3 and S4 gate copies) | PASS, exit 0, 307 checks; 88 configs through prepare | `preflight/gates/G60/07c33ba5fcf0/pcc/` |
| G12 | `07c33ba5fcf0` | PCC, same, smoke archive; rerun at the B9 head | PASS, exit 0, 50 checks | `preflight/gates/G12/07c33ba5fcf0/pcc/` |
| G10 | `07c33ba5fcf0` | PCC, same, smoke archive; rerun (M13 changed) | PASS, exit 0, 114 checks | `preflight/gates/G10/07c33ba5fcf0/pcc/` |
| G11 | `07c33ba5fcf0` | PCC, same, smoke archive; rerun | PASS, exit 0, 89 checks | `preflight/gates/G11/07c33ba5fcf0/pcc/` |
| G13 | `07c33ba5fcf0` | PCC, same, seed 42; rerun | PASS, exit 0, 51 checks | `preflight/gates/G13/07c33ba5fcf0/pcc/` |
| G14 | `07c33ba5fcf0` | PCC, same, seed 1; rerun | PASS, exit 0, 42 checks | `preflight/gates/G14/07c33ba5fcf0/pcc/` |
| G34 | `07c33ba5fcf0` | PCC, same; rerun | PASS, exit 0, 119 checks | `preflight/gates/G34/07c33ba5fcf0/pcc/` |
| G36 | `07c33ba5fcf0` | PCC, same, seeds 1-3, 19 configurations; rerun (prepare changed) | PASS, exit 0, 189 checks; the readings equal the a74cd9d2b977 record (compared by the operator, timing and sha fields excluded; not a gate check) | `preflight/gates/G36/07c33ba5fcf0/pcc/` |
| G37 | `07c33ba5fcf0` | PCC, same; rerun | PASS, exit 0, 137 checks | `preflight/gates/G37/07c33ba5fcf0/pcc/` |
| G11 | `07c33ba5fcf0` | PCG, same, RTX 4060 Ti, default XLA flags (as B5-B7); rerun | PASS, exit 0, 89 checks | `preflight/gates/G11/07c33ba5fcf0/pcg/` |
| G13 | `07c33ba5fcf0` | PCG, same, deterministic GPU ops on both sides (B5-1); rerun | PASS, exit 0, 56 checks; default-flag control again a finding (B5 finding 1) | `preflight/gates/G13/07c33ba5fcf0/pcg/` |
| G14 | `07c33ba5fcf0` | PCG, same, deterministic GPU ops; rerun | PASS, exit 0, 42 checks; default-flag control again a finding | `preflight/gates/G14/07c33ba5fcf0/pcg/` |
| G34 | `07c33ba5fcf0` | PCG, same, deterministic GPU ops; rerun | PASS, exit 0, 119 checks | `preflight/gates/G34/07c33ba5fcf0/pcg/` |
| G36 | `07c33ba5fcf0` | PCG, same, deterministic GPU ops; rerun | PASS, exit 0, 189 checks; all 57 readings equal PCC's (compared by the operator, timing and sha fields excluded; not a gate check) | `preflight/gates/G36/07c33ba5fcf0/pcg/` |
| G57 | `267c6deb053e` | MAC, `~/lab/venvs/mhab1012-cpu/bin/python` (CPython 3.12.13), the recorded command repointed; B10 admission (109) and the wired G58 counts | PASS, exit 0, 66 checks | `preflight/gates/G57/267c6deb053e/mac/` |
| G51 | `267c6deb053e` | MAC, same; static audit | PASS, exit 0, 12 checks | `preflight/gates/G51/267c6deb053e/mac/` |
| G52 | `267c6deb053e` | MAC, scratch venv `cbl1016-g52fix/venv-wandb0310` (wandb 0.31.0); M15 strict level; also `mhab1012-cpu` wandb 0.28.0 (`mac-wandb028/`, 77 checks) | PASS, exit 0, 77 checks | `preflight/gates/G52/267c6deb053e/mac/` |
| G59 | `267c6deb053e` | MAC, `mhab1012-cpu`, empty temporary HOME; configs regenerated | PASS, exit 0, 28 checks | `preflight/gates/G59/267c6deb053e/mac/` |
| G57 | `267c6deb053e` | PCC, `~/lab/venvs/cbl1016-jax2/bin/python`, JAX on CPU, smoke archive; PC part: the runner's launch check | PASS, exit 0, 32 checks | `preflight/gates/G57/267c6deb053e/pcc/` |
| G51 | `267c6deb053e` | PCC, same; static audit and the runtime guard | PASS, exit 0, 25 checks | `preflight/gates/G51/267c6deb053e/pcc/` |
| G50 | `267c6deb053e` | PCC, same; PC part (exported link B14's); CH-8 and F111-BF H2 across a resume | PASS, exit 0, 44 checks | `preflight/gates/G50/267c6deb053e/pcc/` |
| G46 | `267c6deb053e` | PCC, same; controller part (B9) and whole-run part (B10) | PASS, exit 0, 65 checks | `preflight/gates/G46/267c6deb053e/pcc/` |
| G48 | `267c6deb053e` | PCC, same; 3 arms, 12 kill-and-resume chains, controls, constructed ParetoFront/StopIf | PASS, exit 0, 165 checks | `preflight/gates/G48/267c6deb053e/pcc/` |
| G48 | `267c6deb053e` | PCG, same, RTX 4060 Ti, `XLA_FLAGS=--xla_gpu_deterministic_ops=true` (B5-1) | PASS, exit 0, 165 checks | `preflight/gates/G48/267c6deb053e/pcg/` |
| G49 | `267c6deb053e` | PCC, same, JAX on CPU, training archive of record, 109 configs | PASS, exit 0, 130 checks | `preflight/gates/G49/267c6deb053e/pcc/` |
| G07 | `267c6deb053e` | PCC, same; rerun, M3 changed (the guard) | PASS, exit 0, 33 checks | `preflight/gates/G07/267c6deb053e/pcc/` |
| G08 | `267c6deb053e` | PCC, same; rerun | PASS, exit 0, 36 checks | `preflight/gates/G08/267c6deb053e/pcc/` |
| G56 | `267c6deb053e` | PCC part, same (`--pcc`); rerun | PASS, exit 0, 43 checks | `preflight/gates/G56/267c6deb053e/pcc/` |
| G10 | `267c6deb053e` | PCC, same, smoke archive; rerun (M13, M8) | PASS, exit 0, 114 checks | `preflight/gates/G10/267c6deb053e/pcc/` |
| G11 | `267c6deb053e` | PCC, same; rerun | PASS, exit 0, 89 checks | `preflight/gates/G11/267c6deb053e/pcc/` |
| G11 | `267c6deb053e` | PCG, same, default XLA flags (as B5-B9); rerun | PASS, exit 0, 89 checks | `preflight/gates/G11/267c6deb053e/pcg/` |
| G12 | `267c6deb053e` | PCC, same; rerun | PASS, exit 0, 50 checks | `preflight/gates/G12/267c6deb053e/pcc/` |
| G13 | `267c6deb053e` | PCC, same, seed 42; rerun | PASS, exit 0, 51 checks | `preflight/gates/G13/267c6deb053e/pcc/` |
| G13 | `267c6deb053e` | PCG, same, deterministic GPU ops on both sides (B5-1); rerun | PASS, exit 0, 56 checks | `preflight/gates/G13/267c6deb053e/pcg/` |
| G14 | `267c6deb053e` | PCC, same, seed 1; rerun | PASS, exit 0, 42 checks | `preflight/gates/G14/267c6deb053e/pcc/` |
| G14 | `267c6deb053e` | PCG, same, deterministic GPU ops; rerun | PASS, exit 0, 42 checks | `preflight/gates/G14/267c6deb053e/pcg/` |
| G34 | `267c6deb053e` | PCC, same; rerun | PASS, exit 0, 119 checks | `preflight/gates/G34/267c6deb053e/pcc/` |
| G34 | `267c6deb053e` | PCG, same, deterministic GPU ops; rerun | PASS, exit 0, 119 checks | `preflight/gates/G34/267c6deb053e/pcg/` |
| G36 | `267c6deb053e` | PCC, same, seeds 1-3; rerun; readings equal the 07c33ba5fcf0 record (operator comparison, timing, sha and code fields left out) | PASS, exit 0, 189 checks | `preflight/gates/G36/267c6deb053e/pcc/` |
| G36 | `267c6deb053e` | PCG, same; rerun; readings equal PCC's (operator comparison; only frozen_seconds and the device differ) | PASS, exit 0, 189 checks | `preflight/gates/G36/267c6deb053e/pcg/` |
| G37 | `267c6deb053e` | PCC, same, `--device cpu`; rerun (a first run without `--device` ran the pure part only, kept as `pcc-pure-part-only/`) | PASS, exit 0, 137 checks | `preflight/gates/G37/267c6deb053e/pcc/` |
| G43 | `267c6deb053e` | PCC, same; rerun | PASS, exit 0, 40 checks | `preflight/gates/G43/267c6deb053e/pcc/` |
| G44 | `267c6deb053e` | PCC, same; rerun | PASS, exit 0, 34 checks | `preflight/gates/G44/267c6deb053e/pcc/` |
| G45 | `267c6deb053e` | PCC, same; rerun | PASS, exit 0, 25 checks | `preflight/gates/G45/267c6deb053e/pcc/` |
| G47 | `267c6deb053e` | PCC, same; rerun | PASS, exit 0, 44 checks | `preflight/gates/G47/267c6deb053e/pcc/` |
| G58 | `267c6deb053e` | PCC, same; B10 admission (109); counts equal manifests/param-counts-g58.json key for key (19) | PASS, exit 0, 22 checks | `preflight/gates/G58/267c6deb053e/pcc/` |
| G60 | `267c6deb053e` | PCC, same, smoke archive, draws for seeds 1-11; 109 configs through prepare | PASS, exit 0, 370 checks | `preflight/gates/G60/267c6deb053e/pcc/` |
| G52 | `267c6deb053e` | PCC, `cbl1016-jax2` (wandb 0.31.0) through the ephemeral pytest overlay, `TMPDIR` under scratch; freeze unchanged | PASS, exit 0, 77 checks | `preflight/gates/G52/267c6deb053e/pcc/` |

G01 (`code/gates/g01_base.py`) checks, on each machine: the seven `chang/` files and
`vendor/stop_if.py` hash-equal their `reference-code/` sources, the sha256 values in
`code-base.txt` and the prefixes in PREFLIGHT-PLAN §1; the recorded HGQ2 snapshot tree hash
recomputes (`e4dc4ba4912cf5d0f9d79ce63169206b979e21180f0fb7da59f6c56d5901c5e8`, 81 files);
`chang/` and `vendor/` track exactly those eight files; HEAD equals the sha12 named on the command
line; and `git status --porcelain` is empty. Each evidence directory holds `command.txt`,
`interpreter.txt`, `stdout.txt`, `stderr.txt` (empty in both runs) and `exit_code.txt`. Both
directories are byte-identical on the Mac and the PC after Mutagen synced them.

## B0 checks (PREFLIGHT-PLAN §4 row B0)

- **Start condition (plan §0).** Taken from the orchestrator's B0 brief. This session verified
  the STUDY and plan sha256 values above and that `review/arbiter-v0.5.md` and
  `review/constructive-v0.7-delta.md` exist. It did not verify the closure of P1 and P3 or the P5
  record, which live in files this role does not open.
- **PC venv against the lock.** `uv pip freeze` of `~/lab/venvs/cbl1016-jax` (uv 0.11.31) is
  byte-identical to `preflight/env/lock/freeze.txt` (sha256 `bcde70db…b314`, 35 lines; `diff`
  empty); its sorted sha256 is `d256bc3238b6eddaabf88b090da8dba1abd08264592f10a05311a19657167507`,
  equal to the orchestrator's reading. Commands in `code/env/README.md`.
- **PC lab root.** `/home/kaimoe/lab/bnjettag` is a git work tree: remote `origin`
  `git@github.com:kai124138/bnjettag-oct9.git`, HEAD `b9e31c0` (2026-10-10). This campaign's
  directory is untracked there (`??`), and the paused fork's `code/` is not tracked either. A
  lab-root launcher change could therefore be committed on the PC (plan §1, "Lab-root tools");
  nothing was committed, and the choice belongs to B17 and Kai item 1. The PC has no global git
  user name or email; B0 makes no PC commit.
- **Mutagen.** Session `bnjettag`, two-way-safe, "Ignore VCS mode: Ignore", so `.git` never
  syncs. Ignores `/data`, `/local/jev-runtime`, `/local/runs-cache`, `__pycache__`, `.DS_Store`;
  permissions mode portable. Two scan problems, both pre-existing absolute symlinks under the
  paused campaign's `code/.tmp/`; no conflict before the PC steps or after the PC evidence synced.
- **Bundle.** `preflight/pc-setup/code-main.bundle`, sha256
  `f747d076369aac014f37521fcdff67f83d4e14121b41f44ee28e0bdf87c0f124`, equal on both sides; one ref,
  `refs/heads/main` at `1973c8b5…`, complete history. On the PC, following
  `campaigns/2026-10-12-mha-bitnet/SETUP.md` §1: `git init -q -b main`, `.tmp/` added to
  `.git/info/exclude`, `git bundle verify`, `git fetch -q <bundle> main`, `git reset -q FETCH_HEAD`.
  `git status --short` printed nothing; `git rev-parse HEAD` is
  `1973c8b5174125a02cbfd2dcb7d656a90c45dc3e`, equal to the Mac.
- **`.tmp/`** is in `.git/info/exclude` on both sides and in the committed `.gitignore`.
- **Snapshot tree hash.** The value in `code-base.txt` was computed three ways with the same
  result: the Python generator, G01 on both machines, and shell pipelines (BSD `shasum` on the Mac,
  GNU `sha256sum` on the PC).
- **Jev.** The `jev-lab` MCP server failed to connect in this session (ENOENT on its Python
  runtime). `lab_check_protocol` against the STUDY snapshot and `jev_check_methods` were not run.
  B0 generates no config, so there is no config-against-STUDY drift to check yet.
- **Blinding.** None of the excluded paths in the B0 brief was opened, listed or searched. STUDY
  v0.8 was read at §2.2-§2.3, §3.1-§3.3, §5-§7, §8 to §8.6, §8.8-§8.10, §12, §13 and §19.2; §3.4
  was not read. §19.2 was read for items 11 and 12 (S4 option, claim domain); its item 5 contains
  a results-adjacent remark about the da4ml test. It states no outcome, is not reproduced here,
  and is recorded for P3. `docs/interfaces.md` also draws marked fields from §2.2-§2.3, §6, §7,
  §8.8, §8.10 and §19.2, beyond the brief's §3, §5, §8.1, §8.6 and §12.

## B0 choices

```
DECISION: the snapshot tree listing is sorted by relative path (UTF-8 bytes), the order of
          preflight/env/lock/hgq2-0.1.9-installed-files.sha256; the method is written in
          code-base.txt.
ALTERNATIVES: sort the whole "sha256  relpath" line, which orders by hash (gives a different
          tree hash for the same files).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
DECISION: the fork's second commit is env/ (B0 brief), not the shims that PREFLIGHT-PLAN §1 names
          as the second commit; the shims come with M1 in B1.
ALTERNATIVES: none taken; the brief governs B0.
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

G01 asserts more than the plan's row states (plan prefixes, tree hash, tracked set, HEAD); a
failure of any of them fails G01.

## B1 (PREFLIGHT-PLAN §4 row B1): the training lock, M1, M2, gates G02-G04

**What now works.** Chang's model imports and trains under the pinned stack with the two declared
shims bound by campaign code (`cbl/env.py`), and the float32 matmul override is shown to be active
in the model's own compiled einsums on both the PC CPU and the RTX 4060 Ti, with a negative control
that shows the check would catch TF32. The snapshot-against-0.1.9 diff of every module STUDY §7.3
cites is machine-checked (`cbl/refdiff.py`), and every differing hunk in those modules is
classified. Main limitation: all of this is the PC; the runtime image (G05, B16) and the
production GPU product are not covered, and a later JAX, jaxlib or CUDA pin change reruns G03's
GPU row.

### Step 0: wandb added to the training lock

STUDY §12 requires W&B logging in every run; the B0 lock had no `wandb`. On `kaipc` with uv 0.11.31
the B0 `requirements.in` plus `wandb` was compiled with the flags of `preflight/env/README.md:247`
and `-c constraints.txt`, a byte copy of B0's 35-package freeze (`bcde70db…b314`). The resolver
picked `wandb==0.31.0`; that pin was written into `requirements.in` and the lock recompiled.

- All 35 B0 pins are kept, and each keeps exactly its B0 hash set (439 hashes). 31 packages were
  added, wandb and its dependencies; the freeze diff shows additions only.
- The second compile gave the same lock body as the first; a third compile without `-c` gave the
  same pins and hashes (only `# via` comments differ).
- New venv `~/lab/venvs/cbl1016-jax2` (CPython 3.12.13) built with `uv pip sync --require-hashes`;
  its freeze (66 lines, `6dc63572…850b`) equals the lock's pin list byte for byte. The old venv
  `cbl1016-jax` still freezes to `bcde70db…b314` and was not touched.
- In the new venv with `KERAS_BACKEND=jax`: `jax.devices()` is `[CudaDevice(id=0)]`, keras 3.14.0
  reports backend `jax`, hgq 0.1.9 imports, and `import wandb` under `WANDB_MODE=disabled` reports
  0.31.0. No key file was read; wandb was only imported.
- Files: `code/env/training/{requirements.in, constraints.txt, requirements.lock.txt, freeze.txt}`
  and `code/env/README.md` (commit `2e9b2dc9`); every intermediate file and the exact commands in
  `preflight/pc-setup/b1-wandb-lock/` (`commands.txt`, `SHA256SUMS.txt`).

### Gates of record at `bfc558180107`

The Mac bundle `preflight/pc-setup/code-main.bundle` (sha256
`4ebe9e2c6c06a9222b7c05bf9a59a7afd27a55a119d9c7417bea0b022a3b5987`, replacing B0's `f747d076…`)
was equal on both sides after Mutagen synced it; on the PC `git fetch` and `git reset` gave HEAD
`bfc558180107e4c46bbb3ddda18e983f150fa9b5` with an empty `git status`. Each gate checks HEAD and a
clean tree at start and end, and that the interpreter is `cbl1016-jax2` with distributions equal to
`env/training/freeze.txt`. Each evidence directory holds `command.txt`, `interpreter.txt`,
`stdout.txt`, `stderr.txt`, `exit_code.txt` and `result.json`; the four directories hash the same
on the Mac and the PC. `stderr.txt` is empty except on PCG, where it holds only the two
`cuda_executor.cc:1743` driver-version lines (314 bytes) that `preflight/env/README.md:39-41`
describes.

- **G02** (`gates/g02_shims.py`). The installed hgq2 tree (78 files under `hgq/` and `qkeras/`)
  is byte-equal to `env/training/hgq2-0.1.9-installed-files.sha256` (`b38e2cc5…cec9`) and to the
  wheel RECORD, before and after the gate's imports. No installed hgq source contains `StopIf` or
  `QLinformerAttentionT`. With `hgq.utils.sugar` and `hgq.layers` imported and no Chang module
  imported, both names are absent; `bind_shims()` then binds the class from the fork's
  `vendor/stop_if.py` and the placeholder, which raises when constructed; `model` imports from
  the fork's `chang/model.py` only after that, and the `run_train.py:13` import line resolves to
  the vendored StopIf. Negative controls, each in a fresh process: without the shims
  `import model` raises `ImportError` naming `QLinformerAttentionT` and the StopIf import raises
  `ImportError`; a pre-set StopIf or QLinformerAttentionT makes `bind_shims()` refuse ("would
  shadow"); importing Chang's `data` first makes it refuse. Hashes recorded: `cbl/env.py`
  `9f33755944e2868ee0ec08192a089a4404482298b5d9ef720a35bfdbdb3c001e`; `vendor/stop_if.py`
  `ac398006…a3` (equal to `code-base.txt`); installed `hgq/utils/sugar/__init__.py` `06b80a63…4555`
  and `hgq/layers/__init__.py` `b90b1825…c6c9` (the prefixes of `preflight/env/README.md:67-68`);
  `chang/model.py` `7fcb38b4…` and `chang/run_train.py` `bfe36347…` (equal to `code-base.txt`).
  Shim source hashes (ported from): `cbl_shims.py` `dad29ef0…dbf6`, `cbl_common.py` `4c5d9b01…81bd`,
  in the `cbl/env.py` docstring.
- **G03** (`gates/g03_precision.py`), run as PCC and PCG. Readings (engineering, not quotable):
  `jax_default_matmul_precision` is `None` before `import model`, `'tensorfloat32'` after it,
  `'float32'` after the override, after `get_model`, at fit begin, at both epoch ends, at fit end,
  after fit and at gate end. CH-8 was built with Chang's `get_model('xfm', 7, 7, 1e-8, 8, True)`,
  compiled as `run_train.py:97-100` and fitted 2 epochs of 4 steps (batch 2790) on synthetic data
  of Chang's input shape (8 x 3, 5 classes) through hgq's `Dataset` with the device the gate
  passes (`cpu:0` or `gpu:0`, decision U2). In the lowered StableHLO of the model's jitted
  inference forward (18 dot ops) and training-mode loss gradient (82 dot ops) every dot carries
  precision HIGHEST; traced inside an explicit `tensorfloat32` context every dot carries HIGH (the
  negative control), on both devices. Numerical check on every QEinsumDense of the built model
  (FFN 32→24, logits 32→5, MHA query, key, value, attention_output), using the layer's own
  `QEinsumDense.call` taken from under hgq's class wrapper, against a float64 NumPy einsum of its
  own quantized input and kernel cast to float32: as fitted (E1) the error is exactly 0 on both
  devices, because the quantized operands carry at most 10 (input) and 3 (kernel) significant bits,
  which TF32 also represents exactly, so E1 cannot detect TF32; with the kernel quantizer widened
  to b=22, i=2 and the input quantizer to i=4, f=18 (E2, 17-22 significant bits), the maximum
  relative error under the active precision is 6.7e-8 to 1.7e-7 on both devices (tolerance 1e-5).
  On the GPU the same E2 path traced under an explicit `tensorfloat32` request gives 2.8e-4 to
  3.6e-4 for five layers, and for the 64x32x5 logits contraction a result bitwise equal to the
  active one (XLA computes that small dot in float32 under either request). The full layer call
  equals the output quantizer applied to the checked path for Q, K and V, and the path itself for
  the other three. Context: the 512x512 `jnp.matmul` check gives 4.9e-7 (active) and 2.7e-4
  (explicit TF32) on the GPU, 7.7e-7 for both on the CPU.
- **G04** (`gates/g04_refdiff.py`, M2). The snapshot `reference-code/HGQ2/src` re-hashes to
  `e4dc4ba4…c5e8`, 81 files, equal to `code-base.txt` (its `hgq/` part alone:
  `232929d51db2a8cf13ca9726665c038c7a96651482feec576db5eafeb4652edf`); the installed tree
  re-hashes to `b38e2cc5…cec9`, 78 files, byte-equal to the record and to the wheel RECORD. Of 83
  paths, 54 are identical, 22 differ, 5 are snapshot-only and 2 installed-only. All 12 modules of
  the §7.3 "identical" list are identical; all 7 of the "differ" list and the 3 files the "Two
  shims" paragraph cites differ; all 43 hunks of those files are classified in
  `code/cbl/refdiff_known.json` (behavioral, addition or cosmetic), none stale. The human list is
  `preflight/gates/G04/bfc558180107/pcc/refdiff/refdiff.md`, the machine report `report.json`
  beside it, one diff per differing file under `diffs/`.

**Tests** (not a gate; `preflight/gates/tests-b1/bfc558180107/pcc/`; a later `run_all.py` must
skip this directory). `tests/test_env.py` (15) and `tests/test_refdiff.py` (11): 26 passed.
pytest 8.4.2 ran in an ephemeral `uv run --with pytest==8.4.2 --no-project` overlay on top of
`cbl1016-jax2`, and every `cbl.env` scenario ran in a subprocess of the `cbl1016-jax2` interpreter
itself; `uv pip freeze` of `cbl1016-jax2` still equals `env/training/freeze.txt` afterwards
(`freeze_diff_after.txt`).

### Findings (none fails a gate; none needs a STUDY change)

1. Behavioral differences in cited modules, named in §7.3 or the chang review: the 0.1.9 build
   wrapper `check_collapsed_weight` (installed `base.py:118-140`); the snapshot-only EBOPs
   parallelization factor in `einsum_dense.py` (default ratio 1) and its `get_config` key;
   `Quantizer.get_config` serialization (`quantizer.py:64`); the `trace_minmax` verbose guard.
2. Behavioral differences in cited modules that §7.3 does not name: in `layers/core/base.py`, 0.1.9
   cannot build a multi-input layer (QAdd and the other merges) with `enable_iq=False` (two hunks;
   no effect while `enable_iq` is true, as in every xfm layer); in `layers/attn/linformer.py`,
   `QLinformerAttention.get_config` gains a key (unused by xfm). Additions not named: SNN layers,
   `QEinsumDenseT`, and the `lin_kv_proj_mode` option of `QLinformerAttention`.
3. Line citation: the §7.3 fallback bullet cites "0.1.9 `attn/mha.py:387-389`" for
   `full_output_shape`; in installed 0.1.9 those uses are at lines 384-386, and 387-389 are the
   snapshot lines (the +3 offset of chang review C17). A label slip in the STUDY text, suggested
   category C. The other five §7.3 hgq citations check out, including installed `base.py:118-140`,
   snapshot `linformer.py:183` and "stop_if.py imports only Keras".
4. Uncited: `hgq/_alkaid_keras_plugin/` differs in six files and has two snapshot-only files. It is
   on the alkaid/da4ml export path, so it is context for B14 (export environment); not classified.
5. wandb 0.31.0 declares `cwevals==0.0.1b1` as a hard dependency (wandb 0.31.0 PyPI
   `requires_dist`; cwevals: "The official Python library for the CoreWeave Evaluations API",
   author `support@wandb.ai`). Recorded, not acted on.

### Kai item (plan §4 row B1)

JAX 0.9.2 and the CUDA 12 plugin are used as locked. The wandb addition kept every one of those
pins and their hash sets, and G03's GPU row ran in the new venv. A later jax, jaxlib or CUDA pin
change reruns G03 PCG (and every GPU row of later batches).

### B1 choices

```
DECISION: the venv of record is ~/lab/venvs/cbl1016-jax2 (the B0 stack plus wandb 0.31.0 and its
          31 new packages); every PC gate from B1 on uses it. PREFLIGHT-PLAN §3's "in
          ~/lab/venvs/cbl1016-jax" for PCC and PCG, and G05's "pip freeze equal to cbl1016-jax",
          now read cbl1016-jax2: B16 compares the image with cbl1016-jax2.
CONSEQUENCE: cbl1016-jax stays as the evidence of preflight/env/ only.
ALTERNATIVES: install wandb into cbl1016-jax (would change the venv that preflight/env/ records).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
DECISION: env/training/ gains constraints.txt (a byte copy of B0's freeze) beside the four files
          the B1 brief names, so that the lock recompiles from the fork alone with every B0 pin held.
ALTERNATIVES: point -c at preflight/env/lock/freeze.txt, outside the fork.
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
DECISION: G03's E2 negative control on the GPU passes a layer if the explicit-TF32 result is at
          least 10 times worse than the active one and >= 1e-5, or if it is bitwise equal to the
          active result; at least one layer must separate.
CONSEQUENCE: the logits contraction (64x32x5) is checked for float32 accuracy (6.7e-8) but its
          control cannot separate, because XLA computes that dot in float32 under either request.
          The other five QEinsumDense layers and the StableHLO check separate.
ALTERNATIVES: require every layer to separate (the first draft; it failed only on the logits layer
          in a development run of uncommitted code, before any run of record); drop the logits
          layer from the check.
NOTE: this rule was changed after seeing a development output, so it is recorded here and in the
          G03 docstring; the run of record reproduces the case (q_einsum_dense_1:
          tf32_equals_active true; five layers at 2.8e-4 to 3.6e-4 against 7.6e-8 to 1.1e-7).
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO
```

```
DECISION: M2 resolves STUDY §7.3's short name "activation.py" to hgq/layers/activation.py (the
          file the xfm model uses, QAffinedUnaryFunctionLUT); _dais_tracer/layers/activation.py is
          covered by the _dais_tracer/ entry and _alkaid_keras_plugin/activation.py is not cited.
          M2 also classifies hgq/layers/attn/__init__.py, the re-export path of QLinformerAttentionT,
          beside the two files the "Two shims" paragraph cites.
ALTERNATIVES: treat all three activation.py files as cited; all three are identical in the two
          trees, so the outcome would be the same.
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

### Not done in B1

- Jev (`jev-lab` MCP) failed to connect (ENOENT on its Python runtime); `lab_check_protocol` and
  `jev_check_methods` were not run. No configs exist yet, so there is no config drift to check.
- G05 (image) is B16's. G01 was not rerun at `bfc558180107`; every gate reruns at the final sha (B18).

### Blinding (B1)

None of the excluded paths was opened, grepped or displayed by content. Two directory listings,
of `preflight/` and `preflight/env/`, displayed the names `da4ml-evidence-hashes-2026-10-17.txt`,
`da4ml-perpos-evidence-hashes-2026-10-17.txt`, `da4ml-test/`, `results/` and `logs/` (names
only). `mutagen sync list --long` displayed the paths of B11's G52 evidence (scan problems).
STUDY v0.8 was read at §7.3, §12 and §13 in full and its heading list; PREFLIGHT-PLAN at §1-§2, §3
rows G01-G18, §4 rows B0-B16, §8 and §9 items 1-8; `preflight/env/README.md` and `scripts/`;
`review/chang-v0.3.md` header and C15-C19. No hardware-test outcome was seen.

## Lane M merges (orchestrator)

- `fa6b639` merges branch `b3a` (head `a07bc4cdb47f`; M7 pure part `cbl/cost_pure.py`, gate G25) and `8c29340` merges
  branch `b11` (head `f77dbbb15d37`; M15 `cbl/wandb_log.py`, gate G52 Mac part) into `main` after B1 (`bfc558180107`).
  The only conflict: both branches appended a section 7 to `docs/interfaces.md`; b11's became section 8, with a note
  that `cbl/wandb_log.py:14` still says "section 7" (docstring fix left to the next lane-P commit). No code was edited
  in the merge.
- Full rows, commit lists and owner decisions: `preflight/gates/G25/a07bc4cdb47f/PREFLIGHT-rows.md` (B3a-1 u_c max-F
  domain, MEDIUM, FLAG YES; B3a-2 u_c = 0 with no included input, HIGH; B3a-3 float32 combine of kernel and bias terms,
  MEDIUM, FLAG YES, settled by G28 in B7) and `preflight/gates/G52/f77dbbb15d37/PREFLIGHT-rows.md` (B11-1 run-name
  format and run id; B11-2 engineering runs in the STUDY group under job_type "engineering"; B11-3 substring refusal
  rule, fail closed; B11-4 inner key names M8 and M13 must emit).
- Orchestrator checks at the merged sha `8c2934052fed` on the Mac (not evidence of record; every gate reruns at the final
  sha in B18): G01 PASS, G25 PASS, G52 Mac part PASS (77 checks), same interpreters as above.
- G52's PCC row and a G52 run under the lock's wandb 0.31.0 are pending (done at `78c5515df8da`, B5); the training lock has no pytest, so that run
  uses an ephemeral `uv run --with pytest` overlay as B1's tests did.
  (B2: run at `36e0c2df829a`, FAIL from one cause in the gate's readback; see the B2 section.)
- Dependency note: wandb 0.31.0 hard-depends on `cwevals==0.0.1b1` (Apache-2.0, project URL github.com/wandb/core, author
  email support@wandb.ai; checked by the orchestrator in `cbl1016-jax2`); it is content-hashed in the lock and accepted.
- For the PREFLIGHT critical review: G03's GPU control rule was changed after a development run, before any run of
  record (B1 section); the property gated, float32 on every dot op, did not change.

## B2 (PREFLIGHT-PLAN §4 row B2): data preparation, M3, gates G06-G08, and G52 on the PC

**What now works.** The two archives Chang's code reads are prepared on the home PC by his own
script, run unchanged, and every raw file they come from is shown to be byte-identical to the
published Zenodo tarballs. Each jet has a recorded identity (file in glob order, row in that
file), and the archive's content proves that order. The lab's train/validation reader
(`cbl/data.py`, module M3) returns arrays bitwise equal to Chang's `get_data` on the archive of
record, and a kernel-level audit shows that the reader's process opens no held-out file. The
held-out reader refuses unless a freeze manifest exists. Most important limitation: "valid freeze
manifest" means only M15's rule for now (a non-empty JSON object at the given path). Its
completeness is M16's check (G56, batch B3b), so until G56 joins the two, a hand-written dummy
manifest would unseal the reader. Two further limits. The audit covers the reader in a process of
its own; the training runner does not exist yet (M13, B5), and G51 (B10) audits it. The archives
of record exist only in PC scratch; the PVC copy is G09 (B19).

### Provenance: Zenodo record 3602260

- Record metadata from `https://zenodo.org/api/records/3602260`, saved on the PC as
  `~/lab/scratch/cbl1016/zenodo-api/record-3602260.json` (sha256
  `9ae705582175a9a360c456296bc657cab4dd3770b207139727b6aaaec2cd71ee`; title "HLS4ML LHC Jet dataset
  (150 particles)", updated 2020-01-24). It lists md5 and size per file.
- Download. The URLs are those of `prepare_data.sh:15-16`. `aria2c` is not installed on the PC.
  A single curl stream ran at about 350 kB/s, so it was stopped and its partial file deleted. The
  download of record used curl with 8 parallel byte ranges per file (Zenodo answers 206), which
  were then concatenated. Scripts and log: `~/lab/scratch/cbl1016/zenodo-log/`. Nothing was
  uploaded. The complete list of outbound requests: the API record, the aborted single-stream
  download, two probes of the val URL that confirmed range support (a 16-byte range GET and a
  HEAD), and the 16 range requests of record.
- `hls4ml_LHCjet_150p_train.tar.gz`: 2,725,115,104 bytes, equal to the record. Its md5
  `af12d3a500414924e70a73b56fc49aec` equals the record's; sha256
  `cae2338b323c802f649b25b0fe49bd439e23e81f70f75cf58861d3c84679054f`. It holds 62 regular members
  under `train/`, and each equals the on-disk file of the same name in size and sha256.
- `hls4ml_LHCjet_150p_val.tar.gz`: 1,142,180,375 bytes and md5 `ff902e6a02bfb4247a9068bf7ff7f6db`,
  both equal to the record; sha256 `a4277e65d10acbb18cb74f7344cbce72d7fb6bb49116e13037b2e3e3b8d81ac4`.
  It holds 26 regular members under `val/`, all equal to the on-disk files.
- No member is missing on disk, and no on-disk file is missing from its tarball, so every file the
  preparation read is the published file. `jetImage_9_150p_40000_50000.h5` is in neither the
  tarball nor the directory, as `DATASET.md:16-19` says. The member hashes come from one
  streaming `tarfile` pass under `python -I`, with nothing extracted.

### Preparation of record, at `5aae150297a9`

- Evidence: `preflight/pc-setup/b2-data/5aae150297a9/`. It holds the command, interpreter, stdout,
  stderr (empty) and exit code (0); Chang's script logs (`logs/`); the git state before and after
  (HEAD `5aae1502…`, clean tree both times); the manifest; and `stats.json`, which holds the
  training-archive shift and scale values. Those values are kept in evidence, not in the fork.
- `cbl/data.py prepare` runs `chang/tools/prepare_data.py` by path, in a child of the venv of
  record started with `-I -B`, as `-i <dir>/ -o <archive> -j 4` from the output directory, exactly
  as `prepare_data.sh:21-22` do. Before running, it checks the script's sha256 `37f37a56…4d80`
  against `code-base.txt`. No line of the script is patched or wrapped. Inputs are
  `/home/kaimoe/lab/bnjettag-data/{train,val}/` and outputs go to `~/lab/scratch/cbl1016/data/`.
  Neither directory is inside the repository or `$C`.
- Glob order. The script lists its inputs with `Path(inp).glob('*.h5')`. That follows `os.scandir`
  order, which is unsorted on this ext4 directory. The same call made before and after each run
  gave the same list, which the manifest records: training starts `jetImage_2_150p_30000_40000.h5`,
  `jetImage_1_150p_40000_50000.h5`, `jetImage_6_150p_60000_70000.h5`; held-out starts
  `jetImage_9_150p_50000_60000.h5`. The order is proved from content. Archive block k (10,000
  rows) equals file k in glob order: for training files, by the sha256 of the float16 feature
  block and of the label block; for held-out files, by the label block only, because the held-out
  features are never read back. All label blocks are distinct, so the match is unambiguous.
- Archives (engineering, not quotable):

  | archive | bytes | sha256 | datasets | jets per class (5 classes, `jets[:, -6:-1]` order) |
  | --- | ---: | --- | --- | --- |
  | `150c-train.h5` | 1,104,167,353 | `c2dc5b4dfe6a6ece71fc065f5a338c05284ec4b9da41ed20fd2e81d96961b1a7` | feature (620000, 150, 16) float16 lzf; label (620000,) int64 | 124848, 120211, 124937, 124654, 125350 |
  | `150c-test.h5` (held-out) | 448,219,074 | `9748d1aa2af060d2097a49f09da77a9a856ed5a1ce524709087943a571a00c66` | feature (260000, 150, 16) float16 lzf; label (260000,) int64 (metadata) | 52404, 50468, 52235, 52298, 52595 (equal to `DATASET.md:46-48`) |
  | smoke `150c-train.h5` | 33,501,595 | `1e6b221a3c1b5947a92e61b82fc1e01ae3d9614410a255d888f4ce8c27319b1b` | feature (20000, 150, 16) float16; label (20000,) int64 | 3930, 3951, 4009, 4059, 4051 |

  The one-hot assertion of `prepare_data.py:13` holds in all 88 raw files. The raw
  `jetConstituentList` is float64 and is cast to float16 by the script (`:15`).
- Smoke archive (plan §9 item 10). It is built from the first two training files in glob order,
  linked into `~/lab/scratch/cbl1016/data-smoke-src/`, by Chang's script into
  `~/lab/scratch/cbl1016/data-smoke/`. It is an engineering input only: its split and statistics
  differ from those of the archive of record.
- Split of record, with CH-8's arguments (8 constituents; pt, etarel, phirel) after
  `np.random.seed(42)`: 558,000 training and 62,000 validation jets. `order_sha256` is
  `639be200df82ba76ad6b289f946f1bccea3a56fa1767fba0830ccd6bd55aab3b`, `split_sha256`
  `0847a511f435c5060c660dffd55b855e3a65c49f0c8412de7667f6ee16b37f8c` and `stats_sha256`
  `d12c010e956a04bd2a2db76fbfb9f868d022eae1d8685ed1841d85151b5b7b1a` (definitions in
  `code/docs/interfaces.md` §9). All three per-feature stds are finite and nonzero.
- The manifest `code/manifests/data-prep-150c.json` (sha256 `686b15db…7f13`) holds hashes and
  counts only, and is committed at `8a5389b15e76`. Its sha256 is the configs'
  `data_manifest_sha256`.
- Byte-identical archives. A first preparation at `0021c380af88` wrote the same three archives
  byte for byte. It was superseded because its manifest called the per-jet digest dtype float32
  while the raw arrays are float64; the digests themselves were of the stored bytes. Its record is
  kept in `preflight/pc-setup/b2-data/0021c380af88/` (`SUPERSEDED.txt`), and its archives were
  deleted after the comparison. G06 ran the script a third time, again byte-identical.

### Gates of record at `8a5389b15e76`

The Mac bundle (sha256 `dd0067d36abec0660c12215612c7258238a8e0142f5d1a19f03d3715d222a003`) was equal
on both sides after Mutagen synced it. On the PC, `git fetch` and `git reset` gave HEAD
`8a5389b15e76…` with an empty `git status`. Each gate checks HEAD and a clean tree at start and end,
and the venv of record with its freeze. Each evidence directory holds `command.txt`,
`interpreter.txt`, `stdout.txt`, `stderr.txt` (empty in all three), `exit_code.txt` and
`result.json`. Work directories were under `~/lab/scratch/cbl1016/gates-work/` and are deleted by
the gates.

- **G06** (`gates/g06_prep.py`), 51 checks. The manifest is M3's kind, was prepared at an ancestor
  of HEAD, and names Chang's two files at their `code-base.txt` hashes. Python, numpy, h5py, HDF5
  and tqdm equal the manifest's. Glob order, sha256, size, rows, offsets, label block, class counts
  and (training) float16 feature block are re-read for all 62 and 26 raw files and equal the
  manifest. The one-hot assertion holds. The archives of record re-hash, have the stated layout
  and pass the block identity. `read_train_val` reproduces the split record (558,000 and 62,000;
  finite nonzero stds). The smoke archive re-hashes. Chang's script, run again into the work
  directory on both raw directories and on a fresh 2-file link directory, wrote three archives
  byte-identical to those of record, which were then deleted. Both tarballs re-hash: md5 and size
  equal the Zenodo record, sha256 equals the manifest, every on-disk file equals its member, and
  the member order is unchanged.
- **G07** (`gates/g07_reader.py`), 33 checks. X_train (558000, 8, 3) float16, y_train (558000,)
  int32, X_val (62000, 8, 3) float16 and y_val (62000,) int32 from `cbl.data.read_train_val` are
  equal in dtype, shape and bytes to Chang's `get_data` (loaded by path from `chang/data.py`). Each
  side ran in its own process after `np.random.seed(42)`. Chang's side ran on a placeholder
  directory: a link to the archive of record, plus a 16-jet zero file as `150c-test.h5`. A copy
  with one flipped bit fails the comparison. Both reproduce the manifest's split hashes.
  Open-file audit: inotify IN_OPEN watches covered the held-out archive, its directory (by entry
  name), the raw held-out directory and each raw held-out file, from before each process started
  until it ended. They recorded no forbidden event for the reader, for Chang's `get_data` or for
  the decoy process. Positive controls: the reader's h5py open of the training archive was seen;
  `get_data`'s open of the placeholder `150c-test.h5` was seen, which shows the audit sees h5py's
  C-level open; and a decoy opened through a symbolic link elsewhere was seen. The reader's /proc
  fd inventory (6 paths) holds the training archive and no held-out path.
- **G08** (`gates/g08_heldout.py`), 35 checks. Held-out data were touched for hashes and counts
  only. The 26 raw files have the recorded glob order, hashes, rows and label blocks, 260,000 jets
  in all. The held-out archive re-hashes to `9748d1aa…0c66`, its metadata is (260000, 150, 16)
  float16 and (260000,) int64, its per-class counts are as in the table above, and its label
  blocks match the files in order. Its `feature` dataset was not read. Identities are disjoint: no
  file name or file sha256 is shared between the 62 training and 26 held-out files, and none of
  the 880,000 per-jet digests (sha256 of each raw `jetConstituentList` row) is shared.
  The sealed reader, in its own process under the same audit, refused seven cases on the held-out
  archive of record: no, missing, invalid-JSON, empty-object and JSON-list manifest, and a valid
  dummy manifest without, or with a malformed, expected sha256. It also refused two cases on a
  dummy archive: no manifest, and a wrong expected sha256. It recorded no open of any held-out file
  during the window. It is never called on the archive of record with a valid manifest. On a dummy
  pair it returned data: the dummy is the smoke archive as `150c-train.h5`, plus an archive that
  Chang's script wrote from training files 3 and 4 in glob order (`jetImage_6_150p_60000_70000.h5`,
  `jetImage_3_150p_30000_40000.h5`) as `150c-test.h5`. With a valid dummy manifest, all six arrays
  (train, validation and test) equal Chang's `get_data` on the same directory in dtype, shape and
  bytes.
- **Tests** (not a gate; `preflight/gates/tests-b2/8a5389b15e76/pcc/`, also run at `0021c380af88`).
  `tests/test_data.py` ran on synthetic files only: 15 passed, including the inotify test. It used
  the same ephemeral `uv run --with pytest==8.4.2` overlay as B1, with `TMPDIR` under scratch, and
  `uv pip freeze` of `cbl1016-jax2` equals `env/training/freeze.txt` afterwards. A later
  `run_all.py` must skip this directory.

### G52 on the PC (leftover from B11): FAIL, one cause, not fixed here

The gate ran at `36e0c2df829a` in `cbl1016-jax2` (wandb 0.31.0) through the pytest overlay. The
`cbl1016-jax2` freeze was unchanged before and after (both diffs empty). Result: exit 1. 15 checks
passed: HEAD and clean tree, the module's static checks, the unit tests (67 passed under 0.31.0),
the scanner's positive control, and the scans of the children's streams and of the decoy values.
3 checks failed, all from one cause. The gate's record readback (`gates/g52_wandb.py:190-191`)
imports `wandb.sdk.internal.datastore`, which wandb 0.31.0 no longer ships. Its `sdk/internal/`
holds only `__init__.py`, `_generated/`, `incremental_table_util.py`, `job_builder.py` and
`profiler.py`, and no `DataStore` class or `scan_data` function exists anywhere in the installed
package. Both children ("clean" and "decoy") wrote their offline runs and then exited 1 at that
import. With no record header parsed, the producer build id that follows `0.31.0+` at byte offset
50 of each `.wandb` file is not on the allow list, so the scan of the temporary root fails. The
Mac run under 0.28.0 allows the same kind of token as "wandb build id". The checks that read the
records were not evaluated under 0.31.0, which is not the same as passing: every §12 key per trait
combination; project, group, entity, run name and id; the held-out refusal end to end; the
open-path audit; and the absence of `WANDB_API_KEY`. This blocks G52 at the final sha (B18). The
fix belongs to M15's gate: a gate-side reader of the `.wandb` record format that imports no SDK
internals, first shown on the Mac's existing 0.28.0 records to give the same records as 0.28.0's
`datastore`, then rerun on the PC. This run does not implicate `cbl/wandb_log.py`: its unit tests
pass and it wrote offline runs. Diagnosis helper (scratch, not campaign code):
`~/lab/scratch/cbl1016/b2-tools/hex40_context.py`; it prints the 16 bytes before each 40-hex token,
never the token.

### B2 choices

```
B2-1 DECISION: the sealed reader's freeze-manifest rule is M15's, called (cbl.wandb_log.
          freeze_manifest_sha256), not a third rule; M3 adds only the archive check (sha256 equal to
          the committed manifest's value), and both are checked before the archive is touched.
CONSEQUENCE: until G56 (M16, B3b) adds completeness, any non-empty JSON object unseals the reader;
          only discipline and G51 stand between a dummy manifest and the real archive meanwhile.
ALTERNATIVES: M3 checks the section-8.5 items itself (a key list B3b is defining in parallel; three
          rules where one is wanted); refuse always until M16 exists (then G08 could not show the
          positive case).
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO (the orchestrator joins M3, M15 and M16 at G56)
```

```
B2-2 DECISION: the open-file audit is kernel inotify (IN_OPEN on each forbidden file and directory,
          any process, the whole window) plus a /proc fd inventory of the audited process; strace is
          not installed on the PC, and Python audit hooks do not see h5py's C-level opens. The audited
          process at B2 is the reader in its own process, standing in for the training process.
ALTERNATIVES: /proc fd sampling alone (misses short opens); strace (absent; installing it needs root).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B2-3 DECISION: "identities disjoint from training" is checked at three levels: file names, file
          sha256, and a per-jet digest (sha256[:16] of each raw jetConstituentList row). The digest
          reads the raw held-out constituent arrays to hash them (as Chang's preparation reads them
          anyway); only the intersection size (0) is recorded.
ALTERNATIVES: names and file hashes only, which identity (file, row) strictly needs.
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: YES (it touches held-out bytes, hashes only)
```

```
B2-4 DECISION: read_train_val adds two guards that change no output: the global legacy generator
          must be in the state np.random.seed(42) leaves (run_train.py:41, no draw before
          data.py:36), and a zero or nonfinite std raises (STUDY §4). It takes the archive path, not
          a directory, and refuses a file named 150c-test.h5.
ALTERNATIVES: Chang's behaviour without guards (expect_seed=None keeps it available).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B2-5 DECISION: G06 also re-runs Chang's script into a work directory and requires byte-identical
          archives (the plan row asks for "preparation reproduced"); the archives of record stay the
          ones written at 5aae150297a9, and G06's copies are deleted after hashing.
ALTERNATIVES: verify the archives of record only (the plan row read narrowly).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B2-6 DECISION: the smoke archive is the first two training files in glob order, and G08's dummy
          held-out archive the third and fourth, each linked into a fresh directory and prepared by
          Chang's script; "file" in the jet identity is recorded by name and sha256 together.
ALTERNATIVES: any other two files; copies instead of links (same bytes, more disk).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

### Not done in B2

- G09 (the PVC copy) belongs to B19. The archives of record are on no machine but the PC.
- G52 PCC is recorded as FAIL and was not fixed here (above).
- Jev (`jev-lab` MCP) is down: `lab_check_protocol` and `jev_check_methods` were not run. No config
  exists yet, so there is no config drift to check.
- The freeze manifest's schema stays M16's (B3b). `interfaces.md` §9 states the M3 side.

### Blinding (B2)

None of the excluded paths was opened, listed or searched. Read: the B2 brief and the common rules;
`RULES.md`; `docs/methodology/03-phases.md` and the PREFLIGHT parts of `06-review.md`; STUDY v0.8 §4,
§8.5 and §13 and its heading list; PREFLIGHT-PLAN in full (its §0 blinding note names a fragment
without reproducing it); `code/docs/interfaces.md`; `chang/data.py`, `chang/run_train.py:1-80` and
`chang/tools/prepare_data.{py,sh}`; `DATASET.md`; the paused fork's `bnhgq2/derived_view.py`; the
fork's gates, `cbl/wandb_log.py` (head and freeze-manifest functions), `gatelib.py` and
`code-base.txt`. No hardware-test outcome was seen.

## B4 (PREFLIGHT-PLAN §4 row B4): seeding, initial tensors and batch order, M4, gates G10-G12

**What now works.** A run can now be started only in the order STUDY §6 prescribes; any other
order is refused. For a given seed, CH-8's initial weights are drawn once, written to a file whose
content is pinned by a manifest of hashes, and loaded tensor by tensor under names that do not
depend on Keras's automatic layer names (decision U1). Just before training, the Python random
stream that decides the batch order is reseeded from the seed alone (decision U5), and a recorder
logs, for every epoch, which batches the data loader actually fetched. On the home PC, on the CPU
and on the RTX 4060 Ti, two runs of CH-8 with the same seed fetched the same batches in the same
order in every epoch, and that order equals what follows from the reseed when the draws Keras and
hgq make are simulated. A third run that drew three extra random numbers before the reseed,
standing in for an arm whose model build draws more often, fetched the same batches. A guard shows
that no callback changes the global random streams during training, and stops a run whose lab code
does. Main limitation: only CH-8 exists. That all arms of a seed share initial tensors and batch
order is shown for CH-8 against itself only; F000 and the binary arms come in B5 and B6, and the
gates rerun then. The gates also use a stand-in for the training runner (M13, B5) and the 2-file
smoke archive, so the initial-tensor files they draw are not files of record; under U1 those are
drawn on NRP in B19.

### How it works

- **Order (STUDY §6 steps 1-5).** `cbl.rng.SeedFlow` has one method per step and refuses a step
  unless the previous one was the step before it: `np.random.seed(42)` and M3's reader; `random.seed(s)`
  and `keras.utils.set_random_seed(s)` (which also reseeds NumPy); the two hgq Datasets as
  `run_train.py:73-74`, then the model; the initial tensors loaded; the reseed. Building a Dataset
  draws from the stream (`dataset.py:66-79`: one `randint` per Dataset and one shuffle of the
  training batch map), so the flow recomputes those draws from `random.Random(s)` and refuses a
  difference. After every step it records hashes of both global random states.
- **Roles.** Keras names layers with process-wide counters (`q_einsum_dense_batchnorm_2`), so a
  tensor's Keras path depends on what was built before in the same process. A role is built from
  objects instead: the arm's name for the top-level layer, one segment per sublayer, then the
  variable's own name, for example `ffn2/kq/FixedPointQuantizerKBI[0]/b` or `mha/query/kernel`.
  CH-8 has 14 top-level layers and 244 variables, each with one role. F000 and the binary arms
  supply their own role specification (`RoleSpec`) in B5 and B6, declaring which of their tensors
  have no CH-8 counterpart (the unmatched set; STUDY §6 step 4 names the scale quantizers' b and i)
  and which CH-8 tensors they lack (the unused set).
- **The initial-tensor file.** All 244 variables, trainable or not, in an uncompressed `.npz` with
  fixed metadata, so equal tensors give equal bytes, and a JSON manifest: each role's shape, dtype
  and sha256, the file's sha256, and an identity block (seed, device platform and kind, Dataset
  device, package versions, matmul precision, hashes of the five files that determine the draw,
  data identity, the Datasets' construction draws). The loader refuses unless the caller names the
  manifest's sha256 and it matches, the file and every tensor match the manifest, every identity
  field equals the run's, and the unmatched and unused sets equal the arm's declaration; nothing is
  assigned before all of these pass. The file depends on the data: the Datasets draw before the
  model is built, and how many draws they make depends on the archive's length.
- **The recorder.** The training Dataset is an instance of a subclass, created at run time, of the
  installed `hgq.utils.sugar.Dataset`. Its batch map is replaced by a list subclass that records a
  read only while the Dataset's own `__getitem__` runs, so the recorded number is exactly the
  mapped batch index of `dataset.py:88-89`, and the reads made while the map is reshuffled are not
  recorded. No file of Chang's or of hgq2 is changed. The per-epoch hash is the sha256 of the
  little-endian int64 indices in fetch order (`code/docs/interfaces.md` §10).
- **What Keras draws (the reconstruction).** Before epoch 0, Keras's `fit` runs a "symbolic build"
  (a dry pass that fixes the input shapes): it starts the epoch iterator once, which shuffles the
  batch order and fetches two batches (the iterator reads one batch ahead), and then resets the
  iterator, which reshuffles the Dataset's batch map. In each epoch there is one shuffle of the
  batch order, then one fetch per batch, then one reshuffle of the map. Validation draws nothing.
  `cbl.rng.reconstruct_fit_stream` simulates this from the reseed with a private random generator;
  the two pre-fit fetches are kept as a separate record, outside epoch 0.
- **The guard.** Between the reseed and the end of `fit`, every hook a callback overrides is
  wrapped; the module-level random state and NumPy's legacy state are compared before and after
  each call, and a change by lab code raises `StreamViolation`. Chang's callbacks are wrapped as
  released code (a change is recorded, not raised). `guard.watch(label)` does the same around a
  block of lab code, for the frozen evaluation of B7.

### Gates of record at `af5c2a20c28d`

The Mac bundle `preflight/pc-setup/code-main.bundle` (sha256
`b3e7ce4c457fbed8a521b4e0514cc537acc4ece317e3225e4e469fc43a4a88bd`, replacing B2's `dd0067d3…`) was
equal on both sides after Mutagen synced it; on the PC `git fetch` and `git reset` gave HEAD
`af5c2a20c28d895cf1874afc47c0fd9b37f77dfd` with an empty `git status`. Each gate checks HEAD, a
clean tree at start and end, and the venv of record against its freeze, and reads the 2-file smoke
archive (`~/lab/scratch/cbl1016/data-smoke/150c-train.h5`, sha256 `1e6b221a…1b1b`; the gates check it
is neither the archive of record nor the held-out archive). Each step runs in a fresh child process
(`gates/rng_child.py`), as in a pod. Each evidence directory holds `command.txt`, `interpreter.txt`,
`stdout.txt`, `stderr.txt` (empty for all four gate processes), `exit_code.txt`, `result.json` and
`children/` (each child's argv, JSON record, stdout and stderr). The children's stderr holds PBar's
progress lines, with loss and accuracy on the smoke archive (engineering, not quotable), and on PCG
the two `cuda_executor.cc` driver lines per process. Every reading below is engineering, not quotable.

- **G10** (`gates/g10_seedflow.py`, PCC, 75 checks). Three CPU draws (seed 1 twice, seed 2): steps
  in the §6 order; after step 2 random equals `Random(s)` and NumPy equals `RandomState(s)`; the
  gate's own reconstruction of the Datasets' construction draws from `Random(s)` equals the drawn
  values (seed 1: shifts 275 and 967, training map `[3, 6, 1, 5, 2, 0, 4]`) and the stream state
  after them; 244 unique roles; no quantizer has a seed generator. Seed 1 drawn twice gave the same
  file (209,850 bytes, sha256 `6f0074b16e4a…2063`), the same tensors and the same forward probe;
  seed 2 differs from seed 1 in exactly the 11 Glorot kernels (embedding, Q, K, V, O, the two FFN
  contractions, the three head layers, logits), every other tensor equal across the two seeds. A
  CH-8 run for seed 1 repeated steps 1-3: before loading, its own draw equals the file in all 244
  roles; after loading, every variable equals the file; the unmatched and unused sets are empty;
  loading drew from neither stream; the forward probe and the stream state after the model build
  equal the drawing process's; the reseed is `prefit_reseed(1)` = 10458850402844946191 and leaves
  random in `Random(10458850402844946191)`'s state. Refused, each before any variable changed: no
  expected manifest sha256; a wrong one; one flipped bit in the file; the manifest's seed, device
  platform, `n_train`, construction-map hash, `cbl/rng.py` hash or Keras version edited (with the
  expected sha256 of the edited manifest); one role's sha256 edited; a role map that renames one
  layer. The flow refused steps out of order, a bad Dataset device and a config `prefit_reseed` that
  differs from U5. Pending (B5, B6): the unmatched and unused sets of F000 and the binary arms.
- **G11** (`gates/g11_batch_order.py`, PCC and PCG, 75 checks each). Per device: a seed-1 draw on that
  device, then six CH-8 runs of 3 epochs with Chang's compile and callbacks (`run_train.py:79-100`,
  copied in the child) and the lab probe after StopIf. In every run the training Dataset fetched with
  one worker, `drop_last` false; one pre-fit record with 2 fetches; per epoch 7 fetches, the mapped
  indices a permutation equal to the map of the raw indices, the map fixed within the epoch, and
  `on_epoch_begin` drawing nothing; the run's record (pre-fit and per-epoch order, fetches, maps,
  stream states, final state) equals the gate's reconstruction. A and B (same seed, separate
  processes) and C (three extra draws before the reseed) have identical per-epoch hashes (epoch 0
  `batch_index_seq_sha256` `fd56bc17303b…`, on both devices). Controls: D (no reseed) differs from A;
  F (no reseed, 100 extra draws) differs from D in every epoch; A differs from the reconstruction
  under `prefit_reseed(2)`. On PCG the draw ran on the RTX 4060 Ti with Dataset device `gpu:0`.
  Pending (B5, B6): the same hashes across all arms of a seed.
- **G12** (`gates/g12_stream_guard.py`, PCC, 35 checks). Clean 3-epoch run: no lab and no released
  callback changed either stream; the callback order is the released one with the lab probe last;
  every overridden hook was wrapped (StopIf's `on_train_batch_end` among them) and every wrapped
  epoch hook ran 3 times without a change; the guard was armed in `Random(prefit_reseed(1))`'s state
  and disarmed in the run's final state, which equals the reconstruction of the Dataset's and the
  data adapter's draws alone, so nothing else drew between the reseed and the end of `fit`. Controls,
  each stopped by `StreamViolation` (exit 3) in epoch 0, naming itself and no released callback: a lab
  callback calling `random.random()`; one constructing an hgq Dataset during training (it draws,
  `dataset.py:70,77`); one drawing from NumPy's global generator; a draw inside the watched stand-in
  for the frozen-evaluation step. Pending (B7): the frozen evaluation itself.
- **Tests** (not a gate; `preflight/gates/tests-b4/af5c2a20c28d/pcc/` and
  `preflight/gates/tests-b4/07977dfe1ebe/pcc/`; a later `run_all.py` must skip this directory).
  `tests/test_rng.py` through the same ephemeral `uv run --with pytest==8.4.2` overlay as B1 and B2,
  with `CBL_TEST_PYTHON` the venv of record and `TMPDIR` under scratch: 15 passed at `af5c2a20c28d`,
  17 passed at `07977dfe1ebe` (bundle sha256 `7c188290c828fab88c62cb9fb299568f5bc277a37366aa20d3477929f81c6fed`,
  PC HEAD equal, clean tree). The two added cases fit a tiny Keras model on synthetic arrays with
  `steps_per_execution=4` and 8 batches (a multiple of 4, as the archive of record's 200) and 7
  batches (as the smoke archive); in both the recorded fetches, maps and stream states equal
  `reconstruct_fit_stream`, and the final stream state equals its end state. `uv pip freeze` of
  `cbl1016-jax2` equals `env/training/freeze.txt` after both runs.

### Findings (none fails a gate; 2 and 4 are category C wording notes for the STUDY, no design change)

1. **Pre-fit draws.** Keras's symbolic build draws one batch-order shuffle and fetches two batches
   before epoch 0, and the iterator's reset reshuffles the batch map once more
   (`keras/src/trainers/trainer.py:1104-1110`, `backend/jax/trainer.py:418-419`). STUDY §6's account
   of two permutations per epoch from one stream holds for every epoch; these two pre-fit draws come
   on top, after the reseed, identically in every arm. The two pre-fit fetches are not training fetches
   and are recorded apart from epoch 0 (B4-4).
2. **Without the reseed, draw-count differences may or may not change the batch order.** Run E (no
   reseed, one extra `random.random()`, two 32-bit words) fetched different pre-fit batches from D
   but the same batches in all three training epochs, on both devices. `random.shuffle` draws through
   rejection sampling, so the number of words a shuffle consumes varies, and a small offset can be
   absorbed. STUDY §6 step 5 says arms with different draw counts "would get different batch orders"
   without the reseed; in general they do (run F), but it is a matter of chance. The reseed removes
   the question. A category C wording note for the STUDY, not a design issue.
3. **hgq quantizers draw from NumPy at construction** (`hgq/quantizer/internal/base.py:83`,
   identical in the snapshot and in 0.1.9, G04). The value seeds a generator only for stochastic
   rounding modes (`fixed_point_quantizer.py:56-57`); no CH-8 quantizer has a seed generator (G10;
   its round modes are RND and RND_CONV, a development reading), so these draws affect nothing. Arms with different quantizer counts leave NumPy in different states,
   which nothing in training reads; the guard holds NumPy's state fixed during training as well.
4. **For resume (M14, B10).** The Dataset's batch map is state of its own, not part of the `random`
   state that STUDY §6 step 8 and the §13 resume row name; and a fresh `fit(initial_epoch=k)` repeats
   the pre-fit draws of finding 1, which an uninterrupted run made once. For G48 (resumed equals
   uninterrupted) M14 must persist the map and restore stream and map after the symbolic build, for
   example in a callback's `on_train_begin` (`backend/jax/trainer.py:437`), arming the guard after
   that restore. Written into `code/docs/interfaces.md` §10 for B10; not implemented or gated here.
   For the STUDY this is a category C wording note, like finding 2: §6 step 8 says the persisted
   `random` state "restores both consumers, because both draw from it", and the §13 resume row's
   list of persisted state does not name the batch map. The design (resumed equals uninterrupted)
   is unchanged.
5. **CPU and GPU draws differ** in the 11 kernels only: 4,011 of 7,656 kernel elements, at most
   2.98e-8 apart (seed 1, smoke archive, PCC against PCG). This repeats `preflight/env/README.md`
   finding 2 and is why every run loads a file drawn on its own device (U1). The batch order is the
   same on both devices, because the stream is pure Python and both builds draw the same number of
   initializer seeds.
6. The vendored StopIf overrides `on_train_batch_end` (`vendor/stop_if.py`), so Keras dispatches the
   released callbacks' batch-end hooks synchronously; the guard wraps only overridden hooks and
   leaves that choice unchanged.

### Gate design changed during development

G11's control E first had to differ from D. A development run of uncommitted code found that E's
training epochs equal D's (finding 2), so the pass condition moved to F (100 extra draws, 200 words,
an offset the few shuffles of a 3-epoch run cannot absorb), and E is recorded but not a pass
condition. The rule changed after a development output, before any run of record, and is stated in
the G11 docstring; the run of record reproduces the case on both devices (E against D: pre-fit
fetches differ, training epochs equal).

### B4 choices

```
B4-1 DECISION: roles are built by walking objects (top-level semantic name, one segment per tracked
          sublayer, the variable's own name), never from Keras paths; CH-8's 14 top-level names are
          pinned against its class sequence (model.py:190-208) and its 244 variables.
CONSEQUENCE: M5 must give F000's layers CH-8's names, and M6 declares each binary arm's unmatched and
          unused roles in a RoleSpec; G10 checks the declarations when those arms exist.
ALTERNATIVES: Keras variable paths with counters stripped (depends on build order within a process);
          positional order of model.variables (breaks when an arm adds or removes a tensor).
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO
```

```
B4-2 DECISION: the initial-tensor file holds every variable (244 on CH-8: kernels, biases, BN gamma
          and moving moments, quantizer b, i, f, k, i_decay_speed, the per-layer beta and the EBOPs
          counters), not only the trainable or the random ones.
CONSEQUENCE: "every tensor equal at the start" is one hash comparison; the non-random tensors are
          constants, so loading them changes nothing on CH-8 (G10: only the 11 kernels differ by seed).
ALTERNATIVES: trainable variables only; the 11 random kernels only (BN initial state then rests on
          constant initializers, not on the file).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B4-3 DECISION: the loader refuses any difference in the manifest's identity block: seed, device
          platform and kind, Dataset device, package versions, matmul precision, sha256 of
          cbl/rng.py, cbl/env.py, cbl/data.py, chang/model.py and chang/data.py, data identity and
          the Datasets' construction draws. The config field init_tensors is {manifest_path,
          manifest_sha256}, the sha256 null in generated configs and supplied by the stage handoff
          after the B19 draws; the runner refuses null (interfaces.md section 5).
CONSEQUENCE: a change to any of those five files after the draws of record requires new draws; B18
          and B19 repeat on any code change anyway (plan section 4, B19). A run on a GPU product other
          than the drawing one is refused, which is what U1 intends. The device block also compares
          n_devices (the length of jax.devices()): the B19 draw Jobs must see the same number of
          devices as the pack-runner pods (B17, M20), or a run is refused for a reason with no
          physics content; the alternative is to record n_devices without comparing it.
ALTERNATIVES: compare the code sha only (refuses on unrelated commits); record without refusing.
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: YES (it fixes how U1, itself flagged, is enforced)
```

```
B4-4 DECISION: the two fetches of Keras's symbolic build before epoch 0 are a separate pre-fit record;
          the per-epoch batch_index_seq_sha256 covers only the training fetches of that epoch.
ALTERNATIVES: count them in epoch 0 (its hash would then cover batches not trained on).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B4-5 DECISION: the guard also holds NumPy's legacy global state fixed for lab code during training,
          beyond STUDY section 6 step 6's module-level random stream; Chang's released callbacks are
          wrapped and recorded, not stopped.
ALTERNATIVES: random only; also stop on a released callback (none drew in G12).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B4-6 DECISION: SeedFlow does not repeat run_train.py:42's random.seed(42) before the reader; step 2
          seeds random with s and nothing draws in between, so the line has no effect.
ALTERNATIVES: repeat it (no effect either way).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

### Not done in B4

- Pending parts, marked in the gate rows: G10's unmatched and unused sets of F000 and the binary arms
  (B5, B6); G11's identical hashes across all arms of a seed (B5, B6); G12's frozen evaluation (B7).
- G11 and G12 ran with the child's copy of the `run_train.py:79-100` recipe, not with the runner,
  which does not exist yet (M13, B5); both rerun with the runner, and every gate reruns at the final
  sha (B18).
- G10 ran on the CPU only, as the plan row says; the GPU draw path runs in G11 PCG.
- G11 ran on the smoke archive's 7 batches per epoch, which is not a multiple of `steps_per_execution`
  (4); the archive of record has 200 = 50 x 4. That branch is covered by the 8-batch test case at
  `07977dfe1ebe` (a test, not a gate) and by the reading of `backend/jax/trainer.py:227-250`; a gate
  run on real-data epochs of 200 batches belongs to B13's GPU runs on the archive of record.
- The initial-tensor draws of record (U1) belong to B19 on NRP; no file of record exists.
- Jev (`jev-lab` MCP) is down: `lab_check_protocol` and `jev_check_methods` were not run. No config
  exists yet, so there is no config drift to check.

### Blinding (B4)

None of the excluded paths was opened, listed by content or searched. Read: the B4 brief and the
common rules; `RULES.md`; `docs/methodology/03-phases.md` and the PREFLIGHT parts of `06-review.md`;
STUDY v0.8 heading list, §5-§7 (lines 654-859) and §13 (1305-1352); PREFLIGHT-PLAN in full;
`code/docs/interfaces.md`; `cbl/env.py`, `cbl/data.py`, `gates/gatelib.py`, `gates/g03_precision.py`,
the head of `gates/g07_reader.py` and of `tests/test_env.py`; Chang's `run_train.py`, `data.py` and
`model.py`; the snapshot's `dataset.py` and the cited lines of `quantizer/internal/base.py` and
`fixed_point_quantizer.py`; G04's `refdiff.md` and three file entries of its `report.json`;
`preflight/env/README.md`; this file. A listing of `preflight/` displayed the names
`da4ml-evidence-hashes-2026-10-17.txt`, `da4ml-perpos-evidence-hashes-2026-10-17.txt` and
`da4ml-test` (names only). No hardware-test outcome was seen.

## Lane M merges, second round (orchestrator, 2026-10-18)

- After B4 (`07977dfe1ebe`): `950aad0` merges `g52fix` (head `85d6e0f98c7a`), `80c7960` merges `b3b` (head `b2290f3555b1`)
  and `60d5e0d` merges `b15m` (head `d868cad9a07d`). Conflicts only in `docs/interfaces.md`: b3b's section 10 became 11
  and b15m's section 9 became 12, each with a note (`cbl/select.py:311` and `da4ml_clean/` still cite the old numbers).
  No code was edited in the merges.
- Rows and owner decisions: `preflight/gates/G53/b2290f3555b1/PREFLIGHT-rows.md` (B3b-1 to B3b-30; the v0.8 G54 FAIL and
  `G53/1784511c0f34/PREFLIGHT-rows.md` are kept, the latter cited by STUDY v0.9-v0.10), `preflight/gates/G52/85d6e0f98c7a/
  PREFLIGHT-rows.md`, `preflight/gates/G72/d868cad9a07d/PREFLIGHT-rows.md` (B15m-1 to B15m-12; superseded runs at
  `377b8176aec0` and the failing `521c2eae2a93` kept).
- STUDY v0.9 and v0.10 (pre-data amendment from G54; critical review PASS at v0.10, `review/critical-v0.9-amendment.md`):
  the build's STUDY of record is v0.10 (`7180c24f…`); the M16 gates parse v0.9, whose §6 equals v0.10's.
- Orchestrator checks at the merged sha `60d5e0df3599` on the Mac (not evidence of record): G01, G25, G52 (wandb 0.28.0),
  G53, G54, G55, G56 (Mac part, `--study` v0.9) and G72 all PASS.

## B5 (PREFLIGHT-PLAN §4 row B5): leftovers G56 and G52 on the PC, M13 runner, M5 all-Q model, gates G13-G15

**What now works.** The training runner (M13, `cbl/runner.py`) runs Chang's released recipe for an
arm from a config, and for CH-8 it reproduces Chang's own `run_train.py` bit for bit: run on the same
device from the same seed, the two give the same initial weights, the same data, the same batches in
every epoch, and the same loss, gradients, weights, batch-normalization statistics, per-layer EBOPs
and outputs after every epoch. F000, the lab's layer-by-layer rebuild of Chang's model (M5,
`cbl/model.py`), equals CH-8 in the same sense: same structure, same initial draw, same forward and
backward pass, same optimizer step, same cost count and the same three-epoch run. The sealed held-out
reader now also requires M16's complete freeze manifest, and the W&B gate passes under the lock's
wandb 0.31.0. Main limitation: on the RTX 4060 Ti these equalities hold only with XLA's deterministic
GPU operations switched on for both sides; under the pinned stack's default flags two runs of the same
program on the same GPU already differ after the first gradient (finding 1, decision B5-1), which
bears on the NRP determinism record and is Kai's question. Only CH-8 and F000 are built; the binary
arms are B6's, the frozen evaluation B7's, the PID B9's.


### Owner decision on the stack of the bitwise gates (written 2026-10-18T10:55Z, before the gates of record)

STUDY v0.10 §13 (reference-fidelity row and "Stack of the bitwise gates") leaves to the PREFLIGHT owner
on which device and stack the bitwise runner gate runs, to be recorded before it runs. This block is
that record. It was written after development runs at `4d23192` (PC scratch `b5-dev/d1g`, not evidence)
showed two things on the RTX 4060 Ti: under the pinned stack's default XLA flags, two processes of the
same runner program on the same initial tensors and batches diverge from the first training-step
gradient on, in 38 to 40 of 244 variables after each of three epochs (the CPU does not diverge); and
with `XLA_FLAGS=--xla_gpu_deterministic_ops=true` set on both sides, the runner and Chang's
`run_train.py` agree bitwise in every record. Engineering readings, not quotable.

```
B5-1 DECISION: G13 and G14 are bitwise gates on PCC (PC CPU, cbl1016-jax2, default flags) and on PCG
          (RTX 4060 Ti, cbl1016-jax2) with XLA_FLAGS=--xla_gpu_deterministic_ops=true set on both
          sides of every bitwise comparison. On PCG each gate also runs its arms again under the
          default flags and records their comparison as the GPU determinism control: a finding,
          not a pass condition. Neither gate runs on the production GPU product; that is STUDY
          section 13's determinism record (NRP, B19).
CONSEQUENCE: the gates show that the runner's program and F000's program are the release's on both
          devices; they do not show, and the PCG controls are expected to refute, run-to-run
          bitwise determinism of training on the GPU under the default flags. The NRP determinism
          record will then take its "not identical" branch (F000 - CH-8 measures run-to-run
          divergence, STUDY section 13) unless the production stack sets the flag. Setting it in
          production is a change to the STUDY section 7.3 stack (only the float32 matmul override is
          declared there), with a throughput cost not measured here: Kai's or the STUDY's, not
          decided in B5.
ALTERNATIVES: require bitwise GPU equality under the default flags (fails by construction); drop the
          PCG rows (the plan names PCC and PCG); set the flag in the training stack now (an
          undeclared runtime departure).
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: YES
```

### How it works

- **The runner (M13).** `prepare` validates the config (interfaces.md §13), checks that the fork's
  HEAD is the config's `code_sha` with a clean tree, runs M1's setup, imports Chang's `run_train`
  module (hash-checked) for its schedule function, restores the memory fraction that module's top
  level overwrites before the JAX backend starts, and then follows M4's seed flow: M3's reader,
  `random.seed(s)` and `keras.utils.set_random_seed(s)`, the two Datasets with the device the runner
  passes (U2), the model (Chang's `get_model` for CH-8, M5 for F000), the seed's CH-8 file loaded by
  role (U1). The released callbacks and compile are `run_train.py:79-100` with Chang's literal
  arguments restated (they sit inside `__main__`), followed by the appended frozen-evaluation stub.
  `fit` reseeds (U5) with the stream guard armed and trains. W&B goes through M15 (offline or
  disabled). The declared departures are exactly the five of the plan's M13 row; nothing else in the
  recipe differs, which G13 checks against the live release.
- **The appended callback.** A stub of M8 in the place STUDY §7.1 gives it: it writes the epoch's
  batch-index hashes into the logs under new keys, runs an evaluator inside the stream guard's watch,
  refuses if any key a released callback wrote has changed, and keeps a per-epoch record. M8 (B7)
  plugs in as the evaluator.
- **The lab builder (M5).** Chang's six scopes (`get_model`'s three, then `get_transformer`'s three)
  are entered in his order and his layers constructed and called in his order, with a kernel choice per
  group at three hook functions. Under Q each hook builds Chang's own layer; under B it raises until M6
  exists. Because Keras draws an initializer seed from Python's `random` when a layer's initializer is
  made, the same classes in the same order draw the same seeds, so F000's own draw equals CH-8's.
- **The gates.** Every child is a fresh process (`gates/runner_child.py`) under the kernel open-file
  audit of B2. The probe that stands in for M8 hashes every variable by role, the optimizer state, the
  gradient of the release's loss on a fixed batch of 2,790 training jets, the state a training forward
  would write, and the logits, and checks that it changed nothing. G13's reference side is Chang's
  `run_train.py` run by `runpy`, byte-unchanged, with a harness that only remaps the Dataset device on
  the CPU (as the env README's smoke runner did), sets the epochs, appends the same stub and applies the
  same pre-fit reseed. The 2-file smoke archive is used throughout; the held-out archive is watched and
  never opened.

### Gates of record at `46cc25fa9e48` and, for the leftovers, at `78c5515df8da`

The Mac bundle `preflight/pc-setup/code-main.bundle` was rebuilt at each sha and applied on the PC with
`preflight/pc-setup/b5-pc-sync.sh` (bundle sha256 check, `git bundle verify`, `git fetch`, `git reset`,
HEAD and clean tree printed); at `78c5515df8da` the bundle sha256 was
`6668197efa580c20d76174190a4067ecf19cf5717321b23cad6a23f583abf82f`, at `46cc25fa9e48`
`422fdaa6e8230c106fca6f721503da7ffc01d3467f74e32eaabac429794748d4`; PC HEAD equal and the tree clean
both times. Every gate checks HEAD, a clean tree at start and end and the venv of record against its
freeze (G52 through the overlay checks the freeze before and after instead). Evidence directories
hold `command.txt`, `interpreter.txt`, `stdout.txt`, `stderr.txt` (empty for every gate process),
`exit_code.txt`, `result.json` where the gate writes one, and `children/` (each child's argv, JSON
record, stdout, stderr and open-file audit). The children's stderr holds PBar's progress lines with
loss and accuracy on the smoke archive (engineering, not quotable). Every reading below is
engineering, not quotable.

- **G56 PCC part** (`gates/g56_freeze.py --pcc`, 43 checks). The Mac part's checks of the section-8.5
  list and of `cbl.select`'s writer and reader, then: `read_heldout`'s first statement calls
  `check_freeze_manifest`, which calls M15's `freeze_manifest_sha256` and M16's
  `require_freeze_manifest` (static); under the kernel open-file audit the reader of record refuses nine
  manifests on the held-out archive of record (none, missing, invalid JSON, empty object, JSON list, an
  object that passes M15's rule only, a complete manifest without `best`, one whose content no longer
  matches its hash, one with the claim domain unset), the last four by M16's completeness check; the
  audit window holds no open of the held-out archive or any raw held-out file; with a complete
  synthetic manifest (M16's fixture) the reader refuses a placeholder of 16 zero jets under a wrong
  expected sha256 and returns its 16 jets under the right one, naming the manifest's file sha256 and its
  content sha256; the reader of record and `SealedReaderStub` accept and refuse the same ten manifests.
- **G08** (36 checks) reran because its positive control needed a complete manifest after the join; it
  adds the M15-only refusal. **G07** (33 checks) reran unchanged (B5-14).
- **G52 PCC** (77 checks) passes under wandb 0.31.0 with g52fix's standard-library reader: every
  required section-12 key in the clean and decoy children's offline records, project and group, the
  key never read, the held-out refusal end to end; the record header's producer build id
  (`0.31.0+15f8cbc5…`) is the one allow-listed 40-hex token. `cbl1016-jax2`'s freeze is unchanged before
  and after (both diffs empty).
- **G13** (`gates/g13_runner_ref.py`, PCC 51 and PCG 56 checks). Seed 42 (B5-2). The release's own
  draw at fit entry equals the runner's file in all 244 roles, and the runner's own draw before its load
  equals the file. M3's reader gives the same four arrays as Chang's `get_data`; the Datasets' shifts,
  maps and batch counts and the model's 118 flattened layers and 244 roles are the same. The callback
  order is the released six plus `FrozenEvalHook`; each restated argument equals the release's live
  object (PBar template, ParetoFront metrics, sides, file-name format and filter on 20 inputs, beta and
  learning rate at 16 epochs from 0 to 7001, the StopIf condition), and the optimizer and loss
  configurations and `steps_per_execution` agree. In each of 3 epochs the fetched batch-index hashes of
  both equal the reconstruction from `prefit_reseed(42)` (epoch 0 `0c31199efadb…`), and the logs, all
  244 variables (weights, BN moving statistics, per-layer EBOPs, quantizer state), the optimizer
  variables, the gradient by role, the training-mode logits and state and the inference logits are
  bitwise equal; so is the initial-state probe. The runner's guard saw no change of the global streams,
  matmul precision is float32 before and after fit, and M15 opened the disabled run
  `CH-8_nf_s42_gate_ramp_a1_src-<bundle>`. On the CPU Chang's script asked for `gpu:0` twice and the
  harness passed `cpu:0` (the smoke-only remap); on the GPU it passed `gpu:0`. Chang's `get_data`
  opened the placeholder `150c-test.h5` and the runner opened no `150c-test.h5`; no child opened a
  held-out path. Controls: run-B equals run-A in every record; ref-N (no reseed) starts from the same
  weights and differs in batch order from epoch 0 and in every epoch's variables.
- **G14** (`gates/g14_f000.py`, PCC and PCG 42 checks each). F000 and CH-8 have the same 14 top-level
  classes, 118 flattened layers with equal configurations (49 internal quantizers have no
  `get_config`; their classes and variables are compared) and 244 roles with equal shapes, dtypes and
  trainable flags. F000's own draw equals CH-8's file in all 244 roles; it loads them with empty
  unmatched and unused sets. At the initial state the inference logits, training-mode logits, loss,
  gradient by role, the state a training forward writes and its per-layer EBOPs (20 layers) are equal;
  after one `train_on_batch` the loss and accuracy, every variable, the optimizer variables (iterations
  1), the per-layer EBOPs and FreeEBOPs's sum (1,506,839 on the smoke batch, engineering) are equal,
  and the counters equal the training forward's values. In the 3-epoch fit the batch hashes of the two
  arms equal each other and the reconstruction from `prefit_reseed(1)`, and every per-epoch record is
  equal. On the CPU an F000 run logging through M15 in offline mode wrote its offline run file and
  equals the disabled run in every record.
- **G15** (`gates/g15_mha_wrap.py`, PCC, 77 checks). In CH-8, F000 and CH-8 built again in the same
  process the MHA is a `QMultiHeadAttention`; its four projections hold the variables at
  `mha/{query,key,value,attention_output}/…` (kernels (24, 2, 16) three times and (2, 16, 24); a bias
  for Q, V and O, none for K), are configured by the MHA's own `_qkvo_*_conf` objects, and their kq
  configuration equals `_qkvo_kq_conf` field by field; the 89 roles under `mha/` are the same in all
  three although the layer names carry different counters. Every QLayer class of the three models
  (8 classes) carries exactly one QLayerMeta wrap on `call` and one on `build`, and F000's table equals
  CH-8's; a training-mode call adds the same 89 losses in each. Controls made after QEinsumDense was
  instantiated: a subclass inheriting both methods shows 2 and 2, one defining only `call` shows build 2,
  one defining both shows 1 and 1; called in training mode, the doubly wrapped call adds 4 losses
  against the plain layer's 3 (the EBOPs penalty twice).
- **G10** (81 checks) and **G11** (PCC, PCG, 82 checks each) as in B4, plus: F000 built by the runner
  loads draw-s1 with 244 tensors, empty unmatched and unused sets and its own draw equal to the file
  (G10); CH-8 and F000 through the runner fetch the same batches in every epoch, equal to the
  reconstruction and to B4's run A (G11, both devices).
- **Tests** (not a gate; `preflight/gates/tests-b5/46cc25fa9e48/pcc/`). `tests/test_data.py`,
  `tests/test_freeze.py` and `tests/test_runner.py` through the same overlay with `TMPDIR` under
  scratch: 69 passed; freeze unchanged before and after. On the Mac (`mhab1012-cpu`, not evidence):
  `test_data.py` 14 passed and 1 skipped (inotify), `test_runner.py` 29 passed. A later `run_all.py`
  must skip this directory.

### Findings (none fails a gate; finding 1 needs Kai)

1. **The pinned stack is not run-to-run deterministic on the RTX 4060 Ti under the default XLA
   flags.** In G13 PCG, run-C and run-D, the same runner program on the same file and batches in two
   processes, already differ in the gradient of 35 of the 97 trainable variables at the initial state;
   after each of the three epochs 39, 40 and 43 of 244 variables differ (BN moving statistics among
   them). G14 PCG shows the same between CH-8 and F000 under the default flags (76, 80 and 85
   variables). The counts change from one repetition to the next (the development diagnostic of the
   B5-1 block found 38, 39 and 40), as nondeterminism predicts. The batch order is unaffected. With
   `--xla_gpu_deterministic_ops=true` every comparison
   is bitwise equal, so the differences come from the backend's nondeterministic GPU kernels, not from
   the runner or M5. The CPU is deterministic (G13 PCC run-B). Consequences: the STUDY section 13
   determinism record on the production GPU will very likely find CH-8 runs that are not identical, so
   F000 - CH-8 will measure run-to-run divergence there (a branch the STUDY already provides); section
   6's pairing of initial tensors and batch order holds, but bitwise paired trajectories on the GPU do
   not. Whether production should set the flag (a section 7.3 stack change, throughput cost unmeasured)
   is for Kai and the STUDY; see decision B5-1.
2. **hgq2 0.1.9 wraps `build` as well as `call`.** `QLayerMeta.__call__` wraps both once per class at
   its first instantiation (0.1.9 `base.py`; the snapshot wraps `call` only). A lab subclass of an hgq
   layer that inherits either method is wrapped twice: G15's control shows the EBOPs penalty added
   twice through `call`. B6's binary layers and binary MHA subclass (U3) must define their own `call`
   and `build` from the unwrapped originals, with their signatures preserved (a `*args` signature
   without `functools.wraps` loses Keras's shape inference and the layer fails to build, a development
   reading). Written into interfaces.md section 13.
3. **`keras.utils.set_random_seed` (section 6 step 2) also reseeds NumPy and clears Keras's global
   SeedGenerator.** Neither changes CH-8's draw: with s = 42 the runner's draw equals the release's own
   (G13), so the extra step-2 effects are inert for this model.
4. **Importing `run_train` has side effects** (memory fraction '0.5', both streams seeded with 42); the
   runner neutralizes both (B5-11). Recorded for anyone importing Chang's module elsewhere.

### B5 choices

```
B5-2 DECISION: G13 runs at seed 42, a gate-only seed. run_train.py seeds random with 42 before its
          Datasets and model (run_train.py:42, 73-76); the runner's flow with s = 42 seeds it with 42 at
          the same point, after a data read that draws nothing from random, so the runner's file drawn on
          the device must equal the release's own draw. G14, G10 and G11 use seed 1.
ALTERNATIVES: seed 1 with the release's draw replaced by a load of the runner's file in the harness
          (then the initial-weight check would be the gate's own assignment).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B5-3 DECISION: G13's reference side gets decision U5's pre-fit reseed in the gate's fit wrapper, so
          that both sides fetch the same batches; a reference run without it is a control (same initial
          weights, different batches from epoch 0).
ALTERNATIVES: compare only the initial state and one train_on_batch on a fixed batch.
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B5-4 DECISION: the runner accepts a null headline or claim domain only in the engineering stages gate,
          smoke, bench and det, and refuses them in S1, S3 and S4 (interfaces.md section 5, fail closed).
          B5 builds only CH-8 and F000, which train on the native count.
CONSEQUENCE: a binary arm in an engineering stage (B6 on) also needs the headline for its penalty (b);
          B6 tightens the rule to refuse a null headline for binary arms in every stage.
ALTERNATIVES: refuse null everywhere (no gate or smoke could run before B15); a placeholder value.
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO
```

```
B5-5 DECISION: engineering stages may set epochs from 1 to 7000; S1, S3 and S4 require 7000. The smoke
          archive is accepted only in engineering stages; the archive kind (record or smoke) is a runner
          input, and the expected sha256 comes from the committed data manifest.
ALTERNATIVES: an epochs override outside the config.
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B5-6 DECISION: the runner opens M15's run after step 1 of the seed flow (the provenance needs the split
          and order hashes) and before step 2's seeding, and logs with strict=False until M8 writes the
          section-12 keys (B7 sets strict=True). The bitwise gates use mode disabled, in which M15 still
          checks identity and keys; G14 on the CPU runs one F000 run in offline mode and compares it with
          the disabled one.
ALTERNATIVES: open W&B before step 1 (no split hashes yet); log nothing until B7.
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO
```

```
B5-7 DECISION: the runner refuses wandb.mode online during PREFLIGHT; the launcher batch (B17) lifts the
          refusal when the launch route and its credentials are decided.
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO
```

```
B5-8 DECISION: the runner's code identity is the fork's git HEAD, which must equal the config's code_sha,
          with a clean tree.
CONSEQUENCE: a pod without a git checkout (launcher route B, ConfigMaps, or a baked image) needs an
          equivalent identity (a tree hash checked against the handoff record), which B17 and M20 must
          supply before the runner can start there.
ALTERNATIVES: hash the runner's files only (already recorded in run.json as files_sha256).
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO
```

```
B5-9 DECISION: all five variant hooks are stubs that raise NotBuilt. None is a configuration of the
          all-Q path: FP32 needs a per-layer-kind definition of "quantizers off" (kernel, bias and
          datalane quantizers, the two tanh tables, the softmax tables that QSoftmax forces to SAT and
          RND_CONV) that the STUDY does not spell out; W8A8 is decision U6 (LOW, flagged); DS-HGQ is a
          different graph with its own unused set of initial tensors that G10 must gate; GR and BE are
          binary (M6, B6).
ALTERNATIVES: build DS-HGQ now as a graph option.
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B5-10 DECISION: F000's layers carry no names, so a fresh process gives them CH-8's automatic names; roles
          come from CH-8's semantic names pinned by the class sequence (rng.ch8_top_names).
ALTERNATIVES: semantic layer names (readable, but Keras paths and checkpoint layer names would then
          differ from CH-8's).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B5-11 DECISION: the runner imports Chang's run_train module unchanged for cosine_decay_restarts_schedule.
          Its top level sets XLA_PYTHON_CLIENT_MEM_FRACTION to '0.5' and seeds both streams with 42
          (run_train.py:4, 17-18); the runner restores its own memory fraction before the JAX backend
          starts and refuses if the backend has already started; the seeds are replaced by the seed flow.
ALTERNATIVES: copy the schedule function (then G13 would be the only check that it is Chang's).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B5-12 DECISION: Chang's get_data is imported with his module and never called; the runner reads train
          and validation through M3's reader (the declared I/O departure, STUDY section 4). G13 shows the
          arrays equal get_data's on the smoke archive, as G07 does on the archive of record.
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B5-13 DECISION: the gates' gradient is that of the release's loss (sparse categorical cross-entropy from
          logits plus the model's own losses) through model.stateless_call in training mode, not Keras's
          compiled train step: calling the compiled loss before fit would build its metrics early, and
          Keras's fit would then skip the symbolic build that makes the pre-fit draws (B4 finding 1),
          changing the batch order. The training path itself is covered by the per-epoch equality of
          every variable and by G14's train_on_batch step.
ALTERNATIVES: Keras's compute_loss_and_updates under jax.grad.
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO
```

```
B5-14 DECISION: after the G56 join (cbf8eb9, which changes cbl/data.py), G07 and G08 rerun (G08's
          positive control needed a complete manifest, so its gate changed too); G06 does not rerun in B5.
          G06 runs Chang's preparation of all 62 files and calls neither check_freeze_manifest nor
          read_heldout, the only functions the commit changes; B18 reruns it with every gate at the
          final sha.
ALTERNATIVES: rerun G06 now (about the B2 preparation time).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

### Not done in B5

- The binary arms (M6, B6): their RoleSpecs and unmatched sets (G10), their batch hashes (G11), their
  structure and wrap checks (G15), and the binary MHA subclass (U3).
- G12's frozen-evaluation part and the frozen evaluation itself (M8, B7); the stub logs with
  strict=False until then.
- The PID schedule (M12, B9); FP32, W8A8 and DS-HGQ (stubs).
- G06's rerun (decision B5-14); every gate reruns at the final sha in B18.
- Real-data epochs (200 batches) on the archive of record: B13. The GPU rows ran on the smoke archive.
- The production-stack question of decision B5-1 (deterministic GPU ops or not) is Kai's.
- `XLA_FLAGS` is not recorded by `cbl.env.provenance()` or in the runner's `run.json`. In the PCG gates
  the evidence that the deterministic flag was set in each child is the gate's argv line in `stdout.txt`
  and `main_env` in `result.json`, not the child's own record. After finding 1 it is the one environment
  variable that decides whether a GPU trajectory is reproducible: recording it in M1's provenance (and
  the production value in the handoff) is a one-line change for B7 or B13.
- Jev (`jev-lab` MCP) is down: `lab_check_protocol` and `jev_check_methods` were not run. No config of
  record exists yet (M17 is B12), so there is no config drift to check.

### Blinding (B5)

None of the excluded paths was opened, listed by content or searched. Read: the B5 brief and the common
rules; `RULES.md`; `docs/methodology/03-phases.md` and `06-review.md`; STUDY v0.10 heading list, §0-§2.3,
§4-§7, §12-§16 and the opening of §17 (lines 80-339, 628-866, 1291-1430); §3.4 was not read; PREFLIGHT-PLAN in full (its §0
blinding note names a fragment without reproducing it); this file; `code/docs/interfaces.md`; the fork's
`cbl/env.py`, `rng.py`, `data.py`, `wandb_log.py` (API), `select.py` (freeze-manifest functions), the
gates G07, G08, G10, G11, G56, `gatelib.py`, `m16gate.py`, `fileaudit.py`, `rng_child.py`; the B3b rows
(G56 notes) and the G52 fix rows; Chang's `run_train.py`, `model.py`, `data.py`; `vendor/stop_if.py`;
`preflight/env/README.md` and `preflight/env/scripts/cbl_smoke_train.py`; installed hgq2 and Keras
sources on the PC (`mha.py`, `base.py`, `dataset.py`, `ebops.py`, the sugar callbacks, `rng_utils.py`,
`seed_generator.py`, the JAX trainer). A listing of `preflight/` showed the names
`da4ml-evidence-hashes-2026-10-17.txt`, `da4ml-perpos-evidence-hashes-2026-10-17.txt` and `da4ml-test`
(names only). No hardware-test outcome was seen.

## B6 (PREFLIGHT-PLAN §4 row B6): M6 binary layers, M5 binary hooks, gates G16-G24 and the binary parts of G10, G11, G15

**What now works.** Every binary configuration of the screen builds and trains in the lab builder: F100,
the six configurations with a fused-BN binary site in both forms FB and BF, GR and BE (M6,
`cbl/binary.py`, through M5's hooks). Each binary kernel is exactly plus or minus one learned-width
magnitude per layer or per output column, in the training forward as in the exported kernel; no HGQ
kernel quantizer exists at a binary site; the gradients of every variable equal those of Chang's own
layer with its kernel quantizer replaced by the STUDY's map, the moment and bias paths included; the
penalty (b) of both headline branches sends the stated gradient to the scale quantizer's bits; the
binary arms load CH-8's initial tensors by role and fetch CH-8's batches; the runner admits them.
Main limitation: the scale treatment is OD-13, which Kai has not confirmed (STUDY section 19.2 item 2);
it is built under his delegation and isolated in one class (decision B6-0). Second: a binary arm now
needs the headline in every stage, so branch b12's smoke and bench configs for binary arms, generated
with headline null, are refused until section 3.4 fixes it (B6-1); the gates name a branch and test both.

**How it works.**
- **The map.** One class, `LearnedWidthScale` (`cbl.binary.SCALE`), holds the scale quantizer q_g and the
  three maps of STUDY sections 2.2-2.3: at non-BN sites K = q_g(sg(beta)) S + (W - sg(W)); in FB the same
  on U = W s with one beta per column; in BF q_g(sg(beta) |s_c|) sg(sign(s_c) S) + s_c (W - sg(W)); GR
  drops the stop_gradient on the scale's input. S is +1 above the mean and -1 elsewhere (Sign(0) = -1).
- **The layers.** `BinaryEinsumDense` and `BinaryEinsumDenseBatchnorm` re-express Chang's constructor and
  build without the kernel quantizer and build q_g in its place (decision B6-3); the fused-BN layer runs
  Chang's own call, so batch moments, moving statistics, the bias fold and the activation are his code,
  and only the kernel line of `get_fused_qkernel_and_qbias` is replaced. `BinaryMultiHeadAttention`'s
  build is the installed build with binary projections (U3, B6-4). Each class defines its own call and
  build, calling only unwrapped ancestors, so hgq2 0.1.9 wraps each once and `qkernel` is computable when
  its build wrapper evaluates it. The classes register for Keras serialization themselves.
- **The penalty.** A binary layer's `_compute_ebops` returns penalty (b): under H1 Chang's contraction with
  q_g's bits in place of kq's; under H2 the per-column min of section 3.2 with q_g's bits in place of the
  exact width (B6-1, B6-2).
- **The builder and runner.** M5's hooks build the binary layer under B with the arm's form, its headline
  and the GR flag; the binary arms' RoleSpecs map CH-8's tensors by role, with q_g's variables unmatched
  and kq's unused (B6-6). The runner admits the binary arms (B6-7) and records XLA_FLAGS in its
  provenance (B5's note).
- **The gates.** G16-G24 run one fresh child each (`gates/binary_child.py`) on constructed tensors from
  private generators, under the open-file audit of the held-out paths; the GPU rows set XLA's deterministic
  ops (B5-1). G19's reference is Chang's released layer with a gate-written map in place of kq (B6-8).
  G10, G11 and G15 gained binary parts through the runner and the lab builder.

### Design status of the scale treatment (written 2026-10-18, before the gates of record)

```
B6-0 DECISION: B6 implements STUDY decision OD-13 as adopted: the exported magnitude passes through a
          learned-width HGQ quantizer q_g built in the layer's __init__ inside Chang's scopes from the
          kernel configuration of the site, one (b, i) per layer at non-BN sites and per output column at
          fused-BN sites; in BF it quantizes beta |s_c|, after the fold.
CONSEQUENCE: Kai has not confirmed OD-13 (STUDY section 19.2 item 2); it is built under his overnight
          delegation. Everything that depends on the choice is one class, cbl/binary.py
          LearnedWidthScale (used as cbl.binary.SCALE): creation and granularity of q_g, the scale's
          input and its quantization in the three maps (non-BN, FB, BF, with GR), the bits the penalty
          reads, and the per-site record. A different scale treatment replaces that class; G16, G19,
          G20, G22-G24 then rerun, and PREFLIGHT-PLAN section 4's estimate for a rejection (about 10 h)
          shrinks to the class, its gates and B7's counts.
ALTERNATIVES: the v0.2 unrounded float32 beta (rejected in the STUDY: an unpriced scale); quantizing
          beta before the fold in BF (STUDY section 2.3: BF's constants would stay about 24 bits wide).
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: YES (STUDY section 19.2 item 2)
```

### Owner decisions written before the gates of record

```
B6-1 DECISION: the in-trace penalty (b) is implemented in M6 for both branches (cbl.binary.penalty_b, with
          h2_kernel_term and u_bound), selected per layer by the config's headline; M7 (B7) adds the
          independent recompute from quantizer variables (G30, G33) and may move the function.
CONSEQUENCE: a binary layer cannot be built without a branch, so a binary arm needs the headline in every
          stage (decision B5-4's consequence, applied in cbl.model.arm_spec). Gates name a branch and test
          both. Branch b12's smoke and bench configs for binary arms carry headline null and are refused
          until the headline is set; the orchestrator should know this at the merge.
ALTERNATIVES: H1 only until section 3.4 decides (the H2 branch untested until B7); a default branch
          (forbidden: no decision input has a default, interfaces.md section 5).
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO
```

```
B6-2 DECISION: in the H2 penalty, u(t)'s integer part ceil(log2 sum_i 2^I) is computed from exact powers
          of two as M + ceil(log2 sum_i 2^(I_i - M)), M the largest integer bits among the included
          inputs, with no backend log2 or pow, and carries no gradient; F's maximum carries it (STUDY
          section 3.2). Exact when the included integer bits span at most 24 - ceil(log2 n_in); Chang's
          datalane constraint ic = MinMax(0, 12) bounds the span by 12. Whether a column is zero is read
          from q_g's rounded bits, and which branch of the min is active is a comparison; neither has a
          gradient. Ties go to H1, as cbl/cost_pure.py.
ALTERNATIVES: ops.log2 and 2.0**I (an inexact GPU pow could move the ceiling by one, the risk STUDY
          section 3.2 names for the grid).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B6-3 DECISION: both binary kernel layers re-express QEinsumDense.__init__ (einsum_dense.py:29-36) and
          build (:46-53) with q_g in place of the kernel quantizer, calling only unwrapped ancestors
          (QLayerBaseSingleInput's build and get_config); the fused-BN layer calls Chang's call
          (einsum_dense_batchnorm.py:185-211) unwrapped and unchanged and replaces only
          get_fused_qkernel_and_qbias (:158-173) at lines 167-168; the non-BN layer re-expresses
          QEinsumDense.call (:55-66), whose kq call is inline. `kq` raises AttributeError at a binary
          site; q_g's configuration is saved under `kq_conf`, the constructor argument it is built from.
CONSEQUENCE: no kq object exists at a binary site (G18); each class is wrapped once (G15); the moment and
          bias paths are Chang's code (G19). A converter that reads `layer.kq` meets an AttributeError
          (B14's G65 decides the export path; the alkaid plugin reads qkernel and qbias).
ALTERNATIVES: build kq and delete it (STUDY section 2.2: binary layers must not create one); a dummy kq.
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B6-4 DECISION: BinaryMultiHeadAttention.build and _make_output_dense are copies of the installed 0.1.9
          QMultiHeadAttention.build and _make_output_dense with QEinsumDense replaced by BinaryEinsumDense,
          checked equal to the installed source by syntax tree (G22); the projections receive the penalty
          branch and the GR flag through _get_common_kwargs_for_sublayer; call is Chang's, unwrapped.
ALTERNATIVES: rebind hgq's module-level QEinsumDense during the stock build (a runtime patch of hgq, an
          undeclared third shim, STUDY section 7.3); replace the projections after the stock build (U3's
          rejected alternative).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B6-5 DECISION: GR is the GR flag of the three maps at every binary site of F111 (the MHA projections
          included): no stop_gradient on the scale's input (on beta in BF), STUDY section 5. BE binarizes
          the embedding with its own q_g in the arm's form; GR and BE need groups B, B, B.
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B6-6 DECISION: M5 gives the binary arms CH-8's top-level names pinned by the arm's own class sequence;
          the unmatched set is q_g's k, b and i and the unused set kq's k, b and i at the binary sites;
          244 variables for every arm.
CONSEQUENCE: STUDY section 6 step 4 says "b and i"; q_g's uint8 k (constant 1, non-trainable) is also
          unmatched. Recorded as a finding for the STUDY C-item pass (wording), not a design change.
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B6-7 DECISION: the runner admits F100, F001-F111 (FB or BF where a fused-BN site is binary), GR and BE
          (cbl.runner.BUILT_ARMS); GR and BE only in engineering stages, since their campaign form is the
          S2 choice and the runner does not read the S2 record yet. PID-BEST, BEST-C and PID-BEST-C stay
          refused, as do FP32, W8A8 and DS-HGQ. BUILT_IN_B5 is kept unchanged because branch b12's G57
          reads it.
CONSEQUENCE: b12's G57 asserts that every non-CH-8, non-F000 config is refused "only as not built yet";
          after the merge that assertion fails for the binary arms by design, and G57 needs the new list.
ALTERNATIVES: admit GR and BE in S3 with form_source checked against the S2 record (M16's reader, not
          joined to the runner yet).
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO
```

```
B6-8 DECISION: G19's reference is Chang's own layer (QEinsumDense, QEinsumDenseBatchnorm) with its kernel
          quantizer replaced before build by a gate-written map written from STUDY sections 2.2, 2.3 and 5
          (gates/binary_child.py RefMap, which also serves the stock _compute_ebops through bits_); for
          BF a copy of get_fused_qkernel_and_qbias with lines 167-168 replaced, because the stock method
          hands kq only U = W s. Comparison criteria fixed before any run: exact numeric equality of every
          element (byte equality recorded beside it); the two analytic recomputations (GR's added
          derivative at non-BN sites, G20's end-to-end penalty gradient) within 1e-5 of the largest
          magnitude. On the GPU every child runs with XLA_FLAGS=--xla_gpu_deterministic_ops=true (B5-1).
ALTERNATIVES: an analytic reference for the full gradient (the moment path makes it long and itself
          error-prone); comparison within a tolerance (would hide an ordering difference).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B6-9 DECISION: G24's exactness check is the STUDY's: every q_g output lies exactly on its grid (q 2^f an
          integer, |q| <= 2^i - 2^-f) and every on-grid input comes back unchanged; the stricter comparison
          with exact rational RND of arbitrary float32 inputs is recorded beside it with the one kind of
          deviation it showed.
CONSEQUENCE: the restatement was made after a development run (uncommitted gate code, PC scratch
          b6-dev/, not evidence), in which the stricter comparison found that HGQ's float32 RND,
          floor(x 2^f + 0.5) 2^-f, returns 2^-f instead of 0 for inputs within one float32 ulp below 2^-f/2
          (x 2^f + 0.5 rounds to 1), at every b. Such an output is still on the grid, so neither the
          exported kernel's exactness nor w(q) is affected; it is HGQ's rounding, shared with every Chang
          kernel quantizer, and changes at most which grid value a magnitude in that one-ulp interval
          takes. The gate checks that no other kind of deviation occurs. The STUDY's wording ("q_g's own
          output is exact on its grid", section 2.2; critical C16; stage_b_build.py:45-48) is unchanged.
ALTERNATIVES: require exact RND of every input (fails at every b for this reason, with nothing to fix in
          the lab's code); drop the stricter comparison (would hide the finding).
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO (listed for the STUDY C-item pass)
```

### Gates of record at `b2838835b733`

The Mac bundle `preflight/pc-setup/code-main.bundle` (sha256
`46f1d217bf3720dcbcb3e9d414a50ef5f4f5755edb725b51f74a4bd75bb4617b`) was applied on the PC with
`preflight/pc-setup/b5-pc-sync.sh`; PC HEAD `b2838835b733f052f4fae829563178899993f7a0`, equal to the Mac's,
clean tree. All gates ran through `~/lab/scratch/cbl1016/b6-run-gates.sh` (operator code, below), which
checks HEAD and a clean tree and calls `run_gate.sh` per gate; every gate checks HEAD, a clean tree at start
and end and the venv of record against its freeze. Evidence directories hold `command.txt`,
`interpreter.txt`, `stdout.txt`, `stderr.txt` (empty, except G03 PCG's two CUDA driver-version lines, as in
B1), `exit_code.txt`, `result.json` and `children/` (argv, record, stdout, stderr, open-file audit). The
G16-G24 children opened no held-out path (audit: 0 events in the watched paths). Every reading below is
engineering, not quotable.

- **G16** (82 checks per device). In each of the 17 binary configurations one training-mode forward runs
  the map once per binary site (4 to 10 sites), and every kernel it used equals q * S exactly (BF: q *
  sign(s) * S), holds only +q and -q per layer or per column, or 0, with q >= 0 on its grid; qkernel on
  the moving moments satisfies the same; q_g is per layer at non-BN and per column at fused-BN sites.
- **G17** (65). The trainable set of every configuration is CH-8's 97 roles minus kq's b and i plus q_g's
  b and i at its sites, shapes and dtypes otherwise CH-8's; q_g's k is a non-trainable uint8.
- **G18** (48). No kq object, attribute, sublayer or role at any binary site, `kq` raises, and the
  regularized variables there are CH-8's with kq's b and i replaced by q_g's (MonoL1 1e-8); a
  training-mode call adds 89 losses in every configuration, as CH-8; the control finds kq at CH-8's ten
  kernel sites.
- **G19** (76). In all 14 cases (FFN contraction, Q/V projection with its output quantizer, O projection;
  FFN expansion and a head layer in FB and BF; each with GR) the outputs, the gradient of every trainable
  variable (W, gamma, the bias, the input, bias and scale quantizers' variables) and of x, and the updated
  state (moving moments, WRAP integer bits, EBOPs counter) equal those of Chang's layer with kq replaced
  by the gate's map, byte for byte on both devices. So the H1 penalty equals Chang's contraction with
  kq.bits_ replaced by q_g's. GR at non-BN sites matches the analytic added derivative (largest deviation
  3.5e-06 on PCC and 3.6e-06 on PCG, against gradients of order 80; tolerance 1e-5 of that); at fused-BN
  sites GR changes dL/dW and not the forward.
- **G20** (243). For every binary site of F100, F111 (FB, BF), BE (FB, BF) and GR-FB, under H1 and H2,
  at q_g bits 0, 1 and 4, the gradient of beta * (b) with respect to q_g's b equals beta * sum_b (H1), or
  0, beta * sum_b or beta * sum_u by column class and branch (H2), exactly (for example the Q projection
  of F100: sum_b 1344, sum_u 96, all three branches). End to end, beta = 2^-10 adds that gradient to the
  model's training-loss gradient of every q_g b within the declared tolerance, nonzero.
- **G21** (22). Each binary class builds standalone; the 0.1.9 build wrapper evaluated qkernel exactly once
  at the end of each build (four times in the binary MHA, once per projection); the control whose qkernel
  is not computable at the end of build fails inside the wrapper, and passes when made ready.
- **G22** (52). The scope's weight configuration is STUDY section 2.2's q_g configuration field by field;
  in all 17 configurations q_g is `_qkvo_kq_conf` itself at the projections, the scope's weight
  configuration elsewhere, and equal to CH-8's kq configuration at every site; `.keras` reloads restore
  classes, configurations, all 244 variables and the inference logits bytewise; the binary MHA's build
  and output projection equal the installed source by syntax tree with the class replaced.
- **G23** (26). Sign(0) = -1; beta = 0 gives a zero kernel (rounded-to-zero route) with H1 = H2 = 0 and
  NR = penalty H1 = 172032 on the constructed site; negative and zero s_c behave as section 2.3 states in
  both forms; pruned and rounded-to-zero q_g give H1 = H2 = 0, NR 0 or 172032, penalty H1 equal to NR and
  penalty H2 0 or 55296, as section 2.2 states; with 16 saturated columns BF's direct gamma gradient is 0
  there and FB's is not; FB columns hold one magnitude each.
- **G24** (18). For b <= 23 (54 cases) every q_g output is on its grid and every on-grid input comes back
  unchanged; a forced b = 24 changes 6 on-grid inputs in each of three cases; the finding of B6-9 (78
  deviations from exact RND, all inputs within one ulp below half a step) is the same on both devices.
- **G10** (114), **G11** (PCC, PCG 89 each), **G15** (114): as in B5, plus the binary parts. G10: F100,
  F111-FB, F111-BF, GR-BF and BE-FB built by the runner load draw-s1 with 232, 217, 217, 217 and 214
  tensors, the unmatched and unused sets exactly q_g's and kq's k, b, i at the sites the gate lists, their
  own draw equal to the file, q_g at its constant initializers; a RoleSpec declaring empty sets is
  refused before any assignment. G11: F111-FB and F111-BF fetch the same batches as CH-8 and F000 in
  every epoch on both devices. G15: in F100, F111-FB and BE-BF the MHA is the binary subclass, its
  projections binary at CH-8's roles with `/kq/` replaced by `/qg/`, configured by `_qkvo_*_conf`, every
  class wrapped once.
- **Reruns** of modules B6 changed: G02 (34) and G03 (PCC 66, PCG 73) after M1 (env.py records XLA_FLAGS;
  G03's provenance line shows it); G13 (PCC 51, PCG 56) and G14 (42 each) after M13 and M5. All pass as in
  B5; their PCG default-flag controls again show run-to-run differences (B5 finding 1).
- **Tests** (not a gate; `preflight/gates/tests-b6/b2838835b733/pcc/`): `tests/test_runner.py`,
  `tests/test_rng.py` and `tests/test_env.py` through the pytest overlay, 75 passed; the venv's freeze is
  unchanged before and after.

### Findings (none fails a gate)

1. **q_g carries a uint8 k beside b and i** (decision B6-6). The unmatched set of a binary arm is k, b and
   i; STUDY section 6 step 4 says b and i. Wording, for the STUDY C-item pass.
2. **HGQ's float32 RND rounds inputs within one float32 ulp below half a step up to one step** (decision
   B6-9), at every width; the output stays on the grid. It applies to Chang's kernel quantizers as to q_g.
   For the STUDY C-item pass: section 2.2's sentence on exactness holds as worded.
3. **B5 finding 1 recurs** in G13 and G14 PCG at this sha (the default-flag controls).
4. **Branch b12 at the merge** (decisions B6-1, B6-7): its binary smoke and bench configs carry headline
   null and are refused by the B5-4 rule; its G57 asserts that binary arms are refused "only as not built
   yet", which no longer holds. `cbl.runner.BUILT_IN_B5` is unchanged for it; the admitted list is
   `cbl.runner.BUILT_ARMS`.

### Not done in B6

- B7: M7's model part and the independent penalty recompute (G26-G34), the frozen evaluation (M8), the
  initialization readings (G36); the `binary_sites` log keys come from `SCALE.record` there.
- B8: the structure record and the zero-variance detector on binary layers (G38).
- The GPU rows of B13 on real data for binary arms; G58 and G60 on the PC lane after the b12 merge.
- B14: conversion of the binary layers and the MHA subclass (G65); `kq` raises at a binary site (B6-3).
- Jev (`jev-lab` MCP) is down: `lab_check_protocol` and `jev_check_methods` were not run.
- Kai: OD-13 (B6-0), and B5-1's production-stack question, are his.

### Blinding (B6)

None of the excluded paths was opened, listed by content or searched. Read: the B6 brief and the common
rules; `RULES.md`; `docs/methodology/03-phases.md` and the first part of `06-review.md`; STUDY v0.10 heading
list, sections 2, 3.1-3.3, 5, 7.3, 12, 13 (lines 187-422, 658-716, 838-866, 1291-1362); section 3.4 was not
read; PREFLIGHT-PLAN in full; this file; `code/docs/interfaces.md`; `notes/build-todo.md`; the fork's
`cbl/model.py`, `runner.py`, `rng.py` (roles and loader), `env.py` (provenance), `cost_pure.py` (u bound,
column rule, SiteInput), the gates G10, G11, G15, `gatelib.py`, `runnergate.py`, `runner_child.py`,
`fileaudit.py`; G04's `refdiff.md`; `preflight/da4ml-test/scripts/stage_b_build.py` lines 30-60 (the exactness
comment STUDY section 2.2 cites); installed hgq2, Keras and `quantizers` sources on the PC (`base.py`,
`einsum_dense.py`, `einsum_dense_batchnorm.py`, `mha.py`, `quantizer.py`, `config.py`,
`fixed_point_quantizer.py`, `internal/base.py`, `_fixed_point_ops.py`, `constraints`, `regularizers`, the
alkaid plugin's references to kq and qkernel, Keras's `EinsumDense` and `Layer` attribute hooks); in branch
b12's worktree (read-only), the generated config listing, two configs and G57's validate_config part. A
listing of the orchestrator's job directory showed file names only. No hardware-test outcome was seen.

## Lane M merge, third round (orchestrator, 2026-10-18)

- `3a0df98` merges branch `b12` (head `4d9f7cc087e0`; M17 `gen/`, 167 generated configs, gates G57 and G59; the replica token
  in M15 run names, B12-5) into `main` after B6 (`b2838835b733`), with no conflict (b12 had merged main twice; interfaces
  sections 14 (b12) and 15 (B6)). Rows and owner decisions B12-1 to B12-14: `preflight/gates/G57/4d9f7cc087e0/PREFLIGHT-rows.md`
  (supersedes the e8dce7948801, f78d60c576f2 and 08417e78a19c rows).
- Admission under a constructed complete input record plus handoff fields (G57): 66 of 167 configs pass `validate_config`
  (45 S1, 9 smoke, 9 bench, 3 det); 101 are refused only as not built or not admitted (38 PID, B9; 21 BEST-C and 21 CH-8-C, S4;
  15 FP32, W8A8, DS-HGQ; 6 GR and BE in S3, which wait for the S2 form). With inputs unset all 167 are refused; the 14 binary
  smoke and bench configs need the headline (B5-4).
- Orchestrator checks at the merged sha `3a0df9894dae` on the Mac (not evidence of record): G01, G25, G52 (0.28.0), G53-G56
  (`--study` v0.9), G57, G59 and G72 all PASS.

## B7 (PREFLIGHT-PLAN §4 row B7): M7 model part, M8 frozen evaluation, gates G26-G34, G36, G12's frozen part; leftovers G52, G58, G60

**What now works.** Every arm now has its frozen evaluation, built into the runner and run at every epoch end:
validation top-1 as an exact count, and H1, H2 and NR per layer, per group and in total, read from the model's
current quantizer state on the BN moving moments. It calls no layer in training or tracing mode, changes no variable
(checked by a hash of every variable before and after) and draws no random number (checked by the stream guard). A
reloaded checkpoint reproduces its epoch's record exactly. NR equals the native count that HGQ's own compiled
training step writes, layer by layer, on F000 and CH-8, on the CPU and on the RTX 4060 Ti; this settles B3a-3's open
arithmetic question at every state tested. The binary layers' training penalty equals penalty (b) recomputed
independently from the quantizer variables under both headline branches. Initialization readings of H1, H2 and NR
exist for all 19 configurations, seeds 1-3, on both devices. Main limitation: the per-epoch record does not yet hold
the structure and pruning-channel keys or the data-pass diagnostics (B8) or the PID block (B9), so W&B rows stay
non-strict. Second: every reading here is on the 2-file smoke archive or constructed tensors; the evaluation's time
per epoch on the production validation set (62,000 jets) is not measured here (finding 4).

**How it works.**
- **The counts (M7, `cbl/cost.py`).** It walks every counted HGQ layer from `model.layers` with M4's role rule and
  reads, per constant-kernel site, the input quantizer's widths and integer and fractional bits per position and
  input, the exported constants (`qkernel` on moving moments at HGQ sites, the binary map's record at binary sites),
  each constant's quantizer f and bits, and the bias term as the layer computes it; B3a's `cost_pure` turns these
  into exact H1, H2 and NR. Layers without a constant kernel (tanh tables, QAdd, pooling, the MHA's own QK and AV
  counter, the softmax and its tables) enter with the value of their own `_compute_ebops`, cast to uint32 as HGQ
  writes its counter. The shapes passed to `_compute_ebops` are the ones HGQ's call wrapper passes, read from Keras's
  recorded build shapes.
- **Penalty (b), independently.** For each binary layer the training penalty of both branches is recomputed from q_g's
  rounded bits and the input quantizer's bits alone, never from a kernel tensor and never through M6's own penalty
  code, with B3a's u bound and the same float32 combine of the bias term.
- **The evaluation (M8, `cbl/frozen_eval.py`).** One jitted inference-mode stateless call over the validation tensors
  in fixed 2790-row chunks gives the count; the counts, the binary-site record (q_g's widths and values, zero columns by
  route, saturated columns, sign occupancy and flips, single-sign columns) and the variable hashes complete the core.
  The core is cached per epoch, so the PID of B9 and the appended callback read one evaluation. The appended callback
  (still called `FrozenEvalHook`) writes the epoch's record to `epochs.jsonl` and W&B and adds only new Keras log keys.
- **The runner** builds the evaluator in `prepare`; gates keep a hook for a read-only probe run after it. G13's release
  side appends the same callback without the evaluator, so G13's bitwise equality with the release now also shows
  that the evaluation changes nothing in training (B7-8).
- **The gates.** G26-G33 run one fresh child each (`gates/cost_child.py`) on every configuration in five constructed
  states (as built, after five Adam steps, and three states with every quantizer variable, kernel, BN gamma and moving
  moment redrawn, zero widths and pruned scales included), under the open-file audit; G34, G36 and G12's new part run
  the runner on the smoke archive. Every gate of record waited for a window with no other gate process on the PC and
  records the process scan (`concurrency.txt`), because batch B8 ran its own gates there at the same time (finding 3).

### Owner decisions written before the gates of record (drafted 2026-10-18T15:06Z, entered here before the gates of record started at 15:35Z; B7-10 to B7-12 are also stated in the G58 and G60 gate code committed before those gates ran)

```
B7-1 DECISION: the frozen validation top-1 is computed by one jitted model.stateless_call(trainable,
          non_trainable, x, training=False) over the validation Dataset's own tensors (ds_val.x, ds_val.y), in
          fixed 2790-row chunks in order (run_train.py:74's batch), as an exact count of argmax agreement; the
          jitted function also returns whether every non-trainable variable it returns equals its input, which
          must hold. No Keras data adapter, epoch hook or metric is used, and the Dataset's __getitem__ is not called.
CONSEQUENCE: the count does not depend on Keras's validation loop; the checkpoint audit repeats the same
          chunks. Keras's own val_accuracy (a float32 mean) is recorded beside it, not compared as a condition.
ALTERNATIVES: model.predict or evaluate (adapters and callbacks; Keras state); one full-batch call (a
          different compiled shape from the training validation; memory on 62,000 jets).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B7-2 DECISION: G28 compares NR with native at an identical quantizer state: two jitted training-mode
          stateless calls on one constructed batch from the state under test, the second from the first's
          returned state; when every rounded quantizer bit is equal after both, the second call's counters are
          native at that state (its input quantizers were updated before their _compute_ebops, the tables' output
          quantizers after, and neither moved). The returned state is assigned and the frozen counts are compared
          per layer with those counters and with the layers' eager _compute_ebops.
CONSEQUENCE: "native" is the counter as a compiled training step writes it, so the comparison includes XLA's
          fusion of the bias term's multiply and add (B3a-3's open question), on the CPU and, as an added row,
          on the GPU.
ALTERNATIVES: compare with the eager _compute_ebops only (separate multiply and add by construction; would
          not test the fused training step).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B7-3 DECISION: the record's per-layer `native` is the layer's own _compute_ebops(*shapes) * ebops_factor at
          the frozen state, cast to uint32 by the backend as base.py:104-105 writes the counter; the counter
          itself is recorded beside it as `counter`. At a binary site `native` is null (not an HGQ layer,
          STUDY 3.1(a')) and the same call is recorded as `penalty_b`.
CONSEQUENCE: interfaces.md section 7 said B7 "reads from the layer counters"; the counter holds the last
          training-mode call's value, computed at that batch's WRAP integer bits and before the optimizer
          step, so it describes a different state from H1, H2 and NR. Both are in the record; the NativeLayer
          values the counts use are the recompute.
ALTERNATIVES: the counter alone (stale by one step); the recompute alone (loses the training count's parts).
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO
```

```
B7-4 DECISION: per-group subtotals use STUDY 2.2's kernel groups: attention = the four MHA projections, ffn =
          the expansion and the contraction, head = the three hidden-head layers, embedding = the embedding;
          every other counted layer (classifier, the MHA's own QK and AV counter, QSoftmax and its two tables,
          the tanh tables, QAdd, pooling) is `other`. Per-layer subtotals name each of them.
ALTERNATIVES: count the MHA's own counter and the softmax in `attention` (not a binarizable group).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B7-5 DECISION: B7's per-epoch record holds interfaces section 1's run_id, epoch, step, val_correct, val_n,
          val_top1, H1, H2, NR, subtotals, train_count, the four Keras loss and accuracy values, lr, beta, the
          two batch-order hashes, the variable hashes, snap_changed and binary_sites from quantizer state (q_g's
          b, i and value, zero columns by route, saturated columns, sign occupancy, sign-flip rate, single-sign
          columns). Null until later batches: quantizer_structure and pruning_channel (B8, with M10's structure
          record and zero-variance detector), the data-pass binary-site fields and the train-mode against
          frozen top-1 gap (B8), pid (B9). M15's quantizers/*, bn/* and activations/* keys are not written in
          B7. W&B rows stay strict=False (B5-6).
CONSEQUENCE: M15's strict key check would refuse these rows; it stays off until B8 and B9 add their keys.
ALTERNATIVES: build the quantizer-state structure keys here (they would overlap M10's routes and its gate G38).
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO
```

```
B7-6 DECISION: train_count is the int FreeEBOPs writes to logs['ebops'] (ebops.py:11-15), stored as int64;
          interfaces section 1 is retyped from float64.
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B7-7 DECISION: the appended callback adds to Keras's logs only the two batch-order hashes (as in B5) and the
          scalars H1, H2, NR, val_correct, val_n, val_top1; nested blocks go to epochs.jsonl and W&B. It refuses
          a clash with any key a released callback wrote.
CONSEQUENCE: Keras's History keeps these per epoch; nothing released reads them.
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B7-8 DECISION: M8 replaces B5's stub in the runner (the class name stays FrozenEvalHook, the record file is
          epochs.jsonl); gates keep `evaluator_factory` for a read-only probe that runs after M8's evaluation.
          In G13 the release side (run_train.py by runpy) appends the same callback with the same probe but
          without M8's evaluator.
CONSEQUENCE: G13's bitwise equality between the runner, which evaluates every epoch, and the release, which
          does not, also shows that the evaluation changes nothing in training. G10, G11, G13, G14 and G15
          rerun because cbl/runner.py changed.
ALTERNATIVES: M8 on both sides of G13 (would not show non-interference).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B7-9 DECISION: G36's readings are the initial state of each configuration and seed 1-3 as the runner
          prepares it on the device, with CH-8's file drawn there from the smoke archive: kernels from that file,
          BN moving moments 0 and 1, every quantizer at its constant initializer, no training-mode call.
CONSEQUENCE: they are engineering readings, not the draws of record (B19 draws on NRP from the archive of
          record) and not comparable with preflight/env/README.md's per-layer EBOPs, which were read after one
          training-mode forward on the record archive's first batch. For P3: B7 records the readings under the
          fork sha; it does not reproduce the README's table, whose state differs.
ALTERNATIVES: the archive of record (the brief restricts B7 to the smoke archive and constructed inputs).
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: YES (P3: what "reproduces them" asks for)
```

```
B7-10 DECISION: a config's param_count is Keras Model.count_params(), the elements of every variable
          (trainable and non-trainable), as model.summary prints the total; G58's record keeps the trainable and
          non-trainable elements beside it.
ALTERNATIVES: trainable elements only; kernel and bias elements only.
CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO
```

```
B7-11 DECISION: G58 writes the cbl1016-param-counts record to its evidence directory; it is not wired into
          gen/inputs.json and configs/ is not regenerated in B7.
CONSEQUENCE: the configs keep param_count null; filling it is a generator-input change with a regeneration and
          a G57 rerun on the Mac, for the orchestrator (B12's regeneration after B15, or B18).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B7-12 DECISION: G60 runs each admitted config through cbl.runner.prepare on the smoke archive with a gate
          copy that changes code_sha and init_tensors (as gen.configs.with_handoff), wandb mode `disabled` and
          run name null, and, for S1 configs only, stage S1 -> `gate`, because the runner admits the smoke
          archive in engineering stages only. Each copy's changed fields are recorded.
CONSEQUENCE: S1 configs are prepared exactly as generated in every field that enters the build, the
          initialization and the callbacks; their stage-dependent checks (seed range, epochs 7000) are G57's.
ALTERNATIVES: the archive of record (excluded by the brief).
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B7-13 DECISION: the cached evaluation (STUDY 7.2) is memoized by epoch in the evaluator, rechecked against the
          variable hash when a second caller reads it, and the cache keeps the last two epochs; the binary sites'
          sign-flip rate compares with the previous evaluation in the same process (null at its first).
CONSEQUENCE: B9's PID and the appended callback read one evaluation per epoch; after a resume (B10) the first
          record's sign-flip rate is null.
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

```
B7-14 DECISION: beyond the plan's rows, G28 also runs on the GPU path (PCG): B3a-3's open question is whether
          the backend fuses the bias term's multiply and add, and production trains on GPUs.
CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

### Leftovers, each its own gate run

- **G52 PCC at `3a0df9894dae`** (77 checks, PASS): the merged head with B12's replica token in M15, under the lock's
  wandb 0.31.0 through the pytest overlay, `TMPDIR` under scratch, the venv's freeze equal to `env/training/freeze.txt`
  before and after. Run without the `--keep` copy of the offline records, whose absolute symbolic links are Mutagen
  scan problems (B2-B6 notes). Operator script `~/lab/scratch/cbl1016/b7-g52.sh` (sha256 `2942f638…dceda`).
- **G58 at `780b3d908ad6`** (22 checks, PASS): under G57's constructed complete record and synthetic handoff fields,
  `validate_config` admits exactly G57's 66 configs and refuses 101 by reason (38 PID, 15 controls, 21 CH-8-C, 21
  BEST-C, 6 GR and BE in S3), none built. Each admitted config builds on the CPU as the runner builds it (244 variables
  by role, finite logits, an inference call that changes nothing) and reloads from `.keras` with every variable byte for
  byte, the same count_params and the same logits byte for byte; every layer configuration is restored except the
  disclosed beta0 initializer (finding 1). Parameter counts (B7-10), one per model key, engineering: CH-8 and F000
  44,265 (27,899 trainable, equal to `preflight/env/README.md`'s pre-PASS reading), F100 35,061, F001 36,105, F010
  39,756, F011 31,596, F101 26,901, F110 30,552, F111 and GR 22,392, BE 22,248, each the same in FB and BF. Record:
  `preflight/gates/G58/780b3d908ad6/pcc/param-counts.json` (not wired into the configs, B7-11).
- **G60 at `780b3d908ad6`** (217 checks, PASS): for each seed 1-3 the runner draws CH-8's file on the CPU from the smoke
  archive; each of the 66 admitted configs runs the runner's `prepare` (gate copy per B7-12). Every config loads its
  seed's file, the matched roles equal the file, the unmatched roles are exactly q_g's k, b and i at the arm's binary
  sites at their constants (1, 4, 0), the unused roles kq's, and the callback order is the released order plus the
  appended callback, equal to the config's `callbacks` list; per seed every shared role holds one tensor across all
  admitted arms. Rerun at `a74cd9d2b977` (217 checks, PASS), since the appended callback is now M8's.
- Operator script of G58 and G60: `~/lab/scratch/cbl1016/b7-leftovers.sh` (sha256 `252149a2…c7a84`).

### Gates of record at `a74cd9d2b977`

The Mac bundle was applied on the PC with `preflight/pc-setup/b5-pc-sync.sh`; PC HEAD
`a74cd9d2b9770b9b621371e9e87835a774a6826a`, equal to the Mac's, clean tree. All gates ran through
`~/lab/scratch/cbl1016/b7-run-gates.sh` (sha256 `49a4d565…761c3`, not campaign code), which checks HEAD and a clean
tree, waits before each gate for a window with no other gate process on the PC (batch B8 ran its own gates there,
finding 3), records a process scan before and after each gate (`concurrency.txt`; all 26 scans list no other
process), and calls `run_gate.sh`. Every gate checks HEAD, a clean tree at start and end, and the venv against its
freeze. Evidence directories hold `command.txt`, `interpreter.txt`, `stdout.txt`, `stderr.txt` (empty in all 26),
`exit_code.txt`, `result.json`, `concurrency.txt` and `children/` (argv, record, stdout, stderr, open-file audit). No
child opened a held-out path (0 forbidden events in every audit). Every reading below is engineering, not quotable.

- **G26** (109). For CH-8, F000 and the 17 binary configurations, each in five states (95 configuration-states, 1,900
  layer readings), H1, H2 and NR per layer, per group and in total equal an independent implementation in the gate
  (integer widths from the float32 bits, the column rule in Python integers); every native value is the uint32
  truncation of the layer's float32 `_compute_ebops` (non-integer values truncate: tanh1 17.2032 -> 17, add1 873.6 ->
  873); every HGQ site's NR kernel term equals the layer's own float32 einsum of its `ebops_equation`.
- **G27** (109 on each device). H2 <= H1 <= NR at every column, site and layer and in total, in every configuration and
  state; a constant 0.375 under 1-bit quantizer bits raises WidthBoundError.
- **G28** (27 on each device). On F000 and CH-8 in six states (as built, trained, three redrawn states, and a limit
  state where every constant saturates at its quantizer's full width), NR per layer equals the counter a jitted
  training-mode call writes at the same quantizer state (a fixed point of two calls, B7-2) and the layer's eager
  `_compute_ebops`, in all 20 layers; NR equals FreeEBOPs's total; in the limit state H1 = NR at every HGQ site (1,268,059
  in total); F000 holding CH-8's 244 variables gives CH-8's counts layer by layer. No mismatch on either device.
- **G29** (70). In every configuration, as built and redrawn, the counts are identical with no forward and after
  inference forwards on zero, unit and large random inputs, and no variable changes. Control: a training-mode call on
  the large inputs, assigned back, changes them (CH-8: NR 1,159,061 -> 1,177,898).
- **G30** (183). In the 17 binary configurations under H1 and under H2 (636 layer checks): in an eager training-mode call
  every binary layer's `_compute_ebops` value and counter equal penalty (b) recomputed from quantizer variables; after
  each of two jitted Adam steps every counter equals the recompute at that step's quantizer state; redrawing every binary
  kernel changes neither the recompute nor `_compute_ebops`.
- **G31** (51). In CH-8, F000 and every binary configuration the stock public aggregate equals the stock flattened private
  total, both > 0 (CH-8 as built: 1,424,366); the sum of the flattened public aggregates double-counts (2,139,200).
- **G32** (109 on each device). The snap of every exported constant to its quantizer's f changes nothing (727,320
  constants per device); 0.375 moved by one float32 ulp raises SnapError.
- **G33** (149). Every binary layer's `_compute_ebops` equals penalty (b) of its branch exactly; under H1 its NR equals
  the truncated penalty; junk written to every counter leaves H1, H2 and NR unchanged.
- **G34** (119 on each device). CH-8, F111-FB (H1) and F111-BF (H2), 3 epochs on the smoke archive through the runner:
  one record per epoch with the section 1 keys B7 fills; the variable hash unchanged by every evaluation; snap_changed 0
  and H2 <= H1 <= NR; the guard saw no change and its frozen-evaluation block ran once per epoch; the spy saw M8 evaluate
  once per epoch, and every hgq Quantizer call during the evaluations had training None or False; each of the 9 saved
  checkpoints, reloaded in a fresh process, gives the record of its epoch exactly; each mutated reload differs. Recorded
  beside: the frozen validation top-1 equals Keras's float32 val_accuracy at every epoch (for example CH-8 epoch 2,
  611/2000 = 0.3055); evaluation time per epoch after the first (finding 4).
- **G36** (189 on each device). Readings of H1, H2 and NR at initialization, 19 configurations x seeds 1-3, in
  `preflight/gates/G36/a74cd9d2b977/{pcc,pcg}/init-readings.json`; F000's equal CH-8's; NR is 1,348,571 in every
  configuration (all widths at their initializers); seed 1, H1 / H2: CH-8 515,711 / 515,711, F100 543,431 / 543,431,
  F111-FB 584,521 / 512,705, F111-BF 594,209 / 514,337, BE-FB 584,857 / 512,297, BE-BF 591,857 / 511,985 (GR as F111).
  All 57 readings are equal on the CPU and the GPU (B7-9: smoke-archive draws on each device, not the draws of record).
- **G12** (50). Parts 1 and 2 as in B4; part 3: M8's evaluation through the runner for CH-8 and F111-BF ran once per
  epoch inside the watched block with no change, the appended callback is wrapped as lab code, and the stream at the end
  equals the reconstruction of the Dataset's and the adapter's draws alone; the decoy (M8 drawing from `random`) stops
  the run with StreamViolation, exit 3, naming the frozen-evaluation block.
- **Reruns** of gates whose code B7 changed (`cbl/runner.py`, `gates/runner_child.py`): G10 (114), G15 (114), G11 (PCC 89,
  PCG 89), G13 (PCC 51, PCG 56), G14 (42 each), G58 (22), G60 (217), all PASS. G13 now compares the runner with M8
  against the release without it (B7-8), bitwise on both devices; G13 and G14 PCG's default-flag controls again show
  run-to-run differences (B5 finding 1).
- **Tests** (not a gate; `preflight/gates/tests-b7/a74cd9d2b977/pcc/`): `tests/test_runner.py`, `test_rng.py`,
  `test_env.py` and `test_cost_pure.py` through the pytest overlay, 113 passed; the venv's freeze unchanged before and
  after.

### Gates and code restated after a development run

Development runs of committed code at `bc102ac`, `780b3d9`, `c0cb5ec` and `a74cd9d`, and of the uncommitted cost child
from the PC's synced copy, are in `~/lab/scratch/cbl1016/b7-dev/` and are not evidence. No comparison criterion was
changed after a result was seen; what changed:
- **G58's configuration check** (before its run of record): the first development run found the beta0 difference of
  finding 1; the check was restated to require every layer configuration equal except that one named path, with both
  values fixed in the gate (0.0 built, 1e-05 reloaded) and the variables still compared byte for byte.
- **G28's limit state** (construction, not criterion): the first development run's limit state was not at full width at
  the fused-BN sites, because the two training-mode calls of B7-2 moved the moving variance (large x.W moments) and so
  unsaturated W s. The construction now resets the moving variance after the fixed point; native counts do not read the
  moments, so the NR-native comparison is untouched.
- **Code fixes found by development runs**, each its own commit before any run of record: G58's and G60's work paths
  made absolute (the children run in another directory); `config_child prep` leaves `cbl.env.setup()` to the runner
  (`780b3d9`); `runner_child` installs the spy and the decoy after the runner's setup (`a74cd9d`).

### Findings (none fails a gate; none needs a STUDY change)

1. **beta0 of the MHA's softmax after a reload (B7-F1).** QSoftmax and its two tables are created inside
   QMultiHeadAttention.build (installed `mha.py:324-338`, `softmax.py:77-99`) and are not serialized on their own, so
   `load_model` rebuilds them under the global LayerConfigScope default beta0 (1e-5) instead of Chang's `beta0=0`
   (`model.py:295`). Their `beta` variables are restored by value, and BetaScheduler (and the PID) set every beta each
   epoch, so no number changes; only the initializer in `get_config` differs, in Chang's own CH-8 as in every lab arm
   (198 layers across the 66 configs). For M14 (resume) and M18 (export): read beta from the variables, never from the
   configuration.
2. **B3a-3 settled empirically at the states tested.** The per-layer combine trunc_uint32(float32(kernel term) +
   bias term) equals the native counter of a compiled training step in every layer and state of G28, on the CPU and on
   the RTX 4060 Ti, including redrawn states with per-value bias widths whose mean is not dyadic; no fused multiply-add
   difference was observed. This is not shown for the production GPU product and stack, which B19 meets.
3. **Concurrent gate sessions on the PC.** Batch B8 ran gates of record from its own clone on the PC while this batch's
   development runs ran. The open-file audit sees opens by every process, and every gate parent lists the raw held-out
   directory (the manifest's `paths.val_dir`, `/home/kaimoe/lab/bnjettag-data/val`) to build its watches, so each
   batch's processes could appear in the other's audit windows. B8 reports (orchestrator relay) that its gate parents
   listed that directory between 15:11 and 15:36 UTC; the one
   development run of G34 here inside that window showed one foreign event in three of its twelve audit windows (not
   evidence; its records were deleted before the event labels were read), attributed to B8 accordingly, and a rerun
   alone showed none. In the other direction, B8's PC log shows a pass at `fa2311498a0d` with five gates failing while
   this batch's development processes (which list the same directory) were running; B8 owns that diagnosis and reran in
   quiet windows. Every B7 gate of record waited for a quiet window and recorded an empty process scan. Two batches
   sharing the PC need that discipline, or a lock, for every audited gate.
4. **Evaluation time per epoch (engineering, a floor).** After its first, compiling, evaluation, M8 took 0.10-0.18 s per
   epoch on the CPU and 0.26-0.62 s on the RTX 4060 Ti (G34, smoke validation set of 2,000 jets in one chunk; the GPU
   is slower because the counts are many small eager operations). Production evaluates 62,000 jets in 23 chunks every
   epoch; against the 3.53 s per run-epoch at which PREFLIGHT-PLAN §5 item 8 goes to Kai, this is a share of each
   run-epoch for B19's measured rates (G84), and a jitted count extraction is the remedy if it matters.
5. **CPU and GPU initial draws give the same readings.** The ~3e-8 kernel differences between CPU and GPU draws (env
   README finding 2) changed no exported constant: all 57 G36 readings are equal on the two devices.

### Not done in B7

- B8: `quantizer_structure`, `pruning_channel`, the data-pass binary-site fields and `train_frozen_top1_gap` in the
  record (null now, B7-5), the front saver and sidecar (M9), the zero-variance detector (G38). B8's M9 reads the integers
  `val_correct`, `val_n`, `H1`, `H2` and `NR` from the Keras logs (written by the appended callback, `LOG_KEYS`) or from
  `ctx.evaluator.evaluate(epoch)`, which returns them as Python ints.
- B9: the PID calls `ctx.evaluator.headline_value(epoch, headline)` in its `on_epoch_end` and writes `pid`; W&B rows stay
  `strict=False` until B8 and B9 add their keys.
- B10: the evaluator's cache and sign-flip state are per process (B7-13).
- B14: the export audit's entry point (`cbl.frozen_eval.export_audit`) exists; G35 runs on mulder with M18.
- The parameter counts are not in the configs (B7-11). `cbl/select.py:311`'s docstring citation was not touched (B7
  changed no M16 file). Jev (`jev-lab` MCP) is down: `lab_check_protocol` and `jev_check_methods` were not run.
- Kai: B7-9 (what P3's "reproduces them" asks for) is flagged; B3a-3 (flagged in B3a) has its empirical answer
  (finding 2).

### Blinding (B7)

None of the excluded paths was opened, searched or displayed. An `ls` of `preflight/` at the start showed the two
evidence-hash file names (names only; not opened). Read: the B7 brief and the common rules; `RULES.md`;
`docs/methodology/03-phases.md` and the first part of `06-review.md`; `docs/conventions/quantization-and-cost.md`;
STUDY v0.10 sections 2, 3.1-3.3, 3.5, 7.2, 8 to 8.1, 12, 13, 14 and, of 3.4, the frozen headline block and the start of
Addendum A (lines 423-530; file-listing table only, no outcome); PREFLIGHT-PLAN in full; this file; `code/docs/
interfaces.md`; `notes/build-todo.md`; `preflight/env/README.md` lines 139-215 (initialization readings; not excluded);
the fork's `cbl/cost_pure.py`, `binary.py`, `model.py`, `runner.py`, `rng.py` (roles, loader, guard), `wandb_log.py`
(keys), `gen/configs.py` (model key, handoff), gates G12, G13, G25 rows, G57, `gatelib`, `binarygate`, `binary_child`
(parts), `runnergate`, `runner_child`, `rng_child` (parts), `fileaudit`; installed hgq2, Keras and `quantizers` sources
on the PC (`base.py`, `einsum_dense.py`, `einsum_dense_batchnorm.py`, `mha.py`, `softmax.py`, `activation.py`,
`merge.py`, `pooling.py`, `quantizer.py`, `fixed_point_quantizer.py`, `internal/base.py`, `dataset.py`, `sugar/ebops.py`,
`beta_pid.py`); B8's PC operator scripts and logs (`b8-run-quiet.sh` and two logs, read-only, to avoid colliding with
its gates). No hardware-test outcome was seen.

## Merge of b8 (orchestrator, 2026-10-18)

- `f387d65` merges branch `b8` (head `e59dfe74a49e`; M9 `cbl/front.py`, M10 `cbl/structure.py`, M11 `cbl/ablate.py`; gates
  G37-G42) into `main` after B7 (`a74cd9d2b977`). Only conflict: `docs/interfaces.md`, sections 16 (B7) and 17 (b8), both kept.
  b8 worked in a second PC lane (a scratch clone outside the synced tree); its first G41 attempts at `fa2311498a0d` failed
  only the open-file audit, which caught B7's concurrent gate traffic on the held-out directory (kept with ATTEMPT-NOTE.txt);
  the runs of record waited for a quiet window. Rows and owner decisions B8-1 to B8-18:
  `preflight/gates/G37/e59dfe74a49e/PREFLIGHT-rows.md`. The front saver is not yet wired into the runner (B9).
- Orchestrator checks at the merged sha `f387d65e3eb5` on the Mac (not evidence of record): G01, G25, G37, G52 (0.28.0),
  G53-G56 (`--study` v0.9), G57, G59 and G72 all PASS.

## B9 (PREFLIGHT-PLAN §4 row B9): the front saver wired into the runner, M12 lab PID, gates G43-G47, PID admission

**What now works.** Every run through the runner now saves its own front: at each epoch end, inside the frozen-evaluation
step, the B8 front saver receives the epoch's full record and saves front and diagnostic checkpoints with their sidecars,
and at run end it writes the run summary (fronts under H1, H2 and NR, step envelopes with NA, filter outcomes, diagnostic
checkpoints, Chang's released front, and for PID runs the PID selection), checked by M16's reader. The PID arms train:
PID-CH and PID-BEST run Chang's BetaPID with the released gains and λ₀ = 2e-8, steered by the frozen headline count of
each epoch (the same cached evaluation the front and selection use), acted on at the next epoch's start, with its own log
keys, so ParetoFront and StopIf keep reading FreeEBOPs's training count as in ramp runs. The controller's state can be
written and restored so that a resumed controller continues bit for bit. 22 of the 38 generated PID configs pass the
runner's acceptance under G57's constructed complete record. Main limitation: the PID target is 350,000 in headline units
(OD-12), adopted under Kai's delegation and not confirmed by him (STUDY §19.2 item 4). Second: every PID reading here comes
from 14-epoch runs on the 2-file smoke archive, which never reach the 350,000 target or Chang's filter, so the PID selection
and the released front are empty in these runs; the controller is checked against its law, not for convergence.

**How it works.**
- **The saver in the runner (M13, B9-1 to B9-3).** `prepare` binds `cbl.front.FrontSaver` to the run (ablation mode
  `run_end`, B8-17; superseded files deleted, B8-3) and passes the appended callback a read-only step that runs after M8's
  evaluation inside its stream-guard watch: `front.chain_evaluator` on `cbl.frozen_eval.section1_record`, which builds the
  interfaces §1 record from the evaluation already made that epoch (`FrozenEvaluator.cached`, a hash-checked read, no second
  evaluation) and the epoch's logs; a gate's probe follows. The callback order is unchanged. `finish` writes the run summary;
  a failure there is recorded and the run exits 1.
- **The lab PID (M12, `cbl/pid.py`).** `LabBetaPID` subclasses hgq2 0.1.9's `BetaPID` and runs its `on_epoch_begin`
  unchanged. Its `on_epoch_end` writes `beta` as the stock does, reads frozen H(e) through `sensor_reader` (the units choice,
  in one place) inside its own guard watch, which makes M8 compute and cache that epoch's evaluation, reads the stock
  aggregate beside it, and stores H(e) for the next epoch's update. It writes scalar `pid/` keys only and never
  `logs['ebops']`; M8 copies them into the record's `pid` block. `state()` and `restore()` carry integral, previous error, λ,
  epoch and sensed value.
- **The runner's PID path.** The PID takes BetaScheduler's place and index; its sensor and the guard are bound once the
  evaluator exists; the guard wraps it as lab code; `validate_config` checks the `pid` field against `PID_RECORD` (the JSON
  string "inf" for max_beta), the `recipe`'s released values, a headline in every stage and no `ramp`. Released-builder arms
  other than CH-8 load CH-8's file under CH-8's role map with their own arm id (B9-9).
- **Admission (B9-10).** The PID arms were first admitted in engineering stages only; G43-G47 passed at `415c61716290`; then
  `528789d` admitted PID-CH and PID-CH-C in every stage. PID-BEST and PID-BEST-C stay engineering-only with GR and BE, because
  the runner does not read the S2 record yet. G57, G58, G60 and `tests/test_gen.py` state the new admission.
- **The gates.** G43-G47 (`gates/g43_pid_sensor.py` to `g47_pid_keys.py`, children `gates/pid_child.py`, shared
  `gates/pidgate.py`) run the runner for PID-CH (H1), PID-BEST as F111-FB (H1) and as F111-BF (H2), 14 epochs, seed 1, on the
  smoke archive, under the open-file audit, with read-only spies on the evaluator and on every callback's view of
  `logs['ebops']`. G46 replays one run's frozen readings through fresh controllers in separate processes.

### Commits

| sha | subject |
| --- | --- |
| `44d15dec3b980341ac1073a8471c05a6b353b577` | cbl/frozen_eval.py: cached(epoch) for readers after the evaluation, section1_record for the front saver, the pid block from the PID's pid/ log keys [B9] |
| `6da2a1edfc0d6bdfe77059495d7395a39099da84` | cbl/runner.py: wire the front saver (M9) into the frozen-evaluation step and the run summary into finish; callback order unchanged [B9] |
| `70fdb50edbe70a3820040e80340ef5f710dd0424` | cbl/pid.py: M12, hgq2 BetaPID with the frozen-H sensor, its own pid/ log keys, the stock aggregate beside, lambda_0 2e-8, released gains, persisted state [B9] |
| `85f4e600ddf57c36d358a04304d6979079c9cb6a` | cbl/runner.py: the PID schedule, PID config acceptance, released-builder arm ids; PID-CH engineering-only until G43-G47 pass [B9] |
| `262023f074b6ce9c222f51ff5d904925c7caa292` | gates: G43-G47, the lab PID through the runner, pid_child and pidgate [B9] |
| `e998a34abc5e9d5dd523d500ffa7be6460467ca6` | tests: test_pid.py; test_runner.py, PID refusals and the summary's schedule [B9] |
| `415c617162908108ad3f9bad4c32ff26e61a0e3d` | gates: pid_child records each callback's logs['ebops'] after it on its own entry, not on the last event [B9] |
| `528789dfa84a8ab7e104c83d4ecbc8913080d49b` | cbl/runner.py: admit PID-CH and PID-CH-C in every stage after G43-G47 passed at 415c61716290; PID-BEST and PID-BEST-C stay engineering-only (B6-7) [B9] |
| `1694a56fc6559bb9fa3c9b81b34722c5d95212a0` | gates: G57, G58 and G60 state the B9 admission [B9] |
| `b1a9d652c261af75bd45fb01bd583fd3ee31eba8` | tests: test_gen.py, the runner's acceptance under the B9 admission (88 configs) [B9] |
| `07c33ba5fcf095f169eed77e79ea1b61260a9273` | docs: interface notes section 18, lab PID, log keys, record block, state, the runner's PID schedule; front saver wiring as built [B9] |

### Owner decisions on the front-saver wiring (written 2026-10-18T20:59Z, before its gates of record)

```
B9-1 DECISION: the saver runs in the slot the appended callback already runs inside its frozen-evaluation watch
     after M8's evaluation (make_hook's `probe`): cbl.runner.after_evaluation(front.chain_evaluator(section1, saver),
     gate probe), where section1(epoch, logs) is cbl.frozen_eval.section1_record, the interfaces section 1 record of
     the epoch built from the evaluation already made that epoch (FrozenEvaluator.cached, which rechecks the variable
     hash and refuses an epoch not yet evaluated) and the epoch logs; the saver runs first, then a gate's probe
   CONSEQUENCE: the callback order stays the release plus FrozenEvalHook (B8-14); the sidecar's `record` holds the
     whole section 1 record (train_count, log_*, lr, beta, step, run_id; pid in PID runs), not only M8's core; the
     interfaces section 17 text `evaluator = front.chain_evaluator(evaluator, ctx.front)` was written for the B5
     runner, where the evaluator was a callable(epoch, logs); in today's runner M8's evaluator is an object and the
     key integers reach the logs only after that slot runs, so chain_evaluator(ctx.probe, saver) would fail on a
     record without H1; `cached` (not `evaluate`) keeps the evaluation count per epoch at one, which G34's spy checks
   ALTERNATIVES: front.keras_callback as an eighth callback (changes the order G13, G47 and G60 check); a new
     make_hook parameter for the saver
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
B9-2 DECISION: the runner passes ablations='run_end' (B8-17's recommendation) and keep_superseded=False (B8-3), each
     from one named constant in cbl/runner.py (FRONT_ABLATIONS, FRONT_KEEP_SUPERSEDED)
   CONSEQUENCE: B8-17 stays a flagged reading of STUDY section 12 for Kai; changing it is a one-constant change that
     reruns the runner gates
   ALTERNATIVES: every_save
   CONFIDENCE: MEDIUM   FLAG FOR HUMAN: YES (B8-17's flag carries over)
B9-3 DECISION: finish() calls FrontSaver.finalize for the ends completed, stopif, integrity_stop and failed, before
     the W&B summary and run.json; an exception there is recorded in run.json (`front_error`, `front_traceback`),
     `front_summary` is null, and exit 0 becomes exit 1 (failed); run.json is written in every case
   CONSEQUENCE: a run whose fronts cannot be summarized never counts as completed; the training end stays as it was
   ALTERNATIVES: let the exception propagate (run.json would not be written)
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

### Owner decisions on M12 and the PID schedule (written 2026-10-18T21:13Z, before the gates of record G43-G47)

```
B9-4 DECISION: OD-12 is built as adopted: the PID senses the frozen headline count H (H1 or H2, the config's
     headline) and its target is 350,000 in H units; the units choice lives in one place, cbl.pid.SENSOR and
     cbl.pid.sensor_reader, with the config key target_H (gen/configs.py PID, cbl.pid.PID_RECORD)
   CONSEQUENCE: a native-unit target would be a change of sensor_reader (to frozen NR, which equals native on CH-8 and
     F000, or to the stock aggregate) and of that key, a config regeneration and a restated G43; nothing else reads
     the units
   ALTERNATIVES: native units (STUDY 19.2 item 4 excludes them as recommended)
   CONFIDENCE: MEDIUM   FLAG FOR HUMAN: YES (Kai has not confirmed OD-12, STUDY 19.2 item 4)
B9-5 DECISION: the PID writes logs['beta'] as the stock does (beta_pid.py:164; PBar's template reads it) and only
     scalar keys of its own, pid/<field>: sensed_H (its own key for the sensed value), stock_aggregate (logged beside),
     beta, integral, clamp_state, slack, and once acting error, acted_on_H, beta_unclamped; a field without a value
     is left out, never null; it never writes logs['ebops']; M8 assembles the record's pid block from the pid/ keys
     (cbl.frozen_eval.pid_block), so M8 does not import M12
   CONSEQUENCE: the warmup records lack error, acted_on_H and beta_unclamped (M15's strict mode, B10, must allow a
     missing pid/error in warmup); every value passes M8's float.hex audit of the released keys
   ALTERNATIVES: a pid= handle on the appended callback that reads a block from the PID object
   CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO
B9-6 DECISION: the stock on_train_begin runs (init_beta on every layer, beta = init_beta); its stock reading is kept as
     stock_at_train_begin and the stored reading is cleared, because the frozen sensor has no reading before epoch 0;
     warmup >= 1 is required, and on_epoch_begin refuses (PIDError) if the stock law would act without a frozen reading
   CONSEQUENCE: with the released warmup 10 the stock law first acts at epoch 10 on sensed H(9), exactly as the stock
     timing (beta_pid.py:139, :167) with the frozen sensor
   ALTERNATIVES: a frozen evaluation before epoch 0 (an extra evaluation outside the per-epoch cache)
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
B9-7 DECISION: the PID is constructed at BetaScheduler's position in released_callbacks and its sensor and the stream
     guard are bound in prepare once M8's evaluator exists (LabBetaPID.bind; on_train_begin refuses an unbound PID);
     the prepare order is unchanged; the PID is wrapped by the guard as lab code and computes its reading inside
     guard.watch('PID frozen-H sensor')
   CONSEQUENCE: the evaluation of each epoch is computed inside the PID's on_epoch_end (before PBar, ParetoFront and
     StopIf) and reused, cached, by the appended callback and the saver
   ALTERNATIVES: build the evaluator before the callbacks (changes prepare's order, which G13 compares)
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
B9-8 DECISION: validate_config admits schedule pid exactly for the PID arms (PID-CH, PID-BEST, PID-CH-C, PID-BEST-C);
     a PID config's pid field equals cbl.pid.PID_RECORD in value and JSON type (max_beta the string "inf"), it has
     no ramp field, its recipe states the released non-lambda values the runner uses (cbl.runner.PID_RECIPE), and it
     needs the headline in every stage (the sensor and the saver read it); a ramp config has no pid field
   CONSEQUENCE: an engineering PID run cannot start with a null headline (the B5-4 exemption does not extend to it)
   ALTERNATIVES: accept the B5-4 exemption and default the sensor (no default exists in the STUDY)
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
B9-9 DECISION: released-builder arms other than CH-8 (PID-CH; CH-8-C and PID-CH-C) build Chang's model and load CH-8's
     file under CH-8's role map with their own arm id (dataclasses.replace(CH8_ROLE_SPEC, arm_id=...)); the loader's
     arm-id check stays
   ALTERNATIVES: build with arm id CH-8 (the flow's record would misname the arm)
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
B9-10 DECISION: admission in two commits: first the PID arms are built with PID-CH and PID-CH-C admitted in engineering
     stages only (PENDING_GATES_ARMS) and PID-BEST and PID-BEST-C in CHOSEN_FORM_ARMS; once G43-G47 pass, a second
     commit empties PENDING_GATES_ARMS; PID-BEST and PID-BEST-C stay engineering-only, like GR and BE, until the
     runner reads the S2 record (B6-7); CH-8-C stays refused (build-todo item, B9 or B10)
   CONSEQUENCE: of the 38 PID configs, 22 pass validate_config under G57's constructed complete record (PID-CH S3 3,
     PID-CH-C S4 13, smoke 3, bench 3) and 16 are refused only as S2-chosen engineering-only; PID-CH-C is admitted
     while its ramp counterpart CH-8-C is not, as the brief asks for the PID configs; G57, G58, G60 and tests/test_gen.py
     state the new admission
   ALTERNATIVES: admit CH-8-C in the same commit (a one-constant change now that B9-9 exists)
   CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO (the orchestrator decides when CH-8-C is admitted)
B9-11 DECISION: G46 in B9 gates the controller's part of the resume: state() after epoch K-1, written as JSON, restored
     in a fresh process, continues bitwise as the uninterrupted controller on the same frozen readings (K = 5, 10, 12),
     with tamper and no-restore controls; the whole-run resume (G48, M14) is B10's
   ALTERNATIVES: wait for M14 and gate both at once
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
B9-12 DECISION: G43-G47 run on the PC CPU on the 2-file smoke archive, seed 1, 14 epochs (warmup 10 plus four acting
     epochs, so the integral start at epoch 10 and the law are exercised), for PID-CH (H1), PID-BEST as F111-FB (H1) and
     F111-BF (H2); no PCG rows (the GPU PID run is B13's)
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
B9-13 DECISION: the clamp state is recorded by restating the stock's last two steps (10 ** output, damping, clamp) on
     the stock PID's recorded output, and on_epoch_begin raises if the restatement does not give the stock's beta
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

### Gates of record

Three sets, each at a committed sha applied on the PC by bundle (`preflight/pc-setup/b5-pc-sync.sh`, unchanged), PC HEAD
equal to the Mac's and the tree clean before and after every set. Every PC gate ran through
`~/lab/scratch/cbl1016/b9-run-gates.sh` (below), one gate at a time in a window with no other gate process; each evidence
directory holds `concurrency.txt`, the process scans before and after (all empty in B9). No other build lane used the PC.

1. **After the wiring, at `6da2a1edfc0d`:** G12, G13 (PCC, PCG), G14 (PCC, PCG), G34 (PCC, PCG), G37 PCC and G60, chosen
   because they run the runner or M8 or bind the saver (G13, G14 and G37 PCC named in the brief). G13's runner side now
   saves fronts while its release side does not, and the two stay bitwise equal in every weight, gradient, optimizer
   variable and output: the saver changes nothing in training. G34 shows one evaluation per epoch with the saver reading it.
2. **M12, at `415c61716290`, before the admission:** G43-G47 PCC (rows above). With the PID arms admitted in engineering
   stages only.
3. **The B9 head, `07c33ba5fcf0`:** G57 on the Mac (the command recorded at `G57/4d9f7cc087e0/mac/command.txt`, the sha and
   path changed to `main` in `$C/code`); on the PC G43-G47 again, G58 and G60 (they read the admission), and the reruns of
   every gate that runs the runner, prepare or M8: G10, G11 (PCC, PCG), G12, G13 (PCC, PCG), G14 (PCC, PCG), G34 (PCC, PCG),
   G36 (PCC, PCG), G37 PCC; then `tests/test_runner.py`, `test_pid.py`, `test_gen.py` and `test_front.py` through the
   ephemeral pytest overlay (113 passed, exit 0; the venv's freeze equal to `env/training/freeze.txt` before and after;
   `preflight/gates/tests-b9/07c33ba5fcf0/pcc/`). Not rerun, because no module they run changed: G15 (the MHA child builds
   models without the runner), G02-G04, G16-G33 (cbl/binary.py, cost.py and cost_pure.py unchanged), G38-G42 (M10, M11
   unchanged; G37 covers the saver), G52 (M15 unchanged), G59 (configs unchanged).

G57 at the B9 head, under its constructed complete record: 88 configs pass validate_config (S1 45, S3 3 PID-CH, S4 13
PID-CH-C, smoke 12, bench 12, det 3) and 79 are refused only as not built or not admitted (15 controls, 21 CH-8-C, 21
BEST-C, 22 S2-chosen engineering-only: GR and BE in S3 6, PID-BEST in S3 3, PID-BEST-C in S4 13). Of the 38 PID configs,
22 pass and 16 are refused only as S2-chosen engineering-only; with every arm admitted in memory, all 38 meet every other
rule. Before B9 the counts were 66 and 101 (38 of them as PID not built). G58's parameter counts at the B9 head equal the
`780b3d908ad6` record key for key (19 keys). G36's 57 initialization readings equal B7's record and PCC equals PCG; that
equality is an operator comparison of the two `init-readings.json` files with timing and sha fields left out, not a check
the gate prints. M16's reader (`cbl.select.run_from_records` and `check_pid_summary`, run by `FrontSaver.finalize`)
accepted all 20 PID run summaries of the G43-G47 runs at both shas (`front_summary.m16_reader` ok, each with PID selection
COST INFEASIBLE and no checkpoint), the first PID summaries it has read; no gate asserts this field.

Engineering readings, not quotable: the three PID runs' frozen readings stayed above 350,000 in all 14 smoke epochs (for
example PID-CH H1 from 565,850 at epoch 0 to 604,000 at epoch 13), so every PID selection is COST INFEASIBLE and λ moved
only slightly from 2e-8 (PID-CH H1: 1.99999999881e-08 at epoch 10, `G45/415c61716290/pcc/`; 2.00948e-08 after epoch 11,
`G46/415c61716290/pcc/`; the frozen readings in `G43/415c61716290/pcc/`). These readings show the plumbing, not the controller's behavior on the production data.

### Gates and code restated after a development run

- The first development run of G47 (at `e998a34abc5e`, not evidence) failed three checks: the gate child attached each
  callback's after-state to the last event in its sequence, and inside LabBetaPID's on_epoch_end the last event is the
  PID's own evaluation. `415c617` keeps a reference to the callback's own entry. The gate code changed, not the PID; G43,
  which reads only the event positions, passed in that run. G43-G47 of record ran after the fix.
- The interfaces §17 wiring text was restated as built (B9-1).

### Findings (none fails a gate; none needs a STUDY change)

1. **B9-F1, wiring text.** Interface notes §17 "Runner wiring" was written for the B5 runner, where the evaluator slot took
   a callable(epoch, logs); in the B7 runner the integers reach the logs only after that slot runs, so the literal wiring
   would hand the saver a record without H1. Built as B9-1 states; §17 carries a [B9] note.
2. **B9-F2, G47's ParetoFront entry check is empty in these runs.** Chang's filter (val_accuracy > 0.5, ebops < 500,000)
   is not met in 14 smoke epochs, so ParetoFront saved nothing and the check "every saved entry carries the training count"
   holds vacuously. The reads themselves are checked: ParetoFront's filter and StopIf's condition read FreeEBOPs's int at
   every epoch. A run that saves ParetoFront entries comes with B13's real-data PID run.
3. **B9-F3, evaluation timing in PID runs.** In PID runs the epoch's frozen evaluation is computed inside the PID's
   on_epoch_end, before PBar, ParetoFront and StopIf, and reused by the appended callback (G43). In ramp runs it is
   computed in the appended callback, as in B7. The result is the same cached value; only where its time is spent moves.
4. **B9-F4, for M15 strict mode (B10).** Warmup records carry no `pid/error`, `pid/acted_on_H` or `pid/beta_unclamped`
   (B9-5); STUDY §12's required PID keys include `pid/error`, so strict mode must accept its absence in warmup or B10 must
   choose another encoding.
5. **B9-F5, admission asymmetry.** PID-CH-C is admitted while its ramp counterpart CH-8-C is not (B9-10).

### Not done in B9

- CH-8-C's admission (now a one-constant change, B9-9); the parameter-count record is still not wired into the configs
  (B7-11); M15 strict mode (B10, with B9-F4); the runner does not call `gen.configs.check_launchable` yet (B13).
- The whole-run resume, G48, and the storage of `LabBetaPID.state()` and `FrontSaver.snapshot()` in the resume record (M14,
  B10). B10 must also know that the evaluator's cache and the PID's last blocks are per process.
- GPU PID rows: none (B9-12); B13's GPU consolidation includes a PID run.
- Every PID run in B9 used W&B `disabled`; no PID row has yet gone through M15 in `offline` mode to the SDK (G14 covers
  offline for ramp runs only). B13's smoke covers it.
- Kai: OD-12 (B9-4, STUDY §19.2 item 4) and the B8-17 ablation mode (B9-2) remain flagged. Jev (`jev-lab` MCP) is down:
  `lab_check_protocol` and `jev_check_methods` were not run.
- For the orchestrator to strike in `notes/build-todo.md` (not in this batch's writable set): the M9 runner wiring (done,
  `6da2a1e`); `pid.max_beta` read as "inf" (done, `70fdb50`); the 38 PID configs' admission (done, `528789d`: 22 pass, 16
  wait for the S2 reader). New items: B9-F4 for B10; G47's ParetoFront entries for B13 (B9-F2).

### Blinding (B9)

None of the excluded paths was opened, searched or displayed. Read: the B9 brief and the common rules; `RULES.md`; STUDY
v0.10 sections 3.3, 7.1-7.3, 12, 13 (PID and resume rows) and 19.2 items 1-16 (the item list only); PREFLIGHT-PLAN in full;
this file; `code/docs/interfaces.md`; `notes/build-todo.md`; `preflight/gates/G37/e59dfe74a49e/PREFLIGHT-rows.md` (B8's
decisions); the fork's `cbl/runner.py`, `frozen_eval.py`, `front.py`, `cost.py` (aggregates), `rng.py` (RoleSpec, loader,
guard, build_model), `model.py` (arm_spec), `wandb_log.py` (keys, flattening), `gen/configs.py`, `vendor/stop_if.py`,
`chang/run_train.py` (lines 76-104), `chang/model.py` (lines 290-300), gates G13, G14 (part), G34 (part), G37 (part), G57,
G58, G60, `gatelib`, `runnergate`, `runner_child`, `config_child`, `b8_child` (part), tests `test_runner.py`, `test_gen.py`
(part); the installed hgq2 `beta_pid.py`, `beta_scheduler.py`, `ebops.py`, `pareto.py`, `pbar.py` and the `ebops`
properties of `base.py`, `mha.py`, `softmax.py` on the PC; the snapshot's `beta_pid.py` (hash and syntax tree). No
hardware-test outcome was seen.

## B10 (PREFLIGHT-PLAN §4 row B10): M14 resume, the runner's launch check, the held-out guard, M15 strict level, CH-8-C, parameter counts; gates G46 (whole run), G48-G51, G57 PC part

**What now works.** A training run killed at any point (between epochs, in the middle of an epoch, or while it writes
its checkpoint) can be restarted from its own run directory and continues exactly as if it had never stopped: on the
smoke archive, on the PC CPU and on the RTX 4060 Ti, every per-epoch record, the final weights, the optimizer, both
random streams and the batch order, StopIf, ParetoFront, the lab PID and the front saver of a resumed run equal those
of an uninterrupted run (G48; G46 for the PID across a whole run). Every runner run now writes these checkpoints. A
resume refuses another code sha, config, W&B run, checkpoint interval, device or stack, a finished run and a damaged
checkpoint, and changes nothing when it refuses. The runner also refuses, before any training step, a launch copy with
an unset decision input (the launch check, G57's PC part), and in a training process it refuses every open of a
held-out file (G51). Every one of the 109 admitted configs reloads after one CPU epoch on the archive of record with
the same frozen counts and top-1 and logits within 1e-7 (G49), and saved checkpoints carry the training state's hash
through a resume (G50). Main limitation: on the PC CPU a fresh run of a binary arm was twice seen to differ from another
fresh run of the same config (finding B10-F1, origin unknown), so the CPU resume gate compares only chains whose fresh
processes agreed before the resume (B10-14). Second: M15's strict mode now checks every section 12 key the record
carries, but 27 key patterns of STUDY section 12 are produced by no module yet (B10-6), so the per-epoch log is not yet
complete.

**How it works.**
- **Checkpoints (M14, `cbl/resume.py`).** The appended callback writes, at train begin and after every epoch (interval
  `resume_every`, default 1), a checkpoint directory: every variable by role, the optimizer's variables, the module-level
  `random` state, NumPy's state, the training Dataset's `_index_map`, StopIf's counters, ParetoFront's record, the PID
  state, the front saver's snapshot, the evaluator's signs, the byte length and hash of the two append-only logs, and
  hard links of the live checkpoint files; `latest.json` names it last. Two generations are kept.
- **Resume.** `cbl/runner.py run --resume` checks the checkpoint before building anything, then builds the run exactly as
  a fresh one (the same builder, so no `.keras` file is deserialized and finding B7-F1 cannot arise), checks the run
  identity, cuts the logs back to the checkpoint, sets aside what the killed process wrote later, links back files it
  deleted, restores the PID and the saver, and fits from the checkpoint's epoch. Keras repeats its pre-fit draws in a new
  `fit`, so the streams, the map, the model and the optimizer are restored in the appended callback's `on_train_begin`,
  after those draws, and the stream guard is armed right after (STUDY 6 step 8).
- **Launch check, held-out guard, strict level (M13, M3, M15).** `prepare` calls `gen.configs.check_launchable` on the
  launch copy in every stage but `gate`; installs `cbl.data.install_heldout_guard` before any data read (an audit hook on
  `open` and a wrapper of h5py's file open); and opens W&B with `strict="produced"` (B10-6).
- **Configs.** G58's parameter-count record is a generator input (B10-10); 109 configs carry `param_count`. CH-8-C is
  admitted (B10-9).
- **Interface notes** section 19 states the checkpoint layout, the resume contract and the B17 consequences.

### Commits

| sha | subject |
| --- | --- |
| `630f0c39a9adbfe257752fc575dd0a7b064e5bfd` | gen: param_counts input names G58's record at 07c33ba5fcf0, copied byte for byte into manifests/ (B7-11) [B10] |
| `8ff6ad9f2e4482fb4ab26c637c70ecaf9e310f33` | configs: regenerated with G58's parameter counts (109 configs carry param_count; templates and controls stay null) [B10] |
| `6748a536d12d870547402b28fc46b33b6464f87f` | cbl/resume.py: M14, resume checkpoints (model by role, optimizer, random and NumPy states, _index_map, StopIf, ParetoFront, PID, front, signs), rewind of the run directory, refusals on another code sha, config, run id or identity [B10] |
| `c7ad7fe4e5f9ad5ece076c7d1e25e8f2ac924c4f` | cbl/frozen_eval.py: the appended callback's train_begin and after_epoch steps for M14; the evaluator's sign state saved and restored [B10] |
| `881541e35c6644cf9498ee50e8795f8482355e92` | cbl/wandb_log.py: strict level produced, every produced section-12 key required, NOT_PRODUCED named, warmup and first-evaluation row exemptions [B10] |
| `e43c074e935d61ffe29f2bb7a9f1f322ef81c6bf` | cbl/data.py: runtime held-out guard of a training process (audit hook on open, h5py make_fid), by name, resolved name, directory and inode [B10] |
| `4579e6e22f298bc4f84eea13b817dda1a779caec` | cbl/runner.py: check_launchable on the launch copy outside stage gate, the held-out guard, W&B strict produced, M14 resume and checkpoints, CH-8-C admitted [B10] |
| `0dddeb54ce83707a4e0c60f1538c63d8e1aa67ea` | tests: test_resume.py (M14 pure parts), strict level produced, CH-8-C and the launch check, the wired parameter counts and the 109-config admission [B10] |
| `f4237ba846f3e6f002aece469229394cc66eda50` | cbl/resume.py: check_static compares the W&B run name (source bundle, attempt) before anything is built [B10] |
| `0973cc5ba522c126f9072182a3f2ae4a4cfa9494` | cbl/runner.py: a resume refused or failed before fit writes its record under resume/attempts, never run.json; the run name is checked with the static checks [B10] |
| `015d4f8922f393dbb2af846483b25e403c2bf8f3` | tests: test_resume.py, the run-name refusal [B10] |
| `59058e28eef44aed64662d2ef6e5d30a6cffabae` | gates: resume_child, fresh-process child of G46, G48 and G50 (runs, real SIGKILL at a boundary, a checkpoint stage or mid-epoch, resume, final states) [B10] |
| `f8ca91d7dce442b581d02678aebe6dff64f1a236` | gates: G57 states the B10 admission (109 configs, CH-8-C in S4) and checks the wired G58 parameter counts [B10] |
| `4164c4f6037b8da2110a9199e376f6834fddf2b1` | gates: resumegate and resume_child commands for G46-G51 (kill-and-resume chains, comparison, constructed ParetoFront and StopIf, reload, guard probe) [B10] |
| `dca43ffbc656abe739f53bc618f6cfa7092bd44f` | gates: G48, uninterrupted and resumed runs identical (CH-8, F111-FB, PID-CH; boundary, mid-epoch, inside the checkpoint write, twice, offline W&B, link-back; refusal and no-restore controls) [B10] |
| `b44c5bc3196b684057323cd32459a68df33c1849` | gates: G46 whole-run part, PID-CH killed after 5, 10 and 12 epochs and resumed through the runner [B10] |
| `59bf7926034dd7e1ca5ae3eb2130ac5dc0e3d5d9` | gates: G49, metric reload agreement within 1e-7 for every admitted config after one CPU epoch on the archive of record [B10] |
| `e5c37b8f9daf8ccab32f5246bf343e1e409cbf8e` | gates: G50 PC part, training, front and resume checkpoint identity as a hash chain across a resume [B10] |
| `0c23626a5e0068bc0baf8d19352d7317cd4944d7` | gates: G51, static held-out audit of the fork and the runtime guard in a training process [B10] |
| `eed1193c1474c1a14f1f71977801314ffaa6771c` | gates: G57 PC part, a non-gate launch copy with an unset input is refused before any training step [B10] |
| `22a0324c0c3c6a5bf68b31cf89f2c4267e314e4f` | gates: G51 R5 reads subscripts of the manifest's heldout block, not the refusal rules' tokens; gen/protocol.py's row count named [B10] |
| `4c503b071f979a43e6a6fff02ea32f652b5112fd` | gates: resumegate returns no record for a failed child; chain_checked, the fresh-run agreement rule before the resume point (B10-14) [B10] |
| `6e75ad5d1a9d4e160e1f5ce6b9b6bdb7cae155ef` | gates: G48 and G46 check fresh-run agreement before the resume point and rerun a disagreeing chain once (B10-14); the offline case compares around its W&B mode [B10] |
| `622b44514721947efbda3076b7b120b350a1ea1b` | gates: G57 PC part, the gate-copy control has no W&B run stem (M15 names a stage-gate run itself) [B10] |
| `b2af3a7db908cb4017a30bec69499f6ea3b6390e` | gates: G58 and G60 text states the B10 admission (109 configs) [B10] |
| `267c6deb053e6af4a51e3358a1b4d707d1ebfccc` | docs: interface notes section 19, resume checkpoints and resume, the launch check, the held-out guard, M15's strict level, the parameter-count input; CH-8-C in section 18 [B10] |


### Owner decisions (written 2026-10-19T07:41Z, before the gates of record)

```
B10-1 DECISION: a resume never reloads the model from a .keras file. The resumed process runs the runner's prepare as a
     fresh run does (data, seeds, Datasets, the same builder, the seed's initial-tensor file, callbacks, compile) and
     then assigns every variable by role from the checkpoint's variables.npz; the state hash (cbl.rng.variables_sha256
     over the roles) must equal the checkpoint's
   CONSEQUENCE: finding B7-F1 (after load_model the MHA softmax and its two tables carry beta0 1e-5 instead of 0)
     cannot arise in a resume: no configuration is deserialized and every _beta variable is restored by value; G48
     compares every layer's configuration, beta0 included, between the uninterrupted and the resumed process
   ALTERNATIVES: a .keras resume file (the pattern mhab_resume_fit.py), with beta read from the variables
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
B10-2 DECISION: the restore point of the module-level random state, NumPy's legacy state, the training Dataset's
     _index_map, the model, the optimizer, StopIf, ParetoFront and the evaluator's signs is the appended callback's
     on_train_begin, which Keras calls after fit's symbolic build and reset (the pre-fit draws a fresh fit repeats);
     the PID and the front saver are restored before fit; in a resumed process the stream guard is armed right after
     the restore (in a fresh run it is armed at the pre-fit reseed, unchanged)
   CONSEQUENCE: in a resumed process the released callbacks' on_train_begin run before the guard is armed (they draw
     nothing; G12's fresh-run checks are unchanged); the checkpoint is written at the end of the appended callback's
     on_epoch_end, where M4's recorder shows the stream in the state the next epoch starts from (checked per write)
   ALTERNATIVES: a sanctioned-write exception inside the guard; restoring before fit (the pre-fit draws would move
     the stream again)
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
B10-3 DECISION: every runner run writes a checkpoint at train begin and at the end of every resume_every-th epoch
     (default 1, cbl.runner.RESUME_EVERY), at the last epoch and when StopIf stops the run; two generations are kept;
     live front checkpoint files and ParetoFront files are hard-linked (copied where links are refused) into each
     checkpoint; the interval is a runner input recorded in the checkpoint and a resume must name the same
   CONSEQUENCE: per-epoch I/O of the order of the model's and the optimizer's variables plus a state.json that grows
     with the epochs the front saver has seen; B17 and B19 may set another interval (a kill then redoes up to that
     many epochs)
   ALTERNATIVES: a larger default interval
   CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO (B17 may change the interval)
B10-4 DECISION: a resume is refused (exit 2) with nothing in the run directory changed for another code sha
     (checkpoint, config or fork HEAD), another config hash, another interval, another W&B run name (source bundle,
     attempt), another run identity (seed, device kind, stack, matmul precision, draw code, data, Datasets), another
     role set, a finished run (run.json) or a checkpoint file whose sha256 differs; a resume refused or failed before
     fit writes its record under resume/attempts/, never run.json, so the run stays resumable
   CONSEQUENCE: a pod restarted on another GPU product cannot resume (STUDY 6: one GPU product and stack per seed); a
     directory killed before train begin holds no checkpoint and the pack runner (B17) must set it aside and start
     the run again
   ALTERNATIVES: resume across products (bitwise identity lost; the determinism record would decide)
   CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO (for B17)
B10-5 DECISION: on resume the files the killed process wrote after the checkpoint are moved under
     resume/discarded/<n>/, never deleted; epochs.jsonl and front/events.jsonl are cut back to the checkpoint's bytes
     after their prefix hash is checked; live files the killed process deleted are linked back from the checkpoint;
     W&B rows are not rewound (offline: a second directory with the same run id; W&B is an index only, STUDY 12)
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
B10-6 DECISION (M15 strict mode): the build-todo item reads "strict mode stays off until B8 and B9 add their keys";
     B8 and B9 did not add all of them. 27 key patterns of the section 12 catalogue are produced by no module of the
     fork: quantizers/* widths and steps, bn/* scales and moments, activations/* (B7-5), the per-epoch
     quantizer_structure and pruning_channel blocks (M10's quantizer_state is not wired into M8, which leaves both
     null), the data-pass binary-site fields (moment mismatch, FB sign difference, dead channels) and
     train_frozen_top1_gap, per-epoch dwell (only the run summary has it) and sys/alloc_hours (the pack runner's).
     strict=True would refuse every row of every run. Built: strict="produced", the runner's level: every key of the
     catalogue the record carries is required in every row; cbl.wandb_log.NOT_PRODUCED names the 27 patterns and what
     would produce each; cbl.wandb_log.ROW_EXEMPT allows a missing pid/error while pid/clamp_state is warmup (B9-5,
     B9-F4) and a missing sign_flip_rate at the run's epoch 0 (M14 persists the signs across a resume)
   CONSEQUENCE: STUDY 12's per-epoch log is incomplete until a batch produces those keys; that batch then empties
     NOT_PRODUCED and the runner can open with strict=True
   ALTERNATIVES: keep strict=False; build the missing keys here (M8 and M10 work outside M14's scope)
   CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO (an orchestrator item: which batch produces the 27 patterns before S1)
B10-7 DECISION: cbl.runner.prepare calls gen.configs.check_launchable on the config it is given, the launch copy, in
     every stage but gate, right after validate_config and before the environment, the data or W&B
   CONSEQUENCE: smoke, bench and det runs need every input (headline, claim domain, parameter count, initial-tensor
     root) as S1 runs do (interfaces section 14: STUDY 14 treats them as launches); gates use stage gate copies
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
B10-8 DECISION: the runtime held-out guard of G51 is cbl.data.install_heldout_guard, installed by prepare in every
     runner process: an audit hook on the open event and a wrapper of h5py's make_fid, refusing a path whose name or
     resolved name is the held-out archive's or a raw held-out file's, a path under the raw held-out directory of
     record, and a file with the inode of one of those files on the machine; G51's runtime part opens placeholders
     only, so the real held-out archive is never opened even if the guard failed
   CONSEQUENCE: on NRP the inode rule covers the manifest's paths that exist there; the name rules hold everywhere;
     opens made in C by a library other than h5py are not seen by the guard (the kernel audit of the gates is)
   ALTERNATIVES: an LD_PRELOAD or seccomp guard
   CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO
B10-9 DECISION: CH-8-C is admitted in every stage: it is CH-8's arm (released builder, ramp) on the S4 seeds, loading
     CH-8's file under CH-8's role map with its own arm id (B9-9), as PID-CH-C does for PID-CH
   CONSEQUENCE: 109 configs pass validate_config under G57's constructed complete record (88 plus 21 CH-8-C)
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
B10-10 DECISION: the parameter-count record of B7-11 is G58's record at 07c33ba5fcf0, copied byte for byte into the
     fork as manifests/param-counts-g58.json (sha256 b58235ed...960e) and named by gen/inputs.json, so that
     regenerate-and-diff stays a function of the fork; its 19 counts equal the 780b3d908ad6 record key for key
   CONSEQUENCE: 109 configs carry param_count; the S2 templates and the controls (not built by G58) keep it null and
     unset, and check_launchable refuses them
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
B10-11 DECISION (placement): G48 on the PC CPU and on the RTX 4060 Ti under XLA_FLAGS=--xla_gpu_deterministic_ops=true,
     on the smoke archive (the bitwise resume gate on the home PC, STUDY 11 and 13; whether it also runs on the
     production product stays the determinism record's question, B19); G49 on the PC CPU on the archive of record for
     all 109 admitted configs (the GPU path is B13's); G50's PC part only (the exported link is B14's, off the
     launch path since Kai's 2026-10-18 message); G51 on the Mac and the PC CPU
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
B10-12 DECISION: Chang's ParetoFront saves nothing in any smoke run (its filter needs val_accuracy > 0.5 and EBOPs <
     500,000; B9-F2), so its record's persistence, and StopIf's counters, are gated by a constructed part of G48:
     Chang's two callbacks with the runner's restated arguments on twelve constructed epochs, interrupted after epoch
     7 with the process running on to epoch 10, resumed through cbl.resume's state, rewind and restore
   CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO
B10-13 DECISION: the frozen evaluator's per-site signs are persisted, so the first record after a resume carries the
     sign-flip rate of the uninterrupted run (interfaces section 16 said it would be null)
   CONFIDENCE: HIGH   FLAG FOR HUMAN: NO
```

Added 2026-10-19T08:17Z, after a development run of G48 (not evidence) and before the gates of record:

```
B10-14 DECISION: G48 and G46's whole-run part check, per kill-and-resume chain, that the records the killed process
     wrote before the resume point equal the uninterrupted run's (the two fresh processes reached the same state); a
     chain whose fresh processes disagree there says nothing about the resume and is run once more in a new directory,
     at most two attempts; every attempt and every disagreement is reported in the gate's output and record
   CONSEQUENCE: the resume identity is tested on chains whose starting state equals the uninterrupted run's; a fresh-run
     disagreement (finding B10-F1) is counted, never hidden
   ALTERNATIVES: no retry (the gate would fail at random on the CPU binary path for a reason outside M14); running the
     CPU children under an XLA threading flag (not shown to remove the variation)
   CONFIDENCE: MEDIUM   FLAG FOR HUMAN: NO (the critical reviewer should judge it with B10-F1)
```

**Finding B10-F1 (development runs, not evidence; for the orchestrator, B13 and the determinism record).** On the PC CPU,
two of about forty fresh runs of F111-FB (headline H1, seed 1, smoke archive, through the runner) differed from the
others from epoch 0, each in a few width variables of one layer and in nothing else: once the MHA key projection's
input-quantizer `f` and scale-quantizer `b`, once the embedding's input-quantizer `f`, kernel-quantizer `b` and
bias-quantizer `b` (largest differences after 3 to 6 epochs about 0.003 in `f`), with the epoch-0 training loss
differing in the sixth digit and the same initial tensors, batch order, validation loss, validation top-1 and counts;
the two odd runs also differed from each other. The size is of the order of a few Adam steps at lr 3e-3, which would
fit a small difference in those parameters' gradients; that is a guess, not shown. No resume was involved. Not
reproduced by 200 repeated gradient evaluations in one process, nor with a fixed hash seed (15 runs), nor with four
concurrent runs; the lowered gradient program is identical across 24 processes. Origin unknown. CH-8 and PID-CH
fresh runs agreed in every case. Engineering readings, not quotable.

### Gates of record at `267c6deb053e`

All at the B10 head, applied on the PC by bundle (`preflight/pc-setup/b5-pc-sync.sh`, unchanged; bundle sha256
`ccb06838d39a...`), PC HEAD equal to the Mac's and the tree clean before and after. PC gates ran through
`~/lab/scratch/cbl1016/b10-run-gates.sh` one at a time, each after a quiet window with no other python process on the
machine (an export agent's trainings were waited out); every `concurrency.txt` shows empty scans before and after. Mac
gates through the job's `mac_gate.sh` wrapper. Rows above.

- **New gates.** G48 PASS on the PC CPU and on the RTX 4060 Ti (165 checks each): all 12 kill-and-resume chains of CH-8,
  F111-FB (H1) and PID-CH (H1), every kill a SIGKILL (at an epoch boundary, mid-epoch, inside the checkpoint write before
  and after the rename, twice in one run, mid-epoch 0, W&B offline, a deleted live file linked back), equal the
  uninterrupted runs in every compared field; 0 fresh-run disagreements in 12 attempts on each device, so the B10-14 retry
  never ran; the no-restore control differs; the six refusals exit 2 and leave the run directory unchanged; the
  constructed ParetoFront and StopIf sequence resumes exactly. G46 PASS (65 checks): the B9 controller part and the
  whole-run part (PID-CH killed after 5, 10 and 12 epochs, every PID block and the final state equal). G49 PASS (130
  checks): all 109 admitted configs (S1 45, S3 3, S4 34, smoke 12, bench 12, det 3) trained one epoch on the archive of
  record and reloaded in a fresh process with no audit mismatch, logits bitwise equal for all 109 (largest difference
  0.0), validation loss and accuracy equal to Keras's logged values (largest difference 0.0), matmul precision float32
  in both processes. G50 PASS (44 checks): for CH-8 and F111-BF (H2), each across a resume, every sidecar, event line,
  summary entry, file on disk and fresh-process reload names the epoch record's state hash, and both kept resume
  checkpoints and all three checkpoints the resumed process wrote do too; the reloaded files carry beta0 1e-5 where the
  built model has 0 (B7-F1) with equal state hashes; the mutated control differs. In these 6-epoch smoke runs only epoch
  0's checkpoint stayed on disk, so link 1's reload covers one file per arm. G51 PASS on the Mac (12 checks) and the PC
  (25): every file naming a held-out identifier is accounted for, R1-R7 hold, and inside the training process the guard
  refuses all six open attempts on placeholders before the kernel sees an open, names the held-out archive, a hard link
  to it (by inode) and the raw directory, and passes the training archive; the unguarded control's open is seen. G57's PC
  part PASS (32 checks): the committed CH-8 smoke, bench and det and F000 det launch copies are refused at launch (exit
  2, no run directory, no JAX or Keras import), the gate copy trains one epoch with the check skipped, and the
  complete-record smoke copy passes the check and trains its 3 epochs.
- **Reruns** (the runner, M8, M3, M15 or the admission changed): G07, G08, G56 PCC, G10, G11 (PCC, PCG), G12, G13 (PCC,
  PCG), G14 (PCC, PCG), G34 (PCC, PCG), G36 (PCC, PCG), G37 PCC, G43-G45, G47, G58, G60, G52 (Mac twice, PC), G57 and G59
  on the Mac, all PASS with the check counts of B9 except G57 (66, the parameter-count check added), G60 (370, 109 configs)
  and G46 (65, the whole-run part added). `tests/test_runner.py`, `test_pid.py`, `test_gen.py`, `test_front.py`,
  `test_resume.py`, `test_wandb_log.py` and `test_data.py` through the ephemeral pytest overlay: 235 passed, exit 0, the
  venv's freeze equal to `env/training/freeze.txt` before and after (`preflight/gates/tests-b10/267c6deb053e/pcc/`). Not
  rerun, because no module they run changed: G02-G04, G15, G16-G33, G38-G42, G53-G55; and G06, whose preparation
  functions in cbl/data.py are unchanged (B10 added the guard's functions, which only the runner calls).
- Every run through the runner now writes resume checkpoints and logs its W&B rows under `strict="produced"`, so these
  reruns also exercised both on every arm the gates train (CH-8, F000, the binary configurations in both forms and both
  headlines, PID-CH and PID-BEST); G49 did so for all 109 admitted configs. Engineering readings, not quotable.

### Findings (B10-F1 is above; none needs a STUDY change)

2. **B10-F2, the strict-mode premise.** The build-todo item "strict mode stays off until B8 (structure and pruning keys)
   and B9 (PID block) are in" assumed those keys reach the per-epoch record. B9's PID block does; B8's structure and
   pruning keys do not (M10's `quantizer_state` exists but M8 leaves `quantizer_structure` and `pruning_channel` null),
   and the widths and steps, BN moments, activation ranges, data-pass binary-site fields, train-mode gap, per-epoch dwell
   and allocation hours were never assigned to a batch. Built as B10-6; 27 patterns listed in `cbl.wandb_log.NOT_PRODUCED`.
   For the orchestrator: a batch must produce them before S1 (STUDY 12 lists them as logged per epoch).
3. **B10-F3, a resume across GPU products is refused.** The run identity M14 compares includes the device kind and the
   stack (B10-4). A pod restarted on another product cannot resume; B17's launcher must pin the product of a resumed pod
   or start a new attempt. A directory killed before train begin has no checkpoint and must be set aside by the pack runner.
4. **B10-F4, W&B rows after a resume.** Rows the killed process logged after its last checkpoint are logged again by the
   resumed process (offline: in a second directory with the same run id). W&B is an index only (STUDY 12); `epochs.jsonl`
   is cut back and is the record.
5. **B10-F5, ParetoFront in the smoke runs.** As in B9 (B9-F2), Chang's ParetoFront saves nothing in any smoke run, so its
   persistence is shown on constructed epochs (B10-12); a real-data run that saves ParetoFront entries and is resumed
   comes with the NRP kill-and-resume smoke (G83, B19) or B13.

### Not done in B10

- G48 on the production GPU product and stack (the determinism record's, B19); G49's GPU path (B13); G50's exported link
  (B14, off the launch path); G83 (NRP kill and resume, B19).
- The 27 section 12 key patterns of B10-6 (no batch assigned; orchestrator).
- The root cause of B10-F1.
- B17 items: the pack runner's restart policy (`--resume` with the same `--resume-every`, a directory without a
  checkpoint, the GPU product of a resumed pod, B10-4 and B10-F3), and the checkpoint interval of production runs
  (B10-3).
- For the orchestrator to strike in `notes/build-todo.md` (not in this batch's writable set): the M14 items (index map,
  pre-fit draws, `state_sha256` not `ckpt_sha256`), `check_launchable` on the runner's launch path, M15 strict mode
  (settled as B10-6, with the new item B10-F2), the parameter-count record (wired, B10-10), CH-8-C's admission (B10-9).
- Kai: nothing new. OD-12 (B9-4) and the B8-17 ablation mode stay flagged. Jev (`jev-lab` MCP) is down:
  `lab_check_protocol` and `jev_check_methods` were not run.

### Blinding (B10)

None of the excluded paths was opened, searched or displayed. One directory listing of `$C/preflight/` (made to find the
gate directories) showed the file names of the two evidence-hash files and of `da4ml-test/`, `env/`; none was opened and
nothing of their content was seen. Read: the B10 brief and the common rules; `RULES.md`; STUDY v0.10 sections 5, 6,
7.2, 7.3, 11, 12 and 13; PREFLIGHT-PLAN in full; this file's header, gate table, B9 section, B7's decision B7-5 and
finding B7-F1; `preflight/gates/G37/e59dfe74a49e/PREFLIGHT-rows.md` (B8's decisions B8-2, B8-15, B8-17 and its not-done
list); the commands recorded under `preflight/gates/` for G07, G08, G37, G52, G56, G57, G58 and G59; `notes/build-todo.md`;
`code/docs/interfaces.md` in full; the fork's `cbl/runner.py`, `rng.py`, `pid.py`, `front.py`, `frozen_eval.py`,
`wandb_log.py`, `data.py` (parts), `gen/configs.py` and `gen/inputs.py` (parts), `vendor/stop_if.py`,
`chang/run_train.py` (lines 1-40), gates G46, G57, G58, G60 (parts), `gatelib`, `pidgate`, `pid_child`, `runnergate`,
`runner_child`, `config_child` (parts), `rng_child` (part), `fileaudit` (part), tests `test_runner.py`, `test_gen.py`,
`test_wandb_log.py` and `wandb_fixtures.py` (parts); the pattern `campaigns/2026-10-12-mha-bitnet/code/bnhgq2/
mhab_resume_fit.py` (sha256 0e4abde32fea..., read-only); on the PC the installed Keras JAX trainer (`fit`,
`_symbolic_build`, `jax_state_sync`), `callback_list.py`, the learning-rate scheduler, the optimizer base and hgq2's
`pareto.py`, `beta_scheduler.py`, `pbar.py`, `ebops.py`. No hardware-test outcome was seen.

## Operator notes

- The first Mac G01 invocation failed in the shell before Python started (zsh does not split an
  unquoted command string; exit 127, no gate code ran). Its files were deleted and the gate was
  rerun at the same sha; the evidence is from the rerun.
- During the PC venv check one temporary copy of the freeze output was written to `/tmp` on the PC,
  outside the `~/lab/scratch/cbl1016/` rule, and removed at once.
- Commit dates are Mac local time (PDT); evidence timestamps are UTC.
- B1: PC gate runs used the operator wrapper `~/lab/scratch/cbl1016/run_gate.sh` (sha256
  `6b2cedc1a1a05a73cb8b4a70d0c89064a46309c8a86409c6e4637dc3c6596072`, not campaign code); its
  `interpreter.txt` "resolved:" line gives the uv base interpreter that the venv symlink points to,
  unlike B0's.
- B1: development runs of G02, G03 (two drafts) and G04 on the uncommitted tree, from the PC's
  synced copy, were not kept as evidence; three of them wrote their stderr to `/tmp` on the PC,
  outside the scratch rule, and those files were removed. A dev G04 output stays in
  `~/lab/scratch/cbl1016/g04dev/`.
- B1: Mutagen scan problems rose from 2 to 8; the six new ones are absolute symbolic links under
  B11's `preflight/gates/G52/f77dbbb15d37/mac/offline-records/*/run/*/wandb/offline-run-*/logs/debug-core.log`,
  which Mutagen does not sync to the PC. No conflict.
- B2: an early attempt to stop the single-stream download with `pkill -f` matched the ssh session's own
  command line and ended that session (exit 255) before its other steps ran; the curl process was then
  stopped by PID and its partial file deleted. A tmux download unrelated to the lab (not this session's)
  was running on the PC throughout and was not touched.
- B2: the G52 PC run's temporary root went to `/tmp` on the PC (the gate's `tempfile` default) and was
  removed by the gate; a rerun should set `TMPDIR` under scratch. Its `--keep` copy of the offline
  records in `preflight/gates/G52/36e0c2df829a/pcc/offline-records/` holds absolute symbolic links that
  Mutagen does not sync (scan problems now 8 on the Mac side and 6 on the PC side; no conflict).
- B2: development runs of `tests/test_data.py` on the uncommitted tree (Mac `mhab1012-cpu`, and the PC
  overlay) and an h5py write-determinism check in `~/lab/scratch/cbl1016/b2-dev/` were not kept as
  evidence; the check's two files were deleted.
- B2: PC scratch now holds `zenodo/` (both tarballs, kept for G06 reruns up to B18), `data/`,
  `data-smoke/` and `data-smoke-src/` (the archives of record and the smoke links), `b2-prep/` (the
  record run's working copy), `b2-run1/` (the superseded run's manifest and links), `zenodo-api/`,
  `zenodo-log/`, `b2-tools/`, `tmp-tests-b2/` and an empty `gates-work/`.
- B2: on the Mac, `/tmp/cbl-b2-dev/` (outside the writable set) served as the working directory of the
  development pytest runs and held a draft of the B2 section of this file; it was deleted at the end of B2.
- B4: development runs of uncommitted code from the PC's synced copy (variable introspection, the role
  walker prototype, draws, drafts of G10-G12, `tests/test_rng.py`) were kept in
  `~/lab/scratch/cbl1016/b4-dev/` and are not evidence. One development G11 CPU run saw an edit of
  `cbl/rng.py` arrive through Mutagen while it ran; its later children were refused at the draw-code
  identity check, as designed.
- B4: the gates of record ran through the operator script `~/lab/scratch/cbl1016/b4-run-gates.sh`
  (sha256 `c50a6971f7157fc73b86d842ad06f7d190191042debbb63a142c56560167e2e8`, not campaign code),
  which checks HEAD and calls `run_gate.sh` once per gate. The gate work directories
  `~/lab/scratch/cbl1016/gates-work/G1{0,1,2}-af5c2a20c28d-*` are kept: they hold the drawn `.npz`
  files (209,850 bytes each) and their manifests; the evidence directories hold the child records.
- B4: the freeze check after the first test run wrote two temporary freeze listings to `/tmp` on the
  PC (outside the scratch rule), removed at once; the second run wrote its listing under scratch. On
  the Mac, one development listing was written to `/tmp/walk_out.txt` (outside the writable set) and
  deleted.
- B4: Mutagen scan problems on the Mac side rose from 8 to 20; the new ones are absolute symbolic links
  under `preflight/gates/G52/85d6e0f98c7a/`, written by another batch, not B4. No conflict.
- B5: the leftover gates ran through `~/lab/scratch/cbl1016/b5-leftovers.sh` (sha256
  `062c70c9082d240c4888fd945319f05140ca8d33861daaa2796c15580449d248`) at `78c5515df8da`, the gates of record
  through `~/lab/scratch/cbl1016/b5-run-gates.sh` (sha256
  `168678d941057ca8e931dca2979e01d63d0097523ef48937a50811f68d8790d9`) at `46cc25fa9e48`; both check HEAD and a
  clean tree and call `run_gate.sh` per gate (not campaign code). The PC bundle helper
  `preflight/pc-setup/b5-pc-sync.sh` (sha256 `50d13fcea356bd398b33e010b37e274960b8adad83feecb103f15fedea49f136`)
  applies the bundle after checking its sha256.
- B5: development runs of committed code at `4d23192`, `288db48`, `2947d06` and `546f0d7` (draw, runner,
  reference, step and mha children; the GPU determinism diagnostic with and without
  `--xla_gpu_deterministic_ops`; development runs of G10, G11, G13-G15) are in `~/lab/scratch/cbl1016/b5-dev/`
  and are not evidence. Gate work directories `~/lab/scratch/cbl1016/gates-work/G*-46cc25fa9e48-*` keep the
  drawn `.npz` files and the runs' output directories.
- B5: a Mac development pytest run with a `TMPDIR` that did not exist wrote
  `/tmp/pytest-of-kaiyamaguchi/pytest-9` (outside the writable set); it and the `pytest-current` link that
  pointed to it were deleted; `pytest-7` and `pytest-8` there predate this batch and were not touched.
- B5: Mutagen scan problems rose from 20 to 32 on the Mac side and from 6 to 12 on the PC side; the new ones
  are absolute symbolic links in G52's `--keep` copy of the offline records under
  `preflight/gates/G52/78c5515df8da/pcc/offline-records/`, as in B2 and g52fix. No conflict.
- B6: development runs of uncommitted code (from the PC's synced copy) and of committed code at `dd53d6e`,
  `06e5697` and `b283883` (the binary children, the G16-G24 parents and G10, G11, G15 on CPU and GPU) are in
  `~/lab/scratch/cbl1016/b6-dev/` and are not evidence. At `dd53d6e` Mac edits arrived through Mutagen during
  the development run: G15 failed only its clean-tree checks, and the G10 run was stopped.
- B6: the gates of record ran through `~/lab/scratch/cbl1016/b6-run-gates.sh` (sha256
  `ace4da607389bae02ac54a1490707a91ed4bc82e66f6d14671f04a62a186e5ea`, not campaign code) at `b2838835b733`;
  the bundle helper `preflight/pc-setup/b5-pc-sync.sh` was reused unchanged. Gate work directories
  `~/lab/scratch/cbl1016/gates-work/G*-b2838835b733-*` keep the children's work files (G22's `.keras` saves).
- B6: an attempt to stop a development run with `pkill -f` matched the ssh session's own command line and
  ended that session (exit 255), as in B2; the remaining process was then stopped by PID.
- B6: on the Mac, an empty directory `/tmp/cbl-b6-src`, a one-line file `/tmp/b6sha.txt` and a copy of an
  installed source file in the orchestrator's job directory (`mha_build_src.txt`) were written outside the
  writable set and deleted at once; a Mac pytest run wrote `$TMPDIR/pytest-of-kaiyamaguchi/pytest-38` and
  repointed `pytest-current`; both were deleted (`pytest-36` and `pytest-37` predate this batch).
- B6: Mutagen scan problems rose from 32 to 44 on the Mac side and stayed at 12 on the PC side at the end of B6
  (counts read with `mutagen sync list bnjettag`); the new ones were not inspected. No conflict.
- B7: development runs (cost child from the PC's synced copy; G58, G60, G34, G12, G13 and the readings child at
  committed shas `bc102ac`, `780b3d9`, `c0cb5ec`, `a74cd9d`) are in `~/lab/scratch/cbl1016/b7-dev/` and are not evidence;
  the first G34 development run's directory was deleted before its audit events were read (finding 3).
- B7: operator scripts (not campaign code): `~/lab/scratch/cbl1016/b7-g52.sh` (sha256
  `2942f638bed70ff335f6c478d16cf93d537895f24438dcdfd2bed3dd497dceda`, G52 at `3a0df9894dae`),
  `b7-leftovers.sh` (`252149a2c8b614eed1f69a6298795df3a8f6ef853cd8757588d94bed5fbc7a84`, G58 and G60 at `780b3d908ad6`),
  `b7-run-gates.sh` (`49a4d56501fd02189cfb3c7e4902cfca5d5e7fdaf1ecbbcc94d1ed6e376761c3`, the gates of record at
  `a74cd9d2b977`, with the quiet-window wait and process scans kept in `~/lab/scratch/cbl1016/b7-scans/`); the bundle
  helper `preflight/pc-setup/b5-pc-sync.sh` was reused unchanged, and a Mac-side helper outside the lab root bundled,
  waited for Mutagen and called it. Gate work directories `~/lab/scratch/cbl1016/gates-work/G*-{780b3d908ad6,a74cd9d2b977}-*`
  keep the drawn files, run directories, saved checkpoints and `.keras` saves.
- B7: while gates ran on the PC, Mac edits were held outside the lab root (the job's tmp directory) so that Mutagen
  could not dirty the PC tree; PREFLIGHT.md (outside the fork) was edited meanwhile.
- B7: a parallel batch (B8) ran its gates on the PC from `~/lab/scratch/cbl1016/b8-clone`; its operator scripts and logs
  were read, not changed, and none of its processes was touched (finding 3).
- B9: operator scripts (not campaign code): `~/lab/scratch/cbl1016/b9-run-gates.sh` (sha256
  `5e02a6c7e0c577e01a35733f3c4f2a8a114392562eccc7f69c622be29fd1e12d`; HEAD and clean-tree check, quiet-window wait and process
  scans kept in `~/lab/scratch/cbl1016/b9-scans/`, `run_gate.sh` per gate, the pytest overlay for `tests`), logs
  `b9-run-gates-{wiring-6da2a1edfc0d,pid-415c61716290,final-07c33ba5fcf0}.log`; development runner `b9-dev/run-dev.sh`
  (`151223183a1d9bc12ec4346586295ef2c20504b76d90c5bf4b4ba97e24af4e08`). The bundle helper `preflight/pc-setup/b5-pc-sync.sh` was
  reused unchanged. Gate work directories `~/lab/scratch/cbl1016/gates-work/G*-{6da2a1edfc0d,415c61716290,07c33ba5fcf0}-*` keep
  the run directories, saved checkpoints and children's work files.
- B9: development runs at committed shas `6da2a1e` (G13 CPU), `e998a34` (G43-G47; G47 failed on the gate-child bug fixed in
  `415c617`) and `415c617` (G47) are in `~/lab/scratch/cbl1016/b9-dev/` and are not evidence.
- B9: while PC gates ran, new and changed fork files were drafted outside the lab root (the job's tmp directory,
  `b9-work/`) and copied into the fork only between gate sets, so Mutagen could not dirty the PC tree; Mac test runs of the
  drafts used `TMPDIR` there. PREFLIGHT.md (outside the fork) was edited meanwhile. G57's Mac run used the same `TMPDIR`.
- B9: Mutagen scan problems stayed at 56 (Mac side) and 12 (PC side); no conflict.
- B10: operator scripts (not campaign code): `~/lab/scratch/cbl1016/b10-run-gates.sh` (sha256
  `b80644edc75afa84f1fa99a24d8d598feb5bd6baf06afe7503c6887d4e0c596c`; HEAD and clean-tree check, a quiet window with no
  other python process on the machine but the system's and mercaribot before every gate, process scans kept in
  `~/lab/scratch/cbl1016/b10-scans/`, `run_gate.sh` per gate, the pytest overlay for `tests` and G52), log
  `b10-run-gates-267c6deb053e.log`; on the Mac `mac_gate.sh` and the bundle helper `sync_pc.sh` in the job's tmp directory
  (outside the lab root); the PC bundle helper `preflight/pc-setup/b5-pc-sync.sh` reused unchanged.
- B10: a first start of the gates of record (2026-10-19T11:13Z) passed `--device cpu` to G57 and G51, which do not take
  it (argparse exit 2, no gate code ran); stopping the script with `pkill -f` ended the ssh session as in B2 and B6, and
  G50, already started, completed without its after-scan. The three evidence directories (G57, G51, G50 at
  `267c6deb053e`) and their work directories were removed, the log was renamed `...-aborted.log`, and all three ran again
  in the second start (11:25Z); the evidence is from that start.
- B10: development runs (the resume dev chains, the determinism probes of B10-F1 and development runs of G46, G48-G51 and
  G57's PC part at committed shas `f8ca91d`, `22a0324`, `267c6de`) are in `~/lab/scratch/cbl1016/b10-dev/` and are not
  evidence. One development set ran while a Mac edit of `docs/interfaces.md` had reached the PC through Mutagen; its runs
  refused the dirty tree, the edit was moved out of the fork until the set ended, and later fork edits were held until
  no PC run was active.
- B10: G37 was first run without `--device cpu` (the operator script's spec `G37` instead of `G37:cpu`), which runs only
  its pure part (101 checks, PASS); that directory was renamed `G37/267c6deb053e/pcc-pure-part-only/` and G37 ran again
  with `--device cpu` (137 checks, PASS) after the main set; its `concurrency.txt` holds the scans of both runs (the scan
  file name was reused), all empty.
- B10: an export agent ran short trainings from its own clone (`~/lab/scratch/cbl1016/export-clone/`) during the day; no
  gate of record ran while any of its processes ran (the quiet-window wait), and none of its processes was touched.

## Record update, 2026-10-20: coverage and superseded facts

The sections below are reconstructed from the records on 2026-10-20. Paths are relative to
this campaign unless absolute. Earlier sections retain their original wording and evidence;
their pending statements describe those batches, not the current launch state.
All checks and readings below are engineering evidence, not a quotable result.

B11, B12 and B15m already have records in this file's lane-M merge sections and gate table:

| batch | existing section and record | gates of record |
| --- | --- | --- |
| B11, M15 W&B logger | "Lane M merges"; `preflight/gates/G52/f77dbbb15d37/PREFLIGHT-rows.md`, decisions B11-1 to B11-4 | G52 MAC at `f77dbbb15d37`: 77 checks, PASS; the PCC readback failure and subsequent fix are retained in B2/B5 |
| B12, M17 generator and M15 replica token | "Lane M merge, third round"; `preflight/gates/G57/4d9f7cc087e0/PREFLIGHT-rows.md`, decisions B12-1 to B12-14 | At `4d9f7cc087e0`, G57 MAC 62 checks, G59 MAC 28, G52 MAC 77 under each recorded W&B version: PASS; PC follow-ups are in B7 |
| B15m, M19 clean-rerun driver | "Lane M merges, second round"; `preflight/gates/G72/d868cad9a07d/PREFLIGHT-rows.md`, decisions B15m-1 to B15m-12 | G72 MAC at `d868cad9a07d`: 581 checks, PASS against stand-ins; no real export result follows from that test |

These batches are not missing; their owner rows retain the choices, limitations and any
blinding statements recorded there. B17, B18P and B18 had no standalone sections here.

**Dated supersession, 2026-10-20.** Kai confirmed OD-13, OD-15, OD-12 and claim domain
b/default without C3r on 2026-10-19T17:13:14Z. Thus the corresponding open flags in B6 and B9
are historical. The source is lab `.claude/memory/decisions.md`, "2026-10-19 (later, 6)";
`code/gen/inputs.json` carries that provenance. The unknown cause in B10-F1 is superseded
by the loss-id investigation and merged guard below. B10-F2's missing logging catalogue
is reduced by B18P below; full strict logging is still not established.

## B17: pack runner, bundle handoffs and campaign admission

Reconstructed from the records on 2026-10-20. Sources: `code/NOTES-B17.md`,
`preflight/b17/{gate-scope.json,policy-final-checks.json,BRIEF.draft.md}` and the fork's
commit log. Implementation commits start at `3f609a6`; policy correction `6e055b8374ba`
was merged into main by `6cdbd05e65b9`. The note's "uncommitted" text describes its delivery,
before the orchestrator committed it.

M20 (`code/nrp/`) prepares bundle-mode Jobs and supervises one process per arm, preserving
run identity, checkpoints and completed siblings. M21 (`code/tools/`) adds exact config
admission, budget accounting, infrastructure retries and the ConfigMap inventory check
around the unchanged lab harness. Durable NRP state is confined to kai-data under
`/data/cbl1016/`; W&B is offline. Product, packing and resources remain explicit inputs.

### Gates of record at `a474b515d0a5` plus the B17 policy correction

The policy evidence names base HEAD `a474b515d0a5`, with edits still uncommitted
(`preflight/b17/policy-final-checks.json`); it is not a clean-commit gate pass at the merge.
G76/G78 top-level check counts in these original command records are not recorded.

| record | outcome and scope |
| --- | --- |
| `preflight/b17/tests.txt` | 87 passed, 1 skipped, exit 0; policy-correction suite, PC/GPU opt-in skipped |
| `preflight/b17/G76.txt` | harness validation VALID, exit 0; no matching approval, submissions refused |
| `preflight/b17/G78-{prepare,validate,offline-launch}.txt` | each exit 0; real bundle transport preparation/validation and OFFLINE_VALIDATED, no cluster call; G78 only partial |
| `preflight/b17/gate-scope.json` | G77, G79, G80 NOT RUN; staged transport bundle at `f8ca91d7dce4` predates B17 |

### Findings

The diagnostic G78 handoff could not train: the staged bundle lacked B17 and launch inputs
were unset. G76 validated the draft schema, not scientific readiness or authority.
The injected init-container resource ratio was flagged for G77/upstream resolution
(`code/NOTES-B17.md`, B17-8 and B17-11); no successful cluster lint is inferred.

### B17 choices

B17-1 withdraws the finite retry limit. Infrastructure-only relaunches have no count cap
inside the training/reserve budgets; code, integrity and gate failures stop the lineage.
More than six successful infrastructure resubmissions of a pack in a rolling 24 hours
notify through local NOTIFY.log and continue; this is a notification threshold, not a stop.
The counts and policy are from `code/NOTES-B17.md`, B17-1 and its policy verification.

B17-4 uses simultaneous training/reserve accounting; unknown usage holds its hard bound.
B17-9 keeps reviewed ADMISSION outside the code bundle and freezes its hash in BRIEF.
B17-10 requires both the kai ConfigMap inventory and mounted-reference count to be zero
for bundle operation before every wave. D1/D2 below further specify admission/accounting.

### Not done in B17

No U1 draw, training smoke, packing benchmark, production certificate or cluster gate.
No BRIEF/ADMISSION installation or approval. Rates, memory and utilization remain B19's;
the source is `code/NOTES-B17.md`, "Remaining integration and cluster gates".

### Blinding (B17)

The owner note records no network, remote-machine or credential access. Evaluation handles
lifecycle only and does not read held-out data. No new blinding audit was run to reconstruct
this section; the implementation/verification scope is the one in `code/NOTES-B17.md`.

## B18P: epoch diagnostics and pack restart identity

Reconstructed from the records on 2026-10-20. Sources: `code/NOTES-B18P.md`,
`preflight/b18p/{checks.json,tests.txt,pc-tests.txt,G57-command.json,G57.txt}` and the fork
log. Implementation `8e7b64f` and note `2884195` were merged by `107bd9a172dd`.

B18P implements producers for 26 of B10-F2's 27 missing patterns: quantizer/BN summaries,
activation and binary observations, train-mode validation gap, structure, resumable dwell,
and allocation telemetry (`code/NOTES-B18P.md`, pattern table). Only
`pruning_channel/*/native_share_removed` remains in NOT_PRODUCED; strict="produced" stays.
No baseline, denominator or zero-denominator convention was invented for that quantity.

### Gates of record at base `6cdbd05e65b9` with B18P edits

These are development records from the uncommitted worktree, not a clean-commit gate pass:

| record | outcome |
| --- | --- |
| `preflight/b18p/tests.txt` | 596 passed, 1 skipped, 19 deselected, exit 0; Linux-only inotify skipped |
| `preflight/b18p/G57.txt` | 64 PASS checks, 2 FAIL checks, exit 1; only git cleanliness at start/end fails |
| `preflight/b18p/pc-tests.txt` | 19 selected PC cases collected, not executed by this batch |

G57's regenerate-and-diff found 172 files, none missing, extra or different
(`preflight/b18p/G57.txt`). No generator or config source changed in B18P.

### Findings

Helper tests alone did not verify real JAX/Keras diagnostics or whole-run resume.
The epoch record now includes process telemetry, which later exposed the G46/G48 comparator
defect described below. Live `sys/alloc_hours` is supervisor allocation, not a final
all-inclusive pod bill (`code/NOTES-B18P.md`, B18P-10).

### B18P choices

Diagnostics use stateless forwards and variable/RNG guards; frozen counts remain inference-only.
The last-step FB sign observation replays the training batch without assigning updates.
Dwell reuses the resumable front history. These conventions and their limitations are
recorded in B18P-2 through B18P-6 of `code/NOTES-B18P.md`.

Before training, launch identity is published atomically. A launch-identified directory with
no published checkpoint is preserved as a dated `.pretrain-*` sibling and started fresh with
its original product/interval; a corrupt published checkpoint, terminal run or legacy
directory lacking recoverable identity is refused (B18P-7 through B18P-9).
This is the dated exception to B10/B17's earlier blanket no-checkpoint refusal.

### Not done in B18P

The undefined removed-cost share, real-stack producer tests and applicable resume gates,
real PVC interruption test and diagnostic overhead benchmark remained due at delivery.
B18's later PC rows below supply some of that evidence; collection is not execution.

### Blinding (B18P)

`code/NOTES-B18P.md` records no network, cluster, credential or forbidden review/preflight
directory access. This reconstruction reads records only, not arrays or held-out files.

## B18: decision inputs, candidate BRIEF and the B10-F1 integration

Reconstructed from the records on 2026-10-20. Sources: `code/NOTES-B18.md`,
`preflight/b18/{commands.json,checks.json,BRIEF.candidate.md}`, the fork log, and lab
`.claude/memory/decisions.md`, "2026-10-19 (later, 5)" through "(later, 9)".

`3ae52d3` records Kai's confirmations and the narrow null-headline exception for CH-8/F000
ramp runs. `b051a57` regenerates configs against STUDY v0.13; `b52258fb2e5d` merges B18.
Binary arms and PID still need a headline. The complete design is not blanket launch
permission: the candidate enumerates 13 eligible CH-8/F000 identities subject to remaining
inputs, gates and approval (`preflight/b18/BRIEF.candidate.md`, eligible_now).

### Gates of record at base `107bd9a172dd` with B18 edits

These are preparation records, before the clean final pass:

| record | outcome |
| --- | --- |
| `preflight/b18/test-gen.txt` | 37 passed, exit 0 |
| `preflight/b18/tests.txt` | 602 passed, 1 skipped, 19 deselected, exit 0 |
| `preflight/b18/G57.txt` | 65 PASS checks, 3 FAIL checks, exit 1; cleanliness at start/end and configs differing from HEAD |
| `preflight/b18/validate.txt` | VALID, exit 0; no matching approval, submissions refused |

The generation wrote 172 files: 167 configs, four protocol files and MANIFEST
(`preflight/b18/generate.txt`; `code/NOTES-B18.md`). The working-tree G57 failure is retained.

### Findings: B10-F1 cause and merged fix (dated supersession, 2026-10-20)

`preflight/investigations/B10-F1.md` identifies stale Python loss ids in Keras, not a
floating-point or threading cause: a later trace can delete a current layer penalty when
an old id is reused. Logged EBOPs can stay normal while the compiled loss loses the term.
`cbl/lossguard.py` clears those sets before stateless calls and checks the loss-term census.
Kai's B10F1-1 ruling applies it to every runner arm, CH-8 included, and model-building gate
children; STUDY v0.14/v0.15 declares the departure. Keras files on disk are unchanged.

The ticket head's "not merged; waiting for Kai" is historical. Kai ruled on 2026-10-19T21:59:06Z
(decisions entry "(later, 8)"), and `66449e9a94fa` merged the guard. Development census:
109/109 admitted configs, 218/218 census checks; the reduced G49 still failed 57 binary-site
schema comparisons (`code/NOTES-B10F1-CODEX.md`; evidence under
`preflight/investigations/B10-F1/codex/`). That development run is not a full G49 pass.
`c62423f` shares the enrichment-field set between runner and frozen audit, excluding those
diagnostics plus sign_flip_rate from the core reload comparison. Full G49 later passed
at `29a459ce7ee6` (below).

The initial final-pass attempt at `66449e9a94fa` failed G51's static allowlist for five new
documentation/tool files. `29a459ce7ee6` adds reasoned entries. Original stdout/stderr/exit
remain; an overwritten result.json was moved to the correct next-commit evidence directory,
as recorded in `preflight/gates/G51/66449e9a94fa/mac/NOTE-result-json.txt`.

### B18 choices

Keep the headline unset until the recorded export rule permits binary admission; do not
change a penalty mid-run. Keep BRIEF a candidate until final identities and applicable
gates are reviewed. The conditional split-form extension requires its own decisions,
contracts and approval (`code/NOTES-B18.md`, B18-2, B18-3 and B18-5).

### Not done in B18 preparation

The original candidate binds v0.13 and old hashes and has final-identity placeholders;
it is not current authority. `notes/handoff-2026-10-19.md` §4 lists the defects.
B18f's prepared replacement is described below. No production certification follows
from the candidate validator or the reduced B10-F1 census.

### Blinding (B18)

`code/NOTES-B18.md` records no prohibited review/export/da4ml-test access and no consultation
of an export outcome. The final-pass isolation incident is disclosed separately below;
the preparation statement does not cover that later pass.

## B18 final gate pass at `29a459ce7ee6`

Reconstructed from the records on 2026-10-20. The record is
`preflight/b18/final/GATES.md`, regenerated at 2026-10-20T07:36:02Z:
**FAIL 8, PASS 88, PASS WITH SKIPS 1, NOT RUN 33**. These are gate/place or suite rows,
not numbers of individual assertions. Full commit: `29a459ce7ee6714538328f76b2948b8aa170ee32`.
The named record and `results.json` retain the failures; later diagnosis does not change them.

### Gates of record at `29a459ce7ee6`

`preflight/b18/final/runs.json` retains command sources and exact argv. Bundle sha256
`4dc44a2b64f40e5ff89426f3b083cf6282029f23561add19301e3332cedf35d7` matched Mac and PC;
`pc-setup-29a459ce7ee6.txt` records clean PC identity. GPU bitwise rows used deterministic
XLA operations. Mac process scans were sandbox-denied; no Mac quiet-window proof is claimed
(`preflight/b18/final/GATES.md`, provenance and execution limits).

The stale `preflight/b18/final/GATES.stale-20261020T0622Z.md` had FAIL 7, PASS 73,
NOT RUN 49 and RUNNING 1. The PC batch completed and G14 was rerun before the Mac reboot
killed its supervising jobs; the orchestrator then regenerated the record from synced evidence
(`notes/LEDGER.md`, 00:30/00:45 PDT entries). G14 PCC and PCG now PASS, 42 checks each.
The forced-WANDB_MODE PCC attempt remains at
`preflight/gates/G14/29a459ce7ee6/pcc-attempt-forced-wandb-disabled/`.

The following counts are emitted checks (PASS and FAIL combined) or pytest summaries,
from the regenerated GATES record; evidence is under `preflight/gates/<row>/29a459ce7ee6/<place>/`.

| FAIL row | checks/tests | cause and disposition |
| --- | --- | --- |
| G08 pcc | 36 checks | Exception text truncated to 160 characters before checking for "incomplete"; long work path hid the token. Reader refusal and forbidden-open checks passed. Full text retained by `f60d3bc`; full gate rerun due in B18f. |
| tests-b1 pcc | 11 failed, 628 passed, 1 skipped | Mac-absolute lab paths in B17 tests caused FileNotFoundError for harness.py/run_handoff.py. Portable lab-root resolver in `f60d3bc`; PC suite rerun due. Cause: `notes/LEDGER.md`, 00:45 PDT. |
| G46 pcc | 77 checks | Whole-run fresh-agreement comparator included process telemetry. `d8dfe03` excludes all sys/ fields while keeping scientific equality; rerun due. |
| G48 pcc | 204 checks | Same comparator defect, starting at CH-8-boundary2. Offline subcase also met operator-forced WANDB_MODE=disabled. `d8dfe03` plus removal of that operator override; rerun due. |
| G48 pcg | 204 checks | Same telemetry defect on the GPU, including allocation hours; same fix and rerun. This is not evidence of a new training nondeterminism defect. |
| tests-offline mac | 1 failed, 609 passed, 4 skipped, 26 deselected | Sandbox denied os.killpg(SIGTERM). Orchestrator rerun outside the sandbox passed, separately recorded below. |
| G13 pcg | 56 checks | Placeholder IN_OPEN during run-A, without PID attribution. Investigator verdict: artifact of synced work placement, outside the runner; exact historical delayed opener not reconstructed. Unsynced work plus `38f4eba` refusal hardening; rerun due. |
| G51 pcc | 25 checks | Mutagen opened the real held-out inode through the gate's hard link inside synced work. Investigator verdict: placement artifact, not a runner read. Operator fix and `38f4eba`; incident remains for Kai. |

### Findings: field-level evidence and isolated reruns

G46/G48 diagnosis: `notes/LEDGER.md`, 04:12 PDT, and
`preflight/b18/codex/{codex-b18fix2-report.md,b18fix2-replay/output.txt}`.
The original DROP_KEYS removed epoch/frozen timing but left `sys/host_rss_mb` and GPU
`sys/alloc_hours`. Loss, accuracy as float.hex, beta, step, batch-order/index-map hashes,
train_count, subtotals and PID blocks agreed in the inspected records.
The orchestrator's G46 spot-check compared 14 rows on each side: 36 fields identical,
only host RSS different in all 14 rows (ledger). Replay names the saved children
`wr-PID-CH-U` and `wr-PID-CH-boundary10-a2-r`; agreement becomes true at boundary 10
and over all 14 rows after filtering, and false when log_accuracy is perturbed.
This replay verifies the comparator repair on saved CPU records, not fresh G46/G48 execution.

G13/G51 source: `preflight/investigations/G13-G51/TICKET.md` §§1, 2, 4–6.
`preflight/b18/final/prepare_final_commit.py:35-36` moved work inside the synced evidence
tree. G13 run-B/C/D saw no placeholder event; the draw child, which does not read that
placeholder, also saw an event in synced work. G51's draw-s1 event precedes the separate
guard self-test, whose audit was clean. They are different windows, not contradictory logs.
The investigator's later scratch rerun was invalidated by a dirty PC checkout after Mac
edits synced without .git; it is not a passing replacement for either row.

**Held-out incident; Kai ruling PENDING.** Ticket §4.1 records that G51 exposed a hard link
to the 448,219,074-byte held-out archive for about 30 seconds and Mutagen read its bytes.
No campaign training, selection or evaluation path read them; no human open is known.
Whether any content reached the Mac is undetermined. The investigation found no remaining
copy, but that does not exclude transient transfer. The orchestrator proposed treating it as
an incident with a fix rather than a §14 stop; Kai's ruling is not recorded
(`notes/handoff-2026-10-20-0445.md` §4). Neither the artifact verdict nor this document waives it.

Separate orchestrator rows, outside the sandbox, are not additions to the aggregate above:

| row at `29a459ce7ee6` | outcome | source |
| --- | --- | --- |
| G72 mac-orchestrator | PASS, 581 emitted PASS checks | `preflight/gates/G72/29a459ce7ee6/mac-orchestrator/stdout.txt` |
| G51 mac-orchestrator | PASS, 12 emitted PASS checks; static part only | `preflight/gates/G51/29a459ce7ee6/mac-orchestrator/stdout.txt` |
| tests-offline mac-orchestrator | 610 passed, 4 skipped, 26 deselected, exit 0 | `preflight/gates/tests-offline/29a459ce7ee6/mac-orchestrator/`; `notes/LEDGER.md`, 04:12 PDT |

G49 PCC passed over all 109 admitted configs: bitwise-equal reload logits and maximum
logit/loss/accuracy differences 0.0; 130 emitted checks. The population is the admitted
config census on the training/validation archive, not held-out evaluation. Lossguard tests
passed 10; pc_gpu passed 25 with 1 skipped and 614 deselected. The real pack smoke remains
unexecuted because its production path/product requirements do not fit the PC scope
(`preflight/b18/final/GATES.md` and the corresponding test stdout).

### B18 final-pass choices, not done, and blinding

The legacy snapshot checks remain explicit: G25 v0.8; G53–G56 v0.9; G57/G59 v0.8 protocol
references. The old GATES provenance header says v0.14; STUDY of record is v0.15
(`notes/handoff-2026-10-19.md` §§1, 4). Retaining those checks does not certify changed
export or launch policy. NRP/mulder rows, GPU reload, aggregate G85 and advisory G86
remain as individually scoped in GATES; G76 and offline G78 certify schema/transport only.
No ALL_PASS or launch certification is claimed. The held-out incident above qualifies
any blanket isolation statement for this pass.

## B18f: fixes and the final commit

Reconstructed from the records on 2026-10-20. The fork's read-only commit log gives this
linear lineage from `29a459ce7ee6` to `38f4ebae387413c50eaaa747844a3e2d028759c4`:

| commit | change | verification and source |
| --- | --- | --- |
| `f60d3bcd1b69dba3758c3b4fdfce932cb03ac1fe` | G08 checks full refusal text and truncates display only; shared tests/b17 lab-root resolver | Long-path/refusal/resolver regressions 7 passed; tests/b17 99 passed; offline 610 passed, 4 skipped, 26 deselected. `preflight/b18/codex/codex-b18fix-report.md`; full Linux G08 not run by fixer. |
| `d8dfe039382d59a76216ea31f21b75e1113d576b` | Child rows and parent comparisons exclude every sys/ key; scientific equality unchanged | Comparator tests 6 passed; offline 616 passed, 4 skipped, 26 deselected; saved G46 replay above. `preflight/b18/codex/codex-b18fix2-report.md`; fresh gates and GPU replay not run by fixer. |
| `319c52f008718ff628a150a92dde66cd18f4b4fb` | Generator STUDY_VERSION and STUDY_SHA256 point to v0.15 | Snapshot hash checked; generator report stopped at changed derived hashes, with no tests executed by that preparer. `preflight/b18/codex/codex-b18cfg-report.md`. |
| `9411e4fa796d99f8ce1b74b473616927a3fc6207` | Configs-only regeneration against v0.15; stamps and derived hashes only | Orchestrator inspected field differences; generator tests 37 passed; clean G57 68/68 PASS. `notes/LEDGER.md`, 04:25/04:28 PDT and 11:49Z; `preflight/gates/G57/9411e4fa796d/mac-orchestrator/`. |
| `84f372fbceac1747fd45ee865f3142d04b3815e0` | D2: charge zero only for evidenced never-started terminal Jobs; retain measured-final/bound logic | Nine added cases; tests/b17 108 passed; offline 625 passed, 4 skipped, 26 deselected. `preflight/b18/codex/codex-d2-report.md`; orchestrator rerun at tip in ledger 04:28 PDT. |
| `38f4ebae387413c50eaaa747844a3e2d028759c4` | Shared syncroot refusal for work, hard links, temporary paths and audited watch/cwd paths | New tests 18 passed; offline 643 passed, 4 skipped, 26 deselected; synced-work G51 refused before archive access. `preflight/b18/codex/codex-g51fix-report.md`; full Linux gates left to B18f. |

The generator preparer correctly stopped when derived hashes moved. The orchestrator then
verified that no design input changed and split the mixed generator/config commit so the
last configs-changing commit touches configs only. The discarded combined-commit G57 failure
was the configs-only-commit rule, not protocol drift (`notes/LEDGER.md`, 04:25/04:28 PDT).
`code/gen/inputs.json` retains Kai's confirmations and claim domain; headline remains unset.

### B18f choices: D1, D2 and prepared authority

Kai chose the defaults for D1/D2 (lab `.claude/memory/decisions.md`, "2026-10-20 (2)").
D1 restricts installed ADMISSION to the 13 eligible CH-8/F000 identities until the headline;
binary additions require re-approval. This is an installed-file rule, not a change to the
84-config bundled template. BRIEF's eligible_now/headline_hold prose is not an adapter
policy interpreter; the runner's null-headline refusal remains the second guard.

The earlier not-installed status and v1 validation references are superseded as of
2026-10-20 17:30Z: BRIEF candidate v2 was installed as `$C/BRIEF.md`, with
`$C/ADMISSION.json` (13 identities). The pair binds code `c4dc2a80ca2b`, STUDY v0.15
and the current MANIFEST. From the lab root,
`python3 tools/harness.py validate campaigns/2026-10-16-chang-bitlinear` returned VALID
(exit 0), authority sha256 `99199386d0f6b4ebf921c31a7749f833f747d2ae741fb9a91b93169443008f69`,
with no matching approval; submissions remain refused (`preflight/b18f/validate2.txt`;
`notes/LEDGER.md`, 17:30Z and the installation record in the 17:31Z entry).
The original B18 candidate and B18f v1 validation remain historical.

D2 is implemented in `84f372f`: terminal Job observations must exist and all be terminal;
at least one pod attempt must exist, and every attempt must explicitly record
allocation_started False. Only then is never_started true and its charge zero.
Missing pod records do not prove zero allocation. Known final failed usage is measured;
live or unknown usage retains the hard bound. Bucket classification and unfinished-Job
counting are unchanged (`preflight/b18/codex/codex-d2-report.md`).

### Authority prepared for approval one (corrected 2026-10-20 17:58Z)

Critical PREFLIGHT review v1 returned ITERATE (`review/critical-PREFLIGHT-v1.md`);
the 17:58Z ledger correction narrows approval 1 to U1 draws only. BRIEF candidate v3
was installed at 18:07Z as `BRIEF.md` (sha256 fadec31f…, authority sha c6fade11…, `preflight/b18f/validate3.txt` VALID; report `preflight/b18f/codex/codex-b18f-brief3-report.md`); the iteration-2 review (ITERATE, 0 A, 1 B: the draws' GPU product needs a decider and basis) led to revision 4, prepared and reviewed before installation
when available. The v2 installation at 17:30Z is historical execution, not approval
of the corrected scope. The sequence in `notes/LEDGER.md`, 17:58Z, is:

1. **Approval 1: U1 draws only**, for record and smoke archives, seeds 1–3, one product.
   At `c4dc2a80ca2b`, `gen/inputs.json` leaves the `init_tensors` and
   `smoke_init_tensors` roots/platforms and `bench_epochs` null; every non-draw config
   refuses launch. This BRIEF binds STUDY v0.15 by design.
2. **Approval 2: smoke, kill-and-resume, determinism, packing benchmark and measured rate**,
   after the rebind commit set and one gate pass. That set binds the draw outputs
   (initial-tensor roots/platforms), sets `bench_epochs`, regenerates configs, stamps
   STUDY v0.18 (`027ff590…fc68`), and includes headline H1, the two test fixes
   (`c4dc2a80ca2b`, `8626e4b5f4e4`) and the G57 post-outcome branch.
3. **Approval 3: production S1, all arms**, after certification is written into
   ADMISSION and the changed authority is approved. Binary arms also require Kai's
   confirmation of the BF owner record.

The earlier 17:34Z plan to keep main frozen until B19 ends is withdrawn. Main stays
frozen only until the draws end; the rebind commit set then lands on main and its pass
runs on the PC. The S1 hold is enforced through `code/tools/campaign.py`: S1 submits
only in pack mode, refused while `production_certification` is null; changing ADMISSION
invalidates the frozen authority. The 13 installed identities do not confer production
permission (`review/critical-PREFLIGHT-v1.md` A1/B1; `notes/LEDGER.md`, 17:58Z).

### Gates of record at `38f4ebae3874`: completed

The full pass launched 2026-10-20 11:42Z and completed. `preflight/b18f/final/GATES.md`,
Updated 2026-10-20T17:00:19Z, records FAIL 4, NOT RUN 34, PASS 91, PASS WITH SKIPS 1;
96 of 96 assigned rows have exit records. All assigned rows completed; this is not ALL_PASS.
Sources: `preflight/b18f/codex/codex-b18fgates-report.md` and
`preflight/b18f/final/{runs.json,results.json,pc-setup-38f4ebae3874.txt}`.
The launched brief is `preflight/b18f/codex/codex-b18fgates-brief.launched.md`;
its expected offline total of 616 is stale (the recorded total is 643 passed).

Every previously failing gate now passes at `38f4ebae3874` (emitted check counts):

| gate | place and PASS checks | source |
| --- | --- | --- |
| G08 | pcc 36 | `preflight/gates/G08/38f4ebae3874/pcc/` |
| G13 | pcc 51; pcg 56 | `preflight/gates/G13/38f4ebae3874/` |
| G14 | pcc 42; pcg 42 | `preflight/gates/G14/38f4ebae3874/` |
| G46 | pcc 65 | `preflight/gates/G46/38f4ebae3874/pcc/` |
| G48 | pcc 165; pcg 165 | `preflight/gates/G48/38f4ebae3874/` |
| G49 | pcc 130; 109/109 reload checks PASS | `preflight/gates/G49/38f4ebae3874/pcc/` |
| G51 | pcc 25 | `preflight/gates/G51/38f4ebae3874/pcc/` |
| G57 | mac 68; pcc 32 | `preflight/gates/G57/38f4ebae3874/` |

The four FAIL rows are tests-b1, tests-b2, tests-b5 and tests-b10 pcc. All failed
`tests/test_data.py::test_fileaudit_sees_h5py_opens_through_links`: `run_audited` was called
without cwd, falling back to the synced PC checkout and triggering the sync-root refusal
introduced by `38f4eba`. The test-only fix at `c4dc2a80ca2b` adds `cwd=str(tmp_path)` to
both calls (`preflight/b18f/codex/codex-testfix-report.md`; fork commit).
Tests-b1 also failed `tests/test_lossguard.py::test_the_stand_in_reproduces_the_drop_without_the_guard`:
assertion `3 < 3`, because the B10-F1 bug stand-in did not reproduce the loss-term drop on
PC CPU in that suite. The guard's own tests pass (tests-lossguard pcg: 10 passed);
the stand-in passed at `29a459ce7ee6`. Investigation `codex-lossguard-test` (report
`preflight/b18f/codex/codex-lossguard-test-report.md`): repeatable under the exact PC runner
(CPU 0/10, CUDA 0/3), but not universally deterministic; skipped on the Mac (its venv has no JAX).
The stand-in keyed its synthetic-id map by object address without retaining the objects,
so a freed tensor's address was reused and counted. Why it passed at `29a459ce7ee6` is not established;
the earlier claim about preceding FileNotFoundError failures changing the heap layout is withdrawn as unsourced.
The scenario runs in an isolated subprocess. The guard is unaffected (`cbl/lossguard.py:55` clears ids,
`:61-62` raises on census mismatch). Fix (+4/−1, test only; patched PC CPU 10/10, CUDA 3/3) committed on
branch b18fix as `8626e4b5f4e4`; main stays at `c4dc2a80ca2b` for approval 1 (U1 draws only).
This row remains a recorded exception with the established test-bookkeeping cause until the rebind commit
set and its pass (`notes/LEDGER.md`, 17:31Z, corrected 17:58Z).

`preflight/b18f/targeted/GATES.md`, Updated 2026-10-20T17:21:38Z, states its scope and reason:

> **Scope:** Only tests-b1, tests-b2, tests-b5 and tests-b10 pcc were rerun, once each, serially. The fix changes only `tests/test_data.py`: two line replacements add `cwd=str(tmp_path)` to audited child calls. All four rows failed that Linux file-audit link test at `38f4ebae3874`, motivating this targeted scope.

Its header corrects the original rationale: tests-b1 had a second failing test, the
lossguard stand-in. The targeted record preserves the full-pass evidence.

| row at `c4dc2a80ca2b` | outcome | exit |
| --- | --- | --- |
| tests-b1 pcc | FAIL: 1 failed, 671 passed, 1 skipped; lossguard stand-in only | 1 |
| tests-b2 pcc | PASS: 15 passed | 0 |
| tests-b5 pcc | PASS: 86 passed | 0 |
| tests-b10 pcc | PASS: 243 passed | 0 |
| tests-offline mac-orchestrator | PASS WITH SKIPS: 643 passed, 4 skipped, 26 deselected | 0 |
| G72 mac-orchestrator | PASS: 581 checks | 0 |

Evidence: `preflight/gates/<row>/c4dc2a80ca2b/<place>/`, linked by the targeted GATES file.
The Mac rows are existing orchestrator evidence, not rerun by the targeted operator;
other rows were not rerun at this SHA. This targeted scope is not a new full pass.
Work/keep/child scratch/TMPDIR remained outside sync roots; only records were copied into
evidence. PC identity was verified by bundle/reset; no operator-forced WANDB_MODE.
Full and targeted GATES headers retain the execution details and limitations.

Bundle `preflight/pc-setup/code-main-c4dc2a80ca2b.bundle`, sha256
`bbc14fe8ff7b6550369211cc729230fbb7d6430e321b378ffd038384a610d212`, is staged on kai-data at
`/data/cbl1016/code/cbl1016-main-c4dc2a80ca2b.bundle`, with equal in-pod sha256.
The `38f4ebae3874` bundle is also staged at
`/data/cbl1016/code/cbl1016-main-38f4ebae3874.bundle`, sha256
`2c0c6ba8fb489222fadbc2d88f8cbc22264f6e580424add5488764d997d5c2f1`.
Source: `preflight/b18f/staging/STAGING.md`; staging does not confer launch approval.

### Export check outcome (recorded 2026-10-20)

This records STUDY §3.4 v0.4 "Sequence and record": engineering pilot evidence, not
quotable results. Sources: `preflight/export/PREFLIGHT-rows.md` (owner decisions and final
evaluation) and `preflight/export/codex/codex-export-eval-report.md`.
Export/synthesis run commit: `4f828bb5a8df`; gate/check commit: `0dc211ecbacd`.
Evidence root E below: `preflight/export/mulder/4f828bb5a8df/`.

| run | C-synthesis | Vivado | evidence under E |
| --- | --- | --- | --- |
| `init-CH-8-normal` | rc=0 | rc=0 | `init-CH-8-normal/prj/STATUS` |
| `chk-F111-FB-normal` | rc=0 | rc=0 | `chk-F111-FB-normal/prj/STATUS` |
| `chk-F111-FB-factored` | rc=0 | rc=0 | `chk-F111-FB-factored/prj/STATUS` |
| `chk-F111-BF-normal` | rc=124 at 8 h and 86,400 s | never ran | `chk-F111-BF-normal/{prj,prj24}/STATUS` |
| `chk-F111-BF-factored` | rc=124 at 8 h and 86,400 s | never ran | `chk-F111-BF-factored/{prj,prj24}/STATUS` |
| `chk-CH-8-normal` | rc=124 at 8 h and 86,400 s | never ran | `chk-CH-8-normal/{prj,prj24}/STATUS` |

The 86,400 s attempts ended 2026-10-20 09:29:54 PDT. Each attempt's `synth_result.json`
and `vitis_hls.log` accompany STATUS; the driver log's rc=0 entries are status-capture
errors (`preflight/export/operator/eval-observation.json`), not successful synthesis.
G61–G70 at `0dc211ecbacd` each exit 0: `preflight/gates/G61/0dc211ecbacd/mul/` through
`preflight/gates/G70/0dc211ecbacd/mul/` (G6x family plus G70). The combined check exits 1:
`preflight/export/mulder/check/0dc211ecbacd/mul/` (argv, stdout, stderr, exit and result).

Both routes in each form are bit-exact on 4,096 validation jets, zero mismatching values.
Values below are factored / normal; WNS and CLB LUT are Vivado OOC post-opt,
pre-place-and-route, not physical timing closure or C-synthesis LUT estimates.

| form | bit-exact | II=1 | WNS (ns) | CLB LUT | strictly fewer factored LUTs | outcome |
| --- | --- | --- | --- | --- | --- | --- |
| FB | yes / yes | yes / yes | −1.090 / −1.087 | 394,351 / 402,252 | yes | FAIL: criterion 2 (factored WNS ≥ 0) fails |
| BF | yes / yes | no result | no result | no result | no result | FAIL: owner record of 2026-10-20T17:30Z; no C-synthesis result |

Under E, FB reports: `chk-F111-FB-{factored,normal}/prj/vivado/{post_opt_timing,post_opt_util}.rpt`;
II: each project's `prj/myproject_prj/solution1/syn/report/csynth.xml`.
F111-BF: recorded as a failure by the owner record of 2026-10-20T17:30Z; no C-synthesis result.
The governing provision is v0.4 "Outcome and forms": the owner records the attempt as
impossible to complete, with the reason. Missing stages cannot establish a pass.
Both forms fail, so Kai's split-form rule of 2026-10-19T16:28:12Z selects **H1 for every
binary arm, F100 included**. The one-form-passes provisions do not arise.

The owner record was made at 17:30Z (`notes/flags-for-kai.md`, item 0b;
`notes/LEDGER.md`, 17:30Z): BF was impossible to complete within the planned synthesis
budget (8 h, then 24 h; no larger budget in PLAN.md or STUDY §9). This results-informed
record supersedes B14-7a's consequence line (`preflight/export/PREFLIGHT-rows.md`,
lines 80–86); B14-7a is silent on a stopped 24 h copy. It was reported to Kai at about
17:36Z (`notes/LEDGER.md`, 17:48Z). Kai's choice between the default (owner record stands,
no resynthesis) and a 72 h BF attempt remains pending and gates binary admission.
STUDY v0.18 records the outcome in its change log and §3.4
(`review/snapshots/v0.18/STUDY.md`, hash in the status header); critical PASS at iteration 3
was recorded at 17:56Z, and the rebind commit for approval 2 stamps v0.18.
At this trained size, CH-8's normal route and both BF routes did not complete C-synthesis
within 24 h. This is context for VERIFY/REPORT, not a stop.

### Findings and not done in B18f

Fixer tests, saved-record replays and clean Mac rows support the repairs but do not replace
PC execution. G72 and tests-offline remain orchestrator-run rows outside the sandbox.
The first pass-job launch was refused for a non-git work directory; the successful launch
used --skip-git-repo-check (`notes/LEDGER.md`, 11:44Z). That was an operator launch refusal,
not a failed scientific gate. BRIEF v2 and ADMISSION were installed at 17:30Z; after the
17:58Z ITERATE verdict, v3 preparation, re-validation, installation and re-review remain
due before approval 1; the v3 pair was installed at 18:07Z and revision 4 (product decider and basis) is reviewed before it replaces it. Approval and B19 execution remain due under the three-step sequence above.

### Known limitations and open items

- **Lossguard stand-in test (cause found, fix on branch):** tests-b1 pcc fails the unguarded B10-F1
  reproduction test at `c4dc2a80ca2b` because of the stand-in's own id bookkeeping (above); the guard is
  unaffected. Why it passed at `29a459ce7ee6` is not established; the earlier heap-layout
  explanation is withdrawn (isolated subprocess). Fix `8626e4b5f4e4` on b18fix joins the
  rebind commit set (`notes/LEDGER.md`, 17:31Z, corrected 17:58Z).
- **BF owner record (pending Kai's choice):** owner record made 17:30Z under v0.4,
  superseding B14-7a's consequence line; reported to Kai about 17:36Z. The default
  (record stands, no resynthesis) versus a 72 h attempt remains pending; binary admission
  requires his confirmation (`notes/flags-for-kai.md`, item 0b; export outcome above).
- **mid0 resume comparison:** pre_resume_agree compares rows[:0], so fresh-run agreement
  is vacuous at that point. Full-run comparison remains, but it does not independently
  establish scientific initialization/checkpoint state and RNG equality at epoch zero.
  Left unchanged by `d8dfe03`; source `preflight/b18/codex/codex-b18fix2-report.md`.
- **Open attribution:** inotify has no PID; fileaudit samples the child fd inventory and
  cannot attribute every open to a process. Sync-root refusal removes the identified
  exposure, not every possible external opener. Historical G13 run-A timing is unresolved
  (`preflight/investigations/G13-G51/TICKET.md` §§2, 5).
- **Mutagen flag:** Kai's §14 ruling remains pending; transient transfer to the Mac is
  undetermined. The incident is not cleared by a passing rerun.
- **Logging:** native_share_removed still lacks an owner-defined baseline/denominator;
  strict="produced" is not the complete §12 catalogue (`code/NOTES-B18P.md`, B18P-1).
- **B19 evidence:** NRP runtime/data identity (G05/G09), G77 lint and upstream resource
  disposition, in-pod identity G79, fresh G80 inventory, production-product U1 draws,
  determinism, launchable smoke and real PVC kill/resume G83 remain due. Packing G84 must
  measure the actual diagnostics/replay workload, full lifecycle rates, memory and the
  rolling three-hour utilization window. Product/K certification is null until supported
  by those measurements; the PC pack-test skip does not satisfy it. Sources:
  `code/NOTES-B17.md`, remaining gates; `preflight/b18f/BRIEF.candidate.md`, packing and
  certification; `preflight/b18/final/GATES.md`, NOT RUN rows.
- **Binary sequencing:** the export headline and any conditional split-form extension
  remain separate from CH-8/F000 eligibility. Later config/admission changes need their
  recorded checks and renewed authority; final gates alone confer no launch approval
  (`notes/LEDGER.md`, 11:49Z; `preflight/b18f/BRIEF.candidate.md`).

### Blinding (B18f document reconstruction)

This update reads source text, gate summaries, reports, command records and git history.
No dataset arrays or held-out files were opened, no gate or training process was launched,
and no campaign code or worktree was written. Blinding for the live pass must come from its
own completed records. The earlier Mutagen incident remains disclosed above.
