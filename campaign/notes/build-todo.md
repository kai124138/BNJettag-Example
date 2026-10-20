# Cross-batch build items (orchestrator)

Items found in one batch that a later batch must carry. Each names its source and the batch that
takes it. Struck through when done, with the commit.

- `gen.check_launchable` is tested but not called on the launch path. The runner (M13) must call it in
  any non-gate run, and the pack runner and launcher adapter (M20, B17) before submission. Source: B12.
  Takes: B13 for the runner, B17 for M20.
- `pid.max_beta` is written in configs as the JSON string "inf"; M12 (B9) must read it so. Source: B12.
- G57 checks reseeds and W&B names against `cbl.rng` and `cbl.wandb_log` as they are when it runs; any
  change to either module needs configs regenerated before G57 can pass again. Source: B12. Takes: every
  batch that touches M4 or M15; B18.
- `cbl/select.py:311` cites interfaces "section 10" (now 11); `da4ml_clean/` cites "section 9" (now 12).
  Docstring-only fixes. Takes: the next lane-P commit that touches those files, or B18.
- B10 (resume, M14): the Dataset's `_index_map` is separate state, and a resumed `fit` repeats Keras's
  pre-fit draws (B4 finding; interfaces §10). Takes: B10.
- B4 pending parts: G10 and G11 for F000 (B5) and the binary arms (B6); G12 on the frozen evaluation
  (B7); G11 and G12 rerun with the runner M13 (B5); the 200-batch real-data case (B13).
- G52 PCC run with the standard-library reader and G56 PCC part (data.py joined to
  `cbl.select.require_freeze_manifest`). Takes: B5 (in its brief).
- G58 and G60 (configs build on the PC, shared initialization and callback order). Takes: after B6, on
  the PC lane.
- STUDY C items open (non-blocking): critical-v0.9-amendment C1-C4, C6-C9; constructive-v0.7-delta "for
  later" 2, 4, 5, 7, 8; B1 G03 notes (§7.3 two unnamed behavioral differences; the fallback's line cite
  384-386). Takes: a single C-item pass before the PREFLIGHT critical review, or the REPORT.
- `_alkaid_keras_plugin/` differs from the snapshot and is not cited in §7.3; it is on the export path.
  Source: B1 G04. Takes: B14.
- Branch `b12` (head `f78d60c576f2`; M17 generator, configs, G57, G59; B12-5 replica token in M15 run names)
  merges after B5. M15 changed in it, so G52's PCC row must be rerun after that merge, and configs
  regenerated if B5 touched `cbl.rng` or `cbl.wandb_log`. Takes: B6 (leftover), orchestrator at merge.
- Branch `b12` is now at `08417e78a19c` (includes main 46cc25f; configs conform to `validate_config`; rows
  `preflight/gates/G57/08417e78a19c/PREFLIGHT-rows.md`, owner decisions B12-12 to B12-14). Merge after B6;
  rerun G52 PCC after the merge; regenerate configs if B6 touched `cbl.rng`, `cbl.wandb_log` or
  `cbl.runner.RAMP_RECORD`, then rerun G57.
- Runner admission still to come: the 38 PID configs (B9), CH-8-C for the 21 S4 configs (a runner change,
  B9 or B10), and the FP32, W8A8 and DS-HGQ controls (15 configs; `config_code` null in interfaces §5 versus
  F000 derived by `validate_config`, B12-13; whoever builds the controls settles it).
- B6 done at `b2838835b733`: binary parts of B4's pending G10 and G11 are filled (struck from the B4 item).
  For the PREFLIGHT critical review: gates restated after a development run, each disclosed in PREFLIGHT.md
  — G03's GPU control (B1), G11's control E→F (B4), G24's exactness check (B6-9; HGQ's float32 rounding
  returns one step instead of 0 within one ulp below half a step, for Chang's quantizers as for q_g), G20's
  end-to-end tolerance (B6). The reviewer should judge each.
- STUDY C items from B6: q_g also carries a uint8 `k`, so §6 step 4's unmatched set is k, b and i (wording);
  the float32 half-step rounding behaviour above (a §2.2 note).
- B14 (G65): binary layers raise AttributeError on `kq`; the export gate shows whether the converter needs it.
- Branch `b8` (head `e59dfe74a49e`; M9 front, M10 structure, M11 ablations; G37-G42): G41 PCC and G37 PCC at e59dfe7
  wait for a quiet PC (the open-file audit is machine-wide and caught B7's concurrent gate traffic; run
  `cd ~/lab/scratch/cbl1016 && QUIET_POLLS=12 ./b8-run-quiet.sh e59dfe74a49e pcc G41 G37` after B7). Then merge b8.
  Lesson: two PC lanes interfere through the open-file audit; run audited gates one lane at a time.
- Runner wiring for M9 (interfaces §17; after b8 and B7 merge): `Context.front = None`; in `prepare`, before
  `frozen_eval_hook`, `ctx.front = front.register(ctx, ablations=..., keep_superseded=False)` and
  `evaluator = front.chain_evaluator(evaluator, ctx.front)`; in `finish`, `ctx.front.finalize(end=...,
  epochs_completed=..., released_front=front.released_front_of(ctx.released[4], ctx.out_dir))`. Takes: B9.
- B10: `.keras` bytes differ between two saves of one state, so resume compares `state_sha256`, not
  `ckpt_sha256` (B8-2).
- Controls batch: M10/M11 `Graph` refuses DS-HGQ and FP32; they need a route. STUDY C item B8-10: QAdd's branch
  input quantizer is the first downstream WRAP quantizer for every ablation; §8.8 does not list it.
- B7 done at `a74cd9d2b977` (M7 `cbl/cost.py`, M8 `cbl/frozen_eval.py` in the runner; G26-G34, G36, G12 part, G58, G60,
  G52 PCC PASS). B3a-3 settled at the tested states (NR equals the native counter of a compiled step, CPU and 4060 Ti);
  the production GPU product is B19's.
- The parameter-count record `preflight/gates/G58/780b3d908ad6/pcc/param-counts.json` is not wired into the configs:
  regenerate and rerun G57 (B7-11). Takes: B9 or B10.
- M15 strict mode stays off until B8 (structure and pruning keys) and B9 (PID block) are in. Takes: B10.
- For the PREFLIGHT critical review: G13's release side now runs without M8 (B7-8); G58 restated after a development
  run for finding B7-F1 (after a `.keras` reload the softmax and its tables come back with beta0 1e-5 instead of 0;
  beta variables restored by value, no number changes). B7-F1 also matters for M14 (B10) and M18 (B14).
- M8 cost per epoch on the 2,000-jet smoke validation set: about 0.1-0.2 s CPU, 0.3-0.6 s on the 4060 Ti
  (engineering floor; production validation has 62,000 jets); it counts against B19's measured rate.
- Done in B9 (`07c33ba5fcf0`): the M9 runner wiring (`6da2a1e`), `pid.max_beta` read as "inf" (`70fdb50`), PID admission
  (`528789d`; 22 of 38 PID configs pass, PID-BEST and PID-BEST-C engineering-only until the runner reads the S2 record).
  G57 at the head: 88 configs pass, 79 refused by reason.
- Kai, 2026-10-18: mulder is off the path to training (B14, B15 and G35, G50 export part, G61-G75 leave the launch path;
  synthesis of the deciding points follows training). B13 folds into the NRP smoke. `check_launchable` wiring moves to B10.
- Kai, 2026-10-19: everything on NRP stays inside the `kai-data` PVC (`/data/cbl1016/`): code bundle and checkout, archives,
  venv, handoff records, run directories, checkpoints, fronts, epoch records, logs, W&B offline files. No ConfigMaps, no
  hostPath, no other volume except emptyDir for transient caches. Takes: Phase 2 (run_handoff bundle mode and its check),
  M20 pack runner and Job generator (run roots on the PVC), the harness BRIEF (operations block).
- Phase 2 done 2026-10-19 (cluster-ops): `/data/cbl1016/` holds the code bundle (commit f8ca91d7, superseded at B18), the
  training and smoke archives and the venv (`runtime/venv-c8725569731a`); G09 and G05 PASS (rows in
  `preflight/nrp/PREFLIGHT-rows.md`, to merge into PREFLIGHT.md after B10). PVC free space 57 G of 100 G before staging.
- B17 Job generator: keep CPU and memory limits within 1.2 x requests (NRP warned on the venv Job).
- Launch path: the Mac pre-kubectl lint hook could not lint while the Mac token was expired and blocked the helper Pod;
  cluster-ops ran the same lint by hand on the PC (exit 0) and created the Pod from the PC. From now on every create goes
  through a path where the hook runs (the Mac with a valid token; Kai signed in 2026-10-19), or the hook is installed on
  the PC first.
- Branch `b17` (head `a474b51`; Codex gpt-6-astra, committed by the orchestrator; M20 `nrp/`, M21 `tools/`, G76 and the
  offline part of G78 PASS; 51 offline tests pass) merges after B10. Draft BRIEF: `preflight/b17/BRIEF.draft.md`. Before
  Kai's approve, the retry policy changes from "two automatic relaunches per pack lineage" to Kai's instruction ("if
  things fail or queue fails, just loop until it succeeds"): unlimited relaunch for infrastructure failures (eviction,
  preemption, node fault, scheduling timeout) within the GPU-h cap; stop on a code or integrity failure (a fix is a new
  commit). Cluster gates G77, G79, G80 remain. `tests/test_env.py::test_setup_order_records_and_bindings` fails on the
  Mac venv on main too (it needs the PC's JAX setup); not a B17 regression.
- B10 done at `267c6deb053e` (M14 resume, launch check in the runner, held-out guard, strict "produced", CH-8-C admitted,
  parameter counts wired): struck from this list. b17 merged into main at `6cdbd05e65b9`. Open from B10: B10-F1 (2 of ~40 fresh
  F111-FB CPU runs differ from epoch 0; investigator running, ticket `preflight/investigations/B10-F1.md`); B10-F2 (27 §12 W&B
  keys produced by no module) and B10-4/B10-F3 (resumed pods pin the GPU product, a directory killed before train begin is
  set aside, the same --resume-every) on Codex, branch `b18p`.
- b18p merged into main (Codex; 26 of 27 §12 keys produced, `native_share_removed` needs a defined counterfactual, a STUDY
  C item; resumed pods keep the GPU product and --resume-every; pre-train kills set aside). 19 PC test cases deferred
  (`preflight/b18p/pc-tests.txt`): run them on the PC at the final sha (B18). B18 waits for: the B10-F1 investigation,
  the export check outcome (headline per rule v0.4), and Kai's answers on items 2-4 and 12.
