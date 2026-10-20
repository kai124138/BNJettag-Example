"""Unit tests of cbl/wandb_log.py (M15; gate G52). W&B runs here only in disabled mode;
gates/g52_wandb.py runs the offline records in hermetic child processes."""
import json
import sys
from pathlib import Path

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import cbl.wandb_log as w          # noqa: E402
import wandb_fixtures as fx        # noqa: E402

SHA = "0123456789abcdef" * 2 + "01234567"           # a synthetic 40-hex code sha
BUNDLE = fx.SOURCE_BUNDLE


def disabled(cfg):
    return dict(cfg, wandb=dict(cfg["wandb"], mode="disabled"))


# ------------------------------------------------------------------------- run identity

def test_run_name_carries_every_field_and_round_trips():
    name = w.run_name("F011", "BF", 2, "S1", "ramp", 3, BUNDLE)
    assert name == "F011_BF_s2_S1_ramp_a3_src-" + BUNDLE[:12]
    assert w.parse_run_name(name) == {"arm": "F011", "form": "BF", "seed": 2, "stage": "S1",
                                      "schedule": "ramp", "attempt": 3, "source": BUNDLE[:12],
                                      "replica": None}
    assert w.run_name("CH-8", None, 1, "S1", "ramp", 1, BUNDLE).startswith("CH-8_nf_s1_")
    assert w.parse_run_name(w.run_name("PID-BEST-C", "FB", 11, "S4", "pid", 1, BUNDLE))["arm"] == "PID-BEST-C"


@pytest.mark.parametrize("args", [
    ("F012", "FB", 1, "S1", "ramp", 1),       # not a section-5 arm
    ("F011", "XX", 1, "S1", "ramp", 1),       # not a form
    ("PID-CH", None, 1, "S3", "ramp", 1),     # PID arm on the ramp
    ("CH-8", None, 1, "S1", "pid", 1),        # ramp arm under PID
    ("CH-8", None, 4, "S1", "ramp", 1),       # seed outside the screen
    ("CH-8-C", None, 3, "S4", "ramp", 1),     # seed outside S4
    ("CH-8", None, 1, "S2", "ramp", 1),       # S2 trains nothing
    ("CH-8", None, 1, "S1", "ramp", 0),       # attempt from 1
    ("CH-8", None, True, "S1", "ramp", 1),    # bool is not a seed
])
def test_run_name_refuses_bad_identity(args):
    with pytest.raises(w.WandbLogError):
        w.run_name(*args, BUNDLE)


@pytest.mark.parametrize("bundle", [None, "", "abc", BUNDLE.upper(), "g" * 64, BUNDLE + "00"])
def test_source_bundle_has_no_default(bundle):
    with pytest.raises(w.WandbLogError):
        w.run_name("CH-8", None, 1, "S1", "ramp", 1, bundle)


def test_engineering_stages_accept_any_seed():
    assert w.run_name("CH-8", None, 7, "smoke", "ramp", 1, BUNDLE).split("_")[3] == "smoke"


def test_run_id_is_deterministic_and_distinct_across_identities():
    base = w.run_name("F011", "FB", 1, "S1", "ramp", 1, BUNDLE)
    assert w.run_id(base) == w.run_id(base) and w.run_id(base).startswith("cbl") and len(w.run_id(base)) == 15
    others = [w.run_name("F011", "BF", 1, "S1", "ramp", 1, BUNDLE),
              w.run_name("F011", "FB", 2, "S1", "ramp", 1, BUNDLE),
              w.run_name("F011", "FB", 1, "S1", "ramp", 2, BUNDLE),
              w.run_name("F011", "FB", 1, "smoke", "ramp", 1, BUNDLE),
              w.run_name("F011", "FB", 1, "S1", "ramp", 1, fx.fake_sha("another bundle"))]
    assert len({w.run_id(n) for n in [base] + others}) == 6


# ------------------------------------------------- replica token (owner decision B12-5)

def test_replica_token_is_empty_for_single_runs():
    # replica None (every identity with one run) leaves the name and the run id unchanged
    for args in [("F011", "BF", 2, "S1", "ramp", 3), ("CH-8", None, 1, "det", "ramp", 1),
                 ("F000", None, 1, "det", "ramp", 1), ("PID-BEST", "FB", 1, "smoke", "pid", 1)]:
        assert w.run_name(*args, BUNDLE, None) == w.run_name(*args, BUNDLE)
        assert not any(part[:1] == "r" and part[1:].isdigit() for part in w.run_name_stem(*args).split("_"))
    assert w.run_name_stem("CH-8", None, 1, "det", "ramp", 1) == "CH-8_nf_s1_det_ramp_a1"
    assert w.run_name_stem("F000", None, 1, "det", "ramp", 1) == "F000_nf_s1_det_ramp_a1"


def test_replica_token_separates_the_determinism_pair():
    r1 = w.run_name("CH-8", None, 1, "det", "ramp", 1, BUNDLE, 1)
    r2 = w.run_name("CH-8", None, 1, "det", "ramp", 1, BUNDLE, 2)
    f0 = w.run_name("F000", None, 1, "det", "ramp", 1, BUNDLE)
    assert r1 == "CH-8_nf_s1_det_ramp_r1_a1_src-" + BUNDLE[:12]
    assert r2 == "CH-8_nf_s1_det_ramp_r2_a1_src-" + BUNDLE[:12]
    assert len({r1, r2, f0}) == 3 and len({w.run_id(r1), w.run_id(r2), w.run_id(f0)}) == 3
    assert w.parse_run_name(r2)["replica"] == 2 and w.parse_run_name(f0)["replica"] is None
    # a resubmission of replica 2 differs from replica 1 and from replica 2's first attempt
    r2a2 = w.run_name("CH-8", None, 1, "det", "ramp", 2, BUNDLE, 2)
    assert w.parse_run_name(r2a2) == dict(w.parse_run_name(r2), attempt=2)
    assert len({w.run_id(x) for x in (r1, r2, r2a2)}) == 3
    cfg = dict(fx.ch8_config(SHA), stage="det", replica=2)
    assert w.name_from_config(cfg, BUNDLE) == r2
    assert w.name_from_config(dict(cfg, replica=None), BUNDLE) == w.run_name("CH-8", None, 1, "det", "ramp", 1, BUNDLE)


@pytest.mark.parametrize("stage, replica", [("S1", 1), ("S3", 2), ("smoke", 1), ("bench", 1), ("gate", 1),
                                            ("det", 0), ("det", True), ("det", "2"), ("det", 1.0)])
def test_replica_token_refused_outside_det_or_malformed(stage, replica):
    with pytest.raises(w.WandbLogError):
        w.run_name("CH-8", None, 1, stage, "ramp", 1, BUNDLE, replica)


@pytest.mark.parametrize("name", ["CH-8_nf_s1_det_ramp_r0_a1_src-" + BUNDLE[:12],
                                  "CH-8_nf_s1_S1_ramp_r1_a1_src-" + BUNDLE[:12],
                                  "CH-8_nf_s1_det_ramp_r01_a1_src-" + BUNDLE[:12],
                                  "CH-8_nf_s1_det_r1_ramp_a1_src-" + BUNDLE[:12]])
def test_parse_refuses_bad_replica_names(name):
    with pytest.raises(w.WandbLogError):
        w.parse_run_name(name)


def test_disabled_open_names_the_replica(tmp_path):
    cfg = disabled(dict(fx.ch8_config(SHA), stage="det", replica=2))
    with pytest.raises(w.WandbLogError):       # replica 1's stem is not replica 2's
        _open(disabled(dict(cfg, wandb=dict(cfg["wandb"], run_name="CH-8_nf_s1_det_ramp_r1_a1", mode="disabled"))),
              run_dir=tmp_path / "run0")
    run = _open(cfg, run_dir=tmp_path / "run")
    name = run.name
    run.finish()
    assert name == "CH-8_nf_s1_det_ramp_r2_a1_src-" + BUNDLE[:12]
    stem_cfg = disabled(dict(cfg, wandb=dict(cfg["wandb"], run_name="CH-8_nf_s1_det_ramp_r2_a1", mode="disabled")))
    run2 = _open(stem_cfg, run_dir=tmp_path / "run2")
    name2 = run2.name
    run2.finish()
    assert name2 == name


# --------------------------------------------------------------------- held-out refusal

@pytest.mark.parametrize("key", ["test_top1", "val/test/auc", "testset_auc", "test1", "roc_Test",
                                 "heldout_auc", "held_out/top1", "held-out", "HeldOut", "hold_out",
                                 "holdout", "ckpt/x/held.out", "latest"])
def test_names_heldout_flags(key):
    assert w.names_heldout(key)


@pytest.mark.parametrize("key", ["val_top1", "H1", "cost/est", "train_count", "log_val_loss",
                                 "attention_diagnostics/delta_static", "threshold", "holder"])
def test_names_heldout_passes(key):
    assert not w.names_heldout(key)


def test_no_catalogue_key_names_heldout_data():
    assert not [s.pattern for s in w.KEYS if w.names_heldout(s.pattern)]


def test_check_keys_refuses_before_the_freeze(tmp_path):
    keys = ["val_top1", "test_top1"]
    assert w.check_keys(["val_top1"]) is None
    with pytest.raises(w.HeldOutKeyRefused, match="test_top1"):
        w.check_keys(keys, None)
    with pytest.raises(w.HeldOutKeyRefused):
        w.check_keys(keys, tmp_path / "absent.json")
    with pytest.raises(w.HeldOutKeyRefused):
        w.check_keys(keys, tmp_path)                        # a directory is not a manifest
    for bad in ("", "not json", "[]", "{}", "\xff"):
        p = tmp_path / "bad.json"
        p.write_bytes(bad.encode("latin-1"))
        with pytest.raises(w.HeldOutKeyRefused):
            w.check_keys(keys, p)


def test_check_keys_passes_after_the_freeze_and_returns_its_sha(tmp_path):
    p = tmp_path / "freeze.json"
    p.write_text(json.dumps({"dummy": "synthetic section-8.5 manifest"}))
    import hashlib
    assert w.check_keys(["test_top1"], p) == hashlib.sha256(p.read_bytes()).hexdigest()
    assert w.check_keys(["test_top1"], str(p)) == hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- flattening

def test_flatten_nests_and_keeps_strings():
    import numpy as np
    row = w.flatten({"a": {"b": 1, "c": [2.5, True]}, "h_sha256": "ab" * 32, "n": None,
                     "arr": np.array([[1, 2]], dtype=np.int64), "s": np.float32(0.5), "t": (1,)})
    assert row == {"a/b": 1, "a/c/0": 2.5, "a/c/1": 1, "h_sha256": "ab" * 32, "arr/0/0": 1,
                   "arr/0/1": 2, "s": 0.5, "t/0": 1}
    assert w.flatten({"x": 1}, "ckpt") == {"ckpt/x": 1}


@pytest.mark.parametrize("record", [{"a/b": 1, "a": {"b": 2}}, {"_step": 1}, {"a": b"x"},
                                    {"a": object()}, {1: 2}, [1]])
def test_flatten_refuses(record):
    with pytest.raises(w.WandbLogError):
        w.flatten(record)


# -------------------------------------------------------------------- key catalogue

def test_catalogue_maps_every_section12_line():
    lines = {s.study.split()[0] for s in w.KEYS}
    assert {"STUDY:1283", "STUDY:1286", "STUDY:1287", "STUDY:1288", "STUDY:1289", "STUDY:1290",
            "STUDY:1291", "STUDY:1292", "STUDY:1293", "STUDY:1295", "STUDY:1296-1298",
            "STUDY:1299", "STUDY:1301", "STUDY:1303"} <= lines
    assert set(t for s in w.KEYS for t in s.when) <= set(w.TRAITS)


@pytest.mark.parametrize("cfg,want", [
    (fx.config(SHA), {"cost", "quantized", "attention", "binary", "fused_bn_binary", "fb", "pid"}),
    (fx.ch8_config(SHA), {"cost", "quantized", "attention"}),
    (fx.fp32_config(SHA), {"attention"}),
    (fx.config(SHA, arm_id="DS-HGQ", control="DS-HGQ", groups={"attention": "Q", "ffn": "Q", "head": "Q"},
               form=None, schedule="ramp"), {"cost", "quantized"}),
    (fx.config(SHA, arm_id="F100", groups={"attention": "B", "ffn": "Q", "head": "Q"}, form=None,
               schedule="ramp"), {"cost", "quantized", "attention", "binary"}),
    (fx.config(SHA, arm_id="F011", groups={"attention": "Q", "ffn": "B", "head": "B"}, form="BF",
               schedule="ramp"), {"cost", "quantized", "attention", "binary", "fused_bn_binary"}),
])
def test_traits(cfg, want):
    assert w.traits(cfg) == frozenset(want)


@pytest.mark.parametrize("make", [fx.config, fx.ch8_config, fx.fp32_config])
def test_fixture_records_are_complete_and_every_requirement_can_fail(make):
    t = w.traits(make(SHA))
    for kind, row in (("epoch", w.flatten(fx.epoch_record(0, t))),
                      ("checkpoint", w.flatten(fx.checkpoint_sidecar(0, t), "ckpt"))):
        assert w.missing(row, kind, t) == []
        for spec in w.required(kind, t):
            cut = {k: v for k, v in row.items() if not w.matches(k, spec.pattern)}
            assert spec.pattern in w.missing(cut, kind, t), spec


def test_matching_is_per_segment():
    assert w.matches("pruning_channel/ffn/zero_width_frac", "pruning_channel/*/zero_width_frac")
    assert not w.matches("pruning_channel/ffn/input_zero_width_frac", "pruning_channel/*/zero_width_frac")
    assert w.matches("binary_sites/ffn_up/qg_b/2", "binary_sites/*/qg_b")
    assert not w.matches("binary_sites/a/b/qg_b", "binary_sites/*/qg_b")


# ------------------------------------------------------------------- Run.open refusals

def _open(cfg, prov=None, **kw):
    kw.setdefault("env", {})
    return w.Run.open(cfg, fx.provenance(cfg) if prov is None else prov, source_bundle=BUNDLE,
                      run_dir=kw.pop("run_dir", "/nonexistent-g52"), **kw)


@pytest.mark.parametrize("env", [{"WANDB_PROJECT": "other"}, {"WANDB_ENTITY": "other"},
                                 {"WANDB_GROUP": "other"}, {"WANDB_RUN_GROUP": "other"},
                                 {"WANDB_MODE": "offline"}])
def test_open_refuses_conflicting_environment(env):
    with pytest.raises(w.WandbLogError):
        _open(disabled(fx.config(SHA)), env=env)


def test_open_refusals_before_any_sdk_call():
    cfg = disabled(fx.config(SHA))
    with pytest.raises(w.WandbLogError):
        _open(cfg, mode="offline")                               # differs from the config's mode
    with pytest.raises(w.WandbLogError):
        _open(dict(cfg, wandb=dict(cfg["wandb"], mode="dryrun")))
    with pytest.raises(w.WandbLogError):
        _open(dict(cfg, wandb=dict(cfg["wandb"], project="other")))
    with pytest.raises(w.WandbLogError):
        _open(dict(cfg, wandb=dict(cfg["wandb"], run_name="CH-8_nf_s1_S1_ramp_a1")))
    with pytest.raises(w.WandbLogError):
        _open(cfg, expect_run_id="cbl000000000000")
    with pytest.raises(w.WandbLogError):
        _open(cfg, prov=dict(fx.provenance(cfg), code_sha="f" * 40))
    for field in w.PROVENANCE_REQUIRED:
        prov = fx.provenance(cfg)
        del prov[field]
        with pytest.raises(w.IncompleteRecord):
            _open(cfg, prov=prov)
    for field in w.CONFIG_REQUIRED:
        bad = dict(cfg)
        del bad[field]
        with pytest.raises(w.IncompleteRecord):
            _open(bad)
    with pytest.raises(w.HeldOutKeyRefused):
        _open(cfg, prov=dict(fx.provenance(cfg), test_sha256=fx.fake_sha("x")))


# ------------------------------------------------------------------------ disabled mode

def test_disabled_mode_checks_rows_and_writes_nothing(tmp_path):
    before = sorted(tmp_path.rglob("*"))
    cfg = disabled(fx.config(SHA, wandb={"project": fx.PROJECT, "group": fx.GROUP,
                                         "run_name": "PID-BEST_FB_s1_S3_pid_a1", "mode": "disabled"}))
    run = _open(cfg, run_dir=tmp_path / "run")
    assert run.mode == "disabled" and run.name == "PID-BEST_FB_s1_S3_pid_a1_src-" + BUNDLE[:12]
    run.log_epoch(fx.epoch_record(0, run.traits))
    run.log_checkpoint(fx.checkpoint_sidecar(0, run.traits))
    with pytest.raises(w.IncompleteRecord):
        run.log_epoch(fx.without(fx.epoch_record(1, run.traits), "pid/slack"))
    with pytest.raises(w.IncompleteRecord):
        run.log_checkpoint(fx.without(fx.checkpoint_sidecar(1, run.traits), "val_auc_macro"))
    with pytest.raises(w.HeldOutKeyRefused):
        run.log_epoch(dict(fx.epoch_record(1, run.traits), test_top1=0.5))
    with pytest.raises(w.HeldOutKeyRefused):
        run.summary({"heldout/auc": 0.9})
    assert sorted(tmp_path.rglob("*")) == before
    manifest = tmp_path / "freeze.json"
    manifest.write_text(json.dumps({"dummy": True}))
    run.set_freeze_manifest(manifest)
    run.summary({"heldout/auc": 0.9})
    assert run.freeze_sha256 is not None
    run.finish()
    assert sorted(tmp_path.rglob("*")) == [manifest]


# ------------------------------------------------------------- [B10] strict level "produced"

def _strip_not_produced(record, run_traits):
    """The fixture record without every key the fork does not produce yet (cbl.wandb_log.NOT_PRODUCED)."""
    flat = w.flatten(record)
    return {k: v for k, v in flat.items() if not any(w.matches(k, p) for p in w.NOT_PRODUCED)}


def test_not_produced_patterns_are_catalogue_keys():
    pats = {s.pattern for s in w.KEYS}
    assert set(w.NOT_PRODUCED) <= pats
    assert all(p in pats for p, _, _ in w.ROW_EXEMPT)
    assert not any(p.startswith(("H1", "val_", "pid/", "batch_index", "index_map", "subtotals")) for p in w.NOT_PRODUCED)


@pytest.mark.parametrize("make", [fx.config, fx.ch8_config])
def test_produced_level_refuses_every_produced_key_and_passes_the_rest(make):
    t = w.traits(make(SHA))
    row = _strip_not_produced(fx.epoch_record(1, t), t)
    assert w.missing_produced(row, "epoch", t) == []
    assert w.missing(row, "epoch", t) != []                  # the full catalogue still refuses it
    for spec in w.required("epoch", t):
        if spec.pattern in w.NOT_PRODUCED:
            continue
        cut = {k: v for k, v in row.items() if not w.matches(k, spec.pattern)}
        assert spec.pattern in w.missing_produced(cut, "epoch", t), spec


def test_produced_level_row_exemptions_hold_only_under_their_condition():
    t = w.traits(fx.config(SHA))                             # PID-BEST FB: pid and binary traits
    assert {"pid", "binary"} <= set(t)
    row = _strip_not_produced(fx.epoch_record(3, t), t)
    no_err = {k: v for k, v in row.items() if k != "pid/error"}
    assert "pid/error" in w.missing_produced(dict(no_err, **{"pid/clamp_state": "none"}), "epoch", t)
    assert "pid/error" not in w.missing_produced(dict(no_err, **{"pid/clamp_state": "warmup"}), "epoch", t)
    no_flip = {k: v for k, v in row.items() if not w.matches(k, "binary_sites/*/sign_flip_rate")}
    assert "binary_sites/*/sign_flip_rate" in w.missing_produced(no_flip, "epoch", t)
    assert "binary_sites/*/sign_flip_rate" not in w.missing_produced(dict(no_flip, epoch=0), "epoch", t)


def test_produced_level_in_a_disabled_run(tmp_path):
    cfg = disabled(fx.config(SHA))
    run = _open(cfg, run_dir=tmp_path / "run", strict="produced")
    rec = _strip_not_produced(fx.epoch_record(2, run.traits), run.traits)
    run.log_epoch(rec)
    with pytest.raises(w.IncompleteRecord):
        run.log_epoch({k: v for k, v in rec.items() if k != "pid/slack"})
    with pytest.raises(w.IncompleteRecord):
        run.log_epoch(fx.without(rec, "H1"))


@pytest.mark.parametrize("bad", [1, 0, "yes", None, "Produced"])
def test_strict_levels_are_closed(bad):
    with pytest.raises(w.WandbLogError):
        _open(disabled(fx.config(SHA)), strict=bad)
