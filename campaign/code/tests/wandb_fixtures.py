"""Synthetic configs and records for G52 and tests/test_wandb_log.py (no Keras, JAX or data).

The records follow interfaces.md sections 1, 2 and 7 by hand, so that a mismatch between the
section-12 catalogue (`cbl.wandb_log.KEYS`) and the documented record shape shows up as a missing
key. Every value is synthetic; nothing here is a reading of any model.
"""
import copy
import hashlib
import json

PROJECT = "bnjettag-chang-bitlinear"
GROUP = "chang-bitlinear-20261016"
SOURCE_BUNDLE = hashlib.sha256(b"g52 synthetic source bundle").hexdigest()


def fake_sha(label):
    return hashlib.sha256(("g52-synthetic:" + label).encode()).hexdigest()


def config(code_sha, **over):
    """Every trait on: PID-BEST in form FB with all three groups binary (S3, seed 1)."""
    cfg = {
        "schema_version": 1, "campaign": GROUP, "arm_id": "PID-BEST", "builder": "lab",
        "config_code": "F111", "groups": {"attention": "B", "ffn": "B", "head": "B"},
        "embedding": "Q", "form": "FB", "form_source": "s2_choice:" + fake_sha("s2"),
        "best_source": "s2_choice:" + fake_sha("s2"), "control": None, "seed": 1, "stage": "S3",
        "tier": "T1", "schedule": "pid",
        "pid": {"target_H": 350000, "init_beta": 2e-8, "p": 1, "i": 0.002, "d": 0, "warmup": 10,
                "log": True, "max_beta": "inf", "min_beta": 0, "damping": 0},
        "epochs": 7000, "cost_callbacks": True, "headline": None, "claim_domain": None,
        "matmul_precision": "float32", "code_sha": code_sha, "attempt": 1,
        "wandb": {"project": PROJECT, "group": GROUP, "run_name": None, "mode": "offline"},
    }
    cfg.update(over)
    return cfg


def ch8_config(code_sha, **over):
    cfg = config(code_sha, arm_id="CH-8", builder="released", config_code=None, groups=None,
                 embedding=None, form=None, form_source=None, best_source=None, stage="S1",
                 tier="T0", schedule="ramp", pid=None)
    cfg.update(over)
    return cfg


def fp32_config(code_sha, **over):
    cfg = config(code_sha, arm_id="FP32", builder="lab", config_code=None,
                 groups={"attention": "Q", "ffn": "Q", "head": "Q"}, embedding="Q", form=None,
                 form_source=None, best_source=None, control="FP32", stage="S3", tier="T2",
                 schedule="ramp", pid=None, cost_callbacks=False)
    cfg.update(over)
    return cfg


def provenance(cfg):
    return {
        "code_sha": cfg["code_sha"],
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "data_manifest_sha256": fake_sha("data manifest"), "split_sha256": fake_sha("split"),
        "order_sha256": fake_sha("order"), "backend": "jax (synthetic G52 record)",
        "matmul_precision": "float32", "gpu_product": "none (synthetic G52 record)", "pack_k": 1,
    }


def _counts(base):
    return {"H1": base, "H2": base + 7, "NR": base + 11, "native": base + 13}


def epoch_record(epoch, run_traits):
    t = set(run_traits)
    r = {
        "run_id": "synthetic", "epoch": epoch, "step": 50 * (epoch + 1),
        "val_correct": 41000 + epoch, "val_n": 62000, "val_top1": (41000 + epoch) / 62000,
        "log_loss": 0.9, "log_accuracy": 0.66, "log_val_loss": 0.91, "log_val_accuracy": 0.65,
        "lr": 3e-3, "batch_index_seq_sha256": fake_sha(f"batch {epoch}"),
        "index_map_sha256": fake_sha(f"index map {epoch}"),
        "vars_sha256_before": fake_sha(f"vars {epoch}"), "vars_sha256_after": fake_sha(f"vars {epoch}"),
        "snap_changed": 0,
        "bn": {"ffn_up": {"gamma_min": 0.5, "gamma_max": 1.5, "moving_mean_absmax": 0.2,
                          "moving_var_min": 0.8, "moving_var_max": 1.2}},
        "sys": {"epoch_seconds": 3.0, "wall_seconds": 3.0 * (epoch + 1), "host_rss_mb": 900.0,
                "alloc_hours": 0.001 * (epoch + 1)},
    }
    if "cost" in t:
        r.update(beta=2e-8, train_count=330000.0, H1=300000, H2=310000, NR=320000,
                 subtotals={"layer": {"mha.query": _counts(1000), "ffn_up": _counts(2000)},
                            "group": {"attention": _counts(5000), "ffn": _counts(9000)}},
                 dwell={"H1": {"300000-350000": epoch + 1}, "H2": {"300000-350000": epoch + 1},
                        "NR": {"300000-350000": epoch + 1}})
    if "quantized" in t:
        r["quantizers"] = {"ffn_down": {"kq": {"width_mean": 3.2, "width_max": 6, "step_min": 2 ** -6,
                                               "step_max": 2 ** -1}}}
        r["quantizer_structure"] = {"ffn_down": {"kernel_zero_pruned": 0, "kernel_zero_rounded": 0,
                                                 "input_all_zero_width": False,
                                                 "output_all_zero_width": False}}
        if "attention" in t:
            r["quantizer_structure"]["softmax_heads"] = {"h0": {"zero_width_frac": 0.0},
                                                         "h1": {"zero_width_frac": 0.01}}
        r["activations"] = {"mha.softmax_out": {"min": 0.0, "max": 1.0, "zero_width_frac": 0.02}}
        r["pruning_channel"] = {"ffn_down": {"zero_width_frac": 0.1, "native_share_removed": 0.05}}
    if "binary" in t:
        r["binary_sites"] = {
            "mha.query": {"qg_b": 4, "qg_i": 1, "qg_value": 0.25, "zero_cols": {"pruned": 0, "rounded_to_zero": 0},
                          "saturated_cols": 0, "sign_occupancy": 0.5, "sign_flip_rate": 0.01,
                          "moment_mismatch_mean": 0.001, "moment_mismatch_var": 0.002},
        }
        r["pruning_channel"]["mha.query"] = {"input_zero_width_frac": 0.0}
    if "fused_bn_binary" in t:
        r["binary_sites"]["ffn_up"] = {
            "qg_b": [4, 4, 3], "qg_i": [1, 1, 0], "qg_value": [0.25, 0.25, 0.5],
            "zero_cols": {"pruned": 0, "rounded_to_zero": 1}, "saturated_cols": 0,
            "sign_occupancy": 0.5, "sign_flip_rate": 0.02, "moment_mismatch_mean": 0.003,
            "moment_mismatch_var": 0.004, "single_sign_cols": 0, "dead_channel_frac": 0.0}
        r["train_frozen_top1_gap"] = 0.002
    if "fb" in t:
        r["binary_sites"]["ffn_up"]["fb_sign_diff_frac"] = 0.001
    if "pid" in t:
        r["pid"] = {"sensed_H": 300000, "stock_aggregate": 330000.0, "beta": 2e-8, "error": -0.1,
                    "integral": 0.01, "clamp_state": 0, "slack": 50000}
    return r


def checkpoint_sidecar(epoch, run_traits):
    t = set(run_traits)
    s = {
        "run_id": "synthetic", "epoch": epoch, "ckpt_path": f"front/e{epoch:04d}.keras",
        "ckpt_sha256": fake_sha(f"ckpt {epoch}"), "val_correct": 41000 + epoch, "val_n": 62000,
        "val_top1": (41000 + epoch) / 62000, "H1": 300000, "H2": 310000, "NR": 320000,
        "joined": ["H1", "NR"], "chang_filter": {"H1": True, "H2": True, "NR": True},
        "val_auc_macro": 0.9, "val_auc_per_class": [0.91, 0.88, 0.9, 0.93, 0.87],
        "structure_record": {"degenerate": {"mha.query": "none"},
                             "zero_cols": {"ffn_up": {"pruned": 0, "rounded_to_zero": 1}}},
    }
    if "attention" in t:
        a = {"H_attn": 1.9, "all_zero_rows": 0, "negative_rows": 0, "branch_norm": 0.7,
             "delta_attn_mean": 0.01, "delta_attn_zero": 0.03, "delta_static": 0.005,
             "delta_unif": 0.02}
        if "quantized" in t:
            for p in ("q", "k", "v", "o", "qk", "av"):
                a[p] = {"cost": 1000, "width_mean": 3.0}
            a["softmax_zero_width"] = {"h0": 0.0, "h1": 0.01}
            a["degenerate"] = {"q": {"kernel_pruned": 0, "kernel_rounded_to_zero": 0,
                                     "input_zero_width": 0, "output_zero_width": 0, "other": 0}}
        s["attention_diagnostics"] = a
    return s


def without(record, dotted):
    """A deep copy of `record` with the '/'-separated key removed."""
    r = copy.deepcopy(record)
    parts = dotted.split("/")
    d = r
    for p in parts[:-1]:
        d = d[p]
    del d[parts[-1]]
    return r
