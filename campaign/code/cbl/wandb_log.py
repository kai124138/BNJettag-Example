"""W&B logger of chang-bitlinear-20261016 (module M15; STUDY v0.8 section 12; gate G52).

STUDY means `review/snapshots/v0.8/STUDY.md` (sha256 290022fa41a726b8...). Section 12
(STUDY lines 1281-1303) fixes the project `bnjettag-chang-bitlinear` and the group
`chang-bitlinear-20261016`, says that run names carry arm, form, seed, stage, ramp or PID,
attempt and source bundle, lists the logged quantities, and ends "No test metric appears before
the frozen final evaluation". W&B is an index only; the durable evidence is the files.

What this module does
- Run identity. `run_name` builds `<arm>_<form>_s<seed>_<stage>_<schedule>[_r<replica>]_a<attempt>_src-<12 hex>`
  (form `nf` when the config has none; arm ids contain hyphens, so fields are joined by `_`).
  The replica token is empty for every identity with a single run, so those names are unchanged;
  it exists only in stage `det`, where STUDY section 13's determinism record runs the same CH-8
  seed-1 identity twice (config `replica` 1 and 2; owner decision B12-5, batch B12).
  The source bundle is an explicit input with no default (fail closed): it is the content hash of
  the bundle a run was launched from, which is not known when configs are generated, so the
  config's `wandb.run_name` holds the stem without it (interfaces.md section 8). The run id is
  `cbl` plus the first 12 hex of sha256(run name): deterministic, so a resumed attempt continues
  its own run online, and a new attempt, stage or bundle gets a new run.
- Every section-12 key. `KEYS` maps each section-12 item, with its STUDY line, to the key names
  of the M8 frozen record (interfaces.md sections 1, 2, 8). `Run.open` refuses a run config
  without its required keys; `Run.log_epoch` and `Run.log_checkpoint` refuse a row without them
  unless the run was opened with `strict=False`. [B10] `strict="produced"`, the runner's level,
  refuses a row without any required key except the patterns no module produces yet
  (`NOT_PRODUCED`) and the keys a row may lack for a stated reason (`ROW_EXEMPT`). Which conditional items apply comes from
  `traits(cfg)`. Settings: `disable_git`, no code saving, no console capture, quiet.
- Modes. `offline` and `disabled` are exercised in PREFLIGHT; `online` exists in code and is never
  exercised there. The mode comes from the config (`wandb.mode`) or the `mode` argument, never
  from the presence of a credential. A different non-empty `WANDB_MODE`, `WANDB_PROJECT`,
  `WANDB_ENTITY`, `WANDB_GROUP` or `WANDB_RUN_GROUP` in the environment is refused.
- Credentials. This module never reads, prints, logs or sets the W&B credential, never opens a
  credential file, never calls the SDK's login, and never writes the environment. In online mode
  the SDK finds the credential itself.
- Held-out keys. A metric, summary or config key that names test or held-out data (lower-cased
  key contains `test`, or contains `heldout` or `holdout` once `-`, `_`, `.`, `/` and spaces are
  removed) is refused before the section-8.5 freeze manifest exists. The check takes the manifest
  path as input; no path, a missing file, or a file that is not a non-empty JSON object means
  refuse. The rule over-refuses words such as `latest` on purpose (fail closed). Completeness of
  the manifest is M16's check (G56); this module records the manifest's sha256 in the run summary.
- No Keras, JAX or TensorFlow import; `wandb` is imported only inside `Run.open`.

Pattern sources (read-only; that campaign is paused), copied as patterns, not as code:
- `campaigns/2026-10-12-mha-bitnet/code/bnhgq2/mhab_wandb.py`,
  sha256 5b1595f73a3ff2a5f7d3ed2e2b5afa4cd3a1af9575cb1b959ff4af500382fe93 (fork main 6a145873):
  constants with an environment-conflict refusal, a deterministic run id with resume='allow',
  `epoch` as the x axis of every metric (`define_metric`) without passing the W&B step, the
  dict and list flattening, adopt-or-refuse of an already open run, `finish`.
  Not taken: its mode inference from the presence of a credential variable; its TensorFlow device
  and memory readings (the runner passes backend, GPU product and memory); dropping string values
  (section 12 logs per-epoch hashes); the call journal; model artifacts (out of the M15 row).
- `campaigns/2026-10-12-mha-bitnet/code/bnhgq2/wandb_util.py`,
  sha256 3076ff7c39d942c3f778b637d1d8824e23e664978d9c30cd94f3ff9060456d6f:
  the stage-keyed run id (sha256 of the identity, truncated to 12 hex). Not taken:
  `wandb_enabled`, which opens a credential file, and project resolution from the environment.

Entity `kayamaguchi-uc-san-diego` is the lab's (`docs/infrastructure/wandb-layout.md`, identity
table); STUDY section 12 names none. Offline runs ignore `resume`, so after a kill the resumed
process writes a second offline directory with the same run id (M14 must know this before any
offline resume is synced). Local readings are engineering, not quotable.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from fnmatch import fnmatchcase
from pathlib import Path
from typing import NamedTuple

PROJECT = "bnjettag-chang-bitlinear"
GROUP = "chang-bitlinear-20261016"
ENTITY = "kayamaguchi-uc-san-diego"
MODES = ("online", "offline", "disabled")
ARMS = ("CH-8", "F000", "F100", "F001", "F010", "F011", "F101", "F110", "F111", "PID-CH",
        "PID-BEST", "FP32", "W8A8", "DS-HGQ", "GR", "BE", "CH-8-C", "BEST-C", "PID-CH-C",
        "PID-BEST-C")
FORMS = ("FB", "BF")
NO_FORM = "nf"
SCHEDULES = ("ramp", "pid")
CAMPAIGN_STAGES = {"S1": range(1, 4), "S3": range(1, 4), "S4": range(4, 12)}
ENGINEERING_STAGES = ("smoke", "bench", "det", "gate")
REPLICA_STAGES = ("det",)   # the section-13 determinism record runs one identity twice (B12-5)
SETTINGS = {"disable_git": True, "save_code": False, "console": "off", "quiet": True}
CONFIG_REQUIRED = ("arm_id", "seed", "stage", "schedule", "attempt", "cost_callbacks")
ENV_MUST_MATCH = (("WANDB_PROJECT", PROJECT), ("WANDB_ENTITY", ENTITY),
                  ("WANDB_GROUP", GROUP), ("WANDB_RUN_GROUP", GROUP))

_NAME = re.compile(r"^(?P<arm>[A-Z0-9-]+)_(?P<form>FB|BF|nf)_s(?P<seed>[1-9][0-9]*)_"
                   r"(?P<stage>S[134]|smoke|bench|det|gate)_(?P<schedule>ramp|pid)_"
                   r"(?:r(?P<replica>[1-9][0-9]*)_)?a(?P<attempt>[1-9][0-9]*)_src-(?P<source>[0-9a-f]{12})$")
_HEX = re.compile(r"^[0-9a-f]{12,64}$")


class WandbLogError(RuntimeError):
    """A run identity, mode or record that this module refuses."""


class HeldOutKeyRefused(WandbLogError):
    """A key naming test or held-out data, before the section-8.5 freeze manifest exists."""


class IncompleteRecord(WandbLogError):
    """A row without a key that section 12 requires for this run."""


# ---------------------------------------------------------------- run identity (section 12)

def replica_token(replica, stage):
    """'' for an identity with a single run (replica None); '_r<replica>' only in stage det."""
    if replica is None:
        return ""
    if not isinstance(replica, int) or isinstance(replica, bool) or replica < 1:
        raise WandbLogError(f"replica {replica!r} is neither None nor a positive int")
    if stage not in REPLICA_STAGES:
        raise WandbLogError(f"stage {stage!r} has one run per identity; a replica token exists only in "
                            f"{REPLICA_STAGES} (STUDY section 13 determinism record)")
    return f"_r{replica}"


def run_name_stem(arm, form, seed, stage, schedule, attempt, replica=None):
    if arm not in ARMS:
        raise WandbLogError(f"arm {arm!r} is not a STUDY section 5 arm id")
    if form is not None and form not in FORMS:
        raise WandbLogError(f"form {form!r} is neither FB, BF nor None")
    if schedule not in SCHEDULES:
        raise WandbLogError(f"schedule {schedule!r} is neither ramp nor pid")
    if (schedule == "pid") != arm.startswith("PID-"):
        raise WandbLogError(f"arm {arm} with schedule {schedule}: PID arms and only they use pid")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 1:
        raise WandbLogError(f"seed {seed!r} is not a positive int")
    if stage in CAMPAIGN_STAGES:
        if seed not in CAMPAIGN_STAGES[stage]:
            raise WandbLogError(f"seed {seed} is outside stage {stage}'s seeds (STUDY sections 5, 6)")
    elif stage not in ENGINEERING_STAGES:
        raise WandbLogError(f"stage {stage!r} is not S1, S3, S4 or one of {ENGINEERING_STAGES}")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise WandbLogError(f"attempt {attempt!r} is not a positive int")
    return f"{arm}_{form or NO_FORM}_s{seed}_{stage}_{schedule}{replica_token(replica, stage)}_a{attempt}"


def run_name(arm, form, seed, stage, schedule, attempt, source_bundle, replica=None):
    if not isinstance(source_bundle, str) or not _HEX.match(source_bundle):
        raise WandbLogError("source_bundle must be 12-64 lowercase hex (the bundle's sha256); "
                            "there is no default")
    return f"{run_name_stem(arm, form, seed, stage, schedule, attempt, replica)}_src-{source_bundle[:12]}"


def parse_run_name(name):
    m = _NAME.match(name)
    if not m:
        raise WandbLogError(f"{name!r} is not a chang-bitlinear run name")
    d = m.groupdict()
    d["form"] = None if d["form"] == NO_FORM else d["form"]
    d["seed"], d["attempt"] = int(d["seed"]), int(d["attempt"])
    d["replica"] = None if d["replica"] is None else int(d["replica"])
    if run_name(d["arm"], d["form"], d["seed"], d["stage"], d["schedule"], d["attempt"],
                d["source"], d["replica"]) != name:
        raise WandbLogError(f"{name!r} does not rebuild from its own fields")
    return d


def run_id(name):
    return "cbl" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]


def name_from_config(cfg, source_bundle):
    return run_name(cfg["arm_id"], cfg.get("form"), cfg["seed"], cfg["stage"], cfg["schedule"],
                    cfg["attempt"], source_bundle, cfg.get("replica"))


# ---------------------------------------------------------- held-out refusal (section 8.5)

def names_heldout(key):
    low = key.lower()
    squeezed = re.sub(r"[-_./ ]", "", low)
    return "test" in low or "heldout" in squeezed or "holdout" in squeezed


def freeze_manifest_sha256(path):
    """sha256 of the freeze manifest at `path`; HeldOutKeyRefused if there is none."""
    if path is None:
        raise HeldOutKeyRefused("no section-8.5 freeze manifest path was given")
    p = Path(path)
    if not p.is_file():
        raise HeldOutKeyRefused(f"section-8.5 freeze manifest {p} does not exist")
    data = p.read_bytes()
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise HeldOutKeyRefused(f"freeze manifest {p} is not JSON") from None
    if not isinstance(obj, dict) or not obj:
        raise HeldOutKeyRefused(f"freeze manifest {p} is not a non-empty JSON object")
    return hashlib.sha256(data).hexdigest()


def check_keys(keys, freeze_manifest=None):
    """Refuse held-out keys unless the freeze manifest exists. Returns the manifest's sha256
    when a held-out key was allowed, else None."""
    held = sorted(k for k in keys if names_heldout(k))
    if not held:
        return None
    try:
        return freeze_manifest_sha256(freeze_manifest)
    except HeldOutKeyRefused as e:
        raise HeldOutKeyRefused(f"{e}; refused key(s): {held}") from None


# ------------------------------------------------------------------------------ flattening

def flatten(record, prefix=""):
    """dict -> a/b, list, tuple or array -> key/i, bool -> int; int, float and str kept; None
    dropped; anything else raises. Keys are single strings; a leading '_' (W&B internal) or a
    duplicate after flattening raises."""
    if not isinstance(record, dict):
        raise WandbLogError("a record is a dict")
    out = {}

    def put(key, v):
        if v is None:
            return
        if isinstance(v, dict):
            for k, x in v.items():
                if not isinstance(k, str) or not k:
                    raise WandbLogError(f"{key!r}: key {k!r} is not a non-empty str")
                put(f"{key}/{k}" if key else k, x)
            return
        if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
            v = v.tolist()                      # numpy arrays and scalars, without importing numpy
        if isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                put(f"{key}/{i}", x)
            return
        if key.startswith("_"):
            raise WandbLogError(f"{key!r}: keys starting with '_' are W&B's own")
        if key in out:
            raise WandbLogError(f"{key!r} occurs twice after flattening")
        if isinstance(v, bool):
            out[key] = int(v)
        elif isinstance(v, (int, float, str)):
            out[key] = v
        else:
            raise WandbLogError(f"{key!r}: value type {type(v).__name__} is not loggable")

    put(prefix.rstrip("/"), record)
    return out


# ------------------------------------------------------------- the section-12 key catalogue

class KeySpec(NamedTuple):
    pattern: str        # '/'-separated; '*' matches inside one segment; extra trailing segments
                        # in the key (array indices, nested leaves) are allowed
    row: str            # 'epoch', 'checkpoint' (keys under 'ckpt/'), 'config' (run config)
    when: tuple         # traits that must all hold for the key to be required
    study: str          # STUDY v0.8 line and item


def _k(row, when, study, *patterns):
    return tuple(KeySpec(p, row, tuple(when), study) for p in patterns)


_ST = "STUDY:"
KEYS = (
    *_k("epoch", (), _ST + "1286 epoch and step", "epoch", "step"),
    *_k("epoch", (), _ST + "1287 train and validation loss and top-1",
        "log_loss", "log_accuracy", "log_val_loss", "log_val_accuracy",
        "val_correct", "val_n", "val_top1"),
    *_k("epoch", (), _ST + "1288 learning rate", "lr"),
    *_k("epoch", ("cost",), _ST + "1288 lambda", "beta"),
    *_k("epoch", ("cost",), _ST + "1289 training count, frozen H1, H2, NR",
        "train_count", "H1", "H2", "NR"),
    *_k("epoch", ("cost",), _ST + "1289 per-layer and per-group subtotals",
        *(f"subtotals/{lv}/*/{c}" for lv in ("layer", "group") for c in ("H1", "H2", "NR", "native"))),
    *_k("epoch", ("quantized",), _ST + "1290 widths and steps",
        "quantizers/*/*/width_mean", "quantizers/*/*/width_max",
        "quantizers/*/*/step_min", "quantizers/*/*/step_max"),
    *_k("epoch", ("binary",), _ST + "1290 binary sites: q_g b and i, q_g(g) or q_g(g_c), zero "
        "columns by route, saturated columns, sign occupancy, sign-flip rate",
        "binary_sites/*/qg_b", "binary_sites/*/qg_i", "binary_sites/*/qg_value",
        "binary_sites/*/zero_cols/pruned", "binary_sites/*/zero_cols/rounded_to_zero",
        "binary_sites/*/saturated_cols", "binary_sites/*/sign_occupancy",
        "binary_sites/*/sign_flip_rate"),
    *_k("epoch", (), _ST + "1290 BN scales and moments",
        "bn/*/gamma_min", "bn/*/gamma_max", "bn/*/moving_mean_absmax",
        "bn/*/moving_var_min", "bn/*/moving_var_max"),
    *_k("epoch", ("binary",), _ST + "1290 moment mismatch between x.W and x.K",
        "binary_sites/*/moment_mismatch_mean", "binary_sites/*/moment_mismatch_var"),
    *_k("epoch", ("quantized",), _ST + "1291 structure: zero kernels by route, all-zero-width "
        "input or output quantizers",
        "quantizer_structure/*/kernel_zero_pruned", "quantizer_structure/*/kernel_zero_rounded",
        "quantizer_structure/*/input_all_zero_width", "quantizer_structure/*/output_all_zero_width"),
    *_k("epoch", ("quantized", "attention"), _ST + "1291 structure: per-head zero-width fraction of "
        "the softmax output quantizer", "quantizer_structure/softmax_heads/*/zero_width_frac"),
    *_k("epoch", ("quantized",), _ST + "1292 activation ranges and zero-width fractions",
        "activations/*/min", "activations/*/max", "activations/*/zero_width_frac"),
    *_k("epoch", ("quantized",), _ST + "1293 pruning channel: zero-width kernel fraction and the "
        "native cost it removes", "pruning_channel/*/zero_width_frac",
        "pruning_channel/*/native_share_removed"),
    *_k("epoch", ("binary",), _ST + "1293 pruning channel: zero-width inputs at binary sites",
        "pruning_channel/*/input_zero_width_frac"),
    *_k("epoch", ("fb",), _ST + "1295 FB signs differing, batch against moving moments",
        "binary_sites/*/fb_sign_diff_frac"),
    *_k("epoch", ("fused_bn_binary",), _ST + "1296-1298 single-sign columns, dead channels, "
        "train-mode against frozen top-1 gap", "binary_sites/*/single_sign_cols",
        "binary_sites/*/dead_channel_frac", "train_frozen_top1_gap"),
    *_k("epoch", ("cost",), _ST + "1299 epochs spent per grid interval", "dwell/*/*"),
    *_k("epoch", ("pid",), _ST + "1301 PID: sensed H, stock aggregate, error, integral, clamp "
        "state, slack", *(f"pid/{x}" for x in ("sensed_H", "stock_aggregate", "error",
                                              "integral", "clamp_state", "slack"))),
    *_k("epoch", (), _ST + "1301 per-epoch fetched batch-index and _index_map hashes",
        "batch_index_seq_sha256", "index_map_sha256"),
    *_k("epoch", (), _ST + "1301 time, memory and allocation hours",
        "sys/epoch_seconds", "sys/host_rss_mb", "sys/alloc_hours"),
    *_k("checkpoint", (), _ST + "1287 validation macro and per-class AUC for saved candidates",
        "ckpt/ckpt_sha256", "ckpt/val_auc_macro", *(f"ckpt/val_auc_per_class/{i}" for i in range(5))),
    *_k("checkpoint", (), _ST + "1291 zero-variance test in every frozen checkpoint's sidecar",
        "ckpt/structure_record/*"),
    *_k("checkpoint", ("quantized", "attention"), _ST + "1303 Q/K/V/O and QK/AV cost and widths, "
        "softmax zero-width fractions, degenerate Q/K/V/O by route",
        *(f"ckpt/attention_diagnostics/{p}/{x}" for p in ("q", "k", "v", "o", "qk", "av")
          for x in ("cost", "width_mean")),
        "ckpt/attention_diagnostics/softmax_zero_width/*", "ckpt/attention_diagnostics/degenerate/*/*"),
    *_k("checkpoint", ("attention",), _ST + "1303 H_attn and all-zero rows, branch norm, delta_attn "
        "by mean and zero ablation, delta_static, delta_unif",
        *(f"ckpt/attention_diagnostics/{x}" for x in ("H_attn", "all_zero_rows", "branch_norm",
                                                      "delta_attn_mean", "delta_attn_zero",
                                                      "delta_static", "delta_unif"))),
    *_k("config", (), _ST + "1283 run identity (name carries arm, form, seed, stage, ramp or PID, "
        "attempt, source bundle)", *(f"identity/{x}" for x in ("run_name", "run_id", "arm_id",
                                                              "form", "seed", "stage", "schedule",
                                                              "attempt", "source_bundle"))),
    *_k("config", (), _ST + "1301 provenance, split and order hashes",
        "provenance/code_sha", "provenance/config_sha256", "provenance/data_manifest_sha256",
        "provenance/split_sha256", "provenance/order_sha256"),
    *_k("config", (), _ST + "1301 backend and matmul precision, GPU product and K",
        "provenance/backend", "provenance/matmul_precision", "provenance/gpu_product",
        "provenance/pack_k"),
)
TRAITS = ("cost", "quantized", "attention", "binary", "fused_bn_binary", "fb", "pid")

# [B18P] M8/M9/M10/M13/M20 now supply 26 of B10's 27 missing patterns.
# Keep strict="produced" until the one counterfactual below has a STUDY definition.
# Every other applicable pattern is required, with B10's two row exemptions.
NOT_PRODUCED = {
    "pruning_channel/*/native_share_removed":
        "M10: STUDY 12 specifies no unpruned widths/reference checkpoint for the native-cost counterfactual; "
        "zero-width coefficients have zero current cost. Requires an owner/STUDY definition.",
}

# [B10] A key the catalogue requires that a row may lack for a stated reason (strict="produced" only): the PID's
# error exists once the controller acts (B9-5; absent while clamp_state is "warmup"), and a sign-flip rate needs a
# previous evaluation, so the run's epoch 0 has none (M14 persists the signs across a resume).
ROW_EXEMPT = (
    ("pid/error", "PID warmup (B9-5)", lambda row: row.get("pid/clamp_state") == "warmup"),
    ("binary_sites/*/sign_flip_rate", "the run's first evaluation", lambda row: row.get("epoch") == 0),
)
STRICT_LEVELS = (False, True, "produced")
PROVENANCE_REQUIRED = tuple(s.pattern.split("/", 1)[1] for s in KEYS
                            if s.row == "config" and s.pattern.startswith("provenance/"))


def traits(cfg):
    """Which conditional section-12 items apply to a run, from its config (interfaces.md s. 5)."""
    t = set()
    control = cfg.get("control")
    if cfg["cost_callbacks"]:
        t.add("cost")
    if control != "FP32":
        t.add("quantized")
    if control != "DS-HGQ":
        t.add("attention")
    if "B" in (cfg.get("groups") or {}).values() or cfg.get("embedding") == "B":
        t.add("binary")
    if cfg.get("form") in FORMS:                 # a set form means a fused-BN binary site exists
        t.add("fused_bn_binary")
    if cfg.get("form") == "FB":
        t.add("fb")
    if cfg["schedule"] == "pid":
        t.add("pid")
    return frozenset(t)


def matches(key, pattern):
    ks, ps = key.split("/"), pattern.split("/")
    return len(ks) >= len(ps) and all(fnmatchcase(k, p) for k, p in zip(ks, ps))


def required(row, run_traits):
    return [s for s in KEYS if s.row == row and set(s.when) <= set(run_traits)]


def missing(keys, row, run_traits):
    keys = list(keys)
    return [s.pattern for s in required(row, run_traits) if not any(matches(k, s.pattern) for k in keys)]


def missing_produced(row, kind, run_traits):
    """[B10] strict="produced": the catalogue patterns a row lacks, less NOT_PRODUCED and the ROW_EXEMPT keys whose
    condition holds for the row (a flattened dict)."""
    gaps = [p for p in missing(row, kind, run_traits) if p not in NOT_PRODUCED]
    for pattern, _why, cond in ROW_EXEMPT:
        if pattern in gaps and cond(row):
            gaps.remove(pattern)
    return gaps


def host_peak_rss_mb():
    """Peak resident memory of this process in MiB (ru_maxrss is bytes on macOS, KiB on Linux)."""
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == "darwin" else r / 2 ** 10


# ------------------------------------------------------------------------------------- run

class Run:
    """One W&B run of one arm. Build with `Run.open`; rows go through the held-out refusal and
    the completeness check before the SDK sees them."""

    def __init__(self, sdk_run, *, name, rid, mode, run_traits, strict, freeze_manifest):
        if not (strict is False or strict is True or strict == "produced"):
            raise WandbLogError(f"strict must be one of {STRICT_LEVELS}, not {strict!r}")
        self.sdk_run, self.name, self.run_id, self.mode = sdk_run, name, rid, mode
        self.traits, self.strict, self.freeze_manifest = run_traits, strict, freeze_manifest
        self.freeze_sha256 = None

    @classmethod
    def open(cls, cfg, provenance, *, source_bundle, run_dir, mode=None, freeze_manifest=None,
             strict=True, expect_run_id=None, env=None):
        env = os.environ if env is None else env
        if not (strict is False or strict is True or strict == "produced"):
            raise WandbLogError(f"strict must be one of {STRICT_LEVELS}, not {strict!r}")
        absent = [f for f in CONFIG_REQUIRED if f not in cfg]
        if absent:
            raise IncompleteRecord(f"config lacks {absent}")
        wcfg = cfg.get("wandb") or {}
        mode = mode or wcfg.get("mode") or "offline"
        if mode not in MODES:
            raise WandbLogError(f"mode {mode!r} is not one of {MODES}")
        if wcfg.get("mode") not in (None, mode):
            raise WandbLogError(f"mode {mode!r} differs from the config's {wcfg.get('mode')!r}")
        env_mode = (env.get("WANDB_MODE") or "").strip().lower()
        if env_mode and env_mode != mode:
            raise WandbLogError(f"WANDB_MODE={env_mode!r} differs from the requested {mode!r}")
        for var, want in ENV_MUST_MATCH:
            got = env.get(var)
            if got and got != want:
                raise WandbLogError(f"{var}={got!r} differs from {want!r} (STUDY section 12)")
        for field, want in (("project", PROJECT), ("group", GROUP)):
            if wcfg.get(field) not in (None, want):
                raise WandbLogError(f"config wandb.{field}={wcfg.get(field)!r} differs from {want!r}")
        name = name_from_config(cfg, source_bundle)
        stem = name.rsplit("_src-", 1)[0]
        if wcfg.get("run_name") not in (None, stem, name):
            raise WandbLogError(f"config wandb.run_name={wcfg.get('run_name')!r} is neither "
                                f"{stem!r} nor {name!r}")
        rid = run_id(name)
        if expect_run_id is not None and expect_run_id != rid:
            raise WandbLogError(f"persisted run id {expect_run_id!r} differs from {rid!r} of {name}")
        prov = dict(provenance)
        if None not in (cfg.get("code_sha"), prov.get("code_sha")) and prov["code_sha"] != cfg["code_sha"]:
            raise WandbLogError("provenance code_sha differs from the config's code_sha")
        run_traits = traits(cfg)
        identity = {"run_name": name, "run_id": rid, "arm_id": cfg["arm_id"],
                    "form": cfg.get("form") or NO_FORM, "seed": cfg["seed"], "stage": cfg["stage"],
                    "schedule": cfg["schedule"], "attempt": cfg["attempt"],
                    "replica": cfg.get("replica"), "source_bundle": source_bundle}
        config = {"identity": identity, "provenance": prov, "config": cfg}
        flat_config = flatten(config)
        check_keys(flat_config, freeze_manifest)
        gaps = missing(flat_config, "config", run_traits)
        if gaps:
            raise IncompleteRecord(f"run config of {name} lacks {gaps}")

        import wandb
        if wandb.run is not None:
            if wandb.run.id != rid:
                raise WandbLogError(f"another W&B run ({wandb.run.id}) is open in this process")
            return cls(wandb.run, name=name, rid=rid, mode=mode, run_traits=run_traits,
                       strict=strict, freeze_manifest=freeze_manifest)
        stage = cfg["stage"]
        kw = dict(project=PROJECT, entity=ENTITY, group=GROUP,
                  job_type="train" if stage in CAMPAIGN_STAGES else "engineering", name=name,
                  id=rid, mode=mode, config=config, settings=wandb.Settings(**SETTINGS),
                  tags=["chang-bitlinear", cfg["arm_id"], identity["form"], f"s{cfg['seed']}",
                        stage, cfg["schedule"], f"a{cfg['attempt']}"]
                  + ([f"r{cfg['replica']}"] if cfg.get("replica") is not None else []))
        if mode == "online":
            kw["resume"] = "allow"              # offline runs ignore resume (module docstring)
        if mode != "disabled":
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            kw["dir"] = str(run_dir)
        sdk_run = wandb.init(**kw)
        if mode != "disabled":
            sdk_run.define_metric("epoch")
            sdk_run.define_metric("*", step_metric="epoch")
        return cls(sdk_run, name=name, rid=rid, mode=mode, run_traits=run_traits, strict=strict,
                   freeze_manifest=freeze_manifest)

    def set_freeze_manifest(self, path):
        self.freeze_manifest = path

    def _allow(self, row):
        return check_keys(row, self.freeze_manifest)

    def _record_freeze(self, sha):
        """After an allowed held-out row reached the SDK: the manifest that allowed it."""
        if sha is not None and sha != self.freeze_sha256:
            self.freeze_sha256 = sha
            self.sdk_run.summary["freeze/manifest_sha256"] = sha

    def _complete(self, row, kind):
        if self.strict == "produced":
            gaps = missing_produced(row, kind, self.traits)
            if gaps:
                raise IncompleteRecord(f"{kind} row of {self.name} lacks {gaps} (strict: produced keys)")
            return gaps
        gaps = missing(row, kind, self.traits)
        if gaps and self.strict:
            raise IncompleteRecord(f"{kind} row of {self.name} lacks {gaps}")
        return gaps

    def log_epoch(self, record):
        """One history row from the M8 epoch record plus the runner's `sys/*` readings."""
        row = flatten(record)
        if not isinstance(row.get("epoch"), int):
            raise WandbLogError("an epoch row needs an int 'epoch'")
        sha = self._allow(row)
        self._complete(row, "epoch")
        self.sdk_run.log(row)
        self._record_freeze(sha)
        return row

    def log_checkpoint(self, sidecar):
        """One history row from a saved checkpoint's sidecar, its keys under 'ckpt/'."""
        row = flatten(sidecar, "ckpt")
        if not isinstance(row.get("ckpt/epoch"), int):
            raise WandbLogError("a checkpoint row needs an int 'epoch'")
        row["epoch"] = row["ckpt/epoch"]
        sha = self._allow(row)
        self._complete(row, "checkpoint")
        self.sdk_run.log(row)
        self._record_freeze(sha)
        return row

    def summary(self, values):
        row = flatten(values)
        sha = self._allow(row)
        for k, v in row.items():
            self.sdk_run.summary[k] = v
        self._record_freeze(sha)
        return row

    def finish(self, exit_code=0):
        self.sdk_run.finish(exit_code=int(exit_code))
