"""Child process of gates G58 and G60 (PREFLIGHT-PLAN section 3 rows G58 and G60; STUDY v0.10 section 13
"Further required gates" and arms row; modules M17 gen/, M5 cbl/model.py, M6 cbl/binary.py, M13
cbl/runner.py). Gate code, not campaign code.

One invocation is one fresh process on the CPU. It reads launch copies of generated configs that the
parent wrote (the parent decides which configs are admitted; this child never edits a config) and writes
one JSON record.

    build  G58: for every config in --list, the model built as the runner builds it (Chang's
           get_model for the released builder, cbl.model.arm_spec and build for the lab builder, the
           config's headline as the binary penalty branch), its parameter counts (Keras count_params,
           trainable and non-trainable elements), its roles (cbl.rng.roles_of under the arm's RoleSpec),
           a .keras save and keras.models.load_model, and the reloaded model compared with the built one:
           classes, configurations, every variable by role byte for byte, count_params, and the
           inference logits on a constructed batch byte for byte
    prep   G60: cbl.runner.prepare (data, seeds, Datasets, model, the seed's CH-8 initial-tensor file
           loaded by role, callbacks, compile; no fit) for the one config in --config, on the smoke
           archive; records the load record, the callback order, every role's tensor sha256 after the
           load, the values of the unmatched roles, the class sequence

Constructed tensors only in `build` (a private NumPy generator); `prep` reads the smoke archive through
the runner. Every reading is engineering, not quotable.

    <venv>/bin/python -I -B gates/config_child.py build --list LIST.json --work DIR --out JSON
    <venv>/bin/python -I -B gates/config_child.py prep --config CFG.json --archive SMOKE --out-dir DIR \\
        --source-bundle HEX --out JSON
"""

import argparse
import json
import random
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))

PRIVATE_SEED = 20261058
PROBE_N = 64


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n', encoding='utf-8')


def setup():
    from cbl import env

    st = env.setup(mem_fraction='0.5', jax_platforms='cpu')
    import jax
    return {'jax_backend': jax.default_backend(), 'devices': [str(d) for d in jax.devices()],
            'matmul_precision': str(jax.config.jax_default_matmul_precision), 'setup_precision': st.get('matmul_precision')}


def ser(obj):
    """A JSON-ready, address-free form of a configuration (as gates/runner_child.py)."""
    import re

    if isinstance(obj, dict):
        return {str(k): ser(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [ser(x) for x in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, 'get_config'):
        try:
            return {'class': type(obj).__name__, 'config': ser(obj.get_config())}
        except NotImplementedError:
            pass
    return re.sub(r' at 0x[0-9a-f]+', '', repr(obj))


def layer_configs(model):
    out = []
    for ly in model._flatten_layers(include_self=False):
        try:
            c = ly.get_config()
            c.pop('name', None)
            out.append({'class': type(ly).__name__, 'config': ser(c)})
        except NotImplementedError:
            out.append({'class': type(ly).__name__, 'config': None})
    return out


def build_model(cfg):
    """The model of a config, as cbl/runner.py prepare builds it (runner.py, step 2), and its RoleSpec."""
    import keras
    import model as chang_model  # noqa: E402  Chang's module, imported by cbl.env.setup()

    from cbl import model as lab_model
    from cbl import rng

    random.seed(cfg['seed'])
    keras.utils.set_random_seed(cfg['seed'])
    if cfg['builder'] == 'released':
        return chang_model.get_model(*rng.CH8_BUILDER_ARGS), rng.CH8_ROLE_SPEC
    spec = lab_model.arm_spec(cfg['arm_id'], groups=cfg['groups'], embedding=cfg['embedding'], form=cfg['form'],
                              control=cfg['control'], headline=cfg.get('headline'))
    return lab_model.build(spec), lab_model.role_spec(spec)


def config_diffs(a, b, path=''):
    """Paths at which two serialized configurations differ, with both values (repr, truncated)."""
    out = []
    if type(a) is not type(b):
        return [{'path': path, 'built': repr(a)[:120], 'reloaded': repr(b)[:120]}]
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append({'path': f'{path}/{k}', 'built': repr(a.get(k))[:120], 'reloaded': repr(b.get(k))[:120]})
            else:
                out += config_diffs(a[k], b[k], f'{path}/{k}')
    elif isinstance(a, list):
        if len(a) != len(b):
            return [{'path': path, 'built': f'len {len(a)}', 'reloaded': f'len {len(b)}'}]
        for i, (x, y) in enumerate(zip(a, b)):
            out += config_diffs(x, y, f'{path}[{i}]')
    elif a != b:
        out.append({'path': path, 'built': repr(a)[:120], 'reloaded': repr(b)[:120]})
    return out


def layer_config_diffs(m1, m2):
    """Per flattened layer whose configuration differs: index, class and the differing paths."""
    a, b = layer_configs(m1), layer_configs(m2)
    if len(a) != len(b):
        return [{'index': None, 'class': None, 'diffs': [{'path': 'n_layers', 'built': len(a), 'reloaded': len(b)}]}]
    return [{'index': i, 'class': x['class'], 'diffs': config_diffs(x, y)} for i, (x, y) in enumerate(zip(a, b)) if x != y]


def n_elements(variables):
    import math
    return int(sum(math.prod(int(d) for d in v.shape) for v in variables))


def cmd_build(a, rec):
    import keras
    import numpy as np

    from cbl import binary  # noqa: F401  registers the cbl1016 classes for load_model
    from cbl import rng

    items = json.loads(Path(a.list).read_text(encoding='utf-8'))
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    x = np.random.default_rng(PRIVATE_SEED).standard_normal((PROBE_N, 8, 3)).astype(np.float32)
    rows = []
    for it in items:
        cfg = it['config']
        row = {'path': it['path'], 'model_key': it['model_key'], 'label': cfg['label'], 'stage': cfg['stage'],
               'seed': cfg['seed']}
        try:
            model, spec = build_model(cfg)
            roles = rng.roles_of(model, spec)
            row.update(classes=[type(ly).__name__ for ly in model.layers], n_variables=len(model.variables),
                       n_roles=len(roles), count_params=int(model.count_params()),
                       trainable_elements=n_elements(model.trainable_variables),
                       non_trainable_elements=n_elements(model.non_trainable_variables))
            arrays = rng.variable_arrays(roles)
            y0 = np.asarray(keras.ops.convert_to_numpy(model(x, training=False)), dtype=np.float32)
            after_call = rng.variable_arrays(roles)
            row['inference_changed_nothing'] = all(arrays[r].tobytes() == after_call[r].tobytes() for r in roles)
            path = work / f'{cfg["label"]}-s{cfg["seed"]}-{cfg["stage"]}{"-r%d" % cfg["replica"] if cfg.get("replica") else ""}.keras'
            model.save(path)
            row['saved'] = str(path.name)
            row['saved_sha256'] = rng.sha256_file(path)
            m2 = keras.models.load_model(path)
            roles2 = rng.roles_of(m2, spec)
            arrays2 = rng.variable_arrays(roles2)
            diff = sorted(r for r in roles if r not in roles2 or arrays[r].dtype != arrays2[r].dtype
                          or arrays[r].shape != arrays2[r].shape or arrays[r].tobytes() != arrays2[r].tobytes())
            y1 = np.asarray(keras.ops.convert_to_numpy(m2(x, training=False)), dtype=np.float32)
            row.update(reload_classes=[type(ly).__name__ for ly in m2.layers], reload_count_params=int(m2.count_params()),
                       reload_roles_equal=set(roles) == set(roles2), reload_differing_roles=diff,
                       reload_config_diffs=layer_config_diffs(model, m2),
                       logits_sha256=rng.tensor_sha256(y0), reload_logits_sha256=rng.tensor_sha256(y1),
                       logits_equal=y0.tobytes() == y1.tobytes(), logits_finite=bool(np.all(np.isfinite(y0))))
        except Exception as e:  # noqa: BLE001 - recorded per config; the parent fails the gate
            row['error'] = f'{type(e).__name__}: {e}'
            row['traceback'] = traceback.format_exc()[-3000:]
        rows.append(row)
    rec['rows'] = rows
    return 0


def cmd_prep(a, rec):
    from cbl import runner

    cfg = json.loads(Path(a.config).read_text(encoding='utf-8'))
    ctx = runner.Context()
    try:
        runner.prepare(cfg, ctx=ctx, train_archive=a.archive, archive_kind='smoke', device='cpu', out_dir=a.out_dir,
                       source_bundle=a.source_bundle, keep_full=None)
    except runner.ConfigRefused as e:
        rec['refused'] = str(e)
        return 2
    import jax
    import keras  # noqa: F401  after prepare: cbl.env.setup() has set the backend
    import numpy as np

    from cbl import rng

    r = ctx.record
    roles = ctx.roles
    arrays = rng.variable_arrays(roles)
    load = r['load']
    rec.update(label=cfg['label'], stage=cfg['stage'], seed=cfg['seed'], load=load, callback_order=r['callback_order'],
               classes=[type(ly).__name__ for ly in ctx.model.layers],
               roles_sha256={k: rng.tensor_sha256(v) for k, v in arrays.items()},
               unmatched_values={k: np.asarray(arrays[k]).tolist() for k in load['unmatched']},
               unmatched_dtypes={k: str(arrays[k].dtype) for k in load['unmatched']},
               n_roles=len(roles), summary=r['summary'], config_sha256=r['config_sha256'], code=r['code'],
               dataset_device=r['dataset_device'],
               setup={'jax_backend': jax.default_backend(), 'devices': [str(d) for d in jax.devices()],
                      'matmul_precision': str(jax.config.jax_default_matmul_precision)})
    if ctx.wandb_run is not None:
        ctx.wandb_run.finish(exit_code=0)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('cmd', choices=['build', 'prep'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--list')
    ap.add_argument('--work')
    ap.add_argument('--config')
    ap.add_argument('--archive')
    ap.add_argument('--out-dir')
    ap.add_argument('--source-bundle')
    a = ap.parse_args()
    rec = {'cmd': a.cmd, 'args': vars(a)}
    code = 0
    try:
        if a.cmd == 'build':
            rec['setup'] = setup()          # prep: cbl.runner.prepare runs cbl.env.setup() itself
        code = {'build': cmd_build, 'prep': cmd_prep}[a.cmd](a, rec) or 0
    except Exception as e:  # noqa: BLE001 - recorded for the parent, then exit 1
        rec['error'] = f'{type(e).__name__}: {e}'
        rec['traceback'] = traceback.format_exc()
        code = 1
    rec['exit'] = code
    dump(a.out, rec)
    return code


if __name__ == '__main__':
    sys.exit(main())
