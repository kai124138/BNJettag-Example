"""Child process of gate G49 (PREFLIGHT-PLAN section 3 row G49; STUDY v0.10 section 13 "metric reload agreement within
1e-7 with TF32 off"; docs/methodology/03-phases.md phase 2 "every arm's checkpoint reloads within 1e-7 with TF32
off"; modules M8 cbl/frozen_eval.py and M13 cbl/runner.py). Gate code, not campaign code. JAX on the CPU.

    draw    CH-8's initial-tensor file of a seed on the training archive of record (cbl.runner.draw, archive kind
            `record`, decision U1)
    train   the runner (prepare, fit, finish) for one gate copy of a generated config, one epoch on the training
            archive of record; a read-only probe after epoch 0's frozen evaluation saves the model as a .keras file
            and the validation logits over the full validation set (the evaluator's own jitted inference call on the
            validation Dataset's tensors, fixed 2790-row chunks), and keeps the epoch's Keras logs
    reload  a fresh process: the data and the two Datasets as the runner builds them (section 6 steps 1-3, seed of
            the config), the .keras file loaded with keras.models.load_model, then (1) M8's frozen evaluation of the
            loaded model (cbl.frozen_eval.checkpoint_audit's comparison against epoch 0's record), (2) the validation
            logits by the same inference call, (3) Keras's evaluate on the validation Dataset (loss and accuracy),
            and the matmul precision in force

Every reading is engineering, not quotable.
"""

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n', encoding='utf-8')


def logits_of(evaluator):
    """The validation logits by the evaluator's own jitted stateless inference call, chunk by chunk."""
    import keras
    import numpy as np

    m = evaluator.model
    tv = [v.value for v in m.trainable_variables]
    ntv = [v.value for v in m.non_trainable_variables]
    out = []
    for lo, hi in evaluator.bounds:
        y, _ = evaluator._infer(tv, ntv, evaluator.x_val[lo:hi])
        out.append(np.asarray(keras.ops.convert_to_numpy(y), dtype=np.float32))
    return np.concatenate(out, axis=0)


def cmd_draw(a, rec):
    from cbl import runner

    mpath, msha, man, _ = runner.draw(a.seed, train_archive=a.archive, archive_kind='record', device='cpu',
                                      out_dir=a.out_dir, place='PCC gate G49 (archive of record)')
    rec.update(manifest=str(mpath), manifest_sha256=msha, identity=man['identity'], file=man['file'])
    return 0


def cmd_train(a, rec):
    import numpy as np

    from cbl import runner

    cfg = json.loads(Path(a.config).read_text())
    H = {}

    def factory(ctx):
        def probe(epoch, logs):
            if epoch != 0:
                return
            import jax
            ctx.model.save(a.save)
            lg = logits_of(ctx.evaluator)
            np.save(a.logits, lg)
            H.update(logs={k: (v if isinstance(v, str) else float(v)) for k, v in logs.items()},
                     logits_sha256=hashlib.sha256(lg.tobytes()).hexdigest(), logits_shape=list(lg.shape),
                     matmul_precision=str(jax.config.jax_default_matmul_precision))
        return probe

    code, ctx = runner.run(cfg, train_archive=a.archive, archive_kind='record', device='cpu', out_dir=a.out_dir,
                           source_bundle=a.source_bundle, evaluator_factory=factory)
    r = ctx.record
    ep = Path(a.out_dir) / 'epochs.jsonl'
    rows = [json.loads(x) for x in ep.read_text().splitlines() if x] if ep.exists() else []
    rec['runner'] = {k: r.get(k) for k in ('end', 'exit', 'refused', 'error', 'traceback', 'epochs_completed', 'launch_check',
                                           'matmul_precision_before_fit', 'matmul_precision_after_fit', 'config_sha256')}
    rec['row0'] = rows[0] if rows else None
    rec['probe'] = H
    rec['ckpt_sha256'] = hashlib.sha256(Path(a.save).read_bytes()).hexdigest() if Path(a.save).exists() else None
    return code


def cmd_reload(a, rec):
    from cbl import env

    env.setup(mem_fraction='0.5', jax_platforms='cpu')
    import dataclasses

    import jax
    import keras
    import numpy as np

    from cbl import binary  # noqa: F401  registers the cbl1016 classes
    from cbl import frozen_eval, rng
    from cbl import model as lab
    from cbl import runner

    cfg = json.loads(Path(a.config).read_text())
    t = json.loads(Path(a.train_record).read_text())
    man = json.loads((FORK / 'manifests' / 'data-prep-150c.json').read_text())
    flow = rng.SeedFlow(cfg['seed'], dataset_device='cpu:0')
    flow.load_data(a.archive, expect_sha256=man['train']['archive']['sha256'], data_manifest_sha256=cfg['data_manifest_sha256'])
    flow.seed_streams()
    flow.build_datasets(batch_size=runner.TRAIN_BATCH, val_batch_size=runner.VAL_BATCH)
    if cfg['builder'] == 'released':
        spec = dataclasses.replace(rng.CH8_ROLE_SPEC, arm_id=cfg['arm_id'])
    else:
        spec = lab.role_spec(lab.arm_spec(cfg['arm_id'], groups=cfg['groups'], embedding=cfg['embedding'], form=cfg['form'],
                                          control=cfg['control'], headline=cfg.get('headline')))
    model = keras.models.load_model(a.ckpt)
    roles = rng.roles_of(model, spec)
    ev = frozen_eval.FrozenEvaluator(model, roles=roles, top_names=spec.top_names(model), x_val=flow.ds_val.x,
                                     y_val=flow.ds_val.y, batch_size=runner.VAL_BATCH)
    core = ev.evaluate()
    rec['audit_mismatches'] = frozen_eval.audit_mismatches(t['row0'], core)
    rec['core'] = {k: core[k] for k in ('val_correct', 'val_n', 'val_top1', 'H1', 'H2', 'NR', 'vars_sha256_after')}
    lg = logits_of(ev)
    ref = np.load(a.logits)
    rec['logits'] = {'sha256': hashlib.sha256(lg.tobytes()).hexdigest(), 'shape': list(lg.shape),
                     'max_abs_diff': float(np.max(np.abs(lg.astype(np.float64) - ref.astype(np.float64)))) if lg.shape == ref.shape else None,
                     'equal': bool(lg.shape == ref.shape and np.array_equal(lg, ref))}
    ev_res = model.evaluate(flow.ds_val, verbose=0, return_dict=True)
    rec['evaluate'] = {k: float(v) for k, v in ev_res.items()}
    rec['matmul_precision'] = str(jax.config.jax_default_matmul_precision)
    rec['ckpt_sha256'] = hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('cmd', choices=['draw', 'train', 'reload'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--out-dir')
    ap.add_argument('--config')
    ap.add_argument('--source-bundle')
    ap.add_argument('--save')
    ap.add_argument('--logits')
    ap.add_argument('--ckpt')
    ap.add_argument('--train-record')
    a = ap.parse_args()
    rec = {'cmd': a.cmd, 'args': vars(a)}
    code = 0
    try:
        code = {'draw': cmd_draw, 'train': cmd_train, 'reload': cmd_reload}[a.cmd](a, rec) or 0
    except Exception as e:  # noqa: BLE001
        rec['error'] = f'{type(e).__name__}: {e}'
        rec['traceback'] = traceback.format_exc()
        code = 1
    rec['exit'] = code
    dump(a.out, rec)
    return code


if __name__ == '__main__':
    sys.exit(main())
