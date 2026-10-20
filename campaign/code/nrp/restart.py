"""M20 run-directory launch identity and pre-train restart policy. No backend imports."""
from datetime import datetime, timezone
from pathlib import Path
import uuid

from nrp.common import PRODUCTS, ROOT, need, read, positive

LAUNCH = 'launch.json'


def launch_record(arm, plan):
    return {'schema': 1, 'run_name': arm['name'], 'code_sha': plan['code_sha'],
            'config': arm['config'], 'source_bundle': plan['source_bundle'],
            'gpu_product': plan['gpu_product'], 'resume_every': plan['resume_every']}


def recorded(arm, plan, directory=None):
    """Read the original launch, cross-check checkpoint identity if present; never repair corruption."""
    root = Path(directory or arm['out'])
    if not root.exists():
        return None
    need(root.is_dir() and not root.is_symlink(), 'run root must be a real directory')
    lp = root / LAUNCH
    latest = root / 'resume/latest.json'
    st = None
    if latest.exists():
        from cbl.resume import read_latest, ResumeError
        try:
            st, _ = read_latest(root)
        except ResumeError as e:
            raise ValueError(str(e)) from e
    if lp.exists():
        rec = read(lp)
        expected = launch_record(arm, plan)
        need(all(rec.get(k) == expected[k] for k in ('schema', 'run_name', 'code_sha', 'config', 'source_bundle')),
             'run launch identity differs')
    else:
        # B10 checkpoints predate launch.json; their checked identity is authoritative.
        need(st is not None, 'existing run lacks original launch interval/product and checkpoint; inspect legacy run')
        from cbl.runner import config_sha256
        need(st['run']['run_name'] == arm['name'] and st['code_sha'] == plan['code_sha'] and
             st['config_sha256'] == config_sha256(arm['config']), 'checkpoint launch identity differs')
        dev = st['identity']['device']
        need(dev['jax_platform'] == 'gpu' and dev['n_devices'] == 1 and len(dev['device_kind']) == 1,
             'checkpoint is not a single GPU run')
        rec = {**launch_record(arm, plan), 'gpu_product': dev['device_kind'][0].replace(' ', '-'),
               'resume_every': st['every']}
    need(rec['gpu_product'] in PRODUCTS, 'unsupported recorded product')
    positive(rec['resume_every'], 'recorded resume_every', True)
    if st is not None:
        from cbl.runner import config_sha256
        dev = st['identity']['device']
        need(st['every'] == rec['resume_every'] and dev['jax_platform'] == 'gpu' and dev['n_devices'] == 1 and
             [s.replace(' ', '-') for s in dev['device_kind']] == [rec['gpu_product']] and
             st['run']['run_name'] == arm['name'] and st['code_sha'] == plan['code_sha'] and
             st['config_sha256'] == config_sha256(arm['config']), 'launch/checkpoint identity differs')
    return rec


def bind_plan(plan, *, run_state_root=None):
    """Read mounted PVC or a complete captured run tree before emitting a retry Job.

    run_state_root maps /data/cbl1016 to an operator-supplied local capture. No remote fetch.
    Missing/unavailable retry inventory is a refusal, not a guessed GPU product.
    """
    import copy
    out = copy.deepcopy(plan)
    if plan['mode'] == 'draw':
        return out
    root = Path(run_state_root or ROOT)
    if plan['job_attempt'] > 1:
        need(root.is_dir(), 'relaunch requires mounted PVC or complete run_state_root capture')
    products = set()
    for arm in out['arms']:
        path = root / Path(arm['out']).relative_to(ROOT)
        rec = recorded(arm, plan, path)
        if rec:
            products.add(rec['gpu_product'])
            arm['resume_every'] = rec['resume_every']
    need(len(products) <= 1, 'pack contains different original GPU products')
    if products:
        product = products.pop()
        # U1 and the immutable generated config already bind the product. Never rewrite them.
        need(product == plan['gpu_product'], 'selected GPU differs from run-directory product; regenerate spec')
        out['gpu_product'] = product
    return out


def prepare_run(arm, plan):
    """Under the pack's run lock: validate, archive a pre-train kill, and return original interval."""
    root = Path(arm['out'])
    rec = recorded(arm, plan)
    if rec is None:
        return arm.get('resume_every', plan['resume_every'])
    need(rec['gpu_product'] == plan['gpu_product'], 'GPU product differs from original launch')
    need(not (root / 'run.json').exists(), 'terminal run must be handled before restart')
    if not (root / 'resume/latest.json').exists():
        # A named checkpoint without its pointer is an interrupted checkpoint publication;
        # no committed resume state exists. Preserve every byte before starting afresh.
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        target = root.with_name(root.name + '.pretrain-' + stamp + '-' + uuid.uuid4().hex[:8])
        root.rename(target)
    return rec['resume_every']
