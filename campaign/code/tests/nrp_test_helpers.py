"""Constructed M14 checkpoint metadata, never scientific evidence."""
import json
from pathlib import Path
from cbl.resume import KIND, sha256_bytes
from cbl.runner import config_sha256
from nrp.common import write_once


def checkpoint(root, arm, plan, every=None, product=None):
    root = Path(root)
    st = {'kind': KIND, 'schema': 1, 'epochs_done': 0, 'every': every or plan['resume_every'],
          'run': {'run_name': arm['name']}, 'code_sha': plan['code_sha'],
          'config_sha256': config_sha256(arm['config']), 'files': {}, 'links': {},
          'identity': {'device': {'jax_platform': 'gpu', 'n_devices': 1,
                                  'device_kind': [(product or plan['gpu_product']).replace('-', ' ')]}}}
    p = root / 'resume/ckpt-00000/state.json'
    write_once(p, st)
    write_once(root / 'resume/latest.json', {'checkpoint': 'ckpt-00000', 'epochs_done': 0,
                                           'state_sha256': sha256_bytes(p.read_bytes())})
    return p
