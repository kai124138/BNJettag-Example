"""Synthetic decision inputs: never evidence of a scientific decision or GPU gate."""
import json
from pathlib import Path
import sys
import pytest

FORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORK))
from nrp.common import PRODUCTS, init_root


@pytest.fixture
def generated(tmp_path):
    from gen import inputs, configs
    rec = json.loads((FORK / 'gen/inputs.json').read_text())
    rec.update(headline='H1', claim_domain={'base': 'a', 'c3r': False}, bench_epochs=20,
               init_tensors={'root': init_root(PRODUCTS[0], 'record'), 'platform': 'gpu'},
               smoke_init_tensors={'root': init_root(PRODUCTS[0], 'smoke'), 'platform': 'gpu'})
    path = tmp_path / 'synthetic-inputs.json'
    path.write_text(json.dumps(rec))
    inp = inputs.load(path)
    files = configs.generate(inp)
    files['manifests/data-prep-150c.json'] = (FORK / 'manifests/data-prep-150c.json').read_bytes()
    return files


@pytest.fixture
def spec():
    return {'schema': 1, 'pack': 'smoke-k2', 'mode': 'smoke', 'stage': 'smoke', 'job_attempt': 1,
            'gpu_product': PRODUCTS[0], 'archive_kind': 'smoke', 'deterministic_ops': False,
            'mem_fraction': 0.4, 'resume_every': 1, 'excluded_nodes': ['synthetic-excluded'],
            'resources': {'cpu': 4, 'memory_gi': 16, 'deadline_s': 1200, 'ttl_s': 86400, 'cache_gi': 2},
            'single_arm_justification': 'synthetic offline test',
            'approval_ref': 'SYNTHETIC-NOT-APPROVED', 'scientific_gate': 'pending',
            'gate_reference': 'synthetic offline transport test, not scientific clearance',
            'arms': [{'config_path': 'configs/smoke/' + n + '-s1.json', 'init_manifest_sha256': 'a' * 64}
                     for n in ('CH-8', 'F000')]}


@pytest.fixture
def plan(spec, generated):
    from nrp.spec import resolve
    return resolve(spec, generated, 'b' * 40, 'c' * 64)
