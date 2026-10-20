"""Opt-in only. Does not run in the offline set; never stages data or contacts a cluster."""
import os
import json
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.pc_gpu
@pytest.mark.skipif(os.environ.get('CBL_B17_PC_RUN') != '1', reason='PC owner opt-in after B18 required')
def test_real_pack_smoke_on_pc():
    assert sys.platform == 'linux', 'real runner tests belong on the home PC'
    plan = json.loads(Path(os.environ['CBL_B17_PC_PLAN']).read_text())
    assert plan['mode'] == 'smoke' and all(a['config']['epochs'] == 3 for a in plan['arms'])
    python = os.environ['CBL_B17_PC_PYTHON']
    env = dict(os.environ, CBL_PLAN=json.dumps(plan), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([python, '-B', 'nrp/run_pack.py'], env=env, check=False)
    assert result.returncode == 0
    for arm in plan['arms']:
        record = json.loads((Path(arm['out']) / 'run.json').read_text())
        assert record['exit'] == 0 and record['wandb']['provenance']['pack_k'] == plan['k']
