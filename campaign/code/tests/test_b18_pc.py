"""Deferred PC tests of real JAX/Keras producer wiring, mutation guards and last-step logging."""
import os
from pathlib import Path
import subprocess
import sys
import pytest

FORK = Path(__file__).resolve().parents[1]


@pytest.mark.pc_gpu
@pytest.mark.parametrize('arm,form', [('CH-8', 'none'), ('F000', 'none'), ('F111', 'FB'), ('F111', 'BF')])
def test_real_epoch_producers_and_stateless_observer(arm, form):
    result = subprocess.run([os.environ.get('CBL_TEST_PYTHON', sys.executable), '-I', '-B',
                             str(FORK / 'tests/b18_pc_case.py'), arm, form],
                            cwd=FORK, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
