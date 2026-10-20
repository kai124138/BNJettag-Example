"""Tests of M1 `cbl/env.py`. Each scenario that touches jax, keras or hgq runs in its own
subprocess, because their import state is per process. The interpreter is CBL_TEST_PYTHON if set
(the training venv's python), otherwise the one running pytest. JAX runs on the CPU here.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[1]
PY = os.environ.get('CBL_TEST_PYTHON', sys.executable)

PRELUDE = f"""
import json, os, sys, types
sys.dont_write_bytecode = True
sys.path.insert(0, {str(FORK)!r})
def emit(obj):
    print('RESULT ' + json.dumps(obj, default=str), flush=True)
"""


def run(code, extra_env=None, tmp_path=None):
    env = {k: v for k, v in os.environ.items() if k not in ('KERAS_BACKEND', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_MEM_FRACTION')}
    env['JAX_PLATFORMS'] = 'cpu'
    env.update(extra_env or {})
    r = subprocess.run(
        [PY, '-I', '-B', '-c', PRELUDE + textwrap.dedent(code)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path or FORK), timeout=600,
    )
    results = [json.loads(line[len('RESULT '):]) for line in r.stdout.splitlines() if line.startswith('RESULT ')]
    assert r.returncode == 0, f'exit {r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr[-4000:]}'
    assert len(results) == 1, r.stdout
    return results[0]


@pytest.mark.pc_gpu
def test_setup_order_records_and_bindings(tmp_path):
    out = run(
        """
        from cbl import env
        rec = env.setup(mem_fraction='0.5', jax_platforms='cpu')
        import hgq.layers as hl
        import hgq.utils.sugar as hs
        from hgq.utils.sugar import BetaScheduler, Dataset, FreeEBOPs, ParetoFront, PBar, PieceWiseSchedule, StopIf
        from hgq.layers import QLinformerAttentionT
        o = dict(rec['matmul_precision'])
        o['env'] = rec['process_env']['set']
        o['stopif_bound'] = StopIf is hs.StopIf and StopIf.__module__ == env.VENDORED_MODULE_NAME
        o['stopif_file'] = sys.modules[env.VENDORED_MODULE_NAME].__file__
        o['placeholder_bound'] = QLinformerAttentionT is env.QLinformerAttentionT
        try:
            QLinformerAttentionT(2, 4, 16)
            o['placeholder_raises'] = False
        except RuntimeError:
            o['placeholder_raises'] = True
        o['model_file'] = sys.modules['model'].__file__
        o['idempotent'] = env.bind_shims() is rec['shims']
        o['shims'] = rec['shims']
        o['final'] = env.assert_matmul_precision('test end')['jax_default_matmul_precision']
        emit(o)
        """,
        tmp_path=tmp_path,
    )
    assert out['before_import_model'] is None
    assert out['after_import_model'] == 'tensorfloat32'
    assert out['after_override'] == 'float32'
    assert out['final'] == 'float32'
    assert out['env'] == {'KERAS_BACKEND': 'jax', 'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.5', 'JAX_PLATFORMS': 'cpu'}
    assert out['stopif_bound'] and out['placeholder_bound'] and out['placeholder_raises'] and out['idempotent']
    assert Path(out['stopif_file']) == FORK / 'vendor' / 'stop_if.py'
    assert Path(out['model_file']).resolve() == (FORK / 'chang' / 'model.py').resolve()
    assert out['shims']['stopif_vendored_sha256'] == out['shims']['stopif_snapshot_sha256']
    assert out['shims']['bound'] == ['hgq.utils.sugar.StopIf', 'hgq.layers.QLinformerAttentionT']


@pytest.mark.pc_gpu
def test_set_process_env_refuses_after_jax_import(tmp_path):
    out = run(
        """
        import jax
        from cbl import env
        try:
            env.set_process_env(mem_fraction='0.5')
            emit({'raised': None})
        except RuntimeError as e:
            emit({'raised': str(e)})
        """,
        tmp_path=tmp_path,
    )
    assert out['raised'] and 'before jax or keras is imported' in out['raised']


@pytest.mark.parametrize(
    'kwargs',
    [
        {'mem_fraction': '0.5', 'backend': 'tensorflow'},
        {'mem_fraction': '0'},
        {'mem_fraction': '1.5'},
        {'mem_fraction': 'half'},
        {'mem_fraction': '0.5', 'jax_platforms': 'tpu'},
    ],
)
def test_set_process_env_validates_arguments(kwargs):
    # Validation happens before any environment change or import check, so it can run in-process.
    sys.path.insert(0, str(FORK))
    try:
        from cbl import env

        before = dict(os.environ)
        with pytest.raises(ValueError):
            env.set_process_env(**kwargs)
        assert dict(os.environ) == before
    finally:
        sys.path.remove(str(FORK))


@pytest.mark.parametrize(
    'module, name, word',
    [('hgq.utils.sugar', 'StopIf', 'shim 1'), ('hgq.layers', 'QLinformerAttentionT', 'shim 2')],
)
@pytest.mark.pc_gpu
def test_bind_refuses_to_shadow_an_existing_name(tmp_path, module, name, word):
    out = run(
        f"""
        from cbl import env
        env.set_process_env(mem_fraction='0.5', jax_platforms='cpu')
        import importlib
        m = importlib.import_module({module!r})
        sentinel = object()
        setattr(m, {name!r}, sentinel)
        import hgq.layers as hl, hgq.utils.sugar as hs
        try:
            env.bind_shims()
            emit({{'raised': None}})
        except RuntimeError as e:
            emit({{'raised': str(e), 'kept': getattr(m, {name!r}) is sentinel,
                   'other_unbound': not (hasattr(hs, 'StopIf') and hasattr(hl, 'QLinformerAttentionT')),
                   'vendored_loaded': env.VENDORED_MODULE_NAME in sys.modules}})
        """,
        tmp_path=tmp_path,
    )
    assert out['raised'] and 'would shadow' in out['raised'] and word in out['raised']
    assert out['kept'] and out['other_unbound'] and not out['vendored_loaded']


def test_bind_refuses_after_chang_import(tmp_path):
    out = run(
        """
        from cbl import env
        env.set_process_env(mem_fraction='0.5', jax_platforms='cpu')
        sys.modules['data'] = types.ModuleType('data')
        try:
            env.bind_shims()
            emit({'raised': None})
        except RuntimeError as e:
            emit({'raised': str(e)})
        """,
        tmp_path=tmp_path,
    )
    assert out['raised'] and "Chang's module(s) ['data'] already imported" in out['raised']


def test_bind_refuses_a_changed_stop_if(tmp_path):
    bad = tmp_path / 'stop_if.py'
    bad.write_bytes((FORK / 'vendor' / 'stop_if.py').read_bytes() + b'\n')
    out = run(
        f"""
        from pathlib import Path
        from cbl import env
        env.set_process_env(mem_fraction='0.5', jax_platforms='cpu')
        env.VENDORED_STOPIF = Path({str(bad)!r})
        try:
            env.bind_shims()
            emit({{'raised': None}})
        except RuntimeError as e:
            emit({{'raised': str(e), 'hgq_imported': 'hgq' in sys.modules}})
        """,
        tmp_path=tmp_path,
    )
    assert out['raised'] and 'sha256' in out['raised'] and not out['hgq_imported']


def test_bind_refuses_without_process_env(tmp_path):
    out = run(
        """
        from cbl import env
        try:
            env.bind_shims()
            emit({'raised': None})
        except RuntimeError as e:
            emit({'raised': str(e)})
        """,
        tmp_path=tmp_path,
    )
    assert out['raised'] and 'KERAS_BACKEND' in out['raised']


def test_import_chang_model_requires_shims(tmp_path):
    out = run(
        """
        from cbl import env
        env.set_process_env(mem_fraction='0.5', jax_platforms='cpu')
        try:
            env.import_chang_model()
            emit({'raised': None})
        except RuntimeError as e:
            emit({'raised': str(e), 'model_imported': 'model' in sys.modules})
        """,
        tmp_path=tmp_path,
    )
    assert out['raised'] and 'bind_shims() must run' in out['raised'] and not out['model_imported']


@pytest.mark.pc_gpu
def test_chang_import_fails_without_the_shims(tmp_path):
    out = run(
        f"""
        os.environ['KERAS_BACKEND'] = 'jax'
        sys.path.insert(0, {str(FORK / 'chang')!r})
        o = {{}}
        try:
            import model
            o['model'] = None
        except ImportError as e:
            o['model'] = str(e)
        try:
            from hgq.utils.sugar import StopIf
            o['stopif'] = None
        except ImportError as e:
            o['stopif'] = str(e)
        emit(o)
        """,
        tmp_path=tmp_path,
    )
    assert out['model'] and 'QLinformerAttentionT' in out['model']
    assert out['stopif'] and 'StopIf' in out['stopif']


@pytest.mark.pc_gpu
def test_changed_binding_and_precision_are_detected(tmp_path):
    out = run(
        """
        from cbl import env
        env.setup(mem_fraction='0.5', jax_platforms='cpu')
        import jax
        import hgq.utils.sugar as hs
        o = {}
        jax.config.update('jax_default_matmul_precision', 'tensorfloat32')
        try:
            env.assert_matmul_precision('after a change')
            o['precision'] = None
        except RuntimeError as e:
            o['precision'] = str(e)
        hs.StopIf = object()
        try:
            env.bind_shims()
            o['binding'] = None
        except RuntimeError as e:
            o['binding'] = str(e)
        emit(o)
        """,
        tmp_path=tmp_path,
    )
    assert out['precision'] and "'tensorfloat32', not 'float32'" in out['precision']
    assert out['binding'] and 'changed' in out['binding']
