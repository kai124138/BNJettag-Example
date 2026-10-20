"""M1: process environment, the two declared shims, Chang's import and the float32 matmul override.

STUDY v0.8 section 7.3 (backend, "Two shims", binding, the TF32 override, memory fraction);
PREFLIGHT-PLAN section 2 row M1 and section 8 decision U2.

Order of operations, all inside `setup()`:

1. `set_process_env()` sets KERAS_BACKEND and XLA_PYTHON_CLIENT_MEM_FRACTION, the two variables
   Chang's `run_train.py:3-4` sets, and JAX_PLATFORMS only when the caller passes one. It refuses
   once jax, jaxlib or keras is imported.
2. `bind_shims()` imports `hgq.utils.sugar` and `hgq.layers`, refuses if either shim name already
   exists there (a shim fills a gap and never shadows an installed name) or if any of Chang's
   modules is already imported, checks the sha256 of the vendored `vendor/stop_if.py` against the
   snapshot hash and executes exactly the bytes it hashed, then sets the two attributes on the two
   module objects. `from X import Y` reads the module attribute, so Chang's import lines
   (`run_train.py:13`, `model.py:4-18`) resolve to these objects. No installed file and none of
   Chang's files is changed.
3. `import_chang_model()` puts the fork's `chang/` first on `sys.path` (Chang's files import each
   other as top-level modules, `run_train.py:12,15`), checks the sha256 of `chang/model.py` and
   imports it. `model.py:20-23` sets `jax_default_matmul_precision` to 'tensorfloat32' at import.
4. `set_matmul_precision()` sets it to 'float32', reads it back, raises unless it is 'float32',
   and logs it.

The device. This module never chooses one (decision U2): the runner constructs the Datasets and
passes 'gpu:0' or 'cpu:0', as `run_train.py:73-74` does. JAX_PLATFORMS is a guard the caller may
pass so that a GPU run cannot fall back to the CPU silently; it is not the Dataset device.

The memory fraction is a parameter with no default here: STUDY section 7.3 calls it an
operational setting chosen for the measured pack size; the release's value is '0.5'.

Ported in batch B1 from `campaigns/2026-10-16-chang-bitlinear/preflight/env/scripts/`:

    cbl_shims.py   sha256 dad29ef0ab2ba5a280940d9c84425ecbfa68c9577e6e71e2a899f9385b1bdbf6
    cbl_common.py  sha256 4c5d9b0147d390fb2a3de65d18045049fa5fdf4fd6793542bc055f67730981bd

Departures from those two files:
- the vendored StopIf is the fork's `vendor/stop_if.py` (same bytes, `ac398006...`), executed from
  the bytes that were hashed rather than reopened by the import system;
- the memory fraction and JAX_PLATFORMS are parameters (the source fixed '0.5' and always set
  JAX_PLATFORMS from its device argument);
- Chang's directory is the fork's `chang/`, and `model.py` is hash-checked before import;
- binding also refuses once any of Chang's modules is imported; a second `bind_shims()` call in
  the same process returns the first record if the two attributes are still this module's objects;
- `setup()` raises unless `import model` left 'tensorfloat32' (otherwise the module imported is
  not the release's);
- the package list, device list, CUDA-library list and the 512x512 matmul check moved to
  `provenance()` and `matmul_check()`, outside the setup order;
- the smoke runner's Dataset device remap is not ported (decision U2).

Nothing here is quotable; it is environment code.
"""

import hashlib
import importlib
import logging
import os
import platform
import sys
import types
from pathlib import Path

log = logging.getLogger('cbl.env')

FORK_ROOT = Path(__file__).resolve().parents[1]
CHANG_DIR = FORK_ROOT / 'chang'
VENDORED_STOPIF = FORK_ROOT / 'vendor' / 'stop_if.py'

# Both hashes are the ones code-base.txt records (batch B0); gate G01 checks them against
# reference-code/.
STOPIF_SNAPSHOT_SHA256 = 'ac39800657d424ae2b19c897412e1b5971c7e022ea7bcb74960392e6d4a826a3'
CHANG_MODEL_SHA256 = '7fcb38b43efc29d06edb8ae8f4e353cd7688970d4729b5efc57e48c942203b3e'

MATMUL_PRECISION = 'float32'  # STUDY section 7.3, the one declared precision departure
RELEASE_PRECISION = 'tensorfloat32'  # model.py:20-23
ALLOWED_BACKENDS = ('jax',)  # STUDY section 7.3: JAX as released (run_train.py:3)
ALLOWED_JAX_PLATFORMS = (None, 'cpu', 'cuda')

# Chang's seven released files import one another as top-level modules.
CHANG_MODULE_NAMES = ('model', 'data', 'run_train', 'run_test', 'test_utils', 'prepare_data')
VENDORED_MODULE_NAME = 'cbl_vendor_stop_if'
SHIM_NAMES = ('hgq.utils.sugar.StopIf', 'hgq.layers.QLinformerAttentionT')

_STATE = {'process_env': None, 'shims': None, 'stopif': None, 'setup': None}


class QLinformerAttentionT:
    """Placeholder for `hgq.layers.QLinformerAttentionT` (STUDY section 7.3, shim 2).

    The name exists in the HGQ2 snapshot (`reference-code/HGQ2/src/hgq/layers/attn/linformer.py:183`)
    and not in hgq2 0.1.9. `model.py:13` imports it; only `get_llformer` uses it, never the 'xfm'
    graph. Constructing it raises.
    """

    cbl_shim = True

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            'cbl.env shim: QLinformerAttentionT is a placeholder (absent from hgq2 0.1.9, unused by '
            "the 'xfm' graph); it must not be constructed"
        )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def set_process_env(*, mem_fraction, backend='jax', jax_platforms=None, dont_write_bytecode=True) -> dict:
    """Set the backend and memory-fraction variables before jax or keras is imported.

    `mem_fraction` is the XLA_PYTHON_CLIENT_MEM_FRACTION value, a string or a number in (0, 1].
    `jax_platforms` is None (leave JAX_PLATFORMS as it is) or 'cpu' or 'cuda'.
    """
    if backend not in ALLOWED_BACKENDS:
        raise ValueError(f'backend {backend!r} not in {ALLOWED_BACKENDS} (STUDY section 7.3)')
    value = mem_fraction if isinstance(mem_fraction, str) else repr(float(mem_fraction))
    try:
        fraction = float(value)
    except ValueError:
        raise ValueError(f'mem_fraction {mem_fraction!r} is not a number') from None
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f'mem_fraction {mem_fraction!r} is outside (0, 1]')
    if jax_platforms not in ALLOWED_JAX_PLATFORMS:
        raise ValueError(f'jax_platforms {jax_platforms!r} not in {ALLOWED_JAX_PLATFORMS}')
    imported = [m for m in ('jax', 'jaxlib', 'keras') if m in sys.modules]
    if imported:
        raise RuntimeError(f'set_process_env must run before jax or keras is imported; already imported: {imported}')

    keys = ('KERAS_BACKEND', 'XLA_PYTHON_CLIENT_MEM_FRACTION', 'JAX_PLATFORMS')
    previous = {k: os.environ.get(k) for k in keys}
    os.environ['KERAS_BACKEND'] = backend
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = value
    if jax_platforms is not None:
        os.environ['JAX_PLATFORMS'] = jax_platforms
    if dont_write_bytecode:
        sys.dont_write_bytecode = True
    rec = {
        'set': {k: os.environ.get(k) for k in keys},
        'previous': previous,
        'jax_platforms_passed': jax_platforms,
        'dont_write_bytecode': bool(sys.dont_write_bytecode),
    }
    _STATE['process_env'] = rec
    log.info('process env %s', rec['set'])
    return rec


def _bound_ok(hgq_sugar, hgq_layers) -> bool:
    return (
        _STATE['stopif'] is not None
        and getattr(hgq_sugar, 'StopIf', None) is _STATE['stopif']
        and getattr(hgq_layers, 'QLinformerAttentionT', None) is QLinformerAttentionT
    )


def bind_shims() -> dict:
    """Bind StopIf and the QLinformerAttentionT placeholder onto the imported hgq modules.

    Refuses (RuntimeError) if either name already exists on its module, if one of Chang's modules
    is already imported, if KERAS_BACKEND is not 'jax', or if `vendor/stop_if.py` does not hash to
    the snapshot value. Returns a provenance record.
    """
    if _STATE['shims'] is not None:
        import hgq.layers as hgq_layers
        import hgq.utils.sugar as hgq_sugar

        if _bound_ok(hgq_sugar, hgq_layers):
            return _STATE['shims']
        raise RuntimeError('the shim attributes changed after bind_shims() bound them')

    loaded = [n for n in CHANG_MODULE_NAMES if n in sys.modules]
    if loaded:
        raise RuntimeError(
            f"Chang's module(s) {loaded} already imported; the shims must be bound before Chang's "
            'files are imported (STUDY section 7.3 item 3)'
        )
    if os.environ.get('KERAS_BACKEND') != 'jax':
        raise RuntimeError(f"KERAS_BACKEND is {os.environ.get('KERAS_BACKEND')!r}; call set_process_env first")

    source = VENDORED_STOPIF.read_bytes()
    got = sha256_bytes(source)
    if got != STOPIF_SNAPSHOT_SHA256:
        raise RuntimeError(f'{VENDORED_STOPIF} sha256 {got} != snapshot {STOPIF_SNAPSHOT_SHA256}')

    import hgq.layers as hgq_layers
    import hgq.utils.sugar as hgq_sugar
    import keras

    if keras.backend.backend() != 'jax':
        raise RuntimeError(f'keras backend is {keras.backend.backend()!r}, not jax')

    present = [n for n, mod in (('StopIf', hgq_sugar), ('QLinformerAttentionT', hgq_layers)) if hasattr(mod, n)]
    if 'StopIf' in present:
        raise RuntimeError('installed hgq.utils.sugar already has StopIf; shim 1 would shadow it')
    if 'QLinformerAttentionT' in present:
        raise RuntimeError('installed hgq.layers already has QLinformerAttentionT; shim 2 would shadow it')

    mod = types.ModuleType(VENDORED_MODULE_NAME)
    mod.__file__ = str(VENDORED_STOPIF)
    sys.modules[VENDORED_MODULE_NAME] = mod
    try:
        exec(compile(source, str(VENDORED_STOPIF), 'exec'), mod.__dict__)
    except BaseException:
        sys.modules.pop(VENDORED_MODULE_NAME, None)
        raise
    stopif = mod.StopIf

    hgq_sugar.StopIf = stopif
    hgq_layers.QLinformerAttentionT = QLinformerAttentionT
    _STATE['stopif'] = stopif

    this_file = Path(__file__).resolve()
    rec = {
        'bound': list(SHIM_NAMES),
        'absent_before_binding': True,
        'chang_modules_imported_before_binding': [],
        'stopif_vendored_path': str(VENDORED_STOPIF),
        'stopif_vendored_sha256': got,
        'stopif_snapshot_sha256': STOPIF_SNAPSHOT_SHA256,
        'stopif_module_name': VENDORED_MODULE_NAME,
        'shim_file': str(this_file),
        'shim_file_sha256': sha256_file(this_file),
        'hgq_sugar_init': hgq_sugar.__file__,
        'hgq_sugar_init_sha256': sha256_file(hgq_sugar.__file__),
        'hgq_layers_init': hgq_layers.__file__,
        'hgq_layers_init_sha256': sha256_file(hgq_layers.__file__),
    }
    _STATE['shims'] = rec
    log.info('shims bound %s (stop_if.py %s, env.py %s)', rec['bound'], got[:12], rec['shim_file_sha256'][:12])
    return rec


def import_chang_model():
    """Import Chang's `model.py` from the fork's `chang/` after the shims are bound."""
    if _STATE['shims'] is None:
        raise RuntimeError("bind_shims() must run before Chang's model is imported")
    path = (CHANG_DIR / 'model.py').resolve()
    got = sha256_file(path)
    if got != CHANG_MODEL_SHA256:
        raise RuntimeError(f'{path} sha256 {got} != {CHANG_MODEL_SHA256}')
    chang = str(CHANG_DIR.resolve())
    if chang not in sys.path:
        sys.path.insert(0, chang)
    mod = sys.modules.get('model')
    if mod is None:
        mod = importlib.import_module('model')
    if Path(mod.__file__).resolve() != path:
        raise RuntimeError(f"module 'model' comes from {mod.__file__}, not {path}")
    return mod


def matmul_precision():
    import jax

    return jax.config.jax_default_matmul_precision


def log_matmul_precision(where: str) -> dict:
    """Read and log the active default matmul precision (STUDY section 7.3: logged for every run)."""
    rec = {'where': where, 'jax_default_matmul_precision': matmul_precision()}
    log.info('matmul_precision %s %r', where, rec['jax_default_matmul_precision'])
    return rec


def assert_matmul_precision(where: str) -> dict:
    rec = log_matmul_precision(where)
    if rec['jax_default_matmul_precision'] != MATMUL_PRECISION:
        raise RuntimeError(f"matmul precision at {where} is {rec['jax_default_matmul_precision']!r}, not {MATMUL_PRECISION!r}")
    return rec


def set_matmul_precision() -> dict:
    """Apply the declared override: jax_default_matmul_precision = 'float32'."""
    import jax

    before = jax.config.jax_default_matmul_precision
    jax.config.update('jax_default_matmul_precision', MATMUL_PRECISION)
    after = jax.config.jax_default_matmul_precision
    if after != MATMUL_PRECISION:
        raise RuntimeError(f'matmul precision override not active: {after!r}')
    log.info('matmul_precision override %r -> %r', before, after)
    return {'before_override': before, 'after_override': after}


def setup(*, mem_fraction, jax_platforms=None, backend='jax') -> dict:
    """Steps 1-4 of the module docstring, in order. Returns the record of each step."""
    penv = set_process_env(mem_fraction=mem_fraction, backend=backend, jax_platforms=jax_platforms)
    shims = bind_shims()
    import jax

    before_import = jax.config.jax_default_matmul_precision
    model = import_chang_model()
    after_import = jax.config.jax_default_matmul_precision
    if after_import != RELEASE_PRECISION:
        raise RuntimeError(f'after import model the precision is {after_import!r}, not the release value {RELEASE_PRECISION!r}')
    prec = set_matmul_precision()
    rec = {
        'process_env': penv,
        'shims': shims,
        'chang_model': {'file': model.__file__, 'sha256': CHANG_MODEL_SHA256},
        'matmul_precision': {
            'before_import_model': before_import,
            'after_import_model': after_import,
            'after_override': prec['after_override'],
        },
    }
    _STATE['setup'] = rec
    log.info('matmul_precision record %s', rec['matmul_precision'])
    return rec


PROVENANCE_PACKAGES = (
    'jax', 'jaxlib', 'jax-cuda12-plugin', 'jax-cuda12-pjrt', 'keras', 'hgq2', 'quantizers', 'numpy', 'h5py',
    'ml-dtypes', 'scipy', 'opt-einsum', 'optree', 'wandb', 'nvidia-cudnn-cu12', 'nvidia-cublas-cu12',
    'nvidia-cuda-runtime-cu12', 'nvidia-cuda-nvcc-cu12',
)


def loaded_cuda_libs() -> list:
    """Shared objects from CUDA, cuDNN and cuBLAS mapped into this process (Linux only)."""
    keys = ('libcudnn', 'libcublas', 'libcudart', 'libcuda.so', 'libnvrtc', 'libnvJitLink', 'libcufft', 'libcusolver', 'libcusparse')
    out = set()
    try:
        with open('/proc/self/maps') as f:
            for line in f:
                parts = line.split()
                if parts and any(k in parts[-1] for k in keys):
                    out.add(parts[-1])
    except OSError:
        pass
    return sorted(out)


def provenance() -> dict:
    """Interpreter, package versions, backend and devices. Initializes the JAX backend."""
    import importlib.metadata as md

    import jax
    import keras

    pkgs = {}
    for name in PROVENANCE_PACKAGES:
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pkgs[name] = None
    devices = jax.devices()
    rec = {
        'python': sys.version,
        'executable': sys.executable,
        'platform': platform.platform(),
        'packages': pkgs,
        'keras_backend': keras.backend.backend(),
        'jax_backend': jax.default_backend(),
        'jax_devices': [str(d) for d in devices],
        'device_kind': [getattr(d, 'device_kind', None) for d in devices],
        'env': {k: os.environ.get(k) for k in ('KERAS_BACKEND', 'XLA_PYTHON_CLIENT_MEM_FRACTION', 'JAX_PLATFORMS', 'LD_LIBRARY_PATH',
                                               'XLA_FLAGS')},
        'jax_default_matmul_precision': jax.config.jax_default_matmul_precision,
        'loaded_cuda_libs': loaded_cuda_libs(),
    }
    return rec


def matmul_check(n: int = 512, seed: int = 12345) -> dict:
    """Relative error of an n x n x n float32 jnp.matmul against float64 NumPy, under the active
    precision and under an explicit 'tensorfloat32' context (ported from cbl_common._matmul_check).
    Uses a private NumPy generator; never touches the global random streams."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)).astype(np.float32)
    b = rng.standard_normal((n, n)).astype(np.float32)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    out_active = np.asarray(jax.jit(lambda x, y: jnp.matmul(x, y))(a, b)).astype(np.float64)
    with jax.default_matmul_precision(RELEASE_PRECISION):
        out_tf32 = np.asarray(jax.jit(lambda x, y: jnp.matmul(x, y) + 0.0)(a, b)).astype(np.float64)

    def rel(o):
        return float(np.max(np.abs(o - ref)) / np.max(np.abs(ref)))

    return {'n': n, 'rel_err_active_precision': rel(out_active), 'rel_err_explicit_tf32': rel(out_tf32)}
