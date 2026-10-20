"""M4: seed flow, initial tensors by role, batch-order recorder and stream guard.

STUDY means `review/snapshots/v0.8/STUDY.md` (sha256 290022fa41a726b8...), section 6 "RNG flow" steps
1-8, section 7.1 last bullet and the section 13 RNG row; "plan" means PREFLIGHT-PLAN.md (sha256
2f1be9f3...30ff8), section 2 row M4 and section 8 decisions U1 and U5. Gates G10-G12.

What is here, in the order a run uses it

1. `SeedFlow`, the runner-facing API of section 6 steps 1-5. It refuses a step out of order and
   records the module-level Python `random` state and the NumPy legacy state after every step.
   - step 1 `load_data`: `np.random.seed(42)`, then M3's `cbl.data.read_train_val`, which is
     `chang/data.py:7-43` without the held-out lines and refuses unless NumPy is in the state that
     `np.random.seed(42)` leaves.
   - step 2 `seed_streams`: `random.seed(s)` and `keras.utils.set_random_seed(s)`. The latter also
     reseeds NumPy's legacy generator with s (Keras 3.14.0 `keras/src/utils/rng_utils.py:56-63`).
   - step 3 `build_datasets`, then `build_model`: the two hgq `Dataset`s as `run_train.py:73-74`
     (training batch 2790, `shuffle=True`, default `drop_last=False`; validation 2790), with the
     device the caller passes (decision U2), then the model (`run_train.py:76`). The construction
     draws of the Datasets (`dataset.py:66-79`: `randint(0, remainder)` per Dataset and one shuffle
     of the training batch map) are reconstructed from `random.Random(s)` and must equal what was
     drawn; the model build draws Keras initializer seeds from the same stream.
   - step 4 `load_initial_tensors` (decision U1): every arm, CH-8 included, loads the seed's file
     by role. Tensors of the arm with no role in the file are the unmatched set (section 6 step 4:
     the scale quantizers' b and i, which keep their constant initializers); roles of the file the
     arm does not have are the unused set. Both must equal what the arm's `RoleSpec` declares; for
     CH-8 both are empty.
   - step 5 `reseed_before_fit` (decision U5): `random.seed(prefit_reseed(s))` with
     prefit_reseed(s) = int.from_bytes(sha256(b"cbl1016-fit\\0" + str(s).encode()).digest()[:8],
     "big"), the same for every arm of seed s.
2. `draw_initial_tensors`: steps 1-3 with CH-8's released builder (`chang/model.py` `get_model('xfm',
   7, 7, 1e-8, 8, True)`), on the device of the process, then every variable saved by role to a
   deterministic `.npz` with a JSON manifest (role -> shape, dtype, sha256; the file's own sha256;
   the identity of seed, device, stack, data and draw code). The loader refuses a manifest whose
   sha256 differs from the one the caller names, a file whose sha256 differs from the manifest's,
   any tensor whose sha256, shape or dtype differs, and any identity field that differs from the
   run's (seed, device platform and kind, Dataset device, package versions, matmul precision,
   draw-code hashes, data identity and the Datasets' construction draws). The file depends on the
   data: the Datasets draw before the model is built, and their draws depend on the archive's
   length, so a file drawn on the smoke archive is not the file of the archive of record.
3. Roles. `walk_roles(model, top_names)` walks objects, never Keras path strings (the layer names
   in paths carry process-global counters, such as `q_einsum_dense_batchnorm_2` or
   `fixed_point_quantizer_kbi_13`). A role is the arm's semantic name of the top-level layer
   (`model.layers` order), then one segment per tracked sublayer (`Layer._layers` order), then the
   variable's own name. A sublayer segment is `<Class>[k]` when the sublayer carries Keras's
   automatic name for its class (`to_snake_case(class)` with an optional `_<n>` counter; k counts
   siblings of that class), the suffix after `<parent name>_` when the name is derived from the
   parent's (hgq's `<layer>_kq`, `<layer>_iq`, ...), and the name itself otherwise (the MHA
   projections `query`, `key`, `value`, `attention_output`). The walk must cover `model.variables`
   exactly once each, with unique roles. `CH8_ROLE_SPEC` pins CH-8's 14 top-level layers by class
   (`model.py:190-208`) and 244 variables. Other arms (F000 in B5, the binary arms in B6) supply
   their own `RoleSpec` with the same semantic names and their declared unmatched and unused sets.
4. `BatchIndexRecorder` and `recording_dataset_class` (section 6 step 7). The training Dataset is
   an instance of a subclass, made at run time, of the installed `hgq.utils.sugar.Dataset`; no file
   of Chang's or of hgq2 is changed. The subclass replaces the instance's `_index_map` list by a
   list subclass (`_ProbeList`) that records an integer read only while the subclass's own
   `__getitem__` is running, so the recorded value is exactly the one `dataset.py:88-89` read (the
   mapped batch index), and the reads that `random.shuffle` makes while it reshuffles the map are
   not recorded. The subclass also records `on_epoch_begin` and `on_epoch_end` with the random
   state around each. `instrument(dataset)` does the same to an existing instance (for a Dataset
   constructed by Chang's own `run_train.py`). An epoch marker (`EpochMarker`, or the call
   `recorder.mark_keras_epoch(e)` from the appended frozen-evaluation callback) ties each Dataset
   epoch to Keras's epoch number; fetches before the first marker are the pre-fit record.
   Hashes: `batch_index_seq_sha256` and `index_map_sha256` are the sha256 of the little-endian
   int64 bytes of the sequence (`int_seq_sha256`).
5. `reconstruct_fit_stream`: the draws Keras 3.14.0's JAX `fit` makes from the module-level stream
   after the reseed, for one hgq Dataset with `shuffle=True`, `drop_last=False` and one worker:
   - before epoch 0, `_symbolic_build` (`keras/src/trainers/trainer.py:1104-1110`, called at
     `backend/jax/trainer.py:418`) starts the epoch iterator once: the adapter's
     `_finite_generator` shuffles the batch order (`trainers/data_adapters/py_dataset_adapter.py:257-261`)
     and the one-batch-ahead iterator (`backend/jax/trainer.py:1108-1126`) fetches two batches; then
     `epoch_iterator.reset()` (`backend/jax/trainer.py:419`, `trainers/epoch_iterator.py:98-103`)
     calls the Dataset's `on_epoch_end`, which reshuffles `_index_map` (`dataset.py:111-113`);
   - each epoch: one shuffle of the batch order at the epoch's first fetch, then n fetches of
     `_index_map[order[j]]`, then one reshuffle of `_index_map` at the adapter's `on_epoch_end`
     (`trainers/epoch_iterator.py:134`). `JAXEpochIterator.__next__` hands the raw iterator to the
     multi-step train function, which swallows a short last group's StopIteration
     (`backend/jax/trainer.py:227-250`), so this holds whether or not n is a multiple of
     `steps_per_execution`; the iterator made at the end of an epoch is a lazy generator and draws
     nothing. Validation draws nothing (`shuffle=False`, and its Dataset has no `_index_map`).
   It returns the expected fetches per epoch, the map per epoch and the state of the stream at
   every epoch boundary; G11 compares them with the recorder, and G12 with the live state.
6. `StreamGuard` (section 6 step 6). Between the reseed and the end of training it wraps every
   overridden hook of every callback it is given, compares the module-level `random` state and the
   NumPy legacy state before and after each call, and raises `StreamViolation` (or records it) when
   a lab callback changed either. `watch(label)` does the same around a block of lab code, such as
   a frozen-evaluation step (M8, B7). Only hooks the callback's class overrides are wrapped, so
   Keras's choice of synchronous or asynchronous batch-end dispatch (`callbacks/callback_list.py:59-84`
   reads `is_default` of each hook) is the same with and without the guard.

Findings this module depends on (installed 0.1.9, identical to the snapshot where cited):
- hgq quantizers draw `np.random.randint(0, 2**31)` from NumPy's global legacy generator at
  construction (`hgq/quantizer/internal/base.py:83`); the value seeds a Keras SeedGenerator only for
  stochastic rounding modes (`fixed_point_quantizer.py:56-57`). `stochastic_quantizers(model)` lists
  quantizers with a seed generator; on CH-8 there is none (round modes RND and RND_CONV), so the
  NumPy draws of the build affect no computation. Arms with different quantizer counts leave NumPy in
  different states, which nothing in training reads.
- The release seeds `random` with 42 before `get_data` (`run_train.py:42`); section 6 step 2 seeds it
  with s afterwards, with no draw in between, so that line has no effect and is not repeated.

The pure parts (U5, hashing, the reconstructions, the file format, the guard's state comparison)
import only the standard library and NumPy; Keras, JAX and hgq are imported inside the functions
that need them, after `cbl.env.setup()`. Nothing here is quotable; it is run machinery.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import random
import re
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

FORK = Path(__file__).resolve().parents[1]

SPLIT_SEED = 42                      # run_train.py:41
TRAIN_BATCH = 2790                   # run_train.py:67 (default --batch-size), :73
VAL_BATCH = 2790                     # run_train.py:74
CH8_BUILDER_ARGS = ('xfm', 7, 7, 1e-8, 8, True)   # run_train.py:76 with -m xfm -n 8 -f 3
CH8_N_CONSTITUENTS = 8
CH8_PTETAPHI = True
FIT_RESEED_TAG = b'cbl1016-fit\0'    # decision U5
INIT_KIND = 'cbl1016-init-tensors'
INIT_SCHEMA = 1
KERAS_MODELLED = '3.14.0'            # the version whose fit loop reconstruct_fit_stream models
SYMBOLIC_BUILD_FETCHES = 2           # backend/jax/trainer.py:1108-1126, one batch ahead
PROBE_SEED = 20261018                # private generator of the forward probe batch
PROBE_N = 64
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
DRAW_CODE = ('cbl/rng.py', 'cbl/env.py', 'cbl/data.py', 'chang/model.py', 'chang/data.py')
STACK_PACKAGES = ('jax', 'jaxlib', 'jax-cuda12-plugin', 'jax-cuda12-pjrt', 'keras', 'hgq2', 'quantizers',
                  'numpy', 'h5py', 'ml-dtypes')

# CH-8's top-level layers in model.layers order, with the semantic names that are the first
# segment of every role (chang/model.py:190-208).
CH8_TOP = (
    ('InputLayer', 'input'),                      # :190
    ('QEinsumDenseBatchnorm', 'embed'),           # :192
    ('QAffinedUnaryFunctionLUT', 'tanh1'),        # :194
    ('QMultiHeadAttention', 'mha'),               # :195
    ('QAdd', 'add1'),                             # :197
    ('QAffinedUnaryFunctionLUT', 'tanh2'),        # :198
    ('QEinsumDenseBatchnorm', 'ffn1'),            # :199
    ('QEinsumDense', 'ffn2'),                     # :200
    ('QAdd', 'add2'),                             # :202
    ('QGlobalAveragePooling1D', 'pool'),          # :204
    ('QEinsumDenseBatchnorm', 'head1'),           # :205
    ('QEinsumDenseBatchnorm', 'head2'),           # :206
    ('QEinsumDenseBatchnorm', 'head3'),           # :207
    ('QEinsumDense', 'logits'),                   # :208
)
CH8_N_VARIABLES = 244


class FlowError(RuntimeError):
    """A section-6 step was called out of order, or a step's integrity check failed."""


class RoleError(RuntimeError):
    """The model's variables cannot be given unique canonical roles."""


class InitTensorRefused(RuntimeError):
    """The initial-tensor manifest or file does not match what the run expects (decision U1)."""


class StreamViolation(RuntimeError):
    """Lab code changed the module-level random state (or NumPy's) during training (section 6 step 6)."""


# ----------------------------------------------------------------------------------------------
# decision U5 and hashing (pure)


def prefit_reseed(seed: int) -> int:
    """Decision U5: the seed of the module-level `random` stream just before `fit`."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f'seed must be an int, not {seed!r}')
    return int.from_bytes(hashlib.sha256(FIT_RESEED_TAG + str(seed).encode()).digest()[:8], 'big')


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def int_seq_sha256(seq) -> str:
    """sha256 of the little-endian int64 bytes of an integer sequence (batch_index_seq_sha256,
    index_map_sha256)."""
    return sha256_bytes(np.asarray(list(seq), dtype='<i8').tobytes())


def tensor_sha256(arr) -> str:
    """sha256 of the C-ordered bytes of an array; dtype and shape are recorded beside it."""
    return sha256_bytes(np.asarray(arr, order='C').tobytes())


def py_state_sha256(state=None) -> str:
    """sha256 of a module-level (or private) `random` state: version, the 625 words, gauss_next."""
    version, words, gauss = random.getstate() if state is None else state
    return sha256_bytes(json.dumps([version, list(words), gauss]).encode())


def np_state_sha256(state=None) -> str:
    """sha256 of NumPy's legacy global state (MT19937 key, position, gauss cache)."""
    name, key, pos, has_gauss, cached = np.random.get_state(legacy=True) if state is None else state
    return sha256_bytes(json.dumps([name, np.asarray(key, dtype='<u4').tolist(), int(pos), int(has_gauss),
                                    float(cached)]).encode())


def np_state_equals(a, b) -> bool:
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and tuple(a[2:]) == tuple(b[2:])


# ----------------------------------------------------------------------------------------------
# reconstructions of the consumers (pure)


def dataset_construction_draws(rng: random.Random, n_items: int, batch_size: int, shuffle: bool) -> dict:
    """The draws of `Dataset.__init__` -> `Dataset.batch` (dataset.py:57-58, 66-79) on `rng`:
    `randint(0, n_items % batch_size)` always, then one shuffle of `range(n_batches)` when shuffle
    is set (n_batches = ceil(n_items / batch_size) with drop_last=False, dataset.py:60-64)."""
    rem = n_items % batch_size
    n_batches = -(-n_items // batch_size)
    shift = rng.randint(0, rem)
    index_map = None
    if shuffle:
        index_map = list(range(n_batches))
        rng.shuffle(index_map)
    return {'n_batches': n_batches, 'remainder': rem, 'init_shift': shift, 'index_map': index_map}


def reconstruct_datasets(seed: int, n_train: int, n_val: int, batch_size=TRAIN_BATCH, val_batch_size=VAL_BATCH) -> dict:
    """Section 6 steps 2-3 from `random.Random(seed)`: the training Dataset (shuffle=True), then the
    validation Dataset (shuffle=False), as run_train.py:73-74. Returns their draws and the state."""
    r = random.Random(seed)
    tr = dataset_construction_draws(r, n_train, batch_size, True)
    va = dataset_construction_draws(r, n_val, val_batch_size, False)
    return {'train': tr, 'val': va, 'state_sha256': py_state_sha256(r.getstate()), 'state': r.getstate()}


def reconstruct_fit_stream(*, reseed: int | None, index_map, n_batches: int, epochs: int,
                           symbolic_build_fetches: int = SYMBOLIC_BUILD_FETCHES, state=None) -> dict:
    """The module-level stream from the reseed to the end of `fit` under the Keras 3.14.0 JAX
    trainer (module docstring, item 5). `index_map` is the training Dataset's map at the reseed.

    Returns {'prefit': {'order', 'mapped', 'index_map', 'index_map_after', 'state_sha256_at_begin',
    'state_sha256_after_order', 'state_sha256_at_end'}, 'epochs': [{'epoch', 'order', 'mapped',
    'index_map', 'index_map_after', 'batch_index_seq_sha256', 'index_map_sha256',
    'state_sha256_at_begin', 'state_sha256_after_order', 'state_sha256_at_end'}], 'end_state',
    'end_state_sha256'}. The state hashes are those of the module-level stream at the Dataset's
    on_epoch_begin, after the batch-order shuffle, and after the map's reshuffle.
    `symbolic_build_fetches` = 0 models a fit whose model, loss and metrics are already built, so
    that `_symbolic_build` does not iterate (the reset's reshuffle still happens). With `state` (a
    `random.getstate()` tuple) and reseed None, the stream starts from that state instead, as in a
    run without the reseed."""
    if (reseed is None) == (state is None):
        raise ValueError('give exactly one of reseed and state')
    r = random.Random(reseed if reseed is not None else 0)
    if state is not None:
        r.setstate(state)
    m = list(index_map)
    if sorted(m) != list(range(n_batches)):
        raise ValueError('index_map is not a permutation of range(n_batches)')
    pre = {'order': [], 'mapped': [], 'index_map': list(m), 'state_sha256_at_begin': py_state_sha256(r.getstate())}
    if symbolic_build_fetches:
        order = list(range(n_batches))
        r.shuffle(order)
        k = min(symbolic_build_fetches, n_batches)
        pre.update(order=order[:k], mapped=[m[p] for p in order[:k]])
    pre['state_sha256_after_order'] = py_state_sha256(r.getstate())
    r.shuffle(m)                                     # epoch_iterator.reset() -> Dataset.on_epoch_end
    pre['index_map_after'] = list(m)
    pre['state_sha256_at_end'] = py_state_sha256(r.getstate())
    out = []
    for e in range(epochs):
        begin = py_state_sha256(r.getstate())
        order = list(range(n_batches))
        r.shuffle(order)                             # py_dataset_adapter.py:259-261
        mapped = [m[p] for p in order]               # dataset.py:88-89
        ep = {'epoch': e, 'order': order, 'mapped': mapped, 'index_map': list(m),
              'batch_index_seq_sha256': int_seq_sha256(mapped), 'index_map_sha256': int_seq_sha256(m),
              'state_sha256_at_begin': begin, 'state_sha256_after_order': py_state_sha256(r.getstate())}
        r.shuffle(m)                                 # epoch_iterator.py:134 -> dataset.py:111-113
        ep['index_map_after'] = list(m)
        ep['state_sha256_at_end'] = py_state_sha256(r.getstate())
        out.append(ep)
    end = r.getstate()
    return {'reseed': reseed, 'n_batches': n_batches, 'epochs_modelled': epochs,
            'symbolic_build_fetches': symbolic_build_fetches, 'prefit': pre, 'epochs': out,
            'end_state': end, 'end_state_sha256': py_state_sha256(end)}


# ----------------------------------------------------------------------------------------------
# roles


def _to_snake_case(name: str) -> str:
    """Keras's automatic-name rule (keras/src/utils/naming.py `to_snake_case`), copied so that the
    pure part needs no Keras; `walk_roles` checks it against Keras's own function."""
    name = re.sub(r'\W+', '', name)
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z])([A-Z])', r'\1_\2', name).lower()


def segment_name(parent_name: str, child_name: str, child_class: str, counts: dict) -> str:
    """The role segment of one tracked sublayer (module docstring, item 3)."""
    if re.fullmatch(re.escape(_to_snake_case(child_class)) + r'(_\d+)?', child_name):
        k = counts.get(child_class, 0)
        counts[child_class] = k + 1
        return f'{child_class}[{k}]'
    if child_name.startswith(parent_name + '_') and len(child_name) > len(parent_name) + 1:
        return child_name[len(parent_name) + 1:]
    return child_name


def walk_roles(model, top_names) -> dict:
    """{role: variable} for every variable of `model`, in walk order. `top_names` gives one semantic
    name per entry of `model.layers`. Raises RoleError unless the walk covers `model.variables`
    exactly once each with unique roles."""
    from keras.src.utils.naming import to_snake_case

    layers = list(model.layers)
    if len(top_names) != len(layers):
        raise RoleError(f'{len(top_names)} top-level names for {len(layers)} layers')
    if len(set(top_names)) != len(top_names):
        raise RoleError('top-level names are not unique')
    out, seen = {}, {}

    def walk(layer, prefix):
        own = list(layer._trainable_variables) + list(layer._non_trainable_variables)
        names = [v.name for v in own]
        if len(set(names)) != len(names):
            raise RoleError(f'{prefix}: duplicate own variable names {names}')
        for v in own:
            role = f'{prefix}/{v.name}'
            if role in out:
                raise RoleError(f'duplicate role {role}')
            if id(v) in seen:
                raise RoleError(f'variable {v.path} reached twice ({seen[id(v)]} and {role})')
            out[role] = v
            seen[id(v)] = role
        counts = {}
        for sub in layer._layers:
            cls = type(sub).__name__
            if to_snake_case(cls) != _to_snake_case(cls):
                raise RoleError(f'snake-case rule differs from Keras for {cls}')
            walk(sub, prefix + '/' + segment_name(layer.name, sub.name, cls, counts))

    for name, layer in zip(top_names, layers):
        walk(layer, name)
    want = {id(v) for v in model.variables}
    if set(seen) != want or len(model.variables) != len(out):
        raise RoleError(f'walk reached {len(out)} variables, the model has {len(model.variables)}')
    return out


def ch8_top_names(model) -> list:
    """CH-8's semantic top-level names; refuses any other class sequence."""
    got = tuple(type(ly).__name__ for ly in model.layers)
    want = tuple(c for c, _ in CH8_TOP)
    if got != want:
        raise RoleError(f'not the CH-8 graph: layers {got}')
    return [n for _, n in CH8_TOP]


@dataclass(frozen=True)
class RoleSpec:
    """How an arm maps its variables to CH-8's roles (section 6 step 4).

    top_names(model) -> one semantic name per model.layers entry (CH-8's names for the layers the
    arm shares with CH-8). expected_unmatched(model) -> the arm's roles with no CH-8 counterpart
    (the scale quantizers' b and i; constant initializers, no random draw). expected_unused(model)
    -> the CH-8 roles the arm does not have (kq's variables at binary sites). The loader refuses
    unless the observed sets equal the declared ones. n_variables, when set, pins the count."""
    arm_id: str
    top_names: Callable
    expected_unmatched: Callable = field(default=lambda model: frozenset())
    expected_unused: Callable = field(default=lambda model: frozenset())
    n_variables: int | None = None
    pending: str | None = None


CH8_ROLE_SPEC = RoleSpec(arm_id='CH-8', top_names=ch8_top_names, n_variables=CH8_N_VARIABLES)


def roles_of(model, spec: RoleSpec) -> dict:
    roles = walk_roles(model, spec.top_names(model))
    if spec.n_variables is not None and len(roles) != spec.n_variables:
        raise RoleError(f'{spec.arm_id}: {len(roles)} variables, expected {spec.n_variables}')
    return roles


def variable_arrays(roles: dict) -> dict:
    """{role: numpy array} of the current values."""
    import keras

    return {r: np.asarray(keras.ops.convert_to_numpy(v.value)) for r, v in roles.items()}


def variables_sha256(roles: dict) -> str:
    """One hash over (role, dtype, shape, bytes) of every variable, in role order."""
    h = hashlib.sha256()
    for r, a in sorted(variable_arrays(roles).items()):
        h.update(json.dumps([r, str(a.dtype), list(a.shape)]).encode())
        h.update(np.asarray(a, order='C').tobytes())
    return h.hexdigest()


def stochastic_quantizers(model) -> list:
    """Paths of quantizers that hold a Keras SeedGenerator (round modes starting with 'S_')."""
    out = []
    for ly in model._flatten_layers(include_self=False):
        if hasattr(ly, '_seed_gen') and ly._seed_gen is not None:
            out.append(getattr(ly, 'path', ly.name))
    return out


# ----------------------------------------------------------------------------------------------
# the initial-tensor file (pure) and its manifest


def write_tensor_file(path, tensors: dict) -> str:
    """Write {role: array} as an uncompressed .npz with fixed member metadata, so that equal tensors
    give equal bytes. Refuses to overwrite. Returns the file's sha256."""
    path = Path(path)
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_STORED) as zf:
        for role, arr in tensors.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(arr, order='C'), allow_pickle=False)
            zi = zipfile.ZipInfo(role + '.npy', date_time=ZIP_DATE)
            zi.compress_type = zipfile.ZIP_STORED
            zi.create_system = 3
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, buf.getvalue())
    return sha256_file(path)


def read_tensor_file(path) -> dict:
    """{role: array} from a file written by write_tensor_file (no pickles)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def manifest_bytes(man: dict) -> bytes:
    return (json.dumps(man, indent=1, sort_keys=True) + '\n').encode()


def _flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(_flatten(v, key + '.'))
        else:
            out[key] = v
    return out


def identity_mismatches(drawn: dict, run: dict) -> list:
    """Dotted keys whose values differ between the manifest's identity and the run's."""
    a, b = _flatten(drawn), _flatten(run)
    return sorted(k for k in set(a) | set(b) if a.get(k, '<absent>') != b.get(k, '<absent>'))


def _is_hex64(s) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in '0123456789abcdef' for c in s)


def read_manifest(manifest_path, *, expect_manifest_sha256) -> tuple:
    """The manifest dict and its sha256, refusing unless the caller names its sha256 and it matches."""
    if not _is_hex64(expect_manifest_sha256):
        raise InitTensorRefused('no expected manifest sha256 was given (decision U1: the handoff names it); '
                                f'got {expect_manifest_sha256!r}')
    raw = Path(manifest_path).read_bytes()
    got = sha256_bytes(raw)
    if got != expect_manifest_sha256:
        raise InitTensorRefused(f'manifest {manifest_path} sha256 {got} differs from the expected {expect_manifest_sha256}')
    man = json.loads(raw)
    if not isinstance(man, dict) or man.get('kind') != INIT_KIND or man.get('schema') != INIT_SCHEMA:
        raise InitTensorRefused(f'{manifest_path} is not a {INIT_KIND} schema {INIT_SCHEMA} manifest')
    return man, got


def read_checked_tensors(manifest_path, man: dict) -> tuple:
    """The file the manifest names, checked against its sha256 and every role's sha256, dtype and shape."""
    fpath = Path(manifest_path).parent / man['file']['name']
    got = sha256_file(fpath)
    if got != man['file']['sha256']:
        raise InitTensorRefused(f'{fpath} sha256 {got} differs from the manifest {man["file"]["sha256"]}')
    tensors = read_tensor_file(fpath)
    if set(tensors) != set(man['roles']):
        raise InitTensorRefused(f'{fpath}: roles differ from the manifest '
                                f'(file only {sorted(set(tensors) - set(man["roles"]))[:5]}, '
                                f'manifest only {sorted(set(man["roles"]) - set(tensors))[:5]})')
    for role, a in tensors.items():
        e = man['roles'][role]
        if str(a.dtype) != e['dtype'] or list(a.shape) != e['shape'] or tensor_sha256(a) != e['sha256']:
            raise InitTensorRefused(f'{fpath}: tensor {role} differs from its manifest entry')
    return tensors, got


# ----------------------------------------------------------------------------------------------
# the run identity (Keras and JAX imported here)


def stack_versions() -> dict:
    import importlib.metadata as md

    out = {}
    for name in STACK_PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def device_identity(dataset_device: str) -> dict:
    import jax

    devs = jax.devices()
    return {'jax_platform': jax.default_backend(), 'device_kind': sorted({str(d.device_kind) for d in devs}),
            'n_devices': len(devs), 'dataset_device': dataset_device}


def draw_code_sha256() -> dict:
    return {rel: sha256_file(FORK / rel) for rel in DRAW_CODE}


# ----------------------------------------------------------------------------------------------
# the batch-index recorder


class _ProbeList(list):
    """`_index_map` replacement: records an int read only while its owner's __getitem__ runs."""

    __slots__ = ('_cbl_owner',)

    def __getitem__(self, i):
        v = list.__getitem__(self, i)
        owner = getattr(self, '_cbl_owner', None)
        if owner is not None and owner._cbl_fetching and isinstance(i, int):
            owner._cbl_lookups.append(v)
        return v


def _wrap_index_map(ds):
    m = ds._index_map
    if m is not None and not isinstance(m, _ProbeList):
        p = _ProbeList(m)
        p._cbl_owner = ds
        ds._index_map = p


_RECORDING_CLASSES = {}


def recording_dataset_class(base=None):
    """A subclass of `base` (default: the installed `hgq.utils.sugar.Dataset`) that reports fetches
    and epoch hooks to its `_cbl_rec` recorder. Made once per base class."""
    if base is None:
        from hgq.utils.sugar import Dataset as base
    if base in _RECORDING_CLASSES:
        return _RECORDING_CLASSES[base]

    class CblRecordingDataset(base):
        _cbl_fetching = False
        _cbl_rec = None
        _cbl_lookups = ()

        def batch(self, batch_size):
            super().batch(batch_size)
            _wrap_index_map(self)

        def __getitem__(self, idx):
            self._cbl_lookups = []
            self._cbl_fetching = True
            try:
                out = super().__getitem__(idx)
            finally:
                self._cbl_fetching = False
            rec = self._cbl_rec
            if rec is not None:
                rec._fetch(self, idx, self._cbl_lookups)
            return out

        def on_epoch_begin(self):
            rec = self._cbl_rec
            before = random.getstate()
            super().on_epoch_begin()
            if rec is not None:
                rec._epoch_begin(self, before)

        def on_epoch_end(self):
            rec = self._cbl_rec
            if rec is not None:
                rec._epoch_end_before(self)
            super().on_epoch_end()
            if rec is not None:
                rec._epoch_end_after(self)

    CblRecordingDataset.__qualname__ = CblRecordingDataset.__name__ = 'CblRecordingDataset'
    _RECORDING_CLASSES[base] = CblRecordingDataset
    return CblRecordingDataset


class BatchIndexRecorder:
    """Records, for one training Dataset, every fetch (raw index, mapped index, thread) and every
    epoch hook, and groups them by Keras epoch (module docstring, item 4).

    keep_full: None keeps every closed Dataset epoch in full (gates); an int k keeps the last k in
    full and reduces older ones to their view without the index lists (hashes and checks stay),
    so that a 7,000-epoch run does not hold millions of indices."""

    def __init__(self, keep_full: int | None = None):
        self.keep_full = keep_full
        self._segment = None
        self.segments = []
        self._marks = []
        self.anomalies = []
        self.main_thread = threading.get_ident()
        self.dataset_info = None

    # attachment
    def attach(self, ds):
        if ds.workers != 1 or ds.use_multiprocessing:
            raise FlowError(f'the training Dataset must fetch in one worker without multiprocessing '
                            f'(workers {ds.workers}, use_multiprocessing {ds.use_multiprocessing})')
        if type(ds) not in _RECORDING_CLASSES.values():
            ds.__class__ = recording_dataset_class(type(ds))
        cls = type(ds)
        _wrap_index_map(ds)
        ds._cbl_rec = self
        self.dataset_info = {'class': f'{cls.__module__}.{cls.__qualname__}', 'base': f'{cls.__mro__[1].__module__}.{cls.__mro__[1].__qualname__}',
                             'len': len(ds), 'batch_size': ds.batch_size, 'data_len': ds.data_len,
                             'drop_last': bool(ds.drop_last), 'shuffle': bool(ds.shuffle), 'workers': ds.workers,
                             'use_multiprocessing': bool(ds.use_multiprocessing)}
        return ds

    # hooks called by the Dataset subclass
    def _fetch(self, ds, idx, lookups):
        tid = threading.get_ident()
        if ds._index_map is None:
            mapped = idx % len(ds) if idx < 0 else idx
            if lookups:
                self.anomalies.append(f'fetch {idx}: {len(lookups)} map reads without a map')
        else:
            if len(lookups) != 1:
                self.anomalies.append(f'fetch {idx}: {len(lookups)} map reads, expected 1')
            mapped = lookups[0] if lookups else None
        if tid != self.main_thread:
            self.anomalies.append(f'fetch {idx} on thread {tid}, not the main thread')
        if self._segment is None:
            self.anomalies.append(f'fetch {idx} outside an epoch')
            return
        self._segment['raw'].append(int(idx))
        self._segment['mapped'].append(None if mapped is None else int(mapped))

    def _epoch_begin(self, ds, state_before):
        if self._segment is not None:
            self.anomalies.append('on_epoch_begin inside an epoch')
        mark = None
        if self._marks and self._marks[-1][1] is None:
            mark = self._marks[-1][0]
            self._marks[-1] = (mark, len(self.segments))
        m = None if ds._index_map is None else list(ds._index_map)
        self._segment = {'keras_epoch': mark, 'raw': [], 'mapped': [], 'index_map': m,
                         'begin_draws': random.getstate() != state_before,
                         'state_sha256_at_begin': py_state_sha256(state_before)}

    def _epoch_end_before(self, ds):
        seg = self._segment
        if seg is None:
            # reset() calls on_epoch_end with no begin when no iteration happened; record it as its
            # own empty segment so that every reshuffle is accounted for
            seg = self._segment = {'keras_epoch': None, 'raw': [], 'mapped': [], 'index_map': None,
                                   'begin_draws': False, 'state_sha256_at_begin': None, 'no_begin': True}
        m = None if ds._index_map is None else list(ds._index_map)
        seg['index_map_at_end'] = m
        seg['state_sha256_before_end'] = py_state_sha256()

    def _epoch_end_after(self, ds):
        seg = self._segment
        seg['index_map_after_end'] = None if ds._index_map is None else list(ds._index_map)
        seg['state_sha256_after_end'] = py_state_sha256()
        self.segments.append(seg)
        self._segment = None
        if self.keep_full is not None:
            for k in range(max(0, len(self.segments) - self.keep_full)):
                old = self.segments[k]
                if 'compact' not in old:
                    v = self._view(old)
                    for key in ('raw', 'mapped', 'index_map', 'index_map_after_end'):
                        v.pop(key, None)
                    self.segments[k] = {'keras_epoch': old['keras_epoch'], 'compact': v}

    # Keras epoch marks (EpochMarker or the frozen-evaluation callback)
    def mark_keras_epoch(self, epoch: int):
        if self._marks and self._marks[-1][1] is None:
            self.anomalies.append(f'Keras epoch {self._marks[-1][0]} had no Dataset epoch')
        self._marks.append((int(epoch), None))

    # views
    def epoch(self, e: int) -> dict:
        segs = [s for s in self.segments if s['keras_epoch'] == e]
        if len(segs) != 1:
            raise FlowError(f'Keras epoch {e} has {len(segs)} closed Dataset epochs')
        return self._view(segs[0])

    def prefit(self) -> list:
        return [self._view(s) for s in self.segments if s['keras_epoch'] is None]

    @staticmethod
    def _view(s) -> dict:
        if 'compact' in s:
            return dict(s['compact'])
        m = s['index_map']
        out = {'keras_epoch': s['keras_epoch'], 'n_fetches': len(s['mapped']), 'raw': list(s['raw']),
               'mapped': list(s['mapped']),
               'batch_index_seq_sha256': int_seq_sha256([x if x is not None else -1 for x in s['mapped']]),
               'index_map_sha256': None if m is None else int_seq_sha256(m),
               'index_map': m,
               'index_map_unchanged_during_epoch': m == s.get('index_map_at_end'),
               'index_map_after_end': s.get('index_map_after_end'),
               'on_epoch_begin_drew': bool(s['begin_draws']), 'no_begin': bool(s.get('no_begin', False)),
               'state_sha256_at_begin': s['state_sha256_at_begin'],
               'state_sha256_before_end': s.get('state_sha256_before_end'),
               'state_sha256_after_end': s.get('state_sha256_after_end')}
        if m is not None:
            out['mapped_is_permutation'] = sorted(x for x in s['mapped'] if x is not None) == sorted(m) and len(s['mapped']) == len(m)
            out['mapped_equals_map_of_raw'] = all(x == m[r] for r, x in zip(s['raw'], s['mapped']))
        return out

    def summary(self) -> dict:
        epochs = sorted({s['keras_epoch'] for s in self.segments if s['keras_epoch'] is not None})
        return {'dataset': self.dataset_info, 'anomalies': list(self.anomalies),
                'n_segments': len(self.segments), 'prefit': self.prefit(),
                'epochs': [self.epoch(e) for e in epochs], 'open_segment': self._segment is not None}


def instrument(ds, recorder: BatchIndexRecorder | None = None) -> BatchIndexRecorder:
    """Attach a recorder to an existing hgq Dataset (its class is switched to the recording
    subclass of its own class; its `_index_map` list is replaced by a recording copy)."""
    recorder = recorder or BatchIndexRecorder()
    recorder.attach(ds)
    return recorder


def epoch_marker(recorder: BatchIndexRecorder):
    """A Keras callback that marks each Keras epoch in the recorder at on_epoch_begin. In
    production the appended frozen-evaluation callback (M8) calls recorder.mark_keras_epoch
    itself, so that the callback order stays as STUDY section 7.1 states it."""
    import keras

    class EpochMarker(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            recorder.mark_keras_epoch(epoch)

    return EpochMarker()


# ----------------------------------------------------------------------------------------------
# the stream guard


HOOKS = ('on_train_begin', 'on_train_end', 'on_epoch_begin', 'on_epoch_end', 'on_train_batch_begin',
         'on_train_batch_end', 'on_batch_begin', 'on_batch_end', 'on_test_begin', 'on_test_end',
         'on_test_batch_begin', 'on_test_batch_end', 'on_predict_begin', 'on_predict_end',
         'on_predict_batch_begin', 'on_predict_batch_end')


class StreamGuard:
    """Section 6 step 6: lab code must not change the module-level `random` state (nor NumPy's
    legacy global state) between the reseed and the end of training."""

    def __init__(self, *, raise_on_violation=True):
        self.raise_on_violation = raise_on_violation
        self.armed = False
        self.violations = []
        self.calls = {}
        self.wrapped = []
        self.arm_record = None
        self.disarm_record = None

    def arm(self, label='after the pre-fit reseed'):
        self.armed = True
        self.arm_record = {'label': label, 'py_state_sha256': py_state_sha256(), 'np_state_sha256': np_state_sha256(),
                           't': time.time()}

    def disarm(self, label='after fit'):
        self.armed = False
        self.disarm_record = {'label': label, 'py_state_sha256': py_state_sha256(), 'np_state_sha256': np_state_sha256(),
                              't': time.time()}

    def _compare(self, who, owner, hook, py0, np0, may_raise=True):
        py_changed = random.getstate() != py0
        np_changed = not np_state_equals(np.random.get_state(legacy=True), np0)
        key = f'{who}.{hook}'
        c = self.calls.setdefault(key, {'owner': owner, 'calls': 0, 'py_changed': 0, 'np_changed': 0})
        c['calls'] += 1
        c['py_changed'] += int(py_changed)
        c['np_changed'] += int(np_changed)
        if py_changed or np_changed:
            v = {'who': who, 'owner': owner, 'hook': hook, 'python_random': py_changed, 'numpy_global': np_changed}
            self.violations.append(v)
            if owner == 'lab' and self.raise_on_violation and may_raise:
                raise StreamViolation(f'{who}.{hook} changed the '
                                      + ' and '.join(k for k, f in (('module-level random state', py_changed),
                                                                    ('NumPy legacy state', np_changed)) if f)
                                      + ' during training (STUDY section 6 step 6)')

    def wrap(self, callback, *, owner: str, name: str | None = None):
        """Wrap the overridden hooks of `callback` in place; owner is 'lab' or 'released'."""
        from keras.src.utils.python_utils import is_default

        if owner not in ('lab', 'released'):
            raise ValueError(owner)
        who = name or type(callback).__name__
        hooks = []
        for hook in HOOKS:
            meth = getattr(callback, hook, None)
            if meth is None or is_default(meth):
                continue

            def make(meth=meth, hook=hook):
                def wrapped(*args, **kwargs):
                    if not self.armed:
                        return meth(*args, **kwargs)
                    py0, np0 = random.getstate(), np.random.get_state(legacy=True)
                    try:
                        out = meth(*args, **kwargs)
                    except StreamViolation:
                        raise                       # already attributed to an inner watched block
                    except BaseException:
                        self._compare(who, owner, hook, py0, np0, may_raise=False)
                        raise
                    self._compare(who, owner, hook, py0, np0)
                    return out
                wrapped.__name__ = hook
                wrapped.__wrapped__ = meth
                return wrapped

            setattr(callback, hook, make())
            hooks.append(hook)
        self.wrapped.append({'who': who, 'owner': owner, 'class': f'{type(callback).__module__}.{type(callback).__qualname__}',
                             'hooks': hooks})
        return callback

    @contextlib.contextmanager
    def watch(self, label: str, owner: str = 'lab'):
        """Check a block of lab code (for example a frozen-evaluation step) the same way."""
        if not self.armed:
            yield
            return
        py0, np0 = random.getstate(), np.random.get_state(legacy=True)
        try:
            yield
        except StreamViolation:
            raise
        except BaseException:
            self._compare(label, owner, 'watch', py0, np0, may_raise=False)
            raise
        self._compare(label, owner, 'watch', py0, np0)

    def record(self) -> dict:
        return {'armed_at': self.arm_record, 'disarmed_at': self.disarm_record, 'wrapped': list(self.wrapped),
                'calls': dict(self.calls), 'violations': list(self.violations),
                'lab_violations': [v for v in self.violations if v['owner'] == 'lab'],
                'released_violations': [v for v in self.violations if v['owner'] == 'released']}


# ----------------------------------------------------------------------------------------------
# the seed flow


class SeedFlow:
    """Section 6 steps 1-5 for one arm and seed s, in order (module docstring, item 1)."""

    ORDER = ('load_data', 'seed_streams', 'build_datasets', 'build_model', 'load_initial_tensors', 'reseed_before_fit')

    def __init__(self, seed: int, *, dataset_device: str):
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise FlowError(f'seed must be a non-negative int, not {seed!r}')
        if dataset_device not in ('cpu:0', 'gpu:0'):
            raise FlowError(f"dataset_device must be 'cpu:0' or 'gpu:0' (decision U2), not {dataset_device!r}")
        self.seed = seed
        self.dataset_device = dataset_device
        self.done = []
        self.events = []
        self.tv = None
        self.ds_train = self.ds_val = None
        self.recorder = None
        self.model = None
        self.arm_id = None
        self.data_record = None
        self.datasets_record = None
        self.load_record = None
        self.reseed = None
        self.index_map_at_reseed = None

    def _step(self, name, **info):
        k = self.ORDER.index(name)
        if tuple(self.done) != self.ORDER[:k]:
            raise FlowError(f'step {name} called after {self.done}; section 6 order is {list(self.ORDER)}')
        self.done.append(name)
        ev = {'step': name, 'py_state_sha256': py_state_sha256(), 'np_state_sha256': np_state_sha256(), 't': time.time()}
        ev.update(info)
        self.events.append(ev)
        return ev

    def _require(self, name):
        if not (self.done and self.done[-1] == name):
            raise FlowError(f'{name} must be the last step done; done {self.done}; section 6 order is {list(self.ORDER)}')

    # step 1
    def load_data(self, train_archive, *, expect_sha256, data_manifest_sha256=None,
                  n_constituents=CH8_N_CONSTITUENTS, ptetaphi=CH8_PTETAPHI):
        if self.done:
            raise FlowError(f'load_data must be the first step; done {self.done}')
        from cbl import data

        np.random.seed(SPLIT_SEED)                                   # section 6 step 1; run_train.py:41
        tv = data.read_train_val(train_archive, n_constituents, ptetaphi, expect_sha256=expect_sha256,
                                 expect_seed=SPLIT_SEED)
        self.tv = tv
        r = tv.record
        self.data_record = {'train_archive_sha256': r['archive_sha256'], 'data_manifest_sha256': data_manifest_sha256,
                            'order_sha256': r['order_sha256'], 'split_sha256': r['split_sha256'],
                            'stats_sha256': r['stats_sha256'], 'n_train': r['n_train'], 'n_val': r['n_val'],
                            'n_constituents': r['n_constituents'], 'ptetaphi': r['ptetaphi']}
        self._step('load_data', archive=str(train_archive), **{k: v for k, v in self.data_record.items()})
        return tv

    # step 2
    def seed_streams(self):
        self._require('load_data')
        import keras

        random.seed(self.seed)                                       # section 6 step 2
        keras.utils.set_random_seed(self.seed)
        ok = random.getstate() == random.Random(self.seed).getstate()
        ok_np = np_state_equals(np.random.get_state(legacy=True), np.random.RandomState(self.seed).get_state(legacy=True))
        if not (ok and ok_np):
            raise FlowError(f'after seeding, random equals Random({self.seed}): {ok}; NumPy equals RandomState: {ok_np}')
        self._step('seed_streams', random_equals_seed=ok, numpy_equals_seed=ok_np)

    # step 3a
    def build_datasets(self, *, batch_size=TRAIN_BATCH, val_batch_size=VAL_BATCH, record=True):
        self._require('seed_streams')
        cls = recording_dataset_class() if record else None
        if cls is None:
            from hgq.utils.sugar import Dataset as cls
        tv = self.tv
        ds_train = cls(tv.X_train, tv.y_train, batch_size, self.dataset_device, shuffle=True)   # run_train.py:73
        ds_val = cls(tv.X_val, tv.y_val, val_batch_size, self.dataset_device)                    # run_train.py:74
        if record:
            self.recorder = BatchIndexRecorder()
            self.recorder.attach(ds_train)
        exp = reconstruct_datasets(self.seed, len(tv.y_train), len(tv.y_val), batch_size, val_batch_size)
        got_map = list(ds_train._index_map)
        checks = {
            'train_drop_last_false': ds_train.drop_last is False, 'train_shuffle': ds_train.shuffle is True,
            'val_shuffle_false': ds_val.shuffle is False, 'val_index_map_none': ds_val._index_map is None,
            'train_init_shift': ds_train._init_shift == exp['train']['init_shift'],
            'train_index_map': got_map == exp['train']['index_map'],
            'val_init_shift': ds_val._init_shift == exp['val']['init_shift'],
            'state_after_datasets': random.getstate() == exp['state'],
            'n_batches': len(ds_train) == exp['train']['n_batches'],
        }
        if not all(checks.values()):
            raise FlowError(f'Dataset construction draws differ from their reconstruction: '
                            f'{[k for k, v in checks.items() if not v]}')
        self.ds_train, self.ds_val = ds_train, ds_val
        self.datasets_record = {'batch_size': batch_size, 'val_batch_size': val_batch_size,
                                'n_batches': len(ds_train), 'val_n_batches': len(ds_val),
                                'train_remainder': exp['train']['remainder'], 'train_init_shift': ds_train._init_shift,
                                'val_init_shift': ds_val._init_shift,
                                'index_map_sha256_at_construction': int_seq_sha256(got_map),
                                'random_state_sha256_after_datasets': py_state_sha256()}
        self._step('build_datasets', checks=checks, **self.datasets_record)
        return ds_train, ds_val

    # step 3b
    def build_model(self, builder: Callable, *, arm_id: str):
        self._require('build_datasets')
        model = builder()
        self.model, self.arm_id = model, arm_id
        self._step('build_model', arm_id=arm_id, n_variables=len(model.variables),
                   stochastic_quantizers=stochastic_quantizers(model))
        return model

    def run_identity(self) -> dict:
        import jax

        if self.data_record is None or self.datasets_record is None:
            raise FlowError('run identity needs steps 1-3')
        return {'seed': self.seed, 'device': device_identity(self.dataset_device), 'stack': stack_versions(),
                'matmul_precision': str(jax.config.jax_default_matmul_precision), 'draw_code': draw_code_sha256(),
                'data': dict(self.data_record), 'datasets': dict(self.datasets_record)}

    # step 4
    def load_initial_tensors(self, manifest_path, *, expect_manifest_sha256, role_spec: RoleSpec,
                             check_own_draw=False):
        self._require('build_model')
        if role_spec.arm_id != self.arm_id:
            raise FlowError(f'role spec of {role_spec.arm_id} for arm {self.arm_id}')
        rec = load_initial_tensors(self.model, manifest_path, expect_manifest_sha256=expect_manifest_sha256,
                                   role_spec=role_spec, run_identity=self.run_identity(),
                                   check_own_draw=check_own_draw)
        self.load_record = rec
        self._step('load_initial_tensors', **{k: rec[k] for k in ('manifest_sha256', 'file_sha256', 'n_loaded',
                                                                  'unmatched', 'unused', 'loaded_sha256')})
        return rec

    # step 5
    def reseed_before_fit(self, *, expect_prefit_reseed=None, guard: StreamGuard | None = None):
        self._require('load_initial_tensors')
        r = prefit_reseed(self.seed)
        if expect_prefit_reseed is not None and expect_prefit_reseed != r:
            raise FlowError(f'config prefit_reseed {expect_prefit_reseed} differs from U5 value {r} for seed {self.seed}')
        random.seed(r)                                               # section 6 step 5
        if random.getstate() != random.Random(r).getstate():
            raise FlowError('module-level random is not in the state random.seed(prefit_reseed) leaves')
        self.reseed = r
        self.index_map_at_reseed = None if self.ds_train is None else list(self.ds_train._index_map)
        self._step('reseed_before_fit', prefit_reseed=r,
                   index_map_sha256_at_reseed=None if self.index_map_at_reseed is None else int_seq_sha256(self.index_map_at_reseed))
        if guard is not None:
            guard.arm()
        return r

    def record(self) -> dict:
        return {'seed': self.seed, 'arm_id': self.arm_id, 'dataset_device': self.dataset_device, 'steps': list(self.done),
                'events': list(self.events), 'data': self.data_record, 'datasets': self.datasets_record,
                'load': self.load_record, 'prefit_reseed': self.reseed,
                'index_map_at_reseed': self.index_map_at_reseed}


# ----------------------------------------------------------------------------------------------
# draw and load (decision U1)


def probe_forward_sha256(model) -> dict:
    """sha256 of the float32 logits on a fixed batch from a private generator, inference mode; checks
    that the call changes no variable and draws from neither global stream."""
    import keras

    shape = tuple(model.input_shape[1:])
    x = np.random.default_rng(PROBE_SEED).standard_normal((PROBE_N,) + shape).astype(np.float32)
    roles = {f'v{i}': v for i, v in enumerate(model.variables)}
    before = variables_sha256(roles)
    py0, np0 = random.getstate(), np.random.get_state(legacy=True)
    y = np.asarray(keras.ops.convert_to_numpy(model(x, training=False)), dtype=np.float32)
    rec = {'probe_seed': PROBE_SEED, 'shape': [PROBE_N, *shape], 'sha256': tensor_sha256(y),
           'variables_unchanged': variables_sha256(roles) == before, 'random_unchanged': random.getstate() == py0,
           'numpy_unchanged': np_state_equals(np.random.get_state(legacy=True), np0)}
    if not (rec['variables_unchanged'] and rec['random_unchanged'] and rec['numpy_unchanged']):
        raise FlowError(f'the forward probe changed state: {rec}')
    return rec


def draw_initial_tensors(flow: SeedFlow, out_dir, *, place: str, code_sha: str | None,
                         label='engineering, not quotable') -> tuple:
    """Save CH-8's initial tensors for flow.seed after steps 1-3 (decision U1). Writes
    init-s<seed>-<platform>.npz and its .manifest.json into out_dir; returns (manifest path,
    manifest sha256, manifest)."""
    flow._require('build_model')
    if flow.arm_id != 'CH-8':
        raise FlowError(f'the initial tensors of record are CH-8\'s; this flow built {flow.arm_id}')
    model = flow.model
    roles = roles_of(model, CH8_ROLE_SPEC)
    arrays = variable_arrays(roles)
    ident = flow.run_identity()
    stem = f'init-s{flow.seed}-{ident["device"]["jax_platform"]}'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f'{stem}.npz'
    mpath = out_dir / f'{stem}.manifest.json'
    if mpath.exists():
        raise FileExistsError(mpath)
    fsha = write_tensor_file(fpath, arrays)
    if read_tensor_file(fpath).keys() != arrays.keys():
        raise FlowError('written tensor file does not read back')
    probe = probe_forward_sha256(model)
    trainable = {id(v) for v in model.trainable_variables}
    man = {
        'kind': INIT_KIND, 'schema': INIT_SCHEMA, 'label': label, 'place': place,
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'code_sha': code_sha,
        'arm_drawn': 'CH-8', 'builder': 'chang/model.py get_model' + repr(CH8_BUILDER_ARGS),
        'flow': [e['step'] for e in flow.events],
        'identity': ident,
        'random_state_sha256_after_build': flow.events[-1]['py_state_sha256'],
        'stochastic_quantizers': stochastic_quantizers(model),
        'probe_forward': probe,
        'file': {'name': fpath.name, 'sha256': fsha, 'bytes': fpath.stat().st_size},
        'n_roles': len(arrays),
        'roles': {r: {'shape': list(a.shape), 'dtype': str(a.dtype), 'sha256': tensor_sha256(a),
                      'trainable': id(roles[r]) in trainable, 'path_at_draw': roles[r].path}
                  for r, a in arrays.items()},
        'role_order': list(arrays),
    }
    b = manifest_bytes(man)
    with open(mpath, 'xb') as f:
        f.write(b)
    return mpath, sha256_bytes(b), man


def load_initial_tensors(model, manifest_path, *, expect_manifest_sha256, role_spec: RoleSpec,
                         run_identity: dict, check_own_draw=False) -> dict:
    """Section 6 step 4 under decision U1. Every check runs before any variable is assigned."""
    py0, np0 = random.getstate(), np.random.get_state(legacy=True)
    man, msha = read_manifest(manifest_path, expect_manifest_sha256=expect_manifest_sha256)
    bad = identity_mismatches(man['identity'], run_identity)
    if bad:
        a, b = _flatten(man['identity']), _flatten(run_identity)
        raise InitTensorRefused('identity differs between the manifest and the run: '
                                + '; '.join(f'{k}: {a.get(k, "<absent>")!r} != {b.get(k, "<absent>")!r}' for k in bad[:8]))
    tensors, fsha = read_checked_tensors(manifest_path, man)
    roles = roles_of(model, role_spec)
    unmatched = sorted(set(roles) - set(tensors))
    unused = sorted(set(tensors) - set(roles))
    exp_unmatched = sorted(role_spec.expected_unmatched(model))
    exp_unused = sorted(role_spec.expected_unused(model))
    if unmatched != exp_unmatched or unused != exp_unused:
        raise InitTensorRefused(f'{role_spec.arm_id}: unmatched {unmatched[:6]} (declared {exp_unmatched[:6]}), '
                                f'unused {unused[:6]} (declared {exp_unused[:6]})')
    matched = [r for r in roles if r in tensors]
    import keras

    for r in matched:
        v, a = roles[r], tensors[r]
        if tuple(v.shape) != a.shape or keras.backend.standardize_dtype(v.dtype) != str(a.dtype):
            raise InitTensorRefused(f'{r}: variable {tuple(v.shape)} {v.dtype} against file {a.shape} {a.dtype}')
    own = None
    if check_own_draw:
        cur = variable_arrays({r: roles[r] for r in matched})
        diff = [r for r in matched if tensor_sha256(cur[r]) != man['roles'][r]['sha256']]
        own = {'equal_roles': len(matched) - len(diff), 'differing_roles': diff}
    for r in matched:
        roles[r].assign(tensors[r])
    after = variable_arrays({r: roles[r] for r in matched})
    wrong = [r for r in matched if tensor_sha256(after[r]) != man['roles'][r]['sha256']]
    if wrong:
        raise InitTensorRefused(f'after assignment {len(wrong)} variables differ from the file: {wrong[:5]}')
    unmatched_values = {r: tensor_sha256(a) for r, a in variable_arrays({r: roles[r] for r in unmatched}).items()}
    h = hashlib.sha256()
    for r in matched:
        h.update(r.encode() + b'\0' + man['roles'][r]['sha256'].encode())
    rec = {'manifest': str(manifest_path), 'manifest_sha256': msha, 'file_sha256': fsha, 'arm_id': role_spec.arm_id,
           'n_roles_file': len(tensors), 'n_roles_arm': len(roles), 'n_loaded': len(matched),
           'unmatched': unmatched, 'unused': unused, 'unmatched_values_sha256': unmatched_values,
           'loaded_sha256': h.hexdigest(), 'own_draw_before_load': own,
           'random_unchanged_by_load': random.getstate() == py0,
           'numpy_unchanged_by_load': np_state_equals(np.random.get_state(legacy=True), np0),
           'role_spec_pending': role_spec.pending}
    if not (rec['random_unchanged_by_load'] and rec['numpy_unchanged_by_load']):
        raise FlowError('loading the initial tensors drew from a global stream')
    return rec
