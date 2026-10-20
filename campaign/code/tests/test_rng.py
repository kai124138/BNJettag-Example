"""Tests of M4 `cbl/rng.py`. The pure parts run in this process (they import only the standard
library and NumPy). Scenarios that touch jax, keras or hgq run in their own subprocess of
CBL_TEST_PYTHON (the training venv's python) or of the interpreter running pytest, JAX on the CPU.
The gates of record are G10-G12; these tests are not a gate.
"""

import hashlib
import json
import os
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
PY = os.environ.get('CBL_TEST_PYTHON', sys.executable)

from cbl import rng  # noqa: E402

PRELUDE = f"""
import json, os, random, sys
sys.dont_write_bytecode = True
sys.path.insert(0, {str(FORK)!r})
def emit(obj):
    print('RESULT ' + json.dumps(obj, default=str), flush=True)
from cbl import env
env.setup(mem_fraction='0.5', jax_platforms='cpu')
import numpy as np
from cbl import rng
"""


def run(code, tmp_path=None):
    env = {k: v for k, v in os.environ.items() if k not in ('KERAS_BACKEND', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_MEM_FRACTION')}
    env['JAX_PLATFORMS'] = 'cpu'
    r = subprocess.run([PY, '-I', '-B', '-c', PRELUDE + textwrap.dedent(code)], capture_output=True, text=True,
                       env=env, cwd=str(tmp_path or FORK), timeout=600)
    results = [json.loads(line[len('RESULT '):]) for line in r.stdout.splitlines() if line.startswith('RESULT ')]
    assert r.returncode == 0, f'exit {r.returncode}\nstdout:\n{r.stdout[-3000:]}\nstderr:\n{r.stderr[-4000:]}'
    assert len(results) == 1, r.stdout
    return results[0]


# ---------------------------------------------------------------- pure


def test_prefit_reseed_formula():
    for s in range(1, 12):
        want = int.from_bytes(hashlib.sha256(b'cbl1016-fit\0' + str(s).encode()).digest()[:8], 'big')
        assert rng.prefit_reseed(s) == want
    assert len({rng.prefit_reseed(s) for s in range(1, 12)}) == 11
    for bad in (True, '1', 1.0):
        with pytest.raises(TypeError):
            rng.prefit_reseed(bad)


def test_int_seq_sha256_canonical():
    assert rng.int_seq_sha256([3, 0, 2]) == hashlib.sha256(np.array([3, 0, 2], dtype='<i8').tobytes()).hexdigest()
    assert rng.int_seq_sha256([3, 0, 2]) != rng.int_seq_sha256([0, 3, 2])


def test_tensor_file_roundtrip_deterministic_and_keeps_scalars(tmp_path):
    t = {'a/kernel': np.arange(6, dtype=np.float32).reshape(2, 3), 'a/beta': np.array(2e-8, dtype=np.float32),
         'a/k': np.ones((2, 3), np.uint8), 'a/ebops': np.array(0, np.uint32)}
    s1 = rng.write_tensor_file(tmp_path / 'x.npz', t)
    s2 = rng.write_tensor_file(tmp_path / 'y.npz', t)
    assert s1 == s2
    back = rng.read_tensor_file(tmp_path / 'x.npz')
    assert list(back) == list(t)
    for k, v in t.items():
        assert back[k].shape == v.shape and back[k].dtype == v.dtype and rng.tensor_sha256(back[k]) == rng.tensor_sha256(v)
    assert back['a/beta'].shape == ()
    with pytest.raises(FileExistsError):
        rng.write_tensor_file(tmp_path / 'x.npz', t)


def test_dataset_construction_reconstruction_is_a_shuffle_after_randint():
    r = random.Random(5)
    shift = r.randint(0, 18000 % 2790)
    m = list(range(7))
    r.shuffle(m)
    vshift = r.randint(0, 2000)
    got = rng.reconstruct_datasets(5, 18000, 2000)
    assert got['train']['n_batches'] == 7 and got['train']['remainder'] == 1260
    assert got['train']['init_shift'] == shift and got['train']['index_map'] == m
    assert got['val']['init_shift'] == vshift and got['val']['index_map'] is None
    assert got['state'] == r.getstate()


def test_reconstruct_fit_stream_chain_and_state_start():
    m0 = rng.reconstruct_datasets(1, 18000, 2000)['train']['index_map']
    a = rng.reconstruct_fit_stream(reseed=rng.prefit_reseed(1), index_map=m0, n_batches=7, epochs=4)
    b = rng.reconstruct_fit_stream(reseed=None, state=random.Random(rng.prefit_reseed(1)).getstate(), index_map=m0,
                                   n_batches=7, epochs=4)
    assert a['epochs'] == b['epochs'] and a['end_state'] == b['end_state']
    assert len(a['prefit']['mapped']) == 2
    assert a['prefit']['state_sha256_at_end'] == a['epochs'][0]['state_sha256_at_begin']
    for e0, e1 in zip(a['epochs'], a['epochs'][1:]):
        assert e0['state_sha256_at_end'] == e1['state_sha256_at_begin']
        assert e0['index_map_after'] == e1['index_map']
    for e in a['epochs']:
        assert sorted(e['mapped']) == list(range(7)) and e['mapped'] == [e['index_map'][p] for p in e['order']]
    c = rng.reconstruct_fit_stream(reseed=rng.prefit_reseed(2), index_map=m0, n_batches=7, epochs=4)
    assert [e['batch_index_seq_sha256'] for e in c['epochs']] != [e['batch_index_seq_sha256'] for e in a['epochs']]
    with pytest.raises(ValueError):
        rng.reconstruct_fit_stream(reseed=1, state=random.getstate(), index_map=m0, n_batches=7, epochs=1)
    with pytest.raises(ValueError):
        rng.reconstruct_fit_stream(reseed=1, index_map=[0, 0, 1], n_batches=3, epochs=1)


def test_segment_name_rules():
    counts = {}
    assert rng.segment_name('q_add_1', 'multiple_quantizers_1', 'MultipleQuantizers', counts) == 'MultipleQuantizers[0]'
    assert rng.segment_name('x', 'quantizer_2', 'Quantizer', counts) == 'Quantizer[0]'
    assert rng.segment_name('x', 'quantizer_3', 'Quantizer', counts) == 'Quantizer[1]'
    assert rng.segment_name('q_einsum_dense_batchnorm_2', 'q_einsum_dense_batchnorm_2_kq', 'Quantizer', {}) == 'kq'
    assert rng.segment_name('q_softmax', 'q_softmax_exp_table', 'QUnaryFunctionLUT', {}) == 'exp_table'
    assert rng.segment_name('q_multi_head_attention', 'query', 'QEinsumDense', {}) == 'query'
    assert rng.segment_name('p', 'fixed_point_quantizer_kbi', 'FixedPointQuantizerKBI', {}) == 'FixedPointQuantizerKBI[0]'


def test_identity_mismatches():
    a = {'seed': 1, 'device': {'jax_platform': 'cpu', 'device_kind': ['cpu']}, 'data': {'n_train': 18000}}
    assert rng.identity_mismatches(a, json.loads(json.dumps(a))) == []
    b = json.loads(json.dumps(a))
    b['device']['jax_platform'] = 'gpu'
    b['extra'] = 1
    assert rng.identity_mismatches(a, b) == ['device.jax_platform', 'extra']


def test_read_manifest_refusals(tmp_path):
    p = tmp_path / 'm.json'
    p.write_bytes(rng.manifest_bytes({'kind': rng.INIT_KIND, 'schema': rng.INIT_SCHEMA}))
    sha = rng.sha256_file(p)
    for bad in (None, '', 'abc', '0' * 64):
        with pytest.raises(rng.InitTensorRefused):
            rng.read_manifest(p, expect_manifest_sha256=bad)
    man, got = rng.read_manifest(p, expect_manifest_sha256=sha)
    assert got == sha and man['kind'] == rng.INIT_KIND
    q = tmp_path / 'n.json'
    q.write_bytes(rng.manifest_bytes({'kind': 'other', 'schema': 1}))
    with pytest.raises(rng.InitTensorRefused):
        rng.read_manifest(q, expect_manifest_sha256=rng.sha256_file(q))


def test_probe_list_ignores_reads_outside_a_fetch():
    class Owner:
        _cbl_fetching = False
        _cbl_lookups = None

    o = Owner()
    p = rng._ProbeList([4, 1, 3, 0, 2])
    p._cbl_owner = o
    o._cbl_lookups = []
    random.Random(1).shuffle(p)
    assert o._cbl_lookups == []
    o._cbl_fetching = True
    v = p[2]
    assert o._cbl_lookups == [v]


# ---------------------------------------------------------------- with hgq and keras (subprocess)


@pytest.mark.pc_gpu
def test_construction_draws_equal_reconstruction_on_hgq_dataset(tmp_path):
    out = run("""
        from hgq.utils.sugar import Dataset
        random.seed(3)
        x = np.zeros((18000, 8, 3), np.float32); y = np.zeros(18000, np.int32)
        xv = np.zeros((2000, 8, 3), np.float32); yv = np.zeros(2000, np.int32)
        dt = Dataset(x, y, 2790, 'cpu:0', shuffle=True)
        dv = Dataset(xv, yv, 2790, 'cpu:0')
        exp = rng.reconstruct_datasets(3, 18000, 2000)
        emit({'shift': dt._init_shift == exp['train']['init_shift'], 'map': list(dt._index_map) == exp['train']['index_map'],
              'vshift': dv._init_shift == exp['val']['init_shift'], 'state': random.getstate() == exp['state'],
              'drop_last': dt.drop_last, 'vmap': dv._index_map})
    """, tmp_path)
    assert out == {'shift': True, 'map': True, 'vshift': True, 'state': True, 'drop_last': False, 'vmap': None}


@pytest.mark.pc_gpu
def test_recorder_records_mapped_index_and_not_shuffle_reads(tmp_path):
    out = run("""
        x = np.arange(18000 * 2, dtype=np.float32).reshape(18000, 2, 1); y = np.zeros(18000, np.int32)
        random.seed(4)
        cls = rng.recording_dataset_class()
        ds = cls(x, y, 2790, 'cpu:0', shuffle=True)
        rec = rng.BatchIndexRecorder(); rec.attach(ds)
        rec.mark_keras_epoch(0)
        ds.on_epoch_begin()
        order = [5, 0, 3, 6, 1, 2, 4]
        firsts = []
        for i in order:
            bx, _ = ds[i]
            firsts.append(int(np.asarray(bx)[0, 0, 0]) // (2 * 2790))
        m = list(ds._index_map)
        before = random.getstate()
        ds.on_epoch_end()
        r = random.Random(); r.setstate(before); m2 = list(m); r.shuffle(m2)
        e = rec.epoch(0)
        emit({'mapped': e['mapped'] == [m[i] for i in order], 'batches': firsts == [m[i] for i in order],
              'n': e['n_fetches'], 'anomalies': rec.anomalies, 'reshuffle': list(ds._index_map) == m2,
              'state': random.getstate() == r.getstate(), 'begin_drew': e['on_epoch_begin_drew'],
              'perm': e['mapped_is_permutation'], 'cls': type(ds).__name__,
              'hash': e['batch_index_seq_sha256'] == rng.int_seq_sha256([m[i] for i in order])})
    """, tmp_path)
    assert out == {'mapped': True, 'batches': True, 'n': 7, 'anomalies': [], 'reshuffle': True, 'state': True,
                   'begin_drew': False, 'perm': True, 'cls': 'CblRecordingDataset', 'hash': True}


@pytest.mark.pc_gpu
def test_recorder_keep_full_compacts_old_epochs(tmp_path):
    out = run("""
        random.seed(8)
        cls = rng.recording_dataset_class()
        ds = cls(np.zeros((20, 2), np.float32), np.zeros(20, np.int32), 4, 'cpu:0', shuffle=True)
        rec = rng.BatchIndexRecorder(keep_full=1); rec.attach(ds)
        hashes = []
        for e in range(3):
            rec.mark_keras_epoch(e); ds.on_epoch_begin()
            m = list(ds._index_map)
            for i in range(5):
                ds[i]
            hashes.append(rng.int_seq_sha256(m))
            ds.on_epoch_end()
        v0, v2 = rec.epoch(0), rec.epoch(2)
        emit({'old_has_lists': 'mapped' in v0, 'new_has_lists': 'mapped' in v2,
              'hashes': [rec.epoch(e)['batch_index_seq_sha256'] == hashes[e] for e in range(3)],
              'checks': [rec.epoch(e)['mapped_is_permutation'] for e in range(3)]})
    """, tmp_path)
    assert out == {'old_has_lists': False, 'new_has_lists': True, 'hashes': [True, True, True], 'checks': [True, True, True]}


@pytest.mark.pc_gpu
def test_instrument_existing_dataset(tmp_path):
    out = run("""
        from hgq.utils.sugar import Dataset
        random.seed(6)
        ds = Dataset(np.zeros((30, 2), np.float32), np.zeros(30, np.int32), 7, 'cpu:0', shuffle=True)
        m = list(ds._index_map)
        rec = rng.instrument(ds)
        ds.on_epoch_begin(); _ = ds[1]; _ = ds[-1]; ds.on_epoch_end()
        p = rec.prefit()
        emit({'cls': type(ds).__name__, 'base': isinstance(ds, Dataset), 'mapped': p[0]['mapped'], 'want': [m[1], m[4]],
              'anoms': rec.anomalies})
    """, tmp_path)
    assert out['cls'] == 'CblRecordingDataset' and out['base'] is True
    assert out['mapped'] == out['want'] and out['anoms'] == []


@pytest.mark.pc_gpu
def test_guard_wraps_only_overridden_hooks_and_detects(tmp_path):
    out = run("""
        import keras
        from keras.src.utils.python_utils import is_default
        class Bad(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                random.random()
        class Np(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                np.random.rand()
        g = rng.StreamGuard(raise_on_violation=True)
        b, n, rel = Bad(), Np(), Bad()
        g.wrap(b, owner='lab'); g.wrap(n, owner='lab'); g.wrap(rel, owner='released', name='Released')
        o = {'default_kept': is_default(b.on_train_batch_end), 'wrapped': [w['hooks'] for w in g.wrapped]}
        b.on_epoch_end(0)                     # not armed: no check
        o['unarmed'] = len(g.violations)
        g.arm()
        try:
            b.on_epoch_end(0); o['raised'] = False
        except rng.StreamViolation:
            o['raised'] = True
        try:
            n.on_epoch_begin(0); o['np_raised'] = False
        except rng.StreamViolation as e:
            o['np_raised'] = 'NumPy' in str(e)
        rel.on_epoch_end(0)                   # released: recorded, not raised
        try:
            with g.watch('stand-in'):
                random.random()
            o['watch_raised'] = False
        except rng.StreamViolation:
            o['watch_raised'] = True
        with g.watch('clean'):
            pass
        g.disarm()
        r = g.record()
        o['lab'] = [(v['who'], v['hook'], v['python_random'], v['numpy_global']) for v in r['lab_violations']]
        o['released'] = [(v['who'], v['hook']) for v in r['released_violations']]
        emit(o)
    """, tmp_path)
    assert out['default_kept'] is True
    assert out['wrapped'] == [['on_epoch_end'], ['on_epoch_begin'], ['on_epoch_end']]
    assert out['unarmed'] == 0 and out['raised'] and out['np_raised'] and out['watch_raised']
    assert out['lab'] == [['Bad', 'on_epoch_end', True, False], ['Np', 'on_epoch_begin', False, True],
                          ['stand-in', 'watch', True, False]]
    assert out['released'] == [['Released', 'on_epoch_end']]


@pytest.mark.parametrize('n_batches', [8, 7])
@pytest.mark.pc_gpu
def test_fit_stream_matches_reconstruction_multiple_and_not_of_steps_per_execution(tmp_path, n_batches):
    """Keras 3.14.0 JAX fit with steps_per_execution=4 on a tiny model and synthetic arrays: 8 batches
    (a multiple of 4, as the archive of record's 200) and 7 (as the smoke archive). The recorded fetches,
    maps and stream states equal reconstruct_fit_stream, and the final stream state equals its end state."""
    out = run(f"""
        import keras
        n_b, bs = {n_batches}, 10
        x = np.random.default_rng(0).standard_normal((n_b * bs - 3, 4)).astype(np.float32)
        y = (np.arange(len(x)) % 3).astype(np.int32)
        random.seed(9)
        cls = rng.recording_dataset_class()
        dt = cls(x, y, bs, 'cpu:0', shuffle=True)
        dv = cls(x[:25], y[:25], bs, 'cpu:0')
        rec = rng.BatchIndexRecorder(); rec.attach(dt)
        from cbl import lossguard
        model = keras.Sequential([keras.Input((4,)), keras.layers.Dense(3)])
        lossguard.guard(model)
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'], steps_per_execution=4)
        r = rng.prefit_reseed(1)
        random.seed(r)
        m0 = list(dt._index_map)
        model.fit(dt, epochs=3, validation_data=dv, callbacks=[rng.epoch_marker(rec)], verbose=0)
        exp = rng.reconstruct_fit_stream(reseed=r, index_map=m0, n_batches=len(dt), epochs=3)
        s = rec.summary()
        p, q = s['prefit'][0], exp['prefit']
        ok_pre = (len(s['prefit']) == 1 and p['raw'] == q['order'] and p['mapped'] == q['mapped']
                  and p['state_sha256_after_end'] == q['state_sha256_at_end'])
        ok_ep = [e['raw'] == x['order'] and e['mapped'] == x['mapped'] and e['index_map'] == x['index_map']
                 and e['state_sha256_at_begin'] == x['state_sha256_at_begin']
                 and e['state_sha256_after_end'] == x['state_sha256_at_end'] for e, x in zip(s['epochs'], exp['epochs'])]
        emit({{'n': len(dt), 'pre': ok_pre, 'epochs': ok_ep, 'n_epochs': len(s['epochs']),
              'final': rng.py_state_sha256() == exp['end_state_sha256'], 'anoms': s['anomalies']}})
    """, tmp_path)
    assert out == {'n': n_batches, 'pre': True, 'epochs': [True, True, True], 'n_epochs': 3, 'final': True, 'anoms': []}


@pytest.mark.pc_gpu
def test_flow_refuses_out_of_order(tmp_path):
    out = run("""
        o = {}
        for name, fn in (('bad_device', lambda: rng.SeedFlow(1, dataset_device='gpu')),
                         ('bad_seed', lambda: rng.SeedFlow(True, dataset_device='cpu:0'))):
            try:
                fn(); o[name] = False
            except rng.FlowError:
                o[name] = True
        f = rng.SeedFlow(1, dataset_device='cpu:0')
        for name in ('seed_streams', 'build_datasets', 'reseed_before_fit'):
            try:
                getattr(f, name)(); o[name] = False
            except rng.FlowError:
                o[name] = True
        try:
            f.build_model(lambda: None, arm_id='CH-8'); o['build_model'] = False
        except rng.FlowError:
            o['build_model'] = True
        emit(o)
    """, tmp_path)
    assert all(out.values()), out
