"""Tests of `cbl/lossguard.py` (finding B10-F1). Keras runs in a subprocess (its import state is per process, as in
tests/test_env.py); the interpreter is CBL_TEST_PYTHON if set (the training venv's python), otherwise the one
running pytest. JAX runs on the CPU.

The bug needs CPython to hand a new loss tensor the address of a dead tracer from an earlier trace that a later
layer still lists in `_loss_ids`. That happens at random in a real process, so the scenario replaces the `id`
that keras/src/layers/layer.py calls with a deterministic stand-in for address reuse: inside a collecting
StatelessScope the k-th distinct tensor gets id BASE + k + (number of earlier scopes), so each new trace hands
layer j's penalty the id that layer j+1's penalty had in the trace before. Nothing here is quotable.
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

SCENARIO = f"""
import json, sys
sys.dont_write_bytecode = True
sys.path.insert(0, {str(FORK)!r})
from cbl import env
env.set_process_env(mem_fraction='0.5', jax_platforms='cpu')
try:
    import jax, keras
except ImportError as e:
    print('RESULT ' + json.dumps({{'skip': str(e)}})); sys.exit(0)
import numpy as np
from keras import ops
from keras.src import backend
from keras.src.layers import layer as L
from cbl import lossguard

class Pen(keras.layers.Layer):
    enable_ebops = True                      # counted by lossguard.expected_terms, as an hgq layer with beta
    def __init__(self, c, **kw):
        super().__init__(**kw)
        self.c = c
    def build(self, shape):
        self.w = self.add_weight(shape=(shape[-1], shape[-1]), initializer='ones', name='w')
        self._beta = self.add_weight(shape=(), initializer='zeros', trainable=False, name='beta')
    def call(self, x, training=None):
        y = ops.matmul(x, self.w)
        if training:
            self.add_loss(ops.sum(self.w) * 0.0 + self.c)
        return y

def make():
    inp = keras.Input((3,))
    x = Pen(1.0, name='a')(inp)
    x = Pen(10.0, name='b')(x)
    x = Pen(100.0, name='c')(x)
    return keras.Model(inp, x)

REAL_ID, BASE = id, 10 ** 9
ST = {{'scope': None, 'n': -1, 'map': {{}}}}
def fake_id(obj):
    if backend.in_stateless_scope() and backend.is_tensor(obj):
        sc = backend.get_stateless_scope()
        if sc is not ST['scope']:
            ST['scope'], ST['n'], ST['map'] = sc, ST['n'] + 1, {{}}
        return ST['map'].setdefault(REAL_ID(obj), BASE + len(ST['map']) + ST['n'])
    return REAL_ID(obj)

def traces(model, sizes=(2, 5)):
    tv = [v.value for v in model.trainable_variables]
    ntv = [v.value for v in model.non_trainable_variables]
    f = jax.jit(lambda tv, ntv, x: model.stateless_call(tv, ntv, x, training=True, return_losses=True)[2])
    out = []
    for n in sizes:                                   # a new batch size retraces, as the last partial batch does
        losses = f(tv, ntv, np.ones((n, 3), np.float32))
        out.append([len(losses), float(sum(np.asarray(l) for l in losses))])
    return out

res = {{}}
L.id = fake_id
ST.update(scope=None, n=-1, map={{}})
res['unguarded'] = traces(make())
ST.update(scope=None, n=-1, map={{}})
m = make()
res['record'] = lossguard.guard(m)
res['guarded'] = traces(m)
res['counters'] = lossguard.counters(m)
res['idempotent'] = lossguard.guard(m) == res['record']
L.id = REAL_ID
# the census: a collision the guard cannot see (injected inside the trace) must stop the call
m2 = make()
lossguard.guard(m2)
b = m2.get_layer('b')
orig_clear = b._clear_losses
def colliding_clear():
    sc = backend.get_stateless_scope()
    for x in sc.losses:
        b._loss_ids.add(REAL_ID(x))
    return orig_clear()
object.__setattr__(b, '_clear_losses', colliding_clear)
try:
    traces(m2, sizes=(2,))
    res['census'] = 'no error'
except lossguard.LossTermsMismatch as e:
    res['census'] = 'LossTermsMismatch: ' + str(e)
print('RESULT ' + json.dumps(res))
"""


@pytest.fixture(scope='module')
def result(tmp_path_factory):
    env = {k: v for k, v in os.environ.items() if k not in ('KERAS_BACKEND', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_MEM_FRACTION')}
    env['JAX_PLATFORMS'] = 'cpu'
    r = subprocess.run([PY, '-I', '-B', '-c', SCENARIO], capture_output=True, text=True, env=env,
                       cwd=str(tmp_path_factory.mktemp('lossguard')), timeout=600)
    assert r.returncode == 0, f'exit {r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr[-4000:]}'
    out = [json.loads(x[len('RESULT '):]) for x in r.stdout.splitlines() if x.startswith('RESULT ')]
    assert len(out) == 1, r.stdout
    if 'skip' in out[0]:
        pytest.skip(f'keras/jax not importable: {out[0]["skip"]}')
    return out[0]


def test_the_stand_in_reproduces_the_drop_without_the_guard(result):
    first, second = result['unguarded']
    assert first == [3, 111.0]                       # the first trace collects all three penalties
    assert second[0] < 3 and second[1] < 111.0       # the retrace loses penalties to stale ids (B10-F1)


def test_the_guard_keeps_every_penalty_in_every_trace(result):
    assert result['guarded'] == [[3, 111.0], [3, 111.0]]
    assert result['record']['expected_training_terms'] == 3
    assert result['counters']['checked'] == 2 and result['counters']['cleared_ids'] > 0
    assert result['idempotent']


def test_the_census_stops_a_training_call_with_a_missing_term(result):
    assert result['census'].startswith('LossTermsMismatch: a training call returned 2 loss terms')


@pytest.mark.pc_gpu
@pytest.mark.parametrize('builder', ['binary_lab', 'binary_ch8', 'cost_lab', 'cost_ch8',
                                    'b8_lab', 'b8_ch8', 'probe'])
def test_gate_models_keep_penalties_across_training_retraces(tmp_path, builder):
    """Exercise the actual child builders, including cost's shared binary builders and the release Probe.

    The deterministic reused-id injection makes this fail reliably if a guard is removed.
    """
    setup = SCENARIO.split('res = {}')[0].replace(
        "env.set_process_env(mem_fraction='0.5', jax_platforms='cpu')",
        "env.setup(mem_fraction='0.5', jax_platforms='cpu')")
    setup = setup.replace('np.ones((n, 3), np.float32)',
                          'np.ones((n,) + tuple(model.input_shape[1:]), np.float32)')
    scenario = setup + textwrap.dedent(f"""
        sys.path.insert(0, {str(FORK / 'gates')!r})
        import binary_child as bc, cost_child as cc, b8_child as b8, runner_child as rc
        import model as chang_model
        from cbl import rng
        from types import SimpleNamespace
        builder = {builder!r}
        cfg = ('F111-FB', 'F111', dict(attention='B', ffn='B', head='B'), 'Q', 'FB', None)
        if builder == 'binary_lab':
            _, m = bc.build_arm(*cfg)
        elif builder == 'binary_ch8':
            m = bc.build_ch8()
        elif builder == 'cost_lab':
            m, _, _ = cc.build(cfg)
        elif builder == 'cost_ch8':
            m, _, _ = cc.build(('CH-8', 'CH-8', None, None, None, None))
        elif builder.startswith('b8_'):
            m, _ = b8.build('CH-8' if builder == 'b8_ch8' else 'F111-FB')
        else:
            m = chang_model.get_model(*rng.CH8_BUILDER_ARGS)
            ds = SimpleNamespace(x=np.ones((5, 8, 3), np.float32), y=np.zeros(5, np.int32))
            rc.Probe(m, rng.roles_of(m, rng.CH8_ROLE_SPEC), ds, ds)
        assert lossguard.counters(m) is not None, builder
        expected = lossguard.expected_terms(m, True)
        assert expected > 0
        L.id = fake_id
        try:
            results = traces(m)
        finally:
            L.id = REAL_ID
        assert [r[0] for r in results] == [expected, expected], results
        assert lossguard.counters(m)['checked'] == 2
        assert lossguard.counters(m)['cleared_ids'] > 0
        print('RESULT ' + json.dumps(dict(builder=builder, expected=expected, traces=results)))
    """)
    r = subprocess.run([PY, '-I', '-B', '-c', scenario], capture_output=True, text=True,
                       cwd=tmp_path, timeout=600)
    assert r.returncode == 0, f'{builder}: exit {r.returncode}\n{r.stdout}\n{r.stderr}'
