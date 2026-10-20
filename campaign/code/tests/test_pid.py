"""Tests of the pure parts of M12 `cbl/pid.py` and of the runner's acceptance of PID configs (no Keras, JAX or hgq).

The controller itself (the stock law on frozen readings, the log keys, the persisted state) is gated on the PC by
G43-G47 through the runner. These tests check the record, the max_beta string, the refusals, the names the runner,
the generator and M8 share with M12, and validate_config on PID configs. Nothing here is quotable.
"""

import copy
import json
import math
import sys
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

from cbl import pid, runner  # noqa: E402

MANIFEST = FORK / 'manifests' / 'data-prep-150c.json'


def pid_config(**over):
    import runner_child

    from gen import configs as GC
    from gen import release as GR
    cfg = runner_child.engineering_config(arm='CH-8', seed=1, epochs=14, code_sha='a' * 40,
                                          data_manifest_sha256=runner.sha256_file(MANIFEST),
                                          init_manifest_path='/nowhere/init.manifest.json', init_manifest_sha256='b' * 64)
    cfg.update(arm_id='PID-CH', schedule='pid', pid=dict(pid.PID_RECORD), recipe=GC._recipe(GR.release_constants()),
               headline='H1')
    cfg.pop('ramp')
    cfg.update(over)
    return cfg


def test_import_is_standard_library_only():
    import subprocess
    code = ('import sys; sys.path.insert(0, %r); from cbl import pid; '
            'print(sorted(m for m in sys.modules if m.split(".")[0] in ("keras", "jax", "hgq", "tensorflow", "numpy")))'
            % str(FORK))
    out = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == '[]'


def test_record_equals_the_generators():
    from gen import configs as GC
    assert GC.PID == pid.PID_RECORD
    assert json.loads(json.dumps(pid.PID_RECORD)) == pid.PID_RECORD


def test_names_shared_with_runner_generator_and_m8():
    from cbl import frozen_eval
    assert runner.PID_CALLBACK == pid.CALLBACK_NAME == 'LabBetaPID'
    assert runner.PID_ORDER[2] == pid.CALLBACK_NAME and runner.RELEASED_ORDER[2] == 'BetaScheduler'
    assert runner.PID_ORDER[:2] + runner.PID_ORDER[3:] == runner.RELEASED_ORDER[:2] + runner.RELEASED_ORDER[3:]
    assert frozen_eval.PID_LOG_PREFIX == pid.LOG_PREFIX
    assert pid.SENSED_KEY == 'pid/sensed_H' and pid.STOCK_KEY == 'pid/stock_aggregate'
    from gen import configs as GC
    from gen import release as GR
    from gen import arms as A
    spec = next(s for s in A.screen_specs('hold') if s.arm_id == 'PID-CH')
    assert GC._callbacks(spec, GR.release_constants())[2] == pid.CONFIG_NAME


def test_max_beta_string():
    assert pid.parse_max_beta('inf') == math.inf
    assert pid.parse_max_beta(5.0) == 5.0
    for bad in ('Infinity', 'INF', None, math.inf, math.nan, 0, -1.0, True):
        with pytest.raises(pid.PIDConfigError):
            pid.parse_max_beta(bad)


def test_kwargs_and_sources():
    kw = pid.check_record(dict(pid.PID_RECORD))
    assert kw == {'target_ebops': 350000, 'init_beta': 2e-08, 'p': 1.0, 'i': 0.002, 'd': 0.0, 'warmup': 10, 'log': True,
                  'max_beta': math.inf, 'min_beta': 0.0, 'damp_beta_on_target': 0.0}
    assert set(pid.SOURCES) == set(pid.PID_RECORD) == set(pid.KWARG_OF)


@pytest.mark.parametrize('over,needle', [
    ({'init_beta': 0.0}, 'init_beta must be > 0'),
    ({'init_beta': -2e-8}, 'init_beta must be > 0'),
    ({'warmup': 0}, 'warmup must be an int >= 1'),
    ({'warmup': True}, 'warmup must be an int'),
    ({'i': 0.0}, 'i must be nonzero'),
    ({'max_beta': 'Infinity'}, 'max_beta'),
    ({'max_beta': None}, 'max_beta'),
    ({'min_beta': -1.0}, 'min_beta'),
    ({'damping': 1.0}, 'damping'),
    ({'log': 1}, 'log must be true or false'),
    ({'target_H': 350000.0}, 'target_H must be a positive int'),
    ({'target_H': 400000}, 'differs from the record'),
    ({'p': 1}, 'differs from the record'),
    ({'max_beta': 10.0}, 'differs from the record'),
    ({'extra': 1}, 'exactly the keys'),
])
def test_record_refusals(over, needle):
    with pytest.raises(pid.PIDConfigError, match=needle):
        pid.check_record(dict(pid.PID_RECORD, **over))


def test_missing_key_refused():
    rec = dict(pid.PID_RECORD)
    rec.pop('damping')
    with pytest.raises(pid.PIDConfigError, match='exactly the keys'):
        pid.check_record(rec)


def test_sensor_reader_reads_the_headline_of_the_cached_evaluation():
    class Ev:
        calls = []

        def headline_value(self, epoch, headline):
            self.calls.append((epoch, headline))
            return {'H1': 11, 'H2': 7}[headline] + epoch
    ev = Ev()
    r = pid.sensor_reader(ev, 'H2')
    assert r(3) == 10 and r.units == 'H2' and ev.calls == [(3, 'H2')]
    for bad in (None, 'NR', 'H3'):
        with pytest.raises(pid.PIDConfigError):
            pid.sensor_reader(ev, bad)


def test_runner_accepts_a_pid_engineering_config():
    s = runner.validate_config(pid_config(), data_manifest_path=MANIFEST)
    assert s['arm_id'] == 'PID-CH' and s['schedule'] == 'pid' and s['builder'] == 'released'
    best = pid_config(arm_id='PID-BEST', builder='lab', config_code='F111', form='FB', form_source='fixed',
                      groups={'attention': 'B', 'ffn': 'B', 'head': 'B'}, embedding='Q', headline='H2')
    assert runner.validate_config(best)['builder'] == 'lab'


@pytest.mark.parametrize('over,needle', [
    ({'headline': None}, 'needs the headline'),
    ({'pid': dict(pid.PID_RECORD, init_beta=0.0)}, 'init_beta must be > 0'),
    ({'pid': dict(pid.PID_RECORD, max_beta=math.inf)}, 'max_beta'),
    ({'pid': None}, 'exactly the keys'),
    ({'ramp': runner.RAMP_RECORD}, 'has no ramp'),
    ({'recipe': None}, 'recipe must state'),
    ({'schedule': 'ramp'}, 'PID arms and only they'),
    ({'arm_id': 'CH-8'}, 'PID arms and only they'),
])
def test_runner_refuses_pid_variants(over, needle):
    with pytest.raises(runner.ConfigRefused, match=needle):
        runner.validate_config(pid_config(**over))


def test_runner_refuses_a_changed_recipe():
    cfg = pid_config()
    cfg['recipe'] = copy.deepcopy(cfg['recipe'])
    cfg['recipe']['learning_rate'] = 1e-3
    with pytest.raises(runner.ConfigRefused, match='recipe must state'):
        runner.validate_config(cfg)


def test_a_ramp_config_with_a_pid_field_is_refused():
    import runner_child
    cfg = runner_child.engineering_config(arm='CH-8', seed=1, epochs=3, code_sha='a' * 40,
                                          data_manifest_sha256=runner.sha256_file(MANIFEST),
                                          init_manifest_path='/x', init_manifest_sha256='b' * 64)
    cfg['pid'] = dict(pid.PID_RECORD)
    with pytest.raises(runner.ConfigRefused, match='has no pid field'):
        runner.validate_config(cfg)
