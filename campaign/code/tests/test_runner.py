"""Tests of the pure parts of M13 `cbl/runner.py` and M5 `cbl/model.py` (no Keras, JAX or hgq).

The runner restates Chang's literal arguments of `run_train.py:79-104`, which live inside
`__main__` and cannot be imported. These tests read them from the release's syntax tree and compare;
gate G13 compares the live objects on the PC. They also exercise the config refusals and M5's arm
specifications, including the binary arms' (B6). Nothing here is quotable.
"""

import ast
import copy
import inspect
import sys
import textwrap
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

from cbl import model as m  # noqa: E402
from cbl import rng, runner  # noqa: E402

RUN_TRAIN = FORK / 'chang' / 'run_train.py'


def main_block():
    tree = ast.parse(RUN_TRAIN.read_text(encoding='utf-8'))
    return next(n for n in tree.body if isinstance(n, ast.If))


def assigned_call(name):
    for st in main_block().body:
        if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name) and st.targets[0].id == name:
            return st.value
    raise KeyError(name)


def kw(call, key):
    return next(k.value for k in call.keywords if k.arg == key)


def lambda_body(fn):
    src = textwrap.dedent(inspect.getsource(fn))
    ret = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Return))
    return ast.dump(ret.value)


def test_release_file_is_the_base():
    assert runner.sha256_file(RUN_TRAIN) == runner.code_base_sha256('chang/run_train.py')
    assert runner.release_files()['chang/run_train.py'] == runner.code_base_sha256('chang/run_train.py')


def test_restated_callback_arguments_equal_run_train():
    assert ast.literal_eval(assigned_call('pbar').args[0]) == runner.PBAR_FORMAT
    pareto = assigned_call('pareto')
    assert ast.literal_eval(pareto.args[1]) == runner.PARETO_METRICS
    assert ast.literal_eval(pareto.args[2]) == runner.PARETO_SIDES
    assert ast.literal_eval(kw(pareto, 'fname_format')) == runner.PARETO_FNAME
    assert ast.dump(kw(pareto, 'enable_if').body) == lambda_body(runner.pareto_enable_if)
    beta = assigned_call('beta_sched').args[0]
    assert [tuple(x) for x in ast.literal_eval(beta.args[0])] == runner.BETA_POINTS
    stop = assigned_call('stop')
    assert ast.dump(stop.args[0].body) == lambda_body(runner.stop_condition)
    lr = assigned_call('lr_sched').args[0]
    assert ast.unparse(lr.args[0]) == 'args.learning_rate' and ast.literal_eval(lr.args[1]) == runner.LR_FIRST_DECAY_STEPS
    assert {k.arg: ast.literal_eval(k.value) for k in lr.keywords} == runner.LR_KWARGS
    order = [e.id for e in assigned_call('callbacks').elts]
    names = {'ebops': 'FreeEBOPs', 'lr_sched': 'LearningRateScheduler', 'beta_sched': 'BetaScheduler', 'pbar': 'PBar',
             'pareto': 'ParetoFront', 'stop': 'StopIf'}
    assert tuple(names[n] for n in order) == runner.RELEASED_ORDER


def test_restated_parser_defaults_compile_and_fit():
    defaults = {}
    for st in main_block().body:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) and getattr(st.value.func, 'attr', '') == 'add_argument':
            name = ast.literal_eval(st.value.args[0])
            for k in st.value.keywords:
                if k.arg == 'default':
                    defaults[name] = ast.literal_eval(k.value)
    assert defaults['--batch-size'] == runner.TRAIN_BATCH and defaults['--learning-rate'] == runner.LEARNING_RATE
    calls = {getattr(st.value.func, 'attr', ''): st.value for st in main_block().body
             if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)}
    assert ast.literal_eval(kw(calls['compile'], 'steps_per_execution')) == runner.STEPS_PER_EXECUTION
    assert ast.literal_eval(kw(calls['fit'], 'epochs')) == runner.RELEASE_EPOCHS
    assert ast.literal_eval(assigned_call('dataset_val').args[2]) == runner.VAL_BATCH
    gm = assigned_call('model')
    assert [ast.unparse(a) for a in gm.args[1:4]] == ['7', '7', '1e-08'] and rng.CH8_BUILDER_ARGS[1:4] == (7, 7, 1e-8)


def config(**over):
    import runner_child
    cfg = runner_child.engineering_config(arm='F000', seed=1, epochs=3, code_sha='a' * 40,
                                          data_manifest_sha256=runner.sha256_file(FORK / 'manifests' / 'data-prep-150c.json'),
                                          init_manifest_path='/nowhere/init.manifest.json', init_manifest_sha256='b' * 64)
    cfg.update(over)
    return cfg


def test_engineering_config_is_accepted():
    s = runner.validate_config(config(), data_manifest_path=FORK / 'manifests' / 'data-prep-150c.json')
    assert s == {'arm_id': 'F000', 'stage': 'gate', 'engineering': True, 'seed': 1, 'epochs': 3, 'builder': 'lab',
                 'schedule': 'ramp'}
    ch8 = config(arm_id='CH-8', builder='released', config_code=None)
    ch8.pop('groups')
    ch8.pop('embedding')
    assert runner.validate_config(ch8)['builder'] == 'released'


@pytest.mark.parametrize('over,needle', [
    ({'stage': 'S1', 'epochs': 7000}, 'claim_domain is null'),  # CH-8/F000 no longer need headline (v0.4)
    ({'stage': 'S1', 'epochs': 7000, 'headline': 'H1'}, 'claim_domain is null'),
    ({'stage': 'S1', 'headline': 'H1', 'claim_domain': {'base': 'b'}}, 'epochs must be 7000'),
    ({'stage': 'S1', 'seed': 4, 'epochs': 7000}, 'outside stage'),
    ({'stage': 'S4', 'seed': 4, 'epochs': 7000, 'headline': 'H1', 'claim_domain': {'base': 'b'}}, 's4_option'),
    ({'wandb': {'mode': 'online'}}, 'online is refused'),
    ({'schedule': 'pid'}, 'PID arms and only they'),
    ({'arm_id': 'F100', 'config_code': 'F100', 'groups': {'attention': 'B', 'ffn': 'Q', 'head': 'Q'}}, 'needs the headline'),
    ({'arm_id': 'PID-BEST', 'config_code': 'F000'}, 'PID arms and only they'),
    ({'arm_id': 'FP32', 'control': 'FP32', 'config_code': None}, 'not built yet'),
    ({'arm_id': 'BEST-C', 'config_code': 'F000'}, 'not built yet'),
    ({'arm_id': 'GR', 'control': 'GR', 'config_code': 'F111', 'form': 'FB', 'headline': 'H1', 'stage': 'S3',
      'epochs': 7000, 'claim_domain': {'base': 'b'}, 'groups': {'attention': 'B', 'ffn': 'B', 'head': 'B'}},
     'engineering stages only'),
    ({'arm_id': 'GR', 'control': 'GR', 'config_code': 'F110', 'form': 'FB', 'headline': 'H1',
      'groups': {'attention': 'B', 'ffn': 'B', 'head': 'Q'}}, 'groups must all be B'),
    ({'arm_id': 'F111', 'config_code': 'F111', 'form': 'FB', 'groups': {'attention': 'B', 'ffn': 'B', 'head': 'B'}},
     'needs the headline'),
    ({'builder': 'released'}, 'lab builder'),
    ({'config_code': 'F001'}, 'config_code'),
    ({'prefit_reseed': 1}, 'decision U5'),
    ({'init_tensors': {'manifest_path': '/x', 'manifest_sha256': None}}, 'null is refused'),
    ({'ramp': dict(runner.RAMP_RECORD, lr=1e-3)}, 'ramp must equal'),
    ({'epochs': 0}, 'epochs in 1..7000'),
    ({'matmul_precision': 'tensorfloat32'}, 'float32'),
    ({'cost_callbacks': False}, 'FP32'),
    ({'headline': 'H3'}, 'H1 or H2'),
    ({'code_sha': 'abc'}, 'code_sha'),
    ({'attempt': 0}, 'attempt'),
    ({'campaign': 'other'}, 'campaign'),
    ({'schema_version': 2}, 'schema_version'),
])
def test_config_refusals(over, needle):
    with pytest.raises(runner.ConfigRefused, match=needle):
        runner.validate_config(config(**over))


@pytest.mark.parametrize('arm,form,emb,control,code,groups', [
    ('F100', None, 'Q', None, 'F100', 'BQQ'),
    ('F011', 'BF', 'Q', None, 'F011', 'QBB'),
    ('F111', 'FB', 'Q', None, 'F111', 'BBB'),
    ('GR', 'BF', 'Q', 'GR', 'F111', 'BBB'),
    ('BE', 'FB', 'B', 'BE', 'F111', 'BBB'),
])
def test_binary_engineering_configs_are_accepted(arm, form, emb, control, code, groups):
    g = dict(zip(('attention', 'ffn', 'head'), groups))
    cfg = config(arm_id=arm, config_code=code, groups=g, embedding=emb, control=control, form=form, headline='H2')
    s = runner.validate_config(cfg)
    assert s['arm_id'] == arm and s['builder'] == 'lab'
    assert arm in runner.BUILT_ARMS


def test_data_manifest_hash_is_checked(tmp_path):
    other = tmp_path / 'm.json'
    other.write_text('{}')
    with pytest.raises(runner.ConfigRefused, match='data manifest'):
        runner.validate_config(config(), data_manifest_path=other)


def test_config_sha256_is_canonical():
    a = config()
    b = dict(reversed(list(copy.deepcopy(a).items())))
    assert runner.config_sha256(a) == runner.config_sha256(b)
    assert runner.config_sha256(a) != runner.config_sha256(config(attempt=2))


def test_arm_specs():
    q = {'attention': 'Q', 'ffn': 'Q', 'head': 'Q'}
    assert m.arm_spec('F000', groups=q) == m.F000_SPEC
    assert m.arm_spec('F011', groups={'attention': 'Q', 'ffn': 'B', 'head': 'B'}, form='BF', headline='H1').config_code == 'F011'
    assert m.arm_spec('F100', groups={'attention': 'B', 'ffn': 'Q', 'head': 'Q'}, headline='H2').form is None
    assert m.arm_spec('F000', groups=q, headline='H1') == m.F000_SPEC        # the headline is not kept on all-Q arms
    for kwargs, needle in ((dict(arm_id='F100', groups=q), 'configuration F000'),
                           (dict(arm_id='F010', groups={'attention': 'Q', 'ffn': 'B', 'head': 'Q'}), 'form FB or BF'),
                           (dict(arm_id='F100', groups={'attention': 'B', 'ffn': 'Q', 'head': 'Q'}, form='FB'), 'form is set only'),
                           (dict(arm_id='CH-8', groups=q), 'released builder'),
                           (dict(arm_id='F000', groups=q, embedding='B'), 'only in BE'),
                           (dict(arm_id='F000', groups={'attention': 'Q'}), 'groups must map'),
                           (dict(arm_id='F100', groups={'attention': 'B', 'ffn': 'Q', 'head': 'Q'}), 'needs the headline'),
                           (dict(arm_id='F100', groups={'attention': 'B', 'ffn': 'Q', 'head': 'Q'}, headline='H3'),
                            'needs the headline'),
                           (dict(arm_id='BE', groups={'attention': 'B', 'ffn': 'B', 'head': 'Q'}, embedding='B',
                                 control='BE', form='FB', headline='H1'), 'groups must all be B')):
        with pytest.raises(m.SpecError, match=needle):
            m.arm_spec(**kwargs)


def test_unbuilt_paths_raise():
    q = {'attention': 'Q', 'ffn': 'Q', 'head': 'Q'}
    for control in ('FP32', 'W8A8', 'DS-HGQ'):
        with pytest.raises(m.NotBuilt):
            m.build(m.arm_spec(control, groups=q, control=control))
        with pytest.raises(m.NotBuilt):
            m.role_spec(m.arm_spec(control, groups=q, control=control))
    assert m.role_spec(m.F000_SPEC).n_variables == rng.CH8_N_VARIABLES


@pytest.mark.parametrize('arm,form,emb,control,groups,sites', [
    ('F100', None, 'Q', None, 'BQQ', ['mha/query', 'mha/key', 'mha/value', 'mha/attention_output']),
    ('F010', 'FB', 'Q', None, 'QBQ', ['ffn1', 'ffn2']),
    ('F001', 'BF', 'Q', None, 'QQB', ['head1', 'head2', 'head3']),
    ('BE', 'BF', 'B', 'BE', 'BBB', ['embed', 'mha/query', 'mha/key', 'mha/value', 'mha/attention_output', 'ffn1', 'ffn2',
                                    'head1', 'head2', 'head3']),
])
def test_binary_role_specs(arm, form, emb, control, groups, sites):
    g = dict(zip(('attention', 'ffn', 'head'), groups))
    spec = m.arm_spec(arm, groups=g, embedding=emb, form=form, control=control, headline='H1')
    assert m.binary_sites(spec) == sites
    rs = m.role_spec(spec)
    assert rs.n_variables == rng.CH8_N_VARIABLES
    want_unm = {f'{s}/qg/FixedPointQuantizerKBI[0]/{v}' for s in sites for v in ('k', 'b', 'i')}
    assert rs.expected_unmatched(None) == want_unm
    assert rs.expected_unused(None) == {x.replace('/qg/', '/kq/') for x in want_unm}
    seq = m.class_sequence(spec)
    assert len(seq) == len(rng.CH8_TOP) and seq[0] == 'InputLayer' and seq[-1] == 'QEinsumDense'
    assert seq[3] == ('BinaryMultiHeadAttention' if groups[0] == 'B' else 'QMultiHeadAttention')
    assert seq[1] == ('BinaryEinsumDenseBatchnorm' if emb == 'B' else 'QEinsumDenseBatchnorm')


def test_ch8c_is_admitted_as_ch8_on_confirmation_seeds():
    """B10 (owner decision B10-9): CH-8-C is CH-8's released-builder ramp arm on the S4 seeds."""
    from cbl import rng
    cfg = config(arm_id='CH-8-C', builder='released', config_code=None, stage='S4', seed=4, epochs=7000,
                 headline='H1', claim_domain={'base': 'b'}, s4_option='A', prefit_reseed=rng.prefit_reseed(4))
    cfg.pop('groups')
    cfg.pop('embedding')
    s = runner.validate_config(cfg)
    assert s['arm_id'] == 'CH-8-C' and s['builder'] == 'released' and s['schedule'] == 'ramp'
    assert 'CH-8-C' in runner.BUILT_ARMS and 'CH-8-C' in runner.RELEASED_BUILDER_ARMS
    with pytest.raises(runner.ConfigRefused, match='outside stage'):
        runner.validate_config(dict(cfg, seed=3))
    with pytest.raises(runner.ConfigRefused, match='released builder'):
        runner.validate_config(dict(cfg, builder='lab'))


def test_launch_check_runs_in_every_stage_but_gate():
    """B10: prepare calls gen.configs.check_launchable on the launch copy unless the stage is gate."""
    gate = config()
    assert runner.check_launch(gate, runner.validate_config(gate)) == {
        'checked': False, 'reason': 'stage gate: a gate copy, not a launch copy'}
    smoke = config(stage='smoke')
    s = runner.validate_config(smoke)                       # admitted by validate_config (engineering stage) ...
    with pytest.raises(runner.ConfigRefused, match='check_launchable'):
        runner.check_launch(smoke, s)                       # ... and refused at launch (unset inputs)
