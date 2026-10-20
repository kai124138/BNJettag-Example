"""Gate G15 (PREFLIGHT-PLAN section 3 row G15; STUDY v0.10 section 13 "CH-8 runner, F000" row:
"nested MHA kernel mapping; single wrap of the HGQ call"; section 2.2 "Sites"; modules M5
cbl/model.py and M4's roles). PC CPU, venv of record.

One child process (gates/runner_child.py mha) builds CH-8 (Chang's get_model), F000 (M5) and CH-8
again in one process, each after `random.seed(1)` and `keras.utils.set_random_seed(1)`, so that the
third build carries shifted Keras name counters. Checks:
1. Nested MHA kernel mapping by role. In each model the MHA is a `QMultiHeadAttention`; its four
   projections `_query_dense`, `_key_dense`, `_value_dense` and `_output_dense` hold the variables
   at the roles `mha/query/...`, `mha/key/...`, `mha/value/...` and `mha/attention_output/...`
   (the kernel at `<role>/kernel` is the attribute's own kernel object), with kernels (24, 2, 16) for
   Q, K and V and (2, 16, 24) for O, a bias for Q, V and O and none for K (`mha.py:178`); each
   projection's kernel, input, bias and output quantizers are configured by the MHA's own
   `_qkvo_*_conf` objects (`_qkvo_kq_conf` at the snapshot's `mha.py:64`, passed to the projections
   by `_get_common_kwargs_for_sublayer`), and the
   kernel quantizer configuration equals the MHA's `_qkvo_kq_conf` field by field; the role set under
   `mha/` (89 roles) is the same in all three models although the layer names differ.
2. Single wrap of the HGQ call. hgq2's `QLayerMeta.__call__` wraps a class's `call` (EBOPs and the
   output quantizer; installed 0.1.9 `layers/core/base.py`, the snapshot's `base.py:57-114`) and, in
   0.1.9 only, its `build` (`check_collapsed_weight`, 0.1.9 `base.py:118-140`), once, at the class's
   first instantiation. A subclass that inherits an already wrapped method is wrapped again. For every
   QLayer class in each model the number of QLayerMeta wrappers on `call` is what base.py's own
   condition gives (1, or 0 for a class it does not wrap) and on `build` is 1, and F000's table
   equals CH-8's. The number of losses a training-mode call adds is the same in F000 and CH-8.
2b. Binary attention (added in B6, decision U3): in F100, F111-FB and BE-BF the MHA is M6's
   `BinaryMultiHeadAttention`; its four projections are `BinaryEinsumDense` at the same attributes and
   roles (kernels (24, 2, 16) and (2, 16, 24), biases for Q, V and O), with no `_kq`; each projection's
   scale quantizer is configured by the MHA's own `_qkvo_kq_conf` object and equals it field by field,
   and its input, bias and output quantizers by the MHA's `_qkvo_*_conf`; the 89 roles under `mha/` are
   CH-8's with `/kq/` replaced by `/qg/`; every QLayer class of these models, the three binary classes
   included, carries exactly the wraps base.py gives (call 1, build 1); a training-mode call adds as
   many losses as CH-8's.
3. Controls, made after QEinsumDense was instantiated: a subclass inheriting both methods (call and
   build wrapped twice) and one defining only its own call (build wrapped twice) are flagged; one
   defining its own call and build (each wrapped once) passes. Called in training mode, the doubly
   wrapped call adds one more loss than the plain layer (the EBOPs penalty twice). Batch B6's binary
   layers and MHA subclass (decision U3) must define their own call and build, as the passing
   control does; G15 reruns on them.

Every reading is engineering, not quotable. Run:

    <venv>/bin/python -I -B gates/g15_mha_wrap.py --expect-sha12 SHA12 --work DIR --archive SMOKE_ARCHIVE \\
        [--data-manifest PATH] [--json PATH] [--children-out DIR]
"""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import gatelib  # noqa: E402
import rng_child  # noqa: E402
import runner_child  # noqa: E402

GATE = 'G15'
SHAPES = {'query': [24, 2, 16], 'key': [24, 2, 16], 'value': [24, 2, 16], 'attention_output': [2, 16, 24]}
ATTRS = {'query': '_query_dense', 'key': '_key_dense', 'value': '_value_dense', 'attention_output': '_output_dense'}
N_MHA_ROLES = 89


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--expect-sha12', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--data-manifest', default=str(FORK / 'manifests' / 'data-prep-150c.json'))
    ap.add_argument('--json', default=None)
    ap.add_argument('--children-out', default=None)
    a = ap.parse_args()
    g = gatelib.Gate(GATE)
    rec = {'header': g.header(a.expect_sha12)}
    gatelib.check_venv_of_record(g)
    print('label       engineering, not quotable')
    work = Path(a.work)
    g.check(not work.exists(), f'work directory {work} does not exist yet')
    work.mkdir(parents=True)
    inputs = rng_child.smoke_inputs(a.data_manifest, a.archive)
    code, r, argv, _ = runner_child.run_child('mha', 'mha', device='cpu', seed=1, archive=inputs['archive'],
                                              archive_sha=inputs['archive_sha256'],
                                              data_manifest_sha=inputs['data_manifest_sha256'], work=work / 'mha',
                                              children_out=a.children_out)
    g.info('mha argv', ' '.join(argv[3:]))
    if not g.check(code == 0 and r is not None and 'error' not in r, f'mha child exit 0 ({(r or {}).get("error")})'):
        g.check_clean('at end')
        return g.finish(rec, a.json)
    models = r['models']
    g.check(set(models) == {'CH-8', 'F000', 'CH-8 again'}, f'three models built: {sorted(models)}')

    # 1. nested MHA kernel mapping by role
    for name, m in models.items():
        g.check(m['mha_class'] == 'QMultiHeadAttention' and m['n_roles'] == 244,
                f'{name}: the MHA is {m["mha_class"]}; {m["n_roles"]} roles')
        for seg, p in sorted(m['projections'].items()):
            g.check(p['attr'] == ATTRS[seg] and p['kernel_role_is_attr_kernel'] and p['kernel_shape'] == SHAPES[seg],
                    f'{name}: mha/{seg}/kernel is {p["attr"]}.kernel, shape {p["kernel_shape"]}')
            want_bias = seg != 'key'
            g.check(p['has_bias_role'] == want_bias and (p['bias_role_is_attr_bias'] is True if want_bias else True),
                    f'{name}: mha/{seg} has {"a" if want_bias else "no"} bias role'
                    + (' (the attribute\'s bias)' if want_bias else ' (mha.py:178)'))
            confs = [p['kq_config_is_mha_conf'], p['iq_config_is_mha_conf'], p['oq_config_is_mha_conf']]
            if want_bias:
                confs.append(p['bq_config_is_mha_conf'])
            g.check(all(c in (True, None) for c in confs) and p['kq_config_is_mha_conf'] is True,
                    f'{name}: mha/{seg} quantizers are configured by the MHA\'s _qkvo_*_conf objects '
                    f'(kq {p["kq_config_is_mha_conf"]}, iq {p["iq_config_is_mha_conf"]}, bq {p["bq_config_is_mha_conf"]}, '
                    f'oq {p["oq_config_is_mha_conf"]})')
            g.check(p['kq_config'] == m['mha_qkvo_kq_conf'], f'{name}: mha/{seg} kq configuration equals _qkvo_kq_conf field by field')
        g.check(len(m['mha_roles']) == N_MHA_ROLES, f'{name}: {len(m["mha_roles"])} roles under mha/')
    ch8, f000, again = models['CH-8'], models['F000'], models['CH-8 again']
    g.check(ch8['mha_roles'] == f000['mha_roles'] == again['mha_roles'],
            'the roles under mha/ are the same in CH-8, F000 and CH-8 built again')
    g.check(len({ch8['top_layer_names'][3], f000['top_layer_names'][3], again['top_layer_names'][3]}) == 3,
            f'... while the MHA layer names differ by Keras counters: {ch8["top_layer_names"][3]}, '
            f'{f000["top_layer_names"][3]}, {again["top_layer_names"][3]}')
    g.check(all(ch8['projections'][s]['roles'] == f000['projections'][s]['roles'] for s in SHAPES)
            and ch8['mha_qkvo_kq_conf'] == f000['mha_qkvo_kq_conf']
            and all(ch8['projections'][s]['kq_config'] == f000['projections'][s]['kq_config'] for s in SHAPES),
            'F000 against CH-8: per projection the same roles and the same kq configuration')

    # 2. single wrap
    for name, m in models.items():
        bad = {c: w for c, w in m['wraps'].items()
               if w['call_wraps'] != w['call_wraps_expected'] or w['build_wraps'] != w['build_wraps_expected']
               or not w['in_wrapped_cls'] or not w['call_code_file']}
        g.info(f'{name} wraps (class: call/expected, build/expected)',
               {c: f'{w["call_wraps"]}/{w["call_wraps_expected"]}, {w["build_wraps"]}/{w["build_wraps_expected"]}'
                for c, w in sorted(m['wraps'].items())})
        g.check(not bad and len(m['wraps']) >= 8,
                f'{name}: every one of {len(m["wraps"])} QLayer classes carries exactly the QLayerMeta wraps base.py '
                f'gives (call 1 or 0, build 1){"" if not bad else f"; differ: {sorted(bad)}"}')
    g.check(f000['wraps'] == ch8['wraps'], 'F000\'s wrap table equals CH-8\'s')
    g.check(f000['n_losses_training_call'] == ch8['n_losses_training_call'] == again['n_losses_training_call'],
            f'losses added by one training-mode call: CH-8 {ch8["n_losses_training_call"]}, F000 '
            f'{f000["n_losses_training_call"]}, CH-8 again {again["n_losses_training_call"]}')

    # 2b. binary attention (B6)
    bm = r.get('binary_models') or {}
    g.check(set(bm) == {'F100', 'F111-FB', 'BE-BF'}, f'three binary models built: {sorted(bm)}')
    ch8_qg_roles = sorted(x.replace('/kq/', '/qg/') for x in ch8['mha_roles'])
    for name, m in bm.items():
        g.check(m['is_binary_mha'] and m['mha_class'] == 'BinaryMultiHeadAttention' and m['n_roles'] == 244,
                f'{name}: the MHA is {m["mha_class"]}; {m["n_roles"]} roles')
        for seg, p in sorted(m['projections'].items()):
            want_bias = seg != 'key'
            g.check(p['class'] == 'BinaryEinsumDense' and p['attr'] == ATTRS[seg] and p['kernel_role_is_attr_kernel']
                    and p['kernel_shape'] == SHAPES[seg] and not p['has_kq_attr']
                    and p['has_bias_role'] == want_bias and (p['bias_role_is_attr_bias'] is True if want_bias else True),
                    f'{name}: mha/{seg} is a BinaryEinsumDense at {p["attr"]}, kernel {p["kernel_shape"]}, '
                    f'{"a" if want_bias else "no"} bias, no kernel quantizer')
            confs = [p['qg_config_is_mha_conf'], p['iq_config_is_mha_conf'], p['oq_config_is_mha_conf']]
            if want_bias:
                confs.append(p['bq_config_is_mha_conf'])
            g.check(all(c in (True, None) for c in confs) and p['qg_config_is_mha_conf'] is True
                    and p['qg_config'] == m['mha_qkvo_kq_conf'] == ch8['mha_qkvo_kq_conf'],
                    f'{name}: mha/{seg} q_g is configured by the MHA\'s _qkvo_kq_conf object, equal field by field '
                    f'(and to CH-8\'s); iq, bq, oq by the MHA\'s _qkvo_*_conf')
        g.check(m['mha_roles'] == ch8_qg_roles, f'{name}: the 89 roles under mha/ are CH-8\'s with /kq/ replaced by /qg/')
        bad = {c: w for c, w in m['wraps'].items()
               if w['call_wraps'] != w['call_wraps_expected'] or w['build_wraps'] != w['build_wraps_expected']
               or not w['in_wrapped_cls'] or not w['call_code_file']}
        g.info(f'{name} wraps (class: call/expected, build/expected)',
               {c: f'{w["call_wraps"]}/{w["call_wraps_expected"]}, {w["build_wraps"]}/{w["build_wraps_expected"]}'
                for c, w in sorted(m['wraps'].items())})
        g.check(not bad and set(m['binary_classes']) <= set(m['wraps']) and len(m['binary_classes']) >= 2,
                f'{name}: every QLayer class, the binary ones {m["binary_classes"]} included, carries exactly one '
                f'wrap of call and build{"" if not bad else f"; differ: {sorted(bad)}"}')
        g.check(m['n_losses_training_call'] == ch8['n_losses_training_call'],
                f'{name}: a training-mode call adds {m["n_losses_training_call"]} losses, as CH-8')
    rec['binary_models'] = {n: {k: v for k, v in m.items() if k != 'wraps'} for n, m in bm.items()}
    rec['binary_wraps'] = {n: m['wraps'] for n, m in bm.items()}

    # 3. controls
    c = {k.split('.')[-1]: v for k, v in r['controls'].items()}
    calls = {k.split('.')[-1]: v for k, v in r['control_calls'].items()}
    g.info('controls (call/expected, build/expected)', {k: f'{v["call_wraps"]}/{v["call_wraps_expected"]}, '
                                                        f'{v["build_wraps"]}/{v["build_wraps_expected"]}' for k, v in c.items()})
    g.info('control calls (training mode)', calls)
    g.check(r.get('qeinsumdense_wrapped_before_controls') is True, 'controls are made after QEinsumDense was instantiated')
    iw, oc, ob = c.get('InheritsWrapped', {}), c.get('DefinesOwnCall', {}), c.get('DefinesOwnCallAndBuild', {})
    g.check(iw.get('call_wraps') == 2 and iw.get('build_wraps') == 2,
            'negative control: a subclass inheriting the wrapped call and build is flagged (call 2, build 2)')
    g.check(oc.get('call_wraps') == 1 and oc.get('build_wraps') == 2,
            'negative control: a subclass defining only its own call is flagged (build 2)')
    g.check(ob.get('call_wraps') == 1 and ob.get('build_wraps') == 1,
            'positive control: a subclass defining its own call and build is wrapped once each')
    plain = calls.get('QEinsumDense', {})
    g.check(plain.get('ok') and calls.get('InheritsWrapped', {}).get('ok')
            and calls['InheritsWrapped']['n_losses'] == plain['n_losses'] + 1,
            f'the doubly wrapped call adds one more loss than the plain layer in training mode '
            f'({calls.get("InheritsWrapped", {}).get("n_losses")} against {plain.get("n_losses")}: the EBOPs penalty twice)')
    g.check(calls.get('DefinesOwnCallAndBuild', {}).get('ok') and calls['DefinesOwnCallAndBuild']['n_losses'] == plain['n_losses']
            and calls['DefinesOwnCallAndBuild']['ebops'] == plain['ebops'],
            'the once-wrapped subclass adds the same losses and EBOPs as the plain layer')
    rec['models'] = {n: {k: v for k, v in m.items() if k != 'wraps'} for n, m in models.items()}
    rec['wraps'] = {n: m['wraps'] for n, m in models.items()}
    rec['controls'], rec['control_calls'] = r['controls'], r['control_calls']
    g.check_clean('at end')
    return g.finish(rec, a.json)


if __name__ == '__main__':
    sys.exit(main())
