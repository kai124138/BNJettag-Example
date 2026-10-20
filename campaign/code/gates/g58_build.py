"""Gate G58 (PREFLIGHT-PLAN section 3 row G58; STUDY v0.10 section 13 "Further required gates": every build
and serialized reload; 03-phases phase 2 "every config builds on CPU with the parameter count recorded";
modules M17 gen/, M5 cbl/model.py, M6 cbl/binary.py). PC CPU, venv of record.

Every config that the runner's own acceptance (cbl.runner.validate_config) admits under a constructed
complete input record builds on the CPU with its parameter count recorded, and reloads after
serialization; every other config is listed with its reason and not built.

Checks:
  - repository and venv: HEAD equals --expect-sha12, the tree is clean at start and end, the venv of record;
  - the committed configs/ tree equals its regeneration from gen/inputs.json byte for byte;
  - with the constructed complete input record of G57 (gates/g57_configs.py `complete_inputs`, synthetic
    values only: headline H2, a synthetic S2 record and parameter counts) and synthetic handoff fields,
    validate_config admits exactly the configs G57 lists and refuses the others only as not built or not
    admitted, by reason (G57's N_PASS and REFUSALS: since B10, 109 admitted (S1 45, S3 3, S4 34, smoke 12,
    bench 12, det 3) and 58 refused: 15 controls, 21 BEST-C, 22 S2-chosen engineering-only; B9: 88 and 79);
  - in one fresh child (gates/config_child.py build) every admitted config builds on the CPU as the runner
    builds it, with 244 variables walked by role, finite inference logits on a constructed batch, and an
    inference call that changes no variable;
  - parameter counts: Keras count_params (all variables' elements, as model.summary's total), trainable
    and non-trainable elements; one value per model key (gen.configs.model_key) across its configs;
  - serialization: a .keras save and keras.models.load_model restore the classes, every variable by role
    byte for byte, count_params and the inference logits byte for byte, for every admitted config, and every
    layer configuration except one disclosed difference (finding B7-F1, below): the beta0 initializer of the
    MHA's QSoftmax and its two tables, which the MHA's build recreates under the global default.
The parameter-count record (kind cbl1016-param-counts, the gen/inputs.json `param_counts` input) is
written to --record. Wiring it into gen/inputs.json and regenerating configs/ is a generator-input change
for the orchestrator (owner decision B7-11), not done here. Every reading is engineering, not quotable.

    <venv>/bin/python -I -B gates/g58_build.py --expect-sha12 SHA12 --work DIR --record PATH [--json PATH] \\
        [--children-out DIR]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import g57_configs as G57  # noqa: E402
import gatelib  # noqa: E402
from gen import configs as C  # noqa: E402
from gen import inputs as I  # noqa: E402

GATE = 'G58'
N_VARIABLES = 244
EXPECT_REFUSALS = G57.REFUSALS          # B9: the PID arms admitted (G57's statement of the runner's admission)
# preflight/env/README.md "Initialization readings": CH-8, 27,899 trainable of 44,265 variables' elements (pre-PASS
# engineering reading, CPU and GPU); recorded beside this gate's CH-8 count, not a pass condition
ENV_README_CH8 = {'trainable_elements': 27899, 'count_params': 44265}


# Finding B7-F1 (development run, before the gate of record): QSoftmax and its two QUnaryFunctionLUT tables are
# created inside QMultiHeadAttention.build (installed mha.py:324-338, softmax.py:77-99) and are not serialized on their
# own, so load_model rebuilds them under the global LayerConfigScope default beta0 (1e-5) instead of Chang's
# LayerConfigScope(beta0=0) (model.py:295). Their `beta` variables are restored by value (the variable check above),
# and BetaScheduler and the PID set every beta each epoch; only the initializer in get_config differs.
BETA0_PATH = '/config/beta0/config/value'
BETA0_CLASSES = ('QSoftmax', 'QUnaryFunctionLUT')
BETA0_BUILT, BETA0_RELOADED = '0.0', '1e-05'


def beta0_only(d):
    return (d['class'] in BETA0_CLASSES and len(d['diffs']) == 1 and d['diffs'][0]['path'] == BETA0_PATH
            and d['diffs'][0]['built'] == BETA0_BUILT and d['diffs'][0]['reloaded'] == BETA0_RELOADED)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--expect-sha12', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--record', required=True, help='path of the cbl1016-param-counts record to write')
    ap.add_argument('--json', default=None)
    ap.add_argument('--children-out', default=None)
    a = ap.parse_args()
    g = gatelib.Gate(GATE)
    rec = {'header': g.header(a.expect_sha12)}
    gatelib.check_venv_of_record(g)
    print('label       engineering, not quotable')
    work = Path(a.work).resolve()
    g.check(not work.exists(), f'work directory {work} does not exist yet')
    work.mkdir(parents=True)
    head = rec['header']['HEAD']

    inp = I.load()
    d = C.diff_trees(C.generate(inp), C.disk_tree())
    g.check(not any(d.values()), f'the committed configs/ tree equals its regeneration ({len(d["missing"])} missing, '
                                 f'{len(d["extra"])} extra, {len(d["different"])} different)')
    cfgs = C.build_all(inp)
    with tempfile.TemporaryDirectory(dir=str(work)) as td:
        full = C.build_all(G57.complete_inputs(Path(td), cfgs))
    g.check(len(full) == len(cfgs) == 167, f'{len(full)} configs under the constructed complete record')

    from cbl import runner
    handoff = dict(code_sha=head, init_manifest_sha256=G57.HANDOFF['init_manifest_sha256'])
    admitted, refused = [], []
    for c in full:
        try:
            runner.validate_config(C.with_handoff(c, **handoff), data_manifest_path=FORK / 'manifests' / 'data-prep-150c.json')
            admitted.append(c)
        except runner.ConfigRefused as e:
            refused.append((c, str(e)))
    want = sorted(C.config_path(c) for c in full if G57.expected_pass(c))
    got = sorted(C.config_path(c) for c in admitted)
    g.check(got == want and len(got) == G57.N_PASS,
            f'validate_config admits exactly the {len(got)} configs G57 lists (S1 45, S3 3, S4 34, smoke 12, bench 12, det 3)')
    by_reason = {}
    for c, r in refused:
        by_reason.setdefault(G57._reason(c, r), []).append(C.config_path(c))
    counts = {k: len(v) for k, v in sorted(by_reason.items())}
    g.info('refused, by reason (not built)', counts)
    g.check(counts == EXPECT_REFUSALS, f'the other {len(refused)} are refused only as not built or not admitted: 15 '
                                       'controls, 21 BEST-C, 22 S2-chosen engineering-only')
    rec['refused'] = {k: sorted(v) for k, v in by_reason.items()}
    rec['refused_reasons'] = {C.config_path(c): r for c, r in refused}

    items = [{'path': C.config_path(c), 'model_key': C.model_key(c), 'config': C.with_handoff(c, **handoff)} for c in admitted]
    lst = work / 'admitted.json'
    lst.write_text(json.dumps(items, indent=1, sort_keys=True))
    out = work / 'build.json'
    argv = [sys.executable, '-I', '-B', str(FORK / 'gates' / 'config_child.py'), 'build', '--list', str(lst),
            '--work', str(work / 'saves'), '--out', str(out)]
    env = {k: v for k, v in os.environ.items() if k not in ('KERAS_BACKEND', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_FLAGS')}
    g.info('child argv', ' '.join(argv[3:]))
    with open(work / 'build.stdout.txt', 'w') as so, open(work / 'build.stderr.txt', 'w') as se:
        rc = subprocess.run(argv, stdout=so, stderr=se, cwd=str(work), env=env, timeout=7200).returncode
    child = json.loads(out.read_text()) if out.exists() else None
    if a.children_out:
        co = Path(a.children_out)
        co.mkdir(parents=True, exist_ok=True)
        for src in (out, work / 'build.stdout.txt', work / 'build.stderr.txt', lst):
            if src.exists():
                shutil.copy2(src, co / src.name)
        (co / 'build.argv.txt').write_text(' '.join(argv) + '\n')
    ok = rc == 0 and child is not None and 'error' not in child
    g.check(ok, f'build child exit {rc}' + ('' if ok else f' ({(child or {}).get("error")})'))
    st = (child or {}).get('setup', {})
    g.check(st.get('jax_backend') == 'cpu' and st.get('matmul_precision') == 'float32',
            f'child on the CPU ({st.get("devices")}), matmul precision {st.get("matmul_precision")}')
    rows = (child or {}).get('rows', [])
    g.check(len(rows) == len(admitted), f'the child built {len(rows)} of {len(admitted)} admitted configs')
    errs = [(r['path'], r.get('error')) for r in rows if 'error' in r]
    g.check(not errs, f'every admitted config builds and reloads without an exception ({len(errs)} errors)' +
            (f': {errs[:3]}' if errs else ''))
    good = [r for r in rows if 'error' not in r]
    g.check(all(r['n_variables'] == r['n_roles'] == N_VARIABLES for r in good),
            f'every model has {N_VARIABLES} variables, each walked once by role under its RoleSpec')
    g.check(all(r['logits_finite'] and r['inference_changed_nothing'] for r in good),
            'every model gives finite inference logits on the constructed batch, and the inference call changes no variable')
    g.check(all(r['reload_classes'] == r['classes'] and r['reload_roles_equal'] and not r['reload_differing_roles']
                for r in good),
            'every .keras reload restores the classes and every variable by role byte for byte')
    other = [(r['path'], d) for r in good for d in r['reload_config_diffs'] if not beta0_only(d)]
    known = sum(1 for r in good for d in r['reload_config_diffs'] if beta0_only(d))
    g.check(not other, f'every layer configuration is restored, except the beta0 initializer of the MHA\'s QSoftmax and its '
                       f'two tables ({known} layers: built {BETA0_BUILT}, reloaded {BETA0_RELOADED}; finding B7-F1, '
                       f'variables restored by value)' + (f'; other differences: {other[:2]}' if other else ''))
    rec['reload_config_diffs'] = {r['path']: r['reload_config_diffs'] for r in good if r['reload_config_diffs']}
    g.check(all(r['reload_count_params'] == r['count_params'] and r['logits_equal'] for r in good),
            'every reload has the same count_params and the same inference logits byte for byte')
    per_key = {}
    for r in good:
        per_key.setdefault(r['model_key'], set()).add((r['count_params'], r['trainable_elements'], r['non_trainable_elements']))
    g.check(all(len(v) == 1 for v in per_key.values()), f'one parameter count per model key ({len(per_key)} keys)')
    g.check(len(per_key) == 19, f'{len(per_key)} model keys: CH-8, F000, F100, F001-F111 in FB and BF, GR and BE in FB and BF')
    counts_rec = {k: next(iter(v))[0] for k, v in sorted(per_key.items())}
    for k in sorted(per_key):
        cp, tr, nt = next(iter(per_key[k]))
        g.info(f'parameters {k}', f'count_params {cp}, trainable {tr}, non-trainable {nt}')
    ch8 = next(iter(per_key.get('released|-|-|-|-', {(None, None, None)})))
    rec['env_readme_ch8'] = {'readme': ENV_README_CH8, 'this_gate': {'count_params': ch8[0], 'trainable_elements': ch8[1]},
                             'equal': ch8[0] == ENV_README_CH8['count_params'] and ch8[1] == ENV_README_CH8['trainable_elements']}
    g.info('CH-8 against preflight/env/README.md (pre-PASS reading, not a pass condition)', rec['env_readme_ch8'])
    record = {'kind': 'cbl1016-param-counts', 'schema': 1, 'label': 'engineering, not quotable', 'gate': GATE,
              'code_sha': head, 'definition': 'Keras Model.count_params(): elements of every variable (trainable and '
              'non-trainable), as model.summary prints the total; owner decision B7-10',
              'counts': counts_rec,
              'trainable_elements': {k: next(iter(v))[1] for k, v in sorted(per_key.items())},
              'non_trainable_elements': {k: next(iter(v))[2] for k, v in sorted(per_key.items())},
              'configs': {r['path']: r['model_key'] for r in good}}
    Path(a.record).parent.mkdir(parents=True, exist_ok=True)
    Path(a.record).write_text(json.dumps(record, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    g.info('parameter-count record', f'{a.record} ({len(counts_rec)} keys)')
    rec['param_counts'] = record
    g.check_clean('at end')
    return g.finish(rec, a.json)


if __name__ == '__main__':
    sys.exit(main())
