"""Unit tests for da4ml_clean/table.py (gate G72; STUDY v0.8 §3.4 Addendum A step 7, §19.2 item 5).

Run with the Mac CPU interpreter, from anywhere:

    python -I -B -m pytest -q -p no:cacheprovider tests/test_da4ml_clean_table.py

The rows are built from constructed line directories and the G72 stand-in parser
(`tests/da4ml_clean_stubs/scripts/parse_reports.py`), never from a frozen script's output. Every
value is synthetic; engineering, not quotable.
"""

import json
import os
import sys
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK / 'da4ml_clean'))
import table as T  # noqa: E402

STUBS = FORK / 'tests' / 'da4ml_clean_stubs'
CASES_TEXT = (STUBS / 'scripts' / 'cases_bc.txt').read_text()
PARSER = T.load_parser(str(STUBS / 'scripts' / 'parse_reports.py'),
                       T.sha256_file(str(STUBS / 'scripts' / 'parse_reports.py')))
sys.path.insert(0, str(STUBS / 'tools'))
import fake_xilinx as FX  # noqa: E402

CASE = T.parse_cases(CASES_TEXT)[0]
NAME = CASE['name']


def test_stub_case_list_is_the_frozen_one():
    assert T.sha256_file(str(STUBS / 'scripts' / 'cases_bc.txt')) == T.FROZEN_SHA256['cases_bc.txt']


def test_parse_cases_counts_names_pairs():
    cases = T.parse_cases(CASES_TEXT)
    assert len(cases) == 44
    assert sum(c['role'] == 'rule' for c in cases) == 40
    assert [c['line_no'] for c in cases if c['role'] == 'context'] == [23, 28, 33, 38]
    assert cases[42]['name'] == '32x32_NP2w8_het_stock_bias' and cases[43]['name'] == '32x32_NP2w8_het_fact_bias'
    assert cases[0]['pair'] == cases[1]['pair'] == 1 and cases[42]['pair'] == cases[43]['pair'] == 20
    assert all(c['pair'] is None for c in cases if c['role'] == 'context')
    assert len(set(c['pair'] for c in cases if c['role'] == 'rule')) == 20


def test_parse_cases_rejects_bad_lists():
    lines = CASES_TEXT.splitlines()
    with pytest.raises(ValueError):
        T.parse_cases('\n'.join(lines[:-1]))                  # 43 lines, one fact line unpaired
    with pytest.raises(ValueError):
        T.parse_cases('\n'.join(lines[:-1] + ['32x32 NP2w8 het stock bias']))   # two stock lines in a pair
    with pytest.raises(ValueError):
        T.parse_cases('\n'.join(lines[:-1] + ['32x32 NP2w8 het fact']))         # four fields
    with pytest.raises(ValueError):
        T.parse_cases('\n'.join(lines[:-1] + ['32x32 NP2w8 het other bias']))


def test_line_name_matches_run_case():
    assert T.line_name('32x32', 'P2', 'het', 'stock', '-') == '32x32_P2_het_stock'
    assert T.line_name('32x32', 'NP2w8', 'het', 'fact', 'bias') == '32x32_NP2w8_het_fact_bias'


def test_load_expected_embedded_and_file(tmp_path):
    d, src = T.load_expected(None)
    assert list(d) == list(T.HASHED_FILES) and d['env.sh'] == T.ENV_SH_SHA256 and 'embedded' in src
    p = tmp_path / 'e.json'
    p.write_text(json.dumps({k: '0' * 64 for k in T.HASHED_FILES}))
    assert T.load_expected(str(p))[0]['run_case.sh'] == '0' * 64
    p.write_text(json.dumps({k: '0' * 64 for k in T.HASHED_FILES if k != 'env.sh'}))
    with pytest.raises(ValueError):
        T.load_expected(str(p))
    p.write_text(json.dumps(dict({k: '0' * 64 for k in T.HASHED_FILES}, **{'env.sh': 'xyz'})))
    with pytest.raises(ValueError):
        T.load_expected(str(p))


def test_load_parser_refuses_a_wrong_hash():
    with pytest.raises(ValueError):
        T.load_parser(str(STUBS / 'scripts' / 'parse_reports.py'), '0' * 64)


@pytest.mark.parametrize('stdout,status,stage', [
    ('', '', 'build'),
    ('%s build rc=1\n' % NAME, '', 'build'),
    ('%s build rc=0\n' % NAME, '', 'csynth'),
    ('%s build rc=0\n' % NAME, 'csynth start x host y\n', 'csynth'),
    ('%s build rc=0\n' % NAME, 'csynth start x host y\ncsynth rc=1 t\n', 'parse'),
    ('%s build rc=0\n' % NAME, 'csynth start x host y\ncsynth rc=0 t\n', 'vivado'),
    ('%s build rc=0\n' % NAME, 'csynth start x host y\ncsynth rc=0 t\nvivado rc=0 t\n', 'parse'),
])
def test_stage_in_progress(stdout, status, stage):
    assert T.stage_in_progress(T.rc_reports(NAME, stdout, status)) == stage


def test_rc_reports_skip_and_other_lines():
    rc = T.rc_reports(NAME, 'skip %s\n' % NAME, '')
    assert rc['skip'] is True and rc['build_rc'] is None
    rc = T.rc_reports(NAME, 'skip other_line\nother_line build rc=0\n', '')
    assert rc['skip'] is False and rc['build_rc'] is None


def make_line(tmp, stages, csim=0, util=True, timing=True, csynth_xml=True):
    """A constructed line directory with artifacts up to `stages` (build, csynth, vivado)."""
    d = tmp / 'reports' / NAME
    d.mkdir(parents=True)
    out = ''
    status = ''
    if 'build' in stages:
        (d / 'case_info.json').write_text(json.dumps({'csim_mismatch': csim, 'n_values': 4096 * 32,
                                                      'da_kernel_cost_total': 1.0}))
        out += '%s build rc=0\n' % NAME
    if 'csynth' in stages:
        status += 'csynth start t host h\ncsynth rc=0 t\n'
        if csynth_xml:
            FX.write_csynth(str(d), 5)
    if 'vivado' in stages:
        status += 'vivado rc=0 t\n'
        FX.write_vivado(str(d / 'vivado'), 5, opt_util=util)
        if not timing:
            (d / 'vivado' / 'post_opt_timing.rpt').write_text('no table here\n')
    if status:
        (d / 'STATUS').write_text(status)
    sp = tmp / 'stdout'
    sp.write_text(out + ('%s synth rc=0\n' % NAME if 'vivado' in stages else ''))
    return d, sp


def sidecar(**kw):
    sc = {'outcome': 'returned', 'interrupted': None, 'assertion_failures': [], 'policy_applied': None,
          'run_case_rc': 0, 'provenance_checks': {}, 'versions': {'da4ml': '0.5.2'}}
    sc.update(kw)
    return sc


def row(tmp, sc, **kw):
    d, sp = make_line(tmp, **kw)
    return T.build_row(CASE, sc, str(d), str(sp), PARSER)


def test_complete_line(tmp_path):
    r = row(tmp_path, sidecar(), stages=('build', 'csynth', 'vivado'), csim=3)
    assert r['failed_at'] is None and all(v == 'completed' for v in r['stages'].values())
    assert r['values']['csim_mismatch'] == 3 and r['values']['post_opt_LUT'] == 9000005
    assert r['values']['post_opt_WNS_ns'] == 99.0 and r['values']['II_max'] == '1'
    assert not set(T.FORBIDDEN_KEYS) & set(r) and not set(T.FORBIDDEN_KEYS) & set(r['values'])


def test_build_failed(tmp_path):
    d, sp = make_line(tmp_path, stages=())
    sp.write_text('%s build rc=1\n' % NAME)
    r = T.build_row(CASE, sidecar(run_case_rc=1), str(d), str(sp), PARSER)
    assert r['failed_at'] == 'build' and set(r['stages'].values()) == {'failed'}


def test_missing_csynth_report_fails_csynth_and_vivado_even_with_vivado_reports(tmp_path):
    r = row(tmp_path, sidecar(), stages=('build', 'csynth', 'vivado'), csynth_xml=False)
    assert r['failed_at'] == 'csynth' and r['stages']['vivado'] == 'failed' and r['values']['post_opt_LUT'] is None
    assert r['stages']['csim'] == 'completed'


def test_vivado_without_lut_or_wns_fails_vivado(tmp_path):
    r = row(tmp_path, sidecar(), stages=('build', 'csynth', 'vivado'), util=False)
    assert r['failed_at'] == 'vivado'
    r = row(tmp_path / 'b', sidecar(), stages=('build', 'csynth', 'vivado'), timing=False)
    assert r['failed_at'] == 'vivado' and r['values']['post_opt_WNS_ns'] is None


def test_interrupted_line_fails_at_its_stage_whatever_artifacts_exist(tmp_path):
    sc = sidecar(outcome='interrupted', interrupted={'stage': 'csynth'})
    r = row(tmp_path, sc, stages=('build', 'csynth', 'vivado'))
    assert r['failed_at'] == 'csynth' and r['stages']['build'] == r['stages']['csim'] == 'completed'
    assert r['stages']['csynth'] == r['stages']['vivado'] == 'failed'
    assert r['values']['II_max'] is None and r['values']['post_opt_LUT'] is None
    assert r['artifacts_present']['post_opt_util'] is True


def test_interrupted_in_parse_fails_at_vivado(tmp_path):
    sc = sidecar(outcome='interrupted', interrupted={'stage': 'parse'})
    r = row(tmp_path, sc, stages=('build', 'csynth', 'vivado'))
    assert r['failed_at'] == 'vivado' and r['stages']['csynth'] == 'completed'


def test_interrupted_in_build_fails_every_stage(tmp_path):
    sc = sidecar(outcome='interrupted', interrupted={'stage': 'build'})
    r = row(tmp_path, sc, stages=('build', 'csynth', 'vivado'))
    assert r['failed_at'] == 'build' and set(r['stages'].values()) == {'failed'}


def test_fail_line_remedy_fails_every_stage(tmp_path):
    sc = sidecar(assertion_failures=['no_skip'], policy_applied='fail-line')
    r = row(tmp_path, sc, stages=('build', 'csynth', 'vivado'))
    assert r['failed_at'] == 'build' and 'fail-line' in r['fail_cause'] and set(r['stages'].values()) == {'failed'}


def test_not_started_and_not_run(tmp_path):
    r = T.build_row(CASE, None, None, None, PARSER)
    assert r['failed_at'] == 'build' and r['started'] is False and 'not started' in r['fail_cause']
    sc = sidecar(outcome='not_run_assertion_failed', assertion_failures=['proj_dir_absent'], policy_applied='void')
    r = T.build_row(CASE, sc, None, None, PARSER)
    assert r['failed_at'] == 'build' and 'did not run' in r['fail_cause']


def test_synth_result_cross_check_flags_a_difference(tmp_path):
    d, sp = make_line(tmp_path, stages=('build', 'csynth', 'vivado'))
    (d / 'synth_result.json').write_text(json.dumps({'csim_mismatch': 0, 'csynth': {'II_max': '1'},
                                                     'vivado_post_opt': {'LUT': 1, 'WNS_ns': 99.0}}))
    r = T.build_row(CASE, sidecar(), str(d), str(sp), PARSER)
    assert r['synth_result_json'] == {'present': True, 'differs_on': ['post_opt_LUT']}


def test_unreadable_reports_are_no_stage_result(tmp_path):
    d, sp = make_line(tmp_path, stages=('build', 'csynth', 'vivado'))
    xml = d / 'myproject_prj' / 'solution1' / 'syn' / 'report' / 'csynth.xml'
    xml.write_text(xml.read_text()[:40])                      # truncated XML
    r = T.build_row(CASE, sidecar(), str(d), str(sp), PARSER)
    assert r['failed_at'] == 'csynth' and 'unreadable' in r['fail_cause'] and 'csynth' in r['report_errors']
    d2, sp2 = make_line(tmp_path / 'b', stages=('build', 'csynth', 'vivado'))
    os.remove(d2 / 'vivado' / 'post_opt_util.rpt')
    os.mkdir(d2 / 'vivado' / 'post_opt_util.rpt')             # a directory where the report should be
    r = T.build_row(CASE, sidecar(), str(d2), str(sp2), PARSER)
    assert r['failed_at'] == 'vivado' and 'util' in r['report_errors']
