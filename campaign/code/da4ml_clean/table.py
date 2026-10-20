#!/usr/bin/env python3
"""Per-line table of one attempt of the clean da4ml rerun (module M19).

STUDY v0.8 (`review/snapshots/v0.8/STUDY.md`) §3.4 Addendum A step 7, §13 da4ml row and §19.2
item 5 ("Mechanics of an interruption"). The table has 44 rows, one per line of `cases_bc.txt` in
file order; the four `hgq4` lines are labelled context. For the four stages the frozen block names
(build, C-simulation, C-synthesis, Vivado; block "Fail-closed provisions") each row says whether
the line has a stage result and gives the raw values that result carries:

- build: `run_case.sh` printed `<line> build rc=0` and `case_info.json` loads;
- csim: `case_info.json` holds integer `csim_mismatch` and `n_values` (the count of mismatches is
  reported as it is; whether it is zero is not judged here);
- csynth: `STATUS` holds `csynth rc=0` and the C-synthesis report parses, read with the frozen
  `parse_reports.py` function `csynth()`;
- vivado: `STATUS` holds `vivado rc=0`, `post_opt_util.rpt` gives a CLB LUT count and
  `post_opt_timing.rpt` gives a WNS, read with the frozen functions `util()` and `timing()`.

A line without a stage result is failed at that stage and at every later one, and later artifacts
are ignored. A line whose sidecar records `interrupted` is failed at the stage it was stopped in
and at every later one, whatever artifacts exist; a stop after Vivado has reported (the parse
step) is failed at Vivado (owner decision B15m-3). A line failed under the fail-line remedy of a
per-line assertion is failed at every stage. A line with no sidecar did not start in the attempt
and is failed at every stage. Failed lines are kept, unlike `make_table.py`.

The table computes no headline decision: no term of the frozen block (bit-exact, decisive LUT
comparison, timing met), no condition and no aggregate. Applying the rule is gate G74's.

Standard library only; written for Python 3.6 or later. Usage:

    python3 table.py --evidence DIR --attempt N [--expected-hashes FILE]

`--expected-hashes` is accepted only when the run record says dry run. Writes
`<evidence>/attempts/aNN/results/table.json`, `table.md` and `table.sha256`.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import OrderedDict

SCHEMA = 'cbl1016-da4ml-clean-table/1'

# STUDY v0.8 lines 60-72, "Script hashes at the freeze (sha256)", files under
# preflight/da4ml-test/scripts/; env.sh at line 72.
FROZEN_SHA256 = OrderedDict([
    ('cases_bc.txt', '1ba9c416b1c4c6007ad07361bd95b1e902058e66039df89e4b5be1a8928a3191'),
    ('stage_b_build.py', 'db49a4041ff1a9a041e9a3d9c87bded997ff3adc96844366941f981028ba83d6'),
    ('stage_b_build_pos.py', 'f214dc3430a0871ee39119a93a719da3448d0fe9868e4cbae70520e6dc9a00d2'),
    ('run_case.sh', '1253773bd99b58015620e0e99bd12f5e575592d772545a453db43245004169f8'),
    ('synth_one.sh', '32201339fcab6d42cad096b393fa7d41ca0380e5948f718422f4c07869ee9bb9'),
    ('vivado_ooc.tcl', '70d269f430e8fe364ae660f9f13fa81bcb8f7c0a9c373f0a1dcecbe43eb14c32'),
    ('clock.xdc', 'e4ec5851c4fff9a4f8c0caf420e487be9e43c2a8f3ce83cb9954b3ddacb4339f'),
    ('common.py', 'cfb50a22bb95a5746e31ff456bd6b8721e9cdad684babd02c0789361e266a1fa'),
    ('parse_reports.py', 'a889ce8ff21f94eae61f9322e332930e99f22d8beb81eadca647c0922707ae38'),
])
ENV_SH_SHA256 = 'a1c095ef7ab3b6dc32ee07dcf0b9b3c2e1794b8c48d8c04d970614592f459b78'
HASHED_FILES = tuple(FROZEN_SHA256) + ('env.sh',)
# Addendum A step 5: the files whose hashes the sidecar must hold.
SIDECAR_FILES = ('stage_b_build.py', 'common.py', 'run_case.sh', 'synth_one.sh', 'vivado_ooc.tcl',
                 'clock.xdc', 'parse_reports.py', 'env.sh')

STAGES = ('build', 'csim', 'csynth', 'vivado')          # the block's stages, in order
REPORTED_STAGES = ('build', 'csynth', 'vivado', 'parse')  # where a running line can be stopped
STOP_TO_STAGE = {'build': 'build', 'csynth': 'csynth', 'vivado': 'vivado', 'parse': 'vivado'}

N_LINES, N_RULE, N_CONTEXT, N_PAIRS = 44, 40, 4, 20
FORBIDDEN_KEYS = ('H1', 'H2', 'headline', 'condition', 'conditions', 'bit_exact', 'timing_met')


# ----------------------------------------------------------------------------------- helpers

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_or_none(path):
    try:
        return sha256_file(path)
    except (IOError, OSError):
        return None


def read_text(path):
    try:
        with open(path, 'r', errors='replace') as f:
            return f.read()
    except (IOError, OSError):
        return None


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (IOError, OSError, ValueError):
        return None


def dump_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write('\n')
    os.replace(tmp, path)


def load_expected(path=None):
    """The expected sha256 per hashed file: the STUDY v0.8 table embedded above, or, for a dry
    run only, a JSON object with exactly the same keys. Returns (dict, source string)."""
    if path is None:
        d = OrderedDict(FROZEN_SHA256)
        d['env.sh'] = ENV_SH_SHA256
        return d, 'embedded STUDY v0.8 table (lines 60-72)'
    with open(path) as f:
        d = json.load(f)
    if not isinstance(d, dict) or set(d) != set(HASHED_FILES):
        raise ValueError('%s must hold exactly the keys %s' % (path, ', '.join(HASHED_FILES)))
    for k, v in d.items():
        if not (isinstance(v, str) and re.match(r'^[0-9a-f]{64}$', v)):
            raise ValueError('%s: %s is not a sha256 hex digest' % (path, k))
    return OrderedDict((k, d[k]) for k in HASHED_FILES), 'file %s (sha256 %s)' % (path, sha256_file(path))


# ------------------------------------------------------------------------------- case list

def line_name(shape, g, inp, path, bias_field):
    """The project and log name exactly as run_case.sh:7-8 forms it."""
    name = '%s_%s_%s_%s' % (shape, g, inp, path)
    return name + '_bias' if bias_field == 'bias' else name


def parse_cases(text):
    """cases_bc.txt -> list of case dicts in file order, with role and pair id. Raises ValueError
    unless the list has 44 lines: 40 rule lines (stock/fact) in 20 pairs, 4 hgq4 context lines."""
    cases = []
    for n, raw in enumerate(text.splitlines(), 1):
        f = raw.split()
        if len(f) != 5:
            raise ValueError('cases_bc.txt line %d has %d fields: %r' % (n, len(f), raw))
        shape, g, inp, path, bias = f
        if path not in ('stock', 'fact', 'hgq4') or bias not in ('-', 'bias'):
            raise ValueError('cases_bc.txt line %d: path %r or bias %r not recognised' % (n, path, bias))
        cases.append(OrderedDict([
            ('line_no', n), ('name', line_name(shape, g, inp, path, bias)), ('shape', shape), ('g', g),
            ('inp', inp), ('path', path), ('bias_field', bias),
            ('role', 'context' if path == 'hgq4' else 'rule'), ('pair', None)]))
    pairs = OrderedDict()
    for c in cases:
        if c['role'] == 'rule':
            pairs.setdefault((c['shape'], c['g'], c['inp'], c['bias_field']), []).append(c)
    for i, members in enumerate(pairs.values(), 1):
        if sorted(m['path'] for m in members) != ['fact', 'stock']:
            raise ValueError('pair %d is not one stock and one fact line: %s' % (i, [m['name'] for m in members]))
        for m in members:
            m['pair'] = i
    names = [c['name'] for c in cases]
    n_rule = sum(c['role'] == 'rule' for c in cases)
    if len(set(names)) != len(names):
        raise ValueError('line names are not unique')
    if (len(cases), n_rule, len(cases) - n_rule, len(pairs)) != (N_LINES, N_RULE, N_CONTEXT, N_PAIRS):
        raise ValueError('cases_bc.txt has %d lines, %d rule, %d context, %d pairs; expected 44, 40, 4, 20'
                         % (len(cases), n_rule, len(cases) - n_rule, len(pairs)))
    return cases


# ------------------------------------------------------------------------------ stage reports

def _last_int(pattern, text):
    m = re.findall(pattern, text or '', re.M)
    return (int(m[-1]) if m else None), len(m)


def rc_reports(name, stdout_text, status_text):
    """Return codes as run_case.sh prints them (stdout, run_case.sh:10,13,15) and synth_one.sh
    writes them to STATUS (synth_one.sh:7,9,13)."""
    n = re.escape(name)
    build_rc, n_build = _last_int(r'^%s build rc=(\d+)\s*$' % n, stdout_text)
    synth_rc, n_synth = _last_int(r'^%s synth rc=(\d+)\s*$' % n, stdout_text)
    csynth_rc, n_csynth = _last_int(r'^csynth rc=(\d+)\b', status_text)
    vivado_rc, n_vivado = _last_int(r'^vivado rc=(\d+)\b', status_text)
    return OrderedDict([
        ('skip', bool(re.search(r'^skip %s\s*$' % n, stdout_text or '', re.M))),
        ('build_rc', build_rc), ('synth_rc', synth_rc),
        ('csynth_started', bool(re.search(r'^csynth start ', status_text or '', re.M))),
        ('csynth_rc', csynth_rc), ('vivado_rc', vivado_rc),
        ('report_counts', OrderedDict([('build', n_build), ('synth', n_synth), ('csynth', n_csynth),
                                       ('vivado', n_vivado)])),
    ])


def stage_in_progress(rc):
    """The stage a line was in when it stopped: the first stage without a completion report.
    'parse' means Vivado had reported (or C-synthesis had failed) and parse_reports.py was next."""
    if rc['build_rc'] is None or rc['build_rc'] != 0:
        return 'build'
    if rc['csynth_rc'] is None:
        return 'csynth'
    if rc['csynth_rc'] != 0:
        return 'parse'
    if rc['vivado_rc'] is None:
        return 'vivado'
    return 'parse'


# ---------------------------------------------------------------------------------- parser

def load_parser(path, expected_sha):
    """Load the frozen parse_reports.py by path after checking its hash. Only its functions
    csynth(projdir), util(file) and timing(file) are called; its main() is not."""
    got = sha256_or_none(path)
    if got != expected_sha:
        raise ValueError('parse_reports.py at %s has sha256 %s, expected %s' % (path, got, expected_sha))
    spec = importlib.util.spec_from_file_location('cbl1016_frozen_parse_reports', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ('csynth', 'util', 'timing'):
        if not callable(getattr(mod, fn, None)):
            raise ValueError('parse_reports.py has no function %s' % fn)
    return mod


# ------------------------------------------------------------------------------------- rows

def _fail_from(stages, first, cause):
    hit = False
    for s in STAGES:
        hit = hit or s == first
        if hit:
            stages[s] = 'failed'
    return first, cause


def _call(fn, *args):
    """(result, None) or (None, 'ExceptionType: message'): a malformed report is no stage result,
    never a crash of the table."""
    try:
        return fn(*args), None
    except Exception as e:  # noqa: BLE001
        return None, '%s: %s' % (type(e).__name__, e)


def build_row(case, sidecar, line_dir, stdout_path, parser):
    """One table row. `line_dir` is the evidence copy of the line's project artifacts
    (reports/<line>/, same relative layout as $SCR/proj/main/<line>)."""
    row = OrderedDict((k, case[k]) for k in ('line_no', 'name', 'shape', 'g', 'inp', 'path', 'bias_field',
                                             'role', 'pair'))
    stages = OrderedDict((s, 'completed') for s in STAGES)
    values = OrderedDict((k, None) for k in ('csim_mismatch', 'n_values', 'da_kernel_cost_total', 'II_min',
                                             'II_max', 'est_clock_ns', 'post_opt_LUT', 'post_opt_WNS_ns'))
    row['started'] = sidecar is not None
    row['outcome'] = sidecar.get('outcome') if sidecar else None
    row['interrupted_stage'] = (sidecar.get('interrupted') or {}).get('stage') if sidecar else None
    row['assertion_failures'] = list(sidecar.get('assertion_failures') or []) if sidecar else []
    row['policy_applied'] = sidecar.get('policy_applied') if sidecar else None
    failed_at, cause = None, None

    stdout_text = read_text(stdout_path) if stdout_path else None
    status_text = read_text(os.path.join(line_dir, 'STATUS')) if line_dir else None
    rc = rc_reports(case['name'], stdout_text, status_text)
    row['rc'] = OrderedDict([('run_case', (sidecar or {}).get('run_case_rc')), ('build', rc['build_rc']),
                             ('synth', rc['synth_rc']), ('csynth', rc['csynth_rc']), ('vivado', rc['vivado_rc'])])
    row['skip_printed'] = rc['skip']

    info = load_json(os.path.join(line_dir, 'case_info.json')) if line_dir else None
    csynth = vivado_util = vivado_timing = None
    perr = OrderedDict()
    if parser is not None and line_dir:
        if os.path.isfile(os.path.join(line_dir, 'myproject_prj', 'solution1', 'syn', 'report', 'csynth.xml')):
            csynth, perr['csynth'] = _call(parser.csynth, line_dir)
        vivado_util, perr['util'] = _call(parser.util, os.path.join(line_dir, 'vivado', 'post_opt_util.rpt'))
        vivado_timing, perr['timing'] = _call(parser.timing, os.path.join(line_dir, 'vivado', 'post_opt_timing.rpt'))
    perr = OrderedDict((k, v) for k, v in perr.items() if v)
    row['report_errors'] = perr

    if sidecar is None:
        failed_at, cause = _fail_from(stages, 'build', 'not started in this attempt (no sidecar)')
    elif row['policy_applied'] == 'fail-line':
        failed_at, cause = _fail_from(stages, 'build', 'per-line assertion failed, fail-line remedy: %s'
                                      % ', '.join(row['assertion_failures']))
    elif row['outcome'] != 'returned' and not row['interrupted_stage']:
        failed_at, cause = _fail_from(stages, 'build', 'line did not run (%s)' % row['outcome'])
    else:
        # natural stage results, in order
        if rc['build_rc'] != 0 or not isinstance(info, dict):
            failed_at, cause = _fail_from(stages, 'build', 'build rc=%s, case_info.json %s'
                                          % (rc['build_rc'], 'loaded' if isinstance(info, dict) else 'absent'))
        elif not (isinstance(info.get('csim_mismatch'), int) and isinstance(info.get('n_values'), int)
                  and info['n_values'] > 0):
            failed_at, cause = _fail_from(stages, 'csim', 'no C-simulation result in case_info.json')
        elif rc['csynth_rc'] != 0 or not (isinstance(csynth, dict) and csynth.get('II_max') is not None):
            failed_at, cause = _fail_from(stages, 'csynth', 'csynth rc=%s, report %s'
                                          % (rc['csynth_rc'], 'parsed' if csynth else
                                             ('unreadable (%s)' % perr['csynth']) if 'csynth' in perr else 'absent'))
        elif rc['vivado_rc'] != 0 or not (isinstance(vivado_util, dict) and vivado_util.get('LUT') is not None
                                          and isinstance(vivado_timing, dict)
                                          and vivado_timing.get('WNS_ns') is not None):
            failed_at, cause = _fail_from(stages, 'vivado', 'vivado rc=%s, post-opt LUT %s, WNS %s%s'
                                          % (rc['vivado_rc'],
                                             'parsed' if (vivado_util or {}).get('LUT') is not None else 'absent',
                                             'parsed' if (vivado_timing or {}).get('WNS_ns') is not None else 'absent',
                                             ''.join(', %s unreadable (%s)' % (k, perr[k]) for k in ('util', 'timing')
                                                     if k in perr)))
        if row['interrupted_stage']:
            stop = STOP_TO_STAGE.get(row['interrupted_stage'], 'build')
            if failed_at is None or STAGES.index(stop) < STAGES.index(failed_at):
                failed_at, cause = _fail_from(stages, stop, 'interrupted in %s' % row['interrupted_stage'])
            else:
                cause = '%s; interrupted in %s' % (cause, row['interrupted_stage'])

    if stages['csim'] == 'completed':
        values['csim_mismatch'] = info['csim_mismatch']
        values['n_values'] = info['n_values']
        values['da_kernel_cost_total'] = info.get('da_kernel_cost_total')
    if stages['csynth'] == 'completed':
        values['II_min'] = csynth.get('II_min')
        values['II_max'] = csynth.get('II_max')
        values['est_clock_ns'] = csynth.get('est_clock_ns')
    if stages['vivado'] == 'completed':
        values['post_opt_LUT'] = vivado_util.get('LUT')
        values['post_opt_WNS_ns'] = vivado_timing.get('WNS_ns')
    row['stages'] = stages
    row['failed_at'] = failed_at
    row['fail_cause'] = cause
    row['values'] = values
    row['artifacts_present'] = OrderedDict([
        ('case_info', isinstance(info, dict)), ('csynth_report', csynth is not None),
        ('post_opt_util', bool(vivado_util)), ('post_opt_timing', bool(vivado_timing) and
                                               (vivado_timing or {}).get('WNS_ns') is not None)])

    # cross-check against the frozen parser's own output, when parse_reports.py ran
    sr = load_json(os.path.join(line_dir, 'synth_result.json')) if line_dir else None
    differs = []
    if isinstance(sr, dict):
        pairs = (('csim_mismatch', sr.get('csim_mismatch'), values['csim_mismatch']),
                 ('II_max', (sr.get('csynth') or {}).get('II_max'), values['II_max']),
                 ('post_opt_LUT', (sr.get('vivado_post_opt') or {}).get('LUT'), values['post_opt_LUT']),
                 ('post_opt_WNS_ns', (sr.get('vivado_post_opt') or {}).get('WNS_ns'), values['post_opt_WNS_ns']))
        differs = [k for k, a, b in pairs if b is not None and a != b]
    row['synth_result_json'] = OrderedDict([('present', isinstance(sr, dict)), ('differs_on', differs)])

    prov = (sidecar or {}).get('provenance_checks') or {}
    row['provenance'] = OrderedDict([
        ('scripts_before_ok', prov.get('scripts_before_ok')), ('scripts_after_ok', prov.get('scripts_after_ok')),
        ('env_sh_before_ok', prov.get('env_sh_before_ok')), ('env_sh_after_ok', prov.get('env_sh_after_ok')),
        ('da4ml_version', ((sidecar or {}).get('versions') or {}).get('da4ml'))])
    return row


def build_table(evidence, attempt_id, expected=None, expected_source=None):
    """The table of attempt `attempt_id` from the evidence tree alone."""
    run = load_json(os.path.join(evidence, 'run.json'))
    if not isinstance(run, dict):
        raise ValueError('no run record at %s' % os.path.join(evidence, 'run.json'))
    att = [a for a in run.get('attempts', []) if a.get('id') == attempt_id]
    if len(att) != 1:
        raise ValueError('attempt %s not in the run record' % attempt_id)
    att = att[0]
    if expected is None:
        expected, expected_source = load_expected(None)
    scripts = os.path.join(evidence, 'scripts')
    cases_path = os.path.join(scripts, 'cases_bc.txt')
    if sha256_or_none(cases_path) != expected['cases_bc.txt']:
        raise ValueError('cases_bc.txt copy in the evidence does not have the expected sha256')
    cases = parse_cases(read_text(cases_path))
    parser = load_parser(os.path.join(scripts, 'parse_reports.py'), expected['parse_reports.py'])
    adir = os.path.join(evidence, 'attempts', 'a%02d' % attempt_id)
    rows = []
    for c in cases:
        sc = load_json(os.path.join(adir, 'sidecars', c['name'] + '.provenance.json'))
        line_dir = os.path.join(adir, 'reports', c['name'])
        stdout_path = os.path.join(adir, 'stdout', c['name'] + '.stdout')
        rows.append(build_row(c, sc, line_dir if os.path.isdir(line_dir) else None,
                              stdout_path if os.path.isfile(stdout_path) else None, parser))
    me = os.path.abspath(__file__)
    header = OrderedDict([
        ('schema', SCHEMA), ('dry_run', bool(run.get('dry_run'))), ('run_id', run.get('run_id')),
        ('attempt', attempt_id), ('attempt_status', att.get('status')), ('attempt_cause', att.get('cause')),
        ('label', 'candidate rule run (complete)' if att.get('status') == 'complete'
         else 'context (%s); no rule outcome may be read from it' % att.get('status')),
        ('route', run.get('route')), ('da4ml_version', run.get('da4ml_version')),
        ('policies', run.get('policies')), ('cases_sha256', expected['cases_bc.txt']),
        ('expected_hash_source', expected_source),
        ('table_script', OrderedDict([('path', os.path.relpath(me, os.path.dirname(os.path.dirname(me)))),
                                      ('sha256', sha256_file(me))])),
        ('driver_sha256_at_rule_start', att.get('driver_sha256_at_rule_start')),
        ('stages', list(STAGES)),
        ('counts', OrderedDict([('rows', len(rows)), ('rule', sum(r['role'] == 'rule' for r in rows)),
                                ('context', sum(r['role'] == 'context' for r in rows)),
                                ('pairs', len(set(r['pair'] for r in rows if r['pair'] is not None))),
                                ('rows_with_a_failed_stage', sum(r['failed_at'] is not None for r in rows)),
                                ('started', sum(r['started'] for r in rows))])),
        ('note', 'Per-line stage results and raw values only; no term, condition or headline of the frozen '
                 'block is evaluated here (G74 applies the rule).'
                 + (' DRY RUN with stand-in scripts: every value is synthetic; engineering, not quotable.'
                    if run.get('dry_run') else ' Engineering record until G74; not quotable.')),
    ])
    return OrderedDict([('header', header), ('rows', rows)])


def _cell(v):
    return '-' if v is None else str(v)


def render_md(table):
    h = table['header']
    out = ['# da4ml clean rerun, per-line table: run %s, attempt %s (%s)' % (h['run_id'], h['attempt'],
                                                                         h['attempt_status']), '']
    out.append(h['note'])
    out.append('')
    out.append('Label: %s. Route %s, da4ml %s, policies %s. Cases sha256 `%s`. Table script sha256 `%s`.'
               % (h['label'], h['route'], h['da4ml_version'], json.dumps(h['policies'], sort_keys=True),
                  h['cases_sha256'], h['table_script']['sha256']))
    out.append('')
    out.append('| # | line | role | pair | build | csim | csynth | vivado | failed at | cause | csim mismatches / values '
               '| II min / max | post-opt LUT | post-opt WNS (ns) |')
    out.append('| ---: | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |')
    for r in table['rows']:
        st = r['stages']
        v = r['values']
        out.append('| %d | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s / %s | %s / %s | %s | %s |' % (
            r['line_no'], r['name'], r['role'], _cell(r['pair']),
            *[('ok' if st[s] == 'completed' else 'FAILED') for s in STAGES],
            _cell(r['failed_at']), _cell(r['fail_cause']).replace('|', '/'),
            _cell(v['csim_mismatch']), _cell(v['n_values']), _cell(v['II_min']), _cell(v['II_max']),
            _cell(v['post_opt_LUT']), _cell(v['post_opt_WNS_ns'])))
    out.append('')
    return '\n'.join(out)


def write_table(evidence, attempt_id, expected=None, expected_source=None):
    table = build_table(evidence, attempt_id, expected, expected_source)
    rdir = os.path.join(evidence, 'attempts', 'a%02d' % attempt_id, 'results')
    if not os.path.isdir(rdir):
        os.makedirs(rdir)
    jpath, mpath = os.path.join(rdir, 'table.json'), os.path.join(rdir, 'table.md')
    dump_json(table, jpath)
    with open(mpath + '.tmp', 'w') as f:
        f.write(render_md(table))
    os.replace(mpath + '.tmp', mpath)
    lines = ['%s  %s' % (sha256_file(p), os.path.basename(p)) for p in (jpath, mpath)]
    with open(os.path.join(rdir, 'table.sha256.tmp'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    os.replace(os.path.join(rdir, 'table.sha256.tmp'), os.path.join(rdir, 'table.sha256'))
    return table, jpath


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--evidence', required=True)
    ap.add_argument('--attempt', type=int, required=True)
    ap.add_argument('--expected-hashes', default=None, help='dry run only')
    a = ap.parse_args(argv)
    run = load_json(os.path.join(a.evidence, 'run.json')) or {}
    if a.expected_hashes and not run.get('dry_run'):
        print('refused: --expected-hashes is for a dry run only', file=sys.stderr)
        return 2
    expected, source = load_expected(a.expected_hashes)
    table, path = write_table(a.evidence, a.attempt, expected, source)
    print('%s rows=%d failed=%d' % (path, table['header']['counts']['rows'],
                                    table['header']['counts']['rows_with_a_failed_stage']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
