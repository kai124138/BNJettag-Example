"""Gate G60 (PREFLIGHT-PLAN section 3 row G60; STUDY v0.10 section 13 arms row "shared initialization;
callback order", section 6 steps 3-4, section 7.1; modules M17 gen/ and M13 cbl/runner.py). PC CPU, venv
of record, 2-file smoke archive of B2.

Shared initialization and callback order per arm, through the runner. For every config that
cbl.runner.validate_config admits under the constructed complete input record of G57 (66 configs; 88 since B9; 109 since
B10), one
fresh child (gates/config_child.py prep) runs cbl.runner.prepare on the smoke archive: data, seeds, the
two Datasets, the arm's model, the seed's CH-8 initial-tensor file loaded by role, callbacks and compile
(no fit). The seed's file is drawn once per seed by the runner (cbl.runner.draw, decision U1) on the CPU
from the smoke archive, as G10-G14 do.

The child receives a gate copy of the generated config (cbl gen output under the complete record), with
only these fields changed, and the change recorded: code_sha (this HEAD) and init_tensors (the seed's
local manifest path and sha256), as gen.configs.with_handoff makes a launch copy; wandb.mode `disabled`
and wandb.run_name null (M15 accepts a null stem); and, for S1 configs (since B9 also S3 and S4), stage
`gate`, because the runner admits the smoke archive in engineering stages only (owner decision B7-12).
Nothing that enters the build, the initialization or the callbacks is changed.

Checks, per config: prepare completes; the load record names the seed's manifest; the matched roles plus
the unmatched roles are the arm's 244 variables; every matched role's tensor after the load equals the
file's tensor (sha256 by role); the unmatched roles are exactly q_g's k, b and i at the arm's binary sites
with their constant initializers (k = 1, b = 4, i = 0), and none for CH-8 and F000; the callback order is
the released order with the frozen-evaluation callback appended (STUDY 7.1, run_train.py:95), with the lab
PID at BetaScheduler's position in PID configs (STUDY 7.2; since B9), and it is the config's `callbacks` list. Per seed: every role shared by two arms holds the same tensor in both
(shared initialization across all admitted arms of the seed). The children run under the kernel
open-file audit of the held-out paths. Every reading is engineering, not quotable.

    <venv>/bin/python -I -B gates/g60_init_callbacks.py --expect-sha12 SHA12 --work DIR --archive SMOKE_ARCHIVE \\
        [--data-manifest PATH] [--json PATH] [--children-out DIR]
"""

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import fileaudit  # noqa: E402
import g57_configs as G57  # noqa: E402
import gatelib  # noqa: E402
import runner_child  # noqa: E402
import runnergate  # noqa: E402
from gen import configs as C  # noqa: E402
from gen import inputs as I  # noqa: E402

GATE = 'G60'
N_VARIABLES = 244
RELEASED = ['FreeEBOPs', 'LearningRateScheduler', 'BetaScheduler', 'PBar', 'ParetoFront', 'StopIf']
PID_RELEASED = ['FreeEBOPs', 'LearningRateScheduler', 'LabBetaPID', 'PBar', 'ParetoFront', 'StopIf']   # STUDY 7.2 (B9)
APPENDED = 'FrozenEvalHook'
CONFIG_CALLBACK_NAME = {'lab:frozen_evaluation': APPENDED, 'lab:BetaPID': 'LabBetaPID'}
QG_CONST = {'k': 1, 'b': 4, 'i': 0}           # STUDY 2.2: q_g's k0 = 1, b0 = 4, i0 = 0 (constant initializers)


def flatten(v):
    """A nested list (or a scalar) as a flat list."""
    if not isinstance(v, list):
        return [v]
    return [y for x in v for y in flatten(x)]


def binary_site_prefixes(cfg):
    from cbl import model as M

    if cfg['builder'] == 'released':
        return []
    spec = M.arm_spec(cfg['arm_id'], groups=cfg['groups'], embedding=cfg['embedding'], form=cfg['form'],
                      control=cfg['control'], headline=cfg.get('headline'))
    return M.binary_sites(spec)


def gate_copy(cfg, *, head, manifest, manifest_sha):
    out = C.with_handoff(cfg, code_sha=head, init_manifest_sha256=manifest_sha)
    out['init_tensors']['manifest_path'] = str(manifest)
    out['wandb'] = dict(out['wandb'], mode='disabled', run_name=None)
    changed = ['code_sha', 'init_tensors.manifest_path', 'init_tensors.manifest_sha256', 'wandb.mode', 'wandb.run_name']
    if out['stage'] in ('S1', 'S3', 'S4'):
        changed.append(f'stage {out["stage"]} -> gate')
        out['stage'] = 'gate'
    return out, changed


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
    work = Path(a.work).resolve()
    fileaudit.require_unsynced(work)
    g.check(not work.exists(), f'work directory {work} does not exist yet')
    work.mkdir(parents=True)
    head = rec['header']['HEAD']
    inp_s = runnergate.inputs(a.data_manifest, a.archive)
    got = runnergate.sha256_file(a.archive)
    g.check(got == inp_s['archive_sha256'] and got not in (inp_s['record_archive_sha256'], inp_s['heldout_archive_sha256']),
            f'the archive is the smoke archive {got[:12]} (not the archive of record, not the held-out archive)')
    ph = work / 'audit-placeholder'
    ph.mkdir()
    (ph / '150c-test.h5').write_bytes(b'')
    watches = runnergate.watches(inp_s, ph)

    inp = I.load()
    cfgs = C.build_all(inp)
    with tempfile.TemporaryDirectory(dir=str(work)) as td:
        full = C.build_all(G57.complete_inputs(Path(td), cfgs))
    from cbl import runner
    admitted = []
    for c in full:
        try:
            runner.validate_config(C.with_handoff(c, code_sha=head, init_manifest_sha256='ab' * 32),
                                   data_manifest_path=Path(a.data_manifest))
            admitted.append(c)
        except runner.ConfigRefused:
            pass
    g.check(len(admitted) == G57.N_PASS and sorted(C.config_path(c) for c in admitted)
            == sorted(C.config_path(c) for c in full if G57.expected_pass(c)),
            f'{len(admitted)} admitted configs under the constructed complete record, the set G57 lists')
    seeds = sorted({c['seed'] for c in admitted})
    g.info('seeds', seeds)

    draws = {}
    for s in seeds:
        code, d, argv, audit = runner_child.run_child(
            f'draw-s{s}', 'draw', device='cpu', seed=s, archive=a.archive, archive_sha=inp_s['archive_sha256'],
            data_manifest_sha=inp_s['data_manifest_sha256'], work=work / f'draw-s{s}', children_out=a.children_out,
            extra=['--out-dir', work / f'draw-s{s}' / 'init', '--place', 'PCC gate G60'], watches=watches)
        g.info(f'draw-s{s} argv', ' '.join(argv[3:]))
        ok = code == 0 and d is not None and 'error' not in d
        g.check(ok, f'draw-s{s}: CH-8 initial-tensor file drawn on the CPU from the smoke archive, exit {code}')
        fe = runnergate.forbidden_events(audit)
        g.check(not fe, f'draw-s{s}: open-file audit, no open of the held-out archive or any raw held-out file')
        if ok:
            man = json.loads(Path(d['manifest']).read_text(encoding='utf-8'))
            draws[s] = {'manifest': d['manifest'], 'sha256': d['manifest_sha256'],
                        'roles': {r: v['sha256'] for r, v in man['roles'].items()}}
    if len(draws) != len(seeds):
        g.check_clean('at end')
        return g.finish(rec, a.json)

    env = {k: v for k, v in os.environ.items() if k not in ('KERAS_BACKEND', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_FLAGS')}
    per_seed = {s: {} for s in seeds}
    rows = []
    for c in admitted:
        s = c['seed']
        gc, changed = gate_copy(c, head=head, manifest=draws[s]['manifest'], manifest_sha=draws[s]['sha256'])
        name = C.config_path(c).replace('configs/', '').replace('/', '_').replace('.json', '')
        wdir = work / name
        wdir.mkdir()
        cpath = wdir / 'gate-copy.json'
        cpath.write_text(json.dumps(gc, indent=1, sort_keys=True))
        out = wdir / 'prep.json'
        argv = [sys.executable, '-I', '-B', str(FORK / 'gates' / 'config_child.py'), 'prep', '--config', str(cpath),
                '--archive', a.archive, '--out-dir', str(wdir / 'run'), '--source-bundle', inp_s['bundle_sha256'],
                '--out', str(out)]
        r = fileaudit.run_audited(argv, watches=watches, env=env, cwd=str(wdir), timeout=1800)
        (wdir / 'prep.stdout.txt').write_text(r['stdout'])
        (wdir / 'prep.stderr.txt').write_text(r['stderr'])
        audit = {k: r[k] for k in ('events', 'events_before_start', 'fd_paths')}
        child = json.loads(out.read_text()) if out.exists() else None
        if a.children_out:
            co = Path(a.children_out) / name
            co.mkdir(parents=True, exist_ok=True)
            for src in (out, cpath, wdir / 'prep.stdout.txt', wdir / 'prep.stderr.txt'):
                if src.exists():
                    shutil.copy2(src, co / src.name)
            (co / 'prep.audit.json').write_text(json.dumps(audit, indent=1, sort_keys=True) + '\n')
        label = f'{C.config_path(c)}'
        ok = r['returncode'] == 0 and child is not None and 'error' not in child and 'refused' not in child
        g.check(ok, f'{label}: prepare completes through the runner (gate copy changes {changed})' +
                ('' if ok else f' ({(child or {}).get("error") or (child or {}).get("refused")})'))
        fe = runnergate.forbidden_events(audit)
        g.check(not fe, f'{label}: open-file audit, no open of the held-out archive or any raw held-out file')
        if not ok:
            continue
        load = child['load']
        file_roles = draws[s]['roles']
        matched = sorted(set(child['roles_sha256']) & set(file_roles))
        sites = binary_site_prefixes(c)
        want_unmatched = sorted(f'{p}/qg/FixedPointQuantizerKBI[0]/{v}' for p in sites for v in ('k', 'b', 'i'))
        want_unused = sorted(f'{p}/kq/FixedPointQuantizerKBI[0]/{v}' for p in sites for v in ('k', 'b', 'i'))
        uv = child['unmatched_values']
        const_ok = all(all(x == QG_CONST[k.rsplit('/', 1)[1]] for x in flatten(v)) for k, v in uv.items())
        checks = {
            'manifest': load['manifest_sha256'] == draws[s]['sha256'],
            'count': child['n_roles'] == N_VARIABLES and load['n_loaded'] + len(load['unmatched']) == N_VARIABLES,
            'matched_equal_file': all(child['roles_sha256'][x] == file_roles[x] for x in matched)
                                   and len(matched) == load['n_loaded'],
            'unmatched': sorted(load['unmatched']) == want_unmatched and const_ok,
            'unused': sorted(load['unused']) == want_unused,
            'callbacks': child['callback_order'] == (PID_RELEASED if c['schedule'] == 'pid' else RELEASED) + [APPENDED]
                         and [CONFIG_CALLBACK_NAME.get(x, x) for x in c['callbacks']] == child['callback_order'],
        }
        g.check(all(checks.values()), f'{label}: {load["n_loaded"]} roles loaded from draw-s{s} equal the file, '
                                      f'{len(load["unmatched"])} unmatched (q_g k, b, i at {len(sites)} sites, constants), '
                                      f'{len(load["unused"])} unused (kq), callbacks {child["callback_order"]}'
                + ('' if all(checks.values()) else f' FAILED {[k for k, v in checks.items() if not v]}'))
        for x in matched:
            per_seed[s].setdefault(x, set()).add(child['roles_sha256'][x])
        rows.append({'config': C.config_path(c), 'label': c['label'], 'stage': c['stage'], 'seed': s, 'changed': changed,
                     'n_loaded': load['n_loaded'], 'unmatched': len(load['unmatched']), 'unused': len(load['unused']),
                     'callback_order': child['callback_order'], 'loaded_sha256': load['loaded_sha256']})
    rec['rows'] = rows
    for s in seeds:
        n = sum(1 for r in rows if r['seed'] == s)
        split = sorted(x for x, v in per_seed[s].items() if len(v) != 1)
        g.check(n > 0 and not split, f'seed {s}: across its {n} admitted configs, every shared role holds one tensor '
                                     f'({len(per_seed[s])} roles, {len(split)} differing)')
    g.check(len(rows) == len(admitted), f'{len(rows)} of {len(admitted)} admitted configs checked')
    g.check_clean('at end')
    return g.finish(rec, a.json)


if __name__ == '__main__':
    sys.exit(main())
