"""Gate G36 (PREFLIGHT-PLAN section 3 row G36; STUDY v0.10 sections 3.5 and 14 "missing initialization readings of
H1, H2 and NR per configuration in PREFLIGHT" blocks launch; arbiter condition P3; modules M7 and M8 through M13).
PC CPU and PC GPU (RTX 4060 Ti), venv of record, 2-file smoke archive of B2.

Initialization readings of H1, H2 and NR per configuration, recorded as engineering readings under the fork sha.
For each seed 1-3 the runner draws CH-8's initial-tensor file on this device from the smoke archive (decision U1);
for each of the 19 configurations (CH-8, F000, F100, F001-F111 in FB and BF, GR and BE in FB and BF) and each seed,
a fresh child runs the runner's prepare (data, seeds, Datasets, the arm's model, the seed's file loaded by role,
callbacks, compile; no fit) and then M8's frozen evaluation of that initial state
(gates/runner_child.py readings). Binary arms are built with headline H1 (the headline selects only the penalty
(b) branch; H1, H2 and NR read neither). On the GPU every child runs with XLA_FLAGS=--xla_gpu_deterministic_ops=true
(decision B5-1).

The state read is the initial one: kernels from the seed's file (drawn on this device from the smoke archive, so
not the draws of record, which B19 makes on NRP), BN moving moments 0 and 1, every quantizer at its constant
initializer (datalane i0 = 0 and f0 = 7, so input widths of 7 bits, before any training-mode call has moved a WRAP
integer bit). This is a different state from preflight/env/README.md's per-layer EBOPs, which were read after one
training-mode forward on the record archive's first batch; the two are not comparable (owner decision B7-9).

Checks: every child completes and loads its seed's file; the evaluation leaves the variable hash unchanged and
snap_changed is 0; H2 <= H1 <= NR in every reading; F000's readings equal CH-8's per seed. The readings table is
written to --record. Every reading is engineering, not quotable.

    <venv>/bin/python -I -B gates/g36_init_readings.py --device cpu|gpu --expect-sha12 SHA12 --work DIR \\
        --archive SMOKE_ARCHIVE --record PATH [--data-manifest PATH] [--json PATH] [--children-out DIR]
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import gatelib  # noqa: E402
import runnergate as rg  # noqa: E402

GATE = 'G36'
SEEDS = (1, 2, 3)


def configurations():
    """(label, arm, form): the 19 configurations."""
    out = [('CH-8', 'CH-8', None), ('F000', 'F000', None), ('F100', 'F100', None)]
    for code in ('F001', 'F010', 'F011', 'F101', 'F110', 'F111'):
        out += [(f'{code}-{f}', code, f) for f in ('FB', 'BF')]
    out += [(f'GR-{f}', 'GR', f) for f in ('FB', 'BF')] + [(f'BE-{f}', 'BE', f) for f in ('FB', 'BF')]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--device', choices=['cpu', 'gpu'], required=True)
    ap.add_argument('--expect-sha12', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--archive', required=True)
    ap.add_argument('--record', required=True)
    ap.add_argument('--data-manifest', default=str(FORK / 'manifests' / 'data-prep-150c.json'))
    ap.add_argument('--json', default=None)
    ap.add_argument('--children-out', default=None)
    a = ap.parse_args()
    g = gatelib.Gate(GATE)
    rec = {'header': g.header(a.expect_sha12), 'device': a.device}
    gatelib.check_venv_of_record(g)
    print('label       engineering, not quotable')
    work = Path(a.work).resolve()
    rg.fileaudit.require_unsynced(work)
    g.check(not work.exists(), f'work directory {work} does not exist yet')
    work.mkdir(parents=True)
    head = rec['header']['HEAD']
    inp = rg.inputs(a.data_manifest, a.archive)
    got = rg.sha256_file(a.archive)
    g.check(got == inp['archive_sha256'] and got not in (inp['record_archive_sha256'], inp['heldout_archive_sha256']),
            f'the archive is the smoke archive {got[:12]} (not the archive of record, not the held-out archive)')
    ph = work / 'audit-placeholder'
    ph.mkdir()
    (ph / '150c-test.h5').write_bytes(b'')
    watches = rg.watches(inp, ph)
    env_extra = {'XLA_FLAGS': rg.DETERMINISTIC_GPU_FLAG} if a.device == 'gpu' else {}
    ch = rg.Children(g, device=a.device, inp=inp, work=work, children_out=a.children_out, audit_watches=watches,
                     env_extra=env_extra)
    table = {}
    for s in SEEDS:
        d, _ = ch(f'draw-s{s}', 'draw', s, ['--out-dir', work / f'draw-s{s}' / 'init', '--place', f'{a.device.upper()} gate G36'])
        if d is None:
            continue
        for lab, arm, form in configurations():
            extra = rg.draw_args(d, head, inp['bundle_sha256'], epochs=1, arm=arm) + ['--out-dir', work / f'r-{lab}-s{s}' / 'out']
            if form:
                extra += ['--form', form]
            if arm not in ('CH-8', 'F000'):
                extra += ['--headline', 'H1']
            R, _ = ch(f'readings-{lab}-s{s}', 'readings', s, extra)
            if R is None:
                continue
            c = R['core']
            ok = (R['load']['manifest_sha256'] == d['manifest_sha256'] and c['vars_sha256_before'] == c['vars_sha256_after']
                  and c['snap_changed'] == 0 and c['H2'] <= c['H1'] <= c['NR'])
            g.check(ok, f'{lab} seed {s}: loaded draw-s{s}; H1 {c["H1"]}, H2 {c["H2"]}, NR {c["NR"]} (H2 <= H1 <= NR, snap 0, '
                        f'variables unchanged by the evaluation)')
            table.setdefault(lab, {})[s] = {'H1': c['H1'], 'H2': c['H2'], 'NR': c['NR'], 'groups': R['groups'],
                                            'val_correct': c['val_correct'], 'val_n': c['val_n'],
                                            'frozen_seconds': R['frozen_seconds']}
    for s in SEEDS:
        a8, f0 = table.get('CH-8', {}).get(s), table.get('F000', {}).get(s)
        g.check(a8 is not None and f0 is not None and all(a8[k] == f0[k] for k in ('H1', 'H2', 'NR', 'groups')),
                f'seed {s}: F000\'s readings equal CH-8\'s ({None if a8 is None else (a8["H1"], a8["H2"], a8["NR"])})')
    g.check(sum(len(v) for v in table.values()) == len(configurations()) * len(SEEDS),
            f'{sum(len(v) for v in table.values())} of {len(configurations()) * len(SEEDS)} readings')
    for lab in table:
        g.info(f'{lab} (H1, H2, NR) by seed', {s: (v['H1'], v['H2'], v['NR']) for s, v in sorted(table[lab].items())})
    out = {'kind': 'cbl1016-init-readings', 'schema': 1, 'label': 'engineering, not quotable', 'gate': GATE,
           'code_sha': head, 'device': a.device, 'archive': 'smoke', 'archive_sha256': inp['archive_sha256'],
           'state': 'initial: kernels from the seed\'s CH-8 file drawn on this device from the smoke archive; BN moving '
                    'moments 0 and 1; quantizers at their constant initializers; no training-mode call',
           'binary_headline': 'H1 (penalty branch only; the counts do not read it)', 'readings': table}
    Path(a.record).parent.mkdir(parents=True, exist_ok=True)
    Path(a.record).write_text(json.dumps(out, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    rec['readings'] = table
    g.check_clean('at end')
    return g.finish(rec, a.json)


if __name__ == '__main__':
    os.environ.pop('XLA_FLAGS', None)
    sys.exit(main())
