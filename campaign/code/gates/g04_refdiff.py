"""Gate G04 (PREFLIGHT-PLAN section 3; STUDY v0.8 section 7.3 "Reference identity"; module M2
cbl/refdiff.py).

Runs M2 in the venv of record on the PC CPU, against the snapshot `reference-code/HGQ2/src` of
the lab root and the `hgq` package installed in this interpreter (located without importing it),
and writes M2's `report.json`, `refdiff.md` and `diffs/` into --out. Passes when M2 reports no
failure: both trees re-hash to their records (code-base.txt tree line; the installed-files record
and the wheel RECORD), every cited "identical" module is identical, every cited "differ" module
differs, and every hunk of a cited file is classified in `cbl/refdiff_known.json` with no stale
entry. The behavioral differences and the line-citation results are listed as findings, which do
not fail the gate (STUDY section 7.3: a behavioral difference in a cited mechanism is a PREFLIGHT
finding). Also checks the venv of record and a clean `git status` at start and end.

Run:  <venv>/bin/python -I -B gates/g04_refdiff.py --expect-sha12 SHA12 --out DIR [--json PATH] [--lab-root DIR]
"""

import argparse
import importlib.metadata as md
import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import gatelib  # noqa: E402
from cbl import refdiff  # noqa: E402

GATE = 'G04'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--expect-sha12', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--json', default=None)
    ap.add_argument('--lab-root', default=str(gatelib.LAB_ROOT_DEFAULT))
    a = ap.parse_args()
    g = gatelib.Gate(GATE)
    rec = {'header': g.header(a.expect_sha12)}
    gatelib.check_venv_of_record(g)

    g.check(md.version('hgq2') == refdiff.INSTALLED_VERSION, 'installed hgq2 is %s' % refdiff.INSTALLED_VERSION)
    g.check('hgq' not in sys.modules, 'hgq is not imported by this gate')
    site = gatelib.site_packages_of('hgq')
    snapshot = Path(a.lab_root).resolve() / refdiff.SNAPSHOT_TREE_SOURCE
    g.info('snapshot', str(snapshot))
    g.info('site-packages', str(site))
    g.info('registry', '%s sha256 %s' % (refdiff.DEFAULT_KNOWN, refdiff.sha256_bytes(refdiff.DEFAULT_KNOWN.read_bytes())))
    rep = refdiff.evaluate(snapshot, site)
    refdiff.write_outputs(rep, a.out)
    print('')
    print(refdiff.render_md(rep))

    s, i = rep['snapshot'], rep['installed']
    g.check(s['ok'], 'snapshot tree hash %s, %d files, equals code-base.txt' % (s['tree_hash'], s['files']))
    g.check(i['listing_equals_record'], 'installed hgq/ + qkeras/ listing (%d files, %s) byte-equal to the record' % (i['files'], i['listing_sha256']))
    g.check(i['wheel_record']['ok'], 'installed files equal the wheel RECORD')
    n_ident = [c for c in rep['cited'] if c['expect'] == 'identical']
    n_diff = [c for c in rep['cited'] if c['expect'] == 'differ']
    g.check(all(c['outcome'] == 'identical' for c in n_ident), 'all %d cited "identical" modules are identical' % len(n_ident))
    g.check(all(c['outcome'] == 'differ' for c in n_diff), 'all %d cited "differ" modules differ' % len(n_diff))
    g.check(not rep['failures'], 'M2 reports no failure (%d)' % len(rep['failures']))
    for f in rep['failures']:
        g.info('M2 failure', f)
    g.info('findings', len(rep['findings']))
    for f in rep['findings']:
        g.info('finding', f)
    rec['summary'] = rep['summary']
    rec['findings'] = rep['findings']
    rec['m2_failures'] = rep['failures']
    rec['snapshot'] = s
    rec['installed'] = {k: v for k, v in i.items() if k != 'wheel_record'} | {'wheel_record_ok': i['wheel_record']['ok']}
    g.check('hgq' not in sys.modules, 'hgq was still not imported at the end')
    g.check_clean('at end')
    return g.finish(rec, a.json)


if __name__ == '__main__':
    sys.exit(main())
