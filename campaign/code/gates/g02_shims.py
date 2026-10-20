"""Gate G02 (PREFLIGHT-PLAN section 3; STUDY v0.8 section 7.3 "Two shims"; module M1 cbl/env.py).

Checks, in the venv of record on the PC CPU:
- the interpreter is `cbl1016-jax2` and its distributions equal `env/training/freeze.txt`;
- the installed hgq2 tree (hgq/ and qkeras/) equals `env/training/hgq2-0.1.9-installed-files.sha256`
  byte for byte and matches the wheel RECORD, before and after the gate's imports;
- a text search of the installed hgq package finds neither shim name (STUDY section 7.3);
- after `import hgq.utils.sugar` and `import hgq.layers`, and before binding, both names are
  absent and none of Chang's modules is imported; `cbl.env.bind_shims()` then binds exactly the
  vendored StopIf (fork `vendor/stop_if.py`, hash equal to code-base.txt) and the placeholder,
  which raises when constructed; Chang's `model.py` imports from the fork's `chang/` only after
  that, and every name of `run_train.py:13` resolves;
- negative controls, each in its own process: without the shims `import model` and the StopIf
  import fail; a pre-set StopIf or QLinformerAttentionT makes `bind_shims()` refuse; a Chang module
  imported first makes it refuse;
- the shim hashes are recorded: cbl/env.py, vendor/stop_if.py, the installed
  hgq/utils/sugar/__init__.py and hgq/layers/__init__.py, and Chang's model.py and run_train.py;
- `git status --porcelain` is empty at start and at end, and `chang/` has no __pycache__.

Run:  <venv>/bin/python -I -B gates/g02_shims.py --expect-sha12 SHA12 [--json PATH] [--lab-root DIR]
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import gatelib  # noqa: E402
from cbl import refdiff  # noqa: E402

GATE = 'G02'
NAMES_RUN_TRAIN_13 = ['BetaScheduler', 'Dataset', 'FreeEBOPs', 'ParetoFront', 'PBar', 'PieceWiseSchedule', 'StopIf']
# preflight/env/README.md lines 67-68 (B0 evidence): the two installed __init__ files
README_INIT_PREFIXES = {'hgq/utils/sugar/__init__.py': '06b80a63', 'hgq/layers/__init__.py': 'b90b1825'}

NEG_NO_SHIMS = """
import json, os, sys
os.environ['KERAS_BACKEND'] = 'jax'
sys.path.insert(0, %(chang)r)
o = {}
try:
    import model
    o['model'] = None
except ImportError as e:
    o['model'] = '%%s: %%s' %% (type(e).__name__, e)
try:
    from hgq.utils.sugar import StopIf
    o['stopif'] = None
except ImportError as e:
    o['stopif'] = '%%s: %%s' %% (type(e).__name__, e)
print('RESULT ' + json.dumps(o))
"""

NEG_PRESET = """
import json, sys, importlib
sys.path.insert(0, %(fork)r)
from cbl import env
env.set_process_env(mem_fraction='0.5', jax_platforms='cpu')
m = importlib.import_module(%(module)r)
setattr(m, %(name)r, object())
try:
    env.bind_shims()
    o = {'raised': None}
except RuntimeError as e:
    o = {'raised': str(e)}
print('RESULT ' + json.dumps(o))
"""

NEG_CHANG_FIRST = """
import json, sys
sys.path.insert(0, %(fork)r)
from cbl import env
env.set_process_env(mem_fraction='0.5', jax_platforms='cpu')
sys.path.insert(0, %(chang)r)
import data
try:
    env.bind_shims()
    o = {'raised': None}
except RuntimeError as e:
    o = {'raised': str(e)}
print('RESULT ' + json.dumps(o))
"""


def parse_code_base():
    rows = {}
    for line in (FORK / 'code-base.txt').read_text(encoding='utf-8').splitlines():
        parts = line.split('  ')
        if parts[0] == 'file' and len(parts) == 4:
            rows[parts[2]] = parts[1]
    return rows


def installed_state(site):
    files = refdiff.walk(site, refdiff.INSTALLED_TOPS)
    text = refdiff.listing(files)
    return files, text


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--expect-sha12', default=None)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    g = gatelib.Gate(GATE)
    rec = {'header': g.header(a.expect_sha12)}
    gatelib.check_venv_of_record(g)

    # installed hgq2 tree before any hgq import
    site = gatelib.site_packages_of('hgq')
    record_path = FORK / 'env' / 'training' / 'hgq2-0.1.9-installed-files.sha256'
    record_text = record_path.read_text(encoding='utf-8')
    files, text = installed_state(site)
    g.info('site-packages', str(site))
    g.info('installed listing sha256', refdiff.sha256_bytes(text.encode()))
    g.check(text == record_text, 'installed hgq/ + qkeras/ (%d files) byte-equal to %s (sha256 %s)' % (
        len(files), record_path.name, refdiff.sha256_bytes(record_text.encode())))
    wr = refdiff.check_record(site, files)
    g.check(wr['ok'], 'installed files equal the wheel RECORD (%d entries, %d mismatches)' % (wr['record_entries'], len(wr['hash_mismatch'])))

    hits = []
    for rel in sorted(files):
        if rel.startswith('hgq/') and rel.endswith(('.py', '.pyi')):
            body = (site / rel).read_text(encoding='utf-8')
            for name in ('StopIf', 'QLinformerAttentionT'):
                if name in body:
                    hits.append((rel, name))
    g.info('text search hits', hits)
    g.check(not hits, 'no installed hgq source file contains "StopIf" or "QLinformerAttentionT"')

    # hashes recorded
    cb = parse_code_base()
    hashes = {
        'cbl/env.py': refdiff.sha256_bytes((FORK / 'cbl' / 'env.py').read_bytes()),
        'vendor/stop_if.py': refdiff.sha256_bytes((FORK / 'vendor' / 'stop_if.py').read_bytes()),
        'chang/model.py': refdiff.sha256_bytes((FORK / 'chang' / 'model.py').read_bytes()),
        'chang/run_train.py': refdiff.sha256_bytes((FORK / 'chang' / 'run_train.py').read_bytes()),
        'installed hgq/utils/sugar/__init__.py': files['hgq/utils/sugar/__init__.py'],
        'installed hgq/layers/__init__.py': files['hgq/layers/__init__.py'],
    }
    for k, v in hashes.items():
        g.info('sha256 ' + k, v)
    rec['hashes'] = hashes
    for k in ('vendor/stop_if.py', 'chang/model.py', 'chang/run_train.py'):
        g.check(hashes[k] == cb.get(k), '%s equals code-base.txt (%s)' % (k, cb.get(k)))
    for rel, prefix in README_INIT_PREFIXES.items():
        g.check(files[rel].startswith(prefix), 'installed %s starts %s (preflight/env/README.md:67-68)' % (rel, prefix))

    # binding in this process, in the section 7.3 order
    from cbl import env

    g.check(hashes['vendor/stop_if.py'] == env.STOPIF_SNAPSHOT_SHA256, 'cbl.env.STOPIF_SNAPSHOT_SHA256 equals the vendored file')
    g.check(cb.get('chang/model.py') == env.CHANG_MODEL_SHA256, 'cbl.env.CHANG_MODEL_SHA256 equals code-base.txt')
    penv = env.set_process_env(mem_fraction='0.5', jax_platforms='cpu')
    g.info('process env', penv['set'])
    import hgq.layers as hl
    import hgq.utils.sugar as hs

    before = {'StopIf': hasattr(hs, 'StopIf'), 'QLinformerAttentionT': hasattr(hl, 'QLinformerAttentionT')}
    chang_loaded = [n for n in env.CHANG_MODULE_NAMES if n in sys.modules]
    g.check(not any(before.values()), 'before binding, hgq.utils.sugar.StopIf and hgq.layers.QLinformerAttentionT are absent %s' % before)
    g.check(not chang_loaded, "before binding, none of Chang's modules is imported %s" % chang_loaded)
    shims = env.bind_shims()
    rec['shims'] = shims
    g.info('bind record', shims)
    vend = sys.modules.get(env.VENDORED_MODULE_NAME)
    g.check(hs.StopIf is getattr(vend, 'StopIf', None) and Path(vend.__file__).resolve() == (FORK / 'vendor' / 'stop_if.py').resolve(),
            'hgq.utils.sugar.StopIf is the class from the fork vendor/stop_if.py')
    g.check(shims['stopif_vendored_sha256'] == cb.get('vendor/stop_if.py'), 'bound StopIf source sha256 equals code-base.txt')
    g.check(hl.QLinformerAttentionT is env.QLinformerAttentionT, 'hgq.layers.QLinformerAttentionT is the placeholder')
    try:
        hl.QLinformerAttentionT(2, 4, 16)
        raised = None
    except RuntimeError as e:
        raised = str(e)
    g.check(raised is not None, 'constructing the placeholder raises RuntimeError')
    g.check(shims['shim_file_sha256'] == hashes['cbl/env.py'], 'bind record holds the sha256 of cbl/env.py')
    g.check('model' not in sys.modules, "Chang's model is still not imported after binding")
    model = env.import_chang_model()
    g.check(Path(model.__file__).resolve() == (FORK / 'chang' / 'model.py').resolve(), "model imported from the fork's chang/model.py")
    ns = {}
    exec('from hgq.utils.sugar import ' + ', '.join(NAMES_RUN_TRAIN_13), ns)
    g.check(all(n in ns for n in NAMES_RUN_TRAIN_13), 'run_train.py:13 import line resolves: %s' % ', '.join(NAMES_RUN_TRAIN_13))
    g.check(ns['StopIf'] is hs.StopIf, 'the StopIf that run_train.py:13 receives is the vendored one')
    prec = env.set_matmul_precision()
    g.info('matmul precision', prec)

    # negative controls
    chang = str(FORK / 'chang')
    rc, out, err = gatelib.subprocess_python(NEG_NO_SHIMS % {'chang': chang})
    o = gatelib.result_line(out) or {}
    g.info('no-shim run', o)
    g.check(rc == 0 and o.get('model') and 'QLinformerAttentionT' in o['model'],
            'without the shims, import model raises ImportError naming QLinformerAttentionT')
    g.check(rc == 0 and o.get('stopif') and 'StopIf' in o['stopif'], 'without the shims, the StopIf import raises ImportError')
    for module, name, word in (('hgq.utils.sugar', 'StopIf', 'shim 1'), ('hgq.layers', 'QLinformerAttentionT', 'shim 2')):
        rc, out, err = gatelib.subprocess_python(NEG_PRESET % {'fork': str(FORK), 'module': module, 'name': name})
        o = gatelib.result_line(out) or {}
        g.info('preset %s.%s' % (module, name), o)
        g.check(rc == 0 and o.get('raised') and 'would shadow' in o['raised'] and word in o['raised'],
                'bind_shims refuses when %s.%s already exists' % (module, name))
    rc, out, err = gatelib.subprocess_python(NEG_CHANG_FIRST % {'fork': str(FORK), 'chang': chang})
    o = gatelib.result_line(out) or {}
    g.info('chang-first run', o)
    g.check(rc == 0 and o.get('raised') and 'already imported' in o['raised'], "bind_shims refuses once one of Chang's modules is imported")

    # nothing changed
    files2, text2 = installed_state(site)
    g.check(text2 == record_text, 'installed hgq/ + qkeras/ still byte-equal to the record at the end')
    g.check(not (FORK / 'chang' / '__pycache__').exists(), 'chang/ has no __pycache__')
    g.check_clean('at end')
    return g.finish(rec, a.json)


if __name__ == '__main__':
    sys.exit(main())
