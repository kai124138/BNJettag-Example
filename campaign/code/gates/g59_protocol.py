"""Gate G59 (PREFLIGHT-PLAN section 3 row G59, section 4 row B12; 03-phases owner protocol step 6).

The protocol derived from the configs shows no drift against the frozen protocol, checked with
Jev's deterministic function `inspect_protocol(path, baseline_path)` (lab-root
tools/jev/methods.py, read-only) called in this process, because the jev-lab server is down; the
dataset identity hash is filled.

The protocol reference is arbiter P5's `review/snapshots/v0.8/protocol-draft.json` (sha256
8f48e716...; v0.9 and v0.10 did not change it). It is a raw protocol, not a `lab_freeze_protocol`
snapshot, and `inspect_protocol` refuses a raw protocol as `baseline_path` (shown below). So the
gate builds, in this process, the record `freeze_protocol` would build around it
({kind: training-methods-v1, protocol: <reference>, config_sources: {}, policy: policy_snapshot()},
with its digest), writes it into the evidence directory, and passes that file as `baseline_path`.
Nothing is written under local/jev-protocols; freeze_protocol is never called. The policy part is
taken now, so it cannot show a methodology change since the STUDY froze.

Credential isolation: Jev's `read_text` compares every text with the TypeSafe key, which
`common.credential()` reads from $TYPESAFE_API_KEY or ~/.typesafe.env. The gate refuses to run
unless that variable is unset and $HOME holds no .typesafe.env, so no key is ever read.

Checks, on the Mac CPU with no backend:
  - repository: HEAD equals --expect-sha12, clean at start and end;
  - the reference hashes to 8f48e716...; the derived protocols regenerated now equal the
    committed configs/protocol/*.json;
  - screen: inspect_protocol(derived screen, wrapper) has no structural finding and its drift is
    exactly the two dataset.identity_sha256 entries (placeholder -> data manifest sha256);
  - option A claim object against a wrapper around the reference's confirmation_stage: drift only
    dataset.identity_sha256; options B and C: no structural finding, and drift against the
    reference's confirmation_stage with B's or C's arms (the STUDY author's in-memory variants,
    v0.8 plan.md:137-140) only dataset.identity_sha256;
  - with the identity hash reverted to the placeholder, the derived screen is byte-identical to
    the reference file, and the digests equal the STUDY author's recorded ones (screen 3b25814b...,
    A c16e0100..., B 70a702a9..., C cdb34eaa...);
  - dataset identity: identity_sha256 = sha256 of manifests/data-prep-150c.json = 686b15db...7f13;
    the counts the dataset text states (558,000, 62,000, 260,000, 26 files) equal the manifest's;
  - negative controls: one changed config, and one changed prose field, each show as drift.

Run as

    env -u TYPESAFE_API_KEY HOME=<empty dir> python -I -B gates/g59_protocol.py --expect-sha12 SHA12 \
        --lab-root LAB --reference PATH --plan-v08 PATH --evidence DIR

DIR must lie inside LAB (Jev reads only lab paths). Ends with "G59 PASS" (exit 0) or "G59 FAIL".
Every reading is engineering, not quotable.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I does not add the script directory
import gatelib  # noqa: E402

FORK = gatelib.FORK
sys.path.insert(0, str(FORK))
from gen import configs as C  # noqa: E402
from gen import inputs as I  # noqa: E402
from gen import protocol as P  # noqa: E402

REFERENCE_SHA256 = '8f48e716333a4ffb5ca9ae36012480bf6a3484aadc55545ecfdb00fd08d302c8'
DATA_MANIFEST_SHA256 = '686b15db996010cc5eadb7556397488c92ce3b48c2bccdb2a0d947f1011a7f13'
PLACEHOLDER = 'PENDING-PREFLIGHT-DATA-MANIFEST'
# v0.8 plan.md:138-139 ("Protocol check (v0.8, in memory, nothing written")
AUTHOR_DIGESTS = {'screen': '3b25814bdba31978566021b6610a8d668521c598465531111ce721636525e46d',
                  'claim-A': 'c16e01003758294cfa3fe981a6a3b4e5f0cf1751ec83d76ad1db1ef7855e0f49',
                  'claim-B': '70a702a9c86db7b446109dd28be9856680662dbdc31e0a876fc407fef422cb5c',
                  'claim-C': 'cdb34eaa09dca71edd76c76ec0ef70f5f9c1270b61b6c4b3c63256d47247b7a2'}


def revert_identity(obj):
    o = copy.deepcopy(obj)
    o['dataset']['identity_sha256'] = PLACEHOLDER
    if 'confirmation_stage' in o:
        o['confirmation_stage']['dataset']['identity_sha256'] = PLACEHOLDER
    return o


def identity_drift(prefixes):
    return sorted([{'field': f'{p}dataset.identity_sha256', 'before': PLACEHOLDER, 'after': DATA_MANIFEST_SHA256}
                   for p in prefixes], key=lambda x: x['field'])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--expect-sha12', required=True)
    ap.add_argument('--lab-root', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--plan-v08', required=True, help='review/snapshots/v0.8/plan.md (the author digests)')
    ap.add_argument('--evidence', required=True)
    a = ap.parse_args(argv)
    lab = Path(a.lab_root).resolve()
    ev = Path(a.evidence).resolve()
    g = gatelib.Gate('G59')
    g.header(a.expect_sha12)

    home_key = Path(os.environ.get('HOME', '/nonexistent')) / '.typesafe.env'
    if not g.check('TYPESAFE_API_KEY' not in os.environ and not home_key.exists(),
                   'credential isolation: TYPESAFE_API_KEY unset and $HOME has no .typesafe.env (no key is read)'):
        return g.finish()
    g.check(ev.is_relative_to(lab), f'evidence directory lies inside the lab root ({ev.relative_to(lab) if ev.is_relative_to(lab) else ev})')
    ev.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(lab / 'tools'))
    from jev import common as JC  # noqa: E402
    from jev import methods as JM  # noqa: E402
    for mod in (JM, JC):
        f = Path(mod.__file__).resolve()
        g.info(f'{f.relative_to(lab)} sha256', hashlib.sha256(f.read_bytes()).hexdigest())
    g.check(Path(JM.__file__).resolve() == lab / 'tools' / 'jev' / 'methods.py' and JC.ROOT.resolve() == lab,
            'jev.methods is the lab-root tools/jev/methods.py and its ROOT is the lab root')

    ref_raw = Path(a.reference).read_bytes()
    g.check(hashlib.sha256(ref_raw).hexdigest() == REFERENCE_SHA256, 'protocol reference sha256 equals 8f48e716...')
    ref = json.loads(ref_raw)

    inp = I.load()
    files = C.generate(inp)
    names = ('screen', 'claim-A', 'claim-B', 'claim-C')
    derived = {n: json.loads(files[f'configs/protocol/{n}.json']) for n in names}
    same = all((FORK / f'configs/protocol/{n}.json').read_bytes() == files[f'configs/protocol/{n}.json'] for n in names)
    g.check(same, 'the protocols derived now equal the committed configs/protocol/{screen,claim-A,claim-B,claim-C}.json')

    def rel(p):
        return str(Path(p).resolve().relative_to(lab))

    def write(name, obj, raw=None):
        p = ev / name
        p.write_bytes(raw if raw is not None else P.dumps(obj))
        return rel(p)

    def wrapper(name, protocol_obj):
        record = {'kind': 'training-methods-v1', 'protocol': protocol_obj, 'config_sources': {},
                  'policy': JC.policy_snapshot(lab)}
        value = {'sha256': JC.digest(record), 'record': record}
        p = write(name, None, JC.canonical(value) + b'\n')
        g.info(f'{name} sha256 (wrapper record digest)', value['sha256'])
        return p

    # the reference itself is not accepted as baseline_path
    try:
        JM.inspect_protocol(rel(a.reference), rel(a.reference), root=lab)
        raw_refused = False
    except JM.LabError as e:
        raw_refused = 'snapshot hash' in str(e)
        g.info('raw reference as baseline_path', f'LabError: {e}')
    g.check(raw_refused, 'inspect_protocol refuses the raw reference as baseline_path (why the wrapper is built)')

    base_screen = wrapper('baseline-wrapper-screen.json', ref)
    base_conf = wrapper('baseline-wrapper-confirmation-A.json', ref['confirmation_stage'])

    # screen
    p_screen = write('derived-screen.json', None, files['configs/protocol/screen.json'])
    res = JM.inspect_protocol(p_screen, base_screen, root=lab)
    (ev / 'inspect-screen.json').write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    g.info('screen', {k: res[k] for k in ('protocol_sha256', 'structural_valid', 'matches_snapshot', 'scientific_gate',
                                          'launch_authorized')})
    for x in res['drift']:
        g.info('screen drift', x)
    g.check(res['structural_valid'] and res['findings'] == [], 'screen: no structural or policy finding')
    g.check(sorted(res['drift'], key=lambda x: x['field']) == identity_drift(['', 'confirmation_stage.']),
            f'screen: drift is exactly dataset.identity_sha256 and confirmation_stage.dataset.identity_sha256, '
            f'{PLACEHOLDER} -> 686b15db...7f13 ({len(res["drift"])} entries)')
    g.check(res['matches_snapshot'] is False, 'matches_snapshot is false, as it must be while the filled identity hash differs from the placeholder')

    # option A against the reference's confirmation stage
    p_a = write('derived-claim-A.json', None, files['configs/protocol/claim-A.json'])
    ra = JM.inspect_protocol(p_a, base_conf, root=lab)
    (ev / 'inspect-claim-A.json').write_text(json.dumps(ra, indent=1, sort_keys=True) + '\n')
    g.check(ra['structural_valid'] and ra['drift'] == identity_drift(['']),
            f'claim A (20 arms): no structural finding; drift only dataset.identity_sha256 ({len(ra["drift"])} entry)')

    # options B and C against the author's in-memory variants
    for fam in ('B', 'C'):
        # the author's construction: the option-A object with only the arms replaced
        variant = copy.deepcopy(ref['confirmation_stage'])
        variant['arms'] = copy.deepcopy(revert_identity(derived[f'claim-{fam}'])['arms'])
        g.check(JC.digest(variant) == AUTHOR_DIGESTS[f'claim-{fam}'],
                f'claim {fam}: option A object with {fam}\'s derived arms digests to the author\'s {AUTHOR_DIGESTS[f"claim-{fam}"][:8]}... (plan.md v0.8 :139)')
        b = wrapper(f'baseline-wrapper-confirmation-{fam}.json', variant)
        pf = write(f'derived-claim-{fam}.json', None, files[f'configs/protocol/claim-{fam}.json'])
        rf = JM.inspect_protocol(pf, b, root=lab)
        (ev / f'inspect-claim-{fam}.json').write_text(json.dumps(rf, indent=1, sort_keys=True) + '\n')
        n_arms = len(derived[f'claim-{fam}']['arms'])
        g.check(rf['structural_valid'] and rf['drift'] == identity_drift(['']),
                f'claim {fam} ({n_arms} arms): no structural finding (claim scope: >= 3 paired seeds per group, '
                f'interval method); drift only dataset.identity_sha256')

    # byte identity and digests with the placeholder restored
    rs = revert_identity(derived['screen'])
    g.check(P.dumps(rs) == ref_raw, f'with the identity hash reverted, the derived screen is byte-identical to the reference ({len(ref_raw)} bytes)')
    plan = Path(a.plan_v08).read_text(encoding='utf-8')
    for n in names:
        dg = JC.digest(revert_identity(derived[n]))
        g.check(dg == AUTHOR_DIGESTS[n] and AUTHOR_DIGESTS[n] in plan,
                f'{n}: digest with the placeholder restored {dg[:12]}... equals the STUDY author\'s recorded digest (plan.md v0.8)')

    # dataset identity
    mf = FORK / 'manifests' / 'data-prep-150c.json'
    man = json.loads(mf.read_bytes())
    g.check(derived['screen']['dataset']['identity_sha256'] == hashlib.sha256(mf.read_bytes()).hexdigest() == DATA_MANIFEST_SHA256,
            'dataset.identity_sha256 = sha256(manifests/data-prep-150c.json) = 686b15db...7f13 (filled)')
    ds = derived['screen']['dataset']
    counts = (man['split']['n_train'], man['split']['n_val'], man['heldout']['archive']['rows'], len(man['heldout']['files']))
    g.info('manifest counts (n_train, n_val, held-out rows, held-out files)', counts)
    g.check(counts == (558000, 62000, 260000, 26) and '558,000' in ds['train_split'] and '62,000' in ds['validation_split']
            and '260,000' in ds['test_split'] and '26-file' in ds['test_split'],
            'the counts the dataset text states equal the data manifest (558,000 / 62,000 / 260,000 jets, 26 files; counts only)')
    g.check([m['n'] for m in derived['screen']['metrics']] == [260000, 260000, 62000, 260000],
            'metric n from the manifest: 260,000 held-out rows, 62,000 validation jets')

    # negative controls
    cfgs = C.build_all(inp)
    victim = next(c for c in cfgs if c['label'] == 'F011-BF' and c['seed'] == 2)
    victim['epochs'] = 6999
    bad = P.derive_all(cfgs, inp['data_manifest']['json'])['screen']
    pn = write('negative-control-config.json', bad)
    rn = JM.inspect_protocol(pn, base_screen, root=lab)
    extra = [x['field'] for x in rn['drift'] if 'identity_sha256' not in x['field']]
    changed = [(x['name'], y['schedule']['epochs'], y['changed_factors'])
               for x, y in zip(derived['screen']['arms'], bad['arms']) if x != y]
    g.info('negative control (config): drift fields beyond the identity hash', extra)
    g.info('negative control (config): arms that differ', changed)
    g.check(extra == ['arms'] and changed == [('F011-BF-s2', 6999, ['quantization', 'schedule.epochs'])],
            'negative control: F011-BF seed 2 with 6,999 epochs shows as drift in arms (the check can fail)')
    bad2 = copy.deepcopy(derived['screen'])
    bad2['stop_rules'][0] = bad2['stop_rules'][0] + ' (edited)'
    pn2 = write('negative-control-prose.json', bad2)
    rn2 = JM.inspect_protocol(pn2, base_screen, root=lab)
    extra2 = [x['field'] for x in rn2['drift'] if 'identity_sha256' not in x['field']]
    g.check(extra2 == ['stop_rules'], f'negative control: an edited stop rule shows as drift ({extra2})')
    g.check_clean('at end')
    return g.finish()


if __name__ == '__main__':
    sys.exit(main())
