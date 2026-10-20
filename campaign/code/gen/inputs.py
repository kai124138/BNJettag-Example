"""The decision-input record `gen/inputs.json`.

Every input that Kai or a later batch fixes is a key of this file with no default in the code: a
missing key is refused, and a key that is present but null is "unset", which the generator writes
into the configs as null and lists in each config's `unset_inputs` (fail closed at launch,
`gen.configs.check_launchable`). The keys:

  od13/od15/od12      true | null             Kai confirmations, 2026-10-19; unset blocks launch
  headline            "H1" | "H2" | null      STUDY v0.13 section 3.4 rule v0.4, export outcomes
  claim_domain        object | null           Kai, section 19.2 item 12 (validated by
                                              cbl.select.claim_domain_config: base, band_rule,
                                              c3r, c3r_level; interfaces section 11)
  s4_option           "A" | "B" | "C" | null  Kai, section 19.2 item 11, after S2
  s2_record           path | null             the S2 validation record (cbl.s2.s2_record output,
                                              fork-relative or absolute); fixes form and BEST
  item9               "hold" | "cap475"       section 19.2 item 9; "hold" is the STUDY's default
                                              while Kai's answer is open (section 10 "Admission")
  data_manifest       {path, sha256}          batch B2's preparation manifest and the sha256 the
                                              generator must find (refuses a mismatch)
  init_tensors        {root, platform}        decision U1: directory of the per-seed manifests of
                                              record (on kai-data) and the JAX platform string in
                                              their file name init-s<seed>-<platform>.manifest.json
  smoke_init_tensors  {root, platform}        the same for the smoke archive's own draws
  bench_epochs        int | null              epochs per benchmark run (batch B19)
  param_counts        path | null             gate G58's parameter-count record (PC)

No input exists for STUDY section 19.2 items 13-16 (unadopted; the B12 brief).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from gen.arms import ITEM9_POLICIES

FORK = Path(__file__).resolve().parents[1]
INPUTS_REL = 'gen/inputs.json'
KIND = 'cbl1016-gen-inputs'
KEYS = ('kind', 'schema', 'od13', 'od15', 'od12', 'headline', 'claim_domain', 's4_option', 's2_record', 'item9', 'data_manifest',
        'init_tensors', 'smoke_init_tensors', 'bench_epochs', 'param_counts', 'provenance')
_HEX64 = re.compile(r'^[0-9a-f]{64}$')
_PLATFORM = re.compile(r'^[a-z][a-z0-9]{0,15}$')


class InputError(ValueError):
    """gen/inputs.json is malformed or names a value the STUDY does not allow."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve(p: str, fork: Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else fork / q


def _init_block(v, name):
    if not isinstance(v, dict) or set(v) != {'root', 'platform'}:
        raise InputError(f'{name} must be {{"root": ..., "platform": ...}}')
    root, plat = v['root'], v['platform']
    if (root is None) != (plat is None):
        raise InputError(f'{name}: root and platform are set together or not at all')
    if root is not None:
        if not (isinstance(root, str) and root.startswith('/') and not root.endswith('/') and '..' not in root.split('/')):
            raise InputError(f'{name}.root must be an absolute POSIX directory without a trailing /, got {root!r}')
        if not (isinstance(plat, str) and _PLATFORM.match(plat)):
            raise InputError(f'{name}.platform must be a lower-case JAX platform string, got {plat!r}')
    return {'root': root, 'platform': plat}


def load(path: Path | None = None, fork: Path = FORK) -> dict:
    """Read and validate the input record. Returns a dict with the validated values, the file's
    sha256 and the resolved data manifest (its parsed JSON and sha256)."""
    path = Path(path) if path is not None else fork / INPUTS_REL
    raw = path.read_bytes()
    rec = json.loads(raw)
    if not isinstance(rec, dict):
        raise InputError('inputs must be a JSON object')
    missing = [k for k in KEYS if k not in rec]
    extra = sorted(set(rec) - set(KEYS))
    if missing or extra:
        raise InputError(f'inputs keys: missing {missing}, unknown {extra} (no key has a default)')
    if rec['kind'] != KIND or rec['schema'] != 1:
        raise InputError(f'inputs kind/schema must be {KIND}/1')

    out = {'sha256': hashlib.sha256(raw).hexdigest()}
    for key in ('od13', 'od15', 'od12'):
        if rec[key] is not None and rec[key] is not True:
            raise InputError(f'{key} must be true (confirmed) or null (unset)')
        out[key] = rec[key]
    h = rec['headline']
    if h not in (None, 'H1', 'H2'):
        raise InputError(f'headline must be H1, H2 or null, got {h!r}')
    out['headline'] = h

    cd = rec['claim_domain']
    if cd is not None:
        from cbl import select as sel   # NumPy/SciPy only
        try:
            out['claim_domain'] = sel.claim_domain_config(cd).as_dict()
        except sel.UnsetDecision as e:
            raise InputError(f'claim_domain: {e}') from None
    else:
        out['claim_domain'] = None

    s4 = rec['s4_option']
    if s4 not in (None, 'A', 'B', 'C'):
        raise InputError(f's4_option must be A, B, C or null, got {s4!r}')
    out['s4_option'] = s4

    if rec['item9'] not in ITEM9_POLICIES:
        raise InputError(f'item9 must be one of {ITEM9_POLICIES} (no default in the code), got {rec["item9"]!r}')
    out['item9'] = rec['item9']

    dm = rec['data_manifest']
    if not (isinstance(dm, dict) and set(dm) == {'path', 'sha256'} and isinstance(dm['path'], str)
            and isinstance(dm['sha256'], str) and _HEX64.match(dm['sha256'])):
        raise InputError('data_manifest must be {"path": str, "sha256": 64 hex}')
    mpath = _resolve(dm['path'], fork)
    got = sha256_file(mpath)
    if got != dm['sha256']:
        raise InputError(f'data manifest {dm["path"]} sha256 {got} differs from the recorded {dm["sha256"]}')
    manifest = json.loads(mpath.read_bytes())
    if manifest.get('kind') != 'cbl1016-data-prep-manifest' or manifest.get('schema') != 1:
        raise InputError('data manifest is not a cbl1016-data-prep-manifest schema 1')
    out['data_manifest'] = {'path': dm['path'], 'sha256': got, 'json': manifest}

    out['init_tensors'] = _init_block(rec['init_tensors'], 'init_tensors')
    out['smoke_init_tensors'] = _init_block(rec['smoke_init_tensors'], 'smoke_init_tensors')

    be = rec['bench_epochs']
    if be is not None and not (type(be) is int and be >= 1):
        raise InputError(f'bench_epochs must be a positive int or null, got {be!r}')
    out['bench_epochs'] = be

    s2 = rec['s2_record']
    if s2 is not None:
        if not isinstance(s2, str):
            raise InputError('s2_record must be a path or null')
        out['s2_record'] = read_s2_record(_resolve(s2, fork))
    else:
        out['s2_record'] = None

    pc = rec['param_counts']
    if pc is not None:
        if not isinstance(pc, str):
            raise InputError('param_counts must be a path or null')
        ppath = _resolve(pc, fork)
        prec = json.loads(ppath.read_bytes())
        if not (isinstance(prec, dict) and prec.get('kind') == 'cbl1016-param-counts'
                and isinstance(prec.get('counts'), dict)):
            raise InputError('param_counts is not a cbl1016-param-counts record with a counts object')
        out['param_counts'] = {'sha256': sha256_file(ppath), 'counts': prec['counts']}
    else:
        out['param_counts'] = None
    return out


def read_s2_record(path: Path) -> dict:
    """The two S2 choices the templates need: the form (section 8.3) and BEST (section 8.4), from
    the record cbl.s2.s2_record writes (`form_choice.form`, `best.best`), with the file's sha256.
    An INCOMPLETE S2 (empty D_screen, section 8.2) is refused: Kai decides before S3 or S4."""
    raw = Path(path).read_bytes()
    rec = json.loads(raw)
    if not isinstance(rec, dict) or rec.get('stage') != 'S2':
        raise InputError('s2_record is not an S2 record')
    if rec.get('status') == 'INCOMPLETE' or 'form_choice' not in rec or 'best' not in rec:
        raise InputError('s2_record is INCOMPLETE: no form or BEST; Kai decides before any S3 or S4 launch')
    form = rec['form_choice'].get('form')
    best = rec['best'].get('best')
    if form not in ('FB', 'BF'):
        raise InputError(f's2_record form must be FB or BF, got {form!r}')
    if best not in ('F100', 'F001', 'F010', 'F011', 'F101', 'F110', 'F111'):
        raise InputError(f's2_record BEST must be F100 or a fused-BN code, got {best!r}')
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'form': form, 'best': best}
