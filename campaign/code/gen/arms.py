"""The arm catalogue of STUDY v0.10 section 5, in the order of the protocol reference.

Every run the campaign can train is an `ArmSpec` plus a seed (and, for the STUDY section 13
determinism record, a replica). The catalogue holds identities only; `gen.configs` turns each into
a config. Nothing here is a design choice beyond what the STUDY states, except the engineering
sets (smoke and benchmark arm classes), which are owner decision B12-3 in the B12 rows.

Screen (seeds 1-3, section 5 table and section 10 tiers):
  S1  CH-8 (T0); F000, F100 (T1); F001, F010, F011, F101, F110, F111 each -FB and -BF (T1): 45 runs
  S3  PID-CH, PID-BEST (T1); FP32 (T2), W8A8 (T3), DS-HGQ (T4), GR (T5), BE (T6): 21 runs
      (controls in S3 under the section 19.2 item 9 default; under the 475 GPU-h alternative FP32,
      W8A8 and DS-HGQ move to S1, section 5 rows and section 10 "Admission")
S4 templates (section 8.7 table): A = CH-8-C, BEST-C, PID-CH-C, PID-BEST-C on seeds 4-8 (20);
  B = the same four on seeds 4-11 (32); C = CH-8-C and BEST-C on seeds 4-11 (16).
Templates whose model is fixed only by S2 (section 8.3 form, section 8.4 BEST): PID-BEST, GR, BE,
  BEST-C, PID-BEST-C.
Engineering: one smoke and one benchmark identity per arm class; the determinism record's two
  CH-8 seed-1 runs and one F000 seed-1 run (STUDY section 13 "Further required gates").
Not generated: STUDY section 19.2 items 13-16 (unadopted), and the arms section 5 lists as not
  applicable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

GROUPS = ('attention', 'ffn', 'head')          # bit order of the code, section 5 "Codes"
FUSED_CODES = ('F001', 'F010', 'F011', 'F101', 'F110', 'F111')   # reference order
FORMS = ('FB', 'BF')
SCREEN_SEEDS = (1, 2, 3)
S4_SEEDS = {'A': tuple(range(4, 9)), 'B': tuple(range(4, 12)), 'C': tuple(range(4, 12))}  # section 8.7 table
S4_ARMS = {'A': ('CH-8-C', 'BEST-C', 'PID-CH-C', 'PID-BEST-C'),
           'B': ('CH-8-C', 'BEST-C', 'PID-CH-C', 'PID-BEST-C'),
           'C': ('CH-8-C', 'BEST-C')}
CONTROLS = ('FP32', 'W8A8', 'DS-HGQ', 'GR', 'BE')

# templates: what S2 fixes for the arm
T_BEST = 'BEST'                   # BEST (section 8.4) in the chosen form (section 8.3)
T_F111_CHOSEN = 'F111-chosen-form'  # F111 in the chosen form (GR, BE)

ITEM9_POLICIES = ('hold', 'cap475')


def groups_of(code: str) -> dict:
    if not (isinstance(code, str) and len(code) == 4 and code[0] == 'F' and set(code[1:]) <= {'0', '1'}):
        raise ValueError(f'not a configuration code: {code!r}')
    return {g: ('B' if b == '1' else 'Q') for g, b in zip(GROUPS, code[1:])}


def has_fused_bn_binary(code: Optional[str], embedding: Optional[str] = None) -> bool:
    """A fused-BN binary site: FFN expansion (ffn group) or the three hidden-head layers (head
    group), or the BE embedding (section 2.3 "Sites"; section 5 BE)."""
    if embedding == 'B':
        return True
    if code is None:
        return False
    g = groups_of(code)
    return g['ffn'] == 'B' or g['head'] == 'B'


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    label: str                    # protocol group and file stem: CH-8, F000, F001-FB, PID-BEST, ...
    arm_class: str                # training-code path (owner decision B12-3)
    builder: str                  # 'released' or 'lab'
    schedule: str                 # 'ramp' or 'pid'
    config_code: Optional[str] = None
    form: Optional[str] = None
    control: Optional[str] = None
    embedding: Optional[str] = None
    template: Optional[str] = None
    stage: Optional[str] = None
    tier: Optional[str] = None
    admission: Optional[str] = None
    s4_family: Optional[str] = None
    engineering_fixed: bool = False   # smoke and bench: a template's model fixed by definition


def _released(arm_id, label, schedule, cls):
    return ArmSpec(arm_id, label, cls, 'released', schedule)


def _binary_class(code, form):
    if code == 'F000':
        return 'lab-hgq'
    if not has_fused_bn_binary(code):
        return 'binary-nonbn'
    return f'binary-{form}'


def screen_specs(item9: str) -> Tuple[ArmSpec, ...]:
    """The 22 screen arms in protocol-reference order (CH-8, F000, F100, the six fused-BN codes
    with FB before BF, PID-CH, PID-BEST, FP32, W8A8, DS-HGQ, GR, BE)."""
    if item9 not in ITEM9_POLICIES:
        raise ValueError(f'item9 must be one of {ITEM9_POLICIES}, got {item9!r}')
    early = 'S1' if item9 == 'cap475' else 'S3'
    adm_c = 'all-controls-cap475' if item9 == 'cap475' else 'control-slot'
    out = [replace(_released('CH-8', 'CH-8', 'ramp', 'released-ramp'), stage='S1', tier='T0', admission='required'),
           ArmSpec('F000', 'F000', 'lab-hgq', 'lab', 'ramp', config_code='F000', embedding='Q',
                   stage='S1', tier='T1', admission='required'),
           ArmSpec('F100', 'F100', 'binary-nonbn', 'lab', 'ramp', config_code='F100', embedding='Q',
                   stage='S1', tier='T1', admission='required')]
    for code in FUSED_CODES:
        for form in FORMS:
            out.append(ArmSpec(code, f'{code}-{form}', _binary_class(code, form), 'lab', 'ramp',
                               config_code=code, form=form, embedding='Q', stage='S1', tier='T1',
                               admission='required'))
    out += [replace(_released('PID-CH', 'PID-CH', 'pid', 'released-pid'), stage='S3', tier='T1', admission='required'),
            ArmSpec('PID-BEST', 'PID-BEST', 'pid-binary', 'lab', 'pid', template=T_BEST, embedding='Q',
                    stage='S3', tier='T1', admission='required'),
            ArmSpec('FP32', 'FP32', 'fp32', 'lab', 'ramp', control='FP32', embedding='Q', stage=early, tier='T2', admission=adm_c),
            ArmSpec('W8A8', 'W8A8', 'w8a8', 'lab', 'ramp', control='W8A8', embedding='Q', stage=early, tier='T3', admission=adm_c),
            ArmSpec('DS-HGQ', 'DS-HGQ', 'ds-hgq', 'lab', 'ramp', control='DS-HGQ', embedding='Q', stage=early, tier='T4',
                    admission=adm_c),
            ArmSpec('GR', 'GR', 'gr', 'lab', 'ramp', control='GR', embedding='Q', template=T_F111_CHOSEN,
                    stage='S3', tier='T5', admission=adm_c),
            ArmSpec('BE', 'BE', 'be', 'lab', 'ramp', control='BE', embedding='B', template=T_F111_CHOSEN,
                    stage='S3', tier='T6', admission=adm_c)]
    return tuple(out)


def s4_specs(family: str) -> Tuple[ArmSpec, ...]:
    if family not in S4_ARMS:
        raise ValueError(f'S4 family must be A, B or C, got {family!r}')
    table = {
        'CH-8-C': _released('CH-8-C', 'CH-8-C', 'ramp', 'released-ramp'),
        'BEST-C': ArmSpec('BEST-C', 'BEST-C', 'binary-best', 'lab', 'ramp', template=T_BEST, embedding='Q'),
        'PID-CH-C': _released('PID-CH-C', 'PID-CH-C', 'pid', 'released-pid'),
        'PID-BEST-C': ArmSpec('PID-BEST-C', 'PID-BEST-C', 'pid-binary', 'lab', 'pid', template=T_BEST,
                             embedding='Q'),
    }
    return tuple(replace(table[a], stage='S4', tier=None, admission='s4-kai-approval', s4_family=family)
                 for a in S4_ARMS[family])


def engineering_specs() -> Tuple[ArmSpec, ...]:
    """One identity per arm class, for the 2-file, 3-epoch smoke (G82) and the packing benchmark
    (G84). A class is a distinct training-code path: the released builder under the ramp and under
    the PID, the lab all-Q path, the binary non-BN path, each fused-BN form, the PID on a binary
    model in each form, and each control (GR and BE in each form). The S2 templates are fixed by
    definition here: F111, the one configuration that holds every binary site, in each form."""
    f111 = lambda form: dict(config_code='F111', form=form, embedding='Q')  # noqa: E731
    return (
        _released('CH-8', 'CH-8', 'ramp', 'released-ramp'),
        _released('PID-CH', 'PID-CH', 'pid', 'released-pid'),
        ArmSpec('F000', 'F000', 'lab-hgq', 'lab', 'ramp', config_code='F000', embedding='Q'),
        ArmSpec('F100', 'F100', 'binary-nonbn', 'lab', 'ramp', config_code='F100', embedding='Q'),
        ArmSpec('F111', 'F111-FB', 'binary-FB', 'lab', 'ramp', **f111('FB')),
        ArmSpec('F111', 'F111-BF', 'binary-BF', 'lab', 'ramp', **f111('BF')),
        ArmSpec('PID-BEST', 'PID-BEST-F111-FB', 'pid-binary-FB', 'lab', 'pid', template=T_BEST,
                engineering_fixed=True, **f111('FB')),
        ArmSpec('PID-BEST', 'PID-BEST-F111-BF', 'pid-binary-BF', 'lab', 'pid', template=T_BEST,
                engineering_fixed=True, **f111('BF')),
        ArmSpec('FP32', 'FP32', 'fp32', 'lab', 'ramp', control='FP32', embedding='Q'),
        ArmSpec('W8A8', 'W8A8', 'w8a8', 'lab', 'ramp', control='W8A8', embedding='Q'),
        ArmSpec('DS-HGQ', 'DS-HGQ', 'ds-hgq', 'lab', 'ramp', control='DS-HGQ', embedding='Q'),
        ArmSpec('GR', 'GR-FB', 'gr-FB', 'lab', 'ramp', control='GR', template=T_F111_CHOSEN,
                engineering_fixed=True, config_code='F111', form='FB', embedding='Q'),
        ArmSpec('GR', 'GR-BF', 'gr-BF', 'lab', 'ramp', control='GR', template=T_F111_CHOSEN,
                engineering_fixed=True, config_code='F111', form='BF', embedding='Q'),
        ArmSpec('BE', 'BE-FB', 'be-FB', 'lab', 'ramp', control='BE', template=T_F111_CHOSEN,
                engineering_fixed=True, config_code='F111', form='FB', embedding='B'),
        ArmSpec('BE', 'BE-BF', 'be-BF', 'lab', 'ramp', control='BE', template=T_F111_CHOSEN,
                engineering_fixed=True, config_code='F111', form='BF', embedding='B'),
    )


def determinism_specs() -> Tuple[Tuple[ArmSpec, Optional[int]], ...]:
    """Section 13: two CH-8 seed-1 runs and one F000 seed-1 run, same initialization and batch
    order, 20 epochs, on the production GPU product and stack. The CH-8 pair is replica 1 and 2;
    F000 has a single run, so its replica is None like every other identity (owner decision
    B12-5: cbl.wandb_log's replica token is empty for single runs)."""
    ch8 = _released('CH-8', 'CH-8', 'ramp', 'released-ramp')
    f000 = ArmSpec('F000', 'F000', 'lab-hgq', 'lab', 'ramp', config_code='F000', embedding='Q')
    return ((ch8, 1), (ch8, 2), (f000, None))


SMOKE_EPOCHS = 3        # 03-phases phase 2: "the 2-file/3-epoch smoke"
DETERMINISM_EPOCHS = 20  # STUDY section 13 determinism record
ENGINEERING_SEED = 1
