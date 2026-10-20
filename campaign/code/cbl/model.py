"""M5: Chang's xfm graph rebuilt layer by layer inside his scopes, with a kernel choice per group.

STUDY means `review/snapshots/v0.10/STUDY.md` (sha256 7180c24f6b55807e...); "plan" means
PREFLIGHT-PLAN.md (sha256 2f1be9f3...30ff8), section 2 row M5, section 8 decisions U1 and U3, section 9
item 8. Gates G14 (F000 equals CH-8) and G15 (nested MHA mapping by role, single wrap of the HGQ
call). Batch B5 built the all-Q path; batch B6 fills the binary kernel choice (M6, cbl/binary.py) and
the binary arms' RoleSpecs (G10, G16-G24).

What it builds
- `get_model('xfm', 7, 7, 1e-8, 8, True)` (`run_train.py:76` with `-m xfm -n 8 -f 3`) re-expressed:
  get_model's three scopes (`chang/model.py:278-296`), entered in the same order, and inside them
  get_transformer's three scopes (`:185-189`), then the layers of `:190-209` constructed and called
  one by one in Chang's order. Keras draws an initializer seed from the module-level `random`
  stream when a layer's initializer is made (in `__init__`, or in `build` for the MHA's projections,
  which clone the initializer), so the same classes in the same order draw the same seeds: F000's own
  draw equals CH-8's in every tensor, and F000 loads CH-8's file by role anyway (STUDY section 6
  step 4, decision U1).
- No layer is given a name, so in a fresh process the layers carry the same automatic names as
  CH-8's (`q_einsum_dense_batchnorm`, ...). Roles (cbl/rng.py) do not depend on names: F000 uses
  CH-8's semantic top-level names, pinned by the class sequence (`rng.ch8_top_names`), and its
  `RoleSpec` declares empty unmatched and unused sets (244 variables).
- A kernel choice per group (STUDY section 5 "Codes"; section 2.2 "Sites"): `attention` (the MHA's
  four projections), `ffn` (the expansion, a fused-BN site, and the contraction), `head` (the three
  fused-BN head layers); the embedding is `Q` except in BE, and the classifier always stays HGQ.
  `Q` builds Chang's own layer at the site. `B` builds M6's layer there (batch B6), at the hook
  functions `_bn_site` (`BinaryEinsumDenseBatchnorm` in the arm's form), `_dense_site`
  (`BinaryEinsumDense`) and `_attention` (`BinaryMultiHeadAttention`, decision U3), each given the
  config's headline as its penalty (b) branch and the GR flag. The binary layers are constructed with
  the same arguments in the same order inside the same scopes, so they draw the same initializer seeds
  as Chang's layers and their own draw of every shared role equals CH-8's (G10 records it).
- Roles (B6): a binary arm's top-level names are CH-8's, pinned by the arm's class sequence; at each
  binary site the scale quantizer q_g (`<site>/qg/FixedPointQuantizerKBI[0]/{k,b,i}`) is the unmatched
  set and Chang's kernel quantizer (`<site>/kq/FixedPointQuantizerKBI[0]/{k,b,i}`) the unused set
  (STUDY section 6 step 4; q_g's uint8 k beside b and i is a finding against that step's wording). Each
  binary arm has 244 variables, as CH-8, since q_g has as many variables as kq.

Variants (STUDY section 5 "Control definitions"); FP32, W8A8 and DS-HGQ are stubs that raise `NotBuilt`:
- FP32 ("quantizers and cost penalties off"): not a configuration of the all-Q path. Turning the
  quantizers off is a choice per layer kind (kernel and bias quantizers, the datalane quantizers,
  the two tanh tables and the softmax tables, which `QSoftmax` forces to SAT and RND_CONV) that the
  STUDY does not spell out, and its cost callbacks and StopIf are off (section 7.1).
- W8A8: decision U6 (CONFIDENCE LOW, flagged for Kai) changes the datalane quantizer type; it is a
  new quantizer setup, not a kernel choice.
- DS-HGQ: a different graph (tanh 1, Q/K/V/O, softmax, A.V and add 1 removed), with its own top-level
  names and an unused set of initial tensors that G10 has to gate; not configuration-only.
- GR and BE (built in B6): F111 with the scale derivative (OD-7; the GR flag of M6's maps), and F111
  with a binary embedding in the same form with its own q_g. Both need groups B, B, B and a form; the
  form is the S2 choice in campaign stages, which the runner checks (cbl/runner.py).

hgq, keras and Chang's module are imported inside the functions, after `cbl.env.setup()`; this
module imports nothing heavy at import time. Nothing here is quotable; it is model code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

GROUPS = ('attention', 'ffn', 'head')
KINDS = ('Q', 'B')
LAB_ARMS = ('F000', 'F100', 'F001', 'F010', 'F011', 'F101', 'F110', 'F111', 'PID-BEST', 'BEST-C', 'PID-BEST-C',
            'FP32', 'W8A8', 'DS-HGQ', 'GR', 'BE')
CONTROLS = ('FP32', 'W8A8', 'DS-HGQ', 'GR', 'BE')
FORMS = ('FB', 'BF')
HEADLINES = ('H1', 'H2')

# run_train.py:76 with -m xfm -n 8 -f 3: get_model('xfm', 7, 7, 1e-8, 8, True)
BW_K, BW_A, L1_REG, N_CONSTITUENTS, PT_ETA_PHI = 7, 7, 1e-8, 8, True
# get_transformer, model.py:179-183
N_HEADS, KEY_DIM, DIM = 2, 16, 16


class NotBuilt(NotImplementedError):
    """A configuration whose layers are not built (the controls FP32, W8A8 and DS-HGQ)."""


class SpecError(ValueError):
    """An arm specification that is not one of STUDY section 5's configurations."""


@dataclass(frozen=True)
class ArmSpec:
    """What the lab builder needs from a config (interfaces.md section 5)."""
    arm_id: str
    groups: dict = field(default_factory=lambda: {'attention': 'Q', 'ffn': 'Q', 'head': 'Q'})
    embedding: str = 'Q'
    form: str | None = None
    control: str | None = None
    headline: str | None = None        # the penalty (b) branch of binary sites; None on all-Q arms

    @property
    def config_code(self) -> str:
        return 'F' + ''.join('1' if self.groups[g] == 'B' else '0' for g in GROUPS)

    @property
    def binary(self) -> bool:
        return 'B' in self.groups.values() or self.embedding == 'B'


def arm_spec(arm_id, *, groups, embedding='Q', form=None, control=None, headline=None) -> ArmSpec:
    """Validate an arm's builder fields (interfaces.md section 5) and return its spec. A binary arm needs
    the headline, which selects its penalty (b) branch and has no default (STUDY section 3.1(b));
    on an all-Q arm the headline does not enter the build and is not kept in the spec."""
    if arm_id not in LAB_ARMS:
        raise SpecError(f'arm {arm_id!r} is not built by the lab builder (CH-8 is the released builder)')
    if not isinstance(groups, dict) or set(groups) != set(GROUPS) or any(v not in KINDS for v in groups.values()):
        raise SpecError(f'groups must map {GROUPS} to Q or B, not {groups!r}')
    if embedding not in KINDS:
        raise SpecError(f'embedding must be Q or B, not {embedding!r}')
    if control is not None and control not in CONTROLS:
        raise SpecError(f'control {control!r} is not one of {CONTROLS}')
    if (embedding == 'B') != (control == 'BE'):
        raise SpecError('a binary embedding exists only in BE (STUDY section 5)')
    fused_bn_binary = groups['ffn'] == 'B' or groups['head'] == 'B' or embedding == 'B'
    if fused_bn_binary and form not in FORMS:
        raise SpecError('a fused-BN binary site needs form FB or BF (interfaces.md section 5)')
    if not fused_bn_binary and form is not None:
        raise SpecError('form is set only when a fused-BN site is binary (F000 and F100 are identical in both forms)')
    if control in ('GR', 'BE') and set(groups.values()) != {'B'}:
        raise SpecError(f'{control} is F111 in the chosen form (STUDY section 5): groups must all be B, not {groups}')
    binary = 'B' in groups.values() or embedding == 'B'
    if binary and headline not in HEADLINES:
        raise SpecError(f'a binary arm needs the headline {HEADLINES} for its penalty (b) branch; got {headline!r} '
                        '(no default, interfaces.md section 5)')
    if not binary and headline is not None and headline not in HEADLINES:
        raise SpecError(f'headline must be H1 or H2 (or null on an all-Q arm), not {headline!r}')
    spec = ArmSpec(arm_id=arm_id, groups=dict(groups), embedding=embedding, form=form, control=control,
                   headline=headline if binary else None)
    if re.fullmatch(r'F[01]{3}', arm_id) and arm_id != spec.config_code:
        raise SpecError(f'arm {arm_id} has groups {groups}, which is configuration {spec.config_code}')
    return spec


F000_SPEC = ArmSpec(arm_id='F000')


# ----------------------------------------------------------------------------------------------
# the scopes


def release_scopes():
    """get_model's three scopes with run_train.py:76's arguments (chang/model.py:278-295), in the
    order get_model enters them (`with scope0, scope1, scope2`, :296)."""
    from hgq.config import LayerConfigScope, QuantizerConfigScope
    from hgq.constraints import MinMax
    from hgq.regularizers import MonoL1

    scope0 = QuantizerConfigScope(                     # model.py:278-286
        default_q_type='kbi',
        b0=BW_K,
        overflow_mode='wrap',
        i0=0,
        fr=MonoL1(L1_REG),
        ir=MonoL1(L1_REG),
        i_decay_speed=1e-3,
    )
    scope1 = QuantizerConfigScope(                     # model.py:287-294
        default_q_type='kif',
        place='datalane',
        overflow_mode='wrap',
        f0=BW_A,
        fr=MonoL1(L1_REG),
        ic=MinMax(0, 12),
    )
    scope2 = LayerConfigScope(beta0=0)                 # model.py:295
    return scope0, scope1, scope2


def transformer_scopes():
    """get_transformer's three scopes (chang/model.py:185-189) with uq1=False, so
    homogeneous_axis = (0,) (:181)."""
    from hgq.config import QuantizerConfigScope
    from hgq.constraints import Min

    homogeneous_axis = (0,)                            # model.py:181, uq1 False
    return (
        QuantizerConfigScope(place=('weight', 'bias'), overflow_mode='SAT_SYM', b0=4, f0=4),     # :186
        QuantizerConfigScope(place='datalane', homogeneous_axis=homogeneous_axis),               # :187
        QuantizerConfigScope(place='table', homogeneous_axis=homogeneous_axis, bc=Min(4)),       # :188
    )


# ----------------------------------------------------------------------------------------------
# site hooks: Q builds Chang's layer; B builds M6's (cbl/binary.py)


def _binary_kwargs(spec):
    return {'penalty': spec.headline, 'gr': spec.control == 'GR'}


def _bn_site(site, kind, spec, *args, **kwargs):
    """A fused-BN site (embedding, FFN expansion, head layers): QEinsumDenseBatchnorm under Q,
    M6's BinaryEinsumDenseBatchnorm in the arm's form under B."""
    if kind == 'Q':
        from hgq.layers import QEinsumDenseBatchnorm
        return QEinsumDenseBatchnorm(*args, **kwargs)
    from cbl import binary
    return binary.BinaryEinsumDenseBatchnorm(*args, form=spec.form, **_binary_kwargs(spec), **kwargs)


def _dense_site(site, kind, spec, *args, **kwargs):
    """A non-BN site (the FFN contraction): QEinsumDense under Q, M6's BinaryEinsumDense under B."""
    if kind == 'Q':
        from hgq.layers import QEinsumDense
        return QEinsumDense(*args, **kwargs)
    from cbl import binary
    return binary.BinaryEinsumDense(*args, **_binary_kwargs(spec), **kwargs)


def _attention(kind, spec, *args, **kwargs):
    """The MHA (its four projections are the attention group): QMultiHeadAttention under Q, M6's
    BinaryMultiHeadAttention under B, whose build() creates binary projections (decision U3)."""
    if kind == 'Q':
        from hgq.layers import QMultiHeadAttention
        return QMultiHeadAttention(*args, **kwargs)
    from cbl import binary
    return binary.BinaryMultiHeadAttention(*args, **_binary_kwargs(spec), **kwargs)


# ----------------------------------------------------------------------------------------------
# the builder


def build(spec: ArmSpec):
    """The arm's keras.Model: the all-Q path, every binary configuration, GR and BE. Raises NotBuilt for
    FP32, W8A8 and DS-HGQ (module docstring, "Variants")."""
    if spec.control not in (None, 'GR', 'BE'):
        raise NotBuilt(f'control {spec.control}: not built (module docstring, "Variants")')
    import keras
    from hgq.layers import QAdd, QAffinedUnaryFunctionLUT, QEinsumDense, QGlobalAveragePooling1D

    N = N_CONSTITUENTS                                 # model.py:179
    n = 3 if PT_ETA_PHI else 16                        # :180
    h = N_HEADS                                        # :182
    dim = DIM                                          # :183
    g = spec.groups
    scope0, scope1, scope2 = release_scopes()
    with scope0, scope1, scope2:                       # model.py:296
        s0, s1, s2 = transformer_scopes()
        with s0, s1, s2:                               # model.py:185-189
            inp = keras.layers.Input((N, n))                                                          # :190
            x = _bn_site('embed', spec.embedding, spec, '...c,cC->...C', 24, bias_axes='C', activation='relu')(inp)  # :192
            xa = QAffinedUnaryFunctionLUT('tanh')(x)                                                  # :194
            xa = _attention(g['attention'], spec, h, KEY_DIM, dropout=0.0)(xa, xa)                    # :195
            x = QAdd()([x, xa])                                                                       # :197
            xf = xa = QAffinedUnaryFunctionLUT('tanh')(x)                                             # :198
            xf = _bn_site('ffn1', g['ffn'], spec, '...c,cC->...C', dim * h, bias_axes='C', activation='relu')(xf)    # :199
            xf = _dense_site('ffn2', g['ffn'], spec, '...c,cC->...C', 24)(xf)                         # :200
            x = QAdd()([x, xf])                                                                       # :202
            out = QGlobalAveragePooling1D()(x)                                                        # :204
            out = _bn_site('head1', g['head'], spec, '...c,cC->...C', 32, bias_axes='C', activation='relu')(out)   # :205
            out = _bn_site('head2', g['head'], spec, '...c,cC->...C', 32, bias_axes='C', activation='relu')(out)   # :206
            out = _bn_site('head3', g['head'], spec, '...c,cC->...C', 32, bias_axes='C', activation='relu')(out)   # :207
            out = QEinsumDense('...c,cC->...C', 5)(out)                                               # :208, classifier stays HGQ
            model = keras.models.Model(inp, out)                                                      # :209
            return model


# top-level semantic name -> the group that decides its kernel (None: always Chang's layer)
SITE_GROUP = {'embed': 'embedding', 'mha': 'attention', 'ffn1': 'ffn', 'ffn2': 'ffn', 'head1': 'head', 'head2': 'head',
              'head3': 'head'}
BINARY_CLASS = {'embed': 'BinaryEinsumDenseBatchnorm', 'mha': 'BinaryMultiHeadAttention', 'ffn1': 'BinaryEinsumDenseBatchnorm',
                'ffn2': 'BinaryEinsumDense', 'head1': 'BinaryEinsumDenseBatchnorm', 'head2': 'BinaryEinsumDenseBatchnorm',
                'head3': 'BinaryEinsumDenseBatchnorm'}
MHA_PROJECTIONS = ('query', 'key', 'value', 'attention_output')
QUANTIZER_VARIABLES = ('k', 'b', 'i')            # FixedPointQuantizerKBI.build (fixed_point_quantizer.py:215-242)


def _kind(spec: ArmSpec, top: str) -> str:
    grp = SITE_GROUP.get(top)
    if grp is None:
        return 'Q'
    return spec.embedding if grp == 'embedding' else spec.groups[grp]


def binary_sites(spec: ArmSpec) -> list:
    """Role prefixes of the arm's binary kernel layers, in CH-8's layer order (MHA projections nested)."""
    from cbl import rng

    out = []
    for _, top in rng.CH8_TOP:
        if _kind(spec, top) == 'B':
            out += [f'mha/{p}' for p in MHA_PROJECTIONS] if top == 'mha' else [top]
    return out


def class_sequence(spec: ArmSpec) -> tuple:
    """The arm's model.layers classes: CH-8's, with M6's class at every binary top-level site."""
    from cbl import rng

    return tuple(BINARY_CLASS[top] if _kind(spec, top) == 'B' else cls for cls, top in rng.CH8_TOP)


def quantizer_roles(spec: ArmSpec, segment: str) -> frozenset:
    """`<site>/<segment>/FixedPointQuantizerKBI[0]/{k,b,i}` at every binary site (segment qg or kq)."""
    return frozenset(f'{site}/{segment}/FixedPointQuantizerKBI[0]/{v}' for site in binary_sites(spec)
                     for v in QUANTIZER_VARIABLES)


def role_spec(spec: ArmSpec):
    """The arm's RoleSpec (cbl/rng.py section 6 step 4): CH-8's top-level names pinned by the arm's class
    sequence and 244 variables; F000's unmatched and unused sets are empty, a binary arm's are q_g's
    and Chang's kq's variables at its binary sites (module docstring, "Roles")."""
    from cbl import rng

    if spec.control not in (None, 'GR', 'BE'):
        raise NotBuilt(f'RoleSpec of {spec.arm_id} (control {spec.control}): not built')
    want = class_sequence(spec)

    def top_names(model):
        got = tuple(type(ly).__name__ for ly in model.layers)
        if got != want:
            raise rng.RoleError(f'not the {spec.arm_id} ({spec.config_code}) graph: layers {got}, expected {want}')
        return [n for _, n in rng.CH8_TOP]

    if not spec.binary:
        return rng.RoleSpec(arm_id=spec.arm_id, top_names=rng.ch8_top_names, n_variables=rng.CH8_N_VARIABLES)
    unmatched, unused = quantizer_roles(spec, 'qg'), quantizer_roles(spec, 'kq')
    return rng.RoleSpec(arm_id=spec.arm_id, top_names=top_names, expected_unmatched=lambda model: unmatched,
                        expected_unused=lambda model: unused, n_variables=rng.CH8_N_VARIABLES)


def builder(spec: ArmSpec):
    """A zero-argument callable for `SeedFlow.build_model`."""
    return lambda: build(spec)
