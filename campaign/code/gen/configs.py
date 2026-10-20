"""Configs: one JSON object per run (docs/interfaces.md section 5, extended in section 14 [B12]).

`generate(inputs)` returns the whole `configs/` tree as {fork-relative path: bytes}; it is a pure
function of the generator's sources, `gen/inputs.json`, the data manifest, `chang/run_train.py`
and `code-base.txt`. Nothing in it depends on the clock, the host, the interpreter or the git
state, so regenerating and diffing proves that no file under `configs/` was edited by hand (gate
G57).

Fail closed. A decision input that is unset is written as null and listed in the config's
`unset_inputs`; `check_launchable` refuses such a config, and also one with a `launch_blocks`
entry, and one whose handoff fields (`code_sha`, the initial-tensor manifest sha256) are not
supplied. Until the pack runner and the launcher call `check_launchable` (lanes P and L), this is
a tested check, not an enforced one.

The runner (M13, `cbl.runner.validate_config`) accepts a config only with `code_sha` and
`init_tensors.manifest_sha256` set; a committed config cannot carry either (owner decisions B12-1,
U1), so the handoff makes a launch copy with `with_handoff`, and `check_launchable` and the runner
then judge that copy. `ramp` is `cbl.runner.RAMP_RECORD` itself, imported (that module imports only
the standard library at import time), never restated.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from gen import CAMPAIGN, PROTOCOL_REFERENCE_SHA256, STUDY_SHA256, STUDY_VERSION
from gen import arms as A
from gen import release

FORK = Path(__file__).resolve().parents[1]
CONFIG_DIR = 'configs'
SCHEMA_VERSION = 1
FEATURES_PTETAPHI = ['pt', 'etarel', 'phirel']   # data.py:23, columns [5, 8, 11] (STUDY section 4 step 3)
GET_MODEL_ARGS = ['xfm', 7, 7, 1e-08, 8, True]    # STUDY section 2.1: get_model('xfm', 7, 7, 1e-8, 8, True)
PID = {'target_H': 350000, 'init_beta': 2e-08, 'p': 1.0, 'i': 0.002, 'd': 0.0, 'warmup': 10, 'log': True,
       'max_beta': 'inf', 'min_beta': 0.0, 'damping': 0.0}   # STUDY section 7.2 (beta_pid.py:95-106)
LAB_CALLBACKS_RAMP_TAIL = ['lab:frozen_evaluation']
WANDB_PROJECT = 'bnjettag-chang-bitlinear'
WANDB_GROUP = CAMPAIGN
WANDB_MODE = 'offline'   # PREFLIGHT: offline or disabled only (owner decision B5-7); B17 decides online
GEN_SOURCES = ('gen/__init__.py', 'gen/arms.py', 'gen/cli.py', 'gen/configs.py', 'gen/inputs.py',
               'gen/protocol.py', 'gen/protocol_text.json', 'gen/release.py')
HANDOFF_FIELDS = ('code_sha', 'init_tensors.manifest_sha256')
_HEX40 = re.compile(r'^[0-9a-f]{40}$')
_HEX64 = re.compile(r'^[0-9a-f]{64}$')


class UnsetInput(RuntimeError):
    """A config has an unset decision input, a launch block, or no handoff field: refuse to launch."""


def dumps(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def sources_record(fork: Path = FORK) -> dict:
    files = {rel: hashlib.sha256((fork / rel).read_bytes()).hexdigest() for rel in GEN_SOURCES}
    lines = ''.join(f'{files[rel]}  {rel}\n' for rel in sorted(files))
    return {'files': files, 'sha256': hashlib.sha256(lines.encode('utf-8')).hexdigest()}


def model_key(cfg: dict) -> str | None:
    """Identity of the model graph, the key of gate G58's parameter-count record; null while a
    template's model is not fixed by S2."""
    if cfg['template'] is not None and cfg['config_code'] is None:
        return None
    return '|'.join([cfg['builder'], cfg['control'] or '-', cfg['config_code'] or '-', cfg['form'] or '-',
                     cfg['embedding'] or '-'])


def _recipe(rel: dict) -> dict:
    return {'batch_size': rel['batch_size'], 'val_batch_size': rel['val_batch_size'],
            'steps_per_execution': rel['steps_per_execution'], 'learning_rate': rel['learning_rate'],
            'lr_schedule': rel['lr_schedule'], 'optimizer': rel['optimizer'], 'optimizer_args': rel['optimizer_args'],
            'loss': rel['loss'], 'dataset_shuffle': rel['dataset_shuffle'], 'fit_shuffle': rel['fit_shuffle'],
            'released_epochs': rel['epochs'], 'pareto_metrics': rel['pareto_metrics'],
            'pareto_sides': rel['pareto_sides'], 'pareto_filter': rel['pareto_filter'], 'stopif': rel['stopif'],
            'source': rel['source']}


def _ramp() -> dict:
    """interfaces section 5 `ramp`: must equal `cbl.runner.RAMP_RECORD` (interfaces section 13);
    imported, not restated. G57 compares it with what `gen.release` reads from run_train.py."""
    from cbl import runner      # standard library at import time; Keras and JAX only inside functions
    return copy.deepcopy(runner.RAMP_RECORD)


def _callbacks(spec: A.ArmSpec, rel: dict):
    if spec.control == 'FP32':
        return None   # STUDY 7.1: cost callbacks and StopIf off; the remaining list is M13's (B5), open
    cbs = list(rel['callbacks'])
    if spec.schedule == 'pid':
        cbs[cbs.index('BetaScheduler')] = 'lab:BetaPID'    # STUDY 7.2 "Callback order"
    return cbs + LAB_CALLBACKS_RAMP_TAIL


def _data(view: str, manifest: dict) -> dict:
    sp = manifest['split']
    if sp['n_constituents'] != 8 or sp['ptetaphi'] is not True or sp['seed'] != 42:
        raise ValueError('data manifest split is not the N=8 pt/etarel/phirel seed-42 view')
    arc = manifest[view]['archive']
    out = {'view': view, 'archive_name': arc['name'], 'archive_sha256': arc['sha256'], 'archive_rows': arc['rows'],
           'n_constituents': sp['n_constituents'], 'ptetaphi': sp['ptetaphi'], 'features': list(FEATURES_PTETAPHI),
           'split_seed': sp['seed'], 'val_size': sp['val_size']}
    if view == 'train':
        out.update({'n_train': sp['n_train'], 'n_val': sp['n_val'], 'order_sha256': sp['order_sha256'],
                    'split_sha256': sp['split_sha256'], 'stats_sha256': sp['stats_sha256']})
    return out


def _instantiate(spec: A.ArmSpec, s2):
    """Model fields of a template once S2 exists: (config_code, groups, form, form_source,
    best_source, arm_class). BEST templates take BEST's code; GR and BE take F111. The form is the
    chosen form where the configuration has a fused-BN binary site, else null."""
    if spec.template is None:
        form_source = 'fixed' if spec.form is not None else None
        return (spec.config_code, A.groups_of(spec.config_code) if spec.config_code else None, spec.form,
                form_source, None, spec.arm_class)
    if spec.engineering_fixed:
        return (spec.config_code, A.groups_of(spec.config_code), spec.form, 'fixed',
                'fixed' if spec.template == A.T_BEST else None, spec.arm_class)
    if s2 is None:
        return (None, None, None, None, None, spec.arm_class)
    code = s2['best'] if spec.template == A.T_BEST else 'F111'
    fused = A.has_fused_bn_binary(code, spec.embedding)
    form = s2['form'] if fused else None
    tag = f's2_choice:{s2["sha256"]}'
    if spec.template == A.T_BEST:
        cls = (f'pid-binary-{form}' if form else 'pid-binary-nonbn') if spec.schedule == 'pid' \
            else ('binary-nonbn' if not fused else f'binary-{form}')
    else:
        cls = f'{spec.arm_class}-{form}'
    return (code, A.groups_of(code), form, tag if form else None,
            tag if spec.template == A.T_BEST else None, cls)


def build_config(spec: A.ArmSpec, *, seed: int, stage: str, replica, inp: dict, rel: dict, gen_rec: dict) -> dict:
    from cbl import rng                 # NumPy only; prefit_reseed is decision U5
    from cbl import wandb_log as wl     # standard library only

    manifest = inp['data_manifest']['json']
    engineering = stage in ('smoke', 'bench', 'det')
    if engineering:
        epochs = {'smoke': A.SMOKE_EPOCHS, 'det': A.DETERMINISM_EPOCHS, 'bench': inp['bench_epochs']}[stage]
    else:
        epochs = rel['epochs']
    code, groups, form, form_source, best_source, arm_class = _instantiate(spec, inp['s2_record'])
    if spec.control in ('FP32', 'W8A8', 'DS-HGQ'):
        groups = A.groups_of('F000')   # interfaces 5: groups required under the lab builder; no group binary
    if code is not None and A.has_fused_bn_binary(code, spec.embedding) and form is None:
        raise ValueError(f'{spec.label}: a fused-BN binary site needs a form')
    is_template = spec.template is not None and not spec.engineering_fixed
    s2_sha = inp['s2_record']['sha256'] if (is_template and inp['s2_record'] is not None) else None

    init_in = inp['smoke_init_tensors'] if stage == 'smoke' else inp['init_tensors']
    mpath = (f'{init_in["root"]}/init-s{seed}-{init_in["platform"]}.manifest.json'
             if init_in['root'] is not None else None)

    cfg = {
        'schema_version': SCHEMA_VERSION,
        'campaign': CAMPAIGN,
        'arm_id': spec.arm_id,
        'label': spec.label,
        'arm_class': arm_class,
        'builder': spec.builder,
        'get_model_args': list(GET_MODEL_ARGS),
        'config_code': code,
        'groups': groups,
        'embedding': spec.embedding,
        'form': form,
        'form_source': form_source,
        'best_source': best_source,
        'template': spec.template,
        's2_choice_sha256': s2_sha,
        'control': spec.control,
        'seed': seed,
        'replica': replica,
        'stage': stage,
        'tier': None if engineering else spec.tier,
        'admission': 'engineering' if engineering else spec.admission,
        'schedule': spec.schedule,
        'recipe': _recipe(rel),
        'epochs': epochs,
        'cost_callbacks': spec.control != 'FP32',
        'callbacks': _callbacks(spec, rel),
        'headline': inp['headline'],
        'launch_confirmations': {k: inp[k] for k in ('od13', 'od15', 'od12')},
        'claim_domain': inp['claim_domain'],
        'matmul_precision': 'float32',
        'init_tensors': {'manifest_path': mpath, 'manifest_sha256': None},
        'prefit_reseed': rng.prefit_reseed(seed),
        'data_manifest_sha256': inp['data_manifest']['sha256'],
        'data': _data('smoke' if stage == 'smoke' else 'train', manifest),
        'code_sha': None,
        'param_count': None,
        'attempt': 1,
        'generator': {'sources_sha256': gen_rec['sha256'], 'inputs_sha256': inp['sha256'],
                      'study_sha256': STUDY_SHA256, 'study_version': STUDY_VERSION,
                      'protocol_reference_sha256': PROTOCOL_REFERENCE_SHA256,
                      'release_sha256': rel['source']['sha256']},
    }
    if spec.schedule == 'ramp':
        cfg['ramp'] = _ramp()
    else:
        cfg['pid'] = dict(PID)
    if stage == 'S4':
        cfg['s4_family'] = spec.s4_family
        cfg['s4_option'] = inp['s4_option']
    stem_ready = not (is_template and inp['s2_record'] is None)
    cfg['wandb'] = {'project': WANDB_PROJECT, 'group': WANDB_GROUP, 'mode': WANDB_MODE,
                    'run_name': (wl.run_name_stem(spec.arm_id, form, seed, stage, spec.schedule, 1, replica)
                                 if stem_ready else None)}
    key = model_key(cfg)
    if key is not None and inp['param_counts'] is not None:
        cfg['param_count'] = inp['param_counts']['counts'].get(key)

    unset = [k for k, v in cfg['launch_confirmations'].items() if v is not True]
    if cfg['headline'] is None and not headline_independent(cfg):
        unset.append('headline')
    if cfg['claim_domain'] is None:
        unset.append('claim_domain')
    if stage == 'S4' and cfg['s4_option'] is None:
        unset.append('s4_option')
    if is_template and inp['s2_record'] is None:
        unset.append('s2_record')
    if mpath is None:
        unset.append('smoke_init_tensors' if stage == 'smoke' else 'init_tensors')
    if stage == 'bench' and epochs is None:
        unset.append('bench_epochs')
    if cfg['param_count'] is None:
        unset.append('param_counts')
    cfg['unset_inputs'] = unset
    blocks = []
    if stage == 'S4' and inp['s4_option'] is not None and inp['s4_option'] != spec.s4_family:
        blocks.append(f'S4 family {spec.s4_family} is not the chosen option {inp["s4_option"]}')
    cfg['launch_blocks'] = blocks
    cfg['handoff_fields'] = list(HANDOFF_FIELDS)
    return cfg


def config_id(cfg: dict) -> str:
    base = f'{cfg["label"]}-s{cfg["seed"]}'
    return base if cfg['replica'] is None else f'{base}-r{cfg["replica"]}'


def config_path(cfg: dict) -> str:
    sub = f'S4/{cfg["s4_family"]}' if cfg['stage'] == 'S4' else cfg['stage']
    return f'{CONFIG_DIR}/{sub}/{config_id(cfg)}.json'


def catalogue(inp: dict) -> list:
    """(spec, seed, stage, replica) in emission order: the screen in protocol order, the S4
    families A, B, C, then smoke, bench and the determinism record."""
    out = []
    for spec in A.screen_specs(inp['item9']):
        out += [(spec, s, spec.stage, None) for s in A.SCREEN_SEEDS]
    for fam in ('A', 'B', 'C'):
        for spec in A.s4_specs(fam):
            out += [(spec, s, 'S4', None) for s in A.S4_SEEDS[fam]]
    for stage in ('smoke', 'bench'):
        out += [(spec, A.ENGINEERING_SEED, stage, None) for spec in A.engineering_specs()]
    out += [(spec, A.ENGINEERING_SEED, 'det', r) for spec, r in A.determinism_specs()]
    return out


def build_all(inp: dict, fork: Path = FORK) -> list:
    rel = release.release_constants(fork)
    gen_rec = sources_record(fork)
    return [build_config(spec, seed=s, stage=st, replica=r, inp=inp, rel=rel, gen_rec=gen_rec)
            for spec, s, st, r in catalogue(inp)]


def generate(inp: dict, fork: Path = FORK) -> dict:
    """The configs/ tree: every config, the derived protocols and MANIFEST.json."""
    from gen import protocol
    cfgs = build_all(inp, fork)
    files = {}
    for cfg in cfgs:
        p = config_path(cfg)
        if p in files:
            raise ValueError(f'duplicate config path {p}')
        files[p] = dumps(cfg)
    prot = protocol.derive_all(cfgs, inp['data_manifest']['json'], fork)
    for name, obj in prot.items():
        files[f'{CONFIG_DIR}/protocol/{name}.json'] = protocol.dumps(obj)
    gen_rec = sources_record(fork)
    entries = []
    for cfg in cfgs:
        p = config_path(cfg)
        entries.append({'path': p, 'sha256': hashlib.sha256(files[p]).hexdigest(), 'config_id': config_id(cfg),
                        'arm_id': cfg['arm_id'], 'label': cfg['label'], 'stage': cfg['stage'],
                        's4_family': cfg.get('s4_family'), 'seed': cfg['seed'], 'replica': cfg['replica'],
                        'template': cfg['template'], 'unset_inputs': cfg['unset_inputs'],
                        'launch_blocks': cfg['launch_blocks'],
                        'launchable_after_handoff': not cfg['unset_inputs'] and not cfg['launch_blocks']})
    counts = {}
    for e in entries:
        k = f'S4/{e["s4_family"]}' if e['stage'] == 'S4' else e['stage']
        counts[k] = counts.get(k, 0) + 1
    man = {'kind': 'cbl1016-config-manifest', 'schema': 1, 'campaign': CAMPAIGN,
           'study_sha256': STUDY_SHA256, 'study_version': STUDY_VERSION,
           'generator_sources': gen_rec, 'inputs_sha256': inp['sha256'],
           'data_manifest_sha256': inp['data_manifest']['sha256'],
           'counts': dict(sorted(counts.items())), 'n_configs': len(entries),
           'configs': entries,
           'protocol': {f'{CONFIG_DIR}/protocol/{n}.json': hashlib.sha256(files[f'{CONFIG_DIR}/protocol/{n}.json']).hexdigest()
                        for n in prot}}
    files[f'{CONFIG_DIR}/MANIFEST.json'] = dumps(man)
    return files


def disk_tree(fork: Path = FORK) -> dict:
    root = fork / CONFIG_DIR
    if not root.exists():
        return {}
    return {p.relative_to(fork).as_posix(): p.read_bytes() for p in sorted(root.rglob('*')) if p.is_file()}


def diff_trees(want: dict, have: dict) -> dict:
    return {'missing': sorted(set(want) - set(have)), 'extra': sorted(set(have) - set(want)),
            'different': sorted(p for p in set(want) & set(have) if want[p] != have[p])}


def write_tree(files: dict, fork: Path = FORK) -> None:
    """Replace configs/ with `files` (configs/ is generated only, so nothing else lives there)."""
    root = fork / CONFIG_DIR
    if root.exists():
        for p in sorted(root.rglob('*'), reverse=True):
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
    for rel_path, data in sorted(files.items()):
        if not rel_path.startswith(CONFIG_DIR + '/'):
            raise ValueError(f'{rel_path} is outside {CONFIG_DIR}/')
        target = fork / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def with_handoff(cfg: dict, *, code_sha: str, init_manifest_sha256: str) -> dict:
    """The launch copy of a generated config: the two handoff fields filled, nothing else changed.
    The handoff record (B17) keeps it beside the generated config's sha256; configs/ never holds it."""
    if not (isinstance(code_sha, str) and _HEX40.match(code_sha)):
        raise UnsetInput(f'code_sha must be a 40-hex commit, got {code_sha!r}')
    if not (isinstance(init_manifest_sha256, str) and _HEX64.match(init_manifest_sha256)):
        raise UnsetInput(f'init_tensors.manifest_sha256 must be 64-hex, got {init_manifest_sha256!r}')
    out = copy.deepcopy(cfg)
    out['code_sha'] = code_sha
    out['init_tensors']['manifest_sha256'] = init_manifest_sha256
    return out


def headline_independent(cfg: dict) -> bool:
    """v0.4 permits only CH-8/F000 ramp training before the export outcome."""
    return (cfg.get('arm_id') in ('CH-8', 'F000') and cfg.get('schedule') == 'ramp'
            and cfg.get('control') is None and cfg.get('template') is None)


def check_launchable(cfg: dict, *, code_sha: str | None = None, init_manifest_sha256: str | None = None) -> None:
    """Refuse a config that cannot launch: an unset decision input (recomputed from the fields, not
    only read from `unset_inputs`), a launch block, or a missing handoff field. The handoff fields
    are taken from the keywords, or from a launch copy made by `with_handoff`."""
    code_sha = code_sha if code_sha is not None else cfg.get('code_sha')
    if init_manifest_sha256 is None:
        init_manifest_sha256 = (cfg.get('init_tensors') or {}).get('manifest_sha256')
    problems = list(cfg.get('unset_inputs') or [])
    for key in ('od13', 'od15', 'od12'):
        if (cfg.get('launch_confirmations') or {}).get(key) is not True:
            problems.append(key)
    if cfg.get('headline') not in ('H1', 'H2') and not (cfg.get('headline') is None and headline_independent(cfg)):
        problems.append('headline')
    from cbl.select import claim_domain_config, UnsetDecision
    try:
        claim_domain_config(cfg.get('claim_domain'))
    except UnsetDecision as e:
        problems.append(f'claim_domain: {e}')
    if cfg.get('stage') == 'S4':
        if cfg.get('s4_option') not in ('A', 'B', 'C'):
            problems.append('s4_option')
        elif cfg.get('s4_option') != cfg.get('s4_family'):
            problems.append('s4_family differs from s4_option')
    if cfg.get('template') is not None and cfg.get('config_code') is None:
        problems.append('s2_record (template not instantiated)')
    if not (cfg.get('init_tensors') or {}).get('manifest_path'):
        problems.append('init_tensors.manifest_path')
    if cfg.get('epochs') is None:
        problems.append('epochs')
    if cfg.get('param_count') is None:
        problems.append('param_count')
    problems += list(cfg.get('launch_blocks') or [])
    if not (isinstance(code_sha, str) and _HEX40.match(code_sha)):
        problems.append('code_sha (handoff)')
    if not (isinstance(init_manifest_sha256, str) and _HEX64.match(init_manifest_sha256)):
        problems.append('init_tensors.manifest_sha256 (handoff)')
    if problems:
        seen = []
        for p in problems:
            if p not in seen:
                seen.append(p)
        raise UnsetInput(f'{cfg.get("label")} s{cfg.get("seed")} {cfg.get("stage")}: refused, ' + '; '.join(seen))
