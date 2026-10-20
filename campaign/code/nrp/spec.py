"""Pack schema, generated-config binding and launch admission (M20)."""
from pathlib import Path
import json
import copy

from nrp.common import (CAMPAIGN, ROOT, PRODUCTS, STAGES, need, token, positive, hexstr,
                        relative, pvc, digest, init_root, Refused)


def launch_check(cfg):
    from gen.configs import check_launchable, UnsetInput
    try:
        check_launchable(cfg)
    except UnsetInput as e:
        raise Refused(str(e)) from e


def wire_plan(plan):
    """Avoid a third full config copy in the launcher's 100 kB record environment.

    The handoff brief still holds each full launch copy and generated hash. The process
    reconstructs only the two with_handoff fields from immutable checkout bytes.
    """
    out = copy.deepcopy(plan)
    for arm in out['arms']:
        if 'config' in arm:
            arm['init_manifest_sha256'] = arm.pop('config')['init_tensors']['manifest_sha256']
    return out


def expand_plan(plan, code_root):
    from gen.configs import with_handoff
    out = copy.deepcopy(plan)
    for arm in out['arms']:
        if out['mode'] != 'draw' and 'config' not in arm:
            cfg = json.loads((Path(code_root) / relative(arm['config_path'])).read_bytes())
            arm['config'] = with_handoff(cfg, code_sha=out['code_sha'],
                                         init_manifest_sha256=arm.pop('init_manifest_sha256'))
    return out


def resolve(spec, files, commit, bundle_sha):
    """Resolve from exact bundle bytes (keys relative to repository), never local configs.

    Infrastructure attempts change only spec.job_attempt; scientific attempt stays in gen/.
    Draw mode has no training config: its declared config is the data preparation manifest.
    """
    from gen.configs import with_handoff, check_launchable
    from cbl.runner import validate_config
    from cbl.wandb_log import name_from_config

    need(spec.get('schema') == 1, 'pack schema must be 1')
    pack = token(spec['pack'])
    need(len(pack) <= 26 and pack == pack.lower() and '_' not in pack and '.' not in pack, 'pack must be a short DNS label')
    product = spec['gpu_product']
    need(product in PRODUCTS, 'unsupported GPU product/resource key')
    hexstr(commit, 40); hexstr(bundle_sha)
    positive(spec['job_attempt'], 'job_attempt', True)
    mode = spec['mode']
    need(mode in ('pack', 'smoke', 'bench', 'det', 'draw'), 'unknown mode')
    stage = spec['stage']
    need(stage == 'draw' if mode == 'draw' else stage in STAGES, 'invalid stage')
    need(mode in ('pack', 'draw') or mode == stage, 'mode/stage mismatch')
    need(mode != 'pack' or stage in ('S1', 'S3', 'S4'), 'production pack stage required')
    kind = spec['archive_kind']
    need(kind in ('record', 'smoke'), 'invalid archive kind')
    need(kind == ('smoke' if stage == 'smoke' else 'record') or mode == 'draw', 'stage/archive mismatch')
    resources = spec['resources']
    for key in ('cpu', 'memory_gi', 'deadline_s', 'ttl_s', 'cache_gi'):
        positive(resources[key], key, True)
    need(resources['deadline_s'] > 60, 'deadline must leave termination margin')
    need(isinstance(spec.get('excluded_nodes'), list), 'explicit current node exclusions required')
    need(spec.get('approval_ref') and spec.get('gate_reference'), 'approval and gate references required')
    need(spec.get('scientific_gate') in ('pending', 'cleared'), 'scientific gate must be explicit')
    need(type(spec.get('deterministic_ops')) is bool, 'deterministic_ops must be explicitly chosen')
    data_path = 'manifests/data-prep-150c.json'
    data = json.loads(files[data_path])
    arms = []
    if mode == 'draw':
        seeds = spec['seeds']
        need(seeds and len(set(seeds)) == len(seeds) and all(type(s) is int and 1 <= s <= 11 for s in seeds),
             'draw seeds must be unique within 1..11; S4 draws need separate authority')
        for seed in seeds:
            arms.append({'seed': seed, 'name': f'CH-8-s{seed}-draw-{kind}', 'config_path': data_path,
                         'out': init_root(product, kind), 'arm_class': 'initial-tensors'})
        k = 1  # sequential draws, exactly one visible GPU like each training pod
    else:
        seen = set()
        for entry in spec['arms']:
            path = relative(entry['config_path'])
            need(path.startswith('configs/') and path.endswith('.json'), 'only generated configs admitted')
            raw = files[path]
            generated = json.loads(raw)
            need(generated['stage'] == stage, 'mixed stage pack')
            need(generated.get('code_sha') is None and generated['init_tensors']['manifest_sha256'] is None,
                 'expected generated config, not an edited launch copy')
            cfg = with_handoff(generated, code_sha=commit, init_manifest_sha256=hexstr(entry['init_manifest_sha256']))
            launch_check(cfg)  # mandatory even for engineering
            validate_config(cfg)
            need(cfg['data_manifest_sha256'] == digest(files[data_path]), 'preparation manifest mismatch')
            view = 'smoke' if kind == 'smoke' else 'train'
            need(cfg['data']['view'] == view and cfg['data']['archive_sha256'] == data[view]['archive']['sha256'],
                 'archive does not match generated config')
            expected = init_root(product, kind) + f'/init-s{cfg["seed"]}-gpu.manifest.json'
            need(cfg['init_tensors']['manifest_path'] == expected, 'init draw root/product/archive differs')
            name = name_from_config(cfg, bundle_sha)
            token(name)
            need(name not in seen, 'duplicate run identity')
            seen.add(name)
            if stage == 'smoke':
                need(cfg['epochs'] == 3, 'smoke must be three epochs')
            if stage == 'det':
                need(cfg['epochs'] == 20 and cfg['seed'] == 1 and cfg['arm_id'] in ('CH-8', 'F000'), 'invalid determinism run')
            arms.append({'name': name, 'config_path': path, 'generated_sha256': digest(raw), 'config': cfg,
                         'arm_class': cfg['arm_class'], 'seed': cfg['seed'], 'out': f'{ROOT}/runs/{stage}/{name}'})
        k = len(arms)
        positive(k, 'K', True)
        if stage == 'det':
            need(sorted((a['config']['arm_id'], a['config']['replica'] or 0) for a in arms) ==
                 [('CH-8', 1), ('CH-8', 2), ('F000', 0)], 'det needs two CH-8 replicas and one F000')
    need(k != 1 or spec.get('single_arm_justification'), 'one arm needs an explicit engineering/measurement justification')
    fraction = positive(spec['mem_fraction'], 'mem_fraction')
    need(fraction * k <= 0.90, 'aggregate JAX allocation fraction exceeds 90 percent')
    return {'schema': 1, 'campaign': CAMPAIGN, 'pack': pack, 'mode': mode, 'stage': stage,
            'job_attempt': spec['job_attempt'], 'source_bundle': bundle_sha, 'code_sha': commit,
            'gpu_product': product, 'k': k, 'archive_kind': kind, 'arms': arms,
            'mem_fraction': fraction, 'resume_every': positive(spec['resume_every'], 'resume_every', True),
            'deadline_s': resources['deadline_s'], 'deterministic_ops': spec['deterministic_ops']}


def check_execution(plan, code_root):
    """Recheck embedded launch copies against immutable checkout before spawning any child."""
    from gen.configs import with_handoff, check_launchable
    from cbl.runner import validate_config
    from cbl.wandb_log import name_from_config
    need(plan['campaign'] == CAMPAIGN and plan['gpu_product'] in PRODUCTS, 'wrong execution identity')
    need(plan['mode'] in ('pack', 'smoke', 'bench', 'det', 'draw'), 'invalid execution mode')
    need(0 < plan['mem_fraction'] * plan['k'] <= 0.9, 'unsafe memory fraction')
    hexstr(plan['source_bundle']); hexstr(plan['code_sha'], 40)
    names = set()
    for arm in plan['arms']:
        need(Path(arm['out']).resolve().is_relative_to(Path(ROOT).resolve()), 'run path resolves outside campaign PVC')
        raw = (Path(code_root) / relative(arm['config_path'])).read_bytes()
        if plan['mode'] == 'draw':
            need(arm['out'] == init_root(plan['gpu_product'], plan['archive_kind']), 'wrong draw root')
            continue
        cfg = arm['config']
        launch_check(cfg)
        validate_config(cfg)
        need(digest(raw) == arm['generated_sha256'], 'generated config hash mismatch')
        need(with_handoff(json.loads(raw), code_sha=plan['code_sha'],
                          init_manifest_sha256=cfg['init_tensors']['manifest_sha256']) == cfg,
             'launch copy changed fields beyond with_handoff')
        need(cfg['stage'] == plan['stage'], 'stage mismatch')
        need(name_from_config(cfg, plan['source_bundle']) == arm['name'], 'run name mismatch')
        need(arm['out'] == f'{ROOT}/runs/{plan["stage"]}/{arm["name"]}', 'wrong run directory')
        need(arm['name'] not in names, 'duplicate run')
        names.add(arm['name'])
        pvc(cfg['init_tensors']['manifest_path'])
        need(cfg['init_tensors']['manifest_path'] == init_root(plan['gpu_product'], plan['archive_kind']) +
             f'/init-s{cfg["seed"]}-gpu.manifest.json', 'init product/archive mismatch')
    need(plan['k'] == (1 if plan['mode'] == 'draw' else len(plan['arms'])), 'K differs from arm count')
