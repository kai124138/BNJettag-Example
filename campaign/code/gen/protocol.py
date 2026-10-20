"""The protocol derived from the configs, for the drift check against the frozen protocol (G59).

What is computed and what is carried:
- computed from the configs: every arm (name, group, seed, the quantization description from
  the config's builder, code, form, control and template, the schedule's epochs, batch size,
  learning rate, optimizer and cost-control text from the config's recipe and schedule, and the
  changed factors from the config fields that differ from the stage's first arm), the inputs
  (n_constituents and features), the dataset identity hash (`data_manifest_sha256`) and scope;
- computed from the data manifest the configs name: each metric's n (held-out rows, n_val);
- carried verbatim from `gen/protocol_text.json`, which was copied once from the protocol
  reference (sha256 8f48e716...): purpose, status, dataset id and split texts, selection metric
  and rule, metric names, splits and statuses, interval method, stop rules, outputs, and the
  sentence templates of the arm descriptions.
So a drift check against the reference shows a changed config at once, and shows an edit of the
carried prose as a difference too.

Objects: `screen` (the funded screen, with `confirmation_stage` = option A as the reference has
it), and `claim-A`, `claim-B`, `claim-C`, one claim-scope object per S4 template family. The B
and C objects carry option A's prose; STUDY section 8.7 has the claim-scope file rewritten for
the option and claim domain Kai chooses before it is frozen.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

FORK = Path(__file__).resolve().parents[1]
TEXT_REL = 'gen/protocol_text.json'

QUANT_FIELDS = ('builder', 'config_code', 'form', 'control', 'embedding', 'template')


def dumps(obj) -> bytes:
    """The reference's own format: key order as built, indent 1, ASCII."""
    return (json.dumps(obj, indent=1, ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def load_text(fork: Path = FORK) -> dict:
    t = json.loads((fork / TEXT_REL).read_bytes())
    if t.get('kind') != 'cbl1016-protocol-text' or t.get('schema') != 1:
        raise ValueError('protocol_text.json is not a cbl1016-protocol-text schema 1 record')
    return t


def short_float(x: float) -> str:
    """2e-08 -> '2e-8' (the reference's spelling)."""
    s = f'{x:.0e}' if x == float(f'{x:.0e}') else repr(x)
    return re.sub(r'e([+-])0*(\d)', lambda m: 'e' + ('-' if m.group(1) == '-' else '') + m.group(2), s)


def describe_quantization(cfg: dict, text: dict) -> str:
    q = text['arms']['quantization']
    if cfg['template'] == 'BEST':
        return q['BEST_s4'] if cfg['stage'] == 'S4' else q['BEST_screen']
    if cfg['control'] is not None:
        return q[cfg['control']]
    if cfg['builder'] == 'released':
        return q['released']
    code = cfg['config_code']
    if code == 'F000':
        return q['F000']
    if code == 'F100':
        return q['F100']
    names = [q['group_names'][g] for g in ('attention', 'ffn', 'head') if cfg['groups'][g] == 'B']
    return q['fused'].format(groups=q['group_joiner'].join(names), form_text=q['form_text'][cfg['form']])


def schedule_of(cfg: dict, text: dict) -> dict:
    t = text['arms']
    if not cfg['cost_callbacks']:
        cost = t['cost_control']['none']
    elif cfg['schedule'] == 'pid':
        cost = t['cost_control']['pid'].format(target_H=cfg['pid']['target_H'],
                                               init_beta=short_float(cfg['pid']['init_beta']))
    else:
        cost = t['cost_control']['ramp']
    r = cfg['recipe']
    return {'epochs': cfg['epochs'], 'batch_size': r['batch_size'],
            'optimizer': t['optimizer'].format(first_decay_steps=r['lr_schedule']['first_decay_steps']),
            'learning_rate': r['learning_rate'], 'cost_control': cost}


def _quant_identity(cfg: dict) -> tuple:
    return tuple(cfg[f] for f in QUANT_FIELDS)


def derive_arms(cfgs: list, text: dict) -> list:
    arms, first, first_sched = [], None, None
    for cfg in cfgs:
        sched = schedule_of(cfg, text)
        if first is None:
            first, first_sched = cfg, sched
        factors = []
        if _quant_identity(cfg) != _quant_identity(first):
            factors.append('quantization')
        factors += [f'schedule.{k}' for k in sched if sched[k] != first_sched[k]]
        arms.append({'name': f'{cfg["label"]}-s{cfg["seed"]}', 'group': cfg['label'], 'seed': cfg['seed'],
                     'quantization': describe_quantization(cfg, text), 'schedule': sched,
                     'changed_factors': factors})
    return arms


def _common(cfgs: list, manifest: dict) -> tuple:
    dms = {c['data_manifest_sha256'] for c in cfgs}
    ncs = {c['data']['n_constituents'] for c in cfgs}
    feats = {tuple(c['data']['features']) for c in cfgs}
    if len(dms) != 1 or len(ncs) != 1 or len(feats) != 1:
        raise ValueError(f'configs disagree on data identity: {dms} {ncs} {feats}')
    n = {'heldout_rows': manifest['heldout']['archive']['rows'], 'n_val': manifest['split']['n_val']}
    return dms.pop(), ncs.pop(), list(feats.pop()), n


def _object(cfgs: list, manifest: dict, prose: dict, scope: str, with_status: bool) -> dict:
    ident, nc, feats, n = _common(cfgs, manifest)
    obj = {'version': 1}
    if with_status:
        obj['status'] = prose['status']
    obj['purpose'] = prose['purpose']
    obj['scope'] = scope
    d = prose['dataset']
    obj['dataset'] = {'id': d['id'], 'identity_sha256': ident, 'train_split': d['train_split'],
                      'validation_split': d['validation_split'], 'test_split': d['test_split']}
    obj['inputs'] = {'n_constituents': nc, 'features': feats}
    return obj, n


def _tail(obj: dict, prose: dict, n: dict) -> dict:
    obj['selection'] = {'split': 'validation', 'metric': prose['selection']['metric'],
                        'rule': prose['selection']['rule'], 'cost_checkpoint': 'selected'}
    obj['metrics'] = [{'name': m['name'], 'split': m['split'], 'n': n[m['n_source']], 'status': m['status']}
                      for m in prose['metrics']]
    obj['interval_method'] = prose['interval_method']
    obj['stop_rules'] = list(prose['stop_rules'])
    obj['outputs'] = list(prose['outputs'])
    return obj


def derive_claim(cfgs: list, manifest: dict, text: dict) -> dict:
    obj, n = _object(cfgs, manifest, text['claim'], 'claim', True)
    obj['arms'] = derive_arms(cfgs, text)
    return _tail(obj, text['claim'], n)


def derive_screen(cfgs: list, claim_a: dict, manifest: dict, text: dict) -> dict:
    obj, n = _object(cfgs, manifest, text['screen'], 'screen', False)
    obj['arms'] = derive_arms(cfgs, text)
    obj = _tail(obj, text['screen'], n)
    obj['confirmation_stage'] = claim_a
    return obj


def derive_all(cfgs: list, manifest: dict, fork: Path = FORK) -> dict:
    """{'screen': ..., 'claim-A': ..., 'claim-B': ..., 'claim-C': ...} from the configs, in the
    order the generator emits them (screen in reference order; each S4 family by arm, then seed)."""
    text = load_text(fork)
    screen = [c for c in cfgs if c['stage'] in ('S1', 'S3')]
    fam = {f: [c for c in cfgs if c['stage'] == 'S4' and c['s4_family'] == f] for f in ('A', 'B', 'C')}
    claims = {f'claim-{f}': derive_claim(fam[f], manifest, text) for f in ('A', 'B', 'C')}
    out = {'screen': derive_screen(screen, claims['claim-A'], manifest, text)}
    out.update(claims)
    return out
