"""Gate G45 (PREFLIGHT-PLAN section 3 row G45; STUDY v0.10 section 7.2: lambda_0 = 2e-8 > 0 and the sourced gains
"p=1, i=0.002, d=0, warmup 10, log=True, max_beta=inf, min_beta=0 and damping 0 (beta_pid.py:95-106)"; "a positive
lambda_0 is needed: Chang's beta0=0 with init_beta=None reaches log10(0) (model.py:295; beta_pid.py:125-148)"; module
M12 cbl/pid.py). PC CPU, venv of record, 2-file smoke archive.

Children (fresh processes under the open-file audit): `units` (gates/pid_child.py) on a freshly built CH-8 model,
CH-8's initial-tensor file for seed 1, and one PID-CH run (headline H1, 14 epochs) through the runner.

Checks:
1. sources: the installed hgq2 0.1.9 beta_pid.py and the snapshot's reference-code/HGQ2/src/hgq/utils/sugar/
   beta_pid.py hash to the same sha256 (c25dac91..., STUDY 7.3 lists the file as identical); BetaPID.__init__'s
   defaults read from the installed class (inspect.signature) equal those read from the snapshot's syntax tree;
2. cbl.pid.PID_RECORD's gains equal those defaults: p 1.0, i 0.002, d 0.0, warmup 10, log true, max_beta inf,
   min_beta 0.0, damping (damp_beta_on_target) 0.0; init_beta is 2e-8 > 0 (the stock default is None), target_H
   350000 (OD-12); the constructor arguments built from the record carry max_beta = float inf parsed from the
   configs' JSON string "inf";
3. refusals: init_beta 0 or negative, warmup 0, i 0, max_beta "Infinity", a float inf, null, an int p, a float or
   other target, an extra key, log 1 are each refused;
4. the lambda_0 control: every beta of Chang's CH-8 is 0 at build (LayerConfigScope(beta0=0), model.py:295); the
   stock BetaPID with init_beta None starts at beta 0 and raises at epoch == warmup in log10 (math domain error);
5. LabBetaPID sets every layer's beta to 2e-8 (as float32) at on_train_begin and keeps it through the ten warmup
   epochs, logging beta 2e-8 (float64), with no error, acted-on reading or unclamped beta in the warmup blocks;
6. in the PID-CH run through the runner: the recorded constructor arguments and gains are those of 2, the logged
   beta and every layer's beta are 2e-8 in epochs 0-9 (clamp state `warmup`), and the controller acts from epoch 10.
Every reading is engineering, not quotable.

    <venv>/bin/python -I -B gates/g45_pid_init.py --device cpu --expect-sha12 SHA12 --work DIR --archive SMOKE \\
        [--json PATH] [--children-out DIR]
"""

import argparse
import struct
import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import pidgate as pg  # noqa: E402

GATE = 'G45'
BETA_PID_SHA256 = 'c25dac9185b5226dc6b164b9a48780dc943ebd516c2e603eb66a5e2e606c7e13'
STOCK_DEFAULTS = {'init_beta': None, 'p': 1.0, 'i': 0.002, 'd': 0.0, 'warmup': 10, 'log': True, 'max_beta': 'inf',
                  'min_beta': 0.0, 'damp_beta_on_target': 0.0}
RECORD_KEY = {'p': 'p', 'i': 'i', 'd': 'd', 'warmup': 'warmup', 'log': 'log', 'max_beta': 'max_beta',
              'min_beta': 'min_beta', 'damp_beta_on_target': 'damping'}
REFUSED = ('init_beta 0', 'init_beta negative', 'warmup 0', 'max_beta Infinity', 'max_beta float inf', 'max_beta null',
           'i 0', 'target float', 'p int', 'target 400000', 'extra key', 'log 1')


def f32hex(x):
    return struct.unpack('<f', struct.pack('<f', x))[0].hex()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    pg.standard_args(ap)
    a = ap.parse_args()
    G = pg.Gate(GATE, a)
    g = G.g
    U = G.child('units', 'units', [])
    if U is not None:
        ins, snap = U['installed'], U['snapshot']
        g.check(ins['sha256'] == snap['sha256'] == BETA_PID_SHA256,
                f'installed beta_pid.py ({ins["file"]}) and the snapshot\'s ({snap["file"]}) both hash to '
                f'{BETA_PID_SHA256[:12]}...')
        g.check(ins['defaults'] == snap['defaults'] == STOCK_DEFAULTS and ins['required'] == snap['required'] == ['target_ebops'],
                f'BetaPID.__init__ defaults, installed (inspect) and snapshot (syntax tree, lines {snap["lines"]}): '
                f'{ins["defaults"]}')
        rec_ = U['record']
        g.check(all(rec_[RECORD_KEY[k]] == STOCK_DEFAULTS[k] and type(rec_[RECORD_KEY[k]]) is type(STOCK_DEFAULTS[k])
                    for k in RECORD_KEY),
                'cbl.pid.PID_RECORD gains, warmup, log, max_beta, min_beta and damping equal the stock defaults')
        kw = U['kwargs']
        g.check(rec_['init_beta'] == pg.INIT_BETA and rec_['init_beta'] > 0 and STOCK_DEFAULTS['init_beta'] is None
                and rec_['target_H'] == pg.TARGET_H and kw['init_beta'] == pg.INIT_BETA and kw['target_ebops'] == pg.TARGET_H
                and kw['max_beta'] == 'inf' and U['max_beta_parsed'] == ['float', 'inf'],
                f'init_beta 2e-8 > 0 (stock default None), target_H {rec_["target_H"]}; constructor arguments {kw}; '
                f'"inf" parsed to {U["max_beta_parsed"]}')
        ref = U['refusals']
        g.check(all(ref.get(k) for k in REFUSED), f'refused: {", ".join(k for k in REFUSED if ref.get(k))}'
                + ('; NOT refused: ' + ', '.join(k for k in REFUSED if not ref.get(k)) if not all(ref.get(k) for k in REFUSED) else ''))
        sn = U['stock_none']
        g.check(U['chang_betas'] == [(0.0).hex()] and sn['beta_after_train_begin'] == ['float', (0.0).hex()]
                and str(sn['at_warmup'].get('raised', '')).startswith('ValueError: math domain error')
                and 'log10(self.beta)' in str(sn['at_warmup'].get('frame')),
                f'lambda_0 control: Chang\'s CH-8 has every beta {U["chang_betas"]}; the stock BetaPID with init_beta None '
                f'starts at beta 0 and at epoch 10 raises {sn["at_warmup"].get("raised")} in "{sn["at_warmup"].get("frame")}"')
        li, lw = U['lab_initial'], U['lab_warmup']
        want = f32hex(pg.INIT_BETA)
        g.check(li['betas_after_train_begin'] == [want] and li['beta'] == ['float', pg.INIT_BETA.hex()]
                and len(lw) == pg.WARMUP and all(x['betas'] == [want] and x['logs_beta'] == ['float', pg.INIT_BETA.hex()]
                                                 and not {'pid/error', 'pid/acted_on_H', 'pid/beta_unclamped'} & set(x['pid_keys'])
                                                 for x in lw),
                f'LabBetaPID: every layer\'s beta {want} (float32 of 2e-8) after on_train_begin and through the '
                f'{len(lw)} warmup epochs; logged beta 2e-8; no error or acted-on reading in warmup')
        G.rec['units'] = U
    if G.draw_s1() is None:
        return G.finish()
    lab, R = G.run('PID-CH', None, 'H1')
    if R is not None:
        sig = R['runner']['pid']
        g.check(sig['kwargs'] == U['kwargs'] if U else False,
                f'{lab}: the runner\'s PID was built with the constructor arguments of the record ({sig["kwargs"]})')
        g.check(sig['gains'] == {'p': 1.0, 'i': 0.002, 'd': 0.0, 'neg': True} and sig['sensor'] == 'headline'
                and sig['headline'] == 'H1',
                f'{lab}: its PID object holds p 1.0, i 0.002, d 0.0 (error = reading - target, neg true); sensor '
                f'{sig["sensor"]} {sig["headline"]}')
        bl, rows = pg.blocks(R), pg.rows(R)
        betas = {x['epoch']: x['beta_values'] for x in R['betas']}
        warm = range(pg.WARMUP)
        g.check(all(rows[e]['beta'] == pg.INIT_BETA and bl[e]['beta'] == pg.INIT_BETA and bl[e]['clamp_state'] == 'warmup'
                    and betas.get(e) == [f32hex(pg.INIT_BETA)] for e in warm),
                f'{lab}: epochs 0-9 log beta 2e-8, every layer holds it, clamp state warmup')
        g.check(all(bl[e]['clamp_state'] == 'none' and bl[e].get('acted_on_H') is not None for e in range(pg.WARMUP, pg.EPOCHS)),
                f'{lab}: the controller acts from epoch {pg.WARMUP} (clamp state none; beta at epoch 10 '
                f'{bl[pg.WARMUP]["beta"] if pg.WARMUP in bl else None})')
        G.rec['run'] = {'beta': [pg.hexes(bl[e]['beta']) for e in sorted(bl)], 'signature': sig}
    return G.finish()


if __name__ == '__main__':
    sys.exit(main())
