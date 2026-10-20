"""Gate G47 (PREFLIGHT-PLAN section 3 row G47; STUDY v0.10 section 7.2 "Callback order": the PID replaces
BetaScheduler at its position, [FreeEBOPs, LR scheduler, PID, PBar, ParetoFront (released), StopIf], followed by the
frozen-evaluation callback; stock BetaPID.on_epoch_end overwrites logs['ebops'] (beta_pid.py:162-167); here the PID
writes its sensed value under its own key, so ParetoFront and StopIf read FreeEBOPs's training count; module M12
cbl/pid.py through M13 cbl/runner.py). PC CPU, venv of record, 2-file smoke archive.

Children as G43 (gates/pidgate.py; one PID run per arm, 14 epochs, read-only spies), plus `units` for the stock
control.

Checks, per run:
1. callback order [FreeEBOPs, LearningRateScheduler, LabBetaPID, PBar, ParetoFront, StopIf, FrozenEvalHook]: the
   released order (cbl.runner.RELEASED_ORDER) with the PID at BetaScheduler's index 2; the config's callbacks list
   names the same (lab:BetaPID, lab:frozen_evaluation); the stream guard wrapped the PID as lab code;
2. at every epoch end: FreeEBOPs wrote logs['ebops'] as an int; LabBetaPID left it unchanged in type and value and
   added only `beta` and its own `pid/` keys (sensed_H, stock_aggregate, beta, integral, clamp_state, slack; and
   error, acted_on_H, beta_unclamped once acting), never `ebops`; PBar, ParetoFront, StopIf and the appended callback
   saw FreeEBOPs's int; StopIf's condition and ParetoFront's filter read exactly that value;
3. the epoch record's train_count equals that value, and its PID block's sensed_H is a separate field; every entry of
   Chang's ParetoFront record (if it saved any) carries the training count of its epoch;
4. control (units child): the stock BetaPID's on_epoch_end replaces FreeEBOPs's int in logs['ebops'] by its own float
   aggregate (beta_pid.py:166), so the check of 2 would fail for it; LabBetaPID's warmup epochs there leave the int.
Every reading is engineering, not quotable.

    <venv>/bin/python -I -B gates/g47_pid_keys.py --device cpu --expect-sha12 SHA12 --work DIR --archive SMOKE \\
        [--json PATH] [--children-out DIR]
"""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import pidgate as pg  # noqa: E402
from cbl import runner  # noqa: E402

GATE = 'G47'
ORDER = ['FreeEBOPs', 'LearningRateScheduler', 'LabBetaPID', 'PBar', 'ParetoFront', 'StopIf', 'FrozenEvalHook']
CONFIG_NAME = {'lab:BetaPID': 'LabBetaPID', 'lab:frozen_evaluation': 'FrozenEvalHook'}
ALWAYS = {'beta', 'pid/sensed_H', 'pid/stock_aggregate', 'pid/beta', 'pid/integral', 'pid/clamp_state', 'pid/slack'}
ACTING = {'pid/error', 'pid/acted_on_H', 'pid/beta_unclamped'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    pg.standard_args(ap)
    a = ap.parse_args()
    G = pg.Gate(GATE, a)
    g = G.g
    rel = list(runner.RELEASED_ORDER)
    g.check(ORDER[:2] == rel[:2] and ORDER[3:6] == rel[3:] and rel[2] == 'BetaScheduler',
            f'the expected order is the released order {rel} with the PID in BetaScheduler\'s place, then FrozenEvalHook')
    if G.draw_s1() is None:
        return G.finish()
    G.rec['runs'] = {}
    for arm, form, hl in pg.ARMS:
        lab, R = G.run(arm, form, hl)
        if R is None:
            continue
        rr = R['runner']
        g.check(rr['callback_order'] == ORDER and [CONFIG_NAME.get(x, x) for x in R['config']['callbacks']] == ORDER,
                f'{lab}: callback order {rr["callback_order"]}, as the config names it')
        wrapped = {w['who']: w['owner'] for w in rr['guard']['wrapped']}
        g.check(wrapped.get('LabBetaPID') == 'lab' and wrapped.get('FrozenEvalHook') == 'lab'
                and all(wrapped.get(n) == 'released' for n in ('FreeEBOPs', 'LearningRateScheduler', 'PBar', 'ParetoFront', 'StopIf')),
                f'{lab}: the stream guard wrapped LabBetaPID and FrozenEvalHook as lab code, the five others as released')
        seq = R['seq']
        E = pg.EPOCHS
        free = {}
        ok_free, ok_pid, ok_seen, detail = True, True, True, []
        for e in range(E):
            ends = {x['cb']: x for x in seq if x['event'] == 'on_epoch_end' and x['epoch'] == e}
            fe = ends.get('FreeEBOPs', {})
            v = fe.get('ebops_after')
            free[e] = v
            ok_free &= fe.get('ebops_before') is None and isinstance(v, list) and v[0] == 'int' and 'ebops' in fe.get('added', [])
            pe = ends.get('LabBetaPID', {})
            want_added = ALWAYS | (ACTING if e >= pg.WARMUP else set())
            good = pe.get('ebops_before') == v and pe.get('ebops_after') == v and set(pe.get('added', [])) == want_added
            ok_pid &= good
            if not good:
                detail.append({'epoch': e, 'pid': pe})
            seen = [ends.get(n, {}).get('ebops_before') for n in ('PBar', 'ParetoFront', 'StopIf')]
            seen += [ends.get(n, {}).get('ebops_after') for n in ('PBar', 'ParetoFront', 'StopIf')]
            hook = [x['ebops_seen_by_hook'] for x in R['betas'] if x['epoch'] == e]
            ok_seen &= all(s == v for s in seen) and hook == [v]
        g.check(ok_free, f'{lab}: FreeEBOPs wrote logs[\'ebops\'] as an int at every epoch end')
        g.check(ok_pid, f'{lab}: LabBetaPID left logs[\'ebops\'] unchanged in type and value at every epoch end and '
                        f'added only beta and its own pid/ keys' + (f'; {detail[:2]}' if detail else ''))
        g.check(ok_seen, f'{lab}: PBar, ParetoFront, StopIf and the appended callback saw FreeEBOPs\'s int at every epoch')
        sr = [x['ebops'] for x in R['stop_reads']]
        pr = {x['epoch']: x['ebops'] for x in R['pareto_reads']}
        g.check(sr == [free[e] for e in range(E)] and sorted(pr) == list(range(E)) and all(pr[e] == free[e] for e in pr),
                f'{lab}: StopIf\'s condition and ParetoFront\'s filter read FreeEBOPs\'s count at each of the {E} epochs')
        rows, bl = pg.rows(R), pg.blocks(R)
        g.check(all(rows[e]['train_count'] == free[e][1] for e in range(E))
                and all('sensed_H' in bl[e] and 'ebops' not in bl[e] for e in range(E)),
                f'{lab}: the record\'s train_count is FreeEBOPs\'s count; the PID block\'s sensed_H is its own field')
        pf = R['pareto']
        ep_of = [int(p.split('-')[0].split('=')[1]) for p in pf['paths']]
        g.check(len(pf['record']) == len(pf['paths'])
                and all(rr_[1] == ['float', float(free[e][1]).hex()] for rr_, e in zip(pf['record'], ep_of)),
                f'{lab}: Chang\'s ParetoFront record ({len(pf["record"])} entries) carries the training count of its epochs')
        G.rec['runs'][lab] = {'free_ebops': [free[e] for e in range(E)], 'pareto': pf}
    U = G.child('units', 'units', [])
    if U is not None:
        sk, lw = U['stock_log_keys'], U['lab_warmup']
        g.check(sk['ebops_before'] == ['int', 123456] and sk['ebops_after'][0] == 'float' and sk['ebops_after'] != sk['ebops_before']
                and all(x['ebops_after'] == ['int', 123456] for x in lw),
                f'control: the stock BetaPID.on_epoch_end turned logs[\'ebops\'] {sk["ebops_before"]} into its float '
                f'aggregate {sk["ebops_after"]}; LabBetaPID left the int in all {len(lw)} warmup epochs')
    return G.finish()


if __name__ == '__main__':
    sys.exit(main())
