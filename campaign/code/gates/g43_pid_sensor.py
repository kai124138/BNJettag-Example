"""Gate G43 (PREFLIGHT-PLAN section 3 row G43; STUDY v0.10 section 7.2 "Sensor timing (chang C1)": the modified PID
computes frozen H(e) itself in its own on_epoch_end by the shared non-mutating frozen evaluation and acts on it at
on_epoch_begin(e+1); the frozen-evaluation callback reuses the same cached value; gate: sensed(e) equals frozen H(e)
exactly; module M12 cbl/pid.py through M13 cbl/runner.py). PC CPU, venv of record, 2-file smoke archive.

Children (fresh processes under the open-file audit): CH-8's initial-tensor file for seed 1 (gates/runner_child.py
draw), then one PID run per arm of gates/pidgate.py (PID-CH H1, PID-BEST as F111-FB H1 and F111-BF H2), 14 epochs,
through the runner with read-only spies (gates/pid_child.py run).

Checks, per run, bitwise or as exact integers:
1. every epoch's record holds the PID block, and its sensed_H equals the record's frozen headline count (H1 or H2,
   the config's headline) of the same epoch, as the same integer;
2. one evaluation per epoch: the evaluator computed exactly once per epoch (its count equals the epochs), inside
   LabBetaPID's on_epoch_end and before PBar's; the appended callback's evaluation of that epoch was the cached value,
   and the front saver read it through `cached` after it;
3. timing: the reading acted on at on_epoch_begin(e) is sensed_H(e-1) for every e >= warmup (10) and none before;
4. the controller's beta, error and integral of every epoch equal the stock law restated independently in
   gates/pidgate.py `law` on the frozen readings (log10 error, integral start at epoch == warmup, clamp), as float64
   bits; the logged beta equals the block's; every layer's `_beta` holds the epoch's beta (as float32) during it;
5. the stream guard's 'PID frozen-H sensor' block ran once per epoch with no change of the global streams; no lab
   or released violation;
6. control: the same checks against the records shifted by one epoch fail (the comparison can fail).
Every reading is engineering, not quotable.

    <venv>/bin/python -I -B gates/g43_pid_sensor.py --device cpu --expect-sha12 SHA12 --work DIR --archive SMOKE \\
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

GATE = 'G43'


def f32hex(x):
    return struct.unpack('<f', struct.pack('<f', x))[0].hex()


def sensor_equal(R, headline, shift=0):
    rows, bl = pg.rows(R), pg.blocks(R)
    pairs = [(e, e + shift) for e in sorted(rows) if e + shift in rows]
    return bool(pairs) and all(bl[e] is not None and type(bl[e]['sensed_H']) is int
                               and bl[e]['sensed_H'] == rows[f][headline] for e, f in pairs)


def law_equal(R, headline, shift=0):
    rows, bl = pg.rows(R), pg.blocks(R)
    sensed = {e: rows[min(e + shift, max(rows))][headline] for e in rows}
    want = pg.law(sensed, epochs=len(rows))
    bad = []
    for e in sorted(rows):
        b, w = bl[e], want[e]
        got = (pg.hexes(b['beta']), pg.hexes(b.get('error')), pg.hexes(b['integral']), b.get('acted_on_H'))
        exp = (pg.hexes(w['beta']), pg.hexes(w['error']), pg.hexes(w['integral']), w['acted_on_H'])
        if got != exp:
            bad.append({'epoch': e, 'got': got, 'want': exp})
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    pg.standard_args(ap)
    a = ap.parse_args()
    G = pg.Gate(GATE, a)
    g = G.g
    if G.draw_s1() is None:
        return G.finish()
    G.rec['runs'] = {}
    for arm, form, hl in pg.ARMS:
        lab, R = G.run(arm, form, hl)
        if R is None:
            continue
        rows, bl = pg.rows(R), pg.blocks(R)
        E = pg.EPOCHS
        g.check(sorted(rows) == list(range(E)) and all(bl[e] is not None for e in rows),
                f'{lab}: {len(rows)} epoch records, each with the PID block')
        g.check(sensor_equal(R, hl), f'{lab}: sensed_H(e) equals the frozen {hl}(e) of the same epoch\'s record exactly, '
                                     f'as the same integer, at every epoch ({[bl[e]["sensed_H"] for e in sorted(bl)][:4]}...)')
        # 2. one evaluation per epoch, cached for the callback, read by the saver after it
        seq = R['seq']
        ok_order, detail = True, []
        for e in range(E):
            ev = [(i, x) for i, x in enumerate(seq) if x['event'] in ('evaluate', 'cached') and x['epoch'] == e]
            ends = {x['cb']: i for i, x in enumerate(seq) if x['event'] == 'on_epoch_end' and x['epoch'] == e}
            hook = [i for i, x in enumerate(seq) if x['event'] == 'hook_step' and x['epoch'] == e]
            comp = [i for i, x in ev if x['event'] == 'evaluate' and x['computes']]
            cached_eval = [i for i, x in ev if x['event'] == 'evaluate' and not x['computes']]
            cached_read = [i for i, x in ev if x['event'] == 'cached' and not x['inside_evaluate']]
            good = (len(comp) == 1 and ends.get('LabBetaPID', 10 ** 9) < comp[0] < ends.get('PBar', -1)
                    and len(cached_eval) == 1 and cached_eval[0] > ends.get('StopIf', 10 ** 9)
                    and len(cached_read) == 1 and cached_eval[0] < cached_read[0] < (hook[0] if hook else -1))
            ok_order &= good
            if not good:
                detail.append({'epoch': e, 'events': [x for _, x in ev], 'ends': ends, 'hook': hook})
        g.check(R['n_evaluations'] == E and ok_order,
                f'{lab}: the frozen evaluation computed once per epoch ({R["n_evaluations"]} for {E} epochs), inside '
                f'LabBetaPID.on_epoch_end before PBar; the appended callback got the cached value after StopIf, and the '
                f'front saver read it through cached() before the gate probe' + (f'; {detail[:2]}' if detail else ''))
        # 3. timing
        acted = [bl[e].get('acted_on_H') for e in range(E)]
        g.check(all(acted[e] is None for e in range(pg.WARMUP))
                and all(acted[e] == bl[e - 1]['sensed_H'] for e in range(pg.WARMUP, E)),
                f'{lab}: on_epoch_begin(e) acted on sensed_H(e-1) for e >= {pg.WARMUP} and on nothing before')
        # 4. the law, restated
        bad = law_equal(R, hl)
        g.check(not bad, f'{lab}: beta, error and integral of every epoch equal the stock law restated on the frozen '
                         f'readings (float64 bits)' + (f'; first difference {bad[0]}' if bad else ''))
        betas = {x['epoch']: x['beta_values'] for x in R['betas']}
        g.check(all(rows[e]['beta'] == bl[e]['beta'] for e in range(E))
                and all(betas.get(e) == [f32hex(bl[e]['beta'])] for e in range(E)),
                f'{lab}: the logged beta equals the block\'s, and every layer\'s _beta held that beta (float32) during '
                f'the epoch')
        # 5. guard
        gd = R['runner']['guard']
        w = gd['calls'].get('PID frozen-H sensor.watch', {})
        g.check(gd['lab_violations'] == [] and gd['released_violations'] == [] and w.get('calls') == E
                and w.get('py_changed') == 0 and w.get('np_changed') == 0,
                f'{lab}: the stream guard\'s PID sensor block ran {w.get("calls")} times with no change of the global '
                f'streams; no lab or released violation')
        # 6. control
        g.check(not sensor_equal(R, hl, shift=1) and bool(law_equal(R, hl, shift=1)),
                f'{lab}: control: against the readings shifted by one epoch both comparisons fail')
        G.rec['runs'][lab] = {'sensed_H': [bl[e]['sensed_H'] for e in range(E)],
                              'headline': [rows[e][hl] for e in range(E)],
                              'beta': [pg.hexes(bl[e]['beta']) for e in range(E)],
                              'clamp_state': [bl[e]['clamp_state'] for e in range(E)],
                              'n_evaluations': R['n_evaluations'], 'pid_selection': R['front_summary'].get('pid_selection')}
        g.info(f'{lab} engineering readings, not quotable', {'sensed_H': G.rec['runs'][lab]['sensed_H'],
                                                             'pid_selection_status': (R['front_summary'].get('pid_selection') or {}).get('status')})
    return G.finish()


if __name__ == '__main__':
    sys.exit(main())
