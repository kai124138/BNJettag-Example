"""Gate G46, the PID part (PREFLIGHT-PLAN section 3 row G46, modules M12 and M14; STUDY v0.10 section 7.2 "On
resume, the integral, previous error, lambda, epoch and sensed cost are persisted" and the section 13 PID row
"interrupted and resumed integral identical"). PC CPU, venv of record, 2-file smoke archive.

Scope. Two parts. The controller's part (B9): M12's persisted state (`LabBetaPID.state()` and `restore()`) carries the
integral, the previous error, lambda, the epoch and the sensed reading so that an interrupted and resumed
controller continues exactly as an uninterrupted one on the same readings. The general resume of a training run
(model, optimizer, the module-level random state, the ParetoFront record, the StopIf counters, the front saver, the
W&B run; G48) is batch B10's (M14), which stores this state in its resume record; the whole-run equality is G48's.
[B10] The whole-run part: PID-CH (H1, 14 epochs) through the runner, uninterrupted, and interrupted by a real SIGKILL
right after M14's checkpoint of 5 (warmup), 10 (the integral-start epoch) and 12 (acting) epochs, then resumed in a
new process from the run directory (gates/resume_child.py): every epoch's PID block (sensed H, beta, error,
integral, clamp state, slack, the reading acted on, the unclamped beta), the whole per-epoch record and the final PID
state equal the uninterrupted run's.

Children (fresh processes under the open-file audit): CH-8's initial-tensor file for seed 1; one PID-CH run (H1,
14 epochs) through the runner; then gates/pid_child.py replay children on a freshly built CH-8 model with a sensor
that returns that run's frozen readings: `full`; for K in 5 (inside warmup), 10 (the integral-start epoch) and 12
(while acting): `first` (epochs 0..K-1, then the state written as JSON) and `resume` (a new process restores the
JSON, then epochs K..13); `tamper` (K = 12, the integral set to 0) and `fresh` (K = 12, no restore) as controls.

Checks:
1. the replay reproduces the live controller: per epoch beta, error, integral, clamp state, the reading acted on,
   the unclamped beta, the sensed value and the slack of `full` equal the run's PID blocks exactly;
2. for each K: the state written after epoch K-1 holds epoch K-1, sensed_H(K-1), and the integral, previous error
   and beta of the uninterrupted controller at that point (float64 bits, with their hex record); the resumed
   process continues with epochs K..13 equal to `full` in every logged PID key, in the beta on every layer after each
   on_epoch_begin, and in the integral, previous error and beta after each epoch; its final state equals full's;
3. controls: with the integral set to 0 the resumed sequence differs from `full` (the comparison can fail); a fresh
   controller asked to act at epoch 12 without a restored reading is refused (PIDError), never run from nothing.
Every reading is engineering, not quotable.

    <venv>/bin/python -I -B gates/g46_pid_resume.py --device cpu --expect-sha12 SHA12 --work DIR --archive SMOKE \\
        [--json PATH] [--children-out DIR]
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import pidgate as pg  # noqa: E402
import resumegate as RG  # noqa: E402

GATE = 'G46'
KS = (5, 10, 12)
FIELDS = ('beta', 'error', 'integral', 'clamp_state', 'acted_on_H', 'beta_unclamped', 'sensed_H', 'slack')


def typed(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return ['bool', v]
    if isinstance(v, int):
        return ['int', v]
    if isinstance(v, float):
        return ['float', v.hex()]
    return ['str', v]


def logged(ep):
    """The PID keys a replay epoch logged, without the stock aggregate (a fresh model's counters are 0)."""
    return {k: v for k, v in ep['logs'].items() if k.startswith('pid/') and k != 'pid/stock_aggregate'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    pg.standard_args(ap)
    a = ap.parse_args()
    G = pg.Gate(GATE, a)
    g = G.g
    if G.draw_s1() is None:
        return G.finish()
    lab, R = G.run('PID-CH', None, 'H1')
    if R is None:
        return G.finish()
    rows_file = G.work / 'run-rows.json'
    rows_file.write_text(json.dumps({'frozen_rows': R['frozen_rows']}, indent=1, sort_keys=True))
    base = ['--headline', 'H1', '--rows', rows_file]
    F = G.child('replay-full', 'replay', base + ['--mode', 'full'])
    if F is None:
        return G.finish()
    bl = pg.blocks(R)
    E = len(bl)
    bad = []
    for ep in F['epochs']:
        e = ep['epoch']
        want = {f'pid/{k}': typed(bl[e].get(k)) for k in FIELDS if bl[e].get(k) is not None}
        if logged(ep) != want:
            bad.append({'epoch': e, 'replay': logged(ep), 'run': want})
    g.check(len(F['epochs']) == E and not bad,
            f'the replay on the run\'s {E} frozen readings reproduces the live controller\'s PID blocks exactly'
            + (f'; first difference {bad[0]}' if bad else ''))
    full = {ep['epoch']: ep for ep in F['epochs']}
    G.rec['resume'] = {}
    for K in KS:
        st = G.work / f'state-K{K}.json'
        A = G.child(f'replay-first-K{K}', 'replay', base + ['--mode', 'first', '--stop', K, '--state', st])
        B = G.child(f'replay-resume-K{K}', 'replay', base + ['--mode', 'resume', '--state', st])
        if A is None or B is None:
            continue
        s = A['state']
        prev = full[K - 1]
        g.check(s['epoch'] == K - 1 and s['sensed_H'] == bl[K - 1]['sensed_H']
                and [typed(s['integral']), typed(s['prev_error']), typed(s['beta'])]
                == [prev['integral'], prev['prev_error'], prev['beta']]
                and s['hex'] == {'integral': s['integral'].hex(), 'prev_error': s['prev_error'].hex(), 'beta': s['beta'].hex(),
                                 'stock_at_train_begin': None if s['stock_at_train_begin'] is None else float(s['stock_at_train_begin']).hex()}
                and B['state_read'] == s and A['state_file_sha256'] == B['state_file_sha256'],
                f'K={K}: the state after epoch {K - 1} holds epoch {s["epoch"]}, sensed_H {s["sensed_H"]}, integral '
                f'{s["integral"]!r}, previous error {s["prev_error"]!r}, beta {s["beta"]!r} as the uninterrupted '
                f'controller had them; the resumed process read the same file')
        diff = []
        for ep in B['epochs']:
            f = full[ep['epoch']]
            if (logged(ep) != logged(f) or ep['betas_after_begin'] != f['betas_after_begin']
                    or [ep['integral'], ep['prev_error'], ep['beta']] != [f['integral'], f['prev_error'], f['beta']]):
                diff.append(ep['epoch'])
        g.check([ep['epoch'] for ep in B['epochs']] == list(range(K, E)) and not diff
                and B['state']['hex'] == F['state']['hex'] and B['state']['epoch'] == F['state']['epoch'],
                f'K={K}: interrupted after epoch {K - 1} and resumed in a new process, epochs {K}-{E - 1} equal the '
                f'uninterrupted controller (PID keys, every layer\'s beta, integral, previous error, beta); final '
                f'state equal' + (f'; differing epochs {diff}' if diff else ''))
        G.rec['resume'][K] = {'state': s, 'differing_epochs': diff}
    T = G.child('replay-tamper-K12', 'replay', base + ['--mode', 'tamper', '--state', G.work / 'state-K12.json'])
    if T is not None:
        td = [ep['epoch'] for ep in T['epochs'] if logged(ep) != logged(full[ep['epoch']])]
        g.check(bool(td) and td[0] == 12, f'control: with the integral set to 0 the resumed epochs {td} differ from the '
                                           'uninterrupted controller (the comparison can fail)')
    N = G.child('replay-fresh-K12', 'replay', base + ['--mode', 'fresh', '--stop', 12])
    if N is not None:
        g.check(str(N.get('refused') or '').startswith('PIDError') and 'no frozen-H reading' in str(N.get('refused')),
                f'control: a fresh controller asked to act at epoch 12 without a restored reading is refused '
                f'({N.get("refused")})')
    whole_run(G, a)
    return G.finish()


def whole_run(G, a):
    """[B10] The whole-run part (module docstring)."""
    g = G.g
    kids = RG.Kids(g, a, work=G.work / 'whole-run', watches=G.watches, inp=G.inp, draw=G.draw, head=G.head, device='cpu',
                   env={}, rec=G.rec)
    U, _ = RG.uninterrupted(kids, 'wr-PID-CH-U', 'PID-CH', epochs=pg.EPOCHS, headline='H1')
    G.rec['whole_run'] = {}
    for k in (5, 10, 12):
        R, pts, _, log = RG.chain_checked(kids, U, f'wr-PID-CH-boundary{k}', 'PID-CH', epochs=pg.EPOCHS, every=1,
                                          kills=[f'boundary:{k}'], headline='H1')
        g.check(pts == [k], f'whole run, K={k}: killed right after the checkpoint of {k} epochs (latest.json {pts})')
        g.check(log[-1]['pre_resume_agree'], f'whole run, K={k}: the records before the resume point equal the '
                                            f'uninterrupted run\'s (fresh-run agreement, B10-14; attempts {len(log)})')
        if U is None or R is None:
            continue
        ub = [r['pid'] for r in U['rows']]
        rb = [r['pid'] for r in R['rows']]
        d = RG.diff_runs(U, R, resumed_from=k)
        g.check(len(rb) == pg.EPOCHS and rb == ub and (R.get('final') or {}).get('pid') == (U.get('final') or {}).get('pid'),
                f'whole run, K={k}: every epoch\'s PID block and the final PID state equal the uninterrupted run\'s '
                f'(integral after epoch {pg.EPOCHS - 1}: {((R.get("final") or {}).get("pid") or {}).get("integral")!r})')
        RG.checks_of_diff(g, f'whole run, K={k}', d)
        G.rec['whole_run'][k] = {'points': pts, 'differs': sorted(d)}


if __name__ == '__main__':
    sys.exit(main())
