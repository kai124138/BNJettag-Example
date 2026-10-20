"""Gate G37 (PREFLIGHT-PLAN section 3 row G37; STUDY v0.10 section 8.1: the front saver, nondominated under H
with the ties, Chang's filter in H units, the diagnostic checkpoints outside the front, exact sidecars, the step
envelope giving NA and never zero; module M9 cbl/front.py). MAC and PCC.

Pure part (this process; any interpreter with NumPy and SciPy, no Keras): constructed per-epoch frozen records
(interfaces section 1) through `cbl.front.FrontSaver` with stand-in save, state, in-loop and audit functions
that write known bytes. Checks, for hand-built cases with known answers and for random sequences:
- `import cbl.front` loads no Keras, JAX, hgq or TensorFlow (fresh -I interpreter);
- every front equals the brute-force filtered nondominated set (equal pairs keep the earlier epoch) during and
  after the run, under each count with Chang's filter in that count's units; members have distinct top-1 and
  cost; the hand cases' joins, leaves and ties are as derived by hand;
- the diagnostic checkpoints (best top-1 at cost <= 350,000 ignoring the filter; minimum cost) equal brute
  force, are saved, and are outside the front where they fail the filter;
- an epoch is saved exactly when it joins a front or takes a diagnostic role; the sidecar holds the exact
  integers of its record, ckpt_sha256 equals the sha256 of the bytes written, and with keep_superseded false
  the files on disk are exactly the live epochs;
- the summary's step envelope equals brute force, "NA" where nothing is feasible and never 0, slack = g - cost;
  filter outcomes COST INFEASIBLE / NO ACCEPTED CHECKPOINT with the brute-force diagnostic point; dwell equals
  M16's; M16's reader (`run_from_records`, `check_pid_summary`) accepts the summary and refuses three tampered
  copies (so the check can fail);
- the events ledger replays to the incremental state; a saver restored from a snapshot and continued ends in
  the summary of the uninterrupted run;
- refusals: missing or non-integer record fields, an epoch mismatch, an out-of-order epoch, no ablation mode,
  no keep_superseded, a PID run without a headline; chain_evaluator reads the record M8 returns or leaves;
- AUC equals the brute-force pairwise AUC with ties, as an exact fraction.
Keras part (--device cpu; PCC): `gates/b8_child.py g37` under the open-file audit (gates/b8gate.py): the saver
bound to real models through `cbl.front.bind`. Every reading is engineering, not quotable. Run:

    <python> -I -B gates/g37_front.py --expect-sha12 SHA12 --work DIR [--device cpu] [--json PATH] \\
        [--children-out DIR] [--data-manifest PATH]
"""

import argparse
import hashlib
import json
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import gatelib  # noqa: E402

PRIVATE_SEED = 20261018


class Stand:
    """Stand-ins for the Keras functions: the 'state' is the current epoch; the file holds known bytes."""

    def __init__(self):
        self.epoch = None
        self.saves = {}

    def bytes_of(self, epoch):
        return json.dumps({'stand-in checkpoint': epoch}).encode()

    def save_fn(self, path):
        b = self.bytes_of(self.epoch)
        Path(path).write_bytes(b)
        self.saves[self.epoch] = hashlib.sha256(b).hexdigest()

    def state_fn(self):
        return hashlib.sha256(f'state {self.epoch}'.encode()).hexdigest()

    def inloop_fn(self):
        return {'metrics': {'correct': -1, 'n': -1, 'val_auc_per_class': [None] * 5, 'val_auc_macro': None,
                            'logits_sha256': 'stand-in'},
                'structure_record': {'degenerate': {k: None for k in ('Q', 'K', 'V', 'O', 'FFN')}, 'stand_in': True},
                'attention_diagnostics': None, 'seconds': {}}

    def audit_fn(self, path, record):
        return {'stand_in_audit': True, 'delta_attn_mean': 0.0, 'delta_static': 0.0, 'delta_unif': 0.0, 'H_attn': 2.0,
                'overflow': {}, 'file_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def make_saver(F, d, stand, *, schedule='ramp', headline='H1', ablations='run_end', keep=False):
    return F.FrontSaver(d, run_id='g37-run', config_sha256='c' * 64, code_sha='d' * 40, headline=headline,
                        schedule=schedule, save_fn=stand.save_fn, state_fn=stand.state_fn, inloop_fn=stand.inloop_fn,
                        audit_fn=stand.audit_fn, ablations=ablations, keep_superseded=keep)


def rec(e, correct, h1, h2=None, nr=None, n=1000):
    return {'epoch': e, 'val_correct': correct, 'val_n': n, 'H1': h1, 'H2': h1 if h2 is None else h2,
            'NR': (h1 if h2 is None else max(h1, h2)) if nr is None else nr, 'lr': 0.003, 'beta': 2e-8}


def feed(saver, stand, records):
    evs = []
    for r in records:
        stand.epoch = r['epoch']
        evs.append(saver.on_frozen_record(r['epoch'], r))
    return evs


def random_records(seed, n_epochs=400, n=1000, floor=0):
    """Cost falls with noise to `floor`; accuracy drifts with noise, some below the filter; exact repeats and
    near ties."""
    g = random.Random(seed)
    out = []
    acc = 600
    for e in range(n_epochs):
        base = max(floor, int(700000 * (0.993 ** e)) + g.randint(-15000, 15000))
        acc = max(300, min(900, acc + g.randint(-12, 11)))
        h1 = max(0, base)
        h2 = max(0, h1 - g.randint(0, 40000))
        nr = h1 + g.randint(0, 30000)
        if out and g.random() < 0.06:                       # an exact repeat of the previous epoch's values
            p = out[-1]
            out.append({**p, 'epoch': e})
            continue
        if out and g.random() < 0.05:                       # equal accuracy at a new cost
            acc = out[-1]['val_correct']
        out.append(rec(e, acc, h1, h2, nr, n))
    return out


def brute(F, recs):
    """Expected fronts, diagnostics and per-g best (filter ignored) by brute force over Recs."""
    sel = F.sel
    exp = {}
    for c in F.COUNTS:
        fr = F.batch_front(recs, c)
        b350 = [r for r in recs if r.c(c) <= 350000]
        best350 = min(b350, key=lambda r: (-r.top1, r.c(c), r.epoch)) if b350 else None
        mn = min(recs, key=lambda r: (r.c(c), -r.top1, r.epoch)) if recs else None
        env, fo = {}, {}
        for gg in sel.GRID:
            feas = [r for r in fr if r.c(c) <= gg]
            env[gg] = min(feas, key=lambda r: (-r.top1, r.c(c), r.epoch)) if feas else None
            anyg = [r for r in recs if r.c(c) <= gg]
            fo[gg] = ('feasible' if env[gg] else (sel.COST_INFEASIBLE if not anyg else sel.NO_ACCEPTED),
                      min(anyg, key=lambda r: (-r.top1, r.c(c), r.epoch)) if anyg else None)
        exp[c] = {'front': sorted(r.epoch for r in fr), 'best350': best350, 'min': mn, 'env': env, 'fo': fo}
    return exp


def check_summary_against(g, F, summ, exp, stand, label):
    ok_front = ok_env = ok_fo = ok_diag = True
    for c in F.COUNTS:
        got = [p['epoch'] for p in summ['front'][c]]
        ok_front &= got == exp[c]['front']
        for gg in F.GRID:
            e = summ['envelope'][c][str(gg)]
            want = exp[c]['env'][gg]
            if want is None:
                ok_env &= e == 'NA'
            else:
                ok_env &= (isinstance(e, dict) and e['epoch'] == want.epoch and e['cost'] == want.c(c)
                           and e['slack'] == gg - want.c(c) and e['val_correct'] == want.correct
                           and e['ckpt_sha256'] == stand.saves[want.epoch])
            o = summ['filter_outcome'][c][str(gg)]
            wo, wd = exp[c]['fo'][gg]
            ok_fo &= o['outcome'] == wo
            if wo == F.sel.NO_ACCEPTED:
                ok_fo &= o['diagnostic']['epoch'] == wd.epoch and o['diagnostic']['cost'] == wd.c(c)
        dg = summ['diagnostic_ckpts'][c]
        b, m = exp[c]['best350'], exp[c]['min']
        ok_diag &= ((dg['best_at_350000'] is None) if b is None else (dg['best_at_350000']['epoch'] == b.epoch))
        ok_diag &= dg['min_cost']['epoch'] == m.epoch and dg['min_cost']['ckpt_sha256'] == stand.saves[m.epoch]
    never0 = all(e != 0 for c in F.COUNTS for e in summ['envelope'][c].values())
    g.check(ok_front, f'{label}: every front equals the brute-force filtered nondominated set')
    g.check(ok_env and never0, f'{label}: step envelope equals brute force with ties, NA where none, never 0, slack = g - cost')
    g.check(ok_fo, f'{label}: filter outcomes (feasible, COST INFEASIBLE, NO ACCEPTED CHECKPOINT with its diagnostic point)')
    g.check(ok_diag, f'{label}: diagnostic checkpoints equal brute force (best at <= 350,000 ignoring the filter; minimum cost)')


def strip_seconds(o):
    if isinstance(o, dict):
        return {k: strip_seconds(v) for k, v in o.items() if 'seconds' not in k}
    if isinstance(o, list):
        return [strip_seconds(v) for v in o]
    return o


def pure_checks(g, work):
    from cbl import front as F
    from cbl import select as sel

    info = {}
    # 1. import isolation
    code = ('import sys; sys.path.insert(0, %r); import cbl.front; '
            'print(sorted(m for m in ("keras", "jax", "hgq", "tensorflow", "torch") if m in sys.modules))') % str(FORK)
    r = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True, timeout=300)
    g.check(r.returncode == 0 and r.stdout.strip() == '[]',
            f'import cbl.front in a fresh -I interpreter loads none of keras, jax, hgq, tensorflow, torch ({r.stdout.strip()!r})')

    # 2. hand case with known answers
    d = work / 'hand' / 'front'
    st = Stand()
    sv = make_saver(F, d, st)
    hand = [rec(0, 600, 400000),                 # joins every front
            rec(1, 600, 400000),                 # equal pair: the earlier epoch stays, 1 does not join
            rec(2, 600, 390000),                 # equal top-1, lower cost: joins, 0 leaves
            rec(3, 610, 390000),                 # equal cost, higher top-1: joins, 2 leaves
            rec(4, 500, 150000),                 # top-1 0.5 fails the filter (> 0.5): diagnostics only
            rec(5, 700, 500000),                 # cost 500,000 fails the filter (< 500,000)
            rec(6, 700, 499999),                 # joins
            rec(7, 650, 600000, h2=300000, nr=600000),   # H1 and NR fail the filter, H2 joins alone
            rec(8, 200, 120000)]                 # top-1 0.2 at 120,000: new minimum cost, diagnostics only
    evs = feed(sv, st, hand)
    joined = {e['epoch']: e['joined'] for e in evs}
    left = {e['epoch']: e['left'] for e in evs}
    g.check(joined[0] == ['H1', 'H2', 'NR'] and joined[1] == [] and joined[2] == ['H1', 'H2', 'NR']
            and left[2] == {c: [0] for c in F.COUNTS} and joined[3] == ['H1', 'H2', 'NR'] and left[3] == {c: [2] for c in F.COUNTS},
            'hand case: an equal pair keeps the earlier epoch; equal top-1 at lower cost and equal cost at higher top-1 '
            'replace the member')
    g.check(joined[4] == [] and joined[5] == [] and joined[6] == ['H1', 'H2', 'NR'] and joined[7] == ['H2'] and joined[8] == [],
            'hand case: top-1 = 0.5 and cost = 500,000 fail Chang\'s filter; the filter is applied in each count\'s units '
            '(epoch 7 joins H2 only)')
    diag = {e['epoch']: e['diagnostic'] for e in evs}
    g.check('H1:best_at_350000' in diag[4] and 'H1:min_cost' in diag[4] and 'H1:min_cost' in diag[8]
            and 'H1:best_at_350000' not in diag[8] and 'H2:best_at_350000' in diag[7],
            'hand case: diagnostic roles (epoch 4 best at <= 350,000 and minimum cost; epoch 8 minimum cost only, its '
            'top-1 is lower; epoch 7 best at <= 350,000 under H2)')
    saved = sorted(e['epoch'] for e in evs if e['saved'])
    g.check(saved == [0, 2, 3, 4, 6, 7, 8], f'hand case: saved epochs {saved} = joins and diagnostic roles [0, 2, 3, 4, 6, 7, 8]')
    summ = sv.finalize(end='completed', epochs_completed=len(hand))
    fe = {c: [p['epoch'] for p in summ['front'][c]] for c in F.COUNTS}
    g.check(fe == {'H1': [3, 6], 'H2': [6, 7], 'NR': [3, 6]},
            f'hand case: final fronts {fe} (under H2, epoch 7 at 300,000 with top-1 0.65 removes epoch 3)')
    e1 = summ['envelope']['H1']
    fo1 = summ['filter_outcome']['H1']
    g.check(fo1['100000']['outcome'] == sel.COST_INFEASIBLE and e1['100000'] == 'NA'
            and fo1['200000']['outcome'] == sel.NO_ACCEPTED and fo1['200000']['diagnostic']['epoch'] == 4
            and fo1['350000']['outcome'] == sel.NO_ACCEPTED and e1['350000'] == 'NA'
            and e1['400000']['epoch'] == 3 and e1['400000']['slack'] == 10000 and e1['450000']['epoch'] == 3,
            'hand case H1: COST INFEASIBLE at 100,000 (the cheapest epoch costs 120,000); NO ACCEPTED CHECKPOINT at '
            '200,000 to 350,000 with epoch 4 (top-1 0.5) as the diagnostic point; epoch 3 at 400,000 with slack 10,000 '
            'and at 450,000')
    info['hand_H1_filter_outcome'] = fo1
    g.check(summ['envelope']['H2']['300000']['epoch'] == 7 and summ['filter_outcome']['H2']['300000']['outcome'] == 'feasible',
            'hand case H2: epoch 7 (H2 300,000) is the envelope point at 300,000')
    files = sorted(p.name for p in (d / 'ckpt').glob('*.keras'))
    live = sorted(sv.state.live_epochs())
    g.check(files == [f'epoch={e}.keras' for e in live] and live == [3, 4, 6, 7, 8],
            f'hand case, keep_superseded false: files on disk {files} are exactly the live epochs {live}')
    sc = json.loads((d / 'sidecars' / 'epoch=4.json').read_text())
    g.check(sc['val_correct'] == 500 and sc['val_n'] == 1000 and sc['H1'] == 150000 and sc['joined'] == []
            and sorted(sc['diagnostic']) == sorted(['H1:best_at_350000', 'H1:min_cost', 'H2:best_at_350000', 'H2:min_cost',
                                                     'NR:best_at_350000', 'NR:min_cost'])
            and sc['chang_filter'] == {'H1': False, 'H2': False, 'NR': False}
            and sc['ckpt_sha256'] == hashlib.sha256(st.bytes_of(4)).hexdigest(),
            'hand case: the diagnostic checkpoint epoch 4 has its sidecar with exact integers, no front, its roles, the '
            'filter false, and ckpt_sha256 = sha256 of the bytes written')
    g.check(all(p['epoch'] not in (4, 8) for c in F.COUNTS for p in summ['front'][c]),
            'hand case: the diagnostic checkpoints that fail the filter are in no front')
    g.check(summ['m16_reader'] == {'ok': True}, f'hand case: M16 reader on the summary {summ["m16_reader"]}')
    att = sorted(p.name for p in (d / 'attention').glob('*.json'))
    g.check(att == [f'epoch={e}.json' for e in live] and summ['attention_audited'] == live,
            f'hand case, run_end mode: the audit wrote attention diagnostics for the live epochs only ({att})')
    sidecars = F.load_sidecars(d)
    g.check(set(sidecars) == {st.saves[e] for e in saved}
            and all((s['attention_diagnostics'] is not None) == (s['epoch'] in live) for s in sidecars.values()),
            'load_sidecars keys every sidecar by ckpt_sha256 and merges the attention file where it exists')

    # 3. random sequences, both deletion policies
    n_cases = 0
    for seed in range(1, 7):
        recs = random_records(PRIVATE_SEED + seed, floor=250000 if seed >= 5 else 0)
        keep = seed % 2 == 0
        d = work / f'rand{seed}' / 'front'
        st = Stand()
        sv = make_saver(F, d, st, keep=keep)
        ok_inc = True
        R = []
        for k, r in enumerate(recs):
            st.epoch = r['epoch']
            sv.on_frozen_record(r['epoch'], r)
            R.append(F.rec_from(r['epoch'], r))
            if k % 40 == 39:
                for c in F.COUNTS:
                    ok_inc &= sorted(m.epoch for m in sv.state.front[c]) == sorted(x.epoch for x in F.batch_front(R, c))
        exp = brute(F, R)
        label = f'random sequence {seed} ({len(recs)} epochs, keep_superseded {keep}{", cost floor 250,000" if seed >= 5 else ""})'
        g.check(ok_inc, f'{label}: the incremental fronts equal brute force after every 40 epochs')
        distinct = all(len({m.top1 for m in sv.state.front[c]}) == len(sv.state.front[c]) ==
                       len({m.c(c) for m in sv.state.front[c]}) for c in F.COUNTS)
        filt = all(F.passes_filter(m, c) for c in F.COUNTS for m in sv.state.front[c])
        g.check(distinct and filt, f'{label}: members pass the filter in their count and have distinct top-1 and cost')
        evs = F.read_events(d)
        saved_ok = all(e['saved'] == bool(e['joined'] or e['diagnostic']) for e in evs)
        g.check(saved_ok and len(evs) == len(recs), f'{label}: saved exactly at joins and diagnostic roles; one event per epoch')
        summ = sv.finalize(end='completed', epochs_completed=len(recs))
        check_summary_against(g, F, summ, exp, st, label)
        rows = [{'epoch': x.epoch, 'val_correct': x.correct, 'val_n': x.n, c: x.c(c)} for x in R for c in ('H1',)]
        g.check(all(summ['dwell'][c] == sel.dwell([{'epoch': x.epoch, 'val_correct': x.correct, 'val_n': x.n, c: x.c(c)} for x in R], c)
                    for c in F.COUNTS), f'{label}: dwell equals M16\'s')
        del rows
        g.check(summ['m16_reader'] == {'ok': True}, f'{label}: M16\'s run_from_records accepts the summary')
        files = sorted(int(p.name[6:-6]) for p in (d / 'ckpt').glob('*.keras'))
        want_files = sorted(sv.state.live_epochs()) if not keep else sorted(sv.saved)
        g.check(files == want_files, f'{label}: files on disk ({len(files)}) are exactly the '
                                     f'{"saved" if keep else "live"} epochs')
        side_ok = True
        for e, s in sv.saved.items():
            sc = json.loads(sv.sidecar_path(e).read_text())
            r0 = recs[e]
            side_ok &= (sc['val_correct'] == r0['val_correct'] and sc['val_n'] == r0['val_n'] and sc['H1'] == r0['H1']
                        and sc['H2'] == r0['H2'] and sc['NR'] == r0['NR'] and sc['ckpt_sha256'] == st.saves[e]
                        and all(isinstance(sc[k], int) and not isinstance(sc[k], bool) for k in ('val_correct', 'val_n', 'H1', 'H2', 'NR'))
                        and sc['chang_filter'] == {c: F.passes_filter(F.rec_from(e, r0), c) for c in F.COUNTS})
        g.check(side_ok, f'{label}: every sidecar holds its record\'s exact integers, the filter per count and the '
                         f'sha256 of the bytes written')
        for c in F.COUNTS:
            for gg in F.GRID:
                o = summ['filter_outcome'][c][str(gg)]['outcome']
                info.setdefault('outcomes_seen', {}).setdefault(o, 0)
                info['outcomes_seen'][o] += 1
        rp = F.replay(F.recs_from_events(evs))
        g.check(rp.state() == sv.state.state(), f'{label}: the events ledger replays to the incremental state')
        n_cases += 1
        if seed == 1:
            info['random1'] = {'saved': len(sv.saved), 'live': len(sv.state.live_epochs()),
                               'front_sizes': {c: len(sv.state.front[c]) for c in F.COUNTS}}
            # M16 refuses tampered copies (the reader check can fail)
            t1 = json.loads(json.dumps(summ))
            g_na = next((gg for gg in F.GRID if t1['envelope']['H1'][str(gg)] == 'NA'), None)
            g_pt = next((gg for gg in F.GRID if t1['envelope']['H1'][str(gg)] != 'NA'), None)
            bad = []
            if g_pt is not None:
                t = json.loads(json.dumps(summ)); t['envelope']['H1'][str(g_pt)] = 0; bad.append(t)
                t = json.loads(json.dumps(summ))
                others = [p for p in t['front']['H1'] if p['epoch'] != t['envelope']['H1'][str(g_pt)]['epoch']]
                if others:
                    t['envelope']['H1'][str(g_pt)] = dict(others[0], slack=g_pt - others[0]['cost']); bad.append(t)
            t = json.loads(json.dumps(summ))
            t['front']['H1'] = t['front']['H1'] + [{'epoch': 9999, 'ckpt_sha256': 'e' * 64, 'val_correct': 400, 'val_n': 1000, 'cost': 1}]
            bad.append(t)
            refused = [not F.m16_check(t, 'H1', 'ramp')['ok'] for t in bad]
            g.check(len(bad) >= 2 and all(refused), f'M16 refuses {sum(refused)} of {len(bad)} tampered summaries '
                                                    f'(envelope 0, another front point, a point failing the filter)')
            info['na_grid_example'] = g_na
            # snapshot and restore: interrupted at epoch 200 and continued equals uninterrupted
            da = work / 'resumeA' / 'front'
            sa = Stand()
            A = make_saver(F, da, sa)
            feed(A, sa, recs)
            sumA = A.finalize(end='completed', epochs_completed=len(recs))
            db = work / 'resumeB' / 'front'
            sb = Stand()
            B = make_saver(F, db, sb)
            feed(B, sb, recs[:200])
            snap = json.loads(json.dumps(B.snapshot()))
            C = make_saver(F, db, sb)
            C.restore(snap)
            feed(C, sb, recs[200:])
            sumC = C.finalize(end='completed', epochs_completed=len(recs))
            keys = ('front', 'envelope', 'filter_outcome', 'diagnostic_ckpts', 'dwell', 'live_epochs', 'checkpoints')
            g.check(all(strip_seconds(sumA[k]) == strip_seconds(sumC[k]) for k in keys),
                    'a saver restored from a JSON snapshot at epoch 200 and continued ends in the uninterrupted run\'s '
                    'fronts, envelopes, filter outcomes, diagnostics, dwell and checkpoint table')

    # 4. PID run
    recs = random_records(PRIVATE_SEED + 99)
    d = work / 'pid' / 'front'
    st = Stand()
    sv = make_saver(F, d, st, schedule='pid', headline='H2')
    feed(sv, st, recs)
    summ = sv.finalize(end='stopif', epochs_completed=len(recs))
    cfg = {'arm_id': 'PID-CH', 'seed': 1, 'stage': 'gate', 'schedule': 'pid'}
    run = sel.run_from_records(cfg, summ)
    mine = sel.check_pid_summary(run, 'H2')
    g.check(summ['m16_reader'] == {'ok': True} and summ['pid_selection']['count'] == 'H2'
            and (mine['point'] or {}).get('ckpt_sha256') == summ['pid_selection']['ckpt_sha256'],
            f'PID run: pid_selection is the H2 envelope at 350,000 and M16\'s check_pid_summary agrees ({mine["status"]})')

    # 5. refusals
    def refuses(fn):
        try:
            fn()
        except F.FrontError:
            return True
        return False

    d = work / 'refuse' / 'front'
    st = Stand()
    sv = make_saver(F, d, st)
    st.epoch = 0
    r0 = rec(0, 600, 400000)
    g.check(refuses(lambda: sv.on_frozen_record(0, {k: v for k, v in r0.items() if k != 'NR'}))
            and refuses(lambda: sv.on_frozen_record(0, dict(r0, H1=400000.0)))
            and refuses(lambda: sv.on_frozen_record(0, dict(r0, val_correct=True)))
            and refuses(lambda: sv.on_frozen_record(1, r0))
            and refuses(lambda: sv.on_frozen_record(0, dict(r0, val_correct=1001))),
            'refused: a missing count, a float count, a bool count, an epoch that differs from the callback\'s, '
            'correct > n')
    sv.on_frozen_record(0, r0)
    g.check(refuses(lambda: sv.on_frozen_record(0, r0)), 'refused: an epoch offered twice (out of order)')
    g.check(refuses(lambda: make_saver(F, work / 'r2' / 'front', Stand(), ablations=None))
            and refuses(lambda: make_saver(F, work / 'r3' / 'front', Stand(), keep=None))
            and refuses(lambda: make_saver(F, work / 'r4' / 'front', Stand(), schedule='pid', headline=None)),
            'refused: no ablation mode, no keep_superseded, a PID run without a headline (no defaults)')

    # 6. chain_evaluator
    d = work / 'chain' / 'front'
    st = Stand()
    sv = make_saver(F, d, st)
    logs = {'loss': 1.0, **rec(0, 600, 400000)}
    st.epoch = 0
    F.chain_evaluator(lambda e, lg: None, sv)(0, logs)
    st.epoch = 1
    F.chain_evaluator(lambda e, lg: rec(1, 610, 390000), sv)(1, {'loss': 1.0})
    evs = F.read_events(d)
    g.check([e['epoch'] for e in evs] == [0, 1] and evs[1]['val_correct'] == 610,
            'chain_evaluator passes the record M8 leaves in the logs, or the one it returns')

    # 7. AUC
    rng = random.Random(PRIVATE_SEED)
    ok_auc = True
    for trial in range(6):
        n = 300
        y = [rng.randrange(5) for _ in range(n)]
        s = [rng.randrange(7) / 7 for _ in range(n)]       # many ties
        for k in range(5):
            pos = [s[i] for i in range(n) if y[i] == k]
            neg = [s[i] for i in range(n) if y[i] != k]
            bf = Fraction(sum(2 * (a > b) + (a == b) for a in pos for b in neg), 2 * len(pos) * len(neg))
            ok_auc &= F.auc_one_vs_rest(s, y, k) == float(bf)
    g.check(ok_auc, 'one-vs-rest AUC with ties equals the brute-force pairwise AUC exactly (30 cases)')

    # 8. released front parse
    class P:
        paths = [work / 'pareto' / 'epoch=12-val_acc=0.612-train_acc=0.600-ebops=123456-val_loss=1.000.keras']
        record = [[0.6123, 123456.0]]
    rf = F.released_front_of(P, work)
    g.check(rf == [{'epoch': 12, 'train_count': 123456.0, 'log_val_accuracy': 0.6123,
                    'path': 'pareto/epoch=12-val_acc=0.612-train_acc=0.600-ebops=123456-val_loss=1.000.keras'}],
            'released_front_of reads epoch, training count and accuracy of Chang\'s ParetoFront record')
    info['random_cases'] = n_cases
    g.check(set(info['outcomes_seen']) == {'feasible', sel.COST_INFEASIBLE, sel.NO_ACCEPTED},
            f'the random sequences reach all three filter outcomes ({info["outcomes_seen"]})')
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--expect-sha12', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--device', choices=['cpu'], default=None)
    ap.add_argument('--data-manifest', default=str(FORK / 'manifests' / 'data-prep-150c.json'))
    ap.add_argument('--json', default=None)
    ap.add_argument('--children-out', default=None)
    a = ap.parse_args()
    g = gatelib.Gate('G37')
    rec_ = {'header': g.header(a.expect_sha12)}
    print('label       engineering, not quotable')
    work = Path(a.work)
    g.check(not work.exists(), f'work directory {work} does not exist yet')
    work.mkdir(parents=True)
    if a.device:
        gatelib.check_venv_of_record(g)
    rec_['pure'] = pure_checks(g, work / 'pure')
    if a.device:
        import b8gate
        child = b8gate.run_child(g, 'g37', a.device, work, a.data_manifest, a.children_out)
        rec_['child'] = {'setup': (child or {}).get('setup'), 'info': (child or {}).get('info'),
                         'n_checks': len((child or {}).get('checks', []))}
    g.check_clean('at end')
    return g.finish(rec_, a.json)


if __name__ == '__main__':
    sys.exit(main())
