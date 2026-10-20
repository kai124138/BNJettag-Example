"""Gate G50, the PC part (PREFLIGHT-PLAN section 3 row G50; STUDY v0.10 section 13 "Further required gates":
"training, saved and exported checkpoint identity"; modules M14 cbl/resume.py, M9 cbl/front.py, M8 cbl/frozen_eval.py
through M13 cbl/runner.py). PC CPU, venv of record, 2-file smoke archive, seed 1. The exported-checkpoint link of the
chain is the export batch's (M18, B14, mulder) and is not here.

The hash chain. A run's state at the end of epoch e is named by `vars_sha256_after` in its epoch record (M8's frozen
evaluation hashes every variable by role, cbl.rng.variables_sha256). Each saved form of that state must carry the same
name, and a fresh process that loads it must compute it again:
1. training -> front checkpoint: each sidecar's `state_sha256` equals the epoch record's `vars_sha256_after`; its
   `ckpt_sha256` equals the event line's and the run summary's, and the sha256 of the file on disk; the file, loaded by
   keras.models.load_model in a fresh process, hashes by role to the same state hash (the file's bytes are not the
   identity: Keras writes a save time, B8-2);
2. training -> resume checkpoint: each kept checkpoint's `state.json` names the state hash of the epoch it follows,
   its files hash as recorded, and its `variables.npz` assigned by role into a model built as the runner builds the arm
   hashes to the same value; every checkpoint the run wrote (the resumed process's record) names the epoch record's
   hash;
3. training -> run record: run.json's `final_variables_sha256` equals the last epoch record's.
The runs cross a resume (CH-8 and F111-BF with headline H2, 6 epochs, killed by SIGKILL after the checkpoint of 3
epochs and resumed in a new process), so the chain also holds across it. Finding B7-F1 is recorded beside: the
reloaded configurations carry beta0 1e-5 at the MHA's softmax and tables where the built model has 0, and the state
hash, which reads every beta variable by value, is unchanged. Control: a copy of a checkpoint with one kernel entry
changed, saved and reloaded, hashes differently (the chain check can fail). Every reading is engineering, not
quotable.

    <venv>/bin/python -I -B gates/g50_ckpt_chain.py --expect-sha12 SHA12 --work DIR --archive SMOKE [--json PATH]
        [--children-out DIR]
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))
sys.path.insert(0, str(FORK / 'gates'))

import resumegate as RG  # noqa: E402
import runnergate as rgt  # noqa: E402

GATE = 'G50'
ARMS = (('CH-8', None, None), ('F111', 'BF', 'H2'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    RG.standard_args(ap, devices=('cpu',))
    a = ap.parse_args()
    G = RG.Gate(GATE, a)
    g = G.g
    if G.draw_s1() is None:
        return G.finish()
    K = G.kids
    G.rec['chains'] = {}
    for arm, form, headline in ARMS:
        lab = arm if not form else f'{arm}-{form}-{headline}'
        R, pts, d = RG.chain(K, f'chain-{lab}', arm, epochs=6, every=1, kills=['boundary:3'], form=form, headline=headline)
        g.check(pts == [3], f'{lab}: killed after the checkpoint of 3 epochs and resumed ({pts})')
        if R is None:
            continue
        rows = {r['epoch']: r for r in (json.loads(x) for x in (d / 'epochs.jsonl').read_text().splitlines() if x)}
        g.check(sorted(rows) == list(range(6)), f'{lab}: epoch records 0-5 after the resume')
        side = {int(k): v for k, v in (R.get('front') or {}).get('sidecars', {}).items()}
        raw_side = {}
        for p in sorted((d / 'front' / 'sidecars').glob('epoch=*.json')):
            s = json.loads(p.read_text())
            raw_side[s['epoch']] = s
        events = {}
        for x in (d / 'front' / 'events.jsonl').read_text().splitlines():
            if x:
                e = json.loads(x)
                events[e['epoch']] = e
        summ = json.loads((d / 'front' / 'summary.json').read_text())
        ck = summ.get('checkpoints') or {}
        run_json = json.loads((d / 'run.json').read_text())
        on_disk = [e for e, s in raw_side.items() if (d / s['ckpt_path']).exists()]
        keras_files = [str(d / raw_side[e]['ckpt_path']) for e in sorted(on_disk)]
        resume_dirs = sorted(p for p in (d / 'resume').glob('ckpt-*') if p.is_dir())
        states = {p.name: json.loads((p / 'state.json').read_text()) for p in resume_dirs}
        npz = [str(p / 'variables.npz') for p in resume_dirs]
        files = G.work / f'reload-{lab}.json'
        files.write_text(json.dumps({'keras': keras_files, 'npz': npz}))
        extra = ['--out-dir', G.work / 'reload-dummy', '--files', files, '--arm', arm,
                 '--mutate', G.work / f'mutated-{lab}.keras']
        if form:
            extra += ['--form', form, '--headline', headline]
        L = K.child(f'reload-{lab}', 'reload', extra)
        g.check(bool(raw_side) and set(side) == set(raw_side), f'{lab}: {len(raw_side)} sidecars ({sorted(raw_side)}), '
                                                               f'{len(on_disk)} checkpoint files still on disk')
        bad1 = [e for e, s in raw_side.items() if s['state_sha256'] != rows[e]['vars_sha256_after']]
        g.check(not bad1, f'{lab}: link 1, every sidecar\'s state_sha256 equals its epoch record\'s vars_sha256_after'
                          + (f' (differs at {bad1})' if bad1 else ''))
        bad2 = [e for e, s in raw_side.items()
                if not (events.get(e, {}).get('ckpt_sha256') == s['ckpt_sha256'] == (ck.get(str(e)) or {}).get('sha256')
                        and (ck.get(str(e)) or {}).get('state_sha256') == s['state_sha256'])]
        g.check(not bad2, f'{lab}: link 1, every sidecar\'s ckpt_sha256 equals its event line\'s and the run summary\'s, '
                          f'and the summary names the same state_sha256' + (f' (differs at {bad2})' if bad2 else ''))
        bad3 = [e for e in on_disk if rgt.sha256_file(d / raw_side[e]['ckpt_path']) != raw_side[e]['ckpt_sha256']]
        g.check(not bad3, f'{lab}: link 1, every checkpoint file on disk hashes to its recorded ckpt_sha256'
                          + (f' (differs at {bad3})' if bad3 else ''))
        rel = (L or {}).get('reload') or {}
        bad4 = [e for e in on_disk if (rel.get('keras', {}).get(str(d / raw_side[e]['ckpt_path'])) or {}).get('state_sha256')
                != raw_side[e]['state_sha256']]
        g.check(L is not None and not bad4, f'{lab}: link 1, each checkpoint file loaded in a fresh process hashes by role '
                                            f'to its state_sha256 ({len(on_disk)} files)' + (f' (differs at {bad4})' if bad4 else ''))
        bad5 = []
        for name, st in states.items():
            e = st['epochs_done'] - 1
            npzp = str(d / 'resume' / name / 'variables.npz')
            okf = all(rgt.sha256_file(d / 'resume' / name / f) == h for f, h in st['files'].items())
            if not (okf and e in rows and st['variables']['state_sha256'] == rows[e]['vars_sha256_after']
                    and (rel.get('npz', {}).get(npzp) or {}).get('state_sha256') == st['variables']['state_sha256']):
                bad5.append(name)
        g.check(len(states) == 2 and not bad5,
                f'{lab}: link 2, each kept resume checkpoint ({sorted(states)}) names the state hash of the epoch it '
                f'follows, its files hash as recorded, and its variables assigned by role in a fresh process hash to it'
                + (f' (differs: {bad5})' if bad5 else ''))
        wr = (run_json.get('resume') or {}).get('written') or []
        bad6 = [w['epochs_done'] for w in wr if w['state_sha256'] != rows[w['epochs_done'] - 1]['vars_sha256_after']]
        g.check(len(wr) == 3 and not bad6, f'{lab}: link 2, every checkpoint the resumed process wrote ({[w["epochs_done"] for w in wr]}) '
                                           f'names its epoch record\'s state hash' + (f' (differs at {bad6})' if bad6 else ''))
        g.check(run_json['final_variables_sha256'] == rows[5]['vars_sha256_after'],
                f'{lab}: link 3, run.json\'s final_variables_sha256 equals epoch 5\'s vars_sha256_after')
        mut = rel.get('mutated') or {}
        src_state = (rel.get('keras', {}).get(mut.get('source')) or {}).get('state_sha256')
        g.check(bool(mut) and mut.get('state_sha256') not in (None, src_state),
                f'{lab}: control, a copy with one entry of {mut.get("role")} changed hashes differently (the check can fail)')
        b0 = sorted({b for v in rel.get('keras', {}).values() for _, b in v['beta0']})
        built_b0 = sorted({b for _, b in ((R.get('final') or {}).get('layers') or {}).get('beta0', [])})
        g.info(f'{lab} beta0 values (B7-F1, disclosed, not a failure)', {'built model': built_b0, 'reloaded files': b0})
        G.rec['chains'][lab] = {'sidecars': sorted(raw_side), 'on_disk': on_disk, 'resume_checkpoints': sorted(states),
                                'written': [w['epochs_done'] for w in wr], 'beta0_built': built_b0, 'beta0_reloaded': b0}
    return G.finish()


if __name__ == '__main__':
    sys.exit(main())
