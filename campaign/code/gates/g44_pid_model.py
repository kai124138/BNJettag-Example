"""Gate G44 (PREFLIGHT-PLAN section 3 row G44; STUDY v0.10 section 7.2 "Aggregation" (chang C2): the callbacks
receive Chang's functional model object itself, never a wrapper, because get_ebops iterates the top-level
model.layers (beta_pid.py:44-47); gate: the stock aggregate is > 0 after epoch 0; module M12 cbl/pid.py through M13
cbl/runner.py). PC CPU, venv of record, 2-file smoke archive.

Children as G43 (gates/pidgate.py; one PID run per arm, 14 epochs).

Checks, per run:
1. the PID callback's model is the runner's model object (identity, and the id it saw at on_train_begin), which is
   the object the arm's builder returned (Chang's get_model for PID-CH, cbl.model's builder for PID-BEST), a Keras
   Functional with the 14 top-level layers of the xfm graph and the MHA at index 3 (QMultiHeadAttention for PID-CH,
   the binary MHA for PID-BEST F111);
2. the callback is cbl.pid's LabBetaPID, a subclass of hgq2's BetaPID (and BaseBetaPID), whose PID object
   subclasses the stock PID;
3. the stock aggregate (the stock get_ebops, logged beside the sensed value) is > 0 at the end of every epoch,
   epoch 0 included; its value before training (on_train_begin) is recorded;
4. at every epoch end the stock aggregate equals FreeEBOPs's training count of the same epoch (the top-level public
   aggregate against the flattened private total, G31's equality, here inside a PID run);
5. controls on the trained model after fit: the stock aggregate of the model is > 0 and equals the private total;
   the same stock function on a functional wrapper around the model gives 0, so the check of 3 can fail.
Every reading is engineering, not quotable.

    <venv>/bin/python -I -B gates/g44_pid_model.py --device cpu --expect-sha12 SHA12 --work DIR --archive SMOKE \\
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

GATE = 'G44'
FUNCTIONAL = 'keras.src.models.functional.Functional'
MHA = {'PID-CH': 'QMultiHeadAttention', 'PID-BEST': 'BinaryMultiHeadAttention'}


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
        po, m = R['pid_object'], R['model']
        g.check(po['model_is_runner_model'] and po['model_seen']['id'] == po['runner_model_id'] == m['id']
                and m['is_flow_model'],
                f'{lab}: the PID callback received the runner\'s model object itself (identity; the id it saw at '
                f'on_train_begin), the object the builder returned')
        g.check(m['class'] == FUNCTIONAL and po['model_seen']['class'] == FUNCTIONAL and m['n_top_layers'] == 14
                and po['model_seen']['n_top_layers'] == 14 and m['top_classes'][3] == MHA[arm],
                f'{lab}: a Keras Functional with 14 top-level layers, {m["top_classes"][3]} at index 3, not a wrapper')
        g.check(po['class'] == 'LabBetaPID' and po['mro'][1:3] == ['BetaPID', 'BaseBetaPID']
                and po['pid_class'] == 'TracedPID' and po['pid_mro'][1] == 'PID',
                f'{lab}: the callback is cbl.pid.LabBetaPID ({po["mro"]}), its PID object a subclass of the stock PID')
        bl, rows = pg.blocks(R), pg.rows(R)
        E = sorted(bl)
        stock = [bl[e]['stock_aggregate'] for e in E]
        g.check(all(isinstance(x, float) and x > 0 for x in stock),
                f'{lab}: the stock aggregate is > 0 at the end of every epoch, epoch 0 included '
                f'(epoch 0: {stock[0] if stock else None}; before training: {po["stock_at_train_begin"]})')
        g.check(all(bl[e]['stock_aggregate'] == rows[e]['train_count'] and isinstance(rows[e]['train_count'], int)
                    for e in E),
                f'{lab}: at every epoch end the stock aggregate equals FreeEBOPs\'s training count of the epoch '
                '(public aggregate = flattened private total)')
        c = R.get('controls') or {}
        g.check(c.get('stock_aggregate_model', 0) > 0 and c.get('stock_aggregate_model') == c.get('private_total_model')
                and c.get('stock_aggregate_wrapper') == 0.0 and 'wrapper_error' not in c,
                f'{lab}: control after fit: stock aggregate {c.get("stock_aggregate_model")} of the model, equal to the '
                f'private total; {c.get("stock_aggregate_wrapper")} on a functional wrapper '
                f'{c.get("wrapper_top_classes")} (the check can fail)')
        G.rec['runs'][lab] = {'stock_aggregate': stock, 'train_count': [rows[e]['train_count'] for e in E],
                              'stock_at_train_begin': po['stock_at_train_begin'], 'controls': c, 'model': m}
        g.info(f'{lab} engineering readings, not quotable', {'stock_aggregate_epoch0': stock[0] if stock else None,
                                                             'stock_at_train_begin': po['stock_at_train_begin']})
    return G.finish()


if __name__ == '__main__':
    sys.exit(main())
