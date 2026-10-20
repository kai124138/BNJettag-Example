"""Chang's released training constants, read from `chang/run_train.py` by its syntax tree.

`run_train.py` imports keras and hgq at module top (:10-15), so it is never imported here. The
file's sha256 must equal its `code-base.txt` line before anything is read from it. Every value is
taken from a literal in the file, located by the call or argument it belongs to:

  --batch-size default (:67), --learning-rate default (:68), the validation Dataset batch (:74),
  the training Dataset's shuffle (:73), get_model's literal arguments (:76), ParetoFront's
  enable_if (:88), the PieceWiseSchedule list (:90), StopIf's threshold (:91), the cosine
  schedule's arguments (:93), the callback list (:95), Adam() with no argument (:97), the loss
  (:99), steps_per_execution (:100), fit's epochs (:104), the data seed (:41-42).

STUDY v0.10 section 7.1 states the same values; gate G57 compares them with that statement.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

FORK = Path(__file__).resolve().parents[1]
RUN_TRAIN_REL = 'chang/run_train.py'


class ReleaseError(RuntimeError):
    """chang/run_train.py does not hash to code-base.txt, or a value is not where it should be."""


def code_base_sha256(fork_rel: str, fork: Path = FORK) -> str:
    for line in (fork / 'code-base.txt').read_text(encoding='utf-8').splitlines():
        parts = line.split('  ')
        if len(parts) == 4 and parts[0] == 'file' and parts[2] == fork_rel:
            return parts[1]
    raise ReleaseError(f'code-base.txt has no file line for {fork_rel}')


def _lit(node):
    return ast.literal_eval(node)


def _call_name(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ''


def _one(items, what):
    if len(items) != 1:
        raise ReleaseError(f'expected exactly one {what} in {RUN_TRAIN_REL}, found {len(items)}')
    return items[0]


def _kw(call: ast.Call, name: str):
    hits = [k.value for k in call.keywords if k.arg == name]
    return _lit(_one(hits, f'keyword {name}= of {_call_name(call)}()'))


def _compare(lam: ast.Lambda) -> list:
    """`lambda x: x['a'] > 0.5 and x['b'] < 5e5` -> [('a', '>', 0.5), ('b', '<', 500000.0)]."""
    body = lam.body
    terms = body.values if isinstance(body, ast.BoolOp) and isinstance(body.op, ast.And) else [body]
    out = []
    for t in terms:
        if not (isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.left, ast.Subscript)):
            raise ReleaseError('unexpected lambda form in run_train.py')
        key = _lit(t.left.slice)
        op = {ast.Gt: '>', ast.Lt: '<'}.get(type(t.ops[0]))
        if op is None:
            raise ReleaseError('unexpected comparison in run_train.py lambda')
        out.append((key, op, float(_lit(t.comparators[0]))))
    return out


def release_constants(fork: Path = FORK) -> dict:
    path = fork / RUN_TRAIN_REL
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    want = code_base_sha256(RUN_TRAIN_REL, fork)
    if sha != want:
        raise ReleaseError(f'{RUN_TRAIN_REL} sha256 {sha} differs from code-base.txt {want}')
    tree = ast.parse(raw.decode('utf-8'))
    mains = [n for n in tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
             and isinstance(n.test.left, ast.Name) and n.test.left.id == '__name__']
    main = _one(mains, "if __name__ == '__main__' block")
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]

    def by_name(name):
        return [c for c in calls if _call_name(c) == name]

    args = {}
    for c in by_name('add_argument'):
        flag = _lit(c.args[0])
        defaults = [k.value for k in c.keywords if k.arg == 'default']
        if defaults:
            args[flag] = _lit(defaults[0])

    datasets = by_name('Dataset')
    if len(datasets) != 2:
        raise ReleaseError('expected the training and the validation Dataset')
    train_ds, val_ds = datasets
    shuffle = _kw(train_ds, 'shuffle')
    if [k.arg for k in val_ds.keywords]:
        raise ReleaseError('the validation Dataset was expected without keywords')
    val_batch = _lit(val_ds.args[2])
    val_device = _lit(val_ds.args[3])
    train_device = _lit(train_ds.args[3])

    gm = _one(by_name('get_model'), 'get_model call')
    gm_literals = [_lit(a) for a in gm.args[1:4]]

    pareto = _one(by_name('ParetoFront'), 'ParetoFront call')
    pareto_metrics = _lit(pareto.args[1])
    pareto_sides = _lit(pareto.args[2])
    enable = [k.value for k in pareto.keywords if k.arg == 'enable_if']
    pareto_filter = _compare(_one(enable, 'enable_if'))

    pws = _one(by_name('PieceWiseSchedule'), 'PieceWiseSchedule call')
    beta_schedule = [list(t) for t in _lit(pws.args[0])]

    stop = _one(by_name('StopIf'), 'StopIf call')
    stop_rule = _compare(stop.args[0])

    cos = _one(by_name('cosine_decay_restarts_schedule'), 'cosine schedule call')
    lr_schedule = {'function': 'cosine_decay_restarts_schedule', 'first_decay_steps': _lit(cos.args[1]),
                   't_mul': _kw(cos, 't_mul'), 'm_mul': _kw(cos, 'm_mul'), 'alpha': _kw(cos, 'alpha'),
                   'alpha_steps': _kw(cos, 'alpha_steps')}

    assigned = {}
    for n in ast.walk(main):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                and isinstance(n.value, ast.Call):
            assigned[n.targets[0].id] = _call_name(n.value)
    cb_assign = [n for n in ast.walk(main) if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'callbacks']
    cb_list = _one(cb_assign, 'callbacks = [...]').value
    callbacks = [assigned[e.id] for e in cb_list.elts]

    adam = _one(by_name('Adam'), 'Adam call')
    if adam.args or adam.keywords:
        raise ReleaseError('Adam() was expected with defaults')
    loss = _one(by_name('SparseCategoricalCrossentropy'), 'loss call')
    compile_ = _one(by_name('compile'), 'compile call')
    fit = _one(by_name('fit'), 'fit call')
    if [k.arg for k in fit.keywords if k.arg == 'shuffle']:
        raise ReleaseError('fit was expected without shuffle= (Keras default True)')

    seeds = [_lit(c.args[0]) for c in calls if _call_name(c) == 'seed']

    return {
        'source': {'path': RUN_TRAIN_REL, 'sha256': sha},
        'batch_size': args['--batch-size'],
        'val_batch_size': val_batch,
        'learning_rate': args['--learning-rate'],
        'dataset_shuffle': shuffle,
        'dataset_devices': [train_device, val_device],
        'get_model_literals': gm_literals,
        'pareto_metrics': pareto_metrics,
        'pareto_sides': pareto_sides,
        'pareto_filter': [list(t) for t in pareto_filter],
        'beta_schedule': beta_schedule,
        'stopif': [list(t) for t in stop_rule],
        'lr_schedule': lr_schedule,
        'callbacks': callbacks,
        'optimizer': 'Adam',
        'optimizer_args': {},
        'loss': {'class': 'SparseCategoricalCrossentropy', 'from_logits': _kw(loss, 'from_logits')},
        'steps_per_execution': _kw(compile_, 'steps_per_execution'),
        'epochs': _kw(fit, 'epochs'),
        'fit_shuffle': 'keras default (True)',
        'data_seeds_in_main': seeds,
    }
