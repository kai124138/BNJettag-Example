"""Unit tests for cbl/front.py and the pure helpers of cbl/ablate.py (gates G37, G41, G42; STUDY v0.10 sections 8.1
and 8.8). Standard library, NumPy and SciPy only; no Keras.

    python -I -B -m pytest -q -p no:cacheprovider tests/test_front.py

Every case is constructed; every number is an engineering check, not a quotable result.
"""

import hashlib
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

FORK = Path(__file__).resolve().parents[1]
if str(FORK) not in sys.path:
    sys.path.insert(0, str(FORK))

from cbl import ablate  # noqa: E402
from cbl import front as F  # noqa: E402
from cbl import select as sel  # noqa: E402


class Stand:
    def __init__(self):
        self.epoch = None

    def save_fn(self, path):
        Path(path).write_bytes(json.dumps({'e': self.epoch}).encode())

    def state_fn(self):
        return hashlib.sha256(str(self.epoch).encode()).hexdigest()

    def inloop_fn(self):
        return {'metrics': {'correct': 0, 'n': 1, 'val_auc_per_class': [None] * 5, 'val_auc_macro': None,
                            'logits_sha256': 'x'}, 'structure_record': {'degenerate': {}}, 'attention_diagnostics': None}

    def audit_fn(self, path, record):
        return {'delta_attn_mean': 0.0}


def saver(tmp_path, **kw):
    st = Stand()
    args = dict(run_id='t', config_sha256='c' * 64, code_sha='d' * 40, headline='H1', schedule='ramp',
                save_fn=st.save_fn, state_fn=st.state_fn, inloop_fn=st.inloop_fn, audit_fn=st.audit_fn,
                ablations='run_end', keep_superseded=False)
    args.update(kw)
    return F.FrontSaver(tmp_path / 'front', **args), st


def rec(e, c, h1, h2=None, nr=None):
    return {'epoch': e, 'val_correct': c, 'val_n': 1000, 'H1': h1, 'H2': h1 if h2 is None else h2,
            'NR': h1 if nr is None else nr}


def feed(sv, st, rs):
    out = []
    for r in rs:
        st.epoch = r['epoch']
        out.append(sv.on_frozen_record(r['epoch'], r))
    return out


def test_ties_and_filter(tmp_path):
    sv, st = saver(tmp_path)
    ev = feed(sv, st, [rec(0, 600, 400000), rec(1, 600, 400000), rec(2, 600, 390000), rec(3, 500, 100000),
                       rec(4, 700, 500000)])
    assert ev[0]['joined'] == ['H1', 'H2', 'NR']
    assert ev[1]['joined'] == []                       # an equal pair keeps the earlier epoch
    assert ev[2]['joined'] == ['H1', 'H2', 'NR'] and ev[2]['left'] == {'H1': [0], 'H2': [0], 'NR': [0]}
    assert ev[3]['joined'] == [] and 'H1:min_cost' in ev[3]['diagnostic']      # top-1 0.5 fails the filter
    assert ev[4]['joined'] == []                       # cost 500,000 fails the filter
    s = sv.finalize(end='completed', epochs_completed=5)
    assert s['m16_reader'] == {'ok': True}
    assert s['envelope']['H1']['100000'] == 'NA' and s['filter_outcome']['H1']['100000']['outcome'] == sel.NO_ACCEPTED
    assert s['envelope']['H1']['400000']['epoch'] == 2 and s['envelope']['H1']['400000']['slack'] == 10000


def test_random_matches_brute_force(tmp_path):
    g = random.Random(5)
    rs, acc = [], 600
    for e in range(150):
        acc = max(300, min(900, acc + g.randint(-15, 14)))
        h = max(0, int(600000 * 0.99 ** e) + g.randint(-9000, 9000))
        rs.append(rec(e, acc, h, max(0, h - g.randint(0, 20000)), h + g.randint(0, 20000)))
    sv, st = saver(tmp_path)
    feed(sv, st, rs)
    R = [F.rec_from(r['epoch'], r) for r in rs]
    for c in F.COUNTS:
        assert sorted(m.epoch for m in sv.state.front[c]) == sorted(x.epoch for x in F.batch_front(R, c))
    s = sv.finalize(end='completed', epochs_completed=len(rs))
    assert s['m16_reader'] == {'ok': True}
    assert all(e != 0 for c in F.COUNTS for e in s['envelope'][c].values())
    assert F.replay(F.recs_from_events(F.read_events(sv.dir))).state() == sv.state.state()
    files = sorted(int(p.name[6:-6]) for p in (sv.dir / 'ckpt').glob('*.keras'))
    assert files == sorted(sv.state.live_epochs())


def test_refusals(tmp_path):
    sv, st = saver(tmp_path)
    st.epoch = 0
    with pytest.raises(F.FrontError):
        sv.on_frozen_record(0, {'epoch': 0, 'val_correct': 1, 'val_n': 2, 'H1': 1, 'H2': 1})
    with pytest.raises(F.FrontError):
        sv.on_frozen_record(0, dict(rec(0, 600, 1), H1=1.0))
    with pytest.raises(F.FrontError):
        sv.on_frozen_record(3, rec(0, 600, 1))
    with pytest.raises(F.FrontError):
        saver(tmp_path / 'b', ablations=None)
    with pytest.raises(F.FrontError):
        saver(tmp_path / 'c', keep_superseded=None)
    with pytest.raises(F.FrontError):
        saver(tmp_path / 'd', schedule='pid', headline=None)


def test_auc_with_ties():
    g = random.Random(1)
    y = [g.randrange(5) for _ in range(200)]
    s = [g.randrange(4) / 4 for _ in range(200)]
    pos = [s[i] for i in range(200) if y[i] == 2]
    neg = [s[i] for i in range(200) if y[i] != 2]
    want = Fraction(sum(2 * (a > b) + (a == b) for a in pos for b in neg), 2 * len(pos) * len(neg))
    assert F.auc_one_vs_rest(s, y, 2) == float(want)


def test_entropy_and_means():
    h, c = ablate.attention_entropy(np.array([[0.125] * 8, [0.0] * 8, [-0.1, 1.1] + [0.0] * 6, [0.5, 0.5] + [0.0] * 6]))
    assert h.tolist() == [math.log(8.0), math.log(2.0)]
    assert c == {'rows': 4, 'used': 2, 'all_zero_rows': 1, 'negative_rows': 1, 'nan_rows': 0}
    assert ablate.shifted_mean(np.full(7, math.log(8.0))) == math.log(8.0)
    assert ablate.pairwise_sum(np.full((3, 5), 0.1)).tolist() == [ablate.pairwise_sum(np.full(5, 0.1)).item()] * 3


def test_overflow_counts():
    k = np.array([1, 1, 0, 1])
    i = np.array([0, 0, 1, 0])
    f = np.array([3, 3, 2, -1])                          # the last lane is masked (k + i + f = 0)
    x = np.array([0.875, 0.95, -0.25, 5.0])
    r = np.floor(x * 2.0 ** f + 0.5) / 2.0 ** f
    i2 = np.maximum(i, -f)
    w = (r + k * 2.0 ** (i2 + k - 1)) % 2.0 ** (i2 + k) - k * 2.0 ** (i2 + k - 1)
    c = ablate.overflow_from_arrays(x, r, w, k, i, f)
    assert c['overflow'] == c['range_check'] == 2 and c['masked'] == 1 and c['live'] == 3
