"""Offline resume-gate comparisons exclude telemetry but retain exact scientific checks."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gates'))
import resumegate as G  # noqa: E402


def records():
    rows = [
        {'epoch': e, 'log_accuracy': 0.75, 'batch_index_seq_sha256': 'a' * 64,
         'sys/host_rss_mb': 100, 'sys/alloc_hours': 0.01}
        for e in range(2)
    ]
    u = {'rows': rows, 'epoch_records': deepcopy(rows), 'runner': {}}
    return u, deepcopy(u)


def test_telemetry_differences_do_not_enter_either_comparison():
    u, r = records()
    for key in ('rows', 'epoch_records'):
        for row in r[key]:
            row.update({'sys/host_rss_mb': 200, 'sys/alloc_hours': 0.02,
                        'sys/future_telemetry': 123})
    original = deepcopy(r)
    assert G.pre_resume_agree(u, r, 2)
    assert G.diff_runs(u, r, resumed_from=1) == {}
    assert r == original


@pytest.mark.parametrize('field,value', [('log_accuracy', 0.7500000000000001),
                                        ('batch_index_seq_sha256', 'b' * 64)])
@pytest.mark.parametrize('key', ['rows', 'epoch_records'])
def test_scientific_differences_still_fail_exact_comparison(key, field, value):
    u, r = records()
    r[key][0][field] = value
    if key == 'rows':
        assert not G.pre_resume_agree(u, r, 2)
    assert key in G.diff_runs(u, r, resumed_from=1)


def test_missing_compact_records_still_differ_from_empty_records():
    u, r = records()
    u['epoch_records'] = None
    r['epoch_records'] = []
    assert 'epoch_records' in G.diff_runs(u, r, resumed_from=1)
