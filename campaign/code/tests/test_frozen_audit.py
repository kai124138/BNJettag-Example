"""Offline regression coverage for G49's enriched-epoch versus core-only reload audit."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbl import frozen_eval, runner


@pytest.fixture
def core():
    return {
        'val_correct': 3, 'val_n': 4, 'val_top1': .75,
        'H1': 10, 'H2': 20, 'NR': 30, 'snap_changed': False,
        'subtotals': {'mha': {'H1': 10, 'H2': 20, 'NR': 30}},
        'vars_sha256_after': 'a' * 64,
        'binary_sites': {'mha.query': {
            'per_column': True, 'form': 'pre', 'penalty': 'H1', 'gr': False,
            'qg_b': [2, 2], 'qg_i': [1, 1], 'qg_value': [.5, 1.0],
            'zero_cols': {'pruned': 0, 'rounded_to_zero': 0},
            'saturated_cols': 0, 'sign_occupancy': .5,
            'sign_flip_rate': None, 'single_sign_cols': 0,
        }},
    }


@pytest.fixture(params=[False, True], ids=['binary', 'fused_bn'])
def enriched(core, request):
    row = copy.deepcopy(core)
    site = row['binary_sites']['mha.query']
    site.update(moment_mismatch_mean=.1, moment_mismatch_var=.2, sign_flip_rate=.25)
    if request.param:
        site.update(dead_channel_frac=.3, fb_sign_diff_frac=.4)
    return row


def test_enriched_epoch_matches_core_reload(enriched, core):
    assert frozen_eval.BINARY_AUDIT_EXCLUDE == runner.BINARY_ENRICHMENT_FIELDS | {'sign_flip_rate'}
    assert frozen_eval.audit_fields(enriched) == frozen_eval.audit_fields(core)
    assert frozen_eval.audit_mismatches(enriched, core) == []


def test_enriched_epoch_still_detects_changed_core(enriched, core):
    core['binary_sites']['mha.query']['qg_value'][0] = .75
    assert frozen_eval.audit_mismatches(enriched, core) == ['binary_sites']


@pytest.mark.parametrize('side', ['epoch', 'reload'])
def test_unknown_binary_field_is_not_excluded(enriched, core, side):
    row = enriched if side == 'epoch' else core
    row['binary_sites']['mha.query']['unknown_diagnostic'] = .1
    assert frozen_eval.audit_mismatches(enriched, core) == ['binary_sites']
