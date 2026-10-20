"""Offline tests of M8 summaries, M9 dwell and M10 route projection. No backend imports."""
import json
from types import SimpleNamespace
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbl.epoch_diagnostics import width_summary, bn_summary, Moments, moment_difference, observed_calls
from cbl import front, structure, wandb_log as w


def test_quantizer_widths_steps_and_bn_moments_have_physical_values():
    assert width_summary([0, 2, 4], [-1, 0, 3]) == {
        'width_mean': 2, 'width_max': 4, 'step_min': .125, 'step_max': 2}
    assert bn_summary([-2, 3], [-4, 2], [0, 9]) == {
        'gamma_min': -2, 'gamma_max': 3, 'moving_mean_absmax': 4, 'moving_var_min': 0, 'moving_var_max': 9}


def test_population_moments_match_full_validation_with_uneven_chunks():
    raw = np.array([[1, 8], [2, 6], [4, 4], [7, 2], [11, 0]], dtype=float)
    binary = raw * np.array([2, -1]) + 3
    a, b = Moments(), Moments()
    for sl in (slice(0, 2), slice(2, 5)):
        for acc, x in ((a, raw[sl]), (b, binary[sl])):
            acc.add(len(x), x.mean(0), x.var(0))
    np.testing.assert_allclose(a.mean, raw.mean(0))
    np.testing.assert_allclose(a.variance, raw.var(0))
    delta = moment_difference(a, b)
    assert delta['moment_mismatch_mean'] == pytest.approx(np.abs(raw.mean(0) - binary.mean(0)).mean())
    assert delta['moment_mismatch_var'] == pytest.approx(np.abs(raw.var(0) - binary.var(0)).mean())
    assert moment_difference(a, a) == {'moment_mismatch_mean': 0, 'moment_mismatch_var': 0}


def test_observers_restore_exact_calls_on_failure_and_preserve_output():
    class Layer:
        def call(self, x, training=False):
            return x * 2
    ly = Layer()
    seen = []
    with pytest.raises(RuntimeError):
        with observed_calls([(ly, lambda args, kwargs, y: seen.append(y))]):
            assert ly.call(3) == 6
            raise RuntimeError('constructed tracing failure')
    assert seen == [6] and 'call' not in ly.__dict__ and ly.call(4) == 8


def test_structure_projection_keeps_zero_routes_and_absent_quantizers_distinct():
    state = {'layers': {
        'Q': {'kernel': {'pruned': True, 'rounded_to_zero': False},
              'iq': {'zero_width_all': True}, 'oq': {'zero_width_all': False}},
        'O': {'kernel': {'pruned': False, 'rounded_to_zero': True}, 'iq': {}, 'oq': {'present': False}}},
        'softmax_zero_width_fraction': [0.0, 0.75]}
    row = structure.epoch_structure(state)
    assert row['Q'] == {'kernel_zero_pruned': True, 'kernel_zero_rounded': False,
                        'input_all_zero_width': True, 'output_all_zero_width': False}
    assert row['O']['kernel_zero_rounded'] and not row['O']['output_all_zero_width']
    assert row['softmax_heads']['1']['zero_width_frac'] == .75


def test_pruning_fractions_cover_hgq_and_binary_without_invented_native_share(monkeypatch):
    entries = [SimpleNamespace(name=n, kind=k, layer=n) for n, k in [('Q', 'hgq'), ('head1', 'binary'), ('add', 'native')]]
    monkeypatch.setattr(structure, 'kernel_state', lambda l: {'zero_width_coefficients': 3, 'coefficients': 12})
    monkeypatch.setattr(structure, 'quantizer_io_state', lambda l: {'iq': {'input_zero_width_fraction': .5}})
    assert structure.pruning_channels(entries) == {'Q': {'zero_width_frac': .25},
        'head1': {'zero_width_frac': .25, 'input_zero_width_frac': .5}}
    assert set(w.NOT_PRODUCED) == {'pruning_channel/*/native_share_removed'}


def test_dwell_includes_current_once_and_survives_m14_front_restore():
    rows = [{'epoch': i, 'val_correct': 6, 'val_n': 10, 'H1': h, 'H2': h, 'NR': h}
            for i, h in enumerate([100000, 200000, 200001, 450001])]
    state = front.FrontState()
    state.offer(front.rec_from(0, rows[0]))
    state.offer(front.rec_from(1, rows[1]))
    resumed = front.FrontState.from_state(json.loads(json.dumps(state.state())))
    assert resumed.dwell(rows[2]) == state.dwell(rows[2])
    resumed.offer(front.rec_from(2, rows[2]))
    assert resumed.dwell(rows[2]) == state.dwell(rows[2])
    counts = resumed.dwell(rows[3])['H1']
    assert sum(counts.values()) == 4
    assert counts['<=100000'] == counts['(100000,200000]'] == counts['(200000,300000]'] == counts['(450000,inf)'] == 1
