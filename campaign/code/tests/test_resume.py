"""Unit tests of M14's pure parts (cbl/resume.py): state encodings, append-only prefixes, the rewind of a run
directory, ParetoFront and StopIf state, and the refusals of read_latest and check_static. No Keras, JAX or hgq.
The whole-run resume is gate G48's (PC). Engineering, not quotable."""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))

from cbl import resume as R  # noqa: E402
from cbl import rng  # noqa: E402


def test_python_state_round_trips_exactly_and_hashes_as_rng():
    r = random.Random(1234)
    for _ in range(7):
        r.random()
    r.gauss(0, 1)                               # leaves gauss_next set
    st = r.getstate()
    back = R.py_state_from_json(json.loads(json.dumps(R.py_state_to_json(st))))
    assert back == st
    assert R.py_state_sha256(st) == rng.py_state_sha256(st)
    r2 = random.Random()
    r2.setstate(back)
    assert [r2.random() for _ in range(5)] == [r.random() for _ in range(5)]


def test_numpy_state_round_trips_exactly():
    g = np.random.RandomState(7)
    g.standard_normal(3)                         # leaves a cached gaussian
    st = g.get_state(legacy=True)
    back = R.np_state_from_json(json.loads(json.dumps(R.np_state_to_json(st))))
    g2 = np.random.RandomState()
    g2.set_state(back)
    assert np.array_equal(g2.standard_normal(4), g.standard_normal(4))
    assert rng.np_state_sha256(back) == rng.np_state_sha256(st)


def test_prefix_cut_moves_the_tail_and_refuses_another_prefix(tmp_path):
    p = tmp_path / 'epochs.jsonl'
    p.write_bytes(b'{"epoch": 0}\n{"epoch": 1}\n')
    rec = R.prefix_record(p)
    with open(p, 'ab') as f:
        f.write(b'{"epoch": 2}\n')
    info = R.cut_to_prefix(p, rec, tmp_path / 'discard', 'epochs.jsonl')
    assert info['tail_bytes'] == len(b'{"epoch": 2}\n')
    assert p.read_bytes() == b'{"epoch": 0}\n{"epoch": 1}\n'
    assert (tmp_path / 'discard' / 'epochs.jsonl.tail').read_bytes() == b'{"epoch": 2}\n'
    p.write_bytes(b'{"epoch": 9}\n{"epoch": 1}\n')
    with pytest.raises(R.ResumeRefused):
        R.check_prefix(p, rec)
    p.write_bytes(b'{"ep')
    with pytest.raises(R.ResumeRefused):
        R.check_prefix(p, rec)


def test_absent_file_prefix_is_empty(tmp_path):
    rec = R.prefix_record(tmp_path / 'none.jsonl')
    assert rec['bytes'] == 0 and rec['exists'] is False
    assert R.check_prefix(tmp_path / 'none.jsonl', rec)['tail_bytes'] == 0


class FakePareto:
    def __init__(self, path):
        self.path = Path(path)
        self.paths = []
        self.record = []


def test_pareto_state_round_trips_and_refuses_a_foreign_path(tmp_path):
    pf = FakePareto(tmp_path / 'pareto')
    pf.paths = [pf.path / 'epoch=3-a.keras', pf.path / 'epoch=7-b.keras']
    pf.record = [np.array([0.61234567890123, 499999]), np.array([0.7, 123456.0])]
    st = R.pareto_state(pf)
    pf2 = FakePareto(tmp_path / 'pareto')
    R.restore_pareto(pf2, json.loads(json.dumps(st)))
    assert pf2.paths == pf.paths
    assert all(np.array_equal(a, b) and a.dtype == b.dtype for a, b in zip(pf2.record, pf.record))
    assert R.pareto_digest(pf2) == R.pareto_digest(pf)
    pf.paths.append(tmp_path / 'elsewhere' / 'x.keras')
    pf.record.append(np.array([0.9, 1.0]))
    with pytest.raises(R.ResumeError):
        R.pareto_state(pf)


def test_stopif_state():
    class S:
        counter_epochs, counter_steps = 3, 0
    st = R.stopif_state(S())
    t = S()
    t.counter_epochs = 0
    R.restore_stopif(t, st)
    assert (t.counter_epochs, t.counter_steps) == (3, 0)


def _state(run, ck, *, front_files, live, links, pareto_files=(), append=None):
    return {'front_files': sorted(front_files), 'pareto_files': sorted(pareto_files),
            'live': {'front': {k: {} for k in live if k.startswith('front/')},
                     'pareto': {k: {} for k in live if k.startswith('pareto/')}},
            'links': links, 'append_only': append or {}}


def test_rewind_moves_new_files_links_back_deleted_live_ones_and_refuses_lost_ones(tmp_path):
    run = tmp_path / 'run'
    (run / 'front' / 'ckpt').mkdir(parents=True)
    (run / 'front' / 'sidecars').mkdir()
    (run / 'resume').mkdir()
    a = run / 'front' / 'ckpt' / 'epoch=1.keras'
    a.write_bytes(b'A1')
    (run / 'front' / 'sidecars' / 'epoch=1.json').write_text('{}')
    ev = run / 'front' / 'events.jsonl'
    ev.write_bytes(b'e0\ne1\n')
    ep = run / 'epochs.jsonl'
    ep.write_bytes(b'r0\nr1\n')
    append = {'epochs.jsonl': R.prefix_record(ep), 'front/events.jsonl': R.prefix_record(ev)}
    ck = run / 'resume' / R.checkpoint_name(2)
    ck.mkdir()
    R.link_or_copy(a, ck / 'links' / 'front/ckpt/epoch=1.keras')
    links = {'front/ckpt/epoch=1.keras': {'sha256': R.sha256_file(a), 'method': 'link'}}
    st = _state(run, ck, front_files=['front/ckpt/epoch=1.keras', 'front/sidecars/epoch=1.json'],
                live=['front/ckpt/epoch=1.keras'], links=links, append=append)
    # the killed process: two more epochs, a new checkpoint file, epoch 1's file deleted, a newer checkpoint dir
    ep.write_bytes(b'r0\nr1\nr2\nr3\n')
    ev.write_bytes(b'e0\ne1\ne2\ne3\n')
    (run / 'front' / 'ckpt' / 'epoch=3.keras').write_bytes(b'A3')
    (run / 'front' / 'sidecars' / 'epoch=3.json').write_text('{}')
    (run / 'front' / 'summary.json').write_text('{}')
    a.unlink()
    (run / 'resume' / R.checkpoint_name(3)).mkdir()
    (run / 'resume' / (R.TMP_PREFIX + '00004')).mkdir()
    out = R.rewind(run, st, ck, run / 'resume' / 'discarded' / '001')
    assert ep.read_bytes() == b'r0\nr1\n' and ev.read_bytes() == b'e0\ne1\n'
    assert a.read_bytes() == b'A1' and out['linked_back'] == ['front/ckpt/epoch=1.keras']
    assert sorted(R.files_under(run / 'front', run)) == ['front/ckpt/epoch=1.keras', 'front/events.jsonl',
                                                       'front/sidecars/epoch=1.json']
    d = run / 'resume' / 'discarded' / '001'
    assert (d / 'front' / 'ckpt' / 'epoch=3.keras').read_bytes() == b'A3'
    assert (d / 'front' / 'summary.json').exists()
    assert (d / 'resume' / R.checkpoint_name(3)).is_dir() and not (run / 'resume' / R.checkpoint_name(3)).exists()
    assert not (run / 'resume' / (R.TMP_PREFIX + '00004')).exists()
    # a recorded file that is gone and has no copy: refused before anything moves
    (run / 'front' / 'sidecars' / 'epoch=1.json').unlink()
    (run / 'front' / 'sidecars' / 'epoch=9.json').write_text('{}')
    with pytest.raises(R.ResumeRefused):
        R.rewind(run, st, ck, run / 'resume' / 'discarded' / '002')
    assert (run / 'front' / 'sidecars' / 'epoch=9.json').exists()


def test_rewind_refuses_a_live_file_that_changed(tmp_path):
    run = tmp_path / 'run'
    (run / 'front' / 'ckpt').mkdir(parents=True)
    a = run / 'front' / 'ckpt' / 'epoch=1.keras'
    a.write_bytes(b'A1')
    ck = run / 'resume' / R.checkpoint_name(2)
    ck.mkdir(parents=True)
    R.link_or_copy(a, ck / 'links' / 'front/ckpt/epoch=1.keras')
    links = {'front/ckpt/epoch=1.keras': {'sha256': R.sha256_file(a), 'method': 'link'}}
    st = _state(run, ck, front_files=['front/ckpt/epoch=1.keras'], live=['front/ckpt/epoch=1.keras'], links=links)
    a.unlink()
    a.write_bytes(b'XX')
    with pytest.raises(R.ResumeRefused):
        R.rewind(run, st, ck, run / 'resume' / 'discarded' / '001')


def _latest(run, st):
    ck = run / 'resume' / R.checkpoint_name(st['epochs_done'])
    ck.mkdir(parents=True)
    (ck / 'variables.npz').write_bytes(b'v')
    st = dict(st, files={'variables.npz': R.sha256_file(ck / 'variables.npz')}, links={})
    raw = R.json_bytes(st, compact=True)
    (ck / 'state.json').write_bytes(raw)
    (run / 'resume' / R.LATEST).write_bytes(R.json_bytes({'checkpoint': ck.name, 'epochs_done': st['epochs_done'],
                                                          'state_sha256': R.sha256_bytes(raw)}))
    return ck


def test_read_latest_and_its_refusals(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    with pytest.raises(R.ResumeRefused):
        R.read_latest(run)
    st0 = {'kind': R.KIND, 'schema': R.SCHEMA, 'epochs_done': 4, 'code_sha': 'a' * 40, 'config_sha256': 'c' * 64,
           'every': 2}
    ck = _latest(run, st0)
    st, got = R.read_latest(run)
    assert got == ck and st['epochs_done'] == 4
    (ck / 'variables.npz').write_bytes(b'w')
    with pytest.raises(R.ResumeRefused):
        R.read_latest(run)
    (ck / 'variables.npz').write_bytes(b'v')
    (ck / 'state.json').write_bytes((ck / 'state.json').read_bytes().replace(b'"every":2', b'"every":3'))
    with pytest.raises(R.ResumeRefused):
        R.read_latest(run)


def test_check_static_refusals(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    st = {'code_sha': 'a' * 40, 'config_sha256': 'c' * 64, 'every': 2}
    cfg = {'code_sha': 'a' * 40}
    R.check_static(st, cfg=cfg, config_sha256='c' * 64, head='a' * 40, run_dir=run, every=2)
    for kw, needle in ((dict(head='b' * 40), 'code sha'), (dict(config_sha256='d' * 64), 'config sha256'),
                       (dict(every=1), 'interval')):
        args = dict(cfg=cfg, config_sha256='c' * 64, head='a' * 40, run_dir=run, every=2)
        args.update(kw)
        with pytest.raises(R.ResumeRefused, match=needle):
            R.check_static(st, **args)
    with pytest.raises(R.ResumeRefused, match='code sha'):
        R.check_static(st, cfg={'code_sha': 'b' * 40}, config_sha256='c' * 64, head='a' * 40, run_dir=run, every=2)
    st2 = dict(st, run={'run_name': 'CH-8_nf_s1_gate_ramp_a1_src-0123456789ab'})
    R.check_static(st2, cfg=cfg, config_sha256='c' * 64, head='a' * 40, run_dir=run, every=2,
                   run_name='CH-8_nf_s1_gate_ramp_a1_src-0123456789ab')
    with pytest.raises(R.ResumeRefused, match='run name'):
        R.check_static(st2, cfg=cfg, config_sha256='c' * 64, head='a' * 40, run_dir=run, every=2,
                       run_name='CH-8_nf_s1_gate_ramp_a1_src-ffffffffffff')
    (run / 'run.json').write_text('{}')
    with pytest.raises(R.ResumeRefused, match='finished'):
        R.check_static(st, cfg=cfg, config_sha256='c' * 64, head='a' * 40, run_dir=run, every=2)


def test_link_or_copy_and_files_under(tmp_path):
    src = tmp_path / 'a' / 'f.bin'
    src.parent.mkdir()
    src.write_bytes(b'x')
    how = R.link_or_copy(src, tmp_path / 'b' / 'c' / 'f.bin')
    assert how in ('link', 'copy') and (tmp_path / 'b' / 'c' / 'f.bin').read_bytes() == b'x'
    if how == 'link':
        assert os.stat(src).st_ino == os.stat(tmp_path / 'b' / 'c' / 'f.bin').st_ino
    assert R.files_under(tmp_path / 'b', tmp_path) == ['b/c/f.bin']
    assert R.files_under(tmp_path / 'none', tmp_path) == []


def test_launch_identity_is_published_with_directory_before_train_begin(tmp_path):
    out = tmp_path / 'run'
    launch = {'resume_every': 7, 'gpu_product': 'NVIDIA-L40S'}
    R.create_run_directory(out, launch)
    assert json.loads((out / 'launch.json').read_text()) == launch
    assert not (out / 'resume/latest.json').exists()
    with pytest.raises(FileExistsError):
        R.create_run_directory(out, {'resume_every': 1})
    assert json.loads((out / 'launch.json').read_text()) == launch


def test_kill_before_publication_leaves_no_metadata_free_run(tmp_path, monkeypatch):
    def killed(*args):
        raise OSError('simulated pre-publication kill')
    monkeypatch.setattr(R, '_write_bytes_sync', killed)
    out = tmp_path / 'run'
    with pytest.raises(OSError):
        R.create_run_directory(out, {'resume_every': 7})
    assert not out.exists()
    assert list(tmp_path.glob('.run.starting-*'))
