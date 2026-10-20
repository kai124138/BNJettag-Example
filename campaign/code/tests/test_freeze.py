"""Unit tests for the section 8.5 freeze manifest in cbl/select.py (gate G56, Mac part; STUDY v0.8
section 8.5).

Run with the Mac CPU interpreter, from anywhere:

    python -I -B -m pytest -q -p no:cacheprovider tests/test_freeze.py

Constructed cases only (tests/fixtures/m16_synth.py). The sealed reader of record is M3's (B2);
here `SealedReaderStub` stands in for it and opens nothing.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[1]
for p in (str(FORK), str(FORK / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cbl import select as sel  # noqa: E402
from fixtures import m16_synth as syn  # noqa: E402

G = sel.GRID


def complete_manifest(claim=None, s4=None):
    return syn.freeze_example(sel, claim=claim, s4=s4)


def test_complete_manifest_writes_and_opens(tmp_path):
    m = complete_manifest()
    path = tmp_path / "freeze.json"
    sha = sel.write_freeze_manifest(path, m)
    got = sel.require_freeze_manifest(path)
    assert got["manifest_sha256"] == sha and sel.freeze_manifest_problems(got) == []
    stub = sel.SealedReaderStub(path)
    h = got["grid_checkpoints"]["H1"]["F011-FB"]["1"]["350000"]
    assert stub.read(h)["manifest_sha256"] == sha
    with pytest.raises(sel.FreezeRefused):
        stub.read("f" * 64)


def test_reader_refuses_without_manifest(tmp_path):
    with pytest.raises(sel.FreezeRefused):
        sel.SealedReaderStub(None)
    with pytest.raises(sel.FreezeRefused):
        sel.SealedReaderStub(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(sel.FreezeRefused):
        sel.SealedReaderStub(bad)
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(sel.FreezeRefused):
        sel.SealedReaderStub(empty)


@pytest.mark.parametrize("key", [k for k, _ in sel.FREEZE_REQUIRED])
def test_each_required_item_is_required(tmp_path, key):
    m = complete_manifest()
    del m[key]
    assert any(key in p for p in sel.freeze_manifest_problems(dict(m, manifest_sha256="0" * 64)))
    with pytest.raises(sel.SelectError):
        sel.write_freeze_manifest(tmp_path / "f.json", m)
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(sel.FreezeRefused):
        sel.require_freeze_manifest(path)


def test_tampered_manifest_refused(tmp_path):
    path = tmp_path / "freeze.json"
    sel.write_freeze_manifest(path, complete_manifest())
    m = json.loads(path.read_text())
    m["best"]["best"] = "F111"
    path.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(sel.FreezeRefused, match="manifest_sha256"):
        sel.require_freeze_manifest(path)


def test_heldout_key_refused(tmp_path):
    m = complete_manifest()
    m["control_selections"]["FP32"]["1"]["test_top1"] = 0.66
    with pytest.raises(sel.SelectError, match="held-out"):
        sel.write_freeze_manifest(tmp_path / "f.json", m)


def test_grid_checkpoint_gaps_refused(tmp_path):
    m = complete_manifest()
    del m["grid_checkpoints"]["NR"]["F110-BF"]["2"]
    assert any("F110-BF seed 2" in p for p in sel.freeze_manifest_problems(dict(m, manifest_sha256="0" * 64)))
    m = complete_manifest()
    m["grid_checkpoints"]["H2"]["F100"]["3"]["100000"] = 0
    assert any("must be a sha256 or NA" in p for p in sel.freeze_manifest_problems(dict(m, manifest_sha256="0" * 64)))


def test_structure_record_for_every_frozen_point(tmp_path):
    m = complete_manifest()
    h = m["grid_checkpoints"]["H1"]["CH-8"]["1"]["450000"]
    del m["structure_records"][h]
    with pytest.raises(sel.SelectError, match="structure_records"):
        sel.write_freeze_manifest(tmp_path / "f.json", m)


def test_claim_and_s4_decisions(tmp_path):
    m = complete_manifest()
    m["claim_domain"] = None
    with pytest.raises(sel.SelectError, match="claim_domain"):
        sel.write_freeze_manifest(tmp_path / "a.json", m)
    m = complete_manifest()
    m.pop("g_star")
    with pytest.raises(sel.SelectError, match="g_star"):
        sel.write_freeze_manifest(tmp_path / "b.json", m)
    a = sel.claim_domain_config({"base": "a", "c3r": False})
    m = complete_manifest(claim=a)
    assert "g_star" not in m
    sel.write_freeze_manifest(tmp_path / "c.json", m)
    m = complete_manifest(s4={"funded": True})
    with pytest.raises(sel.SelectError, match="s4.option"):
        sel.write_freeze_manifest(tmp_path / "d.json", m)
    m = complete_manifest(s4={"funded": True, "option": "C", "d_conf": {"status": "ok", "allowances": [450000]},
                              "d_band": {"status": "ok", "allowances": [450000]}, "selections": {"BEST-C": {}}})
    sel.write_freeze_manifest(tmp_path / "e.json", m)
    m = complete_manifest(s4={"funded": "yes"})
    with pytest.raises(sel.SelectError, match="s4.funded"):
        sel.write_freeze_manifest(tmp_path / "f.json", m)


def test_freeze_written_once(tmp_path):
    path = tmp_path / "freeze.json"
    sel.write_freeze_manifest(path, complete_manifest())
    with pytest.raises(sel.SelectError, match="written once"):
        sel.write_freeze_manifest(path, complete_manifest())


def test_incomplete_screen_has_no_manifest(tmp_path):
    m = complete_manifest()
    m["d_screen"] = {"status": "INCOMPLETE", "allowances": [], "reason": "empty"}
    with pytest.raises(sel.SelectError, match="d_screen"):
        sel.write_freeze_manifest(tmp_path / "f.json", m)


def test_manifest_holds_no_heldout_value():
    m = complete_manifest()
    assert sel._no_heldout_keys(m) == []
    assert sel._no_heldout_keys({"a": {"HeldOut_top1": 1}}) == ["/a", "/a/HeldOut_top1"][1:]
    assert sel._no_heldout_keys({"x": [{"test": 1}]}) == ["/x[0]/test"]


def test_pid_summary_disagreeing_with_the_rule_is_refused():
    # a PID run whose summary names another checkpoint than the section 8.6 rule cannot be frozen
    import fixtures.m16_synth as fx
    orig = fx.run_records

    def bad_records(*a, **k):
        cfg, sm = orig(*a, **k)
        if k.get("schedule") == "pid" and sm.get("pid_selection"):
            sm["pid_selection"] = dict(sm["pid_selection"], ckpt_sha256="e" * 64)
        return cfg, sm
    fx.run_records = bad_records
    try:
        with pytest.raises(sel.SelectError, match="pid_selection"):
            complete_manifest()
    finally:
        fx.run_records = orig
