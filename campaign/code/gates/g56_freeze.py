"""Gate G56, Mac part (PREFLIGHT-PLAN section 3 row G56, section 4 row B3b; STUDY v0.9 section 8.5).

The section 8.5 freeze manifest is complete, and a reader refuses without it. Checks, on the Mac
CPU with no backend:
  - repository: HEAD equals --expect-sha12 and `git status --porcelain` is empty;
  - the STUDY v0.9 snapshot hashes to its recorded sha256 (61e68527...89df4);
  - the "Freeze before reading" list of STUDY section 8.5 (v0.9 :940-948), parsed from the STUDY text,
    has eight items, and each maps to manifest fields that `cbl.select` requires (`FREEZE_REQUIRED`,
    and the conditional fields: g_star under option (b); s4.d_conf and s4.selections when S4 is
    funded; s4.d_band when funded under (b));
  - a manifest built from constructed screen data (tests/fixtures/m16_synth.py) is written, hashed
    and accepted, and `SealedReaderStub` opens with it and serves only its frozen checkpoints;
  - removing any required field, a gap in the grid checkpoints, a frozen point without its structure
    record, an unset claim domain or S4 decision, a held-out key, a tampered hash, a missing or
    unreadable file, or an empty object each make the writer and the reader refuse;
  - the unit tests `tests/test_freeze.py`, as a sub-step.

PCC part (`--pcc`, batch B5, PC CPU, venv of record): M3's sealed reader of record
`cbl.data.read_heldout` refuses without a complete manifest. Since B5, `cbl.data.check_freeze_manifest`
applies M15's rule and then calls `cbl.select.require_freeze_manifest`. Checks, besides the Mac part's
manifest checks above (the pytest sub-step is left out: the venv of record has no pytest, and the
tests run through the ephemeral pytest overlay as a separate record):
  - static: `read_heldout` calls `check_freeze_manifest` as its first statement, and
    `check_freeze_manifest` calls `freeze_manifest_sha256` (M15) and `require_freeze_manifest` (M16);
  - in a child process of the venv of record under the kernel open-file audit (gates/fileaudit.py),
    the reader is called on the held-out archive of record with no manifest, a missing one, invalid
    JSON, an empty object, a JSON list, an object that passes M15's rule but is not a section-8.5
    manifest, a complete manifest with one required field removed, one whose content no longer
    matches its hash, and one with the claim domain unset: every call is refused with HeldOutSealed,
    and the window holds no open of the held-out archive, its name in its directory, the raw held-out
    directory or any raw held-out file. The reader is never called on the held-out archive with a
    complete manifest;
  - on a 16-jet placeholder of zeros written by the gate (never the held-out archive): with the
    complete manifest and a wrong expected sha256 the reader refuses; with the right one it returns
    16 jets, and its record names both the file sha256 (M15) and the content sha256 (M16) of the
    manifest that allowed the read; the placeholder's open is seen (positive control of the audit);
  - the reader of record and the Mac part's `SealedReaderStub` accept and refuse the same manifests.

Run as

    python -I -B gates/g56_freeze.py --expect-sha12 SHA12 [--study PATH]
    <venv>/bin/python -I -B gates/g56_freeze.py --pcc --expect-sha12 SHA12 --work DIR [--data-manifest PATH] [--study PATH]

Prints one line per check and ends with "G56 PASS" (exit 0) or "G56 FAIL" (exit 1). The Mac part
writes only into a temporary directory that it removes; the PCC part writes into --work, which it
deletes at the end. Every number printed is an engineering reading, not a quotable result.
"""

import argparse
import ast
import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I does not add the script directory
import m16gate as mg  # noqa: E402

sys.path.insert(0, str(mg.FORK))
sys.path.insert(0, str(mg.FORK / "tests"))
from cbl import select as sel  # noqa: E402
from fixtures import m16_synth as syn  # noqa: E402

# STUDY section 8.5 "Freeze before reading" items (v0.9 :941-948), by their opening words, and the
# manifest fields that carry each
ITEMS = (
    ("D_screen and, if S4 ran, D_conf", ["d_screen", "s4.d_conf"]),
    ("the form choice and BEST", ["form_choice", "best"]),
    ("Kai's claim-domain choice (§19.2 item 12) and, under option (b), g* and D_band", ["claim_domain", "g_star", "s4.d_band"]),
    ("every candidate's grid checkpoint hashes in both forms, under H, under the other count and under NR", ["grid_checkpoints"]),
    ("which controls were admitted after S2 (§10), the control selections and the PID selections",
     ["controls_admitted", "control_selections", "pid_selections"]),
    ("each seed's §8.10 CH-8 context checkpoint", ["context_8_10"]),
    ("the structure record of every frozen checkpoint", ["structure_records"]),
    ("Kai's S4 decision and option and, if funded, D_conf and the S4 selections", ["s4", "s4.d_conf", "s4.selections"]),
)
CONDITIONAL = {"g_star", "s4.d_conf", "s4.d_band", "s4.selections"}


def freeze_items(text):
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("**Freeze before reading.**"))
    items = []
    for l in lines[i + 1:]:
        if l.startswith("- "):
            items.append(l[2:])
        elif items and not l.strip():
            break
    return i + 1, items


def refuses_reader(path):
    try:
        sel.SealedReaderStub(path)
    except sel.FreezeRefused:
        return True
    return False


def refuses_writer(path, m):
    try:
        sel.write_freeze_manifest(path, m)
    except sel.SelectError:
        return True
    return False


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--expect-sha12", default=None)
    ap.add_argument("--study", default=str(mg.STUDY_DEFAULT))
    ap.add_argument("--pcc", action="store_true", help="also run the PCC part (sealed reader of record)")
    ap.add_argument("--work", default=None, help="PCC part: work directory (must not exist)")
    ap.add_argument("--data-manifest", default=str(mg.FORK / "manifests" / "data-prep-150c.json"))
    return ap.parse_args()


def main():
    args = parse_args()
    gate = mg.start("G56", args)
    if args.pcc:
        sys.path.insert(0, str(mg.FORK / "gates"))
        import gatelib
        gatelib.check_venv_of_record(gate)
    text = mg.study_text(gate, args.study)
    ck = gate.check

    line, items = freeze_items(text)
    gate.info("section 8.5 freeze list", "STUDY line %d, %d items" % (line, len(items)))
    ck(len(items) == len(ITEMS), "the section 8.5 freeze list has %d items (expected %d)" % (len(items), len(ITEMS)))
    required = {k for k, _ in sel.FREEZE_REQUIRED}
    for (prefix, fields), item in zip(ITEMS, items):
        top = [f for f in fields if f not in CONDITIONAL]
        ck(item.startswith(prefix) and all(f in required for f in top),
           "freeze item '%s...' is carried by %s" % (prefix[:60], ", ".join(fields)))

    with tempfile.TemporaryDirectory(prefix="g56-") as tmp:
        tmp = Path(tmp)
        m = syn.freeze_example(sel)
        path = tmp / "freeze.json"
        sha = sel.write_freeze_manifest(path, m)
        got = sel.require_freeze_manifest(path)
        ck(got["manifest_sha256"] == sha and sel.freeze_manifest_problems(got) == [],
           "a complete manifest from constructed data is written with its content hash %s... and accepted" % sha[:16])
        stub = sel.SealedReaderStub(path)
        frozen = got["grid_checkpoints"]["H1"]["F011-FB"]["1"]["350000"]
        ok_read = stub.read(frozen)["manifest_sha256"] == sha
        try:
            stub.read("f" * 64)
            ok_ref = False
        except sel.FreezeRefused:
            ok_ref = True
        ck(ok_read and ok_ref, "the reader stub opens with the manifest and serves only its frozen checkpoints")
        ck(refuses_reader(None) and refuses_reader(tmp / "absent.json"), "the reader refuses with no manifest path and with a missing file")
        (tmp / "bad.json").write_text("not json", encoding="utf-8")
        (tmp / "empty.json").write_text("{}", encoding="utf-8")
        ck(refuses_reader(tmp / "bad.json") and refuses_reader(tmp / "empty.json"), "the reader refuses an unreadable file and an empty object")
        allok = True
        for k, _ in sel.FREEZE_REQUIRED:
            mm = copy.deepcopy(m)
            del mm[k]
            raw = tmp / ("raw-%s.json" % k)
            raw.write_text(json.dumps(mm), encoding="utf-8")
            allok = allok and refuses_writer(tmp / ("w-%s.json" % k), mm) and refuses_reader(raw)
        ck(allok, "removing any one of the %d required fields makes the writer and the reader refuse" % len(sel.FREEZE_REQUIRED))
        mm = copy.deepcopy(m)
        del mm["grid_checkpoints"]["NR"]["F110-BF"]["2"]
        mm2 = copy.deepcopy(m)
        mm2["structure_records"].pop(m["grid_checkpoints"]["H1"]["CH-8"]["1"]["450000"])
        mm3 = copy.deepcopy(m)
        mm3.pop("g_star")
        mm4 = copy.deepcopy(m)
        mm4["claim_domain"] = None
        mm5 = copy.deepcopy(m)
        mm5["s4"] = {"funded": True}
        mm6 = copy.deepcopy(m)
        mm6["s4"] = {"funded": True, "option": "A", "d_conf": {"status": "ok", "allowances": [450000]}, "selections": {}}
        mm7 = copy.deepcopy(m)
        mm7["control_selections"]["FP32"]["1"]["test_top1"] = 0.66
        ck(all(refuses_writer(tmp / ("c%d.json" % i), x) for i, x in enumerate((mm, mm2, mm3, mm4, mm5, mm6, mm7))),
           "the writer refuses: a grid-checkpoint gap, a frozen point without a structure record, no g* under (b), an unset "
           "claim domain, a funded S4 without option, without D_band under (b), and a held-out key")
        a = sel.claim_domain_config({"base": "a", "c3r": False})
        ma = syn.freeze_example(sel, claim=a)
        ck("g_star" not in ma and not refuses_writer(tmp / "a.json", ma), "under option (a) no g* is required")
        mf = syn.freeze_example(sel, s4={"funded": True, "option": "C", "d_conf": {"status": "ok", "allowances": [450000]},
                                         "d_band": {"status": "ok", "allowances": [450000]}, "selections": {"BEST-C": {}}})
        ck(not refuses_writer(tmp / "f.json", mf), "a funded S4 with option, D_conf, D_band and selections is accepted")
        t = json.loads(path.read_text())
        t["best"]["best"] = "F111"
        (tmp / "tampered.json").write_text(json.dumps(t), encoding="utf-8")
        ck(refuses_reader(tmp / "tampered.json"), "the reader refuses a manifest whose content no longer matches its hash")
        ck(refuses_writer(path, m), "a freeze is written once: an existing manifest is not overwritten")
    if args.pcc:
        pcc_part(gate, args, m)
    else:
        gate.info("G56 PCC part", "run with --pcc on the PC: M3's sealed reader of record, which calls "
                  "cbl.select.require_freeze_manifest since batch B5")
        mg.run_pytest(gate, ["tests/test_freeze.py"])
    return mg.finish(gate)


# ---------------------------------------------------------------------------------------------
# PCC part

SEALED_CHILD = r'''
import json, sys
import numpy as np
sys.dont_write_bytecode = True
spec = json.loads(sys.argv[1])
sys.path.insert(0, spec["fork"])
from cbl import data
from cbl import select as sel
np.random.seed(42)
tv = data.read_train_val(spec["smoke_train"], 8, True, expect_sha256=spec["smoke_sha256"])
kw = dict(shift=tv.shift, scale=tv.scale, n_constituents=8, ptetaphi=True)
res = {}
def attempt(label, archive, manifest, expect):
    try:
        X, y, rec = data.read_heldout(archive, freeze_manifest=manifest, expect_archive_sha256=expect, **kw)
        res[label] = {"outcome": "returned", "n": int(len(y)), "record": rec}
    except data.HeldOutSealed as e:
        res[label] = {"outcome": "refused", "type": "HeldOutSealed", "message": str(e)[:300]}
    except Exception as e:
        res[label] = {"outcome": "other", "type": type(e).__name__, "message": str(e)[:300]}
real, rsha = spec["heldout"], spec["heldout_sha256"]
for label, m in spec["refuse_on_real"].items():
    attempt("real/" + label, real, m, rsha)
attempt("placeholder/complete, wrong expected sha", spec["placeholder"], spec["complete"], spec["smoke_sha256"])
attempt("placeholder/complete, right sha", spec["placeholder"], spec["complete"], spec["placeholder_sha256"])
stub = {}
for label, m in list(spec["refuse_on_real"].items()) + [("complete", spec["complete"])]:
    try:
        sel.SealedReaderStub(m)
        stub[label] = "accepted"
    except sel.FreezeRefused as e:
        stub[label] = "refused"
print("RESULT " + json.dumps({"cases": res, "stub": stub}))
'''


def _calls(fn_node):
    return {(n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None))
            for n in ast.walk(fn_node) if isinstance(n, ast.Call)}


def static_join(gate):
    tree = ast.parse((mg.FORK / "cbl" / "data.py").read_text(encoding="utf-8"))
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    rh, cf = fns.get("read_heldout"), fns.get("check_freeze_manifest")
    body = [b for b in (rh.body if rh else []) if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))]
    first = body[0] if body else None
    first_calls = _calls(first) if first is not None else set()
    gate.check(rh is not None and first_calls == {"check_freeze_manifest"},
               "static: read_heldout's first statement calls check_freeze_manifest and nothing else")
    calls = _calls(cf) if cf else set()
    gate.check({"freeze_manifest_sha256", "require_freeze_manifest"} <= calls,
               "static: check_freeze_manifest calls M15's freeze_manifest_sha256 and M16's require_freeze_manifest "
               "(found %s)" % sorted(c for c in calls if c))


def pcc_part(gate, args, complete_example):
    import numpy as np
    import fileaudit
    from cbl import data
    ck = gate.check
    print("label       engineering, not quotable; held-out archive watched, never read")
    static_join(gate)
    dman = json.loads(Path(args.data_manifest).read_text(encoding="utf-8"))
    P = dman["paths"]
    arch_dir, val_dir = Path(P["out_dir"]), Path(P["val_dir"])
    heldout = arch_dir / data.HELDOUT_ARCHIVE
    smoke = Path(P["smoke_dir"]) / data.TRAIN_ARCHIVE
    ck(heldout.is_file() and val_dir.is_dir(), "held-out archive and raw held-out directory exist (watched; never opened)")
    ck(data.file_sha256(smoke) == dman["smoke"]["archive"]["sha256"],
       "smoke archive hashes to the manifest (its statistics stand in for the training archive's)")
    if not args.work:
        ck(False, "--work is given")
        return
    work = Path(args.work)
    fileaudit.require_unsynced(work)
    ck(not work.exists(), "--work %s does not exist yet" % work)
    work.mkdir(parents=True)
    try:
        import h5py
        ph = work / "placeholder"
        ph.mkdir()
        with h5py.File(ph / data.HELDOUT_ARCHIVE, "w") as f:
            f.create_dataset("feature", data=np.zeros((16, 150, 16), dtype=np.float16))
            f.create_dataset("label", data=np.arange(16) % 5)
        ph_sha = data.file_sha256(ph / data.HELDOUT_ARCHIVE)
        ck(ph_sha != dman["heldout"]["archive"]["sha256"], "the placeholder is not the held-out archive")
        complete = work / "freeze-complete.json"
        content_sha = sel.write_freeze_manifest(complete, complete_example)
        files = {"missing": str(work / "absent.json")}
        for name, text in (("invalid_json", "{not json"), ("empty_object", "{}"), ("json_list", "[1, 2]"),
                           ("m15_only", json.dumps({"G56": "passes M15's rule only; not a section-8.5 manifest"}))):
            (work / (name + ".json")).write_text(text, encoding="utf-8")
            files[name] = str(work / (name + ".json"))
        body = json.loads(complete.read_text())
        for name, edit in (("missing_best", lambda d: d.pop("best")),
                           ("tampered_content", lambda d: d["best"].__setitem__("best", "F111" if d["best"]["best"] != "F111" else "F110")),
                           ("claim_domain_unset", lambda d: d.__setitem__("claim_domain", None))):
            d = copy.deepcopy(body)
            edit(d)
            (work / (name + ".json")).write_text(json.dumps(d), encoding="utf-8")
            files[name] = str(work / (name + ".json"))
        refuse = {"no manifest": None}
        refuse.update(files)
        spec = {"fork": str(mg.FORK), "smoke_train": str(smoke), "smoke_sha256": dman["smoke"]["archive"]["sha256"],
                "heldout": str(heldout), "heldout_sha256": dman["heldout"]["archive"]["sha256"],
                "placeholder": str(ph / data.HELDOUT_ARCHIVE), "placeholder_sha256": ph_sha,
                "complete": str(complete), "refuse_on_real": refuse}
        forbidden = [(heldout, "heldout"), (arch_dir, "data-dir"), (val_dir, "heldout-raw-dir")]
        forbidden += [(val_dir / n, "heldout-raw") for n in data.glob_order(val_dir)]
        watches = forbidden + [(ph / data.HELDOUT_ARCHIVE, "placeholder")]
        env = {k: v for k, v in os.environ.items() if k not in ("KERAS_BACKEND", "JAX_PLATFORMS")}
        env["JAX_PLATFORMS"] = "cpu"
        r = fileaudit.run_audited([sys.executable, "-I", "-B", "-c", SEALED_CHILD, json.dumps(spec)],
                                  watches=watches, env=env, cwd=str(work))
        gate.info("sealed-reader process", {"returncode": r["returncode"], "stderr_tail": r["stderr"][-600:],
                                            "n_events": len(r["events"])})
        ok = ck(r["returncode"] == 0, "sealed-reader child ran (exit %s)" % r["returncode"])
        res = {"cases": {}, "stub": {}}
        if ok:
            res = json.loads([ln for ln in r["stdout"].splitlines() if ln.startswith("RESULT ")][-1][7:])
        cases = res["cases"]
        for label, c in cases.items():
            gate.info(label, {k: v for k, v in c.items() if k != "record"})
        real = {k: v for k, v in cases.items() if k.startswith("real/")}
        ck(len(real) == len(refuse) and all(c["outcome"] == "refused" and c["type"] == "HeldOutSealed" for c in real.values()),
           "the reader of record refuses all %d manifests on the held-out archive of record: none, missing, invalid JSON, "
           "empty object, JSON list, M15-only, a required field removed, content not matching its hash, claim domain unset"
           % len(refuse))
        m16 = [real.get("real/" + k, {}).get("message", "") for k in ("m15_only", "missing_best", "tampered_content", "claim_domain_unset")]
        ck(all("freeze manifest" in msg and "incomplete" in msg for msg in m16),
           "the four manifests that pass M15's rule are refused by M16's completeness check (message names it)")
        wrong = cases.get("placeholder/complete, wrong expected sha", {})
        ck(wrong.get("outcome") == "refused" and "differs" in wrong.get("message", ""),
           "placeholder with the complete manifest and a wrong expected sha256: refused")
        right = cases.get("placeholder/complete, right sha", {})
        rec = right.get("record") or {}
        ck(right.get("outcome") == "returned" and right.get("n") == 16,
           "placeholder with the complete manifest and its sha256: 16 jets returned")
        ck(rec.get("freeze_manifest_sha256") == data.file_sha256(complete) and rec.get("freeze_manifest_content_sha256") == content_sha,
           "the read record names the manifest's file sha256 %s... (M15) and content sha256 %s... (M16)"
           % (data.file_sha256(complete)[:12], content_sha[:12]))
        fe = (fileaudit.events_with(r["events"], {"heldout", "heldout-raw", "heldout-raw-dir"})
              + fileaudit.events_with(r["events"], {"data-dir"}, data.HELDOUT_ARCHIVE)
              + [e for e in r["events"] if e["overflow"]])
        ck(not r["events_before_start"] and not fe,
           "audit window: no open of the held-out archive, its name in the data directory, the raw held-out "
           "directory or any raw held-out file (%d forbidden events)" % len(fe))
        ck(bool(fileaudit.events_with(r["events"], {"placeholder"})), "positive control: the reader's open of the placeholder is seen")
        fd_bad = [x for x in r["fd_paths"] if os.path.realpath(x) == os.path.realpath(heldout)
                  or os.path.realpath(x).startswith(os.path.realpath(val_dir) + os.sep)]
        ck(not fd_bad, "child fd inventory (%d paths) holds no held-out path" % len(r["fd_paths"]))
        stub = res["stub"]
        agree = all((stub.get(k) == "accepted") == (k == "complete") for k in list(refuse) + ["complete"])
        ck(agree and len(stub) == len(refuse) + 1,
           "the reader of record and SealedReaderStub accept and refuse the same %d manifests" % (len(refuse) + 1))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    ck(not work.exists(), "work directory deleted")


if __name__ == "__main__":
    sys.exit(main())
