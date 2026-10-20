"""Gate G52, Mac part (PREFLIGHT-PLAN section 3 row G52; STUDY v0.8 section 12; module M15).

Checks, at a committed sha, that
1. `cbl/wandb_log.py` imports no Keras, JAX or TensorFlow, imports `wandb` only when a run opens,
   and by static reading never names the W&B credential or a credential file, never calls the
   SDK's login, never writes the environment and never prints (AST, docstrings excluded);
2. the unit tests (`tests/test_wandb_log.py`) pass, run under a temporary HOME;
3. offline records, written by hermetic child processes into a temporary root, hold every
   section-12 key the catalogue requires for three trait combinations (PID-BEST in form FB with
   every group binary; CH-8; FP32), with project, group, entity, run name and run id. The
   records are read by this file's own reader (leveldb log framing and protobuf wire format,
   standard library only, no wandb import), so the gate does not depend on SDK internals;
4. a key naming test data is refused with no manifest, a missing manifest and an invalid one,
   and passes once a dummy manifest exists, and only the allowed row reaches the records;
5. no file under the temporary root, and none of the children's stdout or stderr, holds a
   key-like string (a standalone 40-hex token, or `local-` and 20 or more hex): the only 40-hex
   tokens allowed are the code sha the gate passed in and the wandb build id read from the
   record header. One child runs with decoy credentials planted in its temporary HOME (a
   `.netrc` and a `wandb-api-key.txt`, random 40-hex, never a real key); no decoy may appear in
   any output. The scanner is first shown to flag a planted token (positive control);
6. while the module and the SDK run, the child opens no path named `.netrc` or `_netrc` and no
   path matching `*wandb*key*` (an audit hook on the `open` event plus a patched `open`); the
   SDK's own `*key*.py` source files loaded by import from the interpreter's tree are listed as
   exempt. The audit is shown to flag a deliberate open of the decoys after the module phase.
   It covers the Python process only; the SDK's `wandb-core` service process is a separate Go
   binary, covered by the temporary HOME and the output scan, not by the audit;
7. `WANDB_API_KEY` is absent from the children's environment before and after (presence only);
   no test or fixture file passes the mode `online`.

Every reading is engineering, not quotable. Never runs W&B online. Usage:

    <venv>/bin/python -I -B gates/g52_wandb.py [--expect-sha12 SHA12] [--keep DIR]

`--keep DIR` copies the temporary roots, without the planted decoys, to DIR after the scan.
Prints one line per check and ends with "G52 PASS" (exit 0) or "G52 FAIL" (exit 1).
"""
import argparse
import ast
import builtins
import hashlib
import importlib.metadata as md
import io
import json
import os
import platform
import re
import secrets
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from fnmatch import fnmatchcase
from pathlib import Path

GATE = "G52"
FORK = Path(__file__).resolve().parents[1]
MODULE = FORK / "cbl" / "wandb_log.py"
TESTS = FORK / "tests" / "test_wandb_log.py"
FIXTURES = FORK / "tests" / "wandb_fixtures.py"
HEX40 = re.compile(rb"(?<![0-9A-Fa-f])[0-9A-Fa-f]{40}(?![0-9A-Fa-f])")
LOCAL = re.compile(rb"local-[0-9A-Fa-f]{20,}")
BACKENDS = ("keras", "jax", "jaxlib", "tensorflow", "torch")


# ------------------------------------------------------------------------------- helpers

def hermetic_env(root):
    """A child environment built from nothing: no inherited variable, so no credential."""
    env = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "HOME": str(root / "home"),
           "TMPDIR": str(root / "tmp"), "WANDB_DIR": str(root / "run"),
           "WANDB_CACHE_DIR": str(root / "cache"), "WANDB_CONFIG_DIR": str(root / "config"),
           "WANDB_DATA_DIR": str(root / "data")}
    for k in ("home", "tmp", "run", "cache", "config", "data", "audit", "work"):
        (root / k).mkdir(parents=True, exist_ok=True)
    return env


def scan_bytes(data, allowed):
    """(kind, offset, allowed_as) for each key-like token; values are never returned."""
    hits = []
    for m in HEX40.finditer(data):
        tok = m.group().lower().decode()
        hits.append(("hex40", m.start(), allowed.get(tok)))
    for m in LOCAL.finditer(data):
        hits.append(("local", m.start(), None))
    return hits


def scan_tree(root, allowed, skip):
    out, n = [], 0
    for p in sorted(root.rglob("*")):
        if p.is_symlink() or not p.is_file() or any(p == s or s in p.parents for s in skip):
            continue
        n += 1
        for kind, off, why in scan_bytes(p.read_bytes(), allowed):
            out.append((str(p.relative_to(root)), kind, off, why))
    return out, n


def path_violation(path, exempt_roots):
    """'netrc', 'wandb-key', 'exempt' (interpreter code file) or None."""
    if not isinstance(path, str):
        return None
    low = path.lower()
    name = os.path.basename(low)
    if name in (".netrc", "_netrc"):
        return "netrc"
    if fnmatchcase(low, "*wandb*key*"):
        real = os.path.realpath(path)
        if real.endswith((".py", ".pyc", ".so", ".dylib")) and any(
                real.startswith(r + os.sep) for r in exempt_roots):
            return "exempt"
        return "wandb-key"
    return None


# ------------------------------------------- offline record reader (no wandb import at all)
#
# A `run-<id>.wandb` file is a leveldb-format log (github.com/google/leveldb doc/log_format.md):
# a 7-byte file header (b":W&B", magic 0xBEE1 little-endian, version 0), then 32768-byte blocks
# of records, each with a 7-byte header (crc32 of type byte and payload, zlib polynomial as
# W&B uses; payload length; type FULL, FIRST, MIDDLE or LAST), and zero padding when fewer than
# 7 bytes remain in a block. Each reassembled payload is one protobuf `Record` of
# wandb_internal.proto. The field numbers below were read from the descriptors of wandb 0.28.0
# and 0.31.0 and are equal in both; 0.31.0 adds typed values to HistoryItem (fields 3-8) beside
# `value_json` (16), and both encodings are decoded. A changed number fails the checks that read
# it (project, group, keys), it cannot pass them by accident.

LOG_IDENT, LOG_MAGIC, LOG_VERSION, LOG_BLOCK, LOG_HEADER = b":W&B", 0xBEE1, 0, 32768, 7
FULL, FIRST, MIDDLE, LAST = 1, 2, 3, 4
REC_TYPES = {2: "history", 3: "summary", 17: "run", 21: "header"}   # Record.record_type oneof
NO_VALUE = "<no value>"


def log_payloads(path):
    data = Path(path).read_bytes()
    if data[:4] != LOG_IDENT or len(data) < LOG_HEADER or \
            (data[4] | data[5] << 8, data[6]) != (LOG_MAGIC, LOG_VERSION):
        raise ValueError("%s: not a W&B record log (file header)" % path)
    seed = {t: zlib.crc32(bytes([t])) for t in (FULL, FIRST, MIDDLE, LAST)}
    pos, out, pending = LOG_HEADER, [], None
    while pos < len(data):
        left = LOG_BLOCK - pos % LOG_BLOCK
        if left < LOG_HEADER:
            if data[pos:pos + left].strip(b"\0"):
                raise ValueError("%s: non-zero block trailer at %d" % (path, pos))
            pos += left
            continue
        if len(data) - pos < LOG_HEADER:
            raise ValueError("%s: truncated record header at %d" % (path, pos))
        crc, n, typ = struct.unpack("<IHB", data[pos:pos + LOG_HEADER])
        payload = data[pos + LOG_HEADER:pos + LOG_HEADER + n]
        if typ not in seed or len(payload) != n or zlib.crc32(payload, seed[typ]) & 0xFFFFFFFF != crc:
            raise ValueError("%s: bad record (type %d, length %d) at %d" % (path, typ, n, pos))
        pos += LOG_HEADER + n
        if typ == FULL and pending is None:
            out.append(payload)
        elif typ == FIRST and pending is None:
            pending = payload
        elif typ == MIDDLE and pending is not None:
            pending += payload
        elif typ == LAST and pending is not None:
            out.append(pending + payload)
            pending = None
        else:
            raise ValueError("%s: record fragment type %d out of order at %d" % (path, typ, pos))
    if pending is not None:
        raise ValueError("%s: file ends inside a fragmented record" % path)
    return out


def _varint(buf, i):
    shift = value = 0
    while True:
        if i >= len(buf) or shift > 63:
            raise ValueError("bad varint")
        b = buf[i]
        i += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, i
        shift += 7


def pb_fields(buf):
    """Protobuf wire format: [(field number, wire type, value)]; value is an int (varint), 8 or
    4 raw bytes (fixed), or the bytes of a length-delimited field."""
    i, out = 0, []
    while i < len(buf):
        key, i = _varint(buf, i)
        num, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt in (1, 5):
            v, i = buf[i:i + (8 if wt == 1 else 4)], i + (8 if wt == 1 else 4)
        elif wt == 2:
            n, i = _varint(buf, i)
            v, i = buf[i:i + n], i + n
        else:
            raise ValueError("unsupported wire type %d" % wt)
        if i > len(buf):
            raise ValueError("truncated field %d" % num)
        out.append((num, wt, v))
    return out


def _strs(f, num):
    return [v.decode("utf-8") for n, wt, v in f if n == num and wt == 2]


def _str(f, num):
    s = _strs(f, num)
    return s[-1] if s else ""


def _subs(f, num):
    return [pb_fields(v) for n, wt, v in f if n == num and wt == 2]


def _item_key(f):
    return _str(f, 1) or "/".join(_strs(f, 2))          # key, or the joined nested_key


def _item_value(f):
    """value_json (16), else the 0.31.0 typed HistoryItem value: none 3, boolean 4,
    integer 5 (sint64), number 6 (double), text 7, json 8."""
    js = _strs(f, 16)
    if js:
        return json.loads(js[-1])
    for n, wt, v in reversed(f):
        if n == 8 and wt == 2:
            return json.loads(v.decode("utf-8"))
        if n == 7 and wt == 2:
            return v.decode("utf-8")
        if n == 6 and wt == 1:
            return struct.unpack("<d", v)[0]
        if n == 5 and wt == 0:
            return (v >> 1) ^ -(v & 1)
        if n == 4 and wt == 0:
            return bool(v)
        if n == 3 and wt == 0:
            return None
    return NO_VALUE


def read_records(path):
    """The header, run, history and summary records of one offline run file, as dicts."""
    out = []
    for payload in log_payloads(path):
        f = pb_fields(payload)
        kinds = [(n, v) for n, wt, v in f if n in REC_TYPES and wt == 2]
        if not kinds:
            continue
        num, body = kinds[-1]
        g = pb_fields(body)
        kind = REC_TYPES[num]
        if kind == "header":
            vi = _subs(g, 1)
            out.append({"type": kind, "producer": _str(vi[-1], 1) if vi else ""})
        elif kind == "run":                 # RunRecord; config is ConfigRecord.update (ConfigItem)
            items = [(_item_key(c), _str(c, 16)) for cfg in _subs(g, 4) for c in _subs(cfg, 1)]
            out.append({"type": kind, "run_id": _str(g, 1), "entity": _str(g, 2),
                        "project": _str(g, 3), "group": _str(g, 6), "job_type": _str(g, 7),
                        "name": _str(g, 8), "tags": _strs(g, 10), "config": items})
        elif kind == "history":             # HistoryRecord.item (HistoryItem)
            out.append({"type": kind, "items": {_item_key(h): _item_value(h) for h in _subs(g, 1)}})
        elif kind == "summary":             # SummaryRecord.update (SummaryItem)
            out.append({"type": kind, "keys": [_item_key(s) for s in _subs(g, 1)]})
    return out


# --------------------------------------------------------------------------------- child

def child(spec_path):
    """Runs in a hermetic process. Phase 'module': import, open, log, refuse, finish (module
    and SDK). Phase 'readback': parse the offline records. Phase 'selftest': open the decoys."""
    spec = json.loads(Path(spec_path).read_text())
    opens, phase = [], {"name": "pre"}

    def hook(event, args):
        if event == "open":
            p = args[0]
            if isinstance(p, bytes):
                p = os.fsdecode(p)
            opens.append({"via": "audit", "phase": phase["name"], "path": p if isinstance(p, str) else None,
                          "mode": args[1] if isinstance(args[1], str) else None})
        elif event == "subprocess.Popen":
            opens.append({"via": "popen", "phase": phase["name"], "path": os.fsdecode(args[0])
                          if isinstance(args[0], (str, bytes)) else str(args[0]), "mode": None})

    sys.addaudithook(hook)
    real_open = builtins.open

    def audited_open(file, *a, **kw):
        if isinstance(file, (str, bytes, os.PathLike)):
            opens.append({"via": "open", "phase": phase["name"], "path": os.fsdecode(os.fspath(file)),
                          "mode": (a[0] if a else kw.get("mode", "r"))})
        return real_open(file, *a, **kw)

    builtins.open = audited_open
    io.open = audited_open
    result = {"key_env_before": "WANDB_API_KEY" in os.environ}

    phase["name"] = "module"
    sys.path[:0] = [spec["fork"], str(Path(spec["fork"]) / "tests")]
    import cbl.wandb_log as w
    import wandb_fixtures as fx
    result["after_import"] = {m: m in sys.modules for m in BACKENDS + ("wandb",)}
    run_dir = Path(spec["root"]) / "run"
    manifest_dir = Path(spec["root"]) / "work"
    runs = []
    for label, make in (("full", fx.config), ("ch8", fx.ch8_config), ("fp32", fx.fp32_config)):
        cfg = make(spec["code_sha"])
        run = w.Run.open(cfg, fx.provenance(cfg), source_bundle=fx.SOURCE_BUNDLE,
                         run_dir=run_dir / label)
        events = []
        for e in (0, 1):
            run.log_epoch(fx.epoch_record(e, run.traits))
        run.log_checkpoint(fx.checkpoint_sidecar(1, run.traits))
        if label == "full":
            row = dict(fx.epoch_record(2, run.traits), test_top1=0.5)
            for what, manifest in (("none", None), ("missing", manifest_dir / "absent.json"),
                                   ("invalid", manifest_dir / "invalid.json")):
                if what == "invalid":
                    (manifest_dir / "invalid.json").write_text("[]")
                run.set_freeze_manifest(manifest)
                for kind, call in (("epoch", lambda: run.log_epoch(row)),
                                   ("summary", lambda: run.summary({"test/top1": 0.5}))):
                    try:
                        call()
                        events.append([what, kind, "passed"])
                    except w.HeldOutKeyRefused:
                        events.append([what, kind, "refused"])
            dummy = manifest_dir / "freeze-manifest.json"
            dummy.write_text(json.dumps({"G52": "dummy section-8.5 freeze manifest, synthetic"}))
            run.set_freeze_manifest(dummy)
            run.log_epoch(row)
            run.summary({"test/top1": 0.5})
            events.append(["dummy", "epoch+summary", "passed"])
            result["dummy_manifest_sha256"] = hashlib.sha256(dummy.read_bytes()).hexdigest()
            result["freeze_sha256_recorded"] = run.freeze_sha256
        runs.append({"label": label, "name": run.name, "run_id": run.run_id,
                     "traits": sorted(run.traits), "events": events, "dir": str(run_dir / label)})
        run.finish()

    phase["name"] = "readback"
    for r in runs:
        files = sorted(Path(r["dir"]).glob("wandb/offline-run-*/run-*.wandb"))
        r["record_files"] = [str(f) for f in files]
        r["history"], r["summary_keys"], r["config_keys"], r["runrec"] = [], set(), set(), []
        for f in files:
            for rec in read_records(f):
                t = rec["type"]
                if t == "header":
                    r["producer"] = rec["producer"]
                elif t == "run":
                    r["runrec"].append({k: rec[k] for k in ("project", "group", "entity", "name",
                                                            "run_id", "job_type", "tags")})
                    for k, value_json in rec["config"]:
                        if k.startswith("_"):
                            continue                    # the SDK's own entries
                        v = json.loads(value_json)
                        if isinstance(v, dict):
                            r["config_keys"].update(w.flatten(v, k).keys())
                        else:
                            r["config_keys"].add(k)
                elif t == "history":
                    r["history"].append(rec["items"])
                elif t == "summary":
                    r["summary_keys"].update(rec["keys"])
        r["history_keys"] = sorted({k for h in r["history"] for k in h})
        r["heldout_rows"] = [h.get("epoch") for h in r["history"] if any(w.names_heldout(k) for k in h)]
        r["summary_keys"], r["config_keys"] = sorted(r["summary_keys"]), sorted(r["config_keys"])
        ckpt_rows = [h for h in r["history"] if any(k.startswith("ckpt/") for k in h)]
        epoch_rows = [h for h in r["history"] if h not in ckpt_rows]
        r["n_rows"] = {"epoch": len(epoch_rows), "checkpoint": len(ckpt_rows)}
        r["missing"] = {"config": w.missing(r["config_keys"], "config", set(r["traits"]))}
        for row, rows in (("epoch", epoch_rows), ("checkpoint", ckpt_rows)):
            # every row of the kind must hold every key its kind requires
            r["missing"][row] = sorted({m for h in rows for m in w.missing(h, row, set(r["traits"]))}) \
                if rows else ["<no %s row in the records>" % row]
        r["n_required"] = {row: len(w.required(row, set(r["traits"])))
                           for row in ("epoch", "checkpoint", "config")}
        r["summary_freeze"] = "freeze/manifest_sha256" in r["summary_keys"]
        r["n_history_rows"] = len(r["history"])
        del r["history"]
    result["runs"] = runs
    result["expected"] = {"project": w.PROJECT, "group": w.GROUP, "entity": w.ENTITY}
    result["key_env_after"] = "WANDB_API_KEY" in os.environ

    phase["name"] = "selftest"
    for p in spec.get("decoys", []):
        with open(p, "rb"):
            pass                                # opened and closed; the content is never read

    builtins.open = real_open
    io.open = real_open
    result["opens"] = opens
    Path(spec["out"]).write_text(json.dumps(result, default=list))
    return 0


# -------------------------------------------------------------------------------- parent

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--expect-sha12", default=None)
    ap.add_argument("--keep", default=None)
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        return child(a.child)

    failures = []

    def check(ok, what):
        print("%s  %s" % ("PASS" if ok else "FAIL", what))
        if not ok:
            failures.append(what)

    def git(*args):
        r = subprocess.run(["git", "-C", str(FORK)] + list(args), capture_output=True, text=True,
                           env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C",
                                "GIT_PAGER": "cat", "GIT_OPTIONAL_LOCKS": "0",
                                "HOME": os.environ.get("HOME", "/")})
        return r.returncode, r.stdout, r.stderr

    rc, head, _ = git("rev-parse", "HEAD")
    head = head.strip()
    _, status, _ = git("status", "--porcelain")
    print("gate        %s" % GATE)
    print("interpreter %s" % sys.executable)
    print("python      %s" % sys.version.replace("\n", " "))
    print("wandb       %s" % md.version("wandb"))
    print("pytest      %s" % md.version("pytest"))
    print("platform    %s" % platform.platform())
    print("fork        %s" % FORK)
    print("HEAD        %s" % head)
    print("module      cbl/wandb_log.py sha256 %s" % hashlib.sha256(MODULE.read_bytes()).hexdigest())
    print("label       engineering, not quotable; W&B offline and disabled only")
    print("")
    check(rc == 0 and len(head) == 40, "git rev-parse HEAD succeeds")
    if a.expect_sha12:
        check(head[:12] == a.expect_sha12, "HEAD[:12] %s equals --expect-sha12 %s" % (head[:12], a.expect_sha12))
    check(status == "", "working tree clean (the gate runs at a commit)")

    # 1. static reading of the module
    tree = ast.parse(MODULE.read_text())
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
            and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    imports, top_wandb, strings, names, kwargs, env_writes, prints = set(), [], [], set(), set(), [], 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imports.update(x.name.split(".")[0] for x in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            imports.add(n.module.split(".")[0])
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
            strings.append(n.value)
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, ast.keyword) and n.arg:
            kwargs.add(n.arg)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print":
            prints += 1
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.Delete)):
            for t in getattr(n, "targets", [getattr(n, "target", None)]):
                if isinstance(t, ast.Subscript) and "environ" in ast.unparse(t.value):
                    env_writes.append(ast.unparse(t))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and \
                n.func.attr in ("update", "setdefault", "pop", "clear", "popitem") and \
                "environ" in ast.unparse(n.func.value):
            env_writes.append(ast.unparse(n.func))
    top_wandb = [x for x in tree.body if isinstance(x, (ast.Import, ast.ImportFrom)) and "wandb" in ast.unparse(x)]
    bad_strings = [s for s in strings if any(b in s.lower() for b in ("api_key", "netrc", "wandb-api-key"))]
    bad_names = sorted(names & {"login", "relogin", "api_key", "putenv", "unsetenv"})
    bad_kwargs = sorted(kwargs & {"key", "api_key", "relogin", "anonymous"})
    print("      imports: %s" % sorted(imports))
    check(not (imports & set(BACKENDS)), "module imports none of %s" % (BACKENDS,))
    check(not top_wandb, "module has no module-level wandb import")
    check(not bad_strings, "no non-docstring string in the module names the credential or a credential file (%d strings read)" % len(strings))
    check(not bad_names, "module uses no name or attribute login, relogin, api_key, putenv, unsetenv (found %s)" % bad_names)
    check(not bad_kwargs, "module passes no keyword argument key, api_key, relogin, anonymous (found %s)" % bad_kwargs)
    check(not env_writes, "module never writes os.environ (found %s)" % env_writes)
    check(prints == 0, "module never calls print")
    n_online = sum(p.read_text().count("online") for p in (TESTS, FIXTURES))
    check(n_online == 0, "no test or fixture file contains the string 'online' (count %d)" % n_online)

    work = Path(tempfile.mkdtemp(prefix="g52-"))
    try:
        # 2. unit tests under a temporary HOME
        troot = work / "pytest"
        env = hermetic_env(troot)
        r = subprocess.run([sys.executable, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            "-c", os.devnull, "--rootdir", str(FORK), str(TESTS)], cwd=str(troot / "work"), env=env, capture_output=True, text=True)
        print("      pytest exit %d; output follows" % r.returncode)
        for line in (r.stdout + r.stderr).splitlines():
            print("      | %s" % line)
        check(r.returncode == 0, "tests/test_wandb_log.py passes")
        pytest_out = (r.stdout + r.stderr).encode()

        # 5a. positive control of the scanner
        probe = work / "probe"
        probe.mkdir()
        (probe / "planted.txt").write_bytes(b"x " + secrets.token_hex(20).encode() + b" local-"
                                            + secrets.token_hex(20).encode() + b" " + head.encode())
        hits, _ = scan_tree(probe, {head: "code sha"}, [])
        kinds = {(k, why) for _, k, _, why in hits}
        print("      positive control hits (kind, allowed as): %s" % sorted((k, str(why)) for k, why in kinds))
        check(kinds == {("hex40", None), ("local", None), ("hex40", "code sha")},
              "scanner positive control: flags a planted 40-hex and a planted local- token, allows the code sha")

        # 3-7. hermetic offline children
        exempt_roots = sorted({os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)})
        child_results, child_outputs, decoy_values, decoy_files = {}, {}, [], []
        for label in ("clean", "decoy"):
            root = work / label
            env = hermetic_env(root)
            decoys = []
            if label == "decoy":
                fake_hex, fake_local = secrets.token_hex(20), "local-" + secrets.token_hex(20)
                decoy_values += [fake_hex.encode(), fake_local.encode()]
                netrc = root / "home" / ".netrc"
                netrc.write_text("machine api.wandb.ai\n  login user\n  password %s\n" % fake_hex)
                os.chmod(netrc, 0o600)
                keyf = root / "work" / "wandb-api-key.txt"
                keyf.write_text(fake_local + "\n")
                decoys = [str(netrc), str(keyf)]
                decoy_files += [netrc, keyf]
            spec = {"fork": str(FORK), "root": str(root), "code_sha": head, "decoys": decoys,
                    "out": str(root / "audit" / "child.json")}
            (root / "audit" / "spec.json").write_text(json.dumps(spec))
            r = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--child",
                                str(root / "audit" / "spec.json")], cwd=str(root / "work"), env=env,
                               capture_output=True)
            child_outputs[label] = r.stdout + r.stderr
            print("      child %s exit %d; %d bytes of stdout+stderr follow" % (label, r.returncode, len(child_outputs[label])))
            for line in child_outputs[label].decode("utf-8", "replace").splitlines():
                print("      | %s" % line)
            check(r.returncode == 0, "child %s completes" % label)
            if r.returncode == 0:
                child_results[label] = json.loads((root / "audit" / "child.json").read_text())

        allowed = {head: "code sha passed in"}
        for label, res in child_results.items():
            print("")
            print("      -- child %s" % label)
            check(not any(res["after_import"][m] for m in BACKENDS),
                  "%s: importing cbl.wandb_log loads none of %s" % (label, BACKENDS))
            check(not res["after_import"]["wandb"], "%s: importing cbl.wandb_log does not import wandb" % label)
            check(res["key_env_before"] is False and res["key_env_after"] is False,
                  "%s: WANDB_API_KEY absent before and after (presence only)" % label)
            exp = res["expected"]
            for run in res["runs"]:
                tag = "%s/%s" % (label, run["label"])
                prod = run.get("producer", "")
                m = re.search(r"\+([0-9a-f]{40})$", prod)
                if m:
                    allowed[m.group(1)] = "wandb build id (record header producer %s)" % prod
                rr = run["runrec"]
                print("      %s: name %s id %s traits %s record files %d history rows %d" % (
                    tag, run["name"], run["run_id"], run["traits"], len(run["record_files"]), run["n_history_rows"]))
                check(len(run["record_files"]) == 1, "%s: one offline record file written" % tag)
                check(len(rr) >= 1 and all(x["project"] == exp["project"] and x["group"] == exp["group"]
                                           and x["entity"] == exp["entity"] for x in rr),
                      "%s: run record project %s, group %s, entity %s" % (tag, exp["project"], exp["group"], exp["entity"]))
                check(all(x["name"] == run["name"] and x["run_id"] == run["run_id"] for x in rr),
                      "%s: run record name and id equal the module's" % tag)
                for row in ("epoch", "checkpoint", "config"):
                    check(not run["missing"][row], "%s: every required section-12 %s key present%s (%d required; missing %s)"
                          % (tag, row, "" if row == "config" else " in each of its %d %s rows" % (run["n_rows"][row], row),
                             run["n_required"][row], run["missing"][row]))
                if run["label"] == "full":
                    ev = {(e[0], e[1]): e[2] for e in run["events"]}
                    for what in ("none", "missing", "invalid"):
                        check(ev.get((what, "epoch")) == "refused" and ev.get((what, "summary")) == "refused",
                              "%s: test key refused in an epoch row and in the summary with manifest %s" % (tag, what))
                    check(ev.get(("dummy", "epoch+summary")) == "passed", "%s: test key passes once the dummy manifest exists" % tag)
                    check(run["heldout_rows"] == [2], "%s: exactly one history row holds a test key, the allowed one (epochs %s)"
                          % (tag, run["heldout_rows"]))
                    check(run["summary_freeze"] and res["freeze_sha256_recorded"] == res["dummy_manifest_sha256"],
                          "%s: summary records the dummy manifest's sha256" % tag)
                    check("test/top1" in run["summary_keys"], "%s: the allowed summary test key is in the records" % tag)
            # 6. open audit
            viol = [(o["phase"], o["via"], o["path"], path_violation(o["path"], exempt_roots)) for o in res["opens"]]
            mod = [v for v in viol if v[0] in ("pre", "module", "readback") and v[3] in ("netrc", "wandb-key")]
            exempt = sorted({v[2] for v in viol if v[3] == "exempt"})
            selft = [v for v in viol if v[0] == "selftest" and v[3] in ("netrc", "wandb-key")]
            n_audit = sum(1 for o in res["opens"] if o["via"] == "audit")
            n_patch = sum(1 for o in res["opens"] if o["via"] == "open")
            popens = sorted({os.path.basename(o["path"] or "") for o in res["opens"] if o["via"] == "popen"})
            print("      %s: %d audit open events, %d patched-open calls, subprocesses %s" % (label, n_audit, n_patch, popens))
            for p in exempt:
                print("      %s: exempt interpreter code file matching *wandb*key*: %s" % (label, p))
            for v in mod:
                print("      %s: VIOLATION %s" % (label, v))
            check(not mod, "%s: no open of a .netrc/_netrc path or a *wandb*key* path while the module and SDK ran" % label)
            if label == "decoy":
                check(len({v[2] for v in selft}) == 2, "decoy: audit positive control flags the deliberate open of both decoys")

        # 5b. scan every file the children wrote, and their stdout and stderr
        print("")
        for tok, why in sorted(allowed.items(), key=lambda x: x[1]):
            print("      allowed 40-hex: %s (%s)" % (tok, why))
        skip = decoy_files + [probe]
        hits, nfiles = scan_tree(work, allowed, skip)
        bad = [h for h in hits if h[3] is None]
        for f, kind, off, why in hits:
            print("      %s %s at %s:%d" % ("allowed" if why else "UNEXPECTED (value withheld)", kind, f, off)
                  + (" = %s" % why if why else ""))
        check(not bad, "no unexpected key-like token in %d files under the temporary root" % nfiles)
        out_bad = [(lbl, k, o) for lbl, data in list(child_outputs.items()) + [("pytest", pytest_out)]
                   for k, o, why in scan_bytes(data, allowed) if why is None]
        check(not out_bad, "no unexpected key-like token in the children's or pytest's stdout and stderr")
        leaked = []
        for p in sorted(work.rglob("*")):
            if p.is_file() and not p.is_symlink() and p not in decoy_files:
                data = p.read_bytes()
                leaked += [str(p.relative_to(work)) for d in decoy_values if d in data]
        leaked += [lbl for lbl, data in child_outputs.items() for d in decoy_values if d in data]
        check(len(decoy_values) == 2 and not leaked, "no decoy credential appears in any output file or stream (%d places)" % len(leaked))

        if a.keep:
            dest = Path(a.keep)
            dest.mkdir(parents=True, exist_ok=True)
            for label in ("clean", "decoy"):
                shutil.copytree(work / label, dest / label, symlinks=True, dirs_exist_ok=True,
                                ignore=lambda d, names: [n for n in names if Path(d, n) in decoy_files])
            print("      kept %s (decoys removed)" % dest)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("")
    if failures:
        print("%s FAIL (%d failed check(s))" % (GATE, len(failures)))
        return 1
    print("%s PASS" % GATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
