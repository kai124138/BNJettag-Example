"""Shared parts of gates G53-G56 (PREFLIGHT-PLAN section 3; module M16 on the Mac CPU, lane M).

Standard library only, plus `gatelib`. Each gate prints the gatelib header, a line saying that every
number is an engineering record and not quotable, the interpreter's numpy and scipy versions, the
sha256 of the four M16 modules, then one line per check, and ends with "<gate> PASS" or "<gate>
FAIL". Gates write nothing into the fork; evidence is captured by the caller from stdout.
"""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from gatelib import FORK, Gate

STUDY_SHA256 = "61e68527607b76150db4ab9c1e198c20564b8803646de70dc292d39966f89df4"  # STUDY v0.9 (2026-10-18)
STUDY_DEFAULT = FORK.parent / "review" / "snapshots" / "v0.9" / "STUDY.md"
M16 = ("cbl/stats.py", "cbl/select.py", "cbl/s2.py", "cbl/confirm.py")
ALLOWED = {"__future__", "dataclasses", "fractions", "hashlib", "json", "math", "pathlib", "re", "typing", "numpy", "scipy"}
ALLOWED_CBL = {"cbl.select", "cbl.stats"}
BANNED = ("keras", "jax", "jaxlib", "tensorflow", "torch", "hgq", "qkeras", "hls4ml", "da4ml")
BANNED_CBL = ("cbl.env", "cbl.wandb_log", "cbl.refdiff", "cbl.cost_pure")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args(doc):
    ap = argparse.ArgumentParser(description=doc.splitlines()[0])
    ap.add_argument("--expect-sha12", default=None)
    ap.add_argument("--study", default=str(STUDY_DEFAULT))
    return ap.parse_args()


def start(name, args):
    """Gate object with the header printed and the repository checks done."""
    gate = Gate(name)
    print("record      engineering record, not quotable")
    gate.header(args.expect_sha12)
    import numpy
    import scipy
    gate.info("numpy", numpy.__version__)
    gate.info("scipy", scipy.__version__)
    for m in M16:
        gate.info(m, "sha256 " + sha256_file(FORK / m))
    return gate


def study_text(gate, path):
    p = Path(path)
    ok = p.is_file() and sha256_file(p) == STUDY_SHA256
    gate.check(ok, "STUDY snapshot %s hashes to %s..." % (p, STUDY_SHA256[:16]))
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def static_imports(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "cbl":
                for a in node.names:
                    out.add("cbl." + a.name)
            elif node.module:
                out.add(node.module)
    return out


def check_imports(gate):
    """Static: the M16 modules import only the standard library, numpy, scipy and the M16 modules
    cbl.select and cbl.stats. Dynamic: importing all four in a fresh -I interpreter loads no backend
    and no other cbl module."""
    for m in M16:
        imps = static_imports(FORK / m)
        bad = sorted(i for i in imps if i.split(".")[0] not in ALLOWED and i not in ALLOWED_CBL)
        gate.info("static imports of " + m, sorted(imps))
        gate.check(not bad, "%s imports only the standard library, numpy, scipy, cbl.select and cbl.stats%s"
                   % (m, "" if not bad else " (found %s)" % bad))
    code = ("import json, sys; sys.path.insert(0, %r); import cbl.stats, cbl.select, cbl.s2, cbl.confirm; "
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in %r or m in %r); "
            "import importlib.util as u; avail = {k: u.find_spec(k) is not None for k in ('keras', 'jax', 'tensorflow')}; "
            "print(json.dumps({'bad': bad, 'available': avail}))") % (str(FORK), BANNED, BANNED_CBL)
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, cwd=str(FORK))
    try:
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        res = {"bad": ["<no result: %s>" % r.stderr.strip()[-300:]], "available": {}}
    gate.info("importable in this interpreter", res.get("available"))
    gate.check(r.returncode == 0 and not res["bad"],
               "importing cbl.stats, cbl.select, cbl.s2 and cbl.confirm in a fresh -I interpreter loads no backend "
               "(keras, jax, tensorflow, ...) and no cbl.env or cbl.wandb_log")


def run_pytest(gate, files):
    for f in files:
        gate.info(f, "sha256 " + sha256_file(FORK / f))
    r = subprocess.run([sys.executable, "-I", "-B", "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider"] + list(files),
                       capture_output=True, text=True, cwd=str(FORK))
    tail = [l for l in r.stdout.splitlines() if l.strip()][-3:]
    for l in tail:
        print("      | " + l)
    gate.check(r.returncode == 0, "%s pass (pytest exit %d)" % (", ".join(files), r.returncode))


def finish(gate):
    gate.check_clean("at end")
    return gate.finish()
