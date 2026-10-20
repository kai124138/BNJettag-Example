"""Gate G25 (PREFLIGHT-PLAN section 3 row G25, section 4 row B3a; STUDY v0.8 sections 3.2, 3.5).

Checks the pure part of M7, `cbl/cost_pure.py`, on the Mac CPU with no backend:
  - repository: HEAD equals --expect-sha12 and `git status --porcelain` is empty;
  - import isolation: the module imports only the standard library and numpy (static AST check),
    and importing it loads no keras, jax or other backend module, both in a fresh `-I` interpreter
    and in this process (the gate itself imports only the standard library and numpy);
  - the STUDY snapshot hashes to its recorded sha256 and its section 3.5 table parses to the
    values below;
  - w(.) on constructed values: w(1) = 1, w(0.375) = 2, zero, negative values, powers of two over
    the float32 range, and the largest widths the quantizers allow (23 bits) and a float32 holds
    (24 bits); w <= b on every value of every small SAT_SYM grid;
  - the snap: a no-op on the grid, a SnapError off it;
  - the u_c bound: u = 9 at n_in = 24 and 32 for the section 3.5 inputs;
  - the H2 column rule on constructed columns (zero, other, single magnitude on both branches);
  - the section 3.5 table, every row and cell, from its closed formulas and from the module's
    general site functions;
  - the unit tests `tests/test_cost_pure.py`, run as a sub-step with the same interpreter.

Run, with the Mac CPU interpreter, as

    python -I -B gates/g25_cost_pure.py --expect-sha12 SHA12 [--study PATH]

--study defaults to the campaign's `review/snapshots/v0.8/STUDY.md` beside the fork; pass it
explicitly when the fork is a worktree elsewhere. Prints one line per check and ends with
"G25 PASS" (exit 0) or "G25 FAIL" (exit 1). Writes nothing. Every number printed is an
engineering reading, not a quotable result.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

GATE = "G25"
FORK = Path(__file__).resolve().parents[1]
MODULE = FORK / "cbl" / "cost_pure.py"
TESTS = FORK / "tests" / "test_cost_pure.py"
STUDY_SHA256 = "290022fa41a726b8ae66ff302f7910a69c4c148ec27b15c2448cab83fb898ca8"
BANNED = ("keras", "jax", "jaxlib", "tensorflow", "torch", "hgq", "qkeras", "hls4ml", "da4ml")
ALLOWED_IMPORTS = {"__future__", "math", "dataclasses", "fractions", "typing", "numpy"}

# STUDY v0.8 section 3.5 table (review/snapshots/v0.8/STUDY.md:610-613), as printed:
# site -> (native 4-bit, H1 w=24, (H1 w=4, 2, 1), H2 w=24, (H2 w=4, 2, 1))
TABLE = {
    "FFN expansion 24→32": (98304, 589824, (98304, 49152, 24576), 79872, (33792, 29184, 24576)),
    "FFN contraction 32→24": (98304, 589824, (98304, 49152, 24576), 66048, (31488, 28032, 24576)),
    "head 32→32, pooled": (16384, 98304, (16384, 8192, 4096), 11008, (5248, 4672, 4096)),
    "Q, K, V and O together": (393216, 2359296, (393216, 196608, 98304), 305664, (132864, 115584, 98304)),
}
# (P, n_in, n_out) of the sites in each row: N = 8 positions, one pooled position at the head;
# Q, K and V kernels 24x(2*16), O (2*16)x24 (STUDY section 2.2 "Non-BN sites")
SHAPES = {
    "FFN expansion 24→32": [(8, 24, 32)],
    "FFN contraction 32→24": [(8, 32, 24)],
    "head 32→32, pooled": [(1, 32, 32)],
    "Q, K, V and O together": [(8, 24, 32)] * 3 + [(8, 32, 24)],
}
B_BAR, I_IN, F_IN, B_K = 4, 1, 3, 4  # section 3.5 assumptions: 4-bit inputs, I = 1, F = 3; b0 = 4


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    env = dict(os.environ, LC_ALL="C", GIT_PAGER="cat", GIT_OPTIONAL_LOCKS="0")
    r = subprocess.run(["git", "-C", str(FORK)] + list(args), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, env=env, check=False)
    return r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")


def parse_table(study_text):
    """The section 3.5 table rows of the STUDY, as {site: tuple} in the TABLE layout."""
    lines = study_text.splitlines()
    head = [n for n, l in enumerate(lines) if l.startswith("| site | native, 4-bit learned kernel |")]
    if len(head) != 1:
        raise ValueError("section 3.5 table header found %d times" % len(head))
    n0 = head[0]
    rows = {}
    for l in lines[n0 + 2:n0 + 6]:
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if len(cells) != 6:
            raise ValueError("unexpected table row: %r" % l)
        num = lambda s: int(s.replace(",", ""))
        trip = lambda s: tuple(num(x) for x in s.split(" / "))
        rows[cells[0]] = (num(cells[1]), num(cells[2]), trip(cells[3]), num(cells[4]), trip(cells[5]))
    return rows, n0 + 1


def u_closed(n_in):
    # u = ceil(log2(n_in * 2**I)) + F, integer arithmetic independent of the module
    return ((n_in * 2 ** I_IN) - 1).bit_length() + F_IN


def row_closed(row, wv):
    """Section 3.5 closed formulas: native = P n_in n_out b b_k; H1 = P n_in n_out b w;
    H2 per column = min(P n_in b w, P n_in b + P u w)."""
    nat = h1 = h2 = 0
    for P, n_in, n_out in SHAPES[row]:
        nat += P * n_in * n_out * B_BAR * B_K
        h1 += P * n_in * n_out * B_BAR * wv
        h2 += n_out * min(P * n_in * B_BAR * wv, P * n_in * B_BAR + P * u_closed(n_in) * wv)
    return nat, h1, h2


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--expect-sha12", default=None)
    ap.add_argument("--study", default=str(FORK.parent / "review" / "snapshots" / "v0.8" / "STUDY.md"))
    a = ap.parse_args()

    failures = []

    def check(ok, what):
        print("%s  %s" % ("PASS" if ok else "FAIL", what))
        if not ok:
            failures.append(what)
        return ok

    import numpy as np  # the gate's only non-stdlib import before the module

    _, git_version, _ = git("--version")
    rc_head, head, err_head = git("rev-parse", "HEAD")
    head = head.strip()
    _, branch, _ = git("rev-parse", "--abbrev-ref", "HEAD")
    spec = {m: importlib.util.find_spec(m) is not None for m in ("keras", "tensorflow", "jax", "pytest")}
    print("gate        %s (engineering record, not quotable)" % GATE)
    print("interpreter %s" % sys.executable)
    print("python      %s" % sys.version.replace("\n", " "))
    print("platform    %s" % platform.platform())
    print("host        %s" % platform.node())
    print("git         %s" % git_version.strip())
    print("numpy       %s" % np.__version__)
    print("importable  %s" % ", ".join("%s=%s" % kv for kv in sorted(spec.items())))
    print("fork        %s" % FORK)
    print("branch      %s" % branch.strip())
    print("HEAD        %s" % (head if rc_head == 0 else "ERROR " + err_head.strip()))
    print("module      %s sha256 %s" % (MODULE.relative_to(FORK), sha256_file(MODULE)))
    print("tests       %s sha256 %s" % (TESTS.relative_to(FORK), sha256_file(TESTS)))
    print("")

    # A. repository
    check(rc_head == 0 and len(head) == 40, "git rev-parse HEAD succeeds")
    if a.expect_sha12 is not None:
        check(head[:12] == a.expect_sha12, "HEAD[:12] %s equals --expect-sha12 %s" % (head[:12], a.expect_sha12))
    rc, out, _ = git("status", "--porcelain")
    print("      git status --porcelain: %d line(s)" % len(out.splitlines()))
    for line in out.splitlines():
        print("      | %s" % line)
    check(rc == 0 and out == "", "git status --porcelain is empty")

    # B. import isolation
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(n.name.split(".")[0] for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0] if node.level == 0 else ".")
    print("      static imports of cbl/cost_pure.py: %s" % ", ".join(sorted(imported)))
    check(imported <= ALLOWED_IMPORTS, "cbl/cost_pure.py imports only %s" % ", ".join(sorted(ALLOWED_IMPORTS)))
    code = ("import sys, json; sys.path.insert(0, %r); import cbl.cost_pure as m; "
            "print(json.dumps({'file': m.__file__, 'loaded': sorted({k.split('.')[0] for k in sys.modules})}))"
            % str(FORK))
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, check=False)
    try:
        info = json.loads(r.stdout)
    except ValueError:
        info = {"file": None, "loaded": []}
        print("      fresh interpreter stderr: %s" % r.stderr.strip())
    banned_fresh = sorted(set(BANNED) & set(info["loaded"]))
    print("      fresh interpreter: module %s; backend modules loaded: %s" % (info["file"], banned_fresh or "none"))
    check(r.returncode == 0 and info["file"] is not None
          and Path(info["file"]).resolve() == MODULE.resolve() and not banned_fresh,
          "fresh -I interpreter imports cbl.cost_pure from this fork and loads no backend module")
    banned_before = sorted(m for m in sys.modules if m.split(".")[0] in BANNED)
    sys.path.insert(0, str(FORK))
    import cbl.cost_pure as cp  # noqa: E402
    banned_after = sorted(m for m in sys.modules if m.split(".")[0] in BANNED)
    check(not banned_before and not banned_after and "keras" not in sys.modules and "jax" not in sys.modules,
          "after `import cbl.cost_pure` in this process, keras and jax (and %s) are not in sys.modules"
          % ", ".join(BANNED[2:]))
    check(Path(cp.__file__).resolve() == MODULE.resolve(), "imported module file is %s" % MODULE)
    check(cp.STUDY_SHA256 == STUDY_SHA256, "module cites STUDY sha256 %s" % STUDY_SHA256[:16])

    # C. the STUDY snapshot and its section 3.5 table
    study = Path(a.study)
    study_ok = study.is_file() and sha256_file(study) == STUDY_SHA256
    check(study_ok, "STUDY snapshot %s hashes to %s" % (study, STUDY_SHA256))
    if study_ok:
        rows, line = parse_table(study.read_text(encoding="utf-8"))
        print("      section 3.5 table header at STUDY line %d" % line)
        check(rows == TABLE, "section 3.5 table parsed from the STUDY equals the values this gate checks")

    # D. w(.)
    w = cp.w
    check(w(1) == 1, "w(1) = 1 (STUDY section 3.2)")
    check(w(0.375) == 2 and w(Fraction(3, 8)) == 2 and w(np.float32(0.375)) == 2, "w(0.375) = w(3*2^-3) = 2")
    check(w(0) == 0 and w(-0.0) == 0 and w(np.float32(0)) == 0, "w(0) = 0, also for -0.0 and float32 zero")
    check(w(-0.375) == 2 and w(-1) == 1 and w(np.float32(-15 / 16)) == 4, "negative values: w(-q) = w(q)")
    pw = [w(np.float32(2.0 ** k)) for k in range(-149, 128)] + [w(np.float32(-(2.0 ** k))) for k in range(-149, 128)]
    check(set(pw) == {1}, "powers of two 2^-149 .. 2^127 (float32 subnormal to max), both signs: w = 1")
    w23 = [w((2 ** 23 - 1) * Fraction(2) ** (i - 23)) for i in (-8, 0, 12, 30)]
    check(w23 == [23] * 4, "largest kernel grid value (2^23 - 1) * 2^(i - 23), b = 23 = bc max: w = 23")
    check(w(np.float32((2 ** 24 - 1) / 2 ** 24)) == 24 and w(np.float32(np.finfo(np.float32).max)) == 24,
          "24 significant bits (the most a float32 holds): w = 24")
    worst = 0
    for b in range(0, 7):
        for i in range(-3, 10):
            for k in range(-(2 ** b - 1), 2 ** b):
                v = Fraction(k) * Fraction(2) ** (i - b)
                worst = max(worst, w(v) - b)
    check(worst <= 0, "every value on every SAT_SYM grid with b <= 6, i in -3..9 (f < 0 included) has w <= b")
    check(all(w(v) == cp.w_literal(v) for v in (1, 0.375, 3, 0.1, np.float32(0.1), 2.0 ** -40, 12345.5)),
          "w equals floor(log2|q|) - nu(q) + 1 on sample values")

    # E. snap
    check(cp.snap_exact(0.375, 3) == Fraction(3, 8) and cp.snap_exact(16, -3) == 16 and cp.snap_exact(0, -5) == 0,
          "snap to the quantizer's f is a no-op on the grid (f > 0, f < 0, zero)")
    off = np.nextafter(np.float32(15 / 16), np.float32(1))
    try:
        cp.snap_exact(off, 4)
        raised = False
    except cp.SnapError:
        raised = True
    check(raised and w(off) == 24, "one float32 ulp off the 2^-4 grid: snap raises SnapError (w would read 24)")
    try:
        cp.snap_exact(0.375, 2)
        raised = False
    except cp.SnapError:
        raised = True
    check(raised, "0.375 at f = 2: snap raises SnapError")

    # F. u_c
    for n in (24, 32):
        u = cp.u_bound([I_IN] * n, [F_IN] * n, [B_BAR] * n, [True] * n)
        check(u == 9 == u_closed(n), "u = ceil(log2(%d * 2)) + 3 = 9 at n_in = %d (STUDY section 3.5)" % (n, n))

    # G. H2 column rule on constructed columns (2 positions, 2 inputs, b = 4, I = 1, F = 3)
    bb, II, FF = [[4, 4], [4, 4]], [[1, 1], [1, 1]], [[3, 3], [3, 3]]
    check(cp.h2_column(37, "zero", None, bb, II, FF, [False, False])[0] == 0, "H2: a zero column costs 0")
    check(cp.h2_column(37, "other", None, bb, II, FF, [True, True])[0] == 37, "H2: a multi-magnitude column costs H1_c")
    r1 = cp.h2_column(64, "single", Fraction(15, 16), bb, II, FF, [True, True])
    check(r1 == (56, "sign_scale", 16, 10), "H2: single magnitude w = 4: min(64, 16 + 10 * 4) = 56 %r" % (r1,))
    r2 = cp.h2_column(16, "single", Fraction(1, 2), bb, II, FF, [True, True])
    check(r2 == (16, "h1", 16, 10), "H2: single magnitude w = 1: min(16, 16 + 10) = 16 (the floor) %r" % (r2,))

    # H. the section 3.5 table from its formulas and from the general site functions
    rng = np.random.default_rng(35)
    g_by_w = {24: np.float32((2 ** 24 - 1) / 2 ** 24), 4: np.float32(15 / 16), 2: np.float32(0.375), 1: np.float32(0.5)}
    print("      %-24s %10s %10s %26s %10s %26s" % ("site", "native", "H1 w=24", "H1 w=4/2/1", "H2 w=24", "H2 w=4/2/1"))
    for row, exp in TABLE.items():
        closed = {wv: row_closed(row, wv) for wv in (24, 4, 2, 1)}
        got_closed = (closed[24][0], closed[24][1], tuple(closed[wv][1] for wv in (4, 2, 1)),
                      closed[24][2], tuple(closed[wv][2] for wv in (4, 2, 1)))
        general = {}
        nat_general = 0
        for P, n_in, n_out in SHAPES[row]:
            Kq = (rng.integers(-15, 16, (n_in, n_out)) / 16.0).astype(np.float32)  # 4-bit learned kernel
            s = cp.SiteInput(row, "other", "hgq", np.full((P, n_in), B_BAR), np.full((P, n_in), I_IN),
                             np.full((P, n_in), F_IN), Kq, 4, B_K)
            nat_general += cp.site_terms(s).NR  # NR is native at an HGQ site
        for wv, g in g_by_w.items():
            h1 = h2 = 0
            for P, n_in, n_out in SHAPES[row]:
                S = np.where(rng.random((n_in, n_out)) < 0.5, -1.0, 1.0).astype(np.float32)
                s = cp.SiteInput(row, "other", "binary", np.full((P, n_in), B_BAR), np.full((P, n_in), I_IN),
                                 np.full((P, n_in), F_IN), S * g, -cp.lowest_set_bit_exponent(g), 24)
                t = cp.site_terms(s)
                h1, h2 = h1 + t.H1, h2 + t.H2
            general[wv] = (h1, h2)
        got_general = (nat_general, general[24][0], tuple(general[wv][0] for wv in (4, 2, 1)),
                       general[24][1], tuple(general[wv][1] for wv in (4, 2, 1)))
        fmt = lambda tup: "%10d %10d %26s %10d %26s" % (tup[0], tup[1], " / ".join(map(str, tup[2])), tup[3],
                                                         " / ".join(map(str, tup[4])))
        print("      %-24s %s  (closed formulas)" % (row, fmt(got_closed)))
        print("      %-24s %s  (general site functions)" % ("", fmt(got_general)))
        check(got_closed == exp, "section 3.5 row '%s': closed formulas equal the printed row" % row)
        check(got_general == exp, "section 3.5 row '%s': general site functions equal the printed row" % row)

    # I. unit tests
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    cmd = [sys.executable, "-I", "-B", "-m", "pytest", "-v", "--color=no", "-p", "no:cacheprovider",
           "--rootdir", str(FORK), str(TESTS)]
    print("")
    print("      pytest sub-step: %s" % " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(FORK), capture_output=True, text=True, env=env, check=False)
    for line in r.stdout.splitlines():
        print("      | %s" % line)
    for line in r.stderr.splitlines():
        print("      ! %s" % line)
    check(r.returncode == 0, "tests/test_cost_pure.py passes (pytest exit %d)" % r.returncode)

    print("")
    if failures:
        print("%s FAIL (%d failed check(s))" % (GATE, len(failures)))
        return 1
    print("%s PASS" % GATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
