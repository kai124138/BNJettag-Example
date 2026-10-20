"""Gate G01 (PREFLIGHT-PLAN section 3; STUDY v0.8 section 13, reference-fidelity row).

Asserts that Chang's seven released jsc150 files under chang/ and the vendored vendor/stop_if.py
are byte-identical to their sources under the lab's reference-code/ and to the sha256 values
recorded in code-base.txt, that the recorded HGQ2 snapshot tree hash recomputes, and that
`git status --porcelain` of the fork is empty.

Standard library only; no Keras or JAX import. Run from anywhere as

    python3 -I -B gates/g01_base.py [--lab-root DIR] [--expect-sha12 SHA12]

The lab root defaults to four levels above this file (code/gates/ -> code/ -> campaign ->
campaigns/ -> lab root). Prints one line per check and ends with "G01 PASS" (exit 0) or
"G01 FAIL" (exit 1). Writes nothing.
"""

import argparse
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

GATE = "G01"

# (fork path, source path relative to the lab root, sha256 prefix stated in PREFLIGHT-PLAN.md
# section 1 and in the B0 brief). The prefixes are an independent check on code-base.txt.
EXPECTED = [
    ("chang/data.py", "reference-code/HGQ2-examples/jsc150/data.py", "a02743b6"),
    ("chang/model.py", "reference-code/HGQ2-examples/jsc150/model.py", "7fcb38b4"),
    ("chang/run_train.py", "reference-code/HGQ2-examples/jsc150/run_train.py", "bfe36347"),
    ("chang/run_test.py", "reference-code/HGQ2-examples/jsc150/run_test.py", "e4456710"),
    ("chang/test_utils.py", "reference-code/HGQ2-examples/jsc150/test_utils.py", "1c0c4d57"),
    ("chang/tools/prepare_data.py", "reference-code/HGQ2-examples/jsc150/tools/prepare_data.py", "37f37a56"),
    ("chang/tools/prepare_data.sh", "reference-code/HGQ2-examples/jsc150/tools/prepare_data.sh", "82b20099"),
    ("vendor/stop_if.py", "reference-code/HGQ2/src/hgq/utils/sugar/stop_if.py", "ac398006"),
]
TREE = "reference-code/HGQ2/src"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(root):
    """The method stated in code-base.txt: regular files, no __pycache__/ or *.pyc, lines
    '<sha256>  <relpath>' sorted by relpath as UTF-8 bytes, sha256 of the concatenation."""
    root = Path(root)
    rows = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if "__pycache__" in rel.parts or p.name.endswith(".pyc"):
            continue
        if p.is_symlink() or not p.is_file():
            continue
        rows.append((rel.as_posix(), sha256_file(p)))
    rows.sort(key=lambda r: r[0].encode("utf-8"))
    listing = "".join("%s  %s\n" % (h, r) for r, h in rows)
    return hashlib.sha256(listing.encode("utf-8")).hexdigest(), len(rows)


def parse_code_base(path):
    files, trees = [], []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("  ")
        if parts[0] == "file" and len(parts) == 4:
            files.append((parts[2], parts[3], parts[1]))
        elif parts[0] == "tree" and len(parts) == 4 and parts[3].startswith("files="):
            trees.append((parts[2], parts[1], int(parts[3][len("files="):])))
        else:
            raise ValueError("code-base.txt line %d not understood: %r" % (n, line))
    return files, trees


def git(fork, *args):
    env = dict(os.environ, LC_ALL="C", GIT_PAGER="cat", GIT_OPTIONAL_LOCKS="0")
    r = subprocess.run(["git", "-C", str(fork)] + list(args), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, env=env, check=False)
    return r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")


def main():
    fork_default = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lab-root", default=str(Path(__file__).resolve().parents[4]))
    ap.add_argument("--expect-sha12", default=None)
    a = ap.parse_args()
    fork, lab = fork_default, Path(a.lab_root).resolve()

    failures = []

    def check(ok, what):
        print("%s  %s" % ("PASS" if ok else "FAIL", what))
        if not ok:
            failures.append(what)

    _, git_version, _ = git(fork, "--version")
    rc_head, head, err_head = git(fork, "rev-parse", "HEAD")
    head = head.strip()
    print("gate        %s" % GATE)
    print("interpreter %s" % sys.executable)
    print("python      %s" % sys.version.replace("\n", " "))
    print("platform    %s" % platform.platform())
    print("host        %s" % platform.node())
    print("git         %s" % git_version.strip())
    print("fork        %s" % fork)
    print("lab_root    %s" % lab)
    print("HEAD        %s" % (head if rc_head == 0 else "ERROR " + err_head.strip()))
    print("")

    check(rc_head == 0 and len(head) == 40, "git rev-parse HEAD succeeds")
    if a.expect_sha12 is not None:
        check(head[:12] == a.expect_sha12, "HEAD[:12] %s equals --expect-sha12 %s" % (head[:12], a.expect_sha12))

    # code-base.txt entries against the expected set
    files, trees = parse_code_base(fork / "code-base.txt")
    exp_pairs = [(f, s) for f, s, _ in EXPECTED]
    got_pairs = [(f, s) for f, s, _ in files]
    check(sorted(got_pairs) == sorted(exp_pairs) and len(got_pairs) == len(exp_pairs),
          "code-base.txt lists exactly the 8 expected (fork path, source path) pairs")
    recorded = {(f, s): h for f, s, h in files}

    # each file: fork == source == recorded, and the plan prefix
    for fpath, spath, prefix in EXPECTED:
        rec = recorded.get((fpath, spath))
        fp, sp = fork / fpath, lab / spath
        hf = sha256_file(fp) if fp.is_file() else None
        hs = sha256_file(sp) if sp.is_file() else None
        print("      %s  fork %s  source %s  recorded %s" % (fpath, hf, hs, rec))
        check(hf is not None and hf == hs == rec and rec.startswith(prefix),
              "%s byte-identical to %s, equal to code-base.txt, prefix %s" % (fpath, spath, prefix))

    # snapshot tree hash
    check(len(trees) == 1 and trees[0][0] == TREE, "code-base.txt has one tree line for %s" % TREE)
    if trees:
        tpath, thash, tcount = trees[0]
        got_hash, got_count = tree_hash(lab / tpath)
        print("      tree %s  recomputed %s files=%d  recorded %s files=%d"
              % (tpath, got_hash, got_count, thash, tcount))
        check(got_hash == thash and got_count == tcount, "snapshot tree hash recomputes")

    # tracked files under chang/ and vendor/ are exactly the expected ones
    rc, out, err = git(fork, "ls-files", "--", "chang", "vendor")
    tracked = sorted(out.split())
    check(rc == 0 and tracked == sorted(f for f, _, _ in EXPECTED),
          "tracked files under chang/ and vendor/ are exactly the 8 base files")

    # clean working tree
    rc, out, err = git(fork, "status", "--porcelain")
    print("      git status --porcelain: %d line(s)" % len(out.splitlines()))
    for line in out.splitlines():
        print("      | %s" % line)
    check(rc == 0 and out == "", "git status --porcelain is empty")

    print("")
    if failures:
        print("%s FAIL (%d failed check(s))" % (GATE, len(failures)))
        return 1
    print("%s PASS" % GATE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
