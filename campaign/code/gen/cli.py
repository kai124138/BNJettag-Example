"""Command line of the config generator. Run from anywhere:

    python -I -B gen/cli.py generate [--inputs PATH]   write configs/ (replaces the tree)
    python -I -B gen/cli.py check    [--inputs PATH]   regenerate in memory and diff against configs/
    python -I -B gen/cli.py summary  [--inputs PATH]   counts by stage, unset inputs

`check` exits 0 when the committed tree equals the regenerated one byte for byte, else 1, and
prints every missing, extra and different file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORK))   # -I does not add the script directory

from gen import configs as C  # noqa: E402
from gen import inputs as I  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='gen/cli.py')
    ap.add_argument('command', choices=('generate', 'check', 'summary'))
    ap.add_argument('--inputs', default=None, help='decision-input record (default gen/inputs.json)')
    a = ap.parse_args(argv)
    inp = I.load(Path(a.inputs) if a.inputs else None)
    files = C.generate(inp)
    if a.command == 'generate':
        C.write_tree(files)
        print(f'wrote {len(files)} files under {C.CONFIG_DIR}/')
        return 0
    if a.command == 'check':
        d = C.diff_trees(files, C.disk_tree())
        for k in ('missing', 'extra', 'different'):
            for p in d[k]:
                print(f'{k:9s} {p}')
        bad = any(d.values())
        print(f'regenerate-and-diff: {len(files)} files generated; '
              f'{len(d["missing"])} missing, {len(d["extra"])} extra, {len(d["different"])} different')
        return 1 if bad else 0
    man = json.loads(files[f'{C.CONFIG_DIR}/MANIFEST.json'])
    print(json.dumps({'counts': man['counts'], 'n_configs': man['n_configs']}, indent=1))
    unset = sorted({u for e in man['configs'] for u in e['unset_inputs']})
    print('unset inputs present:', unset)
    return 0


if __name__ == '__main__':
    sys.exit(main())
