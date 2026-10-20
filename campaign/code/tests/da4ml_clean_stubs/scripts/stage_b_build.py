"""G72 STAND-IN for stage_b_build.py (dry run only).

Same command line as the frozen script (--shape --g --inp --path --seed [--bias] --out) and the
same interface outputs: the project directory, build_prj.tcl, case_info.json with the keys
parse_reports.py and the table read, and one JSON line on stdout. No model, no hls4ml: every
value is synthetic. Planted faults (one per line, from $HOME/stub-faults/active/<line>):
build_fail, csim_mismatch, hang_build <seconds>.
"""
import argparse
import json
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def faults(name):
    p = os.path.join(os.environ['HOME'], 'stub-faults', 'active', name)
    out = {}
    if os.path.isfile(p):
        for line in open(p):
            f = line.split()
            if f:
                out[f[0]] = f[1:]
    return out


def marker(name, what):
    d = os.path.join(os.environ['HOME'], 'stub-markers')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, '%s.%s' % (name, what)), 'w') as f:
        f.write(str(os.getpid()))


def versions():
    import importlib.metadata as md
    out = {}
    for d in ('hls4ml', 'keras', 'hgq2', 'da4ml'):
        try:
            out[d] = md.version(d)
        except Exception:  # noqa: BLE001
            out[d] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', required=True)
    ap.add_argument('--g', required=True)
    ap.add_argument('--inp', required=True)
    ap.add_argument('--path', required=True, choices=['stock', 'fact', 'hgq4'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bias', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    name = os.path.basename(a.out.rstrip('/'))
    F = faults(name)
    marker(name, 'build.started')
    if 'hang_build' in F:
        marker(name, 'build.hanging')
        time.sleep(float(F['hang_build'][0]))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'build_prj.tcl'), 'w') as f:
        f.write('# G72 stand-in project file\n')
    if 'build_fail' in F:
        print('stand-in build failure (planted fault)', file=sys.stderr)
        sys.exit(1)
    n_in, n_out = C.SHAPES[a.shape]
    syn = zlib.crc32(name.encode()) % 1000
    info = {
        'stand_in': 'G72 dry run; synthetic values',
        'case': {'shape': a.shape, 'g': a.g, 'inp': a.inp, 'path': a.path, 'seed': a.seed, 'bias': a.bias},
        'versions': versions(), 'n_test': 4096, 'n_values': 4096 * n_out,
        'csim_mismatch': 7 if 'csim_mismatch' in F else 0, 'csim_max_abs_diff': 0.0,
        'da_kernel_cost_total': float(900000 + syn),
        'layers': [{'name': 'cmvm', 'class': 'StandIn', 'da_kernel_cost': float(900000 + syn)}],
    }
    with open(os.path.join(a.out, 'case_info.json'), 'w') as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k: info[k] for k in ('case', 'csim_mismatch', 'da_kernel_cost_total')}))


if __name__ == '__main__':
    main()
