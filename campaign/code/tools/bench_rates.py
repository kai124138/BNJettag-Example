"""Read all-inclusive per-class rates; no training imports, physics metrics or cluster calls."""
import argparse
import json
from pathlib import Path
import statistics

from nrp.common import read, need, positive, digest


def rates(directory, lifecycle):
    """lifecycle: captured GPU allocation from Pod start through final container termination.

    A capture is required: process wall alone omits init/queue-with-GPU time. No 1.10
    overhead multiplier is applied (STUDY 10). Host full-horizon projection remains a
    separately reviewed gate. A short utilization record is provisional, never certified.
    """
    root = Path(directory)
    pack = read(root / 'pack.json')
    need(pack['exit'] == 0 and not pack['skipped_complete'], 'benchmark must be successful and fresh')
    need(lifecycle['job'] == pack['job'] and lifecycle['pod'] == pack['pod'], 'lifecycle identity mismatch')
    need(lifecycle.get('source') and lifecycle.get('complete') is True, 'complete lifecycle capture with source required')
    gpu_s = positive(lifecycle['gpu_seconds'], 'pod GPU-seconds')
    need(gpu_s >= pack['wall_seconds'], 'pod lifetime shorter than pack process')
    rows = [read(root / name) for name in pack['telemetry']]
    need(rows and len(rows) == pack['k'], 'missing per-arm telemetry')
    for row in rows:
        need(row['exit'] == 0 and not row['resume_requested'], 'failed/resumed benchmark cannot certify rate')
        need(row['source_bundle'] == pack['source_bundle'] and row['gpu_product'] == pack['gpu_product'],
             'mixed provenance')
        positive(row['epochs_completed_this_process'], 'completed epochs', True)
        positive(row['allocated_gpu_seconds'], 'allocation')
        positive(row['gpu_peak_mib_sampled'], 'sampled GPU memory')
        positive(row['host_peak_mib_sampled'], 'sampled host memory')
    total = sum(r['allocated_gpu_seconds'] for r in rows)
    need(abs(total - pack['wall_seconds']) < 0.01, 'allocation conservation failed')
    by_class = {}
    for row in rows:
        c = by_class.setdefault(row['arm_class'], {'gpu_seconds': 0.0, 'run_epochs': 0})
        c['gpu_seconds'] += row['allocated_gpu_seconds'] * gpu_s / total
        c['run_epochs'] += row['epochs_completed_this_process']
    for row in by_class.values():
        row['gpu_seconds_per_run_epoch'] = row['gpu_seconds'] / row['run_epochs']
    samples = [json.loads(x) for x in (root / 'gpu.jsonl').read_text().splitlines()]
    valid = [s for s in samples if 'utilization_pct' in s]
    need(valid and len(valid) == len(samples), 'missing GPU sample')
    need(all(s['product'] == pack['gpu_product'] and s['total_mib'] > 0 for s in valid), 'GPU sample identity')
    # Each full trailing three-hour window, interval-weighted, gaps > 90 s refuse certification.
    rolling = []
    for end in valid:
        start = end['elapsed_s'] - 10800
        if start < valid[0]['elapsed_s']:
            continue
        weighted = coverage = 0.0
        for a, b in zip(valid, valid[1:]):
            dt = min(end['elapsed_s'], b['elapsed_s']) - max(start, a['elapsed_s'])
            if dt > 0:
                need(0 < b['elapsed_s'] - a['elapsed_s'] <= 90, 'telemetry sampling gap')
                weighted += dt * a['utilization_pct']
                coverage += dt
        need(abs(coverage - 10800) < 0.01, 'incomplete rolling window')
        rolling.append(weighted / coverage)
    peak_fraction = max(s['used_mib'] / s['total_mib'] for s in valid)
    return {'schema': 1, 'status': 'measured', 'gpu_product': pack['gpu_product'], 'k': pack['k'],
            'code_sha': pack['code_sha'], 'source_bundle': pack['source_bundle'], 'classes': by_class,
            'all_inclusive_gpu_seconds': gpu_s, 'lifecycle': lifecycle,
            'device_peak_fraction_sampled': peak_fraction, 'device_memory_pass': peak_fraction <= 0.90,
            'rolling_three_hour_min_utilization_pct': min(rolling) if rolling else None,
            'utilization_pass': bool(rolling) and min(rolling) >= 40,
            'host_projection_pass': False, 'production_certified': False,
            'limitations': ['Sampled peaks are lower bounds.', 'Host full-horizon projection and product gates require B19.',
                            'No production certification from a short benchmark.'],
            'sources_sha256': {p.name: digest(p.read_bytes()) for p in
                [root / 'pack.json', root / 'gpu.jsonl'] + [root / n for n in pack['telemetry']]}}


def project(configs, measured):
    """Per-class all-inclusive estimate; never reduce epochs/seeds to fit."""
    hours = 0.0
    for cfg in configs:
        need(cfg['epochs'] == 7000, 'production horizon must stay 7000')
        r = measured['classes'][cfg['arm_class']]['gpu_seconds_per_run_epoch']
        hours += 7000 * positive(r, 'measured rate') / 3600
    return {'runs': len(configs), 'training_gpu_h': hours, 'with_reserve_gpu_h': hours + 50,
            'fits_400': hours + 50 <= 400, 'formula': 'sum(7000*r_m)/3600 + 50; no planning 1.10 factor'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory'); p.add_argument('--lifecycle', required=True)
    a = p.parse_args(argv)
    print(json.dumps(rates(a.directory, read(a.lifecycle)), indent=1, allow_nan=False))


if __name__ == '__main__':
    main()
