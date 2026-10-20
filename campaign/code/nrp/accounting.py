"""M20 live allocation snapshots; equal shares, not kernel utilization or final pod billing."""
import math
import time
from pathlib import Path

from nrp.common import atom, read, need


def prior_seconds(run_dir):
    """One durable latest snapshot per invocation; never rewind resource usage with epochs."""
    records = [read(p) for p in (Path(run_dir) / 'allocations').glob('*.json')]
    return max((r['prior_seconds'] + r['seconds'] for r in records), default=0.0)


def publish(path, *, seconds, prior, live, now=None):
    need(live >= 1 and seconds >= 0 and prior >= 0, 'invalid allocation snapshot')
    rec = {'seconds': seconds, 'prior_seconds': prior, 'live': live,
           'monotonic': time.monotonic() if now is None else now,
           'basis': 'supervisor wall time shared among live children; pod init/final shutdown reconciled separately'}
    atom(path, rec)
    return rec


def hours(path, *, now=None):
    r = read(path)
    elapsed = (time.monotonic() if now is None else now) - r['monotonic']
    need(elapsed >= 0 and r['live'] >= 1 and all(math.isfinite(r[k]) and r[k] >= 0
         for k in ('seconds', 'prior_seconds')), 'invalid live allocation snapshot')
    return (r['prior_seconds'] + r['seconds'] + elapsed / r['live']) / 3600
