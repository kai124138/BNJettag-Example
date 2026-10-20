"""G80: namespace-wide kai ConfigMaps must equal those mounted by unfinished kai Jobs."""
import argparse
import json
import subprocess
from nrp.common import need, read


def check(configmaps, jobs, *, bundle_only=True):
    need(configmaps.get('kind') in ('List', 'ConfigMapList') and isinstance(configmaps.get('items'), list),
         'need complete ConfigMap inventory, not a count estimate')
    need(jobs.get('kind') in ('List', 'JobList') and isinstance(jobs.get('items'), list), 'need complete Job inventory')
    need(not configmaps.get('metadata', {}).get('continue') and not jobs.get('metadata', {}).get('continue'),
         'paginated inventory is incomplete')
    actual = {x['metadata']['name'] for x in configmaps['items'] if x['metadata']['name'].startswith('kai-')}
    mounted = set()
    unfinished = []
    for job in jobs['items']:
        m = job['metadata']
        if not (m['name'].startswith('kai-') or m.get('labels', {}).get('user') == 'kai'):
            continue
        terminal = any(c.get('type') in ('Complete', 'Failed') and c.get('status') == 'True'
                       for c in job.get('status', {}).get('conditions', []))
        if terminal and not job.get('status', {}).get('active', 0):
            continue
        unfinished.append(m['name'])  # includes unscheduled/pending and terminating unfinished Jobs
        pod = job['spec']['template']['spec']
        used = {v['name'] for c in pod.get('containers', []) + pod.get('initContainers', [])
                for v in c.get('volumeMounts', [])}
        for volume in pod.get('volumes', []):
            if volume['name'] not in used:
                continue
            cm = volume.get('configMap', {}).get('name')
            if cm and cm.startswith('kai-'):
                mounted.add(cm)
            for source in volume.get('projected', {}).get('sources', []):
                cm = source.get('configMap', {}).get('name')
                if cm and cm.startswith('kai-'):
                    mounted.add(cm)
    result = {'configmap_count': len(actual), 'mounted_count': len(mounted),
              'configmaps': sorted(actual), 'mounted': sorted(mounted), 'unfinished_jobs': sorted(unfinished)}
    need(actual == mounted, 'G80 mismatch: ' + json.dumps(result, sort_keys=True))
    need(not bundle_only or not actual, 'bundle-mode wave requires both counts zero; cleanup by cluster owner')
    return dict(result, passed=True, bundle_only=bundle_only)


def live(kubectl='kubectl'):
    """Only metadata is requested for ConfigMaps; no data contents are read or stored."""
    base = [kubectl, '--request-timeout=30s', '-n', 'cms-ml']
    cm = subprocess.run(base + ['get', 'configmaps', '-o', 'name'], capture_output=True, text=True, check=True)
    jobs = subprocess.run(base + ['get', 'jobs', '-o', 'json'], capture_output=True, text=True, check=True)
    inventory = {'kind': 'ConfigMapList', 'items': [{'metadata': {'name': x.split('/', 1)[1]}}
                 for x in cm.stdout.splitlines() if x.strip()]}
    return check(inventory, json.loads(jobs.stdout))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configmaps', required=True)
    p.add_argument('--jobs', required=True)
    p.add_argument('--allow-mounted-legacy', action='store_true')
    a = p.parse_args(argv)
    print(json.dumps(check(read(a.configmaps), read(a.jobs), bundle_only=not a.allow_mounted_legacy), indent=1))


if __name__ == '__main__':
    main()
