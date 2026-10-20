"""Compare the three STUDY section 13 determinism runs without importing an ML backend."""
import argparse
import json
from pathlib import Path
from nrp.common import read, need, digest, encoded, hexstr


def compare(directories):
    runs = {}
    sources = {}
    for directory in map(Path, directories):
        record = read(directory / 'run.json')
        cfg = record['config']
        need(record['exit'] == 0 and record['end'] == 'completed', 'determinism run unfinished or StopIf early')
        need(cfg['stage'] == 'det' and cfg['seed'] == 1 and cfg['epochs'] == 20, 'not section 13 determinism identity')
        key = (cfg['arm_id'], cfg['replica'] or 0)
        need(key not in runs, 'duplicate determinism identity')
        epochs = [json.loads(x) for x in (directory / 'epochs.jsonl').read_text().splitlines()]
        need([e['epoch'] for e in epochs] == list(range(20)), 'need exactly twenty consecutive epochs')
        for e in epochs:
            hexstr(e['vars_sha256_after']); hexstr(e['vars_sha256_before'])
        # Record equality excludes only identity and measured wall-time fields. float.hex
        # distinguishes -0.0 from +0.0 and avoids tolerance-based comparisons.
        filtered = [{k: v for k, v in e.items() if k != 'run_id' and not k.startswith('sys/')} for e in epochs]
        runs[key] = (record, filtered)
        sources[str(directory)] = {f: digest((directory / f).read_bytes()) for f in ('run.json', 'epochs.jsonl')}
    need(set(runs) == {('CH-8', 1), ('CH-8', 2), ('F000', 0)}, 'two CH-8 replicas and one F000 required')
    reference = runs[('CH-8', 1)][0]
    for rec, _ in runs.values():
        need(rec['config']['init_tensors'] == reference['config']['init_tensors'] and
             rec['code']['head'] == reference['code']['head'] and
             rec['wandb']['provenance']['gpu_product'] == reference['wandb']['provenance']['gpu_product'],
             'unpaired determinism inputs')
    def exact(x):
        if isinstance(x, float):
            return {'float_hex': x.hex()}
        if isinstance(x, dict):
            return {k: exact(v) for k, v in x.items()}
        if isinstance(x, list):
            return [exact(v) for v in x]
        return x
    comparisons = []
    for a, b in [(('CH-8', 1), ('CH-8', 2)), (('F000', 0), ('CH-8', 1)), (('F000', 0), ('CH-8', 2))]:
        ra, ea = runs[a]; rb, eb = runs[b]
        fields = sorted({k for x, y in zip(ea, eb) for k in set(x) | set(y) if exact(x.get(k)) != exact(y.get(k))})
        equal = ra['final_variables_sha256'] == rb['final_variables_sha256'] and not fields
        comparisons.append({'a': list(a), 'b': list(b), 'bitwise_equal': equal, 'differing_epoch_fields': fields,
                            'final_variables_equal': ra['final_variables_sha256'] == rb['final_variables_sha256']})
    return {'schema': 1, 'status': 'engineering record, not a gate', 'comparisons': comparisons,
            'f000_bitwise_equal_ch8': all(c['bitwise_equal'] for c in comparisons[1:]), 'sources_sha256': sources,
            'comparison': 'exact variable hashes and exact JSON scalar values; run_id and sys/* timing excluded'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories', nargs=3)
    a = p.parse_args()
    print(json.dumps(compare(a.directories), indent=1))


if __name__ == '__main__':
    main()
