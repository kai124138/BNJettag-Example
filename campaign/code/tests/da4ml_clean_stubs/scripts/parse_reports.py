"""G72 STAND-IN for parse_reports.py (dry run only).

Same command line (parse_reports.py <projdir> ...), the same three report functions with the same
arguments (csynth(projdir), util(file), timing(file)) and the same synth_result.json keys as the
frozen parser's interface, written for the stand-in reports (Vivado-style tables and a
csynth.xml with the element paths the frozen parser reads).
"""
import json
import os
import re
import sys
import xml.etree.ElementTree as ET


def csynth(p):
    f = os.path.join(p, 'myproject_prj', 'solution1', 'syn', 'report', 'csynth.xml')
    if not os.path.exists(f):
        return None
    r = ET.parse(f).getroot()

    def t(path):
        e = r.find(path)
        return None if e is None else e.text

    return {
        'stage': 'C-synthesis estimate (STAND-IN)',
        'LUT': int(t('AreaEstimates/Resources/LUT')),
        'est_clock_ns': float(t('PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod')),
        'latency_worst': int(t('PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency')),
        'II_min': t('PerformanceEstimates/SummaryOfOverallLatency/Interval-min'),
        'II_max': t('PerformanceEstimates/SummaryOfOverallLatency/Interval-max'),
        'part': t('UserAssignments/Part'),
    }


def util(f):
    if not os.path.exists(f):
        return None
    txt = open(f).read()
    m = re.search(r'^\|\s*CLB LUTs\*?\s*\|\s*([\d.]+)\s*\|', txt, re.M)
    return {'LUT': int(float(m.group(1))) if m else None}


def timing(f):
    if not os.path.exists(f):
        return None
    txt = open(f).read()
    m = re.search(r'WNS\(ns\)\s+TNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)', txt, re.S)
    if not m:
        return {'WNS_ns': None}
    return {'WNS_ns': float(m.group(1)), 'TNS_ns': float(m.group(2)), 'failing_endpoints': int(m.group(3))}


def main():
    for p in sys.argv[1:]:
        info = json.load(open(os.path.join(p, 'case_info.json')))
        v = os.path.join(p, 'vivado')
        status = open(os.path.join(p, 'STATUS')).read() if os.path.exists(os.path.join(p, 'STATUS')) else ''
        out = {
            'case': info['case'], 'csim_mismatch': info['csim_mismatch'], 'n_values': info['n_values'],
            'da_kernel_cost_total': info['da_kernel_cost_total'],
            'csynth': csynth(p),
            'vivado_post_opt': {'stage': 'STAND-IN', **(util(os.path.join(v, 'post_opt_util.rpt')) or {}),
                                **(timing(os.path.join(v, 'post_opt_timing.rpt')) or {})},
            'status': status.strip().splitlines(),
        }
        json.dump(out, open(os.path.join(p, 'synth_result.json'), 'w'), indent=1)
        print(os.path.basename(p.rstrip('/')), 'stand-in parsed')


if __name__ == '__main__':
    main()
