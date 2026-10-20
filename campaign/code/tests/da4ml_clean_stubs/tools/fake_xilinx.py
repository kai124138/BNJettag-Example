"""G72 STAND-IN for vitis_hls and vivado (dry run only): writes synthetic reports in the formats
the stand-in parser reads. Planted faults (from $HOME/stub-faults/active/<line>): csynth_fail,
hang_csynth <s>, reports_before_hang, survivor (a child that ignores SIGTERM, in the line's
process group), vivado_fail, vivado_noreport, hang_vivado <s>. Values are obviously synthetic
(a LUT count above nine million, a WNS of 99 ns at a 2.5 ns clock)."""
import os
import subprocess
import sys
import time
import zlib


def faults(name):
    p = os.path.join(os.environ['HOME'], 'stub-faults', 'active', name)
    out = {}
    if os.path.isfile(p):
        for line in open(p):
            f = line.split()
            if f:
                out[f[0]] = f[1:]
    return out


def marker(name, what, text=None):
    d = os.path.join(os.environ['HOME'], 'stub-markers')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, '%s.%s' % (name, what)), 'w') as f:
        f.write(text if text is not None else str(os.getpid()))


def write_csynth(proj, syn):
    d = os.path.join(proj, 'myproject_prj', 'solution1', 'syn', 'report')
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(proj, 'myproject_prj', 'solution1', 'syn', 'verilog'), exist_ok=True)
    xml = ('<profile><ReportVersion><Version>STAND-IN</Version></ReportVersion>'
           '<UserAssignments><Part>xcvu13p-flga2577-2-e</Part><TargetClockPeriod>2.50</TargetClockPeriod></UserAssignments>'
           '<PerformanceEstimates><SummaryOfTimingAnalysis><EstimatedClockPeriod>9.990</EstimatedClockPeriod>'
           '</SummaryOfTimingAnalysis><SummaryOfOverallLatency><Best-caseLatency>3</Best-caseLatency>'
           '<Worst-caseLatency>3</Worst-caseLatency><Interval-min>1</Interval-min><Interval-max>1</Interval-max>'
           '</SummaryOfOverallLatency></PerformanceEstimates><AreaEstimates><Resources><LUT>%d</LUT>'
           '</Resources></AreaEstimates></profile>\n' % (8000000 + syn))
    with open(os.path.join(d, 'csynth.xml'), 'w') as f:
        f.write(xml)


def write_vivado(vdir, syn, opt_util=True):
    os.makedirs(vdir, exist_ok=True)
    util = ('Utilization Design Information (STAND-IN)\n'
            '+----------------------------+---------+-------+------------+-----------+-------+\n'
            '|          Site Type         |   Used  | Fixed | Prohibited | Available | Util%% |\n'
            '+----------------------------+---------+-------+------------+-----------+-------+\n'
            '| CLB LUTs*                  | %7d |     0 |          0 |    230400 |  0.00 |\n'
            '+----------------------------+---------+-------+------------+-----------+-------+\n' % (9000000 + syn))
    tim = ('Design Timing Summary (STAND-IN)\n'
           '    WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints\n'
           '    -------      -------  ---------------------  -------------------\n'
           '     99.000        0.000                      0                  %d\n' % (100 + syn))
    for stage in ('post_synth', 'post_opt'):
        if stage == 'post_opt' and not opt_util:
            continue
        with open(os.path.join(vdir, '%s_util.rpt' % stage), 'w') as f:
            f.write(util)
    for stage in ('post_synth', 'post_opt'):
        with open(os.path.join(vdir, '%s_timing.rpt' % stage), 'w') as f:
            f.write(tim)
    with open(os.path.join(vdir, 'vivado.log'), 'w') as f:
        f.write('STAND-IN vivado log\nOOC_DONE\n')


def main():
    tool = sys.argv[1]
    cwd = os.getcwd()
    proj = cwd if tool == 'vitis_hls' else os.path.dirname(cwd)
    name = os.path.basename(proj)
    F = faults(name)
    syn = zlib.crc32(name.encode()) % 1000
    marker(name, '%s.started' % tool)
    if tool == 'vitis_hls':
        if 'survivor' in F:
            child = subprocess.Popen(['bash', '-c', "trap '' TERM; exec sleep 600"])
            marker(name, 'survivor.pid', str(child.pid))
        if 'reports_before_hang' in F:
            write_csynth(proj, syn)
            write_vivado(os.path.join(proj, 'vivado'), syn)
        if 'hang_csynth' in F:
            marker(name, 'csynth.hanging')
            time.sleep(float(F['hang_csynth'][0]))
        if 'csynth_fail' in F:
            print('stand-in csynth failure (planted fault)')
            return 1
        write_csynth(proj, syn)
        print('stand-in csynth done')
        return 0
    if 'hang_vivado' in F:
        marker(name, 'vivado.hanging')
        time.sleep(float(F['hang_vivado'][0]))
    if 'vivado_fail' in F:
        print('stand-in vivado failure (planted fault)')
        return 1
    write_vivado(cwd, syn, opt_util='vivado_noreport' not in F)
    print('OOC_DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
