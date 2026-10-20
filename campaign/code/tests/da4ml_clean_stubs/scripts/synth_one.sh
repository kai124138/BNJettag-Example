#!/usr/bin/env bash
# G72 STAND-IN for synth_one.sh (dry run only). Mirrors the frozen synth_one.sh:3-14; the Xilinx
# settings files and tools, `timeout` included, are stand-ins under $HOME/stub-xilinx and
# $HOME/stub-tools; `readlink -f` is replaced by a portable form, and dates are UTC.
set -uo pipefail
P=$(cd "$1" && pwd -P)
source "$HOME/scratch/cbl1016-da4ml-test/scripts/env.sh"
cd "$P"
echo "csynth start $(date -u +%Y-%m-%dT%H:%M:%S+00:00) host $(hostname)" >> STATUS
( source "$HOME/stub-xilinx/Vitis/2023.2/settings64.sh" && timeout --signal=TERM 7200 vitis_hls -f build_prj.tcl > csynth_stdout.log 2>&1 ); rc=$?
echo "csynth rc=$rc $(date -u +%Y-%m-%dT%H:%M:%S+00:00)" >> STATUS
[ $rc -ne 0 ] && exit $rc
mkdir -p vivado && cd vivado
( source "$HOME/stub-xilinx/Vivado/2023.2/settings64.sh" && timeout --signal=TERM 7200 vivado -mode batch -nojournal -log vivado.log -source "$SCR/scripts/vivado_ooc.tcl" -tclargs "$P/myproject_prj/solution1/syn/verilog" "$SCR/scripts/clock.xdc" > vivado_stdout.log 2>&1 ); rc=$?
echo "vivado rc=$rc $(date -u +%Y-%m-%dT%H:%M:%S+00:00)" >> ../STATUS
exit $rc
