#!/usr/bin/env bash
# G72 STAND-IN for run_case.sh (dry run only). Mirrors the frozen run_case.sh:4-17 line by line
# and calls the stand-ins, `timeout` included (the stand-in in $HOME/stub-tools moves the build
# into a new process group, as GNU timeout does). Planted faults are read from
# $HOME/stub-faults/<line> and fire once (the file is renamed to <line>.fired).
# usage: run_case.sh <shape> <g> <inp> <path> <bias:-|bias>
set -uo pipefail
source "$HOME/scratch/cbl1016-da4ml-test/scripts/env.sh"
shape=$1; g=$2; inp=$3; path=$4; bias=$5
name="${shape}_${g}_${inp}_${path}"; extra=""
[ "$bias" = "bias" ] && { name="${name}_bias"; extra="--bias"; }
P="$SCR/proj/main/$name"
FD="$HOME/stub-faults"; A="$FD/active/$name"; M="$HOME/stub-markers"; mkdir -p "$FD/active" "$M"; rm -f "$A"
[ -f "$FD/$name" ] && { cp "$FD/$name" "$A"; mv "$FD/$name" "$FD/$name.fired"; }
has() { [ -f "$A" ] && grep -qE "^$1( |$)" "$A"; }
arg() { [ -f "$A" ] && sed -n "s/^$1 //p" "$A" | head -n 1; }
touch "$M/$name.run_case.started"
has no_stdout && exec >/dev/null
has skip && { mkdir -p "$P"; echo '{"stand_in": "written by a planted foreign job"}' > "$P/synth_result.json"; }
has nolog && mv "$SCR/logs" "$SCR/logs.moved-by-stub-fault"
fd=$(arg foreign_dir); [ -n "$fd" ] && mkdir -p "$SCR/proj/main/$fd"
has tamper_envsh && printf '# changed by a planted fault\n' >> "$SCR/scripts/env.sh"
has tamper_common && printf '# changed by a planted fault\n' >> "$SCR/scripts/common.py"
has da4ml_change && "$PY" -I -B -c 'import glob, os, sysconfig
for m in glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "da4ml-*.dist-info", "METADATA")):
    t = open(m).read().splitlines(); open(m, "w").write("\n".join("Version: 9.9.9" if l.startswith("Version: ") else l for l in t) + "\n")'
[ -e "$P/synth_result.json" ] && { echo "skip $name"; exit 0; }
mkdir -p "$SCR/proj/main"
timeout 1800 "$PY" "$SCR/scripts/stage_b_build.py" --shape "$shape" --g "$g" --inp "$inp" --path "$path" --seed 0 $extra --out "$P" > "$SCR/logs/build_$name.log" 2>&1
rc=$?; echo "$name build rc=$rc"
[ $rc -ne 0 ] && exit $rc
bash "$SCR/scripts/synth_one.sh" "$P"; rc=$?; echo "$name synth rc=$rc"
"$PY" "$SCR/scripts/parse_reports.py" "$P" >> "$SCR/logs/parse.log" 2>&1
has survivor_after_return && { ( trap '' TERM; exec sleep 600 ) & echo $! > "$M/$name.survivor.pid"; }
exit $rc
