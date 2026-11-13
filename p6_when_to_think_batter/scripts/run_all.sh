#!/usr/bin/env bash
# Reproduce every milestone, or start from milestone N:  scripts/run_all.sh [N]
# About 1.5 h on a 28-thread desktop.  Logs go to results/*.log.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="env -u PYTHONPATH .venv/bin/python -u"
start=${1:-0}
mkdir -p results
[ "$start" -le 0 ] && $PY scripts/m0_cage.py        2>&1 | tee results/m0.log
[ "$start" -le 1 ] && $PY scripts/m1_hitter.py      2>&1 | tee results/m1.log
[ "$start" -le 2 ] && $PY scripts/m2_worldmodel.py  2>&1 | tee results/m2.log
[ "$start" -le 3 ] && $PY scripts/m3_triggers.py    2>&1 | tee results/m3.log
[ "$start" -le 4 ] && $PY scripts/m4_surprises.py   2>&1 | tee results/m4.log
[ "$start" -le 5 ] && $PY scripts/m5_events.py      2>&1 | tee results/m5.log
[ "$start" -le 6 ] && $PY scripts/rally.py --fresh  2>&1 | tee results/rally.log
$PY scripts/make_reports.py
