#!/usr/bin/env bash
# Wait for the running M2 job, then run M3 and M4 (card and ball in parallel) and write the reports.
cd "$(dirname "$0")/.."
PY="env -u PYTHONPATH .venv/bin/python"
while pgrep -f "scripts/m2_residual.py" > /dev/null; do sleep 20; done
$PY scripts/m3_catch.py 2>&1 | grep -v Polishing > results/m3_catch.log
$PY scripts/m4_triggers.py --objects card 2>&1 | grep -v Polishing > results/m4_card.log &
$PY scripts/m4_triggers.py --objects ball 2>&1 | grep -v Polishing > results/m4_ball.log &
wait
$PY scripts/m4_triggers.py --plot-only > results/m4_plot.log 2>&1
$PY scripts/make_reports.py > results/make_reports.log 2>&1
echo done > results/pipeline_done
