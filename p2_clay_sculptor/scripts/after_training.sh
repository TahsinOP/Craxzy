#!/usr/bin/env bash
# Everything that follows training, in order, each step logged.  Safe to rerun.
# Usage: scripts/after_training.sh            (waits for scripts/train_all.sh if it is still running)
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "[s]cripts/m2_train.py" >/dev/null; do sleep 20; done
PY="env -u PYTHONPATH .venv/bin/python -u"
CAP="systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0"
Q='^WARNING|^impratio|^integrator|^$'
run() { local log=$1; shift; echo "[$(date +%T)] $*"; "$@" 2>&1 | grep --line-buffered -Ev "$Q" > "results/$log"; }
run m2_eval.log $CAP $PY scripts/m2_eval.py
run m3_mpc.log $CAP $PY scripts/m3_mpc.py
run m4_fallback.log $CAP $PY scripts/m4_fallback.py
run stretch.log $CAP $PY scripts/stretch.py --planners structured fallback
run make_reports.log $PY scripts/make_reports.py
run make_figures.log $PY docs/figures/make_figures.py
run docs_videos.log $PY scripts/docs_videos.py
run make_figures.log $PY docs/figures/make_figures.py
echo "[$(date +%T)] after_training done"
