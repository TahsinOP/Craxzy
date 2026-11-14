#!/usr/bin/env bash
# The closed-loop experiments with ten planner seeds (configs/default.yaml), then reports and figures.
# The three-seed results of the first pass are kept in results/three_seeds/.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="env -u PYTHONPATH .venv/bin/python -u"
CAP="systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=0"
Q='^WARNING|^impratio|^integrator|^$|ConstantInputWarning|spearmanr\(x, gain\)'
run() { local log=$1; shift; echo "[$(date +%T)] $*"; "$@" 2>&1 | grep --line-buffered -Ev "$Q" > "results/$log"; }
# Usage: scripts/ten_seeds.sh          everything
#        scripts/ten_seeds.sh resume   only M4, keeping its finished configurations, then reports and figures
if [ "${1:-}" != resume ]; then
  run m1_oracle.log $CAP $PY scripts/m1_oracle.py
  run m3_mpc.log $CAP $PY scripts/m3_mpc.py
  run m4_fallback.log $CAP $PY scripts/m4_fallback.py
else
  cp results/m4_fallback.log results/m4_fallback_before_resume.log 2>/dev/null
  run m4_fallback.log $CAP $PY scripts/m4_fallback.py --resume
fi
run make_reports.log $PY scripts/make_reports.py
run make_figures.log $PY docs/figures/make_figures.py
echo "[$(date +%T)] ten_seeds done"
