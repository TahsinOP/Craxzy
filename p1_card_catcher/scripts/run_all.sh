#!/usr/bin/env bash
# Reproduce every milestone in order.  Usage: scripts/run_all.sh [first-milestone, default 0]
# Budgets are sized for a 16-thread laptop (about 2.5 h in total); raise --n / --steps on a server.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="env -u PYTHONPATH .venv/bin/python"
from=${1:-0}
mkdir -p results
[ "$from" -le 0 ] && { $PY scripts/m0_mount.py; $PY scripts/m0_dataset.py 2>&1 | tee results/m0_dataset.log; $PY scripts/m0_tunnel.py 2>&1 | tee results/m0_tunnel.log; }
[ "$from" -le 1 ] && $PY scripts/m1_ekf.py 2>&1 | tee results/m1_ekf.log
[ "$from" -le 2 ] && $PY scripts/m2_residual.py 2>&1 | grep -v "Polishing" | tee results/m2_residual.log
[ "$from" -le 3 ] && $PY scripts/m3_catch.py 2>&1 | grep -v "Polishing" | tee results/m3_catch.log
[ "$from" -le 4 ] && $PY scripts/m4_triggers.py 2>&1 | grep -v "Polishing" | tee results/m4_triggers.log
echo "all done"
