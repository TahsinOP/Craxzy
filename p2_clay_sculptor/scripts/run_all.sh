#!/usr/bin/env bash
# Reproduce every milestone in order.  Usage: scripts/run_all.sh [first-milestone, default 0]
# Milestone 5 is the stretch goals.  Budgets are sized for a 16-thread laptop with no GPU
# (about 6 h in total: 1.5 h of training, 1 h of oracle planning, 2.4 h for M4); on a server raise --episodes, --epochs and the CEM population.
# systemd-inhibit keeps a laptop awake for the duration.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="env -u PYTHONPATH .venv/bin/python -u"
command -v systemd-inhibit >/dev/null && PY="systemd-inhibit --what=idle:sleep --why=P2-clay $PY"
Q='^WARNING|^impratio|^integrator|^$'
from=${1:-0}
mkdir -p results
[ "$from" -le 0 ] && { $PY scripts/m0_bench.py 2>&1 | grep --line-buffered -Ev "$Q" | tee results/m0_bench.log;
                     $PY scripts/m0_discretisation.py 2>&1 | grep --line-buffered -Ev "$Q" | tee results/m0_discretisation.log; }
[ "$from" -le 1 ] && $PY scripts/m1_oracle.py 2>&1 | grep --line-buffered -Ev "$Q" | tee results/m1_oracle.log
if [ "$from" -le 2 ]; then
  $PY scripts/m2_dataset.py --episodes 1100 --steps 6 --name main 2>&1 | grep --line-buffered -Ev "$Q" | tee results/m2_dataset_main.log
  $PY scripts/m2_dataset.py --episodes 48 --steps 6 --seed 11 --name young_lo --set clay.young=0.75e+5 2>&1 | grep --line-buffered -Ev "$Q" | tee results/m2_dataset_shift.log
  $PY scripts/m2_dataset.py --episodes 48 --steps 6 --seed 12 --name young_hi --set clay.young=3.0e+5 2>&1 | grep --line-buffered -Ev "$Q" | tee -a results/m2_dataset_shift.log
  $PY scripts/m2_dataset.py --episodes 48 --steps 6 --seed 13 --name yield_lo --set clay.yield_strain=0.01 2>&1 | grep --line-buffered -Ev "$Q" | tee -a results/m2_dataset_shift.log
  $PY scripts/m2_dataset.py --episodes 48 --steps 6 --seed 14 --name yield_hi --set clay.yield_strain=0.04 2>&1 | grep --line-buffered -Ev "$Q" | tee -a results/m2_dataset_shift.log
  scripts/train_all.sh 6
  $PY scripts/m2_eval.py 2>&1 | grep --line-buffered -Ev "$Q" | tee results/m2_eval.log
fi
[ "$from" -le 3 ] && $PY scripts/m3_mpc.py 2>&1 | grep --line-buffered -Ev "$Q" | tee results/m3_mpc.log
[ "$from" -le 4 ] && $PY scripts/m4_fallback.py 2>&1 | grep --line-buffered -Ev "$Q" | tee results/m4_fallback.log
[ "$from" -le 5 ] && { $PY scripts/stretch.py --planners oracle --nthread 4 2>&1 | grep --line-buffered -Ev "$Q" | tee results/stretch_oracle.log;
                     $PY scripts/stretch.py --planners structured fallback 2>&1 | grep --line-buffered -Ev "$Q" | tee results/stretch.log; }
$PY scripts/make_reports.py
echo "all done (docs: docs/figures/make_figures.py, scripts/docs_videos.py)"
