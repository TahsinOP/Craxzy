#!/usr/bin/env bash
# Train every model of M2 on the CPU: two processes side by side, four threads each.
# Each process peaks near 1.4 GB and runs under a 2 GB cap, so a mistake ends the job and
# not the desktop session.  (Three processes at batch 32 took 2.5 GB each and the kernel
# started killing other programs; see CLAUDE.md.)
# Usage: scripts/train_all.sh [epochs for the graph networks, default 6]
set -uo pipefail
cd "$(dirname "$0")/.."
EP=${1:-6}
RUN="systemd-run --user --scope -q -p MemoryMax=2G -p MemorySwapMax=0 env -u PYTHONPATH MALLOC_ARENA_MAX=2 .venv/bin/python -u scripts/m2_train.py --threads 4 --epochs $EP --skip-existing"
Q='^WARNING|^impratio|^integrator|^$'
mkdir -p results
( $RUN --kinds structured --ensemble-seeds 0 1 2 3; $RUN --kinds unet --seeds 0 ) 2>&1 | grep --line-buffered -Ev "$Q" > results/m2_train_a.log &
( $RUN --kinds gnn --seeds 0 1 2; $RUN --kinds structured --ensemble-seeds 4; $RUN --kinds unet --seeds 1 2 ) 2>&1 | grep --line-buffered -Ev "$Q" > results/m2_train_b.log &
wait
cat results/m2_train_a.log results/m2_train_b.log > results/m2_train.log
rm -f results/m2_train_a.log results/m2_train_b.log results/m2_train_c.log
grep -h "^saved" results/m2_train.log
