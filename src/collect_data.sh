#!/usr/bin/env bash

set -euo pipefail

echo "[ ] Collecting data..."

BASE_SEED=420000
NUM_SEEDS=5

for i in $(seq 1 $NUM_SEEDS); do
    python3 eval/launch_eval.py --scenario eval/scenarios/scenario_a.yaml --seed $((BASE_SEED + i))
done

echo "[ ] All done."