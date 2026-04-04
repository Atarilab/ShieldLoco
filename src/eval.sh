#!/usr/bin/env bash
# DoomMPC paper sweep
# Axes: mps_safety_weight × obstacle_cost_weight × mps_temperature
# Seeds: --runs N with --seed BASE increments seed per run within each condition
#
# Total conditions = |safety_weight| × |obstacle_cost_weight| × |temperature|
#   3 × 3 × 3 = 27 conditions × 3 runs = 81 runs
# Estimate: ~7-9 min/run (JIT + episode) → ~10 hrs.
# Shrink the grids below to reduce wall time.
#
# Usage (from src/):
#   bash eval.sh
#   bash eval.sh 2>&1 | tee results/sweep_run.log

set -euo pipefail

SCENARIO="eval/scenarios/scenario_a.yaml"
OUT="results/sweep_$(date +%Y%m%d_%H%M%S)"
RUNS=3        # runs per condition (seeds: BASE_SEED, BASE_SEED+1, ...)
BASE_SEED=42

# ---------------------------------------------------------------------------
# Sweep grid  (edit values here; cartesian product is computed automatically)
# ---------------------------------------------------------------------------
SAFETY_WEIGHTS="1000,10000"
OBSTACLE_WEIGHTS="500,1000,5000"
TEMPERATURES="0.01,0.1"
# ---------------------------------------------------------------------------

N_SAFETY=$(echo "$SAFETY_WEIGHTS"  | tr ',' '\n' | wc -l)
N_OBS=$(   echo "$OBSTACLE_WEIGHTS" | tr ',' '\n' | wc -l)
N_TEMP=$(  echo "$TEMPERATURES"     | tr ',' '\n' | wc -l)
N_COND=$(( N_SAFETY * N_OBS * N_TEMP ))
N_TOTAL=$(( N_COND * RUNS ))

echo "[eval.sh] Conditions : $N_COND  (${N_SAFETY}×${N_OBS}×${N_TEMP})"
echo "[eval.sh] Runs/cond  : $RUNS  (seeds $BASE_SEED – $(( BASE_SEED + N_TOTAL - 1 )))"
echo "[eval.sh] Total runs : $N_TOTAL"
echo "[eval.sh] Output dir : $OUT"
echo ""

python3 eval/run_eval.py \
    --scenario "$SCENARIO" \
    --runs     "$RUNS" \
    --seed     "$BASE_SEED" \
    --sweep \
        "mps_safety_weight=$SAFETY_WEIGHTS" \
        "obstacle_cost_weight=$OBSTACLE_WEIGHTS" \
        "mps_temperature=$TEMPERATURES" \
    --out "$OUT"

python3 eval/analyze_results.py "$OUT/results.json" --out "$OUT"
# python3 eval/analyze_results.py results/sweep_*/results.json
# python3 eval/analyze_results.py results/sweep_*/results.json --out results/paper_figs/

echo "[eval.sh] All done. Results: $OUT/results.json"
