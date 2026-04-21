#!/usr/bin/env bash
# DoomMPC eval launcher
#
# Usage (seed-only, one experiment):
#   bash eval.sh
#   bash eval.sh 2>&1 | tee results/go2_mimps/eval.log
#
# Usage (sweep):
#   # Edit SWEEP_DIR and SWEEP below, then run
#   bash eval.sh
#
# Results structure:
#   Seed-only:  results/{NAME}/results.json
#   Sweep:      results/{SWEEP_DIR}/{cond1}/results.json, {cond2}/results.json, ...
#
# To analyze sweeps (each condition is a separate experiment):
#   python3 eval/analyze_results.py results/{SWEEP_DIR}/*/results.json --out results/{SWEEP_DIR}
#
# To analyze seed-only:
#   python3 eval/analyze_results.py results/{NAME}/results.json --out results/{NAME}

set -euo pipefail

SCENARIO="eval/scenarios/scenario_a.yaml"
RUNS=5
BASE_SEED=42

# ---------------------------------------------------------------------------
# Seed-only experiment (edit NAME and OUT, keep sweep commented)
# ---------------------------------------------------------------------------
NAME="go2"
OUT="results/${NAME}"

# ---------------------------------------------------------------------------
# Sweep experiment (uncomment sweep section, edit SWEEP_DIR and sweep axes)
# ---------------------------------------------------------------------------
SWEEP_DIR="results/go2_sweep_contact_policy_latest"    # parent dir for all sweep conditions
# SWEEP="mps_safety_weight=10,100,1000 projection_clearance=0.08 body_sdf_margin=0.1,0.2,0.3"
SWEEP="mps_alg_name=MIMPS,REMIMPS plan_lim=0.25,0.3,0.35 mps_iterations=1,2,3"

# ---------------------------------------------------------------------------

if [[ -n "${SWEEP:-}" ]]; then
    # Sweep mode: each condition gets its own subdirectory
    echo "[eval.sh] Sweep mode: $SWEEP"
    echo "[eval.sh] Output: $SWEEP_DIR/<condition>/results.json"
    echo ""
    python3 eval/run_eval.py \
        --scenario "$SCENARIO" \
        --runs "$RUNS" \
        --seed "$BASE_SEED" \
        --name "$SWEEP_DIR" \
        --sweep $SWEEP \
        --out "$SWEEP_DIR"

    echo ""
    echo "[eval.sh] Analyzing all conditions..."
    python3 eval/analyze_results.py "$SWEEP_DIR"/*/results.json --out "$SWEEP_DIR"
else
    # Seed-only mode
    echo "[eval.sh] Seed-only mode: $NAME"
    echo "[eval.sh] Output: $OUT"
    echo ""
    python3 eval/run_eval.py \
        --scenario "$SCENARIO" \
        --runs "$RUNS" \
        --seed "$BASE_SEED" \
        --name "$NAME" \
        --out "$OUT"

    echo ""
    python3 eval/analyze_results.py "$OUT/results.json" --out "$OUT"
fi

echo "[eval.sh] All done."