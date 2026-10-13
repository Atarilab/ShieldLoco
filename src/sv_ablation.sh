#!/usr/bin/env bash
# Safety-value terminal cost ablation for DoomMPC.
#
# Runs 4 modes: off, heuristic, direct, residual.
#
# Usage:
#   bash sv_ablation.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SCENARIO="eval/scenarios/scenario_a.yaml"
RUNS=5
BASE_SEED=42
OUT="results/sv_ablation"

CHECKPOINT_DIRECT="safety_filter/quad/safety_value_checkpoint/direct.msgpack"
CHECKPOINT_RESIDUAL="safety_filter/quad/safety_value_checkpoint/residual.msgpack"

# ---------------------------------------------------------------------------

mkdir -p "$OUT"

echo "============================================"
echo "Safety-Value Terminal Cost Ablation"
echo "============================================"
echo "Scenario: $SCENARIO"
echo "Runs:     $RUNS"
echo "Output:   $OUT"
echo ""

# 1/4 — No terminal cost
echo "[1/4 off] Running ..."
python3 eval/run_eval.py \
    --scenario "$SCENARIO" --runs "$RUNS" --seed "$BASE_SEED" \
    --sweep "safety_value_mode=off" \
    --out "$OUT/off"

# 2/4 — Heuristic only
echo "[2/4 heuristic] Running ..."
python3 eval/run_eval.py \
    --scenario "$SCENARIO" --runs "$RUNS" --seed "$BASE_SEED" \
    --sweep "safety_value_mode=heuristic" \
    --out "$OUT/heuristic"

# 3/4 — Learned direct
echo "[3/4 direct] Running ..."
python3 eval/run_eval.py \
    --scenario "$SCENARIO" --runs "$RUNS" --seed "$BASE_SEED" \
    --sweep "safety_value_mode=direct safety_value_checkpoint_path=$CHECKPOINT_DIRECT" \
    --out "$OUT/direct"

# 4/4 — Learned residual
echo "[4/4 residual] Running ..."
python3 eval/run_eval.py \
    --scenario "$SCENARIO" --runs "$RUNS" --seed "$BASE_SEED" \
    --sweep "safety_value_mode=residual safety_value_checkpoint_path=$CHECKPOINT_RESIDUAL" \
    --out "$OUT/residual"

echo ""
echo "============================================"
echo "Done. Results in: $OUT/"
echo "============================================"
