#!/usr/bin/env bash
# Reduced-order baseline ablation for the rebuttal.
#
# Compares the safety-filter baselines against each other and the unfiltered
# nominal over the LIPM fix (clearance 0.35 m, swing-feet control):
#
#   1. nominal   : no filter (enabled=false)                      -> tracking only
#   2. CBF-unicycle : DCBF, kinematic centroid model
#   3. CBF-lip      : DCBF with 3D-LIPM + swing-feet centroid      (the fixed baseline)
#   4. CBF-lip-band : CBF-lip with transition-band override        (smooth blend)
#   5. HJ-unicycle  : HJ reach-avoid value + kinematic model
#   6. HJ-lip       : HJ value function with 3D-LIPM dynamics      (the fixed baseline)
#   7. HJ-lip-band  : HJ-lip with transition-band override          (smooth blend)
#
# Reduced-order LIPM + sampling optimizer (the proposed method's MPPI/CEM/Replica
# optimizers driven by the LIPM model instead of full-order MuJoCo):
#
#   8.  LIPM-mppi      : LIPM dynamics + MPPI update
#   9.  LIPM-cem       : LIPM dynamics + CEM update
#   10. LIPM-replica   : LIPM dynamics + replica-exchange MPPI
#
# All baselines run WITHOUT the geometric-projection fallback
# (use_geometric_fallback=false): projection is part of the main method,
# not a baseline, and would contaminate the comparison.
#
# Usage (run from src/ inside the container):
#   bash baselines_ablation.sh
#   RUNS=5 bash baselines_ablation.sh        # fewer runs
#   SCENARIOS="scenario_a" bash baselines_ablation.sh   # scenario subset
#   FORCE=1 bash baselines_ablation.sh       # rerun everything, ignore skip markers
#
# Skip logic: a condition that finished with the same run count is not run again.
# A completion marker is written only after its run_eval invocation succeeds, so
# interrupted/partial runs are automatically redone.
#
# Results:
#   $OUT/<scenario>/<condition>/results.json
#   (one results.json per condition incl. all runs)

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIG (override via env vars)
# ---------------------------------------------------------------------------
RUNS="${RUNS:-10}"
BASE_SEED="${BASE_SEED:-0}"
OUT="${OUT:-results/baselines_ablation}"
SCENARIOS="${SCENARIOS:-scenario_a}"
SCENARIO_DIR="eval/scenarios"

GD_CBF="CBF.use_geometric_fallback=false"
GD_HJ="HJ.use_geometric_fallback=false"
GD_LIPM="LIPM.use_geometric_fallback=false"
BAND_CBF="CBF.use_band=true CBF.band_outer=0.8 CBF.band_inner=0.25"
BAND_HJ="HJ.use_band=true HJ.band_outer=0.8 HJ.band_inner=0.25"
CBF_COMMON="mps_alg_name=CBF CBF.enabled=true
            CBF.clearance=0.35 CBF.max_correction=0.20"
HJ_COMMON="mps_alg_name=HJ HJ.enabled=true
           HJ.clearance=0.35 HJ.max_correction=0.20"
# Reduced-order LIPM uses the same clearance/cost surface as CBF/HJ.
LIPM_COMMON="mps_alg_name=LIPM LIPM.enabled=true LIPM.dynamics_model=lip
             LIPM.clearance=0.35 ${GD_LIPM}"

# <label>|<sweep overrides>  (each becomes its own run_eval invocation)
# NOTE: use_geometric_fallback is read from the nested <ALG>. section only;
# projection is part of the main method, not a baseline, so it stays off.
CONDITIONS=(
  "nominal|mps_alg_name=CBF CBF.enabled=false ${GD_CBF}"
  "cbf_unicycle|$(printf "%s" "$CBF_COMMON") CBF.dynamics_model=unicycle ${GD_CBF}"
  "cbf_lip|$(printf "%s" "$CBF_COMMON") CBF.dynamics_model=lip ${GD_CBF}"
  "cbf_lip_band|$(printf "%s" "$CBF_COMMON") CBF.dynamics_model=lip ${GD_CBF} ${BAND_CBF}"
  "hj_unicycle|$(printf "%s" "$HJ_COMMON") HJ.dynamics_model=unicycle ${GD_HJ}"
  "hj_lip|$(printf "%s" "$HJ_COMMON") HJ.dynamics_model=lip ${GD_HJ}"
  "hj_lip_band|$(printf "%s" "$HJ_COMMON") HJ.dynamics_model=lip ${GD_HJ} ${BAND_HJ}"
  "lipm_mppi|$(printf "%s" "$LIPM_COMMON") LIPM.optimizer=mppi"
  "lipm_cem|$(printf "%s" "$LIPM_COMMON") LIPM.optimizer=cem"
  "lipm_replica|$(printf "%s" "$LIPM_COMMON") LIPM.optimizer=replica"
)

# Optional: run the main method (Shield-Loco/MIMPS) in the same harness/seeds for
# a head-to-head table. Uncomment to enable.
# METHOD_CONDITION="mimps|mps_alg_name=MIMPS"

# ---------------------------------------------------------------------------

mkdir -p "$OUT"

for scen in $SCENARIOS; do
    scenario="${SCENARIO_DIR}/${scen}.yaml"
    if [[ ! -f "$scenario" ]]; then
        echo "[baselines_ablation] Skipping missing scenario: $scenario" >&2
        continue
    fi
    scen_out="${OUT}/${scen}"
    mkdir -p "$scen_out"

    echo "============================================"
    echo "Scenario: $scenario  | runs: $RUNS  seed: $BASE_SEED"
    echo "Output:   $scen_out"
    echo "============================================"

    for cond_spec in "${CONDITIONS[@]}"; do
        label="${cond_spec%%|*}"
        sweep="${cond_spec#*|}"
        marker="${scen_out}/.done_${label}_r${RUNS}"
        if [[ -f "$marker" && "${FORCE:-0}" != "1" ]]; then
            echo "[$scen] $label : SKIP (already run with RUNS=$RUNS)"
            continue
        fi
        echo ""
        echo "-----------------------------"
        echo "[$scen] $label : $sweep"
        echo "-----------------------------"
        python3 eval/run_eval.py \
            --scenario "$scenario" \
            --runs "$RUNS" \
            --seed "$BASE_SEED" \
            --out "$scen_out" \
            --sweep $sweep \
            2>&1 | tee "${scen_out}/${label}.log"
        touch "$marker"
    done

    echo ""
    echo "[baselines_ablation] Analyzing ${scen} conditions..."
    python3 eval/analyze_results.py "${scen_out}"/*/results.json --out "${scen_out}"
done

echo ""
echo "============================================"
echo "Done. Conditions per scenario:"
for scen in $SCENARIOS; do
    echo "  results/baselines_ablation/${scen}/<condition>/results.json"
done
echo "Per-scenario aggregates: results/baselines_ablation/<scenario>/aggregated.csv"
echo "                          results/baselines_ablation/<scenario>/per_run.csv"
echo "============================================"