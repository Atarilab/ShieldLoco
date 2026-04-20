"""
Multi-run eval launcher for DoomMPC.

Launches simulate.py and master_node as subprocesses (mirroring the go2_mps_eval_*
aliases), waits for the episode to finish, then parses the CSVs into a summary dict.

Usage
-----
# single run, scenario A
python eval/run_eval.py --scenario eval/scenarios/scenario_a.yaml --runs 1 --out results/a/

# 5 runs, scenario B
python eval/run_eval.py --scenario eval/scenarios/scenario_b.yaml --runs 5 --out results/b/

# sweep obstacle cost weight (3 runs each)
python eval/run_eval.py --scenario eval/scenarios/scenario_a.yaml --runs 3 \\
    --sweep obstacle_cost_weight=100,300,1000 --out results/sweep_cost/

Notes
-----
- Both processes are launched inside the same Python environment (no ROS rosrun needed;
  uses `python -m master_manager.master_manager.master_node`).
- If you normally use the Docker aliases, run this script from inside the container.
- JIT compilation adds ~2-5 minutes to the first run; subsequent runs in the same process
  would be faster but each subprocess start re-compiles.
"""
import argparse
import copy
import csv
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# SDF helpers (pure NumPy, no JAX/GPU involvement)
# ---------------------------------------------------------------------------

CIRCLE, BOX, CAPSULE = 0, 1, 2


def _sdf_one_np(point_xy, center, R2, half_ext, obs_type):
    """NumPy SDF for one obstacle at one 2D point. No gradient."""
    diff = point_xy - center
    diff_local = R2.T @ diff

    dist_c = np.linalg.norm(diff) + 1e-10
    sdf_c = dist_c - half_ext[0]

    q_b = np.abs(diff_local) - half_ext
    nearest_on_box = np.clip(diff_local, -half_ext, half_ext)
    exterior_b = np.linalg.norm(nearest_on_box - diff_local) > 1e-10
    sdf_b_ext = np.linalg.norm(np.maximum(q_b, 0.0))
    sdf_b_int = np.minimum(np.max(q_b), 0.0)
    sdf_b = sdf_b_ext + sdf_b_int

    clamped_x = np.clip(diff_local[0], -half_ext[1], half_ext[1])
    dk = diff_local - np.array([clamped_x, 0.0])
    dist_k = np.linalg.norm(dk) + 1e-10
    sdf_k = dist_k - half_ext[0]

    if obs_type == BOX:
        return sdf_b
    elif obs_type == CAPSULE:
        return sdf_k
    else:
        return sdf_c


def _build_sdf_from_model(mj_model):
    """Build NumPy SDF data from MuJoCo model. Pure NumPy — no JAX."""
    import mujoco
    ngeom = mj_model.ngeom
    geom_is_hazard = np.zeros(ngeom, dtype=bool)
    for gid in range(ngeom):
        name = mj_model.geom(gid).name or ""
        if name.startswith("obstacle_"):
            gtype = int(mj_model.geom_type[gid])
            size = mj_model.geom_size[gid]
            if gtype == 6:
                height = size[2]
            elif gtype in (3, 5):
                height = size[1]
            else:
                height = size[0]
            if height >= 0.001:
                geom_is_hazard[gid] = True

    hazard_ids = np.flatnonzero(geom_is_hazard).astype(np.int32)
    M = int(hazard_ids.shape[0])
    if M == 0:
        return np.zeros((0, 2)), np.zeros((0, 2, 2)), np.zeros((0, 2)), np.zeros((0,), dtype=np.int32)

    _mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, _mj_data)

    positions = np.zeros((M, 2), dtype=np.float32)
    rotations = np.zeros((M, 2, 2), dtype=np.float32)
    half_extents = np.zeros((M, 2), dtype=np.float32)
    types = np.zeros(M, dtype=np.int32)

    CIRCLE_T = 0
    BOX_T = 1
    CAPSULE_T = 2
    _MJ_TYPE_TO_SDF = {2: CIRCLE_T, 6: BOX_T, 5: CIRCLE_T, 3: CAPSULE_T}

    for i, gid in enumerate(hazard_ids):
        pos3d = _mj_data.geom_xpos[gid]
        mat3d = _mj_data.geom_xmat[gid].reshape(3, 3)
        gtype_mj = int(mj_model.geom_type[gid])
        size = mj_model.geom_size[gid]

        positions[i] = pos3d[:2]
        rotations[i] = mat3d[:2, :2]

        sdf_type = _MJ_TYPE_TO_SDF.get(gtype_mj, CIRCLE_T)
        types[i] = sdf_type

        if sdf_type == CIRCLE_T:
            half_extents[i] = [float(size[0]), 0.0]
        elif sdf_type == BOX_T:
            half_extents[i] = [float(size[0]), float(size[1])]
        elif sdf_type == CAPSULE_T:
            r = float(size[0])
            hl = float(size[1])
            local_z = mat3d[:2, 2]
            xy_norm = float(np.linalg.norm(local_z))
            if xy_norm > 0.15:
                axis_2d = local_z / xy_norm
                rotations[i] = np.array([[axis_2d[0], -axis_2d[1]],
                                          [axis_2d[1],  axis_2d[0]]])
                half_extents[i] = [r, hl * xy_norm]
            else:
                half_extents[i] = [r, 0.0]
                types[i] = CIRCLE_T

    return positions, rotations, half_extents, types


def _compute_sdf_foot_violations(sim_rows, mj_model, clearance=0.07):
    """Count onset SDF foot violations from logged foot positions.

    Pure NumPy — no JAX imported. Builds NumPy SDF data directly from mj_model.
    """
    positions, rotations, half_extents, types = _build_sdf_from_model(mj_model)
    num_obstacles = positions.shape[0]
    if num_obstacles == 0:
        return 0

    flags = []
    for row in sim_rows:
        min_sdf = np.inf
        for foot_i in range(4):
            fx = float(row[f"foot{foot_i}_x"])
            fy = float(row[f"foot{foot_i}_y"])
            point = np.array([fx, fy], dtype=np.float32)
            for obs_i in range(num_obstacles):
                d = _sdf_one_np(
                    point,
                    positions[obs_i],
                    rotations[obs_i],
                    half_extents[obs_i],
                    int(types[obs_i]),
                )
                if d < min_sdf:
                    min_sdf = d
                if min_sdf < clearance:
                    break
            if min_sdf < clearance:
                break
        flags.append(1 if min_sdf < clearance else 0)

    return sum(1 for i in range(1, len(flags)) if flags[i] == 1 and flags[i - 1] == 0)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    scenario_path: str,
    run_id: int,
    out_dir: str,
    ctrl_overrides: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Launch one eval episode (sim + controller subprocesses).

    Returns a summary dict with paper metrics.
    """
    with open(scenario_path) as f:
        scenario_cfg = yaml.safe_load(f)

    # Per-run metric files
    scenario_cfg["sim_metric_file"] = os.path.join(out_dir, f"sim_{run_id}.csv")
    scenario_cfg["controller_metric_file"] = os.path.join(out_dir, f"ctrl_{run_id}.csv")
    os.makedirs(out_dir, exist_ok=True)

    # Write temp scenario config (with per-run paths and seed)
    if seed is not None:
        scenario_cfg["seed"] = seed
    tmp_scenario = os.path.join(out_dir, f"scenario_{run_id}.yaml")
    with open(tmp_scenario, "w") as f:
        yaml.dump(scenario_cfg, f)

    # Optional: write a temp controller config with overrides applied
    tmp_ctrl_cfg = None
    if ctrl_overrides:
        base_ctrl_path = os.path.join(
            os.path.dirname(__file__), "..", "controllers", "config", "rl_mps_go2_cfg.yaml"
        )
        with open(base_ctrl_path) as f:
            ctrl_cfg = yaml.safe_load(f)
        for k, v in ctrl_overrides.items():
            ctrl_cfg[k] = v
        tmp_ctrl_cfg = os.path.join(out_dir, f"ctrl_cfg_{run_id}.yaml")
        with open(tmp_ctrl_cfg, "w") as f:
            yaml.dump(ctrl_cfg, f)

    src_dir = os.path.join(os.path.dirname(__file__), "..")

    # sim gets --seed via CLI; ROS reads seed from the temp YAML (no new CLI arg needed)
    sim_seed_args = ["--seed", str(seed)] if seed is not None else []

    sim_cmd = [
        sys.executable, "simulate.py",
        "--task", "rl-mps-eval-go2",
        "--headless",
        "--log", f"eval_run_{run_id}",
        "--eval-config", os.path.abspath(tmp_scenario),
    ] + sim_seed_args
    ctrl_cmd = [
        "ros2", "run", "master_manager", "master_node",
        "--task", "rl-mps-eval-go2",
        "--log", f"eval_run_{run_id}",
        "--eval-config", os.path.abspath(tmp_scenario),
    ]

    # preexec_fn=os.setsid gives each child its own process group so os.killpg
    # reaches all descendants (including the master_node grandchild of ros2 run).
    popen_kwargs = dict(cwd=src_dir, preexec_fn=os.setsid)

    def killpg_safe(proc, sig):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                pass

    print(f"[run_eval] Run {run_id}: launching sim + controller …")
    sim_proc = subprocess.Popen(sim_cmd, **popen_kwargs)
    # Small delay so DDS channel is ready before controller connects
    time.sleep(1.0)
    ctrl_env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    ctrl_proc = subprocess.Popen(ctrl_cmd, env=ctrl_env, **popen_kwargs)

    # Double reset with delay — robot sometimes needs two resets to stabilize
    # before the controller takes over (avoids startup instability / fall).
    _eval_reset_sig = os.path.join(
        os.path.dirname(__file__), "..", "logs", "eval", "reset_signal.txt"
    )
    os.makedirs(os.path.dirname(_eval_reset_sig), exist_ok=True)
    with open(_eval_reset_sig, "w") as f:
        f.write("reset")
    time.sleep(0.5)
    with open(_eval_reset_sig, "w") as f:
        f.write("reset")
    print(f"[run_eval] Run {run_id}: double reset written (0.5s gap)", flush=True)

    max_wait = scenario_cfg.get("max_time", 120.0) + 600  # generous: 10 min for JIT
    try:
        sim_proc.wait(timeout=max_wait)
    except subprocess.TimeoutExpired:
        print(f"[run_eval] Timeout waiting for sim (run {run_id})", flush=True)
    finally:
        killpg_safe(ctrl_proc, signal.SIGTERM)
        try:
            ctrl_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            killpg_safe(ctrl_proc, signal.SIGKILL)
        # Wait for GPU memory to be reclaimed before the next run starts
        time.sleep(5.0)

    summary = parse_episode(
        sim_csv=scenario_cfg["sim_metric_file"],
        ctrl_csv=scenario_cfg["controller_metric_file"],
        scenario_cfg=scenario_cfg,
        scenario_path=scenario_path,
        run_id=run_id,
        ctrl_overrides=ctrl_overrides or {},
    )
    print(f"[run_eval] Run {run_id} done: {summary}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------

def parse_episode(
    sim_csv: str,
    ctrl_csv: str,
    scenario_cfg: dict,
    scenario_path: str,
    run_id: int,
    ctrl_overrides: dict,
) -> Dict[str, Any]:
    import math
    summary: Dict[str, Any] = {"run_id": run_id, "overrides": ctrl_overrides}

    # ---------- sim CSV ----------
    sim_rows = _read_csv(sim_csv)
    if not sim_rows:
        summary["error"] = "no_sim_data"
        return summary

    # --- constraint violations ---
    # Count onset events (0→1 transitions) rather than total flagged steps
    col_vals = [int(r.get("collision", 0)) for r in sim_rows]
    fall_vals = [int(r.get("fall", 0)) for r in sim_rows]
    foot_col_vals = [int(r.get("foot_collision", 0)) for r in sim_rows]

    def _onset_count(vals):
        return sum(1 for i in range(1, len(vals)) if vals[i] == 1 and vals[i-1] == 0)

    collision_events = _onset_count(col_vals)
    fall_events = _onset_count(fall_vals)
    foot_collision_events = _onset_count(foot_col_vals)

    summary["collision_events"] = collision_events
    summary["foot_collision_events"] = foot_collision_events
    summary["fall_events"] = fall_events
    summary["constraint_violations"] = collision_events + fall_events
    summary["collision_steps"] = sum(col_vals)
    summary["fall_steps"] = sum(fall_vals)

    # --- SDF foot violations (post-processing, no JAX/GPU) ---
    scene_model = scenario_cfg.get("scene_model")
    if scene_model:
        model_base = os.path.join(os.path.dirname(scenario_path), "..", "..", scene_model)
        if os.path.exists(model_base):
            try:
                mj_model = mujoco.MjModel.from_xml_path(model_base)
                summary["foot_collision_events_sdf"] = _compute_sdf_foot_violations(
                    sim_rows, mj_model, clearance=0.07
                )
            except Exception:
                summary["foot_collision_events_sdf"] = None
        else:
            summary["foot_collision_events_sdf"] = None
    else:
        summary["foot_collision_events_sdf"] = None

    # --- goal ---
    goal_dists = [float(r["goal_dist"]) for r in sim_rows]
    summary["final_goal_dist"] = goal_dists[-1]
    summary["min_goal_dist"] = min(goal_dists)
    threshold = float(scenario_cfg.get("goal_threshold", 0.5))
    reached_rows = [r for r in sim_rows if float(r["goal_dist"]) < threshold]
    summary["goal_reached"] = len(reached_rows) > 0
    summary["time_to_goal"] = float(reached_rows[0]["t"]) if reached_rows else None
    summary["episode_duration"] = float(sim_rows[-1]["t"])

    # --- path length and efficiency ---
    xs = [float(r["x"]) for r in sim_rows]
    ys = [float(r["y"]) for r in sim_rows]
    path_len = sum(
        math.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1])
        for i in range(1, len(xs))
    )
    straight_line = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    summary["path_length"] = round(path_len, 4)
    summary["path_efficiency"] = round(straight_line / path_len, 4) if path_len > 0 else None

    # --- speed ---
    if "vx" in sim_rows[0]:
        speeds = [math.hypot(float(r["vx"]), float(r["vy"])) for r in sim_rows]
        summary["mean_speed"] = round(sum(speeds) / len(speeds), 4)
        summary["max_speed"] = round(max(speeds), 4)
    else:
        summary["mean_speed"] = None
        summary["max_speed"] = None

    # --- heading error ---
    if "heading_error" in sim_rows[0]:
        h_errs = [abs(float(r["heading_error"])) for r in sim_rows]
        summary["mean_abs_heading_error"] = round(sum(h_errs) / len(h_errs), 4)
    else:
        summary["mean_abs_heading_error"] = None

    # --- obstacle proximity ---
    ball_dists = [float(r["ball_dist"]) for r in sim_rows if float(r.get("ball_dist", -1)) >= 0]
    summary["min_ball_dist"] = round(min(ball_dists), 4) if ball_dists else None

    summary["survived"] = summary["constraint_violations"] == 0

    # ---------- controller CSV ----------
    ctrl_rows = _read_csv(ctrl_csv)
    if ctrl_rows:
        # tracking cost (filter any residual NaN strings)
        costs = []
        for r in ctrl_rows:
            try:
                v = float(r["tracking_cost"])
                if math.isfinite(v):
                    costs.append(v)
            except (ValueError, KeyError):
                pass
        summary["mean_tracking_cost"] = round(sum(costs) / len(costs), 6) if costs else None
        summary["total_tracking_cost"] = round(sum(costs), 4) if costs else None

        # MPC engagement rate
        mpc_vals = [int(r.get("mpc_active", 0)) for r in ctrl_rows]
        summary["mpc_engagement_rate"] = round(sum(mpc_vals) / len(mpc_vals), 4)

        # Replanning rate: count unique plan_seq transitions per second
        seqs = [int(r["plan_seq"]) for r in ctrl_rows if r.get("plan_seq", "") != ""]
        replan_events = sum(1 for i in range(1, len(seqs)) if seqs[i] != seqs[i-1])
        ep_dur = float(ctrl_rows[-1]["t"]) if ctrl_rows else 1.0
        summary["replanning_rate_hz"] = round(replan_events / ep_dur, 4) if ep_dur > 0 else None

        # Velocity stats from controller (body-frame forward velocity)
        if "vel_x" in ctrl_rows[0]:
            fwd_vels = [float(r["vel_x"]) for r in ctrl_rows]
            summary["mean_forward_vel"] = round(sum(fwd_vels) / len(fwd_vels), 4)

        # Height stability (std dev of base height during episode)
        if "height" in ctrl_rows[0]:
            heights = [float(r["height"]) for r in ctrl_rows]
            mean_h = sum(heights) / len(heights)
            std_h = math.sqrt(sum((h - mean_h)**2 for h in heights) / len(heights))
            summary["mean_height"] = round(mean_h, 4)
            summary["height_std"] = round(std_h, 4)

        summary["num_ctrl_steps"] = len(ctrl_rows)
    else:
        summary["mean_tracking_cost"] = None
        summary["total_tracking_cost"] = None
        summary["mpc_engagement_rate"] = None
        summary["replanning_rate_hz"] = None

    return summary


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_sweep_arg(sweep_str: str) -> List[Dict[str, Any]]:
    """
    Parse "--sweep key1=v1,v2,v3 key2=a,b" into a list of override dicts.
    Each unique combination gets its own set of runs.

    For now supports a single key with multiple values:
        --sweep obstacle_cost_weight=100,300,1000
    produces [{"obstacle_cost_weight": 100}, {"obstacle_cost_weight": 300}, ...]
    """
    if not sweep_str:
        return [{}]
    overrides_list = [{}]
    for token in sweep_str.split():
        k, vals_str = token.split("=", 1)
        vals = [_coerce(v) for v in vals_str.split(",")]
        new_list = []
        for existing in overrides_list:
            for v in vals:
                entry = copy.copy(existing)
                entry[k] = v
                new_list.append(entry)
        overrides_list = new_list
    return overrides_list


def _coerce(s: str) -> Any:
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def main():
    parser = argparse.ArgumentParser(description="DoomMPC eval runner")
    parser.add_argument("--scenario", required=True, help="Path to scenario YAML")
    parser.add_argument("--runs", type=int, default=1, help="Runs per condition")
    parser.add_argument("--out", default="results/eval", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed (incremented per run)")
    parser.add_argument(
        "--sweep",
        default="",
        nargs="*",
        help="Hyperparameter sweep: 'key=v1,v2,v3' (space-separated keys)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Experiment name for output directory (overrides default label)",
    )
    args = parser.parse_args()

    sweep_str = " ".join(args.sweep) if args.sweep else ""
    conditions = parse_sweep_arg(sweep_str)
    sweep_name = sweep_str.replace(" ", "_") if sweep_str else ""

    # Determine experiment base name (used as prefix for all condition names)
    exp_base = args.name or "baseline"

    # For sweeps: each condition gets its own subdirectory + results.json.
    # For seed-only: all runs go directly to args.out.
    # Results structure:
    #   Seed-only:  results/exp_name/results.json
    #   Sweep:      results/exp_name/cond1/results.json, results/exp_name/cond2/results.json, ...
    os.makedirs(args.out, exist_ok=True)

    # Save MPS controller config at experiment top level
    mps_cfg_src = os.path.join(os.path.dirname(__file__), "..", "controllers", "config", "rl_mps_go2_cfg.yaml")
    mps_cfg_dst = os.path.join(args.out, "controller_cfg.yaml")
    shutil.copy2(mps_cfg_src, mps_cfg_dst)
    print(f"[run_eval] Controller config saved to {mps_cfg_dst}")

    # Save sweep metadata if sweeping
    if sweep_str:
        sweep_info = {
            "sweep_str": sweep_str,
            "conditions": conditions,
            "runs_per_condition": args.runs,
            "base_seed": args.seed,
        }
        with open(os.path.join(args.out, "sweep_config.json"), "w") as f:
            json.dump(sweep_info, f, indent=2)

    all_results_dirs = []
    run_id = 0
    for condition in conditions:
        # Build condition label from sweep params, or use exp_base for seed-only
        if sweep_str:
            cond_label = "_".join(f"{k}{v}" for k, v in condition.items())
            experiment_name = f"{exp_base}_{cond_label}"
            cond_dir = os.path.join(args.out, cond_label)
        else:
            experiment_name = exp_base
            cond_dir = args.out

        os.makedirs(cond_dir, exist_ok=True)
        # Save MPS config into each condition dir too (so each is self-contained)
        shutil.copy2(mps_cfg_src, os.path.join(cond_dir, "controller_cfg.yaml"))

        run_summaries = []
        for _ in range(args.runs):
            run_seed = args.seed + run_id if args.seed is not None else None
            summary = run_episode(
                scenario_path=args.scenario,
                run_id=run_id,
                out_dir=cond_dir,
                ctrl_overrides=condition if condition else None,
                seed=run_seed,
            )
            run_summaries.append(summary)
            run_id += 1

        # Save per-condition results.json (self-contained experiment)
        results_path = os.path.join(cond_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump({"experiment_name": experiment_name, "summaries": run_summaries}, f, indent=2)
        print(f"[run_eval] Condition '{experiment_name}' saved to {results_path}")
        all_results_dirs.append(cond_dir)

    # Final output path for the whole experiment
    if sweep_str:
        final_msg = (f"[run_eval] All done. Sweep complete — "
                     f"{len(conditions)} conditions × {args.runs} runs. "
                     f"Results in: {args.out}/")
        print(final_msg)
        print(f"[run_eval] To analyze all conditions:")
        print(f"  python3 eval/analyze_results.py {args.out}/*/results.json --out {args.out}")
    else:
        print(f"\n[run_eval] All done. Results saved to {args.out}")


if __name__ == "__main__":
    main()
