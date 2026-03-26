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
    collision_events = sum(1 for i in range(1, len(col_vals)) if col_vals[i] == 1 and col_vals[i-1] == 0)
    fall_events = sum(1 for i in range(1, len(fall_vals)) if fall_vals[i] == 1 and fall_vals[i-1] == 0)
    summary["collision_events"] = collision_events
    summary["fall_events"] = fall_events
    summary["constraint_violations"] = collision_events + fall_events
    summary["collision_steps"] = sum(col_vals)
    summary["fall_steps"] = sum(fall_vals)

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
    args = parser.parse_args()

    sweep_str = " ".join(args.sweep) if args.sweep else ""
    conditions = parse_sweep_arg(sweep_str)

    all_summaries = []
    run_id = 0
    for condition in conditions:
        label = "_".join(f"{k}{v}" for k, v in condition.items()) if condition else "baseline"
        cond_dir = os.path.join(args.out, label or "baseline")
        for _ in range(args.runs):
            run_seed = args.seed + run_id if args.seed is not None else None
            summary = run_episode(
                scenario_path=args.scenario,
                run_id=run_id,
                out_dir=cond_dir,
                ctrl_overrides=condition if condition else None,
                seed=run_seed,
            )
            all_summaries.append(summary)
            run_id += 1

    # Save all summaries to JSON
    results_path = os.path.join(args.out, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n[run_eval] All done. Results saved to {results_path}")

    # Quick table
    print("\n--- Summary ---")
    for s in all_summaries:
        print(
            f"  run={s['run_id']}  "
            f"goal={s.get('goal_reached')}  "
            f"t_goal={s.get('time_to_goal')}  "
            f"violations={s.get('constraint_violations')}  "
            f"mean_cost={s.get('mean_tracking_cost'):.4f if s.get('mean_tracking_cost') else 'N/A'}  "
            f"overrides={s.get('overrides')}"
        )


if __name__ == "__main__":
    main()
