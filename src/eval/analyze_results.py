#!/usr/bin/env python3
"""
Analyze Shield-Loco eval results.

Usage:
    python eval/analyze_results.py results/sweep_*/results.json
    python eval/analyze_results.py results/sweep_20260327_145629/results.json --out results/analysis/
    python eval/analyze_results.py results/sweep_*/results.json --group-by mps_safety_weight
"""
import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from glob import glob
from typing import Any

import yaml


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

# ── Metrics shown in the console table and written to aggregated.csv ──────────
METRICS = [
    ("constraint_violations",   "Collisions",    "float"),
    ("foot_collision_events",   "FootLanding",   "float"),
    ("foot_collision_events_sdf", "FootSDF",     "float"),
    ("mean_tracking_cost",      "Track.Cost",    "float"),
    ("mean_optimizer_cost",     "Opt.Cost",      "float"),
    ("mean_opt_time_ms",        "OptTime(ms)",   "float"),
]

METRIC_KEYS = [m[0] for m in METRICS]


def _parse_run_from_dir(run_dir: str) -> list[dict]:
    """Parse all completed runs from a directory (one per sim_*.csv)."""
    import math

    sim_files = sorted(glob(os.path.join(run_dir, "sim_*.csv")))
    if not sim_files:
        return []

    results = []
    for sim_csv in sim_files:
        m = re.search(r"_(\d+)\.csv$", os.path.basename(sim_csv))
        run_id = int(m.group(1)) if m else 0

        ctrl_csv = os.path.join(run_dir, f"ctrl_{run_id}.csv")
        if not os.path.exists(ctrl_csv):
            ctrl_csv = None

        sim_rows = _read_csv(sim_csv)
        if not sim_rows:
            continue

        row: dict[str, Any] = {"run_id": run_id, "_source": run_dir}

        col_vals = [int(r.get("collision", 0)) for r in sim_rows]
        fall_vals = [int(r.get("fall", 0)) for r in sim_rows]
        foot_col_vals = [int(r.get("foot_collision", 0)) for r in sim_rows]

        def _onset_count(vals):
            return sum(1 for i in range(1, len(vals)) if vals[i] == 1 and vals[i-1] == 0)

        row["collision_events"] = _onset_count(col_vals)
        row["fall_events"] = _onset_count(fall_vals)
        row["foot_collision_events"] = _onset_count(foot_col_vals)
        row["constraint_violations"] = row["collision_events"] + row["fall_events"]
        row["collision_steps"] = sum(col_vals)
        row["fall_steps"] = sum(fall_vals)

        goal_dists = [float(r["goal_dist"]) for r in sim_rows]
        row["final_goal_dist"] = goal_dists[-1]
        row["min_goal_dist"] = min(goal_dists)
        threshold = 0.5
        reached_rows = [r for r in sim_rows if float(r["goal_dist"]) < threshold]
        row["goal_reached"] = len(reached_rows) > 0
        row["time_to_goal"] = float(reached_rows[0]["t"]) if reached_rows else None
        row["episode_duration"] = float(sim_rows[-1]["t"])

        xs = [float(r["x"]) for r in sim_rows]
        ys = [float(r["y"]) for r in sim_rows]
        path_len = sum(math.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1]) for i in range(1, len(xs)))
        straight_line = math.hypot(xs[-1]-xs[0], ys[-1]-ys[0])
        row["path_length"] = round(path_len, 4)
        row["path_efficiency"] = round(straight_line / path_len, 4) if path_len > 0 else None

        if "vx" in sim_rows[0]:
            speeds = [math.hypot(float(r["vx"]), float(r["vy"])) for r in sim_rows]
            row["mean_speed"] = round(sum(speeds) / len(speeds), 4)
            row["max_speed"] = round(max(speeds), 4)
        else:
            row["mean_speed"] = None
            row["max_speed"] = None

        if "heading_error" in sim_rows[0]:
            h_errs = [abs(float(r["heading_error"])) for r in sim_rows]
            row["mean_abs_heading_error"] = round(sum(h_errs) / len(h_errs), 4)
        else:
            row["mean_abs_heading_error"] = None

        ball_dists = [float(r["ball_dist"]) for r in sim_rows if float(r.get("ball_dist", -1)) >= 0]
        row["min_ball_dist"] = round(min(ball_dists), 4) if ball_dists else None

        row["survived"] = row["constraint_violations"] == 0

        if ctrl_csv:
            ctrl_rows = _read_csv(ctrl_csv)
            if ctrl_rows:
                costs = []
                for r in ctrl_rows:
                    try:
                        v = float(r["tracking_cost"])
                        if math.isfinite(v):
                            costs.append(v)
                    except (ValueError, KeyError):
                        pass
                row["mean_tracking_cost"] = round(sum(costs) / len(costs), 6) if costs else None

                opt_costs = []
                for r in ctrl_rows:
                    try:
                        v = float(r["optimizer_cost"])
                        if math.isfinite(v):
                            opt_costs.append(v)
                    except (ValueError, KeyError):
                        pass
                row["mean_optimizer_cost"] = round(sum(opt_costs) / len(opt_costs), 6) if opt_costs else None

                opt_times = []
                for r in ctrl_rows:
                    try:
                        v = float(r["opt_elapsed_ms"])
                        if math.isfinite(v):
                            opt_times.append(v)
                    except (ValueError, KeyError):
                        pass
                row["mean_opt_time_ms"] = round(sum(opt_times) / len(opt_times), 2) if opt_times else None

                row["num_ctrl_steps"] = len(ctrl_rows)
            else:
                row["mean_tracking_cost"] = None
                row["mean_optimizer_cost"] = None
                row["mean_opt_time_ms"] = None
        else:
            row["mean_tracking_cost"] = None
            row["mean_optimizer_cost"] = None
            row["mean_opt_time_ms"] = None

        results.append(row)
    return results


def _extract_overrides_from_dir(run_dir: str) -> dict:
    """Extract sweep overrides from ctrl_cfg_X.yaml or scenario_X.yaml in run_dir."""
    SWEEP_KEYS = ("mps_safety_weight", "obstacle_clearance", "projection_clearance",
                  "constraint_margin", "mps_plan_horizon", "plan_lim", "mps_noise_level")
    for fname in sorted(os.listdir(run_dir)):
        if fname.startswith("ctrl_cfg_") and fname.endswith(".yaml"):
            cfg_path = os.path.join(run_dir, fname)
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            return {k: cfg[k] for k in SWEEP_KEYS if k in cfg}
        if fname.startswith("scenario_") and fname.endswith(".yaml"):
            cfg_path = os.path.join(run_dir, fname)
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            return {k: cfg[k] for k in SWEEP_KEYS if k in cfg}
    return {}


def generate_stub_results(results_dir: str, expected_runs: int) -> tuple[list[dict], list[dict]]:
    """Returns (completed_runs, incomplete_stubs) from subdirectories."""
    completed = []
    stubs = []
    if not os.path.exists(results_dir):
        return completed, stubs
    for entry in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, entry)
        if not os.path.isdir(run_dir):
            continue

        parsed_runs = _parse_run_from_dir(run_dir)
        overrides = _extract_overrides_from_dir(run_dir)
        if parsed_runs:
            for parsed in parsed_runs:
                parsed["overrides"] = overrides
                completed.append(parsed)
        else:
            stub = {
                "run_id": len(stubs),
                "overrides": overrides,
                "_source": run_dir,
                "error": "incomplete_run",
                "constraint_violations": None,
                "foot_collision_events": None,
                "foot_collision_events_sdf": None,
                "mean_tracking_cost": None,
                "mean_optimizer_cost": None,
                "mean_opt_time_ms": None,
            }
            stubs.append(stub)
    return completed, stubs


def load_results(paths: list[str]) -> tuple[list[dict], str | None]:
    data = []
    experiment_name = None
    for p in paths:
        matched = sorted(glob(p))
        for f in matched:
            if not os.path.exists(f) or os.path.getsize(f) == 0:
                print(f"[analyze] Skipping empty/missing file: {f}")
                continue
            try:
                with open(f) as fh:
                    loaded = json.load(fh)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[analyze] Skipping corrupted file: {f} ({e})")
                continue
            if isinstance(loaded, dict) and "summaries" in loaded:
                experiment_name = loaded.get("experiment_name") or experiment_name
                runs = loaded["summaries"]
            elif isinstance(loaded, list):
                runs = loaded
            else:
                runs = [loaded]
            for row in runs:
                row["_source"] = f
            data.extend(runs)
    if not data:
        print(f"[analyze] Warning: No data found in: {paths} (will rely on stubs if --stub-incomplete is set)")
    else:
        print(f"[analyze] Loaded {len(data)} runs from {len(paths)} file(s)")
    return data, experiment_name


def group_data(data: list[dict], group_keys: list[str] | None) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in data:
        overrides = row.get("overrides") or {}
        if group_keys:
            key = tuple((k, overrides.get(k)) for k in group_keys)
        else:
            key = tuple(sorted(overrides.items()))
        groups[key].append(row)
    return dict(sorted(groups.items()))


def stats(vals: list[float]) -> tuple[float, float]:
    mu = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mu, sd


def format_cell(key: str, kind: str, rows: list[dict], col_w: int) -> str:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return f"{'—':>{col_w}}"
    if kind == "bool":
        pct = 100 * sum(bool(v) for v in vals) / len(rows)
        return f"{pct:>{col_w-1}.0f}%"
    mu, sd = stats([float(v) for v in vals])
    cell = f"{mu:.3f}±{sd:.3f}"
    return f"{cell:>{col_w}}"


def print_table(groups: dict[tuple, list[dict]], experiment_name: str | None = None):
    COL_W = 14
    cond_w = 120

    def _label(key: tuple) -> str:
        if key:
            return "  ".join(f"{k}={v}" for k, v in key)
        return experiment_name or "baseline"

    header = f"{'Condition':<{cond_w}}" + "".join(f"  {label:>{COL_W}}" for _, label, _ in METRICS)
    print(f"\n{header}")
    print("─" * len(header))

    for key, rows in groups.items():
        label = _label(key)[:cond_w]
        row_str = f"{label:<{cond_w}}"
        for mkey, _, kind in METRICS:
            row_str += "  " + format_cell(mkey, kind, rows, COL_W)
        print(row_str)

    print(f"\n  n runs shown: {sum(len(v) for v in groups.values())}")


def write_csv(groups: dict[tuple, list[dict]], out_path: str, experiment_name: str | None = None):
    def _cond_label(key: tuple) -> str:
        if key:
            return "  ".join(f"{k}={v}" for k, v in key)
        return experiment_name or "baseline"

    fieldnames = (
        ["condition", "n_runs"]
        + [f"{k}_mean" for k in METRIC_KEYS]
        + [f"{k}_std"  for k in METRIC_KEYS]
    )
    rows_out = []
    for key, rows in groups.items():
        row_out: dict[str, Any] = {
            "condition": _cond_label(key),
            "n_runs":    len(rows),
        }
        for mkey, _, kind in METRICS:
            vals = [float(r[mkey]) for r in rows if r.get(mkey) is not None]
            if kind == "bool":
                vals = [float(r[mkey]) for r in rows if r.get(mkey) is not None]
            row_out[f"{mkey}_mean"] = round(statistics.mean(vals), 5) if vals else None
            row_out[f"{mkey}_std"]  = round(statistics.stdev(vals), 5) if len(vals) > 1 else 0.0
        rows_out.append(row_out)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"[analyze] Aggregated CSV → {out_path}")


def write_per_run_csv(data: list[dict], out_path: str):
    """Flat CSV with one row per run — useful for seaborn/pandas plotting."""
    if not data:
        return
    all_keys = set()
    for row in data:
        all_keys.update(row.keys())
    # Flatten overrides dict into top-level columns
    fieldnames = (
        ["run_id", "_source"]
        + sorted(set(k for r in data for k in r.get("overrides", {})))
        + [k for k in sorted(all_keys) if k not in ("run_id", "_source", "overrides")]
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in data:
            flat = {k: v for k, v in row.items() if k != "overrides"}
            flat.update(row.get("overrides") or {})
            w.writerow(flat)
    print(f"[analyze] Per-run CSV    → {out_path}")


def print_best(groups: dict[tuple, list[dict]], experiment_name: str | None = None):
    print("\n── Best condition per metric ──")
    for mkey, label, kind in METRICS:
        best_key, best_val = None, None
        for key, rows in groups.items():
            vals = [float(r[mkey]) for r in rows if r.get(mkey) is not None]
            if not vals:
                continue
            mu = statistics.mean(vals)
            higher_better = False
            if best_val is None or (mu > best_val if higher_better else mu < best_val):
                best_val = mu
                best_key = key
        if best_key is not None:
            cond = "  ".join(f"{k}={v}" for k, v in best_key) or (experiment_name or "baseline")
            print(f"  {label:<16}: {best_val:.4f}  ← {cond}")


def _inject_experiment_name(data: list[dict], experiment_name: str | None) -> list[dict]:
    """Inject experiment_name as a synthetic override so grouping works."""
    if experiment_name is None:
        return data
    for row in data:
        overrides = dict(row.get("overrides") or {})
        overrides["_experiment"] = experiment_name
        row["overrides"] = overrides
    return data


def main():
    parser = argparse.ArgumentParser(description="Analyze Shield-Loco eval results.json")
    parser.add_argument("results", nargs="+", help="Path(s) to results.json (glob ok, e.g. results/exp/*/results.json)")
    parser.add_argument("--out", default=None, help="Output directory for CSVs (default: next to first results.json)")
    parser.add_argument(
        "--group-by",
        nargs="*",
        metavar="KEY",
        help="Override keys to group by (default: _experiment for multi-exp, sweep params for single-exp)",
    )
    parser.add_argument(
        "--stub-incomplete",
        metavar="DIR",
        help="Generate stub entries for incomplete runs found in this results directory",
    )
    parser.add_argument(
        "--write-stubs",
        metavar="PATH",
        help="Write stub results.json to this path (used with --stub-incomplete)",
    )
    args = parser.parse_args()

    all_data = []
    unique_experiments = set()
    for path_spec in args.results:
        matched = sorted(glob(path_spec))
        if not matched:
            print(f"[analyze] No files matching: {path_spec}")
            continue
        for f in matched:
            if not os.path.exists(f) or os.path.getsize(f) == 0:
                print(f"[analyze] Skipping empty/missing: {f}")
                continue
            try:
                with open(f) as fh:
                    loaded = json.load(fh)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[analyze] Skipping corrupted: {f} ({e})")
                continue
            exp_name = loaded.get("experiment_name") if isinstance(loaded, dict) else None
            runs = loaded.get("summaries", loaded if isinstance(loaded, list) else [])
            if isinstance(runs, dict):
                runs = [runs]
            for row in runs:
                row["_source"] = f
            unique_experiments.add(exp_name)
            all_data.extend(_inject_experiment_name(runs, exp_name))

    if not all_data and not args.stub_incomplete:
        sys.exit(f"No data found in: {args.results} and --stub-incomplete not set")

    if args.stub_incomplete:
        completed_dirs, stubs = generate_stub_results(args.stub_incomplete, expected_runs=0)
        print(f"[analyze] Parsed {len(completed_dirs)} completed runs from subdirectories")
        print(f"[analyze] Generated {len(stubs)} stub entries for incomplete runs")
        all_data.extend(completed_dirs)
        if args.write_stubs:
            combined = completed_dirs + stubs
            os.makedirs(os.path.dirname(args.write_stubs) or ".", exist_ok=True)
            with open(args.write_stubs, "w") as f:
                json.dump(combined, f, indent=2)
            print(f"[analyze] Combined results written to {args.write_stubs}")
        if stubs:
            print(f"[analyze] Warning: {len(stubs)} runs incomplete (marked with error=incomplete_run)")
        all_data.extend(stubs)

    # Default grouping: if multiple experiments loaded, group by _experiment.
    # If only one experiment loaded (no glob), group by sweep params (overrides).
    group_keys = args.group_by
    if group_keys is None and len(unique_experiments) > 1:
        group_keys = ["_experiment"]

    groups = group_data(all_data, group_keys)

    # Single experiment_name for table header (first one, all same if one experiment)
    experiment_name = list(unique_experiments)[0] if len(unique_experiments) == 1 else None

    print_table(groups, experiment_name)
    print_best(groups, experiment_name)

    out_dir = args.out or os.path.dirname(os.path.abspath(args.results[0]))
    write_csv(groups=groups, out_path=os.path.join(out_dir, "aggregated.csv"), experiment_name=experiment_name)
    write_per_run_csv(data=all_data, out_path=os.path.join(out_dir, "per_run.csv"))


if __name__ == "__main__":
    main()
