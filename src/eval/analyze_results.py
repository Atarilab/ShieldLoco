#!/usr/bin/env python3
"""
Analyze DoomMPC eval results.

Usage:
    python eval/analyze_results.py results/sweep_*/results.json
    python eval/analyze_results.py results/sweep_20260327_145629/results.json --out results/analysis/
    python eval/analyze_results.py results/sweep_*/results.json --group-by mps_safety_weight
"""
import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from glob import glob
from typing import Any

# ── Metrics shown in the console table and written to aggregated.csv ──────────
METRICS = [
    ("goal_reached",          "Goal%",       "bool"),
    ("time_to_goal",          "T_goal(s)",   "float"),
    ("constraint_violations", "Collisions",  "float"),
    ("foot_collision_events", "FootLanding", "float"),
    ("mean_tracking_cost",    "Track.Cost",  "float"),
    ("path_efficiency",       "PathEff",     "float"),
    ("mean_speed",            "Speed(m/s)",  "float"),
    ("min_ball_dist",         "BallDist",    "float"),
    ("mean_abs_heading_error","HeadErr",     "float"),
    ("height_std",            "HgtStd",      "float"),
]

METRIC_KEYS = [m[0] for m in METRICS]


def load_results(paths: list[str]) -> list[dict]:
    data = []
    for p in paths:
        for f in sorted(glob(p)):
            with open(f) as fh:
                runs = json.load(fh)
            for row in runs:
                row["_source"] = f
            data.extend(runs)
    if not data:
        sys.exit(f"No data found in: {paths}")
    print(f"[analyze] Loaded {len(data)} runs from {len(paths)} file(s)")
    return data


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


def print_table(groups: dict[tuple, list[dict]]):
    COL_W = 14
    cond_w = 70

    header = f"{'Condition':<{cond_w}}" + "".join(f"  {label:>{COL_W}}" for _, label, _ in METRICS)
    print(f"\n{header}")
    print("─" * len(header))

    for key, rows in groups.items():
        label = "  ".join(f"{k}={v}" for k, v in key) or "baseline"
        label = label[:cond_w]
        row_str = f"{label:<{cond_w}}"
        for mkey, _, kind in METRICS:
            row_str += "  " + format_cell(mkey, kind, rows, COL_W)
        print(row_str)

    print(f"\n  n runs shown: {sum(len(v) for v in groups.values())}")


def write_csv(groups: dict[tuple, list[dict]], out_path: str):
    fieldnames = (
        ["condition", "n_runs"]
        + [f"{k}_mean" for k in METRIC_KEYS]
        + [f"{k}_std"  for k in METRIC_KEYS]
    )
    rows_out = []
    for key, rows in groups.items():
        row_out: dict[str, Any] = {
            "condition": "  ".join(f"{k}={v}" for k, v in key) or "baseline",
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


def print_best(groups: dict[tuple, list[dict]]):
    print("\n── Best condition per metric ──")
    for mkey, label, kind in METRICS:
        best_key, best_val = None, None
        for key, rows in groups.items():
            vals = [float(r[mkey]) for r in rows if r.get(mkey) is not None]
            if not vals:
                continue
            mu = statistics.mean(vals)
            # goal_reached and path_efficiency: higher is better; others: lower is better
            higher_better = mkey in ("goal_reached", "path_efficiency")
            if best_val is None or (mu > best_val if higher_better else mu < best_val):
                best_val = mu
                best_key = key
        if best_key is not None:
            cond = "  ".join(f"{k}={v}" for k, v in best_key) or "baseline"
            print(f"  {label:<16}: {best_val:.4f}  ← {cond}")


def main():
    parser = argparse.ArgumentParser(description="Analyze DoomMPC eval results.json")
    parser.add_argument("results", nargs="+", help="Path(s) to results.json (glob ok)")
    parser.add_argument("--out", default=None, help="Output directory for CSVs (default: next to first results.json)")
    parser.add_argument(
        "--group-by",
        nargs="*",
        metavar="KEY",
        help="Override keys to group by (default: all keys in overrides dict)",
    )
    args = parser.parse_args()

    data = load_results(args.results)
    groups = group_data(data, args.group_by)

    print_table(groups)
    print_best(groups)

    out_dir = args.out or os.path.dirname(os.path.abspath(args.results[0]))
    write_csv(groups=groups, out_path=os.path.join(out_dir, "aggregated.csv"))
    write_per_run_csv(data=data, out_path=os.path.join(out_dir, "per_run.csv"))


if __name__ == "__main__":
    main()
