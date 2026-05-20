"""
Benchmark: dual projected-gradient QP vs iterative greedy projection.

Implements two variants of the foot-projection QP:

  A) single-linearize  – 1 SDF eval, then dual PGD (fastest, inexact for large shifts)
  B) sqp               – 2–3 SQP outer iters (re-linearize after each QP solve)

And compares against the current iterative greedy method (3 SDF evals).

Run from src/:
    python safety_filter/tests/test_foot_projection_qp.py

The test uses the cluttered indoor scene from obstacles.xml (~50 obstacles at
x∈[0.8, 6.0]m, y∈[-0.5, 0.5]m, sizes 0.004–0.08m).
"""

import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_deterministic_ops=true "
    "--xla_gpu_autotune_level=0"
)
    # Kills the code for some reason
    # "--xla_gpu_enable_async_all_gather=true "
    # "--xla_gpu_enable_async_collectives=true "

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import re
import sys
import time
import numpy as np
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import jax.numpy as jnp

from safety_filter.quad.foot_projection import (
    ObstacleSDF,
    _sdf_and_grad_one,
    _sdf_and_grad_all,
    _batch_sdf_and_grad,
    project_plan_feet,
    project_plan_feet_qp,
    project_plan_feet_qp_sqp,
    _project_one_pairwise,
    BOX, CIRCLE, CAPSULE,
)

# ---------------------------------------------------------------------------
# Cluttered‑scene helpers
# ---------------------------------------------------------------------------

_R_CLUTTER = re.compile(r"<!--(.*?)-->", re.DOTALL)


def _build_cluttered_xml() -> str:
    """Read obstacles.xml, return MJCF with only the CLUTTERED INDOOR SCENE.

    This is the ~50‑obstacle scene at the bottom of obstacles.xml with unique
    names (obstacle_z1_00 … obstacle_z4_13), not the small ~20‑obstacle
    'Cluttered scene' above it that shares names with the Navigation scene.
    """
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    xml_path = os.path.join(src_dir, "robots", "go2", "obstacles.xml")
    with open(xml_path) as f:
        raw = f.read()

    # Find the CLUTTERED INDOOR SCENE section
    marker = "CLUTTERED INDOOR SCENE"
    idx = raw.find(marker)
    if idx < 0:
        raise RuntimeError(f"Could not find '{marker}' in {xml_path}")
    section = raw[idx:]  # everything from CLUTTERED INDOOR SCENE to EOF

    # Uncomment all <!-- ... --> blocks
    section = _R_CLUTTER.sub(r"\1", section)

    # Extract all <geom ...> lines
    geom_lines = []
    for line in section.split("\n"):
        stripped = line.strip()
        if stripped.startswith("<geom "):
            geom_lines.append(stripped)

    scene_tpl = """<mujoco model="bench">
  <worldbody>
    <geom name="floor" size="0 0 0.05" type="plane"/>
{obstacles}
  </worldbody>
</mujoco>"""
    return scene_tpl.format(obstacles="\n".join(geom_lines))


def load_cluttered_sdf():
    """Build an ObstacleSDF for the ~50-obstacle cluttered indoor scene."""
    xml_str = _build_cluttered_xml()
    model = mujoco.MjModel.from_xml_string(xml_str)
    sdf = ObstacleSDF.from_mujoco_model(
        model, hazard_prefixes=("obstacle",), min_height=0.0001
    )
    return sdf


# ---------------------------------------------------------------------------
# Correctness helpers
# ---------------------------------------------------------------------------

def _violation_rate(positions, sdf, r_safe):
    """Fraction of points with min SDF < r_safe (unsatisfied clearance)."""
    if sdf.num_obstacles == 0:
        return 0.0
    sdfs, _ = _batch_sdf_and_grad(positions, sdf)
    min_sdf = jnp.min(sdfs, axis=1)
    return jnp.mean(min_sdf < r_safe - 1e-4).item()


def _max_violation(positions, sdf, r_safe):
    """Maximum clearance shortfall over all points."""
    if sdf.num_obstacles == 0:
        return 0.0
    sdfs, _ = _batch_sdf_and_grad(positions, sdf)
    min_sdf = jnp.min(sdfs, axis=1)
    shortfall = jnp.maximum(r_safe - min_sdf, 0.0)
    return jnp.max(shortfall).item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_debug_pairwise(sdf):
    """Debug: show that pairwise KKT solves the QP exactly."""
    print("\n" + "=" * 60)
    print("Debug: Pairwise KKT projection check")
    print("=" * 60)

    R_SAFE = 0.07

    obs_pos = np.array(sdf.positions[0])
    test_pt = obs_pos + np.array([0.0, 0.0], dtype=np.float32)

    p_ref_j = jnp.array(test_pt)
    sdfs, grads = _sdf_and_grad_all(p_ref_j, sdf)

    print(f"  Test point: ({test_pt[0]:.3f}, {test_pt[1]:.3f})")
    print(f"  Min SDF at test point: {float(jnp.min(sdfs)):.4f}")
    print(f"  Violated: {int(jnp.sum(sdfs < R_SAFE))} / {sdf.num_obstacles} constraints")

    # Pairwise KKT
    p_proj = _project_one_pairwise(p_ref_j, sdfs, grads, R_SAFE)
    p_proj_np = np.array(p_proj)
    sdfs_proj, _ = _sdf_and_grad_all(jnp.array(p_proj_np), sdf)
    min_sdf_proj = float(jnp.min(sdfs_proj))
    print(f"  Pairwise KKT:    min SDF={min_sdf_proj:.5f}  safe={min_sdf_proj >= R_SAFE - 1e-4}  "
          f"||Δp||={np.linalg.norm(p_proj_np - test_pt):.4f}")

    # Greedy iterative (existing method)
    plans = np.zeros((1, 1, 12), dtype=np.float32)
    for f in range(4):
        plans[0, 0, f*3:f*3+2] = test_pt
        plans[0, 0, f*3+2] = 0.1
    proj_iter = np.array(project_plan_feet(plans, sdf, R_SAFE, n_iters=3))
    after_iter = proj_iter[0, 0, :2]
    sdfs_iter, _ = _sdf_and_grad_all(jnp.array(after_iter), sdf)
    min_sdf_iter = float(jnp.min(sdfs_iter))
    print(f"  Greedy iterative: min SDF={min_sdf_iter:.5f}  safe={min_sdf_iter >= R_SAFE - 1e-4}  "
          f"||Δp||={np.linalg.norm(after_iter - test_pt):.4f}")

    # Test sdf satisfaction (linearized)
    v = R_SAFE - sdfs
    b_rhs = jnp.einsum("mj,j->m", grads, p_ref_j) + R_SAFE - sdfs
    lin_check = grads @ p_proj >= b_rhs - 1e-5
    print(f"  Linearized check at proj: {int(jnp.sum(~lin_check))} / {sdf.num_obstacles} violated")


def test_correctness_on_grid(sdf):
    """Compare QP output vs iterative method on a dense grid.

    Both should produce nearly identical results (≤1mm differences) since
    the iterative method already does 3 greedy projection iterations which
    approximate the QP solution well for 1‑constraint violations.
    """
    print("\n" + "=" * 60)
    print("Test: Correctness on grid  (QP single  vs  iterative)")
    print("=" * 60)

    X_MIN, X_MAX = 0.5, 6.5
    Y_MIN, Y_MAX = -0.7, 0.7
    GRID = 0.05
    R_SAFE = 0.07

    grid_x = np.arange(X_MIN, X_MAX + GRID, GRID)
    grid_y = np.arange(Y_MIN, Y_MAX + GRID, GRID)
    XX, YY = np.meshgrid(grid_x, grid_y, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1).astype(np.float32)
    N = pts.shape[0]

    plans = np.zeros((N, 1, 12), dtype=np.float32)
    for f in range(4):
        plans[:, 0, f * 3: f * 3 + 2] = pts
        plans[:, 0, f * 3 + 2] = 0.1

    # Run both methods
    proj_iter = np.array(project_plan_feet(plans, sdf, R_SAFE, n_iters=3))
    proj_qp_s = np.array(project_plan_feet_qp(plans, sdf, R_SAFE))
    proj_qp_sqp = np.array(project_plan_feet_qp_sqp(plans, sdf, R_SAFE, n_outer=2))
    proj_qp_nest = np.array(project_plan_feet_qp(plans, sdf, R_SAFE))

    after_iter = proj_iter[:, 0, :2]
    after_qp_s = proj_qp_s[:, 0, :2]
    after_qp_sqp = proj_qp_sqp[:, 0, :2]
    after_qp_nest = proj_qp_nest[:, 0, :2]

    # Violation rates
    v_iter = _violation_rate(after_iter, sdf, R_SAFE)
    v_qp_s = _violation_rate(after_qp_s, sdf, R_SAFE)
    v_qp_sqp = _violation_rate(after_qp_sqp, sdf, R_SAFE)
    v_qp_nest = _violation_rate(after_qp_nest, sdf, R_SAFE)

    # Max diff from iterative
    diff_s = np.max(np.linalg.norm(after_qp_s - after_iter, axis=1))
    diff_sqp = np.max(np.linalg.norm(after_qp_sqp - after_iter, axis=1))
    diff_nest = np.max(np.linalg.norm(after_qp_nest - after_iter, axis=1))

    print(f"  {N} grid points ({grid_x.size}×{grid_y.size}), R_safe={R_SAFE}")
    print(f"  M={sdf.num_obstacles} obstacles")
    print()
    print(f"  {'Method':<18} {'violation rate':>14} {'max Δ vs iterative':>22}")
    print(f"  {'─'*18} {'─'*14} {'─'*22}")
    print(f"  {'iterative (3×SDF)':<18} {v_iter:>14.6f} {'(baseline)':>22}")
    print(f"  {'QP single':<18} {v_qp_s:>14.6f} {diff_s:>22.6f}")
    print(f"  {'QP SQP':<18} {v_qp_sqp:>14.6f} {diff_sqp:>22.6f}")
    print(f"  {'QP Nesterov':<18} {v_qp_nest:>14.6f} {diff_nest:>22.6f}")

    best_viol = min(v_qp_s, v_qp_sqp)
    if best_viol <= v_iter + 1e-4:
        print(f"\n  ✓ QP methods match or beat iterative method on violation rate")
    else:
        print(f"\n  ⚠️  QP methods have higher violation rate than iterative")

    if diff_s < 0.001:
        print(f"  ✓ QP single matches iterative within <1mm")
    else:
        print(f"  ⚠️  QP single deviates from iterative by {diff_s*1000:.1f}mm")


def test_corner_case(sdf):
    """Test point squeezed between two obstacles — where greedy oscillates."""
    print("\n" + "=" * 60)
    print("Test: Corner case  (point between two obstacles)")
    print("=" * 60)

    R_SAFE = 0.07

    # Find two nearby obstacles for the corner test
    positions = np.array(sdf.positions)
    radii = np.array(sdf.half_extents[:, 0])

    # Pick obstacles 2 and 21 (from cluttered scene — two boxes near (4.0, 0.0)
    # and (3.0, 0.0))
    if sdf.num_obstacles < 3:
        print("  ⚠️  Not enough obstacles — skipping")
        return

    # Place a point exactly between two close obstacles
    obs_a = positions[2]  # ~ (3.0, 0.0)
    obs_b = positions[5]  # ~ (4.0, 0.0)

    midpoint = (obs_a + obs_b) / 2.0
    # Push into the gap — this foot is deep between two boxes
    pt = midpoint + np.array([0.0, 0.0], dtype=np.float32)

    plans = np.zeros((1, 1, 12), dtype=np.float32)
    for f in range(4):
        plans[0, 0, f * 3: f * 3 + 2] = pt
        plans[0, 0, f * 3 + 2] = 0.1

    proj_iter = np.array(project_plan_feet(plans, sdf, R_SAFE, n_iters=3))
    proj_qp_s = np.array(project_plan_feet_qp(plans, sdf, R_SAFE))
    proj_qp_sqp = np.array(project_plan_feet_qp_sqp(plans, sdf, R_SAFE, n_outer=2))

    after_iter = proj_iter[0, 0, :2]
    after_qp_s = proj_qp_s[0, 0, :2]
    after_qp_sqp = proj_qp_sqp[0, 0, :2]

    sdfs_iter, _ = _batch_sdf_and_grad(after_iter[None, :], sdf)
    sdfs_qp_s, _ = _batch_sdf_and_grad(after_qp_s[None, :], sdf)
    sdfs_qp_sqp, _ = _batch_sdf_and_grad(after_qp_sqp[None, :], sdf)

    min_iter = float(np.min(sdfs_iter))
    min_qp_s = float(np.min(sdfs_qp_s))
    min_qp_sqp = float(np.min(sdfs_qp_sqp))

    print(f"  Test point: ({pt[0]:.2f}, {pt[1]:.2f})")
    print(f"  Obstacle A: ({obs_a[0]:.2f}, {obs_a[1]:.2f}) r={radii[2]:.3f}")
    print(f"  Obstacle B: ({obs_b[0]:.2f}, {obs_b[1]:.2f}) r={radii[5]:.3f}")
    print()
    print(f"  {'Method':<18} {'min SDF':>10} {'safe?':>8} {'correction norm':>16}")
    print(f"  {'─'*18} {'─'*10} {'─'*8} {'─'*16}")
    print(f"  {'iterative':<18} {min_iter:>10.4f} {str(min_iter>=R_SAFE):>8} "
          f"{np.linalg.norm(after_iter - pt):>16.4f}")
    print(f"  {'QP single':<18} {min_qp_s:>10.4f} {str(min_qp_s>=R_SAFE):>8} "
          f"{np.linalg.norm(after_qp_s - pt):>16.4f}")
    print(f"  {'QP SQP':<18} {min_qp_sqp:>10.4f} {str(min_qp_sqp>=R_SAFE):>8} "
          f"{np.linalg.norm(after_qp_sqp - pt):>16.4f}")

    if min_qp_sqp >= R_SAFE - 1e-3:
        print(f"\n  ✓ QP SQP achieves clearance")
    if min_iter >= R_SAFE - 1e-3:
        print(f"  ✓ Iterative achieves clearance")


# ---------------------------------------------------------------------------
# Speed benchmarks
# ---------------------------------------------------------------------------

def _time_call(fn, *args, warmup=3, trials=10):
    """Time a function (which has internal JIT) by calling and blocking.

    Args:
        fn: The function to time (should be internally JIT-compiled).
        *args: Positional arguments passed directly.
    """
    for _ in range(warmup):
        _ = fn(*args)
        if hasattr(_, "block_until_ready"):
            _.block_until_ready()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        result = fn(*args)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000  # ms


def benchmark(sdf, r_safe=0.07):
    """Speed comparison with realistic MPPI shapes."""
    print("\n" + "=" * 60)
    print("Speed Benchmark  (cluttered scene, M=%d obstacles)" % sdf.num_obstacles)
    print("=" * 60)

    # NOTE: project_plan_feet, project_plan_feet_qp etc. are already
    # JIT-compiled internally (via vmap + jit on inner kernels), so we time
    # the raw Python call which dispatches to pre-compiled XLA.
    # The SDF object argument is handled via Python closure in the inner jit.

    from safety_filter.quad.foot_projection import project_plan_feet as _proj_iter

    # ---- Small batch (1 replica) ----
    N_SMALL = 300
    H = 6
    plans_small = np.zeros((N_SMALL, H, 12), dtype=np.float32)
    rng = np.random.RandomState(42)
    for i in range(N_SMALL):
        # Random foot positions in cluttered zone
        x = rng.uniform(0.8, 6.0)
        y = rng.uniform(-0.5, 0.5)
        for h in range(H):
            for f in range(4):
                fx = x + rng.normal(0, 0.05)
                fy = y + rng.normal(0, 0.05)
                plans_small[i, h, f * 3: f * 3 + 2] = [fx, fy]
                plans_small[i, h, f * 3 + 2] = 0.02 + rng.uniform(0, 0.05)

    N_feet_small = N_SMALL * H * 4  # 7200

    print(f"\n  Small batch: {N_SMALL} samples × {H} horizon × 4 feet = {N_feet_small} feet")
    print(f"  M={sdf.num_obstacles} obstacles")
    print()

    # Time iterative method (baseline)
    t_iter_s, s_iter_s = _time_call(_proj_iter, plans_small, sdf, r_safe, 3)
    print(f"  {'Method':<24} {'time (ms)':>10} {'±std':>8} {'SDF evals':>10}")
    print(f"  {'─'*24} {'─'*10} {'─'*8} {'─'*10}")
    print(f"  {'iterative (3×SDF)':<24} {t_iter_s:>10.3f} {s_iter_s:>8.3f} {'3':>10}")

    t_qp_s, s_qp_s = _time_call(project_plan_feet_qp, plans_small, sdf, r_safe)
    print(f"  {'QP single (1×SDF)':<24} {t_qp_s:>10.3f} {s_qp_s:>8.3f} {'1':>10}")

    t_qp_sqp, s_qp_sqp = _time_call(project_plan_feet_qp_sqp, plans_small, sdf, r_safe, 2)
    print(f"  {'QP SQP (2×SDF)':<24} {t_qp_sqp:>10.3f} {s_qp_sqp:>8.3f} {'2':>10}")

    t_qp_nest, s_qp_nest = _time_call(project_plan_feet_qp, plans_small, sdf, r_safe)
    print(f"  {'QP Nesterov (1×SDF)':<24} {t_qp_nest:>10.3f} {s_qp_nest:>8.3f} {'1':>10}")

    # ---- Large batch (10 replicas, REMIMPS) ----
    K = 10
    N_LARGE = K * N_SMALL
    plans_large = np.tile(plans_small, (K, 1, 1))  # (3000, 6, 12)
    N_feet_large = N_LARGE * H * 4  # 72000

    print(f"\n  Large batch: {N_LARGE} samples × {H} horizon × 4 feet = {N_feet_large} feet")
    print()

    t_iter_l, s_iter_l = _time_call(_proj_iter, plans_large, sdf, r_safe, 3)
    print(f"  {'iterative (3×SDF)':<24} {t_iter_l:>10.3f} {s_iter_l:>8.3f} {'3':>10}")

    t_qp_l, s_qp_l = _time_call(project_plan_feet_qp, plans_large, sdf, r_safe)
    print(f"  {'QP single (1×SDF)':<24} {t_qp_l:>10.3f} {s_qp_l:>8.3f} {'1':>10}")

    t_qp_sqp_l, s_qp_sqp_l = _time_call(project_plan_feet_qp_sqp, plans_large, sdf, r_safe, 2)
    print(f"  {'QP SQP (2×SDF)':<24} {t_qp_sqp_l:>10.3f} {s_qp_sqp_l:>8.3f} {'2':>10}")

    t_qp_nest_l, s_qp_nest_l = _time_call(project_plan_feet_qp, plans_large, sdf, r_safe)
    print(f"  {'QP Nesterov (1×SDF)':<24} {t_qp_nest_l:>10.3f} {s_qp_nest_l:>8.3f} {'1':>10}")

    # ---- Summary ----
    print(f"\n  {'═══ Summary (large batch) ═══':>48}")
    speedup_s = t_iter_l / t_qp_l
    speedup_sqp = t_iter_l / t_qp_sqp_l
    speedup_nest = t_iter_l / t_qp_nest_l
    print(f"  Speedup vs iterative:")
    print(f"    QP single:    {speedup_s:.2f}×  (1 SDF eval vs 3)")
    print(f"    QP SQP:       {speedup_sqp:.2f}×  (2 SDF evals vs 3)")
    print(f"    QP Nesterov:  {speedup_nest:.2f}×  (1 SDF eval vs 3)")

    # Verify correctness on large batch (spot check)
    proj_iter = np.array(_proj_iter(plans_large, sdf, r_safe, 3))
    proj_qp = np.array(project_plan_feet_qp(plans_large, sdf, r_safe))
    proj_qp_sqp = np.array(project_plan_feet_qp_sqp(plans_large, sdf, r_safe, 2))

    flat_iter = proj_iter.reshape(-1, 2)
    flat_qp = proj_qp.reshape(-1, 2)
    flat_qp_sqp = proj_qp_sqp.reshape(-1, 2)

    v_iter = _violation_rate(flat_iter, sdf, r_safe)
    v_qp = _violation_rate(flat_qp, sdf, r_safe)
    v_qp_sqp_f = _violation_rate(flat_qp_sqp, sdf, r_safe)
    max_v_iter = _max_violation(flat_iter, sdf, r_safe)
    max_v_qp = _max_violation(flat_qp, sdf, r_safe)
    max_v_qp_sqp = _max_violation(flat_qp_sqp, sdf, r_safe)

    print(f"\n  {'═══ Violation check ═══':>48}")
    print(f"  {'Method':<24} {'violation rate':>14} {'max shortfall (m)':>18}")
    print(f"  {'─'*24} {'─'*14} {'─'*18}")
    print(f"  {'iterative':<24} {v_iter:>14.6f} {max_v_iter:>18.6f}")
    print(f"  {'QP single':<24} {v_qp:>14.6f} {max_v_qp:>18.6f}")
    print(f"  {'QP SQP':<24} {v_qp_sqp_f:>14.6f} {max_v_qp_sqp:>18.6f}")

    return {
        "small": {
            "iterative": t_iter_s,
            "qp_single": t_qp_s,
            "qp_sqp": t_qp_sqp,
            "qp_nest": t_qp_nest,
        },
        "large": {
            "iterative": t_iter_l,
            "qp_single": t_qp_l,
            "qp_sqp": t_qp_sqp_l,
            "qp_nest": t_qp_nest_l,
        },
        "violations": {
            "iterative": (v_iter, max_v_iter),
            "qp_single": (v_qp, max_v_qp),
            "qp_sqp": (v_qp_sqp_f, max_v_qp_sqp),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("QP Foot Projection — Benchmark & Correctness")
    print("=" * 60)

    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    os.chdir(src_dir)
    print(f"Working directory: {src_dir}")

    # Load cluttered scene SDF
    print("\nLoading cluttered indoor scene...")
    sdf = load_cluttered_sdf()
    print(f"  Extracted {sdf.num_obstacles} obstacles")
    print(f"  Types: BOX={int((sdf.types == BOX).sum())}, "
          f"CIRCLE={int((sdf.types == CIRCLE).sum())}, "
          f"CAPSULE={int((sdf.types == CAPSULE).sum())}")

    # Tests
    test_debug_pairwise(sdf)
    test_correctness_on_grid(sdf)
    test_corner_case(sdf)

    # Benchmark
    results = benchmark(sdf)

    print("\n" + "=" * 60)
    print("Done. Key findings:")
    print("=" * 60)
    if results["large"]["qp_single"] < results["large"]["iterative"]:
        gain = results["large"]["iterative"] / results["large"]["qp_single"]
        print(f"  ✓ QP single is {gain:.1f}× faster than iterative "
              f"({results['large']['qp_single']:.1f}ms vs {results['large']['iterative']:.1f}ms)")
    else:
        ratio = results["large"]["qp_single"] / results["large"]["iterative"]
        print(f"  QP single is {ratio:.1f}× slower than iterative "
              f"({results['large']['qp_single']:.1f}ms vs {results['large']['iterative']:.1f}ms)")


if __name__ == "__main__":
    main()
