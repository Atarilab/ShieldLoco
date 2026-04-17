"""
Test foot_projection.py in isolation.

Tests the SDF evaluation and projection functions by sampling a grid of points
and verifying:
    1. SDF values are computed correctly (sign matches inside/outside)
    2. Projection pushes points out of obstacles to the boundary
    3. Arc checking detects crossing arcs
    4. Full projection with arc check works end-to-end

Run from src/:
    python safety_filter/tests/test_foot_projection.py

Outputs:
    Printed diagnostics, assertions, and visualization plots.
    - sdf_heatmap.png   : SDF values color-coded (blue=inside, yellow=outside, green=boundary)
    - projection_test.png: Before/after foot positions for projection tests
    - arc_checking.png  : Arc crossing visualization with corrections
"""

import os
import sys
import numpy as np
import mujoco

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.collections as mcoll
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not available — skipping plots")

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from safety_filter.quad.foot_projection import (
    ObstacleSDF,
    _sdf_and_grad_one,
    _sdf_and_grad_all,
    _batch_sdf_and_grad,
    project_plan_feet,
    check_and_fix_swing_arc,
    project_plan_feet_with_arc_check,
    CIRCLE, BOX, CAPSULE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCENE_XML = "robots/go2/scene.xml"

# Grid bounds matching the obstacle course
X_MIN, X_MAX = 0.0, 7.0
Y_MIN, Y_MAX = -0.8, 0.8
GRID_RES = 0.02  # 2 cm for good visualization quality

# Clearance used in projection (must match what the safety filter uses)
R_SAFE = 0.05  # projection_clearance from rl_mps_go2_cfg.yaml

# Foot radius for reference
FOOT_RADIUS = 0.022

# Zone definitions (from obstacles.xml comments)
ZONES = [
    ("Zone 0", 0.0, 0.7),
    ("Zone 1", 0.7, 1.6),
    ("Zone 2", 1.8, 3.0),
    ("Zone 3", 3.2, 4.5),
    ("Zone 4", 4.6, 6.0),
    ("Post",   6.0, 7.0),
]

# ---------------------------------------------------------------------------
# Helper: reference NumPy SDF implementations for validation
# ---------------------------------------------------------------------------

def _quat_to_yaw(quat_wxyz: np.ndarray) -> float:
    """Extract yaw from wxyz quaternion."""
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _rot2d(yaw: float) -> np.ndarray:
    """2x2 rotation matrix for given yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def reference_sdf_circle(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Reference circle SDF: signed distance to circle edge."""
    return np.linalg.norm(points - center[None, :], axis=-1) - radius


def reference_sdf_box(points: np.ndarray, center: np.ndarray, R2: np.ndarray,
                      half_extents: np.ndarray) -> np.ndarray:
    """Reference box SDF (Inigo Quilez)."""
    d = points - center[None, :]
    d_local = d @ R2
    q = np.abs(d_local) - half_extents
    outer = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inner = np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)
    return outer + inner


# ---------------------------------------------------------------------------
# Test 1: ObstacleSDF extraction
# ---------------------------------------------------------------------------

def test_sdf_extraction(model: mujoco.MjModel):
    """Test that ObstacleSDF is extracted correctly from the model."""
    print("\n" + "=" * 60)
    print("Test 1: ObstacleSDF extraction")
    print("=" * 60)

    sdf = ObstacleSDF.from_mujoco_model(model)

    print(f"  Extracted {sdf.num_obstacles} obstacles")
    assert sdf.num_obstacles > 0, "No obstacles extracted! Check hazard_prefixes and min_height."

    # Check arrays have correct shapes
    assert sdf.positions.shape[1] == 2, "positions should be (M, 2)"
    assert sdf.rotations.shape[1] == 2, "rotations should be (M, 2, 2)"
    assert sdf.rotations.shape[2] == 2, "rotations should be (M, 2, 2)"
    assert sdf.half_extents.shape[1] == 2, "half_extents should be (M, 2)"
    assert sdf.types.shape[0] == sdf.num_obstacles, "types should be (M,)"

    # Check types are valid
    assert all(t in (CIRCLE, BOX, CAPSULE) for t in sdf.types), "Invalid obstacle types found"

    print(f"  ✓ Positions shape: {sdf.positions.shape}")
    print(f"  ✓ Rotations shape: {sdf.rotations.shape}")
    print(f"  ✓ Half extents shape: {sdf.half_extents.shape}")
    print(f"  ✓ Types: {dict(zip(*np.unique(sdf.types, return_counts=True)))}")

    return sdf


# ---------------------------------------------------------------------------
# Test 2: SDF correctness on grid
# ---------------------------------------------------------------------------

def test_sdf_on_grid(model: mujoco.MjModel, sdf: ObstacleSDF):
    """Test SDF values against reference implementations at grid points."""
    print("\n" + "=" * 60)
    print("Test 2: SDF correctness on grid")
    print("=" * 60)

    grid_x = np.arange(X_MIN, X_MAX + GRID_RES, GRID_RES)
    grid_y = np.arange(Y_MIN, Y_MAX + GRID_RES, GRID_RES)
    XX, YY = np.meshgrid(grid_x, grid_y, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)  # (N, 2)

    # Compute SDFs using our JAX implementation
    sdfs_jax, grads_jax = _batch_sdf_and_grad(pts, sdf)
    min_sdfs_jax = np.min(sdfs_jax, axis=1)  # (N,)

    # Compute reference SDFs using NumPy
    sdfs_ref = np.full(pts.shape[0], np.inf)
    for i in range(sdf.num_obstacles):
        pos = np.array(sdf.positions[i])
        rot = np.array(sdf.rotations[i])
        he = np.array(sdf.half_extents[i])
        typ = int(sdf.types[i])

        if typ == CIRCLE:
            sdf_i = reference_sdf_circle(pts, pos, he[0])
        elif typ == BOX:
            sdf_i = reference_sdf_box(pts, pos, rot, he)
        else:
            # Capsule approximation: treat as circle
            sdf_i = reference_sdf_circle(pts, pos, he[0])

        sdfs_ref = np.minimum(sdfs_ref, sdf_i)

    # Compare
    errors = np.abs(min_sdfs_jax - sdfs_ref)
    max_err = np.max(errors)
    mean_err = np.mean(errors)

    print(f"  Grid points: {pts.shape[0]}")
    print(f"  Max absolute error: {max_err:.6f} m")
    print(f"  Mean absolute error: {mean_err:.6f} m")

    if max_err > 1e-4:
        print(f"  ⚠️  Large errors detected! Max error {max_err:.6f} m")
        worst_idx = np.argmax(errors)
        print(f"  Worst point: {pts[worst_idx]}, jax={min_sdfs_jax[worst_idx]:.4f}, ref={sdfs_ref[worst_idx]:.4f}")
    else:
        print(f"  ✓ SDF values match reference implementation")

    inside_mask = sdfs_ref < -FOOT_RADIUS
    if np.any(inside_mask):
        jax_signs = np.sign(min_sdfs_jax[inside_mask])
        ref_signs = np.sign(sdfs_ref[inside_mask])
        sign_match = np.all(jax_signs == ref_signs)
        print(f"  ✓ Inside-obstacle signs match: {sign_match}")

    return pts, min_sdfs_jax, XX, YY, grid_x, grid_y


# ---------------------------------------------------------------------------
# Test 3: Projection correctness
# ---------------------------------------------------------------------------

def test_projection_on_grid(sdf: ObstacleSDF):
    """Test project_plan_feet on a dense grid of points across the obstacle course.

    Creates one plan per grid point (H=1, all 4 feet at that point), projects
    all plans in one batched call, then verifies every projected foot satisfies
    SDF >= R_SAFE.
    """
    print("\n" + "=" * 60)
    print("Test 3: Projection on grid")
    print("=" * 60)

    GRID_COARSE = 0.1  # 10 cm — dense enough for visual clarity, fast enough to run
    grid_x = np.arange(X_MIN, X_MAX + GRID_COARSE, GRID_COARSE)
    grid_y = np.arange(Y_MIN, Y_MAX + GRID_COARSE, GRID_COARSE)
    XX, YY = np.meshgrid(grid_x, grid_y, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1).astype(np.float32)  # (N, 2)
    N = pts.shape[0]

    # Build plans: (N, H=1, 12) — all 4 feet at the same grid point per plan
    plans = np.zeros((N, 1, 12), dtype=np.float32)
    for f in range(4):
        plans[:, 0, f * 3 : f * 3 + 2] = pts
        plans[:, 0, f * 3 + 2] = 0.1  # z

    print(f"  Grid: {N} points ({grid_x.size} × {grid_y.size}), projecting...")
    projected = np.array(project_plan_feet(plans, sdf, R_SAFE))

    # Use foot 0 (FL) for evaluation — all 4 are identical so one suffices
    before_xy = plans[:, 0, 0:2]      # (N, 2)
    after_xy = projected[:, 0, 0:2]   # (N, 2)

    if sdf.num_obstacles > 0:
        sdfs_before, _ = _batch_sdf_and_grad(before_xy, sdf)  # (N, M)
        min_sdf_before = np.min(sdfs_before, axis=1)           # (N,)
        inside_mask = min_sdf_before < 0.0                      # geometrically inside obstacle

        sdfs_after, _ = _batch_sdf_and_grad(after_xy, sdf)
        min_sdf_after = np.min(sdfs_after, axis=1)
        violation_idx = np.where(min_sdf_after < R_SAFE - 1e-3)[0]

        print(f"  Inside obstacle (SDF < 0): {int(np.sum(inside_mask))} / {N}")
        print(f"  Violations remaining after projection: {len(violation_idx)}")
        for i in violation_idx[:5]:
            print(f"    Point {pts[i]}: SDF before={min_sdf_before[i]:.4f}, after={min_sdf_after[i]:.4f}")
        if len(violation_idx) == 0:
            print(f"  ✓ All projected points are safe")
        else:
            print(f"  ⚠️  {len(violation_idx)} points still inside after projection")
    else:
        inside_mask = np.zeros(N, dtype=bool)
        min_sdf_before = np.full(N, np.inf, dtype=np.float32)
        min_sdf_after = np.full(N, np.inf, dtype=np.float32)
        print(f"  ✓ No obstacles — projection is a no-op")

    return {
        "before": before_xy,
        "after": after_xy,
        "inside_mask": inside_mask,
        "min_sdf_before": min_sdf_before,
        "min_sdf_after": min_sdf_after,
    }


# ---------------------------------------------------------------------------
# Test 4: Arc checking
# ---------------------------------------------------------------------------

def test_arc_checking(sdf: ObstacleSDF):
    """Test check_and_fix_swing_arc for various scenarios.

    Note: check_and_fix_swing_arc processes 4 feet at once (shape (4, 2)),
    so we test with properly shaped arrays. This tests the full vmapped
    version used in project_plan_feet_with_arc_check.
    """
    print("\n" + "=" * 60)
    print("Test 4: Arc checking")
    print("=" * 60)

    arc_results = {}

    if sdf.num_obstacles == 0:
        print("  ⚠️  No obstacles, skipping arc check test")
        return arc_results

    # Find an obstacle center
    obs_idx = 0
    center = np.array(sdf.positions[obs_idx])
    radius = float(sdf.half_extents[obs_idx][0])

    # check_and_fix_swing_arc expects (4, 2) - one row per foot
    # All 4 feet start at same position for this simple test
    foot_starts = np.tile(center + np.array([-(radius + 0.2), 0.0]), (4, 1))  # (4, 2)
    foot_ends = np.tile(center + np.array([radius + 0.2, 0.0]), (4, 1))        # (4, 2)

    # Test case 1: Arc that CLEARLY crosses obstacle
    # Note: check_and_fix_swing_arc is already JAX-compiled internally via vmap/scan
    # so we can call it directly
    p_corrected = np.array(check_and_fix_swing_arc(foot_starts, foot_ends, sdf, R_SAFE))

    arc_length = np.linalg.norm(foot_ends[0] - foot_starts[0])
    correction_magnitude = np.linalg.norm(p_corrected[0] - foot_ends[0])
    print(f"  Crossing arc test (4 feet, all same):")
    print(f"    Arc length: {arc_length:.3f} m")
    print(f"    Correction magnitude: {correction_magnitude:.4f} m")

    arc_results["crossing"] = {
        "from": foot_starts[0], "to": foot_ends[0], "corrected": p_corrected[0],
        "obstacle_center": center, "obstacle_radius": radius,
    }

    if correction_magnitude > 0.01:
        print(f"  ✓ Arc was corrected (expected for crossing arc)")
    else:
        print(f"  ⚠️  Arc was NOT corrected but should have been")

    # Test case 2: Arc that CLEARLY does NOT cross
    foot_starts_safe = np.tile(center + np.array([-(radius + 0.3), 0.0]), (4, 1))
    foot_ends_safe = np.tile(center + np.array([-(radius + 0.1), 0.0]), (4, 1))  # stays to the left
    p_corrected_safe = np.array(check_and_fix_swing_arc(foot_starts_safe, foot_ends_safe, sdf, R_SAFE))

    correction_magnitude_safe = np.linalg.norm(p_corrected_safe[0] - foot_ends_safe[0])
    print(f"\n  Safe arc test (no crossing):")
    print(f"    Correction magnitude: {correction_magnitude_safe:.6f} m")

    arc_results["safe"] = {
        "from": foot_starts_safe[0], "to": foot_ends_safe[0], "corrected": p_corrected_safe[0],
        "obstacle_center": center, "obstacle_radius": radius,
    }

    if correction_magnitude_safe < 1e-4:
        print(f"  ✓ Arc was NOT corrected (expected for safe arc)")
    else:
        print(f"  ⚠️  Arc was corrected but should not have been")

    # Test case 3: Arc just grazing the edge
    foot_starts_graze = np.tile(center + np.array([-(radius + 0.15), 0.0]), (4, 1))
    foot_ends_graze = np.tile(center + np.array([radius + 0.15, 0.0]), (4, 1))
    p_corrected_graze = np.array(check_and_fix_swing_arc(foot_starts_graze, foot_ends_graze, sdf, R_SAFE))
    correction_magnitude_graze = np.linalg.norm(p_corrected_graze[0] - foot_ends_graze[0])
    print(f"\n  Grazing arc test:")
    print(f"    Correction magnitude: {correction_magnitude_graze:.4f} m")

    arc_results["grazing"] = {
        "from": foot_starts_graze[0], "to": foot_ends_graze[0], "corrected": p_corrected_graze[0],
        "obstacle_center": center, "obstacle_radius": radius,
    }

    return arc_results


# ---------------------------------------------------------------------------
# Test 5: Full projection with arc check
# ---------------------------------------------------------------------------

def test_projection_with_arc_check(sdf: ObstacleSDF):
    """Test project_plan_feet_with_arc_check on a grid.

    For each grid point (the planned landing), current feet are placed STRIDE
    metres behind in x (simulating forward approach).  Arc correction should
    fire wherever the straight-line swing from current → landing crosses an
    obstacle, even if the landing itself is geometrically safe.
    """
    print("\n" + "=" * 60)
    print("Test 5: Projection with arc check (grid)")
    print("=" * 60)

    GRID_COARSE = 0.1
    STRIDE = 0.3  # approach offset in x

    grid_x = np.arange(X_MIN, X_MAX + GRID_COARSE, GRID_COARSE)
    grid_y = np.arange(Y_MIN, Y_MAX + GRID_COARSE, GRID_COARSE)
    XX, YY = np.meshgrid(grid_x, grid_y, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1).astype(np.float32)  # (N, 2)
    N = pts.shape[0]

    # Plans: all 4 feet landing at grid point, H=1
    plans = np.zeros((N, 1, 12), dtype=np.float32)
    for f in range(4):
        plans[:, 0, f * 3 : f * 3 + 2] = pts
        plans[:, 0, f * 3 + 2] = 0.1

    # Current feet: all 4 feet STRIDE behind in x  →  (N, 4, 2)
    current_pts = pts + np.array([-STRIDE, 0.0], dtype=np.float32)
    current_feet = np.stack([current_pts] * 4, axis=1)  # (N, 4, 2)

    print(f"  Grid: {N} points ({grid_x.size} × {grid_y.size}), stride={STRIDE}m, projecting...")
    projected = np.array(project_plan_feet_with_arc_check(plans, sdf, R_SAFE, current_feet))

    before_xy = plans[:, 0, 0:2]       # planned landing (N, 2)
    after_xy  = projected[:, 0, 0:2]   # arc-corrected landing (N, 2)

    # Arc correction fired wherever the foot actually moved
    arc_fired = np.linalg.norm(after_xy - before_xy, axis=1) > 1e-4

    if sdf.num_obstacles > 0:
        sdfs_after, _ = _batch_sdf_and_grad(after_xy, sdf)
        min_sdf_after = np.min(sdfs_after, axis=1)
        violation_idx = np.where(min_sdf_after < R_SAFE - 1e-3)[0]

        print(f"  Arc correction fired: {int(np.sum(arc_fired))} / {N}")
        print(f"  Violations after correction: {len(violation_idx)}")
        for i in violation_idx[:5]:
            print(f"    Point {pts[i]}: SDF after={min_sdf_after[i]:.4f}")
        if len(violation_idx) == 0:
            print(f"  ✓ All arc-corrected positions are safe")
        else:
            print(f"  ⚠️  {len(violation_idx)} arc corrections still unsafe")
    else:
        min_sdf_after = np.full(N, np.inf, dtype=np.float32)

    return {
        "pts": pts,
        "current_pts": current_pts,
        "before": before_xy,
        "after": after_xy,
        "arc_fired": arc_fired,
        "min_sdf_after": min_sdf_after,
        "stride": STRIDE,
    }


# ---------------------------------------------------------------------------
# Test 6: JAX gradient correctness
# ---------------------------------------------------------------------------

def test_gradient_correctness(sdf: ObstacleSDF):
    """Test that SDF gradients are unit vectors (as they should be)."""
    print("\n" + "=" * 60)
    print("Test 6: Gradient correctness (unit vectors)")
    print("=" * 60)

    grid_x = np.arange(X_MIN, X_MAX + GRID_RES * 2, GRID_RES * 2)
    grid_y = np.arange(Y_MIN, Y_MAX + GRID_RES * 2, GRID_RES * 2)
    XX, YY = np.meshgrid(grid_x, grid_y, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)

    _, grads = _batch_sdf_and_grad(pts, sdf)
    grad_norms = np.linalg.norm(grads, axis=2)  # (N, M)

    sdfs, _ = _batch_sdf_and_grad(pts, sdf)
    nearest_obs = np.argmin(sdfs, axis=1)
    nearest_grad_norms = grad_norms[np.arange(grad_norms.shape[0]), nearest_obs]

    errors = np.abs(nearest_grad_norms - 1.0)
    max_err = np.max(errors)
    mean_err = np.mean(errors)

    print(f"  Sampled {pts.shape[0]} points")
    print(f"  Max gradient magnitude error: {max_err:.6f}")
    print(f"  Mean gradient magnitude error: {mean_err:.6f}")

    if max_err > 1e-4:
        print(f"  ⚠️  Gradients are not unit vectors!")
    else:
        print(f"  ✓ Gradients are unit vectors")


# ---------------------------------------------------------------------------
# Test 7: Edge cases
# ---------------------------------------------------------------------------

def test_edge_cases(sdf: ObstacleSDF):
    """Test edge cases: empty SDF, stationary feet, etc."""
    print("\n" + "=" * 60)
    print("Test 7: Edge cases")
    print("=" * 60)

    # Edge case 1: Empty SDF (no obstacles)
    empty_sdf = ObstacleSDF(
        positions=np.zeros((0, 2), dtype=np.float32),
        rotations=np.zeros((0, 2, 2), dtype=np.float32),
        half_extents=np.zeros((0, 2), dtype=np.float32),
        types=np.zeros((0,), dtype=np.int32),
    )

    plans = np.array([[
        [1.0, 2.0, 0.1,  1.0, 2.0, 0.1,  1.0, 2.0, 0.1,  1.0, 2.0, 0.1],
        [1.0, 2.0, 0.1,  1.0, 2.0, 0.1,  1.0, 2.0, 0.1,  1.0, 2.0, 0.1],
        [1.0, 2.0, 0.1,  1.0, 2.0, 0.1,  1.0, 2.0, 0.1,  1.0, 2.0, 0.1],
    ]])  # (1, H=3, 12)
    current_feet = np.array([
        [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]
    ])  # (4, 2)

    result = project_plan_feet(plans, empty_sdf, R_SAFE)
    assert np.allclose(result, plans), "Empty SDF should return plans unchanged"
    print(f"  ✓ Empty SDF returns plans unchanged")

    result_arc = project_plan_feet_with_arc_check(plans, empty_sdf, R_SAFE, current_feet)
    assert np.allclose(result_arc, plans), "Empty SDF should return plans unchanged with arc check"
    print(f"  ✓ Empty SDF with arc check returns plans unchanged")

    # Edge case 2: Stationary feet (zero displacement)
    # check_and_fix_swing_arc expects (4, 2) - 4 feet at once
    if sdf.num_obstacles > 0:
        # All 4 feet stationary at same position
        stationary_pos = np.array([[3.5, 0.0]] * 4)  # (4, 2)
        p_corrected = np.array(check_and_fix_swing_arc(stationary_pos, stationary_pos, sdf, R_SAFE))
        assert np.allclose(p_corrected, stationary_pos), "Stationary foot should not be corrected"
        print(f"  ✓ Stationary feet are not corrected")

    print(f"  ✓ All edge cases handled correctly")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _get_obstacle_patches(sdf: ObstacleSDF, ax, X_MIN, X_MAX, Y_MIN, Y_MAX):
    """Draw obstacle outlines only (no fill — SDF heatmap already shows interior via color)."""
    for i in range(sdf.num_obstacles):
        pos = sdf.positions[i]
        rot = sdf.rotations[i]
        he = sdf.half_extents[i]
        typ = sdf.types[i]

        if typ == CIRCLE:
            ax.add_patch(plt.Circle(
                (pos[0], pos[1]), he[0],
                fill=False, edgecolor="black", linewidth=0.8, zorder=3,
            ))

        elif typ == BOX:
            from matplotlib.patches import Rectangle
            angle = float(np.arctan2(rot[1, 0], rot[0, 0]))
            w, h = he[0] * 2, he[1] * 2
            ax.add_patch(Rectangle(
                (pos[0] - he[0], pos[1] - he[1]), w, h,
                fill=False, edgecolor="black", linewidth=0.8, zorder=3,
                transform=matplotlib.transforms.Affine2D().rotate_around(pos[0], pos[1], angle)
                + ax.transData,
            ))

        elif typ == CAPSULE:
            r, hl = float(he[0]), float(he[1])
            angle = float(np.arctan2(rot[1, 0], rot[0, 0]))
            c, s = np.cos(angle), np.sin(angle)
            theta = np.linspace(0, np.pi, 24)
            x_loc = np.concatenate([hl + r * np.cos(theta), -hl + r * np.cos(theta + np.pi)])
            y_loc = np.concatenate([r * np.sin(theta), r * np.sin(theta + np.pi)])
            x_w = c * x_loc - s * y_loc + pos[0]
            y_w = s * x_loc + c * y_loc + pos[1]
            ax.plot(np.append(x_w, x_w[0]), np.append(y_w, y_w[0]),
                    color="black", linewidth=0.8, zorder=3)


def plot_sdf_heatmap(sdfs_grid: np.ndarray, XX: np.ndarray, YY: np.ndarray,
                      sdf: ObstacleSDF, grid_x: np.ndarray, grid_y: np.ndarray,
                      src_dir: str):
    """Plot SDF as a heatmap with obstacle outlines and clearance boundaries."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(1, 1, figsize=(16, 4))

    # Reshape SDF grid to match meshgrid output
    # XX, YY have shape (Nx, Ny) from meshgrid(..., indexing='ij')
    sdf_grid = sdfs_grid.reshape(XX.shape)

    # For imshow: data shape (Ny, Nx) = (rows, cols)
    # imshow shows data.T since our grid is (Nx, Ny) but imshow expects (Ny, Nx)
    vmax = 0.5  # Clip display at 0.5m
    im = ax.imshow(
        sdf_grid.T, origin="lower", aspect="auto",
        extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
        cmap="RdYlGn_r", vmin=-vmax, vmax=vmax,
        alpha=0.8,
    )
    cbar = plt.colorbar(im, ax=ax, label="SDF (m)")

    # Draw R_SAFE boundary in orange
    # For contour, we need XX, YY, data all with same shape
    # XX, YY are (Nx, Ny), sdf_grid is (Nx, Ny), but contour expects (Ny, Nx) if using xy indexing
    # So we pass sdf_grid (not transposed) since contour will handle the extent
    contour_levels = [0.0, R_SAFE]
    cs = ax.contour(
        XX, YY, sdf_grid,
        levels=contour_levels,
        colors=["green", "orange"],
        linewidths=[1.5, 1.0],
        linestyles=["solid", "dashed"],
        extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt={0.0: "SDF=0", R_SAFE: f"SDF=R_safe={R_SAFE}"})

    # Draw obstacles
    _get_obstacle_patches(sdf, ax, X_MIN, X_MAX, Y_MIN, Y_MAX)

    # Zone boundaries
    for _, z_x0, z_x1 in ZONES:
        ax.axvline(z_x0, color="blue", linewidth=0.5, linestyle="--", alpha=0.3)

    ax.set_title(f"SDF Heatmap (green=SDF>0 safe, red=SDF<0 inside) | R_safe={R_SAFE}m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)

    plt.tight_layout()
    out_path = os.path.join(src_dir, "sdf_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved SDF heatmap → {out_path}")
    plt.close(fig)


def plot_projection_test(proj_results: dict, sdf: ObstacleSDF, src_dir: str):
    """Plot grid projection test: before/after positions for every inside point."""
    if not HAS_MPL or not proj_results:
        return

    fig, ax = plt.subplots(1, 1, figsize=(16, 4))

    # SDF background
    grid_x = np.arange(X_MIN, X_MAX + GRID_RES, GRID_RES)
    grid_y = np.arange(Y_MIN, Y_MAX + GRID_RES, GRID_RES)
    XX, YY = np.meshgrid(grid_x, grid_y, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    sdfs_jax, _ = _batch_sdf_and_grad(pts, sdf)
    min_sdfs = np.min(sdfs_jax, axis=1).reshape(XX.shape)

    im = ax.imshow(
        min_sdfs.T, origin="lower", aspect="auto",
        extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
        cmap="RdYlGn_r", vmin=-0.3, vmax=0.3, alpha=0.5,
    )
    plt.colorbar(im, ax=ax, label="SDF (m)")

    # R_SAFE contour (single source of truth for the clearance boundary)
    ax.contour(XX, YY, min_sdfs, levels=[R_SAFE],
               colors=["orange"], linewidths=[1.0], linestyles=["--"])

    _get_obstacle_patches(sdf, ax, X_MIN, X_MAX, Y_MIN, Y_MAX)

    before = proj_results["before"]          # (N, 2)
    after = proj_results["after"]            # (N, 2)
    inside_mask = proj_results["inside_mask"]  # (N,) bool

    inside_before = before[inside_mask]   # (K, 2)
    inside_after  = after[inside_mask]    # (K, 2)

    # Arrows: before → after (sub-sample if too many)
    step = max(1, len(inside_before) // 200)
    for i in range(0, len(inside_before), step):
        ax.annotate(
            "", xy=inside_after[i], xytext=inside_before[i],
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.8, alpha=0.6),
            zorder=10,
        )

    n_inside = int(np.sum(inside_mask))
    ax.scatter(inside_before[:, 0], inside_before[:, 1],
               c="red", s=15, marker="x", label=f"Before (SDF<0, {n_inside} pts)", zorder=11)
    ax.scatter(inside_after[:, 0], inside_after[:, 1],
               c="limegreen", s=15, marker="o", label="After (projected)", zorder=11)

    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Projection Test: grid ({len(before)} pts, 10 cm spacing) — SDF<0 → Projected")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)

    plt.tight_layout()
    out_path = os.path.join(src_dir, "projection_test.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved projection test → {out_path}")
    plt.close(fig)


def plot_arc_checking(arc_results: dict, sdf: ObstacleSDF, src_dir: str):
    """Plot arc crossing visualization with corrections."""
    if not HAS_MPL or not arc_results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (key, test) in enumerate(arc_results.items()):
        ax = axes[idx]
        center = test["obstacle_center"]
        radius = test["obstacle_radius"]
        p_from = test["from"]
        p_to = test["to"]
        p_corrected = test["corrected"]

        # Draw SDF background
        grid_x = np.arange(X_MIN, X_MAX + GRID_RES, GRID_RES)
        grid_y = np.arange(Y_MIN, Y_MAX + GRID_RES, GRID_RES)
        XX, YY = np.meshgrid(grid_x, grid_y, indexing="ij")
        pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)
        sdfs_jax, _ = _batch_sdf_and_grad(pts, sdf)
        min_sdfs = np.min(sdfs_jax, axis=1).reshape(XX.shape)

        im = ax.imshow(
            min_sdfs.T, origin="lower", aspect="auto",
            extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
            cmap="RdYlGn_r", vmin=-0.3, vmax=0.3, alpha=0.5,
        )

        # Draw obstacle
        circle = plt.Circle((center[0], center[1]), radius,
                           fill=True, color="gray", alpha=0.3, zorder=2)
        ax.add_patch(circle)
        # Inflated boundary
        inflated = plt.Circle((center[0], center[1]), radius + R_SAFE,
                              fill=False, color="orange", linewidth=1.0, linestyle="--")
        ax.add_patch(inflated)

        # Draw original arc
        ax.plot([p_from[0], p_to[0]], [p_from[1], p_to[1]],
               "r-", linewidth=2, label="Original arc")

        # Draw corrected landing
        ax.plot(p_to[0], p_to[1], "rx", markersize=10, label="Original landing")
        ax.plot(p_corrected[0], p_corrected[1], "go", markersize=10, label="Corrected landing")

        # Draw corrected arc
        ax.plot([p_from[0], p_corrected[0]], [p_from[1], p_corrected[1]],
               "g--", linewidth=1.5, label="Corrected arc")

        # Mark start/end points
        ax.plot(p_from[0], p_from[1], "bs", markersize=8, label="Start")

        ax.set_title(f"Arc test: {key}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(center[0] - 0.6, center[0] + 0.6)
        ax.set_ylim(center[1] - 0.4, center[1] + 0.4)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_aspect("equal")

    plt.tight_layout()
    out_path = os.path.join(src_dir, "arc_checking.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved arc checking visualization → {out_path}")
    plt.close(fig)


def plot_projection_with_arc_e2e(e2e_results: dict, sdf: ObstacleSDF, src_dir: str):
    """Plot arc-correction grid test: planned landings that triggered arc correction."""
    if not HAS_MPL or not e2e_results:
        return

    before     = e2e_results["before"]      # (N, 2) planned landing
    after      = e2e_results["after"]       # (N, 2) arc-corrected landing
    arc_fired  = e2e_results["arc_fired"]   # (N,) bool
    stride     = e2e_results["stride"]

    fig, ax = plt.subplots(1, 1, figsize=(16, 4))

    # SDF background
    grid_x = np.arange(X_MIN, X_MAX + GRID_RES, GRID_RES)
    grid_y = np.arange(Y_MIN, Y_MAX + GRID_RES, GRID_RES)
    XX, YY = np.meshgrid(grid_x, grid_y, indexing="ij")
    bg_pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    sdfs_jax, _ = _batch_sdf_and_grad(bg_pts, sdf)
    min_sdfs = np.min(sdfs_jax, axis=1).reshape(XX.shape)

    im = ax.imshow(
        min_sdfs.T, origin="lower", aspect="auto",
        extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
        cmap="RdYlGn_r", vmin=-0.3, vmax=0.3, alpha=0.5,
    )
    plt.colorbar(im, ax=ax, label="SDF (m)")

    ax.contour(XX, YY, min_sdfs, levels=[R_SAFE],
               colors=["orange"], linewidths=[1.0], linestyles=["--"])
    _get_obstacle_patches(sdf, ax, X_MIN, X_MAX, Y_MIN, Y_MAX)

    fired_before = before[arc_fired]
    fired_after  = after[arc_fired]

    step = max(1, len(fired_before) // 200)
    for i in range(0, len(fired_before), step):
        ax.annotate("", xy=fired_after[i], xytext=fired_before[i],
                    arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.8, alpha=0.6),
                    zorder=10)

    n_fired = int(np.sum(arc_fired))
    ax.scatter(fired_before[:, 0], fired_before[:, 1],
               c="red", s=15, marker="x",
               label=f"Planned landing ({n_fired} arc-corrected)", zorder=11)
    ax.scatter(fired_after[:, 0], fired_after[:, 1],
               c="limegreen", s=15, marker="o",
               label="Arc-corrected landing", zorder=11)

    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"Arc-correction grid test ({len(before)} pts, stride={stride}m) "
        f"— Planned → Near-side when swing crosses obstacle"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)

    plt.tight_layout()
    out_path = os.path.join(src_dir, "projection_e2e.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved arc-correction grid visualization → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Foot Projection Module Tests")
    print("=" * 60)

    # Change to src/ so relative XML paths resolve
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    os.chdir(src_dir)
    print(f"Working directory: {src_dir}")

    print(f"\nLoading scene: {os.path.join(src_dir, SCENE_XML)}")
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    print(f"Model loaded: {model.nq} DOFs, {model.ngeom} geoms")

    # Run tests
    sdf = test_sdf_extraction(model)
    pts, sdfs_grid, XX, YY, grid_x, grid_y = test_sdf_on_grid(model, sdf)
    proj_results = test_projection_on_grid(sdf)
    arc_results = test_arc_checking(sdf)
    e2e_results = test_projection_with_arc_check(sdf)
    test_gradient_correctness(sdf)
    test_edge_cases(sdf)

    # Generate visualizations
    if HAS_MPL:
        print("\n" + "=" * 60)
        print("Generating visualizations...")
        print("=" * 60)

        plot_sdf_heatmap(sdfs_grid, XX, YY, sdf, grid_x, grid_y, src_dir)
        plot_projection_test(proj_results, sdf, src_dir)
        plot_arc_checking(arc_results, sdf, src_dir)
        plot_projection_with_arc_e2e(e2e_results, sdf, src_dir)

        print("\n" + "=" * 60)
        print(f"Visualizations saved to {src_dir}/")
        print("  - sdf_heatmap.png   : SDF color map with obstacle outlines")
        print("  - projection_test.png: Before/after projection for inside points")
        print("  - arc_checking.png  : Crossing vs safe arc examples")
        print("  - projection_e2e.png : End-to-end projection with arc fix")
        print("=" * 60)
    else:
        print("\n[skip] Install matplotlib to generate visualizations")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
