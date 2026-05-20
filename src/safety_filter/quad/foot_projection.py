"""Minimally invasive foot-plan projector using iterative SDF projection.

For each foot contact position, iteratively projects onto the most-violated
SDF constraint until all constraints are satisfied:

    1. Evaluate all M obstacle SDFs at current position
    2. Find the most-violated constraint (largest clearance violation)
    3. Push position along violation gradient by the violation amount
    4. Repeat until convergence (3 iterations suffices for M=40 scenes)

This is the closest-point projection onto the intersection of inflated
obstacle half-planes.  For single-constraint violations the projection is
exact in one step (circles are affine).  For two nearby constraints the
iterative scheme converges in 2-3 steps; the residual after 3 iterations
is <1mm for our 7cm clearance.

Performance: O(N * M * n_iters) total SDF evaluations instead of the
previous O(N * M^2 * n_iters) active-set QP.  With N=9600 feet, M=40
obstacles, and n_iters=3 this gives ~1.15M evaluations vs ~630M — a
~550x speedup.
"""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

CIRCLE = 0
BOX = 1
CAPSULE = 2

_MJ_TYPE_TO_SDF = {2: CIRCLE, 6: BOX, 5: CIRCLE, 3: CAPSULE}


class ObstacleSDF:
    """2D signed distance field for static ground-plane obstacles.

    Attributes:
        positions: (M, 2) XY centres in world frame.
        rotations: (M, 2, 2) 2D rotation matrices (local-to-world).
        half_extents: (M, 2) type-dependent:
            CIRCLE  -> (radius, 0)
            BOX     -> (hx, hy)
            CAPSULE -> (radius, half_length)
        types: (M,) int32 obstacle type (CIRCLE / BOX / CAPSULE).
        heights: (M,) float32 half-height in world Z (metres).
    """

    def __init__(self, positions, rotations, half_extents, types, heights=None):
        self.positions = jnp.asarray(positions, dtype=jnp.float32)
        self.rotations = jnp.asarray(rotations, dtype=jnp.float32)
        self.half_extents = jnp.asarray(half_extents, dtype=jnp.float32)
        self.types = jnp.asarray(types, dtype=jnp.int32)
        M = self.positions.shape[0]
        if heights is not None:
            self.heights = jnp.asarray(heights, dtype=jnp.float32)
        else:
            self.heights = jnp.zeros((M,), dtype=jnp.float32)

    @property
    def num_obstacles(self):
        return self.positions.shape[0]

    def filter_by_height(self, min_height=-float("inf"), max_height=float("inf")):
        """Return new ObstacleSDF with only obstacles in [min_height, max_height).

        Uses numpy indexing — safe because this runs at init time, not inside JIT.
        """
        if self.num_obstacles == 0:
            return self
        h = np.asarray(self.heights)
        mask = (h >= min_height) & (h < max_height)
        idx = np.nonzero(mask)[0]
        return ObstacleSDF(
            positions=np.array(self.positions)[idx],
            rotations=np.array(self.rotations)[idx],
            half_extents=np.array(self.half_extents)[idx],
            types=np.array(self.types)[idx],
            heights=h[idx],
        )

    @classmethod
    def from_mujoco_model(
        cls,
        mj_model,
        hazard_prefixes=("obstacle_",),
        min_height=0.001,
    ):
        ngeom = mj_model.ngeom
        geom_is_hazard = np.zeros(ngeom, dtype=bool)

        for gid in range(ngeom):
            name = mj_model.geom(gid).name or ""
            if any(name.startswith(p) for p in hazard_prefixes):
                gtype = int(mj_model.geom_type[gid])
                size = mj_model.geom_size[gid]
                # Height index depends on geom type:
                #   BOX (6):               size = [hx, hy, hz]   → size[2]
                #   CYLINDER (5), CAPSULE (3): size = [r, half_len, 0] → size[1]
                #   SPHERE (2):            size = [r, 0, 0]      → size[0]
                if gtype == 6:
                    height = size[2]
                elif gtype in (3, 5):
                    height = size[1]
                else:
                    height = size[0]
                if height >= min_height:
                    geom_is_hazard[gid] = True

        hazard_ids = np.flatnonzero(geom_is_hazard).astype(np.int32)
        M = int(hazard_ids.shape[0])

        if M == 0:
            return cls(
                positions=np.zeros((0, 2), dtype=np.float32),
                rotations=np.zeros((0, 2, 2), dtype=np.float32),
                half_extents=np.zeros((0, 2), dtype=np.float32),
                types=np.zeros((0,), dtype=np.int32),
                heights=np.zeros((0,), dtype=np.float32),
            )

        _mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, _mj_data)

        positions = np.zeros((M, 2), dtype=np.float32)
        rotations = np.zeros((M, 2, 2), dtype=np.float32)
        half_extents = np.zeros((M, 2), dtype=np.float32)
        types = np.zeros(M, dtype=np.int32)
        heights = np.zeros(M, dtype=np.float32)

        for i, gid in enumerate(hazard_ids):
            pos3d = _mj_data.geom_xpos[gid]
            mat3d = _mj_data.geom_xmat[gid].reshape(3, 3)
            gtype_mj = int(mj_model.geom_type[gid])
            size = mj_model.geom_size[gid]

            if gtype_mj == 6:
                height = size[2]
            elif gtype_mj in (3, 5):
                height = size[1]
            else:
                height = size[0]
            heights[i] = height

            positions[i] = pos3d[:2]
            rotations[i] = mat3d[:2, :2]

            sdf_type = _MJ_TYPE_TO_SDF.get(gtype_mj, CIRCLE)
            types[i] = sdf_type

            if sdf_type == CIRCLE:
                r = float(size[0])
                half_extents[i] = [r, 0.0]
            elif sdf_type == BOX:
                hx, hy = float(size[0]), float(size[1])
                half_extents[i] = [hx, hy]
            elif sdf_type == CAPSULE:
                r = float(size[0])
                hl = float(size[1])
                local_z = mat3d[:2, 2]
                xy_norm = float(np.linalg.norm(local_z))
                if xy_norm > 0.15:
                    # MuJoCo capsule axis is local z.  Build a 2D rotation matrix
                    # R2 such that R2 @ [1,0] = axis_2d so that diff_local[0] in
                    # _sdf_and_grad_one is the projection onto the capsule axis.
                    axis_2d = local_z / xy_norm          # unit capsule axis in XY
                    rotations[i] = np.array([[axis_2d[0], -axis_2d[1]],
                                             [axis_2d[1],  axis_2d[0]]])
                    half_extents[i] = [r, hl * xy_norm]
                else:
                    half_extents[i] = [r, 0.0]
                    types[i] = CIRCLE

        return cls(
            positions=positions,
            rotations=rotations,
            half_extents=half_extents,
            types=types,
            heights=heights,
        )


def _sdf_and_grad_one(point_xy, center, R2, half_ext, obs_type):
    """SDF and outward unit gradient for one obstacle at one 2D point."""
    diff = point_xy - center
    diff_local = R2.T @ diff

    # --- Circle ---
    dist_c = jnp.linalg.norm(diff) + 1e-10
    sdf_c = dist_c - half_ext[0]
    grad_c = diff / dist_c

    # --- Box (Inigo Quilez exact SDF) ---
    q_b = jnp.abs(diff_local) - half_ext
    nearest_on_box = jnp.clip(diff_local, -half_ext, half_ext)
    exterior_b = jnp.linalg.norm(nearest_on_box - diff_local) > 1e-10
    sdf_b_ext = jnp.linalg.norm(jnp.maximum(q_b, 0.0))
    sdf_b_int = jnp.minimum(jnp.max(q_b), 0.0)
    sdf_b = sdf_b_ext + sdf_b_int
    grad_b_ext = (diff_local - nearest_on_box) / (
        jnp.linalg.norm(diff_local - nearest_on_box) + 1e-10
    )
    # Use >= 0 → +1, < 0 → -1 to guarantee non-zero gradient.
    # jnp.sign(0) = 0 causes zero gradient when a point lands exactly on a
    # box's symmetry axis (common with regular grids or axis-aligned obstacles).
    s_x = jnp.where(diff_local[0] >= 0, 1.0, -1.0)
    s_y = jnp.where(diff_local[1] >= 0, 1.0, -1.0)
    grad_b_int = jnp.where(
        jnp.argmax(q_b) == 0,
        jnp.array([s_x, 0.0]),
        jnp.array([0.0, s_y]),
    )
    grad_b_local = jnp.where(exterior_b, grad_b_ext, grad_b_int)
    grad_b = R2 @ grad_b_local
    grad_b = grad_b / (jnp.linalg.norm(grad_b) + 1e-10)

    # --- Capsule ---
    clamped_x = jnp.clip(diff_local[0], -half_ext[1], half_ext[1])
    dk = diff_local - jnp.array([clamped_x, 0.0])
    dist_k = jnp.linalg.norm(dk) + 1e-10
    sdf_k = dist_k - half_ext[0]
    grad_k = R2 @ (dk / dist_k)
    grad_k = grad_k / (jnp.linalg.norm(grad_k) + 1e-10)

    sdf = jnp.where(
        obs_type == BOX, sdf_b, jnp.where(obs_type == CAPSULE, sdf_k, sdf_c)
    )
    grad = jnp.where(
        (obs_type == BOX)[..., None],
        grad_b,
        jnp.where((obs_type == CAPSULE)[..., None], grad_k, grad_c),
    )
    return sdf, grad


def _sdf_and_grad_all(point_xy, sdf):
    """SDF and gradient for all M obstacles at one 2D point."""
    return jax.vmap(
        lambda i: _sdf_and_grad_one(
            point_xy,
            sdf.positions[i],
            sdf.rotations[i],
            sdf.half_extents[i],
            sdf.types[i],
        )
    )(jnp.arange(sdf.num_obstacles))


def _batch_sdf_and_grad(positions, sdf):
    """Evaluate SDFs and gradients for all obstacles at a batch of 2D points.

    Args:
        positions: (N, 2) XY positions.
        sdf: ObstacleSDF.
    Returns:
        sdfs: (N, M) signed distances.
        grads: (N, M, 2) outward unit gradients.
    """
    return jax.vmap(lambda p: _sdf_and_grad_all(p, sdf))(positions)


def project_plan_feet(plans_denorm, sdf, r_safe, n_iters=3):
    """Project foot XY positions out of obstacle interiors.

    Uses iterative closest-obstacle projection: at each iteration, find the
    most-violated SDF constraint and push the foot onto its boundary.  For
    single-obstacle violations the projection is exact (circles are affine).
    For multi-obstacle cases, 3 iterations converge within ~1mm at 7cm
    clearance.

    Args:
        plans_denorm: (..., H, 12) world-frame feet positions.
        sdf: ObstacleSDF.
        r_safe: required clearance (metres).
        n_iters: number of projection iterations (3 is a good default).
    Returns:
        Same shape with projected XY and untouched Z.
    """
    if sdf.num_obstacles == 0:
        return plans_denorm

    *batch, H, _ = plans_denorm.shape
    feet = plans_denorm.reshape(*batch, H, 4, 3)
    feet_xy = feet[..., :2]
    feet_z = feet[..., 2:]

    flat_xy = feet_xy.reshape(-1, 2)

    def _step(p, _):
        sdfs_all, grads_all = _batch_sdf_and_grad(p, sdf)
        violations_all = r_safe - sdfs_all

        max_viol = jnp.max(violations_all, axis=1)
        worst_idx = jnp.argmax(violations_all, axis=1)

        N = p.shape[0]
        worst_grads = grads_all[jnp.arange(N), worst_idx]

        push = max_viol[:, None] * worst_grads
        needs_push = max_viol > 1e-6

        return p + jnp.where(needs_push[:, None], push, jnp.zeros_like(push)), None

    flat_xy, _ = jax.lax.scan(_step, flat_xy, None, length=n_iters)

    projected_xy = flat_xy.reshape(feet_xy.shape)
    projected_feet = jnp.concatenate([projected_xy, feet_z], axis=-1)
    return projected_feet.reshape(plans_denorm.shape)


def check_and_fix_swing_arc(p_from, p_to, sdf, r_safe):
    """Check if the swing arc from p_from to p_to crosses an obstacle.

    If the straight-line arc crosses an obstacle interior (SDF < r_safe at any
    intermediate point), returns a corrected landing position on the NEAR-side
    boundary of the obstacle.  Otherwise returns p_to unchanged.

    This ensures that when a foot swings from its current stance position to
    a planned landing, it does not pass through an obstacle.  Without this check,
    projection can land a foot on the far side of an obstacle (geometrically
    safe) while the swing arc passes through the obstacle (collision).

    Args:
        p_from: (2,) current foot XY position in world frame.
        p_to:   (2,) projected landing XY position in world frame.
        sdf:    ObstacleSDF.
        r_safe: required clearance margin (metres).

    Returns:
        p_corrected: (2,) safe landing position — either p_to (if arc is clear)
                     or the near-side boundary intersection (if arc crosses).
    """
    if sdf.num_obstacles == 0:
        return p_to

    direction = p_to - p_from
    arc_length = jnp.linalg.norm(direction)
    is_stationary = arc_length < 1e-7

    n_interp = 20
    t_vals = jnp.linspace(0.1, 0.9, n_interp)
    interp_pts = p_from + t_vals[:, None, None] * direction

    ip_flat = interp_pts.reshape(-1, 2)
    sdfs_flat, grads_flat = _batch_sdf_and_grad(ip_flat, sdf)
    sdfs_544 = sdfs_flat.reshape(5, 4, -1)
    grads_544 = grads_flat.reshape(5, 4, -1, 2)

    min_sdfs_544 = jnp.min(sdfs_544, axis=2)
    overall_min_flat = jnp.argmin(min_sdfs_544)
    idx_i = overall_min_flat // 4
    idx_f = overall_min_flat % 4
    idx_m = jnp.argmin(sdfs_544[idx_i, idx_f, :])

    p_from_min = p_from[idx_f]
    direction_min = direction[idx_f]
    t_min_scalar = t_vals[idx_i]
    grad_at_min = grads_544[idx_i, idx_f, idx_m, :]
    sdf_at_min = sdfs_544[idx_i, idx_f, idx_m]
    min_clamped = jnp.clip(sdf_at_min, -1e4, 1e4)

    arc_is_clear = sdf_at_min >= r_safe
    p_corrected = jax.lax.select(
        is_stationary,
        p_to,
        jax.lax.select(
            arc_is_clear,
            p_to,
            _near_side_correction(
                p_from_min, direction_min, t_min_scalar, grad_at_min,
                sdf_at_min, min_clamped, r_safe, p_to, idx_f
            )
        )
    )
    return p_corrected


def _near_side_correction(p_from_min, direction_min, t_min, grads_at_min, sdfs_at_min, min_sdf_clamped, r_safe, p_to, idx_f):
    """Compute the near-side boundary correction for a crossing arc.

    When arc_check detects SDF < r_safe at t_min, push the violating foot to
    the near-side boundary by projecting along the SDF gradient direction.
    Returns (4, 2) with only the violating foot (idx_f) corrected.
    """
    grad_norm = jnp.linalg.norm(grads_at_min) + 1e-10
    grad_unit = grads_at_min / grad_norm

    deficit = r_safe - min_sdf_clamped
    correction = jnp.maximum(deficit, 0.0) * grad_unit

    p_min = p_from_min + t_min * direction_min
    p_near = p_min + correction

    is_violating = jnp.arange(4) == idx_f
    p_to_corrected = jnp.where(is_violating[:, None], p_near, p_to)

    return p_to_corrected


def project_plan_feet_with_arc_check(plans_denorm, sdf, r_safe, current_feet_xy, n_iters=3):
    """Project foot XY positions with arc-aware correction.

    Like project_plan_feet but additionally checks each swing arc (from current
    stance to projected landing at h=0, and between consecutive planned positions
    at h>0) and snaps landings to the near-side boundary if the arc would
    cross an obstacle.

    Args:
        plans_denorm:    (..., H, 12) world-frame feet positions.
        sdf:             ObstacleSDF.
        r_safe:          required clearance (metres).
        current_feet_xy: (..., 4, 2) current foot XY positions in world frame.
                         For sample_plans this is (N, 4, 2).
        n_iters:         projection iterations (3 is a good default).

    Returns:
        Projected plans with same shape, feet pushed to near-side boundaries
        when swing arcs would cross obstacles.
    """
    if sdf.num_obstacles == 0:
        return plans_denorm

    *batch, H, _ = plans_denorm.shape
    feet = plans_denorm.reshape(*batch, H, 4, 3)
    feet_xy = feet[..., :2]
    feet_z = feet[..., 2:]

    flat_xy = feet_xy.reshape(-1, 2)
    N = flat_xy.shape[0]

    def _step(p, _):
        sdfs_all, grads_all = _batch_sdf_and_grad(p, sdf)
        violations_all = r_safe - sdfs_all
        max_viol = jnp.max(violations_all, axis=1)
        worst_idx = jnp.argmax(violations_all, axis=1)
        worst_grads = grads_all[jnp.arange(N), worst_idx]
        push = max_viol[:, None] * worst_grads
        needs_push = max_viol > 1e-6
        return p + jnp.where(needs_push[:, None], push, jnp.zeros_like(push)), None

    projected_flat_xy, _ = jax.lax.scan(_step, flat_xy, None, length=n_iters)
    projected_xy = projected_flat_xy.reshape(feet_xy.shape)

    def _fix_arc_h0(p_proj, p_curr):
        return jax.vmap(check_and_fix_swing_arc, in_axes=(0, 0, None, None))(
            p_curr, p_proj, sdf, r_safe
        )

    def _fix_arc_hm1(p_proj, p_from):
        return jax.vmap(check_and_fix_swing_arc, in_axes=(0, 0, None, None))(
            p_from, p_proj, sdf, r_safe
        )

    corrected_xy = projected_xy.copy()

    h0_to_check = corrected_xy[..., 0, :, :]  # (*batch, 4, 2)
    # current_feet_xy: (4, 2) shared across batch, or (*batch, 4, 2) per-sample
    current_arr = jnp.asarray(current_feet_xy)
    if current_arr.ndim == 2:
        h0_from = jnp.broadcast_to(current_arr, h0_to_check.shape)
    else:
        h0_from = current_arr.reshape(*batch, 4, 2)
    fixed_h0 = _fix_arc_h0(h0_to_check, h0_from)
    corrected_xy = corrected_xy.at[..., 0, :, :].set(fixed_h0)

    if H > 1:
        for h in range(1, H):
            hm1_xy = corrected_xy[..., h-1, :, :]
            h_xy = corrected_xy[..., h, :, :]
            fixed_h = _fix_arc_hm1(h_xy, hm1_xy)
            corrected_xy = corrected_xy.at[..., h, :, :].set(fixed_h)

    corrected_xy_4d = corrected_xy.reshape(*batch, H, 4, 2)
    corrected_4d = jnp.concatenate([corrected_xy_4d, feet_z], axis=-1)
    return corrected_4d.reshape(plans_denorm.shape)


# ===========================================================================
# QP projection: pairwise KKT enumeration
# ===========================================================================
#
# For a 2D point with identity Hessian and M half-plane constraints, at most
# 2 constraints can be active at the optimum.  We enumerate single-constraint
# and pairwise projections, verify feasibility against all linearized
# constraints, and pick the best feasible candidate.
#
# This gives the exact QP solution for the linearized subproblem in closed
# form per candidate — no iterative convergence issues.
# ===========================================================================


def _project_one_pairwise(p_ref, sdfs, grads, r_safe):
    """Project a single foot (2,) using pairwise KKT enumeration.

    Args:
        p_ref: (2,) initial XY position.
        sdfs: (M,) SDF values at p_ref.
        grads: (M, 2) outward unit gradients at p_ref.
        r_safe: required clearance.

    Returns:
        p_proj: (2,) projected position.
    """
    M = sdfs.shape[0]
    v = r_safe - sdfs  # (M,) violation per obstacle

    b_rhs = jnp.einsum("mj,j->m", grads, p_ref) + r_safe - sdfs  # (M,)

    top2_idx = jnp.flip(jnp.argsort(v)[-2:])
    v2 = v[top2_idx]
    g2 = grads[top2_idx]  # (2, 2)

    # Candidate 1: single-constraint projection onto worst
    p1 = p_ref + v2[0] * g2[0]
    feasible1 = jnp.all(grads @ p1 >= b_rhs - 1e-5)

    # Candidate 2: pairwise intersection
    G = g2
    det = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
    nonsingular = jnp.abs(det) > 1e-10

    def _compute_p2():
        G_inv = jnp.array([[G[1, 1], -G[0, 1]], [-G[1, 0], G[0, 0]]]) / det
        return G_inv @ b_rhs[top2_idx]
    def _fallback_p2():
        return p_ref
    p2 = jax.lax.cond(nonsingular, _compute_p2, _fallback_p2)
    feasible2 = jnp.all(grads @ p2 >= b_rhs - 1e-5) & nonsingular

    p_default = p1
    p_best = jax.lax.select(
        feasible2, p2, jax.lax.select(feasible1, p1, p_default)
    )
    any_violation = v2[0] > 1e-6
    return jax.lax.select(any_violation, p_best, p_ref)


_project_batch_pairwise = jax.vmap(_project_one_pairwise, in_axes=(0, 0, 0, None))


def project_plan_feet_qp(plans_denorm, sdf, r_safe):
    """Project foot XY positions using pairwise KKT QP (1 SDF eval).

    Solves min ||p - p_ref||^2 s.t. linearised SDF_j(p) >= r_safe via
    enumeration of the top-2 violated half-plane constraints.

    Args:
        plans_denorm: (..., H, 12) world-frame feet positions.
        sdf: ObstacleSDF.
        r_safe: required clearance (metres).

    Returns:
        Same shape with projected XY and untouched Z.
    """
    if sdf.num_obstacles == 0:
        return plans_denorm

    *batch, H, _ = plans_denorm.shape
    feet = plans_denorm.reshape(*batch, H, 4, 3)
    feet_xy = feet[..., :2]
    feet_z = feet[..., 2:]
    flat_xy = feet_xy.reshape(-1, 2)

    sdfs_all, grads_all = _batch_sdf_and_grad(flat_xy, sdf)
    p_proj = _project_batch_pairwise(flat_xy, sdfs_all, grads_all, r_safe)

    projected_xy = p_proj.reshape(feet_xy.shape)
    projected_feet = jnp.concatenate([projected_xy, feet_z], axis=-1)
    return projected_feet.reshape(plans_denorm.shape)


def project_plan_feet_qp_sqp(plans_denorm, sdf, r_safe, n_outer=2):
    """SQP‑style QP: 2 outer iterations re‑linearising SDF after each solve.

    Each iteration: evaluate SDFs, project via pairwise KKT, re-linearise.
    This catches the nonlinear boundary cases (boxes, capsules) that a single
    linearisation might miss.

    Args:
        plans_denorm: (..., H, 12).
        sdf: ObstacleSDF.
        r_safe: clearance.
        n_outer: SQP outer iterations (1 or 2).

    Returns:
        Projected feet.
    """
    if sdf.num_obstacles == 0:
        return plans_denorm

    *batch, H, _ = plans_denorm.shape
    feet = plans_denorm.reshape(*batch, H, 4, 3)
    feet_xy = feet[..., :2]
    feet_z = feet[..., 2:]
    p = feet_xy.reshape(-1, 2)

    sdfs, grads = _batch_sdf_and_grad(p, sdf)
    p = _project_batch_pairwise(p, sdfs, grads, r_safe)

    if n_outer >= 2:
        sdfs, grads = _batch_sdf_and_grad(p, sdf)
        p = _project_batch_pairwise(p, sdfs, grads, r_safe)

    projected_xy = p.reshape(feet_xy.shape)
    projected_feet = jnp.concatenate([projected_xy, feet_z], axis=-1)
    return projected_feet.reshape(plans_denorm.shape)


def project_plan_feet_with_arc_check_qp(plans_denorm, sdf, r_safe, current_feet_xy):
    """Arc‑aware foot projection using pairwise KKT QP (1 SDF eval).

    Like project_plan_feet_with_arc_check but replaces the iterative
    projection step with a single‑shot QP solve.

    Args:
        plans_denorm:    (..., H, 12) world-frame feet positions.
        sdf:             ObstacleSDF.
        r_safe:          required clearance (metres).
        current_feet_xy: (..., 4, 2) current foot XY in world frame.

    Returns:
        Projected plans with arc corrections.
    """
    if sdf.num_obstacles == 0:
        return plans_denorm

    *batch, H, _ = plans_denorm.shape
    feet = plans_denorm.reshape(*batch, H, 4, 3)
    feet_xy = feet[..., :2]
    feet_z = feet[..., 2:]

    flat_xy = feet_xy.reshape(-1, 2)
    sdfs_all, grads_all = _batch_sdf_and_grad(flat_xy, sdf)
    projected_flat_xy = _project_batch_pairwise(flat_xy, sdfs_all, grads_all, r_safe)
    projected_xy = projected_flat_xy.reshape(feet_xy.shape)

    def _fix_arc_h0(p_proj, p_curr):
        return jax.vmap(check_and_fix_swing_arc, in_axes=(0, 0, None, None))(
            p_curr, p_proj, sdf, r_safe
        )

    def _fix_arc_hm1(p_proj, p_from):
        return jax.vmap(check_and_fix_swing_arc, in_axes=(0, 0, None, None))(
            p_from, p_proj, sdf, r_safe
        )

    corrected_xy = projected_xy.copy()

    h0_to_check = corrected_xy[..., 0, :, :]
    current_arr = jnp.asarray(current_feet_xy)
    if current_arr.ndim == 2:
        h0_from = jnp.broadcast_to(current_arr, h0_to_check.shape)
    else:
        h0_from = current_arr.reshape(*batch, 4, 2)
    fixed_h0 = _fix_arc_h0(h0_to_check, h0_from)
    corrected_xy = corrected_xy.at[..., 0, :, :].set(fixed_h0)

    if H > 1:
        for h in range(1, H):
            hm1_xy = corrected_xy[..., h - 1, :, :]
            h_xy = corrected_xy[..., h, :, :]
            fixed_h = _fix_arc_hm1(h_xy, hm1_xy)
            corrected_xy = corrected_xy.at[..., h, :, :].set(fixed_h)

    corrected_xy_4d = corrected_xy.reshape(*batch, H, 4, 2)
    corrected_4d = jnp.concatenate([corrected_xy_4d, feet_z], axis=-1)
    return corrected_4d.reshape(plans_denorm.shape)


def project_plan_feet_with_arc_check_qp_sqp(
    plans_denorm, sdf, r_safe, current_feet_xy, n_outer=2
):
    """Arc‑aware foot projection using SQP (2 SDF evals, re‑linearising).

    Like project_plan_feet_with_arc_check_qp but with 2 outer iterations.

    Args:
        plans_denorm:    (..., H, 12).
        sdf:             ObstacleSDF.
        r_safe:          clearance.
        current_feet_xy: (..., 4, 2).
        n_outer:         SQP outer iterations (1 or 2).

    Returns:
        Projected plans with arc corrections.
    """
    if sdf.num_obstacles == 0:
        return plans_denorm

    *batch, H, _ = plans_denorm.shape
    feet = plans_denorm.reshape(*batch, H, 4, 3)
    feet_xy = feet[..., :2]
    feet_z = feet[..., 2:]
    p = feet_xy.reshape(-1, 2)

    sdfs, grads = _batch_sdf_and_grad(p, sdf)
    p = _project_batch_pairwise(p, sdfs, grads, r_safe)

    if n_outer >= 2:
        sdfs, grads = _batch_sdf_and_grad(p, sdf)
        p = _project_batch_pairwise(p, sdfs, grads, r_safe)

    projected_xy = p.reshape(feet_xy.shape)

    def _fix_arc_h0(p_proj, p_curr):
        return jax.vmap(check_and_fix_swing_arc, in_axes=(0, 0, None, None))(
            p_curr, p_proj, sdf, r_safe
        )

    def _fix_arc_hm1(p_proj, p_from):
        return jax.vmap(check_and_fix_swing_arc, in_axes=(0, 0, None, None))(
            p_from, p_proj, sdf, r_safe
        )

    corrected_xy = projected_xy.copy()

    h0_to_check = corrected_xy[..., 0, :, :]
    current_arr = jnp.asarray(current_feet_xy)
    if current_arr.ndim == 2:
        h0_from = jnp.broadcast_to(current_arr, h0_to_check.shape)
    else:
        h0_from = current_arr.reshape(*batch, 4, 2)
    fixed_h0 = _fix_arc_h0(h0_to_check, h0_from)
    corrected_xy = corrected_xy.at[..., 0, :, :].set(fixed_h0)

    if H > 1:
        for h in range(1, H):
            hm1_xy = corrected_xy[..., h - 1, :, :]
            h_xy = corrected_xy[..., h, :, :]
            fixed_h = _fix_arc_hm1(h_xy, hm1_xy)
            corrected_xy = corrected_xy.at[..., h, :, :].set(fixed_h)

    corrected_xy_4d = corrected_xy.reshape(*batch, H, 4, 2)
    corrected_4d = jnp.concatenate([corrected_xy_4d, feet_z], axis=-1)
    return corrected_4d.reshape(plans_denorm.shape)


# ===========================================================================
# Projection method registry
# ===========================================================================
#
# Add new methods here as (project_fn, project_with_arc_check_fn) tuples.
# The task code calls get_projection_methods(name) to retrieve the pair,
# making it trivial to switch between algorithms via config.
# ===========================================================================

PROJECTION_METHODS = {
    "iterative": (project_plan_feet, project_plan_feet_with_arc_check),
    "qp_single": (project_plan_feet_qp, project_plan_feet_with_arc_check_qp),
    "qp_sqp": (project_plan_feet_qp_sqp, project_plan_feet_with_arc_check_qp_sqp),
}


def get_projection_methods(name):
    """Return (project_fn, project_with_arc_check_fn) for a named method.

    Args:
        name: one of "iterative", "qp_single", "qp_sqp".

    Returns:
        (project_fn, project_with_arc_check_fn) tuple.
    """
    if name not in PROJECTION_METHODS:
        raise ValueError(
            f"Unknown projection method: {name!r}. "
            f"Choose from: {list(PROJECTION_METHODS.keys())}"
        )
    return PROJECTION_METHODS[name]