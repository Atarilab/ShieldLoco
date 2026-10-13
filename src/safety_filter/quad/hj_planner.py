"""
HJ Reachability Safety Filter for Footstep Planning.

Discrete-time reach-avoid value function on a 4D grid for CoM
dynamics (configurable: 3D-LIP or unicycle).  The value function
V(s) = max(min(h(s), max_u V(A@s+B@u))) is computed via value
iteration at planner init using the actual obstacle SDF from
task.py — no external dependencies, pure JAX.

At runtime the filter linearises V around the nominal footstep and solves
a 2D QP (iterative half-plane projection) to find the minimally-invasive
footstep correction  preserving V(next CoM) >= 0.

Citations:
  - Bansal et al. 2017, "Hamilton-Jacobi Reachability: A Brief Overview"
    (CDC, 2017) — HJ reachability formulation.
  - Fisac et al. 2019, "A General Safety Framework for Learning-Based
    Control in Uncertain Robotic Systems" (TAC, 2019) — QP-based safety
    filter with value-function constraints.
"""

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from typing import Tuple, Optional

from .bases.alg import QuadAlgSamplingParams, QuadSamplingBasedPlanner, QuadAlgTrajectory
from .task import QuadRLTaskParams, QuadRLTask
from .foot_projection import _batch_sdf_and_grad
from .cbf_planner import (
    _make_lip_matrices,
    _make_unicycle_matrices,
    _project_onto_half_plane,
    _blend_nominal_to_band,
    _DEFAULT_CBF_CLEARANCE,
    _DEFAULT_BAND_OUTER,
    _DEFAULT_BAND_INNER,
)

from safety_filter.bases.risk import RiskStrategy


_DEFAULT_COM_HEIGHT = 0.35
_DEFAULT_MAX_CORRECTION = 0.10


@dataclass
class HJParams(QuadAlgSamplingParams):
    best_plan: Optional[jax.Array] = None
    exec_plan: Optional[jax.Array] = None


class QuadHJPlanner(QuadSamplingBasedPlanner):
    """HJ reachability safety filter for footstep planning.

    Precomputes a discrete-time reach-avoid value function V(s) on a 4D
    grid (x, vx, y, vy) via value iteration at init time.  At runtime,
    the same linearised-QP structure as the CBF planner is used but with
    the HJ value function in place of the SDF barrier function.

    Supports configurable dynamics models:
      - "lip"      : 3D-LIP inverted pendulum with gait-aware stance centroid
      - "unicycle" : kinematic point-mass CoM = footstep centroid
    """

    def __init__(
        self,
        task: QuadRLTask,
        num_samples: int = 1,
        noise_level: float = 0.6,
        num_randomizations: int = 1,
        enabled: bool = True,
        use_geometric_fallback: bool = True,
        risk_strategy: Optional[RiskStrategy] = None,
        seed: int = 0,
        plan_horizon: int = 10,
        iterations: int = 1,
        gamma: float = 0.99,
        safety_weight: float = 1e3,
        collect_data: bool = False,
        top_fraction: float = 0.01,
        use_momentum: bool = False,
        momentum: float = 0.5,
        p_max: float = 1.0,
        c_max_decay: float = 0.9,
        include_terminal_cost: bool = True,
        hj_grid: Tuple[int, int, int, int] = (25, 15, 25, 15),
        hj_grid_bounds: float = 2.0,
        hj_vel_bounds: float = 2.0,
        hj_n_u_per_dim: int = 6,
        hj_vi_iters: int = 30,
        hj_foot_bounds: float = 0.25,
        dynamics_model: str = "lip",
        clearance: float = _DEFAULT_CBF_CLEARANCE,
        max_correction: float = _DEFAULT_MAX_CORRECTION,
        use_band: bool = False,
        band_outer: float = _DEFAULT_BAND_OUTER,
        band_inner: float = _DEFAULT_BAND_INNER,
        **kwargs,
    ) -> None:
        super().__init__(
            task=task,
            num_samples=num_samples,
            noise_level=noise_level,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            iterations=iterations,
            gamma=gamma,
            safety_weight=safety_weight,
            collect_data=collect_data,
            top_fraction=top_fraction,
            use_momentum=use_momentum,
            momentum=momentum,
            p_max=p_max,
            c_max_decay=c_max_decay,
            include_terminal_cost=include_terminal_cost,
        )

        self._enabled = enabled
        self._use_geometric_fallback = use_geometric_fallback
        self._max_correction = max_correction
        self._hj_clearance = clearance
        self._use_band = use_band
        self._band_outer = band_outer
        self._band_inner = band_inner

        T = self.task.command_duration
        self._dynamics_model = dynamics_model
        if dynamics_model == "unicycle":
            self._A, self._B_full = _make_unicycle_matrices(T)
        else:
            self._A, self._B_full = _make_lip_matrices(T, _DEFAULT_COM_HEIGHT)

        self._B_xy = self._B_full[jnp.array([0, 2]), :]

        self._hj_grid = hj_grid
        self._grid_bounds = hj_grid_bounds
        self._vel_bounds = hj_vel_bounds
        self._n_u_per_dim = hj_n_u_per_dim
        self._vi_iters = hj_vi_iters
        self._foot_bounds = hj_foot_bounds

        if self._enabled and self.task.obstacle_sdf.num_obstacles > 0:
            self._grid, self._V = self._compute_value_function()
        else:
            self._V, self._grid = None, None

    # =====================================================================
    # HJ VALUE FUNCTION
    # =====================================================================

    def _compute_value_function(self):
        """Discrete-time reach-avoid value iteration on a 4D grid.

        Solves the fixed-point equation:

            V(s) = min(h(s),  max_{u in U} V(A@s + B@u))

        where h(s) = SDF(s_xy) - clearance uses the actual obstacle SDF
        from task.py.  The grid is centred on the obstacle cluster so that
        relevant CoM states fall within the domain.

        Returns: (grid, V_arr)
            grid: tuple (x, vx, y, vy) of 1-D axis arrays.
            V_arr: (nx, nvx, ny, nvy) value function array.
        """
        nx, nvx, ny, nvy = self._hj_grid

        obs_pos = self.task.obstacle_sdf.positions
        obs_center = jnp.mean(obs_pos, axis=0)

        x  = jnp.linspace(obs_center[0] - self._grid_bounds, obs_center[0] + self._grid_bounds, nx)
        vx = jnp.linspace(-self._vel_bounds, self._vel_bounds, nvx)
        y  = jnp.linspace(obs_center[1] - self._grid_bounds, obs_center[1] + self._grid_bounds, ny)
        vy = jnp.linspace(-self._vel_bounds, self._vel_bounds, nvy)

        X, VX, Y, VY = jnp.meshgrid(x, vx, y, vy, indexing="ij")
        states = jnp.stack([X.ravel(), VX.ravel(), Y.ravel(), VY.ravel()], axis=-1)
        N = states.shape[0]

        sdfs, _ = _batch_sdf_and_grad(states[:, [0, 2]], self.task.obstacle_sdf)
        h_vals = jnp.min(sdfs, axis=-1) - self._hj_clearance

        u_1d = jnp.linspace(-self._foot_bounds, self._foot_bounds, self._n_u_per_dim)
        UX, UY = jnp.meshgrid(u_1d, u_1d, indexing="ij")
        u_flat = jnp.stack([UX.ravel(), UY.ravel()], axis=-1)
        Nu = u_flat.shape[0]

        drift   = states @ self._A.T                 # (N, 4)
        control = u_flat @ self._B_full.T            # (Nu, 4)

        grid_starts = jnp.array([x[0], vx[0], y[0], vy[0]])
        grid_steps  = jnp.array([x[1]-x[0], vx[1]-vx[0], y[1]-y[0], vy[1]-vy[0]])
        grid_sizes  = jnp.array([nx, nvx, ny, nvy])
        strides     = jnp.array([nvx*ny*nvy, ny*nvy, nvy, 1])
        grid_lo     = grid_starts
        grid_hi     = jnp.array([x[-1], vx[-1], y[-1], vy[-1]])

        @jax.jit
        def _iterate(V):
            def _body(V, _):
                def _footstep(max_V, u_ctrl):
                    s_next = drift + u_ctrl[None, :]
                    s_clip = jnp.clip(s_next, grid_lo, grid_hi)
                    idx_f  = jnp.round((s_clip - grid_starts) / grid_steps).astype(jnp.int32)
                    idx_f  = jnp.clip(idx_f, 0, grid_sizes - 1)
                    flat   = jnp.sum(idx_f * strides[None, :], axis=-1)
                    return jnp.maximum(max_V, V[flat]), None

                max_V, _ = jax.lax.scan(_footstep, jnp.full(N, -1e6), control, length=Nu)
                return jnp.minimum(h_vals, max_V), None

            V_final, _ = jax.lax.scan(_body, V, None, length=self._vi_iters)
            return V_final

        V = _iterate(h_vals)
        return (x, vx, y, vy), V.reshape(nx, nvx, ny, nvy)

    # =====================================================================
    # GRID QUERY HELPERS
    # =====================================================================

    def _nearest_idx(self, s: jax.Array) -> jax.Array:
        """Batch nearest-neighbour index into flattened V.

        Args:
            s: (N, 4) query states.

        Returns:
            (N,) flat integer indices into V.ravel().
        """
        x, vx, y, vy = self._grid
        nx, nvx, ny, nvy = self._V.shape

        grid_starts = jnp.array([x[0], vx[0], y[0], vy[0]])
        grid_steps  = jnp.array([x[1]-x[0], vx[1]-vx[0], y[1]-y[0], vy[1]-vy[0]])
        grid_sizes  = jnp.array([nx, nvx, ny, nvy])
        strides     = jnp.array([nvx*ny*nvy, ny*nvy, nvy, 1])
        grid_lo     = grid_starts
        grid_hi     = jnp.array([x[-1], vx[-1], y[-1], vy[-1]])

        s_clip = jnp.clip(s, grid_lo, grid_hi)
        idx_f  = jnp.round((s_clip - grid_starts) / grid_steps).astype(jnp.int32)
        idx_f  = jnp.clip(idx_f, 0, grid_sizes - 1)
        return jnp.sum(idx_f * strides[None, :], axis=-1)

    def _query_V(self, s: jax.Array) -> jax.Array:
        """Query V at a single state.

        Args:
            s: (4,) query state [x, vx, y, vy].

        Returns:
            Scalar V(s).
        """
        idx = self._nearest_idx(s[None, :])[0]
        return self._V.ravel()[idx]

    def _grad_V(self, s: jax.Array) -> jax.Array:
        """Gradient dV/ds at s via central finite differences on the grid.

        Args:
            s: (4,) query state.

        Returns:
            (4,) gradient [dV/dx, dV/dvx, dV/dy, dV/dvy].
        """
        x, vx, y, vy = self._grid
        nx, nvx, ny, nvy = self._V.shape

        idx = self._nearest_idx(s[None, :])[0]

        s_nvx_ny_nvy = nvx * ny * nvy
        s_ny_nvy     = ny * nvy

        ix  = idx // s_nvx_ny_nvy
        rem = idx % s_nvx_ny_nvy
        ivx = rem // s_ny_nvy
        rem = rem % s_ny_nvy
        iy  = rem // nvy
        ivy = rem % nvy

        dx  = x[1] - x[0]
        dvx = vx[1] - vx[0]
        dy  = y[1] - y[0]
        dvy = vy[1] - vy[0]

        V_arr = self._V

        ix_p1 = jnp.clip(ix + 1, 0, nx - 1)
        ix_m1 = jnp.clip(ix - 1, 0, nx - 1)
        gx    = jnp.where(
            (ix > 0) & (ix < nx - 1),
            (V_arr[ix_p1, ivx, iy, ivy] - V_arr[ix_m1, ivx, iy, ivy]) / (2.0 * dx),
            0.0,
        )

        ivx_p1 = jnp.clip(ivx + 1, 0, nvx - 1)
        ivx_m1 = jnp.clip(ivx - 1, 0, nvx - 1)
        gvx    = jnp.where(
            (ivx > 0) & (ivx < nvx - 1),
            (V_arr[ix, ivx_p1, iy, ivy] - V_arr[ix, ivx_m1, iy, ivy]) / (2.0 * dvx),
            0.0,
        )

        iy_p1 = jnp.clip(iy + 1, 0, ny - 1)
        iy_m1 = jnp.clip(iy - 1, 0, ny - 1)
        gy    = jnp.where(
            (iy > 0) & (iy < ny - 1),
            (V_arr[ix, ivx, iy_p1, ivy] - V_arr[ix, ivx, iy_m1, ivy]) / (2.0 * dy),
            0.0,
        )

        ivy_p1 = jnp.clip(ivy + 1, 0, nvy - 1)
        ivy_m1 = jnp.clip(ivy - 1, 0, nvy - 1)
        gvy    = jnp.where(
            (ivy > 0) & (ivy < nvy - 1),
            (V_arr[ix, ivx, iy, ivy_p1] - V_arr[ix, ivx, iy, ivy_m1]) / (2.0 * dvy),
            0.0,
        )

        return jnp.array([gx, gvx, gy, gvy])

    # =====================================================================
    # RUNTIME SAFETY FILTER
    # =====================================================================

    def _hj_correct_step(
        self, plan_h: jax.Array, s_k: jax.Array,
        stance_mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        """HJ value-based footstep correction for one horizon step.

        Evaluates V at the predicted next CoM under the nominal footstep.
        If V < 0 (state is in the BRT), corrects the footstep by solving
        a 2D QP with the linearised V constraint.

        The discrete control is the centroid of the SWING feet — the feet
        about to step and become the next support pivots.  The correction
        is applied to the swing-feet targets; stance feet are never moved.

        Args:
            plan_h: (plan_dim,) normalised nominal footstep plan for step h.
            s_k:    (4,) current CoM state.
            stance_mask: (4,) boolean, True = stance foot. If None, all 4 used.

        Returns:
            (plan_dim,) corrected normalised plan.
        """
        if self._V is None or self.task.obstacle_sdf.num_obstacles == 0:
            return plan_h

        ref_world = plan_h * self.task.plan_lim
        feet_world = ref_world.reshape(4, 3)

        if stance_mask is not None:
            swing_f = 1.0 - stance_mask.astype(jnp.float32)
            n_swing = jnp.sum(swing_f)
        else:
            swing_f = jnp.ones(4, dtype=jnp.float32)
            n_swing = jnp.array(4.0)

        def _correct():
            u_avg_nom = jnp.sum(swing_f[:, None] * feet_world[:, :2], axis=0) / n_swing

            s_nom = self._A @ s_k + self._B_full @ u_avg_nom
            V_nom = self._query_V(s_nom)

            grad = self._grad_V(s_nom)
            grad_B = self._B_full.T @ grad          # (2,)  = B_full^T (4×2) @ grad (4,)
            grad_B = grad_B[:2]                     # keep only the u_x, u_y components

            A_hj = -grad_B[None, :]
            b_hj = V_nom - grad_B @ u_avg_nom

            u_avg_safe = _project_onto_half_plane(
                u_avg_nom,
                A_hj,
                b_hj.reshape(1,),
                u_avg_nom - self._max_correction,
                u_avg_nom + self._max_correction,
            )

            correction = u_avg_safe - u_avg_nom
            safe_feet_xy = feet_world[:, :2] + swing_f[:, None] * correction[None, :]
            safe_feet = jnp.concatenate([safe_feet_xy, feet_world[:, 2:]], axis=-1)
            return safe_feet.reshape(-1) / self.task.plan_lim

        return jax.lax.cond(n_swing > 0.0, _correct, lambda: plan_h)

    # =====================================================================
    # GAIT HELPERS
    # =====================================================================

    def _get_stance_mask(self, params: HJParams, step_idx: int) -> Optional[jax.Array]:
        """Return (4,) boolean stance mask for horizon step step_idx.

        Only for the 3D-LIP model: stance mask = which feet are on the ground
        at this phase.  Unicycle model returns None (all-feet centroid).
        """
        if self._dynamics_model == "unicycle":
            return None

        tp = params.task_params
        gait = jnp.asarray(tp.current_gait, dtype=jnp.int32)
        counter = jnp.asarray(tp.goal_completion_counter, dtype=jnp.int32)
        num_phases = self.task.gait_num_phases[gait]
        phase_idx = jnp.mod(counter + jnp.asarray(step_idx, dtype=jnp.int32), num_phases)

        gait_pattern = jax.lax.dynamic_index_in_dim(
            self.task.gait_patterns, gait, axis=0, keepdims=False
        )
        return jax.lax.dynamic_index_in_dim(
            gait_pattern, phase_idx, axis=0, keepdims=False
        )

    # =====================================================================
    # SAMPLING-BASED PLANNER INTERFACE
    # =====================================================================

    def init_params(
        self,
        task_params: QuadRLTaskParams,
        initial_mean: Optional[jax.Array] = None,
        ref_traj: Optional[jax.Array] = None,
        seed: int = 0,
    ) -> HJParams:
        base = super().init_params(task_params, initial_mean, ref_traj, seed)
        abs_plan = base.ref + base.mean
        return HJParams(
            mean=base.mean,
            best_plan=abs_plan,
            exec_plan=self.task.get_exec_plan(abs_plan),
            ref=base.ref,
            rng=base.rng,
            task_params=base.task_params,
            c_max=base.c_max,
            train_data=None,
        )

    def prepare_optimize(self, state: mjx.Data, params: HJParams) -> HJParams:
        abs_plan = params.best_plan
        abs_plan = jnp.concatenate([abs_plan[1:], params.ref[-1:]], axis=0)
        abs_plan = jnp.clip(abs_plan, self.task.u_min, self.task.u_max)
        new_mean = abs_plan - params.ref
        return params.replace(
            mean=new_mean,
            best_plan=abs_plan,
            exec_plan=self.task.get_exec_plan(abs_plan),
        )

    def update_with_plan(self, state: mjx.Data, params: HJParams) -> HJParams:
        if not self._enabled:
            return params.replace(
                mean=jnp.zeros_like(params.mean),
                best_plan=params.ref,
            )

        base_pos = state.xpos[self.task.base_body_id]
        base_vel = state.qvel
        s_k = jnp.array([base_pos[0], base_vel[0], base_pos[1], base_vel[1]])

        def _step(plan_out, h):
            ref_h = params.ref[h]
            stance_mask = self._get_stance_mask(params, h)
            safe_h = self._hj_correct_step(ref_h, s_k, stance_mask=stance_mask)
            return plan_out.at[h].set(safe_h), None

        H = self.plan_horizon
        safe_plan, _ = jax.lax.scan(
            _step, jnp.zeros_like(params.ref), jnp.arange(H), length=H,
        )

        if self._use_band:
            com_xy = s_k[jnp.array([0, 2])]
            safe_plan = _blend_nominal_to_band(
                params.ref, safe_plan, com_xy, self.task.obstacle_sdf,
                self._band_outer, self._band_inner,
            )

        new_mean = safe_plan - params.ref
        return params.replace(mean=new_mean, best_plan=safe_plan)

    def update_without_plan(self, params: HJParams) -> HJParams:
        zero_offset = jnp.zeros_like(params.ref)
        return params.replace(
            mean=zero_offset,
            best_plan=params.ref,
            exec_plan=self.task.get_exec_plan(params.ref),
            train_data=None,
        )

    def sample_plans(self, params: HJParams) -> Tuple[jax.Array, HJParams]:
        return params.ref[None, :, :], params

    def update_params(
        self, params: HJParams, rollouts: QuadAlgTrajectory
    ) -> HJParams:
        return params

    def finalize_best_plan(self, params: HJParams) -> HJParams:
        abs_plan = params.best_plan
        if self._use_geometric_fallback:
            abs_plan = self.task.project_plan_to_safe(abs_plan)
        exec_plan = self.task.get_exec_plan(abs_plan)
        return params.replace(exec_plan=exec_plan)
