"""
Reduced-order LIPM planning baseline driven by a sampling optimizer.

This baseline implements the same 3D-LIP / unicycle CoM dynamics used by the
CBF and HJ safety filters, but instead of a deterministic closed-form filter it
feeds the reduced-order model to a *general-purpose sampling optimizer*
(MPPI / CEM / Replica-exchange MPPI), exactly the optimizers used by the
full-order method.  Rollouts are performed on the reduced-order CoM model plus
the obstacle SDF — no MuJoCo physics, no learned policy.

Purpose: isolate the effect of the *dynamics model* (reduced-order LIPM vs.
full-order MuJoCo) while keeping the optimizer identical to the proposed
method.  This is the "LIPM model + my optimizer" ablation baseline.

The class is self-contained and modular, mirroring the structure of
QuadCBFPlanner / QuadHJPlanner:

  - reuses the shared LIPM matrix helpers (_make_lip_matrices, ...),
  - reuses the shared SDF / foot-projection utilities via the task,
  - subclasses QuadSamplingBasedPlanner so the worker pipeline (init_params,
    prepare_optimize, update_with_plan, finalize_best_plan, exec_plan publish)
    is identical to every other planner,
  - does NOT modify MIMPS / CEMPS / REMIMPS or any existing planner.

Config:  optimizer in {"mppi", "cem", "replica"} selects the update rule.
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
    _blend_nominal_to_band,
    _DEFAULT_CBF_CLEARANCE,
    _DEFAULT_BAND_OUTER,
    _DEFAULT_BAND_INNER,
)

from safety_filter.bases.risk import RiskStrategy


_DEFAULT_COM_HEIGHT = 0.35
_DEFAULT_OPTIMIZER = "mppi"


@dataclass
class LIPMPlannerParams(QuadAlgSamplingParams):
    """Policy parameters for the reduced-order LIPM sampling planner.

    In "mppi" / "cem" mode the active state lives in ``mean`` / ``std`` /
    ``best_plan`` (shapes (H, plan_dim)).  In "replica" mode the active state
    lives in the per-replica arrays (K, H, plan_dim) and the base-class fields
    mirror the coldest replica for upstream callers.
    """
    std: jax.Array = jnp.zeros((1, 1))
    best_plan: Optional[jax.Array] = None
    exec_plan: Optional[jax.Array] = None

    # Replica-exchange state (dummy defaults; real shapes set in init_params)
    temperature: jax.Array = jnp.zeros(1)          # (K,)
    swap_every: int = 1
    mean_replicas: jax.Array = jnp.zeros((1, 1, 1))       # (K, H, plan_dim)
    best_plan_replicas: jax.Array = jnp.zeros((1, 1, 1))  # (K, H, plan_dim)
    std_replicas: jax.Array = jnp.zeros((1, 1, 1))        # (K, H, plan_dim)


class QuadLIPMPlanner(QuadSamplingBasedPlanner):
    """Sampling optimizer over the reduced-order LIPM / unicycle CoM model.

    At each cycle the planner:
      1. samples N footstep plans around the warm-started mean (+ ref),
      2. propagates the CoM through the reduced-order LIPM dynamics using the
         swing-feet centroid as the control, evaluating SDF-based safety cost
         at the predicted CoM and feet plus a tracking cost,
      3. updates the sampling distribution with the selected optimizer rule
         ("mppi" | "cem" | "replica").

    No full-order physics and no learned policy is ever invoked.
    """

    def __init__(
        self,
        task: QuadRLTask,
        num_samples: int = 256,
        noise_level: float = 0.6,
        num_randomizations: int = 1,
        enabled: bool = True,
        use_geometric_fallback: bool = True,
        risk_strategy: Optional[RiskStrategy] = None,
        seed: int = 0,
        plan_horizon: int = 10,
        iterations: int = 3,
        gamma: float = 0.99,
        safety_weight: float = 1e3,
        collect_data: bool = False,
        top_fraction: float = 0.01,
        use_momentum: bool = False,
        momentum: float = 0.5,
        p_max: float = 1.0,
        c_max_decay: float = 0.9,
        include_terminal_cost: bool = True,
        std_decay: float = 0.9,
        optimizer: str = _DEFAULT_OPTIMIZER,
        temperature: float = 1.0,
        elite_fraction: float = 0.1,
        cross_entropy_decay: float = 0.8,
        num_replicas: int = 3,
        temp_min: float = 0.1,
        temp_max: float = 2.0,
        swap_every: int = 1,
        dynamics_model: str = "lip",
        clearance: float = _DEFAULT_CBF_CLEARANCE,
        cost_w_body: float = 10.0,
        cost_w_foot: float = 5.0,
        cost_w_track: float = 1.0,
        cost_w_terminal: float = 10.0,
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
            std_decay=std_decay,
        )

        if optimizer not in ("mppi", "cem", "replica"):
            raise ValueError(
                f"optimizer must be 'mppi', 'cem', or 'replica', got {optimizer!r}"
            )
        if optimizer == "replica" and num_replicas < 2:
            raise ValueError("num_replicas must be >= 2 for replica exchange")

        self._enabled = enabled
        self._use_geometric_fallback = use_geometric_fallback
        self._optimizer = optimizer
        self._temperature = temperature
        self._elite_fraction = elite_fraction
        self._cross_entropy_decay = cross_entropy_decay
        self._num_replicas = num_replicas
        self._temp_min = temp_min
        self._temp_max = temp_max
        self._swap_every = max(1, swap_every)

        self._dynamics_model = dynamics_model
        T = self.task.command_duration
        if dynamics_model == "unicycle":
            self._A, self._B_full = _make_unicycle_matrices(T)
        else:
            self._A, self._B_full = _make_lip_matrices(T, _DEFAULT_COM_HEIGHT)
        self._B_xy = self._B_full[jnp.array([0, 2]), :]

        self._clearance = clearance
        self._w_body = cost_w_body
        self._w_foot = cost_w_foot
        self._w_track = cost_w_track
        self._w_terminal = cost_w_terminal

        self._use_band = use_band
        self._band_outer = band_outer
        self._band_inner = band_inner

        # Replica ladder only needed in replica mode.
        exponents = jnp.linspace(0.0, 1.0, self._num_replicas)
        self._temperature_ladder = temp_min * (temp_max / temp_min) ** exponents
        self._samples_per_replica = max(1, num_samples // self._num_replicas)

    # =====================================================================
    # GAIT HELPERS
    # =====================================================================

    def _get_stance_mask(self, params: LIPMPlannerParams, step_idx: int) -> Optional[jax.Array]:
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

    def _get_stance_masks(self, task_params: QuadRLTaskParams, gait_counter: int) -> jax.Array:
        """Return (H, 4) boolean stance masks over the horizon."""
        if self._dynamics_model == "unicycle":
            return jnp.zeros((self.plan_horizon, 4), dtype=jnp.bool_)

        gait = jnp.asarray(task_params.current_gait, dtype=jnp.int32)
        counter = jnp.asarray(gait_counter, dtype=jnp.int32)
        num_phases = self.task.gait_num_phases[gait]
        gait_pattern = jax.lax.dynamic_index_in_dim(
            self.task.gait_patterns, gait, axis=0, keepdims=False
        )

        H = self.plan_horizon
        masks = []
        for h in range(H):
            phase_idx = jnp.mod(counter + h, num_phases)
            masks.append(
                jax.lax.dynamic_index_in_dim(gait_pattern, phase_idx, axis=0, keepdims=False)
            )
        return jnp.stack(masks, axis=0)

    # =====================================================================
    # REDUCED-ORDER ROLLOUT
    # =====================================================================

    def _rollout_reduced(
        self,
        task_params: QuadRLTaskParams,
        s_k: jax.Array,
        plans: jax.Array,
        ref: jax.Array,
    ) -> jax.Array:
        """Evaluate sample plans on the reduced-order CoM model.

        Args:
            task_params: current task params (gait phase source).
            s_k:    (4,) current CoM state [x, vx, y, vy].
            plans:  (N, H, plan_dim) normalised absolute footstep plans.
            ref:    (H, plan_dim) normalised reference plans.

        Returns:
            costs:  (N,) discounted cumulative cost per sample.
        """
        N = plans.shape[0]
        plan_lim = self.task.plan_lim
        sdf = self.task.obstacle_sdf

        s0 = jnp.broadcast_to(s_k, (N, 4))
        plans_f = jnp.transpose(
            (plans.reshape(N, self.plan_horizon, 4, 3) * plan_lim), (1, 0, 2, 3)
        )  # (H, N, 4, 3)
        ref_f = ref.reshape(self.plan_horizon, 4, 3) * plan_lim
        stances = self._get_stance_masks(
            task_params, task_params.goal_completion_counter
        )

        def _sdf_penalty(points):
            if sdf.num_obstacles == 0:
                return jnp.zeros(points.shape[:-1], dtype=jnp.float32)
            flat = points.reshape(-1, 2)
            sdfs, _ = _batch_sdf_and_grad(flat, sdf)
            sdfs = sdfs.reshape(points.shape[:-1] + (sdfs.shape[-1],))
            return jax.nn.softplus(self._clearance - jnp.min(sdfs, axis=-1))

        def _step_body(carry, xs):
            s, acc, disc = carry
            plan_h, ref_h, stance = xs
            feet = plan_h  # (N, 4, 3)
            swing = 1.0 - stance.astype(jnp.float32)
            n_swing = jnp.maximum(jnp.sum(swing), 1e-6)
            u_avg = jnp.sum(swing[None, :, None] * feet[:, :, :2], axis=1) / n_swing  # (N, 2)

            s_next = s @ self._A.T + u_avg @ self._B_full.T  # (N, 4)
            com_xy = s_next[:, [0, 2]]

            c_body = self._w_body * _sdf_penalty(com_xy)  # (N,)

            feet_xy = feet[:, :, :2]  # (N, 4, 2)
            c_foot = self._w_foot * jnp.sum(
                swing[None, :] * _sdf_penalty(feet_xy), axis=-1
            )

            c_track = self._w_track * jnp.sum(
                (feet[:, :, :2] - ref_h[None, :, :2]) ** 2, axis=(1, 2)
            )

            acc = acc + disc * (c_body + c_foot + c_track)
            return (s_next, acc, disc * self.gamma), None

        init = (s0, jnp.zeros(N, dtype=jnp.float32), jnp.ones(N, dtype=jnp.float32))
        (s_f, costs, disc), _ = jax.lax.scan(
            _step_body, init, (plans_f, ref_f, stances), length=self.plan_horizon
        )

        if self.include_terminal_cost:
            c_term = self._w_terminal * _sdf_penalty(s_f[:, [0, 2]])
            costs += disc * c_term

        return costs

    # =====================================================================
    # OPTIMIZER UPDATES
    # =====================================================================

    def _mppi_update(
        self, params: LIPMPlannerParams, plans: jax.Array, costs: jax.Array
    ) -> LIPMPlannerParams:
        """Temperature-weighted softmax update (mirrors MIMPS.update_params)."""
        plans = plans[: self.num_samples]
        costs = costs[: self.num_samples]

        increments = plans - params.ref
        costs_safe = jnp.where(jnp.isfinite(costs), costs, jnp.max(costs, initial=0.0) + 1e3)
        cost_std = jnp.std(costs_safe)
        costs_normed = (costs_safe - jnp.mean(costs_safe)) / (cost_std + 1e-6)
        costs_normed = jnp.clip(costs_normed, -100.0, 100.0)
        weights = jax.nn.softmax(-costs_normed / self._temperature, axis=0)
        raw_mean = jnp.sum(weights[:, None, None] * increments, axis=0)

        if self.use_momentum:
            new_mean = raw_mean + (self.momentum / (1 - self.momentum)) * (raw_mean - params.mean)
            diffs = increments - new_mean[None, :, :]
            s = jnp.sum(weights[:, None, None] * (diffs ** 2), axis=0)
            old_variance = params.std ** 2
            alpha = (self.momentum + 1) * 0.1
            new_variance = s + (1 / (self.momentum + 1 - alpha)) * (
                self.momentum * (old_variance - s) + alpha * s
            )
            new_std = jnp.sqrt(jnp.maximum(new_variance, (0.1 * self.noise_level) ** 2))
        else:
            new_mean = raw_mean
            new_std = params.std

        new_best = plans[jnp.argmin(costs)]

        return params.replace(mean=new_mean, std=new_std, best_plan=new_best)

    def _cem_update(
        self, params: LIPMPlannerParams, plans: jax.Array, costs: jax.Array
    ) -> LIPMPlannerParams:
        """Elite-fraction mean/std refit (mirrors CEMPS.update_params)."""
        plans = plans[: self.num_samples]
        costs = costs[: self.num_samples]

        increments = plans - params.ref
        num_elites = max(1, int(self.num_samples * self._elite_fraction))
        elite_indices = jnp.argsort(costs)[:num_elites]

        elite_increments = increments[elite_indices]
        raw_mean = jnp.mean(elite_increments, axis=0)

        if self.use_momentum:
            new_mean = raw_mean + (self.momentum / (1 - self.momentum)) * (raw_mean - params.mean)
            s = jnp.mean((elite_increments - new_mean[None, :, :]) ** 2, axis=0)
            old_variance = params.std ** 2
            alpha = (self.momentum + 1) * 0.1
            new_variance = s + (1 / (self.momentum + 1 - alpha)) * (
                self.momentum * (old_variance - s) + alpha * s
            )
            new_std = jnp.sqrt(jnp.maximum(new_variance, (0.1 * self.noise_level) ** 2))
        else:
            new_mean = raw_mean
            new_std = jnp.std(elite_increments, axis=0)
            new_std = self._cross_entropy_decay * new_std + (1 - self._cross_entropy_decay) * self.noise_level
            new_std = jnp.maximum(new_std, 0.1 * self.noise_level)

        new_best = plans[jnp.argmin(costs)]

        return params.replace(mean=new_mean, std=new_std, best_plan=new_best)

    # =====================================================================
    # SAMPLING-BASED PLANNER INTERFACE
    # =====================================================================

    def init_params(
        self,
        task_params: QuadRLTaskParams,
        initial_mean: Optional[jax.Array] = None,
        ref_traj: Optional[jax.Array] = None,
        seed: int = 0,
    ) -> LIPMPlannerParams:
        base = super().init_params(task_params, initial_mean, ref_traj, seed)

        H = self.plan_horizon
        plan_dim = self.task.plan_dim
        K = self._num_replicas
        std = jnp.full((H, plan_dim), self.noise_level, dtype=jnp.float32)

        abs_plan = base.ref + base.mean

        if self._optimizer == "replica":
            means = jnp.zeros((K, H, plan_dim), dtype=jnp.float32)
            if initial_mean is not None:
                means = means.at[0].set(initial_mean)
            best_plans = jnp.zeros((K, H, plan_dim), dtype=jnp.float32)
            best_plans = best_plans.at[0].set(abs_plan)
            rng = base.rng
            for k in range(1, K):
                rng, subrng = jax.random.split(rng)
                perturb = jax.random.normal(subrng, (H, plan_dim)) * 0.1
                best_plans = best_plans.at[k].set(base.ref + perturb)
            stds = jnp.full((K, H, plan_dim), self.noise_level, dtype=jnp.float32)

            return LIPMPlannerParams(
                mean=means[0],
                std=stds[0],
                best_plan=best_plans[0],
                exec_plan=self.task.get_exec_plan(best_plans[0]),
                ref=base.ref,
                rng=rng,
                task_params=base.task_params,
                c_max=base.c_max,
                train_data=None,
                temperature=self._temperature_ladder,
                swap_every=self._swap_every,
                mean_replicas=means,
                best_plan_replicas=best_plans,
                std_replicas=stds,
            )

        return LIPMPlannerParams(
            mean=base.mean,
            std=std,
            best_plan=abs_plan,
            exec_plan=self.task.get_exec_plan(abs_plan),
            ref=base.ref,
            rng=base.rng,
            task_params=base.task_params,
            c_max=base.c_max,
            train_data=None,
            temperature=self._temperature_ladder,
            swap_every=self._swap_every,
            mean_replicas=jnp.zeros((1, H, plan_dim), dtype=jnp.float32),
            best_plan_replicas=jnp.zeros((1, H, plan_dim), dtype=jnp.float32),
            std_replicas=jnp.zeros((1, H, plan_dim), dtype=jnp.float32),
        )

    def prepare_optimize(self, state: mjx.Data, params: LIPMPlannerParams) -> LIPMPlannerParams:
        if self._optimizer == "replica":
            return self._prepare_optimize_replica(params)
        return self._prepare_optimize_single(params)

    def _prepare_optimize_single(self, params: LIPMPlannerParams) -> LIPMPlannerParams:
        abs_plan = params.best_plan
        abs_plan = jnp.concatenate([abs_plan[1:], params.ref[-1:]], axis=0)
        abs_plan = jnp.clip(abs_plan, self.task.u_min, self.task.u_max)
        new_mean = abs_plan - params.ref

        std_shifted = jnp.concatenate([params.std[1:], params.std[-1:]], axis=0)
        new_std = self.std_decay * std_shifted + (1 - self.std_decay) * self.noise_level
        new_std = jnp.maximum(new_std, 0.1 * self.noise_level)
        has_zero_std = jnp.all(new_std == 0.0)
        new_std = jnp.where(has_zero_std, jnp.full_like(new_std, self.noise_level), new_std)

        return params.replace(
            mean=new_mean,
            std=new_std,
            best_plan=abs_plan,
            exec_plan=self.task.get_exec_plan(abs_plan),
        )

    def _prepare_optimize_replica(self, params: LIPMPlannerParams) -> LIPMPlannerParams:
        K = self._num_replicas
        caller_reset = jnp.all(params.mean == 0.0) & jnp.all(params.best_plan == params.ref)

        ref_tail = jnp.repeat(params.ref[None, -1:, :], K, axis=0)
        best_shifted = jnp.concatenate([params.best_plan_replicas[:, 1:, :], ref_tail], axis=1)
        best_shifted = jnp.clip(best_shifted, self.task.u_min, self.task.u_max)
        mean_shifted = best_shifted - params.ref[None, :, :]

        std_shifted = jnp.concatenate(
            [params.std_replicas[:, 1:, :], params.std_replicas[:, -1:, :]], axis=1
        )
        std_decayed = self.std_decay * std_shifted + (1 - self.std_decay) * self.noise_level
        std_decayed = jnp.maximum(std_decayed, 0.1 * self.noise_level)
        has_zero_std = jnp.all(std_decayed == 0.0, axis=(1, 2))
        std_decayed = jnp.where(
            has_zero_std[:, None, None],
            jnp.full_like(std_decayed, self.noise_level),
            std_decayed,
        )

        best_shifted = jnp.where(
            caller_reset,
            jnp.broadcast_to(params.ref[None, :, :], best_shifted.shape),
            best_shifted,
        )
        mean_shifted = jnp.where(caller_reset, jnp.zeros_like(mean_shifted), mean_shifted)
        std_decayed = jnp.where(
            caller_reset, jnp.full_like(std_decayed, self.noise_level), std_decayed
        )

        has_nan = jnp.any(~jnp.isfinite(best_shifted))
        best_shifted = jnp.where(
            has_nan, jnp.broadcast_to(params.ref[None, :, :], best_shifted.shape), best_shifted
        )
        mean_shifted = jnp.where(has_nan, jnp.zeros_like(mean_shifted), mean_shifted)
        std_decayed = jnp.where(
            has_nan, jnp.full_like(std_decayed, self.noise_level), std_decayed
        )

        best_cold = best_shifted[0]
        return params.replace(
            mean=mean_shifted[0],
            best_plan=best_cold,
            exec_plan=self.task.get_exec_plan(best_cold),
            mean_replicas=mean_shifted,
            best_plan_replicas=best_shifted,
            std_replicas=std_decayed,
        )

    def sample_plans(self, params: LIPMPlannerParams) -> Tuple[jax.Array, LIPMPlannerParams]:
        if self._optimizer == "replica":
            return self._sample_plans_replica(params)
        return self._sample_plans_single(params)

    def _sample_plans_single(self, params: LIPMPlannerParams) -> Tuple[jax.Array, LIPMPlannerParams]:
        rng, noise_rng = jax.random.split(params.rng)
        noise = self.quad_noise(noise_rng)

        noise_raw = noise / jnp.maximum(self.noise_level, 1e-6)
        knots = params.mean + params.ref + noise_raw * params.std[None, :, :]

        knots = knots.at[-1].set(params.ref)

        current_feet_xy = params.task_params.init_foot_positions[:, :2]
        knots = self.task.project_plan_to_safe_with_arc_check(knots, current_feet_xy)

        return knots, params.replace(rng=rng)

    def _sample_plans_replica(self, params: LIPMPlannerParams) -> Tuple[jax.Array, LIPMPlannerParams]:
        K = self._num_replicas
        N_k = self._samples_per_replica

        def sample_replica(k, rng_in):
            rng_out, noise_rng = jax.random.split(rng_in)
            noise = self.quad_noise(noise_rng, N_k)
            noise_raw = noise / jnp.maximum(self.noise_level, 1e-6)
            temp_scale = jnp.sqrt(params.temperature[k] / self._temp_min)
            noise_scaled = noise_raw * params.std_replicas[k] * temp_scale

            mean_k = jnp.where(
                jnp.all(jnp.isfinite(params.mean_replicas[k])),
                params.mean_replicas[k],
                jnp.zeros_like(params.mean_replicas[k]),
            )
            plan_k = mean_k + params.ref + noise_scaled
            plan_k = plan_k.at[-1].set(params.ref)

            current_feet_xy = params.task_params.init_foot_positions[:, :2]
            plan_k = self.task.project_plan_to_safe_with_arc_check(plan_k, current_feet_xy)
            return plan_k, rng_out

        rngs = jax.random.split(params.rng, K)
        plans, rngs_out = jax.vmap(sample_replica, in_axes=(0, 0))(jnp.arange(K), rngs)

        return plans, params.replace(rng=rngs_out[0])

    def update_with_plan(self, state: mjx.Data, params: LIPMPlannerParams) -> LIPMPlannerParams:
        if not self._enabled:
            return params.replace(
                mean=jnp.zeros_like(params.mean),
                best_plan=params.ref,
            )

        base_pos = state.xpos[self.task.base_body_id]
        base_vel = state.qvel
        s_k = jnp.array([base_pos[0], base_vel[0], base_pos[1], base_vel[1]])

        if self._optimizer == "replica":
            params = self._update_replica(s_k, params)
        else:
            params = self._update_single(s_k, params)

        if self._use_band and self.task.obstacle_sdf.num_obstacles > 0:
            com_xy = s_k[jnp.array([0, 2])]
            params = params.replace(
                best_plan=_blend_nominal_to_band(
                    params.ref,
                    params.best_plan,
                    com_xy,
                    self.task.obstacle_sdf,
                    self._band_outer,
                    self._band_inner,
                )
            )

        return params

    def _update_single(self, s_k: jax.Array, params: LIPMPlannerParams) -> LIPMPlannerParams:
        def _optimize_scan_body(params, _):
            plans, params = self.sample_plans(params)
            costs = self._rollout_reduced(params.task_params, s_k, plans, params.ref)
            if self._optimizer == "cem":
                params = self._cem_update(params, plans, costs)
            else:
                params = self._mppi_update(params, plans, costs)
            return params, None

        params, _ = jax.lax.scan(
            _optimize_scan_body, params, jnp.arange(self.iterations), length=self.iterations
        )
        return params

    def _update_replica(self, s_k: jax.Array, params: LIPMPlannerParams) -> LIPMPlannerParams:
        K = self._num_replicas
        N_k = self._samples_per_replica
        H = self.plan_horizon
        plan_dim = self.task.plan_dim

        def _optimize_scan_body(carry, iter_idx):
            params_iter = carry

            plans, params_iter = self.sample_plans(params_iter)
            plans_flat = plans.reshape(K * N_k, H, plan_dim)
            costs_flat = self._rollout_reduced(params_iter.task_params, s_k, plans_flat, params_iter.ref)
            costs = costs_flat.reshape(K, N_k)

            def update_replica(k):
                plans_k = plans[k]
                costs_k = costs[k]
                T_k = params_iter.temperature[k]

                all_valid = jnp.all(jnp.isfinite(costs_k))
                cost_std = jnp.std(costs_k)
                costs_normed = (costs_k - jnp.mean(costs_k)) / (cost_std + 1e-6)
                costs_normed = jnp.clip(costs_normed, -100.0, 100.0)
                weights = jax.nn.softmax(-costs_normed / T_k, axis=0)

                increments = plans_k - params_iter.ref
                raw_mean_k = jnp.sum(weights[:, None, None] * increments, axis=0)

                if self.use_momentum:
                    new_mean_k = raw_mean_k + (self.momentum / (1 - self.momentum)) * (
                        raw_mean_k - params_iter.mean_replicas[k]
                    )
                    diffs = increments - new_mean_k[None, :, :]
                    s_k_ = jnp.sum(weights[:, None, None] * (diffs ** 2), axis=0)
                    old_variance = params_iter.std_replicas[k] ** 2
                    alpha = (self.momentum + 1) * 0.1
                    new_variance = s_k_ + (1 / (self.momentum + 1 - alpha)) * (
                        self.momentum * (old_variance - s_k_) + alpha * s_k_
                    )
                    new_std_k = jnp.sqrt(jnp.maximum(new_variance, (0.1 * self.noise_level) ** 2))
                else:
                    new_mean_k = raw_mean_k
                    new_std_k = params_iter.std_replicas[k]

                best_idx = jnp.argmin(jnp.where(jnp.isfinite(costs_k), costs_k, jnp.inf))
                new_best_plan_k = plans_k[best_idx]
                replica_cost = jnp.mean(costs_k)

                new_mean_k = jnp.where(all_valid, new_mean_k, params_iter.mean_replicas[k])
                new_best_plan_k = jnp.where(
                    all_valid, new_best_plan_k, params_iter.best_plan_replicas[k]
                )
                new_std_k = jnp.where(all_valid, new_std_k, params_iter.std_replicas[k])
                replica_cost = jnp.where(all_valid, replica_cost, jnp.inf)

                return new_mean_k, new_best_plan_k, new_std_k, replica_cost

            new_means, new_best_plans, new_stds, replica_costs = jax.vmap(update_replica)(
                jnp.arange(K)
            )

            params_iter = params_iter.replace(
                mean_replicas=new_means,
                best_plan_replicas=new_best_plans,
                std_replicas=new_stds,
            )

            should_swap = jnp.mod(iter_idx, params_iter.swap_every) == 0

            def do_swap(p):
                def swap_pair(i, j, means_in, stds_in, rng_in):
                    ci = replica_costs[i]
                    cj = replica_costs[j]
                    Ti = p.temperature[i]
                    Tj = p.temperature[j]

                    bad = (~jnp.isfinite(ci)) | (~jnp.isfinite(cj))
                    log_accept = (ci - cj) * (1.0 / Ti - 1.0 / Tj)
                    log_accept = jnp.clip(log_accept, -20.0, 20.0)
                    accept_prob = jnp.minimum(1.0, jnp.exp(log_accept))

                    rng_out, subrng = jax.random.split(rng_in)
                    accept = (~bad) & (jax.random.uniform(subrng) < accept_prob)

                    new_mean_i = jnp.where(accept, means_in[j], means_in[i])
                    new_mean_j = jnp.where(accept, means_in[i], means_in[j])
                    new_std_i = jnp.where(accept, stds_in[j], stds_in[i])
                    new_std_j = jnp.where(accept, stds_in[i], stds_in[j])

                    return (
                        means_in.at[i].set(new_mean_i).at[j].set(new_mean_j),
                        stds_in.at[i].set(new_std_i).at[j].set(new_std_j),
                        rng_out,
                    )

                means_curr = p.mean_replicas
                stds_curr = p.std_replicas
                rng_curr = p.rng

                for i in range(0, K - 1, 2):
                    means_curr, stds_curr, rng_curr = swap_pair(
                        i, i + 1, means_curr, stds_curr, rng_curr
                    )
                for i in range(1, K - 1, 2):
                    means_curr, stds_curr, rng_curr = swap_pair(
                        i, i + 1, means_curr, stds_curr, rng_curr
                    )

                return p.replace(mean_replicas=means_curr, std_replicas=stds_curr, rng=rng_curr)

            def no_swap(p):
                return p

            params_iter = jax.lax.cond(should_swap, do_swap, no_swap, params_iter)
            return params_iter, None

        params_final, _ = jax.lax.scan(
            f=_optimize_scan_body,
            init=params,
            xs=jnp.arange(self.iterations),
            length=self.iterations,
        )

        return params_final.replace(
            mean=params_final.mean_replicas[0],
            best_plan=params_final.best_plan_replicas[0],
        )

    def update_without_plan(self, params: LIPMPlannerParams) -> LIPMPlannerParams:
        zero_offset = jnp.zeros_like(params.ref)
        H = self.plan_horizon
        plan_dim = self.task.plan_dim

        if self._optimizer == "replica":
            K = self._num_replicas
            zero_means = jnp.zeros((K, H, plan_dim), dtype=jnp.float32)
            best_plans = jnp.repeat(params.ref[None, :, :], K, axis=0)
            stds = jnp.full((K, H, plan_dim), self.noise_level, dtype=jnp.float32)
            return params.replace(
                mean=zero_means[0],
                best_plan=best_plans[0],
                exec_plan=self.task.get_exec_plan(params.ref),
                train_data=None,
                mean_replicas=zero_means,
                best_plan_replicas=best_plans,
                std_replicas=stds,
            )

        zero_std = jnp.full((H, plan_dim), self.noise_level, dtype=jnp.float32)
        return params.replace(
            mean=zero_offset,
            std=zero_std,
            best_plan=params.ref,
            exec_plan=self.task.get_exec_plan(params.ref),
            train_data=None,
        )

    def update_params(
        self, params: LIPMPlannerParams, rollouts: QuadAlgTrajectory
    ) -> LIPMPlannerParams:
        """No-op: updates happen inside update_with_plan via the reduced-order model."""
        return params

    def finalize_best_plan(self, params: LIPMPlannerParams) -> LIPMPlannerParams:
        abs_plan = params.best_plan
        if self._use_geometric_fallback:
            abs_plan = self.task.project_plan_to_safe(abs_plan)
        exec_plan = self.task.get_exec_plan(abs_plan)
        return params.replace(exec_plan=exec_plan)

    def f(
        self, state: mjx.Data, params: QuadAlgSamplingParams, plans: jax.Array
    ) -> QuadAlgSamplingParams:
        """Reduced-order planner evaluates rollouts internally; f() must not be used."""
        raise RuntimeError(
            "QuadLIPMPlanner.f() must not be called directly. "
            "Rollouts are performed on the reduced-order LIPM model inside update_with_plan()."
        )