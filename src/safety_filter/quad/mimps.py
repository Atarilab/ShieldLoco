import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from typing import Dict, Tuple, Optional

from .bases.alg import QuadAlgSamplingParams, QuadSamplingBasedPlanner, QuadAlgTrajectory
from .task import QuadRLTaskParams, QuadRLTask

from safety_filter.bases.risk import RiskStrategy


# MIMPS = MMPI MPS
@dataclass
class MIMPSParams(QuadAlgSamplingParams):
    """Policy parameters for model-predictive path integral control.

    Same as SamplingParams

    Attributes:
        best_plan: The best plan spline knots found so far.
        min_violations: Constraint sum of the best (argmin) sample from the last MPPI iteration.
                        0.0 means at least one sample was collision-free → safe to override.
    """
    best_plan: Optional[jax.Array] = None
    exec_plan: Optional[jax.Array] = None
    # min_violations: Optional[jax.Array] = None


class QuadMIMPSPlanner(QuadSamplingBasedPlanner):
    def __init__(
        self,
        task: QuadRLTask,
        num_samples: int,
        noise_level: float,
        temperature: float,
        num_randomizations: int = 1,
        risk_strategy: Optional[RiskStrategy] = None,
        seed: int = 0,
        plan_horizon: int = 10,
        iterations: int = 1,
        gamma: float = 0.99,
        safety_weight: float = 1e3,
        collect_data: bool = False,
        top_fraction: float = 0.01,
        p_max: float = 1.0,
        c_max_decay: float = 0.9,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The timesteps of horizon for the rollout.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
        """

        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,  # type: ignore
            seed=seed,
            plan_horizon=plan_horizon,
            iterations=iterations,
            gamma=gamma,
            safety_weight=safety_weight,
            collect_data=collect_data,
            top_fraction=top_fraction,
            p_max=p_max,
            c_max_decay=c_max_decay,
        )

        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        self.noise_level = noise_level
        self.num_samples = num_samples
        self.temperature = temperature
        self.gamma = gamma
        self.safety_weight = safety_weight
        self.collect_data = collect_data
        self.top_fraction = top_fraction

    def update_with_plan(self, state, params):
        def _optimize_scan_body(params, _):
            # Sample random control sequences from spline knots
            plans, params = self.sample_plans(params)

            params = self.f(state, params, plans)

            # Only return train_data from last iteration to save memory
            return params, _

        # Don't accumulate all iterations - only keep final iteration
        params, _ = jax.lax.scan(
            f=_optimize_scan_body,
            init=params,
            xs=jnp.arange(self.iterations),
            length=self.iterations,
        )

        return params

    def update_params(self, params: MIMPSParams, rollouts: QuadAlgTrajectory) -> MIMPSParams:
        """Update the mean with an exponentially weighted average.
        
        CaT (Constraints as Termination) is applied in get_rollouts using
        feet-only collision flags.  Argmin selection uses whole-body collision
        counts to pick the sample with fewest physical contacts.
        """
        # Just sum costs - CaT already applied in rollout
        costs = jnp.sum(rollouts.costs, axis=1)

        # Dual constraint flags: (N, H+1, 2) = [feet_flag, body_count]
        constraint_flags = rollouts.constraint_flags.astype(jnp.float32)
        feet_flags = constraint_flags[..., 0]   # (N, H+1) binary foot-hazard contacts
        # body_flags = constraint_flags[..., 1]    # (N, H+1) whole-body contact counts

        # Update c_max: adapt to observed FEET-ONLY violations (discriminative)
        batch_max_violation = jnp.max(feet_flags)
        new_c_max = jnp.maximum(batch_max_violation, 1.0)

        # First num_samples are from mean around ref
        costs_mean = costs[: self.num_samples]

        # Shape: (num_samples, H, plan_dim)
        plans_mean = rollouts.plans[: self.num_samples]

        # Update mean (around ref)
        # Shape: (num_samples, H, plan_dim)
        increments_mean = plans_mean - params.ref
        # Shape: (num_samples,)
        # Normalize costs by their std before weighting so that `temperature` is a
        # dimensionless sharpness parameter independent of cost magnitude.  Without
        # this, large safety_weight values cause exp(-cost/T) to underflow to 0 for
        # all-but-the-best sample, collapsing MPPI to pure argmin selection and
        # producing NaN weighted averages when costs contain outliers.
        #
        # Guard against NaN/Inf: if any cost is NaN (e.g. from corrupted physics
        # state), replace it with 0 for std computation and clip normalized costs
        # to prevent exp overflow in softmax.
        costs_mean_safe = jnp.where(jnp.isfinite(costs_mean), costs_mean, 0.0)
        cost_std = jnp.std(costs_mean_safe)
        costs_normed = (costs_mean_safe - jnp.mean(costs_mean_safe)) / (cost_std + 1e-6)
        costs_normed = jnp.clip(costs_normed, -100.0, 100.0)  # prevent exp overflow
        weights_mean = jax.nn.softmax(-costs_normed / self.temperature, axis=0)
        # Shape: (num_plans, plan_dim)
        new_mean = jnp.sum(weights_mean[:, None, None] * increments_mean, axis=0)


        new_best = rollouts.plans[jnp.argmin(costs)]


        # Argmin selection uses WHOLE-BODY collision count to pick the
        # sample with fewest physical contacts (swing arcs, calf brushes, etc.)
        # body_sums = jnp.sum(body_flags, axis=1)  # (N,) total body contacts per sample
        # best_index = jnp.argmin(body_sums)
        # new_best = rollouts.plans[best_index]  # Shape: (H, plan_dim)

        # # Tie-breaking: if all samples have similar body contact counts, fall
        # # back to the stance-anchored plan (plans[-2]) for EXECUTION.
        # # best_plan is intentionally kept as argmin (not stance) so that
        # # prepare_optimize warm-starts from the optimizer's current search
        # # direction rather than resetting to stance every tied cycle.
        # # exec_plan is overridden in finalize_best_plan, which in
        # # QuadMIMPSPlanner projects exec_plan (not best_plan) for execution.
        # range_ratio = (jnp.max(body_sums) - jnp.min(body_sums)) / (jnp.mean(body_sums) + 1e-6)
        # constraint_tied = range_ratio < 0.01
        # stance_plan = rollouts.plans[-2]  # second-to-last sample = stance-anchored
        # exec_plan = jax.lax.cond(
        #     constraint_tied,
        #     lambda: self.task.get_exec_plan(stance_plan),   # stance-anchored: feet stay put
        #     lambda: self.task.get_exec_plan(new_best),       # avoidance argmin
        # )

        return params.replace(  # type: ignore
            c_max=new_c_max,
            mean=new_mean,
            best_plan=new_best,
            # exec_plan=exec_plan,
            # min_violations=body_sums[best_index],
        )

    #### Can be called with super/modified by child classes ####

    def init_params(
        self,
        task_params: QuadRLTaskParams,
        initial_mean: Optional[jax.Array] = None,
        ref_traj: Optional[jax.Array] = None,
        seed: int = 0,
    ) -> MIMPSParams:
        """Initialize the policy parameters."""
        _params = super().init_params(task_params, initial_mean, ref_traj, seed)

        abs_plan = (_params.ref + _params.mean).copy()

        return MIMPSParams(
            mean=_params.mean,
            best_plan=abs_plan,
            exec_plan=self.task.get_exec_plan(abs_plan),
            # min_violations=jnp.zeros(()),
            ref=_params.ref,
            rng=_params.rng,
            task_params=_params.task_params,
            c_max=_params.c_max,
            train_data=self._empty_train_data(self.num_samples) if self.collect_data else None,
        )

    def sample_plans(self, params: MIMPSParams) -> Tuple[jax.Array, MIMPSParams]:
        """Sample morphologically and temporally coherent plans.

        Body-level xy noise shifts all 4 feet together — the dominant mode for
        obstacle avoidance.  A constant + linear-trend basis keeps the body
        shift smooth across horizon steps.  Per-foot residuals allow small
        individual adjustments within the morphological constraint.
        """
        rng, body_const_rng, body_slope_rng, foot_rng = jax.random.split(params.rng, 4)

        N = self.num_samples
        H = self.plan_horizon

        # --- 3-zone exploration: low / med / high noise multipliers ---
        # Low zone  refines near the current mean (exploitation).
        # Med zone  uses the nominal noise_level.
        # High zone explores far from the mean to escape large obstacles.
        N_low  = N // 3
        N_med  = N // 3
        N_high = N - N_low - N_med
        zone_scales = jnp.concatenate([
            jnp.full((N_low,),  0.2 * self.noise_level),
            jnp.full((N_med,),  0.5 * self.noise_level),
            jnp.full((N_high,), 1.5 * self.noise_level),
        ])  # (N,)

        # --- Body-level xy noise (all 4 feet shift together) ---
        body_const = jax.random.normal(body_const_rng, (N, 1, 2))          # (N, 1, 2)
        body_slope = jax.random.normal(body_slope_rng, (N, 1, 2)) * 0.3   # (N, 1, 2)
        t = jnp.linspace(0.0, 1.0, H).reshape(1, H, 1)                   # (1, H, 1)
        body_xy = body_const + body_slope * t                              # (N, H, 2)

        # Tile to all 4 feet: (N, H, 4, 2)
        body_xy_4feet = jnp.broadcast_to(body_xy[:, :, None, :], (N, H, 4, 2))

        # --- Per-foot residual noise (small individual adjustment) ---
        foot_noise = jax.random.normal(foot_rng, (N, H, 4, 3))
        foot_xy = foot_noise[..., :2]
        foot_z  = foot_noise[..., 2:] * 0.01  # z stays near ground

        # --- Combine and apply per-sample zone scale ---
        combined_xy = 0.7 * body_xy_4feet + 0.3 * foot_xy   # (N, H, 4, 2)
        combined = jnp.concatenate([combined_xy, foot_z], axis=-1)  # (N, H, 4, 3)
        noise = combined.reshape(N, H, self.task.plan_dim)
        noise = noise * zone_scales[:, None, None]  # (N, H, plan_dim)

        knots = params.mean + params.ref + noise

        # Last sample is always the plain reference (zero exploration)
        knots = knots.at[-1].set(params.ref)

        # Second-to-last sample anchors all feet at current stance positions.
        # Guarantees at least one near-side option when the warm-start has
        # drifted to the far side of an obstacle.  Without this, ALL samples
        # project to the same (far) side after the warm-start crosses through
        # an obstacle, giving MPPI zero near-side plans to select from.
        stance_world = params.task_params.init_foot_positions  # (4, 3)
        stance_plan_world = jnp.tile(
            stance_world.reshape(1, self.task.plan_dim), (H, 1)
        )  # (H, plan_dim)
        stance_plan_norm = stance_plan_world / self.task.plan_lim
        knots = knots.at[-2].set(stance_plan_norm)

        current_feet_xy = params.task_params.init_foot_positions[:, :2]
        knots = self.task.project_plan_to_safe_with_arc_check(knots, current_feet_xy)
        # knots = self.task.project_plan_to_safe(knots)

        return knots, params.replace(rng=rng)  # type: ignore

    def prepare_optimize(self, state: mjx.Data, params: MIMPSParams) -> MIMPSParams:
        # Use best_plan (stored in absolute normalized coords) for warm-starting.
        # This is critical: mean is an offset from ref, but ref changes between
        # planning cycles (goal_idx advances, reference positions update).
        # Using ref + mean would corrupt the absolute plan by (new_ref - old_ref).
        # best_plan is ref-independent, so it survives ref changes correctly.
        abs_plan = params.best_plan
        abs_plan = jnp.concatenate([abs_plan[1:], params.ref[-1:]], axis=0)

        abs_plan = jnp.clip(abs_plan, self.task.u_min, self.task.u_max)

        # Project warm-start plan to safe set so MPPI starts from
        # a feasible reference and doesn't warm-start into an obstacle.
        # abs_plan = self.task.project_plan_to_safe(abs_plan)

        new_mean = abs_plan - params.ref

        params = params.replace(  # type: ignore
            mean=new_mean,
            best_plan=abs_plan,
            exec_plan=self.task.get_exec_plan(abs_plan),
        )

        return params

    def update_without_plan(self, params: MIMPSParams):
        zero_offset = jnp.zeros_like(params.ref)
        return params.replace(  # type: ignore
            mean=zero_offset,
            best_plan=params.ref,
            exec_plan=self.task.get_exec_plan(params.ref),
            train_data=(
                self._empty_train_data(self.num_samples) if self.collect_data else None
            )
        )

    def finalize_best_plan(self, params: MIMPSParams) -> MIMPSParams:
        """Finalize the best plan by clip and denormalizing if necessary."""

        """Materialize the controller-facing execution plan without mutating warm-start state."""
        abs_plan = params.mean + params.ref

        abs_plan = self.task.project_plan_to_safe(abs_plan)

        exec_plan = self.task.get_exec_plan(abs_plan)

        # # Store the MPPI weighted-mean absolute plan for warm-starting the next cycle.
        # # This is smoother than the argmin best_plan from update_params.
        # return params.replace(exec_plan=exec_plan, best_plan=abs_plan)  # type: ignore

        return params.replace(exec_plan=exec_plan)

    # def finalize_best_plan(self, params: MIMPSParams) -> MIMPSParams:
    #     """Project exec_plan to the safe set and finalize for execution.

    #     exec_plan is set in update_params with tie-breaking applied:
    #     - Normally: exec_plan = get_exec_plan(argmin_plan)
    #     - When all samples tie (equal body contacts): exec_plan = get_exec_plan(stance_plan)

    #     best_plan is always the argmin plan, preserved for warm-starting next cycle
    #     via prepare_optimize.  Separating exec_plan from best_plan ensures the
    #     tie-breaking only affects execution, not the optimizer's search state.

    #     The exec plan is projected to the safe set so that even the selected
    #     sample — which may still clip an obstacle corner — has its feet pushed to
    #     the nearest safe position.
    #     """
    #     # Recover the denormalized plan from the current exec_plan for projection.
    #     # exec_plan has shape (4, H, 3) from get_exec_plan; invert back to (H, plan_dim).
    #     # Simpler: project best_plan (always argmin, not stance) for safe warm-start,
    #     # but project exec_plan's underlying plan for actual execution.
    #     #
    #     # exec_plan is already (4, H, 3); to project we need (H, plan_dim) normalized form.
    #     # Re-derive from params.best_plan if exec plan == argmin, or leave as-is if stance.
    #     # Safest: just project exec_plan's normalized equivalent via the stored exec_plan.
    #     #
    #     # Since exec_plan is get_exec_plan(plan) = denorm(plan).reshape(4,H,3).T-ish,
    #     # we cannot easily invert it.  Instead, project best_plan for the warm-start
    #     # (done in prepare_optimize) and project exec_plan by re-deriving from exec_plan.
    #     #
    #     # Practical approach: exec_plan carries the tie-breaking choice.  Project it
    #     # by converting back to (H, plan_dim) normalized, projecting, re-applying get_exec_plan.
    #     # exec_plan shape: (4, H, 3) world-frame → normalize → (H, plan_dim) → project → exec.
    #     exec_world = params.exec_plan                          # (4, H, 3)
    #     H = exec_world.shape[1]
    #     exec_norm = (exec_world.transpose(1, 0, 2).reshape(H, self.task.plan_dim)
    #                  / self.task.plan_lim)                    # (H, plan_dim) normalized
    #     projected_norm = self.task.project_plan_to_safe(exec_norm[None])[0]  # (H, plan_dim)
    #     exec_plan = self.task.get_exec_plan(projected_norm)

    #     return params.replace(exec_plan=exec_plan)
