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

    Same as QuadAlgSamplingParams (best_plan, exec_plan now inherited from base).

    Attributes:
        best_plan: The best plan spline knots found so far.
        exec_plan: Materialized execution plan for controller output.
    """
    pass


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
        use_momentum: bool = False,
        momentum: float = 0.5,
        top_fraction: float = 0.01,
        p_max: float = 1.0,
        c_max_decay: float = 0.9,
        include_terminal_cost: bool = True,
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
            num_samples=num_samples,
            noise_level=noise_level,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,  # type: ignore
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

        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        self.noise_level = noise_level
        self.num_samples = num_samples
        self.temperature = temperature
        self.gamma = gamma
        self.safety_weight = safety_weight
        self.collect_data = collect_data
        self.top_fraction = top_fraction

        self.use_momentum = use_momentum
        self.momentum = momentum

    def update_with_plan(self, state, params):
        def _optimize_scan_body(params, _):
            # Sample random control sequences from spline knots
            plans, params = self.sample_plans(params)

            params = self.f(state, params, plans)

            # f() sets train_data — extract it and remove from carry
            # so that the scan carry type stays consistent across iterations.
            train_data = params.train_data
            params = params.replace(train_data=None)

            return params, train_data

        # Don't accumulate all iterations - only keep final iteration
        init_no_train = params.replace(train_data=None)
        params, train_data_out = jax.lax.scan(
            f=_optimize_scan_body,
            init=init_no_train,
            xs=jnp.arange(self.iterations),
            length=self.iterations,
        )

        # Restore train_data from the last scan iteration.
        # jax.lax.scan always stacks ys along a new leading axis.
        if train_data_out is not None:
            def _take_last(x):
                return x[-1]
            params = params.replace(
                train_data=jax.tree.map(_take_last, train_data_out)
            )

        return params

    def update_params(self, params: MIMPSParams, rollouts: QuadAlgTrajectory) -> MIMPSParams:
        """Update the mean with an exponentially weighted average.
        
        CaT (Constraints as Termination) is applied in get_rollouts using
        feet-only collision flags.  Argmin selection uses whole-body collision
        counts to pick the sample with fewest physical contacts.
        """
        costs = jnp.sum(rollouts.costs, axis=1)

        costs_mean = costs[: self.num_samples]
        plans_mean = rollouts.plans[: self.num_samples]

        increments_mean = plans_mean - params.ref
        costs_mean_safe = jnp.where(jnp.isfinite(costs_mean), costs_mean, 0.0)
        cost_std = jnp.std(costs_mean_safe)
        costs_normed = (costs_mean_safe - jnp.mean(costs_mean_safe)) / (cost_std + 1e-6)
        costs_normed = jnp.clip(costs_normed, -100.0, 100.0)
        weights_mean = jax.nn.softmax(-costs_normed / self.temperature, axis=0)
        new_mean = jnp.sum(weights_mean[:, None, None] * increments_mean, axis=0)

        new_best = rollouts.plans[jnp.argmin(costs)]

        if self.use_momentum:
            # S Sanokowski, et. al. formulation
            new_mean = new_mean + (self.momentum / (1 - self.momentum)) * (params.mean - new_mean)

            # TODO: add variance updates to MPPI and Replica Exchange
            # Select alpha < (self.momentum + 1) to avoid singularities
            # new_variance = new_variance + (1 / (self.momentum + 1 - alpha)) * (self.momentum * (params.variance - new_variance) + alpha * new_variance)
        
        return params.replace(
            mean=new_mean,
            best_plan=new_best,
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
        rng, noise_rng = jax.random.split(params.rng)
        noise = self.quad_noise(noise_rng)

        knots = params.mean + params.ref + noise

        # Last sample is always the plain reference (zero exploration)
        knots = knots.at[-1].set(params.ref)

        # Second-to-last sample anchors all feet at current stance positions.
        # Guarantees at least one near-side option when the warm-start has
        # drifted to the far side of an obstacle.  Without this, ALL samples
        # project to the same (far) side after the warm-start crosses through
        # an obstacle, giving MPPI zero near-side plans to select from.
        # stance_world = params.task_params.init_foot_positions  # (4, 3)
        # stance_plan_world = jnp.tile(
        #     stance_world.reshape(1, self.task.plan_dim), (H, 1)
        # )  # (H, plan_dim)
        # stance_plan_norm = stance_plan_world / self.task.plan_lim
        # knots = knots.at[-2].set(stance_plan_norm)

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
