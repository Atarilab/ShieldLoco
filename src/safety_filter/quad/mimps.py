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
    """
    best_plan: Optional[jax.Array] = None
    exec_plan: Optional[jax.Array] = None


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
        
        CaT (Constraints as Termination) is now applied in get_rollouts - 
        constraint violations modulate the discount factor during rollout.
        This just computes the MPPI weights and updates c_max.
        """
        # Just sum costs - CaT already applied in rollout
        costs = jnp.sum(rollouts.costs, axis=1)

        # Update c_max: adapt to observed constraint violations
        constraint_flags = rollouts.constraint_flags.astype(jnp.float32)
        batch_max_violation = jnp.max(constraint_flags)
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
        cost_std = jnp.std(costs_mean)
        costs_normed = (costs_mean - jnp.mean(costs_mean)) / (cost_std + 1e-6)
        weights_mean = jax.nn.softmax(-costs_normed / self.temperature, axis=0)
        # Shape: (num_plans, plan_dim)
        new_mean = jnp.sum(weights_mean[:, None, None] * increments_mean, axis=0)

        # TODO: sample from the mean
        # Get best knots (lowest cost among all)
        best_index = jnp.argmin(costs)
        best_plan = rollouts.plans[best_index]  # Shape: (H, plan_dim)

        return params.replace(  # type: ignore
            c_max=new_c_max,
            mean=new_mean,
            best_plan=best_plan,
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
        rng, body_const_rng, body_slope_rng, foot_rng = jax.random.split(params.rng, 4)

        N = self.num_samples
        H = self.plan_horizon

        # --- Body-level xy noise (all 4 feet shift together) ---
        # Constant + gentle linear trend → temporally coherent across horizon.
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

        # --- Combine: body dominates xy, foot residuals are small ---
        combined_xy = 0.3 * body_xy_4feet + 0.7 * foot_xy   # (N, H, 4, 2)
        combined = jnp.concatenate([combined_xy, foot_z], axis=-1)  # (N, H, 4, 3)

        # Flatten to plan_dim: (N, H, 12)
        noise = combined.reshape(N, H, self.task.plan_dim)

        # Analytically-guided mean bias: shift all samples toward the dodge
        # position predicted from the dynamic obstacle trajectory.  This centres the
        # MPPI sampling distribution around whole-body avoidance so the
        # optimizer can find physics-feasible dodge plans in a single cycle.
        dyna_safety_bias = self.task.get_dynamic_safety_bias(
            params.task_params, params.ref
        )  # (H, plan_dim)

        # Shape: (num_samples, H, plan_dim)
        mean = (params.mean + dyna_safety_bias) + params.ref + self.noise_level * noise

        return mean, params.replace(rng=rng)  # type: ignore

    def prepare_optimize(self, state: mjx.Data, params: MIMPSParams) -> MIMPSParams:
        # # Use best_plan (stored in absolute normalized coords) for warm-starting.
        # # This is critical: mean is an offset from ref, but ref changes between
        # # planning cycles (goal_idx advances, reference positions update).
        # # Using ref + mean would corrupt the absolute plan by (new_ref - old_ref).
        # # best_plan is ref-independent, so it survives ref changes correctly.
        # abs_plan = params.best_plan
        # abs_plan = jnp.concatenate([abs_plan[1:], params.ref[-1:]], axis=0)

        # Shift the previous best plan forward by one step to preserve corridor structure.
        # Append the last reference control to the end of the shifted plan to maintain the same length.
        abs_plan = params.ref + params.mean # Shape: (H, plan_dim)
        abs_plan = jnp.concatenate([abs_plan[1:], params.ref[-1:]], axis=0)

        abs_plan = jnp.clip(abs_plan, self.task.u_min, self.task.u_max)
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

        exec_plan = self.task.get_exec_plan(abs_plan)

        # # Store the MPPI weighted-mean absolute plan for warm-starting the next cycle.
        # # This is smoother than the argmin best_plan from update_params.
        # return params.replace(exec_plan=exec_plan, best_plan=abs_plan)  # type: ignore

        return params.replace(exec_plan=exec_plan)  # type: ignore
