import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from typing import Dict, Tuple, Optional

from .bases.alg import QuadAlgSamplingParams, QuadSamplingBasedPlanner, QuadAlgTrajectory
from .task import QuadRLTaskParams, QuadRLTask

from safety_filter.bases.risk import RiskStrategy


# MIMPS = MPPI-based Model Predictive Shielding (MPS)
@dataclass
class MIMPSParams(QuadAlgSamplingParams):
    """Policy parameters for model-predictive path integral control.

    Attributes:
        best_plan: The best plan spline knots found so far.
        exec_plan: Materialized execution plan for controller output.
        std: Per-element standard deviation of sampling distribution, shape (H, plan_dim).
    """
    std: jax.Array = jnp.zeros((1, 1))


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
        use_momentum: bool = False,
        momentum: float = 0.5,
        p_max: float = 1.0,
        c_max_decay: float = 0.9,
        include_terminal_cost: bool = True,
        std_decay: float = 0.9,
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
            std_decay=std_decay,
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
        raw_mean = jnp.sum(weights_mean[:, None, None] * increments_mean, axis=0)

        new_best = rollouts.plans[jnp.argmin(costs)]

        if self.use_momentum:
            new_mean = raw_mean + (self.momentum / (1 - self.momentum)) * (raw_mean - params.mean)

            diffs = increments_mean - new_mean[None, :, :]
            s = jnp.sum(weights_mean[:, None, None] * (diffs ** 2), axis=0)

            old_variance = params.std ** 2
            alpha = (self.momentum + 1) * 0.1
            new_variance = s + (1 / (self.momentum + 1 - alpha)) * (
                self.momentum * (old_variance - s) + alpha * s
            )
            new_std = jnp.sqrt(jnp.maximum(new_variance, (0.1 * self.noise_level) ** 2))
        else:
            new_mean = raw_mean
            new_std = params.std
        
        # ── constraint diagnostics over all samples ──
        n_total = rollouts.constraint_flags.shape[0]
        feet_flags  = rollouts.constraint_flags[:, :, 0]   # (N, H+1)
        body_counts = rollouts.constraint_flags[:, :, 1]   # (N, H+1)
        total_feet  = jnp.sum(feet_flags, axis=1)           # (N,)
        total_body  = jnp.sum(body_counts, axis=1)          # (N,)
        has_foot = total_feet > 0
        has_body = total_body > 0
        n_clean     = jnp.sum(~has_foot & ~has_body).astype(jnp.float32)
        n_foot_only = jnp.sum( has_foot & ~has_body).astype(jnp.float32)
        n_body_only = jnp.sum(~has_foot &  has_body).astype(jnp.float32)
        n_both      = jnp.sum( has_foot &  has_body).astype(jnp.float32)
        first_hit   = jnp.argmax(feet_flags > 0, axis=1).astype(jnp.float32)  # (N,)
        never_hit   = total_feet == 0
        first_hit_mean = jnp.where(
            jnp.any(~never_hit),
            jnp.sum(jnp.where(~never_hit, first_hit, 0.0)) / jnp.maximum(jnp.sum(~never_hit).astype(jnp.float32), 1.0),
            -1.0,
        )
        diag = jnp.array([
            n_clean, n_foot_only, n_body_only, n_both,
            jnp.mean(total_feet), jnp.mean(total_body),
            first_hit_mean,
        ], dtype=jnp.float32)

        return params.replace(
            mean=new_mean,
            std=new_std,
            best_plan=new_best,
            diag=diag,
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

        H = self.plan_horizon
        plan_dim = self.task.plan_dim
        abs_plan = (_params.ref + _params.mean).copy()
        std = jnp.full((H, plan_dim), self.noise_level, dtype=jnp.float32)

        return MIMPSParams(
            mean=_params.mean,
            std=std,
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

        # quad_noise bakes in self.noise_level as a fixed scale factor.
        # Normalize so that params.std directly controls the per-element scale.
        noise_raw = noise / jnp.maximum(self.noise_level, 1e-6)
        knots = params.mean + params.ref + noise_raw * params.std[None, :, :]

        # Last sample is always the plain reference (zero exploration)
        knots = knots.at[-1].set(params.ref)

        current_feet_xy = params.task_params.init_foot_positions[:, :2]
        knots = self.task.project_plan_to_safe_with_arc_check(knots, current_feet_xy)

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

        new_mean = abs_plan - params.ref

        # Warm-start std: shift forward and decay toward noise_level
        std_shifted = jnp.concatenate([params.std[1:], params.std[-1:]], axis=0)
        new_std = self.std_decay * std_shifted + (1 - self.std_decay) * self.noise_level
        new_std = jnp.maximum(new_std, 0.1 * self.noise_level)
        # Safety: if std was reset to zeros (external reference change), restore to noise_level
        has_zero_std = jnp.all(new_std == 0.0)
        new_std = jnp.where(has_zero_std, jnp.full_like(new_std, self.noise_level), new_std)

        params = params.replace(  # type: ignore
            mean=new_mean,
            std=new_std,
            best_plan=abs_plan,
            exec_plan=self.task.get_exec_plan(abs_plan),
        )

        return params

    def update_without_plan(self, params: MIMPSParams):
        zero_offset = jnp.zeros_like(params.ref)
        H = self.plan_horizon
        plan_dim = self.task.plan_dim
        zero_std = jnp.full((H, plan_dim), self.noise_level, dtype=jnp.float32)
        return params.replace(  # type: ignore
            mean=zero_offset,
            std=zero_std,
            best_plan=params.ref,
            exec_plan=self.task.get_exec_plan(params.ref),
            train_data=(
                self._empty_train_data(self.num_samples) if self.collect_data else None
            )
        )

