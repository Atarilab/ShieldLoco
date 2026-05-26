"""Cross-Entropy Method MPC optimizer (CEMPS).

CEM optimizes a Gaussian sampling distribution over control sequences by:
1. Sampling N plans from N(mean + ref, std)
2. Evaluating all samples via full-simulation rollout
3. Selecting the top elite_fraction samples (lowest cost)
4. Fitting new mean and std to the elite subset
5. Repeating for iterations cycles

Unlike MPPI which uses soft temperature-weighted averaging over ALL samples,
CEM uses hard selection — only elites shape the next distribution.
"""

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from typing import Tuple, Optional

from .bases.alg import QuadAlgSamplingParams, QuadSamplingBasedPlanner, QuadAlgTrajectory
from .task import QuadRLTaskParams, QuadRLTask

from safety_filter.bases.risk import RiskStrategy


@dataclass
class CEMPSParams(QuadAlgSamplingParams):
    """Policy parameters for Cross-Entropy Method MPC.

    Attributes:
        mean: Offset around the reference trajectory, shape (H, plan_dim).
        std: Per-element standard deviation of sampling distribution, shape (H, plan_dim).
        best_plan: Absolute normalized plan for warm-starting, shape (H, plan_dim).
        exec_plan: Materialized execution plan (4, H, 3) for controller output.
    """
    std: jax.Array = jnp.zeros((1, 1))
    best_plan: Optional[jax.Array] = None
    exec_plan: Optional[jax.Array] = None


class QuadCEMPSPlanner(QuadSamplingBasedPlanner):
    """Cross-Entropy Method planner for contact-space optimization.

    Runs iterative CEM optimization:
    - Sample plans from Gaussian distribution parameterized by (mean, std)
    - Evaluate via full-simulation rollout
    - Select elite fraction (lowest cost samples)
    - Refit distribution parameters to elites
    - Repeat for iterations cycles
    """

    def __init__(
        self,
        task: QuadRLTask,
        num_samples: int,
        noise_level: float,
        elite_fraction: float = 0.1,
        cross_entropy_decay: float = 0.8,
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
        """Initialize CEM planner.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample per iteration.
            noise_level: Initial standard deviation for sampling distribution.
            elite_fraction: Fraction of samples to keep as elites (default 0.1).
            cross_entropy_decay: Decay rate for std toward noise_level (0 = no decay).
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combine costs from different randomizations.
            seed: The random seed for domain randomization.
            plan_horizon: The timesteps of horizon for the rollout.
            iterations: The number of CEM refinement cycles.
            gamma: Discount factor for cost accumulation.
            safety_weight: Weight for CaT (Constraints as Termination) penalty.
            collect_data: Whether to collect training data.
            top_fraction: Fraction of best samples for training data.
            p_max: Maximum termination probability for CaT.
            c_max_decay: Decay rate for c_max adaptation.
        """
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

        self.elite_fraction = elite_fraction
        self.cross_entropy_decay = cross_entropy_decay

    def init_params(
        self,
        task_params: QuadRLTaskParams,
        initial_mean: Optional[jax.Array] = None,
        ref_traj: Optional[jax.Array] = None,
        seed: int = 0,
    ) -> CEMPSParams:
        """Initialize CEM parameters."""
        base_params = super().init_params(task_params, initial_mean, ref_traj, seed)

        H = self.plan_horizon
        plan_dim = self.task.plan_dim

        mean = base_params.mean
        std = jnp.full((H, plan_dim), self.noise_level, dtype=jnp.float32)

        abs_plan = base_params.ref + mean
        exec_plan = self.task.get_exec_plan(abs_plan)

        return CEMPSParams(
            mean=mean,
            std=std,
            best_plan=abs_plan,
            exec_plan=exec_plan,
            rng=base_params.rng,
            ref=base_params.ref,
            task_params=base_params.task_params,
            c_max=base_params.c_max,
            train_data=self._empty_train_data(self.num_samples) if self.collect_data else None,
        )

    def prepare_optimize(self, state: mjx.Data, params: CEMPSParams) -> CEMPSParams:
        """Warm-start: shift best_plan forward and std forward + decay toward noise_level."""
        abs_plan = params.best_plan
        abs_plan = jnp.concatenate([abs_plan[1:], params.ref[-1:]], axis=0)
        abs_plan = jnp.clip(abs_plan, self.task.u_min, self.task.u_max)

        H = self.plan_horizon
        plan_dim = self.task.plan_dim

        new_mean = abs_plan - params.ref

        # Shift std forward and decay toward noise_level
        std_shifted = jnp.concatenate([params.std[1:], params.std[-1:]], axis=0)
        new_std = self.std_decay * std_shifted + (1 - self.std_decay) * self.noise_level
        new_std = jnp.maximum(new_std, 0.1 * self.noise_level)

        params = params.replace(
            mean=new_mean,
            std=new_std,
            best_plan=abs_plan,
            exec_plan=self.task.get_exec_plan(abs_plan),
        )

        return params

    def sample_plans(self, params: CEMPSParams) -> Tuple[jax.Array, CEMPSParams]:
        """Sample plans from Gaussian distribution N(mean + ref, std)."""
        rng, sample_rng = jax.random.split(params.rng)

        noise = self.quad_noise(sample_rng)
        # quad_noise bakes in self.noise_level as a fixed scale factor.
        # For CEM we need raw (unit-variance-ish) noise so that the adaptive
        # params.std directly controls the sampling standard deviation.
        noise_raw = noise / jnp.maximum(self.noise_level, 1e-6)
        knots = params.mean[None, :, :] + params.ref[None, :, :] + noise_raw * params.std[None, :, :]

        knots = knots.at[-1].set(params.ref)

        current_feet_xy = params.task_params.init_foot_positions[:, :2]
        knots = self.task.project_plan_to_safe_with_arc_check(knots, current_feet_xy)

        return knots, params.replace(rng=rng)

    def update_with_plan(self, state: mjx.Data, params: CEMPSParams) -> CEMPSParams:
        """Run CEM optimization loop over iterations cycles."""

        def _cem_scan_body(params_iter, _):
            plans, params_iter = self.sample_plans(params_iter)

            params_iter = self.f(state, params_iter, plans)

            return params_iter, _

        params, _ = jax.lax.scan(
            f=_cem_scan_body,
            init=params,
            xs=jnp.arange(self.iterations),
            length=self.iterations,
        )

        return params

    def update_params(self, params: CEMPSParams, rollouts: QuadAlgTrajectory) -> CEMPSParams:
        """Update mean and std by fitting to elite samples."""
        costs = jnp.sum(rollouts.costs, axis=1)

        plans = rollouts.plans
        increments = plans - params.ref[None, :, :]

        num_elites = max(1, int(self.num_samples * self.elite_fraction))
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
            new_std = self.cross_entropy_decay * new_std + (1 - self.cross_entropy_decay) * self.noise_level
            new_std = jnp.maximum(new_std, 0.1 * self.noise_level)

        new_best = plans[jnp.argmin(costs)]

        return params.replace(
            mean=new_mean,
            std=new_std,
            best_plan=new_best,
        )

    def update_without_plan(self, params: CEMPSParams) -> CEMPSParams:
        """Reset to zero offset (pure reference tracking)."""
        H = self.plan_horizon
        plan_dim = self.task.plan_dim

        zero_mean = jnp.zeros((H, plan_dim), dtype=jnp.float32)
        zero_std = jnp.full((H, plan_dim), self.noise_level, dtype=jnp.float32)

        return params.replace(
            mean=zero_mean,
            std=zero_std,
            best_plan=params.ref,
            exec_plan=self.task.get_exec_plan(params.ref),
            train_data=(
                self._empty_train_data(self.num_samples) if self.collect_data else None
            ),
        )

    def finalize_best_plan(self, params: CEMPSParams) -> CEMPSParams:
        """Project exec_plan to the safe set and finalize for execution."""
        abs_plan = params.best_plan
        abs_plan = self.task.project_plan_to_safe(abs_plan)
        exec_plan = self.task.get_exec_plan(abs_plan)
        return params.replace(exec_plan=exec_plan)
