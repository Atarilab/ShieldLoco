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
        plan_horizon: float = 1.0,
        iterations: int = 1,
        gamma: float = 0.99,
        safety_weight: float = 1e3,
        collect_data: bool = False,
        top_fraction: float = 0.01,
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
            plan_horizon: The time horizon for the rollout in seconds.
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
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(
            rollouts.costs + self.safety_weight * rollouts.constraint_flags,
            axis=1,
        )

        # N.B. jax.nn.softmax takes care of details like baseline subtraction.

        # First num_samples are from mean around ref
        costs_mean = costs[: self.num_samples]

        # Shape: (num_samples, H, plan_dim)
        plans_mean = rollouts.plans[: self.num_samples]


        # Update mean (around ref)
        # Shape: (num_samples, H, plan_dim)
        increments_mean = plans_mean - params.ref
        # Shape: (num_samples,)
        weights_mean = jax.nn.softmax(-costs_mean / self.temperature, axis=0)
        # Shape: (num_plans, plan_dim)
        new_mean = jnp.sum(weights_mean[:, None, None] * increments_mean, axis=0)


        # TODO: sample from the mean
        # Get best knots (lowest cost among all)
        best_index = jnp.argmin(costs)
        best_plan = rollouts.plans[best_index]  # Shape: (H, plan_dim)

        return params.replace(  # type: ignore
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

        # best_plan should be absolute controls, not offset-only.
        best_plan = (_params.ref + _params.mean).copy()

        return MIMPSParams(
            mean=_params.mean,
            best_plan=best_plan,
            ref=_params.ref,
            rng=_params.rng,
            task_params=_params.task_params,
            train_data=self._empty_train_data(self.num_samples) if self.collect_data else None,
        )

    def sample_plans(self, params: MIMPSParams) -> Tuple[jax.Array, MIMPSParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.plan_steps,
                self.task.plan_dim,
            ),
        )
        # Shape: (num_samples, H, plan_dim)
        mean = params.mean + params.ref + self.noise_level * noise

        return mean, params.replace(rng=rng)  # type: ignore

    def prepare_optimize(self, state: mjx.Data, params: MIMPSParams) -> MIMPSParams:
        # mean is an offset; clip in absolute control space, then map back.
        abs_plan = params.ref + params.mean
        abs_plan = jnp.clip(abs_plan, self.task.u_min, self.task.u_max)
        new_mean = abs_plan - params.ref

        params = params.replace(  # type: ignore
            mean=new_mean
        )

        return params

    def update_without_plan(self, params: MIMPSParams):
        zero_offset = jnp.zeros_like(params.ref)
        return params.replace(  # type: ignore
            mean=zero_offset,
            best_plan=params.ref,
            train_data=(
                self._empty_train_data(self.num_samples) if self.collect_data else None
            )
        )

    def finalize_best_plan(self, params: MIMPSParams) -> MIMPSParams:
        """Finalize the best plan by clip and denormalizing if necessary."""
        best_plan = self.task.get_exec_plan(params.best_plan)

        return params.replace(best_plan=best_plan)  # type: ignore
