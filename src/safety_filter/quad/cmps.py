import jax
import jax.numpy as jnp
from mujoco import mjx

from typing import Any, Tuple, Optional

from .bases.alg import QuadAlgSamplingParams, QuadSamplingBasedPlanner, QuadAlgTrajectory
from .task import QuadRLTaskParams, QuadRLTask

from safety_filter.bases.risk import RiskStrategy


class QuadCMPSPlanner(QuadSamplingBasedPlanner):
    """Consensus-based optimization planner stub.
    
    Consensus optimization finds safe plans by:
    1. Each foot operates as independent agent
    2. Agents iteratively agree on joint plan
    3. Step size controls convergence speed vs stability
    """

    def __init__(
        self,
        task: QuadRLTask,
        num_samples: int,
        consensus_iters: int = 10,
        step_size: float = 0.5,
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
        include_terminal_cost: bool = True,
    ) -> None:
        """Initialize consensus planner.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            consensus_iters: Number of consensus iterations per optimization step.
            step_size: Step size for consensus updates (0 < step_size <= 1).
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combine costs from different randomizations.
            seed: The random seed for domain randomization.
            plan_horizon: The timesteps of horizon for the rollout.
            iterations: The number of optimization iterations to perform.
            gamma: Discount factor for cost accumulation.
            safety_weight: Weight for CaT (Constraints as Termination) penalty.
            collect_data: Whether to collect training data.
            top_fraction: Fraction of best samples for training data.
            p_max: Maximum termination probability for CaT.
            c_max_decay: Decay rate for c_max adaptation.
        """
        self.consensus_iters = consensus_iters
        self.step_size = step_size

        super().__init__(
            task=task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            iterations=iterations,
            gamma=gamma,
            safety_weight=safety_weight,
            collect_data=collect_data,
            top_fraction=top_fraction,
            p_max=p_max,
            c_max_decay=c_max_decay,
            include_terminal_cost=include_terminal_cost,
        )

        self.num_samples = num_samples

    def prepare_optimize(self, state: mjx.Data, params: QuadAlgSamplingParams) -> QuadAlgSamplingParams:
        raise NotImplementedError(".prepare_optimize not implemented")

    def update_with_plan(self, state: mjx.Data, params: QuadAlgSamplingParams) -> QuadAlgSamplingParams:
        raise NotImplementedError(".update_with_plan not implemented")

    def update_without_plan(self, params: QuadAlgSamplingParams) -> QuadAlgSamplingParams:
        raise NotImplementedError(".update_without_plan not implemented")

    def sample_plans(self, params: QuadAlgSamplingParams) -> Tuple[jax.Array, QuadAlgSamplingParams]:
        raise NotImplementedError(".sample_plans not implemented")

    def update_params(self, params: QuadAlgSamplingParams, rollouts: QuadAlgTrajectory) -> QuadAlgSamplingParams:
        raise NotImplementedError(".update_params not implemented")

    def finalize_best_plan(self, params: QuadAlgSamplingParams) -> QuadAlgSamplingParams:
        raise NotImplementedError(".finalize_best_plan not implemented")

    def init_params(
        self,
        task_params: QuadRLTaskParams,
        initial_mean: Optional[jax.Array] = None,
        ref_traj: Optional[jax.Array] = None,
        seed: int = 0,
    ) -> QuadAlgSamplingParams:
        raise NotImplementedError(".init_params not implemented")