from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Literal, Optional, Tuple


import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from safety_filter.quad.task import QuadRLTask
from safety_filter.quad.task import QuadRLTaskParams

from safety_filter.bases.risk import AverageCost, RiskStrategy


@dataclass
class QuadAlgState:
    """
    Data class for storing relevant state information during rollouts.
    """

    qpos: jax.Array  # Joint positions
    qvel: jax.Array  # Joint velocities
    xpos: jax.Array  # Site positions
    xquat: jax.Array  # Site orientations (quaternions)


@dataclass
class QuadAlgTrainData:
    """
    Data class for storing training data.
    """

    states: QuadAlgState  # shape: (num_traj, horizon + 1, state_dim)
    plans: jax.Array  # shape: (num_traj, horizon, plan_dim)
    costs: jax.Array  # shape: (num_traj, horizon + 1)
    constraint_flags: jax.Array  # shape: (num_traj, horizon + 1)

    optimized: bool = True


@dataclass
class QuadAlgTrajectory:
    """Data class for storing rollout data.

    Throughout, H denotes the number of control steps (given by the times at
    which the control spline is interpolated).

    Attributes:
        plans: Plan actions of shape (num_rollouts, H, plan_dim).
        costs: Costs of shape (num_rollouts, H+1).
        constraint_flags: Optional binary flags indicating constraint violations, shape (num_rollouts, H+1).
    """

    plans: jax.Array
    costs: jax.Array
    constraint_flags: jax.Array  # shape (num_rollouts, H+1)

    def __len__(self):
        """Return the number of time steps in the trajectory (T)."""
        return self.costs.shape[-1] - 1


@dataclass
class QuadAlgSamplingParams:
    """Parameters for sampling-based control algorithms.

    Attributes:
        mean: The mean of the plan distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
        ref: The reference trajectory that we are sampling around.

        task_params: Optional task parameters that can be updated during plan call
        
    """
    mean: jax.Array
    rng: jax.Array
    ref: jax.Array

    task_params: QuadRLTaskParams

    train_data: Optional[QuadAlgTrainData] = None


class QuadSamplingBasedPlanner(ABC):
    """An abstract sampling-based MPC algorithm interface."""

    def __init__(
        self,
        task: QuadRLTask,
        num_randomizations: int,
        risk_strategy: RiskStrategy,
        seed: int,
        plan_horizon: float,
        iterations: int = 1,
        gamma: float = 0.99,
        safety_weight: float = 1e3,
        collect_data: bool = False,
        top_fraction: float = 0.01,        
    ) -> None:
        """Initialize the MPC controller.

        Args:
            task: The task instance defining the dynamics and costs.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            iterations: The number of optimization iterations to perform.
            alpha_mean: The smoothing factor for the mean update.
            alpha_cov: The smoothing factor for the covariance update.
        """
        self.task = task
        self.num_randomizations = max(num_randomizations, 1)

        # Risk strategy defaults to average cost
        if risk_strategy is None:
            risk_strategy = AverageCost()
        self.risk_strategy = risk_strategy

        # time-related variables
        # NOTE: we always interpret self.task.model as the controller's
        # internal model, not the model used for simulation. dt is the
        # time between spline queries.
        self.plan_horizon = plan_horizon
        self.dt = self.task.dt
        self.plan_steps = int(round(self.plan_horizon / self.dt))


        # Use a single model (no domain randomization) by default
        self.model = task.model
        self.randomized_axes = None

        # Number of optimization iterations
        if iterations < 1:
            raise ValueError("iterations must be greater than 0!")

        self.iterations = iterations

        if self.num_randomizations > 1:
            # Make domain randomized models
            rng = jax.random.key(seed)
            rng, subrng = jax.random.split(rng)
            subrngs = jax.random.split(subrng, num_randomizations)
            randomizations = jax.vmap(self.task.domain_randomize_model)(subrngs)
            self.model = self.task.model.tree_replace(randomizations) # type: ignore

            # Keep track of which elements of the model have randomization
            self.randomized_axes = jax.tree.map(lambda x: None, self.task.model)
            self.randomized_axes = self.randomized_axes.tree_replace(
                {key: 0 for key in randomizations.keys()}
            )
        
        self.safety_weight = safety_weight
        self.collect_data = collect_data
        self.top_fraction = top_fraction
        self.gamma = gamma

    def init_params(
        self, task_params: QuadRLTaskParams, initial_mean: Optional[jax.Array] = None, ref_traj: Optional[jax.Array] = None, seed: int = 0
    ) -> QuadAlgSamplingParams:
        """Initialize the policy parameters, U = [u₀, u₁, ... ] ~ π(params).

        Args:
            initial_mean: The initial mean of the plan distribution.
            ref_traj: The initial reference trajectory.
            seed: The random seed for initializing the policy parameters.

        Returns:
            The initial policy parameters.
        """
        rng = jax.random.key(seed)
        
        mean = (
            initial_mean
            if initial_mean is not None
            else jnp.zeros((self.plan_steps, self.task.plan_dim))
        )
        assert mean.shape == (self.plan_steps, self.task.plan_dim), (
            f"Initial mean must have shape (plan_steps, plan_dim), got {mean.shape}"
        )

        ref = (
            ref_traj
            if ref_traj is not None
            else jnp.zeros((self.plan_steps, self.task.plan_dim))
        )
        assert ref.shape == (self.plan_steps, self.task.plan_dim), (
            f"Initial reference trajectory must have shape (plan_steps, plan_dim), got {ref.shape}"
        )

        return QuadAlgSamplingParams(mean=mean, rng=rng, ref=ref, task_params=task_params)

    def optimize(self, state: mjx.Data, params: QuadAlgSamplingParams, solve: bool) -> QuadAlgSamplingParams:
        """Perform an optimization step to update the policy parameters.

        Args:
            state: The initial state x₀.
            params: The current policy parameters, U ~ π(params).

        Returns:
            Updated policy parameters
        """
        params = self.prepare_optimize(state, params)

        def _optimize(params):
            return self.update_with_plan(state, params)

        def _no_optimize(params):
            return self.update_without_plan(params)

        # lax.cond
        params = jax.lax.cond(
            solve,
            _optimize,
            _no_optimize,
            params,
        )

        params = self.finalize_best_plan(params)        

        return params

    def f(self, state: mjx.Data, params: QuadAlgSamplingParams, plans: jax.Array) -> QuadAlgSamplingParams:
        # Clip final plans between min and max plan val
        # (self.num_rollouts, plan_steps, plan_dim)
        plans = jnp.clip(plans, self.task.u_min, self.task.u_max)

        # Roll out the control sequences, applying domain randomizations and
        # combining costs using self.risk_strategy.
        rng, dr_rng = jax.random.split(params.rng)
        # (self.num_rollouts, H, plan_dim)
        rollouts, train_data = self.rollout_with_randomizations(
            params, state, plans, dr_rng
        )
        params = params.replace(rng=rng)  # type: ignore

        # Update the policy parameters based on the combined costs
        params = self.update_params(params, rollouts)

        return params.replace(train_data=train_data)  # type: ignore

    def rollout_with_randomizations(
        self,
        params: QuadAlgSamplingParams,
        state: mjx.Data,
        plans: jax.Array,
        rng: jax.Array,
    ) -> Tuple[QuadAlgTrajectory, Optional[QuadAlgTrainData]]:
        """Compute rollout costs, applying domain randomizations.

        Args:
            params: The current policy parameters.
            state: The initial state x₀.
            plans: The control spline plans, (num rollouts, num_knots, plan_dim).
            rng: The random number generator key for randomizing initial states.

        Returns:
            Trajectory and TrainData
        """

        # Set the initial state for each rollout.
        states = jax.vmap(lambda _, x: x, in_axes=(0, None))(
            jnp.arange(self.num_randomizations), state
        )

        if self.num_randomizations > 1:
            # Randomize the initial states for each domain randomization
            subrngs = jax.random.split(rng, self.num_randomizations)
            randomizations = jax.vmap(self.task.domain_randomize_data)(states, subrngs)
            states = states.tree_replace(randomizations)

        # Apply the control sequences, parallelized over both rollouts and
        # domain randomizations.
        true_states_traj, rollouts = jax.vmap(
            self.eval_rollouts,
            in_axes=(self.randomized_axes, 0, None, None),
        )(self.model, states, params, plans)  # type: ignore

        # Combine the costs from different domain randomizations using the
        # specified risk strategy.
        costs = self.risk_strategy.combine_costs(rollouts.costs)
        constraint_flags = self.risk_strategy.combine_costs(rollouts.constraint_flags)
        plans = rollouts.plans[0]  # identical over randomizations

        if self.collect_data:

            states_traj = QuadAlgState(
                qpos=true_states_traj.qpos,
                qvel=true_states_traj.qvel,
                xpos=true_states_traj.xpos,
                xquat=true_states_traj.xquat,
            )

            states_traj_flat = jax.tree_util.tree_map(
                self._merge_randomizations_leaf, states_traj
            )
            # Compute summed cost per trajectory and select top fraction (lowest cost)
            R = rollouts.costs.shape[0]
            N = rollouts.costs.shape[1]
            B = int((R * N) * self.top_fraction)

            per_traj_costs = jnp.sum(
                rollouts.costs + self.safety_weight * rollouts.constraint_flags,
                axis=2,
            )  # shape: (R, N)
            costs_flat = per_traj_costs.reshape(-1)  # shape: (R*N,)

            sorted_indices = jnp.argsort(costs_flat)
            top_indices = sorted_indices[:B]

            # Flatten rollout data and index by selected top trajectories
            plans_flat = rollouts.plans.reshape(
                -1, rollouts.plans.shape[2], rollouts.plans.shape[3]
            )
            costs_flat = rollouts.costs.reshape(-1, rollouts.costs.shape[2])
            constraint_flags_flat = rollouts.constraint_flags.reshape(
                -1, rollouts.constraint_flags.shape[2]
            )

            states_selected = jax.tree_util.tree_map(
                lambda x: jnp.take(x, top_indices, axis=0), states_traj_flat
            )
            plans_selected = jnp.take(plans_flat, top_indices, axis=0)
            costs_selected = jnp.take(costs_flat, top_indices, axis=0)
            constraint_flags_selected = jnp.take(
                constraint_flags_flat, top_indices, axis=0
            )

            train_data = QuadAlgTrainData(
                states=states_selected,
                plans=plans_selected,
                costs=costs_selected,
                constraint_flags=constraint_flags_selected,
                optimized=True,
            )

        else:

            train_data = None

        return (
            rollouts.replace(  # type: ignore
                costs=costs,
                plans=plans,
                constraint_flags=constraint_flags,
            ),
            train_data,
        )

    @partial(jax.vmap, in_axes=(None, None, None, None, 0))
    def eval_rollouts(
        self,
        model: mjx.Model,
        state: mjx.Data,
        params: QuadAlgSamplingParams,
        plans: jax.Array,
    ) -> Tuple[mjx.Data, QuadAlgTrajectory]:
        """Rollout control sequences (in parallel) with optional value bootstrap."""

        states, final_state, task_costs, constraint_flags, discount = self.get_rollouts(
            model, state, params.task_params, plans
        )

        return states, QuadAlgTrajectory(
            plans=plans,
            costs=task_costs,
            constraint_flags=constraint_flags,
        )

    def get_rollouts(
        self,
        model: mjx.Model,
        state: mjx.Data,
        task_params: QuadRLTaskParams,
        plans: jax.Array,
    ) -> Tuple[mjx.Data, mjx.Data, jax.Array, jax.Array, jax.Array]:
        """Rollout control sequences (in parallel) with optional value bootstrap."""

        task_costs = jnp.zeros(self.plan_steps, dtype=jnp.float32)
        constraint_flags = jnp.zeros(self.plan_steps, dtype=jnp.float32)
        states_list = []
        
        x = state
        tp = task_params
        discount = jnp.asarray(1.0)
        
        for i in range(self.plan_steps - 1):
            a = plans[i:i+2] # shape (2, plan_dim)

            a_denormalized = self.task.denormalize(a)

            tp, x_next = self.task.dynamics(model, tp, x, a_denormalized)

            task_cost = self.dt * self.task.running_cost(tp, x, a_denormalized) * discount

            constraint_flag = self.task.constraint_check(x) * discount
            
            states_list.append(x)
            task_costs = task_costs.at[i].set(task_cost)
            constraint_flags = constraint_flags.at[i].set(constraint_flag)
            
            discount = discount * self.gamma
            x = x_next
        
        final_state = x
        states = jax.tree_util.tree_map(
            lambda x0, *xs: jnp.concatenate([x0[None, ...]] + [x[None, ...] for x in xs], axis=0),
            state,
            *states_list,
        )

        terminal_task_cost = self.dt * self.task.terminal_cost(tp, final_state) * discount
        terminal_flag = self.task.constraint_check(final_state) * discount

        task_costs = jnp.append(task_costs, terminal_task_cost)
        constraint_flags = jnp.append(constraint_flags, terminal_flag)

        return states, final_state, task_costs, constraint_flags, discount * self.gamma

    def _empty_train_data(self, N: int) -> QuadAlgTrainData:
        R = self.num_randomizations

        H = self.plan_steps
        B = R * N
        T = H + 1

        def make_leaf(x):
            return jnp.zeros((B, H) + x.shape, dtype=x.dtype)

        state = QuadAlgState(
            qpos=jnp.zeros(self.task.data.qpos.shape),  # type: ignore
            qvel=jnp.zeros(self.task.data.qvel.shape),  # type: ignore
            xpos=jnp.zeros(self.task.data.xpos.shape),  # type: ignore
            xquat=jnp.zeros(self.task.data.xquat.shape),  # type: ignore
        )
        dummy_states = jax.tree_util.tree_map(make_leaf, state)
        dummy_plans = jnp.zeros((B, H, self.task.plan_dim), dtype=jnp.float32)
        dummy_costs = jnp.zeros((B, T), dtype=jnp.float32)
        dummy_constraint_flags = jnp.zeros((B, T), dtype=jnp.float32)

        return QuadAlgTrainData(
            states=dummy_states,
            plans=dummy_plans,
            costs=dummy_costs,
            constraint_flags=dummy_constraint_flags,
            optimized=False,
        )

    # States traj is not identical over randomizations, so we need to keep all of them
    # Make a MPSTrainData object
    @staticmethod
    def _merge_randomizations_leaf(x):
        return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])

    @abstractmethod
    def prepare_optimize(self, state: mjx.Data, params: Any) -> Any:
        """Prepare for optimization by computing any necessary parameters (e.g., warm-start)."""

    @abstractmethod
    def update_with_plan(self, state: mjx.Data, params: Any) -> Any:
        """Update the policy parameters by performing a planning step."""

    @abstractmethod
    def update_without_plan(self, params: Any) -> Any:
        """Update the policy parameters without performing a planning step (e.g., apply warm-start)."""

    @abstractmethod
    def sample_plans(self, params: Any) -> Tuple[jax.Array, Any]:
        """Sample a set of control spline knots U ~ π(params).

        Args:
            params: Parameters of the policy distribution (e.g., mean, std).

        Returns:
            Control spline knots U, size (num rollouts, num_knots).
            Updated parameters (e.g., with a new PRNG key).
        """

    @abstractmethod
    def update_params(self, params: Any, rollouts: QuadAlgTrajectory) -> Any:
        """Update the policy parameters π(params) using the rollouts.

        Args:
            params: The current policy parameters.
            rollouts: The rollouts obtained from the current policy.

        Returns:
            The updated policy parameters.
        """

    @abstractmethod
    def finalize_best_plan(self, params: Any) -> Any:
        """Finalize the best plan after optimization (e.g., clip and denormalize).

        Args:
            params: The policy parameters containing the best plan.

        Returns:
            Updated parameters with finalized best plan.
        """
