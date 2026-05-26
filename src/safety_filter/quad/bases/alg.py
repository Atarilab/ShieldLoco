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
    plans: jax.Array
    costs: jax.Array
    constraint_flags: jax.Array  # shape (num_rollouts, H+1, 2): [feet_flag, body_count]
    heuristic_values: Optional[jax.Array] = None  # shape (num_rollouts, H+1): V_h at each state

    optimized: bool = True


@dataclass
class QuadAlgTrajectory:
    """Data class for storing rollout data.

    Throughout, H denotes the number of control steps (given by the times at
    which the control spline is interpolated).

    Attributes:
        plans: Plan actions of shape (num_rollouts, H, plan_dim).
        costs: Costs of shape (num_rollouts, H+1).
        constraint_flags: Dual constraint flags, shape (num_rollouts, H+1, 2).
            [..., 0] = feet-only binary flag (for CaT penalty).
            [..., 1] = whole-body collision count (for argmin selection).
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
        mean: Offset around the reference trajectory (not absolute controls).
              Absolute plan is typically ref + mean (+ noise).
        rng: The pseudo-random number generator key.
        ref: The reference trajectory that we are sampling around.

        task_params: Optional task parameters that can be updated during plan call
        c_max: Adaptive maximum violation threshold for CaT (Constraints as Termination).
        
        best_plan: Absolute normalized plan for warm-starting (ref-independent).
        exec_plan: Materialized execution plan (4, H, 3) for controller output.
        
    """
    mean: jax.Array
    rng: jax.Array
    ref: jax.Array

    task_params: QuadRLTaskParams

    c_max: float = 1.0

    train_data: Optional[QuadAlgTrainData] = None
    
    best_plan: Optional[jax.Array] = None
    exec_plan: Optional[jax.Array] = None

    diag: jax.Array = jnp.zeros(7, dtype=jnp.float32)


class QuadSamplingBasedPlanner(ABC):
    """An abstract sampling-based MPC algorithm interface."""

    def __init__(
        self,
        task: QuadRLTask,
        num_samples: int,
        noise_level: float,
        num_randomizations: int,
        risk_strategy: RiskStrategy,
        seed: int,
        plan_horizon: int,
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
        """Initialize the MPC controller.

        Args:
            task: The task instance defining the dynamics and costs.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
            seed: The random seed for domain randomization.
            plan_horizon: The timesteps of horizon for the rollout.
            iterations: The number of optimization iterations to perform.
            alpha_mean: The smoothing factor for the mean update.
            alpha_cov: The smoothing factor for the covariance update.
        """
        self.task = task
        self.num_randomizations = max(num_randomizations, 1)

        self.num_samples = num_samples
        self.noise_level = noise_level

        # Risk strategy defaults to average cost
        if risk_strategy is None:
            risk_strategy = AverageCost()
        self.risk_strategy = risk_strategy

        # time-related variables
        # NOTE: we always interpret self.task.model as the controller's
        # internal model, not the model used for simulation.
        # self.plan_horizon = plan_horizon
        self.dt = self.task.dt * self.task.rollout_decimation
        self.plan_horizon = plan_horizon


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
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.std_decay = std_decay
        self.gamma = gamma
        self.p_max = p_max
        self.c_max_decay = c_max_decay
        self.include_terminal_cost = include_terminal_cost

    @staticmethod
    def update_c_max(params, rollouts, p_max, c_max_decay):
        """Update c_max based on observed constraint violations (CaT adaptation).
        
        Uses feet-only flags (discriminative) to avoid weight collapse.
        c_max adapts to observed batch maximum violation.
        
        Args:
            params: Current sampling params with c_max field.
            rollouts: QuadAlgTrajectory with constraint_flags (N, H+1, 2).
            p_max: Maximum termination probability.
            c_max_decay: Decay rate for c_max when no violations observed.
            
        Returns:
            Updated params with new c_max value.
        """
        feet_flags = rollouts.constraint_flags[..., 0]
        batch_max_violation = jnp.max(feet_flags)
        new_c_max = jnp.maximum(batch_max_violation, 1.0)
        return params.replace(c_max=new_c_max)

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
            else jnp.zeros((self.plan_horizon, self.task.plan_dim))
        )
        assert mean.shape == (self.plan_horizon, self.task.plan_dim), (
            f"Initial mean must have shape (plan_steps, plan_dim), got {mean.shape}"
        )

        ref = (
            ref_traj
            if ref_traj is not None
            else jnp.zeros((self.plan_horizon, self.task.plan_dim))
        )
        assert ref.shape == (self.plan_horizon, self.task.plan_dim), (
            f"Initial reference trajectory must have shape (plan_steps, plan_dim), got {ref.shape}"
        )

        return QuadAlgSamplingParams(mean=mean, rng=rng, ref=ref, task_params=task_params, c_max=1.0)

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

        params = self.update_c_max(params, rollouts, self.p_max, self.c_max_decay)

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
            B = max(1, int((R * N) * self.top_fraction))

            per_traj_costs = jnp.sum(
                rollouts.costs,
                axis=2,
            )  # shape: (R, N)
            # Note: constraint handling is now done via CaT in get_rollouts
            costs_flat = per_traj_costs.reshape(-1)  # shape: (R*N,)

            sorted_indices = jnp.argsort(costs_flat)
            top_indices = sorted_indices[:B]

            # Flatten rollout data and index by selected top trajectories
            plans_flat = rollouts.plans.reshape(
                -1, rollouts.plans.shape[2], rollouts.plans.shape[3]
            )
            costs_flat = rollouts.costs.reshape(-1, rollouts.costs.shape[2])
            constraint_flags_flat = rollouts.constraint_flags.reshape(
                -1, rollouts.constraint_flags.shape[2], rollouts.constraint_flags.shape[3]
            )

            states_selected = jax.tree_util.tree_map(
                lambda x: jnp.take(x, top_indices, axis=0), states_traj_flat
            )
            plans_selected = jnp.take(plans_flat, top_indices, axis=0)
            costs_selected = jnp.take(costs_flat, top_indices, axis=0)
            constraint_flags_selected = jnp.take(
                constraint_flags_flat, top_indices, axis=0
            )

            # Compute heuristic values for all selected states so that offline
            # training can ablate between direct and residual modes without
            # recomputing SDFs.
            from safety_filter.quad.safety_value import _vectorized_heuristic

            B_sel = states_selected.xpos.shape[0]
            T_sel = states_selected.xpos.shape[1]
            xpos_flat = states_selected.xpos.reshape(
                B_sel * T_sel, *states_selected.xpos.shape[2:]
            )
            xquat_flat = states_selected.xquat.reshape(
                B_sel * T_sel, *states_selected.xquat.shape[2:]
            )
            qvel_flat = states_selected.qvel.reshape(
                B_sel * T_sel, *states_selected.qvel.shape[2:]
            )
            heuristic_values = _vectorized_heuristic(
                self.task, xpos_flat, xquat_flat, qvel_flat
            )
            heuristic_values = heuristic_values.reshape(B_sel, T_sel)

            train_data = QuadAlgTrainData(
                states=states_selected,
                plans=plans_selected,
                costs=costs_selected,
                constraint_flags=constraint_flags_selected,
                heuristic_values=heuristic_values,
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
            model, state, params, plans
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
        params: QuadAlgSamplingParams,
        plans: jax.Array,
    ) -> Tuple[mjx.Data, mjx.Data, jax.Array, jax.Array, jax.Array]:
        """Rollout control sequences (in parallel) with optional value bootstrap.
        
        Uses CaT (Constraints as Termination): constraint violations modulate the 
        discount factor to penalize trajectories that violate constraints early.
        c_max is read from params.c_max.
        """

        task_costs = jnp.zeros(self.plan_horizon, dtype=jnp.float32)
        constraint_flags = jnp.zeros((self.plan_horizon, 2), dtype=jnp.float32)

        x = state
        tp = params.task_params
        discount = jnp.asarray(1.0)

        # Pad plans by repeating the last row so every horizon step gets a unique
        # 2-step window. Without padding, steps H-2 and H-1 both clip to the same
        # window [H-2:H], meaning the optimizer has no control over the final step.
        plans_padded = jnp.concatenate([plans, plans[-1:]], axis=0)      # (H+1, plan_dim)
        ref_padded = jnp.concatenate([params.ref, params.ref[-1:]], axis=0)  # (H+1, plan_dim)

        # Scan over horizon steps instead of unrolling a Python for loop.
        # Unrolling plan_horizon=8 steps × rollout_decimation=10 sub-steps = 80
        # sequential mjx.step kernels in the XLA graph; scan emits a single
        # while-loop, reducing dispatch overhead and peak memory.
        #
        # IMPORTANT: only emit QuadAlgState (4 small arrays) as the scan's stacked
        # output, NOT the full mjx.Data.  Full mjx.Data includes efc_J of shape
        # (njmax, nv) = (204800, 18).  Stacked over plan_horizon=8 steps and vmapped
        # over 400 samples that would be 8 × 400 × 204800 × 18 × 4 bytes ≈ 47 GB.
        # QuadAlgState carries only qpos/qvel/xpos/xquat — the only fields ever read
        # downstream (collect_data path in rollout_with_randomizations).
        def _trim(s: mjx.Data) -> QuadAlgState:
            return QuadAlgState(qpos=s.qpos, qvel=s.qvel, xpos=s.xpos, xquat=s.xquat)

        def _plan_step(carry, i):
            tp, x, task_costs, constraint_flags, discount = carry
            a     = jax.lax.dynamic_slice(plans_padded, (i, 0), (2, self.task.plan_dim))
            a_ref = jax.lax.dynamic_slice(ref_padded,   (i, 0), (2, self.task.plan_dim))
            tp, x_next, task_cost, constraint_flag = self.task.forward(model, tp, x, a, a_ref, step_idx=i)
            
            # Dual constraint: constraint_flag is (2,) = [feet_flag, body_count]
            # CaT penalty uses feet-only flag (discriminative, avoids weight collapse)
            feet_flag = constraint_flag[0]
            
            c_max = params.c_max
            normalized = jnp.clip(feet_flag / c_max, 0.0, 1.0)
            termination_prob = self.p_max * normalized
            
            # Store both flags for downstream argmin selection
            constraint_flags = constraint_flags.at[i].set(constraint_flag)
            
            # CaT penalty: add termination penalty directly to this step's cost
            # This penalizes trajectories that violate constraints EARLY
            penalty = self.safety_weight * termination_prob
            
            # Apply normal discount for future costs + add penalty to this step
            effective_cost = (task_cost + penalty) * discount * self.dt 
            
            task_costs = task_costs.at[i].set(effective_cost)
            
            # Update discount for next step
            new_discount = discount * self.gamma
            
            return (tp, x_next, task_costs, constraint_flags, new_discount), _trim(x_next)

        (tp, final_state, task_costs, constraint_flags, discount), states_stacked = jax.lax.scan(
            _plan_step,
            (tp, x, task_costs, constraint_flags, discount),
            jnp.arange(self.plan_horizon),
        )

        # Prepend initial state to get shape (plan_horizon+1, ...) matching original.
        states = jax.tree_util.tree_map(
            lambda x0, xs: jnp.concatenate([x0[None, ...], xs], axis=0),
            _trim(state),
            states_stacked,
        )

        terminal_flags = self.task.constraint_check(final_state)

        # CaT penalty for terminal step uses feet-only flag
        c_max = params.c_max
        terminal_feet = terminal_flags[0]
        terminal_normalized = jnp.clip(terminal_feet / c_max, 0.0, 1.0)
        terminal_term_prob = self.p_max * terminal_normalized

        terminal_penalty = self.safety_weight * terminal_term_prob

        if self.include_terminal_cost:
            terminal_cost = self.dt * self.task.terminal_cost(tp, final_state)
            terminal_cost_with_penalty = (terminal_cost + terminal_penalty) * discount
        else:
            # Disable terminal cost in planner optimization.
            # The heuristic is still stored separately in train_data for training.
            terminal_cost_with_penalty = terminal_penalty * discount

        task_costs = jnp.append(task_costs, terminal_cost_with_penalty)
        constraint_flags = jnp.concatenate([constraint_flags, terminal_flags[None, :]], axis=0)

        return states, final_state, task_costs, constraint_flags, discount * self.gamma

    def _empty_train_data(self, N: int) -> QuadAlgTrainData:
        R = self.num_randomizations

        H = self.plan_horizon
        B = max(1, int((R * N) * self.top_fraction))

        def make_leaf(x):
            # H+1 to match the rollout output: initial state + H stepped states
            return jnp.zeros((B, H + 1) + x.shape, dtype=x.dtype)

        state = QuadAlgState(
            qpos=jnp.zeros(self.task.data.qpos.shape),  # type: ignore
            qvel=jnp.zeros(self.task.data.qvel.shape),  # type: ignore
            xpos=jnp.zeros(self.task.data.xpos.shape),  # type: ignore
            xquat=jnp.zeros(self.task.data.xquat.shape),  # type: ignore
        )
        dummy_states = jax.tree_util.tree_map(make_leaf, state)
        # Match rollout outputs exactly:
        # - plans: (B, H, plan_dim)
        # - costs: (B, H + 1) including terminal value
        # - constraint_flags: (B, H + 1, 2) dual flags [feet_flag, body_count]
        dummy_plans = jnp.zeros((B, H, self.task.plan_dim), dtype=jnp.float32)
        dummy_costs = jnp.zeros((B, H + 1), dtype=jnp.float32)
        dummy_constraint_flags = jnp.zeros((B, H + 1, 2), dtype=jnp.float32)
        dummy_heuristics = jnp.zeros((B, H + 1), dtype=jnp.float32)

        return QuadAlgTrainData(
            states=dummy_states,
            plans=dummy_plans,
            costs=dummy_costs,
            constraint_flags=dummy_constraint_flags,
            heuristic_values=dummy_heuristics,
            optimized=False,
        )

    def finalize_best_plan(self, params: Any) -> Any:
        """Finalize the best plan by clip and denormalizing if necessary."""

        """Materialize the controller-facing execution plan without mutating warm-start state."""
        abs_plan = params.mean + params.ref

        abs_plan = self.task.project_plan_to_safe(abs_plan)

        exec_plan = self.task.get_exec_plan(abs_plan)

        return params.replace(exec_plan=exec_plan)

    def quad_noise(self, rng, N=None):
        rng, body_const_rng, body_slope_rng, foot_rng = jax.random.split(rng, 4)

        N = N if N is not None else self.num_samples
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

        return noise

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

