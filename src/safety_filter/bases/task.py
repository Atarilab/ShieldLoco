from abc import ABC, abstractmethod
from typing import Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from flax.struct import dataclass


@dataclass
class TaskParams:
    """A dataclass to hold any additional parameters needed for the task."""


class Task(ABC):
    """An abstract task interface, defining the dynamics and cost functions.

    The task is a discrete-time optimal control problem of the form

        minᵤ ϕ(x_{T+1}) + ∑ₜ ℓ(xₜ, uₜ)
        s.t. xₜ₊₁ = f(xₜ, uₜ)

    where the dynamics f(xₜ, uₜ) are defined by a MuJoCo model, and the costs
    ℓ(xₜ, uₜ) and ϕ(x_{T+1}) are defined by the task instance itself.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,  # type: ignore
        trace_sites: Sequence[str] | None = None,
    ) -> None:
        """Set the model and simulation parameters.

        Args:
            mj_model: The MuJoCo model to use for simulation.
            trace_sites: A list of site names to visualize with traces.

        Note: many other simulator parameters, e.g., simulator time step,
              Newton iterations, etc., are set in the model itself.
        """
        assert isinstance(mj_model, mujoco.MjModel)  # type: ignore
        self.mj_model = mj_model
        self.model = mjx.put_model(mj_model, impl="warp")
        self.data: mjx.Data | None = None

        # Set joint position limits
        self.j_min = jnp.where(
            mj_model.actuator_ctrllimited, mj_model.actuator_ctrlrange[:, 0], -jnp.inf
        )
        self.j_max = jnp.where(
            mj_model.actuator_ctrllimited, mj_model.actuator_ctrlrange[:, 1], jnp.inf
        )
        # print("J min: ", self.j_min)
        # print("J max: ", self.j_max)

        # Simulation timestep
        self.dt = mj_model.opt.timestep

        # Get site IDs for points we want to trace
        trace_sites = trace_sites or []
        self.trace_site_ids = jnp.array(
            [mj_model.site(name).id for name in trace_sites]
        )

        self.u_min = -1
        self.u_max = 1

    def get_reference_plan(self, **kwargs):
        """Get the reference plan (in optimization) for the current task."""
        return kwargs.get("ref", None)

    def get_exec_plan(self, plan: jax.Array) -> jax.Array:
        """Get the executable plan by reshaping and denormalizing the optimized plan."""
        return plan

    @abstractmethod
    def normalize(self, plan: jax.Array) -> jax.Array:
        """Normalize the control plan if necessary."""
        
    @abstractmethod
    def denormalize(self, plan: jax.Array) -> jax.Array:
        """Denormalize the control plan if necessary."""

    @abstractmethod
    def dynamics(
        self,
        model: mjx.Model,
        task_params: TaskParams,
        state: mjx.Data,
        control: jax.Array,
    ) -> Tuple[TaskParams, mjx.Data]:
        """The dynamics function f(xₜ, uₜ) that maps the current state and control to
        the next state.

        Args:
            model: The MuJoCo model to use for simulation.
            task_params: The task parameters.
            state: The current state xₜ.
            control: The control action uₜ.

        Returns:
            A tuple of (new_task_params, next_state), where new_task_params are
            any updated task parameters after taking the action, and next_state
            is the resulting state xₜ₊₁ after applying the dynamics.
        """

    @abstractmethod
    def running_cost(
        self, task_params: TaskParams, state: mjx.Data, control: jax.Array
    ) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ).

        Args:
            task_params: The task parameters.
            state: The current state xₜ.
            control: The control action uₜ.

        Returns:
            The scalar running cost ℓ(xₜ, uₜ)
        """
        pass

    @abstractmethod
    def terminal_cost(self, task_params: TaskParams, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T).

        Args:
            task_params: The task parameters.
            state: The final state x_T.

        Returns:
            The scalar terminal cost ϕ(x_T).
        """
        pass

    @abstractmethod
    def constraint_check(self, state: mjx.Data) -> jax.Array:
        """Check for constraint violations in the state.

        This method should return a scalar value representing the degree of
        constraint violation. A value of 0 means no violation, while positive
        values indicate the amount of violation.

        Args:
            state: The current state xₜ.

        Returns:
            A scalar representing the degree of constraint violation.
        """
        pass

    def get_trace_sites(self, state: mjx.Data) -> jax.Array:
        """Get the positions of the trace sites at the current time step.

        Args:
            state: The current state xₜ.

        Returns:
            The positions of the trace sites at the current time step.
        """
        if len(self.trace_site_ids) == 0:
            return jnp.zeros((0, 3))

        return state.site_xpos[self.trace_site_ids]

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Generate randomized model parameters for domain randomization.

        Returns a dictionary of randomized model parameters, that can be used
        with `mjx.Model.tree_replace` to create a new randomized model.

        For example, we might set the `model.geom_friction` values by returning
        `{"geom_friction": new_frictions, ...}`.

        The default behavior is to return an empty dictionary, which means no
        randomization is applied.

        Args:
            rng: A random number generator key.

        Returns:
            A dictionary of randomized model parameters.
        """
        return {}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Generate randomized data elements for domain randomization.

        This is the place where we could randomize the initial state and other
        `data` elements. Like `domain_randomize_model`, this method should
        return a dictionary that can be used with `mjx.Data.tree_replace`.

        Args:
            data: The base data instance holding the current state.
            rng: A random number generator key.

        Returns:
            A dictionary of randomized data elements.
        """
        return {}
