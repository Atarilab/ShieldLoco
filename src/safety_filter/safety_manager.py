import threading
import time
from typing import Callable, Dict, Optional, Any

import numpy as np

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from utils.thread_manager import ThreadManager

from utils.mps import update_params

import functools

from controllers.policies.policy import TorchScriptPolicy


class SafetyManager:
    """Safety manager that runs RL policy planning in a separate thread to find safe contact locations."""

    def __init__(
        self,
        mj_model: mujoco.MjModel, # type: ignore
        rl_policy: Any,
        rl_config: Dict[str, Any],
        rate: float = 20.0,
        logger=None,
    ):
        """Initialize the safety manager.

        Args:
            rl_policy: The RL policy to use for safety checking.
            rl_config: The RL configuration to use for safety checking.
            rate: The rate at which to run the safety filter (Hz). Default is 20 Hz.
            logger: Optional logger for debugging.
        """
        self.mj_model = mj_model

        self.rl_config = rl_config
        self.torch_script_policy = rl_policy

        self.rate = rate

        self.logger = logger

        self._thread_manager = ThreadManager(logger=logger)
        self._thread_manager.add_thread("safety_filter", self._safety_loop)

        self._safety_lock = threading.RLock()
        self._latest_state: Optional[mjx.Data] = None
        self._safety_override: Optional[Any] = None

        self.updated_state = False

    def init_thread(self):
        self.setup_jax_policy(self.torch_script_policy)

        # set the task
        task_name = self.rl_config.get("mps_task_name", "QuadRLTask")
        if task_name == "QuadRLTask":
            from .quad.task import QuadRLTask, QuadRLTaskParams
            self.task = QuadRLTask(self.mj_model, self.rl_policy_jax, self.rl_config)

            self.task.data = mjx.make_data(self.task.mj_model, impl="warp", nconmax=1000, njmax=1000)

            self.task_params = QuadRLTaskParams()

            alg_name = self.rl_config.get("mps_alg_name", "MIMPS")
            if alg_name == "MIMPS":
                from .quad.mimps import QuadMIMPSPlanner
                self.planner = QuadMIMPSPlanner(
                    task=self.task,
                    num_samples=self.rl_config.get("mps_num_samples", 1000),
                    noise_level=self.rl_config.get("mps_noise_level", 0.5),
                    temperature=self.rl_config.get("mps_temperature", 0.1),
                    plan_horizon=self.rl_config.get("mps_plan_horizon", 1.0),
                    iterations=self.rl_config.get("mps_iterations", 3),
                    gamma=self.rl_config.get("mps_gamma", 0.99),
                    safety_weight=self.rl_config.get("mps_safety_weight", 1e3),
                    collect_data=self.rl_config.get("mps_collect_data", False),
                    top_fraction=self.rl_config.get("mps_top_fraction", 0.01),
                )
            else:
                raise ValueError(f"Unsupported algorithm name: {alg_name}")

        else:
            raise ValueError(f"Unsupported task name: {task_name}")

        self.jit_forward = jax.jit(mjx.forward)
        self.jit_step = jax.jit(mjx.step)
        self.jit_optimize = jax.jit(self.planner.optimize)
        self.jit_constraint_check = jax.jit(self.task.constraint_check)

        random_seed = np.random.randint(0, 2**31)
        self.ref = None
        self.plan = None
        self.policy_params = self.planner.init_params(
            task_params=self.task_params, seed=random_seed
        )

    def setup_jax_policy(self, rl_policy: TorchScriptPolicy):
        import torch
        import torchax
        from torchax.interop import jax_view

        # Get the actual TorchScript model from the wrapper
        ts_policy = rl_policy.policy
        torch_weights = ts_policy.state_dict()

        task_name = self.rl_config.get("mps_task_name", "QuadRLTask")
        if task_name == "QuadRLTask":
            from .quad.task import get_torch_policy
            t_policy = get_torch_policy()

            t_policy.load_state_dict(torch_weights)

            # Enable torchax globally
            torchax.enable_globally()
            torchax.enable_performance_mode()

            t_policy.to('jax')
            torch_weights = t_policy.state_dict()

            def run_rl_policy(weights, obs):
                # First call the model with passed in weights
                return torch.func.functional_call(t_policy, weights, args=(obs,))

            jax_weights = jax_view(torch_weights)

            self.rl_policy_jax = functools.partial(
                    jax.jit(jax_view(run_rl_policy)),
                    jax_weights
                )

        else:
            raise ValueError(f"Unsupported task name: {task_name}")

    def start(self):
        """Start the safety filter thread."""
        self._thread_manager.start()

    def stop(self, timeout: float = 2.0):
        """Stop the safety filter thread.

        Args:
            timeout: Maximum time to wait for thread to finish.
        """
        self._thread_manager.stop(timeout=timeout)

    def update_state(self, **kwargs):
        """
            Update the latest robot state for safety checking.
            Called from main thread
        """
        with self._safety_lock:
            qpos = kwargs.get("qpos")
            qvel = kwargs.get("qvel")
            time = kwargs.get("time")
            ref = kwargs.get("ref")

            # update task_params
            self.task_params = self.task.update_params(self.task_params, kwargs)

            # Update task_data
            self.task.data = update_params(
                self.task.data,
                qpos=jnp.asarray(qpos),
                qvel=jnp.asarray(qvel),
                time=jnp.asarray(time),
            )

            self.plan = ref

            self.ref = self.task.get_reference(ref=ref)
            self.updated_state = True

    def check(self) -> Any:
        """Check for safety and return override command if needed.

        This method runs the safety RL policy to find safe contact locations.

        Args:
            state: Optional state to check. If None, uses the latest state.

        Returns:
            Safety override command if unsafe, None if safe.

        Raises:
            NotImplementedError: This method needs to be implemented.
        """
        # raise NotImplementedError(
        #     "Safety check not implemented. "
        #     "This method should run the RL policy to find safe contact locations."
        # )
        return True

    def is_running(self) -> bool:
        """Check if the safety filter is running.

        Returns:
            True if running, False otherwise.
        """
        return self._thread_manager.is_running()

    def _safety_loop(self):
        """Main safety filter loop running in a separate thread."""

        with self._safety_lock:
            self.init_thread()

        while self._thread_manager.should_continue():
            start_time = time.perf_counter()

            if self.updated_state:
                with self._safety_lock:
                    # check safety
                    override = self.check()

                    self.policy_params = self.jit_optimize(
                        self.task.data,
                        self.policy_params,
                        override
                    )

                    self.plan = np.asarray(self.policy_params.best_plan)
                self.updated_state = False

            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, (1.0 / self.rate) - elapsed)
            time.sleep(sleep_time)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.stop()
        except:
            pass
