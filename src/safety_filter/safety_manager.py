import multiprocessing as mp
import time
from typing import Callable, Dict, Optional, Any
import sys

import numpy as np

import mujoco

from utils.mps import update_params

import functools
import traceback

from controllers.policies.policy import TorchScriptPolicy


def guard_errors(context: str, *, stop_on_error: bool = True):
    """Decorator to handle repetitive try/except + stop + logging."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            try:
                return fn(self, *args, **kwargs)
            except Exception as e:
                tb = traceback.format_exc()
                self._log_error(f"Error {context}: {e}")
                self._log_error(tb)
                if stop_on_error:
                    self.stop()
        return wrapper
    return decorator


class SafetyManager:
    """Safety manager that runs RL policy planning in a separate thread to find safe contact locations."""

    def __init__(
        self,
        mj_model: mujoco.MjModel, # type: ignore
        rl_policy: Any,
        rl_config: Dict[str, Any],
        rate: float = 50.0,
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
        self._pending_log_messages = []

        self._ctx = mp.get_context("fork")
        self._manager: Optional[Any] = None
        self._shared_state: Optional[Any] = None
        self._shared_result: Optional[Any] = None
        self._state_event: Optional[Any] = None
        self._stop_event: Optional[Any] = None
        self._process: Optional[Any] = None
        self._last_result_seq = -1
        self._plan_count = 0
        self._last_health_log_time = 0.0
        self._health_log_interval = 5.0
        self._first_plan_logged = False
        self._worker_error_logged = False
        self._latest_plan_info = None

        self.thread_initialized = False
        self.updated_state = False
        self.updated_plan = False
        self.override = False
        self.exec_plan = None
        self._last_train_data = None

    def _log_error(self, msg: str):
        if self.logger:
            self.logger.error(msg)
        else:
            self._pending_log_messages.append(("error", msg))
            print(msg, file=sys.stderr, flush=True)

    def _log_info(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            self._pending_log_messages.append(("info", msg))

    def set_logger(self, logger):
        self.logger = logger
        if self.logger and self._pending_log_messages:
            for level, message in self._pending_log_messages:
                if level == "error":
                    self.logger.error(message)
                else:
                    self.logger.info(message)
            self._pending_log_messages.clear()

    @guard_errors("initializing thread")
    def init_thread(self):
        import jax
        from mujoco import mjx

        self.setup_jax_policy(self.torch_script_policy)

        # set the task
        task_name = self.rl_config.get("mps_task_name", "QuadRLTask")
        if task_name == "QuadRLTask":
            from .quad.task import QuadRLTask, QuadRLTaskParams
            self.task = QuadRLTask(self.mj_model, self.rl_policy_jax, self.rl_config)

            self.task.data = mjx.make_data(self.task.mj_model, impl="warp", nconmax=50000, njmax=2000)

            self.task_params = self.task.make_task_params()

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
        self.exec_plan = None
        self.policy_params = self.planner.init_params(
            task_params=self.task_params, seed=random_seed
        )

    @guard_errors("setting up JAX policy")
    def setup_jax_policy(self, rl_policy: TorchScriptPolicy):
        import jax
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

    @guard_errors("starting safety filter thread")
    def start(self):
        """Start the safety filter worker process."""
        if self._process is not None and self._process.is_alive():
            return

        self._manager = self._ctx.Manager()
        self._shared_state = self._manager.dict({"payload": None, "seq": 0})
        self._shared_result = self._manager.dict(
            {
                "plan": None,
                "override": False,
                "seq": -1,
                "error": None,
                "reference_generation_id": None,
                "reference_step_seq": None,
            }
        )
        self._state_event = self._ctx.Event()
        self._stop_event = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_safety_worker_main,
            args=(
                self.mj_model,
                self.torch_script_policy,
                self.rl_config,
                self.rate,
                self._shared_state,
                self._shared_result,
                self._state_event,
                self._stop_event,
            ),
            daemon=True,
        )
        assert self._process is not None
        self._process.start()
        self.thread_initialized = True
        self._plan_count = 0
        self._last_health_log_time = time.time()
        self._first_plan_logged = False
        self._worker_error_logged = False
        self._log_info(
            f"Safety worker started (pid={self._process.pid}, rate={self.rate:.1f}Hz)"
        )

    @guard_errors("stopping safety filter thread")
    def stop(self, timeout: float = 2.0):
        """Stop the safety filter thread.

        Args:
            timeout: Maximum time to wait for thread to finish.
        """
        if self._stop_event is not None:
            self._stop_event.set()

        # Only join if process was actually started
        if self._process is not None and hasattr(self._process, '_popen'):
            try:
                self._process.join(timeout=timeout)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=1.0)
            except Exception as e:
                print(f"[SafetyManager] Error joining process: {e}", flush=True)

        self._process = None
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception as e:
                print(f"[SafetyManager] Error shutting down manager: {e}", flush=True)
        self._manager = None
        self._stop_event = None
        self._state_event = None
        self._shared_state = None
        self._shared_result = None
        self._last_result_seq = -1
        self._plan_count = 0
        self._first_plan_logged = False
        self._worker_error_logged = False
        self._latest_plan_info = None
        self.thread_initialized = False
        self._log_info("Safety worker stopped")

    @guard_errors("updating state")
    def update_state(self, **kwargs):
        """
            Update the latest robot state for safety checking.
            Called from main thread
        """
        self.updated_state = True
        self.updated_plan = False

        if self._shared_state is None:
            return

        payload = {
            "qpos": kwargs.get("qpos"),
            "qvel": kwargs.get("qvel"),
            "time": kwargs.get("time"),
            "xpos": kwargs.get("xpos"),
            "xquat": kwargs.get("xquat"),

            "ref": kwargs.get("ref"),
            "reference_generation_id": kwargs.get("reference_generation_id"),
            "reference_step_seq": kwargs.get("reference_step_seq"),
            "goal_idx": kwargs.get("goal_idx"),
            "goal_completion_counter": kwargs.get("goal_completion_counter"),
            "contact_plan": kwargs.get("contact_plan"),
            "current_gait": kwargs.get("current_gait"),
            "in_transition": kwargs.get("in_transition"),
            "dynamic_obstacle_positions": kwargs.get("dynamic_obstacle_positions"),
            "dynamic_obstacle_active": kwargs.get("dynamic_obstacle_active"),
            "dynamic_obstacle_radii": kwargs.get("dynamic_obstacle_radii"),
        }
        current_seq = int(self._shared_state.get("seq", 0)) + 1
        self._shared_state["payload"] = payload
        self._shared_state["seq"] = current_seq
        if self._state_event is not None:
            self._state_event.set()

    @guard_errors("checking safety")
    def check(self):
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

        # TODO: add a real check
        self.override = True

    @guard_errors("checking if safety filter is running")
    def is_running(self) -> bool:
        """Check if the safety filter is running.

        Returns:
            True if running, False otherwise.
        """
        if self._process is None or not self.thread_initialized:
            return False

        alive = self._process.is_alive()
        if not alive and not self._worker_error_logged:
            self._poll_results()
            worker_error = None
            if self._shared_result is not None:
                worker_error = self._shared_result.get("error")
            if worker_error:
                self._log_error(f"Safety worker crashed:\n{worker_error}")
            else:
                self._log_error(
                    f"Safety worker exited unexpectedly with exitcode={self._process.exitcode}"
                )
            self._worker_error_logged = True

        return alive

    @guard_errors("polling safety result", stop_on_error=False)
    def _poll_results(self):
        if self._shared_result is None:
            return

        worker_error = self._shared_result.get("error")
        if worker_error and not self._worker_error_logged:
            self._log_error(f"Safety worker reported error:\n{worker_error}")
            self._worker_error_logged = True

        result_seq = int(self._shared_result.get("seq", -1))
        if result_seq <= self._last_result_seq:
            return

        self._last_result_seq = result_seq
        self.exec_plan = self._shared_result.get("plan")
        self.override = bool(self._shared_result.get("override", False))
        self._latest_plan_info = {
            "plan": self.exec_plan,
            "override": self.override,
            "result_seq": result_seq,
            "reference_generation_id": self._shared_result.get("reference_generation_id"),
            "reference_step_seq": self._shared_result.get("reference_step_seq"),
        }
        
        # Fetch train_data COM trajectory
        train_data_dict = self._shared_result.get("train_data")
        if train_data_dict is not None:
            self._last_train_data = train_data_dict
        else:
            self._last_train_data = None
            
        if self.exec_plan is not None:
            self.updated_plan = self.exec_plan is not None
            self.updated_state = False
            self._plan_count += 1

            if not self._first_plan_logged:
                self._first_plan_logged = True
                self._log_info(f"Safety worker produced first plan (seq={result_seq})")

            now = time.time()
            if now - self._last_health_log_time >= self._health_log_interval:
                self._last_health_log_time = now
                self._log_info(
                    f"Safety heartbeat: running={self.is_running()}, plans={self._plan_count}, latest_seq={result_seq}"
                )

    @guard_errors("getting latest safety plan", stop_on_error=False)
    def get_latest_plan(self):
        plan_info = self.get_latest_plan_info()
        if plan_info is None:
            return None
        return plan_info.get("plan")

    @guard_errors("getting latest safety plan metadata", stop_on_error=False)
    def get_latest_plan_info(self):
        self._poll_results()
        if self.updated_plan and self.override:
            self.updated_plan = False
            return self._latest_plan_info
        return None

    @guard_errors("getting train data", stop_on_error=False)
    def get_train_data(self):
        self._poll_results()
        return self._last_train_data

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


def _setup_jax_policy(rl_policy: TorchScriptPolicy, rl_config: Dict[str, Any]):
    import jax
    import torch
    import torchax
    from torchax.interop import jax_view

    ts_policy = rl_policy.policy
    torch_weights = ts_policy.state_dict()

    task_name = rl_config.get("mps_task_name", "QuadRLTask")
    if task_name != "QuadRLTask":
        raise ValueError(f"Unsupported task name: {task_name}")

    from .quad.task import get_torch_policy

    t_policy = get_torch_policy()
    t_policy.load_state_dict(torch_weights)

    torchax.enable_globally()
    torchax.enable_performance_mode()

    t_policy.to("jax")
    torch_weights = t_policy.state_dict()

    def run_rl_policy(weights, obs):
        return torch.func.functional_call(t_policy, weights, args=(obs,))

    jax_weights = jax_view(torch_weights)
    return functools.partial(jax_view(run_rl_policy), jax_weights)


def _init_worker_objects(mj_model, rl_policy, rl_config):
    import jax
    from mujoco import mjx

    rl_policy_jax = _setup_jax_policy(rl_policy, rl_config)

    task_name = rl_config.get("mps_task_name", "QuadRLTask")
    if task_name != "QuadRLTask":
        raise ValueError(f"Unsupported task name: {task_name}")

    from .quad.task import QuadRLTask
    from .quad.mimps import QuadMIMPSPlanner

    task = QuadRLTask(mj_model, rl_policy_jax, rl_config)
    task.data = mjx.make_data(task.mj_model, impl="warp", nconmax=20000, njmax=1000)
    task_params = task.make_task_params()

    planner = QuadMIMPSPlanner(
        task=task,
        num_samples=rl_config.get("mps_num_samples", 1000),
        noise_level=rl_config.get("mps_noise_level", 0.5),
        temperature=rl_config.get("mps_temperature", 0.1),
        plan_horizon=rl_config.get("mps_plan_horizon", 1.0),
        iterations=rl_config.get("mps_iterations", 3),
        gamma=rl_config.get("mps_gamma", 0.99),
        safety_weight=rl_config.get("mps_safety_weight", 1e3),
        collect_data=rl_config.get("mps_collect_data", False),
        top_fraction=rl_config.get("mps_top_fraction", 0.01),
    )

    random_seed = np.random.randint(0, 2**31)
    policy_params = planner.init_params(task_params=task_params, seed=random_seed)
    jit_optimize = jax.jit(planner.optimize)
    return task, task_params, policy_params, jit_optimize


# TODO: is it possible to decouple update and opt? might give a bump in speed
def _safety_worker_main(
    mj_model,
    rl_policy,
    rl_config,
    rate: float,
    shared_state: Any,
    shared_result: Any,
    state_event: Any,
    stop_event: Any,
):
    try:
        import jax.numpy as jnp
        from mujoco import mjx
        import jax

        task, task_params, policy_params, jit_optimize = _init_worker_objects(mj_model, rl_policy, rl_config)
        jit_mjx_forward = jax.jit(mjx.forward)

        period = 1.0 / max(rate, 1e-3)
        last_processed_seq = -1
        while not stop_event.is_set():
            loop_start = time.perf_counter()
            state_event.wait(timeout=period)
            state_event.clear()

            latest_seq = int(shared_state.get("seq", -1))
            if latest_seq <= last_processed_seq:
                continue

            latest_state = shared_state.get("payload")
            if latest_state is None:
                continue

            last_processed_seq = latest_seq
            task.data = update_params(
                task.data,
                qpos=jnp.asarray(latest_state.get("qpos")),
                qvel=jnp.asarray(latest_state.get("qvel")),
                xpos=jnp.asarray(latest_state.get("xpos")),
                xquat=jnp.asarray(latest_state.get("xquat")),
                time=jnp.asarray(latest_state.get("time")),
            )
            # Refresh derived kinematics/contact fields after injecting raw state.
            # task.data = jit_mjx_forward(task.model, task.data)
            
            task_params = task.update_params(task_params, latest_state)
            ref_plan_normalized = task.get_reference_plan(
                ref=latest_state.get("ref"),
                goal_idx=latest_state.get("goal_idx", 0),
            )
            policy_params = update_params(
                policy_params,
                task_params=task_params,
                ref=ref_plan_normalized,
            )

            override = True
            policy_params = jit_optimize(task.data, policy_params, override)
            exec_plan = np.asarray(policy_params.exec_plan)

            # Share train_data COM trajectory if available
            if hasattr(policy_params, 'train_data') and policy_params.train_data is not None:
                qpos_traj = np.asarray(policy_params.train_data.states.qpos)
                costs_traj = np.asarray(policy_params.train_data.costs)
                constraint_traj = np.asarray(policy_params.train_data.constraint_flags)

                if qpos_traj.ndim == 3 and qpos_traj.shape[0] > 0:
                    if costs_traj.ndim >= 2 and constraint_traj.ndim >= 2:
                        combined = costs_traj + float(rl_config.get("mps_safety_weight", 1e3)) * constraint_traj
                        per_traj_obj = np.sum(combined, axis=1)
                        best_idx = int(np.argmin(per_traj_obj))
                    else:
                        best_idx = 0

                    best_qpos = qpos_traj[best_idx]
                    best_costs = costs_traj[best_idx] if costs_traj.ndim >= 2 else costs_traj
                    best_constraints = (
                        constraint_traj[best_idx] if constraint_traj.ndim >= 2 else constraint_traj
                    )
                elif qpos_traj.ndim == 2:
                    best_qpos = qpos_traj
                    best_costs = costs_traj if costs_traj.ndim == 1 else None
                    best_constraints = (
                        constraint_traj if constraint_traj.ndim == 1 else None
                    )
                else:
                    best_qpos = None
                    best_costs = None
                    best_constraints = None

                if best_qpos is not None:
                    com_traj = best_qpos[:, :3][None, ...]
                    shared_result["train_data"] = {
                        "com": com_traj,
                        "qpos": best_qpos,
                        "costs": best_costs,
                        "constraint_flags": best_constraints,
                    }
                else:
                    shared_result["train_data"] = None
            else:
                shared_result["train_data"] = None

            shared_result["plan"] = exec_plan
            shared_result["override"] = override
            shared_result["seq"] = latest_seq
            shared_result["reference_generation_id"] = latest_state.get("reference_generation_id")
            shared_result["reference_step_seq"] = latest_state.get("reference_step_seq")

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.0, period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        tb = traceback.format_exc()
        shared_result["error"] = f"{e}\n{tb}"
