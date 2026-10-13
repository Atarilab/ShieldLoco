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


# ---------------------------------------------------------------------------
# Monkey-patch: fix multiprocessing "spawn" failures in fd-heavy environments
# (e.g. ROS2, where DDS middleware opens many sockets / pipes).
#
# PROBLEM
#   Python ≤ 3.10 can hit
#       ValueError: bad value(s) in fds_to_keep
#   when multiprocessing.Manager() (or any spawn-context action) tries to
#   launch its internal _resource_tracker helper process.  The tracker
#   collects every open file descriptor via /proc/self/fd and feeds them to
#   _posixsubprocess.fork_exec(), which rejects descriptors that are
#   negative, ≥ os.sysconf("SC_OPEN_MAX"), or already closed by the DDS
#   layer between the scan and the actual fork+exec.
#
# WHY NOT "fork"?
#   The safety worker process initialises JAX + CUDA from scratch.  Forking
#   a CUDA-enabled parent would duplicate the GPU context, which CUDA
#   forbids.  "spawn" is therefore required — but the spawn codepath relies
#   on spawnv_passfds, which is where the crash happens.
#
# FIX
#   Wrap spawnv_passfds so that each file descriptor is validated with a
#   cheap os.fstat() call *before* reaching fork_exec.  Descriptors that no
#   longer exist (closed or never valid) are silently dropped.  If none
#   survive the check we fall back to plain subprocess.Popen, which works
#   without passing any fds.
# ---------------------------------------------------------------------------
def _apply_spawnv_passfds_patch():
    """
    Inject a safe wrapper around multiprocessing.util.spawnv_passfds so that
    the resource_tracker helper launched by Manager() / Process() survives in
    environments where the parent holds stale or out-of-range file descriptors
    (common in ROS2 deployments).

    This function is idempotent — calling it multiple times is harmless.
    """
    import os as _os
    import subprocess as _subprocess
    from multiprocessing import util as _mp_util

    # Guard against repeated patching across multiple SafetyManager instances.
    if getattr(_mp_util.spawnv_passfds, "_doom_safety_patched", False):
        return

    # Keep a reference to the original C-backed function (native code in
    # _posixsubprocess.so).  We need it for the happy path where all fds are
    # valid.
    _original_spawnv_passfds = _mp_util.spawnv_passfds

    def _safe_spawnv_passfds(path, args, passfds):
        """
        Replacement for multiprocessing.util.spawnv_passfds.

        Iterates over every file descriptor that the resource tracker wants
        to forward to the child and sanity-checks each one with os.fstat().
        This call fails with OSError (EBADF) if the descriptor was closed
        between the /proc/self/fd scan and now — exactly the condition that
        triggers "bad value(s) in fds_to_keep" inside _posixsubprocess.

        Any fd that fails the check is simply excluded from the list; the
        resource tracker's internal pipe fd is always valid, so the list
        should never become empty under normal circumstances.
        """
        # First pass: keep only descriptors that the kernel still recognises
        # as open.  os.fstat(fd) is cheap (one syscall) and is the exact
        # check that fork_exec performs internally.
        clean: list[int] = []
        for fd in passfds:
            try:
                _os.fstat(fd)          # raises OSError (EBADF) if closed
            except OSError:
                continue                # fd was closed → skip it
            if fd < 0:
                continue                # paranoia: negative fd
            clean.append(fd)

        if not clean:
            # Degenerate case: every descriptor the tracker collected is now
            # closed.  Fall back to spawning without any fd passing.  Under
            # Python 3.10 this signals the tracker to reopen the pipe later.
            proc = _subprocess.Popen(
                [path] + args,
                close_fds=True,
            )
            return proc.pid

        # Happy path: forward the sanitised list to the C extension.
        return _original_spawnv_passfds(path, args, clean)

    # Mark the function so repeated calls are no-ops.
    _safe_spawnv_passfds._doom_safety_patched = True

    # Replace the module-level function.  All subsequent calls inside
    # multiprocessing (Manager, Process, resource_tracker, …) will
    # transparently go through our safe wrapper.
    _mp_util.spawnv_passfds = _safe_spawnv_passfds


# Apply the patch once at import time, before any multiprocessing objects are
# created.  The patch is safe to call unconditionally — it is a no-op after
# the first invocation.
_apply_spawnv_passfds_patch()


def _dump_train_data(train_data, output_dir: str, seq: int, seed: int = 0):
    """Save a QuadAlgTrainData object to disk as a pickle (numpy arrays)."""
    import pickle
    from pathlib import Path

    out = Path(output_dir) / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"rollout_{seq:06d}.pkl"

    # Convert JAX arrays to numpy for pickle compatibility and smaller files
    import jax

    def to_np(x):
        if hasattr(x, "__array__"):
            return np.array(x)
        return x

    np_data = jax.tree_util.tree_map(to_np, train_data)
    with open(path, "wb") as f:
        pickle.dump(np_data, f)


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


PLANNER_REGISTRY = {
    "MIMPS": "QuadMIMPSPlanner",
    "CEMPS": "QuadCEMPSPlanner",
    "DIAL_MPC": "QuadDPMSPlanner",
    "CONSENSUS": "QuadCMPSPlanner",
    "REMIMPS": "QuadREMIMPSPlanner",
    "CBF": "QuadCBFPlanner",
    "HJ": "QuadHJPlanner",
    "LIPM": "QuadLIPMPlanner",
}


def create_planner(alg_name, task, config):
    """Factory function to create a planner based on algorithm name.
    
    Args:
        alg_name: Algorithm name (MIMPS, CEMPS, DMPS, CMPS)
        task: QuadRLTask instance
        config: Configuration dict with common + optimizer-specific params
        
    Returns:
        Planner instance of the appropriate type
    """
    from .quad.mimps import QuadMIMPSPlanner
    from .quad.cemps import QuadCEMPSPlanner
    from .quad.dmps import QuadDPMSPlanner
    from .quad.cmps import QuadCMPSPlanner
    from .quad.remimps import QuadREMIMPSPlanner
    from .quad.cbf_planner import QuadCBFPlanner
    from .quad.hj_planner import QuadHJPlanner
    from .quad.lipm_planner import QuadLIPMPlanner

    registry = {
        "MIMPS": QuadMIMPSPlanner,
        "CEMPS": QuadCEMPSPlanner,
        "DMPS": QuadDPMSPlanner,
        "CMPS": QuadCMPSPlanner,
        "REMIMPS": QuadREMIMPSPlanner,
        "CBF": QuadCBFPlanner,
        "HJ": QuadHJPlanner,
        "LIPM": QuadLIPMPlanner,
    }
    
    if alg_name not in registry:
        raise ValueError(f"Unsupported algorithm: {alg_name}. Choose from {list(registry.keys())}")
    
    planner_cls = registry[alg_name]
    
    common_params = {
        "num_samples": config.get("mps_num_samples", 800),
        "plan_horizon": config.get("horizon_length", config.get("mps_plan_horizon", 6)),
        "iterations": config.get("mps_iterations", 2),
        "gamma": config.get("mps_gamma", 0.99),
        "safety_weight": config.get("mps_safety_weight", 1.0),
        "collect_data": config.get("mps_collect_data", False),
        "top_fraction": config.get("mps_top_fraction", 0.01),
        "momentum": config.get("mps_momentum", 0.5),
        "use_momentum": config.get("mps_use_momentum", True),
        "std_decay": config.get("mps_std_decay", 0.9),
        "include_terminal_cost": config.get(
            "safety_value_mode", "heuristic"
        ) != "off",
    }
    
    optimizer_params = config.get(alg_name, {})

    # Bool flags may arrive as strings ('true'/'false') from sweep overrides or
    # YAML. Coerce so that a truthy string "false" can never silently enable a
    # filter ("nominal" planns). Non-bool/int keys are passed through untouched.
    for _key in ("enabled", "use_geometric_fallback", "use_band"):
        if _key in optimizer_params:
            _v = optimizer_params[_key]
            if isinstance(_v, str):
                _low = _v.strip().lower()
                if _low == "true":
                    optimizer_params[_key] = True
                elif _low == "false":
                    optimizer_params[_key] = False

    return planner_cls(task=task, **common_params, **optimizer_params)


class SafetyManager:
    """Safety manager that runs RL policy planning in a separate thread to find safe contact locations."""

    def __init__(
        self,
        mj_model: mujoco.MjModel, # type: ignore
        mj_model_path: str,
        rl_policy: Any,
        rl_config: Dict[str, Any],
        rate: float = 50.0,
        logger=None,
    ):
        """Initialize the safety manager.

        Args:
            mj_model: MuJoCo model (used for thread-mode init).
            mj_model_path: File path to the MuJoCo XML model (used for worker process).
            rl_policy: The RL policy to use for safety checking.
            rl_config: The RL configuration to use for safety checking.
            rate: The rate at which to run the safety filter (Hz). Default is 20 Hz.
            logger: Optional logger for debugging.
        """
        self.mj_model = mj_model
        self.mj_model_path = mj_model_path

        self.rl_config = rl_config
        self.torch_script_policy = rl_policy
        self.model_path = getattr(rl_policy, 'model_path', None)

        self.rate = rate

        self.logger = logger
        self._pending_log_messages = []

        self._ctx = mp.get_context("spawn")
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

            _num_samples    = self.rl_config.get("mps_num_samples", 400)
            _ncon_per_world = self.rl_config.get("mps_ncon_per_world", 32)  # empirical max: 23
            _njmax          = self.rl_config.get("mps_njmax", 512)           # per-world; empirical max: 161
            self.task.data = mjx.make_data(
                self.task.mj_model,
                impl="warp",
                nconmax=_ncon_per_world * _num_samples,
                njmax=_njmax,
            )

            self.task_params = self.task.make_task_params()

            alg_name = self.rl_config.get("mps_alg_name", "MIMPS")
            self.planner = create_planner(alg_name, self.task, self.rl_config)

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
                "logs": [],
            }
        )
        self._state_event = self._ctx.Event()
        self._stop_event = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_safety_worker_main,
            args=(
                self.mj_model_path,
                self.model_path,
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
            "dynamic_obstacle_p0": kwargs.get("dynamic_obstacle_p0"),
            "dynamic_obstacle_v0": kwargs.get("dynamic_obstacle_v0"),
            "dynamic_obstacle_t0": kwargs.get("dynamic_obstacle_t0"),
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

        # Drain logs unconditionally (worker may log before first plan)
        logs = self._shared_result.get("logs", [])
        for log_msg in logs:
            self._log_info(log_msg)
        if logs:
            self._shared_result["logs"] = []

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
        if self.updated_plan:
            self.updated_plan = False
            return self._latest_plan_info
        return None

    @guard_errors("getting train data", stop_on_error=False)
    def get_train_data(self):
        self._poll_results()
        return self._last_train_data

    @guard_errors("getting safety logs", stop_on_error=False)
    def get_logs(self):
        if self._shared_result is None:
            return []
        return self._shared_result.get("logs", [])

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


def _init_worker_objects(mj_model_path, rl_policy, rl_config):
    import mujoco
    import jax
    from mujoco import mjx

    mj_model = mujoco.MjModel.from_xml_path(mj_model_path)

    rl_policy_jax = _setup_jax_policy(rl_policy, rl_config)

    task_name = rl_config.get("mps_task_name", "QuadRLTask")
    if task_name != "QuadRLTask":
        raise ValueError(f"Unsupported task name: {task_name}")

    from .quad.task import QuadRLTask

    task = QuadRLTask(mj_model, rl_policy_jax, rl_config)
    # Buffer sizing for the MJX warp backend — two different semantics:
    #
    # nconmax = TOTAL contacts across ALL parallel worlds (must scale with num_samples).
    #   Evidence: "increase nconmax to 22" with nconmax=64 total means ~22 contacts/world
    #   × 400 worlds = 8800 needed >> 64 total.
    #
    # njmax = PER-WORLD efc rows (must NOT scale with num_samples).
    #   Evidence: "increase njmax to 161" = per-world nefc count needed.
    #   After vmap over N samples, efc_J becomes (N, njmax, nv).
    #   njmax=512 → (400, 512, 18) × 4 bytes = 14.7 MB — fine.
    #   njmax=512×400=204800 → (400, 204800, 18) = 5.9 GB → OOM (previous mistake).
    _num_samples    = rl_config.get("mps_num_samples", 400)
    _ncon_per_world = rl_config.get("mps_ncon_per_world", 32)  # empirical max: 23
    _njmax          = rl_config.get("mps_njmax", 512)           # per-world; empirical max: 161
    task.data = mjx.make_data(
        task.mj_model,
        impl="warp",
        nconmax=_ncon_per_world * _num_samples,
        njmax=_njmax,
    )
    print(
        f"[WORKER] Buffer: nconmax={_ncon_per_world * _num_samples} "
        f"({_ncon_per_world}×{_num_samples}), njmax={_njmax}, nbody={task.mj_model.nbody}",
        flush=True,
    )
    task_params = task.make_task_params()

    alg_name = rl_config.get("mps_alg_name", "MIMPS")
    planner = create_planner(alg_name, task, rl_config)

    random_seed = int(rl_config.get("seed", np.random.randint(0, 2**31)))
    policy_params = planner.init_params(task_params=task_params, seed=random_seed)
    jit_optimize = jax.jit(planner.optimize)
    jit_dynamics = jax.jit(task.dynamics)
    jit_mjx_forward = jax.jit(mjx.forward)
    return task, task_params, policy_params, jit_optimize, jit_dynamics, jit_mjx_forward, random_seed


# TODO: is it possible to decouple update and opt? might give a bump in speed
def _add_log(shared_result, message):
    logs = shared_result.get("logs", [])
    logs.append(message)
    shared_result["logs"] = logs


def _safety_worker_main(
    mj_model_path,
    model_path,
    rl_config,
    rate: float,
    shared_state: Any,
    shared_result: Any,
    state_event: Any,
    stop_event: Any,
):
    import os
    import signal as _signal
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_triton_gemm_any=true "
        "--xla_gpu_autotune_level=4"
    )
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Signal handler: set stop_event on SIGTERM so the worker can exit cleanly
    # rather than being killed mid-JAX-call and leaking GPU memory.
    def _on_term(signum, frame):
        stop_event.set()
    _signal.signal(_signal.SIGTERM, _on_term)

    try:
        from controllers.policies.policy import TorchScriptPolicy
        import jax.numpy as jnp
        from mujoco import mjx
        import jax

        rl_policy = TorchScriptPolicy(model_path, device="cuda")

        task, task_params, policy_params, jit_optimize, jit_dynamics, jit_mjx_forward, random_seed = _init_worker_objects(mj_model_path, rl_policy, rl_config)

        # ------------------------------------------------------------------
        # Safety-value setup (only for modes that use a learned network)
        # ------------------------------------------------------------------
        sv_mode = rl_config.get("safety_value_mode", "heuristic")
        task.safety_value_mode = sv_mode

        if sv_mode in ("direct", "residual"):
            # Create network with correct architecture
            sv_hidden_dim = rl_config.get("safety_value_hidden_dim", 256)
            sv_use_layer_norm = rl_config.get("safety_value_use_layer_norm", False)
            import safety_filter.quad.safety_value as _sv_mod
            task.safety_value_net = _sv_mod.SafetyValueNet(
                hidden_dim=sv_hidden_dim,
                use_layer_norm=sv_use_layer_norm,
            )

            # Load pre-trained checkpoint
            from pathlib import Path
            from safety_filter.quad.safety_value import load_checkpoint

            ckpt = rl_config.get("safety_value_checkpoint_path")
            if ckpt is not None and Path(ckpt).is_file():
                sv_params = load_checkpoint(ckpt)
                task_params = task_params.replace(
                    safety_value_params=sv_params,
                )
            else:
                print(
                    f"[WORKER] checkpoint not found for mode='{sv_mode}'. "
                    "Falling back to heuristic.",
                    flush=True,
                )
                task.safety_value_mode = "heuristic"

        period = 1.0 / max(rate, 1e-3)
        last_processed_seq = -1
        last_reference_generation_id = None
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

            current_generation_id = latest_state.get("reference_generation_id")
            if current_generation_id != last_reference_generation_id:
                # Reference trajectory changed (e.g. step size or heading command).
                # Reset warm-start to the new reference so MPPI doesn't spend several
                # 300ms cycles converging from the old stale plan.
                #
                # For REMIMPS we must also reset the internal per-replica arrays.
                if hasattr(policy_params, "mean_replicas"):
                    K = policy_params.mean_replicas.shape[0]
                    best_plans = jnp.repeat(
                        ref_plan_normalized[None, :, :], K, axis=0
                    )
                    means = jnp.zeros_like(best_plans)
                    stds = jnp.zeros_like(best_plans)
                    updates = dict(
                        best_plan=ref_plan_normalized,
                        mean=means[0],
                        mean_replicas=means,
                        best_plan_replicas=best_plans,
                        std_replicas=stds,
                    )
                    if hasattr(policy_params, "std"):
                        updates["std"] = jnp.zeros_like(policy_params.std)
                    policy_params = policy_params.replace(**updates)  # type: ignore
                else:
                    updates = dict(
                        best_plan=ref_plan_normalized,
                        mean=jnp.zeros_like(policy_params.mean),
                    )
                    if hasattr(policy_params, "std"):
                        updates["std"] = jnp.zeros_like(policy_params.std)
                    policy_params = policy_params.replace(**updates)  # type: ignore
                last_reference_generation_id = current_generation_id

            policy_params = update_params(
                policy_params,
                task_params=task_params,
                ref=ref_plan_normalized,
            )

            # Latency compensation: plan from the predicted state 1 plan step ahead.
            # The planner takes ~250ms and each plan step is ~300ms, so stale_steps=1
            # always. By forward-simulating one step before optimizing, plan step 0
            # corresponds to the state when the plan actually arrives → controller
            # applies step 0 directly with no horizon shift needed.
            _, data_ahead, _, _ = jit_dynamics(
                task.model, task_params, task.data, policy_params.best_plan[:2]
            )
            policy_params = jit_optimize(data_ahead, policy_params, True)

            # ── constraint diagnostic log (every 5th cycle) ──
            # if latest_seq % 5 == 0 and hasattr(policy_params, "diag"):
            #     d = np.asarray(policy_params.diag)
            #     total = d[0] + d[1] + d[2] + d[3]
            #     pct_clean = (d[0] / total * 100) if total > 0 else 0.0
            #     _add_log(shared_result,
            #         f"[DIAG-FULL] N={int(total):.0f} "
            #         f"clean={d[0]:.0f}({pct_clean:.0f}%) "
            #         f"foot={d[1]:.0f} body={d[2]:.0f} both={d[3]:.0f} | "
            #         f"mean_feet_hits={d[4]:.2f} mean_body_hits={d[5]:.2f} "
            #         f"first_foot_hit@step={d[6]:.1f} "
            #         f"c_max={np.asarray(policy_params.c_max):.2f}"
            #     )

            # Optionally dump raw train_data to disk for offline pre-training.
            # Throttled: only dump every 5th cycle to avoid IO overhead.
            data_dir = rl_config.get("safety_value_data_dir")
            if (
                data_dir
                and (latest_seq % 5 == 0)
                and hasattr(policy_params, 'train_data')
                and policy_params.train_data is not None
            ):
                try:
                    _dump_train_data(policy_params.train_data, data_dir, latest_seq, random_seed)
                except Exception as e:
                    _add_log(shared_result, f"[SAFETY] Data dump failed: {e}")

            override = True
            exec_plan = np.asarray(policy_params.exec_plan)

            # _add_log(shared_result, f"[SAFETY] Optimization completed: exec_plan shape={exec_plan.shape}, override={override}")

            best_costs = None

            # Share train_data COM trajectory if available
            if hasattr(policy_params, 'train_data') and policy_params.train_data is not None:
                qpos_traj = np.asarray(policy_params.train_data.states.qpos)
                costs_traj = np.asarray(policy_params.train_data.costs)
                constraint_traj = np.asarray(policy_params.train_data.constraint_flags)

                if qpos_traj.ndim == 3 and qpos_traj.shape[0] > 0:
                    if costs_traj.ndim > 1:
                        # _add_log(shared_result, f"[SAFETY] WARNING: costs_traj is multidimensional {costs_traj.shape}, using axis=0 argmin")
                        best_idx = np.argmin(costs_traj, axis=0)[0]
                        if best_idx >= qpos_traj.shape[0]:
                            best_idx = 0
                        best_qpos = qpos_traj[best_idx]
                        best_costs = costs_traj[best_idx]
                        best_constraints = (
                            constraint_traj[best_idx] if constraint_traj.ndim > 1 else constraint_traj
                        )
                    elif costs_traj.shape[0] != qpos_traj.shape[0]:
                        # _add_log(shared_result, f"[SAFETY] WARNING: costs_traj size ({costs_traj.shape[0]}) != qpos_traj size ({qpos_traj.shape[0]}), skipping best_qpos selection")
                        best_qpos = None
                        best_costs = None
                        best_constraints = None
                    else:
                        best_idx = np.argmin(costs_traj)
                        best_qpos = qpos_traj[best_idx]
                        best_costs = costs_traj[best_idx] if costs_traj.ndim >= 2 else costs_traj
                        best_constraints = (
                            constraint_traj[best_idx] if constraint_traj.ndim >= 2 else constraint_traj
                        )
                    
                    # # ── constraint diagnostics (every 5th cycle) ──
                    # if constraint_traj.ndim == 3 and constraint_traj.shape[2] == 2:
                    #     feet_flags  = constraint_traj[:, :, 0]   # (B, H+1)
                    #     body_counts = constraint_traj[:, :, 1]   # (B, H+1)
                    #     total_feet  = feet_flags.sum(axis=1)      # (B,)
                    #     total_body  = body_counts.sum(axis=1)     # (B,)
                    #     has_foot    = total_feet > 0
                    #     has_body    = total_body > 0
                    #     n_clean     = int(np.sum(~has_foot & ~has_body))
                    #     n_foot_only = int(np.sum( has_foot & ~has_body))
                    #     n_body_only = int(np.sum(~has_foot &  has_body))
                    #     n_both      = int(np.sum( has_foot &  has_body))
                    #     B_total     = feet_flags.shape[0]

                    #     # argmin sample (if available)
                    #     bc_feet = -1.0
                    #     bc_body = -1.0
                    #     if best_constraints is not None:
                    #         bc_feet = best_constraints[:, 0].sum()
                    #         bc_body = best_constraints[:, 1].sum()

                    #     # timing: first horizon step with a foot collision
                    #     first_hit = np.argmax(feet_flags > 0, axis=1)       # (B,); 0 = never
                    #     never_hit = total_feet == 0
                    #     first_hit_mean = first_hit[~never_hit].mean() if (~never_hit).any() else -1.0

                    #     _add_log(shared_result,
                    #         f"[DIAG] top-{B_total}: "
                    #         f"clean={n_clean} foot={n_foot_only} body={n_body_only} both={n_both} | "
                    #         f"best: feet_hits={bc_feet:.0f} body_hits={bc_body:.0f} | "
                    #         f"mean_feet_hits={total_feet.mean():.1f} "
                    #         f"max_feet_hits={total_feet.max():.0f} "
                    #         f"first_hit@step={first_hit_mean:.1f}"
                    #     )

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
                        "reference_step_seq": latest_state.get("reference_step_seq"),
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

            # Share best optimizer cost for eval logging
            if best_costs is not None:
                shared_result["best_optimizer_cost"] = float(np.asarray(best_costs).sum())

            # _add_log(shared_result, f"[SAFETY] === Result Summary === seq={latest_seq}, override={override}, plan_shape={exec_plan.shape}")

            elapsed = time.perf_counter() - loop_start
            shared_result["opt_elapsed_ms"] = float(elapsed * 1000)
            _add_log(shared_result, f"[SAFETY] Optimization elapsed time: {elapsed*1000:.2f}ms (target period: {period*1000:.2f}ms)")
            sleep_time = max(0.0, period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        tb = traceback.format_exc()
        shared_result["error"] = f"{e}\n{tb}"
