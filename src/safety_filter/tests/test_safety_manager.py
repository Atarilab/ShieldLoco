"""
Test SafetyManager - verifies safety filter integration with MPPI planner.
"""

import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_deterministic_ops=true "
    "--xla_gpu_autotune_level=0"
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import time
import numpy as np
import mujoco

from utils.config_loader import load_config


def get_rl_config():
    """Load rl_config dictionary directly from RL-MPS controller config."""
    rl_config = load_config("controllers/config/rl_mps_go2_cfg.yaml")

    for key in [
        "joint_obs_unitree_to_isaac_mapping",
        "default_joint_pos",
        "action_offset",
        "action_mapping",
        "j_pd_min",
        "j_pd_max",
    ]:
        rl_config[key] = np.asarray(rl_config[key])

    # Keep GPU/Warp path, but reduce optimization load for test determinism.
    rl_config["mps_num_samples"] = min(int(rl_config.get("mps_num_samples", 1000)), 128)
    rl_config["mps_iterations"] = min(int(rl_config.get("mps_iterations", 3)), 1)
    rl_config["mps_collect_data"] = True

    return rl_config


def get_policy(rl_config):
    """Load the real RL policy."""
    from controllers.policies.policy import TorchScriptPolicy
    
    policy_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "controllers", "policies", rl_config["policy_path"]
    )
    policy_path = os.path.normpath(policy_path)
    
    device = "cpu"
    policy = TorchScriptPolicy(policy_path, device)
    
    return policy


def main():
    """Run safety manager tests."""
    print("=" * 60)
    print("Testing SafetyManager")
    print("=" * 60)
    
    # Load MuJoCo model
    xml_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "robots", "go2", "scene_mjx.xml"
    )
    xml_path = os.path.normpath(xml_path)
    print(f"Loading model from: {xml_path}")
    
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    print(f"Model loaded: {mj_model.nq} DOFs, {mj_model.nu} actuators")
    
    # Load config directly from controller YAML
    rl_config = get_rl_config()
    
    # Load policy
    print("\nLoading policy...")
    policy = get_policy(rl_config)
    print(f"Policy loaded: {type(policy)}")



    # Create initial state
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = 0  # Reset position
    mj_data.qvel[:] = 0  # Reset velocity
    mj_data.time = 0.0
    
    # Create reference plan with forward movement (trot gait)
    horizon_length = int(rl_config["mps_plan_horizon"] / rl_config["timestep"])
    rl_config["horizon_length"] = horizon_length
    
    # Default stance offset for GO2 (FL, FR, RL, RR)
    default_stance = np.array([
        [0.1934, 0.1465, 0.0],   # FL
        [0.1934, -0.1465, 0.0],  # FR
        [-0.1934, 0.1465, 0.0],  # RL
        [-0.1934, -0.1465, 0.0], # RR
    ])
    
    # Create reference with forward movement (stride = 0.1m per step)
    ref = np.zeros((4, horizon_length, 3))
    stride = 0.1  # forward stride per timestep
    for t in range(horizon_length):
        ref[:, t, 0] = default_stance[:, 0] + t * stride  # forward in X
        ref[:, t, 1] = default_stance[:, 1]  # lateral stays same
        ref[:, t, 2] = 0.1  # foot height
    
    print(f"\nReference plan shape: {ref.shape}")
    print(f"Ref at t=0: FL={ref[0, 0]}, FR={ref[1, 0]}")
    print(f"Ref at t={horizon_length-1}: FL={ref[0, horizon_length-1]}, FR={ref[1, horizon_length-1]}")

    # Create SafetyManager
    print("\nInitializing SafetyManager...")
    from safety_filter.safety_manager import SafetyManager
    
    safety_manager = SafetyManager(
        mj_model=mj_model,
        rl_policy=policy,
        rl_config=rl_config,
        rate=20.0,
        logger=None
    )
    print("SafetyManager initialized")
    
    # Test thread start/stop
    print("\n--- Verification: Thread Start/Stop ---")
    print("Starting safety thread...")
    safety_manager.start()
    time.sleep(10.0)  # Give thread time to initialize (init_thread runs in thread)
    
    assert safety_manager.is_running(), "Safety thread not running"
    print("✓ Safety thread started")
    
    # Test state update (after thread has initialized)
    print("\n--- Verification: State Update ---")
    
    
    # Update state with trot gait (gait=2) and forward movement
    print("\nSending state update with trot gait and forward movement...", flush=True)
    safety_manager.update_state(
        ref=ref,
        qpos=mj_data.qpos,
        qvel=mj_data.qvel,
        time=mj_data.time,
        command_start_time=0.0,
        goal_idx=0,
        contact_plan=np.array([1, 0, 0, 1, 0, 1, 1, 0]),  # Will be updated by gait
        current_gait=2,  # trot
        goal_completion_counter=0,
        in_transition=False,
    )
    print("✓ State update successful")
    
    # Note: With MP implementation, task_params is inside worker process
    # Cannot access directly - verify via public API instead
    
    # Wait for first plan (first optimize pass can take longer due JIT/warmup)
    print("Waiting for first plan (up to 900s)...")
    deadline = time.time() + 900.0
    plan = None
    while time.time() < deadline and safety_manager.is_running():
        plan = safety_manager.get_latest_plan()
        if plan is not None:
            break
        time.sleep(0.1)

    print("Checking plan update...")
    if plan is None:
        worker = getattr(safety_manager, "_process", None)
        exit_code = worker.exitcode if worker is not None else None
        raise AssertionError(
            f"No plan received within timeout. is_running={safety_manager.is_running()}, worker_exitcode={exit_code}"
        )
    print(f"✓ Plan shape: {plan.shape}")
    assert plan.shape == (4, horizon_length, 3), f"Expected {(4, horizon_length, 3)}, got {plan.shape}"
    
    # Get base trajectory + rollout diagnostics from train_data
    print("\n--- Checking Rollout Trajectory ---")
    train_data = safety_manager.get_train_data()
    if train_data is not None and train_data.get("com") is not None:
        com = train_data["com"]
        print(f"Base trajectory shape: {com.shape}")
        print(f"Base at t=0: {com[0, 0]}")
        print(f"Base at t=mid: {com[0, com.shape[1]//2]}")
        print(f"Base at t=end: {com[0, -1]}")
        
        # Check movement in each direction
        movement = com[0, -1] - com[0, 0]
        print(f"Base movement: X={movement[0]:.4f}, Y={movement[1]:.4f}, Z={movement[2]:.4f}")
    else:
        print("WARNING: No train_data or COM available!")
    
    # Test thread stop
    print("\nStopping safety thread...")
    safety_manager.stop(timeout=2.0)
    
    assert not safety_manager.is_running(), "Safety thread still running"
    print("✓ Safety thread stopped")
    
    # Note: With MP implementation, task/task_params are inside worker process
    # These internal tests cannot run from main process
    
    print("\n--- Skipped (MP internal) ---")
    print("⚠ constraint_check, running_cost, terminal_cost tests skipped")
    print("  (These require access to worker-internal objects)")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
