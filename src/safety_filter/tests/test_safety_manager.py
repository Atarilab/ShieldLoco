"""
Test SafetyManager - verifies safety filter integration with MPPI planner.
"""

import os
import time
import numpy as np
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
import torch

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_deterministic_ops=true "
    "--xla_gpu_autotune_level=0"
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"


def get_rl_config():
    """Create rl_config dictionary for SafetyManager."""
    
    # Robot joint configuration
    default_joint_pos = np.array([
        0.1, -0.1, 0.1, -0.1,  # FL hip, FR hip, RL hip, RR hip
        0.8, 0.8, 1.0, 1.0,    # FL thigh, FR thigh, RL thigh, RR thigh
        -1.5, -1.5, -1.5, -1.5  # FL calf, FR calf, RL calf, RR calf
    ])
    
    action_offset = default_joint_pos.copy()
    action_scale = 0.35
    
    # Joint position limits (approximate for Go2)
    j_pd_min = np.array([
        -1.0472, -1.0472, -1.0472, -1.0472,  # hip
        -2.775, -2.775, -2.775, -2.775,       # thigh
        -2.775, -2.775, -2.775, -2.775         # calf
    ])
    j_pd_max = np.array([
        1.0472, 1.0472, 1.0472, 1.0472,  # hip
        2.775, 2.775, 2.775, 2.775,      # thigh
        -0.524, -0.524, -0.524, -0.524    # calf (negative!)
    ])
    
    # Foot site IDs - will be set after loading model
    foot_ids = np.array([0, 1, 2, 3])  # Placeholder, updated below
    
    # Joint observation mapping (Unitree to IsaacLab)
    joint_obs_unitree_to_isaac_mapping = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    # Action mapping
    action_mapping = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    rl_config = {
        # Physics
        "timestep": 0.01,
        
        # Observation
        "obs_horizon": 2,
        "command_duration": 0.35,
        
        # Robot config
        "joint_obs_unitree_to_isaac_mapping": joint_obs_unitree_to_isaac_mapping,
        "default_joint_pos": default_joint_pos,
        "foot_ids": foot_ids,
        
        # Action limits
        "plan_lim": 0.1,
        "act_rl_clip": 0.5,
        "action_offset": action_offset,
        "action_scale": action_scale,
        "action_mapping": action_mapping,
        
        # PD control
        "j_pd_min": j_pd_min,
        "j_pd_max": j_pd_max,
        "kp": 30.0,
        "kd": 0.5,
        
        # MPC
        "decimation": 4,
        
        # MPS Planner config
        "mps_task_name": "QuadRLTask",
        "mps_alg_name": "MIMPS",
        "mps_num_samples": 100,
        "mps_noise_level": 0.5,
        "mps_temperature": 0.1,
        "mps_plan_horizon": 0.1,
        "mps_iterations": 1,
        "mps_gamma": 0.99,
        "mps_safety_weight": 1e3,
        "mps_collect_data": True,
        "mps_top_fraction": 0.01,
        
        # Obstacles (from scene.xml)
        "obstacles": [
            [1.5, -0.3, 0.2, 0.15, 0.15, 0.2],
            [3.0, 0.5, 0.2, 0.15, 0.15, 0.2],
            [4.0, 0.0, 0.2, 0.15, 0.15, 0.2],
            [4.7, -0.8, 0.2, 0.15, 0.15, 0.2],
            [5.5, 0.0, 0.2, 0.15, 0.15, 0.2],
        ],
        "safety_margin": 0.15,
        "robot_radius": 0.15,
    }
    
    return rl_config


def get_policy():
    """Load the real RL policy."""
    from controllers.policies.policy import TorchScriptPolicy
    
    policy_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "controllers", "policies", "locomotion_go2", "contact_policy_stanceSeperate.pt"
    )
    policy_path = os.path.normpath(policy_path)
    
    device = "cpu"  # Change to "cuda:0" if available
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
    
    # Get foot site IDs
    foot_site_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    foot_ids = []
    for name in foot_site_names:
        try:
            foot_ids.append(mj_model.site(name).id)
        except:
            print(f"Warning: Could not find site {name}, using placeholder")
            foot_ids.append(len(foot_ids))
    print(f"Foot site IDs: {foot_ids}")
    
    # Get config and update foot_ids
    rl_config = get_rl_config()
    rl_config["foot_ids"] = np.array(foot_ids)
    
    # Load policy
    print("\nLoading policy...")
    policy = get_policy()
    print(f"Policy loaded: {type(policy)}")



    # Create initial state
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = 0  # Reset position
    mj_data.qvel[:] = 0  # Reset velocity
    mj_data.time = 0.0
    
    # Convert to MJX
    model = mjx.put_model(mj_model, impl="warp")
    data = mjx.make_data(mj_model, impl="warp", nconmax=1000, njmax=1000)
    
    # Update MJX data from MJ data
    data = data.replace(
        qpos=jnp.array(mj_data.qpos),
        qvel=jnp.array(mj_data.qvel),
        time=jnp.array(mj_data.time)
    )
    
    # Create initial reference plan
    horizon_length = int(rl_config["mps_plan_horizon"] / rl_config["timestep"])
    rl_config["horizon_length"] = horizon_length
    ref = np.zeros((4, horizon_length, 3))
    ref[:, :, 2] = 0.1  # Foot height



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
    
    
    # Update state
    safety_manager.update_state(
        ref=ref,
        qpos=mj_data.qpos,
        qvel=mj_data.qvel,
        time=mj_data.time,
        command_start_time=0.0,
        goal_idx=0,
        contact_plan=np.zeros(8),
        current_gait=0,
        goal_completion_counter=0,
        in_transition=False,
    )
    print("✓ State update successful")
    
    # Verify task_params were updated
    assert safety_manager.task_params.command_start_time == 0.0
    assert safety_manager.task_params.goal_idx == 0
    print("✓ Task params updated correctly")
    
    # Let it run for a bit
    print("Letting thread run for 2 seconds...")
    time.sleep(2.0)
    
    # Check if plan was updated
    print("Checking plan update...")
    with safety_manager._safety_lock:
        plan = safety_manager.exec_plan
    
    if plan is not None:
        print(f"✓ Plan shape: {plan.shape}")
        assert plan.shape == (horizon_length, 12), f"Expected {(horizon_length, 12)}, got {plan.shape}"
    else:
        print("⚠ Plan is None (may need more time or check() implementation)")
    
    # Plot 2D trajectory from train_data
    print("\n--- Plotting 2D Trajectory ---")
    try:
        with safety_manager._safety_lock:
            train_data = safety_manager.policy_params.train_data
        
        if train_data is not None:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Get x,y from qpos (first 2 elements are x, y of base)
            qpos = np.array(train_data.states.qpos)
            num_traj = qpos.shape[0]
            
            # Plot all trajectories
            plt.figure(figsize=(8, 8))
            for i in range(num_traj):
                plt.plot(qpos[i, :, 0], qpos[i, :, 1], 'b-', alpha=0.3)
            
            # Mark start and end points
            plt.plot(qpos[:, 0, 0], qpos[:, 0, 1], 'go', label='Start', markersize=10)
            plt.plot(qpos[:, -1, 0], qpos[:, -1, 1], 'r*', label='End', markersize=10)
            
            plt.xlabel('X position')
            plt.ylabel('Y position')
            plt.title('Robot 2D Trajectory (from train_data)')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            
            plot_path = os.path.join(os.path.dirname(__file__), 'trajectory_plot.png')
            plt.savefig(plot_path)
            print(f"✓ Trajectory plot saved to {plot_path}")
        else:
            print("⚠ train_data is None (collect_data may be False)")
    except Exception as e:
        print(f"⚠ Plotting error: {e}")
    
    # Test thread stop
    print("\nStopping safety thread...")
    safety_manager.stop(timeout=2.0)
    
    assert not safety_manager.is_running(), "Safety thread still running"
    print("✓ Safety thread stopped")
    
    # Test constraint_check (if implemented)
    print("\n--- Verification: Constraint Check ---")
    try:
        with safety_manager._safety_lock:
            task_data = safety_manager.task.data
        constraint = jax.jit(safety_manager.task.constraint_check)(task_data)
        print(f"✓ Constraint check result: {constraint}")
    except NotImplementedError:
        print("⚠ constraint_check not implemented yet")
    except Exception as e:
        print(f"⚠ constraint_check error: {e}")
    
    # Test running_cost and terminal_cost (if implemented)
    print("\n--- Verification: Cost Functions ---")
    try:
        # Create dummy plan segment (updated API expects 2-point segment + ref segment)
        plan_dim = getattr(safety_manager.task, "plan_dim", 12)
        dummy_plan_segment = jnp.zeros((2, plan_dim))
        dummy_ref_segment = jnp.zeros_like(dummy_plan_segment)
        with safety_manager._safety_lock:
            task_params = safety_manager.task_params
            task_data = safety_manager.task.data
        running_cost = safety_manager.task.running_cost(
            task_params,
            task_data,
            dummy_plan_segment,
            dummy_ref_segment,
        )
        print(f"✓ Running cost: {running_cost}")
    except NotImplementedError:
        print("⚠ running_cost not implemented yet")
    except Exception as e:
        print(f"⚠ running_cost error: {e}")
    
    try:
        with safety_manager._safety_lock:
            task_params = safety_manager.task_params
            task_data = safety_manager.task.data
        terminal_cost = safety_manager.task.terminal_cost(task_params, task_data)
        print(f"✓ Terminal cost: {terminal_cost}")
    except NotImplementedError:
        print("⚠ terminal_cost not implemented yet")
    except Exception as e:
        print(f"⚠ terminal_cost error: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
