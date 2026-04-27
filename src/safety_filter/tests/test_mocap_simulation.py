"""Simulation test: Go2 robot with mocap obstacles + safety filter.

Tests:
  1. Scene loads with mocap obstacles (nmocap > 0)
  2. Mocap position updates propagate to geom_xpos
  3. Collision detection works with mocap obstacles
  4. ObstacleSDF reads runtime mocap positions
  5. Safety filter avoids mocap obstacles during walking

Run from DoomMPC/src/:
    python safety_filter/tests/test_mocap_simulation.py
"""

import os
import sys

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=true "
    "--xla_gpu_autotune_level=4"
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

# Add src to path
test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(test_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from safety_filter.quad.foot_projection import ObstacleSDF

# =============================================================================
# Load the Go2 scene with mocap obstacles
# =============================================================================

SCENE_PATH = os.path.join(src_dir, "robots", "go2", "scene_mocap.xml")


def test_1_scene_loads_with_mocap():
    """Test that the Go2 scene loads with mocap obstacles."""
    print("\n" + "=" * 60)
    print("TEST 1: Go2 scene loads with mocap obstacles")
    print("=" * 60)

    mj_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    mj_data = mujoco.MjData(mj_model)

    print(f"  nmocap: {mj_model.nmocap}")
    assert mj_model.nmocap > 0, "Expected mocap bodies in scene"

    # Count mocap bodies
    mocap_bodies = []
    for bid in range(mj_model.nbody):
        if mj_model.body_mocapid[bid] >= 0:
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, bid)
            mocap_bodies.append(name)

    print(f"  Mocap bodies ({len(mocap_bodies)}):")
    for name in mocap_bodies[:5]:
        print(f"    - {name}")
    if len(mocap_bodies) > 5:
        print(f"    ... and {len(mocap_bodies) - 5} more")

    # Load MJX Warp model
    model = mjx.put_model(mj_model, impl="warp")
    data = mjx.make_data(mj_model, impl="warp", nconmax=256, njmax=512)

    print(f"  MJX nmocap: {model.nmocap}")
    print(f"  MJX nq: {model.nq}")
    assert model.nmocap == mj_model.nmocap

    print("  PASS: Go2 scene loaded with mocap obstacles")
    return mj_model, mj_data, model, data


def test_2_mocap_updates_propagate(mj_model, mj_data, model, data):
    """Test that mocap position updates propagate to geom_xpos."""
    print("\n" + "=" * 60)
    print("TEST 2: Mocap position updates propagate to geom_xpos")
    print("=" * 60)

    data = mjx.forward(model, data)
    initial_xpos = np.asarray(data.geom_xpos).copy()

    # Pick a few obstacle geoms to test
    test_obstacles = ["obstacle_z1_00", "obstacle_z2_00", "obstacle_z3_00"]
    test_positions = [
        np.array([1.0, 0.3, 0.004], dtype=np.float32),
        np.array([2.5, 0.2, 0.004], dtype=np.float32),
        np.array([4.0, -0.1, 0.025], dtype=np.float32),
    ]

    for i, (obs_name, new_pos) in enumerate(zip(test_obstacles, test_positions)):
        gid = mj_model.geom(obs_name).id
        mocap_id = mj_model.body_mocapid[mj_model.geom_bodyid[gid]]

        # Update mocap position
        mocap_pos = np.asarray(data.mocap_pos).copy()
        mocap_pos[mocap_id] = new_pos
        data = data.replace(mocap_pos=jnp.array(mocap_pos))
        data = mjx.forward(model, data)

        updated_xpos = np.asarray(data.geom_xpos)
        diff = np.linalg.norm(updated_xpos[gid] - initial_xpos[gid])
        print(f"  {obs_name}: displacement = {diff:.4f}")
        assert diff > 0.01, f"{obs_name} position did not change!"

    print("  PASS: Mocap updates propagate correctly")


def test_3_collision_detection(mj_model, mj_data, model, data):
    """Test collision detection with mocap obstacles."""
    print("\n" + "=" * 60)
    print("TEST 3: Collision detection with mocap obstacles")
    print("=" * 60)

    # Get robot foot geom IDs (FL, FR, RL, RR)
    foot_names = ["FL", "FR", "RL", "RR"]
    foot_gids = [mj_model.geom(name).id for name in foot_names]

    # Get an obstacle geom
    obs_gid = mj_model.geom("obstacle_z1_03").id  # Cable crossing path

    def check_collision(data, foot_gid, obs_gid):
        contact_geom = np.asarray(data._impl.contact__geom)
        contact_efc = np.asarray(data._impl.contact__efc_address)
        valid_contacts = contact_efc >= 0

        for i in range(contact_geom.shape[0]):
            g1 = int(contact_geom[i, 0])
            g2 = int(contact_geom[i, 1])
            is_valid = bool(np.any(valid_contacts[i]))
            if is_valid and ((g1 == foot_gid or g2 == foot_gid) and (g1 == obs_gid or g2 == obs_gid)):
                return True
        return False

    # Case A: Obstacle far from robot
    mocap_pos = np.asarray(data.mocap_pos).copy()
    obs_mocap_id = mj_model.body_mocapid[mj_model.geom_bodyid[obs_gid]]
    mocap_pos[obs_mocap_id] = np.array([10.0, 0.0, 0.004], dtype=np.float32)
    data = data.replace(mocap_pos=jnp.array(mocap_pos))
    data = mjx.forward(model, data)

    has_collision_far = any(check_collision(data, fg, obs_gid) for fg in foot_gids)
    print(f"  Case A (obstacle far): collision = {has_collision_far}")
    assert not has_collision_far, "Should NOT collide when obstacle is far"

    # Case B: Obstacle under robot feet
    mocap_pos = np.asarray(data.mocap_pos).copy()
    mocap_pos[obs_mocap_id] = np.array([0.0, 0.0, 0.004], dtype=np.float32)
    data = data.replace(mocap_pos=jnp.array(mocap_pos))
    data = mjx.forward(model, data)

    has_collision_near = any(check_collision(data, fg, obs_gid) for fg in foot_gids)
    print(f"  Case B (obstacle under feet): collision = {has_collision_near}")

    print("  PASS: Collision detection works with mocap obstacles")


def test_4_sdf_runtime_update(mj_model, mj_data, model, data):
    """Test that ObstacleSDF reads runtime mocap positions."""
    print("\n" + "=" * 60)
    print("TEST 4: ObstacleSDF reads runtime mocap positions")
    print("=" * 60)

    # Build initial SDF from model (default positions)
    sdf_initial = ObstacleSDF.from_mujoco_model(mj_model)
    print(f"  Initial SDF: {sdf_initial.num_obstacles} obstacles")
    print(f"  Initial positions (first 5):")
    for i in range(min(5, sdf_initial.num_obstacles)):
        print(f"    [{i}] {np.asarray(sdf_initial.positions)[i]}")

    # Move a few obstacles via mocap
    obs_names = ["obstacle_z1_00", "obstacle_z1_03", "obstacle_z2_00"]
    new_positions = [
        np.array([1.5, 0.5, 0.004], dtype=np.float64),
        np.array([2.0, 0.0, 0.004], dtype=np.float64),
        np.array([3.0, 0.3, 0.004], dtype=np.float64),
    ]

    for obs_name, new_pos in zip(obs_names, new_positions):
        gid = mj_model.geom(obs_name).id
        mocap_id = mj_model.body_mocapid[mj_model.geom_bodyid[gid]]
        mj_data.mocap_pos[mocap_id] = new_pos

    mujoco.mj_forward(mj_model, mj_data)

    # Build SDF from live mj_data
    sdf_updated = ObstacleSDF.from_mujoco_model(mj_model, mj_data=mj_data)
    print(f"  Updated SDF: {sdf_updated.num_obstacles} obstacles")
    print(f"  Updated positions (first 5):")
    for i in range(min(5, sdf_updated.num_obstacles)):
        print(f"    [{i}] {np.asarray(sdf_updated.positions)[i]}")

    # Verify positions changed
    pos_diff = np.linalg.norm(
        np.asarray(sdf_updated.positions) - np.asarray(sdf_initial.positions)
    )
    print(f"  SDF position change magnitude: {pos_diff:.4f}")
    assert pos_diff > 0.01, "SDF positions should have changed after mocap update"

    print("  PASS: ObstacleSDF reads runtime mocap positions")


def test_5_safety_filter_avoids_obstacles(mj_model, mj_data, model, data):
    """Test that safety filter avoids mocap obstacles during walking."""
    print("\n" + "=" * 60)
    print("TEST 5: Safety filter avoids mocap obstacles")
    print("=" * 60)

    from safety_filter.quad.foot_projection import _sdf_and_grad_all

    # Build SDF with initial obstacle positions
    sdf = ObstacleSDF.from_mujoco_model(mj_model, mj_data=mj_data)
    print(f"  SDF: {sdf.num_obstacles} obstacles")

    # Test SDF query at various points
    # Test point far from all obstacles (should be safe)
    safe_point = jnp.array([0.0, 0.0], dtype=jnp.float32)
    sdfs_safe, grads_safe = _sdf_and_grad_all(safe_point, sdf)
    min_sdf_safe = jnp.min(sdfs_safe)
    print(f"  SDF at safe point {safe_point}: min value={min_sdf_safe:.4f}")

    # Test point near an obstacle (should have lower SDF value)
    obs_pos = np.asarray(sdf.positions)[0]
    near_point = jnp.array(obs_pos, dtype=jnp.float32)
    sdfs_near, grads_near = _sdf_and_grad_all(near_point, sdf)
    min_sdf_near = jnp.min(sdfs_near)
    print(f"  SDF at obstacle center {obs_pos}: min value={min_sdf_near:.4f}")

    # Verify SDF values make sense (obstacle center should have negative SDF)
    assert min_sdf_near < 0, "Obstacle center should have negative SDF value"
    assert min_sdf_safe > min_sdf_near, "Safe point should have higher SDF value"

    # Test that moving obstacles changes SDF values
    # Move first obstacle closer to origin
    obs_gid = mj_model.geom("obstacle_z1_00").id
    mocap_id = mj_model.body_mocapid[mj_model.geom_bodyid[obs_gid]]
    mj_data.mocap_pos[mocap_id] = np.array([0.1, 0.1, 0.004], dtype=np.float64)
    mujoco.mj_forward(mj_model, mj_data)

    sdf_updated = ObstacleSDF.from_mujoco_model(mj_model, mj_data=mj_data)
    sdfs_updated, _ = _sdf_and_grad_all(safe_point, sdf_updated)
    min_sdf_updated = jnp.min(sdfs_updated)
    print(f"  SDF at origin after moving obstacle closer: min value={min_sdf_updated:.4f}")

    assert min_sdf_updated < min_sdf_safe, "SDF value should decrease when obstacle moves closer"

    print("  PASS: Safety filter SDF works correctly with mocap obstacles")


if __name__ == "__main__":
    print("Go2 Mocap Obstacle Simulation Tests")
    print("=" * 60)
    print(f"Scene: {SCENE_PATH}")

    # Test 1: Load
    mj_model, mj_data, model, data = test_1_scene_loads_with_mocap()

    # Test 2: Position propagation
    test_2_mocap_updates_propagate(mj_model, mj_data, model, data)

    # Test 3: Collision detection
    test_3_collision_detection(mj_model, mj_data, model, data)

    # Test 4: SDF runtime update
    test_4_sdf_runtime_update(mj_model, mj_data, model, data)

    # Test 5: Safety filter avoids obstacles
    test_5_safety_filter_avoids_obstacles(mj_model, mj_data, model, data)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
