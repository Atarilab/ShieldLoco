"""Prototype: Verify mocap bodies work with MJX Warp for dynamic obstacles.

Tests:
  1. Mocap bodies load correctly in MJX Warp (nmocap > 0)
  2. Mocap position updates propagate to geom_xpos after mjx.forward
  3. Collision detection works when robot contacts a mocap obstacle
  4. ObstacleSDF can be rebuilt with runtime geom positions

Run from DoomMPC/src/:
    python safety_filter/tests/test_mocap_obstacles.py
"""

import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=true "
    "--xla_gpu_autotune_level=4"
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

# =============================================================================
# Minimal test scene: sliding box robot + 2 mocap obstacle bodies
# =============================================================================

TEST_SCENE = """
<mujoco>
  <option timestep="0.02" integrator="implicitfast" solver="CG" iterations="4"/>

  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.05" rgba="0.3 0.3 0.3 1"/>

    <!-- Simple robot: a box on a slide joint (1 DOF along x) -->
    <body name="robot" pos="0 0 0.3">
      <joint name="robot_x" type="slide" axis="1 0 0" damping="10"/>
      <geom name="robot_body" type="box" size="0.15 0.15 0.15" rgba="0.2 0.6 0.2 1" mass="5"/>
    </body>

    <!-- Mocap obstacle 1: box at (1.0, 0, 0.2) -->
    <body name="obstacle_0" mocap="true" pos="1.0 0 0.2">
      <geom name="obstacle_0" type="box" size="0.15 0.15 0.2" rgba="0.8 0.2 0.2 1"/>
    </body>

    <!-- Mocap obstacle 2: box at (2.0, 0, 0.2) -->
    <body name="obstacle_1" mocap="true" pos="2.0 0 0.2">
      <geom name="obstacle_1" type="box" size="0.15 0.15 0.2" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_1_mocap_bodies_load():
    """Test that mocap bodies are recognized by both MuJoCo and MJX."""
    print("\n" + "=" * 60)
    print("TEST 1: Mocap bodies load correctly")
    print("=" * 60)

    mj_model = mujoco.MjModel.from_xml_string(TEST_SCENE)
    mj_data = mujoco.MjData(mj_model)

    # MuJoCo checks
    print(f"  nmocap (MuJoCo): {mj_model.nmocap}")
    assert mj_model.nmocap == 2, f"Expected 2 mocap bodies, got {mj_model.nmocap}"

    for i in range(mj_model.nmocap):
        # Find which body has mocapid == i
        for bid in range(mj_model.nbody):
            if mj_model.body_mocapid[bid] == i:
                body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, bid)
                print(f"  mocap[{i}] -> body '{body_name}' (id={bid})")
                break

    # MJX Warp checks
    model = mjx.put_model(mj_model, impl="warp")
    data = mjx.make_data(mj_model, impl="warp", nconmax=64, njmax=128)

    print(f"  nmocap (MJX model): {model.nmocap}")
    assert model.nmocap == 2, f"Expected 2 mocap bodies in MJX, got {model.nmocap}"
    # In MJX, mocap DOFs are NOT in qpos — they have separate mocap_pos/mocap_quat fields
    print(f"  nq (MJX): {model.nq}  (joint DOFs only; mocap uses separate mocap_pos/mocap_quat)")
    assert model.nq == 1, f"Expected nq=1 (1 joint DOF, mocap separate), got {model.nq}"

    print("  PASS: Mocap bodies loaded correctly in MJX Warp")
    return mj_model, mj_data, model, data


def test_2_mocap_position_updates_propagate(mj_model, mj_data, model, data):
    """Test that updating mocap_pos propagates to geom_xpos via mjx.forward."""
    print("\n" + "=" * 60)
    print("TEST 2: Mocap position updates propagate to geom_xpos")
    print("=" * 60)

    # Initial forward pass
    data = mjx.forward(model, data)
    initial_geom_xpos = np.asarray(data.geom_xpos)

    # Find obstacle geom IDs
    obs0_gid = mj_model.geom("obstacle_0").id
    obs1_gid = mj_model.geom("obstacle_1").id

    print(f"  obstacle_0 geom_id: {obs0_gid}")
    print(f"  obstacle_1 geom_id: {obs1_gid}")
    print(f"  Initial obstacle_0 xpos: {initial_geom_xpos[obs0_gid]}")
    print(f"  Initial obstacle_1 xpos: {initial_geom_xpos[obs1_gid]}")

    # Update mocap positions: move obstacle_0 to (0.5, 0.5, 0.2)
    new_obs0_pos = np.array([0.5, 0.5, 0.2], dtype=np.float32)
    new_obs1_pos = np.array([2.0, -0.3, 0.3], dtype=np.float32)

    # In MJX, mocap positions are stored in data.mocap_pos (nmocap, 3)
    # and data.mocap_quat (nmocap, 4), NOT in qpos.
    mocap_pos = np.asarray(data.mocap_pos).copy()
    mocap_pos[0] = new_obs0_pos
    mocap_pos[1] = new_obs1_pos

    data = data.replace(mocap_pos=jnp.array(mocap_pos))

    # Forward pass to propagate kinematics
    data = mjx.forward(model, data)
    updated_geom_xpos = np.asarray(data.geom_xpos)

    print(f"  Updated obstacle_0 xpos: {updated_geom_xpos[obs0_gid]}")
    print(f"  Updated obstacle_1 xpos: {updated_geom_xpos[obs1_gid]}")

    # Verify positions changed
    obs0_diff = np.linalg.norm(updated_geom_xpos[obs0_gid] - initial_geom_xpos[obs0_gid])
    obs1_diff = np.linalg.norm(updated_geom_xpos[obs1_gid] - initial_geom_xpos[obs1_gid])

    print(f"  obstacle_0 displacement: {obs0_diff:.4f}")
    print(f"  obstacle_1 displacement: {obs1_diff:.4f}")

    assert obs0_diff > 0.01, "obstacle_0 position did not change!"
    assert obs1_diff > 0.01, "obstacle_1 position did not change!"

    # Verify positions match what we set (within tolerance)
    np.testing.assert_allclose(
        updated_geom_xpos[obs0_gid][:3], new_obs0_pos, atol=1e-5,
        err_msg="obstacle_0 geom_xpos doesn't match set mocap position"
    )
    np.testing.assert_allclose(
        updated_geom_xpos[obs1_gid][:3], new_obs1_pos, atol=1e-5,
        err_msg="obstacle_1 geom_xpos doesn't match set mocap position"
    )

    print("  PASS: Mocap position updates propagate correctly to geom_xpos")


def test_3_collision_detection_with_mocap(mj_model, mj_data, model, data):
    """Test that collision detection works when robot contacts a mocap obstacle."""
    print("\n" + "=" * 60)
    print("TEST 3: Collision detection with mocap obstacles")
    print("=" * 60)

    # Place obstacle_0 directly in front of robot so they overlap
    # Robot is at x=0, size 0.15. Obstacle at x=0.2, size 0.15 → overlap
    overlap_pos = np.array([0.2, 0.0, 0.2], dtype=np.float32)
    far_pos = np.array([5.0, 0.0, 0.2], dtype=np.float32)

    obs0_gid = mj_model.geom("obstacle_0").id
    robot_gid = mj_model.geom("robot_body").id

    # --- Case A: Obstacle far away → no collision ---
    mocap_pos = np.asarray(data.mocap_pos).copy()
    mocap_pos[0] = far_pos
    mocap_pos[1] = far_pos
    data = data.replace(mocap_pos=jnp.array(mocap_pos))
    data = mjx.forward(model, data)

    # Check contacts via Warp's private contact data
    contact_geom = np.asarray(data._impl.contact__geom)
    contact_efc = np.asarray(data._impl.contact__efc_address)
    valid_contacts = contact_efc >= 0

    obs0_contact_far = False
    for i in range(contact_geom.shape[0]):
        g1 = int(contact_geom[i, 0])
        g2 = int(contact_geom[i, 1])
        is_valid = bool(np.any(valid_contacts[i]))
        if is_valid and ((g1 == obs0_gid or g2 == obs0_gid) and (g1 == robot_gid or g2 == robot_gid)):
            obs0_contact_far = True
            break

    print(f"  Case A (obstacle far): robot-obstacle_0 collision = {obs0_contact_far}")
    assert not obs0_contact_far, "Should NOT have collision when obstacle is far"

    # --- Case B: Obstacle overlapping robot → collision ---
    mocap_pos = np.asarray(data.mocap_pos).copy()
    mocap_pos[0] = overlap_pos
    data = data.replace(mocap_pos=jnp.array(mocap_pos))
    data = mjx.forward(model, data)

    contact_geom = np.asarray(data._impl.contact__geom)
    contact_efc = np.asarray(data._impl.contact__efc_address)
    valid_contacts = contact_efc >= 0

    obs0_contact_near = False
    for i in range(contact_geom.shape[0]):
        g1 = int(contact_geom[i, 0])
        g2 = int(contact_geom[i, 1])
        is_valid = bool(np.any(valid_contacts[i]))
        if is_valid and ((g1 == obs0_gid or g2 == obs0_gid) and (g1 == robot_gid or g2 == robot_gid)):
            obs0_contact_near = True
            break

    print(f"  Case B (obstacle overlapping): robot-obstacle_0 collision = {obs0_contact_near}")
    assert obs0_contact_near, "Should HAVE collision when obstacle overlaps robot"

    print("  PASS: Collision detection works correctly with mocap obstacles")


def test_4_obstacle_sdf_runtime_update(mj_model, mj_data, model, data):
    """Test that ObstacleSDF can be updated with runtime geom positions.

    NOTE: ObstacleSDF.from_mujoco_model() reads from a FRESH MjData (static defaults).
    To get runtime positions, we must either:
      (a) Pass the live MjData to from_mujoco_model, or
      (b) Add an update_positions() method that takes geom_xpos/geom_xmat from MJX.

    This test demonstrates (a) and shows what (b) would look like.
    """
    print("\n" + "=" * 60)
    print("TEST 4: ObstacleSDF runtime update with mocap positions")
    print("=" * 60)

    import sys
    # Add the src directory to path (handles both local and Docker paths)
    test_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(os.path.dirname(test_dir))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from safety_filter.quad.foot_projection import ObstacleSDF

    # Build initial SDF from model (obstacles at default positions)
    sdf_initial = ObstacleSDF.from_mujoco_model(mj_model)
    print(f"  Initial SDF: {sdf_initial.num_obstacles} obstacles")
    print(f"  Initial positions: {np.asarray(sdf_initial.positions)}")

    # Move obstacles via mocap in mj_data
    new_obs0_pos = np.array([0.5, 0.5, 0.2], dtype=np.float64)
    new_obs1_pos = np.array([2.0, -0.3, 0.3], dtype=np.float64)

    mocap0_id = mj_model.body_mocapid[mj_model.body("obstacle_0").id]
    mocap1_id = mj_model.body_mocapid[mj_model.body("obstacle_1").id]

    # Update mj_data mocap positions and run mj_forward
    mj_data.mocap_pos[mocap0_id] = new_obs0_pos
    mj_data.mocap_pos[mocap1_id] = new_obs1_pos
    mujoco.mj_forward(mj_model, mj_data)

    print(f"  Moved obstacles via mocap:")
    print(f"    obstacle_0: {new_obs0_pos}")
    print(f"    obstacle_1: {new_obs1_pos}")
    print(f"  mj_data.geom_xpos[obstacle_0]: {mj_data.geom_xpos[mj_model.geom('obstacle_0').id]}")

    # --- Approach (a): Build SDF from live mj_data ---
    # Modify from_mujoco_model to accept optional mj_data
    sdf_updated = ObstacleSDF.from_mujoco_model(mj_model, mj_data=mj_data)
    print(f"  Updated SDF (from live mj_data): {sdf_updated.num_obstacles} obstacles")
    print(f"  Updated positions: {np.asarray(sdf_updated.positions)}")

    # Verify SDF positions changed
    pos_diff = np.linalg.norm(
        np.asarray(sdf_updated.positions) - np.asarray(sdf_initial.positions)
    )
    print(f"  SDF position change magnitude: {pos_diff:.4f}")
    assert pos_diff > 0.01, "SDF positions should have changed after mocap update"

    # Verify SDF positions match mocap positions (XY only)
    expected_positions = np.array([
        new_obs0_pos[:2],
        new_obs1_pos[:2],
    ], dtype=np.float32)
    actual_positions = np.asarray(sdf_updated.positions)
    np.testing.assert_allclose(
        actual_positions, expected_positions, atol=1e-4,
        err_msg="SDF positions don't match mocap XY positions"
    )

    print("  PASS: ObstacleSDF correctly reflects mocap-updated obstacle positions")
    print("  NOTE: Requires from_mujoco_model(mj_model, mj_data=...) signature")


def test_5_jit_compatibility(mj_model, model, data):
    """Test that mocap_pos updates work inside JIT-compiled functions."""
    print("\n" + "=" * 60)
    print("TEST 5: JIT compatibility with mocap updates")
    print("=" * 60)

    @jax.jit
    def step_with_mocap(data, mocap_positions):
        """Update mocap positions and step simulation."""
        # mocap_positions: (nmocap, 3)
        data = data.replace(mocap_pos=mocap_positions)
        data = mjx.forward(model, data)
        data = mjx.step(model, data)
        return data

    # Initial data
    data = mjx.make_data(mj_model, impl="warp", nconmax=64, njmax=128)

    # Set mocap positions and run JIT step
    mocap_pos = jnp.array([
        [0.5, 0.0, 0.2],
        [2.0, 0.0, 0.2],
    ], dtype=jnp.float32)

    # First call (compilation)
    data = step_with_mocap(data, mocap_pos)
    print(f"  First JIT step completed, robot qpos[0] = {np.asarray(data.qpos)[0]:.4f}")

    # Second call (cached)
    mocap_pos2 = jnp.array([
        [0.3, 0.1, 0.2],
        [1.8, -0.1, 0.2],
    ], dtype=jnp.float32)
    data = step_with_mocap(data, mocap_pos2)
    print(f"  Second JIT step completed, robot qpos[0] = {np.asarray(data.qpos)[0]:.4f}")

    print("  PASS: JIT compilation works with mocap position updates")


if __name__ == "__main__":
    print("Mocap + MJX Warp Compatibility Tests")
    print("=" * 60)

    # Test 1: Load
    mj_model, mj_data, model, data = test_1_mocap_bodies_load()

    # Test 2: Position propagation
    test_2_mocap_position_updates_propagate(mj_model, mj_data, model, data)

    # Test 3: Collision detection
    test_3_collision_detection_with_mocap(mj_model, mj_data, model, data)

    # Test 4: SDF runtime update
    test_4_obstacle_sdf_runtime_update(mj_model, mj_data, model, data)

    # Test 5: JIT compatibility
    test_5_jit_compatibility(mj_model, model, data)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
