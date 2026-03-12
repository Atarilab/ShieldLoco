"""
BallLauncher: Sim-only ball spawning and lifecycle management.

Manages pre-allocated freejoint ball bodies in MuJoCo. Balls are initially
hidden underground and activated on demand by setting qpos/qvel on the
corresponding freejoint DOFs.

Hardware note: This class is sim-only. On hardware, dynamic obstacle
positions come from mocap via DynamicObstacleTracker.update_from_mocap().
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import mujoco
import numpy as np


@dataclass
class BallSlot:
    """State for a single pre-allocated ball body."""
    body_id: int  # MuJoCo body id
    qpos_adr: int  # Start index into mj_data.qpos for freejoint (7 DOFs: pos + quat)
    qvel_adr: int  # Start index into mj_data.qvel for freejoint (6 DOFs: lin + ang)
    active: bool = False
    launch_time: float = 0.0
    landed_time: Optional[float] = None
    p0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 0.05


class BallLauncher:
    """Manages launching and lifecycle of pre-allocated ball projectiles in MuJoCo sim."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        max_balls: int = 5,
        ball_radius: float = 0.05,
        launch_distance_range: tuple = (4.0, 6.0),
        launch_height_range: tuple = (2.0, 3.0),
        launch_speed_range: tuple = (2.0, 4.0),
        ball_lifetime: float = 3.0,
        ground_z_threshold: float = 0.1,
    ):
        self.max_balls = max_balls
        self.ball_radius = ball_radius
        self.launch_distance_range = launch_distance_range
        self.launch_height_range = launch_height_range
        self.launch_speed_range = launch_speed_range
        self.ball_lifetime = ball_lifetime
        self.ground_z_threshold = ground_z_threshold
        self._next_slot = 0

        # Resolve ball body/joint IDs from the model
        self.slots: List[BallSlot] = []
        for i in range(max_balls):
            body_name = f"ball_{i}"
            joint_name = f"ball_joint_{i}"

            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

            if body_id < 0 or joint_id < 0:
                raise ValueError(
                    f"Ball body/joint '{body_name}'/'{joint_name}' not found in model. "
                    "Ensure obstacles.xml has the pre-allocated ball bodies."
                )

            qpos_adr = mj_model.jnt_qposadr[joint_id]
            qvel_adr = mj_model.jnt_dofadr[joint_id]

            self.slots.append(BallSlot(
                body_id=body_id,
                qpos_adr=qpos_adr,
                qvel_adr=qvel_adr,
                radius=ball_radius,
            ))

    def launch(self, robot_pos: np.ndarray, mj_data: mujoco.MjData) -> Optional[int]:
        """Launch a ball toward the robot's current position.

        Picks the next slot (round-robin), computes a random launch origin at a
        random compass angle, and sets initial velocity to arc toward robot_pos.

        Args:
            robot_pos: Current robot base position (x, y, z).
            mj_data: MuJoCo data to write qpos/qvel into.

        Returns:
            Slot index of the launched ball, or None if launch failed.
        """
        slot_idx = self._next_slot
        self._next_slot = (self._next_slot + 1) % self.max_balls
        slot = self.slots[slot_idx]

        # Deactivate if it was previously active
        if slot.active:
            self._deactivate_slot(slot, mj_data)

        # Random launch origin: random compass angle, configurable distance and height
        rng = np.random.default_rng()
        angle = rng.uniform(0, 2 * np.pi)
        distance = rng.uniform(*self.launch_distance_range)
        height = rng.uniform(*self.launch_height_range)

        launch_pos = np.array([
            robot_pos[0] + distance * np.cos(angle),
            robot_pos[1] + distance * np.sin(angle),
            height,
        ])

        # Compute initial velocity to reach robot_pos in a lobbed arc
        # Using projectile kinematics: choose flight_time, solve for v0
        speed = rng.uniform(*self.launch_speed_range)
        delta = robot_pos[:2] - launch_pos[:2]
        horizontal_dist = np.linalg.norm(delta)

        # Flight time from horizontal distance and speed
        flight_time = horizontal_dist / max(speed, 0.1)
        flight_time = max(flight_time, 0.5)  # At least 0.5s flight

        # Horizontal velocity
        vx = delta[0] / flight_time
        vy = delta[1] / flight_time

        # Vertical velocity: solve z_target = z_launch + vz*t - 0.5*g*t^2
        # vz = (z_target - z_launch + 0.5*g*t^2) / t
        g = 9.81
        target_z = robot_pos[2]  # Aim at robot height
        vz = (target_z - launch_pos[2] + 0.5 * g * flight_time**2) / flight_time

        v0 = np.array([vx, vy, vz])

        # Set freejoint state in mj_data
        qa = slot.qpos_adr
        slot.p0 = launch_pos.copy()
        slot.v0 = v0.copy()
        slot.launch_time = time.time()
        slot.landed_time = None
        slot.active = True

        # Freejoint qpos: [x, y, z, qw, qx, qy, qz]
        mj_data.qpos[qa:qa + 3] = launch_pos
        mj_data.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]  # Unit quaternion

        # Freejoint qvel: [vx, vy, vz, wx, wy, wz]
        va = slot.qvel_adr
        mj_data.qvel[va:va + 3] = v0
        mj_data.qvel[va + 3:va + 6] = 0.0  # No angular velocity

        return slot_idx

    def tick(self, mj_data: mujoco.MjData, current_time: float):
        """Update ball lifecycle: detect landing and auto-deactivate old balls.

        Called every sim step from the simulation thread.

        Args:
            mj_data: Current MuJoCo data.
            current_time: Current wall-clock time (time.time()).
        """
        for slot in self.slots:
            if not slot.active:
                continue

            # Read current ball position from mj_data
            ball_z = mj_data.qpos[slot.qpos_adr + 2]

            # Detect landing: ball is near ground level
            if ball_z < self.ground_z_threshold and slot.landed_time is None:
                slot.landed_time = current_time

            # Auto-deactivate after ball_lifetime seconds on the ground
            if slot.landed_time is not None:
                if current_time - slot.landed_time > self.ball_lifetime:
                    self._deactivate_slot(slot, mj_data)

    def get_active_ball_states(self) -> List[dict]:
        """Return state of all active balls.

        Returns:
            List of dicts with keys: slot_idx, p0, v0, launch_time, radius, active.
        """
        states = []
        for i, slot in enumerate(self.slots):
            states.append({
                "slot_idx": i,
                "p0": slot.p0.copy(),
                "v0": slot.v0.copy(),
                "launch_time": slot.launch_time,
                "radius": slot.radius,
                "active": slot.active,
            })
        return states

    def _deactivate_slot(self, slot: BallSlot, mj_data: mujoco.MjData):
        """Move a ball back underground and mark inactive."""
        qa = slot.qpos_adr
        va = slot.qvel_adr

        mj_data.qpos[qa:qa + 3] = [0.0, 0.0, -10.0]
        mj_data.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
        mj_data.qvel[va:va + 6] = 0.0

        slot.active = False
        slot.landed_time = None
