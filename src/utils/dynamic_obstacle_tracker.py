"""
DynamicObstacleTracker: Source-agnostic trajectory forecaster.

Produces pre-computed predicted positions at the planner's rollout timesteps.
The planner receives a fixed-size (max_obstacles, H, 3) tensor and evaluates
cost — it never does any physics or prediction.

Two update paths (same output format):
  - update_from_sim(ball_launcher, mj_data): reads live sim state
  - update_from_mocap(observations): accepts observed positions+velocities from
    motion capture, fits trajectory (stub for hardware implementation)
"""

import time
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from utils.ball_launcher import BallLauncher


class DynamicObstacleTracker:
    """Forecasts dynamic obstacle trajectories for the MPS planner.

    Outputs a fixed-size tensor of predicted positions at each rollout
    timestep, plus active mask and radii. This tensor is passed to the
    planner which uses it as a pure lookup for cost evaluation.
    """

    GRAVITY = np.array([0.0, 0.0, -9.81])

    def __init__(
        self,
        max_obstacles: int = 5,
        horizon: int = 10,
        rollout_dt: float = 0.3,
        ball_radius: float = 0.05,
    ):
        """
        Args:
            max_obstacles: Max number of simultaneous dynamic obstacles (fixed for JIT).
            horizon: Number of rollout steps in the planner.
            rollout_dt: Time between rollout steps (command_duration).
            ball_radius: Default ball radius.
        """
        self.max_obstacles = max_obstacles
        self.horizon = horizon
        self.rollout_dt = rollout_dt
        self.default_radius = ball_radius

        # Internal state: current observed obstacle state.
        self._position = np.zeros((max_obstacles, 3), dtype=np.float32)
        self._velocity = np.zeros((max_obstacles, 3), dtype=np.float32)
        self._observed_time = np.zeros(max_obstacles, dtype=np.float32)
        self._active = np.zeros(max_obstacles, dtype=bool)
        self._radius = np.full(max_obstacles, ball_radius, dtype=np.float32)

    def update_from_sim(
        self,
        ball_launcher: "BallLauncher",
        mj_data,
        current_time: Optional[float] = None,
    ):
        """Sync internal state from live sim state.

        Args:
            ball_launcher: The sim's BallLauncher instance.
            mj_data: Live MuJoCo data holding the current ball state.
        """
        snapshot = ball_launcher.get_obstacle_snapshot(mj_data)
        self.update_from_observations(
            {
                "positions": snapshot[:, 0:3],
                "velocities": snapshot[:, 3:6],
                "radii": snapshot[:, 6],
                "active": snapshot[:, 7] > 0.5,
            },
            current_time=current_time,
        )

    def update_from_observations(
        self,
        observations: dict,
        current_time: Optional[float] = None,
    ):
        """Update tracker from observed obstacle position and velocity."""
        if current_time is None:
            current_time = time.time()

        positions = np.asarray(observations.get("positions", []), dtype=np.float32)
        velocities = np.asarray(observations.get("velocities", []), dtype=np.float32)
        radii = np.asarray(observations.get("radii", []), dtype=np.float32)
        active = np.asarray(observations.get("active", []), dtype=bool)

        self._position.fill(0.0)
        self._velocity.fill(0.0)
        self._observed_time.fill(0.0)
        self._active[:] = False
        self._radius.fill(self.default_radius)

        num_obstacles = min(len(positions), self.max_obstacles)
        if num_obstacles <= 0:
            return

        for i in range(num_obstacles):
            radius = (
                float(radii[i])
                if radii.ndim > 0 and i < len(radii)
                else float(self.default_radius)
            )
            self._radius[i] = radius
            if not active[i]:
                continue

            self._position[i] = positions[i]
            self._velocity[i] = velocities[i]
            self._observed_time[i] = float(current_time)
            self._active[i] = True

    def update_from_mocap(self, observations: dict):
        """Update from motion capture observations (hardware path).

        Expected format:
            observations = {
                "positions": np.ndarray (N, 3),  # current observed positions
                "velocities": np.ndarray (N, 3), # current observed velocities
                "radii": np.ndarray (N,),         # observed radii
                "active": np.ndarray (N,),        # active flags
            }

        For now, we treat the observation as instantaneous state and use
        ballistic prediction from that point forward (constant gravity,
        no drag). A Kalman filter or polynomial fit could improve this.
        """
        self.update_from_observations(observations, current_time=time.time())

    def get_predicted_positions(
        self,
        current_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute predicted obstacle positions at each rollout timestep.

        For each active obstacle, predicts position at:
            t_i = current_time + (i + 1) * rollout_dt,  for i in 0..horizon-1

        Uses ballistic equation:
            p(t) = p_obs + v_obs * dt + 0.5 * g * dt^2
        where dt is measured from the latest observation time.

        Inactive obstacles get zero positions (masked out by active flags).

        Args:
            current_time: Current wall-clock time (same clock as observation time).

        Returns:
            positions: (max_obstacles, horizon, 3) predicted positions
            active: (max_obstacles,) boolean mask
            radii: (max_obstacles,) ball radii
        """
        if current_time is None:
            current_time = time.time()

        positions = np.zeros((self.max_obstacles, self.horizon, 3), dtype=np.float32)

        for i in range(self.max_obstacles):
            if not self._active[i]:
                continue

            lag = max(float(current_time) - float(self._observed_time[i]), 0.0)
            for h in range(self.horizon):
                dt = lag + (h + 1) * self.rollout_dt

                # Ballistic trajectory: p(t) = p0 + v0*dt + 0.5*g*dt^2
                pos = self._position[i] + self._velocity[i] * dt + 0.5 * self.GRAVITY * dt**2

                # Clamp to ground (ball doesn't go underground)
                pos[2] = max(pos[2], self._radius[i])

                positions[i, h] = pos

        return positions, self._active.copy(), self._radius.copy()

    def has_active_obstacles(self) -> bool:
        """Check if any obstacles are currently active."""
        return np.any(self._active)
