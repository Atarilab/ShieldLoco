"""
DynamicObstacleTracker: Source-agnostic obstacle state tracker.

Maintains per-obstacle state (position, velocity at observation time) and
exposes it for the MPC to use for on-the-fly ballistic prediction.

The MPC receives (p0, v0, lag) and computes p(t) = p0 + v0*t + 0.5*g*t^2
analytically at any rollout step — no horizon limit, no precomputed arrays.

Two update paths (same output format):
  - update_from_sim(ball_launcher, mj_data): reads live sim state
  - update_from_mocap(observations): accepts observed positions+velocities from
    motion capture (stub for hardware implementation)
"""

import time
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from utils.ball_launcher import BallLauncher


class DynamicObstacleTracker:
    """Tracks dynamic obstacle state for the MPS planner.

    Stores per-obstacle (position, velocity) at the time of last observation.
    The MPC evaluates ball position analytically at each rollout step using
    the ballistic equation — no fixed horizon required.
    """

    def __init__(
        self,
        max_obstacles: int = 5,
        ball_radius: float = 0.05,
    ):
        """
        Args:
            max_obstacles: Max number of simultaneous dynamic obstacles (fixed for JIT).
            ball_radius: Default ball radius.
        """
        self.max_obstacles = max_obstacles
        self.default_radius = ball_radius

        # Internal state: obstacle state at last observation time.
        self._position = np.zeros((max_obstacles, 3), dtype=np.float32)
        self._velocity = np.zeros((max_obstacles, 3), dtype=np.float32)
        self._observed_time = np.zeros(max_obstacles, dtype=np.float64)
        self._active = np.zeros(max_obstacles, dtype=bool)
        self._radius = np.full(max_obstacles, ball_radius, dtype=np.float32)

    def update_from_sim(
        self,
        ball_launcher: "BallLauncher",
        mj_data,
        current_time: Optional[float] = None,
    ):
        """Sync internal state from live sim state."""
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
        """Update from motion capture observations (hardware path)."""
        self.update_from_observations(observations, current_time=time.time())

    def get_obstacle_state(
        self,
        current_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return current obstacle state for on-the-fly MPC prediction.

        The MPC computes ball positions analytically at each rollout step:
            p(t) = p0 + v0 * (t0 + t_stage) + 0.5 * g * (t0 + t_stage)^2
        where t0 is the lag returned here and t_stage is the rollout step offset.

        Args:
            current_time: Current wall-clock time (same clock as observation time).

        Returns:
            p0:     (max_obstacles, 3)  position at observation time
            v0:     (max_obstacles, 3)  velocity at observation time
            t0:     (max_obstacles,)    seconds elapsed since observation (lag)
            radii:  (max_obstacles,)    ball radii
            active: (max_obstacles,)    boolean active mask
        """
        if current_time is None:
            current_time = time.time()

        t0 = np.zeros(self.max_obstacles, dtype=np.float32)
        for i in range(self.max_obstacles):
            if self._active[i]:
                t0[i] = max(float(current_time) - float(self._observed_time[i]), 0.0)

        return (
            self._position.copy(),
            self._velocity.copy(),
            t0,
            self._radius.copy(),
            self._active.copy(),
        )

    def has_active_obstacles(self) -> bool:
        """Check if any obstacles are currently active."""
        return bool(np.any(self._active))
