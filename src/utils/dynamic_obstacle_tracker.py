"""
DynamicObstacleTracker: Source-agnostic trajectory forecaster.

Produces pre-computed predicted positions at the planner's rollout timesteps.
The planner receives a fixed-size (max_obstacles, H, 3) tensor and evaluates
cost — it never does any physics or prediction.

Two update paths (same output format):
  - update_from_sim(ball_launcher): reads BallLauncher state, uses ballistic eq
  - update_from_mocap(observations): accepts observed positions+velocities from
    motion capture, fits trajectory (stub for hardware implementation)
"""

import time
from typing import Optional, Tuple

import numpy as np

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

        # Internal state: per-obstacle launch params (for sim / ballistic model)
        self._p0 = np.zeros((max_obstacles, 3))
        self._v0 = np.zeros((max_obstacles, 3))
        self._launch_time = np.zeros(max_obstacles)
        self._active = np.zeros(max_obstacles, dtype=bool)
        self._radius = np.full(max_obstacles, ball_radius)

    def update_from_sim(self, ball_launcher: BallLauncher):
        """Sync internal state from BallLauncher (sim path).

        Args:
            ball_launcher: The sim's BallLauncher instance.
        """
        states = ball_launcher.get_active_ball_states()
        for i, s in enumerate(states):
            if i >= self.max_obstacles:
                break
            self._p0[i] = s["p0"]
            self._v0[i] = s["v0"]
            self._launch_time[i] = s["launch_time"]
            self._active[i] = s["active"]
            self._radius[i] = s["radius"]

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
        n = min(len(observations.get("positions", [])), self.max_obstacles)

        # Reset all slots
        self._active[:] = False

        for i in range(n):
            if not observations["active"][i]:
                continue
            self._p0[i] = observations["positions"][i]
            self._v0[i] = observations["velocities"][i]
            self._launch_time[i] = time.time()  # treat "now" as launch time
            self._active[i] = True
            if "radii" in observations:
                self._radius[i] = observations["radii"][i]

    def get_predicted_positions(
        self, current_time: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute predicted obstacle positions at each rollout timestep.

        For each active obstacle, predicts position at:
            t_i = current_time + i * rollout_dt,  for i in 0..horizon-1

        Uses ballistic equation:
            p(t) = p0 + v0 * dt + 0.5 * g * dt^2
        where dt = t_i - launch_time.

        Inactive obstacles get zero positions (masked out by active flags).

        Args:
            current_time: Current wall-clock time (same clock as launch_time).

        Returns:
            positions: (max_obstacles, horizon, 3) predicted positions
            active: (max_obstacles,) boolean mask
            radii: (max_obstacles,) ball radii
        """
        positions = np.zeros((self.max_obstacles, self.horizon, 3))

        for i in range(self.max_obstacles):
            if not self._active[i]:
                continue

            for h in range(self.horizon):
                t = current_time + h * self.rollout_dt
                dt = t - self._launch_time[i]
                if dt < 0:
                    dt = 0.0

                # Ballistic trajectory: p(t) = p0 + v0*dt + 0.5*g*dt^2
                pos = self._p0[i] + self._v0[i] * dt + 0.5 * self.GRAVITY * dt**2

                # Clamp to ground (ball doesn't go underground)
                pos[2] = max(pos[2], self._radius[i])

                positions[i, h] = pos

        return positions, self._active.copy(), self._radius.copy()

    def has_active_obstacles(self) -> bool:
        """Check if any obstacles are currently active."""
        return np.any(self._active)
