"""
SimEvalHooks — sim-side eval logic as a single callback object.

Plugged into SimRobotInterface via the eval_hooks parameter.  The sim calls
on_step() once per mj_step.  No eval-specific state lives inside
SimRobotInterface itself.

Responsibilities:
  - Log base position, collision flag, fall flag, goal distance, ball distance
    every step to a CSV.
  - Auto-throw a ball at a configurable time (scenario B).
  - Set stop_event when the episode ends (goal reached or timeout).
"""
import csv
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import mujoco
import numpy as np


@dataclass
class SimEvalConfig:
    scenario: str = "static_nav"           # "static_nav" | "dynamic_obs"
    goal_position: List[float] = field(default_factory=lambda: [6.5, 0.0])
    goal_threshold: float = 0.5            # metres (xy only)
    max_time: float = 120.0                # seconds after eval start
    ball_throw_delay: float = 5.0          # seconds after eval start (scenario B)
    settle_time: float = 5.0               # seconds to wait after resets before starting timer
    sim_metric_file: str = "logs/eval/sim.csv"
    controller_metric_file: str = "logs/eval/ctrl.csv"
    done_file: str = "logs/eval/eval_done.txt"  # written by controller when eval is complete

    @classmethod
    def from_dict(cls, d: dict) -> "SimEvalConfig":
        return cls(
            scenario=d.get("scenario", "static_nav"),
            goal_position=d.get("goal_position", [6.5, 0.0]),
            goal_threshold=float(d.get("goal_threshold", 0.5)),
            max_time=float(d.get("max_time", 120.0)),
            ball_throw_delay=float(d.get("ball_throw_delay", 5.0)),
            settle_time=float(d.get("settle_time", 3.0)),
            sim_metric_file=d.get("sim_metric_file", "logs/eval/sim.csv"),
            controller_metric_file=d.get("controller_metric_file", "logs/eval/ctrl.csv"),
            done_file=d.get("done_file", "logs/eval/eval_done.txt"),
        )


class SimEvalHooks:
    """
    Eval callback object for SimRobotInterface.

    Construction should happen *after* the SimRobotInterface has initialised
    its mj_model / mj_data / ball_launcher / stop_event fields.
    """

    def __init__(
        self,
        config: SimEvalConfig,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        ball_launcher,           # utils.ball_launcher.BallLauncher
        stop_event,              # threading.Event
    ):
        self.cfg = config
        self.mj_data = mj_data
        self.ball_launcher = ball_launcher
        self.stop_event = stop_event
        # Phase tracking:
        #   start_time is None  → waiting for ctrl CSV (JIT not done yet)
        #   _reset_at is set    → resets firing during settle_time
        #   start_time is set   → eval running, timer active
        self.start_time: Optional[float] = None
        self._reset_at: Optional[float] = None   # wall time when ctrl CSV was first detected
        self._ball_thrown = False
        self._done = False

        # Pre-cache geom categorisation (same logic as QuadRLTask.__init__)
        self._setup_geom_masks(mj_model)
        self._base_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        import os
        os.makedirs(os.path.dirname(config.sim_metric_file) or ".", exist_ok=True)
        # Delete stale signals from previous runs so they don't trigger
        # an immediate false-start or early termination.
        for stale in (config.controller_metric_file, config.done_file):
            try:
                os.remove(stale)
            except FileNotFoundError:
                pass

        self._file = open(config.sim_metric_file, "w", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=["t", "x", "y", "z", "vx", "vy", "heading_error",
                        "collision", "foot_collision", "fall", "goal_dist", "ball_dist"],
        )
        self._writer.writeheader()
        self._step = 0

    # ------------------------------------------------------------------
    # Main callback — called once per mj_step inside the sim thread
    # ------------------------------------------------------------------

    def on_step(self):
        if self._done:
            return

        # Phase 1: wait for ctrl CSV (JIT done, eval controller auto-started).
        # The sim itself handles the reset (via reset_signal.txt in sim_robot_interface).
        if self._reset_at is None:
            if not os.path.exists(self.cfg.controller_metric_file):
                return  # JIT still compiling — skip
            self._reset_at = time.time()
            return

        # Phase 2: wait settle_time after the reset before starting the timer.
        if self.start_time is None:
            if time.time() - self._reset_at < self.cfg.settle_time:
                return
            self.start_time = time.time()

        t = time.time() - self.start_time
        base_pos = self.mj_data.qpos[:3].copy()
        base_vel = self.mj_data.qvel[:2].copy()  # world-frame vx, vy

        collision = int(self._check_collision())
        foot_collision = int(self._check_foot_collision())
        fall = int(self._check_fall())
        goal_pos = np.array(self.cfg.goal_position[:2])
        goal_dist = float(np.linalg.norm(base_pos[:2] - goal_pos))
        ball_dist = self._min_ball_distance(base_pos)

        # Heading error: angle between robot yaw and direction to goal [rad, in (-pi, pi)]
        quat = self.mj_data.qpos[3:7]  # [w, x, y, z]
        yaw = float(np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]),
                               1 - 2*(quat[2]**2 + quat[3]**2)))
        goal_angle = float(np.arctan2(goal_pos[1] - base_pos[1], goal_pos[0] - base_pos[0]))
        heading_err = float((goal_angle - yaw + np.pi) % (2*np.pi) - np.pi)

        self._writer.writerow({
            "t": f"{t:.4f}",
            "x": f"{base_pos[0]:.4f}",
            "y": f"{base_pos[1]:.4f}",
            "z": f"{base_pos[2]:.4f}",
            "vx": f"{base_vel[0]:.4f}",
            "vy": f"{base_vel[1]:.4f}",
            "heading_error": f"{heading_err:.4f}",
            "collision": collision,
            "foot_collision": foot_collision,
            "fall": fall,
            "goal_dist": f"{goal_dist:.4f}",
            "ball_dist": f"{ball_dist:.4f}",
        })

        # Flush every ~200 steps (~1 s at 200 Hz)
        self._step += 1
        if self._step % 200 == 0:
            self._file.flush()

        # Scenario B: auto-throw ball
        if self.cfg.scenario == "dynamic_obs" and not self._ball_thrown:
            if t > self.cfg.ball_throw_delay:
                robot_pos = self.mj_data.xpos[self._base_body_id].copy()
                self.ball_launcher.launch(robot_pos, self.mj_data)
                self._ball_thrown = True

        # Episode termination:
        # - goal_reached: write done_file immediately so the ROS node also stops fast
        # - timed_out: controller writes done_file after max_time, we pick it up here
        goal_reached = goal_dist < self.cfg.goal_threshold
        ros_done = os.path.exists(self.cfg.done_file)
        if goal_reached and not ros_done:
            with open(self.cfg.done_file, "w") as _f:
                _f.write(f"goal_reached:{t:.2f}\n")
        if goal_reached or ros_done:
            self._close()
            self.stop_event.set()

    # ------------------------------------------------------------------
    # Geom masks — mirroring QuadRLTask.__init__ (no JAX needed here)
    # ------------------------------------------------------------------

    def _setup_geom_masks(self, mj_model: mujoco.MjModel):
        ngeom = mj_model.ngeom
        geom_is_hazard = np.zeros(ngeom, dtype=bool)
        geom_is_robot = np.zeros(ngeom, dtype=bool)
        geom_is_foot = np.zeros(ngeom, dtype=bool)

        hazard_prefixes = ("obstacle_",)
        base_link_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        # Foot contact spheres are on the calf bodies (e.g. FL_calf), not the
        # foot bodies (FL_foot) which are kinematic endpoints with no geoms.
        foot_body_ids = {
            bid for bid in range(mj_model.nbody)
            if (mj_model.body(bid).name or "").endswith("_calf")
        }

        for gid in range(ngeom):
            geom_name = mj_model.geom(gid).name or ""
            if any(geom_name.startswith(p) for p in hazard_prefixes):
                geom_is_hazard[gid] = True
            body_id = mj_model.geom_bodyid[gid]
            if mj_model.body_rootid[body_id] == base_link_id:
                geom_is_robot[gid] = True
            if body_id in foot_body_ids:
                geom_is_foot[gid] = True

        self._geom_is_hazard = geom_is_hazard
        self._geom_is_robot = geom_is_robot
        self._geom_is_foot = geom_is_foot

    # ------------------------------------------------------------------
    # Constraint checks (NumPy / pure MuJoCo — no JAX)
    # ------------------------------------------------------------------

    def _check_collision(self) -> bool:
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (self._geom_is_robot[g1] and self._geom_is_hazard[g2]) or \
               (self._geom_is_robot[g2] and self._geom_is_hazard[g1]):
                return True
        return False

    def _check_foot_collision(self) -> bool:
        """True if a foot geom is in contact with an obstacle geom.

        MuJoCo's constraint solver prevents real penetration (c.dist ≈ 0 for all
        resolved contacts), so depth thresholds are ineffective. Filtering to foot
        geoms only is sufficient — they are small spheres at the leg tips and rarely
        graze obstacles during swing unless there is actual impact.
        """
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (self._geom_is_foot[g1] and self._geom_is_hazard[g2]) or \
               (self._geom_is_foot[g2] and self._geom_is_hazard[g1]):
                return True
        return False

    def _check_fall(self) -> bool:
        # xquat is stored as (w, x, y, z) in MuJoCo
        q = self.mj_data.xquat[self._base_body_id]
        w, x, y, z = q[0], q[1], q[2], q[3]
        # z-component of the rotated up-vector [0,0,1]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        return up_z < 0.5

    def _min_ball_distance(self, base_pos: np.ndarray) -> float:
        min_dist = -1.0
        for slot in self.ball_launcher.slots:
            if slot.active:
                ball_pos = self.mj_data.qpos[slot.qpos_adr: slot.qpos_adr + 3]
                dist = float(np.linalg.norm(base_pos - ball_pos))
                if min_dist < 0 or dist < min_dist:
                    min_dist = dist
        return min_dist

    # ------------------------------------------------------------------

    def _close(self):
        if not self._done:
            self._done = True
            self._file.flush()
            self._file.close()

    def __del__(self):
        self._close()
