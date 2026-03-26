"""
Eval subclass of RLQuadrupedMPSController.

Adds automatic startup (trot + forward + MPC once JIT compilation is done) and
per-step metric logging to CSV.  All eval-specific state is encapsulated in
EvalConfig / EvalMetrics — zero new instance variables on the base class.

Usage: registered under the "rl-mps-eval-go2" task (see tasks/task_configs.py).
The master_node passes the eval YAML fields via configs["eval_config"].
"""
import csv
import time
from dataclasses import dataclass, field
from typing import IO, Optional

from controllers.rl_mps_controller import RLQuadrupedMPSController


@dataclass
class EvalConfig:
    start_gait: str = "trot"
    start_move: str = "up"                          # "up"=forward, "center"=stay
    min_plans: int = 2                              # wait for N plans before auto-starting
    max_time: float = 120.0                         # seconds; writes done_file when elapsed
    controller_metric_file: str = "logs/eval/ctrl.csv"
    done_file: str = "logs/eval/eval_done.txt"

    @classmethod
    def from_dict(cls, d: dict) -> "EvalConfig":
        return cls(
            start_gait=d.get("start_gait", "trot"),
            start_move=d.get("start_move", "up"),
            min_plans=int(d.get("min_plans", 2)),
            max_time=float(d.get("max_time", 120.0)),
            controller_metric_file=d.get("controller_metric_file", "logs/eval/ctrl.csv"),
            done_file=d.get("done_file", "logs/eval/eval_done.txt"),
        )


@dataclass
class EvalMetrics:
    started: bool = False
    start_time: Optional[float] = None
    file: Optional[IO] = None
    writer: Optional[csv.DictWriter] = None


class RLQuadrupedMPSEvalController(RLQuadrupedMPSController):
    """
    Drop-in eval variant of the MPS controller.

    After JAX JIT compilation is done (safety_manager._plan_count > 0), it
    automatically:
      1. Switches to the configured start_gait (default: trot).
      2. Enables MPC.
      3. Issues a move command (default: forward).
      4. Opens the metric CSV and logs tracking cost + plan sequence every step.
    """

    def __init__(self, robot, configs, debug=False):
        super().__init__(robot=robot, configs=configs, debug=debug)
        raw = configs.get("eval_config", {}) or {}
        self.eval_cfg = EvalConfig.from_dict(raw)
        self.eval_m = EvalMetrics()

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def compute_lowlevelcmd(self, state):
        # Auto-start once the planner has produced enough plans to be stable
        if not self.eval_m.started:
            self._eval_autostart()

        cmd = super().compute_lowlevelcmd(state)

        if self.eval_m.started:
            self._eval_log_step(state)

        return cmd

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _eval_autostart(self):
        """Issue startup commands and open the metric CSV."""
        import os
        # Use absolute path anchored to src/ so path is correct regardless of CWD.
        _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _reset_sig = os.path.join(_src, "logs", "eval", "reset_signal.txt")
        os.makedirs(os.path.dirname(self.eval_cfg.controller_metric_file) or ".", exist_ok=True)
        # Remove stale signals from any previous run
        for stale in (self.eval_cfg.done_file, _reset_sig):
            try:
                os.remove(stale)
            except FileNotFoundError:
                pass
        # Signal the sim to reset (picked up by sim_robot_interface regardless of eval mode)
        os.makedirs(os.path.dirname(_reset_sig), exist_ok=True)
        with open(_reset_sig, "w") as f:
            f.write("reset")
        if self.logger:
            self.logger.info(f"[EvalController] Reset signal written: {_reset_sig}")

        if self.safety_manager._plan_count >= self.eval_cfg.min_plans:
            move_cmd = {}
            if self.eval_cfg.start_move != "center":
                move_cmd = {"move": self.eval_cfg.start_move}

            self.change_commands({
                "gait": self.eval_cfg.start_gait,
                "mpc": "on",
                **move_cmd,
            })

            self.eval_m.started = True
            self.eval_m.start_time = time.time()
            self.eval_m.file = open(self.eval_cfg.controller_metric_file, "w", newline="")
            self.eval_m.writer = csv.DictWriter(
                self.eval_m.file,
                fieldnames=["t", "vel_x", "vel_y", "height",
                            "tracking_cost", "plan_seq", "mpc_active"],
            )
            self.eval_m.writer.writeheader()
            if self.logger:
                self.logger.info(
                    f"[EvalController] Auto-started: gait={self.eval_cfg.start_gait}, "
                    f"move={self.eval_cfg.start_move}, mpc=on"
                )

    def _eval_log_step(self, state):
        """Write one row to the controller metric CSV. Writes done_file on timeout."""
        import os, torch
        t = time.time() - self.eval_m.start_time
        if t > self.eval_cfg.max_time and not os.path.exists(self.eval_cfg.done_file):
            self.eval_m.file.flush()
            with open(self.eval_cfg.done_file, "w") as f:
                f.write(f"{t:.2f}\n")
            if self.logger:
                self.logger.info(f"[EvalController] max_time {self.eval_cfg.max_time}s elapsed — wrote done signal")

        diff = self.future_feet_positions_w - self.reference_feet_positions_w
        tracking_cost = float(torch.nan_to_num(diff ** 2, nan=0.0).sum())

        # Robot state for paper metrics
        lin_vel = state.get("robot/base_lin_vel_b", [0.0, 0.0, 0.0])
        base_pos = state.get("robot/base_pos_w", [0.0, 0.0, 0.0])

        self.eval_m.writer.writerow({
            "t": f"{t:.4f}",
            "vel_x": f"{float(lin_vel[0]):.4f}",
            "vel_y": f"{float(lin_vel[1]):.4f}",
            "height": f"{float(base_pos[2]):.4f}",
            "tracking_cost": f"{tracking_cost:.6f}",
            "plan_seq": self.safety_manager._last_result_seq,
            "mpc_active": int(self.apply_mpc_contact_plan),
        })
        # Flush periodically (every ~1 s at 200 Hz = 200 steps)
        if self.counter % 200 == 0:
            self.eval_m.file.flush()

    def __del__(self):
        if self.eval_m.file and not self.eval_m.file.closed:
            self.eval_m.file.flush()
            self.eval_m.file.close()
        try:
            super().__del__()
        except AttributeError:
            pass
