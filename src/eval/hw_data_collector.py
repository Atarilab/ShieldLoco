#!/usr/bin/env python3
"""
hw_data_collector.py — Data collection for DoomMPC safety-filter experiments.

Supports both hardware (Vicon) and simulation modes.

Hardware mode (default):
    Subscribes to Vicon Position + /mps_optimizer_metrics.
    Vicon provides ground-truth base pose.

Simulation mode (--sim):
    No Vicon — uses controller's own base position estimate from the
    /mps_optimizer_metrics message (indices 9-11: base_x, base_y, base_z).
    No heading/quaternion data available.

Saves controller config, scene XMLs, and optionally records a rosbag.

Usage:
    # Hardware:
    python3 eval/hw_data_collector.py --label static_nav
    python3 eval/hw_data_collector.py --label dynamic_obs --rosbag

    # Simulation (testing):
    python3 eval/hw_data_collector.py --sim --task rl-mps-eval-go2 --label static_nav
    python3 eval/hw_data_collector.py --sim --label static_nav --rosbag

Output directory:  {out}/{alg}_{interface}_{label}_{timestamp}/
"""

import argparse
import csv
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# Vicon import is hw-only — late-import inside node when mode != "sim"
# from vicon_receiver.msg import Position

# ---------------------------------------------------------------------------
# Paths relative to this script (src/eval/)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_SCRIPT_DIR)
_GO2_DIR = os.path.join(_SRC_DIR, "robots", "go2")
_TASK_CONFIG_PATH = os.path.join(_SRC_DIR, "tasks", "task_configs.py")

# XMLs to snapshot
_XML_FILES = [
    "scene.xml",
    "scene_mjx.xml",
    "go2.xml",
    "go2_mjx.xml",
    "obstacles_hw.xml",
    "balls.xml",
    "obstacles.xml",
]

# Optimizer message layout (12 floats from controller)
#  [0-3]: tracking_cost, optimizer_cost, opt_elapsed_ms, mpc_active
#  [4]:   plan_seq
#  [5-7]: ctrl_vx, ctrl_vy, ctrl_height
#  [8]:   ctrl_timestamp
#  [9-11]: base_x, base_y, base_z (controller's world-frame position estimate)
_OPT_IDX = {
    "tracking_cost": 0,
    "optimizer_cost": 1,
    "opt_elapsed_ms": 2,
    "mpc_active": 3,
    "plan_seq": 4,
    "ctrl_vx": 5,
    "ctrl_vy": 6,
    "ctrl_height": 7,
    "ctrl_timestamp": 8,
    "base_x": 9,
    "base_y": 10,
    "base_z": 11,
}

CSV_FIELDNAMES = [
    "t",
    "x", "y", "z",
    "qw", "qx", "qy", "qz",
    "heading",
    "tracking_cost",
    "optimizer_cost",
    "opt_elapsed_ms",
    "mpc_active",
    "plan_seq",
    "ctrl_vx", "ctrl_vy", "ctrl_height",
    "frame_number",
]


def _load_task_config(task_name: str) -> Dict[str, str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("task_configs", _TASK_CONFIG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    cfg = mod.TASK_CONFIG.get(task_name)
    if cfg is None:
        print(f"[collector] ERROR: task '{task_name}' not found in TASK_CONFIG")
        sys.exit(1)
    return cfg


def _resolve_out_dir(
    out_parent: str,
    controller_cfg_path: str,
    robot_iface_cfg_path: str,
    label: str,
) -> str:
    with open(controller_cfg_path) as f:
        ctrl = yaml.safe_load(f)
    alg = ctrl.get("mps_alg_name", "MIMPS")

    with open(robot_iface_cfg_path) as f:
        riface = yaml.safe_load(f)
    interface_raw = (riface.get("INTERFACE", "") or "").lower()
    interface = "hw" if interface_raw == "real" else "sim"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{alg}_{interface}_{label}_{ts}"
    return os.path.join(out_parent, name)


def _save_configs_xmls(out_dir: str, controller_cfg_path: str, robot_iface_cfg_path: str):
    cfg_dir = os.path.join(out_dir, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    shutil.copy2(controller_cfg_path, os.path.join(cfg_dir, "controller_cfg.yaml"))
    shutil.copy2(robot_iface_cfg_path, os.path.join(cfg_dir, "robot_interface_cfg.yaml"))

    xml_dir = os.path.join(out_dir, "xmls")
    os.makedirs(xml_dir, exist_ok=True)
    for fname in _XML_FILES:
        src = os.path.join(_GO2_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(xml_dir, fname))


def _write_meta(out_dir: str, label: str, task: str, mode: str, start_time_epoch: float):
    meta = {
        "experiment_label": label,
        "task": task,
        "mode": mode,
        "start_time_epoch": start_time_epoch,
        "start_time_iso": datetime.fromtimestamp(start_time_epoch).isoformat(),
    }
    with open(os.path.join(out_dir, "experiment_config.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _get_float(d: list, idx: int, default: float = 0.0) -> float:
    return float(d[idx]) if len(d) > idx else default


def _get_int(d: list, idx: int, default: int = 0) -> int:
    return int(d[idx]) if len(d) > idx else default


# ---------------------------------------------------------------------------
# ROS2 collector node
# ---------------------------------------------------------------------------

class DataCollector(Node):
    def __init__(
        self,
        out_dir: str,
        mode: str,
        vicon_topic: str = "/vicon/Go2/root",
        start_rosbag: bool = False,
        rosbag_topics: Optional[List[str]] = None,
    ):
        super().__init__("data_collector")
        self._mode = mode
        self._out_dir = out_dir
        self._start_time: Optional[float] = None
        self._csv_file = open(os.path.join(out_dir, "eval_data.csv"), "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDNAMES)
        self._csv_writer.writeheader()
        self._csv_step = 0

        self._latest_opt_msg: Optional[Float32MultiArray] = None

        # Hardware mode: Vicon subscription
        self._vicon_sub = None
        if mode == "hw":
            from vicon_receiver.msg import Position
            self._latest_vicon_msg: Any = None
            self._vicon_sub = self.create_subscription(
                Position, vicon_topic, self._vicon_callback, 10,
            )

        # Optimizer metrics subscription (both modes)
        self._opt_sub = self.create_subscription(
            Float32MultiArray, "/mps_optimizer_metrics", self._opt_callback, 10,
        )

        # Rosbag
        self._bag_proc: Optional[subprocess.Popen] = None
        if start_rosbag:
            default_topics = (
                ["/tf", "/mps_optimizer_metrics"]
                if mode == "sim"
                else ["/vicon/Go2/root", "/mps_optimizer_metrics", "/tf"]
            )
            self._start_rosbag(rosbag_topics or default_topics)

        self.get_logger().info(f"Collector [{mode}] started. Output: {out_dir}")
        if mode == "hw":
            self.get_logger().info(f"  Vicon topic: {vicon_topic}")
        if start_rosbag:
            self.get_logger().info(f"  Rosbag: recording to {out_dir}/rosbag/")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _opt_callback(self, msg: Float32MultiArray):
        self._latest_opt_msg = msg
        self._try_write_row()

    def _vicon_callback(self, msg):
        self._latest_vicon_msg = msg
        if self._start_time is None:
            self._start_time = time.time()
        self._try_write_row()

    # ------------------------------------------------------------------
    # Row assembly
    # ------------------------------------------------------------------

    def _try_write_row(self):
        """Write a CSV row. In sim mode we write on every opt message.
        In hw mode we wait until both Vicon and opt data are available."""
        if self._latest_opt_msg is None:
            return

        # Latch start_time on first write
        if self._start_time is None:
            self._start_time = time.time()

        opt = self._latest_opt_msg
        d = opt.data
        t = time.time() - self._start_time

        if self._mode == "hw":
            vicon = getattr(self, "_latest_vicon_msg", None)
            if vicon is None:
                return
            x = vicon.x_trans * 0.001
            y = vicon.y_trans * 0.001
            z = vicon.z_trans * 0.001
            qx = vicon.x_rot
            qy = vicon.y_rot
            qz = vicon.z_rot
            qw = vicon.w
            frame_number = vicon.frame_number
            heading = float(math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            ))
        else:
            # Sim mode: use controller's own base position estimate
            x = _get_float(d, _OPT_IDX["base_x"])
            y = _get_float(d, _OPT_IDX["base_y"])
            z = _get_float(d, _OPT_IDX["base_z"])
            qx = 0.0
            qy = 0.0
            qz = 0.0
            qw = 1.0
            heading = 0.0
            frame_number = -1

        tracking_cost = _get_float(d, _OPT_IDX["tracking_cost"])
        optimizer_cost = _get_float(d, _OPT_IDX["optimizer_cost"])
        opt_elapsed_ms = _get_float(d, _OPT_IDX["opt_elapsed_ms"])
        mpc_active = _get_int(d, _OPT_IDX["mpc_active"])
        plan_seq = _get_int(d, _OPT_IDX["plan_seq"], -1)
        ctrl_vx = _get_float(d, _OPT_IDX["ctrl_vx"])
        ctrl_vy = _get_float(d, _OPT_IDX["ctrl_vy"])
        ctrl_height = _get_float(d, _OPT_IDX["ctrl_height"])

        self._csv_writer.writerow({
            "t": f"{t:.4f}",
            "x": f"{x:.4f}",
            "y": f"{y:.4f}",
            "z": f"{z:.4f}",
            "qw": f"{qw:.6f}",
            "qx": f"{qx:.6f}",
            "qy": f"{qy:.6f}",
            "qz": f"{qz:.6f}",
            "heading": f"{heading:.4f}",
            "tracking_cost": f"{tracking_cost:.6f}",
            "optimizer_cost": f"{optimizer_cost:.6f}",
            "opt_elapsed_ms": f"{opt_elapsed_ms:.2f}",
            "mpc_active": mpc_active,
            "plan_seq": plan_seq,
            "ctrl_vx": f"{ctrl_vx:.4f}",
            "ctrl_vy": f"{ctrl_vy:.4f}",
            "ctrl_height": f"{ctrl_height:.4f}",
            "frame_number": frame_number,
        })

        self._csv_step += 1
        if self._csv_step % 200 == 0:
            self._csv_file.flush()

        self._latest_opt_msg = None
        if self._mode == "hw":
            self._latest_vicon_msg = None

    # ------------------------------------------------------------------
    # Rosbag
    # ------------------------------------------------------------------

    def _start_rosbag(self, topics: List[str]):
        bag_dir = os.path.join(self._out_dir, "rosbag")
        os.makedirs(bag_dir, exist_ok=True)
        bag_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cmd = ["ros2", "bag", "record", "-o", os.path.join(bag_dir, bag_name)] + topics
        self._bag_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.get_logger().info(f"Rosbag started (pid={self._bag_proc.pid})")

    def _stop_rosbag(self):
        if self._bag_proc is None:
            return
        try:
            self._bag_proc.send_signal(signal.SIGINT)
            self._bag_proc.wait(timeout=1)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            pass
        if self._bag_proc is not None and self._bag_proc.poll() is None:
            try:
                self._bag_proc.kill()
                self._bag_proc.wait(timeout=1)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                pass
        self._bag_proc = None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self):
        self._stop_rosbag()
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()
        self.get_logger().info(f"Collector stopped. Data → {self._out_dir}")

    def __del__(self):
        self.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DoomMPC data collector (sim + hw)")
    parser.add_argument("--label", required=True, help="Experiment label (e.g. static_nav)")
    parser.add_argument(
        "--task", default="rl-mps-real-go2",
        help="Task name for resolving configs (default: rl-mps-real-go2)",
    )
    parser.add_argument("--out", default="results/hw_eval", help="Parent output directory")
    parser.add_argument(
        "--sim", action="store_true",
        help="Simulation mode: skip Vicon, use controller state for position",
    )
    parser.add_argument(
        "--vicon-topic", default="/vicon/Go2/root",
        help="Vicon Position topic (hw mode only, default: /vicon/Go2/root)",
    )
    parser.add_argument("--rosbag", action="store_true", help="Record rosbag in parallel")
    parser.add_argument(
        "--rosbag-topics", nargs="+", default=None,
        help="Topics for rosbag (default depends on --sim flag)",
    )
    args = parser.parse_args()

    if not args.sim and args.task == "rl-mps-real-go2":
        args.task = "rl-mps-real-go2"

    # Resolve task configs
    task_cfg = _load_task_config(args.task)
    controller_cfg_path = os.path.join(_SRC_DIR, task_cfg["controller"])
    robot_iface_cfg_path = os.path.join(_SRC_DIR, task_cfg["robot_interface"])

    if not os.path.exists(controller_cfg_path):
        print(f"[collector] ERROR: controller config not found: {controller_cfg_path}")
        sys.exit(1)
    if not os.path.exists(robot_iface_cfg_path):
        print(f"[collector] ERROR: robot interface config not found: {robot_iface_cfg_path}")
        sys.exit(1)

    mode = "sim" if args.sim else "hw"

    out_dir = _resolve_out_dir(args.out, controller_cfg_path, robot_iface_cfg_path, args.label)
    os.makedirs(out_dir, exist_ok=True)

    _save_configs_xmls(out_dir, controller_cfg_path, robot_iface_cfg_path)
    _write_meta(out_dir, args.label, args.task, mode, time.time())

    print(f"[collector] Output:  {out_dir}")
    print(f"[collector] Mode:    {mode}")
    print(f"[collector] Label:   {args.label}")
    print(f"[collector] Task:    {args.task}")
    if mode == "hw":
        print(f"[collector] Vicon:   {args.vicon_topic}")
    if args.rosbag:
        print(f"[collector] Rosbag:  recording")

    rclpy.init(args=sys.argv)
    node = DataCollector(
        out_dir=out_dir,
        mode=mode,
        vicon_topic=args.vicon_topic,
        start_rosbag=args.rosbag,
        rosbag_topics=args.rosbag_topics,
    )

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
