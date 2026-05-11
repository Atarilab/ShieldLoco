#!/usr/bin/env python3
"""
Generate obstacle XML and update robot initial position from Vicon data.

Subscribes to the robot's Vicon topic to capture its XY position and updates
go2.xml / go2_mjx.xml so the robot spawns at that location. Simultaneously
auto-discovers all other /vicon/* topics and writes obstacles_hw.xml.

Usage:
  python gen_obstacles_xml.py                          # defaults
  python gen_obstacles_xml.py --size 0.1 0.1 0.15      # custom box size
  python gen_obstacles_xml.py --height 0.0              # obstacles at z=0 (ground)
  python gen_obstacles_xml.py --robot-topic /vicon/Go2/root

Revert: git checkout go2.xml go2_mjx.xml obstacles_hw.xml
"""

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from vicon_receiver.msg import Position

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOOM_SRC_DIR = os.path.dirname(SCRIPT_DIR)
GO2_DIR = os.path.join(DOOM_SRC_DIR, "robots", "go2")
GO2_XML = os.path.join(GO2_DIR, "go2.xml")
GO2_MJX_XML = os.path.join(GO2_DIR, "go2_mjx.xml")
OBSTACLES_HW_XML = os.path.join(GO2_DIR, "obstacles_hw.xml")


class SceneCaptureNode(Node):
    def __init__(
        self,
        robot_topic: str,
        robot_exclude: Tuple[str, ...],
        num_samples: int,
    ):
        super().__init__("scene_capture")
        self._robot_topic = robot_topic
        self._num_samples = num_samples

        self._robot_positions: List[Tuple[float, float]] = []
        self._robot_quats: List[Tuple[float, float, float, float]] = []
        self._obstacle_positions: Dict[str, List[Tuple[float, float, float]]] = (
            defaultdict(list)
        )
        self._robot_found = False
        self._obstacle_topic_count = 0
        self._done = False

        all_topics = self._discover_topics()
        obstacle_topics = [
            t for t in all_topics if t != robot_topic and t not in robot_exclude
        ]

        if robot_topic in all_topics:
            self._robot_found = True
            self.get_logger().info(f"Robot topic found: {robot_topic}")
            self.create_subscription(
                Position,
                robot_topic,
                lambda msg: self._robot_callback(msg),
                10,
            )
        else:
            self.get_logger().warn(
                f"Robot topic '{robot_topic}' not found — robot XML will NOT be updated."
            )

        self._obstacle_topic_count = len(obstacle_topics)
        if obstacle_topics:
            self.get_logger().info(
                f"Found {self._obstacle_topic_count} obstacle topic(s): {obstacle_topics}"
            )
            for topic in obstacle_topics:
                self.create_subscription(
                    Position,
                    topic,
                    lambda msg, t=topic: self._obstacle_callback(t, msg),
                    10,
                )
        else:
            self.get_logger().warn(
                "No obstacle topics found — obstacles_hw.xml will be empty."
            )
            self._done = True

    def _discover_topics(self) -> List[str]:
        topics: List[str] = []
        for _ in range(60):
            topic_types = self.get_topic_names_and_types()
            for topic, _types in topic_types:
                if topic.startswith("/vicon/") and topic not in topics:
                    topics.append(topic)
            if topics:
                self.get_logger().info(f"Discovered {len(topics)} Vicon topic(s)")
                return topics
            time.sleep(0.5)
        return topics

    def _robot_callback(self, msg: Position):
        if len(self._robot_positions) >= self._num_samples:
            return
        self._robot_positions.append(
            (msg.x_trans * 0.001, msg.y_trans * 0.001)
        )
        self._robot_quats.append(
            (msg.x_rot, msg.y_rot, msg.z_rot, msg.w)
        )

    def _obstacle_callback(self, topic: str, msg: Position):
        if len(self._obstacle_positions[topic]) >= self._num_samples:
            return
        self._obstacle_positions[topic].append(
            (msg.x_trans * 0.001, msg.y_trans * 0.001, msg.z_trans * 0.001)
        )

    def all_done(self) -> bool:
        if self._done:
            return True
        robot_ready = (not self._robot_found) or (
            len(self._robot_positions) >= self._num_samples
        )
        obstacles_ready = (
            self._obstacle_topic_count == 0
            or all(
                len(s) >= self._num_samples
                for s in self._obstacle_positions.values()
            )
        )
        return robot_ready and obstacles_ready

    def get_robot_pose(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        if not self._robot_positions:
            return None
        arr = np.array(self._robot_positions)
        median_pos = np.median(arr, axis=0)
        if self._robot_quats:
            quat_arr = np.array(self._robot_quats)
            yaws = np.arctan2(
                2.0 * (quat_arr[:, 3] * quat_arr[:, 2] + quat_arr[:, 0] * quat_arr[:, 1]),
                1.0 - 2.0 * (quat_arr[:, 1] ** 2 + quat_arr[:, 2] ** 2),
            )
            median_yaw = float(np.median(yaws))
            qw = np.cos(median_yaw / 2.0)
            qz = np.sin(median_yaw / 2.0)
        else:
            qw, qz = 1.0, 0.0
        return (
            float(median_pos[0]),
            float(median_pos[1]),
            float(qw), 0.0, 0.0, float(qz),
        )

    def get_robot_xy(self) -> Optional[Tuple[float, float]]:
        if not self._robot_positions:
            return None
        arr = np.array(self._robot_positions)
        median = np.median(arr, axis=0)
        return (float(median[0]), float(median[1]))

    def get_obstacle_positions(self) -> Dict[str, np.ndarray]:
        result = {}
        for topic, samples in self._obstacle_positions.items():
            if not samples:
                continue
            arr = np.array(samples)
            result[topic] = np.median(arr, axis=0)
        return result


def _short_name(topic: str) -> str:
    parts = [p for p in topic.strip("/").split("/") if p]
    return parts[-1] if parts else topic.replace("/", "_")


_BASE_LINK_RE = re.compile(
    r'(<body\s+name="base_link"\s+pos=)"([^"]*)"(\s+quat="[^"]*")?'
)


def _update_robot_pose_in_xml(
    xml_path: str, x: float, y: float, qw: float, qx: float, qy: float, qz: float
):
    with open(xml_path, "r") as f:
        content = f.read()

    replacement = rf'\1"{x:.4f} {y:.4f} 0.445" quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}"'
    new_content, count = _BASE_LINK_RE.subn(replacement, content)

    if count == 0:
        print(f"WARNING: base_link body not found in {xml_path} — skipping.")
        return

    os.rename(xml_path, xml_path + ".bak")
    with open(xml_path, "w") as f:
        f.write(new_content)
    os.remove(xml_path + ".bak")
    print(
        f"Updated robot initial pose in {xml_path}: x={x:.4f} y={y:.4f} "
        f"quat=({qw:.4f} {qx:.4f} {qy:.4f} {qz:.4f})"
    )


def write_obstacles_xml(
    positions: Dict[str, np.ndarray],
    output_path: str,
    size: Tuple[float, float, float],
    height: Optional[float],
    rgba: Tuple[float, float, float, float],
):
    lines = ["<mujocoinclude>", ""]
    lines.append(
        "    <!-- Hardware obstacles auto-generated by gen_obstacles_xml.py -->"
    )
    lines.append(
        f"    <!-- {len(positions)} obstacle(s) captured at {time.strftime('%Y-%m-%d %H:%M:%S')} -->"
    )
    lines.append("")

    hx, hy, hz = size

    for i, (topic, pos) in enumerate(sorted(positions.items())):
        name = _short_name(topic)
        x, y, z_measured = float(pos[0]), float(pos[1]), float(pos[2])
        z = height if height is not None else z_measured
        r, g, b, a = rgba

        line = (
            f'    <geom name="obstacle_hw_{i}" '
            f'pos="{x:.4f} {y:.4f} {z:.4f}" '
            f'type="box" '
            f'size="{hx:.4f} {hy:.4f} {hz:.4f}" '
            f'rgba="{r:.2f} {g:.2f} {b:.2f} {a:.2f}" '
            f'friction="1.0"/>'
        )
        line += f"  <!-- {name} ({topic}) -->"
        lines.append(line)

    lines.append("")
    lines.append("</mujocoinclude>")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {len(positions)} obstacle(s) to {output_path}")


def write_empty_obstacles_xml(output_path: str):
    lines = [
        "<mujocoinclude>",
        "    <!-- Hardware obstacles — populated by gen_obstacles_xml.py -->",
        "</mujocoinclude>",
        "",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote empty obstacle file to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Capture scene from Vicon: update robot initial XY + generate obstacles XML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Revert: git checkout go2.xml go2_mjx.xml obstacles_hw.xml",
    )
    parser.add_argument(
        "--robot-topic",
        default="/vicon/Go2/root",
        help="Vicon topic for robot XY capture (default: /vicon/Go2/root)",
    )
    parser.add_argument(
        "--robot-exclude",
        nargs="+",
        default=["/vicon/Go2/Go2"],
        help="Extra robot topics to exclude from obstacles",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Position samples per topic (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Max seconds to wait for samples (default: 15)",
    )
    parser.add_argument(
        "--size",
        nargs=3,
        type=float,
        default=[0.15, 0.15, 0.2],
        metavar=("HX", "HY", "HZ"),
        help="Box half-extents in meters (default: 0.15 0.15 0.2)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="Override Z position in meters. If not set, uses Vicon marker Z.",
    )
    parser.add_argument(
        "--rgba",
        nargs=4,
        type=float,
        default=[0.8, 0.2, 0.2, 1.0],
        metavar=("R", "G", "B", "A"),
        help="Obstacle color (default: 0.8 0.2 0.2 1.0 = red)",
    )

    args = parser.parse_args()

    rclpy.init(args=sys.argv)
    node = SceneCaptureNode(
        robot_topic=args.robot_topic,
        robot_exclude=tuple(args.robot_exclude),
        num_samples=args.num_samples,
    )

    start = time.time()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.all_done():
                break
            if time.time() - start > args.timeout:
                node.get_logger().warn(
                    f"Timeout ({args.timeout}s) reached. Using collected samples."
                )
                break
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")

    robot_pose = node.get_robot_pose()
    obstacle_positions = node.get_obstacle_positions()
    node.destroy_node()
    rclpy.shutdown()

    if robot_pose is not None:
        rx, ry, qw, qx, qy, qz = robot_pose
        _update_robot_pose_in_xml(GO2_XML, rx, ry, qw, qx, qy, qz)
        _update_robot_pose_in_xml(GO2_MJX_XML, rx, ry, qw, qx, qy, qz)
    else:
        print("No robot position captured — robot XML unchanged.")

    if obstacle_positions:
        write_obstacles_xml(
            obstacle_positions,
            OBSTACLES_HW_XML,
            size=tuple(args.size),
            height=args.height,
            rgba=tuple(args.rgba),
        )
    else:
        write_empty_obstacles_xml(OBSTACLES_HW_XML)

    print("\nDone. Start the controller whenever ready.\n"
          "To revert: git checkout go2.xml go2_mjx.xml obstacles_hw.xml")


if __name__ == "__main__":
    main()
