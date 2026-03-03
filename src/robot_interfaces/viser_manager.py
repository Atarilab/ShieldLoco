import time
from dataclasses import dataclass, field
from threading import Thread
from typing import Any, Optional

import mujoco
import numpy as np

try:
    import viser
except ImportError:
    viser = None


@dataclass
class ViserConfig:
    backend: str
    host: str
    port: int
    dt: float
    share_url: bool
    show_collision_geoms: bool
    collision_geom_group: Optional[int]
    collision_opacity: float
    collision_wireframe: bool
    collision_color_rgb: tuple[int, int, int]
    show_planned_contact_locations: bool
    planned_contact_topic: str
    planned_contact_radius: float
    planned_contact_color_rgb: tuple[int, int, int]
    show_qpos_ghost: bool
    qpos_trajectory_topic: str
    qpos_ghost_stride: int
    qpos_ghost_max_steps: int
    qpos_ghost_opacity: float
    qpos_ghost_wireframe: bool
    qpos_ghost_color_rgb: tuple[int, int, int]


@dataclass
class ViserState:
    server: Any = None
    viewer_url: Optional[str] = None
    share_link: Optional[str] = None
    collision_body_ids: list[int] = field(default_factory=list)
    collision_body_handles: dict[int, Any] = field(default_factory=dict)
    reset_button: Any = None
    opt_contact_handles: dict[int, Any] = field(default_factory=dict)
    opt_contact_positions: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    opt_contact_last_update: float = 0.0
    opt_contact_node: Any = None
    opt_contact_subscriber: Any = None
    opt_contact_thread: Optional[Thread] = None
    rclpy_initialized_here: bool = False
    opt_contact_horizon_length: int = 0
    # COM trajectory visualization
    com_trajectory_handles: dict[int, Any] = field(default_factory=dict)
    com_trajectory_positions: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    com_trajectory_last_update: float = 0.0
    com_trajectory_subscriber: Any = None
    qpos_trajectory_subscriber: Any = None
    qpos_ghost_body_ids: list[int] = field(default_factory=list)
    qpos_ghost_body_handles: dict[tuple[int, int], Any] = field(default_factory=dict)
    qpos_ghost_body_poses: dict[tuple[int, int], tuple[tuple[float, float, float], tuple[float, float, float, float]]] = field(default_factory=dict)
    qpos_ghost_steps: int = 0
    qpos_ghost_last_update: float = 0.0
    ghost_mj_data: Any = None


class ViserManager:
    def __init__(self, config, robot_name, mj_model, mj_data, locker, stop_event, request_reset_callback):
        collision_group = config.get("VISER_COLLISION_GEOM_GROUP", None)
        self.cfg = ViserConfig(
            backend=str(config.get("VISER_BACKEND", "none")).lower(),
            host=str(config.get("VISER_HOST", "0.0.0.0")),
            port=int(config.get("VISER_PORT", 8080)),
            dt=float(config.get("VISER_DT", config["VIEWER_DT"])),
            share_url=bool(config.get("VISER_SHARE_URL", False)),
            show_collision_geoms=bool(config.get("VISER_SHOW_COLLISION_GEOMS", False)),
            collision_geom_group=None if collision_group is None else int(collision_group),
            collision_opacity=float(config.get("VISER_COLLISION_OPACITY", 0.35)),
            collision_wireframe=bool(config.get("VISER_COLLISION_WIREFRAME", False)),
            collision_color_rgb=tuple(config.get("VISER_COLLISION_COLOR_RGB", [255, 120, 80])),
            show_planned_contact_locations=bool(config.get("VISER_SHOW_PLANNED_CONTACT_LOCATIONS", True)),
            planned_contact_topic=str(config.get("VISER_PLANNED_CONTACT_TOPIC", "opt_contact_locations")),
            planned_contact_radius=float(config.get("VISER_PLANNED_CONTACT_RADIUS", 0.04)),
            planned_contact_color_rgb=tuple(config.get("VISER_PLANNED_CONTACT_COLOR_RGB", [0, 255, 0])),
            show_qpos_ghost=bool(config.get("VISER_SHOW_QPOS_GHOST", True)),
            qpos_trajectory_topic=str(config.get("VISER_QPOS_TRAJECTORY_TOPIC", "qpos_trajectory")),
            qpos_ghost_stride=max(1, int(config.get("VISER_QPOS_GHOST_STRIDE", 2))),
            qpos_ghost_max_steps=max(1, int(config.get("VISER_QPOS_GHOST_MAX_STEPS", 12))),
            qpos_ghost_opacity=float(config.get("VISER_QPOS_GHOST_OPACITY", 0.18)),
            qpos_ghost_wireframe=bool(config.get("VISER_QPOS_GHOST_WIREFRAME", False)),
            qpos_ghost_color_rgb=tuple(config.get("VISER_QPOS_GHOST_COLOR_RGB", [130, 130, 255])),
        )
        self.state = ViserState()

        self.robot_name = robot_name
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.locker = locker
        self.stop_event = stop_event
        self.request_reset_callback = request_reset_callback

    def is_enabled(self):
        return self.cfg.backend == "viser"

    def has_server(self):
        return self.state.server is not None

    def initialize(self):
        if not self.is_enabled():
            return
        self._initialize_web_viewer()
        self._initialize_web_opt_contact_locations()
        self._initialize_opt_contact_subscriber()

    def _initialize_web_viewer(self):
        if viser is None:
            raise ImportError(
                "VISER_BACKEND=viser requested, but `viser` is not installed. "
                "Install it with: pip install viser"
            )

        self.state.server = viser.ViserServer(
            host=self.cfg.host,
            port=self.cfg.port,
            label=f"DOOM {self.robot_name}",
        )

        host = self.state.server.get_host()
        port = self.state.server.get_port()
        display_host = "localhost" if host in ("0.0.0.0", "::") else host
        self.state.viewer_url = f"http://{display_host}:{port}"

        self.state.server.scene.set_up_direction("+z")
        self.state.server.scene.add_grid("/world/grid", width=8.0, height=8.0, plane="xy")

        try:
            self.state.reset_button = self.state.server.gui.add_button("Reset Robot")
            self.state.reset_button.on_click(lambda _: self.request_reset_callback())
        except Exception:
            self.state.reset_button = None

        if self.cfg.show_collision_geoms:
            self._initialize_web_collision_geometries()

        if self.cfg.show_qpos_ghost:
            self._initialize_web_qpos_ghosts()

        if self.cfg.share_url:
            self.state.share_link = self.state.server.request_share_url(verbose=False)

    def _initialize_web_opt_contact_locations(self):
        if self.state.server is None or not self.cfg.show_planned_contact_locations:
            return

        foot_colors = [
            (255, 0, 0),    # Red for FL
            (0, 255, 0),    # Green for FR
            (0, 0, 255),    # Blue for RL
            (255, 255, 0),  # Yellow for RR
        ]

        max_horizon = 30
        self.state.opt_contact_horizon_length = max_horizon

        for foot_idx, foot_name in enumerate(["FL", "FR", "RL", "RR"]):
            for step in range(max_horizon):
                handle = self.state.server.scene.add_icosphere(
                    f"/robot/opt_contact_locations/{foot_name}_step{step}",
                    radius=self.cfg.planned_contact_radius * 0.8,
                    color=foot_colors[foot_idx],
                    position=(0.0, 0.0, 0.0),
                    visible=False,
                )
                self.state.opt_contact_handles[foot_idx * max_horizon + step] = handle

        # Initialize COM trajectory handles (max 20 steps)
        com_horizon = 20
        for step in range(com_horizon):
            handle = self.state.server.scene.add_icosphere(
                f"/robot/com_trajectory/step{step}",
                radius=0.015,
                color=(0, 255, 255),  # Cyan
                position=(0.0, 0.0, 0.0),
                visible=False,
            )
            self.state.com_trajectory_handles[step] = handle

    def _initialize_opt_contact_subscriber(self):
        if not self.cfg.show_planned_contact_locations and not self.cfg.show_qpos_ghost:
            return

        try:
            import rclpy
            from std_msgs.msg import Float32MultiArray
            from visualization_msgs.msg import MarkerArray

            if not rclpy.ok():
                rclpy.init(args=None)
                self.state.rclpy_initialized_here = True

            self.state.opt_contact_node = rclpy.create_node("viser_opt_contact_locations_subscriber")
            if self.cfg.show_planned_contact_locations:
                self.state.opt_contact_subscriber = self.state.opt_contact_node.create_subscription(
                    MarkerArray,
                    self.cfg.planned_contact_topic,
                    self._on_opt_contact_locations,
                    10,
                )
                self.state.com_trajectory_subscriber = self.state.opt_contact_node.create_subscription(
                    MarkerArray,
                    "com_trajectory",
                    self._on_com_trajectory,
                    10,
                )

            if self.cfg.show_qpos_ghost:
                self.state.qpos_trajectory_subscriber = self.state.opt_contact_node.create_subscription(
                    Float32MultiArray,
                    self.cfg.qpos_trajectory_topic,
                    self._on_qpos_trajectory,
                    10,
                )
            self.state.opt_contact_thread = Thread(target=self._opt_contact_spin_thread, daemon=True)
            self.state.opt_contact_thread.start()
        except Exception as e:
            print(f"Failed to initialize opt contact subscriber for Viser: {e}", flush=True)

    def _opt_contact_spin_thread(self):
        try:
            import rclpy

            while not self.stop_event.is_set() and self.state.opt_contact_node is not None:
                rclpy.spin_once(self.state.opt_contact_node, timeout_sec=0.05)
        except Exception as e:
            print(f"Opt contact spin thread stopped: {e}", flush=True)

    def _on_opt_contact_locations(self, msg):
        positions = {}
        foot_order = ["FL", "FR", "RL", "RR"]
        foot_to_markers = {name: [] for name in foot_order}

        for marker in msg.markers:
            ns = str(marker.ns)
            foot_name = None
            prefix = "opt_foot_trajectory_"
            if ns.startswith(prefix):
                suffix = ns[len(prefix):]
                if suffix in foot_to_markers:
                    foot_name = suffix

            if foot_name is None:
                continue

            pos = (
                float(marker.pose.position.x),
                float(marker.pose.position.y),
                float(marker.pose.position.z),
            )
            foot_to_markers[foot_name].append((int(marker.id), pos))

        max_horizon = self.state.opt_contact_horizon_length
        for foot_idx, foot_name in enumerate(foot_order):
            marker_list = sorted(foot_to_markers[foot_name], key=lambda x: x[0])
            for step, (_, pos) in enumerate(marker_list[:max_horizon]):
                handle_id = foot_idx * max_horizon + step
                positions[handle_id] = pos

        with self.locker:
            self.state.opt_contact_positions = positions
            self.state.opt_contact_last_update = time.time()

    def _on_com_trajectory(self, msg):
        positions = {}
        for marker in msg.markers:
            positions[int(marker.id)] = (
                float(marker.pose.position.x),
                float(marker.pose.position.y),
                float(marker.pose.position.z),
            )

        with self.locker:
            self.state.com_trajectory_positions = positions
            self.state.com_trajectory_last_update = time.time()

    def _on_qpos_trajectory(self, msg):
        if not self.cfg.show_qpos_ghost or self.state.ghost_mj_data is None:
            return

        try:
            raw = np.asarray(msg.data, dtype=np.float64)
            if raw.size == 0:
                return

            nq = int(self.mj_model.nq)
            num_steps = 0
            qpos_dim = nq

            if len(msg.layout.dim) >= 2:
                num_steps = int(msg.layout.dim[0].size)
                qpos_dim = int(msg.layout.dim[1].size)
            elif nq > 0 and raw.size % nq == 0:
                num_steps = int(raw.size // nq)
                qpos_dim = nq

            if num_steps <= 0 or qpos_dim <= 0 or raw.size < (num_steps * qpos_dim):
                return

            qpos_steps = raw[: num_steps * qpos_dim].reshape(num_steps, qpos_dim)
            qpos_steps = qpos_steps[:: self.cfg.qpos_ghost_stride]
            qpos_steps = qpos_steps[: self.cfg.qpos_ghost_max_steps]

            ghost_poses = self._compute_qpos_ghost_body_poses(qpos_steps)
            with self.locker:
                self.state.qpos_ghost_body_poses = ghost_poses
                self.state.qpos_ghost_steps = int(qpos_steps.shape[0])
                self.state.qpos_ghost_last_update = time.time()
        except Exception as e:
            print(f"Failed to parse qpos trajectory for Viser ghost: {e}", flush=True)

    def _should_visualize_collision_geom(self, geom_id):
        geom_contype = int(self.mj_model.geom_contype[geom_id])
        geom_conaffinity = int(self.mj_model.geom_conaffinity[geom_id])
        if geom_contype == 0 and geom_conaffinity == 0:
            return False

        if self.cfg.collision_geom_group is not None:
            geom_group = int(self.mj_model.geom_group[geom_id])
            if geom_group != self.cfg.collision_geom_group:
                return False

        return True

    def _should_visualize_qpos_ghost_geom(self, geom_id):
        if not self._should_visualize_collision_geom(geom_id):
            return False

        geom_body_id = int(self.mj_model.geom_bodyid[geom_id])
        # Exclude world geoms from ghost overlay to avoid duplicate static scene visuals.
        if geom_body_id <= 0:
            return False

        geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        geom_name = (geom_name or "").lower()
        if geom_name == "floor":
            return False

        excluded_prefixes = ("obstacle_", "destination_", "finish_")
        if any(geom_name.startswith(prefix) for prefix in excluded_prefixes):
            return False

        return True

    def _initialize_web_collision_geometries(self):
        if self.state.server is None:
            return

        added_body_ids = set()
        for geom_id in range(self.mj_model.ngeom):
            if not self._should_visualize_collision_geom(geom_id):
                continue

            geom_body_id = int(self.mj_model.geom_bodyid[geom_id])
            if geom_body_id < 0:
                continue

            body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, geom_body_id)
            if body_name is None:
                body_name = f"body_{geom_body_id}"

            body_frame_path = f"/robot/collisions/bodies/{body_name}"
            if geom_body_id not in self.state.collision_body_handles:
                self.state.collision_body_handles[geom_body_id] = self.state.server.scene.add_frame(
                    body_frame_path,
                    show_axes=False,
                )

            if geom_body_id not in added_body_ids:
                self.state.collision_body_ids.append(geom_body_id)
                added_body_ids.add(geom_body_id)

            geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name is None:
                geom_name = f"geom_{geom_id}"

            geom_path = f"{body_frame_path}/{geom_name}_{geom_id}"
            self._add_collision_geom_visual(
                geom_id=geom_id,
                geom_path=geom_path,
                color=self.cfg.collision_color_rgb,
                opacity=self.cfg.collision_opacity,
                wireframe=self.cfg.collision_wireframe,
            )

    def _add_collision_geom_visual(self, geom_id, geom_path, color, opacity, wireframe):
        if self.state.server is None:
            return

        geom_type = int(self.mj_model.geom_type[geom_id])
        geom_pos = tuple(self.mj_model.geom_pos[geom_id].tolist())
        geom_quat = tuple(self.mj_model.geom_quat[geom_id].tolist())
        geom_size = self.mj_model.geom_size[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius = float(geom_size[0])
            self.state.server.scene.add_icosphere(
                geom_path,
                radius=radius,
                color=color,
                opacity=opacity,
                wireframe=wireframe,
                wxyz=geom_quat,
                position=geom_pos,
            )
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            dimensions = (
                float(2.0 * geom_size[0]),
                float(2.0 * geom_size[1]),
                float(2.0 * geom_size[2]),
            )
            self.state.server.scene.add_box(
                geom_path,
                dimensions=dimensions,
                color=color,
                opacity=opacity,
                wireframe=wireframe,
                wxyz=geom_quat,
                position=geom_pos,
            )
        elif geom_type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            radius = float(geom_size[0])
            half_length = float(geom_size[1])
            height = float(2.0 * half_length)
            self.state.server.scene.add_cylinder(
                geom_path,
                radius=radius,
                height=height,
                color=color,
                opacity=opacity,
                wireframe=wireframe,
                wxyz=geom_quat,
                position=geom_pos,
            )

    def _initialize_web_qpos_ghosts(self):
        if self.state.server is None:
            return

        added_body_ids = set()
        self.state.qpos_ghost_body_ids = []
        for geom_id in range(self.mj_model.ngeom):
            if not self._should_visualize_qpos_ghost_geom(geom_id):
                continue
            geom_body_id = int(self.mj_model.geom_bodyid[geom_id])
            if geom_body_id < 0 or geom_body_id in added_body_ids:
                continue
            self.state.qpos_ghost_body_ids.append(geom_body_id)
            added_body_ids.add(geom_body_id)

        self.state.ghost_mj_data = mujoco.MjData(self.mj_model)

        for step in range(self.cfg.qpos_ghost_max_steps):
            for body_id in self.state.qpos_ghost_body_ids:
                body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name is None:
                    body_name = f"body_{body_id}"

                body_frame_path = f"/robot/qpos_ghost/step_{step}/bodies/{body_name}"
                body_handle = self.state.server.scene.add_frame(
                    body_frame_path,
                    show_axes=False,
                    visible=False,
                )
                self.state.qpos_ghost_body_handles[(step, body_id)] = body_handle

            for geom_id in range(self.mj_model.ngeom):
                if not self._should_visualize_qpos_ghost_geom(geom_id):
                    continue

                geom_body_id = int(self.mj_model.geom_bodyid[geom_id])
                if geom_body_id < 0:
                    continue

                body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, geom_body_id)
                if body_name is None:
                    body_name = f"body_{geom_body_id}"

                geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                if geom_name is None:
                    geom_name = f"geom_{geom_id}"

                geom_path = f"/robot/qpos_ghost/step_{step}/bodies/{body_name}/{geom_name}_{geom_id}"
                self._add_collision_geom_visual(
                    geom_id=geom_id,
                    geom_path=geom_path,
                    color=self.cfg.qpos_ghost_color_rgb,
                    opacity=self.cfg.qpos_ghost_opacity,
                    wireframe=self.cfg.qpos_ghost_wireframe,
                )

    def _compute_qpos_ghost_body_poses(self, qpos_steps):
        if self.state.ghost_mj_data is None:
            return {}

        body_poses = {}
        max_nq = int(self.mj_model.nq)
        tracked_ids = list(self.state.qpos_ghost_body_ids)

        for step_idx, qpos in enumerate(qpos_steps):
            step_qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
            nq = min(max_nq, step_qpos.shape[0])

            self.state.ghost_mj_data.qpos[:] = 0.0
            self.state.ghost_mj_data.qpos[:nq] = step_qpos[:nq]
            mujoco.mj_forward(self.mj_model, self.state.ghost_mj_data)

            for body_id in tracked_ids:
                body_poses[(step_idx, body_id)] = (
                    tuple(np.asarray(self.state.ghost_mj_data.xpos[body_id]).tolist()),
                    tuple(np.asarray(self.state.ghost_mj_data.xquat[body_id]).tolist()),
                )

        return body_poses

    def _snapshot_collision_body_poses(self):
        tracked_body_poses = {}
        tracked_ids = set(self.state.collision_body_ids)

        for body_id in tracked_ids:
            tracked_body_poses[body_id] = (
                np.array(self.mj_data.xpos[body_id], copy=True),
                np.array(self.mj_data.xquat[body_id], copy=True),
            )
        return tracked_body_poses

    def run(self):
        if self.state.server is None:
            return

        while not self.stop_event.is_set():
            with self.locker:
                tracked_body_poses = self._snapshot_collision_body_poses()

            with self.state.server.atomic():
                for body_id, (body_pos, body_quat) in tracked_body_poses.items():
                    collision_body_handle = self.state.collision_body_handles.get(body_id)
                    if collision_body_handle is not None:
                        collision_body_handle.position = tuple(body_pos.tolist())
                        collision_body_handle.wxyz = tuple(body_quat.tolist())

                if self.cfg.show_planned_contact_locations and self.state.opt_contact_handles:
                    with self.locker:
                        opt_positions = dict(self.state.opt_contact_positions)
                        has_recent_update = (time.time() - self.state.opt_contact_last_update) < 1.0

                    if has_recent_update:
                        for marker_id, pos in opt_positions.items():
                            handle = self.state.opt_contact_handles.get(marker_id)
                            if handle is not None:
                                handle.position = pos
                                handle.visible = True
                    else:
                        for handle in self.state.opt_contact_handles.values():
                            handle.visible = False

                # Update COM trajectory visualization
                if self.state.com_trajectory_handles:
                    with self.locker:
                        com_positions = dict(self.state.com_trajectory_positions)
                        has_com_update = (time.time() - self.state.com_trajectory_last_update) < 1.0

                    if has_com_update:
                        for marker_id, pos in com_positions.items():
                            handle = self.state.com_trajectory_handles.get(marker_id)
                            if handle is not None:
                                handle.position = pos
                                handle.visible = True
                    else:
                        for handle in self.state.com_trajectory_handles.values():
                            handle.visible = False

                if self.cfg.show_qpos_ghost and self.state.qpos_ghost_body_handles:
                    with self.locker:
                        ghost_poses = dict(self.state.qpos_ghost_body_poses)
                        ghost_steps = int(self.state.qpos_ghost_steps)
                        has_ghost_update = (time.time() - self.state.qpos_ghost_last_update) < 1.0

                    for (step, body_id), handle in self.state.qpos_ghost_body_handles.items():
                        pose = ghost_poses.get((step, body_id))
                        if has_ghost_update and step < ghost_steps and pose is not None:
                            body_pos, body_quat = pose
                            handle.position = body_pos
                            handle.wxyz = body_quat
                            handle.visible = True
                        else:
                            handle.visible = False

            time.sleep(self.cfg.dt)

    def get_viewer_url(self):
        return self.state.viewer_url

    def get_share_url(self):
        return self.state.share_link

    def stop(self):
        if self.state.server is not None:
            self.state.server.stop()

        if self.state.opt_contact_thread is not None:
            self.state.opt_contact_thread.join(timeout=1.0)

        if self.state.opt_contact_node is not None:
            try:
                self.state.opt_contact_node.destroy_node()
            except Exception:
                pass
            self.state.opt_contact_node = None

        if self.state.rclpy_initialized_here:
            try:
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass
