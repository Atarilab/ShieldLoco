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

try:
    import trimesh
except ImportError:
    trimesh = None


@dataclass
class ViserConfig:
    backend: str
    host: str
    port: int
    dt: float
    share_url: bool
    show_collision_geoms: bool
    collision_geom_group: Optional[int]
    collision_wireframe: bool
    show_planned_contact_locations: bool
    planned_contact_topic: str
    planned_contact_radius: float
    
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
    mesh_handles: dict[int, Any] = field(default_factory=dict)  # geom_id -> mesh handle
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
    follow_enabled: bool = True
    follow_target_pos: Optional[np.ndarray] = None
    # Ball visualization
    ball_handles: dict[int, Any] = field(default_factory=dict)  # slot_idx -> sphere handle
    ball_geom_ids: list[int] = field(default_factory=list)  # geom IDs for each ball


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
            collision_wireframe=bool(config.get("VISER_COLLISION_WIREFRAME", False)),
            show_planned_contact_locations=bool(config.get("VISER_SHOW_PLANNED_CONTACT_LOCATIONS", True)),
            planned_contact_topic=str(config.get("VISER_PLANNED_CONTACT_TOPIC", "opt_contact_locations")),
            planned_contact_radius=float(config.get("VISER_PLANNED_CONTACT_RADIUS", 0.04)),
            
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
        self.mesh_cache = {}  # Cache loaded meshes to avoid reloading
        self.ball_launcher = None
        self.base_body_id = None

    def _load_mesh_from_mujoco(self, geom_id: int):
        """Load mesh data from MuJoCo model and cache it as a trimesh object.
        
        Extracts vertex and face data from MuJoCo's internal mesh storage and converts
        to trimesh format for rendering with viser.add_mesh_simple().
        """
        if trimesh is None:
            return None
        
        if geom_id in self.mesh_cache:
            return self.mesh_cache[geom_id]
        
        try:
            mesh_id = int(self.mj_model.geom_dataid[geom_id])
            if mesh_id < 0:
                return None
            
            mesh_vert_adr = int(self.mj_model.mesh_vertadr[mesh_id])
            mesh_vert_num = int(self.mj_model.mesh_vertnum[mesh_id])
            mesh_face_adr = int(self.mj_model.mesh_faceadr[mesh_id])
            mesh_face_num = int(self.mj_model.mesh_facenum[mesh_id])
            
            if mesh_vert_num <= 0 or mesh_face_num <= 0:
                return None
            
            vertices = np.asarray(
                self.mj_model.mesh_vert[mesh_vert_adr : mesh_vert_adr + mesh_vert_num],
                dtype=np.float64
            )
            
            if vertices.size != mesh_vert_num * 3:
                return None
            
            vertices = vertices.reshape(mesh_vert_num, 3)
            
            faces = np.asarray(
                self.mj_model.mesh_face[mesh_face_adr : mesh_face_adr + mesh_face_num],
                dtype=np.uint32
            )
            
            if faces.size != mesh_face_num * 3:
                return None
                
            faces = faces.reshape(mesh_face_num, 3)
            
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            self.mesh_cache[geom_id] = mesh
            return mesh
            
        except Exception:
            return None

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
        self._setup_client_callbacks()

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

        try:
            self.state.server.gui.add_button("Follow Robot").on_click(
                lambda _: self._toggle_follow()
            )
        except Exception as e:
            print(f"Failed to add follow button: {e}", flush=True)

        if self.cfg.show_collision_geoms:
            self._initialize_web_collision_geometries()

        if self.cfg.show_qpos_ghost:
            self._initialize_web_qpos_ghosts()

        if self.cfg.share_url:
            self.state.share_link = self.state.server.request_share_url(verbose=False)

    def set_ball_launcher(self, ball_launcher, base_body_id):
        """Attach a BallLauncher and add a throw button to the web UI."""
        self.ball_launcher = ball_launcher
        self.base_body_id = base_body_id

        # Create ball visualization spheres
        self.state.ball_geom_ids = []
        for i in range(ball_launcher.max_balls):
            geom_name = f"ball_geom_{i}"
            geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            self.state.ball_geom_ids.append(geom_id)
            if self.state.server is not None and geom_id >= 0:
                try:
                    handle = self.state.server.scene.add_icosphere(
                        f"/robot/balls/{geom_name}",
                        radius=float(ball_launcher.ball_radius),
                        color=(255, 128, 0),
                        position=(0.0, 0.0, -10.0),
                        visible=False,
                    )
                    self.state.ball_handles[i] = handle
                except Exception:
                    pass

        if self.state.server is None:
            return
        try:
            btn = self.state.server.gui.add_button("Throw Ball")
            btn.on_click(lambda _: self._handle_throw_ball())
        except Exception as e:
            print(f"Failed to add throw ball button: {e}", flush=True)

    def _handle_throw_ball(self):
        """Throw a ball at the robot's current position."""
        if self.ball_launcher is None or self.base_body_id is None:
            return
        with self.locker:
            robot_pos = self.mj_data.xpos[self.base_body_id].copy()
            self.ball_launcher.launch(robot_pos, self.mj_data)

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
            # Skip the first foothold marker: it typically sits under the robot
            # and reads like a fake "ball" in Viser when the qpos ghost is also enabled.
            marker_list = marker_list[1:]
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

            # Build sparse samples across the full horizon and always include the final state.
            stride = max(1, int(self.cfg.qpos_ghost_stride))
            max_steps = max(1, int(self.cfg.qpos_ghost_max_steps))

            candidate_indices = np.arange(0, num_steps, stride, dtype=np.int32)
            if candidate_indices.size == 0 or candidate_indices[-1] != (num_steps - 1):
                candidate_indices = np.append(candidate_indices, num_steps - 1)

            if candidate_indices.size > max_steps:
                select_positions = np.linspace(0, candidate_indices.size - 1, num=max_steps)
                select_positions = np.rint(select_positions).astype(np.int32)
                selected_indices = candidate_indices[select_positions]
                selected_indices[-1] = num_steps - 1
                selected_indices = np.unique(selected_indices)
                if selected_indices[-1] != (num_steps - 1):
                    selected_indices = np.append(selected_indices, num_steps - 1)
            else:
                selected_indices = candidate_indices

            qpos_steps = qpos_steps[selected_indices]

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
        geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        geom_name = (geom_name or "").lower()

        # Show visual-only markers for known prefixes even if contype/conaffinity are zero
        if geom_contype == 0 and geom_conaffinity == 0:
            if geom_name.startswith(("destination_", "finish_")):
                return True
            return False

        if self.cfg.collision_geom_group is not None:
            geom_group = int(self.mj_model.geom_group[geom_id])
            if geom_group != self.cfg.collision_geom_group:
                return False

        return True
    
    def _is_robot_geom(self, geom_id):
        """Check if a geom belongs to the robot (not world/obstacles).
        
        Robot bodies have body_id > 0. World/global body is 0.
        """
        body_id = int(self.mj_model.geom_bodyid[geom_id])
        return body_id > 0

    def _should_visualize_qpos_ghost_geom(self, geom_id):
        if not self._should_visualize_collision_geom(geom_id):
            return False

        geom_body_id = int(self.mj_model.geom_bodyid[geom_id])
        # Only visualize for robot bodies, not world geoms
        if geom_body_id <= 0:
            return False

        geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        geom_name = (geom_name or "").lower()
        if geom_name == "floor":
            return False

        # The qpos ghost comes from the planner model, which does not include the
        # sim-only projectile balls. Rendering ball geoms from the sim model here
        # creates bogus ghost balls at the origin.
        excluded_prefixes = ("obstacle_", "destination_", "finish_", "ball_")
        if any(geom_name.startswith(prefix) for prefix in excluded_prefixes):
            return False

        return True

    def _add_mesh_visualization(self, geom_id: int):
        """Add mesh visualization for a geometry mesh.
        
        Creates viser mesh handle at origin with identity rotation, then updates
        position/rotation in animation loop from simulation data.
        """
        if self.state.server is None:
            return
        
        try:
            mesh = self._load_mesh_from_mujoco(geom_id)
            if mesh is None:
                return
            
            geom_rgba = self.mj_model.geom_rgba[geom_id]
            geom_color = (float(geom_rgba[0]), float(geom_rgba[1]), float(geom_rgba[2]))
            geom_opacity = float(geom_rgba[3])
            
            body_id = int(self.mj_model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name is None:
                body_name = f"body_{body_id}"
            
            geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name is None:
                geom_name = f"geom_{geom_id}"
            
            geom_path = f"/robot/{body_name}/{geom_name}_{geom_id}"
            
            handle = self.state.server.scene.add_mesh_simple(
                geom_path,
                vertices=mesh.vertices,
                faces=mesh.faces,
                color=geom_color,
                opacity=geom_opacity,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=(0.0, 0.0, 0.0),
            )
            
            self.state.mesh_handles[geom_id] = handle
            
        except Exception:
            pass

    def _initialize_web_collision_geometries(self):
        """Initialize visualization of robot meshes and scene collision primitives.
        
        - Robot meshes (all body_id > 0) rendered as detailed meshes
        - Scene obstacles (body_id == 0) rendered as collision primitives
        """
        if self.state.server is None:
            return

        for geom_id in range(self.mj_model.ngeom):
            geom_type = int(self.mj_model.geom_type[geom_id])
            
            if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                self._add_mesh_visualization(geom_id)
                continue
            
            # Skip collision primitive rendering for robot geoms
            if self._is_robot_geom(geom_id):
                continue
            
            # Only visualize collidable obstacle geoms
            if not self._should_visualize_collision_geom(geom_id):
                continue

            geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name is None:
                geom_name = f"geom_{geom_id}"

            geom_path = f"/robot/collision_geoms/{geom_name}_{geom_id}"
            
            geom_rgba = self.mj_model.geom_rgba[geom_id]
            geom_color = (geom_rgba[0], geom_rgba[1], geom_rgba[2])
            geom_opacity = geom_rgba[3]
            
            self._add_collision_geom_visual(
                geom_id=geom_id,
                geom_path=geom_path,
                color=geom_color,
                opacity=geom_opacity,
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
        with self.locker:
            base_qpos = np.asarray(self.mj_data.qpos, dtype=np.float64).copy()

        for step_idx, qpos in enumerate(qpos_steps):
            step_qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
            nq = min(max_nq, step_qpos.shape[0])

            self.state.ghost_mj_data.qpos[:] = base_qpos
            self.state.ghost_mj_data.qpos[:nq] = step_qpos[:nq]
            mujoco.mj_forward(self.mj_model, self.state.ghost_mj_data)

            for body_id in tracked_ids:
                body_poses[(step_idx, body_id)] = (
                    tuple(np.asarray(self.state.ghost_mj_data.xpos[body_id]).tolist()),
                    tuple(np.asarray(self.state.ghost_mj_data.xquat[body_id]).tolist()),
                )

        return body_poses

    def _snapshot_mesh_geom_poses(self):
        """Get current poses of all mesh geometries from simulation data.
        
        Extracts world-frame position and rotation for each mesh geometry.
        
        Returns:
            dict: Mapping geom_id -> (position_array, quaternion_tuple)
        """
        mesh_poses = {}
        
        handle_ids = list(self.state.mesh_handles.keys())
        if not handle_ids:
            return mesh_poses
        
        for geom_id in handle_ids:
            try:
                # Check array bounds
                if geom_id >= len(self.mj_data.geom_xpos):
                    continue
                
                # Get world pose directly from MuJoCo (already composed with body transform)
                geom_pos = np.array(self.mj_data.geom_xpos[geom_id], copy=True)
                
                # Get rotation matrix and flatten to 9-element vector for mju_mat2Quat
                geom_mat = np.array(self.mj_data.geom_xmat[geom_id], copy=True)
                geom_mat_flat = geom_mat.flatten()  # Convert 3x3 to flat 9-element array
                
                # Convert rotation matrix to quaternion
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, geom_mat_flat)
                quat_wxyz = tuple(quat.tolist())
                
                mesh_poses[geom_id] = (geom_pos, quat_wxyz)
            except Exception:
                pass
        
        return mesh_poses


    def run(self):
        """Main visualization animation loop.
        
        Thread-safe updates to mesh positions/rotations and other visualization
        elements (contact markers, trajectories, etc.) at each simulation frame.
        Uses server.atomic() for batched scene updates.
        """
        if self.state.server is None:
            return

        while not self.stop_event.is_set():
            try:
                with self.locker:
                    mesh_poses = self._snapshot_mesh_geom_poses()
                    # Update follow target to robot's base position (body 1)
                    if self.mj_model.nbody > 1:
                        self.state.follow_target_pos = np.array(self.mj_data.xpos[1], copy=True)
                    # Snapshot ball positions
                    ball_positions = {}
                    for i, geom_id in enumerate(self.state.ball_geom_ids):
                        if geom_id >= 0 and i in self.state.ball_handles:
                            ball_positions[i] = tuple(self.mj_data.geom_xpos[geom_id].tolist())

                with self.state.server.atomic():
                    # Update mesh poses from simulation data
                    for geom_id, (geom_pos, geom_quat) in mesh_poses.items():
                        mesh_handle = self.state.mesh_handles.get(geom_id)
                        if mesh_handle is not None:
                            mesh_handle.position = tuple(geom_pos.tolist())
                            mesh_handle.wxyz = geom_quat

                    # Update ball positions
                    for i, pos in ball_positions.items():
                        handle = self.state.ball_handles.get(i)
                        if handle is not None:
                            # Only show balls that are above ground (not hidden at z=-10 or far away)
                            if pos[2] > -1.0 and abs(pos[0]) < 50.0:
                                handle.position = pos
                                handle.visible = True
                            else:
                                handle.visible = False

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

            except Exception as e:
                pass

            time.sleep(self.cfg.dt)

    def get_viewer_url(self):
        return self.state.viewer_url

    def get_share_url(self):
        return self.state.share_link

    def _toggle_follow(self):
        self.state.follow_enabled = not self.state.follow_enabled

    def _setup_client_callbacks(self):
        """Start background thread to update all clients' cameras for follow mode."""
        if self.state.server is None:
            return
        
        def update_all_clients():
            while not self.stop_event.is_set():
                if self.state.follow_enabled and self.state.follow_target_pos is not None:
                    try:
                        clients = self.state.server.get_clients()
                        for client in clients.values():
                            try:
                                body_pos = self.state.follow_target_pos
                                cam_offset = np.array([-0.8, -0.8, 0.8])
                                cam_pos = body_pos + cam_offset
                                cam_pos[2] = 0.8
                                
                                with client.atomic():
                                    client.camera.position = tuple(cam_pos)
                                    client.camera.look_at = tuple(body_pos)
                            except Exception:
                                pass
                    except Exception:
                        pass
                
                time.sleep(self.cfg.dt)
        
        t = Thread(target=update_all_clients, daemon=True)
        t.start()

    def stop(self):
        # Stop recording if active
        try:
            self._stop_recording()
        except Exception:
            pass

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
