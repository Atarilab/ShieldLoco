# robot_interfaces/sim_robot_interface.py
import os
import threading
from threading import Thread
import time
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
try:
    import viser
except ImportError:
    viser = None

from utils.unitree_sdk2py_bridge import UnitreeSdk2Bridge


class SimRobotInterface:
    def __init__(self, config, headless=False):
        self.stop_event = threading.Event()
        self.reset_requested = threading.Event()

        self.robot_name = config["ROBOT"]
        self.robot_scene = os.path.join(os.getcwd(), "robots", self.robot_name, config["SCENE"])

        # Initialize Mujoco model and data
        self.mj_model = mujoco.MjModel.from_xml_path(self.robot_scene)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Check for object body in the scene
        self.object_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "object")

        # Set the timestep for the simulation
        self.mj_model.opt.timestep = config["SIMULATION_DT"]

        # Base link
        self.base_link_name = config["BASE_LINK"]
        self.base_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.base_link_name)

        # Fixed base position configuration
        self.fixed_base_pos = config.get("FIXED_BASE_POS", None)  # [x, y, z] or None for no fixed position

        # Elastic band
        use_elastic_band = config.get("ELASTIC_BAND", False)
        self.elastic_band = ElasticBand() if use_elastic_band else None
        # Initialize the viewer based on configuration
        viewer_kwargs: dict[str, object] = {
            "show_left_ui": False,
            "show_right_ui": False,
            "key_callback": self._viewer_key_callback,
        }
        if use_elastic_band and self.elastic_band is not None:
            self.band_link = self.mj_model.body(self.base_link_name)

        # Set up viewer (skip if headless mode)
        self.viewer = None
        self.headless = headless
        if not headless:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data, **viewer_kwargs)

        # Optional web viewer (Viser)
        self.web_viewer_backend = str(config.get("WEB_VIEWER_BACKEND", "none")).lower()
        self.web_viewer_host = config.get("WEB_VIEWER_HOST", "0.0.0.0")
        self.web_viewer_port = int(config.get("WEB_VIEWER_PORT", 8080))
        self.web_viewer_dt = float(config.get("WEB_VIEWER_DT", config["VIEWER_DT"]))
        self.web_viewer_share_url = bool(config.get("WEB_VIEWER_SHARE_URL", False))
        self.web_viewer_show_base = bool(config.get("WEB_VIEWER_SHOW_BASE", True))
        self.web_viewer_show_frames = bool(config.get("WEB_VIEWER_SHOW_FRAMES", True))
        self.web_viewer_tracked_bodies = config.get("WEB_VIEWER_TRACKED_BODIES", None)
        self.web_viewer_auto_track_filters = [
            str(value).lower() for value in config.get("WEB_VIEWER_AUTO_TRACK_FILTERS", ["foot", "toe"])
        ]
        self.web_viewer_base_axes_length = float(config.get("WEB_VIEWER_BASE_AXES_LENGTH", 0.3))
        self.web_viewer_base_axes_radius = float(config.get("WEB_VIEWER_BASE_AXES_RADIUS", 0.015))
        self.web_viewer_body_axes_length = float(config.get("WEB_VIEWER_BODY_AXES_LENGTH", 0.16))
        self.web_viewer_body_axes_radius = float(config.get("WEB_VIEWER_BODY_AXES_RADIUS", 0.008))
        self.web_viewer_show_collisions = bool(config.get("WEB_VIEWER_SHOW_COLLISIONS", False))
        collision_group = config.get("WEB_VIEWER_COLLISION_GROUP", None)
        self.web_viewer_collision_group = None if collision_group is None else int(collision_group)
        self.web_viewer_collision_opacity = float(config.get("WEB_VIEWER_COLLISION_OPACITY", 0.35))
        self.web_viewer_collision_wireframe = bool(config.get("WEB_VIEWER_COLLISION_WIREFRAME", False))
        raw_collision_color = config.get("WEB_VIEWER_COLLISION_COLOR", [255, 120, 80])
        self.web_viewer_collision_color = tuple(raw_collision_color)
        self.web_viewer_show_opt_contact_locations = bool(config.get("WEB_VIEWER_SHOW_OPT_CONTACT_LOCATIONS", True))
        self.web_viewer_opt_contact_topic = str(config.get("WEB_VIEWER_OPT_CONTACT_TOPIC", "opt_contact_locations"))
        self.web_viewer_opt_contact_radius = float(config.get("WEB_VIEWER_OPT_CONTACT_RADIUS", 0.04))
        raw_opt_contact_color = config.get("WEB_VIEWER_OPT_CONTACT_COLOR", [0, 255, 0])
        self.web_viewer_opt_contact_color = tuple(raw_opt_contact_color)
        self.web_viewer_show_actual_contact_locations = bool(
            config.get("WEB_VIEWER_SHOW_ACTUAL_CONTACT_LOCATIONS", True)
        )
        self.web_viewer_actual_contact_radius = float(config.get("WEB_VIEWER_ACTUAL_CONTACT_RADIUS", 0.04))
        raw_actual_contact_color = config.get("WEB_VIEWER_ACTUAL_CONTACT_COLOR", [0, 120, 255])
        self.web_viewer_actual_contact_color = tuple(raw_actual_contact_color)
        self.web_viewer_actual_contact_bodies = [
            str(name) for name in config.get("WEB_VIEWER_ACTUAL_CONTACT_BODIES", ["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
        ]
        self.web_server = None
        self.web_viewer_url = None
        self.web_viewer_share_link = None
        self.web_tracked_body_ids = []
        self.web_tracked_body_handles = {}
        self.web_collision_body_ids = []
        self.web_collision_body_handles = {}
        self.web_reset_button = None
        self.web_toggle_frames_button = None
        self.web_opt_contact_handles = {}
        self.web_opt_contact_positions = {}
        self.web_opt_contact_last_update = 0.0
        self.web_actual_contact_handles = {}
        self.web_actual_contact_body_ids = {}
        self.opt_contact_node = None
        self.opt_contact_subscriber = None
        self.opt_contact_thread: Optional[Thread] = None
        self._rclpy_initialized_here = False

        # Shared lock used by simulation/viewer/web/subscriber threads
        self.locker = threading.Lock()

        if self.web_viewer_backend == "viser":
            self._initialize_web_viewer()
            self._initialize_web_opt_contact_locations()
            self._initialize_web_actual_contact_locations()
            self._initialize_opt_contact_subscriber()

        self.dim_motor_sensor_ = 3 * self.mj_model.nu

        # Other configuration variables
        self.simulate_dt = config["SIMULATION_DT"]
        self.viewer_dt = config["VIEWER_DT"]
        self.domain_id = config["DOMAIN_ID"]
        self.interface = config["INTERFACE"]

        self.print_scene_info = config.get("PRINT_SCENE_INFORMATION", False)

        # Start threads
        self.viewer_thread = Thread(target=self._physics_viewer_thread)
        self.sim_thread = Thread(target=self._simulation_thread)
        self.web_thread = Thread(target=self._web_viewer_thread) if self.web_server is not None else None

        self.viewer_thread.start()
        self.sim_thread.start()
        if self.web_thread is not None:
            self.web_thread.start()

    def _initialize_web_viewer(self):
        if viser is None:
            raise ImportError(
                "WEB_VIEWER_BACKEND=viser requested, but `viser` is not installed. "
                "Install it with: pip install viser"
            )

        self.web_server = viser.ViserServer(
            host=self.web_viewer_host,
            port=self.web_viewer_port,
            label=f"DOOM {self.robot_name}",
        )

        host = self.web_server.get_host()
        port = self.web_server.get_port()
        display_host = "localhost" if host in ("0.0.0.0", "::") else host
        self.web_viewer_url = f"http://{display_host}:{port}"

        self.web_server.scene.set_up_direction("+z")
        self.web_server.scene.add_grid("/world/grid", width=8.0, height=8.0, plane="xy")

        try:
            self.web_reset_button = self.web_server.gui.add_button("Reset Robot")
            self.web_reset_button.on_click(lambda _: self.request_reset())
        except Exception:
            self.web_reset_button = None

        try:
            self.web_toggle_frames_button = self.web_server.gui.add_button("Toggle Frames")
            self.web_toggle_frames_button.on_click(lambda _: self._toggle_web_frames_visibility())
        except Exception:
            self.web_toggle_frames_button = None

        self._initialize_web_body_frames()
        if self.web_viewer_show_collisions:
            self._initialize_web_collision_geometries()

        if self.web_viewer_share_url:
            self.web_viewer_share_link = self.web_server.request_share_url(verbose=False)

    def _initialize_web_opt_contact_locations(self):
        if self.web_server is None or not self.web_viewer_show_opt_contact_locations:
            return

        for foot_idx, foot_name in enumerate(["FL", "FR", "RL", "RR"]):
            handle = self.web_server.scene.add_icosphere(
                f"/robot/opt_contact_locations/{foot_name}",
                radius=self.web_viewer_opt_contact_radius,
                color=self.web_viewer_opt_contact_color,
                position=(0.0, 0.0, 0.0),
                visible=False,
            )
            self.web_opt_contact_handles[foot_idx] = handle

    def _initialize_web_actual_contact_locations(self):
        if self.web_server is None or not self.web_viewer_show_actual_contact_locations:
            return

        for body_name in self.web_viewer_actual_contact_bodies:
            body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                continue

            handle = self.web_server.scene.add_icosphere(
                f"/robot/actual_contact_locations/{body_name}",
                radius=self.web_viewer_actual_contact_radius,
                color=self.web_viewer_actual_contact_color,
                position=(0.0, 0.0, 0.0),
                visible=True,
            )
            self.web_actual_contact_body_ids[body_name] = body_id
            self.web_actual_contact_handles[body_name] = handle

    def _initialize_opt_contact_subscriber(self):
        if not self.web_viewer_show_opt_contact_locations:
            return

        try:
            import rclpy
            from visualization_msgs.msg import MarkerArray

            if not rclpy.ok():
                rclpy.init(args=None)
                self._rclpy_initialized_here = True

            self.opt_contact_node = rclpy.create_node("viser_opt_contact_locations_subscriber")
            self.opt_contact_subscriber = self.opt_contact_node.create_subscription(
                MarkerArray,
                self.web_viewer_opt_contact_topic,
                self._on_opt_contact_locations,
                10,
            )
            self.opt_contact_thread = Thread(target=self._opt_contact_spin_thread, daemon=True)
            self.opt_contact_thread.start()
        except Exception as e:
            print(f"Failed to initialize opt contact subscriber for Viser: {e}", flush=True)

    def _opt_contact_spin_thread(self):
        try:
            import rclpy

            while not self.stop_event.is_set() and self.opt_contact_node is not None:
                rclpy.spin_once(self.opt_contact_node, timeout_sec=0.05)
        except Exception as e:
            print(f"Opt contact spin thread stopped: {e}", flush=True)

    def _on_opt_contact_locations(self, msg):
        positions = {}
        for marker in msg.markers:
            positions[int(marker.id)] = (
                float(marker.pose.position.x),
                float(marker.pose.position.y),
                float(marker.pose.position.z),
            )

        lock = getattr(self, "locker", None)
        if lock is None:
            return

        with lock:
            self.web_opt_contact_positions = positions
            self.web_opt_contact_last_update = time.time()

    def _resolve_web_tracked_body_names(self):
        selected_body_names = []

        if self.web_viewer_tracked_bodies is not None:
            selected_body_names.extend([str(name) for name in self.web_viewer_tracked_bodies])
        else:
            for body_id in range(1, self.mj_model.nbody):
                body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name is None:
                    continue
                lowered_name = body_name.lower()
                if any(token in lowered_name for token in self.web_viewer_auto_track_filters):
                    selected_body_names.append(body_name)

        if self.web_viewer_show_base and self.base_body_id >= 0:
            selected_body_names.insert(0, self.base_link_name)

        unique_names = []
        seen = set()
        for body_name in selected_body_names:
            if body_name in seen:
                continue
            seen.add(body_name)
            unique_names.append(body_name)
        return unique_names

    def _initialize_web_body_frames(self):
        if self.web_server is None:
            return

        selected_body_names = self._resolve_web_tracked_body_names()

        for body_name in selected_body_names:
            body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                continue
            self.web_tracked_body_ids.append(body_id)
            if body_id == self.base_body_id:
                axes_length = self.web_viewer_base_axes_length
                axes_radius = self.web_viewer_base_axes_radius
                frame_path = "/robot/base"
            else:
                axes_length = self.web_viewer_body_axes_length
                axes_radius = self.web_viewer_body_axes_radius
                frame_path = f"/robot/frames/{body_name}"

            self.web_tracked_body_handles[body_id] = self.web_server.scene.add_frame(
                frame_path,
                axes_length=axes_length,
                axes_radius=axes_radius,
                visible=self.web_viewer_show_frames,
            )

    def _set_web_frames_visibility(self, visible: bool):
        self.web_viewer_show_frames = visible
        for body_handle in self.web_tracked_body_handles.values():
            body_handle.visible = visible

    def _toggle_web_frames_visibility(self):
        self._set_web_frames_visibility(not self.web_viewer_show_frames)

    def _should_visualize_collision_geom(self, geom_id):
        geom_contype = int(self.mj_model.geom_contype[geom_id])
        geom_conaffinity = int(self.mj_model.geom_conaffinity[geom_id])
        if geom_contype == 0 and geom_conaffinity == 0:
            return False

        if self.web_viewer_collision_group is not None:
            geom_group = int(self.mj_model.geom_group[geom_id])
            if geom_group != self.web_viewer_collision_group:
                return False

        return True

    def _initialize_web_collision_geometries(self):
        if self.web_server is None:
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
            if geom_body_id not in self.web_collision_body_handles:
                self.web_collision_body_handles[geom_body_id] = self.web_server.scene.add_frame(
                    body_frame_path,
                    show_axes=False,
                )

            if geom_body_id not in added_body_ids:
                self.web_collision_body_ids.append(geom_body_id)
                added_body_ids.add(geom_body_id)

            geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name is None:
                geom_name = f"geom_{geom_id}"

            geom_path = f"{body_frame_path}/{geom_name}_{geom_id}"
            geom_type = int(self.mj_model.geom_type[geom_id])
            geom_pos = tuple(self.mj_model.geom_pos[geom_id].tolist())
            geom_quat = tuple(self.mj_model.geom_quat[geom_id].tolist())
            geom_size = self.mj_model.geom_size[geom_id]

            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                radius = float(geom_size[0])
                self.web_server.scene.add_icosphere(
                    geom_path,
                    radius=radius,
                    color=self.web_viewer_collision_color,
                    opacity=self.web_viewer_collision_opacity,
                    wireframe=self.web_viewer_collision_wireframe,
                    wxyz=geom_quat,
                    position=geom_pos,
                )
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                dimensions = (
                    float(2.0 * geom_size[0]),
                    float(2.0 * geom_size[1]),
                    float(2.0 * geom_size[2]),
                )
                self.web_server.scene.add_box(
                    geom_path,
                    dimensions=dimensions,
                    color=self.web_viewer_collision_color,
                    opacity=self.web_viewer_collision_opacity,
                    wireframe=self.web_viewer_collision_wireframe,
                    wxyz=geom_quat,
                    position=geom_pos,
                )
            elif geom_type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
                radius = float(geom_size[0])
                half_length = float(geom_size[1])
                height = float(2.0 * half_length)
                self.web_server.scene.add_cylinder(
                    geom_path,
                    radius=radius,
                    height=height,
                    color=self.web_viewer_collision_color,
                    opacity=self.web_viewer_collision_opacity,
                    wireframe=self.web_viewer_collision_wireframe,
                    wxyz=geom_quat,
                    position=geom_pos,
                )

    def _simulation_thread(self):
        ChannelFactoryInitialize(self.domain_id, self.interface)
        unitree = UnitreeSdk2Bridge(self.mj_model, self.mj_data, robot=self.robot_name, object=self.object_id)

        self._apply_fixed_base_pose()

        if self.print_scene_info:
            unitree.PrintSceneInformation()

        while not self.stop_event.is_set():
            if self.viewer is not None and not self.viewer.is_running():
                break

            step_start = time.perf_counter()

            with self.locker:
                if self.reset_requested.is_set():
                    self.reset_requested.clear()
                    self._reset_simulation_state()

                if self.elastic_band:
                    if self.elastic_band.enable:
                        force = self.elastic_band.Advance(self.mj_data.qpos[:3], self.mj_data.qvel[:3])
                        self.mj_data.xfrc_applied[self.band_link.id][:3] = force

                mujoco.mj_step(self.mj_model, self.mj_data)

            time_until_next_step = self.mj_model.opt.timestep - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def _physics_viewer_thread(self):
        if self.viewer is None:
            return

        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
        while not self.stop_event.is_set() and self.viewer.is_running():
            with self.locker:
                self._update_mujoco_viewer_camera()

                self.viewer.sync()
            time.sleep(self.viewer_dt)

    def _update_mujoco_viewer_camera(self):
        if self.viewer is None or self.base_body_id < 0 or self.fixed_base_pos is not None:
            return
        robot_pos = self.mj_data.xpos[self.base_body_id]
        self.viewer.cam.lookat[:] = robot_pos
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -20

    def _snapshot_tracked_body_poses(self):
        tracked_body_poses = {}
        tracked_ids = set(self.web_tracked_body_ids)
        tracked_ids.update(self.web_collision_body_ids)

        for body_id in tracked_ids:
            tracked_body_poses[body_id] = (
                np.array(self.mj_data.xpos[body_id], copy=True),
                np.array(self.mj_data.xquat[body_id], copy=True),
            )
        return tracked_body_poses

    def _web_viewer_thread(self):
        if self.web_server is None:
            return

        while not self.stop_event.is_set():
            with self.locker:
                tracked_body_poses = self._snapshot_tracked_body_poses()

            with self.web_server.atomic():
                for body_id, (body_pos, body_quat) in tracked_body_poses.items():
                    body_handle = self.web_tracked_body_handles.get(body_id)
                    if body_handle is not None:
                        body_handle.position = tuple(body_pos.tolist())
                        body_handle.wxyz = tuple(body_quat.tolist())

                    collision_body_handle = self.web_collision_body_handles.get(body_id)
                    if collision_body_handle is not None:
                        collision_body_handle.position = tuple(body_pos.tolist())
                        collision_body_handle.wxyz = tuple(body_quat.tolist())

                if self.web_viewer_show_actual_contact_locations and self.web_actual_contact_handles:
                    for body_name, handle in self.web_actual_contact_handles.items():
                        body_id = self.web_actual_contact_body_ids.get(body_name)
                        if body_id is None:
                            continue
                        handle.position = tuple(self.mj_data.xpos[body_id].tolist())
                        handle.visible = True

                if self.web_viewer_show_opt_contact_locations and self.web_opt_contact_handles:
                    with self.locker:
                        opt_positions = dict(self.web_opt_contact_positions)
                        has_recent_update = (time.time() - self.web_opt_contact_last_update) < 1.0

                    for foot_idx, handle in self.web_opt_contact_handles.items():
                        if has_recent_update and foot_idx in opt_positions:
                            handle.position = opt_positions[foot_idx]
                            handle.visible = True
                        else:
                            handle.visible = False

            time.sleep(self.web_viewer_dt)

    def get_web_viewer_url(self):
        return self.web_viewer_url

    def get_web_viewer_share_url(self):
        return self.web_viewer_share_link

    def request_reset(self):
        """Request a simulation reset on the simulation thread."""
        self.reset_requested.set()

    def _apply_fixed_base_pose(self):
        """Apply fixed base pose constraints if configured."""
        if self.fixed_base_pos is None:
            return

        base_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.base_link_name)
        if base_body_id < 0:
            return

        self.mj_data.qpos[:3] = np.array(self.fixed_base_pos)
        self.mj_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.mj_data.qvel[:3] = np.zeros(3)
        self.mj_data.qvel[3:6] = np.zeros(3)

    def _reset_simulation_state(self):
        """Reset simulation state to model defaults (MuJoCo Backspace equivalent)."""
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self._apply_fixed_base_pose()
        if self.mj_data.ctrl is not None:
            self.mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _viewer_key_callback(self, key):
        """Handle viewer keyboard shortcuts for simulation control."""
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_BACKSPACE:
            self.request_reset()
            return

        if self.elastic_band is not None:
            self.elastic_band.MujuocoKeyCallback(key)

    def stop(self, logger):
        """Gracefully stop the simulation and viewer threads."""
        self.stop_event.set()

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.close()

        if self.web_server is not None:
            self.web_server.stop()

        if self.opt_contact_thread is not None:
            self.opt_contact_thread.join(timeout=1.0)

        if self.opt_contact_node is not None:
            try:
                self.opt_contact_node.destroy_node()
            except Exception:
                pass
            self.opt_contact_node = None

        if self._rclpy_initialized_here:
            try:
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

        self.viewer_thread.join()
        self.sim_thread.join()
        if self.web_thread is not None:
            self.web_thread.join()
        logger.info("Simulation exited successfully.")


class ElasticBand:

    def __init__(self):
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0, 0, 2.5])
        self.length = 0
        self.enable = True

    def Advance(self, x, dx):
        """
        Args:
          δx: desired position - current position
          dx: current velocity
        """
        # self.point[:2] = x[:2]
        δx = self.point - x
        distance = np.linalg.norm(δx)
        direction = δx / distance
        v = np.dot(dx, direction)
        f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return f

    def MujuocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable
