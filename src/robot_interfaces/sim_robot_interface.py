# robot_interfaces/sim_robot_interface.py
import os
import threading
from threading import Thread
import time

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
        }
        if use_elastic_band and self.elastic_band is not None:
            self.band_link = self.mj_model.body(self.base_link_name)
            viewer_kwargs["key_callback"] = self.elastic_band.MujuocoKeyCallback

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
        self.web_server = None
        self.web_viewer_url = None
        self.web_viewer_share_link = None
        self.web_base_handle = None
        self.web_object_handle = None

        if self.web_viewer_backend == "viser":
            self._initialize_web_viewer()

        self.dim_motor_sensor_ = 3 * self.mj_model.nu
        self.locker = threading.Lock()

        # Other configuration variables
        self.simulate_dt = config["SIMULATION_DT"]
        self.viewer_dt = config["VIEWER_DT"]
        self.domain_id = config["DOMAIN_ID"]
        self.interface = config["INTERFACE"]

        self.print_scene_info = config.get("PRINT_SCENE_INFORMATION", False)

        # Start threads
        self.viewer_thread = Thread(target=self._physics_viewer_thread)
        self.sim_thread = Thread(target=self._simulation_thread)
        self.web_thread = Thread(target=self._web_viewer_thread)

        self.viewer_thread.start()
        self.sim_thread.start()
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
        self.web_base_handle = self.web_server.scene.add_frame(
            "/robot/base",
            axes_length=0.3,
            axes_radius=0.015,
        )
        if self.object_id >= 0:
            self.web_object_handle = self.web_server.scene.add_icosphere(
                "/world/object",
                radius=0.06,
                color=(255, 140, 0),
            )

        if self.web_viewer_share_url:
            self.web_viewer_share_link = self.web_server.request_share_url(verbose=False)

    def _simulation_thread(self):
        ChannelFactoryInitialize(self.domain_id, self.interface)
        unitree = UnitreeSdk2Bridge(self.mj_model, self.mj_data, robot=self.robot_name, object=self.object_id)

        # Set fixed base position for the robot if configured
        if self.fixed_base_pos is not None:
            base_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.base_link_name)
            if base_body_id >= 0:
                # Set the initial position (x, y, z) for the base link
                self.mj_data.qpos[:3] = np.array(self.fixed_base_pos)
                # Set orientation to unit quaternion (no rotation - identity)
                self.mj_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
                # Reset velocities to zero
                self.mj_data.qvel[:3] = np.zeros(3)
                self.mj_data.qvel[3:6] = np.zeros(3)  # angular velocities

        if self.print_scene_info:
            unitree.PrintSceneInformation()

        while not self.stop_event.is_set():
            if self.viewer is not None and not self.viewer.is_running():
                break

            step_start = time.perf_counter()

            with self.locker:
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
                # Find the robot body (might need adjustment based on your exact model)
                robot_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.base_link_name)
                if robot_body_id >= 0 and self.fixed_base_pos is None:
                    robot_pos = self.mj_data.xpos[robot_body_id]

                    # Set camera lookat point
                    self.viewer.cam.lookat[:] = robot_pos

                    # Optional: soft zoom and angle
                    self.viewer.cam.distance = 3.0
                    self.viewer.cam.elevation = -20

                self.viewer.sync()
            time.sleep(self.viewer_dt)

    def _web_viewer_thread(self):
        if self.web_server is None:
            return

        while not self.stop_event.is_set():
            with self.locker:
                if self.base_body_id >= 0:
                    base_pos = np.array(self.mj_data.xpos[self.base_body_id], copy=True)
                    base_quat = np.array(self.mj_data.xquat[self.base_body_id], copy=True)
                else:
                    base_pos = None
                    base_quat = None

                if self.object_id >= 0:
                    object_pos = np.array(self.mj_data.xpos[self.object_id], copy=True)
                else:
                    object_pos = None

            with self.web_server.atomic():
                if self.web_base_handle is not None and base_pos is not None and base_quat is not None:
                    self.web_base_handle.position = tuple(base_pos.tolist())
                    self.web_base_handle.wxyz = tuple(base_quat.tolist())

                if self.web_object_handle is not None and object_pos is not None:
                    self.web_object_handle.position = tuple(object_pos.tolist())

            time.sleep(self.web_viewer_dt)

    def get_web_viewer_url(self):
        return self.web_viewer_url

    def get_web_viewer_share_url(self):
        return self.web_viewer_share_link

    def stop(self, logger):
        """Gracefully stop the simulation and viewer threads."""
        self.stop_event.set()

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.close()

        if self.web_server is not None:
            self.web_server.stop()

        self.viewer_thread.join()
        self.sim_thread.join()
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
