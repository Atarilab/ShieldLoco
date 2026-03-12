# robot_interfaces/sim_robot_interface.py
import os
import threading
from threading import Thread
import time

import mujoco
import mujoco.viewer
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from robot_interfaces.viser_manager import ViserManager
from utils.ball_launcher import BallLauncher
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

        # Shared lock used by simulation/viewer/web/subscriber threads
        self.locker = threading.Lock()
        self.viser_manager = ViserManager(
            config=config,
            robot_name=self.robot_name,
            mj_model=self.mj_model,
            mj_data=self.mj_data,
            locker=self.locker,
            stop_event=self.stop_event,
            request_reset_callback=self.request_reset,
        )
        self.viser_manager.initialize()

        self.dim_motor_sensor_ = 3 * self.mj_model.nu

        # Other configuration variables
        self.simulate_dt = config["SIMULATION_DT"]
        self.viewer_dt = config["VIEWER_DT"]
        self.domain_id = config["DOMAIN_ID"]
        self.interface = config["INTERFACE"]

        self.print_scene_info = config.get("PRINT_SCENE_INFORMATION", False)

        # Ball launcher for dynamic obstacle testing
        self.ball_launcher = BallLauncher(
            mj_model=self.mj_model,
            max_balls=5,
            ball_radius=0.05,
            launch_distance_range=(4.0, 6.0),
            launch_height_range=(2.0, 3.0),
            launch_speed_range=(2.0, 4.0),
            ball_lifetime=3.0,
        )
        self.viser_manager.set_ball_launcher(self.ball_launcher, self.base_body_id)

        # Start threads
        self.viewer_thread = Thread(target=self._physics_viewer_thread)
        self.sim_thread = Thread(target=self._simulation_thread)
        self.web_thread = Thread(target=self.viser_manager.run) if self.viser_manager.has_server() else None

        self.viewer_thread.start()
        self.sim_thread.start()
        if self.web_thread is not None:
            self.web_thread.start()

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

                # Tick ball launcher to auto-deactivate grounded balls
                self.ball_launcher.tick(self.mj_data, time.time())

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

    def get_web_viewer_url(self):
        return self.viser_manager.get_viewer_url()

    def get_web_viewer_share_url(self):
        return self.viser_manager.get_share_url()

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

        # Press 'T' to throw a ball at the robot
        if key == glfw.KEY_T:
            with self.locker:
                robot_pos = self.mj_data.xpos[self.base_body_id].copy()
                self.ball_launcher.launch(robot_pos, self.mj_data)
            return

        if self.elastic_band is not None:
            self.elastic_band.MujuocoKeyCallback(key)

    def stop(self, logger):
        """Gracefully stop the simulation and viewer threads."""
        self.stop_event.set()

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.close()

        self.viser_manager.stop()

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
