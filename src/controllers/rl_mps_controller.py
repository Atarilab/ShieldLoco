import threading
import time
from typing import TYPE_CHECKING, Any, Dict

from std_msgs.msg import ColorRGBA, Float32MultiArray, MultiArrayDimension
import torch
from visualization_msgs.msg import Marker, MarkerArray

from commands.command_manager import CommandTerm
from controllers.rl_controller_base import RLControllerBase
from state_manager.obs_manager import ObsTerm

from utils.frequency_tracker import FrequencyTracker

from safety_filter.safety_manager import SafetyManager

if TYPE_CHECKING:
    from robots.robot_base import RobotBase
    from commands.command_manager import CommandManager

import numpy as np

import mujoco


class RLQuadrupedMPSController(RLControllerBase):
    """
    MPS Contact-conditioned RL Locomotion Controller
    Uses contact-explicit reinforcement learning policy
    """

    def __init__(self, robot: "RobotBase", configs: Dict[str, Any], debug: bool = False):
        super().__init__(robot=robot, configs=configs, debug=debug)

        # Initialize publshers for visualization
        self._init_publishers(configs)

        self.policy_stiffness = torch.tensor([self.Kp] * len(self.robot.actuated_joint_names), device=self.device)
        self.policy_damping = torch.tensor([self.Kd] * len(self.robot.actuated_joint_names), device=self.device)
        self.policy_joint_indices = [
            self.robot.actuated_joint_names.index(joint_name) for joint_name in self.robot.actuated_joint_names
        ]

        # Contact command parameters
        self.command_duration = configs["controller_config"]["command_duration"]  # Duration of each contact plan in seconds
        self.pending_command_duration = self.command_duration  # Initialize pending command duration
        self.command_duration_change_pending = False  # Flag to indicate pending command duration change
        self.current_contact_plan = torch.ones((2, 4), dtype=torch.bool, device=self.device)  # Current contact pattern
        self.command_start_time = time.time()  # When the current plan started
        # Thread synchronization for gait changes
        self._gait_lock = threading.RLock()  # Reentrant lock for gait changes
        self._gait_change_event = threading.Event()  # Event to signal gait changes

        # Default offset values ( stride lengths and stance widths would be add to these values )
        self.default_offset = torch.tensor(
            [
                (0.1934, 0.1465, 0.0),
                (0.1934, -0.1465, 0.0),
                (-0.1934, 0.1465, 0.0),
                (-0.1934, -0.1465, 0.0),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.current_offset = self.default_offset.clone()
        self.pending_stance_width_front = self.current_offset[0, 1]
        self.pending_stance_width_rear = self.current_offset[0, 1]
        self.stance_width_change_pending = False

        # Horizon planning
        self.future_feet_positions_init_frame = None
        self.horizon_length = configs["controller_config"]["horizon_length"]
        self.feet_step_size = configs["controller_config"]["feet_step_size"]
        self.base_link = self.robot.base_link
        self.lateral_pos = 0.0
        self.lateral_step_size = 0.0
        self.pending_lateral_step_size = 0.0
        self.lateral_step_size_change_pending = False
        self.pending_heading = 0.0
        self.pending_step_size = 0.0

        self.future_feet_positions_w = torch.zeros(4, self.horizon_length, 3, device=self.device)
        self.reference_feet_positions_w = torch.zeros(4, self.horizon_length, 3, device=self.device)
        self.future_feet_positions_b = torch.zeros(4, self.horizon_length, 3, device=self.device)
        self.apply_mpc_contact_plan = configs["controller_config"].get("apply_mpc_contact_plan", True)
        self.reference_generation_id = 0
        self.reference_step_seq = 0

        # Heading command for feet positions
        self.heading_command = 0.0  # Default heading (in radians)
        self.heading_command_lock = threading.RLock()  # Lock for heading command updates
        self.heading_change_pending = False  # Flag to indicate pending heading change

        # Define gait patterns (FL, FR, RL, RR)
        self.gait_patterns = {
            "transition": torch.tensor(
                [
                    [True, True, True, True],  # Phase 1: All legs in contact
                    [True, True, True, True],  # Phase 2: All legs in contact
                ]
            ),
            "stance": torch.tensor(
                [
                    [True, True, True, True],  # Phase 1: All legs in contact
                    [True, True, True, True],  # Phase 2: All legs in contact
                ]
            ),
            "trot": torch.tensor(
                [
                    [False, True, True, False],  # Phase 2: FR and RL in contact
                    [True, False, False, True],  # Phase 1: FL and RR in contact
                ]
            ),
            "trot-jump1": torch.tensor(
                [
                    [False, True, True, False],  # Phase 2: FR and RL in contact
                    [False, False, False, False],  # Phase 1: FL and RR in contact
                ]
            ),
            "trot-jump2": torch.tensor(
                [
                    [True, False, False, True],  # Phase 2: FR and RL in contact
                    [False, False, False, False],  # Phase 1: FL and RR in contact
                ]
            ),
            "pace": torch.tensor(
                [
                    [True, False, True, False],  # Phase 1: FL and RL in contact
                    [False, True, False, True],  # Phase 2: FR and RR in contact
                ]
            ),
            "pace-jump1": torch.tensor(
                [
                    [False, False, False, False],  # Phase 2: All legs in flight
                    [True, False, True, False],  # Phase 1: FL and RL in contact
                ]
            ),
            "pace-jump2": torch.tensor(
                [
                    [False, False, False, False],  # Phase 2: All legs in flight
                    [False, True, False, True],  # Phase 1: FR and RR in contact
                ]
            ),
            "bound": torch.tensor(
                [
                    [True, True, False, False],  # Phase 1: Front legs in contact
                    [False, False, True, True],  # Phase 2: Back legs in contact
                ]
            ),
            "random1": torch.tensor(
                [
                    [True, False, False, False],
                    [False, True, True, True],
                ]
            ),
            "random2": torch.tensor(
                [
                    [False, True, False, False],
                    [True, False, True, True],
                ]
            ),
            "random3": torch.tensor(
                [
                    [False, False, True, False],
                    [True, True, False, True],
                ]
            ),
            "random4": torch.tensor(
                [
                    [False, False, False, True],
                    [True, True, True, False],
                ]
            ),
            "jump": torch.tensor(
                [
                    [False, False, False, False],  # Phase 2: All legs in flight
                    [True, True, True, True],  # Phase 1: All legs in contact
                ]
            ),
            "crawl-anticlockwise": torch.tensor(
                [
                    [True, True, True, False],  # Phase 1: FL, FR, RL in contact, RR moving
                    [True, True, False, True],  # Phase 2: FL, FR, RR in contact, RL moving
                    [True, False, True, True],  # Phase 3: FL, RL, RR in contact, FR moving
                    [False, True, True, True],  # Phase 4: FR, RL, RR in contact, FL moving
                ]
            ),
            "crawl-anticlockwise2": torch.tensor(
                [
                    [True, True, True, False],  # Phase 1: FL, FR, RL in contact, RR moving
                    [True, True, False, True],  # Phase 2: FL, FR, RR in contact, RL moving
                    [False, True, True, True],  # Phase 4: FR, RL, RR in contact, FL moving
                    [True, False, True, True],  # Phase 3: FL, RL, RR in contact, FR moving
                ]
            ),
            "crawl-clockwise": torch.tensor(
                [
                    [True, True, True, False],  # Phase 2: FL, FR, RL in contact, RR moving
                    [False, True, True, True],  # Phase 1: FR, RL, RR in contact, FL moving
                    [True, False, True, True],  # Phase 4: FL, RL, RR in contact, FR moving
                    [True, True, False, True],  # Phase 3: FL, FR, RR in contact, RL moving
                ]
            ),
            "trot-fly": torch.tensor(
                [
                    [False, True, True, False],  # Phase 2: FR and RL in contact
                    [False, False, False, False],  # Phase 1: FL and RR in contact
                    [True, False, False, True],  # Phase 4: FR and RL in contact
                    [False, False, False, False],  # Phase 3: FL and RR in contact
                ]
            ),
        }
        self.current_gait = "stance"  # Default gait
        self.pending_gait_change = None  # Store pending gait change
        self.in_transition = False  # Flag to indicate if we're in a transition phase
        self.transition_counter = 0  # Counter to track resample steps during transition
        self.transition_duration = configs["controller_config"][
            "transition_duration"
        ]  # Number of resample steps for transition (increased from 2)
        self.transition_progress = 0.0
        self.transition_start_gait = None  # Starting gait for transition
        self.transition_end_gait = None  # Ending gait for transition
        self.time_left = self.command_duration
        self.current_contact_plan = self.gait_patterns[self.current_gait].to(self.device)
        self.current_goal_idx = 0
        self.goal_completion_counter = 0
        self.joint_pos_targets = self.default_joint_pos.clone()

        # TODO
        # # Frequency tracking (logger will be set later in set_cmd_manager)
        # self._frequency_tracker = FrequencyTracker(
        #     name="compute_lowlevelcmd",
        #     log_interval=2.0,  # Reduced for easier testing
        #     logger=self.logger  # Will be updated when logger is available
        # )

        xml_path_rel_to_pkg = "robots/go2/scene_mjx.xml"
        abs_xml_path = '/home/atari/workspace/DOOM/src/' + xml_path_rel_to_pkg
        mj_model = mujoco.MjModel.from_xml_path(abs_xml_path)
        self.safety_manager = SafetyManager(
            mj_model=mj_model,
            rl_policy=self.policy,
            rl_config=configs["controller_config"],
            rate=50,
            logger=self.logger,
        )

    def set_cmd_manager(self, cmd_manager: "CommandManager"):
        """
        Configure command manager for the controller.

        :param obs_manager: Command manager instance
        """
        super().set_cmd_manager(cmd_manager)

        safety_manager = getattr(self, "safety_manager", None)
        if safety_manager is not None:
            safety_manager.set_logger(self.logger)

    def register_observations(self):
        """
        Register observations for contact-conditioned locomotion.
        Includes contact pattern and timing information.
        """
        from state_manager.observations import (
            ang_vel_b,
            contact_locations_b,
            contact_plan,
            contact_time_left,
            ee_pos_rel_b,
            joint_pos_rel,
            joint_vel,
            last_action,
            lin_vel_b,
            projected_gravity_b,
        )

        self.obs_manager.register("lin_vel_b", ObsTerm(lin_vel_b, obs_dim=3, device=self.device))
        self.obs_manager.register("ang_vel_b", ObsTerm(ang_vel_b, obs_dim=3, device=self.device))
        self.obs_manager.register("projected_gravity", ObsTerm(projected_gravity_b, obs_dim=3, device=self.device))
        self.obs_manager.register(
            "contact_locations",
            ObsTerm(
                contact_locations_b,
                obs_dim=24,
                params={
                    "mj_model": self.robot.mj_model,
                    "future_feet_positions_w": lambda: self.future_feet_positions_w,
                    "obs_horizon": 2,
                    "current_goal_idx": lambda: self.current_goal_idx,
                },
                device=self.device,
            ),
        )
        self.obs_manager.register(
            "contact_time_left",
            ObsTerm(
                contact_time_left, params={"contact_time_left": lambda: self.time_left}, obs_dim=1, device=self.device
            ),
        )
        self.obs_manager.register(
            "contact_plan",
            ObsTerm(
                contact_plan,
                params={"contact_plan": lambda: self.current_contact_plan},
                obs_dim=8,
                device=self.device,
            ),
        )
        self.obs_manager.register(
            "joint_pos",
            ObsTerm(
                joint_pos_rel,
                params={
                    "default_joint_pos": self.default_joint_pos,
                    "mapping": self.joint_obs_unitree_to_isaac_mapping,
                },
                obs_dim=12,
                device=self.device,
            ),
        )
        self.obs_manager.register(
            "joint_vel",
            ObsTerm(
                joint_vel, params={"mapping": self.joint_obs_unitree_to_isaac_mapping}, obs_dim=12, device=self.device
            ),
        )
        self.obs_manager.register(
            "ee_pos_rel_b",
            ObsTerm(
                ee_pos_rel_b,
                params={
                    "mj_model": self.robot.mj_model,
                    "future_feet_positions_w": lambda: self.future_feet_positions_w,
                    "current_goal_idx": lambda: self.current_goal_idx,
                },
                obs_dim=4,
                device=self.device,
            ),
        )
        self.obs_manager.register(
            "last_action",
            ObsTerm(
                last_action, params={"last_action": lambda: self.action_term.raw_action}, obs_dim=12, device=self.device
            ),
        )

    def register_commands(self):
        """Register gait selection as button group."""

        self.command_manager.register_button_command(
            name="gait",
            description="Select Gait Pattern",
            options=["stance", "trot"],
            default_value="stance",
        )
        self.command_manager.register_button_command(
            name="move",
            description="Move Robot",
            options=["up", "down", "left", "center", "right", "turn-left", "turn-right"],
            default_value="center",
        )
        self.command_manager.register_button_command(
            name="mpc",
            description="MPC Contact Plan",
            options=["off", "on"],
            default_value="on" if self.apply_mpc_contact_plan else "off",
        )

    def set_mode(self):
        """Runs when the mode is changed in the UI.
        Generates the future feet positions in the init frame for horizon planning.
        """
        # Call the base class set_mode to activate the controller
        super().set_mode()

        # Set default gait to stance when switching back to RL controller
        with self._gait_lock:
            self.current_gait = "stance"
            self.current_contact_plan = self.gait_patterns["stance"]

            self.pending_gait_change = None
            self.in_transition = False
            self.transition_counter = 0
            self.transition_progress = 0.0
            self.transition_start_gait = None
            self.transition_end_gait = None
            self.feet_step_size = 0.0
            self.command_duration = 0.35
            self.heading_command = 0.0
            self.lateral_step_size = 0.0
            self.pending_lateral_step_size = 0.0
            self.lateral_step_size_change_pending = False

        base_pos_w = torch.tensor(
            self.robot.mj_model.get_body_position_world("base_link"), dtype=torch.float32, device=self.device
        )
        self.lateral_pos = base_pos_w[1]
        self.generate_future_feet_positions(pos=base_pos_w)

        self.joint_pos_targets = self.default_joint_pos.clone()

        self.safety_manager.set_logger(self.logger)

        self.safety_manager.start()

        # TODO: see what can be done here
        # # update when manager starts running
        # while not self.safety_manager.is_running():
        #     time.sleep(0.1)  # Small delay to avoid busy waiting
        #     print("Waiting for safety manager to start...")

        # self.safety_manager.update_state(
        #     ref=self.future_feet_positions_w.detach().cpu().numpy(),
        #     qpos=self.robot.mj_model.data.qpos,
        #     qvel=self.robot.mj_model.data.qvel,
        #     time=time.time(),

        #     command_start_time=self.command_start_time,
        #     goal_idx=self.current_goal_idx,
        #     contact_plan=self.current_contact_plan.detach().cpu().numpy(),
        #     current_gait=self.current_gait,
        #     goal_completion_counter=self.goal_completion_counter,
        #     in_transition=self.in_transition,
        # )

    def compute_lowlevelcmd(self, state):
        """
        Compute motor commands using the learned policy.

        :param state: Current robot state
        :return: Motor commands dictionary
        """
        try:
            self.counter += 1

            if self.robot.mj_model is not None:
                self.robot.mj_model.update(state)

            start_time = time.perf_counter()

            self.update_state(state)

            # Update lateral_pos from current robot position
            if self.robot.mj_model is not None:
                base_pos_w = self.robot.mj_model.get_body_position_world(self.base_link)
                self.lateral_pos = base_pos_w[1]

            # Update contact plan timing with thread safety
            with self._gait_lock:
                current_time = time.time()
                elapsed = current_time - self.command_start_time
                self.time_left = max(0, self.command_duration - elapsed)
                if elapsed >= self.command_duration:
                    self.command_start_time = current_time
                    self._resample_commands()
                    # # self._update_desired_ee_positions()
                    # self._update_action_scale()

            # update state
            if self.safety_manager.is_running():
                safety_gait_id = {
                    "stance": 0,
                    "transition": 1,
                    "trot": 2,
                    "pace": 3,
                    "bound": 4,
                    "jump": 5,
                }.get(self.current_gait, 0)
                self.safety_manager.update_state(
                    ref=self.reference_feet_positions_w.detach().cpu().numpy(),
                    qpos=self.robot.mj_model.data.qpos,
                    qvel=self.robot.mj_model.data.qvel,
                    xpos=self.robot.mj_model.data.xpos,
                    xquat=self.robot.mj_model.data.xquat,
                    time=current_time,
                    reference_generation_id=self.reference_generation_id,
                    reference_step_seq=self.reference_step_seq,
                    goal_idx=self.current_goal_idx,
                    goal_completion_counter=self.goal_completion_counter,
                    contact_plan=self.current_contact_plan.detach().cpu().numpy(),
                    current_gait=safety_gait_id,
                    in_transition=self.in_transition,
                )

                plan_info = self.safety_manager.get_latest_plan_info()
                if plan_info is not None:
                    opt_future_feet_positions_w = torch.tensor(
                        plan_info["plan"], dtype=torch.float32, device=self.device
                    )
                    plan_generation_id = plan_info.get("reference_generation_id")
                    plan_step_seq = plan_info.get("reference_step_seq")

                    if self.visualize["opt_contact_locations"]:
                        self.pub_opt_contact_locations(opt_future_feet_positions_w)

                    if plan_generation_id == self.reference_generation_id:
                        with self._gait_lock:
                            aligned_plan = self._align_optimized_contact_plan(
                                opt_future_feet_positions_w, plan_step_seq
                            )
                            if aligned_plan is not None:
                                self._apply_optimized_contact_plan(aligned_plan)
                    elif self.logger is not None:
                        self.logger.debug(
                            "Discarded obsolete MPC contact plan generation: "
                            f"plan_generation_id={plan_generation_id}, current_generation_id={self.reference_generation_id}"
                        )

                # Publish COM trajectory from train_data
                if self.visualize["opt_contact_locations"]:
                    train_data = self.safety_manager.get_train_data()
                    if train_data is not None and "com" in train_data:
                        com_traj = train_data["com"]  # shape: (num_traj, horizon+1, 3)
                                                
                        self.pub_com_trajectory(com_traj)
                    if train_data is not None and "qpos" in train_data:
                        self.pub_qpos_trajectory(train_data["qpos"])

                    # if train_data is not None and "constraint_flags" in train_data:
                    #     violated = train_data["constraint_flags"].any()
                    #     self.logger.debug(f"Constraint Violated: {violated}")

            if self.visualize["current_contact_locations"]:
                self.pub_current_contact_locations()

            if self.visualize["foot_forces"]:
                self.pub_foot_forces(self.latest_state["robot/foot_forces"])
                self.pub_contact_status(self.latest_state["robot/foot_forces"])
            if self.visualize["contact_plan"]:
                self.pub_contact_plan()

            # self.logger.debug(f"Time left: {self.time_left:.2f} seconds")

            # If we do not use threading, we need to compute the obs first, pass it to the policy, and then compute the joint pos targets
            # else we can compute the joint pos targets directly from the obs tensor

            if self.counter % self.decimation == 0:
                if not self.use_threading:
                    obs_tensor = self.obs_manager.compute_full_tensor(self.latest_state, batch_idx=0)
                    self.joint_pos_targets = self.compute_joint_pos_targets_from_policy(obs_tensor)
                else:
                    self.joint_pos_targets = self.compute_joint_pos_targets()

            # Clip the joint pos targets for safety
            if hasattr(self, "soft_dof_pos_limit"):
                self.joint_pos_targets = self._clip_dof_pos(self.joint_pos_targets)

            # Prepare motor commands
            self.cmd = {
                f"motor_{i}": {
                    "q": self.joint_pos_targets[i],
                    "kp": self.Kp,
                    "dq": 0.0,
                    "kd": self.Kd,
                    "tau": 0.0,
                }
                for i in range(self.robot.num_joints)
            }

            # Track command preparation time
            self.cmd_preparation_time = time.perf_counter() - start_time

            return self.cmd

        except Exception as e:
            self.logger.error(f"Error computing low level commands: {e}. Defaulting to default joint positions.")
            self.cmd = {
                f"motor_{i}": {
                    "q": self.default_joint_pos[i],
                    "kp": 0.0,
                    "dq": 0.0,
                    "kd": 0.0,
                    "tau": 0.0,
                }
                for i in range(self.robot.num_joints)
            }
            return self.cmd

    def _resample_commands(self):
        """Resample the commands for the controller with thread safety."""
        try:
            with self._gait_lock:
                # Apply pending step size change if any

                if hasattr(self, "step_size_change_pending") and self.step_size_change_pending:
                    self.feet_step_size = self.pending_step_size
                    self.step_size_change_pending = False
                    if self.logger is not None:
                        self.logger.debug(f"Applied step size change: {self.feet_step_size:.2f} meters")
                    # Regenerate future feet positions with new step size
                    avg_base_xy = self.future_feet_positions_w[:, self.current_goal_idx].mean(dim=0).clone()
                    self.generate_future_feet_positions(pos=avg_base_xy)

                # Apply pending heading change if any
                if self.heading_change_pending:
                    self.heading_command = self.pending_heading
                    self.heading_change_pending = False
                    if self.logger is not None:
                        self.logger.debug(f"Applied heading change: {self.heading_command:.2f} radians")
                    # Regenerate future feet positions with new heading
                    self.generate_future_feet_positions()

                if self.lateral_step_size_change_pending:
                    self.lateral_step_size = self.pending_lateral_step_size
                    self.lateral_step_size_change_pending = False
                    if self.logger is not None:
                        self.logger.debug(f"Applied lateral step size change: {self.lateral_step_size:.2f} meters")
                    # Regenerate future feet positions with new lateral step size
                    avg_base_xy = self.future_feet_positions_w[:, self.current_goal_idx].mean(dim=0).clone()
                    self.generate_future_feet_positions(pos=avg_base_xy)

                # Apply pending command duration change if any
                if self.command_duration_change_pending:
                    self.command_duration = self.pending_command_duration
                    self.command_duration_change_pending = False
                    if self.logger is not None:
                        self.logger.debug(f"Applied command duration change: {self.command_duration:.2f} seconds")

                # Apply pending offset change if any
                if self.stance_width_change_pending:
                    self.current_offset[0, 1] = self.pending_stance_width_front
                    self.current_offset[1, 1] = -self.pending_stance_width_front
                    self.current_offset[2, 1] = self.pending_stance_width_rear
                    self.current_offset[3, 1] = -self.pending_stance_width_rear

                    self.stance_width_change_pending = False
                    if self.logger is not None:
                        self.logger.debug(f"Applied stance width change: {self.current_offset}")
                    # Regenerate future feet positions with new stance width
                    avg_base_xy = self.future_feet_positions_w[:, self.current_goal_idx].mean(dim=0).clone()

                    self.generate_future_feet_positions(pos=avg_base_xy)

                # Handle gait transitions
                if self.pending_gait_change is not None:
                    if not self.in_transition:
                        # Start transition phase
                        self.in_transition = True
                        self.transition_counter = 0
                        self.transition_progress = 0.0
                        self.transition_start_gait = self.current_gait
                        self.transition_end_gait = self.pending_gait_change

                        # For safety, start with a stable stance if transitioning to a dynamic gait
                        if self.pending_gait_change in [
                            gait for gait in self.gait_patterns.keys() if gait not in ["transition", "stance"]
                        ]:
                            self.current_gait = "transition"
                            self.current_contact_plan = self.gait_patterns["transition"]

                        if self.transition_duration == 0:
                            self.in_transition = False
                            self.current_gait = self.pending_gait_change
                            self.current_contact_plan = self.gait_patterns[self.pending_gait_change]

                            # Call set_mode() if the pending gait is "stance"
                            if self.pending_gait_change == "stance":
                                avg_base_xy = self.future_feet_positions_w[:, self.current_goal_idx].mean(dim=0).clone()
                                self.generate_future_feet_positions(pos=avg_base_xy)

                            self.pending_gait_change = None
                            self.transition_start_gait = None
                            self.transition_end_gait = None

                            # Signal that the gait change is complete
                            self._gait_change_event.set()

                            self.logger.debug(f"Transition complete, applied gait: {self.current_gait}")

                            # Signal that the gait change is complete
                    else:
                        # Increment transition counter and update progress
                        self.transition_counter += 1
                        self.transition_progress = min(1.0, self.transition_counter / self.transition_duration)

                        # Only apply the pending gait after transition duration
                        if self.transition_counter >= self.transition_duration:
                            # Transition phase complete - apply the pending gait
                            self.in_transition = False
                            self.current_gait = self.pending_gait_change
                            self.current_contact_plan = self.gait_patterns[self.pending_gait_change]

                            # Call set_mode() if the pending gait is "stance"
                            if self.pending_gait_change == "stance":
                                avg_base_xy = self.future_feet_positions_w[:, self.current_goal_idx].mean(dim=0).clone()
                                self.generate_future_feet_positions(pos=avg_base_xy)

                            self.pending_gait_change = None
                            self.transition_start_gait = None
                            self.transition_end_gait = None

                            # Signal that the gait change is complete
                            self._gait_change_event.set()

                            self.logger.debug(f"Transition complete, applied gait: {self.current_gait}")

                # Normal gait progression (only if not in transition phase)
                if not self.in_transition and self.current_gait not in ["stance", "transition"]:
                    self.goal_completion_counter += 1
                    should_advance_reference = False

                    # Handle different gait phases
                    if "crawl" in self.current_gait:
                        # For crawl gait or trot fly, cycle through 4 phases
                        should_advance_reference = self.goal_completion_counter % 4 == 0
                        self.current_contact_plan = torch.stack(
                            [
                                self.gait_patterns[self.current_gait][(self.goal_completion_counter) % 4],
                                self.gait_patterns[self.current_gait][(self.goal_completion_counter + 1) % 4],
                            ]
                        )
                    # elif self.current_gait == "jump":
                    #     self.current_goal_idx += 1 if (self.goal_completion_counter) % 4 == 0 else 0
                    #     self.current_contact_plan = torch.stack(
                    #         [
                    #             self.current_contact_plan[1],
                    #             self.current_contact_plan[0],
                    #         ]
                    #     )

                    else:
                        # For other gaits (2 phases), alternate between phases
                        should_advance_reference = (
                            self.goal_completion_counter % 2 != 0 and self.goal_completion_counter > 0
                        )
                        self.current_contact_plan = torch.stack(
                            [
                                self.current_contact_plan[1],
                                self.current_contact_plan[0],
                            ]
                        )

                    if should_advance_reference:
                        self._advance_future_feet_positions()

        except Exception as e:
            if hasattr(self, "command_manager") and self.command_manager and hasattr(self.command_manager, "logger"):
                self.logger.error(f"Error resampling commands: {e}")
            else:
                print(f"Error resampling commands: {e}")

    def _get_reference_motion_terms(self):
        offset = self.current_offset.clone()

        with self.heading_command_lock:
            target_yaw = torch.tensor(self.heading_command, dtype=torch.float32, device=self.device)

        cos_yaw = torch.cos(target_yaw)
        sin_yaw = torch.sin(target_yaw)
        rotation_matrix = torch.tensor(
            [[cos_yaw.item(), -sin_yaw.item(), 0], [sin_yaw.item(), cos_yaw.item(), 0], [0, 0, 1]],
            dtype=torch.float32,
            device=self.device,
        )
        rotated_offset = torch.matmul(rotation_matrix, offset.T).T

        stride_delta = self.feet_step_size * self.command_duration
        lateral_delta = self.lateral_step_size * self.command_duration
        step_delta = torch.stack(
            [
                stride_delta * cos_yaw - lateral_delta * sin_yaw,
                stride_delta * sin_yaw + lateral_delta * cos_yaw,
                torch.tensor(0.0, dtype=torch.float32, device=self.device),
            ]
        )

        return rotated_offset, step_delta

    def _append_future_step(self):
        _, step_delta = self._get_reference_motion_terms()
        next_step = self.reference_feet_positions_w[:, -1, :] + step_delta.unsqueeze(0)
        next_step[:, 2] = 0.01
        self.reference_feet_positions_w[:, -1, :] = next_step

    def _copy_reference_to_active_horizon(self):
        self.future_feet_positions_w[:] = self.reference_feet_positions_w

    def _apply_optimized_contact_plan(self, optimized_plan: torch.Tensor):
        if not self.apply_mpc_contact_plan:
            return

        if tuple(optimized_plan.shape) != tuple(self.future_feet_positions_w.shape):
            if self.logger is not None:
                self.logger.error(
                    "Ignoring optimized contact plan with unexpected shape: "
                    f"got {tuple(optimized_plan.shape)}, expected {tuple(self.future_feet_positions_w.shape)}"
                )
            return

        self.future_feet_positions_w[:] = optimized_plan

    def _align_optimized_contact_plan(
        self, optimized_plan: torch.Tensor, plan_step_seq: int | None
    ) -> torch.Tensor | None:
        if plan_step_seq is None:
            return optimized_plan

        stale_steps = int(self.reference_step_seq) - int(plan_step_seq)
        if stale_steps < 0:
            stale_steps = 0

        if stale_steps == 0:
            return optimized_plan

        if stale_steps >= self.horizon_length:
            if self.logger is not None:
                self.logger.debug(
                    "Discarded stale MPC contact plan beyond horizon: "
                    f"plan_step_seq={plan_step_seq}, current_step_seq={self.reference_step_seq}"
                )
            return None

        aligned_plan = optimized_plan.clone()
        aligned_plan[:, :-stale_steps, :] = optimized_plan[:, stale_steps:, :]
        aligned_plan[:, -stale_steps:, :] = self.reference_feet_positions_w[:, -stale_steps:, :]

        if self.logger is not None:
            self.logger.debug(
                "Applied delayed MPC contact plan with horizon shift: "
                f"plan_step_seq={plan_step_seq}, current_step_seq={self.reference_step_seq}, stale_steps={stale_steps}"
            )

        return aligned_plan

    def _bump_reference_generation(self):
        self.reference_generation_id += 1
        self.reference_step_seq = 0

    def _bump_reference_step_seq(self):
        self.reference_step_seq += 1

    def _advance_horizon_buffer(
        self, buffer: torch.Tensor, tail_value: torch.Tensor | None = None
    ):
        if self.horizon_length <= 0:
            return

        if self.horizon_length > 1:
            buffer[:, :-1, :] = buffer[:, 1:, :].clone()

        if tail_value is None:
            _, step_delta = self._get_reference_motion_terms()
            next_step = buffer[:, -1, :] + step_delta.unsqueeze(0)
            next_step[:, 2] = 0.01
            buffer[:, -1, :] = next_step
        else:
            buffer[:, -1, :] = tail_value

    def _advance_future_feet_positions(self):
        self._advance_horizon_buffer(self.reference_feet_positions_w)
        self._advance_horizon_buffer(
            self.future_feet_positions_w,
            tail_value=self.reference_feet_positions_w[:, -1, :].clone(),
        )
        self.current_goal_idx = 0
        self._bump_reference_step_seq()

    def generate_future_feet_positions(self, pos=None):
        """
        Generates future feet positions for the controller.
        Uses the heading_command to determine the direction of movement.
        """
        if pos is None:
            pos = torch.zeros(3, device=self.device)

        self.goal_completion_counter = 0
        self.current_goal_idx = 0

        # Check if pos is a tensor of zeros (default case)
        if torch.all(pos == 0):
            # Get the robot's current position
            robot_pos = torch.tensor(
                self.robot.mj_model.get_body_position_world(self.base_link), dtype=torch.float32, device=self.device
            )
            # Use robot's actual current X and Y positions
            pos = torch.tensor([robot_pos[0], robot_pos[1], 0.0], dtype=torch.float32, device=self.device)

        pos = torch.as_tensor(pos, dtype=torch.float32, device=self.device)
        rotated_offset, step_delta = self._get_reference_motion_terms()

        step_indices = torch.arange(self.horizon_length, dtype=torch.float32, device=self.device).view(1, -1, 1)
        self.reference_feet_positions_w[:] = pos.view(1, 1, 3) + rotated_offset.unsqueeze(1)
        self.reference_feet_positions_w[:] += step_indices * step_delta.view(1, 1, 3)

        # self.future_feet_positions_w[:, :, 2] = -0.02 if self.interface == "real" else 0.08

        # TODO BUG: should be what the terrain height or something
        self.reference_feet_positions_w[:, :, 2] = 0.01
        self._copy_reference_to_active_horizon()
        self._bump_reference_generation()

        # if self.current_gait in ["trot", "pace"]:
        #     self.future_feet_positions_w[[1, 2], :, 0] -= (self.feet_step_size / 2) * direction_x
        #     self.future_feet_positions_w[[1, 2], :, 1] -= (self.feet_step_size / 2) * direction_y

        # if self.current_gait in ["pace"]:
        #     self.future_feet_positions_w[[0, 3], :, 0] += 0.12 * direction_x
        #     self.future_feet_positions_w[[0, 3], :, 1] += 0.12 * direction_y
        # self.future_feet_positions_init_frame = self.robot.mj_model.transform_world_to_init_frame(
        #     self.future_feet_positions_w.numpy()
        # )

        if self.visualize["future_feet_positions"]:
            self.pub_future_feet_positions()

    """
    Joystick mappings and callbacks        
    """

    def get_joystick_mappings(self):
        """
        Define joystick button mappings for gait changes and heading control.

        Returns:
            Dict mapping button names to callback functions.
        """
        return {
            # Reset to stance
            "L2-R2": lambda: self.change_commands({"gait": "stance"}),
            # Normal Gaits
            "A": lambda: self.change_commands({"gait": "trot"}),
            "B": lambda: self.change_commands({"gait": "pace"}),
            "X": lambda: self.change_commands({"gait": "bound"}),
            "Y": lambda: self.change_commands({"gait": "jump"}),
            # Crawl Gaits
            "L2-Y": lambda: self.change_commands({"gait": "crawl-anticlockwise"}),
            "L2-A": lambda: self.change_commands({"gait": "crawl-clockwise"}),
            # Pace-Jump Gaits
            "R1-X": lambda: self.change_commands({"gait": "pace-jump1"}),
            "R1-B": lambda: self.change_commands({"gait": "pace-jump2"}),
            # Trot-Jump Gaits
            "R1-Y": lambda: self.change_commands({"gait": "trot-jump1"}),
            "R1-A": lambda: self.change_commands({"gait": "trot-jump2"}),
            # "L1-X": lambda: self.change_commands({"gait": "trot-fly"}),
            # OOD Gaits
            "R2-Y": lambda: self.change_commands({"gait": "random1"}),
            "R2-X": lambda: self.change_commands({"gait": "random2"}),
            "R2-A": lambda: self.change_commands({"gait": "random3"}),
            "R2-B": lambda: self.change_commands({"gait": "random4"}),
            # Step Size
            "up": lambda: self._handle_step_size_change("up"),
            "down": lambda: self._handle_step_size_change("down"),
            # Lateral Position
            "left": lambda: self._handle_lateral_pos_change("left"),
            "right": lambda: self._handle_lateral_pos_change("right"),
            # Command Duration
            "L1-up": lambda: self._handle_command_duration_change("increase"),
            "L1-down": lambda: self._handle_command_duration_change("decrease"),
            "L1-R1": lambda: self._handle_reset(),
            # Offset Adjustment
            "L1-left": lambda: self._handle_stance_width_change("front", "decrease"),
            "L1-right": lambda: self._handle_stance_width_change("front", "increase"),
            "R1-left": lambda: self._handle_stance_width_change("rear", "decrease"),
            "R1-right": lambda: self._handle_stance_width_change("rear", "increase"),
        }

    """
    Handler functions for changing the contact commands for the rl-contact-locomotion controller
    - gait, step size, heading, lateral position, command duration, stance width
    """

    def change_commands(self, new_commands: Dict[str, Any]):
        """Change the robot's contact commands with thread safety.

        This method handles changes to the robot's gait pattern. When a new gait is requested,
        it is stored as a pending change rather than applied immediately. The actual gait
        transition happens during the next resampling phase to ensure smooth transitions.

        Args:
            new_commands: Dictionary containing command updates. Currently supports:
                - 'gait': String specifying the new gait pattern (e.g. 'trot', 'pace', etc.)

        Raises:
            ValueError: If an invalid gait pattern is specified
        """
        try:
            if "mpc" in new_commands:
                mpc_mode = str(new_commands["mpc"]).lower()
                if mpc_mode in ["off", "on"]:
                    with self._gait_lock:
                        self.apply_mpc_contact_plan = mpc_mode == "on"
                        if not self.apply_mpc_contact_plan:
                            self._copy_reference_to_active_horizon()

                    if self.logger is not None:
                        self.logger.info(
                            "MPC contact plan "
                            f"{'enabled' if self.apply_mpc_contact_plan else 'disabled'}"
                        )

            if "gait" in new_commands:
                new_gait = new_commands["gait"].lower()
                if new_gait in self.gait_patterns and new_gait != self.current_gait:
                    with self._gait_lock:
                        # Only set pending change if it's different from current gait
                        # and we're not already in a transition to this gait
                        if self.pending_gait_change != new_gait:
                            self.pending_gait_change = new_gait
                            self._gait_change_event.clear()  # Reset event for new change

                            # If we're already in a transition, reset it to start fresh
                            if self.in_transition:
                                self.transition_counter = 0
                                self.transition_progress = 0.0
                                self.transition_start_gait = self.current_gait
                                self.transition_end_gait = new_gait

            if "move" in new_commands:
                move_command = str(new_commands["move"]).lower()
                if move_command in ["up", "down"]:
                    self._handle_step_size_change(move_command)
                elif move_command == "center":
                    with self._gait_lock:
                        # Reset current values immediately
                        self.feet_step_size = 0.0
                        self.lateral_step_size = 0.0
                        # Also reset pending values
                        self.pending_step_size = 0.0
                        self.step_size_change_pending = True
                        self.pending_lateral_step_size = 0.0
                        self.lateral_step_size_change_pending = True
                        # Don't reset heading - keep current heading
                elif move_command in ["left", "right"]:
                    self._handle_lateral_pos_change(move_command)
                elif move_command == "turn-left":
                    self._handle_heading_change("left")
                elif move_command == "turn-right":
                    self._handle_heading_change("right")

        except Exception as e:
            self.logger.error(f"Contact command update failed: {e}")

    def _handle_step_size_change(self, direction: str):
        """Handle forwardstep size changes."""
        new_step_size = self._compute_discrete_step(
            current_value=self.feet_step_size,
            direction=direction,
            step=0.2,
            minimum=-0.6,
            maximum=0.6,
        )

        # Set the pending step size change instead of applying it immediately
        with self._gait_lock:
            self.pending_step_size = new_step_size
            self.step_size_change_pending = True

    def _handle_heading_change(self, direction: str):
        """
        Handle heading changes using directional buttons.

        Args:
            direction: String indicating the direction to turn ("left" or "right")
        """
        heading_change = 0.05  # radians per press

        with self._gait_lock:
            if direction == "left":
                self.pending_heading = self.heading_command + heading_change
            elif direction == "right":
                self.pending_heading = self.heading_command - heading_change

            # Wrap heading to [-pi, pi] to prevent unbounded growth
            self.pending_heading = (self.pending_heading + np.pi) % (2 * np.pi) - np.pi

            self.heading_change_pending = True

    def _handle_lateral_pos_change(self, direction: str):
        """Handle lateral step size changes."""
        with self._gait_lock:
            self.pending_lateral_step_size = self._compute_discrete_step(
                current_value=self.lateral_step_size,
                direction=direction,
                step=0.2,
                minimum=-0.6,
                maximum=0.6,
            )
            self.lateral_step_size_change_pending = True

    @staticmethod
    def _compute_discrete_step(current_value: float, direction: str, step: float, minimum: float, maximum: float) -> float:
        """Compute one discrete command step with clamping."""
        if direction in ["up", "left"]:
            updated = current_value + step
        elif direction in ["down", "right"]:
            updated = current_value - step
        else:
            updated = current_value
        return max(minimum, min(updated, maximum))

    def _handle_command_duration_change(self, direction: str):
        """
        Handle command duration changes.

        Args:
            direction: String indicating whether to increase or decrease the duration
        """
        # Define duration change amount in seconds
        duration_change = 0.1  # 50ms change

        with self._gait_lock:
            if direction == "increase":
                # Increase command duration
                self.pending_command_duration = min(1.0, self.command_duration + duration_change)
            elif direction == "decrease":
                # Decrease command duration (with a minimum value)
                self.pending_command_duration = max(0.1, self.command_duration - duration_change)
            elif direction == "default":
                self.pending_command_duration = 0.35

            # Set the pending flag
            self.command_duration_change_pending = True

            if self.logger is not None:
                self.logger.debug(f"Pending command duration change: {self.pending_command_duration:.2f} seconds")

    def _handle_stance_width_change(self, side: str, direction: str):
        """Handle offset changes."""
        # Define offset change amount
        stance_width_change = 0.2 

        with self._gait_lock:
            if direction == "increase":
                # Increase offset values
                if side == "front":
                    self.pending_stance_width_front = (
                        self.current_offset[0, 1] + stance_width_change
                    )  # Modify y-coordinate (stance width)
                elif side == "rear":
                    self.pending_stance_width_rear = (
                        self.current_offset[2, 1] + stance_width_change
                    )  # Modify y-coordinate (stance width)
            elif direction == "decrease":
                # Decrease offset values
                if side == "front":
                    self.pending_stance_width_front = (
                        self.current_offset[0, 1] - stance_width_change
                    )  # Modify y-coordinate (stance width)
                elif side == "rear":
                    self.pending_stance_width_rear = (
                        self.current_offset[2, 1] - stance_width_change
                    )  # Modify y-coordinate (stance width)

            # Set bounds for stance width (0.1m to 0.3m)
            self.pending_stance_width_front = max(0.05, min(0.3, self.pending_stance_width_front))
            self.pending_stance_width_rear = max(0.05, min(0.3, self.pending_stance_width_rear))

            # Set the pending flag
            self.stance_width_change_pending = True

            if self.logger is not None:
                self.logger.debug(
                    f"Pending stance width change: {self.pending_stance_width_front:.3f} meters (front) and {self.pending_stance_width_rear:.3f} meters (rear)"
                )

    def _handle_reset(self):
        """Reset command duration and offset values to default."""
        with self._gait_lock:
            # Reset command duration
            self.pending_command_duration = 0.35
            self.command_duration_change_pending = True

            # Reset offset values
            self.pending_stance_width_front = self.default_offset[0, 1]
            self.pending_stance_width_rear = self.default_offset[0, 1]
            self.stance_width_change_pending = True

            if self.logger is not None:
                self.logger.debug("Reset command duration and stance width values to default")

    """
    Helper functions to initialize publishers for visualization.
    """

    def _init_publishers(self, configs):
        """
        Initialize publishers for visualization.
        """
        # Initialize visualization flags from configs
        self.visualize = {
            key: configs["controller_config"]["visualize"][key] for key in configs["controller_config"]["visualize"]
        }

        if self.visualize["future_feet_positions"]:
            self.feet_trajectory_pub = self.create_publisher(MarkerArray, "feet_trajectories", 10)
        if self.visualize["current_contact_locations"]:
            self.contact_locations_pub = self.create_publisher(MarkerArray, "contact_locations", 10)
        if self.visualize["opt_contact_locations"]:
            self.opt_contact_locations_pub = self.create_publisher(MarkerArray, "opt_contact_locations", 10)
            self.com_trajectory_pub = self.create_publisher(MarkerArray, "com_trajectory", 10)
            self.qpos_trajectory_pub = self.create_publisher(Float32MultiArray, "qpos_trajectory", 10)
        if self.visualize["feet_error"]:
            self.feet_error_pub = self.create_publisher(MarkerArray, "feet_error", 10)
            self.feet_error_norm_pub = self.create_publisher(Float32MultiArray, "feet_error_norm", 10)
        if self.visualize["contact_plan"]:
            self.contact_plan_pub = self.create_publisher(Float32MultiArray, "contact_plan", 10)
        if self.visualize["foot_forces"]:
            self.foot_forces_pub = self.create_publisher(Float32MultiArray, "foot_forces", 10)
        if self.visualize["contact_status"]:
            self.contact_status_pub = self.create_publisher(Float32MultiArray, "contact_status", 10)

    def pub_feet_error(self):
        """
        Visualizes the feet error in the world frame.
        """
        try:
            # Publish feet locations
            feet_errors_msg = MarkerArray()
            ee_names = ["FL", "FR", "RL", "RR"]
            colors = [
                ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),  # Red for FL
                ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),  # Green for FR
                ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # Blue for RL
                ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),  # Yellow for RR
            ]

            # Get feet positions from the state manager
            feet_errors = (
                self.robot.mj_model.get_ee_positions_w()
                - self.future_feet_positions_w[:, self.current_goal_idx].numpy()
            )
            if feet_errors is not None:
                for i, (name, color) in enumerate(zip(ee_names, colors)):
                    marker = Marker()
                    marker.header.frame_id = "world"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = "feet_locations"
                    marker.id = i
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD

                    # Set the scale of the sphere
                    marker.scale.x = 0.05  # radius
                    marker.scale.y = 0.05
                    marker.scale.z = 0.05

                    # Set color
                    marker.color = color

                    # Set position
                    marker.pose.position.x = float(feet_errors[i][0])
                    marker.pose.position.y = float(feet_errors[i][1])
                    marker.pose.position.z = float(feet_errors[i][2])

                    # Set orientation (identity quaternion)
                    marker.pose.orientation.w = 1.0
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0

                    feet_errors_msg.markers.append(marker)

            # Publish the marker array
            self.feet_error_pub.publish(feet_errors_msg)
        except Exception as e:
            self.logger.error(f"Failed to publish feet error: {e}")

    def pub_future_feet_positions(self):
        """
        Visualizes the future feet positions in the world frame.
        """
        try:
            marker_array = MarkerArray()

            # Colors for each foot trajectory
            colors = [
                ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),  # Red for FL
                ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),  # Green for FR
                ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # Blue for RL
                ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),  # Yellow for RR
            ]

            foot_names = ["FL", "FR", "RL", "RR"]

            # Transpose the tensor to match the expected shape (num_steps, 4, 3)
            positions_for_viz = self.future_feet_positions_w.permute(1, 0, 2).numpy()

            for foot_idx in range(len(foot_names)):
                # Create a marker for each future position of each foot
                for step in range(positions_for_viz.shape[0]):
                    marker = Marker()
                    marker.header.frame_id = "world"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = f"foot_trajectory_{foot_names[foot_idx]}"
                    marker.id = foot_idx * positions_for_viz.shape[0] + step
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD

                    # Set the scale of the sphere
                    marker.scale.x = 0.02  # radius
                    marker.scale.y = 0.02
                    marker.scale.z = 0.02

                    # Set the color
                    marker.color = colors[foot_idx]

                    # Set position
                    marker.pose.position.x = float(positions_for_viz[step, foot_idx, 0])
                    marker.pose.position.y = float(positions_for_viz[step, foot_idx, 1])
                    marker.pose.position.z = float(positions_for_viz[step, foot_idx, 2])

                    # Set orientation (identity quaternion)
                    marker.pose.orientation.w = 1.0
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0

                    marker_array.markers.append(marker)

            # Publish the marker array
            self.feet_trajectory_pub.publish(marker_array)
            if self.logger is not None:
                self.logger.debug("Published future feet positions marker array")

        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to publish future feet positions: {e}")

    def pub_opt_contact_locations(self, opt_contact_locations):
        """
        Visualizes the optimized contact locations in the world frame.
        Shows ALL time steps in the horizon at once.
        Colors the markers by foot:
        - Red: FL
        - Green: FR
        - Blue: RL
        - Yellow: RR
        """
        try:
            marker_array = MarkerArray()

            colors = [
                ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),  # Red for FL
                ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),  # Green for FR
                ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # Blue for RL
                ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),  # Yellow for RR
            ]

            foot_names = ["FL", "FR", "RL", "RR"]

            positions_for_viz = opt_contact_locations.permute(1, 0, 2).numpy()

            for foot_idx in range(len(foot_names)):
                for step in range(positions_for_viz.shape[0]):
                    marker = Marker()
                    marker.header.frame_id = "world"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = f"opt_foot_trajectory_{foot_names[foot_idx]}"
                    marker.id = foot_idx * self.horizon_length + step
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD

                    marker.scale.x = 0.03
                    marker.scale.y = 0.03
                    marker.scale.z = 0.03

                    marker.color = colors[foot_idx]

                    marker.pose.position.x = float(positions_for_viz[step, foot_idx, 0])
                    marker.pose.position.y = float(positions_for_viz[step, foot_idx, 1])
                    marker.pose.position.z = float(positions_for_viz[step, foot_idx, 2])

                    marker.pose.orientation.w = 1.0
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0

                    marker_array.markers.append(marker)

            self.opt_contact_locations_pub.publish(marker_array)

        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to publish optimized contact locations: {e}")

    def pub_com_trajectory(self, com_traj):
        """
        Visualizes the COM trajectory from collected train_data.
        com_traj: shape (num_traj, horizon+1, 3) - x, y, z positions
        """
        try:
            marker_array = MarkerArray()

            # Get first trajectory (best one)
            traj = com_traj[0]  # shape: (horizon+1, 3)

            for step in range(traj.shape[0]):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "com_trajectory"
                marker.id = step
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD

                marker.scale.x = 0.02
                marker.scale.y = 0.02
                marker.scale.z = 0.02

                # Cyan color for COM trajectory
                marker.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)

                marker.pose.position.x = float(traj[step, 0])
                marker.pose.position.y = float(traj[step, 1])
                marker.pose.position.z = float(traj[step, 2])

                marker.pose.orientation.w = 1.0
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0

                marker_array.markers.append(marker)

            self.com_trajectory_pub.publish(marker_array)

        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to publish COM trajectory: {e}")

    def pub_qpos_trajectory(self, qpos_traj):
        """
        Publishes predicted qpos trajectory from train_data.

        qpos_traj can be:
        - (num_traj, horizon+1, nq): first trajectory is used
        - (horizon+1, nq)
        """
        try:
            qpos_arr = np.asarray(qpos_traj, dtype=np.float32)
            if qpos_arr.ndim == 3:
                qpos_arr = qpos_arr[0]

            if qpos_arr.ndim != 2:
                return

            msg = Float32MultiArray()
            msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
            msg.layout.dim[0].label = "step"
            msg.layout.dim[0].size = int(qpos_arr.shape[0])
            msg.layout.dim[0].stride = int(qpos_arr.size)
            msg.layout.dim[1].label = "qpos"
            msg.layout.dim[1].size = int(qpos_arr.shape[1])
            msg.layout.dim[1].stride = int(qpos_arr.shape[1])
            msg.data = qpos_arr.reshape(-1).tolist()

            self.qpos_trajectory_pub.publish(msg)
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to publish qpos trajectory: {e}")

    def pub_current_contact_locations(self):
        """
        Visualizes the current contact locations in the world frame.
        Colors the markers based on the current contact plan:
        - Green: for feet that are not in contact (contact_plan is False)
        - Black: for feet that are in contact (contact_plan is True)
        """
        try:
            marker_array = MarkerArray()

            # Get current desired positions for each foot
            current_positions = self.future_feet_positions_w[:, self.current_goal_idx]

            # Create markers for each foot
            for foot_idx in range(4):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "current_contact_locations"
                marker.id = foot_idx
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD

                # Set the scale of the sphere
                marker.scale.x = 0.05  # radius
                marker.scale.y = 0.05
                marker.scale.z = 0.05

                # Set color based on contact plan
                if self.current_contact_plan[0, foot_idx]:
                    # Black for feet in contact
                    marker.color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0)
                else:
                    # Green for feet not in contact
                    marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)

                # Set position using desired positions
                marker.pose.position.x = float(current_positions[foot_idx, 0])
                marker.pose.position.y = float(current_positions[foot_idx, 1])
                marker.pose.position.z = float(current_positions[foot_idx, 2])

                # Set orientation (identity quaternion)
                marker.pose.orientation.w = 1.0
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0

                marker_array.markers.append(marker)

            # Publish the marker array
            self.contact_locations_pub.publish(marker_array)

        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to publish current contact locations: {e}")

    def pub_feet_error_norm(self):
        """
        Publishes the error norm (magnitude) for each foot.
        """
        try:
            # Calculate error norms directly
            feet_errors = torch.linalg.norm(
                self.robot.mj_model.get_ee_positions_w() - self.future_feet_positions_w[:, self.current_goal_idx],
                axis=-1,
            )

            # Create Float32MultiArray message
            msg = Float32MultiArray()
            msg.layout.dim = [MultiArrayDimension()]
            msg.layout.dim[0].label = "feet"
            msg.layout.dim[0].size = 4
            msg.layout.dim[0].stride = 4
            msg.data = [float(x) for x in feet_errors]

            # Publish the message
            self.feet_error_norm_pub.publish(msg)
        except Exception as e:
            self.logger.error(f"Failed to publish feet error norm: {e}")

    def pub_contact_plan(self):
        """
        Publishes the boolean contact plan values.
        The contact plan is a boolean tensor indicating which feet are in contact.
        """
        try:
            # Create Float32MultiArray message
            msg = Float32MultiArray()
            msg.layout.dim = [MultiArrayDimension()]
            msg.layout.dim[0].label = "feet"
            msg.layout.dim[0].size = 4
            msg.layout.dim[0].stride = 4

            # Convert boolean contact plan to float values (0.0 for False, 1.0 for True)
            contact_plan_values = [float(x) for x in self.current_contact_plan[0]]
            msg.data = contact_plan_values

            # Publish the message
            self.contact_plan_pub.publish(msg)

        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"Failed to publish contact plan: {e}")

    def pub_contact_status(self, foot_forces):
        """
        Publishes the foot forces in the world frame.
        """
        try:
            # Calculate foot forces directly
            msg = Float32MultiArray()
            msg.layout.dim = [MultiArrayDimension()]
            msg.layout.dim[0].label = "feet"
            msg.layout.dim[0].size = 4
            msg.layout.dim[0].stride = 4
            mapped_foot_forces = [foot_forces[1], foot_forces[0], foot_forces[3], foot_forces[2]]
            msg.data = [float(x > 30) for x in mapped_foot_forces]
            # self.logger.debug(f"Publishing foot forces: {foot_forces}")

            # Publish the message
            self.contact_status_pub.publish(msg)
        except Exception as e:
            self.logger.error(f"Failed to publish contact status: {e}")

    def pub_foot_forces(self, foot_forces):
        """
        Publishes the foot forces in the world frame.
        """
        try:
            # Calculate foot forces directly
            msg = Float32MultiArray()
            msg.layout.dim = [MultiArrayDimension()]
            msg.layout.dim[0].label = "feet"
            msg.layout.dim[0].size = 4
            msg.layout.dim[0].stride = 4
            mapped_foot_forces = [foot_forces[1], foot_forces[0], foot_forces[3], foot_forces[2]]
            msg.data = [float(x) for x in mapped_foot_forces]
            # self.logger.debug(f"Publishing foot forces: {foot_forces}")

            # Publish the message
            self.foot_forces_pub.publish(msg)
        except Exception as e:
            self.logger.error(f"Failed to publish foot forces: {e}")

