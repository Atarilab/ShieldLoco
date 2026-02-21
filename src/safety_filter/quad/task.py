import copy
from typing import Any, Dict, Sequence

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from utils.mps import *
from safety_filter.bases.task import Task, TaskParams

from flax.struct import dataclass


class QuadRLTaskParams(TaskParams):
    """Task parameters specific to the MPSQuadRL task."""

    command_start_time: float = 0.0
    goal_idx: int = 0
    contact_plan: jax.Array = jnp.zeros((4, 2))  # Example shape, adjust as needed
    last_action: jax.Array = jnp.zeros(12)  # Assuming 12-dimensional action space

    current_gait: int = 0  # 0=stance, 1=transition, 2=trot, 3=pace, etc.
    goal_completion_counter: int = 0
    in_transition: bool = False


class PhysicsMetaData:
    """Full config for Go2"""

    # ------------- Physics -------------

    _JACOBIAN_MAP = {
        "auto": mujoco.mjtJacobian.mjJAC_AUTO,  # type: ignore
        "dense": mujoco.mjtJacobian.mjJAC_DENSE,  # type: ignore
        "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,  # type: ignore
    }

    _CONE_MAP = {
        "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,  # type: ignore
        "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,  # type: ignore
    }

    _INTEGRATOR_MAP = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,  # type: ignore
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,  # type: ignore
    }

    _SOLVER_MAP = {
        "newton": mujoco.mjtSolver.mjSOL_NEWTON,  # type: ignore
        "cg": mujoco.mjtSolver.mjSOL_CG,  # type: ignore
        "pgs": mujoco.mjtSolver.mjSOL_PGS,  # type: ignore
    }

    iterations: int = 8
    ls_iterations: int = 8
    ls_tolerance: float = 1e-5
    tolerance: float = 1e-8
    integrator: str = "implicitfast"
    impratio: float = 1.0
    cone: str = "pyramidal"
    jacobian: str = "auto"
    solver: str = "newton"


class QuadRLTask(Task):
    """
    A Dynamics class that takes the following input and ouput

    Input:
        mjx.Data: the current state of the system
        action_mps: the MPS action (delta of contact_locations)

    Output:
        mjx.Data: the next state of the system after applying the action_mps

    Internal Logic:
        1. action_mps (delta of contact_locations) # TODO: contact_time_left, contact_plan
            contact_locations: (4, 2, 3) 4 is legs, 2 is reference length, and 3 is xyz position.
            action_mps: (4, 2, 3) delta of contact_locations
        2. action_mps + state to observation_rl
        3. observation_rl to action_rl (using the learned policy)
        4. action_rl to joint_pd (using the learned policy)
        5. joint_pd to next_state (using the MuJoCo dynamics)


    PLAN:
        1. goal indx and contact loc

    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,  # type: ignore
        rl_policy: callable,  # type: ignore
        rl_config: Dict[str, any],  # type: ignore
        trace_sites: Sequence[str] | None = None,
    ):
        # copy
        mj_model = copy.deepcopy(mj_model)
        self.meta_data = PhysicsMetaData()

        # Set MJX physics for all meta_data
        mj_model.opt.timestep = rl_config["timestep"]
        mj_model.opt.integrator = self.meta_data._INTEGRATOR_MAP[
            self.meta_data.integrator
        ]
        mj_model.opt.impratio = self.meta_data.impratio
        mj_model.opt.cone = self.meta_data._CONE_MAP[self.meta_data.cone]
        mj_model.opt.jacobian = self.meta_data._JACOBIAN_MAP[self.meta_data.jacobian]
        mj_model.opt.solver = self.meta_data._SOLVER_MAP[self.meta_data.solver]
        mj_model.opt.iterations = self.meta_data.iterations
        mj_model.opt.tolerance = self.meta_data.tolerance
        mj_model.opt.ls_iterations = self.meta_data.ls_iterations
        mj_model.opt.ls_tolerance = self.meta_data.ls_tolerance

        super().__init__(mj_model, trace_sites)

        self.rl_policy = rl_policy
        self.rl_config = rl_config

        self.obs_horizon = rl_config["obs_horizon"]
        self.command_duration = rl_config["command_duration"]
        self.joint_obs_unitree_to_isaac_mapping = rl_config[
            "joint_obs_unitree_to_isaac_mapping"
        ]
        self.default_joint_pos = rl_config["default_joint_pos"]

        self.num_joints = len(self.default_joint_pos)

        self.foot_ids = rl_config["foot_ids"]

        self.plan_lim = rl_config["plan_lim"]
        self.act_rl_clip = rl_config["act_rl_clip"]
        self.act_rl_offset = rl_config["action_offset"]
        self.act_rl_scale = rl_config["action_scale"]
        self.act_rl_mapping = rl_config["action_mapping"]

        self.j_pd_min = rl_config["j_pd_min"]
        self.j_pd_max = rl_config["j_pd_max"]

        self.kp = rl_config["kp"]
        self.kd = rl_config["kd"]

        self.decimation = rl_config["decimation"]

        self.gait_patterns = {
            0: jnp.array(
                [[True, True, True, True], [True, True, True, True]]
            ),  # stance
            1: jnp.array(
                [[True, True, True, True], [True, True, True, True]]
            ),  # transition
            2: jnp.array(
                [[False, True, True, False], [True, False, False, True]]
            ),  # trot
            3: jnp.array(
                [[True, False, True, False], [False, True, False, True]]
            ),  # pace
            4: jnp.array(
                [[True, True, False, False], [False, False, True, True]]
            ),  # bound
            5: jnp.array(
                [[False, False, False, False], [True, True, True, True]]
            ),  # jump
            # Add crawl/trot-fly patterns (4 phases) as needed
        }

        self.plan_dim = 4 * 3  # 4 legs, 3D position
        self.horizon_length = rl_config["horizon_length"]

    def normalize(self, plan: jax.Array) -> jax.Array:
        """Normalize the plan to be within [-1, 1] based on plan_lim."""
        return jnp.clip(plan / self.plan_lim, -1.0, 1.0)
    
    def denormalize(self, plan: jax.Array) -> jax.Array:
        """Denormalize the plan from [-1, 1] back to the original scale."""
        return jnp.clip(plan * self.plan_lim, -self.plan_lim, self.plan_lim)

    def update_goal(self, task_params: QuadRLTaskParams) -> QuadRLTaskParams:
        """Update goal_idx and contact_plan based on gait phase progression."""
        gait = task_params.current_gait
        counter = task_params.goal_completion_counter
        goal_idx = task_params.goal_idx

        # Skip if stance or transition
        if gait <= 1:
            return task_params

        # Determine number of phases for current gait
        num_phases = 2  # default for trot, pace, bound, jump
        if gait >= 6:  # crawl patterns have 4 phases
            num_phases = 4

        new_counter = counter + 1

        # Update goal_idx based on phase completion
        if num_phases == 4:
            new_goal_idx = goal_idx + (1 if new_counter % 4 == 0 else 0)
        else:
            new_goal_idx = goal_idx + (
                1 if new_counter % 2 != 0 and new_counter > 0 else 0
            )

        # Safety reset if exceeds horizon
        if new_goal_idx >= self.horizon_length:
            new_goal_idx = 0
            new_counter = 0

        # Update contact_plan
        phase_idx = new_counter % num_phases
        next_phase_idx = (new_counter + 1) % num_phases

        current_phase = self.gait_patterns[gait][phase_idx]
        next_phase = self.gait_patterns[gait][next_phase_idx]

        new_contact_plan = jnp.concatenate([current_phase, next_phase]).flatten()

        return task_params.replace(  # type: ignore
            goal_completion_counter=new_counter,
            goal_idx=new_goal_idx,
            contact_plan=new_contact_plan,
        )

    def get_observation(
        self, task_params: QuadRLTaskParams, state: mjx.Data, action_mps: jax.Array
    ) -> jax.Array:
        # Base position and orientation
        base_pos = state.qpos[:3]
        base_quat = state.qpos[3:7]

        # Assuming the first 7 are base pos and quat
        joint_pos = state.qpos[-self.num_joints :]
        # Assuming the first 6 are base lin and ang vel
        joint_vel = state.qvel[-self.num_joints :]

        current_feet_pos_w = state.xpos[self.foot_ids]

        mapping = self.joint_obs_unitree_to_isaac_mapping

        # 1. lin_vel_b = 3
        lin_vel_b = state.qvel[:3]

        # 2. ang_vel_b = 3
        ang_vel_b = state.qvel[3:6]

        # 3. projected_gravity = 3
        gravity_dir = jnp.array([0, 0, -1])
        projected_gravity = quat_rotate_inv(base_quat, gravity_dir)

        # 4. contact_locations_b (4, 2, 3) = 24
        # world -> base: p_b = R(q)^T (p_w - p_base)
        rel = action_mps - base_pos  # broadcast over (4, 2, 3)
        rel_flat = rel.reshape(-1, 3)  # (8, 3)
        feet_b_flat = jax.vmap(lambda p: quat_rotate_inv(base_quat, p))(rel_flat)
        contact_locations_b = feet_b_flat.reshape(4, self.obs_horizon, 3)

        # 5. contact_time_left = 1
        elapsed = state.time - task_params.command_start_time
        time_left = jnp.maximum(0.0, self.command_duration - elapsed)

        if elapsed > self.command_duration:
            # If the command duration has elapsed, we can reset the command start time
            # to the current time, so that the next command starts immediately.
            task_params = task_params.replace(command_start_time=state.time)  # type: ignore

        # 6. contact_plan = 8
        contact_plan = task_params.contact_plan.reshape(-1)  # (8,)

        # 7. joint_pos_rel = 12
        joint_pos_rel = joint_pos[mapping] - self.default_joint_pos

        # 8. joint_vel = 12
        joint_vel_mapped = joint_vel[mapping]

        # 9. ee_pos_rel_b = 4
        feet_pos_rel_b = (
            action_mps[:, task_params.goal_idx] - current_feet_pos_w
        )  # (4, 3)
        ee_pos_rel_b = jnp.linalg.norm(feet_pos_rel_b, axis=-1)  # (4,)

        # 10. last_action = 12
        last_action = task_params.last_action

        # Concatenate all components to form the final observation
        observation_rl = jnp.concatenate(
            [
                lin_vel_b,
                ang_vel_b,
                projected_gravity,
                contact_locations_b.reshape(-1),
                time_left,
                contact_plan,
                joint_pos_rel,
                joint_vel_mapped,
                ee_pos_rel_b,
                last_action,
            ]
        )

        return task_params, observation_rl

    def compute_low_level_cmd(self, action_rl: jax.Array) -> jax.Array:
        action_rl = jnp.clip(action_rl, -self.act_rl_clip, self.act_rl_clip)

        joint_pos_targets = (action_rl * self.act_rl_scale + self.act_rl_offset)[
            self.act_rl_mapping
        ]

        joint_pos_targets = jnp.clip(joint_pos_targets, self.j_pd_min, self.j_pd_max)

        return joint_pos_targets

    def rl_forward(
        self, task_params: QuadRLTaskParams, state: mjx.Data, plan_mps: jax.Array
    ) -> mjx.Data:
        """
        Since this is called inside the planner, state is as it should be but
            plan_mps shape is (2, plan_dim) and not (4, 2, 3), where plan_dim is 4*3
        We will need to reshape it inside the function.
        But the variables should remain consistent while reshaping.
        """
        # convert carefully
        # plan_dim = ([ref1, ref2], [FLx, FLy, FLz, FRx, FRy, FRz, RLx, RLy, RLz, RRx, RRy, RRz]) = 12
        # action_mps = ([FL, FR, RL, RR], [ref1, ref2], [x, y, z]) = (4, 2, 3)
        action_mps = jnp.transpose(plan_mps.reshape(2, 4, 3), (1, 0, 2))

        # Step 1: Convert MPS action and state to RL observation
        task_params, observation_rl = self.get_observation(
            task_params, state, action_mps
        )

        # Step 2: Get RL action from the learned policy
        action_rl = self.rl_policy(observation_rl)
        task_params = task_params.replace(last_action=action_rl)  # type: ignore

        # Step 3: Convert RL action to joint PD targets
        joint_pd = self.compute_low_level_cmd(action_rl)

        return task_params, joint_pd

    def dynamics(
        self,
        model: mjx.Model,
        task_params: QuadRLTaskParams,
        state: mjx.Data,
        plan_mps: jax.Array,
    ) -> mjx.Data:

        # TODO: check if pending change check is required. See IsaacLab code
        # TODO: currently gait is fixed during rollout
        # Update goal
        tp = self.update_goal(task_params)

        tp, u = self.rl_forward(tp, state, plan_mps)

        # PD Loop
        for _ in range(self.decimation):
            # Here we would apply the joint PD targets to the MuJoCo simulation and step forward

            q = state.qpos
            dq = state.qvel

            q_des = u
            dq_des = jnp.zeros_like(dq)

            torq = self.kp * (q_des - q) + self.kd * (dq_des - dq)  # PD control law
            torq = jnp.clip(
                torq, self.j_min, self.j_max
            )  # Clip the torques to prevent excessive forces

            state = state.replace(ctrl=torq)

            state = mjx.step(model, state)

        return tp, state
