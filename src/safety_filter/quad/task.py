import copy
from typing import Any, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from utils.mps import *
from safety_filter.bases.task import Task, TaskParams

from flax.struct import dataclass


def get_torch_policy():
    import torch
    import torch.nn as nn

    class ActorNetwork(nn.Module):
        def __init__(self, obs_dim=82, action_dim=12):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, 512),
                nn.ELU(),
                nn.Linear(512, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, action_dim),
            )

        def forward(self, x):
            return self.actor(x)

    model = ActorNetwork()
    return model


@dataclass
class QuadRLTaskParams(TaskParams):
    """Task parameters specific to the MPSQuadRL task."""

    goal_idx: int = 0
    contact_plan: jax.Array = jnp.zeros(8)
    last_action: jax.Array = jnp.zeros(12)  # Assuming 12-dimensional action space

    current_gait: int = 0  # 0=stance, 1=transition, 2=trot, 3=pace, etc.
    goal_completion_counter: int = 0
    in_transition: bool = False
    lowlevel_step_counter: int = 0
    last_q_des: jax.Array = jnp.zeros(12)


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
        plan_segment: the MPS action (delta of contact_locations)

    Output:
        mjx.Data: the next state of the system after applying the plan_segment

    Internal Logic:
        1. plan_segment (delta of contact_locations) # TODO: contact_time_left, contact_plan
            contact_locations: (4, 2, 3) 4 is legs, 2 is reference length, and 3 is xyz position.
            plan_segment: (4, 2, 3) delta of contact_locations
        2. plan_segment + state to observation_rl
        3. observation_rl to action_rl (using the learned policy)
        4. action_rl to joint_pd (using the learned policy)
        5. joint_pd to next_state (using the MuJoCo dynamics)


    PLAN:
        1. goal indx and contact loc

    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,  # type: ignore
        rl_policy: Callable,
        rl_config: Dict[str, Any],
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

        self.obs_horizon = rl_config["obs_horizon"]
        self.command_duration = rl_config["command_duration"]
        self.joint_obs_unitree_to_isaac_mapping = np.asarray(
            rl_config["joint_obs_unitree_to_isaac_mapping"], dtype=np.int32
        )
        self.default_joint_pos = jnp.asarray(
            rl_config["default_joint_pos"], dtype=jnp.float32
        )

        self.num_joints = len(self.default_joint_pos)
        actuator_joint_ids = np.asarray(mj_model.actuator_trnid[:, 0], dtype=np.int32)
        self.joint_qpos_indices = np.asarray(
            mj_model.jnt_qposadr[actuator_joint_ids[: self.num_joints]], dtype=np.int32
        )
        self.joint_qvel_indices = np.asarray(
            mj_model.jnt_dofadr[actuator_joint_ids[: self.num_joints]], dtype=np.int32
        )
        actuator_joint_ids = actuator_joint_ids[: self.num_joints]

        foot_names = rl_config["foot_names"]
        self.foot_ids = np.asarray(
            [mj_model.body(foot).id for foot in foot_names], dtype=np.int32
        )

        self.plan_lim = rl_config["plan_lim"]
        self.act_rl_clip = rl_config["act_rl_clip"]
        self.act_rl_offset = jnp.asarray(rl_config["action_offset"])
        self.act_rl_scale = rl_config["action_scale"]
        self.act_rl_mapping = np.asarray(rl_config["action_mapping"], dtype=np.int32)
        self.obstacle_clearance = float(rl_config.get("obstacle_clearance", 0.05))
        self.obstacle_cost_weight = float(rl_config.get("obstacle_cost_weight", 100.0))
        self.obstacle_cost_scale = float(rl_config.get("obstacle_cost_scale", 0.3))

        # We optimize around absolute reference contact locations in normalized space.
        # Keep planner bounds unbounded to avoid saturating valid world-frame references.
        self.u_min = -jnp.inf
        self.u_max = jnp.inf

        soft_limit_factor = float(rl_config.get("soft_dof_limit_factor", 0.95))
        joint_lower = np.asarray(
            mj_model.jnt_range[actuator_joint_ids, 0], dtype=np.float32
        )
        joint_upper = np.asarray(
            mj_model.jnt_range[actuator_joint_ids, 1], dtype=np.float32
        )
        joint_center = 0.5 * (joint_lower + joint_upper)
        joint_span = joint_upper - joint_lower
        soft_joint_min = joint_center - 0.5 * joint_span * soft_limit_factor
        soft_joint_max = joint_center + 0.5 * joint_span * soft_limit_factor

        cfg_jpd_min = np.asarray(rl_config.get("j_pd_min", -np.inf), dtype=np.float32)
        cfg_jpd_max = np.asarray(rl_config.get("j_pd_max", np.inf), dtype=np.float32)
        if cfg_jpd_min.ndim == 0:
            cfg_jpd_min = np.full(self.num_joints, float(cfg_jpd_min), dtype=np.float32)
        if cfg_jpd_max.ndim == 0:
            cfg_jpd_max = np.full(self.num_joints, float(cfg_jpd_max), dtype=np.float32)

        self.j_pd_min = jnp.asarray(
            np.maximum(soft_joint_min, cfg_jpd_min), dtype=jnp.float32
        )
        self.j_pd_max = jnp.asarray(
            np.minimum(soft_joint_max, cfg_jpd_max), dtype=jnp.float32
        )

        self.kp = rl_config["kp"]
        self.kd = rl_config["kd"]

        sim_dt = mj_model.opt.timestep
        decimation = rl_config["decimation"]
        # Freq at which the main controller runs (RL + PD)
        main_controller_dt = rl_config["control_dt"]
        # It is called every decimation steps of main_controller_dt
        rl_policy_dt = main_controller_dt * decimation
        # rollout_decimation is how many sim steps we should take
        # it should be roughly equal to the command duration
        self.rollout_decimation = max(1, int(round(self.command_duration / sim_dt)))
        self.policy_decimation = max(1, int(round(rl_policy_dt / sim_dt)))

        self.gait_patterns = jnp.array(
            [
                [[True, True, True, True], [True, True, True, True]],  # 0: stance
                [[True, True, True, True], [True, True, True, True]],  # 1: transition
                [[False, True, True, False], [True, False, False, True]],  # 2: trot
                [[True, False, True, False], [False, True, False, True]],  # 3: pace
                [[True, True, False, False], [False, False, True, True]],  # 4: bound
                [[False, False, False, False], [True, True, True, True]],  # 5: jump
            ]
        )
        self.plan_dim = 4 * 3  # 4 legs, 3D position
        self.horizon_length = rl_config["horizon_length"]

        self.body_name_to_id = {mj_model.body(i).name: i for i in range(mj_model.nbody)}
        self.base_body_id = self.body_name_to_id.get("base_link", None)
        if self.base_body_id is None:
            raise KeyError("No 'base_link' found in model bodies for constraint check")
        ngeom = mj_model.ngeom
        geom_is_hazard_np = np.zeros(ngeom, dtype=bool)
        geom_is_robot_np = np.zeros(ngeom, dtype=bool)

        # Make it xml specific remember!
        hazard_prefixes = ("obstacle_",)
        base_link_id = mj_model.body("base_link").id

        for gid in range(ngeom):
            # -- HAZARD CHECK --
            # Handle the case where a geom might not have a name (returns None or empty)
            geom_name = mj_model.geom(gid).name or ""

            if any(geom_name.startswith(p) for p in hazard_prefixes):
                geom_is_hazard_np[gid] = True

            # -- ROBOT CHECK --
            # Find which body this geom belongs to
            body_id = mj_model.geom_bodyid[gid]

            # Check if this body's ultimate parent is the base_link
            if mj_model.body_rootid[body_id] == base_link_id:
                geom_is_robot_np[gid] = True

        # Convert to JAX arrays and store them
        self.geom_is_hazard = jnp.array(geom_is_hazard_np)
        self.geom_is_robot = jnp.array(geom_is_robot_np)

        hazard_geom_ids = np.flatnonzero(geom_is_hazard_np).astype(np.int32)
        hazard_geom_body_ids = np.asarray(mj_model.geom_bodyid[hazard_geom_ids], dtype=np.int32)
        hazard_geom_pos_local = np.asarray(mj_model.geom_pos[hazard_geom_ids], dtype=np.float32)
        if hasattr(mj_model, "geom_rbound"):
            hazard_geom_radii = np.asarray(mj_model.geom_rbound[hazard_geom_ids], dtype=np.float32)
        else:
            hazard_geom_size = np.asarray(mj_model.geom_size[hazard_geom_ids], dtype=np.float32)
            hazard_geom_radii = np.linalg.norm(hazard_geom_size, axis=1)

        self.num_hazard_geoms = int(hazard_geom_ids.shape[0])
        self.hazard_geom_body_ids = jnp.asarray(hazard_geom_body_ids, dtype=jnp.int32)
        self.hazard_geom_pos_local = jnp.asarray(hazard_geom_pos_local, dtype=jnp.float32)
        self.hazard_geom_radii = jnp.asarray(
            np.maximum(hazard_geom_radii, 0.0), dtype=jnp.float32
        )

        self.robot_radius = float(rl_config.get("robot_radius", 0.3))

    def normalize(self, plan: jax.Array) -> jax.Array:
        """Normalize plan values by plan_lim without saturating absolute references."""
        return plan / self.plan_lim

    def denormalize(self, plan: jax.Array) -> jax.Array:
        """Denormalize normalized plan values back to world-frame units."""
        return plan * self.plan_lim

    @partial(jax.jit, static_argnames=["self"])
    def get_exec_plan(self, plan: jax.Array) -> jax.Array:
        """Get the executable plan by reshaping and denormalizing the best_plan."""
        plan_denorm = self.denormalize(plan)

        # Reshape from (H, plan_dim) to (H, 4, 3)
        plan_reshaped = plan_denorm.reshape(-1, 4, 3)

        return jnp.transpose(plan_reshaped, (1, 0, 2))  # Shape: (4, H, 3)

    @partial(jax.jit, static_argnames=["self"])
    def get_reference_plan(self, **kwargs):
        """Get the reference trajectory for the MPS planner."""
        ref = kwargs.get("ref", None)
        goal_idx = kwargs.get("goal_idx", 0)
        if ref is not None:
            goal_idx = jnp.asarray(goal_idx, dtype=jnp.int32)
            ref_horizon = ref.shape[1]
            safe_goal_idx = jnp.mod(goal_idx, ref_horizon)
            shifted_ref = jnp.roll(ref, shift=-safe_goal_idx, axis=1)

            # contact_locations_w = ([FL, FR, RL, RR], [ref1, ref2], [x, y, z]) = (4, 2, 3)
            # plan_dim = ([ref1, ref2], [FLx, FLy, FLz, FRx, FRy, FRz, RLx, RLy, RLz, RRx, RRy, RRz]) = 12
            ref_plan = jnp.transpose(shifted_ref, (1, 0, 2)).reshape(
                -1, self.plan_dim
            )  # Shape: (H, plan_dim)

            return self.normalize(ref_plan)
        else:
            raise ValueError("Reference plan not provided in kwargs.")

    @partial(jax.jit, static_argnames=["self"])
    def update_params(self, task_params: QuadRLTaskParams, kwargs):
        current_gait = kwargs.get("current_gait")
        in_transition = kwargs.get("in_transition")
        goal_idx = kwargs.get("goal_idx")
        goal_completion_counter = kwargs.get("goal_completion_counter")
        contact_plan = kwargs.get("contact_plan")

        gait = jnp.asarray(current_gait, dtype=jnp.int32)
        synced_goal_idx = jnp.asarray(
            task_params.goal_idx if goal_idx is None else goal_idx, dtype=jnp.int32
        )
        synced_counter = jnp.asarray(
            task_params.goal_completion_counter
            if goal_completion_counter is None
            else goal_completion_counter,
            dtype=jnp.int32,
        )

        # Prime contact_plan from the gait pattern so plan step 0 sees the
        # correct phase immediately (update_goal() won't fire for i=0 since
        # lowlevel_step_counter is reset to 0 and should_resample needs counter>0).
        if contact_plan is None:
            primed_contact_plan = self._build_contact_plan(gait, synced_counter)
            # For stance/transition (gait<=1) keep the existing contact_plan.
            primed_contact_plan = jnp.where(
                gait <= 1, task_params.contact_plan, primed_contact_plan
            )
        else:
            primed_contact_plan = jnp.asarray(contact_plan, dtype=jnp.float32).reshape(-1)

        return update_params(
            task_params,
            goal_idx=synced_goal_idx,
            goal_completion_counter=synced_counter,
            contact_plan=primed_contact_plan,
            current_gait=gait,
            in_transition=jnp.asarray(in_transition, dtype=jnp.bool_),
            lowlevel_step_counter=jnp.asarray(0, dtype=jnp.int32),
            last_q_des=task_params.last_q_des,
        )

    def _build_contact_plan(
        self, gait: jax.Array, counter: jax.Array
    ) -> jax.Array:
        """Construct an 8-element contact_plan for the given gait and phase counter."""
        gait_pattern = jax.lax.dynamic_index_in_dim(
            self.gait_patterns, gait, axis=0, keepdims=False
        )
        phase_idx = jnp.mod(counter, 2)
        next_phase_idx = 1 - phase_idx
        current_phase = jax.lax.dynamic_index_in_dim(
            gait_pattern, phase_idx, axis=0, keepdims=False
        )
        next_phase = jax.lax.dynamic_index_in_dim(
            gait_pattern, next_phase_idx, axis=0, keepdims=False
        )
        return jnp.concatenate([current_phase, next_phase]).astype(jnp.float32)

    def plan_seg_to_contact_locations(self, plan_segment: jax.Array, H: int = 2) -> jax.Array:
        # plan_segment shape is (H, plan_dim) and not (4, H, 3), where plan_dim is 4*3
        # contact_locations_w = ([FL, FR, RL, RR], [ref1, ref2, ..., refH], [x, y, z]) = (4, H, 3)
        return jnp.transpose(plan_segment.reshape(H, 4, 3), (1, 0, 2))

    def update_goal(self, task_params: QuadRLTaskParams) -> QuadRLTaskParams:
        """Update goal_idx and contact_plan based on gait phase progression."""
        gait = jnp.asarray(task_params.current_gait, dtype=jnp.int32)
        counter = jnp.asarray(task_params.goal_completion_counter, dtype=jnp.int32)
        goal_idx = jnp.asarray(task_params.goal_idx, dtype=jnp.int32)

        def _update_goal(params):
            new_counter = counter + 1
            new_goal_idx = goal_idx + 1

            # Safety reset if exceeds horizon
            reset = new_goal_idx >= self.horizon_length
            new_goal_idx = jnp.where(reset, 0, new_goal_idx)
            new_counter = jnp.where(reset, 0, new_counter)

            new_contact_plan = self._build_contact_plan(gait, new_counter)

            return params.replace(  # type: ignore
                goal_completion_counter=new_counter,
                goal_idx=new_goal_idx,
                contact_plan=new_contact_plan,
            )

        # Skip if stance or transition
        task_params = jax.lax.cond(
            gait <= 1, lambda _: task_params, lambda _: _update_goal(task_params), None
        )

        return task_params

    def get_observation(
        self,
        task_params: QuadRLTaskParams,
        state: mjx.Data,
        contact_locations_w: jax.Array,
    ) -> Tuple[QuadRLTaskParams, jax.Array]:
        # Base position and orientation
        base_pos = state.qpos[:3]
        base_quat = state.qpos[3:7]

        # Use actuator-ordered joint states (matches Unitree low-level command/sensor order).
        joint_pos = state.qpos[self.joint_qpos_indices]
        joint_vel = state.qvel[self.joint_qvel_indices]

        current_feet_pos_w = state.xpos[self.foot_ids]

        mapping = self.joint_obs_unitree_to_isaac_mapping

        # 1. lin_vel_b = 3
        lin_vel_b = quat_rotate_inv(base_quat, state.qvel[:3])

        # 2. ang_vel_b = 3
        # Already in base? TODO: verify
        ang_vel_b = state.qvel[3:6]

        # 3. projected_gravity = 3
        gravity_dir = jnp.array([0, 0, -1])
        projected_gravity = quat_rotate_inv(base_quat, gravity_dir)

        # 4. contact_locations_b (4, 2, 3) = 24
        # world -> base: p_b = R(q)^T (p_w - p_base)
        rel = contact_locations_w - base_pos  # broadcast over (4, 2, 3)
        rel_flat = rel.reshape(-1, 3)  # (8, 3)
        feet_b_flat = jax.vmap(lambda p: quat_rotate_inv(base_quat, p))(rel_flat)
        contact_locations_b = feet_b_flat.reshape(4, self.obs_horizon, 3)

        # 5. contact_time_left = 1
        # Use the step counter to compute time remaining in the current command
        # period.  Avoids float32 precision loss when state.time holds a large
        # Unix timestamp (ULP ~200 s >> command_duration = 0.3 s).
        steps_since_resample = jnp.asarray(
            jnp.mod(task_params.lowlevel_step_counter, self.rollout_decimation),
            dtype=jnp.float32,
        )
        time_left = jnp.maximum(
            0.0, self.command_duration - steps_since_resample * self.dt
        )
        time_left = jnp.expand_dims(time_left, axis=-1)

        # 6. contact_plan = 8
        contact_plan = task_params.contact_plan.reshape(-1)  # (8,)

        # 7. joint_pos_rel = 12
        joint_pos_rel = joint_pos[mapping] - self.default_joint_pos

        # 8. joint_vel = 12
        joint_vel_mapped = joint_vel[mapping]

        # 9. ee_pos_rel_b = 4
        feet_pos_rel_b = (
            jax.lax.dynamic_index_in_dim(contact_locations_w, 0, axis=1, keepdims=False)
            - current_feet_pos_w
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
        self, task_params: QuadRLTaskParams, state: mjx.Data, plan_segment: jax.Array
    ) -> Tuple[QuadRLTaskParams, jax.Array]:
        """
        Since this is called inside the planner, state is as it should be but
            plan_segment shape is (2, plan_dim) and not (4, 2, 3), where plan_dim is 4*3
        We will need to reshape it inside the function.
        But the variables should remain consistent while reshaping.
        """
        # convert carefully
        contact_locations_w = self.plan_seg_to_contact_locations(plan_segment)

        # Step 1: Convert MPS action and state to RL observation
        task_params, observation_rl = self.get_observation(
            task_params, state, contact_locations_w
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
        plan_segment: jax.Array,
        plan_segment_ref: jax.Array | None = None,
    ) -> Tuple[QuadRLTaskParams, mjx.Data]:
        # If ref is optional, default to zeros (or task's current ref convention)
        if plan_segment_ref is None:
            plan_segment_ref = jnp.zeros_like(plan_segment)

        plan_segment_denorm = self.denormalize(plan_segment)

        def _update_policy(tp, s):
            tp_new, q_des_new = self.rl_forward(tp, s, plan_segment_denorm)
            tp_new = tp_new.replace(last_q_des=q_des_new)  # type: ignore
            return tp_new, q_des_new

        def _hold_policy(tp, s):
            return tp, tp.last_q_des

        for _ in range(self.rollout_decimation):
            # Use the integer step counter to detect command-period boundaries.
            # Wall-clock timestamps passed as state.time are Unix epoch values
            # (~1.7e9 s); in float32 their ULP is ~200 s, so sub-second elapsed
            # time comparisons always round to zero and should_resample never fires
            # after the first reset.  The integer counter has no precision issue.
            # Fire at the very first sub-step of every plan step EXCEPT step 0
            # (counter==0) so that goal_idx starts at 0 for the first plan step.
            counter = task_params.lowlevel_step_counter
            should_resample = jnp.logical_and(
                jnp.equal(jnp.mod(counter, self.rollout_decimation), 0),
                counter > 0,
            )
            task_params = jax.lax.cond(
                should_resample,
                lambda p: self.update_goal(p),
                lambda p: p,
                task_params,
            )

            should_update_policy = jnp.equal(
                jnp.mod(task_params.lowlevel_step_counter, self.policy_decimation), 0
            )

            task_params, q_des = jax.lax.cond(
                should_update_policy,
                lambda _: _update_policy(task_params, state),
                lambda _: _hold_policy(task_params, state),
                operand=None,
            )

            # Use actuator-ordered joint states to match master low-level PD path.
            q = state.qpos[self.joint_qpos_indices]
            dq = state.qvel[self.joint_qvel_indices]

            dq_des = jnp.zeros_like(dq)

            torq = self.kp * (q_des - q) + self.kd * (dq_des - dq)  # PD control law
            # Clip the torques to prevent excessive forces
            torq = jnp.clip(torq, self.j_min, self.j_max)

            # Step physics
            state = state.replace(ctrl=torq)
            state = mjx.step(model, state)
            task_params = task_params.replace(lowlevel_step_counter=task_params.lowlevel_step_counter + 1)  # type: ignore

        return task_params, state

    @partial(jax.jit, static_argnames=["self"])
    def constraint_check(self, state: mjx.Data) -> jax.Array:
        """
        Safety cost: Penalizes if any robot body collides with any hazard.
        """
        data_warp = state._impl  # DataWarp

        # Extract contact arrays
        geom_pairs = data_warp.contact__geom
        efc_address = data_warp.contact__efc_address

        # Active contact slot if *any* of the 4 constraint dims is active.
        valid = (efc_address >= 0).any(axis=-1)  # (ncon,)

        g1 = geom_pairs[..., 0]  # (ncon,)
        g2 = geom_pairs[..., 1]  # (ncon,)

        # Look up categories using the arrays we built in Python
        is_hazard_1 = self.geom_is_hazard[g1]
        is_hazard_2 = self.geom_is_hazard[g2]
        is_robot_1 = self.geom_is_robot[g1]
        is_robot_2 = self.geom_is_robot[g2]

        # Logical OR: Collision happens if (G1 is robot AND G2 is hazard) OR vice versa
        robot_hazard_collision = (is_robot_1 & is_hazard_2) | (is_robot_2 & is_hazard_1)

        # Filter by valid contacts
        bad_contacts = robot_hazard_collision & valid
        collides = jnp.any(bad_contacts)  # scalar boolean

        # Check for falling
        base_quat = state.xquat[self.base_body_id]

        up_vector = quat_rotate(base_quat, jnp.array([0.0, 0.0, 1.0]))
        fall_down = up_vector[2] < 0.5

        # Return 1.0 if failure condition met, else 0.0
        collision_check = jnp.where(collides | fall_down, 1.0, 0.0)

        return collision_check

    def _hazard_geom_centers_world(self, state: mjx.Data) -> jax.Array:
        body_pos = state.xpos[self.hazard_geom_body_ids]
        body_quat = state.xquat[self.hazard_geom_body_ids]
        geom_offset_world = jax.vmap(quat_rotate)(body_quat, self.hazard_geom_pos_local)
        return body_pos + geom_offset_world

    def _contact_obstacle_cost(
        self, task_params: QuadRLTaskParams, state: mjx.Data, contact_locations_w: jax.Array
    ) -> jax.Array:
        if self.num_hazard_geoms == 0:
            return jnp.array(0.0, dtype=jnp.float32)
        
        hazard_centers = self._hazard_geom_centers_world(state) # (num_hazards, 3)

        body_pos = state.xpos[self.base_body_id]  # (3,)
        contact_points = contact_locations_w.reshape(-1, 3)

        scale = self.obstacle_cost_scale

        # --- Body cost (exponential, long-range) ---
        body_dist_offsets = body_pos - hazard_centers  # (num_hazards, 3)
        body_center_distances = jnp.linalg.norm(body_dist_offsets, axis=-1)
        body_signed_clearance = body_center_distances - self.hazard_geom_radii - self.robot_radius  # (num_hazards,)
        body_contact_cost = self.obstacle_cost_weight * jnp.sum(
            jnp.exp(-body_signed_clearance / scale)
        )

        # --- Foot cost (exponential, per foot-hazard pair) ---
        foot_dist_offsets = contact_points[:, None, :] - hazard_centers[None, :, :]  # (num_foot, num_hazards, 3)
        foot_center_distances = jnp.linalg.norm(foot_dist_offsets, axis=-1)  # (num_foot, num_hazards)
        foot_signed_clearance = foot_center_distances - self.hazard_geom_radii[None, :]  # (num_foot, num_hazards)
        foot_contact_cost = self.obstacle_cost_weight * jnp.sum(
            jnp.exp(-foot_signed_clearance / scale)
        )

        return body_contact_cost + foot_contact_cost

    def terminal_cost(
        self, task_params: QuadRLTaskParams, state: mjx.Data
    ) -> jax.Array:
        return jnp.array(0.0)

    def running_cost(
        self,
        task_params: QuadRLTaskParams,
        state: mjx.Data,
        plan_segment: jax.Array,
        plan_segment_ref: jax.Array | None = None,
    ) -> jax.Array:
        if plan_segment_ref is None:
            plan_segment_ref = jnp.zeros_like(plan_segment)

        # Penalize deviation from reference contact locations.
        # Avoid absolute-plan magnitude penalties that bias contacts toward origin.
        tracking_cost = jnp.sum((plan_segment - plan_segment_ref) ** 2)
        contact_locations_w = self.plan_seg_to_contact_locations(
            self.denormalize(plan_segment)
        )
        obstacle_cost = self._contact_obstacle_cost(
            task_params, state, contact_locations_w
        )
        return tracking_cost + obstacle_cost
