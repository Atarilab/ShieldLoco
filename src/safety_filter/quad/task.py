import copy
from functools import partial
from typing import Any, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from utils.mps import *
from safety_filter.bases.task import Task, TaskParams
from safety_filter.quad.foot_projection import ObstacleSDF, project_plan_feet, project_plan_feet_with_arc_check, _sdf_and_grad_all, _batch_sdf_and_grad

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
    # Ball state at last observation: MPC computes p(t) = p0 + v0*(t0+t) + 0.5*g*(t0+t)^2
    dynamic_obstacle_p0: jax.Array = jnp.zeros((0, 3))      # position at obs time
    dynamic_obstacle_v0: jax.Array = jnp.zeros((0, 3))      # velocity at obs time
    dynamic_obstacle_t0: jax.Array = jnp.zeros((0,))        # lag since obs (seconds)
    dynamic_obstacle_radii: jax.Array = jnp.zeros((0,))
    dynamic_obstacle_active: jax.Array = jnp.zeros((0,), dtype=jnp.bool_)
    # Current foot positions in world frame at start of each planning cycle.
    # Used by foot_step_cost to penalize large foot displacements that cause
    # the RL policy to take intermediate contacts through obstacles.
    init_foot_positions: jax.Array = jnp.zeros((4, 3))


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
        # Allow overriding solver iterations from config for speed tuning.
        # Reducing from 8 to 4 roughly halves per-step Newton solve cost.
        mj_model.opt.iterations = int(rl_config.get("physics_iterations", self.meta_data.iterations))
        mj_model.opt.tolerance = self.meta_data.tolerance
        mj_model.opt.ls_iterations = int(rl_config.get("physics_ls_iterations", self.meta_data.ls_iterations))
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

        self.max_dynamic_obstacles = int(rl_config.get("max_dynamic_obstacles", 0))

        self.projection_clearance = float(rl_config.get("projection_clearance", 0.07))
        self.projection_iters = int(rl_config.get("projection_iters", 3))

        self.foot_step_weight = float(rl_config.get("foot_step_weight", 50.0))
        self.foot_chain_weight = float(rl_config.get("foot_chain_weight", 30.0))
        self.body_sdf_weight = float(rl_config.get("body_sdf_weight", 5.0))
        self.body_sdf_margin = float(rl_config.get("body_sdf_margin", 0.10))
        self.body_sdf_scale = float(rl_config.get("body_sdf_scale", 0.10))
        self.foot_path_sdf_weight = float(rl_config.get("foot_path_sdf_weight", 30.0))
        self.foot_path_sdf_scale = float(rl_config.get("foot_path_sdf_scale", self.body_sdf_scale))


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
        # For MPC rollouts the RL policy only needs to fire once per plan step (not every
        # sub-step). `mps_policy_decimation` defaults to rollout_decimation so the policy
        # is called once per horizon step.  The real-robot decimation (rl_policy_dt /
        # sim_dt ≈ 1) is intentionally *not* used here because it causes 10× more NN
        # inference than needed.
        self.policy_decimation = int(rl_config.get("mps_policy_decimation", self.rollout_decimation))

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

        # Dual constraint check: always compute both feet-only and full-body
        # collision signals.  Feet-only (discriminative, avoids weight collapse)
        # is used for CaT penalty; full-body (exhaustive) is used for argmin
        # selection so MPPI picks the sample with least whole-body contact.
        foot_geom_names = {"FL", "FR", "RL", "RR"}
        geom_is_foot_np = np.array(
            [(mj_model.geom(gid).name or "") in foot_geom_names for gid in range(ngeom)],
            dtype=bool,
        )
        self.geom_is_foot = jnp.array(geom_is_foot_np)

        # 2D SDF for foot-plan projection onto obstacle-free region.
        min_obs_h = float(rl_config.get("constraint_min_obstacle_height", 0.001))
        self.obstacle_sdf = ObstacleSDF.from_mujoco_model(
            mj_model,
            hazard_prefixes=("obstacle_",),
            min_height=min_obs_h,
        )

    def make_task_params(self) -> QuadRLTaskParams:
        n = self.max_dynamic_obstacles
        return QuadRLTaskParams(
            dynamic_obstacle_p0=jnp.zeros((n, 3), dtype=jnp.float32),
            dynamic_obstacle_v0=jnp.zeros((n, 3), dtype=jnp.float32),
            dynamic_obstacle_t0=jnp.zeros((n,), dtype=jnp.float32),
            dynamic_obstacle_radii=jnp.zeros((n,), dtype=jnp.float32),
            dynamic_obstacle_active=jnp.zeros((n,), dtype=jnp.bool_),
        )

    def normalize(self, plan: jax.Array) -> jax.Array:
        """Normalize plan values by plan_lim without saturating absolute references."""
        return plan / self.plan_lim

    def denormalize(self, plan: jax.Array) -> jax.Array:
        """Denormalize normalized plan values back to world-frame units."""
        return plan * self.plan_lim

    def project_plan_to_safe(self, plans_normalized: jax.Array) -> jax.Array:
        """Project foot XY positions to satisfy obstacle clearance.

        Solves min ||p - p_ref||^2 s.t. SDF(p) >= r_safe for all obstacles,
        using the active-set QP in foot_projection.py.

        Args:
            plans_normalized: (..., H, plan_dim) normalised contact plans.
        Returns:
            Projected plans with same shape, feet pushed out of obstacles.
        """
        if self.obstacle_sdf.num_obstacles == 0:
            return plans_normalized

        plans_world = self.denormalize(plans_normalized)
        projected_world = project_plan_feet(
            plans_world,
            self.obstacle_sdf,
            self.projection_clearance,
            self.projection_iters,
        )
        return self.normalize(projected_world)

    def project_plan_to_safe_with_arc_check(
        self, plans_normalized: jax.Array, current_feet_world: jax.Array
    ) -> jax.Array:
        """Project foot XY positions with arc-aware swing collision avoidance.

        Like project_plan_to_safe but additionally checks each swing arc
        (current stance -> h=0, and h-1 -> h for h>0) and snaps landings
        to the near-side boundary if the arc would cross an obstacle.

        This addresses the key failure mode where geometric projection lands
        a foot on the far side of an obstacle (SDF >= r_safe at landing) but
        the swing arc from current stance passes through the obstacle interior.

        Args:
            plans_normalized:   (..., H, plan_dim) normalised contact plans.
            current_feet_world: (..., 4, 2) current foot XY in world frame.
                                For sample_plans this is (N, 4, 2).

        Returns:
            Projected plans with arc-corrected XY, same shape.
        """
        if self.obstacle_sdf.num_obstacles == 0:
            return plans_normalized

        plans_world = self.denormalize(plans_normalized)
        projected_world = project_plan_feet_with_arc_check(
            plans_world,
            self.obstacle_sdf,
            self.projection_clearance,
            current_feet_world,
            self.projection_iters,
        )
        return self.normalize(projected_world)

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
        dynamic_obstacle_p0 = kwargs.get("dynamic_obstacle_p0")
        dynamic_obstacle_v0 = kwargs.get("dynamic_obstacle_v0")
        dynamic_obstacle_t0 = kwargs.get("dynamic_obstacle_t0")
        dynamic_obstacle_active = kwargs.get("dynamic_obstacle_active")
        dynamic_obstacle_radii = kwargs.get("dynamic_obstacle_radii")

        # Extract current foot positions from the full body xpos array.
        # Used by foot_step_cost to penalise large first-step displacements.
        xpos = kwargs.get("xpos")
        init_foot_positions = (
            jnp.asarray(xpos)[self.foot_ids]
            if xpos is not None
            else task_params.init_foot_positions
        )

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

        def _sync(new_val, existing, dtype):
            return existing if new_val is None else jnp.asarray(new_val, dtype=dtype)

        return update_params(
            task_params,
            goal_idx=synced_goal_idx,
            goal_completion_counter=synced_counter,
            contact_plan=primed_contact_plan,
            current_gait=gait,
            in_transition=jnp.asarray(in_transition, dtype=jnp.bool_),
            lowlevel_step_counter=jnp.asarray(0, dtype=jnp.int32),
            last_q_des=task_params.last_q_des,
            dynamic_obstacle_p0=_sync(dynamic_obstacle_p0, task_params.dynamic_obstacle_p0, jnp.float32),
            dynamic_obstacle_v0=_sync(dynamic_obstacle_v0, task_params.dynamic_obstacle_v0, jnp.float32),
            dynamic_obstacle_t0=_sync(dynamic_obstacle_t0, task_params.dynamic_obstacle_t0, jnp.float32),
            dynamic_obstacle_active=_sync(dynamic_obstacle_active, task_params.dynamic_obstacle_active, jnp.bool_),
            dynamic_obstacle_radii=_sync(dynamic_obstacle_radii, task_params.dynamic_obstacle_radii, jnp.float32),
            init_foot_positions=init_foot_positions,
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
    ) -> Tuple[QuadRLTaskParams, mjx.Data, jax.Array]:
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

        # Use jax.lax.scan instead of a Python for loop so XLA compiles the body
        # once (a while-loop) rather than unrolling rollout_decimation copies of
        # mjx.step into the HLO graph.  Unrolling 10 steps × 8 horizon = 80
        # sequential kernel launches; scan reduces this to a single while-loop.
        def _substep(carry, _):
            tp, s, hits = carry

            # Advance goal at the first sub-step of every plan step (except step 0).
            counter = tp.lowlevel_step_counter
            should_resample = jnp.logical_and(
                jnp.equal(jnp.mod(counter, self.rollout_decimation), 0),
                counter > 0,
            )
            tp = jax.lax.cond(
                should_resample, lambda p: self.update_goal(p), lambda p: p, tp
            )

            # Call RL policy only every policy_decimation sub-steps.
            should_update_policy = jnp.equal(
                jnp.mod(tp.lowlevel_step_counter, self.policy_decimation), 0
            )
            tp, q_des = jax.lax.cond(
                should_update_policy,
                lambda args: _update_policy(*args),
                lambda args: _hold_policy(*args),
                (tp, s),
            )

            q = s.qpos[self.joint_qpos_indices]
            dq = s.qvel[self.joint_qvel_indices]
            torq = jnp.clip(self.kp * (q_des - q) + self.kd * -dq, self.j_min, self.j_max)
            s = mjx.step(model, s.replace(ctrl=torq))
            tp = tp.replace(lowlevel_step_counter=tp.lowlevel_step_counter + 1)  # type: ignore
            hits = hits + self.constraint_check(s)
            return (tp, s, hits), None

        (task_params, state, constraint_hits), _ = jax.lax.scan(
            _substep,
            (task_params, state, jnp.zeros(2)),
            None,
            length=self.rollout_decimation,
        )
        return task_params, state, constraint_hits

    @partial(jax.jit, static_argnames=["self"])
    def constraint_check(self, state: mjx.Data) -> jax.Array:
        """Dual collision indicator: [feet_flag, body_collision_count].

        feet_flag (scalar): 1.0 if ANY foot-sphere geom contacts a hazard or
            the robot has fallen, else 0.0.  Used for CaT penalty —
            discriminative because swing-phase contacts don't trigger it.

        body_collision_count (scalar): number of distinct robot-hazard
            contact pairs.  Fall adds 1e3 so a fallen robot is always worse
            than any brushing.  Used for argmin selection — MPPI picks the
            sample with the fewest whole-body contacts.
        """
        data_warp = state._impl

        geom_pairs = data_warp.contact__geom
        efc_address = data_warp.contact__efc_address

        valid = (efc_address >= 0).any(axis=-1)

        g1 = geom_pairs[..., 0]
        g2 = geom_pairs[..., 1]

        is_hazard_1 = self.geom_is_hazard[g1]
        is_hazard_2 = self.geom_is_hazard[g2]

        is_foot_1 = self.geom_is_foot[g1]
        is_foot_2 = self.geom_is_foot[g2]

        is_robot_1 = self.geom_is_robot[g1]
        is_robot_2 = self.geom_is_robot[g2]

        feet_hazard = (is_foot_1 & is_hazard_2) | (is_foot_2 & is_hazard_1)
        robot_hazard = (is_robot_1 & is_hazard_2) | (is_robot_2 & is_hazard_1)

        # Filter by valid contacts
        feet_hit = jnp.any(feet_hazard & valid)
        body_count = jnp.sum(robot_hazard & valid)

        base_quat = state.xquat[self.base_body_id]
        up_vector = quat_rotate(base_quat, jnp.array([0.0, 0.0, 1.0]))
        fall = up_vector[2] < 0.5

        feet_flag = jnp.where(feet_hit | fall, 1.0, 0.0)
        # Clamp body_count to prevent overflow in downstream cost accumulation.
        # Values > 1e6 indicate a pathological physics state (theoretical max ~800
        # contacts; 11000+ seen before NaN cascade).  Clamping to 1e6 is safely
        # above any physical value while preventing float overflow.
        body_count_clamped = jnp.minimum(body_count, 1e6)
        body_flag = jnp.astype(body_count_clamped, jnp.float32) + jnp.where(fall, 1e3, 0.0)

        return jnp.array([feet_flag, body_flag])


    def foot_step_cost(
        self, task_params: QuadRLTaskParams, plan_segment: jax.Array,
        step_idx: jax.Array | None = None
    ) -> jax.Array:
        """Penalise displacement between planned first-step feet and current stance.

        Only fires at step_idx == 0 (the first horizon step transitioning from
        current stance).  At later steps the robot has already moved, so comparing
        to init_foot_positions penalises forward progress.

        Cost is computed in world frame (metres²) so that foot_step_weight is
        independent of plan_lim and stays comparable in magnitude to the tracking
        cost (~3-6 per step).
        """
        foot_step0_world = self.denormalize(plan_segment[0]).reshape(4, 3)
        disp_xy = foot_step0_world[:, :2] - task_params.init_foot_positions[:, :2]
        cost = self.foot_step_weight * jnp.sum(disp_xy ** 2)
        if step_idx is None:
            return cost
        return jax.lax.select(step_idx == 0, cost, jnp.array(0.0))

    def foot_chain_cost(self, task_params: QuadRLTaskParams, plan_segment: jax.Array) -> jax.Array:
        """Penalize step-to-step foot displacement across the full horizon.

        Cost = Σ_{h=1}^{H} Σ_{f=1}^{4} ||foot_h_f − foot_{h-1}_f||²

        Unlike foot_step_cost (only fires at goal_idx==0, penalizes distance from
        CURRENT stance), this penalizes the SHAPE of the plan: large jumps between
        consecutive planned steps encourage "shuffling" — each step lands near where
        the previous step landed.

        Fires at ALL goal_idx values — plan-shape regularizer, not cycle-dependent.
        A higher weight forces the robot to take many small steps rather than
        reaching far in any single step.

        Option C (saved for later): variance cost — penalize the spatial spread
        of foot positions across the horizon, e.g. Σ var(foot_h_f across h).
        Would push all horizon steps to cluster near current feet without
        penalizing smooth forward progressions.
        """
        # Work in world frame (metres) so foot_chain_weight is independent of plan_lim
        # and stays comparable to tracking cost.
        n_steps = plan_segment.shape[0]
        feet = self.denormalize(plan_segment).reshape(n_steps, 4, 3)
        deltas = feet[1:] - feet[:-1]   # (n_steps-1, 4, 3)
        return self.foot_chain_weight * jnp.sum(deltas ** 2)

    def foot_path_sdf_cost(
        self, task_params: QuadRLTaskParams, plan_segment: jax.Array
    ) -> jax.Array:
        """Penalize foot step transitions that cross through obstacle interiors.

        For each foot, interpolates along the step[0] -> step[1] transition
        and evaluates the 2D obstacle SDF at interior points.  If any point
        is within projection_clearance of an obstacle boundary, a softplus
        penalty fires.

        Only evaluates consecutive planned step transitions (not stance ->
        step[0], which is handled by foot_step_cost).  This avoids stale
        stance comparisons at later horizon steps where the path from initial
        stance to deep-planned positions always crosses obstacles regardless.
        """
        if self.obstacle_sdf.num_obstacles == 0:
            return jnp.array(0.0)

        feet_curr = self.denormalize(plan_segment[0]).reshape(4, 3)[:, :2]  # (4, 2)
        feet_next = self.denormalize(plan_segment[1]).reshape(4, 3)[:, :2]  # (4, 2)

        n_interp = 5
        t_vals = jnp.linspace(0.0, 1.0, n_interp + 2)[1:-1]

        interp = (1.0 - t_vals[:, None, None]) * feet_curr[None] + t_vals[:, None, None] * feet_next[None]

        flat_xy = interp.reshape(-1, 2)  # (n_interp * 4, 2)

        sdfs, _ = _batch_sdf_and_grad(flat_xy, self.obstacle_sdf)
        min_sdfs = jnp.min(sdfs, axis=1)

        violations = jnp.maximum(0.0, self.projection_clearance - min_sdfs)
        violations = jnp.clip(violations, -1e4, 1e4)
        cost = jnp.sum(jax.nn.softplus(violations / self.foot_path_sdf_scale) * self.foot_path_sdf_scale)

        return self.foot_path_sdf_weight * cost

    def body_sdf_cost(self, state: mjx.Data) -> jax.Array:
        """Soft penalty for the robot body being near obstacle clusters.

        Evaluates 2D SDF at the torso ground projection (base_link XY).
        Provides a smooth gradient for MPPI to route through less-dense regions
        even though individual foot landing positions are safe (via projection).

        This does NOT cause weight collapse — it's a continuous soft cost, not
        an all-or-nothing binary flag.
        """
        if self.obstacle_sdf.num_obstacles == 0:
            return jnp.array(0.0)

        body_xy = state.xpos[self.base_body_id, :2]
        sdfs, _ = _sdf_and_grad_all(body_xy, self.obstacle_sdf)
        min_sdf = jnp.min(sdfs)
        violation = jnp.maximum(0.0, self.body_sdf_margin - min_sdf)
        # Clamp before softplus to prevent overflow.  softplus(x) ~ x for small x,
        # but exp overflows at x > ~36.7 in float32.  Clamping to [-1e4, 1e4]
        # keeps the result safe for any physically plausible SDF value.
        violation_safe = jnp.clip(violation, -1e4, 1e4)
        return self.body_sdf_weight * jax.nn.softplus(violation_safe / self.body_sdf_scale) * self.body_sdf_scale

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
        step_idx: jax.Array | None = None,
    ) -> jax.Array:
        if plan_segment_ref is None:
            plan_segment_ref = jnp.zeros_like(plan_segment)

        tracking_cost = jnp.sum((plan_segment - plan_segment_ref) ** 2)
        foot_step = self.foot_step_cost(task_params, plan_segment, step_idx)
        foot_chain = self.foot_chain_cost(task_params, plan_segment)
        foot_path = self.foot_path_sdf_cost(task_params, plan_segment)
        body_sdf = self.body_sdf_cost(state)

        return tracking_cost + foot_step + foot_chain + foot_path + body_sdf

    def forward(self, model, tp, x, a, a_ref, step_idx=None):
        tp, x_next, constraint_hits = self.dynamics(model, tp, x, a, a_ref)

        task_cost = self.running_cost(tp, x_next, a, a_ref, step_idx=step_idx)

        return tp, x_next, task_cost, constraint_hits
