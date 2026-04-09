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
    # Ball state at last observation: MPC computes p(t) = p0 + v0*(t0+t) + 0.5*g*(t0+t)^2
    dynamic_obstacle_p0: jax.Array = jnp.zeros((0, 3))      # position at obs time
    dynamic_obstacle_v0: jax.Array = jnp.zeros((0, 3))      # velocity at obs time
    dynamic_obstacle_t0: jax.Array = jnp.zeros((0,))        # lag since obs (seconds)
    dynamic_obstacle_radii: jax.Array = jnp.zeros((0,))
    dynamic_obstacle_active: jax.Array = jnp.zeros((0,), dtype=jnp.bool_)


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
        self.obstacle_clearance = float(rl_config.get("obstacle_clearance", 0.05))
        self.obstacle_cost_weight = float(rl_config.get("obstacle_cost_weight", 100.0))
        self.obstacle_cost_scale = float(rl_config.get("obstacle_cost_scale", 0.3))
        self.max_dynamic_obstacles = int(rl_config.get("max_dynamic_obstacles", 0))
        self.dynamic_obstacle_cost_weight = float(rl_config.get("dynamic_obs_cost_weight", 0.0))
        self.dynamic_obstacle_cost_scale = float(rl_config.get("dynamic_obs_cost_scale", 0.15))
        self.mpc_latency_compensation = float(rl_config.get("mpc_latency_compensation", 0.0))
        # DPCBF parabolic safety: safe iff d_perp > sqrt(2 * k * ||v|| * approach_dist) + r_safe
        # k=0 disables heading cost; increase k to react earlier to fast approaching balls.
        self.dpcbf_k = float(rl_config.get("dpcbf_k", 0.0))

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

        # 2D (xy) bounding radius for ground-plane obstacle cost.
        # Avoids inflating obstacles by the z half-size (e.g. box 0.15×0.15×0.2
        # has 3D rbound=0.29 but 2D footprint radius=0.21).
        hazard_geom_size = np.asarray(mj_model.geom_size[hazard_geom_ids], dtype=np.float32)
        hazard_geom_radii_2d = np.linalg.norm(hazard_geom_size[:, :2], axis=1)
        self.hazard_geom_radii_2d = jnp.asarray(
            np.maximum(hazard_geom_radii_2d, 0.0), dtype=jnp.float32
        )

        self.robot_radius = float(rl_config.get("robot_radius", 0.3))

        # Robot body IDs and per-body bounding radii for whole-body aerial obstacle checks.
        # Uses all robot geoms (root = base_link) grouped by body.
        robot_geom_ids = np.flatnonzero(geom_is_robot_np).astype(np.int32)
        robot_body_ids_np = np.unique(mj_model.geom_bodyid[robot_geom_ids]).astype(np.int32)
        robot_body_radii_np = []
        for bid in robot_body_ids_np:
            mask = mj_model.geom_bodyid[robot_geom_ids] == bid
            geoms = robot_geom_ids[mask]
            if hasattr(mj_model, "geom_rbound"):
                r = float(np.maximum(mj_model.geom_rbound[geoms].max(), 0.05))
            else:
                r = float(np.maximum(np.linalg.norm(mj_model.geom_size[geoms], axis=1).max(), 0.05))
            robot_body_radii_np.append(r)
        self.robot_body_ids = jnp.asarray(robot_body_ids_np, dtype=jnp.int32)
        self.robot_body_radii = jnp.asarray(robot_body_radii_np, dtype=jnp.float32)


        # Map MuJoCo geom types to our convention
        # mjtGeom: 0=plane, 2=sphere, 3=capsule, 5=cylinder, 6=box
        _type_map = {2: 0, 6: 1, 5: 2, 3: 0}
        hazard_geom_types_mj = mj_model.geom_type[hazard_geom_ids]
        self.hazard_types = jnp.asarray(
            [_type_map.get(int(t), 0) for t in hazard_geom_types_mj], dtype=jnp.int32
        )
        self.hazard_half_extents = jnp.asarray(
            mj_model.geom_size[hazard_geom_ids], dtype=jnp.float32
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
            (task_params, state, jnp.array(0.0)),
            None,
            length=self.rollout_decimation,
        )
        return task_params, state, constraint_hits

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

    def _contact_obstacle_cost(self, task_params, state, contact_locations_w):
        if self.num_hazard_geoms == 0:
            return jnp.array(0.0)

        hazard_centers = self._hazard_geom_centers_world(state)
        body_pos = state.xpos[self.base_body_id]

        foot_points = contact_locations_w.reshape(-1, 3)

        return contact_obstacle_cost_sdf(
            body_pos_xy=body_pos[:2],
            foot_positions=foot_points,
            obstacle_centers=hazard_centers,
            obstacle_half_extents=self.hazard_half_extents,
            obstacle_types=self.hazard_types,              
            obstacle_radii=self.hazard_geom_radii_2d,
            robot_radius=self.robot_radius,
            foot_radius=0.07,
            margin=self.obstacle_clearance,
            cost_weight=self.obstacle_cost_weight,
            cost_fn="softplus",
            alpha=1.0 / self.obstacle_cost_scale,
            use_2d=True,  # ground-plane check for footstep planning
        )

    def _dynamic_obstacle_cost(
        self, task_params: QuadRLTaskParams, state: mjx.Data, contact_locations_w: jax.Array
    ) -> jax.Array:
        if (
            self.dynamic_obstacle_cost_weight <= 0.0
            or task_params.dynamic_obstacle_p0.shape[0] == 0
        ):
            return jnp.array(0.0, dtype=jnp.float32)

        obstacle_radii = task_params.dynamic_obstacle_radii   # (N,)
        obstacle_active = task_params.dynamic_obstacle_active.astype(jnp.float32)  # (N,)
        W = self.dynamic_obstacle_cost_weight
        scale = self.dynamic_obstacle_cost_scale
        _EXP_MAX = 20.0

        body_pos = state.xpos[self.base_body_id]  # (3,)

        # ── Stage-based cost ────────────────────────────────────────────────────
        # Ball position at this rollout stage (ballistic projection).
        # dt = lag-since-observation + MPS solve latency + time-into-horizon
        stage_idx = task_params.lowlevel_step_counter // self.rollout_decimation - 1
        stage_idx = jnp.clip(stage_idx, 0, self.horizon_length - 1)
        stage_time = stage_idx.astype(jnp.float32) * self.command_duration
        dt = task_params.dynamic_obstacle_t0 + self.mpc_latency_compensation + stage_time  # (N,)

        g = jnp.array([0.0, 0.0, -9.81], dtype=jnp.float32)
        ball_pos_stage = (
            task_params.dynamic_obstacle_p0
            + task_params.dynamic_obstacle_v0 * dt[:, None]
            + 0.5 * g[None, :] * (dt[:, None] ** 2)
        )
        # Clamp ball z to ground so underground balls don't create phantom attraction.
        ball_pos_stage = ball_pos_stage.at[:, 2].set(
            jnp.maximum(ball_pos_stage[:, 2], obstacle_radii)
        )

        # ── Whole-body check: all robot bodies vs predicted ball (3-D) ──────────
        # robot_body_ids covers base, hips, thighs, calves, and feet — every
        # body whose root is base_link.  Using 3-D distance is correct here
        # because the ball flies through the air and can hit from any direction.
        robot_body_pos = state.xpos[self.robot_body_ids]  # (B, 3)
        robot_ball_dist = jnp.linalg.norm(
            robot_body_pos[:, None, :] - ball_pos_stage[None, :, :], axis=-1
        )  # (B, N)
        robot_clearance = (
            robot_ball_dist - obstacle_radii[None, :] - self.robot_body_radii[:, None] - self.obstacle_clearance
        )
        stage_cost = W * jnp.sum(
            obstacle_active[None, :] * jnp.exp(jnp.clip(-robot_clearance / scale, a_max=_EXP_MAX))
        )

        # ── Ballistic impact position cost ──────────────────────────────────────
        # Computes WHERE the ball will be when it descends to the robot's body
        # height (z = body_pos.z) using the full ballistic arc.  Measures the
        # HORIZONTAL distance from that impact position to the robot body.
        # This correctly handles lobbed throws (high v0_z) where the straight-
        # line velocity direction points upward and misses the robot entirely.
        #
        # Works for any trajectory: line-drive, lob, drop — all correctly
        # mapped to a 2D horizontal clearance check at robot height.
        # Ballistic impact: where does the ball land at the robot's body height?
        # Uses the full parabolic arc — correctly handles lobs (high v0_z) where
        # the instantaneous velocity direction points skyward and doesn't predict
        # where the ball will actually be at robot height.
        p0 = task_params.dynamic_obstacle_p0   # (N, 3)
        v0 = task_params.dynamic_obstacle_v0   # (N, 3)
        p0_z = p0[:, 2]
        v0_z = v0[:, 2]
        robot_z = body_pos[2]

        # Solve: p0_z + v0_z*t - 4.905*t² = robot_z  →  descent root (larger t)
        disc = v0_z ** 2 + 4.0 * 4.905 * (p0_z - robot_z)     # (N,)
        t_impact = (v0_z + jnp.sqrt(jnp.maximum(disc, 0.0))) / (2.0 * 4.905)  # (N,)

        impact_xy = p0[:, :2] + v0[:, :2] * t_impact[:, None]  # (N, 2)  gravity only affects z
        horiz_dist = jnp.linalg.norm(impact_xy - body_pos[None, :2], axis=-1)  # (N,)
        horiz_clearance = horiz_dist - obstacle_radii - self.robot_radius - self.obstacle_clearance

        valid_arc = (disc > 0).astype(jnp.float32)  # ball actually reaches robot height
        heading_cost = W * jnp.sum(
            obstacle_active * valid_arc
            * jnp.exp(jnp.clip(-horiz_clearance / scale, a_max=_EXP_MAX))
        )

        return stage_cost + heading_cost

    def get_dynamic_safety_bias(
        self, task_params: QuadRLTaskParams, ref: jax.Array
    ) -> jax.Array:
        """Returns (H, plan_dim) normalized plan-space mean bias to avoid
        threatening ballistic obstacles.  Called from MIMPS.sample_plans so
        that all samples are centred around the analytically-predicted dodge
        rather than the nominal trajectory — enabling single-cycle whole-body
        avoidance.  Returns zeros when no obstacle is active or threatening.

        Body position is estimated as the XY centroid of the reference feet at
        horizon step 0.  The bias shifts ALL four feet by the same XY offset
        (rigid-body translation), which is the dominant mode for body-level
        obstacle avoidance.
        """
        if task_params.dynamic_obstacle_p0.shape[0] == 0:
            return jnp.zeros((self.horizon_length, self.plan_dim), dtype=jnp.float32)

        # ── Estimate robot body XY from reference foot centroid ──────────────
        # ref: (H, plan_dim) normalized.  Step-0 feet → (4, 3) world coords.
        step0_feet = (ref[0] * self.plan_lim).reshape(4, 3)   # (4, 3)
        body_xy = step0_feet[:, :2].mean(axis=0)               # (2,)
        body_z  = jnp.array(0.35, dtype=jnp.float32)          # nominal torso height

        # ── Per-obstacle ballistic impact check ──────────────────────────────
        p0  = task_params.dynamic_obstacle_p0    # (N, 3)
        v0  = task_params.dynamic_obstacle_v0    # (N, 3)
        radii = task_params.dynamic_obstacle_radii  # (N,)
        active = task_params.dynamic_obstacle_active.astype(jnp.float32)  # (N,)

        p0_z = p0[:, 2]
        v0_z = v0[:, 2]

        # Time for ball to descend to body_z (descent root of ballistic arc).
        disc = v0_z ** 2 + 4.0 * 4.905 * (p0_z - body_z)          # (N,)
        disc_safe = jnp.maximum(disc, 0.0)
        t_imp = (v0_z + jnp.sqrt(disc_safe)) / (2.0 * 4.905)       # (N,)

        # Impact XY position
        impact_xy = p0[:, :2] + v0[:, :2] * t_imp[:, None]         # (N, 2)

        # Vector and distance from body to impact
        to_impact = impact_xy - body_xy[None, :]                    # (N, 2)
        horiz_dist = jnp.linalg.norm(to_impact, axis=-1)            # (N,)
        horiz_dist_safe = jnp.maximum(horiz_dist, 1e-6)

        r_safe = radii + self.robot_radius + self.obstacle_clearance + 0.1  # (N,)
        deficit = jnp.maximum(r_safe - horiz_dist, 0.0)             # (N,) >0 when threatening

        # Dodge direction: directly away from impact point
        dodge_dir = -to_impact / horiz_dist_safe[:, None]           # (N, 2)

        # Mask: only fire for active obstacles with valid arc and actual threat
        valid = active * (disc > 0).astype(jnp.float32) * (t_imp > 0).astype(jnp.float32)
        per_obs_offset = valid[:, None] * deficit[:, None] * dodge_dir  # (N, 2)

        # Aggregate: sum offsets (handles multiple balls)
        total_xy = per_obs_offset.sum(axis=0)                       # (2,)

        # Clamp to plan_lim so the bias stays within one normalized unit.
        # Larger values are clipped by f() anyway but can corrupt warm-start mean.
        total_xy_norm = total_xy / self.plan_lim
        total_xy_norm = jnp.clip(total_xy_norm, -1.0, 1.0)

        # ── Convert to normalized plan-space bias ────────────────────────────
        # Shift all 4 feet by the same XY amount (rigid body translation).
        # plan_dim layout: [foot0_x, foot0_y, foot0_z, foot1_x, …] × 4
        foot_bias = jnp.zeros(self.plan_dim, dtype=jnp.float32)
        for f in range(4):
            foot_bias = foot_bias.at[f * 3    ].set(total_xy_norm[0])
            foot_bias = foot_bias.at[f * 3 + 1].set(total_xy_norm[1])

        # Constant across all horizon steps (simple and stable)
        return jnp.broadcast_to(foot_bias[None, :], (self.horizon_length, self.plan_dim))

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

        static_obstacle_cost = self._contact_obstacle_cost(
            task_params, state, contact_locations_w
        )

        dynamic_obstacle_cost = self._dynamic_obstacle_cost(
            task_params, state, contact_locations_w
        )

        return tracking_cost + static_obstacle_cost #+ dynamic_obstacle_cost





# =============================================================================
# 1. SDF Primitives (negative inside, positive outside)
# =============================================================================

def sdf_sphere(point: jax.Array, center: jax.Array, radius: float) -> jax.Array:
    return jnp.linalg.norm(point - center) - radius


def sdf_box_3d(point: jax.Array, center: jax.Array, half_extents: jax.Array,
               rotation: jax.Array = None) -> jax.Array:
    """Exact SDF for a 3D box (oriented or axis-aligned). Inigo Quilez formula."""
    p = point - center
    if rotation is not None:
        p = rotation.T @ p
    q = jnp.abs(p) - half_extents
    return jnp.linalg.norm(jnp.maximum(q, 0.0)) + jnp.minimum(jnp.max(q), 0.0)


def sdf_box_2d(point_xy: jax.Array, center_xy: jax.Array,
               half_extents_xy: jax.Array) -> jax.Array:
    """Exact SDF for a 2D box (ground-plane foothold check)."""
    q = jnp.abs(point_xy - center_xy) - half_extents_xy
    return jnp.linalg.norm(jnp.maximum(q, 0.0)) + jnp.minimum(jnp.max(q), 0.0)


def sdf_cylinder(point: jax.Array, center: jax.Array, radius: float,
                 half_height: float) -> jax.Array:
    """Exact SDF for an upright (z-axis) cylinder."""
    p = point - center
    d_radial = jnp.linalg.norm(p[:2]) - radius
    d_axial = jnp.abs(p[2]) - half_height
    outside = jnp.linalg.norm(jnp.maximum(jnp.array([d_radial, d_axial]), 0.0))
    inside = jnp.minimum(jnp.maximum(d_radial, d_axial), 0.0)
    return outside + inside


def sdf_capsule(point: jax.Array, a: jax.Array, b: jax.Array,
                radius: float) -> jax.Array:
    """Exact SDF for a capsule (sphere-swept line segment)."""
    pa = point - a
    ba = b - a
    t = jnp.clip(jnp.dot(pa, ba) / (jnp.dot(ba, ba) + 1e-8), 0.0, 1.0)
    return jnp.linalg.norm(point - (a + t * ba)) - radius


# =============================================================================
# 2. Cost Wrappers: "zero outside, active inside" (SemLoco-style)
# =============================================================================
#
# SemLoco's 1_col(pk) is the limiting case: step function at the boundary.
# Below are smooth approximations ordered by how closely they match that
# step-function behavior, all sharing the property:
#   cost ≈ 0 when sdf > margin (safely outside)
#   cost > 0 when sdf < 0 (inside obstacle)
#   gradient points toward nearest surface when inside

def cost_indicator_smooth(sdf_val: jax.Array, margin: float = 0.02,
                          sharpness: float = 50.0) -> jax.Array:
    """
    Smooth approximation of SemLoco's binary indicator 1_col.

    Uses a sigmoid: σ(sharpness * (margin - sdf))
      - ≈ 1.0 deep inside the obstacle (sdf << 0)
      - ≈ 0.5 at the boundary (sdf = margin)
      - ≈ 0.0 safely outside (sdf >> margin)

    As sharpness → ∞, this converges to 1_col exactly.
    But unlike 1_col, it's C^∞ differentiable everywhere.

    This is the closest match to SemLoco's formulation.
    Best for: when you want near-binary inside/outside but need gradients.
    """
    return jax.nn.sigmoid(sharpness * (margin - sdf_val))


def cost_softplus(sdf_val: jax.Array, margin: float = 0.02,
                  alpha: float = 20.0) -> jax.Array:
    """
    Softplus cost: (1/α) * log(1 + exp(α * (margin - sdf)))

    Unlike the sigmoid above, this grows linearly as you go deeper
    inside the obstacle — stronger gradient to escape from deep penetration.
    Near the boundary, it's smooth and near-zero outside.

    Best for: sampling-based planners (MPPI/MPS) where you want both
    the "zero outside" property AND a strong escape gradient inside.
    """
    return jax.nn.softplus(alpha * (margin - sdf_val)) / alpha


def cost_hinge_squared(sdf_val: jax.Array, margin: float = 0.02) -> jax.Array:
    """
    Squared hinge: max(margin - sdf, 0)^2

    Exactly zero when sdf > margin (hard cutoff, not exponential decay).
    Quadratic ramp inside. C^1 at the boundary.

    Best for: when you want STRICTLY zero cost outside (no exponential tail).
    This is what TrajOpt effectively uses.
    """
    return jnp.maximum(margin - sdf_val, 0.0) ** 2


# =============================================================================
# 3. Drop-in replacement for QuadRLTask._contact_obstacle_cost
# =============================================================================

def contact_obstacle_cost_sdf(
    body_pos_xy: jax.Array,            # (2,) robot body XY position
    foot_positions: jax.Array,         # (N, 3) or (N, 2) foot contact positions
    obstacle_centers: jax.Array,       # (M, 3) obstacle centers in world
    obstacle_half_extents: jax.Array,  # (M, 3) half-extents [hx, hy, hz]
    obstacle_types: jax.Array,         # (M,) int: 0=sphere, 1=box, 2=cylinder
    obstacle_radii: jax.Array,         # (M,) radii for spheres/cylinders
    robot_radius: float = 0.3,
    foot_radius: float = 0.02,        # dilates AABB by foot radius
    margin: float = 0.2,             # safety margin (uses rfoot)
    cost_weight: float = 100.0,       # w_col in SemLoco
    cost_fn: str = "softplus",        # "sigmoid", "softplus", "hinge_sq"
    alpha: float = 20.0,              # sharpness parameter
    use_2d: bool = True,              # ground-plane only (like SemLoco)
) -> jax.Array:
    """
    This computes smooth SDF-based cost for continuous optimization.

    The effective margin = foot_radius + margin

    Args:
        body_pos_xy: Robot body center (for body-level avoidance)
        foot_positions: Foot contact locations in world frame
        obstacle_centers: Obstacle center positions
        obstacle_half_extents: Half-sizes for box obstacles
        obstacle_types: 0=sphere, 1=box, 2=cylinder per obstacle
        obstacle_radii: Radii for sphere/cylinder obstacles
        robot_radius: Body bounding radius
        foot_radius: Foot bounding radius (SemLoco's rfoot)
        margin: Additional safety margin beyond foot_radius
        cost_weight: Penalty weight (SemLoco's w_col)
        cost_fn: Which smooth approximation to use
        alpha: Sharpness (higher = closer to binary indicator)
        use_2d: If True, only check XY plane (ground-plane foothold)

    Returns:
        Scalar cost (0.0 when all feet are safely outside obstacles)
    """
    n_obs = obstacle_centers.shape[0]
    if n_obs == 0:
        return jnp.array(0.0, dtype=jnp.float32)

    effective_margin = foot_radius + margin

    # Select cost wrapper
    if cost_fn == "sigmoid":
        _cost = lambda s: cost_indicator_smooth(s, effective_margin, alpha)
    elif cost_fn == "hinge_sq":
        _cost = lambda s: cost_hinge_squared(s, effective_margin)
    else:  # softplus (default, recommended)
        _cost = lambda s: cost_softplus(s, effective_margin, alpha)

    def _sdf_one(point, obs_idx):
        """SDF from one point to one obstacle."""
        c = obstacle_centers[obs_idx]
        he = obstacle_half_extents[obs_idx]
        r = obstacle_radii[obs_idx]
        otype = obstacle_types[obs_idx]

        if use_2d:
            # Ground-plane check: foot XY vs obstacle XY footprint
            sdf_s = jnp.linalg.norm(point[:2] - c[:2]) - r
            sdf_b = sdf_box_2d(point[:2], c[:2], he[:2])
            sdf_c = jnp.linalg.norm(point[:2] - c[:2]) - r  # cylinder = circle in 2D
        else:
            # Full 3D check
            sdf_s = sdf_sphere(point, c, r)
            sdf_b = sdf_box_3d(point, c, he)
            sdf_c = sdf_cylinder(point, c, r, he[2])

        return jnp.where(otype == 0, sdf_s,
               jnp.where(otype == 1, sdf_b, sdf_c))

    obs_indices = jnp.arange(n_obs)

    # --- Per-foot cost ---
    def _foot_cost(foot_pos):
        sdfs = jax.vmap(lambda idx: _sdf_one(foot_pos, idx))(obs_indices)
        return jnp.sum(_cost(sdfs))

    foot_cost = jnp.sum(jax.vmap(_foot_cost)(foot_positions))

    # --- Body cost (optional) ---
    def _body_sdf(obs_idx):
        c = obstacle_centers[obs_idx]
        he = obstacle_half_extents[obs_idx]
        r = obstacle_radii[obs_idx]
        otype = obstacle_types[obs_idx]
        # Body is a circle in 2D
        sdf_s = jnp.linalg.norm(body_pos_xy - c[:2]) - r - robot_radius
        sdf_b = sdf_box_2d(body_pos_xy, c[:2], he[:2]) - robot_radius
        sdf_c = sdf_s  # same as sphere in 2D
        return jnp.where(otype == 0, sdf_s,
               jnp.where(otype == 1, sdf_b, sdf_c))

    body_sdfs = jax.vmap(_body_sdf)(obs_indices)
    body_cost = jnp.sum(_cost(body_sdfs))

    return cost_weight * (foot_cost + body_cost)

