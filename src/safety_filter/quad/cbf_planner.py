"""
3D-LIP Linear DCBF footstep safety filter.

Paper: Peng, C. et al. "Real-Time Safe Bipedal Robot Navigation
using Linear Discrete Control Barrier Functions" (IROS 2024).

Replaces MPPI sampling with a deterministic CBF-QP over footstep
contact locations using 3D-LIP dynamics.  Single-step receding-horizon,
closed-form QP (iterative half-plane projection), no learning.

Supports configurable dynamics models:
  - "lip"      : 3D-LIP inverted pendulum (Peng et al.)
  - "unicycle" : kinematic point-mass CoM = footstep centroid
"""

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from typing import Tuple, Optional

from .bases.alg import QuadAlgSamplingParams, QuadSamplingBasedPlanner, QuadAlgTrajectory
from .task import QuadRLTaskParams, QuadRLTask
from .foot_projection import _sdf_and_grad_all

from safety_filter.bases.risk import RiskStrategy


_DEFAULT_COM_HEIGHT = 0.35
_DEFAULT_DCBF_GAMMA = 0.5
_DEFAULT_MAX_CORRECTION = 0.20


@dataclass
class CBFParams(QuadAlgSamplingParams):
    best_plan: Optional[jax.Array] = None
    exec_plan: Optional[jax.Array] = None


def _make_lip_matrices(T: float, z_c: float = _DEFAULT_COM_HEIGHT) -> Tuple[jax.Array, jax.Array]:
    omega = jnp.sqrt(9.81 / z_c)
    ch = jnp.cosh(omega * T)
    sh = jnp.sinh(omega * T)
    A = jnp.array([
        [ch,   sh / omega, 0,   0          ],
        [omega * sh, ch,   0,   0          ],
        [0,    0,          ch,  sh / omega ],
        [0,    0,          omega * sh,  ch ],
    ])
    B = jnp.array([
        [1 - ch,     0          ],
        [-omega * sh,  0        ],
        [0,           1 - ch    ],
        [0,           -omega * sh],
    ])
    return A, B


def _make_unicycle_matrices(T: float) -> Tuple[jax.Array, jax.Array]:
    A = jnp.array([
        [0.0, 0.0, 0.0, 0.0],
        [-1.0 / T, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0 / T, 0.0],
    ])
    B = jnp.array([
        [1.0, 0.0],
        [1.0 / T, 0.0],
        [0.0, 1.0],
        [0.0, 1.0 / T],
    ])
    return A, B


def _project_onto_half_plane(
    u: jax.Array,
    A: jax.Array,
    b: jax.Array,
    u_min: jax.Array,
    u_max: jax.Array,
    n_iters: int = 5,
) -> jax.Array:
    """Project u onto intersection of half-planes A @ u <= b.

    Iterative projection onto the most-violated constraint.
    """
    def _step(u, _):
        residuals = A @ u - b
        worst_idx = jnp.argmax(residuals)
        max_viol = residuals[worst_idx]
        A_worst = A[worst_idx]
        A_norm_sq = jnp.sum(A_worst ** 2) + 1e-10
        u_proj = u - jnp.maximum(max_viol, 0.0) / A_norm_sq * A_worst
        return jnp.clip(u_proj, u_min, u_max), None

    u, _ = jax.lax.scan(_step, u, None, length=n_iters)
    return u


def _cbf_correct_plan(
    plan_h: jax.Array,
    s_k: jax.Array,
    plan_lim: float,
    A: jax.Array,
    B: jax.Array,
    B_xy: jax.Array,
    sdf,
    clearance: float,
    gamma: float,
    max_correction: float,
    stance_mask: Optional[jax.Array] = None,
) -> jax.Array:
    """Apply a DCBF correction using predicted CoM (one-step lookahead).

    Evaluates h at the predicted next CoM s_nom = A@s_k + B@u_avg_nom,
    so the constraint activates preventively when the next CoM would be
    unsafe, even if the current CoM is safe.

    If stance_mask is provided (4,), only stance feet are averaged into
    the centroid and only stance feet receive the correction.
    """
    ref_world = plan_h * plan_lim
    feet_world = ref_world.reshape(4, 3)

    if stance_mask is not None:
        mask_f = stance_mask.astype(jnp.float32)
        n_stance = jnp.maximum(jnp.sum(mask_f), 1.0)
        u_avg_nom = jnp.sum(mask_f[:, None] * feet_world[:, :2], axis=0) / n_stance
    else:
        u_avg_nom = jnp.mean(feet_world[:, :2], axis=0)

    if sdf.num_obstacles == 0:
        return plan_h

    # h_k: safety at current CoM (for RHS of DCBF)
    com_xy = s_k[jnp.array([0, 2])]
    sdfs_k, _ = _sdf_and_grad_all(com_xy, sdf)
    h_k = jnp.min(sdfs_k) - clearance

    # h_nom: safety at predicted next CoM under nominal footstep
    s_nom = A @ s_k + B @ u_avg_nom
    s_nom_xy = s_nom[jnp.array([0, 2])]
    sdfs_nom, grads_nom = _sdf_and_grad_all(s_nom_xy, sdf)
    h_nom = jnp.min(sdfs_nom) - clearance
    worst_idx = jnp.argmin(sdfs_nom)
    dh_dcom = grads_nom[worst_idx]

    grad_B = jnp.array([
        dh_dcom[0] * B_xy[0, 0] + dh_dcom[1] * B_xy[1, 0],
        dh_dcom[0] * B_xy[0, 1] + dh_dcom[1] * B_xy[1, 1],
    ])

    A_cbf = -grad_B[None, :]
    b_cbf = h_nom - (1.0 - gamma) * h_k - grad_B @ u_avg_nom
    u_avg_safe = _project_onto_half_plane(u_avg_nom, A_cbf, b_cbf,
                                           u_avg_nom - max_correction,
                                           u_avg_nom + max_correction)

    correction = u_avg_safe - u_avg_nom
    mask_f = stance_mask.astype(jnp.float32) if stance_mask is not None else jnp.ones(4, dtype=jnp.float32)
    safe_feet_xy = feet_world[:, :2] + mask_f[:, None] * correction[None, :]
    safe_feet = jnp.concatenate([safe_feet_xy, feet_world[:, 2:]], axis=-1)
    return safe_feet.reshape(-1) / plan_lim


class QuadCBFPlanner(QuadSamplingBasedPlanner):
    """DCBF footstep safety filter with configurable dynamics model.

    Deterministic CBF-QP replaces MPPI sampling.  Computes safe footstep
    positions by projecting the nominal reference onto DCBF half-planes
    derived from CoM dynamics (3D-LIP or unicycle) and SDF obstacle constraints.
    """

    def __init__(
        self,
        task: QuadRLTask,
        num_samples: int = 1,
        noise_level: float = 0.6,
        cross_entropy_decay: float = _DEFAULT_DCBF_GAMMA,
        num_randomizations: int = 1,
        enabled: bool = True,
        use_geometric_fallback: bool = True,
        risk_strategy: Optional[RiskStrategy] = None,
        seed: int = 0,
        plan_horizon: int = 10,
        iterations: int = 1,
        gamma: float = 0.99,
        safety_weight: float = 1e3,
        collect_data: bool = False,
        top_fraction: float = 0.01,
        use_momentum: bool = False,
        momentum: float = 0.5,
        p_max: float = 1.0,
        c_max_decay: float = 0.9,
        include_terminal_cost: bool = True,
        dynamics_model: str = "lip",
        **kwargs,
    ) -> None:
        super().__init__(
            task=task,
            num_samples=num_samples,
            noise_level=noise_level,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            iterations=iterations,
            gamma=gamma,
            safety_weight=safety_weight,
            collect_data=collect_data,
            top_fraction=top_fraction,
            use_momentum=use_momentum,
            momentum=momentum,
            p_max=p_max,
            c_max_decay=c_max_decay,
            include_terminal_cost=include_terminal_cost,
        )

        self._enabled = enabled
        self._use_geometric_fallback = use_geometric_fallback
        self.dcbf_gamma = cross_entropy_decay
        T = self.task.command_duration

        self._dynamics_model = dynamics_model
        if dynamics_model == "unicycle":
            self._A, self._B_full = _make_unicycle_matrices(T)
        else:
            self._A, self._B_full = _make_lip_matrices(T, _DEFAULT_COM_HEIGHT)

        self._B_xy = self._B_full[jnp.array([0, 2]), :]
        self._max_correction = _DEFAULT_MAX_CORRECTION

    def init_params(
        self,
        task_params: QuadRLTaskParams,
        initial_mean: Optional[jax.Array] = None,
        ref_traj: Optional[jax.Array] = None,
        seed: int = 0,
    ) -> CBFParams:
        base = super().init_params(task_params, initial_mean, ref_traj, seed)
        abs_plan = base.ref + base.mean
        return CBFParams(
            mean=base.mean,
            best_plan=abs_plan,
            exec_plan=self.task.get_exec_plan(abs_plan),
            ref=base.ref,
            rng=base.rng,
            task_params=base.task_params,
            c_max=base.c_max,
            train_data=None,
        )

    def prepare_optimize(self, state: mjx.Data, params: CBFParams) -> CBFParams:
        abs_plan = params.best_plan
        abs_plan = jnp.concatenate([abs_plan[1:], params.ref[-1:]], axis=0)
        abs_plan = jnp.clip(abs_plan, self.task.u_min, self.task.u_max)
        new_mean = abs_plan - params.ref
        return params.replace(
            mean=new_mean,
            best_plan=abs_plan,
            exec_plan=self.task.get_exec_plan(abs_plan),
        )

    def _get_stance_mask(self, params: CBFParams, step_idx: int) -> Optional[jax.Array]:
        """Return (4,) boolean stance mask for horizon step step_idx.

        Only for the 3D-LIP model: stance mask = which feet are on the ground
        at this phase.  Unicycle model returns None (all-feet centroid).
        """
        if self._dynamics_model == "unicycle":
            return None

        tp = params.task_params
        gait = jnp.asarray(tp.current_gait, dtype=jnp.int32)
        counter = jnp.asarray(tp.goal_completion_counter, dtype=jnp.int32)
        num_phases = self.task.gait_num_phases[gait]
        phase_idx = jnp.mod(counter + jnp.asarray(step_idx, dtype=jnp.int32), num_phases)

        gait_pattern = jax.lax.dynamic_index_in_dim(
            self.task.gait_patterns, gait, axis=0, keepdims=False
        )
        return jax.lax.dynamic_index_in_dim(
            gait_pattern, phase_idx, axis=0, keepdims=False
        )

    def update_with_plan(self, state: mjx.Data, params: CBFParams) -> CBFParams:
        if not self._enabled:
            return params.replace(
                mean=jnp.zeros_like(params.mean),
                best_plan=params.ref,
            )

        base_pos = state.xpos[self.task.base_body_id]
        base_vel = state.qvel
        s_k = jnp.array([base_pos[0], base_vel[0], base_pos[1], base_vel[1]])

        sdf = self.task.obstacle_sdf
        clearance = self.task.body_sdf_margin

        if sdf.num_obstacles == 0:
            safe_plan = params.ref
        else:
            def _step(plan_out, h):
                ref_h = params.ref[h]
                stance_mask = self._get_stance_mask(params, h)
                safe_h = _cbf_correct_plan(
                    ref_h, s_k, self.task.plan_lim,
                    self._A, self._B_full, self._B_xy,
                    sdf, clearance, self.dcbf_gamma, self._max_correction,
                    stance_mask=stance_mask,
                )
                return plan_out.at[h].set(safe_h), None

            H = self.plan_horizon
            safe_plan, _ = jax.lax.scan(
                _step, jnp.zeros_like(params.ref), jnp.arange(H), length=H,
            )

        new_mean = safe_plan - params.ref
        return params.replace(mean=new_mean, best_plan=safe_plan)

    def update_without_plan(self, params: CBFParams) -> CBFParams:
        zero_offset = jnp.zeros_like(params.ref)
        return params.replace(
            mean=zero_offset,
            best_plan=params.ref,
            exec_plan=self.task.get_exec_plan(params.ref),
            train_data=None,
        )

    def sample_plans(self, params: CBFParams) -> Tuple[jax.Array, CBFParams]:
        return params.ref[None, :, :], params

    def update_params(self, params: CBFParams, rollouts: QuadAlgTrajectory) -> CBFParams:
        return params

    def finalize_best_plan(self, params: CBFParams) -> CBFParams:
        abs_plan = params.best_plan
        if self._use_geometric_fallback:
            abs_plan = self.task.project_plan_to_safe(abs_plan)
        exec_plan = self.task.get_exec_plan(abs_plan)
        return params.replace(exec_plan=exec_plan)
