"""
3D-LIP Linear DCBF footstep safety filter.

Paper: Peng, C. et al. "Real-Time Safe Bipedal Robot Navigation
using Linear Discrete Control Barrier Functions" (IROS 2024).

Replaces MPPI sampling with a deterministic CBF-QP over footstep
contact locations using 3D-LIP dynamics.  Single-step receding-horizon,
closed-form QP (iterative half-plane projection), no learning.
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
_DEFAULT_MAX_CORRECTION = 0.10


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
    B_xy: jax.Array,
    sdf,
    clearance: float,
    gamma: float,
    max_correction: float,
) -> jax.Array:
    """Apply a mild DCBF correction to one horizon step of the plan.

    Evaluates the DCBF at the CURRENT CoM position (not a dynamics prediction)
    to avoid 3D-LIP model mismatch compounding across steps.  The geometric
    `project_plan_to_safe` (applied in finalize_best_plan) handles large
    obstacle clearance; this CBF adds a small velocity-aware adjustment.

    The DCBF constraint:
        ∇h(s_k)^T B_xy (u_avg - u_avg_nom) >= -γ * h(s_k)

    Only activates when h(s_k) < 0 (CoM inside obstacle) or h(s_k) is small.
    Correction is clamped to prevent foot oscillation.
    """
    ref_world = plan_h * plan_lim
    feet_world = ref_world.reshape(4, 3)
    u_avg_nom = jnp.mean(feet_world[:, :2], axis=0)

    if sdf.num_obstacles == 0:
        return plan_h

    com_xy = s_k[jnp.array([0, 2])]
    sdfs, grads = _sdf_and_grad_all(com_xy, sdf)
    h_k = jnp.min(sdfs) - clearance
    worst_idx = jnp.argmin(sdfs)
    dh_dcom = grads[worst_idx]

    grad_B = jnp.array([
        dh_dcom[0] * B_xy[0, 0] + dh_dcom[1] * B_xy[1, 0],
        dh_dcom[0] * B_xy[0, 1] + dh_dcom[1] * B_xy[1, 1],
    ])

    A_cbf = -grad_B[None, :]
    b_cbf = gamma * h_k - grad_B @ u_avg_nom
    u_avg_safe = _project_onto_half_plane(u_avg_nom, A_cbf, b_cbf,
                                           u_avg_nom - max_correction,
                                           u_avg_nom + max_correction)

    correction = u_avg_safe - u_avg_nom
    safe_feet_xy = feet_world[:, :2] + correction[None, :]
    safe_feet = jnp.concatenate([safe_feet_xy, feet_world[:, 2:]], axis=-1)
    return safe_feet.reshape(-1) / plan_lim


class QuadCBFPlanner(QuadSamplingBasedPlanner):
    """3D-LIP LDCBF footstep safety filter.

    Deterministic CBF-QP replaces MPPI sampling.  Computes safe footstep
    positions by projecting the nominal reference onto DCBF half-planes
    derived from 3D-LIP CoM dynamics and SDF obstacle constraints.
    """

    def __init__(
        self,
        task: QuadRLTask,
        num_samples: int = 1,
        noise_level: float = 0.6,
        cross_entropy_decay: float = _DEFAULT_DCBF_GAMMA,
        num_randomizations: int = 1,
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

        self.dcbf_gamma = cross_entropy_decay
        T = self.task.command_duration
        _, B = _make_lip_matrices(T, _DEFAULT_COM_HEIGHT)
        self._B_xy = B[jnp.array([0, 2]), :]
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

    def update_with_plan(self, state: mjx.Data, params: CBFParams) -> CBFParams:
        base_pos = state.xpos[self.task.base_body_id]
        base_vel = state.qvel
        s_k = jnp.array([base_pos[0], base_vel[0], base_pos[1], base_vel[1]])

        sdf = self.task.obstacle_sdf
        clearance = self.task.projection_clearance

        if sdf.num_obstacles == 0:
            safe_plan = params.ref
        else:
            def _step(plan_out, h):
                ref_h = params.ref[h]
                safe_h = _cbf_correct_plan(
                    ref_h, s_k, self.task.plan_lim, self._B_xy,
                    sdf, clearance, self.dcbf_gamma, self._max_correction,
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
        abs_plan = self.task.project_plan_to_safe(params.best_plan)
        exec_plan = self.task.get_exec_plan(abs_plan)
        return params.replace(exec_plan=exec_plan)
