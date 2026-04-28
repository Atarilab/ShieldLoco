"""Safety terminal value function: heuristic + learned residual.

Decomposition:  V(x) = V_heuristic(x) + Delta(x)

V_heuristic: analytic SDF-based prior, zero parameters, scene-invariant.
Delta: small Flax MLP, trained on MPPI rollout data.

Controlled by ``safety_value_mode`` in config:
    "off"       – no terminal cost, planner ignores value
    "heuristic" – SDF heuristic only (no network)
    "direct"    – learned V(x) net replaces heuristic
    "residual"  – learned Delta(x) net corrects heuristic
"""

from typing import Tuple, Any, Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

from safety_filter.quad.foot_projection import (
    _sdf_and_grad_all,
    _batch_sdf_and_grad,
)


# =============================================================================
# CONFIG
# =============================================================================


@dataclass
class SafetyValueConfig:
    """Safety terminal value hyperparameters."""

    mode: str = "heuristic"  # "off" | "heuristic" | "direct" | "residual"

    # Network architecture
    hidden_dim: int = 128
    use_layer_norm: bool = False

    # Feature preprocessing
    use_feature_norm: bool = False

    # Training
    n_step: int = 3
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 256
    num_epochs: int = 30
    grad_clip: float = 1.0
    target_clip: float = 5000.0

    # Checkpoint for warm-start
    checkpoint_path: Optional[str] = None


# =============================================================================
# NETWORK
# =============================================================================


class SafetyValueNet(nn.Module):
    """Two-hidden-layer MLP for V(x) or Delta(x)."""

    hidden_dim: int = 128
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.hidden_dim)(x)
        if self.use_layer_norm:
            h = nn.LayerNorm()(h)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim)(h)
        if self.use_layer_norm:
            h = nn.LayerNorm()(h)
        h = nn.relu(h)
        out = nn.Dense(1)(h)
        return out.squeeze()


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================


def _extract_single_features(task, state) -> jnp.ndarray:
    """Scalar feature extraction for terminal_cost()."""
    base_xy = state.xpos[task.base_body_id, :2]
    base_quat = state.xquat[task.base_body_id]
    yaw = jnp.arctan2(
        2.0 * (base_quat[0] * base_quat[3] + base_quat[1] * base_quat[2]),
        1.0 - 2.0 * (base_quat[2] ** 2 + base_quat[3] ** 2),
    )
    base_vel = state.qvel[:2]
    foot_xy = state.xpos[task.foot_ids, :2]  # (4, 2)

    if task.obstacle_sdf.num_obstacles > 0:
        sdfs_b, grads_b = _sdf_and_grad_all(base_xy, task.obstacle_sdf)
        min_sdf_b = jnp.min(sdfs_b)
        nearest_idx = jnp.argmin(sdfs_b)
        nearest_rel = task.obstacle_sdf.positions[nearest_idx] - base_xy
        nearest_grad = grads_b[nearest_idx]
        k = min(3, task.obstacle_sdf.num_obstacles)
        topk_idx = jnp.argsort(sdfs_b)[:k]
        topk_sdfs = sdfs_b[topk_idx]
        topk_rel = task.obstacle_sdf.positions[topk_idx] - base_xy
        pad = 3 - k
        if pad > 0:
            topk_sdfs = jnp.concatenate([topk_sdfs, jnp.full(pad, 10.0)])
            topk_rel = jnp.concatenate([topk_rel, jnp.zeros((pad, 2))], axis=0)
    else:
        min_sdf_b = jnp.array(10.0)
        nearest_rel = jnp.zeros(2)
        nearest_grad = jnp.zeros(2)
        topk_sdfs = jnp.full(3, 10.0)
        topk_rel = jnp.zeros((3, 2))

    if task.obstacle_sdf.num_obstacles > 0:
        sdfs_f, grads_f = _batch_sdf_and_grad(foot_xy, task.obstacle_sdf)
        min_sdfs_f = jnp.min(sdfs_f, axis=1)  # (4,)
        nearest_idx_f = jnp.argmin(sdfs_f, axis=1)
        nearest_rel_f = task.obstacle_sdf.positions[nearest_idx_f] - foot_xy
    else:
        min_sdfs_f = jnp.full(4, 10.0)
        nearest_rel_f = jnp.zeros((4, 2))

    return jnp.concatenate([
        base_xy, jnp.array([yaw]), base_vel,
        foot_xy.reshape(-1),
        jnp.array([min_sdf_b]), nearest_rel, nearest_grad,
        topk_sdfs, topk_rel.reshape(-1),
        min_sdfs_f, nearest_rel_f.reshape(-1),
    ])


def _vectorized_extract_features(task, xpos, xquat, qvel) -> jnp.ndarray:
    """Vectorized feature extraction over a batch of states.

    Returns:
        features: (N, 39)
    """
    N = xpos.shape[0]
    base_xy = xpos[:, task.base_body_id, :2]
    base_quat = xquat[:, task.base_body_id, :]
    yaw = jnp.arctan2(
        2.0 * (base_quat[:, 0] * base_quat[:, 3]
               + base_quat[:, 1] * base_quat[:, 2]),
        1.0 - 2.0 * (base_quat[:, 2] ** 2 + base_quat[:, 3] ** 2),
    )
    base_vel = qvel[:, :2]
    foot_xy = xpos[:, task.foot_ids, :2]  # (N, 4, 2)

    if task.obstacle_sdf.num_obstacles > 0:
        sdfs_b, grads_b = _batch_sdf_and_grad(base_xy, task.obstacle_sdf)
        min_sdf_b = jnp.min(sdfs_b, axis=1)
        nearest_idx = jnp.argmin(sdfs_b, axis=1)
        nearest_rel = task.obstacle_sdf.positions[nearest_idx] - base_xy
        nearest_grad = grads_b[jnp.arange(N), nearest_idx, :]

        k = min(3, task.obstacle_sdf.num_obstacles)
        topk_idx = jnp.argsort(sdfs_b, axis=1)[:, :k]
        topk_sdfs = jnp.take_along_axis(sdfs_b, topk_idx, axis=1)
        topk_pos = task.obstacle_sdf.positions[topk_idx]
        topk_rel = topk_pos - base_xy[:, None, :]

        pad = 3 - k
        if pad > 0:
            topk_sdfs = jnp.concatenate(
                [topk_sdfs, jnp.full((N, pad), 10.0)], axis=1
            )
            topk_rel = jnp.concatenate(
                [topk_rel, jnp.zeros((N, pad, 2))], axis=1
            )
    else:
        min_sdf_b = jnp.full(N, 10.0)
        nearest_rel = jnp.zeros((N, 2))
        nearest_grad = jnp.zeros((N, 2))
        topk_sdfs = jnp.full((N, 3), 10.0)
        topk_rel = jnp.zeros((N, 3, 2))

    if task.obstacle_sdf.num_obstacles > 0:
        feet_flat = foot_xy.reshape(N * 4, 2)
        sdfs_f, _ = _batch_sdf_and_grad(feet_flat, task.obstacle_sdf)
        sdfs_f = sdfs_f.reshape(N, 4, -1)
        min_sdfs_f = jnp.min(sdfs_f, axis=2)
        nearest_idx_f = jnp.argmin(sdfs_f, axis=2)
        nearest_pos_f = task.obstacle_sdf.positions[nearest_idx_f]
        nearest_rel_f = nearest_pos_f - foot_xy
    else:
        min_sdfs_f = jnp.full((N, 4), 10.0)
        nearest_rel_f = jnp.zeros((N, 4, 2))

    features = jnp.concatenate([
        base_xy,
        yaw[:, None],
        base_vel,
        foot_xy.reshape(N, -1),
        min_sdf_b[:, None],
        nearest_rel,
        nearest_grad,
        topk_sdfs,
        topk_rel.reshape(N, -1),
        min_sdfs_f,
        nearest_rel_f.reshape(N, -1),
    ], axis=1)

    return features  # (N, 39)


# =============================================================================
# FEATURE NORMALIZATION
# =============================================================================


def normalize_features(
    features: jnp.ndarray, stats: Optional[dict] = None
) -> Tuple[jnp.ndarray, dict]:
    """Normalize features to zero mean / unit variance.

    Returns:
        normalized_features, stats_dict
    """
    if stats is None:
        mean = jnp.mean(features, axis=0, keepdims=True)
        std = jnp.std(features, axis=0, keepdims=True) + 1e-6
        stats = {"mean": mean, "std": std}
    else:
        mean = stats["mean"]
        std = stats["std"]

    return (features - mean) / std, stats


# =============================================================================
# HEURISTIC
# =============================================================================


def _vectorized_heuristic(task, xpos, xquat, qvel) -> jnp.ndarray:
    """Vectorized SDF heuristic over a batch of states."""
    N = xpos.shape[0]
    base_xy = xpos[:, task.base_body_id, :2]

    if task.obstacle_sdf.num_obstacles == 0:
        return jnp.zeros(N)

    sdfs_b, _ = _batch_sdf_and_grad(base_xy, task.obstacle_sdf)
    min_sdf_b = jnp.min(sdfs_b, axis=1)

    foot_xy = xpos[:, task.foot_ids, :2]
    feet_flat = foot_xy.reshape(N * 4, 2)
    sdfs_f, _ = _batch_sdf_and_grad(feet_flat, task.obstacle_sdf)
    sdfs_f = sdfs_f.reshape(N, 4, -1)
    min_sdfs_f = jnp.min(sdfs_f, axis=2)

    scale = task.foot_path_sdf_scale + 1e-6
    body_penalty = jax.nn.softplus(task.projection_clearance - min_sdf_b) / scale
    feet_penalty = jnp.sum(
        jax.nn.softplus(task.projection_clearance - min_sdfs_f), axis=1
    ) / scale

    h = (body_penalty + feet_penalty) * task.horizon_length
    return h


def compute_heuristic_terminal_cost(task, state) -> jnp.ndarray:
    """Scalar SDF heuristic terminal cost."""
    if task.obstacle_sdf.num_obstacles == 0:
        return jnp.array(0.0)

    base_xy = state.xpos[task.base_body_id, :2]
    sdfs_b, _ = _sdf_and_grad_all(base_xy, task.obstacle_sdf)
    min_sdf_b = jnp.min(sdfs_b)

    foot_xy = state.xpos[task.foot_ids, :2]
    sdfs_f, _ = _batch_sdf_and_grad(foot_xy, task.obstacle_sdf)
    min_sdfs_f = jnp.min(sdfs_f, axis=1)

    scale = task.foot_path_sdf_scale + 1e-6
    body_penalty = jax.nn.softplus(task.projection_clearance - min_sdf_b) / scale
    feet_penalty = jnp.sum(
        jax.nn.softplus(task.projection_clearance - min_sdfs_f)
    ) / scale

    h = (body_penalty + feet_penalty) * task.horizon_length
    return h


# =============================================================================
# N-STEP TARGETS
# =============================================================================


def compute_nstep_targets(
    task,
    train_data: Any,
    cfg: SafetyValueConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute n-step TD training targets from MPPI rollout data.

    mode="direct"   -> target = n-step return (network predicts V(x) directly)
    mode="residual" -> target = n-step return - V_h(x_t) (network predicts Delta)

    If ``train_data.heuristic_values`` is present, it is used directly;
    otherwise heuristics are recomputed from stored states for backward
    compatibility with old rollout pickles.

    Returns:
        features: (N, 39)
        targets : (N,)
    """
    B, H_plus_1 = train_data.costs.shape
    costs = train_data.costs  # (B, H+1)

    # Flatten batch x time so vectorized extractors see (N, ...) not (B, T, ...)
    xpos = train_data.states.xpos          # (B, T, nbody, 3)
    xquat = train_data.states.xquat        # (B, T, nbody, 4)
    qvel = train_data.states.qvel          # (B, T, nv)

    xpos_flat = xpos.reshape(B * H_plus_1, *xpos.shape[2:])
    xquat_flat = xquat.reshape(B * H_plus_1, *xquat.shape[2:])
    qvel_flat = qvel.reshape(B * H_plus_1, *qvel.shape[2:])

    features_flat = _vectorized_extract_features(task, xpos_flat, xquat_flat, qvel_flat)
    features = features_flat.reshape(B, H_plus_1, -1)

    # Use stored heuristics if available (avoids recomputation and guarantees
    # consistency with the planner config used during data collection).
    if (
        hasattr(train_data, "heuristic_values")
        and train_data.heuristic_values is not None
    ):
        heuristics = train_data.heuristic_values  # (B, H+1)
    else:
        heuristics_flat = _vectorized_heuristic(task, xpos_flat, xquat_flat, qvel_flat)
        heuristics = heuristics_flat.reshape(B, H_plus_1)

    n = cfg.n_step
    gamma = cfg.gamma
    targets = jnp.zeros((B, H_plus_1))

    for t in range(H_plus_1):
        end = min(t + n, H_plus_1)

        # costs[t:end] already contain gamma^{t+j} factors.
        # Summing and dividing by gamma^t recovers sum_j raw_{t+j} * gamma^j.
        step_sum = jnp.sum(costs[:, t:end], axis=1) / (gamma ** t)

        if t + n < H_plus_1:
            bootstrap = (gamma ** n) * heuristics[:, t + n]
        else:
            bootstrap = jnp.zeros(B)

        nstep_return = step_sum + bootstrap

        if cfg.mode == "direct":
            targets = targets.at[:, t].set(nstep_return)
        else:
            # residual mode (default)
            targets = targets.at[:, t].set(nstep_return - heuristics[:, t])

    features_flat = features.reshape(-1, features.shape[-1])
    targets_flat = targets.reshape(-1)

    return features_flat, targets_flat


# =============================================================================
# TRAINING
# =============================================================================


def _build_network(cfg: SafetyValueConfig):
    """Return a network instance matching the config."""
    return SafetyValueNet(
        hidden_dim=cfg.hidden_dim,
        use_layer_norm=cfg.use_layer_norm,
    )


def init_network(rng: jax.Array, feature_dim: int, cfg: SafetyValueConfig):
    """Initialize the network and return (params, apply_fn)."""
    net = _build_network(cfg)
    dummy = jnp.ones((1, feature_dim))
    params = net.init(rng, dummy)
    return params, net.apply


def make_train_step(apply_fn, cfg: SafetyValueConfig):
    """Build a JITted training step with gradient clipping.

    Returns:
        train_step: JITted function (params, opt_state, features, targets, costs)
        optimizer:  optax.GradientTransformation
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adam(cfg.learning_rate),
    )

    @jax.jit
    def train_step(params, opt_state, features, targets, costs):
        def loss_fn(p):
            preds = apply_fn(p, features)
            return jnp.mean((preds - targets) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step, optimizer


# =============================================================================
# SERIALISATION HELPERS
# =============================================================================


import numpy as np
from pathlib import Path
from flax.serialization import msgpack_serialize, msgpack_restore


def params_to_dict(params):
    """Convert Flax params to a plain dict of numpy arrays."""
    return jax.tree_util.tree_map(lambda x: np.array(x), params)


def params_from_dict(d):
    """Restore Flax params from a plain dict."""
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), d)


def save_checkpoint(params, path: str):
    """Save params to disk using Flax msgpack (handles nested dicts)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(msgpack_serialize(params))


def load_checkpoint(path: str):
    """Load params from disk."""
    with open(path, "rb") as f:
        return msgpack_restore(f.read())


def save_feature_stats(stats: dict, path: str):
    """Save feature normalization stats."""
    np.savez(path, mean=np.array(stats["mean"]), std=np.array(stats["std"]))


def load_feature_stats(path: str) -> dict:
    """Load feature normalization stats."""
    data = np.load(path)
    return {"mean": jnp.asarray(data["mean"]), "std": jnp.asarray(data["std"])}
