"""Safety terminal value function: structured heuristic + learned residual.

Decomposition:  V(x) = V_heuristic(x) + Delta(x)

V_heuristic  : analytic SDF-based prior, zero parameters, scene-invariant.
Delta        : small Flax MLP residual, trained on MPPI rollout data.

The residual target is n-step return minus heuristic:
    Delta_target(x_t) = [ sum_{k=t}^{t+n} gamma^{k-t} c_k + gamma^n V_h(x_{t+n}) ]
                        - V_h(x_t)

This is n-step TD learning on the residual.  The heuristic provides a strong
prior so the network only learns what the analytic model misses.

------------------------------------------------------------------------
ABLATION DESIGN
------------------------------------------------------------------------
Every learnable enhancement is guarded by a flag in ``SafetyValueConfig``
so you can toggle it systematically:

    use_learned_residual      – baseline: heuristic-only when False
    use_feature_norm          – normalize input features per-batch
    use_layer_norm            – LayerNorm in the MLP
    use_residual_connection   – skip connection in the MLP
    use_descent_penalty       – Baltussen-style soft descent penalty
    use_online_adaptation     – inline SGD updates during deployment
    n_step                    – TD target horizon (1, 3, 5, ...)
    hidden_dim                – MLP width (32, 64, 128, ...)

Recommended ablation protocol:
    1. Heuristic-only  (use_learned_residual=False)
    2. Residual, all bells off
    3. + feature_norm
    4. + layer_norm
    5. + residual_connection
    6. + descent_penalty   (usually hurts; see note below)
    7. + online_adaptation

NOTE ON THE DESCENT PENALTY
---------------------------
Baltussen et al. (CDC 2025) enforce a Lyapunov-style descent condition
    V(x_{t+1}) - V(x_t) <= -cost_t
via scenario optimization.  This is rigorous for **stabilization to a fixed
origin**.  DoomMPC is **trajectory tracking through changing obstacle scenes**;
the "value" is cost-to-go, not a stabilizing Lyapunov function.  Enforcing
descent can conflict with the residual learning the true cost-to-go geometry.
Therefore the penalty is **disabled by default** and should be treated as an
experimental ablation only.
"""

from typing import Tuple, Any, Optional
from dataclasses import dataclass
from functools import partial

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
    """Hyper-parameters for the learned safety terminal value.

    Every field is a single toggle/dial so you can run clean ablations.
    """

    # ---- Master switch (backward compatibility) -----------------------------
    use_learned_residual: bool = True
    """If False, terminal_cost() always returns the heuristic."""

    # ---- Training mode -------------------------------------------------------
    training_mode: str = "residual"
    """How to form training targets.

    ``direct``   -> network predicts V(x) directly.
                    Target = n-step return (bootstrapped with heuristic).
    ``residual`` -> network predicts Delta(x) = V(x) - V_h(x).
                    Target = n-step return - V_h(x_t).
    """

    # ---- Deployment mode -----------------------------------------------------
    deployment_mode: str = "residual"
    """How to combine network output with heuristic at inference.

    ``heuristic`` -> V(x) = V_h(x)  (ignore network).
    ``direct``    -> V(x) = network(x).
    ``residual``  -> V(x) = V_h(x) + network(x).
    """

    # ---- Network architecture ------------------------------------------------
    hidden_dim: int = 128
    use_layer_norm: bool = False
    use_residual_connection: bool = False

    # ---- Training ------------------------------------------------------------
    n_step: int = 3
    gamma: float = 0.99
    learning_rate: float = 3e-4
    lr_decay_epochs: int = 30    # cosine decay to 0 over this many epochs
    batch_size: int = 256
    num_epochs: int = 30         # offline pre-training only
    grad_clip: float = 1.0       # global norm clipping
    target_clip: float = 5000.0  # clip residual targets to this range

    # ---- Feature preprocessing -----------------------------------------------
    use_feature_norm: bool = False

    # ---- Descent penalty (Baltussen et al.) ----------------------------------
    # WARNING: see module docstring.  Default OFF.
    use_descent_penalty: bool = False
    descent_penalty_weight: float = 0.1

    # ---- Online inline adaptation --------------------------------------------
    use_online_adaptation: bool = True
    train_every: int = 5
    n_inner_steps: int = 1
    time_budget_ms: float = 5.0

    # ---- Warm-start from checkpoint ------------------------------------------
    checkpoint_path: Optional[str] = None


# =============================================================================
# NETWORK
# =============================================================================


class SafetyValueNet(nn.Module):
    """Tiny MLP for the residual Delta(x)."""

    hidden_dim: int = 128
    use_layer_norm: bool = False
    use_residual_connection: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.hidden_dim)(x)
        if self.use_layer_norm:
            h = nn.LayerNorm()(h)
        h = nn.relu(h)

        if self.use_residual_connection:
            residual = h
            h = nn.Dense(self.hidden_dim)(h)
            if self.use_layer_norm:
                h = nn.LayerNorm()(h)
            h = nn.relu(h)
            h = h + residual
        else:
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
    """Vectorized heuristic over a batch of states."""
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
    """Scalar heuristic terminal cost."""
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
    n: int = 3,
    gamma: float = 0.99,
    training_mode: str = "residual",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute n-step TD training targets from MPPI rollout data.

    Supports two training modes:

        ``direct``   -> target = G_t^{(n)}  (network predicts V(x) directly)
        ``residual`` -> target = G_t^{(n)} - V_h(x_t)  (network predicts residual)

    where  G_t^{(n)} = sum_{k=t}^{t+n-1} gamma^{k-t} c_k + gamma^n * V_h(x_{t+n}).

    If ``train_data.heuristic_values`` is present (new data format), it is used
    directly; otherwise heuristics are recomputed from the stored states for
    backward compatibility with old rollout pickles.

    Returns:
        features: (N, 39)
        targets : (N,)
    """
    B, H_plus_1 = train_data.costs.shape
    costs = train_data.costs  # (B, H+1)

    # Flatten batch × time so vectorized extractors see (N, ...) not (B, T, ...)
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

    # NOTE: costs[k] in train_data are already pre-discounted by gamma^k
    # in the planner (see get_rollouts: effective_cost = ... * discount).
    # To recover the standard n-step TD target we must divide by gamma^t.
    targets = jnp.zeros((B, H_plus_1))

    for t in range(H_plus_1):
        end = min(t + n, H_plus_1)

        # costs[t:end] already contain gamma^{t+j} factors.
        # Summing and dividing by gamma^t gives sum_j raw_{t+j} * gamma^j * dt.
        step_sum = jnp.sum(costs[:, t:end], axis=1) / (gamma ** t)

        if t + n < H_plus_1:
            bootstrap = (gamma ** n) * heuristics[:, t + n]
        else:
            bootstrap = jnp.zeros(B)

        nstep_return = step_sum + bootstrap

        if training_mode == "direct":
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
        use_residual_connection=cfg.use_residual_connection,
    )


def init_network(rng: jax.Array, feature_dim: int, cfg: SafetyValueConfig):
    """Initialize the residual network and return (params, apply_fn)."""
    net = _build_network(cfg)
    dummy = jnp.ones((1, feature_dim))
    params = net.init(rng, dummy)
    return params, net.apply


def compute_descent_penalty(
    apply_fn, params, features: jnp.ndarray, costs: jnp.ndarray
) -> jnp.ndarray:
    """Soft descent penalty (Baltussen et al., CDC 2025).

    Penalizes  V(x_{t+1}) - V(x_t) + cost_t > 0  along consecutive samples.
    See module docstring for why this is experimental.
    """
    preds = apply_fn(params, features)
    delta_v = preds[1:] - preds[:-1]
    violation = delta_v + costs[:-1]
    return jnp.mean(jax.nn.relu(violation))


def make_train_step(apply_fn, cfg: SafetyValueConfig):
    """Build a JITted training step with gradient clipping and constant LR.

    Returns:
        train_step: JITted function (params, opt_state, features, targets, costs)
        optimizer:  optax.GradientTransformation (caller must init with optimizer.init(params))
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adam(cfg.learning_rate),
    )

    @jax.jit
    def train_step(params, opt_state, features, targets, costs):
        def loss_fn(p):
            preds = apply_fn(p, features)
            mse = jnp.mean((preds - targets) ** 2)

            if cfg.use_descent_penalty and cfg.descent_penalty_weight > 0.0:
                descent = compute_descent_penalty(apply_fn, p, features, costs)
                return mse + cfg.descent_penalty_weight * descent
            return mse

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step, optimizer


def train_safety_value(
    task,
    train_data: Any,
    cfg: SafetyValueConfig,
    rng_key: Optional[jax.Array] = None,
) -> Tuple[Any, Any, Optional[dict]]:
    """Train the residual network on a batch of MPPI rollouts.

    Returns:
        params    : trained Flax parameters
        apply_fn  : network apply function
        stats     : feature normalization statistics (or None)
    """
    rng = rng_key if rng_key is not None else jax.random.PRNGKey(0)

    features, targets = compute_nstep_targets(
        task,
        train_data,
        n=cfg.n_step,
        gamma=cfg.gamma,
        training_mode=cfg.training_mode,
    )

    feat_stats = None
    if cfg.use_feature_norm:
        features, feat_stats = normalize_features(features)

    N = features.shape[0]
    params, apply_fn = init_network(rng, features.shape[-1], cfg)
    train_step, optimizer = make_train_step(apply_fn, cfg)
    opt_state = optimizer.init(params)

    costs = train_data.costs.reshape(-1)
    if costs.shape[0] != N:
        costs = jnp.broadcast_to(costs[:1], (N,))

    for epoch in range(cfg.num_epochs):
        rng, subrng = jax.random.split(rng)
        perm = jax.random.permutation(subrng, N)
        features_shuf = features[perm]
        targets_shuf = targets[perm]
        costs_shuf = costs[perm]

        epoch_loss = 0.0
        num_batches = 0
        for i in range(0, N, cfg.batch_size):
            end = min(i + cfg.batch_size, N)
            f_batch = features_shuf[i:end]
            t_batch = targets_shuf[i:end]
            c_batch = costs_shuf[i:end]
            params, opt_state, loss = train_step(
                params, opt_state, f_batch, t_batch, c_batch
            )
            epoch_loss += float(loss)
            num_batches += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | loss = {epoch_loss / num_batches:.4f}")

    preds = apply_fn(params, features)
    final_mse = float(jnp.mean((preds - targets) ** 2))
    print(f"Final MSE = {final_mse:.4f}")

    return params, apply_fn, feat_stats


# =============================================================================
# SERIALISATION HELPERS
# =============================================================================


def params_to_dict(params):
    """Convert Flax params to a plain dict of numpy arrays."""
    return jax.tree_util.tree_map(lambda x: np.array(x), params)


def params_from_dict(d):
    """Restore Flax params from a plain dict."""
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), d)


import numpy as np
from pathlib import Path
from flax.serialization import msgpack_serialize, msgpack_restore


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


def normalize_targets(targets: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
    """Standardize targets to zero mean / unit variance."""
    mean = jnp.mean(targets)
    std = jnp.std(targets) + 1e-6
    return (targets - mean) / std, {"mean": mean, "std": std}


def denormalize_targets(targets_norm: jnp.ndarray, stats: dict) -> jnp.ndarray:
    """Convert standardized targets back to original scale."""
    return targets_norm * stats["std"] + stats["mean"]


# =============================================================================
# INLINE TRAINER (deterministic, cycle-skipping, JIT, zero threads)
# =============================================================================


class InlineSafetyValueTrainer:
    """Inline trainer that performs a tiny number of JITted gradient steps.

    No Python threads.  Training happens on the same GPU stream as MPPI,
    but only every ``cfg.train_every`` planning cycles.
    """

    def __init__(self, cfg: SafetyValueConfig):
        self.cfg = cfg
        rng = jax.random.PRNGKey(0)

        self._feat_stats: Optional[dict] = None
        self._target_stats: Optional[dict] = None

        ckpt = cfg.checkpoint_path
        if ckpt is not None and Path(ckpt).is_file():
            self._params = load_checkpoint(ckpt)
            net = _build_network(cfg)
            self._apply_fn = lambda p, x: net.apply(p, x)
            # Load pre-computed feature stats if they exist next to checkpoint
            stats_path = str(Path(ckpt).with_suffix(".stats.npz"))
            if Path(stats_path).exists():
                self._feat_stats = load_feature_stats(stats_path)
            # Load target normalization stats if present
            target_stats_path = str(Path(ckpt).with_suffix(".targets.npz"))
            if Path(target_stats_path).exists():
                data = np.load(target_stats_path)
                self._target_stats = {
                    "mean": float(data["mean"]),
                    "std": float(data["std"]),
                }
        else:
            if ckpt is not None and not Path(ckpt).exists():
                print(
                    f"[InlineSafetyValueTrainer] WARNING: checkpoint_path='{ckpt}' "
                    "does not exist.  Falling back to random initialization.",
                    flush=True,
                )
            self._params, self._apply_fn = init_network(rng, 39, cfg)

        self._optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip),
            optax.adam(cfg.learning_rate),
        )
        self._opt_state = self._optimizer.init(self._params)
        self._rng = rng
        self._jit_step = jax.jit(self._make_step_fn())

    def _make_step_fn(self):
        def _step(params_opt, features, targets, costs):
            params, opt_state = params_opt

            def loss_fn(p):
                preds = self._apply_fn(p, features)
                mse = jnp.mean((preds - targets) ** 2)

                if self.cfg.use_descent_penalty and self.cfg.descent_penalty_weight > 0.0:
                    descent = compute_descent_penalty(
                        self._apply_fn, p, features, costs
                    )
                    return mse + self.cfg.descent_penalty_weight * descent
                return mse

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = self._optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), loss

        return _step

    def maybe_train(
        self,
        features: jnp.ndarray,
        targets: jnp.ndarray,
        costs: jnp.ndarray,
        step_counter: int,
    ) -> Optional[Any]:
        """Conditionally train and return updated params, or None."""
        if not self.cfg.use_online_adaptation:
            return None
        if step_counter % self.cfg.train_every != 0:
            return None

        N = features.shape[0]
        if N == 0:
            return None

        if self.cfg.use_feature_norm:
            features, self._feat_stats = normalize_features(
                features, self._feat_stats
            )

        # Online target normalization: compute mean/std from current batch if
        # we don't have offline pre-computed stats.  This centers targets so
        # the network learns pattern, not mean-fighting.
        if self._target_stats is None:
            t_mean = float(jnp.mean(targets))
            t_std = float(jnp.std(targets)) + 1e-6
            self._target_stats = {"mean": t_mean, "std": t_std}
        targets_norm = (targets - self._target_stats["mean"]) / self._target_stats["std"]

        self._rng, subrng = jax.random.split(self._rng)
        perm = jax.random.permutation(subrng, N)
        end = min(self.cfg.batch_size, N)
        fb = features[perm[:end]]
        tb = targets_norm[perm[:end]]
        cb = costs[perm[:end]]

        for _ in range(self.cfg.n_inner_steps):
            (self._params, self._opt_state), _ = self._jit_step(
                (self._params, self._opt_state), fb, tb, cb
            )

        return self._params

    def get_params(self) -> Any:
        """Return the current params without training."""
        return self._params
