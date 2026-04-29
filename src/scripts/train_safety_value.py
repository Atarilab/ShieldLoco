#!/usr/bin/env python3
"""Offline pre-training script for learned safety terminal value function.

Usage:
    # Use config file as source of truth (recommended)
    python scripts/train_safety_value.py \\
        --config controllers/config/rl_mps_go2_cfg.yaml \\
        --data-dir data/safety_value_rollouts/ \\
        --output checkpoints/safety_value_pretrain.msgpack

    # Override specific hyperparameters via CLI
    python scripts/train_safety_value.py \\
        --config controllers/config/rl_mps_go2_cfg.yaml \\
        --data-dir data/safety_value_rollouts/ \\
        --mode direct \\
        --num-epochs 50 \\
        --lr 1e-4

The script reads all safety_value_* hyperparameters from the config file.
CLI arguments only override when explicitly provided.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from safety_filter.quad.safety_value import (
    SafetyValueConfig,
    compute_nstep_targets,
    init_network,
    make_train_step,
    normalize_features,
    save_checkpoint,
    save_feature_stats,
)
from safety_filter.quad.task import QuadRLTask


def parse_args():
    p = argparse.ArgumentParser(description="Offline pre-train safety value")
    default_cfg = str(PROJECT_ROOT / "controllers" / "config" / "rl_mps_go2_cfg.yaml")
    p.add_argument("--config", type=str, default=default_cfg,
                   help="YAML config file (source of truth for hyperparameters)")
    p.add_argument("--data-dir", type=str, required=True,
                   help="Directory with .pkl rollout files")
    p.add_argument("--output", type=str,
                   default="logs/checkpoints/safety_value_pretrain.msgpack")

    # Override arguments — all default to None so we know when user provided them.
    p.add_argument("--mode", type=str, default=None,
                   choices=["direct", "residual"],
                   help="Training mode (default: from config file)")
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-step", type=int, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--layer-norm", action="store_true", default=False)
    p.add_argument("--no-layer-norm", action="store_true", default=False)
    p.add_argument("--feature-norm", action="store_true", default=False)
    p.add_argument("--no-feature-norm", action="store_true", default=False)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--target-clip", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--filter-seed", type=int, default=None)
    p.add_argument("--cache-dir", type=str, default=".cache/safety_value")
    p.add_argument("--no-cache", action="store_true", default=False)
    return p.parse_args()


def build_config(rl_config: dict, args) -> SafetyValueConfig:
    """Build SafetyValueConfig from config file, allowing CLI overrides."""

    def _get(key, arg, default):
        if arg is not None:
            return arg
        return rl_config.get(key, default)

    def _bool(key, enable_flag, disable_flag, default):
        if enable_flag:
            return True
        if disable_flag:
            return False
        return rl_config.get(key, default)

    return SafetyValueConfig(
        mode=_get("safety_value_mode", args.mode, "residual"),
        hidden_dim=int(_get("safety_value_hidden_dim", args.hidden_dim, 128)),
        use_layer_norm=_bool(
            "safety_value_use_layer_norm", args.layer_norm, args.no_layer_norm, False
        ),
        use_feature_norm=_bool(
            "safety_value_use_feature_norm", args.feature_norm, args.no_feature_norm, False
        ),
        n_step=int(_get("horizon_length", _get("safety_value_n_step", args.n_step, 3))),
        gamma=float(_get("mps_gamma", _get("safety_value_gamma", args.gamma, 0.99))),
        learning_rate=float(_get("safety_value_lr", args.lr, 3e-4)),
        batch_size=int(_get("safety_value_batch_size", args.batch_size, 2048)),
        num_epochs=int(_get("safety_value_num_epochs", args.num_epochs, 30)),
        grad_clip=float(_get("safety_value_grad_clip", args.grad_clip, 1.0)),
        target_clip=float(_get("safety_value_target_clip", args.target_clip, 5000.0)),
    )


def load_rollout_data(data_dir: str, filter_seed: int | None = None):
    """Load all QuadAlgTrainData pickles from directory (recurses into seed_*/ subdirs)."""
    data_path = Path(data_dir)

    if filter_seed is not None:
        target = data_path / f"seed_{filter_seed}"
        files = sorted(target.glob("*.pkl")) if target.exists() else []
        print(f"Loading from {target} … found {len(files)} files")
    else:
        files = sorted(data_path.rglob("*.pkl"))
        seed_dirs = {}
        for f in files:
            seed_dir = f.parent.name if f.parent != data_path else "(root)"
            seed_dirs.setdefault(seed_dir, 0)
            seed_dirs[seed_dir] += 1
        for seed_dir, count in sorted(seed_dirs.items()):
            print(f"  {seed_dir}: {count} files")

    if not files:
        raise ValueError(f"No .pkl files found in {data_dir}")

    all_data = []
    for f in files:
        with open(f, "rb") as fh:
            all_data.append(pickle.load(fh))
    return all_data


def main():
    args = parse_args()
    rng = jax.random.PRNGKey(args.seed)

    with open(args.config, "r") as f:
        rl_config = yaml.safe_load(f)

    cfg = build_config(rl_config, args)

    print("=" * 60)
    print("Offline Safety Value Pre-Training")
    print("=" * 60)
    print(f"Config file: {args.config}")
    print(f"Data dir: {args.data_dir}")
    print(f"Mode: {cfg.mode}")
    print(f"Epochs: {cfg.num_epochs} | LR: {cfg.learning_rate} | Batch: {cfg.batch_size}")
    print(f"N-step: {cfg.n_step} | Gamma: {cfg.gamma}")
    print(f"Hidden dim: {cfg.hidden_dim} | LayerNorm: {cfg.use_layer_norm}")
    print(f"Feature norm: {cfg.use_feature_norm}")
    print("=" * 60)

    # Load rollout data
    rollout_data = load_rollout_data(args.data_dir, filter_seed=args.filter_seed)
    print(f"Loaded {len(rollout_data)} rollout files total")

    # Build a dummy task just for feature extraction.
    import mujoco
    mj_model_path = PROJECT_ROOT / "robots/go2/scene.xml"
    if not mj_model_path.exists():
        raise FileNotFoundError(
            f"MJCF not found at {mj_model_path}. "
            "Pass the correct --model-path or edit this script."
        )
    mj_model = mujoco.MjModel.from_xml_path(str(mj_model_path))

    dummy_policy = lambda x: jnp.zeros(12)
    task = QuadRLTask(mj_model, dummy_policy, rl_config)

    # Cache key
    import hashlib
    cache_key_str = (
        f"{args.data_dir}:{args.filter_seed}:{cfg.n_step}:{cfg.gamma}:"
        f"{cfg.use_feature_norm}:{len(rollout_data)}:{cfg.mode}"
    )
    cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.npz"

    if cache_path.exists() and not args.no_cache:
        print(f"Loading cached features/targets from {cache_path}")
        cache = np.load(cache_path)
        features = jnp.asarray(cache["features"])
        targets = jnp.asarray(cache["targets"])
        costs = jnp.asarray(cache["costs"])
        print(f"  Cached samples: {features.shape[0]}")
    else:
        CHUNK_SIZE = 50
        print(f"Computing n-step targets in chunks of {CHUNK_SIZE} rollouts …")
        from safety_filter.quad.bases.alg import QuadAlgTrainData, QuadAlgState

        all_features = []
        all_targets = []
        all_costs = []

        for chunk_start in range(0, len(rollout_data), CHUNK_SIZE):
            chunk = rollout_data[chunk_start:chunk_start + CHUNK_SIZE]
            all_costs_raw = [d.costs for d in chunk]
            max_h = max(c.shape[1] for c in all_costs_raw)

            def _pad(arr, target_T):
                if arr.shape[1] < target_T:
                    pad_w = target_T - arr.shape[1]
                    return jnp.pad(arr, ((0, 0), (0, pad_w), (0, 0), (0, 0)))
                return arr

            batched_data = QuadAlgTrainData(
                states=QuadAlgState(
                    qpos=jnp.concatenate([d.states.qpos for d in chunk], axis=0),
                    qvel=jnp.concatenate([d.states.qvel for d in chunk], axis=0),
                    xpos=jnp.concatenate([d.states.xpos for d in chunk], axis=0),
                    xquat=jnp.concatenate([d.states.xquat for d in chunk], axis=0),
                ),
                plans=jnp.concatenate([d.plans for d in chunk], axis=0),
                costs=jnp.concatenate([_pad(c, max_h) for c in all_costs_raw], axis=0),
                constraint_flags=jnp.concatenate(
                    [_pad(d.constraint_flags, max_h) for d in chunk], axis=0
                ),
            )
            features_chunk, targets_chunk = compute_nstep_targets(
                task, batched_data, cfg,
            )
            costs_chunk = batched_data.costs.reshape(-1)
            if costs_chunk.shape[0] != features_chunk.shape[0]:
                costs_chunk = jnp.broadcast_to(costs_chunk[:1], (features_chunk.shape[0],))

            all_features.append(features_chunk)
            all_targets.append(targets_chunk)
            all_costs.append(costs_chunk)
            print(f"  Chunk {chunk_start // CHUNK_SIZE + 1}/{(len(rollout_data) - 1) // CHUNK_SIZE + 1}: {features_chunk.shape[0]} samples")

        features = jnp.concatenate(all_features, axis=0)
        targets = jnp.concatenate(all_targets, axis=0)
        costs = jnp.concatenate(all_costs, axis=0)
        print(f"  Total samples after target computation: {features.shape[0]}")

        np.savez_compressed(
            cache_path,
            features=np.array(features),
            targets=np.array(targets),
            costs=np.array(costs),
        )
        print(f"  Cached to {cache_path}")

    # NaN / Inf diagnostics
    def _report_bad(name, arr):
        n_nan = int(jnp.sum(jnp.isnan(arr)))
        n_inf = int(jnp.sum(jnp.isinf(arr)))
        if n_nan or n_inf:
            print(f"  WARNING: {name} contains {n_nan} NaN, {n_inf} Inf")
        return n_nan, n_inf

    _report_bad("features", features)
    _report_bad("targets", targets)
    _report_bad("costs", costs)

    valid = (
        jnp.all(jnp.isfinite(features), axis=1)
        & jnp.isfinite(targets)
        & jnp.isfinite(costs)
    )
    n_invalid = int(jnp.sum(~valid))
    if n_invalid > 0:
        print(f"  Removing {n_invalid} invalid samples ({n_invalid/valid.shape[0]*100:.2f}%)")
        features = features[valid]
        targets = targets[valid]
        costs = costs[valid]

    t_min = float(jnp.min(targets))
    t_max = float(jnp.max(targets))
    targets = jnp.clip(targets, -cfg.target_clip, cfg.target_clip)
    print(
        f"  Target range: [{t_min:.1f}, {t_max:.1f}]  "
        f"→ clipped to [{-cfg.target_clip:.1f}, {cfg.target_clip:.1f}]"
    )
    n_clipped = int(jnp.sum((targets <= -cfg.target_clip) | (targets >= cfg.target_clip)))
    if n_clipped > 0:
        print(f"  {n_clipped} samples hit clip boundary ({n_clipped/targets.shape[0]*100:.2f}%)")

    feat_stats = None
    if cfg.use_feature_norm:
        features, feat_stats = normalize_features(features)
        print(f"Total samples: {features.shape[0]} | Feature dim: {features.shape[1]} (normalized)")
    else:
        print(f"Total samples: {features.shape[0]} | Feature dim: {features.shape[1]}")

    # Train / validation split
    N = features.shape[0]
    rng, split_rng = jax.random.split(rng)
    perm = jax.random.permutation(split_rng, N)
    n_val = max(1, int(N * 0.1))
    train_idx = perm[n_val:]
    val_idx = perm[:n_val]

    f_train, f_val = features[train_idx], features[val_idx]
    t_train, t_val = targets[train_idx], targets[val_idx]
    c_train, c_val = costs[train_idx], costs[val_idx]
    N_train = f_train.shape[0]

    # Initialize network
    params, apply_fn = init_network(rng, features.shape[-1], cfg)
    train_step, optimizer = make_train_step(apply_fn, cfg)
    opt_state = optimizer.init(params)
    print(f"Train: {N_train} | Val: {n_val}")

    def _eval_metrics(p, f, t):
        preds = apply_fn(p, f)
        mse = float(jnp.mean((preds - t) ** 2))
        rmse = mse ** 0.5
        t_var = float(jnp.var(t))
        r2 = 1.0 - mse / (t_var + 1e-12)
        return mse, rmse, r2

    mean_baseline_mse = float(jnp.mean((t_val - jnp.mean(t_train)) ** 2))
    print(f"Mean-baseline MSE on val: {mean_baseline_mse:.2f}")

    # Diagnostic: first gradient step (not jitted — avoids optax schedule issues)
    def _diag_step(p, o, f, t, c):
        def loss_fn(params):
            preds = apply_fn(params, f)
            return jnp.mean((preds - t) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(p)
        grad_norm = jnp.sqrt(jnp.sum(jnp.array([jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)])))
        p_new, o_new, _ = train_step(p, o, f, t, c)
        preds_after = apply_fn(p_new, f)
        return loss, grad_norm, jnp.mean(preds_after), jnp.std(preds_after), p_new, o_new

    diag_loss, diag_gn, diag_mean, diag_std, params, opt_state = _diag_step(
        params, opt_state,
        f_train[:cfg.batch_size], t_train[:cfg.batch_size], c_train[:cfg.batch_size]
    )
    print(
        f"\nDIAG: first-batch loss={float(diag_loss):.2f} "
        f"grad_norm={float(diag_gn):.4f} "
        f"pred_mean(norm)={float(diag_mean):.3f} pred_std(norm)={float(diag_std):.3f}"
    )
    if float(diag_gn) < 1e-6:
        print("WARNING: gradient norm is near-zero. Model will not learn.")

    # Training loop
    best_val_r2 = -1e9
    for epoch in range(cfg.num_epochs):
        rng, subrng = jax.random.split(rng)
        perm = jax.random.permutation(subrng, N_train)
        f_shuf = f_train[perm]
        t_shuf = t_train[perm]
        c_shuf = c_train[perm]

        epoch_loss = 0.0
        num_batches = 0
        for i in range(0, N_train, cfg.batch_size):
            end = min(i + cfg.batch_size, N_train)
            params, opt_state, loss = train_step(
                params, opt_state, f_shuf[i:end], t_shuf[i:end], c_shuf[i:end]
            )
            epoch_loss += float(loss)
            num_batches += 1

        if epoch % 10 == 0 or epoch == cfg.num_epochs - 1:
            val_mse, val_rmse, val_r2 = _eval_metrics(params, f_val, t_val)
            best_val_r2 = max(best_val_r2, val_r2)
            print(
                f"Epoch {epoch:3d} | train_loss={epoch_loss/num_batches:.2f} "
                f"| val_RMSE={val_rmse:.2f} val_R²={val_r2:.4f}"
            )

    val_mse, val_rmse, val_r2 = _eval_metrics(params, f_val, t_val)
    print(f"\nFinal validation  MSE={val_mse:.2f} RMSE={val_rmse:.2f} R²={val_r2:.4f}")
    print(f"Best validation R² during training: {best_val_r2:.4f}")

    preds_val = apply_fn(params, f_val)
    print(
        f"Residual range on val: [{float(jnp.min(preds_val)):.2f}, "
        f"{float(jnp.max(preds_val)):.2f}]"
    )

    os.makedirs(Path(args.output).parent, exist_ok=True)
    save_checkpoint(params, args.output)
    print(f"Checkpoint saved to {args.output}")

    if feat_stats is not None:
        stats_path = str(Path(args.output).with_suffix(".stats.npz"))
        save_feature_stats(feat_stats, stats_path)
        print(f"Feature stats saved to {stats_path}")


if __name__ == "__main__":
    main()
