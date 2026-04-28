"""Replica Exchange MIMPS (REMIMPS) optimizer.

REMIMPS runs K parallel MIMPS replicas at different temperatures.  The key
novelty is a mathematically principled swap between replicas: the *mean* of
each replica (its search-distribution centre) is treated as the state, and its
energy is the empirical average cost of the samples drawn from it.  Swaps are
accepted with the standard Metropolis criterion for parallel tempering.

This formulation is rigorous because the mean is a persistent state across
iterations, and its energy is a well-defined scalar computed from the same
population used for the MPPI update.

References:
- Dong & Tong (JMLR 2022): "Replica Exchange for Non-Convex Optimization"
- Syed et al. (NeurIPS 2022): "Parallel Tempering With a Variational Reference"
- DIAL-MPC (Xue et al., ICRA 2025): Diffusion-style annealing connection
"""

from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from .bases.alg import QuadAlgSamplingParams, QuadSamplingBasedPlanner, QuadAlgTrajectory
from .task import QuadRLTaskParams, QuadRLTask

from safety_filter.bases.risk import RiskStrategy


@dataclass
class REMIMPSParams(QuadAlgSamplingParams):
    """Policy parameters for Replica Exchange MIMPS.

    The base-class fields ``mean`` and ``best_plan`` are kept as the **coldest**
    replica's state (shape ``(H, plan_dim)``) so that upstream callers (e.g.
    ``safety_manager.py``) never need to know about the internal per-replica
    arrays.  The full per-replica state lives in ``mean_replicas`` and
    ``best_plan_replicas`` (shape ``(K, H, plan_dim)``).

    Attributes:
        mean: Coldest-replica plan offset, shape (H, plan_dim).
        best_plan: Coldest-replica best absolute plan, shape (H, plan_dim).
        temperature: Temperature ladder, shape (K,).  Lower = colder.
        swap_every: Propose swaps every N iterations.
        mean_replicas: Per-replica means, shape (K, H, plan_dim).
        best_plan_replicas: Per-replica best plans, shape (K, H, plan_dim).
    """
    # All fields need defaults because the base class already has defaults.
    temperature: jax.Array = jnp.zeros(1)  # (K,)
    swap_every: int = 1
    # Dummy defaults — real shapes are set in init_params
    mean_replicas: jax.Array = jnp.zeros((1, 1, 1))  # (K, H, plan_dim)
    best_plan_replicas: jax.Array = jnp.zeros((1, 1, 1))  # (K, H, plan_dim)


class QuadREMIMPSPlanner(QuadSamplingBasedPlanner):
    """Replica Exchange MIMPS planner.

    Runs K parallel MPPI chains at different temperature levels.
    - Cold replicas (low T): exploitation, local refinement
    - Hot replicas (high T): exploration, escaping local minima

    Periodically proposes swaps between adjacent replicas.  A swap exchanges
    the *means* (search centres) and is accepted with the Metropolis
    probability derived from the empirical average cost of each mean's
    population.

    Args:
        num_replicas: Number of temperature levels (default 3).
        temp_min: Coldest temperature (default 0.1).
        temp_max: Hottest temperature (default 2.0).
        swap_every: Swap proposal frequency (default 1 = every iteration).
    """

    def __init__(
        self,
        task: QuadRLTask,
        num_samples: int,
        noise_level: float,
        temperature: float,  # Not used directly — replicas have individual temps
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
        num_replicas: int = 3,
        temp_min: float = 0.1,
        temp_max: float = 2.0,
        swap_every: int = 1,
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

        if num_replicas < 2:
            raise ValueError("num_replicas must be >= 2 for replica exchange")

        self.num_replicas = num_replicas
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.swap_every = max(1, swap_every)

        # Geometric temperature ladder
        exponents = jnp.linspace(0.0, 1.0, self.num_replicas)
        self.temperature = temp_min * (temp_max / temp_min) ** exponents

        # Keep total rollout budget fixed for fair comparison with single MIMPS
        self.samples_per_replica = max(1, num_samples // self.num_replicas)
        # Use the exact batch size we'll feed into rollouts so that
        # _empty_train_data() produces the same shape as rollout_with_randomizations().
        self._num_samples_total = self.num_replicas * self.samples_per_replica

    def _build_temperature_ladder(self) -> jax.Array:
        """Build geometric temperature ladder."""
        exponents = jnp.linspace(0.0, 1.0, self.num_replicas)
        return self.temp_min * (self.temp_max / self.temp_min) ** exponents

    # ------------------------------------------------------------------
    # Parameter initialisation
    # ------------------------------------------------------------------

    def init_params(
        self,
        task_params: QuadRLTaskParams,
        initial_mean: Optional[jax.Array] = None,
        ref_traj: Optional[jax.Array] = None,
        seed: int = 0,
    ) -> REMIMPSParams:
        """Initialize REMIMPS parameters for all replicas."""
        base_params = super().init_params(task_params, initial_mean, ref_traj, seed)

        H = self.plan_horizon
        plan_dim = self.task.plan_dim
        K = self.num_replicas

        # Per-replica means (offset from ref)
        means = jnp.zeros((K, H, plan_dim), dtype=jnp.float32)
        if initial_mean is not None:
            means = means.at[0].set(initial_mean)

        # Per-replica best plans (absolute) — coldest starts with ref, others perturbed
        best_plans = jnp.zeros((K, H, plan_dim), dtype=jnp.float32)
        abs_plan_cold = base_params.ref + means[0]
        best_plans = best_plans.at[0].set(abs_plan_cold)

        rng = base_params.rng
        for k in range(1, K):
            rng, subrng = jax.random.split(rng)
            perturb = jax.random.normal(subrng, (H, plan_dim)) * 0.1
            best_plans = best_plans.at[k].set(base_params.ref + perturb)

        temp_ladder = self._build_temperature_ladder()

        return REMIMPSParams(
            mean=means[0],  # coldest only — public interface
            best_plan=best_plans[0],  # coldest only — public interface
            temperature=temp_ladder,
            swap_every=self.swap_every,
            mean_replicas=means,
            best_plan_replicas=best_plans,
            rng=rng,
            ref=base_params.ref,
            task_params=base_params.task_params,
            c_max=base_params.c_max,
            train_data=(
                self._empty_train_data(self._num_samples_total)
                if self.collect_data
                else None
            ),
            exec_plan=self.task.get_exec_plan(abs_plan_cold),
        )

    # ------------------------------------------------------------------
    # Warm-start / reset
    # ------------------------------------------------------------------

    def prepare_optimize(self, state: mjx.Data, params: REMIMPSParams) -> REMIMPSParams:
        """Warm-start all replicas: shift best plans forward and project."""
        K = self.num_replicas

        # Detect external caller reset (e.g. safety_manager reference change).
        # Caller sets mean=zeros and best_plan=ref when resetting.
        caller_reset = jnp.all(params.mean == 0.0) & jnp.all(
            params.best_plan == params.ref
        )

        # Shift each replica's best_plan forward by one timestep
        # Tile ref tail to (K, 1, D) so it matches best_plan_replicas along axis 0
        ref_tail = jnp.repeat(params.ref[None, -1:, :], K, axis=0)
        best_shifted = jnp.concatenate(
            [params.best_plan_replicas[:, 1:, :], ref_tail], axis=1
        )
        best_shifted = jnp.clip(best_shifted, self.task.u_min, self.task.u_max)

        # Recompute means as offset from the current reference
        mean_shifted = best_shifted - params.ref[None, :, :]

        # If caller reset, propagate to all hot replicas so they don't warm-start
        # from stale plans belonging to the previous reference trajectory.
        best_shifted = jnp.where(
            caller_reset,
            jnp.broadcast_to(params.ref[None, :, :], best_shifted.shape),
            best_shifted,
        )
        mean_shifted = jnp.where(
            caller_reset,
            jnp.zeros_like(mean_shifted),
            mean_shifted,
        )

        # If any replica has NaN/Inf in its best plan (physics divergence
        # in a previous cycle), reset ALL replicas to the safe reference.
        # This is conservative but prevents NaN from persisting.
        has_nan = jnp.any(~jnp.isfinite(best_shifted))
        best_shifted = jnp.where(
            has_nan,
            jnp.broadcast_to(params.ref[None, :, :], best_shifted.shape),
            best_shifted,
        )
        mean_shifted = jnp.where(
            has_nan,
            jnp.zeros_like(mean_shifted),
            mean_shifted,
        )

        # Sync coldest replica back to public interface
        mean_cold = mean_shifted[0]
        best_cold = best_shifted[0]
        exec_plan = self.task.get_exec_plan(best_cold)

        return params.replace(
            mean=mean_cold,
            best_plan=best_cold,
            mean_replicas=mean_shifted,
            best_plan_replicas=best_shifted,
            exec_plan=exec_plan,
        )

    def update_without_plan(self, params: REMIMPSParams) -> REMIMPSParams:
        """Reset all replicas to zero offset (pure reference tracking)."""
        K = self.num_replicas
        H = self.plan_horizon
        plan_dim = self.task.plan_dim

        zero_means = jnp.zeros((K, H, plan_dim), dtype=jnp.float32)
        best_plans = jnp.repeat(params.ref[None, :, :], K, axis=0)

        return params.replace(
            mean=zero_means[0],
            best_plan=best_plans[0],
            mean_replicas=zero_means,
            best_plan_replicas=best_plans,
            exec_plan=self.task.get_exec_plan(params.ref),
            train_data=(
                self._empty_train_data(self._num_samples_total)
                if self.collect_data
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_plans(self, params: REMIMPSParams) -> Tuple[jax.Array, REMIMPSParams]:
        """Sample plans for all K replicas with temperature-scaled noise.

        Returns:
            plans: (K, N_k, H, plan_dim)
            params: Updated with new RNG.
        """
        K = self.num_replicas
        rng = params.rng

        def sample_replica(k, rng_in):
            """Sample plans for a single replica."""
            rng_out, noise_rng = jax.random.split(rng_in)

            noise = self.quad_noise(noise_rng, self.samples_per_replica)

            # Scale noise by temperature: hot replicas explore more
            temp_scale = jnp.sqrt(params.temperature[k] / self.temp_min)
            noise_scaled = noise * temp_scale

            # Mean + ref + noise.  Guard against stale NaN means.
            mean_k = jnp.where(
                jnp.all(jnp.isfinite(params.mean_replicas[k])),
                params.mean_replicas[k],
                jnp.zeros_like(params.mean_replicas[k]),
            )
            plan_k = mean_k + params.ref + noise_scaled

            # Anchor last sample to reference (guaranteed safe)
            plan_k = plan_k.at[-1].set(params.ref)

            # Project to safe set
            current_feet_xy = params.task_params.init_foot_positions[:, :2]
            plan_k = self.task.project_plan_to_safe_with_arc_check(
                plan_k, current_feet_xy
            )

            return plan_k, rng_out

        rngs = jax.random.split(rng, K)
        plans, rngs_out = jax.vmap(sample_replica, in_axes=(0, 0))(
            jnp.arange(K), rngs
        )

        return plans, params.replace(rng=rngs_out[0])

    # ------------------------------------------------------------------
    # Core optimisation loop
    # ------------------------------------------------------------------

    def update_with_plan(self, state: mjx.Data, params: REMIMPSParams) -> REMIMPSParams:
        """Run REMIMPS optimisation with periodic replica swaps."""

        K = self.num_replicas
        N_k = self.samples_per_replica
        H = self.plan_horizon
        plan_dim = self.task.plan_dim

        def _optimize_scan_body(carry, iter_idx):
            params_iter = carry

            # ---- 1. Sample plans for all replicas ----
            plans, params_iter = self.sample_plans(params_iter)
            # plans: (K, N_k, H, plan_dim)

            # ---- 2. Flatten and rollout ONCE ----
            # Using rollout_with_randomizations handles domain randomisation,
            # risk strategies, and train_data collection automatically.
            plans_flat = plans.reshape(K * N_k, H, plan_dim)
            rng, dr_rng = jax.random.split(params_iter.rng)
            rollouts_flat, train_data = self.rollout_with_randomizations(
                params_iter, state, plans_flat, dr_rng
            )
            params_iter = params_iter.replace(rng=rng, train_data=train_data)

            # ---- 3. Adapt CaT threshold ----
            params_iter = self.update_c_max(
                params_iter, rollouts_flat, self.p_max, self.c_max_decay
            )

            # ---- 4. Reshape per replica ----
            summed_costs_flat = jnp.sum(rollouts_flat.costs, axis=1)  # (K*N_k,)
            summed_costs = summed_costs_flat.reshape(K, N_k)
            plans = rollouts_flat.plans.reshape(K, N_k, H, plan_dim)

            # ---- 5. Per-replica MPPI update ----
            def update_replica(k):
                plans_k = plans[k]  # (N_k, H, plan_dim)
                costs_k = summed_costs[k]  # (N_k,)
                T_k = params_iter.temperature[k]

                # Guard against NaN/Inf costs (physics divergence etc.)
                all_valid = jnp.all(jnp.isfinite(costs_k))

                # Normalise costs (independent of absolute scale)
                cost_std = jnp.std(costs_k)
                costs_normed = (costs_k - jnp.mean(costs_k)) / (cost_std + 1e-6)
                costs_normed = jnp.clip(costs_normed, -100.0, 100.0)

                # Temperature-weighted softmax
                weights = jax.nn.softmax(-costs_normed / T_k, axis=0)
                increments = plans_k - params_iter.ref
                new_mean_k = jnp.sum(
                    weights[:, None, None] * increments, axis=0
                )


                if self.use_momentum:
                    # S Sanokowski, et. al. formulation
                    new_mean_k = new_mean_k + (self.momentum / (1 - self.momentum)) * (params_iter.mean_replicas[k] - new_mean_k)

                    # TODO: add variance updates to MPPI and Replica Exchange
                    # Select alpha < (self.momentum + 1) to avoid singularities
                    # new_variance = new_variance + (1 / (self.momentum + 1 - alpha)) * (self.momentum * (params.variance - new_variance) + alpha * new_variance)

                # Best plan for warm-starting
                best_idx = jnp.argmin(costs_k)
                new_best_plan_k = plans_k[best_idx]

                # Replica energy = empirical average cost of population
                replica_cost = jnp.mean(costs_k)

                # If any cost is non-finite, discard this update and keep the
                # previous mean/best_plan.  Setting energy to +inf guarantees
                # the swap logic will reject any exchange involving this replica.
                new_mean_k = jnp.where(
                    all_valid, new_mean_k, params_iter.mean_replicas[k]
                )
                new_best_plan_k = jnp.where(
                    all_valid, new_best_plan_k, params_iter.best_plan_replicas[k]
                )
                replica_cost = jnp.where(all_valid, replica_cost, jnp.inf)

                return new_mean_k, new_best_plan_k, replica_cost

            k_range = jnp.arange(K)
            new_means, new_best_plans, replica_costs = jax.vmap(
                update_replica
            )(k_range)

            params_iter = params_iter.replace(
                mean_replicas=new_means,
                best_plan_replicas=new_best_plans,
            )

            # ---- 6. Replica exchange (swap means) ----
            should_swap = jnp.mod(iter_idx, params_iter.swap_every) == 0

            def do_swap(p):
                means = p.mean_replicas
                temp = p.temperature
                costs = replica_costs
                rng = p.rng

                def swap_pair(i, j, means_in, rng_in):
                    ci = costs[i]
                    cj = costs[j]
                    Ti = temp[i]
                    Tj = temp[j]

                    # Reject swap if either energy is non-finite.
                    bad = (~jnp.isfinite(ci)) | (~jnp.isfinite(cj))

                    # Metropolis acceptance for swapping means.
                    # Energy of mean = empirical average cost of its population.
                    log_accept = (ci - cj) * (1.0 / Ti - 1.0 / Tj)
                    log_accept = jnp.clip(log_accept, -20.0, 20.0)
                    accept_prob = jnp.minimum(1.0, jnp.exp(log_accept))

                    rng_out, subrng = jax.random.split(rng_in)
                    accept = (~bad) & (jax.random.uniform(subrng) < accept_prob)

                    new_mean_i = jnp.where(accept, means_in[j], means_in[i])
                    new_mean_j = jnp.where(accept, means_in[i], means_in[j])

                    means_out = (
                        means_in.at[i].set(new_mean_i).at[j].set(new_mean_j)
                    )
                    return means_out, rng_out

                means_curr = means
                rng_curr = rng

                # Even-indexed pairs: (0,1), (2,3), ...
                for i in range(0, K - 1, 2):
                    means_curr, rng_curr = swap_pair(
                        i, i + 1, means_curr, rng_curr
                    )

                # Odd-indexed pairs: (1,2), (3,4), ...
                for i in range(1, K - 1, 2):
                    means_curr, rng_curr = swap_pair(
                        i, i + 1, means_curr, rng_curr
                    )

                return p.replace(mean_replicas=means_curr, rng=rng_curr)

            def no_swap(p):
                return p

            params_iter = jax.lax.cond(should_swap, do_swap, no_swap, params_iter)

            return params_iter, None

        params_final, _ = jax.lax.scan(
            f=_optimize_scan_body,
            init=params,
            xs=jnp.arange(self.iterations),
            length=self.iterations,
        )

        # Sync coldest replica back to public interface
        return params_final.replace(
            mean=params_final.mean_replicas[0],
            best_plan=params_final.best_plan_replicas[0],
        )

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finalize_best_plan(self, params: REMIMPSParams) -> REMIMPSParams:
        """Finalize best plan from coldest replica."""
        abs_plan_cold = params.best_plan_replicas[0]
        abs_plan_cold = self.task.project_plan_to_safe(abs_plan_cold)
        exec_plan = self.task.get_exec_plan(abs_plan_cold)
        return params.replace(exec_plan=exec_plan)

    # ------------------------------------------------------------------
    # Safety overrides — these should never be called in the normal flow
    # ------------------------------------------------------------------

    def f(
        self, state: mjx.Data, params: QuadAlgSamplingParams, plans: jax.Array
    ) -> QuadAlgSamplingParams:
        """REMIMPS handles rollouts inside update_with_plan; do not call f() directly."""
        raise RuntimeError(
            "QuadREMIMPSPlanner.f() must not be called directly. "
            "Rollouts are orchestrated inside update_with_plan()."
        )

    def update_params(
        self, params: REMIMPSParams, rollouts: QuadAlgTrajectory
    ) -> REMIMPSParams:
        """REMIMPS performs updates inside update_with_plan via vmapped replicas."""
        # No-op for API compatibility; actual logic lives in update_with_plan.
        return params
