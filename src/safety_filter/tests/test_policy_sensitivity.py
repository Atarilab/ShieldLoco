"""
Standalone policy sensitivity analysis.

Loads the frozen RL policy and computes:
  1. Jacobian (∂action/∂obs) for each observation component
  2. Perturbation sweep: vary key obs fields and measure action deviation
  3. Observation logging helper (unpack 82-dim obs into named fields)

Run from src/:
    python safety_filter/tests/test_policy_sensitivity.py

Outputs:
    Printed diagnostics saved to policy_sensitivity_results.pkl
"""

import os
import sys
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

POLICY_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../controllers/policies/locomotion_go2/contact_policy_stanceSeperate.pt",
    # "../../controllers/policies/locomotion_go2/contact_policy_latest.pt",
    # "../../controllers/policies/locomotion_go2/contact_policy_feetRandom.pt",
    # "../../controllers/policies/locomotion_go2/contact_policy_stance2.pt",
)

OBS_DIM = 82
ACT_DIM = 12

OBS_FIELDS = {
    "lin_vel_b": (0, 3),
    "ang_vel_b": (3, 6),
    "projected_gravity": (6, 9),
    "contact_locations_b": (9, 33),
    "time_left": (33, 34),
    "contact_plan": (34, 42),
    "joint_pos_rel": (42, 54),
    "joint_vel": (54, 66),
    "ee_pos_rel_b": (66, 70),
    "last_action": (70, 82),
}

DEFAULT_JOINT_POS = np.array([
    0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5
], dtype=np.float32)

ACTION_SCALE = 0.35
ACTION_MAPPING = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]


def load_policy(path=None, device="cpu"):
    path = path or POLICY_PATH
    policy = torch.jit.load(path, map_location=device)
    policy.eval()
    return policy


def unpack_obs(obs):
    result = {}
    for name, (start, end) in OBS_FIELDS.items():
        result[name] = obs[start:end]
    return result


def make_synthetic_obs(
    contact_locations_b=None,
    contact_plan=None,
    ee_pos_rel_b=None,
    time_left=0.15,
):
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    if contact_locations_b is None:
        default_offsets = np.array([
            [0.1934, 0.1465, 0.0],
            [0.1934, -0.1465, 0.0],
            [-0.1934, 0.1465, 0.0],
            [-0.1934, -0.1465, 0.0],
        ], dtype=np.float32)
        contact_locations_b = np.tile(default_offsets.reshape(-1), 2)
    obs[9:33] = contact_locations_b
    if contact_plan is None:
        contact_plan = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.float32)
    obs[34:42] = contact_plan
    if ee_pos_rel_b is None:
        ee_pos_rel_b = np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    obs[66:70] = ee_pos_rel_b
    obs[33] = time_left
    obs[42:54] = DEFAULT_JOINT_POS
    obs[70:82] = DEFAULT_JOINT_POS
    return obs


def jacobian_analysis(policy, obs_np):
    obs_tensor = torch.tensor(obs_np, dtype=torch.float32, requires_grad=True)
    action = policy(obs_tensor.unsqueeze(0)).squeeze(0)
    jacobian = torch.zeros(ACT_DIM, OBS_DIM)
    for i in range(ACT_DIM):
        if obs_tensor.grad is not None:
            obs_tensor.grad.zero_()
        action[i].backward(retain_graph=True)
        jacobian[i] = obs_tensor.grad.clone()
    return jacobian.numpy(), action.detach().numpy()


def perturbation_sweep(policy, obs_np, fields=None, ranges=None):
    if fields is None:
        fields = ["contact_locations_b", "ee_pos_rel_b", "time_left"]
    if ranges is None:
        ranges = {
            "contact_locations_b": np.linspace(-0.3, 0.3, 61),
            "ee_pos_rel_b": np.linspace(0.01, 0.4, 40),
            "time_left": np.linspace(0.0, 0.3, 31),
        }

    base_action = policy(torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)).squeeze(0).detach().numpy()
    results = {}

    for field in fields:
        sweep_range = ranges.get(field)
        if sweep_range is None:
            continue
        start, end = OBS_FIELDS[field]
        actions = []
        for val in sweep_range:
            obs_mod = obs_np.copy()
            if field in ("contact_locations_b",):
                obs_mod[start:end] = obs_np[start:end] + val
            else:
                obs_mod[start:end] = val if (end - start) == 1 else np.full(end - start, val)
            a = policy(torch.tensor(obs_mod, dtype=torch.float32).unsqueeze(0)).squeeze(0).detach().numpy()
            actions.append(a)
        actions = np.array(actions)
        deviation = np.linalg.norm(actions - base_action, axis=1)
        results[field] = {
            "range": sweep_range,
            "actions": actions,
            "deviation": deviation,
            "base_action": base_action,
        }

    return results


def contact_distance_sweep(policy, obs_np):
    base_offsets = np.array([
        [0.1934, 0.1465, 0.0],
        [0.1934, -0.1465, 0.0],
        [-0.1934, 0.1465, 0.0],
        [-0.1934, -0.1465, 0.0],
    ], dtype=np.float32)

    distances = np.linspace(-0.2, 0.4, 61)
    results = {"forward": {}, "lateral": {}}

    for direction in ["forward", "lateral"]:
        actions = []
        for d in distances:
            obs_mod = obs_np.copy()
            offsets = base_offsets.copy()
            if direction == "forward":
                offsets[:, 0] += d
            else:
                offsets[:, 1] += d
            obs_mod[9:33] = np.tile(offsets.reshape(-1), 2)
            ee = np.linalg.norm(offsets.reshape(4, 3), axis=1)
            obs_mod[66:70] = ee
            a = policy(torch.tensor(obs_mod, dtype=torch.float32).unsqueeze(0)).squeeze(0).detach().numpy()
            actions.append(a)
        actions = np.array(actions)
        base_action = actions[len(distances) // 2]
        deviation = np.linalg.norm(actions - base_action, axis=1)
        results[direction] = {
            "distances": distances,
            "actions": actions,
            "deviation": deviation,
        }

    return results


def per_foot_sensitivity(policy, obs_np):
    foot_names = ["FL", "FR", "RL", "RR"]
    base_offsets = np.array([
        [0.1934, 0.1465, 0.0],
        [0.1934, -0.1465, 0.0],
        [-0.1934, 0.1465, 0.0],
        [-0.1934, -0.1465, 0.0],
    ], dtype=np.float32)

    base_action = policy(torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)).squeeze(0).detach().numpy()
    deltas = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
    results = {}

    for fi, foot in enumerate(foot_names):
        deviations_forward = []
        deviations_lateral = []
        for delta in deltas:
            offsets = base_offsets.copy()
            offsets_fwd = base_offsets.copy()
            offsets_fwd[fi, 0] += delta
            obs_fwd = obs_np.copy()
            obs_fwd[9:33] = np.tile(offsets_fwd.reshape(-1), 2)
            obs_fwd[66 + fi] = np.linalg.norm(offsets_fwd[fi])
            a_fwd = policy(torch.tensor(obs_fwd, dtype=torch.float32).unsqueeze(0)).squeeze(0).detach().numpy()
            deviations_forward.append(np.linalg.norm(a_fwd - base_action))

            offsets_lat = base_offsets.copy()
            offsets_lat[fi, 1] += delta
            obs_lat = obs_np.copy()
            obs_lat[9:33] = np.tile(offsets_lat.reshape(-1), 2)
            obs_lat[66 + fi] = np.linalg.norm(offsets_lat[fi])
            a_lat = policy(torch.tensor(obs_lat, dtype=torch.float32).unsqueeze(0)).squeeze(0).detach().numpy()
            deviations_lateral.append(np.linalg.norm(a_lat - base_action))

        results[foot] = {
            "deltas": deltas,
            "forward_deviation": np.array(deviations_forward),
            "lateral_deviation": np.array(deviations_lateral),
        }

    return results


def swing_vs_stance_sensitivity(policy, obs_np):
    trot_swing_FR_RL = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.float32)
    trot_swing_FL_RR = np.array([1, 0, 0, 1, 0, 1, 1, 0], dtype=np.float32)
    stance_all = np.ones(8, dtype=np.float32)

    base_action = policy(torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)).squeeze(0).detach().numpy()

    plans = {
        "stance_all": stance_all,
        "trot_swing_FR_RL": trot_swing_FR_RL,
        "trot_swing_FL_RR": trot_swing_FL_RR,
    }

    deltas = np.linspace(0.01, 0.3, 30)
    results = {}

    for plan_name, plan in plans.items():
        obs_base = obs_np.copy()
        obs_base[34:42] = plan
        base_a = policy(torch.tensor(obs_base, dtype=torch.float32).unsqueeze(0)).squeeze(0).detach().numpy()

        fwd_devs = []
        for delta in deltas:
            offsets = np.array([
                [0.1934 + delta, 0.1465, 0.0],
                [0.1934, -0.1465, 0.0],
                [-0.1934, 0.1465, 0.0],
                [-0.1934, -0.1465, 0.0],
            ], dtype=np.float32)
            obs_mod = obs_base.copy()
            obs_mod[9:33] = np.tile(offsets.reshape(-1), 2)
            obs_mod[66:70] = np.linalg.norm(offsets, axis=1)
            a = policy(torch.tensor(obs_mod, dtype=torch.float32).unsqueeze(0)).squeeze(0).detach().numpy()
            fwd_devs.append(np.linalg.norm(a - base_a))

        results[plan_name] = {
            "deltas": deltas,
            "forward_deviations": np.array(fwd_devs),
        }

    return results


def main():
    print("=" * 60)
    print("Policy Sensitivity Analysis")
    print("=" * 60)

    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    os.chdir(src_dir)
    print(f"Working directory: {src_dir}")

    policy_path = os.path.abspath(POLICY_PATH)
    if not os.path.exists(policy_path):
        print(f"ERROR: Policy not found at {policy_path}")
        print("Please ensure the policy file exists.")
        return

    print(f"Loading policy from: {policy_path}")
    policy = load_policy(policy_path)
    print(f"Policy loaded successfully")

    base_obs = make_synthetic_obs()
    print(f"\nBase observation shape: {base_obs.shape}")
    unpacked = unpack_obs(base_obs)
    for name, val in unpacked.items():
        if val.ndim == 1 and val.shape[0] <= 8:
            print(f"  {name}: shape={val.shape}, values={val}")
        else:
            print(f"  {name}: shape={val.shape}, mean={val.mean():.4f}, std={val.std():.4f}")

    base_action = policy(torch.tensor(base_obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).detach().numpy()
    print(f"\nBase action shape: {base_action.shape}")
    print(f"  min={base_action.min():.4f}, max={base_action.max():.4f}, mean={base_action.mean():.4f}")

    print("\n" + "-" * 60)
    print("1. Jacobian Analysis (∂action/∂obs)")
    print("-" * 60)
    jacobian, action = jacobian_analysis(policy, base_obs)
    print(f"Jacobian shape: {jacobian.shape}")

    for name, (start, end) in OBS_FIELDS.items():
        sub_jac = jacobian[:, start:end]
        frob = np.linalg.norm(sub_jac)
        max_val = np.max(np.abs(sub_jac))
        row_max = np.max(np.abs(sub_jac), axis=1)
        most_sensitive_act = np.argmax(row_max)
        print(f"  {name:25s}: Frob={frob:.4f}, max_abs={max_val:.4f}, "
              f"most_sensitive_act=joint[{most_sensitive_act}] "
              f"(sens={row_max[most_sensitive_act]:.4f})")

    print("\n" + "-" * 60)
    print("2. Perturbation Sweep")
    print("-" * 60)
    sweep_results = perturbation_sweep(policy, base_obs)
    for field, data in sweep_results.items():
        max_dev = np.max(data["deviation"])
        argmax_dev = np.argmax(data["deviation"])
        print(f"  {field:25s}: max_deviation={max_dev:.4f} "
              f"at sweep_val={data['range'][argmax_dev]:.4f}")

    print("\n" + "-" * 60)
    print("3. Contact Distance Sweep (forward/lateral)")
    print("-" * 60)
    dist_results = contact_distance_sweep(policy, base_obs)
    for direction, data in dist_results.items():
        max_dev = np.max(data["deviation"])
        argmax_idx = np.argmax(data["deviation"])
        print(f"  {direction:10s}: max_deviation={max_dev:.4f} "
              f"at distance={data['distances'][argmax_idx]:.3f}")

    print("\n" + "-" * 60)
    print("4. Per-Foot Sensitivity")
    print("-" * 60)
    foot_results = per_foot_sensitivity(policy, base_obs)
    for foot, data in foot_results.items():
        print(f"  {foot}: fwd_dev @ 0.1m={data['forward_deviation'][3]:.4f}, "
              f"lat_dev @ 0.1m={data['lateral_deviation'][3]:.4f}, "
              f"fwd_dev @ 0.3m={data['forward_deviation'][-1]:.4f}, "
              f"lat_dev @ 0.3m={data['lateral_deviation'][-1]:.4f}")

    print("\n" + "-" * 60)
    print("5. Swing vs Stance Sensitivity")
    print("-" * 60)
    swing_results = swing_vs_stance_sensitivity(policy, base_obs)
    for plan_name, data in swing_results.items():
        max_dev = np.max(data["forward_deviations"])
        print(f"  {plan_name:25s}: max_fwd_dev={max_dev:.4f}")

    all_results = {
        "base_obs": base_obs,
        "base_action": action,
        "jacobian": jacobian,
        "sweep_results": sweep_results,
        "distance_sweep": dist_results,
        "per_foot": foot_results,
        "swing_vs_stance": swing_results,
    }

    out_path = os.path.join(src_dir, "policy_sensitivity_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {out_path}")

    print("\n" + "-" * 60)
    print("Key Findings Summary")
    print("-" * 60)

    jac_norms = {}
    for name, (start, end) in OBS_FIELDS.items():
        jac_norms[name] = np.linalg.norm(jacobian[:, start:end])
    sorted_fields = sorted(jac_norms.items(), key=lambda x: x[1], reverse=True)
    print("  Observation sensitivity ranking (by Jacobian Frobenius norm):")
    for name, norm in sorted_fields:
        print(f"    {name:25s}: {norm:.4f}")

    most_important = sorted_fields[0][0]
    print(f"\n  Most sensitive input: {most_important}")
    print(f"  This confirms that the policy responds strongest to {most_important}.")
    print(f"  Large contact displacements OOD => unpredictable swing height.")


if __name__ == "__main__":
    main()