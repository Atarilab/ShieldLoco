import os
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_deterministic_ops=true "
    "--xla_gpu_autotune_level=0"
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"


import torch
import torch.nn as nn
# 1. Load TorchScript model
ts_model = torch.jit.load("controllers/policies/locomotion_go2/contact_policy_latest.pt")
# ts_model = torch.jit.load("controllers/policies/locomotion_go2/contact_policy_stanceSeperate.pt")
print("=== TorchScript model structure ===")
print(ts_model)
print("\n=== TorchScript code ===")
print(ts_model.code)
state_dict = ts_model.state_dict()
dummy_input = torch.randn(1, 82)
output = ts_model(dummy_input)
print("Original model output:", output)

# # 2. Map keys correctly (actor.X → net.X)
# actor_weights = {}
# for k, v in state_dict.items():
#     if k.startswith('actor.'):
#         new_key = k.replace('actor.', 'net.')
#         actor_weights[new_key] = v

# 3. Create model with matching architecture
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim=82, action_dim=12):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),  # ELU, not ReLU!
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )
    
    def forward(self, x):
        return self.actor(x)

# 4. Load weights
model = ActorNetwork()
result = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", result.missing_keys)
print("Unexpected keys:", result.unexpected_keys)

# Verify ALL weights match
print("\n=== Weight comparison ===")
for key in state_dict:
    ts_weight = state_dict[key]
    model_weight = model
    for k in key.split('.'):
        model_weight = getattr(model_weight, k) if hasattr(model_weight, k) else model_weight[int(k)]
    if not torch.allclose(ts_weight, model_weight):
        print(f"MISMATCH: {key}")

# Check if weights actually loaded
print("\nOriginal fc1 weight norm:", state_dict['actor.0.weight'].norm())
print("Loaded fc1 weight norm:", model.actor[0].weight.norm())
print("Normal model output:", model(dummy_input))
assert torch.allclose(model(dummy_input), output, atol=1e-5), "Loaded model output differs from original!"


# 5. Now use torchax!
import torchax
torchax.enable_globally()
model.to('jax')

# 6. Test forward pass
output = ts_model(dummy_input)
dummy_input_jax = dummy_input.to('jax')
output_jax = model(dummy_input_jax)

# 7. Compare outputs
print("PyTorch output:", output)
print("JAX output:", output_jax)
assert torch.allclose(output.to('cpu'), output_jax.to('cpu'), atol=1e-5), "Outputs differ!"
