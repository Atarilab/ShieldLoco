import torch
import torch.nn as nn
# 1. Load TorchScript model
ts_model = torch.jit.load("controllers/policies/locomotion_go2/contact_policy_stanceSeperate.pt")
state_dict = ts_model.state_dict()
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
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    
    def forward(self, x):
        return self.actor(x)
# 4. Load weights
model = ActorNetwork()
model.load_state_dict(state_dict)
# 5. Now use torchax!
import torchax
torchax.enable_globally()
model.to('jax')