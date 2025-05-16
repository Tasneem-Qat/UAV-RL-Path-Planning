import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HIDDEN_DIM

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Shared base network
        self.base = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(HIDDEN_DIM, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        hidden = self.base(x)
        mean = self.actor_mean(hidden)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        value = self.critic(hidden).squeeze(-1)
        return mean, std, value