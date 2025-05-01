#Contains the network architectures for the Actor and Critic used in maddpg_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HIDDEN_DIM

#The network responsible for outputting actions for the drone to take
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_DIM)
        self.ln1 = nn.LayerNorm(HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.ln1(x)
        x = F.relu(self.fc2(x))
        #For continuous action, we often use tanh to constrain outputs(keep action in range)
        x = torch.tanh(self.fc3(x))
        return x

#The network responsible for measuring how good the Actor's action was
class Critic(nn.Module):
    """
    Centralized Critic for multi-agent:
    input_dim could be sum of states + sum of actions from all agents.
    """
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        #Takes an input that is a concatenation of all agents’ states and actions
        self.fc1 = nn.Linear(input_dim, HIDDEN_DIM*2)
        self.ln1 = nn.LayerNorm(HIDDEN_DIM*2)
        self.fc2 = nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM*2)
        self.fc3 = nn.Linear(HIDDEN_DIM*2, 1)
    
    #Final output is a single scalar Q-value: quality of combined state-action pair
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.ln1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
