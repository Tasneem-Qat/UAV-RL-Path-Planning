import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from config import *
from networks import ActorCritic

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=ACTOR_LR)
        self.clip_epsilon = CLIP_EPSILON
        self.entropy_coef = INITIAL_ENTROPY_COEF

    def act(self, state):
        # Add batch dimension for neural network processing
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)  # Shape [1, 13]
        
        with torch.no_grad():
            mean, std, value = self.policy(state_tensor)
        
        # Create distribution and sample action
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        
        # Remove batch dimension before returning
        action = action.squeeze(0)  # Converts [1, 3] → [3]
        action_np = action.cpu().numpy()
        action_np = np.clip(action_np, -1.0, 1.0)
        
        log_prob = dist.log_prob(action).sum(-1)
        
        return action_np, log_prob.item(), value.item()

    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.FloatTensor(actions).to(DEVICE)
        old_log_probs = torch.FloatTensor(old_log_probs).to(DEVICE)
        returns = torch.FloatTensor(returns).to(DEVICE)
        advantages = torch.FloatTensor(advantages).to(DEVICE)

        for _ in range(PPO_EPOCHS):
            mean, std, values = self.policy(states)
            values_clipped = values.view(-1).clamp(-10, 10)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().mean()

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = 0.5 * F.mse_loss(values_clipped, returns)
            self.critic_optimizer = optim.Adam(
                self.policy.critic.parameters(), 
                lr=CRITIC_LR
            )
            loss = actor_loss + critic_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), 
                max_norm=0.5  # Tighter than actor
            )
            self.optimizer.step()

        return actor_loss.item(), critic_loss.item()
    
    def save_checkpoint(self, path):
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])