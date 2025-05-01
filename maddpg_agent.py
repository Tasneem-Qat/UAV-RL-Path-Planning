#Implements an individual agent that uses the MADDPG algorithm
import torch
import torch.optim as optim
import numpy as np

from networks import Actor, Critic
from config import (DEVICE, GAMMA, TAU, ACTOR_LR, CRITIC_LR, 
                    NUM_AGENTS, STATE_DIM, ACTION_DIM, GRAD_CLIP)

class MADDPGAgent:
    def __init__(self, agent_index):
        """
        Each agent has:
        - index so that it can be individually tracked and updated
        - local actor
        - local critic (ours is centralized so it sees all states+actions)
        - target actor
        - target critic
        - optimizers
        """
        self.agent_index = agent_index
        
        #Actor network instantiation
        #Initializes online actor (for selecting actions) and target network (for computing target Q-values)
        self.actor = Actor(STATE_DIM, ACTION_DIM).to(DEVICE)
        self.target_actor = Actor(STATE_DIM, ACTION_DIM).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        
        # Critic network instantiation
        critic_input_dim = (STATE_DIM + ACTION_DIM) * NUM_AGENTS
        self.critic = Critic(critic_input_dim).to(DEVICE)
        self.target_critic = Critic(critic_input_dim).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        
        #Copies parameters to target
        self._hard_update(self.target_actor, self.actor)
        self._hard_update(self.target_critic, self.critic)

    def act(self, state, noise=0.0):
        """
        Converts provided state into a tensor and passes it through actor network
        state: np.array (state_dim,)
        Returns: action as np.array (action_dim,)
        """
        state_t = torch.FloatTensor(state).to(DEVICE)
        action = self.actor(state_t).detach().cpu().numpy()
        #Adds exploration Gaussian noise
        action += noise * np.random.randn(*action.shape)
        return action

    def update(self, obs, actions, rewards, next_obs, dones, all_agents, is_weights, episode_num):
        """
        takes a batch of transitions for all agents and a list of all agent objects to correctly compute target actions
        obs: shape [batch_size, num_agents, state_dim]
        actions: shape [batch_size, num_agents, action_dim]
        rewards: shape [batch_size, num_agents]
        next_obs: shape [batch_size, num_agents, state_dim]
        dones: shape [batch_size, num_agents]
        all_agents: list of all agent objects (for target actions, etc.)
        """
        #Converts to torch tensors
        obs_t = torch.FloatTensor(obs).to(DEVICE)
        actions_t = torch.FloatTensor(actions).to(DEVICE)
        rewards_t = torch.FloatTensor(rewards[:, self.agent_index]).unsqueeze(1).to(DEVICE)
        next_obs_t = torch.FloatTensor(next_obs).to(DEVICE)
        dones_t = torch.FloatTensor(dones[:, 0]).unsqueeze(1).to(DEVICE)
        is_weights_t = is_weights.to(DEVICE)
        
        # —————— Critic Update ——————
        #Builds centralized input for critic and flattens all states and all actions
        full_obs = obs_t.view(obs_t.size(0), -1)
        full_actions = actions_t.view(actions_t.size(0), -1)
        
        #Next actions (target actors)
        target_next_actions = []
        for i, agent in enumerate(all_agents):
            next_state_i = next_obs_t[:, i, :]
            target_next_action_i = agent.target_actor(next_state_i)
            target_next_actions.append(target_next_action_i)
        target_next_actions = torch.cat(target_next_actions, dim=1)

        full_next_obs = next_obs_t.view(next_obs_t.size(0), -1)
        target_critic_input = torch.cat((full_next_obs, target_next_actions), dim=1)
        
        with torch.no_grad():
            target_q_next = self.target_critic(target_critic_input)
            target_q = rewards_t + GAMMA * (1 - dones_t) * target_q_next
        
        critic_input = torch.cat((full_obs, full_actions), dim=1)
        # q_value = self.critic(critic_input)
        
        # critic_loss = torch.mean((q_value - target_q) ** 2)
        
        q_value = self.critic(critic_input)
        critic_loss = torch.mean((q_value - target_q) ** 2 * is_weights_t)

        # Compute TD errors for each sample
        td_errors = (q_value - target_q).detach().abs().squeeze().cpu().numpy()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # critic_grad_norms = []
        # for name, param in self.critic.named_parameters():
        #     if param.grad is not None:
        #         critic_grad_norms.append(param.grad.norm().item())
        
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 
            max_norm=GRAD_CLIP,  # From config.py
            norm_type=2  # L2 norm (standard)
        )
        # critic_grad_norms = [p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None] 
        
        self.critic_optimizer.step()
        
        # —————— Actor Update ——————
        #Tries to maximize Q; we differentiate w.r.t. the agent's actions
        #So we need the agent’s current policy action but other agents’ actions fixed
        current_actions = []
        for i, agent in enumerate(all_agents):
            current_actions.append(agent.actor(obs_t[:, i, :]))
        current_actions = torch.cat(current_actions, dim=1)
        
        actor_input = torch.cat((full_obs, current_actions), dim=1)
        actor_loss = -self.critic(actor_input).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP*0.5)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            max_norm=GRAD_CLIP,  # Same as critic
            norm_type=2
        )
        
        # for p in self.actor.parameters():
        #     print(p.grad.norm().item())
    
        self.actor_optimizer.step()
        
        # print(f"Agent {self.agent_index} | Critic Loss: {critic_loss.item():.4f}")
        # print(f"Agent {self.agent_index} | Actor Loss: {actor_loss.item():.4f}")
        
        # # Check gradients
        # for name, param in self.actor.named_parameters():
        #     if param.grad is None:
        #         print(f"Actor {self.agent_index} | {name}: No gradient!")
        #     else:
        #         print(f"Actor {self.agent_index} | {name}: Grad norm {param.grad.norm().item():.4f}")

        # —————— Target Networks Update ——————
        self._soft_update(self.target_actor, self.actor, TAU)
        self._soft_update(self.target_critic, self.critic, TAU)
        
        
        return critic_loss.item(), actor_loss.item(), td_errors

    def _soft_update(self, target, source, tau):
        """
        gradually adjusts the target network’s parameters
        to be closer to the online network’s parameters
        and helps smooth out fluctuations in the network parameters
        """
        for target_param, local_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def _hard_update(self, target, source):
        """
        It ensures that the target networks start out perfectly
        synchronized with the online networks
        This is important to have a good starting point before
        beginning the training process
        """
        for target_param, local_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(local_param.data)
