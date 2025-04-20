import random
import numpy as np
#We are using Python’s deque for a rolling buffer and random for sampling
from collections import deque
from config import REPLAY_BUFFER_SIZE

#This class implements a replay buffer that stores transitions for experience replay in train.py
class ReplayBuffer:
    def __init__(self, buffer_size=REPLAY_BUFFER_SIZE):
        #Initializes a replay buffer with maximum size
        self.buffer_size = buffer_size
        #Deque automatically discards oldest data once capacity is exceeded
        self.buffer = deque(maxlen=self.buffer_size)
    
    def add(self, obs, actions, rewards, next_obs, dones):
        """
        Appends a tuple representing one experience (observation, action, reward, next observation, done flag) for all agents
        obs:         list of observations for each agent [num_agents, state_dim]
        actions:     list of actions for each agent [num_agents, action_dim]
        rewards:     list of rewards for each agent [num_agents]
        next_obs:    list of next observations [num_agents, state_dim]
        dones:       list of bool for each agent
        """
        self.buffer.append((obs, actions, rewards, next_obs, dones))
    
    #Randomly samples a minibatch of experiences
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = [], [], [], [], []
        
        for experience in batch:
            o, a, r, no, d = experience
            obs.append(o)
            actions.append(a)
            rewards.append(r)
            next_obs.append(no)
            dones.append(d)
        
        #Converts to numpy arrays, useful for feeding into PyTorch models
        obs = np.array(obs)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_obs = np.array(next_obs)
        dones = np.array(dones)
        
        return obs, actions, rewards, next_obs, dones
    
    #Checks the current size of the replay buffer
    def __len__(self):
        return len(self.buffer)
