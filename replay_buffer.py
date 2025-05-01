import random
import numpy as np
#We are using Python’s deque for a rolling buffer and random for sampling
from collections import deque
from config import PER_ALPHA, PER_BETA, PER_BETA_INCREMENT, PER_EPSILON, REPLAY_BUFFER_SIZE

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
    
class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, alpha=PER_ALPHA, beta=PER_BETA, beta_increment=PER_BETA_INCREMENT, epsilon=PER_EPSILON):
        self.tree = SumTree(buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

    def add(self, obs, actions, rewards, next_obs, dones):
        experience = (obs, actions, rewards, next_obs, dones)
        priority = self.max_priority  # Initial priority
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        indices = []
        experiences = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority, experience = self.tree.get(s)
            indices.append(idx)
            experiences.append(experience)
            priorities.append(priority)

        # Importance sampling weights
        sampling_probs = np.array(priorities) / self.tree.total()
        is_weights = np.power(len(self.tree.data) * sampling_probs, -self.beta)
        is_weights /= is_weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        return indices, experiences, is_weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            clipped_priority = np.maximum(priority + self.epsilon, 1e-6)
            clipped_priority = clipped_priority ** self.alpha
            self.tree.update(idx, clipped_priority)
            if clipped_priority > self.max_priority:
                self.max_priority = clipped_priority

    def __len__(self):
        return self.tree.n_entries
        
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = []
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(left + 1, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        if len(self.data) < self.capacity:
            self.data.append(data)
        else:
            self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])
