import torch

#Random number generator seed used for torch and numpy in train.py to reproduce issues
SEED = 123

#Selects GPU if available for better performace
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#_______________________________________________________________________________________

#Environment Parameters:

#Number of drone agents present
NUM_AGENTS = 1

#Specifies the state dimensions for each drone, 6 values: position, velocity, goal
STATE_DIM = 6

#Number of actions that the drone can take. 3 values: velocity in three directions
ACTION_DIM = 3 
#_______________________________________________________________________________________

#Training Settings:

#Episodes of training (like epochs)
MAX_EPISODES = 10000

#Max steps in each episode
MAX_STEPS = 1000000

#Discount factor: determines how much the RL agent cares about distant future rewards relative to those in the immediate future        
GAMMA = 0.95

#Soft update rate: prevents sudden shifts in learned Q-values, for more reliable and steady learning
TAU = 0.01

#Learning rate for actor
ACTOR_LR =  0.0001

#Learning rate for critic
CRITIC_LR =  0.0003

#Gradient clipping
GRAD_CLIP = 1

#Training batch size
BATCH_SIZE = 64

#How often to do a training step
UPDATE_FREQUENCY = 20

#Buffer which stores experience in replay_buffer.py
REPLAY_BUFFER_SIZE = int(2e5)
#_______________________________________________________________________________________

#Neural Network Settings:

#Hidden layer dimension for the nerual networks in networks.py
HIDDEN_DIM = 256        
