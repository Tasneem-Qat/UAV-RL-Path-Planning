import torch

#Random number generator seed used for torch and numpy in train.py to reproduce issues
SEED = 123

#Selects GPU if available for better performace
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#_______________________________________________________________________________________

#Environment Parameters:

#Number of drone agents present
NUM_AGENTS = 2

#Specifies the state dimensions for each drone, 9 values: # 6 (pos + vel) + 3 (rel_goal) + 4 (sensors)
STATE_DIM = 13

#Number of actions that the drone can take. 3 values: velocity in three directions
ACTION_DIM = 3 
#_______________________________________________________________________________________

#Training Settings:

#Episodes of training (like epochs)
MAX_EPISODES = 2000

#Max steps in each episode
MAX_STEPS = 120

#Discount factor: determines how much the RL agent cares about distant future rewards relative to those in the immediate future        
GAMMA = 0.95

#Soft update rate: prevents sudden shifts in learned Q-values, for more reliable and steady learning
TAU = 0.01

#Learning rate for actor
ACTOR_LR =  0.0001

#Learning rate for critic
CRITIC_LR =  0.0001

#Gradient clipping
GRAD_CLIP = 1.0

#Training batch size
BATCH_SIZE = 64

#How often to do a training step
UPDATE_FREQUENCY = 20

#Buffer which stores experience in replay_buffer.py
REPLAY_BUFFER_SIZE = int(1e4)
#_______________________________________________________________________________________

#Neural Network Settings:

#Hidden layer dimension for the nerual networks in networks.py
HIDDEN_DIM = 128        

#Priority Experience Replay Parameters
PER_ALPHA = 0.6  # Priority exponent
PER_BETA = 0.4    # Initial importance sampling exponent
PER_BETA_INCREMENT = 0.001
PER_EPSILON = 1e-6