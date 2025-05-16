import torch

# Random number generator seed
SEED = 123

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment Parameters
NUM_AGENTS = 1  # Single agent
STATE_DIM = 13
ACTION_DIM = 3

# Training Settings
MAX_EPISODES = 2000
MAX_STEPS = 120
GAMMA = 0.95
ACTOR_LR = 0.0001
CRITIC_LR = 0.0005
GRAD_CLIP = 1.0
BATCH_SIZE = 64

# PPO-specific parameters
CLIP_EPSILON = 0.1
GAE_LAMBDA = 0.95

INITIAL_ENTROPY_COEF = 0.01     # Starting value
MIN_ENTROPY_COEF = 0.003        # Minimum value
ENTROPY_DECAY_RATE = 0.002      # Controls decay speed (higher = faster decay)

PPO_EPOCHS = 4
NUM_STEPS = 2048       # Steps per policy update
MINIBATCH_SIZE = 64
ADV_GAMMA = 0.99

# Neural Network
HIDDEN_DIM = 128