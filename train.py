import numpy as np
import torch
import random
import os
import csv
from airsim_env import AirSimEnv
from ppo_agent import PPOAgent
from config import *

def compute_advantages(rewards, values, dones, next_value):
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    next_value = next_value
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + ADV_GAMMA * next_value * (1 - dones[t]) - values[t]
        advantages[t] = delta + ADV_GAMMA * GAE_LAMBDA * (1 - dones[t]) * last_advantage
        last_advantage = advantages[t]
        next_value = values[t]
    returns = advantages + values
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return advantages, returns

def main():
    # Set seeds
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    env = AirSimEnv()
    agent = PPOAgent(STATE_DIM, ACTION_DIM)
    
    checkpoint_path = "weights/ppo_agent_ep200.pth"  # latest checkpoint
    if os.path.exists(checkpoint_path):
        agent.load_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint: {checkpoint_path}")
        
    # Training metrics storage
    os.makedirs("results", exist_ok=True)
    with open("results/training_log.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total Reward", "Actor Loss", "Critic Loss",
                         "Collision Count", "Completion Time", "Termination State",
                         "Step Count", "Distance To Goal"
                         ])
    
    # Training loop
    for ep in range(MAX_EPISODES):
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        state = env.reset()
        episode_reward = 0
        step_count = 0
        # Calculate entropy coefficient with exponential decay
        agent.entropy_coef = max(
            MIN_ENTROPY_COEF,
            INITIAL_ENTROPY_COEF * np.exp(-ENTROPY_DECAY_RATE * ep)
        )
        
        # Collect trajectory
        for step in range(NUM_STEPS):
            step_count += 1
            action, log_prob, value = agent.act(state)
            next_state, reward, done, details = env.step(action)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            
            state = next_state
            episode_reward += reward
            
            if done:
                print("Episode terminated because: ", details)
                break
        
        # Compute advantages and returns
        with torch.no_grad():
            _, _, last_value = agent.policy(torch.FloatTensor(state).to(DEVICE))
            
        advantages, returns = compute_advantages(
            np.array(rewards),
            np.array(values),
            np.array(dones),
            last_value.cpu().numpy()
        )
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        log_probs = np.array(log_probs)
        returns = np.array(returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy
        actor_losses = []
        critic_losses = []
        for _ in range(PPO_EPOCHS):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                idx = indices[start:end]
                
                a_loss, c_loss = agent.update(
                    states[idx],
                    actions[idx],
                    log_probs[idx],
                    returns[idx],
                    advantages[idx]
                )
                actor_losses.append(a_loss)
                critic_losses.append(c_loss)
                
        # Logging
        avg_actor_loss = np.mean(actor_losses)
        avg_critic_loss = np.mean(critic_losses)
        
        dist_to_goal = env.dist_to_goal
        
        with open("results/training_log.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, episode_reward, avg_actor_loss, avg_critic_loss,
                             env.collision_counter, 
                             f"{env.completion_time:.2f}" if env.completion_time else "N/A",
                             details, step_count, dist_to_goal
                             ])
        
        # Save model
        if (ep + 1) % 20 == 0:
            os.makedirs("weights", exist_ok=True)
            agent.save_checkpoint(f"weights/ppo_agent_ep{ep+1}.pth")
            print(f"Saved checkpoint at episode {ep+1}")

        print(f"Episode {ep} | Reward: {episode_reward:.2f} | Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f}")

if __name__ == "__main__":
    main()