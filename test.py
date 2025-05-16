import csv
import numpy as np
import torch
import os
from airsim_env import AirSimEnv
from ppo_agent import PPOAgent
from networks import ActorCritic
from config import STATE_DIM, ACTION_DIM, DEVICE, MAX_STEPS

def test_trained_agent(model_path):
    # Initialize environment and agent
    env = AirSimEnv()
    policy = ActorCritic(STATE_DIM, ACTION_DIM).to(DEVICE)
    # Load checkpoint and extract policy weights
    checkpoint = torch.load(model_path)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()

    step_counter = 0

    # Testing metrics storage
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/testing_log.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total Reward",
                         "Collision Count", "Completion Time", "Termination State",
                         "Step Count", "Distance To Goal"
                         ])
        
    # Run 10 test episodes
    for ep in range(50):
        state = env.reset()
        episode_reward = 0
        done = False
        
        for step in range(MAX_STEPS):
            step_counter += 1
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(DEVICE)
                mean, std, _ = policy(state_tensor)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample().cpu().numpy()

            next_state, reward, done, details = env.step(action)
            episode_reward += reward
            state = next_state

            if done:
                break
        
        with open("test_results/testing_log.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, episode_reward,
                             env.collision_counter,
                             f"{env.completion_time:.2f}" if env.completion_time else "N/A",
                             details, step_counter, env.dist_to_goal
                             ])
            
        print(f"Test Episode {ep+1} | Total Reward: {episode_reward:.2f}")

if __name__ == "__main__":
    # Load your best saved model
    test_trained_agent("weights/ppo_agent_ep680.pth")