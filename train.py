import numpy as np
import torch
import random
import os
import csv

from airsim_env import AirSimMultiAgentEnv
from maddpg_agent import MADDPGAgent
from replay_buffer import ReplayBuffer
from config import (MAX_EPISODES, MAX_STEPS, BATCH_SIZE, UPDATE_FREQUENCY, 
                    NUM_AGENTS, SEED)

def main():
    #Set seeds for reproducibility
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    #Creates environment
    env = AirSimMultiAgentEnv()
    
    #Creates multiple agents
    agents = [MADDPGAgent(i) for i in range(NUM_AGENTS)]
    
    #Creates replay buffer
    memory = ReplayBuffer()

    episode_critic_loss = [0.0] * NUM_AGENTS
    episode_actor_loss = [0.0] * NUM_AGENTS
    episode_avg_critic_grad = [0.0] * NUM_AGENTS
    episode_max_critic_grad = [0.0] * NUM_AGENTS
    episode_avg_actor_grad = [0.0] * NUM_AGENTS
    episode_max_actor_grad = [0.0] * NUM_AGENTS
    num_updates_per_agent = [0] * NUM_AGENTS
    episode_critic_gn = [0.0] * NUM_AGENTS
    episode_actor_gn = [0.0] * NUM_AGENTS
    avg_critic_gn = [0.0] * NUM_AGENTS
    avg_actor_gn = [0.0] * NUM_AGENTS
    
    # Create a results directory
    os.makedirs("results", exist_ok=True)
    
    with open("results/training_log.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Episode", "Reward 1", "Reward 2",
            "Critic Loss 1", "Critic Loss 2",
            "Actor Loss 1", "Actor Loss 2" 
        ])
        
    #Training loop
    for ep in range(MAX_EPISODES):
        obs = env.reset()  #Shape: [num_agents, state_dim]
        episode_reward = np.zeros(NUM_AGENTS)
        
        for step in range(MAX_STEPS):
            #Each agent acts
            actions = []
            for i, agent in enumerate(agents):
                # adding some noise for exploration
                noise = max(0.1, (1 - ep / MAX_EPISODES))
                action = agent.act(obs[i], noise=noise)
                actions.append(action)
            actions = np.array(actions)
            
            #Steps environment
            next_obs, rewards, dones, info = env.step(actions)
            
            #Stores transition
            memory.add(obs, actions, rewards, next_obs, dones)
            # print(f"Buffer size: {len(memory)}")

            obs = next_obs
            episode_reward += rewards

            # print("Obs shape:", obs.shape)  # Should be [num_agents, state_dim]
            # print("Actions shape:", actions.shape)  # [num_agents, action_dim]

            #Checks done
            if dones[0]:
                print("Episode terminated because: ", info)
                break
            
            #Learns every [UPDATE_FREQUENCY] steps
            if len(memory) > BATCH_SIZE and step % UPDATE_FREQUENCY == 0:
                for agent_i, agent in enumerate(agents):
                    # Sample and update
                    b_obs, b_actions, b_rewards, b_next_obs, b_dones = memory.sample(BATCH_SIZE)
                    
                    # Modified to receive gradient norms from agent.update()
                    (critic_loss, actor_loss) = agent.update(b_obs, b_actions, b_rewards, b_next_obs, b_dones, agents, episode_num=ep)
                    
                    # Track per-agent metrics
                    episode_critic_loss[agent_i] += critic_loss
                    episode_actor_loss[agent_i] += actor_loss
                    num_updates_per_agent[agent_i] += 1
            

        # Inside the training loop (train.py), after each episode:
        if (ep + 1) % 20 == 0:
        # Create the directory if it doesn't exist
            os.makedirs("weights", exist_ok=True)
            for i, agent in enumerate(agents):
                torch.save(
                    agent.actor.state_dict(), 
                    f"weights/agent{i}_actor_ep{ep}.pth"  # Save to weights/
                )
            print("Saved model checkpoints!") 
               
        avg_critic_loss = [
            episode_critic_loss[i] / num_updates_per_agent[i] 
            if num_updates_per_agent[i] > 0 else 0.0 
            for i in range(NUM_AGENTS)
        ]
        avg_actor_loss = [
            episode_actor_loss[i] / num_updates_per_agent[i] 
            if num_updates_per_agent[i] > 0 else 0.0 
            for i in range(NUM_AGENTS)
        ]
        
        with open("results/training_log.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                ep, 
                episode_reward[0], episode_reward[1],
                avg_critic_loss[0], avg_critic_loss[1],
                avg_actor_loss[0], avg_actor_loss[1]
            ])
        
        print(f"Episode {ep} | Rewards: {episode_reward}")

if __name__ == "__main__":
    main()
