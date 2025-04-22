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

    episode_critic_loss = 0.0
    episode_actor_loss = 0.0
    num_updates = 0
    episode_avg_critic_grad = 0
    episode_max_critic_grad = 0
    episode_avg_actor_grad = 0
    episode_max_actor_grad = 0
     
    # Create a results directory
    os.makedirs("results", exist_ok=True)
    
    with open("results/training_log.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Reward", "Critic Loss", "Actor Loss",
                        "Avg Critic Grad", "Max Critic Grad",
                        "Avg Actor Grad", "Max Actor Grad"])
        
    #Training loop
    for ep in range(MAX_EPISODES):
        obs = env.reset()  #Shape: [num_agents, state_dim]
        episode_reward = np.zeros(NUM_AGENTS)
        
        for step in range(MAX_STEPS):
            #Each agent acts
            actions = []
            for i, agent in enumerate(agents):
                # adding some noise for exploration
                noise = max(0.5, (1 - ep / MAX_EPISODES*0.5))
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
                print(f"Episode terminated because: {info}")
                break
            
            #Learns every [UPDATE_FREQUENCY] steps
            if len(memory) > BATCH_SIZE and step % UPDATE_FREQUENCY == 0:
                for agent_i, agent in enumerate(agents):
                    #Samples from memory (replay buffer)
                    b_obs, b_actions, b_rewards, b_next_obs, b_dones = memory.sample(BATCH_SIZE)
                    #Updates agent
                    critic_loss, actor_loss, avg_cg, max_cg, avg_ag, max_ag = agent.update(
                    b_obs, b_actions, b_rewards, b_next_obs, b_dones, agents)
                    episode_critic_loss += critic_loss
                    episode_actor_loss += actor_loss
                    num_updates += 1
                    episode_avg_critic_grad += avg_cg
                    episode_max_critic_grad += max_cg
                    episode_avg_actor_grad += avg_ag
                    episode_max_actor_grad += max_ag
            

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
               
        avg_critic_loss = (episode_critic_loss / num_updates) if (num_updates > 0) else 0.0
        avg_actor_loss = (episode_actor_loss / num_updates) if (num_updates > 0) else 0.0
        
        with open("results/training_log.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                ep, 
                episode_reward[0],
                avg_critic_loss,
                avg_actor_loss,
                (episode_avg_critic_grad / num_updates) if (num_updates > 0) else 0.0,
                (episode_max_critic_grad / num_updates) if (num_updates > 0) else 0.0,
                (episode_avg_actor_grad / num_updates) if (num_updates > 0) else 0.0,
                (episode_max_actor_grad / num_updates) if (num_updates > 0) else 0.0
            ])
        
        print(f"Episode {ep} | Rewards: {episode_reward}")

if __name__ == "__main__":
    main()
