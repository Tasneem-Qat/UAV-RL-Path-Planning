import numpy as np
import torch
from airsim_env import AirSimMultiAgentEnv
from maddpg_agent import MADDPGAgent
from config import NUM_AGENTS, MAX_STEPS

def test_trained_agents(agent_paths):
    """
    agent_paths: list of file paths to the saved agent models 
                 e.g. ['agent0_actor.pth', 'agent1_actor.pth', ...]
    """
    # Create environment
    env = AirSimMultiAgentEnv()

    # Create Agents
    agents = [MADDPGAgent(i) for i in range(NUM_AGENTS)]
    
    # Load actor weights
    for i, path in enumerate(agent_paths):
        agents[i].actor.load_state_dict(torch.load(path))
        agents[i].actor.eval()
    
    # Evaluate
    for ep in range(10):  # e.g. 10 test episodes
        obs = env.reset()
        episode_reward = np.zeros(NUM_AGENTS)
        
        for step in range(MAX_STEPS):
            actions = []
            for i, agent in enumerate(agents):
                action = agent.act(obs[i], noise=0.0)  # no noise for testing
                actions.append(action)
            
            actions = np.array(actions)
            next_obs, rewards, dones, info = env.step(actions)
            
            obs = next_obs
            episode_reward += rewards
            
            if dones[0]:
                print("Episode terminated because: ", info)
                break
        
        print(f"[TEST] Episode {ep} - Rewards: {episode_reward}")

if __name__ == "__main__":
    # Provide a list of saved models for each agent
    model_paths = ["weights/agent0_actor_ep1500.pth", "weights/agent1_actor_ep2000.pth"]
    test_trained_agents(model_paths)
