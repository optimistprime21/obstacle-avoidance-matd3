"""
Multi-agent TD3 testing script for the ComplexEnv environment. Replace ComplexEnv with GateEnv in order to 
change the test environment. In that case do not forget to change the world name in the launch file as well.

This script loads trained actor models for two agents, runs evaluation for 25 episodes,
logs outcomes and rewards, and generates summary plots.
"""

import time
import csv
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from test_envs import ComplexEnv


def orthogonal_init(layer, gain=1.0):
    """
    Orthogonally initialize weights and set biases to zero for a given layer.
    Args:
        layer (nn.Module): The layer to initialize.
        gain (float): Scaling factor for orthogonal initialization.
    """
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor(nn.Module):
    """
    Actor network for a single agent.
    Maps observations to actions.
    """
    def __init__(self, agent_id, obs_dim_n, action_dim_n, use_orthogonal_init):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim_n[agent_id], 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim_n[agent_id])
        if use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class MATD3:
    """
    Minimal multi-agent TD3 actor wrapper for testing.
    Loads the trained actor model and selects actions.
    """
    def __init__(self, agent_id, action_dim_n, max_action, obs_dim_n, use_orthogonal_init):
        self.agent_id = agent_id
        self.max_action = max_action
        self.action_dim = action_dim_n[agent_id]
        self.actor = Actor(agent_id, obs_dim_n, action_dim_n, use_orthogonal_init)

    def choose_action(self, obs, noise_std):
        """
        Select an action for the agent given its observation, with optional Gaussian noise.
        Args:
            obs (np.ndarray): Observation for the agent.
            noise_std (float): Standard deviation of Gaussian noise for exploration.
        Returns:
            np.ndarray: Action clipped to valid range.
        """
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0)
        a = self.actor(obs).data.numpy().flatten()
        return (a + np.random.normal(0, noise_std, size=self.action_dim)).clip(-self.max_action, self.max_action)

    def load(self, step, agent_id):
        """
        Load the trained actor model parameters from disk.
        Args:
            step (int): Training step of the model.
            agent_id (int): Agent identifier.
        """
        model_path = f".../models/MATD3_actor_number_1_step_{step}k_agent_{agent_id}.pth"
        self.actor.load_state_dict(torch.load(model_path))


# === Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
max_ep_steps = 500
num_test_episodes = 25

environment_dim = 20
robot_dim = 4
state_dim = environment_dim + robot_dim
obs_dim_n = [state_dim, state_dim]
action_dim_n = [2, 2]

# Initialize environment
env = ComplexEnv(".../obstacle-avoidance-matd3/launch/test_scenario.launch", environment_dim)
time.sleep(5)  # Wait for environment to initialize

torch.manual_seed(seed)
np.random.seed(seed)

max_action = 1
use_orthogonal_init = True

# Load agents and their trained models
step = 350 # Choose the training step of the model that will be tested. 350 means 350k
agents = [MATD3(i, action_dim_n, max_action, obs_dim_n, use_orthogonal_init) for i in range(2)]
for i, agent in enumerate(agents):
    try:
        agent.load(step, i)
    except Exception as e:
        raise ValueError(f"Could not load the stored model parameters of agent {i}: {e}")


# === Testing Loop ===
episode_counter = 0
episode_timesteps = 0
obs_n = env.reset()
success_log = []
total_rewards = [0.0, 0.0]

while episode_counter < num_test_episodes:
    # Agents select actions (no exploration noise during testing)
    actions = [agent.choose_action(obs, noise_std=0) for agent, obs in zip(agents, obs_n)]
    next_obs_n, rewards, dones, targets = env.step(actions)

    # Accumulate rewards for each agent
    for i in range(2):
        total_rewards[i] += rewards[i]

    episode_timesteps += 1
    # End episode if max steps reached
    if episode_timesteps >= max_ep_steps:
        dones = [True, True]

    # On episode termination, log results and reset environment
    if all(dones):
        episode_result = {"episode": episode_counter}
        for i in range(2):
            if env.collisions[i]:
                status = "collision"
            elif env.targets[i]:
                status = "success"
            else:
                status = "timeout"
            episode_result[f"agent_{i}_status"] = status
            episode_result[f"agent_{i}_reward"] = round(total_rewards[i], 2)
        success_log.append(episode_result)
        episode_counter += 1
        obs_n = env.reset()
        episode_timesteps = 0
        total_rewards = [0.0, 0.0]
    else:
        obs_n = next_obs_n


# === Export Results to CSV ===
os.makedirs("test_logs", exist_ok=True)
csv_path = os.path.join("test_logs", "test_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["episode", "agent_0_status", "agent_0_reward", "agent_1_status", "agent_1_reward"])
    writer.writeheader()
    for entry in success_log:
        writer.writerow(entry)

print(f"\nResults saved to: {csv_path}")


# === Plot Bar Charts of Agent Outcomes ===
df = pd.DataFrame(success_log)

def plot_agent_status_counts(agent_id):
    """
    Plot and save a bar chart of outcome counts for the specified agent.
    Args:
        agent_id (int): Agent identifier (0 or 1).
    """
    status_counts = df[f"agent_{agent_id}_status"].value_counts()
    statuses = ["success", "collision", "timeout"]
    counts = [status_counts.get(s, 0) for s in statuses]

    plt.bar(statuses, counts, color=["green", "red", "gray"])
    plt.title(f"Agent {agent_id} Outcomes over {num_test_episodes} Episodes")
    plt.ylabel("Count")
    plt.xlabel("Outcome")
    plt.savefig(f"test_logs/agent_{agent_id}_outcome_bar.png")
    plt.clf()

for i in range(2):
    plot_agent_status_counts(i)

print("Bar plots saved to 'test_logs/'")
