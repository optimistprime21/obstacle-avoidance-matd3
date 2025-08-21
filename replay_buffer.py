import torch
import numpy as np

class ReplayBuffer:
    """
    ReplayBuffer stores experiences for multi-agent reinforcement learning.
    It supports storing transitions and sampling batches for training.
    """

    def __init__(self, N, buffer_size, batch_size, obs_dim_n, action_dim_n):
        """
        Args:
            N (int): Number of agents.
            buffer_size (int): Maximum number of transitions to store.
            batch_size (int): Number of samples per batch.
            obs_dim_n (list): List of observation dimensions for each agent.
            action_dim_n (list): List of action dimensions for each agent.
        """
        self.N = N  # Number of agents
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.count = 0  # Pointer to the current index for storing transitions
        self.current_size = 0  # Number of transitions currently stored

        # Initialize buffers for each agent
        self.buffer_obs_n = [np.empty((buffer_size, obs_dim_n[agent_id])) for agent_id in range(N)]
        self.buffer_a_n = [np.empty((buffer_size, action_dim_n[agent_id])) for agent_id in range(N)]
        self.buffer_r_n = [np.empty((buffer_size, 1)) for _ in range(N)]
        self.buffer_s_next_n = [np.empty((buffer_size, obs_dim_n[agent_id])) for agent_id in range(N)]
        self.buffer_done_n = [np.empty((buffer_size, 1)) for _ in range(N)]

    def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n):
        """
        Store a transition for all agents in the buffer.

        Args:
            obs_n (list): List of observations for each agent.
            a_n (list): List of actions for each agent.
            r_n (list): List of rewards for each agent.
            obs_next_n (list): List of next observations for each agent.
            done_n (list): List of done flags for each agent.
        """
        for agent_id in range(self.N):
            self.buffer_obs_n[agent_id][self.count] = obs_n[agent_id]
            self.buffer_a_n[agent_id][self.count] = a_n[agent_id]
            self.buffer_r_n[agent_id][self.count] = r_n[agent_id]
            self.buffer_s_next_n[agent_id][self.count] = obs_next_n[agent_id]
            self.buffer_done_n[agent_id][self.count] = done_n[agent_id]
        # Move pointer forward and wrap around if buffer is full
        self.count = (self.count + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self):
        """
        Sample a batch of transitions for all agents.

        Returns:
            tuple: Batch of (obs_n, a_n, r_n, obs_next_n, done_n) for each agent,
                   each as a list of torch tensors.
        """
        # Randomly select indices for sampling
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = [], [], [], [], []
        for agent_id in range(self.N):
            batch_obs_n.append(torch.tensor(self.buffer_obs_n[agent_id][index], dtype=torch.float))
            batch_a_n.append(torch.tensor(self.buffer_a_n[agent_id][index], dtype=torch.float))
            batch_r_n.append(torch.tensor(self.buffer_r_n[agent_id][index], dtype=torch.float))
            batch_obs_next_n.append(torch.tensor(self.buffer_s_next_n[agent_id][index], dtype=torch.float))
            batch_done_n.append(torch.tensor(self.buffer_done_n[agent_id][index], dtype=torch.float))

        return batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n
