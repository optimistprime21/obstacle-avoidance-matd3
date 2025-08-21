import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

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
        a = torch.tanh(self.fc3(x))
        return a

class Critic(nn.Module):
    """
    Critic network shared by all agents.
    Evaluates the value of joint state-action pairs.
    Implements double Q-learning.
    """
    def __init__(self, obs_dim_n, action_dim_n, use_orthogonal_init):
        super(Critic, self).__init__()
        input_dim = sum(obs_dim_n) + sum(action_dim_n)
        # First Q-network
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        # Second Q-network
        self.fc4 = nn.Linear(input_dim, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 1)
        if use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)
            orthogonal_init(self.fc5)
            orthogonal_init(self.fc6)

    def forward(self, s, a):
        """
        Forward pass for both Q-networks.
        Args:
            s (list of tensors): Observations for all agents.
            a (list of tensors): Actions for all agents.
        Returns:
            q1, q2: Q-values from both networks.
        """
        s = torch.cat(s, dim=1)
        a = torch.cat(a, dim=1)
        s_a = torch.cat([s, a], dim=1)

        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(s_a))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, s, a):
        """
        Forward pass for the first Q-network only.
        Args:
            s (list of tensors): Observations for all agents.
            a (list of tensors): Actions for all agents.
        Returns:
            q1: Q-value from the first network.
        """
        s = torch.cat(s, dim=1)
        a = torch.cat(a, dim=1)
        s_a = torch.cat([s, a], dim=1)
        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1

class MATD3(object):
    """
    Multi-Agent TD3 algorithm implementation.
    Each agent has its own actor, but shares the critic.
    Handles action selection, training, and model saving.
    """
    def __init__(self, agent_id, action_dim_n, max_action, lr_a, lr_c, gamma, 
                 tau, use_grad_clip, policy_noise, noise_clip, policy_update_freq,
                 obs_dim_n, use_orthogonal_init):
        self.N = 2  # Number of agents
        self.agent_id = agent_id
        self.max_action = max_action
        self.action_dim = action_dim_n[agent_id]
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma
        self.tau = tau
        self.use_grad_clip = use_grad_clip
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_update_freq = policy_update_freq
        self.actor_pointer = 0

        # Create actor and critic networks (and their targets)
        self.actor = Actor(agent_id, obs_dim_n, action_dim_n, use_orthogonal_init)
        self.critic = Critic(obs_dim_n, action_dim_n, use_orthogonal_init)
        self.actor_target = Actor(agent_id, obs_dim_n, action_dim_n, use_orthogonal_init)
        self.critic_target = Critic(obs_dim_n, action_dim_n, use_orthogonal_init)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        # TensorBoard writer for logging
        self.writer = SummaryWriter(log_dir=f'runs/MATD3/agent_{agent_id}')
        self.iter_count = 0  # To track training steps

    def choose_action(self, obs, noise_std):
        """
        Select an action for the agent given its observation, with optional exploration noise.
        Args:
            obs (np.ndarray): Observation for the agent.
            noise_std (float): Standard deviation of Gaussian noise for exploration.
        Returns:
            np.ndarray: Action clipped to valid range.
        """
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0)
        with torch.no_grad():
            a = self.actor(obs).data.numpy().flatten()
        # Add exploration noise and clip to valid action range
        a = (a + np.random.normal(0, noise_std, size=self.action_dim)).clip(-self.max_action, self.max_action)
        return a

    def train(self, replay_buffer, agent_n):
        """
        Train the agent using a batch of experiences from the replay buffer.
        Args:
            replay_buffer (ReplayBuffer): Buffer containing experiences.
            agent_n (list): List of all agent instances.
        """
        self.actor_pointer += 1
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample() 

        # --- Compute target Q-values using target networks ---
        with torch.no_grad():
            batch_a_next_n = []
            for i in range(self.N):
                batch_a_next = agent_n[i].actor_target(batch_obs_next_n[i])
                noise = (torch.randn_like(batch_a_next) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                batch_a_next = (batch_a_next + noise).clamp(-self.max_action, self.max_action)
                batch_a_next_n.append(batch_a_next)

            Q1_next, Q2_next = self.critic_target(batch_obs_next_n, batch_a_next_n)
            target_Q = batch_r_n[self.agent_id] + self.gamma * (1 - batch_done_n[self.agent_id]) * torch.min(Q1_next, Q2_next)

        # --- Compute current Q-values and critic loss ---
        current_Q1, current_Q2 = self.critic(batch_obs_n, batch_a_n)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        av_Q = torch.mean(target_Q).item()
        max_Q = torch.max(target_Q).item()

        # --- Optimize the critic network ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # --- Delayed policy updates for the actor ---
        if self.actor_pointer % self.policy_update_freq == 0:
            print("Policy is getting updated")
            # Update only the current agent's action
            batch_a_n[self.agent_id] = self.actor(batch_obs_n[self.agent_id])
            actor_loss = -self.critic.Q1(batch_obs_n, batch_a_n).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()

            # --- Soft update target networks ---
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # --- Logging to TensorBoard ---
            self.iter_count += 1
            self.writer.add_scalar("critic_loss", critic_loss.item(), self.iter_count)
            self.writer.add_scalar("actor_loss", actor_loss.item(), self.iter_count)
            self.writer.add_scalar("avg_Q", av_Q, self.iter_count)
            self.writer.add_scalar("max_Q", max_Q, self.iter_count)     

    def save_model(self, number, total_steps, agent_id):
        """
        Save the actor model to disk.
        Args:
            number (int): Experiment number.
            total_steps (int): Current training step.
            agent_id (int): Agent identifier.
        """
        torch.save(
            self.actor.state_dict(),
            f"./model/MATD3_actor_number_{number}_step_{int(total_steps / 1000)}k_agent_{agent_id}.pth"
        )
