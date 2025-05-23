import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter


# Initializes weights orthogonally (helps with stable training), and biases to zero. 
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


# Different agents have different observation dimensions and action dimensions, so we need to use 'agent_id' to distinguish them
class Actor(nn.Module):
    def __init__(self, agent_id, obs_dim_n, action_dim_n, hidden_dim, max_action, use_orthogonal_init):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(obs_dim_n[agent_id], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim_n[agent_id])
        if use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        a = self.max_action * torch.tanh(self.fc3(x))

        return a


# The critic network is shared by all agents, so the input dimension is the sum of the observation dimensions and action dimensions of all agents
class Critic(nn.Module):
    def __init__(self, obs_dim_n, action_dim_n, hidden_dim, use_orthogonal_init):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(sum(obs_dim_n) + sum(action_dim_n), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.fc4 = nn.Linear(sum(obs_dim_n) + sum(action_dim_n), hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        if use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)
            orthogonal_init(self.fc5)
            orthogonal_init(self.fc6)


    def forward(self, s, a):
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
        s = torch.cat(s, dim=1)
        a = torch.cat(a, dim=1)
        s_a = torch.cat([s, a], dim=1)
        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        return q1
    


"""
    Each agent holds:
    - Its own actor & target actor
    - Its own critic & target critic
    - agent_id to know its slice in obs_n, action_n etc.
"""
class MATD3(object):

    def __init__(self, agent_id, action_dim_n, max_action, lr_a, lr_c, gamma, 
                 tau, use_grad_clip, policy_noise, noise_clip, policy_update_freq,
                 obs_dim_n, hidden_dim, use_orthogonal_init):
        self.N = 2
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

        # Create an individual actor and critic for each agent according to the 'agent_id'
        self.actor = Actor(agent_id, obs_dim_n, action_dim_n, hidden_dim, max_action, use_orthogonal_init)
        self.critic = Critic(obs_dim_n, action_dim_n, hidden_dim, use_orthogonal_init)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

        self.writer = SummaryWriter(log_dir=f'runs/MATD3/agent_{agent_id}')
        self.iter_count = 0  # To track training steps


    # Each agent selects actions based on its own local observations (add noise for exploration)
    def choose_action(self, obs, noise_std):
        obs = torch.unsqueeze(torch.tensor(obs, dtype=torch.float), 0)
        a = self.actor(obs).data.numpy().flatten()
        a = (a + np.random.normal(0, noise_std, size=self.action_dim)).clip(-self.max_action, self.max_action)
        return a

    def train(self, replay_buffer, agent_n):
        self.actor_pointer += 1
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

        # Compute target_Q
        with torch.no_grad():  # target_Q has no gradient
            # Trick 1:target policy smoothing
            batch_a_next_n = []
            for i in range(self.N):
                batch_a_next = agent_n[i].actor_target(batch_obs_next_n[i])
                noise = (torch.randn_like(batch_a_next) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                batch_a_next = (batch_a_next + noise).clamp(-self.max_action, self.max_action)
                batch_a_next_n.append(batch_a_next)

            # Trick 2:clipped double Q-learning
            Q1_next, Q2_next = self.critic_target(batch_obs_next_n, batch_a_next_n)
            target_Q = batch_r_n[self.agent_id] + self.gamma * (1 - batch_done_n[self.agent_id]) * torch.min(Q1_next, Q2_next)  # shape:(batch_size,1)

        # Compute current_Q
        current_Q1, current_Q2 = self.critic(batch_obs_n, batch_a_n)  # shape:(batch_size,1)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        av_Q = torch.mean(target_Q).item()
        max_Q = torch.max(target_Q).item()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # Trick 3:delayed policy updates
        if self.actor_pointer % self.policy_update_freq == 0:
            print("Policy is getting updated")
            # Reselect the actions of the agent corresponding to 'agent_id', the actions of other agents remain unchanged
            batch_a_n[self.agent_id] = self.actor(batch_obs_n[self.agent_id])
            actor_loss = -self.critic.Q1(batch_obs_n, batch_a_n).mean()  # Only use Q1

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()

            # Softly update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # Logging scalar values to TensorBoard
            self.iter_count += 1
            self.writer.add_scalar("critic_loss", critic_loss.item(), self.iter_count)
            self.writer.add_scalar("actor_loss", actor_loss.item(), self.iter_count)
            self.writer.add_scalar("avg_Q", av_Q, self.iter_count)
            self.writer.add_scalar("max_Q", max_Q, self.iter_count)     

    def save_model(self, number, total_steps, agent_id):
        torch.save(self.actor.state_dict(), "./model/MATD3_actor_number_{}_step_{}k_agent_{}.pth".format(number, int(total_steps / 1000), agent_id))

