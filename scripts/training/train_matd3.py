import torch
import numpy as np
import copy
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from matd3_env import GazeboEnv
from matd3_network import MATD3

class Runner:
    """
    Runner class manages the training and evaluation of multi-agent TD3 in the Gazebo environment.
    """

    def __init__(
        self,
        number,
        device,
        seed,
        episode_limit,
        max_train_steps,
        evaluate_freq,
        evaluate_times,
        max_action,
        lr_a,
        lr_c,
        gamma,
        tau,
        use_grad_clip,
        policy_noise,
        noise_clip,
        policy_update_freq,
        use_orthogonal_init,
        batch_size,
        buffer_size,
        noise_std_init,
        noise_std_min,
        noise_std_decay,
        use_noise_decay
    ):
        # Environment and agent setup
        self.N = 2  # Number of agents
        self.device = device
        self.number = number
        self.seed = seed
        self.batch_size = batch_size
        self.episode_limit = episode_limit
        self.max_train_steps = max_train_steps
        self.evaluate_freq = evaluate_freq
        self.evaluate_times = evaluate_times
        self.total_steps = 0

        # Exploration noise parameters
        self.noise_std = noise_std_init
        self.noise_std_min = noise_std_min
        self.noise_std_decay = noise_std_decay
        self.use_noise_decay = use_noise_decay

        # State and action dimensions
        self.environment_dim = 20
        self.robot_dim = 4
        self.state_dim = self.environment_dim + self.robot_dim
        self.obs_dim_n = [self.state_dim] * self.N
        self.action_dim_n = [2] * self.N

        # Logging and evaluation
        self.evaluate_rewards = [[] for _ in range(self.N)]  # Stores evaluation rewards for each agent
        self.writer = SummaryWriter(log_dir=f'runs/MATD3/MATD3_number_{self.number}_seed_{self.seed}')

        # Set random seeds for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Initialize environment and agents
        self.env = GazeboEnv(".../launch/two_robots.launch", self.environment_dim)
        self.agent_n = [
            MATD3(
                agent_id,
                self.action_dim_n,
                max_action,
                lr_a,
                lr_c,
                gamma,
                tau,
                use_grad_clip,
                policy_noise,
                noise_clip,
                policy_update_freq,
                self.obs_dim_n,
                use_orthogonal_init
            )
            for agent_id in range(self.N)
        ]

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(self.N, buffer_size, self.batch_size, self.obs_dim_n, self.action_dim_n)

    def run(self):
        """
        Main training loop. Handles episodes, agent training, evaluation, and logging.
        """
        self.evaluate_policy()  # Initial evaluation before training
        episode_no = 0

        while self.total_steps < self.max_train_steps:
            print(f"Episode {episode_no} started")
            print(f"Total steps: {self.total_steps}")
            obs_n = self.env.reset()  # Reset environment and get initial observations

            for t in range(self.episode_limit):
                print(f"Timestep: {t}")

                # Agents select actions based on their observations (with exploration noise)
                a_n = [
                    agent.choose_action(obs, noise_std=self.noise_std)
                    for agent, obs in zip(self.agent_n, obs_n)
                ]

                # Step environment with selected actions
                obs_next_n, r_n, done_n, _ = self.env.step(copy.deepcopy(a_n), False)

                # Store transition in replay buffer
                self.replay_buffer.store_transition(obs_n, a_n, r_n, obs_next_n, done_n)
                obs_n = obs_next_n
                self.total_steps += 1

                # Decay exploration noise if enabled
                if self.use_noise_decay:
                    self.noise_std = max(self.noise_std - self.noise_std_decay, self.noise_std_min)

                print(f"Replay buffer current size: {self.replay_buffer.current_size}")

                # Train agents if enough samples in buffer
                if self.replay_buffer.current_size > self.batch_size:
                    for agent_id, agent in enumerate(self.agent_n):
                        agent.train(self.replay_buffer, self.agent_n)
                        print(f"Agent {agent_id} is getting trained")

                # Evaluate policy at specified frequency
                if self.total_steps % self.evaluate_freq == 0:
                    self.evaluate_policy()

                # End episode if all agents are done
                if all(done_n):
                    break

            # Export episode rewards to CSV for analysis
            self.env.export_rewards_to_csv()
            episode_no += 1

        # Close environment after training
        self.env.close()

    def evaluate_policy(self):
        """
        Evaluate the current policy of the agents and log results.
        """
        print("Evaluating policy...")
        evaluate_rewards = [0.0 for _ in range(self.N)]  # Store total rewards for each agent

        for eval_idx in range(self.evaluate_times):
            print(f"Evaluation no: {eval_idx}")
            obs_n = self.env.reset()
            episode_rewards = [0.0 for _ in range(self.N)]

            for _ in range(self.episode_limit):
                # Agents select actions without exploration noise during evaluation
                a_n = [
                    agent.choose_action(obs, noise_std=0)
                    for agent, obs in zip(self.agent_n, obs_n)
                ]
                obs_next_n, r_n, done_n, _ = self.env.step(copy.deepcopy(a_n), True)

                # Accumulate rewards for each agent
                for agent_id in range(self.N):
                    episode_rewards[agent_id] += r_n[agent_id]

                obs_n = obs_next_n

                # End evaluation episode if all agents are done
                if all(done_n):
                    break

            # Add episode rewards to total evaluation rewards
            for agent_id in range(self.N):
                evaluate_rewards[agent_id] += episode_rewards[agent_id]

        # Average rewards across evaluation runs and log them
        for agent_id in range(self.N):
            avg_reward = evaluate_rewards[agent_id] / self.evaluate_times
            self.evaluate_rewards[agent_id].append(avg_reward)
            print(f"Agent {agent_id} - Avg Eval Reward: {avg_reward:.2f}")
            self.writer.add_scalar(
                f'evaluate_step_rewards_agent_{agent_id}',
                avg_reward,
                global_step=self.total_steps
            )

        # Save evaluation rewards to disk for later analysis
        for agent_id in range(self.N):
            np.save(f'./data_train/MATD3_agent{agent_id}.npy', np.array(self.evaluate_rewards[agent_id]))

        # Save model checkpoints for each agent
        for agent_id in range(self.N):
            self.agent_n[agent_id].save_model(self.number, self.total_steps, agent_id)

if __name__ == '__main__':

    # Hyperparameters for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_train_steps = int(5e6)           # Maximum number of training steps
    episode_limit = 500                  # Maximum number of steps per episode
    evaluate_freq = 5000                 # Evaluate the policy every 'evaluate_freq' steps
    evaluate_times = 10                  # Number of evaluation runs
    max_action = 1.0
    buffer_size = int(1e6)               # Capacity of the replay buffer
    batch_size = 1024                    # Batch size for training
    noise_std_init = 1                   # Initial std of Gaussian noise for exploration
    noise_std_min = 0.1                  # Minimum std of Gaussian noise for exploration
    noise_decay_steps = 500000           # Steps before noise_std decays to minimum
    use_noise_decay = True               # Whether to decay the noise_std
    lr_a = 5e-4                          # Learning rate for actor
    lr_c = 5e-4                          # Learning rate for critic
    gamma = 0.95                         # Discount factor
    tau = 0.005                          # Soft update rate for target network
    use_orthogonal_init = True           # Use orthogonal initialization
    use_grad_clip = True                 # Use gradient clipping
    policy_noise = 0.2                   # Target policy smoothing noise
    noise_clip = 0.5                     # Clip noise for target policy
    policy_update_freq = 2               # Frequency of policy updates
    number = 1                           # Experiment number
    seed = 0                             # Random seed

    # Calculate noise decay per step
    noise_std_decay = (noise_std_init - noise_std_min) / noise_decay_steps

    # Create runner and start training
    runner = Runner(
        number, device, seed, episode_limit, max_train_steps, evaluate_freq, evaluate_times, max_action, lr_a, lr_c,
        gamma, tau, use_grad_clip, policy_noise, noise_clip, policy_update_freq, use_orthogonal_init,
        batch_size, buffer_size, noise_std_init, noise_std_min, noise_std_decay, use_noise_decay
    )
    runner.run()
