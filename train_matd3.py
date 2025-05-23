import torch
import numpy as np
import copy
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from matd3_env import GazeboEnv
from matd3_network import MATD3




class Runner:
    def __init__(self, number, seed, episode_limit, max_train_steps, evaluate_freq, evaluate_times, max_action, lr_a, lr_c, 
                 gamma, tau, use_grad_clip, policy_noise, noise_clip, policy_update_freq, 
                 batch_size, buffer_size, noise_std_init, noise_std_min, noise_std_decay, use_noise_decay):

        self.number = number
        self.seed = seed
        self.batch_size = batch_size
        self.episode_limit = episode_limit
        self.max_train_steps = max_train_steps
        self.evaluate_freq = evaluate_freq
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        self.noise_std = noise_std_init  # Initialize noise_std
        self.noise_std_min = noise_std_min
        self.noise_std_decay = noise_std_decay
        self.use_noise_decay = use_noise_decay
        self.evaluate_times = evaluate_times

        #self.env = make_env(env_name, discrete=False)  # Continuous action space
        self.env = GazeboEnv("/home/sila/catkin_ws/src/sac_marl/launch/two_robots.launch", 20)

        self.N = 2 # The number of agents
        self.obs_dim_n = [self.env.observation_space[i].shape[0] for i in range(self.N)]  # obs dimensions of N agents
        self.action_dim_n = [self.env.action_space[i].shape[0] for i in range(self.N)]  # actions dimensions of N agents
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.action_dim_n))

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create N agents
        self.agent_n = [MATD3(agent_id, self.action_dim_n, max_action, lr_a, lr_c, gamma, tau, use_grad_clip, policy_noise, 
                              noise_clip, policy_update_freq, self.obs_dim_n, hidden_dim, use_orthogonal_init) for agent_id in range(self.N)]

        self.replay_buffer = ReplayBuffer(self.N, buffer_size, self.batch_size, self.obs_dim_n, self.action_dim_n)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='runs/MATD3/MATD3_number_{}_seed_{}'.format(self.number, self.seed))


    def run(self, ):

        self.evaluate_policy()
        episode_no = 0
        # Training loop
        while self.total_steps < self.max_train_steps:

            print("Episode {} started".format(episode_no))
            print("Total steps: {}".format(self.total_steps))
            obs_n = self.env.reset()

            for i in range(self.episode_limit):

                print("Timestep: {}".format(i))
                # Each agent selects actions based on its own local observations(add noise for exploration)
                a_n = [agent.choose_action(obs, noise_std=self.noise_std) for agent, obs in zip(self.agent_n, obs_n)]
                
                obs_next_n, r_n, done_n, _ = self.env.step(copy.deepcopy(a_n))

                # Store the transition
                self.replay_buffer.store_transition(obs_n, a_n, r_n, obs_next_n, done_n)
                obs_n = obs_next_n
                self.total_steps += 1

                # Decay noise_std
                if self.use_noise_decay:
                    self.noise_std = self.noise_std - self.noise_std_decay if self.noise_std - self.noise_std_decay > self.noise_std_min else self.noise_std_min

                print("Replay buffer current size: {}".format(self.replay_buffer.current_size))
                if self.replay_buffer.current_size > self.batch_size:
                    # Train each agent individually
                    for agent_id in range(self.N):
                        self.agent_n[agent_id].train(self.replay_buffer, self.agent_n)
                        print("Agent {} is getting trained".format(agent_id))

                if self.total_steps % self.evaluate_freq == 0:
                    self.evaluate_policy()

                if all(done_n):
                    break

            episode_no += 1
        self.env.close()
        #self.env_evaluate.close()

    def evaluate_policy(self, ):

        print("Evaluating policy...")
        evaluate_reward = 0

        for i in range(self.evaluate_times):
            print("Evaluation no: {}".format(i))
            obs_n = self.env.reset()
            episode_reward = 0

            for _ in range(self.episode_limit):
                a_n = [agent.choose_action(obs, noise_std=0) for agent, obs in zip(self.agent_n, obs_n)]  # We do not add noise when evaluating
                obs_next_n, r_n, done_n, _ = self.env.step(copy.deepcopy(a_n))
                episode_reward += r_n[0]
                obs_n = obs_next_n
                if all(done_n):
                    break
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)

        print("total_steps:{} \t evaluate_reward:{} \t noise_std:{}".format(self.total_steps, evaluate_reward, self.noise_std))

        self.writer.add_scalar('evaluate_step_rewards_MATD3', evaluate_reward, global_step=self.total_steps)

        # Save the rewards and models
        np.save('./data_train/MATD3_number_{}_seed_{}.npy'.format(self.number, self.seed), np.array(self.evaluate_rewards))

        for agent_id in range(self.N):
            self.agent_n[agent_id].save_model(self.number, self.total_steps, agent_id)





if __name__ == '__main__':
    
    # Hyperparameters
    max_train_steps = int(3e6) # Maximum number of training steps
    episode_limit = 300 # Maximum number of steps per episode /25
    evaluate_freq = 5000 # Evaluate the policy every 'evaluate_freq' steps
    evaluate_times = 10 # Evaluate times /3
    max_action = 1.0
    buffer_size = int(1e6) # The capacity of the replay buffer
    batch_size = 1024 # Batch size
    hidden_dim=64 # The number of neurons in hidden layers of the neural network
    noise_std_init=1 # The std of Gaussian noise for exploration /0.2
    noise_std_min=0.1 # The std of Gaussian noise for exploration /0.05
    noise_decay_steps = int(3e6) #How many steps before the noise_std decays to the minimum 3e5
    use_noise_decay = True # Whether to decay the noise_std
    lr_a = 5e-4 # Learning rate of actor
    lr_c = 5e-4 # Learning rate of critic
    gamma = 0.95 # Discount factor
    tau = 0.01 # Softly update the target network
    use_orthogonal_init = True # Orthogonal initialization
    use_grad_clip = True #Gradient clip
    # --------------------------------------MATD3--------------------------------------------------------------------
    policy_noise=0.2 # Target policy smoothing
    noise_clip=0.5 # Clip noise
    policy_update_freq=2 # The frequency of policy updates

    number=1
    seed=0
    
    noise_std_decay = (noise_std_init - noise_std_min) / noise_decay_steps

    runner = Runner(number, seed, episode_limit, max_train_steps, evaluate_freq, evaluate_times, max_action, lr_a, lr_c, 
                 gamma, tau, use_grad_clip, policy_noise, noise_clip, policy_update_freq, 
                 batch_size, buffer_size, noise_std_init, noise_std_min, noise_std_decay, use_noise_decay)
    runner.run()