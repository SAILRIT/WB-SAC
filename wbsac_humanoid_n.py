
# -*- coding: utf-8 -*-
# !pip install gymnasium
# !pip install gymnasium[mujoco]
# !pip install pot

import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
import gymnasium as gym
import warnings
import matplotlib.pyplot as plt
import argparse
import pickle

warnings.filterwarnings("ignore")

print("Is CUDA available?", T.cuda.is_available())
print("CUDA device count:", T.cuda.device_count())
print("CUDA device name:", T.cuda.get_device_name(0) if T.cuda.is_available() else "No GPU detected")
print("CUDA current device:", T.cuda.current_device() if T.cuda.is_available() else "No GPU detected")

device = T.device("cuda" if T.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#########################################
# Wasserstein Barycenter Computation
#########################################
def gaussian_wasserstein_barycenter(mu_actor, sigma_actor, mu_explorer, sigma_explorer, weights=(0.5, 0.5)):
    w_actor, w_explorer = weights
    mu_barycenter = w_actor * mu_actor + w_explorer * mu_explorer
    sigma_barycenter_squared = (w_actor * (sigma_actor**2 + mu_actor**2) +
                                w_explorer * (sigma_explorer**2 + mu_explorer**2)
                                - mu_barycenter**2)
    if not isinstance(sigma_barycenter_squared, T.Tensor):
        sigma_barycenter_squared = T.tensor(sigma_barycenter_squared, dtype=T.float32)
    sigma_barycenter_squared = T.clamp(sigma_barycenter_squared, min=1e-6)
    sigma_barycenter = T.sqrt(sigma_barycenter_squared)
    return mu_barycenter, sigma_barycenter

#########################################
# Replay Buffer and Network Definitions
#########################################
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, states_, dones

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims[0] + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, n_actions=2, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.sigma = nn.Linear(fc2_dims, n_actions)
        self.max_action = max_action
        self.reparam_noise = 1e-6
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        sigma = T.clamp(self.sigma(x), min=self.reparam_noise, max=1)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)
        actions = dist.rsample() if reparameterize else dist.sample()
        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        log_probs = dist.log_prob(actions)
        log_probs -= T.log(1 - T.tanh(actions).pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        return action, log_probs

class PolicyExploreNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, n_actions, fc1_dims=256, fc2_dims=256):
        super(PolicyExploreNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.sigma = nn.Linear(fc2_dims, n_actions)
        self.max_action = max_action
        self.reparam_noise = 1e-6
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        sigma = T.clamp(self.sigma(x), min=self.reparam_noise, max=1)
        return mu, sigma

    def sample_normal_explorer(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)
        actions = dist.rsample() if reparameterize else dist.sample()
        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        log_probs = dist.log_prob(actions)
        log_probs -= T.log(1 - T.tanh(actions).pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        return action, log_probs

#########################################
# Agent Definition
#########################################
class Agent:
    def __init__(self, alpha, beta, input_dims, env, gamma=0.99, tau=0.005,
                 batch_size=256, reward_scale=20):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.scale = reward_scale

        self.memory = ReplayBuffer(max_size=1_000_000, input_shape=input_dims, n_actions=env.action_space.shape[0])
        self.actor = ActorNetwork(alpha, input_dims, max_action=env.action_space.high,
                                    n_actions=env.action_space.shape[0])
        self.policy_explore = PolicyExploreNetwork(alpha, input_dims, max_action=env.action_space.high,
                                                   n_actions=env.action_space.shape[0])

        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=env.action_space.shape[0])
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=env.action_space.shape[0])

        # Initialize target critics
        self.critic_1_target = CriticNetwork(beta, input_dims, n_actions=env.action_space.shape[0])
        self.critic_2_target = CriticNetwork(beta, input_dims, n_actions=env.action_space.shape[0])
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

    def choose_action(self, observation, t, total_steps):
        state = T.tensor([observation], dtype=T.float32).to(self.actor.device)
        # Obtain parameters from both actor and explorer
        actor_mu, actor_sigma = self.actor.forward(state)
        explorer_mu, explorer_sigma = self.policy_explore.forward(state)
        # Determine weights that change over training (example: linear annealing)
        a = min(0.9, 0.1 + (10 * t / total_steps))
        b = 1 - a
        # Compute the Wasserstein barycenter
        barycenter_mu, barycenter_sigma = gaussian_wasserstein_barycenter(
            actor_mu.detach(), actor_sigma.detach(),
            explorer_mu.detach(), explorer_sigma.detach(),
            weights=(b, a)
        )
        # Ensure they are tensors on the proper device
        barycenter_mu = T.tensor(barycenter_mu, dtype=T.float32).to(self.actor.device)
        barycenter_sigma = T.tensor(barycenter_sigma, dtype=T.float32).to(self.actor.device)
        # Sample from the barycenter distribution
        barycenter_dist = Normal(barycenter_mu, barycenter_sigma)
        action = barycenter_dist.sample()
        action = T.tanh(action) * T.tensor(self.actor.max_action).to(self.actor.device)
        return action.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_target_networks(self):
        for target, source in [(self.critic_1_target, self.critic_1),
                               (self.critic_2_target, self.critic_2)]:
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        rewards = T.tensor(rewards, dtype=T.float32).to(self.actor.device)
        dones = T.tensor(dones, dtype=T.float32).to(self.actor.device)
        state = T.tensor(states, dtype=T.float32).to(self.actor.device)
        state_ = T.tensor(states_, dtype=T.float32).to(self.actor.device)
        action = T.tensor(actions, dtype=T.float32).to(self.actor.device)

        with T.no_grad():
            next_actions, next_log_probs = self.actor.sample_normal(state_, reparameterize=False)
            next_log_probs = next_log_probs.view(-1)
            q1_next = self.critic_1_target(state_, next_actions).view(-1)
            q2_next = self.critic_2_target(state_, next_actions).view(-1)
            min_q_next = T.min(q1_next, q2_next)
            q_hat = self.scale * rewards + self.gamma * (min_q_next - next_log_probs) * (1 - dones)

        # Update Critics
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q1_val = self.critic_1(state, action).view(-1)
        q2_val = self.critic_2(state, action).view(-1)
        critic_loss = 0.5 * ((q1_val - q_hat.detach())**2 + (q2_val - q_hat.detach())**2).mean()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # Update Explorer Policy with its own objective
        self.policy_explore.optimizer.zero_grad()
        actions_explor, _ = self.policy_explore.sample_normal_explorer(state, reparameterize=True)
        q1_values = self.critic_1(state, actions_explor).view(-1)
        q2_values = self.critic_2(state, actions_explor).view(-1)
        q_min = T.min(q1_values, q2_values)
        q_max = T.max(q1_values, q2_values)
        q_diff = T.abs(q_max - q_min)
        q_mean = (q1_values + q2_values) / 2
        objective = q_mean + 0.5 * q_diff
        policy_loss = -objective.mean()
        policy_loss.backward()
        self.policy_explore.optimizer.step()

        # Update Actor Network
        self.actor.optimizer.zero_grad()
        actions_new, log_probs_new = self.actor.sample_normal(state, reparameterize=True)
        log_probs_new = log_probs_new.view(-1)
        q1_new = self.critic_1(state, actions_new).view(-1)
        q2_new = self.critic_2(state, actions_new).view(-1)
        critic_value_new = T.min(q1_new, q2_new)
        actor_loss = (log_probs_new - critic_value_new).mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Soft update target networks
        self.update_target_networks()

#########################################
# Evaluation Function (Actor-Only)
#########################################
def evaluate_actor(agent, env_name='Humanoid-v5', seed=0, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.reset(seed=seed + 100)
    total_reward = 0.0
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        while not (done or truncated):
            state_t = T.tensor([state], dtype=T.float32).to(agent.actor.device)
            actor_mu, _ = agent.actor.forward(state_t)
            action = T.tanh(actor_mu) * T.tensor(agent.actor.max_action).to(agent.actor.device)
            action = action.cpu().detach().numpy()[0]
            state, reward, done, truncated, _ = eval_env.step(action)
            episode_reward += reward
        total_reward += episode_reward
    avg_reward = total_reward / eval_episodes
    print("---------------------------------------")
    print(f"Actor-only evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

#########################################
# Main Training Loop
#########################################
# Create directory for saving results
save_dir = 'results'
os.makedirs(save_dir, exist_ok=True)

# Hyperparameters and training configuration
total_steps = 3_000_000
max_steps_per_episode = 1000
eval_interval = 5000  # Evaluate every 5,000 steps
alpha = 0.0003
beta = 0.0003

parser = argparse.ArgumentParser(description="WB-SAC with actor-only evaluation")
parser.add_argument("-r", metavar="N", type=int, help="Index to pick from the rand_num")
args = parser.parse_args()

# Setting the random seed for reproducibility
rand_num = list(range(1, 1000))
print("Number of elements in the random seed list %d" % len(rand_num))
print("The index from random seed list : %d" % args.r)
r = rand_num[args.r] if args.r is not None else 1
print("Value picked: %d" % r)

np.random.seed(r)
T.manual_seed(r)

env_name = 'Humanoid-v5'
env = gym.make(env_name)
env.reset(seed=r)

agent = Agent(alpha=alpha,
              beta=beta,
              input_dims=(env.observation_space.shape[0],),
              env=env)

score_history = []        # To track barycenter returns per episode
actor_eval_history = []   # To track actor-only evaluation returns
steps = 0
episode = 0

while steps < total_steps:
    observation, _ = env.reset()
    done = False
    truncated = False
    score = 0.0
    episode_steps = 0

    while not (done or truncated) and episode_steps < max_steps_per_episode and steps < total_steps:
        # Action selected via barycenter of actor and explorer
        action = agent.choose_action(observation, t=steps, total_steps=total_steps)
        observation_, reward, done, truncated, _ = env.step(action)
        score += reward

        agent.remember(observation, action, reward, observation_, done or truncated)
        agent.learn()

        observation = observation_
        steps += 1
        episode_steps += 1

        # Evaluation check every eval_interval steps
        if steps % eval_interval == 0:
            print(f"Seed {r}, Step {steps}")
            actor_only_return = evaluate_actor(agent, env_name=env_name, seed=r, eval_episodes=10)
            actor_eval_history.append(actor_only_return)
            with open(os.path.join(save_dir, f'actor_eval_seed_{r}.pkl'), 'wb') as f:
                pickle.dump((steps, actor_eval_history), f)

    score_history.append(score)
    episode += 1

# Optionally, save final actor-only results
with open(os.path.join(save_dir, f'actor_eval_seed_{r}.pkl'), 'wb') as f:
    pickle.dump(actor_eval_history, f)
