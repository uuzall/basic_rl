import gymnasium as gym
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import time 

from tqdm import tqdm, trange

device = 'cpu' 

def make_env(env_id, seed, idx, run_name): 
	def thunk(): 
		env = gym.make(env_id)
		env.action_space.seed(seed) 
		env.observation_space.seed(seed) 
		return env 
	return thunk() 

class diag_gaussian_distribution(): 
	def __init__(self, action_dim): 
		super().__init__() 
		self.action_dim = action_dim 
		self.mean_actions = None 
		self.log_std = None 
		
	def proba_distribution_net(self, latent_dim, log_std_init=0.0): 
		mean_actions = nn.Linear(latent_dim, self.action_dim) 
		log_std = nn.Parameter(torch.ones(self.action_dim)*log_std_init, requires_grad=True) 
		return mean_actions, log_std 
	
	def proba_distribution(self, mean_actions, log_std): 
		action_std = torch.ones_like(mean_actions) * log_std.exp() 
		self.distribution = torch.distributions.normal.Normal(mean_actions, action_std)
		return self 
	
	def log_prob(self, actions): 
		log_prob = self.distribution.log_prob(actions) 
		if len(log_prob) > 1: 
			return log_prob.sum(dim=1) 
		else: 
			return log_prob.sum() 
		
	def entropy(self): 
		entropy = self.distribution.entropy() 
		if len(entropy) > 1: 
			return entropy.sum(dim=1) 
		else: 
			return entropy.sum() 
		
	def sample(self): 
		return self.distribution.rsample() 
	
	def mode(self): 
		return self.distribution.mean 
	
	def actions_from_params(self, mean_actions, log_std, deterministic=False): 
		self.proba_distribution(mean_actions, log_std) 
		if deterministic: 
			return self.mode() 
		return self.sample() 
	
	def log_prob_from_params(self, mean_actions, log_std): 
		actions = self.actions_from_params(mean_actions, log_std) 
		log_prob = self.log_prob(actions) 
		return actions, log_prob 

hidden_dim = 64

def layer_init(layer, std=np.sqrt(2), bias_const=0.0): 
	nn.init.orthogonal_(layer.weight, std) 
	nn.init.constant_(layer.bias, bias_const)
	return layer 

class Agent(nn.Module): 
	def __init__(self, envs): 
		super().__init__() 
		self.state_dist = diag_gaussian_distribution(action_dim=np.array(envs.single_action_space.shape).prod())

		self.critic = nn.Sequential(
			layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_dim)), 
			nn.Tanh(), 
			layer_init(nn.Linear(hidden_dim, hidden_dim)), 
			nn.Tanh(), 
			layer_init(nn.Linear(hidden_dim, 1), std=1.0)
		)

		self.actor = nn.Sequential(
			layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_dim)), 
			nn.Tanh(), 
			layer_init(nn.Linear(hidden_dim, hidden_dim)), 
			nn.Tanh(), 
		)

		self.actor_mean, self.actor_logstd = self.state_dist.proba_distribution_net(latent_dim=hidden_dim)

	def get_value(self, x): 
		return self.critic(x) 
	
	def get_action_and_value(self, x, pre_action=None): 
		action = self.actor(x) 
		action_mean = self.actor_mean(action) 
		actions, log_prob = self.state_dist.log_prob_from_params(mean_actions=action_mean, log_std=self.actor_logstd)
		entropy = self.state_dist.entropy() 
		if pre_action is None: 
			return actions, log_prob, entropy, self.critic(x) 
		else: 
			log_prob = self.state_dist.log_prob(pre_action)
			return pre_action, log_prob, entropy, self.critic(x) 

envs = gym.vector.SyncVectorEnv([
    lambda: make_env('Pusher-v4', 0+i, i, 'the first') for i in range(4)
])

def get_action(x): 
	with torch.no_grad(): 
		action = agent.actor(x) 
		action_mean = agent.actor_mean(action) 
		actions = agent.state_dist.actions_from_params(mean_actions=action_mean, log_std=agent.actor_logstd, deterministic=True)
	return actions

list_models = os.listdir('models/pusher_ppo_diag/')

agent = Agent(envs).to(device) 
# print() 
# print(agent.load_state_dict(torch.load('models/pusher_ppo_diag/64.pth'))) 

env = gym.make('Pusher-v4', render_mode='human')
state, info = env.reset() 

loop = tqdm()

for model_name in range(64, 1920, 64): 
	agent.load_state_dict(torch.load(f'models/pusher_ppo_diag/{model_name}.pth'))
	while True: 
		action = get_action(torch.from_numpy(state).float().to(device).unsqueeze(0)) 
		time.sleep(0.001)
		state, reward, terminated, truncated, _ = env.step(action[0].cpu().numpy())
		loop.set_postfix(model=model_name)
		if terminated or truncated: 
			state, info = env.reset()
			break