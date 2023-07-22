import gymnasium as gym
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm, trange

device = 'cpu' 

def make_env(env_id, seed, idx, run_name): 
	def thunk(): 
		env = gym.make(env_id)
		env.action_space.seed(seed) 
		env.observation_space.seed(seed) 
		return env 
	return thunk() 

hidden_dim = 1024

class state_dependent_noise_distribution(): 
	def __init__(self, action_dim, full_std=True, use_expln=False, learn_features=False, epsilon=1e-6): 
		self.action_dim = action_dim 
		self.latent_sde_dim = None 
		self.mean_actions = None 
		self.log_std = None 
		self.weights_dist = None 
		self.exploration_mat = None 
		self.exploration_matrices = None 
		self._latent_sde = None 
		self.use_expln = use_expln
		self.full_std = full_std
		self.epsilon = epsilon
		self.learn_features = learn_features
		self.bijector = None # because squash_output = False

	def get_std(self, log_std): 
		if self.use_expln: 
			below_threshold = torch.exp(log_std) * (log_std <= 0) 
			safe_log_std = log_std * (log_std > 0) + self.epsilon 
			above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0) 
			std = below_threshold + above_threshold 
		else: 
			std = torch.exp(log_std) 

		if self.full_std: 
			return std 
		return torch.ones(self.latent_sde_dim, self.action_dim).to(log_std.device) * std

	def sample_weights(self, log_std, batch_size=1): 
		std = self.get_std(log_std)
		self.weights_dist = torch.distributions.normal.Normal(torch.zeros_like(std), std)
		# reparametrization trick 
		self.exploration_mat = self.weights_dist.rsample() 
		self.exploration_matrices = self.weights_dist.rsample((batch_size,))

	def proba_distribution_net(self, latent_dim, log_std_init=-2.0, latent_sde_dim=None): 
		mean_actions_net = nn.Linear(latent_dim, self.action_dim)
		self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim 
		log_std = torch.ones(self.latent_sde_dim, self.action_dim) if self.full_std else torch.ones(self.latent_sde_dim, 1) 
		log_std = nn.Parameter(log_std * log_std_init, requires_grad=True) 
		self.sample_weights(log_std) 
		return mean_actions_net, log_std 
	
	def proba_distribution(self, mean_actions, log_std, latent_sde): 
		self._latent_sde = latent_sde if self.learn_features else latent_sde.detach() 
		# print(self._latent_sde.shape, self.get_std(log_std).shape)
		variance = torch.mm(self._latent_sde**2, self.get_std(log_std)**2) 
		self.distribution = torch.distributions.normal.Normal(mean_actions, torch.sqrt(variance + self.epsilon)) 
		return self 
	
	def log_prob(self, actions): 
		if self.bijector is None: 
			gaussian_actions = actions 

		log_prob = self.distribution.log_prob(gaussian_actions) 
		
		if len(log_prob.shape) > 1: 
			log_prob = log_prob.sum(dim=1) 
		else: 
			log_prob = log_prob.sum() 
		return log_prob

	def entropy(self): 
		if len(self.distribution.entropy()) > 1: 
			return self.distribution.entropy().sum(dim=1) 
		else: 
			return self.distribution.entropy().sum()
	
	def sample(self): 
		noise = self.get_noise(self._latent_sde) 
		actions = self.distribution.mean + noise 
		return actions 

	def mode(self): 
		return self.distribution.mean 
	
	def get_noise(self, latent_sde): 
		latent_sde = latent_sde if self.learn_features else latent_sde.detach() 

		if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices): 
			return torch.mm(latent_sde, self.exploration_mat) 
		
		latent_sde = latent_sde.unsqueeze(1) 
		noise = torch.bmm(latent_sde, self.exploration_matrices) 
		return noise.squeeze(dim=1) 
	
	def actions_from_params(self, mean_actions, log_std, latent_sde, deterministic=False): 
		self.proba_distribution(mean_actions, log_std, latent_sde)
		if deterministic: 
			return self.mode()
		return self.sample() 
	
	def log_prob_from_params(self, mean_actions, log_std, latent_sde): 
		actions = self.actions_from_params(mean_actions, log_std, latent_sde)
		log_prob = self.log_prob(actions) 
		return actions, log_prob 

def layer_init(layer, std=np.sqrt(2), bias_const=0.0): 
	nn.init.orthogonal_(layer.weight, std) 
	nn.init.constant_(layer.bias, bias_const)
	return layer 

class Agent(nn.Module): 
	def __init__(self, envs): 
		super().__init__() 
		self.state_dist = state_dependent_noise_distribution(action_dim=np.array(envs.single_action_space.shape).prod())

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
			# layer_init(nn.Linear(hidden_dim, np.array(envs.single_action_space.shape).prod()), std=0.01)
		)
		# self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

		self.actor_mean, self.actor_logstd = self.state_dist.proba_distribution_net(latent_dim=hidden_dim)

	def get_value(self, x): 
		return self.critic(x) 
	
	def get_action_and_value(self, x, pre_action=None): 
		action = self.actor(x) 
		action_mean = self.actor_mean(action) 
		actions, log_prob = self.state_dist.log_prob_from_params(mean_actions=action_mean, log_std=self.actor_logstd, latent_sde=action)
		entropy = self.state_dist.entropy() 
		if pre_action is None: 
			return actions, log_prob, entropy, self.critic(x) 
		else: 
			log_prob = self.state_dist.log_prob(pre_action)
			return pre_action, log_prob, entropy, self.critic(x) 

envs = gym.vector.SyncVectorEnv([
    lambda: make_env('Pusher-v4', 0+i, i, 'the first') for i in range(4)
])

agent = Agent(envs).to(device) 
print(agent.load_state_dict(torch.load('models/pusher_ppo_state/best_performing.pth'))) 

env = gym.make('Pusher-v4', render_mode='human')
# env = gym.wrappers.ClipAction(env) 
# env = gym.wrappers.NormalizeObservation(env) 
# env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
# env = gym.wrappers.NormalizeReward(env, gamma=0.99)
# env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
state, info = env.reset() 

def get_action(x): 
	with torch.no_grad(): 
		action = agent.actor(x) 
		action_mean = agent.actor_mean(action) 
		actions = agent.state_dist.actions_from_params(mean_actions=action_mean, log_std=agent.actor_logstd, latent_sde=action, deterministic=True)
	return actions

loop = tqdm()
# while True: 
for _ in range(512): 
	action = get_action(torch.from_numpy(state).float().to(device).unsqueeze(0)) 
	state, reward, terminated, truncated, _ = env.step(action[0].cpu().numpy())
	
	loop.set_postfix(reward=reward)
	if terminated or truncated: 
		state, info = env.reset()
