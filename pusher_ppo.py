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

def layer_init(layer, std=np.sqrt(2), bias_const=0.0): 
	nn.init.orthogonal_(layer.weight, std) 
	nn.init.constant_(layer.bias, bias_const)
	return layer 

class Agent(nn.Module): 
	def __init__(self, envs): 
		super().__init__() 

		self.critic = nn.Sequential(
			layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_dim)), 
			nn.Tanh(), 
			layer_init(nn.Linear(hidden_dim, hidden_dim)), 
			nn.Tanh(), 
			layer_init(nn.Linear(hidden_dim, 1), std=1.0)
		)

		self.actor_mean = nn.Sequential(
			layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_dim)), 
			nn.Tanh(), 
			layer_init(nn.Linear(hidden_dim, hidden_dim)), 
			nn.Tanh(), 
			layer_init(nn.Linear(hidden_dim, np.array(envs.single_action_space.shape).prod()), std=0.01)
		)
		self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

	def get_value(self, x): 
		return self.critic(x) 
	
	def get_action_and_value(self, x, pre_action=None): 
		pass 

envs = gym.vector.SyncVectorEnv([
    lambda: make_env('Pusher-v4', 0+i, i, 'the first') for i in range(4)
])

agent = Agent(envs).to(device) 
# print(agent.load_state_dict(torch.load('models/pusher_ppo/final.pth'))) 
agent.load_state_dict(torch.load('models/pusher_ppo/1408.pth'))

env = gym.make('Pusher-v4', render_mode='human')
state, info = env.reset() 

loop = tqdm()
while True: 
# for _ in range(512): 
	with torch.no_grad(): 
		action = agent.actor_mean(torch.from_numpy(state).float().unsqueeze(0).to(device))
		# action = agent.actor_mean(action)
		# action, _, _, _ = agent.get_action_and_value(torch.from_numpy(state).float().to(device))
	state, reward, terminated, truncated, _ = env.step(action[0].cpu().numpy())
	
	loop.set_postfix(reward=reward)
	if terminated or truncated: 
		state, info = env.reset()
