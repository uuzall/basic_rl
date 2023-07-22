import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm, trange
import numpy as np 

def make_env(env_id, seed, idx, run_name): 
	def thunk(): 
		env = gym.make(env_id, render_mode='rgb_array')
		env = gym.wrappers.RecordEpisodeStatistics(env) 
		env = gym.wrappers.RecordVideo(env, f'videos/ppo/cartpole')
		env.action_space.seed(seed) 
		env.observation_space.seed(seed) 
		return env 
	return thunk() 

device = 'cpu'

def layer_init(layer, std=np.sqrt(2), bias_const=0.0): 
	nn.init.orthogonal_(layer.weight, std) 
	nn.init.constant_(layer.bias, bias_const)
	return layer 

class Agent(nn.Module): 
	def __init__(self, envs): 
		super().__init__() 
		self.critic = nn.Sequential(
			layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)), 
			nn.Tanh(), 
			layer_init(nn.Linear(64, 64)), 
			nn.Tanh(), 
			layer_init(nn.Linear(64, 1), std=1.0)
		)

		self.actor = nn.Sequential(
			layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)), 
			nn.Tanh(), 
			layer_init(nn.Linear(64, 64)), 
			nn.Tanh(), 
			layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
		)

	def get_value(self, x): 
		return self.critic(x) 
	
	def get_action_and_value(self, x, action=None): 
		logits = self.actor(x) 
		probs = torch.distributions.categorical.Categorical(logits=logits) 
		if action is None: 
			action = probs.sample() 
		return action, probs.log_prob(action), probs.entropy(), self.critic(x) 
	
seed = 1337 
num_envs = 4

envs = gym.vector.SyncVectorEnv([
    lambda: make_env('CartPole-v1', seed+0, 0, 'the first'), 
    lambda: make_env('CartPole-v1', seed+1, 1, 'the first'), 
    lambda: make_env('CartPole-v1', seed+2, 2, 'the first'), 
    lambda: make_env('CartPole-v1', seed+3, 3, 'the first')
])

agent = Agent(envs).to(device) 
print(agent.load_state_dict(torch.load('models/cartpole_ppo/4096_train.pth')))

env = gym.make('CartPole-v1', render_mode='human') 

state, info = env.reset() 
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) 

while True: 
	with torch.no_grad(): 
		action = agent.actor(state).argmax(keepdims=True)
		# action = env.action_space.sample()

	observation, _, terminated, truncated, _  = env.step(action.item()) 
	state = torch.tensor(observation, dtype=torch.float32, device=device) 

	if terminated or truncated: 
		state, info = env.reset() 
		state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) 