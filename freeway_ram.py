import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from tqdm import tqdm, trange

device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

env = gym.make('ALE/Freeway-ram-v5', render_mode='human')
RAM_mask = np.asarray([14, 16, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117])

class dqn(nn.Module): 
	def __init__(self, n_action=env.action_space.n): 
		super().__init__() 

		self.net = nn.Sequential(
			nn.Linear(12, 8), nn.ReLU(), 
			nn.Linear(8, 4), nn.ReLU(), 
			nn.Linear(4, n_action)
		)
	
	def forward(self, x): 
		out = self.net(x/255.) 
		return out 
	
def layer_init(layer, std=np.sqrt(2), bias_const=0.0): 
	nn.init.orthogonal_(layer.weight, std) 
	nn.init.constant_(layer.bias, bias_const)
	return layer 

class Agent(nn.Module): 
	def __init__(self): 
		super().__init__() 
		self.critic = nn.Sequential(
			layer_init(nn.Linear(RAM_mask.size, 64)), #np.array(envs.single_observation_space.shape).prod(), 64)), 
			nn.Tanh(), 
			layer_init(nn.Linear(64, 64)), 
			nn.Tanh(), 
			layer_init(nn.Linear(64, 1), std=1.0)
		)

		self.actor = nn.Sequential(
			layer_init(nn.Linear(RAM_mask.size, 64)), 
			nn.Tanh(), 
			layer_init(nn.Linear(64, 64)), 
			nn.Tanh(), 
			layer_init(nn.Linear(64, 3), std=0.01)
		)

	def get_value(self, x): 
		return self.critic(x) 
	
	def get_action_and_value(self, x, action=None): 
		logits = self.actor(x) 
		probs = torch.distributions.categorical.Categorical(logits=logits) 
		if action is None: 
			action = probs.sample() 
		return action, probs.log_prob(action), probs.entropy(), self.critic(x) 

policy_net = Agent().to(device) 
# print(policy_net.load_state_dict(torch.load('models/freeway/449_2047_highest_score.pth'))) 
print(policy_net.load_state_dict(torch.load('models/freeway_ppo/minus_100.pth'))) 

state, info = env.reset() 
state = torch.tensor(state[RAM_mask], dtype=torch.float32, device=device).unsqueeze(0) 

while True: 
	with torch.no_grad(): 
		action = policy_net.actor(state).argmax(keepdims=True)
	observation, _, terminated, truncated, _ = env.step(action.item())
	state = torch.tensor(observation[RAM_mask], dtype=torch.float32, device=device).unsqueeze(0) 

	if terminated or truncated: 
		state, info = env.reset() 
		state = torch.tensor(state[RAM_mask], dtype=torch.float32, device=device).unsqueeze(0)
		break 
