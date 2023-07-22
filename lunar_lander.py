import gymnasium as gym 
from tqdm import trange
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class dqn(nn.Module): 
	def __init__(self, n_observations, n_actions): 
		super().__init__() 
		
		self.layer1 = nn.Linear(n_observations, 128) 
		self.layer2 = nn.Linear(128, 128) 
		self.layer3 = nn.Linear(128, n_actions) 

	def forward(self, x): 
		x = F.relu(self.layer1(x)) 
		x = F.relu(self.layer2(x)) 
		return self.layer3(x) 

env = gym.make('LunarLander-v2', render_mode='human')

policy_net = dqn(n_observations=len(env.reset()[0]), n_actions=env.action_space.n)

print(policy_net.load_state_dict(torch.load('models/policy_net_lunar_lander')))

observation, info = env.reset()

while True: 
	observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
	action = policy_net(observation).argmax(keepdim=True)
	observation, _, terminated, truncated, _ = env.step(action.item())
	if terminated or truncated: 
		break 
env.close() 




# for _ in (loop := trange(10000)):
# 	action = env.action_space.sample()  # agent policy that uses the observation and info
# 	observation, reward, terminated, truncated, info = env.step(action)
# 	loop.set_postfix(reward=reward)

# 	if terminated or truncated:
# 		observation, info = env.reset()

# env.close() 