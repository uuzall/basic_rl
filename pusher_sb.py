import gymnasium as gym
from collections import namedtuple, deque
import stable_baselines3 as sb 

import torch

from tqdm import tqdm, trange

device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('Pusher-v4', render_mode='human')
model = sb.PPO.load('models/pusher_ppo/sb_10000', env=env, device=device)
print(model)

vec_env = model.get_env() 
obs = vec_env.reset() 

# for i in range(256): 
while True: 
	action, _states = model.predict(obs, deterministic=True)
	obs, reward, done, info = vec_env.step(action) 
	# vec_env.render() 
	print(_states)
	break

env.close()