# Implementing Reinforcement Learning Algorithms from Scratch

This is a repo that implements RL algorithms from scratch. It was meant for learning and not teaching. There are no comments, and the code might not make a lot of sense but you might learn something if you want to implement RL algorithms yourself. 

One basic thing that applies to all the files in this repo is that the Notebooks train the models and the .py files run the models in 'human' render_mode. \
Let's go through all the things I did in this repo in chronological order: \
* Lunar Lander using a basic Deep Q-Learning algorithm.
* Freeway using Deep Q-Learning. At first, I used CNNs and the RGB rendering to train the model but it did not work. Then I used the RAM outputs of the game to train the model and it kinda works (if you call pressing the forward button and never leaving it working).
* CartPole using the Proximal Policy Optimization (PPO). 
* Pusher using PPO. This was my final challenge. It was very hard to make it work. 
  * At first I tried using the vanilla PPO that I learned, and it did not work. 
  * Then I used "stable_baselines3" to train a PPO algorithm to make sure that it will actually work if I went through this path (it did). This is saved as "pusher_stablebaselines.ipynb" and "pusher_sb.py". 
  * Then I went through the source code for stable baselines and tried to retrace where I went wrong. I found out that they used special probability distributions to give basic probabilities of the actions more depth and more information than the basic Linear layer that I was using to model the probability. My probabilies had less information hence it did not work. The default distribution stable baselines used was Diagonal Gaussian Distribution. This is saved as "pusher_ppo_gaussian.ipynb". 
  * Then I implemented the generalized State Dependent Exploration (gSDE) from this paper (https://arxiv.org/abs/2005.05719) as mentioned in the source code as a substitute for the Gaussian Distribution. It still did not work. This is saved as "pusher_ppo.ipynb".

This concludes my learning path for Basic Reinforcement Learning. 