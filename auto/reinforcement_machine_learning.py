# What I want. I want to be able to feed the values for desired_v, dt, current v, last throttle percent, to replace the pid system to get the our desired_v in shortest steps... not sure how ima do this
# Also why i choose reinforcemnt learning is cause I heard it used in simulating data before for something u can't easily get the answer for. While this may be easier to get the answer for do to formulas I want to see how it does.
# reading: 
# https://www.geeksforgeeks.org/deep-learning/reinforcement-learning-using-pytorch/#reinforcement-learning-with-pytorch
# https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


import math
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pid_template import make_car, update
from torch import nn
from torch.distributions import Normal

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print("Device used: " + device)

class Env_For_Car():
	def __init__(self, max_v=100, max_dt=0.3, steps=550):
		self.currentCar = None
		self.max_v = max_v * 1.2
		self.max_dt = max_dt * 1.2
		self.steps = steps
		self.prev_reward = 0

	def create_car(self, desired_v, dt):
		self.currentCar = make_car(desired_v=desired_v, dt=dt)
		self.prev_reward = 0

	def get_state(self):
		return [(self.currentCar["desired_v"] / self.max_v), (self.currentCar["dt"] / self.max_dt), (self.currentCar["v"] / self.max_v), self.currentCar["last_throttle_perc"]]

	def step(self, throttle_perc):
		old_error = abs(self.currentCar["desired_v"] - self.currentCar["v"])
		update(self.currentCar, throttle_perc)
		done = False
		if (self.currentCar["step"] >= self.steps):
			done = True

		error = abs(self.currentCar["desired_v"] - self.currentCar["v"])
		reward = -error / self.max_v

		if error < 0.1 * self.max_v:
			reward += 0.05

		self.prev_reward = reward

		return(self.get_state(), reward, done)
	
		

class Policy_Gradient(nn.Module):
	# n_observations number on inputs, n_actions number of outputs
	def __init__(self, n_observations):
		super().__init__()
		# hidden layer is 128 for each of these which is 128 tensors holding the weight and baisas like from last tutorial (first goes len(n_observations) then to 128 tensor or whatever I have it set to at the time)
		self.layer1 = nn.Linear(n_observations, 128)
		self.layer2 = nn.Linear(128, 64)
		self.layer3 = nn.Linear(64, 1)

	# I still don't really get this... lookup later, it what is called when feeding the data to the model
	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		return torch.tanh(self.layer3(x))

class RL_Project():
	def __init__(self, seed=17):
		self.episode_rewards = []

		random.seed(seed)
		torch.manual_seed(seed)

		self.policy = Policy_Gradient(n_observations=4).to(device)
		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

	# so the model doesn't think the first -5 was horrible to get the a positive score. It taking over the last decisions the future score so lets say 5 mph it slowly going to 60 mph but like that a negative score or if overshooting but if necessary then it won't be as bad and the model think it did horribly just accelerating
	def compute_discounted_rewards(self, rewards, gamma=0.99):
		discounted_rewards = []
		R = 0
		for r in reversed(rewards):
			R = r + gamma * R
			discounted_rewards.insert(0, R)
		discounted_rewards = torch.tensor(discounted_rewards).to(device)
		discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
		return discounted_rewards

	def save_model(self, path="car_throttle_policy.pth"):
		torch.save(self.policy.state_dict(), path)
		print(f"Model saved to {path}")

	def load_model(self, path="car_throttle_policy.pth"):
		self.policy.load_state_dict(torch.load(path, map_location=device))
		self.policy.eval()   # important: put network in evaluation mode
		print(f"Model loaded from {path}")

	def get_desired_throttle_from_loaded_model(self, env_car: Env_For_Car):
		state = torch.FloatTensor(env_car.get_state()).unsqueeze(0).to(device)
		return self.policy(state).item()
	

	def train(self, env_car, updates=100, batch_size=16):
		for updatet in range(updates):
			batch_losses = []
			batch_total_rewards = []

			for _ in range(batch_size):
				desired_v = round(random.uniform(0, 100), 1)
				dt = round(random.uniform(0.1, 0.3), 2)
				env_car.create_car(desired_v=desired_v, dt=dt)

				log_probs, rewards = [], []
				done = False
				while not done:
					state = torch.FloatTensor(env_car.get_state()).unsqueeze(0).to(device)
					mean = self.policy(state)
					std = 0.5 * (1 - updatet / updates) + 0.05
					dist = Normal(loc=mean, scale=std)

					action = dist.sample()
					log_probs.append(dist.log_prob(action).view(-1))
					_, reward, done = env_car.step(torch.clamp(action, -1.0, 1.0).item())
					rewards.append(reward)

				returns = self.compute_discounted_rewards(rewards).to(device)
				log_probs = torch.cat(log_probs)
				batch_losses.append(-(log_probs * returns).sum())
				batch_total_rewards.append(sum(rewards))

			loss = torch.stack(batch_losses).mean()
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			if updatet % 5 == 0:
				print(f"Update {updatet}, avg reward: {sum(batch_total_rewards)/batch_size:.2f}")