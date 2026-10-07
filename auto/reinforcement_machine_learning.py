# What I want. I want to be able to feed the values for desired_v, dt, current v, last throttle percent, to replace the pid system to get the our desired_v in shortest steps... not sure how ima do this
# Also why i choose reinforcemnt learning is cause I heard it used in simulating data before for something u can't easily get the answer for. While this may be easier to get the answer for do to formulas I want to see how it does.
# reading: 
# https://www.geeksforgeeks.org/deep-learning/reinforcement-learning-using-pytorch/#reinforcement-learning-with-pytorch
# https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pid_template import make_car, update
from torch import nn
from torch.distributions import Normal

# device = "cuda" if torch.cuda.is_available() else "cpu"
# settings cpu cause it actually taking longer to load to gpu then it worth since such a small model
device = "cpu"
print("Device used: " + device)

class Env_For_Car():
	def __init__(self, max_v:float=100, max_dt:float=0.3, steps:int=250):
		self.currentCar = None
		self.max_v = max_v * 1.2
		self.max_dt = max_dt * 1.2
		self.steps = steps
		self.prev_reward = 0

	def create_car(self, desired_v:float, dt:float):
		self.currentCar = make_car(desired_v=desired_v, dt=dt)
		self.prev_reward = 0

	def get_state(self):
		error = self.currentCar["desired_v"] - self.currentCar["v"]
		return [(self.currentCar["desired_v"] / self.max_v), (error / self.max_v), (max(-1.0, min(1.0, error / 5.0))), (self.currentCar["dt"] / self.max_dt), (self.currentCar["v"] / self.max_v), self.currentCar["last_throttle_perc"]]

	def step(self, throttle_perc:float):
		# old_error = abs(self.currentCar["desired_v"] - self.currentCar["v"])
		update(self.currentCar, throttle_perc)
		done = False
		if (self.currentCar["step"] >= self.steps):
			done = True

		error = abs(self.currentCar["desired_v"] - self.currentCar["v"])
		reward = -error / self.max_v

		#try changing these values as it gets closer like load model and have these more strict/better reward
		reward += 0.1 * (0.2 / max(error, 0.1))
		if(error > 0.1):
			reward -= 0.02 * (self.currentCar["step"] * self.currentCar["dt"])

		self.prev_reward = reward

		return(self.get_state(), reward, done)
	
		

class Policy_Gradient(nn.Module):
	# n_observations number on inputs, n_actions number of outputs
	def __init__(self, n_observations:int):
		super().__init__()
		# hidden layer is 128 for each of these which is 128 tensors holding the weight and baisas like from last tutorial (first goes len(n_observations) then to 128 tensor or whatever I have it set to at the time)
		self.layer1 = nn.Linear(n_observations, 32)
		self.layer2 = nn.Linear(32, 32)
		self.layer3 = nn.Linear(32, 1)

	# when calling the model to make a preidction it send the data through layer 1 layer 2... then at the end its a sigle value and tanh converts it to -1,1
	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		return torch.tanh(self.layer3(x))

class RL_Project():
	def __init__(self, seed:int=17):
		self.episode_rewards = []

		random.seed(seed)
		torch.manual_seed(seed)

		self.policy = Policy_Gradient(n_observations=6).to(device) #set the model and gradient descent
		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3) # optimizer for the loss
		self.average_reward = []
		self.average_reward_times = []

	# so the model doesn't think the first -5 was horrible to get the a positive score. 
	# It taking over the last decisions the future score so lets say 5 mph it slowly going to 60 mph but like that a negative score or 
	# if overshooting but if necessary then it won't be as bad and the model think it did horribly just accelerating
	# this is also based on the gamma number, a higher gamma number means it less greedy about what it just did and will have less pressure about making the wrong mistake in one move
	# so lets say a value of .9 gamma every time it does poorly it would freak out and make a lot of changes to model even tho that inevitable to get the desired velocity and good rewards
	def compute_discounted_rewards(self, rewards:float, gamma:float=0.99):
		discounted_rewards = []
		report_card = 0 # its called report card cause it scoring the model lol
		for r in reversed(rewards): #reverses it
			report_card = r + gamma * report_card
			discounted_rewards.insert(0, report_card) # have to put to front since we just reversed it
		discounted_rewards = torch.tensor(discounted_rewards).to(device) # converts the rewards to tensors so model can read
		return discounted_rewards

	def save_model(self, path:str="car_throttle_policy.pth"):
		# saves the model to the path provided
		torch.save(self.policy.state_dict(), path)
		print(f"Model saved to {path}")

	def load_model(self, mode:str, path:str="car_throttle_policy.pth"):
		# loads the model from the provided path as either train or eval based on what is input for mode
		self.policy.load_state_dict(torch.load(path, map_location=device))
		if(mode.lower() == "train"):
			self.policy.train()
		elif(mode.lower() == "eval"):
			self.policy.eval()
		else:
			print("load_model mode must be `train` or `eval` passed in as a str")
			return
		print(f"Model loaded from {path} as {mode} mode")

	def get_desired_throttle_from_loaded_model(self, env_car:Env_For_Car):
		# gets the output based on the current car env states
		state = torch.FloatTensor(env_car.get_state()).unsqueeze(0).to(device)
		return self.policy(state).item()
	

	def train(self, env_car:Env_For_Car, show_training_graphs:bool, updates:int=100, batch_size:int=16, gamma:float=0.99):
		# loops through each update
		for updatet in range(updates):
			all_log_probs = []
			all_returns = []
			batch_total_rewards = []
			
			for _ in range(batch_size):
				#make a random desired velocity and desired dt then creates the car enviroment with
				desired_v = round(random.uniform(0, 100), 1)
				dt = round(random.uniform(0.1, 0.3), 2)
				env_car.create_car(desired_v=desired_v, dt=dt)

				log_probs, rewards = [], []
				done = False
				std = 0.5 * (1 - updatet / updates) + 0.05 #creates the randomness in it so the model doesn't get scared and just lock into not discovering anything need. Also it gets lower over time of training so model hopefully locking in its strategy
				while not done:
					state = torch.FloatTensor(env_car.get_state()).unsqueeze(0).to(device) # sets the current state of the enviroments variables to float tensors to be inputs for model
					mean = self.policy(state) # does forward function of the policy to get the output of the model using the tensor states
					dist = Normal(loc=mean, scale=std) # creates the upper and lower bound of the range from the mean (models guess). so essentailly range from model +- std

					action = dist.sample() # chooses based on a bell curve(still no clue what this means lol) but its sometihng like its more likely to pick a number closer to the mean then farther and more to the bounds of std
					log_probs.append(dist.log_prob(action).view(-1)) # gives how probability this "randomized value" from the bell was actually to be chosen by the model with that state
					_, reward, done = env_car.step(torch.clamp(action, -1.0, 1.0).item()) # steps and gets rewards and if done
					rewards.append(reward) # appends rewards

				all_returns.append(self.compute_discounted_rewards(rewards, gamma)) # appends  scoring for game
				all_log_probs.append(torch.cat(log_probs)) # appends probability for game of choosing this results, read how log_probs is append to for more
				batch_total_rewards.append(sum(rewards)) # total rewards for batch appended

			log_probs = torch.cat(all_log_probs) # make them single tensors rather then a bunch of them
			returns = torch.cat(all_returns) # make them single tensors rather then a bunch of them
			returns = (returns - returns.mean()) / (returns.std() + 1e-8) # normilizes the rewards so the model isn't seeing huge numbers per scoring... also the +1e-8 so it doesn't crash cause it accidently would divide by 0
			loss = -(log_probs * returns).sum() / batch_size # it figuring out how to make the good actions stand out and be done again in model while bad not
			self.optimizer.zero_grad() # get rids of old gradient data in model
			loss.backward() # does backpropigation like the learning phase is how I would word this
			self.optimizer.step() # does gradient descent and will shift the weights in the model 

			if updatet % 5 == 0:
				#printing stuff every 5 updates 
				print(f"Update {updatet}, avg reward: {sum(batch_total_rewards)/batch_size:.2f}")
				self.average_reward.append(round(sum(batch_total_rewards)/batch_size, 2))
				self.average_reward_times.append(updatet)

		if(show_training_graphs):
			#graphs reward overtime along with saves it
			plt.figure(figsize=(10, 7))
			plt.scatter(self.average_reward_times, self.average_reward, c="orange", s=4, label="Average reward over update")
			plt.title("Average reward over update")
			plt.savefig("auto/Average_Reward_Over_Update_Last_Training_Run.png")
			plt.show()