import matplotlib.pyplot as plt
from reinforcement_machine_learning import Env_For_Car, RL_Project

# VARIABLES

UPDATES = 5000 # training - how many updates or batches you want to run during trianing
BATCH_SIZE = 32 # training - how many episodes (games it plays) you want it to do per update during training
GAMMA = 0.99 # training - how much you want the model to pay attention to its current decision... higher means it will look further ahead and a single bad idea it will think overall it was for a better reward (bad wording, covered easily online what gamma is)

MAX_DT = 0.3 # training - max dt (delay time... like how often it will update per second... 0.1 is every 0.1 seconds) it will ever see (lowest is hard set to 0.1)
MAX_V = 100 # training and testing - max velocity it will ever see (lowest is hard set to 0)
MAX_THROTTLE_FORCE = 10000 # training and testing - max throttle force, min is set to 3000
MAX_FRICTION = 2.5 # training and testing - max friction number, min hard set to 1.0
STEPS = 1000 # training and testing - how many dt per episode/game

DESIRED_V = 20 # testing - desired velocity to hit
DESIRED_DT = 0.1 # testing - dt during testing game
MASS = 1000 # testing - M
FRICTION = 2 # testing - friction number
THROTTLE_FORCE = 5000 # testing - max throttle due 


TOGGLE_TRAINING = True # training - toggle where model with train (true means it will)
LOAD_PREVIOUSE_MODEL_FOR_TRAINING = False # training - toggles whether to load the previouse model for training or to make a new one from random tensors (True means it will start a new)

# TRAINING
if (TOGGLE_TRAINING):

	CarEnv = Env_For_Car(max_v=MAX_V, max_dt=MAX_DT, max_throttle_force=MAX_THROTTLE_FORCE, max_friction=MAX_FRICTION, steps=STEPS)

	rl_prj = RL_Project()
	if (LOAD_PREVIOUSE_MODEL_FOR_TRAINING):
		rl_prj.load_model("train", "auto/pid-reinforcement/car_throttle_policy.pth")

	rl_prj.train(CarEnv, show_training_graphs=True, updates=UPDATES, batch_size=BATCH_SIZE, gamma=GAMMA)
	rl_prj.save_model("auto/pid-reinforcement/car_throttle_policy.pth")


#TESTING
# create car enviroment obj
CarEnv = Env_For_Car(max_v=MAX_V, max_dt=MAX_DT, max_throttle_force=MAX_THROTTLE_FORCE, max_friction=MAX_FRICTION, steps=STEPS)

# create reinforcement learning obj and loads the model
rl_prj = RL_Project()
rl_prj.load_model("eval", "auto/pid-reinforcement/car_throttle_policy.pth")

velocities = []
rewards = []
times = []
errors = []

# create car enviroment
car = CarEnv.create_car(desired_v=DESIRED_V, dt=DESIRED_DT, max_throttle_force=THROTTLE_FORCE, friction=FRICTION)

# runs the model for each step
P, I, D = rl_prj.get_pid_from_loaded_model(CarEnv)
print(f"P: {P}, I: {I}, D: {D}")
for steps in range(CarEnv.steps):
	_, reward, _ = CarEnv.step(P, I, D)
	rewards.append(reward)
	velocities.append(CarEnv.currentCar["v"])
	times.append(CarEnv.currentCar["t"])
	error = abs(CarEnv.currentCar["desired_v"] - CarEnv.currentCar["v"])
	errors.append(error)

print(f"Final Velocity: {velocities[-1]}")
print(f"Total Reward: {sum(rewards)}")

#plots the fiture of velocities over time and reward over time
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

ax1.scatter(times, velocities, c="orange", s=4, label="Velocities")
ax1.set_title("Velocities")
ax1.set_xlabel("Time")
ax1.set_ylabel("Velocity")
ax1.axhline(y=CarEnv.currentCar["desired_v"], c="r", linestyle="--", linewidth="2")
ax1.legend()

ax2.scatter(times, rewards, c="b", s=4, label="Reward")
ax2.set_title("Reward")
ax2.set_xlabel("Time")
ax2.set_ylabel("Reward")
ax2.legend()

ax3.scatter(times, errors, c="g", s=4, label="Errors")
ax3.set_title("Error")
ax3.set_xlabel("Time")
ax3.set_ylabel("Error")
ax3.legend()

plt.tight_layout()
plt.savefig("auto/pid-reinforcement/Velocity Over Time And Reward Over Time.png")
plt.show()