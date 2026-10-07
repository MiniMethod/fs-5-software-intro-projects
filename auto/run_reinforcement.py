import matplotlib.pyplot as plt
from reinforcement_machine_learning import Env_For_Car, RL_Project

# VARIABLES

UPDATES = 2000
BATCH_SIZE = 16
GAMMA = 0.99

MAX_DT = 0.3
MAX_V = 100
STEPS = 550

DESIRED_V = 20
DESIRED_DT = 0.1

TOGGLE_TRAINING = True #True means on False means off
LOAD_PREVIOUSE_MODEL_FOR_TRAINING = False #True mean will train using previouse model False means it will generate new model


# TRAINING
if (TOGGLE_TRAINING):

	CarEnv = Env_For_Car(max_v=MAX_V, max_dt=MAX_DT, steps=STEPS)

	rl_prj = RL_Project()
	if (LOAD_PREVIOUSE_MODEL_FOR_TRAINING):
		rl_prj.load_model("train", "auto/car_throttle_policy.pth")

	rl_prj.train(CarEnv, show_training_graphs=True, updates=UPDATES, batch_size=BATCH_SIZE, gamma=GAMMA)
	rl_prj.save_model("car_throttle_policy.pth")


#TESTING
# create car enviroment obj
CarEnv = Env_For_Car(max_v=MAX_V, max_dt=MAX_DT, steps=STEPS)

# create reinforcement learning obj and loads the model
rl_prj = RL_Project()
rl_prj.load_model("eval", "auto/car_throttle_policy.pth")

velocities = []
rewards = []
times = []

# create car enviroment
car = CarEnv.create_car(desired_v=DESIRED_V, dt=DESIRED_DT)

# runs the model for each step
for steps in range(CarEnv.steps):
	throttle_perc = rl_prj.get_desired_throttle_from_loaded_model(CarEnv)
	_, reward, _ = CarEnv.step(throttle_perc)
	rewards.append(reward)
	velocities.append(CarEnv.currentCar["v"])
	times.append(CarEnv.currentCar["t"])

print(F"Total Reward: {sum(rewards)}")

#plots the fiture of velocities over time and reward over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

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

plt.tight_layout()
plt.savefig("auto/Velocity Over Time And Reward Over Time.png")
plt.show()