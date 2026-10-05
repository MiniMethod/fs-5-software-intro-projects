# import matplotlib.pyplot as plt
# from pid_template import make_car
# from pid_template import update
# from pid_template import calculate_desired_acceleration
# from pid_template import acceleration_to_throttle_percentage

# # Notes to self---
# # Ku = 20 , Tu or Pu = 0.2 people change how it said for some reason... for Ziegler Nicholas method
# # Ku was when it started doing sin patter
# # Tu how often each sine completes which was 2 steps 

# K_P = 11.76
# K_I = 10
# K_D = 2.5

# STEPS = 550

# car = make_car(desired_v=20.0, dt=0.1)

# #WRITE CODE HERE

# velocities = []
# errors = []
# times = []

# for steps in range(STEPS):
# 	acceleration_desired, error = calculate_desired_acceleration(car, K_P, K_I, K_D)
# 	throttle_perc = acceleration_to_throttle_percentage(acceleration_desired)
# 	update(car, throttle_perc)
# 	velocities.append(car["v"])
# 	errors.append(error)
# 	times.append(car["t"])


# plt.figure(figsize=(10, 7))

# plt.scatter(times, velocities, c="orange", s=4, label="Velocities")
# # plt.scatter(times, errors, c="b", s=4, label="Errors")
# plt.show()





import matplotlib.pyplot as plt
import torch
from reinforcement_machine_learning import Env_For_Car, RL_Project

# TRAINING
CarEnv = Env_For_Car(max_v=100, max_dt=0.3, steps=550)

rl_prj = RL_Project()

rl_prj.train(CarEnv, updates=320, batch_size=16)
rl_prj.save_model("car_throttle_policy.pth")


#TESTING
CarEnv = Env_For_Car(max_v=100, max_dt=0.3, steps=550)

rl_prj = RL_Project()
rl_prj.load_model("car_throttle_policy.pth")

velocities = []
rewards = []
times = []

car = CarEnv.create_car(desired_v=20, dt=0.1)

for steps in range(CarEnv.steps):
	throttle_perc = rl_prj.get_desired_throttle_from_loaded_model(CarEnv)
	_, reward, _ = CarEnv.step(throttle_perc)
	rewards.append(reward)
	velocities.append(CarEnv.currentCar["v"])
	times.append(CarEnv.currentCar["t"])

print(F"Total Reward: {sum(rewards)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(times, velocities, c="orange", s=4, label="Velocities")
ax1.set_title("Velocities")
ax1.set_xlabel("Time")
ax1.set_ylabel("Velocity")
ax1.legend()

ax2.scatter(times, rewards, c="b", s=4, label="Reward")
ax2.set_title("Reward")
ax2.set_xlabel("Time")
ax2.set_ylabel("Reward")
ax2.legend()

plt.tight_layout()
plt.show()
	