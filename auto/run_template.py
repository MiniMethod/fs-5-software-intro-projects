import matplotlib.pyplot as plt
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage

# Notes to self---
# Ku = 20 , Tu or Pu = 2 people change how it said for some reason... for Ziegler Nicholas method
# Ku was when it started doing sin patter
# Tu how often each sine completes which was 2 steps 

K_P = 11.76
K_I = 10
K_D = 2.5

STEPS = 550

car = make_car(desired_v=20.0, dt=0.1)

#WRITE CODE HERE

velocities = []
errors = []
times = []

for steps in range(STEPS):
	acceleration_desired, error = calculate_desired_acceleration(car, K_P, K_I, K_D)
	throttle_perc = acceleration_to_throttle_percentage(acceleration_desired)
	update(car, throttle_perc)
	velocities.append(car["v"])
	errors.append(error)
	times.append(car["t"])


plt.plot(velocities)
plt.plot(errors)
plt.show()