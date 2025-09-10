# 50 m by 50 m area with static and dynamic obstacles
#    - l Static obstacles: Rectangle with minimum edge length of 1 m and maximum edge length of 15 m. Maximum of 10 static obstacles.
#    - K Dynamic obstacles: Circle with radius d_obs_r = . Moving at minimum speed of 0.5 m/s and maximum speed of 2 m/s. Maximum of 10 dynamic obstacles. Intiallized with random position (from a given patch) and random velocity and heading direction. Obstacle stops when it reaches the boundary of the area. Obstacles stops after collision with other obstacles. Should have option to change the heading of the obstacle after collision or after some time.
# Unicycle robot with 2 m/s speed


import numpy as np
import matplotlib.pyplot as plt
import casadi as ca