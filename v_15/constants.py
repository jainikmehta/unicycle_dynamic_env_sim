"""
constants.py

Central file for all simulation parameters.
"""

import numpy as np

# --- Simulation Parameters ---
MAX_SIM_STEPS = 500         # Max steps per episode
DT = 0.1                    # Timestep duration (s)
N = 20                      # NMPC prediction horizon

# --- Environment Parameters ---
X_LIM = 50.0                # Environment width (m)
Y_LIM = 50.0                # Environment height (m)
GRID_RESOLUTION = 1.0       # Occupancy grid resolution (m)
MIN_START_GOAL_DIST = 25.0  # Min distance between robot start and goal

# --- Robot Parameters ---
ROBOT_RADIUS = 1.0          # (m)
MAX_LINEAR_VEL = 5.0        # (m/s)
MAX_ANGULAR_VEL = np.pi / 2   # (rad/s)
MAX_LINEAR_ACCEL = 5.0      # (m/s^2)
MAX_ANGULAR_ACCEL = np.pi / 2 # (rad/s^2)
BRAKING_ACCELERATION = -2.5 # Fallback braking (m/s^2)

# --- Goal Parameters ---
GOAL_RADIUS = 1.5           # (m)

# --- Obstacle Parameters ---
D_OBS_RADIUS = 1.0          # Dynamic obstacle radius (m)
STATIC_OBS_MIN_SIZE = 5.0   # (m)
STATIC_OBS_MAX_SIZE = 10.0  # (m)

# --- Uncertainty Parameters ---
OBSTACLE_POS_NOISE_STD = 0.15 # (m)
UNCERTAINTY_GROWTH_RATE = 0.1 # Variance added per second
HEADING_NOISE_RANGE = np.pi / 18 # +/- 10 degrees
SIGMA_BOUND_LOWER = 0.5     # Lower bound for random uncertainty scaling
SIGMA_BOUND_UPPER = 3.0     # Upper bound for random uncertainty scaling

# --- NMPC Parameters ---
# FIX: Increased Q_path to enforce stricter path following
Q_path = np.diag([50.0, 50.0])  # State cost (path tracking)
# FIX: Added Q_vel to incentivize forward motion and prevent getting "stuck"
Q_vel = 0.1                     # Velocity reward
R = np.diag([0.1, 0.1])         # Control cost (a, alpha)
D_SAFE = 1.0                    # Safety margin (m)
# Use a reasonable gain for the discrete-time CBF
CBF_GAMMA = 2.0                 # CBF gain (alpha)

# --- RRT* Parameters ---
RRT_MAX_ITER = 1500
RRT_STEP_SIZE = 2.0
RRT_SEARCH_RADIUS = 4.0
RRT_GOAL_SAMPLE_RATE = 0.1
PATH_DEVIATION_THRESHOLD = 5.0  # When to trigger a replan