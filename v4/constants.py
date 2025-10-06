import numpy as np

# --- Simulation Parameters ---
MAX_SIM_STEPS = 350
DT = 0.1
N = 100  # Prediction horizon

# --- Obstacle Parameters ---
K = 5 # Number of dynamic obstacles
L = 3 # Number of static obstacles
D_OBS_R = 1.0

# --- Robot and Goal Parameters ---
GOAL_RADIUS = 1.5
ROBOT_RADIUS = 1.0
MIN_START_GOAL_DIST = 25.0
MAX_LINEAR_VEL = 3.0
MAX_ANGULAR_VEL = np.pi / 2

# --- NMPC Parameters ---
Q = np.diag([2.0, 2.0, 0.1])
R = np.diag([0.1, 0.05])
SLACK_PENALTY = 100
D_SAFE = 3.0

# --- Environment Parameters ---
X_LIM = 50.0
Y_LIM = 50.0
GRID_RESOLUTION = 1.0