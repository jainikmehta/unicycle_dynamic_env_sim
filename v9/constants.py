import numpy as np

# --- Simulation Parameters ---
MAX_SIM_STEPS = 200
DT = 0.1
N = 20  # Prediction horizon

# --- Obstacle Parameters ---
K = 5 # Number of dynamic obstacles
L = 3 # Number of static obstacles
D_OBS_R = 1.0

# --- Uncertainty Parameters ---
OBSTACLE_POS_NOISE_STD = 0.15  # meters
UNCERTAINTY_GROWTH_RATE = 0.1 # How much variance is added per second
HEADING_NOISE_RANGE = np.pi / 18 # +/- 10 degrees in radians
SIGMA_BOUND_LOWER = 0.5
SIGMA_BOUND_UPPER = 3.0

# --- Robot and Goal Parameters ---
GOAL_RADIUS = 1.5
ROBOT_RADIUS = 1.0
MIN_START_GOAL_DIST = 25.0
MAX_LINEAR_VEL = 3.0
MAX_ANGULAR_VEL = np.pi / 2
MAX_LINEAR_ACCEL = 1.0  # m/s^2
MAX_ANGULAR_ACCEL = np.pi / 4  # rad/s^2

# --- NMPC Parameters ---
Q_path = np.diag([25.0, 25.0]) # Weights for path tracking error
R = np.diag([0.1, 0.05])     # Weights for control input
SLACK_PENALTY = 1000
D_SAFE = 1.0
CBF_GAMMA = 0.7 # Damping factor for higher-order CBF constraint

# --- Environment Parameters ---
X_LIM = 50.0
Y_LIM = 50.0
GRID_RESOLUTION = 1.0

