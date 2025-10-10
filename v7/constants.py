import numpy as np

# --- Simulation Parameters ---
MAX_SIM_STEPS = 200
DT = 0.1
N = 20  # Prediction horizon

# --- Obstacle Parameters ---
K = 5 # Number of dynamic obstacles
L = 3 # Number of static obstacles
D_OBS_R = 1.0
NOISY_OBSTACLES = True # SET TO TRUE/FALSE to enable/disable noise

# --- Uncertainty Parameters (if NOISY_OBSTACLES is True) ---
OBSTACLE_POS_NOISE_STD = 0.15  # meters
UNCERTAINTY_GROWTH_RATE = 0.1 # How much variance is added per second
SIGMA_BOUND = 3.0              # e.g., 3.0 means controller respects 3-sigma bound

# --- Robot and Goal Parameters ---
GOAL_RADIUS = 1.5
ROBOT_RADIUS = 1.0
MIN_START_GOAL_DIST = 25.0
MAX_LINEAR_VEL = 3.0
MAX_ANGULAR_VEL = np.pi / 2

# --- NMPC Parameters ---
Q_path = np.diag([5.0, 5.0]) # Weights for path tracking error
R = np.diag([0.1, 0.05])     # Weights for control input
SLACK_PENALTY = 1000
D_SAFE = 3.0

# --- Environment Parameters ---
X_LIM = 50.0
Y_LIM = 50.0
GRID_RESOLUTION = 1.0