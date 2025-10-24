import numpy as np

# --- Simulation Parameters ---
MAX_SIM_STEPS = 500
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
MAX_LINEAR_VEL = 5.0
MAX_ANGULAR_VEL = np.pi / 2
MAX_LINEAR_ACCEL = 5.0  # m/s^2
MAX_ANGULAR_ACCEL = np.pi / 4  # rad/s^2
BRAKING_ACCELERATION = -2.5 # m/s^2, used for safety stop

# --- NMPC Parameters ---
Q_path = np.diag([0.5, 0.5]) # Weight for path tracking error
Q_vel = 0.1 # Weight for maximizing forward velocity
R = np.diag([0.1, 0.05]) # Weight for control inputs
D_SAFE = 1.0 # Safety distance
CBF_GAMMA = 1.0 # CBF parameter

# --- Environment Parameters ---
X_LIM = 50.0
Y_LIM = 50.0
GRID_RESOLUTION = 1.0
MIN_SPAWN_H_VALUE = 2.0

# --- RRT* Replanning Parameters ---
PATH_DEVIATION_THRESHOLD = 5.0 # Threshold for replanning RRT*
RRT_MAX_ITER = 1500
RRT_STEP_SIZE = 2.0
RRT_SEARCH_RADIUS = 4.0
RRT_GOAL_SAMPLE_RATE = 0.1