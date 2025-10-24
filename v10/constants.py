import numpy as np

# --- Simulation Parameters ---
MAX_SIM_STEPS = 500
DT = 0.1
N = 50  # Prediction horizon

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
Q_path = np.diag([0.5, 0.5])
R = np.diag([0.1, 0.05])
D_SAFE = 1.0
CBF_GAMMA = 1.0

# --- Environment Parameters ---
X_LIM = 50.0
Y_LIM = 50.0
GRID_RESOLUTION = 1.0
MIN_SPAWN_H_VALUE = 2.0

# --- RRT* Replanning Parameters ---
PATH_CROSSING_THRESHOLD = 2.0
H_RECOVERY_THRESHOLD = 0.3
H_NORMAL_THRESHOLD = 1.0
CONSERVATIVE_SAFETY_MARGIN = D_SAFE + 2.0

# --- Intermediate Goal Parameters ---
INTERMEDIATE_GOAL_LOOKAHEAD_DISTANCE = 8.0 # How far along the RRT* path to look for a new goal
INTERMEDIATE_GOAL_REACH_THRESHOLD = 1.5   # How close the robot needs to be to a goal to have "reached" it
INTERMEDIATE_GOAL_COLLISION_RADIUS = 2.0  # Safety buffer when checking for goal collisions