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

# --- NMPC Parameters ---
Q_path = np.diag([50.0, 50.0])    # Path tracking weight (reduced for smoother motion)
R = np.diag([0.1, 0.05])          # Weights for control input
D_SAFE = 1.0                       # Safety distance (reduced from 1.5 to allow more freedom)
CBF_GAMMA = 1.0                    # Damping factor for higher-order CBF (increased for more conservative behavior)

# --- Environment Parameters ---
X_LIM = 50.0
Y_LIM = 50.0
GRID_RESOLUTION = 1.0

# --- RRT* Replanning Parameters ---
PATH_CROSSING_THRESHOLD = 2.0  # Distance threshold to check if obstacle crosses path
H_RECOVERY_THRESHOLD = 0.3
H_NORMAL_THRESHOLD = 1.0
CONSERVATIVE_SAFETY_MARGIN = D_SAFE + 2.0