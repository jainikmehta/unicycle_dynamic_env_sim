# Maintainer: Jainik Mehta
# Contact: jainikjmehta@gmail.com
# Website: https://jainikmehta.com
# Last Update Date: September 2025

# Description: A simulator for training different RL algorithms for a unicycle robot in a 2D environment with static and dynamic obstacles.
# Simulator includes:
# Controller for the robot - NMPC controller
# Prediction block for dynamic obstacles motion
# Visualization of the environment
# Logging of the robot and obstacles positions
# Per step reward estimation model training
# Training and testing of RL algorithms


# A configuration file for a simulation environment with the following specifications:
# 50 m by 50 m area with static and dynamic obstacles
#    - l Static obstacles: Rectangle with minimum edge length of 1 m and maximum edge length of 15 m. Maximum of 10 static obstacles.
#    - K Dynamic obstacles: Circle with radius d_obs_r = . Moving at minimum speed of 0.5 m/s and maximum speed of 2 m/s. Maximum of 10 dynamic obstacles. Intiallized with random position (from a given patch) and random velocity and heading direction. 
# 
# 
# Obstacle changes heading when it reaches the boundary of the area or after collision with another obstacle.


# Robot:
# Unicycle robot with 2 m/s speed

# Logging file
# Logs the position of the robot and each obstacles at each time step

# Object Dynamics file
# Provides the next position of robot and obstacles given the current position and control inputs

# Prediction file
# Provides prediction of dynamic obstacles motion along with uncertainty estimation.


# Plotting file


import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import random
from dynamics import unicycle_dynamics, obstacle_dyanmics
from prediciton import predict_obstacle_motion
from plotting import plot_environment
from utils import grid_cell
HEADLESS = False  # Set to True to disable plotting for faster training

# Environment parameters
X_LIM = 50  # Environment size in x direction 50 m
Y_LIM = 50  # Environment size in y direction 50 m
GRID_RESOLUTION = 1.0 # Grid resolution in meters
DT = 0.1  # Time step in seconds
N = 20  # Prediction horizon
MAX_EPISODES = 1000  # Number of episodes for trainings
EPISODE_LENGTH = 1000  # Length of each episode in time steps
N = 20  # Prediction horizon
# Border buffer for static obstacle placement
BORDER_BUFFER = 15.0 # Minimum distance from the border to place static obstacles. This should be at least the maximum static obstacle size.

# Dynamic Obstacles
K = 5  # Number of dynamic obstacles


L = 3  # Number of static obstacles
D_OBS_R = 1.0  # Radius of dynamic obstacles
D_SAFE = 0.5  # Safety distance
HEADING_NOISE = 0.1  # Noise in obstacle heading change (radians)

# Robot and Goal
STATE = np.array([5.0, 5.0, 0.0])  # Initial state of the robot [x, y, theta]
GOAL = np.array([45.0, 45.0])  # Goal position
GOAL_RADIUS = 1.0  # Radius to consider goal reached
ROBOT_RADIUS = 1.0  # Radius of the robot
MAX_LINEAR_VEL = 2.0  # Maximum linear velocity of the robot
MAX_ANGULAR_VEL = np.pi / 4  # Maximum angular velocity of the robot
# NMPC parameters
Q = np.diag([1, 1, 0.1])  # State cost weights
R = np.diag([0.1, 0.1])  # Control cost weights



# Initialize Robot and Dynamic Obstacles (modeled as circular objects)
class robot_class():
    def __init__(self, state = STATE):
        self.state = state  # state = [x, y, theta]
        self.radius = ROBOT_RADIUS
        self.max_linear_vel = MAX_LINEAR_VEL
        self.max_angular_vel = MAX_ANGULAR_VEL
        self.trajectory = [state]  # To log the trajectory
        self.control_inputs = []  # To log control inputs

    def __repr__(self):
        return f"RobotState (x={self.state[0]}, y={self.state[1]}, theta={self.state[2]})"

    def update_state(self, control_input):
        self.state = unicycle_dynamics(self.state, control_input, DT)
        self.trajectory.append(self.state)
        self.control_inputs.append(control_input)

class goal_class():
    def __init__(self, state = GOAL):
        self.state = state  # state = [x, y]
        self.x = state[0]
        self.y = state[1]
        self.radius = GOAL_RADIUS

    def __repr__(self):
        return f"GoalPosition (x={self.state[0]}, y={self.state[1]})"

class dynamic_obstacle_class():
    def __init__(self, grid):
        # Sample random empty grid cell for obstacle placement
        # grid.plot_cells()  # Optional: visualize grid cells
        empty_cells = np.argwhere(grid.grid_cells == 0)
        cell_idx = random.choice(range(len(empty_cells)))
        cell = empty_cells[cell_idx]
        self.x = (cell[0] + 0.5) * GRID_RESOLUTION
        self.y = (cell[1] + 0.5) * GRID_RESOLUTION
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.v = np.random.uniform(0.5, 2.0)  # Random speed between 0.5 m/s and 2 m/s
        self.omega = 0
        self.state = [self.x, self.y, self.theta]  # state = [x, y, theta] observerd at timestep t
        self.control_input = [self.v, self.omega]  # control_input = [v, omega]
        self.radius = D_OBS_R
        self.trajectory = [self.state]  # To log the trajectory
        self.predicted_traj = []  # To log the next N predicted states

    def __repr__(self):
        return f"DynamicObstacleState (x={self.state[0]}, y={self.state[1]}, theta={self.state[2]})"
    
    def update_state(self):
        self.state = obstacle_dyanmics(self.state, self.control_input, DT, HEADING_NOISE)
        self.trajectory.append(self.state)  

    def pred_state(self, control_input):
        self.next_N_states = (self.state, control_input, DT)
        self.predicted_traj.append(self.state)

class static_obstacle_class():
    def __init__(self, grid=None):
        # Allow grid-aware sampling to avoid overlaps with already placed objects
        # Static obstacle size in meters
        self.width = np.random.uniform(10.0, 15.0)
        self.height = np.random.uniform(10.0, 15.0)
        # Convert size to cell counts (at least 1 cell)
        w_cells = max(1, int(np.ceil(self.width / GRID_RESOLUTION)))
        h_cells = max(1, int(np.ceil(self.height / GRID_RESOLUTION)))

        if grid is not None:
            # find all top-left positions where a w_cells x h_cells patch fits and is free
            placed = False
            candidates = []
            max_x_start = grid.grid_width - w_cells
            max_y_start = grid.grid_height - h_cells
            buffer_cells = 1  # extra margin so static obstacles don't touch
            if max_x_start >= 0 and max_y_start >= 0:
                for xs in range(0, max_x_start + 1):
                    for ys in range(0, max_y_start + 1):
                        # compute expanded patch including buffer
                        bx0 = max(0, xs - buffer_cells)
                        by0 = max(0, ys - buffer_cells)
                        bx1 = min(grid.grid_width - 1, xs + w_cells - 1 + buffer_cells)
                        by1 = min(grid.grid_height - 1, ys + h_cells - 1 + buffer_cells)
                        if np.all(grid.grid_cells[bx0:bx1+1, by0:by1+1] == 0):
                            candidates.append((xs, ys))

            if candidates:
                x_start, y_start = random.choice(candidates)
                x_end = x_start + w_cells - 1
                y_end = y_start + h_cells - 1
                # mark those cells as occupied immediately (only the obstacle area, not buffer)
                grid.grid_cells[x_start:x_end+1, y_start:y_end+1] = 1
                # store cell indices on the obstacle and compute center coordinates
                self.x_start = x_start
                self.y_start = y_start
                self.x_end = x_end
                self.y_end = y_end
                # compute center in meters
                cx = (x_start + x_end + 1) * 0.5 * GRID_RESOLUTION
                cy = (y_start + y_end + 1) * 0.5 * GRID_RESOLUTION
                self.x = cx
                self.y = cy
                self.state = [self.x, self.y]
                placed = True

            if not placed:
                # Fallback: choose a random center in buffer and compute cell indices
                self.x = np.random.uniform(BORDER_BUFFER, X_LIM - BORDER_BUFFER)
                self.y = np.random.uniform(BORDER_BUFFER, Y_LIM - BORDER_BUFFER)
                # compute occupied cells and mark them
                x_start = max(0, int((self.x - self.width / 2) / GRID_RESOLUTION) - 1)
                x_end = min(grid.grid_width - 1, int((self.x + self.width / 2) / GRID_RESOLUTION) + 1)
                y_start = max(0, int((self.y - self.height / 2) / GRID_RESOLUTION) - 1)
                y_end = min(grid.grid_height - 1, int((self.y + self.height / 2) / GRID_RESOLUTION) + 1)
                grid.grid_cells[x_start:x_end+1, y_start:y_end+1] = 1
                self.state = [self.x, self.y]
                self.x_start = x_start
                self.y_start = y_start
                self.x_end = x_end
                self.y_end = y_end
        else:
            # No grid given: fallback to a simple random center
            self.x = np.random.uniform(BORDER_BUFFER, X_LIM - BORDER_BUFFER)
            self.y = np.random.uniform(BORDER_BUFFER, Y_LIM - BORDER_BUFFER)
            self.state = [self.x, self.y]
            # compute cell indices without marking
            self.x_start = max(0, int((self.x - self.width / 2) / GRID_RESOLUTION) - 1)
            self.x_end = min(int(X_LIM / GRID_RESOLUTION) - 1, int((self.x + self.width / 2) / GRID_RESOLUTION) + 1)
            self.y_start = max(0, int((self.y - self.height / 2) / GRID_RESOLUTION) - 1)
            self.y_end = min(int(Y_LIM / GRID_RESOLUTION) - 1, int((self.y + self.height / 2) / GRID_RESOLUTION) + 1)

    def __repr__(self):
        return f"StaticObstacle (x={self.x}, y={self.y}, width={self.width}, height={self.height})"



# Collision checking between two circular objects (robot and dynamic obstacle)
def check_collision_cc(object1, object2):
    dist = np.linalg.norm(np.array(object1.state[:2]) - np.array(object2.state[:2]))
    return dist < (object1.radius + object2.radius + D_SAFE)

def check_collision_rc(object1, static_obstacle):
    circle_x, circle_y = object1.state[0], object1.state[1]
    rect_x, rect_y = static_obstacle.state[0], static_obstacle.state[1]
    rect_w, rect_h = static_obstacle.width, static_obstacle.height

    closest_x = np.clip(circle_x, rect_x, rect_x + rect_w)
    closest_y = np.clip(circle_y, rect_y, rect_y + rect_h)

    dist = np.linalg.norm(np.array([circle_x, circle_y]) - np.array([closest_x, closest_y]))
    return dist < (object1.radius + D_SAFE)

# collsion checking between walls and circular objects (robot and dynamic obstacle)
def check_collision_cw(object1):
    x, y = object1.state[0], object1.state[1]
    if x - object1.radius < 0 or x + object1.radius > X_LIM or y - object1.radius < 0 or y + object1.radius > Y_LIM:
        return True
    return False   



if __name__== "__main__":
# Simulation run
# Collect data for training per step reward model.
# Each episode starts with random initialization of robot and obstacles.
# For each time step in the episode:
#   - Predict the motion of each dynamic obstacle over a prediction horizon N given their current state and control inputs.
#   - Estimate uncertainty scaling factor associated with each prediction. RL takes as input robot state, predicted obstacle states and uncertainty. The input dimension is calculated as follows:
#       - Robot state: 3 (x, y, theta)
#       - Dynamic obstacle: (2 (mu_x, mu_y) + 2 (sigma_x, sigma_y) )* N (prediction horizon) * K (number of dynamic obstacles)
# The obstacles prediction and uncertainty is flattened and concatenated with robot state to form the input to RL agent. Robot state is always first in flattened input vector. First closest obstacle is second in the input vector. We only consider top 10 closest obstacles to the robot. If K < 10, we pad the input with zeros.

#   - Total input dimension = 3 + K * (3 * N + 1
#   - RL outputs unceratinity scaling factor that scales unceratinity value between 0.25 sigma to 2 sigma range for each obstacle and each time step in the prediction horizon. The output dimension is K * N * 2.
#  - This scaled uncertainty is used to formulate CBF constraints in the NMPC controller.
#   - Robot NMPC controller computes control inputs based on current state and predicted obstacle states with scaled unceratainity.
# Each step RL agent doesn't recieve any reward signal. At the end of the episode, a reward is calculated based on the following criteria:
#   - +1000 for reaching the goal
#   - -100 for collision with any obstacle
#   - -1 for each time step to encourage faster reaching to the goal




    for episode in range(MAX_EPISODES):
        # Initialize grid only for initial object placement
        grid = grid_cell(X_LIM, Y_LIM, GRID_RESOLUTION)
        # Place robot and goal first at random free cells, ensuring they are far apart
        min_separation = 15.0  # meters: minimum distance between robot and goal
        empty_cells = np.argwhere(grid.grid_cells == 0)
        if len(empty_cells) == 0:
            raise RuntimeError('No free cells available to place robot and goal')

        # Sample robot position
        cell_idx = random.choice(range(len(empty_cells)))
        cell = empty_cells[cell_idx]
        rx = (cell[0] + 0.5) * GRID_RESOLUTION
        ry = (cell[1] + 0.5) * GRID_RESOLUTION
        rtheta = np.random.uniform(-np.pi, np.pi)
        robot = robot_class(state = np.array([rx, ry, rtheta]))
        grid.update_grid(robot, obstacle_type='c')

        # Sample goal position ensuring it's sufficiently far from the robot
        goal_placed = False
        tries = 0
        max_goal_tries = 200
        while not goal_placed and tries < max_goal_tries:
            tries += 1
            empty_cells = np.argwhere(grid.grid_cells == 0)
            if len(empty_cells) == 0:
                break
            cell_idx = random.choice(range(len(empty_cells)))
            cell = empty_cells[cell_idx]
            gx = (cell[0] + 0.5) * GRID_RESOLUTION
            gy = (cell[1] + 0.5) * GRID_RESOLUTION
            if np.linalg.norm(np.array([rx, ry]) - np.array([gx, gy])) >= min_separation:
                goal = goal_class(state = np.array([gx, gy]))
                grid.update_grid(goal, obstacle_type='c')
                goal_placed = True

        if not goal_placed:
            # last resort: place goal deterministically far (corner opposite) and update grid
            gx = X_LIM - rx
            gy = Y_LIM - ry
            goal = goal_class(state = np.array([gx, gy]))
            grid.update_grid(goal, obstacle_type='c')

        # Initialize dynamic obstacles at random positions and headings but not overlapping with robot/goal
        dynamic_obstacles = []
        while len(dynamic_obstacles) < K:
            obs = dynamic_obstacle_class(grid)
            dynamic_obstacles.append(obs)
            grid.update_grid(obs, obstacle_type='c')

        # Finally place static obstacles (rectangles) in remaining free space
        static_obstacles = [static_obstacle_class(grid) for _ in range(L)]
        
        if not HEADLESS:
            # Show environment for 10 seconds then close, then continue to next episode
            plot_environment(robot, goal, static_obstacles, dynamic_obstacles, duration=5)

        # # Run the episode
        # for t in range(EPISODE_LENGTH):
        #     # Predict obstacle motion
        #     obstacle_states = [obs.state for obs in dynamic_obstacles]
        #     obstacle_controls = [obs.control_input for obs in dynamic_obstacles]
        #     predictions = predict_obstacle_motion(obstacle_states, obstacle_controls, K, N, DT)

        #     # Here we would call the RL agent to get uncertainty scaling factors
        #     # For now, we use a placeholder of ones
        #     uncertainty_scaling = np.ones((K, N, 2))

        #     # Here we would call the NMPC controller to get control inputs for the robot
        #     # For now, we use a placeholder of zero control inputs
        #     control_input = np.array([0.0, 0.0])  # [linear_velocity, angular_velocity]

        #     # Update robot state
        #     robot.update_state(control_input)

        #     # Update dynamic obstacles states
        #     for obs in dynamic_obstacles:
        #         obs.update_state()

        #     # Check for collisions

        # else:
        #     print(f"Episode {episode}: Completed without collision")



# Testing RL