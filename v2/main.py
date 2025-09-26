# Maintainer: Jainik Mehta
# Contact: jainikjmehta@gmail.com
# Website: https://jainikmehta.com
# Last Update Date: September 2025

# Description: A simulator for a unicycle robot in a 2D environment with static and dynamic obstacles.

import numpy as np
import random
from plotting import plot_environment
from utils import Grid

# --- Environment Parameters ---
X_LIM = 50.0
Y_LIM = 50.0
GRID_RESOLUTION = 1.0
DT = 0.1
MAX_EPISODES = 10

# --- Obstacle Parameters ---
K = 5  # Number of dynamic obstacles
L = 3  # Number of static obstacles
D_OBS_R = 1.0
D_SAFE = 1.5

# --- Robot and Goal Parameters ---
GOAL_RADIUS = 1.0
ROBOT_RADIUS = 1.0
MIN_START_GOAL_DIST = 25.0

# --- Simulation Classes ---
class Robot:
    def __init__(self, x, y, theta):
        self.state = np.array([x, y, theta])
        self.radius = ROBOT_RADIUS

class Goal:
    def __init__(self, x, y):
        self.state = np.array([x, y])
        self.radius = GOAL_RADIUS

class DynamicObstacle:
    def __init__(self, x, y):
        self.state = np.array([x, y, np.random.uniform(-np.pi, np.pi)])
        self.velocity = np.random.uniform(0.5, 2.0)
        self.radius = D_OBS_R

class StaticObstacle:
    def __init__(self, x, y, w, h):
        self.center = np.array([x, y])
        self.width = w
        self.height = h

def setup_environment(grid):
    """
    Places obstacles, goal, and robot in the environment grid without collisions.
    """
    # 1. Place Static Obstacles
    static_obstacles = []
    for _ in range(L):
        w = np.random.uniform(5.0, 10.0)
        h = np.random.uniform(5.0, 10.0)
        w_cells = int(np.ceil(w / grid.resolution))
        h_cells = int(np.ceil(h / grid.resolution))

        pos = grid.find_free_rect_space(w_cells, h_cells)
        if pos:
            x_start, y_start = pos
            center_x = (x_start + w_cells / 2) * grid.resolution
            center_y = (y_start + h_cells / 2) * grid.resolution
            
            obstacle = StaticObstacle(center_x, center_y, w, h)
            static_obstacles.append(obstacle)
            grid.mark_rect_as_occupied(x_start, y_start, w_cells, h_cells, D_SAFE)

    # 2. Place Goal
    goal_pos = grid.find_random_free_cell()
    if goal_pos is None: raise Exception("No space left for goal.")
    gx, gy = (goal_pos + 0.5) * grid.resolution
    goal = Goal(gx, gy)
    grid.mark_circle_as_occupied(gx, gy, goal.radius, D_SAFE)

    # 3. Place Robot (far from goal)
    robot = None
    for _ in range(100):
        robot_pos = grid.find_random_free_cell()
        if robot_pos is not None:
            rx, ry = (robot_pos + 0.5) * grid.resolution
            if np.linalg.norm(np.array([rx, ry]) - goal.state) >= MIN_START_GOAL_DIST:
                theta = np.random.uniform(-np.pi, np.pi)
                robot = Robot(rx, ry, theta)
                grid.mark_circle_as_occupied(rx, ry, robot.radius, D_SAFE)
                break
    if robot is None: raise Exception("Could not place robot far enough from goal.")

    # 4. Place Dynamic Obstacles
    dynamic_obstacles = []
    for _ in range(K):
        obs_pos = grid.find_random_free_cell()
        if obs_pos is not None:
            ox, oy = (obs_pos + 0.5) * grid.resolution
            obstacle = DynamicObstacle(ox, oy)
            dynamic_obstacles.append(obstacle)
            grid.mark_circle_as_occupied(ox, oy, obstacle.radius, D_SAFE)

    return robot, goal, static_obstacles, dynamic_obstacles

if __name__ == "__main__":
    for episode in range(MAX_EPISODES):
        print(f"--- Setting up Episode {episode + 1} ---")
        try:
            grid = Grid(X_LIM, Y_LIM, GRID_RESOLUTION, border_width=2)
            robot, goal, static_obs, dynamic_obs = setup_environment(grid)
            
            print("Environment setup successful.")
            plot_environment(robot, goal, static_obs, dynamic_obs, duration=5)

        except Exception as e:
            print(f"Error setting up episode: {e}")
            print("Skipping to next episode.")