"""
env_setup.py

Handles the creation of the simulation environment using an occupancy grid.
"""

import numpy as np
import random
import constants as const
from simulation_entities import Robot, Goal, StaticObstacle, DynamicObstacle

class Grid:
    """Helper class for tracking occupied space to avoid impossible spawns."""
    def __init__(self, x_lim, y_lim, resolution, border_width=1):
        self.resolution = resolution
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.grid_width = int(x_lim / resolution)
        self.grid_height = int(y_lim / resolution)
        # 0 = free, 1 = occupied
        self.cells = np.zeros((self.grid_width, self.grid_height), dtype=int)
        
        if border_width > 0:
            self.cells[0:border_width, :] = 1  # Left
            self.cells[-border_width:, :] = 1 # Right
            self.cells[:, 0:border_width] = 1  # Bottom
            self.cells[:, -border_width:] = 1 # Top

    def _to_grid_coords(self, x, y):
        return int(x / self.resolution), int(y / self.resolution)

    def mark_rect_as_occupied(self, x_start, y_start, w_cells, h_cells, safety_dist):
        """Marks a rectangle plus a safety buffer as occupied."""
        buffer_cells = int(np.ceil(safety_dist / self.resolution))
        x_start_buf = max(0, x_start - buffer_cells)
        y_start_buf = max(0, y_start - buffer_cells)
        x_end_buf = min(self.grid_width, x_start + w_cells + buffer_cells)
        y_end_buf = min(self.grid_height, y_start + h_cells + buffer_cells)
        self.cells[x_start_buf:x_end_buf, y_start_buf:y_end_buf] = 1

    def mark_circle_as_occupied(self, x, y, radius, safety_dist):
        """Marks a circular area plus a safety buffer as occupied."""
        total_radius = radius + safety_dist
        x_min_cell, y_min_cell = self._to_grid_coords(x - total_radius, y - total_radius)
        x_max_cell, y_max_cell = self._to_grid_coords(x + total_radius, y + total_radius)

        for i in range(max(0, x_min_cell), min(self.grid_width, x_max_cell + 1)):
            for j in range(max(0, y_min_cell), min(self.grid_height, y_max_cell + 1)):
                cell_x = (i + 0.5) * self.resolution
                cell_y = (j + 0.5) * self.resolution
                if np.linalg.norm([x - cell_x, y - cell_y]) <= total_radius:
                    self.cells[i, j] = 1
    
    def find_free_rect_space(self, w_cells, h_cells):
        """Finds a top-left cell index for a free rectangular area."""
        candidates = []
        for r in range(self.grid_width - w_cells + 1):
            for c in range(self.grid_height - h_cells + 1):
                if not np.any(self.cells[r:r+w_cells, c:c+h_cells]):
                    candidates.append((r, c))
        return random.choice(candidates) if candidates else None

    def find_random_free_cell(self):
        """Finds the indices of a random free cell."""
        free_cells = np.argwhere(self.cells == 0)
        return random.choice(free_cells) if len(free_cells) > 0 else None

def setup_environment(num_static, num_dynamic):
    """
    Creates and places all simulation entities in a valid configuration.
    """
    grid = Grid(const.X_LIM, const.Y_LIM, const.GRID_RESOLUTION, border_width=2)
    static_obstacles = []
    dynamic_obstacles = []
    
    # Place Static Obstacles
    for _ in range(num_static):
        w = np.random.uniform(const.STATIC_OBS_MIN_SIZE, const.STATIC_OBS_MAX_SIZE)
        h = np.random.uniform(const.STATIC_OBS_MIN_SIZE, const.STATIC_OBS_MAX_SIZE)
        w_cells = int(np.ceil(w / grid.resolution))
        h_cells = int(np.ceil(h / grid.resolution))
        pos = grid.find_free_rect_space(w_cells, h_cells)
        if pos:
            x_start, y_start = pos
            center_x = (x_start + w_cells / 2) * grid.resolution
            center_y = (y_start + h_cells / 2) * grid.resolution
            obs = StaticObstacle(center_x, center_y, w, h)
            static_obstacles.append(obs)
            grid.mark_rect_as_occupied(x_start, y_start, w_cells, h_cells, const.D_SAFE)
    
    # Place Goal
    goal_pos = grid.find_random_free_cell()
    if goal_pos is None: raise Exception("No space for goal.")
    gx = (goal_pos[0] + 0.5) * grid.resolution
    gy = (goal_pos[1] + 0.5) * grid.resolution
    goal = Goal(gx, gy)
    grid.mark_circle_as_occupied(gx, gy, goal.radius, const.D_SAFE)

    # Place Robot
    robot = None
    for _ in range(100): # Try 100 times to find a valid spot
        robot_pos = grid.find_random_free_cell()
        if robot_pos is not None:
            rx = (robot_pos[0] + 0.5) * grid.resolution
            ry = (robot_pos[1] + 0.5) * grid.resolution
            if np.linalg.norm([rx-gx, ry-gy]) >= const.MIN_START_GOAL_DIST:
                robot = Robot(rx, ry, np.random.uniform(-np.pi, np.pi))
                grid.mark_circle_as_occupied(rx, ry, robot.radius, const.D_SAFE)
                break
    if robot is None: raise Exception("Could not place robot.")
    
    # Place Dynamic Obstacles
    for _ in range(num_dynamic):
        obs_pos = grid.find_random_free_cell()
        if obs_pos is not None:
            ox = (obs_pos[0] + 0.5) * grid.resolution
            oy = (obs_pos[1] + 0.5) * grid.resolution
            dynamic_obstacles.append(DynamicObstacle(ox, oy))
            grid.mark_circle_as_occupied(ox, oy, const.D_OBS_RADIUS, const.D_SAFE)
            
    return robot, goal, static_obstacles, dynamic_obstacles

def check_collision(robot, dynamic_obstacles, static_obstacles):
    """Checks if the robot has collided with the *true* state of any obstacle."""
    # Dynamic obstacles
    for obs in dynamic_obstacles:
        if np.linalg.norm(robot.state[:2] - obs.true_state[:2]) < robot.radius + obs.radius:
            return True, "dynamic"
            
    # Static obstacles
    for obs in static_obstacles:
        closest_x = np.clip(robot.state[0], obs.center[0] - obs.width/2, obs.center[0] + obs.width/2)
        closest_y = np.clip(robot.state[1], obs.center[1] - obs.height/2, obs.center[1] + obs.height/2)
        if np.linalg.norm(robot.state[:2] - np.array([closest_x, closest_y])) < robot.radius:
            return True, "static"
            
    return False, None