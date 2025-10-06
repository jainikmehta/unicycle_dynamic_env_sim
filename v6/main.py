import numpy as np
import os
import imageio.v2 as imageio
import random
import argparse
from utils import Grid
from plotting import plot_environment
from nmpc_utils import nmpc_solver
from rrt_star_planner import RRTStar
import constants as const

# --- Simulation Classes (Same as before) ---
class Robot:
    def __init__(self, x, y, theta):
        self.state = np.array([x, y, theta])
        self.radius = const.ROBOT_RADIUS
        self.trajectory = [self.state.copy()]
    def update_state(self, u, dt):
        self.state[0] += dt * u[0] * np.cos(self.state[2])
        self.state[1] += dt * u[0] * np.sin(self.state[2])
        self.state[2] += dt * u[1]
        self.trajectory.append(self.state.copy())

class Goal:
    def __init__(self, x, y):
        self.state = np.array([x, y])
        self.radius = const.GOAL_RADIUS

class DynamicObstacle:
    def __init__(self, x, y):
        self.state = np.array([x, y, np.random.uniform(-np.pi, np.pi)])
        self.velocity = np.random.uniform(0.5, 2.0)
        self.radius = const.D_OBS_R
    def update_state(self, dt, static_obstacles, x_lim, y_lim):
        next_x = self.state[0] + dt * self.velocity * np.cos(self.state[2])
        next_y = self.state[1] + dt * self.velocity * np.sin(self.state[2])
        if not (self.radius < next_x < x_lim - self.radius and self.radius < next_y < y_lim - self.radius):
            self.state[2] += np.pi; return
        for obs in static_obstacles:
            closest_x = np.clip(next_x, obs.center[0] - obs.width/2, obs.center[0] + obs.width/2)
            closest_y = np.clip(next_y, obs.center[1] - obs.height/2, obs.center[1] + obs.height/2)
            if np.sqrt((next_x - closest_x)**2 + (next_y - closest_y)**2) < self.radius:
                self.state[2] += np.pi; return
        self.state[0], self.state[1] = next_x, next_y

class StaticObstacle:
    def __init__(self, x, y, w, h):
        self.center = np.array([x, y])
        self.width = w; self.height = h

# --- Environment Setup (Same as before) ---
def setup_environment(grid):
    # This function remains unchanged
    static_obstacles = []
    for _ in range(const.L):
        w, h = np.random.uniform(5.0, 10.0), np.random.uniform(5.0, 10.0)
        w_cells, h_cells = int(np.ceil(w/grid.resolution)), int(np.ceil(h/grid.resolution))
        pos = grid.find_free_rect_space(w_cells, h_cells)
        if pos:
            x_start, y_start = pos
            center_x = (x_start + w_cells/2)*grid.resolution
            center_y = (y_start + h_cells/2)*grid.resolution
            obstacle = StaticObstacle(center_x, center_y, w, h)
            static_obstacles.append(obstacle)
            grid.mark_rect_as_occupied(x_start, y_start, w_cells, h_cells, const.D_SAFE)
    goal_pos = grid.find_random_free_cell()
    if goal_pos is None: raise Exception("No space for goal.")
    gx, gy = (goal_pos+0.5)*grid.resolution; goal = Goal(gx,gy)
    grid.mark_circle_as_occupied(gx, gy, goal.radius, const.D_SAFE)
    robot = None
    for _ in range(100):
        robot_pos = grid.find_random_free_cell()
        if robot_pos is not None:
            rx,ry = (robot_pos+0.5)*grid.resolution
            if np.linalg.norm([rx-gx, ry-gy]) >= const.MIN_START_GOAL_DIST:
                robot = Robot(rx,ry,np.random.uniform(-np.pi,np.pi))
                grid.mark_circle_as_occupied(rx,ry,robot.radius,const.D_SAFE); break
    if robot is None: raise Exception("Could not place robot.")
    dynamic_obstacles = []
    for _ in range(const.K):
        obs_pos = grid.find_random_free_cell()
        if obs_pos is not None:
            ox,oy = (obs_pos+0.5)*grid.resolution
            dynamic_obstacles.append(DynamicObstacle(ox,oy))
            grid.mark_circle_as_occupied(ox,oy,const.D_OBS_R,const.D_SAFE)
    return robot, goal, static_obstacles, dynamic_obstacles

def generate_reference_trajectory(robot_state, path):
    # Finds the closest point on the path and returns the next N points as a reference
    current_pos = robot_state[:2]
    distances = np.linalg.norm(path - current_pos, axis=1)
    closest_idx = np.argmin(distances)
    
    ref_path = np.zeros((2, const.N + 1))
    for i in range(const.N + 1):
        idx = min(closest_idx + i, len(path) - 1)
        ref_path[:, i] = path[idx]
    return ref_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NMPC simulation.")
    parser.add_argument('--seed', type=int, help='Seed for random number generator.')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 10000)
    random.seed(seed); np.random.seed(seed)
    
    print("--- Setting up Episode ---")
    if not os.path.exists('frames'): os.makedirs('frames')
    
    try:
        grid = Grid(const.X_LIM, const.Y_LIM, const.GRID_RESOLUTION, border_width=2)
        robot, goal, static_obstacles, dynamic_obstacles = setup_environment(grid)
        print("Environment setup successful.")

        control_history = []
        rrt_path = None
        
        for t in range(const.MAX_SIM_STEPS):
            print(f"--- Timestep {t} ---")

            # 1. Plan with RRT*
            print("Planning with RRT*...")
            rrt = RRTStar(start=robot.state[:2], goal=goal.state,
                          obstacles_static=static_obstacles, obstacles_dynamic=dynamic_obstacles,
                          bounds=[0, const.X_LIM, 0, const.Y_LIM], safe_dist=const.ROBOT_RADIUS + const.D_SAFE)
            new_path = rrt.plan()
            if new_path is not None:
                rrt_path = new_path
            
            if rrt_path is None:
                print("RRT* failed to find a path, stopping.")
                break
            
            # 2. Generate reference trajectory for NMPC
            x_ref = generate_reference_trajectory(robot.state, rrt_path)

            # 3. Solve NMPC
            u_optimal, x_predicted = nmpc_solver(robot.state, x_ref, dynamic_obstacles, static_obstacles)
            robot.update_state(u_optimal, const.DT)
            
            for obs in dynamic_obstacles:
                obs.update_state(const.DT, static_obstacles, const.X_LIM, const.Y_LIM)
            
            control_history.append(u_optimal)

            # 4. Plotting
            frame_path = f"frames/frame_{t:03d}.png"
            plot_environment(robot, goal, static_obstacles, dynamic_obstacles, 
                             x_predicted, rrt_path, control_history, save_path=frame_path)
            
            if np.linalg.norm(robot.state[:2] - goal.state) < goal.radius:
                print(f"Goal reached at timestep {t}!"); break
        
        # Animation generation
        final_timestep = t
        print(f"\n--- Generating Animation ---")
        print(f"Simulation Seed: {seed}")
        frames = [imageio.imread(f) for i in range(final_timestep + 1) if os.path.exists(f:=(f"frames/frame_{i:03d}.png"))]
        anim_file = f"simulation_{seed}_{random.randint(1000,9999)}.gif"
        imageio.mimsave(anim_file, frames, fps=10)
        print(f"Animation saved as {anim_file}")

    except Exception as e:
        print(f"An error occurred: {e}")