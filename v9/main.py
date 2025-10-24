import numpy as np
import os
import imageio.v2 as imageio
import random
import argparse
from utils import Grid
from plotting import plot_environment
from nmpc_utils import nmpc_solver
from rrt_star_planner import RRTStar
from data_logger import DataLogger
import constants as const

# --- Simulation Classes ---
class Robot:
    def __init__(self, x, y, theta):
        self.state = np.array([x, y, theta, 0.0, 0.0]) # state: [x, y, theta, v, w]
        self.radius = const.ROBOT_RADIUS
        self.trajectory = [self.state.copy()]

    def update_state(self, u, dt):
        # u = [linear_acceleration, angular_acceleration]
        v = self.state[3]
        w = self.state[4]
        
        # Update state using second-order dynamics
        self.state[0] += dt * v * np.cos(self.state[2])
        self.state[1] += dt * v * np.sin(self.state[2])
        self.state[2] += dt * w
        self.state[3] += dt * u[0]
        self.state[4] += dt * u[1]

        # Clamp velocities to their maximum values
        self.state[3] = np.clip(self.state[3], 0, const.MAX_LINEAR_VEL)
        self.state[4] = np.clip(self.state[4], -const.MAX_ANGULAR_VEL, const.MAX_ANGULAR_VEL)

        self.trajectory.append(self.state.copy())

class Goal:
    def __init__(self, x, y):
        self.state = np.array([x, y])
        self.radius = const.GOAL_RADIUS

class DynamicObstacle:
    def __init__(self, x, y):
        self.true_state = np.array([x, y, np.random.uniform(-np.pi, np.pi)])
        self.velocity = np.random.uniform(0.5, 2.0)
        self.radius = const.D_OBS_R
        
        # Initialize uncertainty variables
        self.measured_state = self.true_state.copy()
        self.cov = np.diag([const.OBSTACLE_POS_NOISE_STD**2, const.OBSTACLE_POS_NOISE_STD**2])
        self.predicted_path = np.zeros((2, const.N + 1))
        self.predicted_cov = [np.zeros((2,2)) for _ in range(const.N + 1)]
        self.sigma_bounds = np.zeros(const.N + 1)  # Store sigma bounds for each prediction step

        # History for plotting
        self.true_trajectory = [self.true_state.copy()]
        self.measure() # Initial measurement
        self.measured_trajectory = [self.measured_state.copy()]

    def measure(self):
        # Simulate noisy measurement
        noise = np.random.normal(0, const.OBSTACLE_POS_NOISE_STD, 2)
        self.measured_state[:2] = self.true_state[:2] + noise
        self.measured_state[2] = self.true_state[2] # Assume perfect heading measurement
        self.cov = np.diag([const.OBSTACLE_POS_NOISE_STD**2, const.OBSTACLE_POS_NOISE_STD**2])

    def update_state(self, dt, static_obstacles, x_lim, y_lim):
        # Update true state
        next_x = self.true_state[0] + dt * self.velocity * np.cos(self.true_state[2])
        next_y = self.true_state[1] + dt * self.velocity * np.sin(self.true_state[2])
        
        # Wall and static obstacle collision logic for true state
        if not (self.radius < next_x < x_lim-self.radius and self.radius < next_y < y_lim-self.radius):
            self.true_state[2] += np.pi + np.random.uniform(-const.HEADING_NOISE_RANGE, const.HEADING_NOISE_RANGE)
            return
        for obs in static_obstacles:
            closest_x=np.clip(next_x, obs.center[0]-obs.width/2, obs.center[0]+obs.width/2)
            closest_y=np.clip(next_y, obs.center[1]-obs.height/2, obs.center[1]+obs.height/2)
            if np.sqrt((next_x-closest_x)**2 + (next_y-closest_y)**2) < self.radius:
                self.true_state[2] += np.pi + np.random.uniform(-const.HEADING_NOISE_RANGE, const.HEADING_NOISE_RANGE)
                return
        self.true_state[0], self.true_state[1] = next_x, next_y
        
        # Generate new measurement
        self.measure()
        
        # Store history
        self.true_trajectory.append(self.true_state.copy())
        self.measured_trajectory.append(self.measured_state.copy())

    def predict_future_path(self):
        # Predict path and covariance based on the LATEST measurement
        self.predicted_path[:, 0] = self.measured_state[:2]
        self.predicted_cov[0] = self.cov
        self.sigma_bounds = np.random.uniform(const.SIGMA_BOUND_LOWER, const.SIGMA_BOUND_UPPER, const.N + 1)
        
        for i in range(1, const.N + 1):
            self.predicted_path[0, i] = self.predicted_path[0, i-1] + const.DT * self.velocity * np.cos(self.measured_state[2])
            self.predicted_path[1, i] = self.predicted_path[1, i-1] + const.DT * self.velocity * np.sin(self.measured_state[2])
            
            # Propagate uncertainty
            added_variance = const.UNCERTAINTY_GROWTH_RATE * (i * const.DT)
            self.predicted_cov[i] = self.cov + np.diag([added_variance, added_variance])

class StaticObstacle:
    def __init__(self, x, y, w, h):
        self.center = np.array([x, y]); self.width = w; self.height = h

# --- Environment Setup (Unchanged) ---
def setup_environment(grid):
    static_obstacles = []
    for _ in range(const.L):
        w,h = np.random.uniform(5.0, 10.0), np.random.uniform(5.0, 10.0)
        w_cells, h_cells = int(np.ceil(w/grid.resolution)), int(np.ceil(h/grid.resolution))
        pos = grid.find_free_rect_space(w_cells, h_cells)
        if pos:
            x_start,y_start=pos; center_x=(x_start+w_cells/2)*grid.resolution; center_y=(y_start+h_cells/2)*grid.resolution
            obstacle = StaticObstacle(center_x, center_y, w, h); static_obstacles.append(obstacle)
            grid.mark_rect_as_occupied(x_start, y_start, w_cells, h_cells, const.D_SAFE)
    goal_pos = grid.find_random_free_cell()
    if goal_pos is None: raise Exception("No space for goal.")
    gx,gy=(goal_pos+0.5)*grid.resolution; goal=Goal(gx,gy)
    grid.mark_circle_as_occupied(gx,gy,goal.radius,const.D_SAFE)
    robot=None
    for _ in range(100):
        robot_pos = grid.find_random_free_cell()
        if robot_pos is not None:
            rx,ry=(robot_pos+0.5)*grid.resolution
            if np.linalg.norm([rx-gx,ry-gy]) >= const.MIN_START_GOAL_DIST:
                robot=Robot(rx,ry,np.random.uniform(-np.pi,np.pi))
                grid.mark_circle_as_occupied(rx,ry,robot.radius,const.D_SAFE); break
    if robot is None: raise Exception("Could not place robot.")
    dynamic_obstacles = []
    for _ in range(const.K):
        obs_pos = grid.find_random_free_cell()
        if obs_pos is not None:
            ox,oy=(obs_pos+0.5)*grid.resolution; dynamic_obstacles.append(DynamicObstacle(ox,oy))
            grid.mark_circle_as_occupied(ox,oy,const.D_OBS_R,const.D_SAFE)
    return robot, goal, static_obstacles, dynamic_obstacles

def generate_reference_trajectory(robot_state, path):
    current_pos = robot_state[:2]
    distances = np.linalg.norm(path - current_pos, axis=1)
    closest_idx = np.argmin(distances)
    ref_path = np.zeros((2, const.N + 1))
    for i in range(const.N + 1):
        idx = min(closest_idx + i, len(path) - 1)
        ref_path[:, i] = path[idx]
    return ref_path
    
def check_collision(robot, dynamic_obstacles, static_obstacles):
    # Check collision with dynamic obstacles
    for obs in dynamic_obstacles:
        if np.linalg.norm(robot.state[:2] - obs.true_state[:2]) < robot.radius + obs.radius:
            return True, "dynamic"
            
    # Check collision with static obstacles
    for obs in static_obstacles:
        closest_x = np.clip(robot.state[0], obs.center[0] - obs.width/2, obs.center[0] + obs.width/2)
        closest_y = np.clip(robot.state[1], obs.center[1] - obs.height/2, obs.center[1] + obs.height/2)
        if np.linalg.norm(robot.state[:2] - np.array([closest_x, closest_y])) < robot.radius:
            return True, "static"
            
    return False, None

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
        
        random_suffix = random.randint(1000, 9999)
        logger = DataLogger(f"log_{seed}_{random_suffix}.txt")

        control_history = []; h_dyn_history = []; h_stat_history = []; rrt_path = None
        
        for t in range(const.MAX_SIM_STEPS):
            print(f"--- Timestep {t} ---")

            # 1. Update obstacles and predict their future paths + uncertainty
            for obs in dynamic_obstacles:
                obs.update_state(const.DT, static_obstacles, const.X_LIM, const.Y_LIM)
                obs.predict_future_path()

            # 2. Plan with RRT*
            print("Planning with RRT*...")
            rrt = RRTStar(start=robot.state[:2], goal=goal.state,
                          obstacles_static=static_obstacles,
                          bounds=[0, const.X_LIM, 0, const.Y_LIM], safe_dist=const.ROBOT_RADIUS + const.D_SAFE)
            new_path = rrt.plan()
            if new_path is not None: rrt_path = new_path
            if rrt_path is None: print("RRT* failed to find a path, stopping."); break
            
            # 3. Generate reference for NMPC and solve
            x_ref = generate_reference_trajectory(robot.state, rrt_path)
            u_optimal, x_predicted, h_dyn, h_stat = nmpc_solver(robot.state, x_ref, dynamic_obstacles, static_obstacles)

            if u_optimal is None:
                print("NMPC solver failed to find a solution. Halting simulation.")
                break
            
            robot.update_state(u_optimal, const.DT)
            
            # 4. Check for collisions
            collided, obs_type = check_collision(robot, dynamic_obstacles, static_obstacles)
            if collided:
                print(f"Collision detected with a {obs_type} obstacle at timestep {t}. Halting simulation.")
                break

            # 5. Log data for this timestep
            logger.log_timestep(t, robot, goal, dynamic_obstacles)
            control_history.append(u_optimal); h_dyn_history.append(min(h_dyn) if h_dyn else 0); h_stat_history.append(min(h_stat) if h_stat else 0)

            # 6. Plotting
            frame_path = f"frames/frame_{t:03d}.png"
            plot_environment(robot, goal, static_obstacles, dynamic_obstacles, 
                             x_predicted, rrt_path, control_history, h_dyn_history, h_stat_history, save_path=frame_path)
            
            if np.linalg.norm(robot.state[:2] - goal.state) < goal.radius:
                print(f"Goal reached at timestep {t}!"); break
        
        logger.close()
        final_timestep = t
        print(f"\n--- Generating Animation ---"); print(f"Simulation Seed: {seed}")
        frames = [imageio.imread(f) for i in range(final_timestep + 1) if os.path.exists(f:=(f"frames/frame_{i:03d}.png"))]
        anim_file = f"simulation_{seed}_{random_suffix}.gif"; imageio.mimsave(anim_file, frames, fps=10)
        print(f"Animation saved as {anim_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

