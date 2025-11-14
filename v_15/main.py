"""
main.py

The main entry point for the NMPC simulation.
Run this file to start the simulation.

Example:
    python main.py --seed 123 --log
    python main.py --dynamic_obs 10 --static_obs 5
"""

import numpy as np
import os
import imageio.v2 as imageio
import random
import argparse
import time

# Import our new modules
import constants as const
from simulation_entities import Robot, Goal, DynamicObstacle, StaticObstacle
from env_setup import setup_environment, check_collision
from rrt_star_planner import RRTStar
from nmpc_controller import nmpc_solver, generate_reference_trajectory
from data_logger import DataLogger, DummyLogger
from plotting import plot_simulation_step

def is_path_safe(rrt_path, dynamic_obstacles):
    """
    Checks if the given RRT path is safe against the *current*
    measured positions of dynamic obstacles.
    """
    if rrt_path is None:
        return False
        
    for node in rrt_path:
        for obs in dynamic_obstacles:
            dist = np.linalg.norm(node - obs.measured_state[:2])
            # Check if any path node is inside the safety buffer of a dynamic obs
            if dist < (obs.radius + const.ROBOT_RADIUS + const.D_SAFE):
                return False # Path is unsafe
    return True # Path is safe

def run_simulation(args):
    """
    Main simulation loop.
    """
    # --- Setup ---
    # Set random seed for reproducibility
    seed = args.seed if args.seed is not None else random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)
    print(f"--- Running Simulation with Seed: {seed} ---")
    
    # Setup directories
    frame_dir = f"frames_seed_{seed}"
    if os.path.exists(frame_dir):
        # Clear old frames
        for f in os.listdir(frame_dir): os.remove(os.path.join(frame_dir, f))
    else:
        os.makedirs(frame_dir)

    # Initialize Logger
    if args.log:
        logger = DataLogger(f"sim_log_{seed}.txt")
    else:
        logger = DummyLogger()
        
    # --- Environment Setup ---
    try:
        robot, goal, static_obstacles, dynamic_obstacles = setup_environment(
            num_static=args.static_obs, 
            num_dynamic=args.dynamic_obs
        )
        print(f"Environment setup successful: {args.static_obs} static, {args.dynamic_obs} dynamic.")
    except Exception as e:
        print(f"Failed to setup environment: {e}")
        return

    # --- Initial Path Plan ---
    print("Planning initial RRT* path...")
    rrt = RRTStar(start=robot.state[:2], goal=goal.state,
                  obstacles_static=static_obstacles,
                  obstacles_dynamic=dynamic_obstacles, # FIX: Pass dynamic obs
                  bounds=[0, const.X_LIM, 0, const.Y_LIM])
    rrt_path = rrt.plan()
    if rrt_path is None:
        print("RRT* failed to find an initial path. Halting.")
        return

    # --- Simulation Loop ---
    control_history = []
    h_dyn_history = []
    h_stat_history = []
    last_solution = None # For NMPC warm start (now stores w_opt)
    
    sim_start_time = time.time()
    for t in range(const.MAX_SIM_STEPS):
        print(f"--- Timestep {t} / {const.MAX_SIM_STEPS} ---")
        
        # 1. Update Obstacles and Predict Future
        for obs in dynamic_obstacles:
            obs.update_state(const.DT, static_obstacles, (const.X_LIM, const.Y_LIM))
            obs.predict_future_path(const.N, const.DT)

        # 2. Check for RRT* Replanning
        dist_from_path = np.min(np.linalg.norm(rrt_path - robot.state[:2], axis=1))
        
        # FIX: Replan if path is unsafe OR robot has strayed
        if not is_path_safe(rrt_path, dynamic_obstacles) or dist_from_path > const.PATH_DEVIATION_THRESHOLD:
            if not is_path_safe(rrt_path, dynamic_obstacles):
                print("RRT* path is now unsafe. Replanning...")
            else:
                print("Deviated from path, replanning RRT*...")
                
            rrt = RRTStar(start=robot.state[:2], goal=goal.state,
                          obstacles_static=static_obstacles,
                          obstacles_dynamic=dynamic_obstacles, # FIX: Pass dynamic obs
                          bounds=[0, const.X_LIM, 0, const.Y_LIM])
            new_path = rrt.plan()
            if new_path is None:
                print("RRT* replanning failed. Continuing with old path.")
            else:
                rrt_path = new_path

        # 3. Solve NMPC
        # Generate reference for NMPC
        x_ref = generate_reference_trajectory(robot.state, rrt_path, const.N, const.DT)
        
        # Solve
        # The signature now matches the v10-style solver
        u_optimal, x_predicted, h_dyn, h_stat, last_solution = nmpc_solver(
            robot.state, x_ref, dynamic_obstacles, static_obstacles, last_solution
        )
        
        # 4. Update Robot State
        robot.update_state(u_optimal, const.DT)
        
        # 5. Check for Failures
        collided, obs_type = check_collision(robot, dynamic_obstacles, static_obstacles)
        if collided:
            print(f"!!! Collision detected with a {obs_type} obstacle at timestep {t}. Halting. !!!")
            break
            
        if np.linalg.norm(robot.state[:2] - goal.state) < goal.radius:
            print(f"*** Goal reached at timestep {t}! ***")
            break
            
        # 6. Log Data
        logger.log_timestep(t, robot, goal, dynamic_obstacles)
        control_history.append(u_optimal)
        h_dyn_history.append(h_dyn)
        h_stat_history.append(h_stat)

        # 7. Plotting
        frame_path = os.path.join(frame_dir, f"frame_{t:04d}.png")
        plot_simulation_step(robot, goal, static_obstacles, dynamic_obstacles,
                             x_predicted, rrt_path, control_history,
                             h_dyn_history, h_stat_history, t, save_path=frame_path)
    
    # --- End of Simulation ---
    sim_end_time = time.time()
    total_time = sim_end_time - sim_start_time
    avg_step_time = total_time / (t + 1)
    print(f"\n--- Simulation Finished ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per step: {avg_step_time:.3f}s")
    
    logger.close()
    
    # Generate GIF
    print("Generating animation...")
    frames = []
    for i in range(t + 1):
        frame_file = os.path.join(frame_dir, f"frame_{i:04d}.png")
        if os.path.exists(frame_file):
            frames.append(imageio.imread(frame_file))
            
    if frames:
        anim_file = f"simulation_{seed}.gif"
        imageio.mimsave(anim_file, frames, fps=10)
        print(f"Animation saved as {anim_file}")
    else:
        print("No frames found to generate animation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NMPC simulation with dynamic obstacles.")
    parser.add_argument('--seed', type=int, help='Seed for random number generator.')
    parser.add_argument('--log', action='store_true', help='Enable data logging.')
    parser.add_argument('--static_obs', type=int, default=3, help='Number of static obstacles.')
    parser.add_argument('--dynamic_obs', type=int, default=5, help='Number of dynamic obstacles.')
    
    args = parser.parse_args()
    run_simulation(args)