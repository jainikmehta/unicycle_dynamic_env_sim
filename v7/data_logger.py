import numpy as np
import constants as const

class DataLogger:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self._write_header()

    def _write_header(self):
        self.file.write("--- Simulation Log ---\n")
        self.file.write(f"DT: {const.DT}, N: {const.N}, D_SAFE: {const.D_SAFE}\n")
        if const.NOISY_OBSTACLES:
            self.file.write(f"NOISY_OBSTACLES: True, SIGMA_BOUND: {const.SIGMA_BOUND}\n")
        self.file.write("-" * 20 + "\n")

    def log_timestep(self, t, robot, goal, dynamic_obstacles, static_obstacles):
        self.file.write(f"--- Timestep {t} ---\n")
        
        # Log Robot State
        self.file.write(f"Robot: pos=({robot.state[0]:.2f}, {robot.state[1]:.2f}), theta={robot.state[2]:.2f}\n")
        
        # Log Goal State (only once needed, but logging per step is fine)
        self.file.write(f"Goal: pos=({goal.state[0]:.2f}, {goal.state[1]:.2f})\n")

        # Log Dynamic Obstacles
        self.file.write("Dynamic Obstacles:\n")
        for i, obs in enumerate(dynamic_obstacles):
            log_str = f"  ID {i}: true_pos=({obs.true_state[0]:.2f}, {obs.true_state[1]:.2f})"
            if const.NOISY_OBSTACLES:
                log_str += f", measured_pos=({obs.measured_state[0]:.2f}, {obs.measured_state[1]:.2f})"
                log_str += f", cov_diag=({obs.cov[0,0]:.3f}, {obs.cov[1,1]:.3f})"
            self.file.write(log_str + "\n")

        # Log Static Obstacles
        if t == 0: # Only need to log these once
            self.file.write("Static Obstacles:\n")
            for i, obs in enumerate(static_obstacles):
                self.file.write(f"  ID {i}: center=({obs.center[0]:.2f}, {obs.center[1]:.2f}), size=({obs.width:.2f}, {obs.height:.2f})\n")
        
        self.file.write("\n")

    def close(self):
        self.file.close()