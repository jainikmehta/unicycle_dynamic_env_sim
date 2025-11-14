"""
data_logger.py

Provides a logger to save simulation data in the requested format.
Includes a DummyLogger to disable logging cleanly.
"""

import numpy as np
import constants as const

class DataLogger:
    """Logs simulation data to a file."""
    def __init__(self, filename):
        try:
            self.file = open(filename, 'w')
            print(f"Data logger initialized, writing to {filename}")
        except IOError as e:
            print(f"Failed to open log file: {e}")
            self.file = None

    def log_timestep(self, t, robot, goal, dynamic_obstacles):
        """Write a concise snapshot for time t."""
        if self.file is None:
            return
            
        self.file.write(f"For t = {t}:\n")
        
        # Log Robot and Goal State
        r = robot.state
        self.file.write(f"robot_true_state: ({r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f}, {r[3]:.3f}, {r[4]:.3f}, 0.00)\n")
        g = goal.state
        self.file.write(f"goal_state: ({g[0]:.2f}, {g[1]:.2f}, 0.00, 0.000, 0.000, 0.00)\n")

        # Sort dynamic obstacles by distance to the robot for consistent ordering
        sorted_obstacles = sorted(
            dynamic_obstacles, 
            key=lambda obs: np.linalg.norm(robot.state[:2] - obs.measured_state[:2])
        )

        # Log Dynamic Obstacles and their horizon predictions
        for i, obs in enumerate(sorted_obstacles):
            m = obs.measured_state
            unc_x = np.sqrt(obs.cov[0, 0])
            unc_y = np.sqrt(obs.cov[1, 1])
            sig_b = obs.sigma_bounds[0]
            self.file.write(
                f"obstacle_{i+1}_measured_state: ({m[0]:.2f}, {m[1]:.2f}, {m[2]:.2f}, {unc_x:.3f}, {unc_y:.3f}, {sig_b:.2f})\n"
            )

            for n in range(1, const.N + 1):
                mean_x, mean_y = obs.predicted_path[:, n]
                theta = obs.measured_state[2] # Assumed constant heading in prediction
                unc_x_n = np.sqrt(obs.predicted_cov[n][0, 0])
                unc_y_n = np.sqrt(obs.predicted_cov[n][1, 1])
                sig_b_n = obs.sigma_bounds[n]
                self.file.write(
                    f"obstacle_{i+1}_predicted_state_{n}: ({mean_x:.2f}, {mean_y:.2f}, {theta:.2f}, {unc_x_n:.3f}, {unc_y_n:.3f}, {sig_b_n:.2f})\n"
                )

        self.file.write("\n")
        self.file.flush()

    def close(self):
        """Close the underlying file handle."""
        if self.file:
            self.file.close()
            print("Data logger closed.")

class DummyLogger:
    """A logger that does nothing. Used when logging is disabled."""
    def log_timestep(self, t, robot, goal, dynamic_obstacles):
        pass # Do nothing
    def close(self):
        pass # Do nothing