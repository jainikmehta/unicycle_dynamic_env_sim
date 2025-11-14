import numpy as np
import constants as const


class DataLogger:
    """Simple file-based logger for simulation timesteps.

    The logger writes robot/goal states and each dynamic obstacle's
    measured and predicted states (including uncertainty) in a readable
    line-based format used for post-run analysis.
    """

    def __init__(self, filename):
        # Open log file (overwrites existing file)
        self.file = open(filename, 'w')

    def log_timestep(self, t, robot, goal, dynamic_obstacles):
        """Write a concise snapshot for time t.

        Logs the robot and goal states first, then each dynamic obstacle
        sorted by proximity to the robot. Predicted states for the
        NMPC horizon are written with their covariance-derived
        uncertainty values.
        """
        self.file.write(f"For t = {t}:\n")

        # Log Robot and Goal State (robot state may include velocities)
        self.file.write(f"robot_true_state: ({robot.state[0]:.2f}, {robot.state[1]:.2f}, {robot.state[2]:.2f}, 0.000, 0.000, 0.00)\n")
        self.file.write(f"goal_state: ({goal.state[0]:.2f}, {goal.state[1]:.2f}, 0.00, 0.000, 0.000, 0.00)\n")

        # Sort dynamic obstacles by distance to the robot for consistent ordering
        sorted_obstacles = sorted(dynamic_obstacles, key=lambda obs: np.linalg.norm(robot.state[:2] - obs.measured_state[:2]))

        # Log Dynamic Obstacles and their horizon predictions
        for i, obs in enumerate(sorted_obstacles):
            uncertainty_x = np.sqrt(obs.cov[0, 0])
            uncertainty_y = np.sqrt(obs.cov[1, 1])
            sigma_bound = obs.sigma_bounds[0]
            self.file.write(
                f"obstacle_{i+1}_measured_state: ({obs.measured_state[0]:.2f}, {obs.measured_state[1]:.2f}, {obs.measured_state[2]:.2f}, {uncertainty_x:.3f}, {uncertainty_y:.3f}, {sigma_bound:.2f})\n"
            )

            # Predicted states for 1..N (skip 0 - current measurement already logged)
            for n in range(1, const.N + 1):
                mean_x, mean_y = obs.predicted_path[:, n]
                theta = obs.measured_state[2]
                uncertainty_x = np.sqrt(obs.predicted_cov[n][0, 0])
                uncertainty_y = np.sqrt(obs.predicted_cov[n][1, 1])
                sigma_bound = obs.sigma_bounds[n]
                self.file.write(
                    f"obstacle_{i+1}_predicted_state_{n}: ({mean_x:.2f}, {mean_y:.2f}, {theta:.2f}, {uncertainty_x:.3f}, {uncertainty_y:.3f}, {sigma_bound:.2f})\n"
                )

        self.file.write("\n")

    def close(self):
        """Close the underlying file handle."""
        self.file.close()