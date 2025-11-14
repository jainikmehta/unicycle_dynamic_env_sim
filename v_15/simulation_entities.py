"""
simulation_entities.py

Contains the classes for all actors in the simulation.
"""

import numpy as np
import constants as const

class Robot:
    """Represents the unicycle robot with second-order dynamics."""
    def __init__(self, x, y, theta):
        # state: [x, y, theta, v, w]
        self.state = np.array([x, y, theta, 0.1, 0.0]) 
        self.radius = const.ROBOT_RADIUS
        self.trajectory = [self.state.copy()] # History of states

    def update_state(self, u, dt):
        """
        Updates the robot's state using the second-order unicycle model.
        u = [linear_acceleration, angular_acceleration]
        """
        x, y, theta, v, w = self.state
        a, alpha = u
        
        # Update state
        self.state[0] = x + dt * v * np.cos(theta)
        self.state[1] = y + dt * v * np.sin(theta)
        self.state[2] = theta + dt * w
        self.state[3] = v + dt * a
        self.state[4] = w + dt * alpha

        # Clamp velocities
        self.state[3] = np.clip(self.state[3], 0, const.MAX_LINEAR_VEL)
        self.state[4] = np.clip(self.state[4], -const.MAX_ANGULAR_VEL, const.MAX_ANGULAR_VEL)
        
        self.trajectory.append(self.state.copy())

class Goal:
    """Represents the goal location."""
    def __init__(self, x, y):
        self.state = np.array([x, y])
        self.radius = const.GOAL_RADIUS

class StaticObstacle:
    """Represents a static rectangular obstacle."""
    def __init__(self, x, y, w, h):
        self.center = np.array([x, y])
        self.width = w
        self.height = h

class DynamicObstacle:
    """Represents a dynamic obstacle with uncertainty."""
    def __init__(self, x, y):
        # The "ground truth" state
        self.true_state = np.array([x, y, np.random.uniform(-np.pi, np.pi)])
        self.velocity = np.random.uniform(0.5, 2.0)
        self.radius = const.D_OBS_RADIUS
        
        # The "sensed" state (what the robot knows)
        self.measured_state = self.true_state.copy()
        self.cov = np.diag([const.OBSTACLE_POS_NOISE_STD**2] * 2)
        
        # NMPC predictions
        self.predicted_path = np.zeros((2, const.N + 1))
        self.predicted_cov = [np.zeros((2,2)) for _ in range(const.N + 1)]
        self.sigma_bounds = np.zeros(const.N + 1)
        
        # History for plotting
        self.true_trajectory = [self.true_state.copy()]
        self.measure() # Initial measurement
        self.measured_trajectory = [self.measured_state.copy()]

    def measure(self):
        """Simulate noisy position measurement."""
        noise = np.random.normal(0, const.OBSTACLE_POS_NOISE_STD, 2)
        self.measured_state[:2] = self.true_state[:2] + noise
        # Assume heading is observable (or estimated well)
        self.measured_state[2] = self.true_state[2] 
        self.cov = np.diag([const.OBSTACLE_POS_NOISE_STD**2] * 2)

    def update_state(self, dt, static_obstacles, world_bounds):
        """Update the obstacle's true state (the "truth" model)."""
        next_x = self.true_state[0] + dt * self.velocity * np.cos(self.true_state[2])
        next_y = self.true_state[1] + dt * self.velocity * np.sin(self.true_state[2])
        
        # Collision checking (walls and static obstacles)
        collided = False
        if not (self.radius < next_x < world_bounds[0]-self.radius and 
                self.radius < next_y < world_bounds[1]-self.radius):
            collided = True
            
        for obs in static_obstacles:
            closest_x = np.clip(next_x, obs.center[0]-obs.width/2, obs.center[0]+obs.width/2)
            closest_y = np.clip(next_y, obs.center[1]-obs.height/2, obs.center[1]+obs.height/2)
            if np.linalg.norm([next_x - closest_x, next_y - closest_y]) < self.radius:
                collided = True
                break
        
        if collided:
            # Turn around and add noise
            self.true_state[2] += np.pi + np.random.uniform(-const.HEADING_NOISE_RANGE, const.HEADING_NOISE_RANGE)
        else:
            self.true_state[0], self.true_state[1] = next_x, next_y
        
        # Generate new measurement
        self.measure()
        
        # Store history
        self.true_trajectory.append(self.true_state.copy())
        self.measured_trajectory.append(self.measured_state.copy())

    def predict_future_path(self, N, dt):
        """
        Predict future path and covariance based on the LATEST measurement.
        This is what the NMPC controller will use.
        """
        # Start prediction from the latest measurement
        self.predicted_path[:, 0] = self.measured_state[:2]
        self.predicted_cov[0] = self.cov
        
        # Generate new random sigma bounds for this prediction horizon
        self.sigma_bounds = np.random.uniform(const.SIGMA_BOUND_LOWER, const.SIGMA_BOUND_UPPER, N + 1)
        self.sigma_bounds[0] = const.SIGMA_BOUND_UPPER # Be conservative at k=0
        
        v_obs_x = self.velocity * np.cos(self.measured_state[2])
        v_obs_y = self.velocity * np.sin(self.measured_state[2])
        
        for i in range(1, N + 1):
            # Predict mean path (constant velocity model)
            self.predicted_path[0, i] = self.predicted_path[0, i-1] + dt * v_obs_x
            self.predicted_path[1, i] = self.predicted_path[1, i-1] + dt * v_obs_y
            
            # Propagate uncertainty (simple growth model)
            added_variance = const.UNCERTAINTY_GROWTH_RATE * (i * dt)
            self.predicted_cov[i] = self.cov + np.diag([added_variance, added_variance])