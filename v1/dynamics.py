import casadi as cas
import numpy as np

# Define the dynamics function (discrete time) - NO CHANGE
def unicycle_dynamics(x, u, dt):
    return cas.vertcat(
        x[0] + dt * u[0] * cas.cos(x[2]),  # x_{k+1}
        x[1] + dt * u[0] * cas.sin(x[2]),  # y_{k+1}
        x[2] + dt * u[1]                  # θ_{k+1}
    )

# Define the dynamics function (discrete time) - NO CHANGE
def obstacle_dyanmics(x, u, dt, heading_noise = 0.0):
    if heading_noise > 0:
        x[2] += np.random.uniform(0, heading_noise)  # Add noise to heading
    return cas.vertcat(
        x[0] + dt * u[0] * cas.cos(x[2]),  # x_{k+1}
        x[1] + dt * u[0] * cas.sin(x[2]),  # y_{k+1}
        x[2] + dt * u[1]                  # θ_{k+1}
    )