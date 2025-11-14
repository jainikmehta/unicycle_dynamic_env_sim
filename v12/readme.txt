# How Version 12 Works

## Key Improvements from v11
- Enhanced uncertainty modeling with adaptive sigma bounds for dynamic obstacles
- Improved CBF constraints using higher-order safety conditions
- Removed slack variables from NMPC formulation for better computational efficiency
- Added detailed logging of uncertainty evolution and safety margins
- Improved visualization of uncertainty bounds and predicted trajectories

## NMPC Controller (nmpc_utils.py)
- Uses model predictive control with 20-step horizon (N = 20) and 0.1s timestep
- Cost function balances path tracking (Q_path = diag[0.5, 0.5]) and control effort (R = diag[0.1, 0.05])
- State vector includes [x, y, theta, v, w] with acceleration control inputs [a, alpha]
- Dynamic constraints enforce unicycle motion model with acceleration-based control
- Higher-order CBF constraints with damping factor (CBF_GAMMA = 0.9) for smoother safety enforcement

## Dynamic Obstacle Handling
- Position measurement noise: 0.15m standard deviation
- Uncertainty growth rate: 0.1 per second
- Heading noise range: ±10° (π/18 radians)
- Adaptive sigma bounds: [0.5, 3.0] for uncertainty scaling
- Predicted covariance evolution tracked for N-step horizon
- Effective safety radius combines physical size, uncertainty, and safety margin

## Robot Parameters
- Maximum linear velocity: 5.0 m/s
- Maximum angular velocity: π/2 rad/s
- Maximum linear acceleration: 5.0 m/s²
- Maximum angular acceleration: π/2 rad/s²
- Robot radius: 1.0m
- Goal acceptance radius: 1.5m
- Safety distance (D_SAFE): 2.0m from all obstacles

## Environment Setup
- 50m × 50m environment with 1.0m grid resolution
- 5 dynamic obstacles (K) with 1.0m radius
- 3 static obstacles (L)
- Minimum start-goal distance: 25.0m
- Comprehensive data logging with obstacle tracking and uncertainty evolution
- Enhanced visualization with uncertainty ellipses and safety bounds
