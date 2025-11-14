# How Version 11 Works

## Key Improvements from v10
- Path following uses dynamic trajectory generation instead of intermediate goals
- RRT* plan updates are now condition-based (only when path deviation > PATH_DEVIATION_THRESHOLD)
- Added velocity maximization term in NMPC cost function (Q_vel parameter)
- More efficient CBF constraints using squared distances for safety checks
- Simplified reference path generation using closest point on RRT* path

## NMPC Controller (nmpc_utils.py)
- Uses model predictive control with 20-step horizon (N = 20) and 0.1s timestep
- Cost function balances path tracking (Q_path), forward velocity (Q_vel), and control effort (R)
- State vector includes [x, y, theta, v, w] with acceleration control inputs [a, alpha]
- Dynamic constraints enforce unicycle motion model with acceleration-based control
- Safety uses squared-distance CBFs for both static and dynamic obstacles (more numerically stable)

## RRT* Path Planning (rrt_star_planner.py)
- Only replans when robot deviates > 5.0m from current path (PATH_DEVIATION_THRESHOLD)
- Uses 1500 iterations per planning cycle (RRT_MAX_ITER)
- Step size of 2.0m (RRT_STEP_SIZE) with 4.0m search radius for rewiring
- 10% goal biasing (RRT_GOAL_SAMPLE_RATE) for efficient goal-directed search
- Considers both static and dynamic obstacles with D_SAFE buffer

## Dynamic Obstacles
- Constant velocity motion with noisy heading changes
- Position measurement noise: 0.15m std dev
- Uncertainty grows at 0.1 per second
- Heading changes have ±10° noise
- Sigma bounds for uncertainty scaling: [0.5, 3.0]

## Environment and Robot Parameters
- 50m × 50m environment
- Robot: 1.0m radius, max 5.0 m/s speed
- Goal radius: 1.5m
- Minimum start-goal distance: 25.0m
- Obstacles: 3 static rectangles, 5 dynamic circles (1.0m radius)
- Safety distance (D_SAFE): 1.0m from all obstacles
