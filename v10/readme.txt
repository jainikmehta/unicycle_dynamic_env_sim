# Key Feature

- Provides Intermediate Goals
- RRT update the plan only when current plan becomes unsafe.

# NMPC Constraints (nmpc_utils.py)

- Initial state constraints enforce that the optimization starts from the current robot state
- Dynamic constraints ensure robot motion follows unicycle dynamics (x, y, theta, v, w) with acceleration inputs (a, alpha)
- State bounds limit robot position (within map), velocity (0 to MAX_LINEAR_VEL), and angular velocity (-MAX_ANGULAR_VEL to MAX_ANGULAR_VEL)
- Control bounds restrict linear acceleration (-MAX_LINEAR_ACCEL to MAX_LINEAR_ACCEL) and angular acceleration (-MAX_ANGULAR_ACCEL to MAX_ANGULAR_ACCEL)
- Safety constraints use Control Barrier Functions (CBFs) that maintain minimum distances from obstacles, considering their uncertainties and safety margins

# RRT* Path Planning Updates (rrt_star_planner.py)

- RRT* replanning triggers when the robot's current path becomes unsafe (h-value drops below H_RECOVERY_THRESHOLD)
- Each planning call uses 1500 iterations to find an optimal path through the current obstacle configuration
- The planner generates waypoints considering both static obstacles (with safety margins) and current positions of dynamic obstacles
- Path costs are continuously optimized through rewiring, improving path quality during the 1500 iterations
- After finding a path, intermediate goals are selected along it using INTERMEDIATE_GOAL_LOOKAHEAD_DISTANCE (8.0m) to guide the NMPC
