# How Version 13 Works

## Overview
Version 13 is a hybrid of v11 and v10. It utilizes the robust path planning and simulation structure of v11 but reverts to the Higher-Order Control Barrier Function (HO-CBF) logic from v10 for obstacle avoidance constraints. Additionally, path tracking costs have been significantly increased to enforce stricter adherence to the RRT* plan.

## Key Changes in v13
1. **NMPC Logic (nmpc_utils.py):** - Replaced v11's distance-squared safety constraints with v10's Higher-Order CBF formulation.
   - Constraints now explicitly account for the time derivative of the barrier function (`dh/dt + alpha * h >= 0`), using the obstacle's velocity to predict safety evolution.
2. **Tighter Path Following (constants.py):**
   - Increased `Q_path` weight from 0.5 to 10.0. This heavily penalizes any deviation from the RRT* reference path, ensuring the robot prioritizes the global plan over local cost minimization.
3. **Removed Intermediate Goals:**
   - Uses v11's direct path reference generation instead of v10's intermediate goal heuristics.

## Simulation Components
- **Planner:** RRT* (from v11) with `1500` iterations.
- **Controller:** NMPC with `20` step horizon, HO-CBF safety constraints, and high path tracking penalties.
- **Obstacles:** - Dynamic: Constant velocity with noise, uncertainty modeled via sigma bounds.
  - Static: Rectangles with safety buffers.