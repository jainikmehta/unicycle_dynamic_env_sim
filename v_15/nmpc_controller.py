"""
nmpc_controller.py

Defines the NMPC optimization problem using CasADi's `nlpsol` interface.
Implements a robust, discrete-time First-Order CBF without slack variables.
"""

import casadi as ca
import numpy as np
import constants as const

def generate_reference_trajectory(robot_state, rrt_path, N, dt):
    """
    Generates a reference trajectory for the NMPC horizon
    based on the robot's current position on the RRT* path.
    """
    ref_path = np.zeros((2, N + 1))
    current_pos = robot_state[:2]
    
    # Find the closest point on the RRT path
    distances = np.linalg.norm(rrt_path - current_pos, axis=1)
    closest_idx = np.argmin(distances)
    
    # Create a reference path by looking ahead on the RRT path
    for i in range(N + 1):
        idx = min(closest_idx + i, len(rrt_path) - 1)
        ref_path[:, i] = rrt_path[idx]
        
    return ref_path

def nmpc_solver(robot_state, x_ref, dynamic_obstacles, static_obstacles, initial_guess=None):
    """
    Builds and solves the NMPC optimization problem using CasADi.
    """
    
    # --- Decision variables ---
    X = ca.MX.sym('X', 5, const.N + 1)  # State: [x, y, theta, v, w]
    U = ca.MX.sym('U', 2, const.N)      # Control: [a, alpha]

    # Stack all decision variables into a single flat vector
    opt_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

    # --- Parameters ---
    x0 = robot_state
    X_ref = x_ref # This is passed in from main.py

    # --- Cost function ---
    cost = 0
    
    # Path tracking, velocity reward, and control effort cost
    for k in range(const.N):
        # Path tracking cost
        error = X[:2, k] - X_ref[:, k]
        cost += ca.mtimes([error.T, const.Q_path, error])
        
        # FIX: Add velocity reward to prevent getting stuck
        cost -= const.Q_vel * X[3, k] # Penalize low velocity
        
        # Control effort cost
        cost += ca.mtimes([U[:, k].T, const.R, U[:, k]])
    
    # Terminal cost (only path tracking)
    terminal_error = X[:2, const.N] - X_ref[:, const.N]
    cost += ca.mtimes([terminal_error.T, const.Q_path, terminal_error])

    # --- Constraints ---
    g = []  # Constraint expressions
    lbg = []  # Lower bounds for constraints
    ubg = []  # Upper bounds for constraints

    # Initial state constraint
    g.append(X[:, 0] - x0)
    lbg.extend([0] * 5)
    ubg.extend([0] * 5)

    # --- Dynamics constraints ---
    for k in range(const.N):
        x, y, theta, v, w = X[0,k], X[1,k], X[2,k], X[3,k], X[4,k]
        a, alpha = U[0,k], U[1,k]

        # Second-order dynamics
        x_next = ca.vertcat(
            x + const.DT * v * ca.cos(theta),
            y + const.DT * v * ca.sin(theta),
            theta + const.DT * w,
            v + const.DT * a,
            w + const.DT * alpha
        )
        
        g.append(X[:, k+1] - x_next)
        lbg.extend([0] * 5)
        ubg.extend([0] * 5)

    # --- Robust Discrete-Time CBF Constraints ---
    h_values_dyn_expr = []
    h_values_stat_expr = []
    alpha_cbf = const.CBF_GAMMA # CBF gain
    
    # --- 1. CBF for Dynamic Obstacles ---
    for i, obs in enumerate(dynamic_obstacles):
        for k in range(const.N): # From k=0 to N-1
            # h(x_k)
            obs_pos_k = obs.predicted_path[:, k]
            cov_k = obs.predicted_cov[k]
            sigma_k = obs.sigma_bounds[k]
            unc_rad_k = sigma_k * ca.sqrt(ca.trace(cov_k))
            eff_rad_k = obs.radius + unc_rad_k
            dist_sq_k = (X[0, k] - obs_pos_k[0])**2 + (X[1, k] - obs_pos_k[1])**2
            
            # FIX: Use linear distance (sqrt) to match static obstacle formulation
            dist_k = ca.sqrt(dist_sq_k + 1e-8)
            h_k = dist_k - (const.ROBOT_RADIUS + eff_rad_k + const.D_SAFE)
            
            # h(x_{k+1})
            obs_pos_k1 = obs.predicted_path[:, k+1]
            cov_k1 = obs.predicted_cov[k+1]
            sigma_k1 = obs.sigma_bounds[k+1]
            unc_rad_k1 = sigma_k1 * ca.sqrt(ca.trace(cov_k1))
            eff_rad_k1 = obs.radius + unc_rad_k1
            dist_sq_k1 = (X[0, k+1] - obs_pos_k1[0])**2 + (X[1, k+1] - obs_pos_k1[1])**2

            # FIX: Use linear distance (sqrt) to match static obstacle formulation
            dist_k1 = ca.sqrt(dist_sq_k1 + 1e-8)
            h_k1 = dist_k1 - (const.ROBOT_RADIUS + eff_rad_k1 + const.D_SAFE)

            # Constraint: h(x_{k+1}) >= (1 - alpha*dt) * h(x_k)
            g.append(h_k1 - (1 - alpha_cbf * const.DT) * h_k)
            lbg.append(0)
            ubg.append(ca.inf)
            
            h_values_dyn_expr.append(h_k) # Log h at step k

    # --- 2. CBF for Static Obstacles ---
    for i, obs in enumerate(static_obstacles):
        for k in range(const.N): # From k=0 to N-1
            
            # --- h(x_k) ---
            # FIX: Re-implementing the v10 logic correctly (using sqrt)
            dist_x_k = ca.fabs(X[0, k] - obs.center[0]) - obs.width / 2
            dist_y_k = ca.fabs(X[1, k] - obs.center[1]) - obs.height / 2
            h_outside_sq_k = (ca.fmax(0, dist_x_k))**2 + (ca.fmax(0, dist_y_k))**2
            # FIX: Correct penalty logic
            penalty_inside_k = ca.fmin(0, dist_x_k) + ca.fmin(0, dist_y_k)
            # FIX: Use sqrt to get linear distance h, not h^2
            h_k = ca.sqrt(h_outside_sq_k + 1e-8) - (const.ROBOT_RADIUS + const.D_SAFE) + penalty_inside_k

            # --- h(x_{k+1}) ---
            dist_x_k1 = ca.fabs(X[0, k+1] - obs.center[0]) - obs.width / 2
            dist_y_k1 = ca.fabs(X[1, k+1] - obs.center[1]) - obs.height / 2
            h_outside_sq_k1 = (ca.fmax(0, dist_x_k1))**2 + (ca.fmax(0, dist_y_k1))**2
            # FIX: Correct penalty logic
            penalty_inside_k1 = ca.fmin(0, dist_x_k1) + ca.fmin(0, dist_y_k1)
            # FIX: Use sqrt to get linear distance h, not h^2
            h_k1 = ca.sqrt(h_outside_sq_k1 + 1e-8) - (const.ROBOT_RADIUS + const.D_SAFE) + penalty_inside_k1

            # Constraint: h(x_{k+1}) >= (1 - alpha*dt) * h(x_k)
            g.append(h_k1 - (1 - alpha_cbf * const.DT) * h_k)
            lbg.append(0)
            ubg.append(ca.inf)
            
            h_values_stat_expr.append(h_k) # Log h at step k

    # --- State and Control Bounds (placed in lbx/ubx for nlpsol) ---
    lbx = []
    ubx = []
    
    # State bounds (for X)
    for k in range(const.N + 1):
        lbx.extend([0, 0, -ca.inf, 0, -const.MAX_ANGULAR_VEL])
        ubx.extend([const.X_LIM, const.Y_LIM, ca.inf, const.MAX_LINEAR_VEL, const.MAX_ANGULAR_VEL])
    
    # Control bounds (for U)
    for k in range(const.N):
        lbx.extend([-const.MAX_LINEAR_ACCEL, -const.MAX_ANGULAR_ACCEL])
        ubx.extend([const.MAX_LINEAR_ACCEL, const.MAX_ANGULAR_ACCEL])

    # --- Setup NLP ---
    nlp = {
        'x': opt_variables,
        'f': cost,
        'g': ca.vertcat(*g)
    }

    # Solver options
    opts = {
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.sb': 'yes',
        'ipopt.max_iter': 1000,
        'ipopt.acceptable_tol': 1e-6,
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # --- Set Initial Guess ---
    if initial_guess is None:
        initial_guess = np.zeros(opt_variables.shape[0])
        # State guess
        for k in range(const.N + 1):
            idx_start = k * 5
            initial_guess[idx_start:idx_start+5] = robot_state
        
        # Control guess (encourage forward motion)
        control_start_idx = 5 * (const.N + 1)
        for k in range(const.N):
            initial_guess[control_start_idx + k*2] = 0.1 # Small forward accel
    
    try:
        sol = solver(x0=initial_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        
        stats = solver.stats()
        if not stats['success']:
            if stats['return_status'] not in ['Solve_Succeeded', 'Solved_To_Acceptable_Level']:
                print(f"Solver warning: {stats['return_status']}")
                # If it fails, return a braking maneuver
                raise Exception(f"Solver failed: {stats['return_status']}")
        
        w_opt = sol['x'].full().flatten()
        
        # Reshape solution from flat vector
        X_sol = w_opt[:5*(const.N+1)].reshape((5, const.N+1), order='F')
        U_sol = w_opt[5*(const.N+1):].reshape((2, const.N), order='F')
        
        # Evaluate h-values for logging
        h_dyn = []
        h_stat = []
        
        h_dyn_func = ca.Function('h_dyn', [opt_variables], [ca.vertcat(*h_values_dyn_expr)])
        h_stat_func = ca.Function('h_stat', [opt_variables], [ca.vertcat(*h_values_stat_expr)])
        
        h_dyn = h_dyn_func(w_opt).full().flatten().tolist()
        h_stat = h_stat_func(w_opt).full().flatten().tolist()
        
        return U_sol[:, 0], X_sol, h_dyn, h_stat, w_opt
        
    except Exception as e:
        print(f"Solver failed: {e}")
        # Return fallback values
        fallback_traj = np.tile(robot_state.reshape(5, 1), (1, const.N + 1))
        # Apply braking
        u_brake = np.array([const.BRAKING_ACCELERATION, 0.0])
        if robot_state[3] < 0.1: # If already stopped, don't brake
             u_brake[0] = 0.0
             
        return u_brake, fallback_traj, [], [], None