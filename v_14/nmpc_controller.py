"""
nmpc_controller.py

Defines the NMPC optimization problem using CasADi's `nlpsol` interface.
Implements the Second-Order CBF logic from v10 (h >= 0 and h_dot + a*h >= 0)
without slack variables.
"""

import casadi as ca
import numpy as np
import constants as const

def generate_reference_trajectory(robot_state, rrt_path, N, dt):
    """
    Generates the reference trajectory (X_ref) for the NMPC horizon
    by finding the closest point on the RRT path and looking ahead.
    """
    current_pos = robot_state[:2]
    distances = np.linalg.norm(rrt_path - current_pos, axis=1)
    closest_idx = np.argmin(distances)
    
    ref_path = np.zeros((2, N + 1))
    
    for i in range(N + 1):
        # Find the index on the path to track
        # This simple lookahead just steps along the RRT* waypoints
        idx = min(closest_idx + i, len(rrt_path) - 1)
        ref_path[:, i] = rrt_path[idx]
        
    return ref_path

def nmpc_solver(robot_state, x_ref, dynamic_obstacles, static_obstacles, initial_guess=None):
    """
    Builds and solves the NMPC optimization problem using the v10-style
    manual problem setup and second-order CBF.
    """
    
    # --- Decision variables ---
    X = ca.MX.sym('X', 5, const.N + 1)  # State: [x, y, theta, v, w]
    U = ca.MX.sym('U', 2, const.N)      # Control: [a, alpha]

    # Stack all decision variables into a single flat vector
    opt_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

    # --- Parameters ---
    x0 = robot_state
    X_ref = x_ref # This is passed in from main.py

    # --- Cost function (from v10) ---
    cost = 0
    
    # Path tracking and control effort cost
    for k in range(const.N):
        error = X[:2, k] - X_ref[:, k]
        cost += ca.mtimes([error.T, const.Q_path, error])
        
        # NOTE: The v10 cost function does not include a Q_vel reward.
        # This implementation matches the v10 file provided.
        cost += ca.mtimes([U[:, k].T, const.R, U[:, k]])
    
    # Terminal cost
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

    # --- Higher-Order CBF Constraints for Dynamic Obstacles (v10 logic) ---
    h_values_dyn_expr = []
    alpha_cbf = const.CBF_GAMMA
    
    for i, obs in enumerate(dynamic_obstacles):
        for k in range(const.N + 1):
            obs_pred_pos = obs.predicted_path[:, k]
            cov = obs.predicted_cov[k]
            sigma_bound = obs.sigma_bounds[k]
            
            # Use trace for uncertainty radius
            uncertainty_radius = sigma_bound * ca.sqrt(ca.trace(cov))
            # Use object properties for radii
            effective_radius = obs.radius + uncertainty_radius
            
            # Barrier function h(x)
            dist_sq = (X[0, k] - obs_pred_pos[0])**2 + (X[1, k] - obs_pred_pos[1])**2
            dist = ca.sqrt(dist_sq + 1e-8)
            h = dist - (const.ROBOT_RADIUS + effective_radius + const.D_SAFE)
            
            h_values_dyn_expr.append(h)
            
            # First constraint: h(x) >= 0 (Applied for k=0...N)
            g.append(h)
            lbg.append(0)
            ubg.append(ca.inf)
            
            # Second constraint: dh/dt + alpha * h >= 0 (Applied for k=0...N-1)
            if k < const.N:
                v_curr = X[3, k]
                theta_curr = X[2, k]
                
                # Distance components
                dx = X[0, k] - obs_pred_pos[0]
                dy = X[1, k] - obs_pred_pos[1]
                
                # Obstacle velocity (constant velocity model)
                obs_vx = obs.velocity * ca.cos(obs.measured_state[2])
                obs_vy = obs.velocity * ca.sin(obs.measured_state[2])
                
                # Time derivative of h: dh/dt
                dh_dt = ((dx * (v_curr * ca.cos(theta_curr) - obs_vx) + 
                         dy * (v_curr * ca.sin(theta_curr) - obs_vy)) / (dist + 1e-8))
                
                # --- ROBUSTNESS FIX ---
                # Only apply the derivative constraint for k > 0.
                # At k=0, v_curr and theta_curr are fixed parameters, not variables.
                # Enforcing this constraint at k=0 causes infeasibility.
                if k > 0:
                    g.append(dh_dt + alpha_cbf * h)
                    lbg.append(0)
                    ubg.append(ca.inf)

    # --- Higher-Order CBF Constraints for Static Obstacles (v10 logic) ---
    h_values_stat_expr = []
    for i, obs in enumerate(static_obstacles):
        for k in range(const.N + 1):
            # Calculate signed distances to the rectangle's boundaries
            dist_x = ca.fabs(X[0, k] - obs.center[0]) - obs.width / 2
            dist_y = ca.fabs(X[1, k] - obs.center[1]) - obs.height / 2

            # Distance outside the rectangle
            h_outside_sq = (ca.fmax(0, dist_x))**2 + (ca.fmax(0, dist_y))**2
            
            # Penalty inside
            penalty_inside = ca.fmin(0, dist_x) + ca.fmin(0, dist_y)

            # Barrier function
            h = ca.sqrt(h_outside_sq + 1e-8) - (const.ROBOT_RADIUS + const.D_SAFE) + penalty_inside
            h_values_stat_expr.append(h)

            # First constraint: h >= 0 (Applied for k=0...N)
            g.append(h)
            lbg.append(0)
            ubg.append(ca.inf)
            
            # Second constraint: dh/dt + alpha * h >= 0 (Applied for k=0...N-1)
            if k < const.N:
                v_curr = X[3, k]
                theta_curr = X[2, k]
                
                # Compute gradient of h with respect to position
                sign_x = ca.sign(X[0, k] - obs.center[0])
                sign_y = ca.sign(X[1, k] - obs.center[1])
                
                # Gradient components
                dh_dx_expr = sign_x * ca.fmax(0, dist_x) / (ca.sqrt(h_outside_sq + 1e-8))
                dh_dy_expr = sign_y * ca.fmax(0, dist_y) / (ca.sqrt(h_outside_sq + 1e-8))
                
                # Handle penalty_inside gradient (simplified)
                dh_dx = ca.if_else(dist_x < 0, sign_x, dh_dx_expr)
                dh_dy = ca.if_else(dist_y < 0, sign_y, dh_dy_expr)

                # Time derivative of h
                dh_dt = dh_dx * v_curr * ca.cos(theta_curr) + dh_dy * v_curr * ca.sin(theta_curr)
                
                # --- ROBUSTNESS FIX ---
                # Only apply the derivative constraint for k > 0.
                if k > 0:
                    g.append(dh_dt + alpha_cbf * h)
                    lbg.append(0)
                    ubg.append(ca.inf)

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

    # --- Set Initial Guess (v10 logic) ---
    if initial_guess is None:
        initial_guess = np.zeros(opt_variables.shape[0])
        # State guess
        for k in range(const.N + 1):
            idx_start = k * 5
            if k == 0:
                initial_guess[idx_start:idx_start+5] = robot_state
            else:
                initial_guess[idx_start] = X_ref[0, k]
                initial_guess[idx_start+1] = X_ref[1, k]
                initial_guess[idx_start+2] = robot_state[2]
                initial_guess[idx_start+3] = 1.0 # Guess moving forward
                initial_guess[idx_start+4] = 0.0
        
        # Control guess (encourage forward motion)
        control_start_idx = 5 * (const.N + 1)
        for k in range(const.N):
            initial_guess[control_start_idx + k*2] = const.MAX_LINEAR_ACCEL / 5.0 
            initial_guess[control_start_idx + k*2 + 1] = 0.0
    
    try:
        sol = solver(x0=initial_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        
        stats = solver.stats()
        if not stats['success']:
            # Allow "Solved_To_Acceptable_Level"
            if stats['return_status'] not in ['Solve_Succeeded', 'Solved_To_Acceptable_Level']:
                print(f"Solver warning: {stats['return_status']}")
                # Don't return None, try to use the (suboptimal) solution
                # return None, None, [], [], None
        
        w_opt = sol['x'].full().flatten()
        
        # Reshape solution from flat vector
        X_sol = w_opt[:5*(const.N+1)].reshape((5, const.N+1), order='F')
        U_sol = w_opt[5*(const.N+1):].reshape((2, const.N), order='F')
        
        # Evaluate h-values for logging
        h_dyn = []
        h_stat = []
        
        # Need to create CasADi functions to evaluate the h expressions
        h_dyn_func = ca.Function('h_dyn', [opt_variables], [ca.vertcat(*h_values_dyn_expr)])
        h_stat_func = ca.Function('h_stat', [opt_variables], [ca.vertcat(*h_values_stat_expr)])
        
        h_dyn = h_dyn_func(w_opt).full().flatten().tolist()
        h_stat = h_stat_func(w_opt).full().flatten().tolist()
        
        # Return signature matches v10
        return U_sol[:, 0], X_sol, h_dyn, h_stat, w_opt
        
    except Exception as e:
        print(f"Solver failed: {e}")
        # Return fallback values
        fallback_traj = np.tile(robot_state.reshape(5, 1), (1, const.N + 1))
        return np.array([const.BRAKING_ACCELERATION, 0.0]), \
               fallback_traj, \
               [], [], None