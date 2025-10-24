import casadi as ca
import numpy as np
import constants as const

def nmpc_solver(robot_state, x_ref, dynamic_obstacles, static_obstacles, initial_guess=None):
    # --- Decision variables ---
    X = ca.MX.sym('X', 5, const.N + 1)  # State: [x, y, theta, v, w]
    U = ca.MX.sym('U', 2, const.N)      # Control: [a, alpha]

    # Stack all decision variables
    opt_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

    # --- Parameters ---
    x0 = robot_state
    X_ref = x_ref

    # --- Cost function ---
    cost = 0
    
    # Path tracking cost
    for k in range(const.N):
        error = X[:2, k] - X_ref[:, k]
        cost += ca.mtimes([error.T, const.Q_path, error])
        cost += ca.mtimes([U[:, k].T, const.R, U[:, k]])
    
    # Terminal cost
    terminal_error = X[:2, const.N] - X_ref[:, const.N]
    cost += ca.mtimes([terminal_error.T, const.Q_path, terminal_error])

    # --- Constraints ---
    g = []  # Constraint expressions
    lbg = []  # Lower bounds
    ubg = []  # Upper bounds

    # Initial state constraint
    g.append(X[:, 0] - x0)
    lbg.extend([0, 0, 0, 0, 0])
    ubg.extend([0, 0, 0, 0, 0])

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
        lbg.extend([0, 0, 0, 0, 0])
        ubg.extend([0, 0, 0, 0, 0])

    # --- Higher-Order CBF Constraints for Dynamic Obstacles ---
    h_values_dyn_expr = []
    for i, obs in enumerate(dynamic_obstacles):
        for k in range(const.N + 1):
            obs_pred_pos = obs.predicted_path[:, k]
            cov = obs.predicted_cov[k]
            sigma_bound = obs.sigma_bounds[k]
            uncertainty_radius = sigma_bound * ca.sqrt(ca.trace(cov))
            effective_radius = obs.radius + uncertainty_radius
            
            # Barrier function h(x)
            dist_sq = (X[0, k] - obs_pred_pos[0])**2 + (X[1, k] - obs_pred_pos[1])**2
            dist = ca.sqrt(dist_sq + 1e-8)
            h = dist - (const.ROBOT_RADIUS + effective_radius + const.D_SAFE)
            
            h_values_dyn_expr.append(h)
            
            # First constraint: h(x) >= 0
            g.append(h)
            lbg.append(0)
            ubg.append(ca.inf)
            
            # Second constraint: Higher-order CBF to maintain safety margin
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
                
                # Higher-order CBF: dh/dt + alpha * h >= 0
                alpha = const.CBF_GAMMA
                g.append(dh_dt + alpha * h)
                lbg.append(0)
                ubg.append(ca.inf)

    # --- Higher-Order CBF Constraints for Static Obstacles ---
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

            # First constraint: h >= 0
            g.append(h)
            lbg.append(0)
            ubg.append(ca.inf)
            
            # Second constraint: Higher-order CBF
            if k < const.N:
                v_curr = X[3, k]
                theta_curr = X[2, k]
                
                # Compute gradient of h with respect to position
                sign_x = ca.sign(X[0, k] - obs.center[0])
                sign_y = ca.sign(X[1, k] - obs.center[1])
                
                # Gradient components
                dh_dx = sign_x * ca.fmax(0, dist_x) / (ca.sqrt(h_outside_sq + 1e-8))
                dh_dy = sign_y * ca.fmax(0, dist_y) / (ca.sqrt(h_outside_sq + 1e-8))
                
                # Time derivative of h
                dh_dt = dh_dx * v_curr * ca.cos(theta_curr) + dh_dy * v_curr * ca.sin(theta_curr)
                
                # Higher-order CBF constraint: dh/dt + alpha * h >= 0
                alpha = const.CBF_GAMMA
                g.append(dh_dt + alpha * h)
                lbg.append(0)
                ubg.append(ca.inf)

    # --- State bounds ---
    lbx = []
    ubx = []
    
    # State bounds for all time steps
    for k in range(const.N + 1):
        lbx.extend([0, 0, -ca.inf, 0, -const.MAX_ANGULAR_VEL])
        ubx.extend([const.X_LIM, const.Y_LIM, ca.inf, const.MAX_LINEAR_VEL, const.MAX_ANGULAR_VEL])
    
    # Control bounds
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
        'ipopt.acceptable_obj_change_tol': 1e-6,
        'ipopt.mu_strategy': 'adaptive',
        'ipopt.acceptable_iter': 10
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    if initial_guess is None:
        initial_guess = np.zeros(opt_variables.shape[0])
        for k in range(const.N + 1):
            if k == 0:
                initial_guess[k*5:(k+1)*5] = robot_state
            else:
                initial_guess[k*5] = X_ref[0, k]
                initial_guess[k*5+1] = X_ref[1, k]
                initial_guess[k*5+2] = robot_state[2]
                initial_guess[k*5+3] = 1.0
                initial_guess[k*5+4] = 0.0
        
        # --- MODIFICATION START ---
        # A better initial guess for controls to encourage forward motion
        control_start_idx = 5 * (const.N + 1)
        for k in range(const.N):
            initial_guess[control_start_idx + k*2] = const.MAX_LINEAR_ACCEL / 2 # Strong initial forward acceleration
            initial_guess[control_start_idx + k*2 + 1] = 0.0
        # --- MODIFICATION END ---
    
    try:
        sol = solver(x0=initial_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        
        stats = solver.stats()
        if not stats['success']:
            print(f"Solver warning: {stats['return_status']}")
            return None, None, [], [], None
        
        w_opt = sol['x'].full().flatten()
        
        X_sol = w_opt[:5*(const.N+1)].reshape((5, const.N+1), order='F')
        U_sol = w_opt[5*(const.N+1):].reshape((2, const.N), order='F')
        
        h_dyn = []
        h_stat = []
        
        for h_expr in h_values_dyn_expr:
            h_func = ca.Function('h', [opt_variables], [h_expr])
            h_val = float(h_func(w_opt))
            h_dyn.append(h_val)
        
        for h_expr in h_values_stat_expr:
            h_func = ca.Function('h', [opt_variables], [h_expr])
            h_val = float(h_func(w_opt))
            h_stat.append(h_val)
        
        return U_sol[:, 0], X_sol, h_dyn, h_stat, w_opt
        
    except Exception as e:
        print(f"Solver failed: {e}")
        return None, None, [], [], None