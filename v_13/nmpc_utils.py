import casadi as ca
import numpy as np
import constants as const

def nmpc_solver(robot_state, x_ref, dynamic_obstacles, static_obstacles, initial_guess=None):
    """
    V13 Solver: V11 Structure + V10 Second-Order CBF Logic
    FIX: Skips derivative constraints at k=0 to prevent infeasibility.
    """
    # --- Decision variables ---
    X = ca.MX.sym('X', 5, const.N + 1)  # State: [x, y, theta, v, w]
    U = ca.MX.sym('U', 2, const.N)      # Control: [a, alpha]

    # Stack decision variables
    opt_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

    # --- Parameters ---
    x0 = robot_state
    X_ref = x_ref

    # --- Cost function ---
    cost = 0
    for k in range(const.N):
        error = X[:2, k] - X_ref[:, k]
        # High penalty for path deviation (Requested feature)
        cost += ca.mtimes([error.T, const.Q_path, error])
        
        # Reward forward velocity (from v11) to encourage movement
        if hasattr(const, 'Q_vel'):
            cost -= const.Q_vel * X[3, k]
            
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
            
            # 1. Position Safety Constraint: h(x) >= 0
            g.append(h)
            lbg.append(0)
            ubg.append(ca.inf)
            
            # 2. Higher-Order CBF Constraint: dh/dt + alpha * h >= 0
            # CRITICAL FIX: Only enforce for k > 0. 
            # At k=0, 'dh/dt' depends on initial velocity (fixed parameter), so the solver cannot 
            # satisfy this constraint if the initial state is already "violating" the rate limit.
            if k > 0:
                v_curr = X[3, k]
                theta_curr = X[2, k]
                
                dx = X[0, k] - obs_pred_pos[0]
                dy = X[1, k] - obs_pred_pos[1]
                
                obs_vx = obs.velocity * ca.cos(obs.measured_state[2])
                obs_vy = obs.velocity * ca.sin(obs.measured_state[2])
                
                dh_dt = ((dx * (v_curr * ca.cos(theta_curr) - obs_vx) + 
                         dy * (v_curr * ca.sin(theta_curr) - obs_vy)) / (dist + 1e-8))
                
                alpha = const.CBF_GAMMA
                g.append(dh_dt + alpha * h)
                lbg.append(0)
                ubg.append(ca.inf)

    # --- Higher-Order CBF Constraints for Static Obstacles ---
    h_values_stat_expr = []
    for i, obs in enumerate(static_obstacles):
        for k in range(const.N + 1):
            dist_x = ca.fabs(X[0, k] - obs.center[0]) - obs.width / 2
            dist_y = ca.fabs(X[1, k] - obs.center[1]) - obs.height / 2

            h_outside_sq = (ca.fmax(0, dist_x))**2 + (ca.fmax(0, dist_y))**2
            penalty_inside = ca.fmin(0, dist_x) + ca.fmin(0, dist_y)

            h = ca.sqrt(h_outside_sq + 1e-8) - (const.ROBOT_RADIUS + const.D_SAFE) + penalty_inside
            h_values_stat_expr.append(h)

            # 1. Position Safety Constraint: h >= 0
            g.append(h)
            lbg.append(0)
            ubg.append(ca.inf)
            
            # 2. Higher-Order CBF Constraint
            # CRITICAL FIX: Only enforce for k > 0
            if k > 0:
                v_curr = X[3, k]
                theta_curr = X[2, k]
                
                sign_x = ca.sign(X[0, k] - obs.center[0])
                sign_y = ca.sign(X[1, k] - obs.center[1])
                
                dh_dx = sign_x * ca.fmax(0, dist_x) / (ca.sqrt(h_outside_sq + 1e-8))
                dh_dy = sign_y * ca.fmax(0, dist_y) / (ca.sqrt(h_outside_sq + 1e-8))
                
                dh_dt = dh_dx * v_curr * ca.cos(theta_curr) + dh_dy * v_curr * ca.sin(theta_curr)
                
                alpha = const.CBF_GAMMA
                g.append(dh_dt + alpha * h)
                lbg.append(0)
                ubg.append(ca.inf)

    # --- State bounds ---
    lbx = []
    ubx = []
    for k in range(const.N + 1):
        lbx.extend([0, 0, -ca.inf, 0, -const.MAX_ANGULAR_VEL])
        ubx.extend([const.X_LIM, const.Y_LIM, ca.inf, const.MAX_LINEAR_VEL, const.MAX_ANGULAR_VEL])
    
    for k in range(const.N):
        lbx.extend([-const.MAX_LINEAR_ACCEL, -const.MAX_ANGULAR_ACCEL])
        ubx.extend([const.MAX_LINEAR_ACCEL, const.MAX_ANGULAR_ACCEL])

    # --- Solver ---
    nlp = {'x': opt_variables, 'f': cost, 'g': ca.vertcat(*g)}
    opts = {
        'ipopt.print_level': 0, 
        'print_time': 0, 
        'ipopt.sb': 'yes',
        'ipopt.max_iter': 2000,
        'ipopt.acceptable_tol': 1e-4,
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    if initial_guess is None:
        initial_guess = np.zeros(opt_variables.shape[0])
        for k in range(const.N + 1):
            initial_guess[k*5:(k+1)*5] = robot_state

    try:
        sol = solver(x0=initial_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        
        if not solver.stats()['success']:
            # Return best attempt if solver struggled but didn't crash hard
            # (Useful when near obstacles)
            pass

        w_opt = sol['x'].full().flatten()
        X_sol = w_opt[:5*(const.N+1)].reshape((5, const.N+1), order='F')
        U_sol = w_opt[5*(const.N+1):].reshape((2, const.N), order='F')
        
        # Diagnostics
        h_dyn = [ca.Function('h', [opt_variables], [expr])(w_opt).full()[0,0] for expr in h_values_dyn_expr]
        h_stat = [ca.Function('h', [opt_variables], [expr])(w_opt).full()[0,0] for expr in h_values_stat_expr]
        
        return U_sol[:, 0], X_sol, h_dyn, h_stat, w_opt
        
    except Exception as e:
        print(f"Solver failed: {e}")
        return None, None, [], [], None