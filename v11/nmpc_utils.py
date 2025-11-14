"""NMPC utilities (v11).

This module provides a single entry function `nmpc_solver` that builds and
solves an optimization problem using CasADi/IPOPT. The function documents
expected inputs and returns; internal sections are commented for clarity.
"""

import casadi as ca
import numpy as np
import constants as const


def nmpc_solver(robot_state, x_ref, dynamic_obstacles, static_obstacles, initial_guess=None):
    """Build and solve the NMPC optimization problem.

    Args:
        robot_state (array): current robot state [x,y,theta,v,w]
        x_ref (2xN+1 array): reference positions for the prediction horizon
        dynamic_obstacles (list): obstacles with predicted_path/predicted_cov
        static_obstacles (list): static rectangular obstacles (center,width,height)
        initial_guess (np.array, optional): warm-start vector for the solver

    Returns:
        tuple: (first_control, X_solution, h_dyn_list, h_stat_list, raw_solution)
               or (None, None, [], [], None) on solver failure.
    """

    # --- Decision variables (states and inputs over the horizon) ---
    X = ca.MX.sym('X', 5, const.N + 1)  # State: [x, y, theta, v, w]
    U = ca.MX.sym('U', 2, const.N)      # Control: [a, alpha]

    # Stack decision variables into a single optimization vector
    opt_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

    # --- Parameters / local copies ---
    x0 = robot_state
    X_ref = x_ref

    # --- Cost function: path tracking + (optional) velocity term + control effort ---
    cost = 0
    for k in range(const.N):
        error = X[:2, k] - X_ref[:, k]
        cost += ca.mtimes([error.T, const.Q_path, error])
        # If Q_vel exists in constants, reward forward speed
        if hasattr(const, 'Q_vel'):
            cost -= const.Q_vel * X[3, k]
        cost += ca.mtimes([U[:, k].T, const.R, U[:, k]])

    # Terminal tracking cost
    terminal_error = X[:2, const.N] - X_ref[:, const.N]
    cost += ca.mtimes([terminal_error.T, const.Q_path, terminal_error])

    # --- Constraints containers ---
    g = []  # constraint expressions
    lbg = []
    ubg = []

    # Initial state equality constraint
    g.append(X[:, 0] - x0)
    lbg.extend([0, 0, 0, 0, 0])
    ubg.extend([0, 0, 0, 0, 0])

    # --- Dynamics constraints (discrete-time second-order unicycle model) ---
    for k in range(const.N):
        x, y, theta, v, w = X[0, k], X[1, k], X[2, k], X[3, k], X[4, k]
        a, alpha = U[0, k], U[1, k]

        x_next = ca.vertcat(
            x + const.DT * v * ca.cos(theta),
            y + const.DT * v * ca.sin(theta),
            theta + const.DT * w,
            v + const.DT * a,
            w + const.DT * alpha,
        )

        g.append(X[:, k + 1] - x_next)
        lbg.extend([0, 0, 0, 0, 0])
        ubg.extend([0, 0, 0, 0, 0])

    # --- CBF-style safety constraints for dynamic obstacles ---
    h_values_dyn_expr = []
    for obs in dynamic_obstacles:
        for k in range(const.N + 1):
            obs_pred_pos = obs.predicted_path[:, k]
            cov = obs.predicted_cov[k]
            sigma_bound = obs.sigma_bounds[k]
            # uncertainty radius derived from covariance trace scaled by sigma bound
            uncertainty_radius = sigma_bound * ca.sqrt(ca.trace(cov))
            effective_radius = obs.radius + uncertainty_radius

            dist_sq = (X[0, k] - obs_pred_pos[0]) ** 2 + (X[1, k] - obs_pred_pos[1]) ** 2
            h = dist_sq - (const.ROBOT_RADIUS + effective_radius + const.D_SAFE) ** 2

            h_values_dyn_expr.append(h)
            g.append(h)
            lbg.append(0)
            ubg.append(ca.inf)

    # --- Safety constraints for static rectangular obstacles ---
    h_values_stat_expr = []
    for obs in static_obstacles:
        for k in range(const.N + 1):
            dist_x = ca.fabs(X[0, k] - obs.center[0]) - obs.width / 2
            dist_y = ca.fabs(X[1, k] - obs.center[1]) - obs.height / 2

            h_outside_sq = (ca.fmax(0, dist_x)) ** 2 + (ca.fmax(0, dist_y)) ** 2
            h = h_outside_sq - (const.ROBOT_RADIUS + const.D_SAFE) ** 2

            h_values_stat_expr.append(h)
            g.append(h)
            lbg.append(0)
            ubg.append(ca.inf)

    # --- State and control bounds ---
    lbx = []
    ubx = []
    for k in range(const.N + 1):
        lbx.extend([0, 0, -ca.inf, 0, -const.MAX_ANGULAR_VEL])
        ubx.extend([const.X_LIM, const.Y_LIM, ca.inf, const.MAX_LINEAR_VEL, const.MAX_ANGULAR_VEL])

    for k in range(const.N):
        lbx.extend([-const.MAX_LINEAR_ACCEL, -const.MAX_ANGULAR_ACCEL])
        ubx.extend([const.MAX_LINEAR_ACCEL, const.MAX_ANGULAR_ACCEL])

    # --- Setup NLP and solver options ---
    nlp = {'x': opt_variables, 'f': cost, 'g': ca.vertcat(*g)}
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # --- Initial guess / warm start ---
    if initial_guess is None:
        initial_guess = np.zeros(opt_variables.shape[0])
        # use current robot state repeated along the state portion
        for k in range(const.N + 1):
            initial_guess[k * 5:(k + 1) * 5] = robot_state

    # --- Solve the NLP ---
    try:
        sol = solver(x0=initial_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        X_sol = w_opt[:5 * (const.N + 1)].reshape((5, const.N + 1), order='F')
        U_sol = w_opt[5 * (const.N + 1):].reshape((2, const.N), order='F')

        # Evaluate CBF h-values for diagnostics
        h_dyn = [ca.Function('h', [opt_variables], [h_expr])(w_opt).full()[0, 0] for h_expr in h_values_dyn_expr]
        h_stat = [ca.Function('h', [opt_variables], [h_expr])(w_opt).full()[0, 0] for h_expr in h_values_stat_expr]

        return U_sol[:, 0], X_sol, h_dyn, h_stat, w_opt

    except Exception as e:
        # On failure return sentinel values for caller to handle
        print(f"Solver failed: {e}")
        return None, None, [], [], None