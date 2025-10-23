import casadi as ca
import numpy as np
import constants as const

def nmpc_solver(robot_state, x_ref, dynamic_obstacles, static_obstacles):
    opti = ca.Opti()

    # --- Decision variables ---
    X = opti.variable(5, const.N + 1) # State: [x, y, theta, v, w]
    U = opti.variable(2, const.N)     # Control: [a, alpha]
    S_dyn = opti.variable(len(dynamic_obstacles), const.N + 1)
    S_stat = opti.variable(len(static_obstacles), const.N + 1)

    # --- Parameters ---
    x0 = opti.parameter(5, 1)
    X_ref = opti.parameter(2, const.N + 1) # Reference path from RRT*

    # --- Cost function ---
    cost = 0
    for k in range(const.N):
        error = X[:2, k] - X_ref[:, k]
        cost += ca.mtimes([error.T, const.Q_path, error])
        cost += ca.mtimes([U[:, k].T, const.R, U[:, k]])
    cost += const.SLACK_PENALTY * (ca.sumsqr(S_dyn) + ca.sumsqr(S_stat))
    opti.minimize(cost)

    # --- Dynamics constraints ---
    for k in range(const.N):
        x, y, theta, v, w = X[0,k], X[1,k], X[2,k], X[3,k], X[4,k]
        a, alpha = U[0,k], U[1,k]

        # Second-order dynamics
        opti.subject_to(X[0, k+1] == x + const.DT * v * ca.cos(theta))
        opti.subject_to(X[1, k+1] == y + const.DT * v * ca.sin(theta))
        opti.subject_to(X[2, k+1] == theta + const.DT * w)
        opti.subject_to(X[3, k+1] == v + const.DT * a)
        opti.subject_to(X[4, k+1] == w + const.DT * alpha)

    # --- CBF Constraints ---
    h_values_dyn_expr = []
    h_values_stat_expr = []

    # Dynamic obstacles
    for i, obs in enumerate(dynamic_obstacles):
        h_sequence = []
        for k in range(const.N + 1):
            obs_pred_pos = obs.predicted_path[:, k]
            cov = obs.predicted_cov[k]
            sigma_bound = obs.sigma_bounds[k]
            uncertainty_radius = sigma_bound * ca.sqrt(ca.trace(cov)) # CORRECTED LINE
            effective_radius = obs.radius + uncertainty_radius
            dist_sq = (X[0, k] - obs_pred_pos[0])**2 + (X[1, k] - obs_pred_pos[1])**2
            h = dist_sq - (const.ROBOT_RADIUS + effective_radius + const.D_SAFE)**2
            h_sequence.append(h)
            h_values_dyn_expr.append(h)

        # Apply constraints for this obstacle using slack variables
        opti.subject_to(h_sequence[0] >= S_dyn[i, 0])
        opti.subject_to(S_dyn[i, 0] >= 0)
        for k in range(1, const.N + 1):
            opti.subject_to(h_sequence[k] - (1 - const.CBF_GAMMA) * h_sequence[k-1] >= S_dyn[i, k])
            opti.subject_to(S_dyn[i, k] >= 0)

    # Static obstacles
    for i, obs in enumerate(static_obstacles):
        h_sequence = []
        for k in range(const.N + 1):
            dx = ca.fabs(X[0, k] - obs.center[0]) - obs.width / 2
            dy = ca.fabs(X[1, k] - obs.center[1]) - obs.height / 2
            h = (ca.fmax(0, dx))**2 + (ca.fmax(0, dy))**2 - (const.ROBOT_RADIUS + const.D_SAFE)**2
            h_sequence.append(h)
            h_values_stat_expr.append(h)

        # Apply constraints for this obstacle using slack variables
        opti.subject_to(h_sequence[0] >= S_stat[i, 0])
        opti.subject_to(S_stat[i, 0] >= 0)
        for k in range(1, const.N + 1):
            opti.subject_to(h_sequence[k] - (1 - const.CBF_GAMMA) * h_sequence[k-1] >= S_stat[i, k])
            opti.subject_to(S_stat[i, k] >= 0)

    # --- Other constraints ---
    # State bounds
    opti.subject_to(opti.bounded(0, X[0, :], const.X_LIM))
    opti.subject_to(opti.bounded(0, X[1, :], const.Y_LIM))
    opti.subject_to(opti.bounded(0, X[3, :], const.MAX_LINEAR_VEL)) # Linear velocity
    opti.subject_to(opti.bounded(-const.MAX_ANGULAR_VEL, X[4, :], const.MAX_ANGULAR_VEL)) # Angular velocity

    # Control input bounds (accelerations)
    opti.subject_to(opti.bounded(-const.MAX_LINEAR_ACCEL, U[0, :], const.MAX_LINEAR_ACCEL))
    opti.subject_to(opti.bounded(-const.MAX_ANGULAR_ACCEL, U[1, :], const.MAX_ANGULAR_ACCEL))

    opti.subject_to(X[:, 0] == x0)

    # --- Solver setup ---
    opti.set_value(x0, robot_state); opti.set_value(X_ref, x_ref)
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}; opti.solver('ipopt', opts)

    try:
        sol = opti.solve()
        h_dyn = [sol.value(h) for h in h_values_dyn_expr]
        h_stat = [sol.value(h) for h in h_values_stat_expr]
        return sol.value(U)[:, 0], sol.value(X), h_dyn, h_stat
    except:
        print("Solver failed. Returning zero control.")
        # Try to debug by getting the values of the variables
        h_dyn = [opti.debug.value(h) for h in h_values_dyn_expr]
        h_stat = [opti.debug.value(h) for h in h_values_stat_expr]
        return None, None, h_dyn, h_stat