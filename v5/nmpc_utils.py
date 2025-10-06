import casadi as ca
import numpy as np
import constants as const

def nmpc_solver(robot_state, goal_state, dynamic_obstacles, static_obstacles):
    """
    Solves the NMPC problem with a penalty for being "stuck".
    """
    opti = ca.Opti()
    
    # --- Decision variables ---
    X = opti.variable(3, const.N + 1)
    U = opti.variable(2, const.N)
    S_dyn = opti.variable(len(dynamic_obstacles), const.N + 1) 
    S_stat = opti.variable(len(static_obstacles), const.N + 1)

    # --- Parameters ---
    x0 = opti.parameter(3, 1)
    goal = opti.parameter(2, 1)

    # --- Cost function ---
    cost = 0
    stuck_penalty_cost = 0
    for k in range(const.N):
        # Goal cost (zero inside the goal radius)
        dist_to_goal = ca.norm_2(X[:2, k] - goal)
        goal_cost = ca.fmax(0, dist_to_goal - const.GOAL_RADIUS)**2
        cost += const.Q[0, 0] * goal_cost

        # Control cost
        cost += ca.mtimes([U[:, k].T, const.R, U[:, k]])

        # Penalty for not moving away from the starting point of the horizon
        dist_from_start = ca.norm_2(X[:2, k+1] - x0[:2])
        stuck_penalty = ca.fmax(0, const.STUCK_DISTANCE_THRESHOLD - dist_from_start)**2
        stuck_penalty_cost += stuck_penalty

    cost += const.SLACK_PENALTY * (ca.sumsqr(S_dyn) + ca.sumsqr(S_stat))
    cost += const.STUCK_PENALTY * stuck_penalty_cost
    opti.minimize(cost)

    # --- Dynamics constraints ---
    for k in range(const.N):
        opti.subject_to(X[0, k+1] == X[0, k] + const.DT * U[0, k] * ca.cos(X[2, k]))
        opti.subject_to(X[1, k+1] == X[1, k] + const.DT * U[0, k] * ca.sin(X[2, k]))
        opti.subject_to(X[2, k+1] == X[2, k] + const.DT * U[1, k])

    # --- CBF Constraints ---
    h_values_dyn_expr = []
    h_values_stat_expr = []
    for k in range(const.N + 1):
        # Dynamic obstacles
        for i, obs in enumerate(dynamic_obstacles):
            obs_pred_x = obs.state[0] + k * const.DT * obs.velocity * np.cos(obs.state[2])
            obs_pred_y = obs.state[1] + k * const.DT * obs.velocity * np.sin(obs.state[2])
            
            h = (X[0, k] - obs_pred_x)**2 + (X[1, k] - obs_pred_y)**2 - (const.ROBOT_RADIUS + obs.radius + const.D_SAFE)**2
            opti.subject_to(h >= -S_dyn[i, k])
            opti.subject_to(S_dyn[i, k] >= 0)
            h_values_dyn_expr.append(h)

        # Static obstacles
        for i, obs in enumerate(static_obstacles):
            dx = ca.fabs(X[0, k] - obs.center[0]) - obs.width / 2
            dy = ca.fabs(X[1, k] - obs.center[1]) - obs.height / 2
            
            h = (ca.fmax(0, dx))**2 + (ca.fmax(0, dy))**2 - (const.ROBOT_RADIUS + const.D_SAFE)**2
            opti.subject_to(h >= -S_stat[i, k])
            opti.subject_to(S_stat[i, k] >= 0)
            h_values_stat_expr.append(h)

    # --- Other constraints ---
    opti.subject_to(opti.bounded(0, X[0, :], const.X_LIM))
    opti.subject_to(opti.bounded(0, X[1, :], const.Y_LIM))
    opti.subject_to(opti.bounded(0, U[0, :], const.MAX_LINEAR_VEL))
    opti.subject_to(opti.bounded(-const.MAX_ANGULAR_VEL, U[1, :], const.MAX_ANGULAR_VEL))
    opti.subject_to(X[:, 0] == x0)

    # --- Solver setup ---
    opti.set_value(x0, robot_state)
    opti.set_value(goal, goal_state)
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver('ipopt', opts)

    try:
        sol = opti.solve()
        h_dyn_vals = [sol.value(h) for h in h_values_dyn_expr]
        h_stat_vals = [sol.value(h) for h in h_values_stat_expr]
        return sol.value(U)[:, 0], sol.value(X), h_dyn_vals, h_stat_vals
    except:
        print("Solver failed. Returning zero control.")
        return np.zeros(2), np.tile(robot_state, (1, const.N + 1)).T, [-1], [-1]