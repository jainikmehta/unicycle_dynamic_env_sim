import casadi as ca
import numpy as np
import constants as const

def nmpc_solver(robot_state, x_ref, dynamic_obstacles, static_obstacles):
    opti = ca.Opti()
    
    # --- Decision variables ---
    X = opti.variable(3, const.N + 1)
    U = opti.variable(2, const.N)
    S_dyn = opti.variable(len(dynamic_obstacles), const.N + 1) 
    S_stat = opti.variable(len(static_obstacles), const.N + 1)

    # --- Parameters ---
    x0 = opti.parameter(3, 1)
    X_ref = opti.parameter(2, const.N + 1) # Reference path from RRT*

    # --- Cost function ---
    cost = 0
    for k in range(const.N):
        # Path tracking cost
        error = X[:2, k] - X_ref[:, k]
        cost += ca.mtimes([error.T, const.Q_path, error])
        # Control cost
        cost += ca.mtimes([U[:, k].T, const.R, U[:, k]])
    
    cost += const.SLACK_PENALTY * (ca.sumsqr(S_dyn) + ca.sumsqr(S_stat))
    opti.minimize(cost)

    # --- Dynamics constraints ---
    for k in range(const.N):
        opti.subject_to(X[0, k+1] == X[0, k] + const.DT * U[0, k] * ca.cos(X[2, k]))
        opti.subject_to(X[1, k+1] == X[1, k] + const.DT * U[0, k] * ca.sin(X[2, k]))
        opti.subject_to(X[2, k+1] == X[2, k] + const.DT * U[1, k])

    # --- CBF Constraints (Unchanged) ---
    for k in range(const.N + 1):
        for i, obs in enumerate(dynamic_obstacles):
            obs_pred_x = obs.state[0] + k * const.DT * obs.velocity * np.cos(obs.state[2])
            obs_pred_y = obs.state[1] + k * const.DT * obs.velocity * np.sin(obs.state[2])
            h = (X[0, k] - obs_pred_x)**2 + (X[1, k] - obs_pred_y)**2 - (const.ROBOT_RADIUS + obs.radius + const.D_SAFE)**2
            opti.subject_to(h >= -S_dyn[i, k]); opti.subject_to(S_dyn[i,k] >= 0)
        for i, obs in enumerate(static_obstacles):
            dx = ca.fabs(X[0, k] - obs.center[0]) - obs.width / 2
            dy = ca.fabs(X[1, k] - obs.center[1]) - obs.height / 2
            h = (ca.fmax(0, dx))**2 + (ca.fmax(0, dy))**2 - (const.ROBOT_RADIUS + const.D_SAFE)**2
            opti.subject_to(h >= -S_stat[i, k]); opti.subject_to(S_stat[i,k] >= 0)

    # --- Other constraints ---
    opti.subject_to(opti.bounded(0, X[0, :], const.X_LIM))
    opti.subject_to(opti.bounded(0, X[1, :], const.Y_LIM))
    opti.subject_to(opti.bounded(0, U[0, :], const.MAX_LINEAR_VEL))
    opti.subject_to(opti.bounded(-const.MAX_ANGULAR_VEL, U[1, :], const.MAX_ANGULAR_VEL))
    opti.subject_to(X[:, 0] == x0)

    # --- Solver setup ---
    opti.set_value(x0, robot_state)
    opti.set_value(X_ref, x_ref)
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver('ipopt', opts)

    try:
        sol = opti.solve()
        return sol.value(U)[:, 0], sol.value(X)
    except:
        print("Solver failed. Returning zero control.")
        return np.zeros(2), np.tile(robot_state, (1, const.N + 1)).T