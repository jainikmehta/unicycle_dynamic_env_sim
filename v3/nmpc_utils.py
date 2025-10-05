import casadi as ca
import numpy as np

# --- NMPC & Simulation Parameters ---
DT = 0.1
N = 20
Q = np.diag([2.0, 2.0, 0.1])
R = np.diag([0.1, 0.05])
SLACK_PENALTY = 1000

# --- Robot & Environment Parameters ---
ROBOT_RADIUS = 1.0
MAX_LINEAR_VEL = 3.0
MAX_ANGULAR_VEL = np.pi / 2
X_LIM = 50.0
Y_LIM = 50.0
D_SAFE = 3

def nmpc_solver(robot_state, goal_state, dynamic_obstacles, static_obstacles):
    """
    Solves the Non-linear Model Predictive Control problem with corrected CBF.
    """
    opti = ca.Opti()
    
    # --- Decision variables ---
    X = opti.variable(3, N + 1)
    U = opti.variable(2, N)
    S_dyn = opti.variable(len(dynamic_obstacles), N + 1) 
    S_stat = opti.variable(len(static_obstacles), N + 1)

    # --- Parameters ---
    x0 = opti.parameter(3, 1)
    goal = opti.parameter(2, 1)

    # --- Cost function ---
    cost = 0
    for k in range(N):
        cost += ca.mtimes([(X[:2, k] - goal).T, Q[:2, :2], (X[:2, k] - goal)])
        cost += ca.mtimes([U[:, k].T, R, U[:, k]])
    cost += SLACK_PENALTY * (ca.sumsqr(S_dyn) + ca.sumsqr(S_stat))
    opti.minimize(cost)

    # --- Dynamics constraints ---
    for k in range(N):
        opti.subject_to(X[0, k+1] == X[0, k] + DT * U[0, k] * ca.cos(X[2, k]))
        opti.subject_to(X[1, k+1] == X[1, k] + DT * U[0, k] * ca.sin(X[2, k]))
        opti.subject_to(X[2, k+1] == X[2, k] + DT * U[1, k])

    # --- CBF Constraints (with corrections) ---
    for k in range(N + 1):
        # Dynamic obstacles
        for i, obs in enumerate(dynamic_obstacles):
            obs_pred_x = obs.state[0] + k * DT * obs.velocity * np.cos(obs.state[2])
            obs_pred_y = obs.state[1] + k * DT * obs.velocity * np.sin(obs.state[2])
            
            h = (X[0, k] - obs_pred_x)**2 + (X[1, k] - obs_pred_y)**2 - (ROBOT_RADIUS + obs.radius + D_SAFE)**2
            opti.subject_to(h >= S_dyn[i, k])
            opti.subject_to(S_dyn[i, k] >= 0)

        # Static obstacles
        for i, obs in enumerate(static_obstacles):
            dx = ca.fabs(X[0, k] - obs.center[0]) - obs.width / 2
            dy = ca.fabs(X[1, k] - obs.center[1]) - obs.height / 2
            
            # --- THIS IS THE CORRECTED LINE ---
            # It now includes D_SAFE to create a proper buffer zone.
            h = (ca.fmax(0, dx))**2 + (ca.fmax(0, dy))**2 - (ROBOT_RADIUS + D_SAFE)**2
            # opti.subject_to(h >= 0)
            opti.subject_to(h >= S_stat[i, k])
            opti.subject_to(S_stat[i, k] >= 0)

    # --- Other constraints ---
    opti.subject_to(opti.bounded(0, X[0, :], X_LIM))
    opti.subject_to(opti.bounded(0, X[1, :], Y_LIM))
    opti.subject_to(opti.bounded(0, U[0, :], MAX_LINEAR_VEL))
    opti.subject_to(opti.bounded(-MAX_ANGULAR_VEL, U[1, :], MAX_ANGULAR_VEL))
    opti.subject_to(X[:, 0] == x0)

    # --- Solver setup ---
    opti.set_value(x0, robot_state)
    opti.set_value(goal, goal_state)
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver('ipopt', opts)

    try:
        sol = opti.solve()
        return sol.value(U)[:, 0], sol.value(X)
    except:
        print("Solver failed. Returning zero control.")
        return np.zeros(2), np.tile(robot_state, (1, N + 1)).T