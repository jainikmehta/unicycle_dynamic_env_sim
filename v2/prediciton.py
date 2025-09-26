
# Using dynamics.py predict the motion of each dynamic obstacle k over a prediction horizon N given their current state and control inputs. Return dictionary of predicted states and associated uncertainty for each obstacle.

import casadi as ca
from dynamics import obstacle_dyanmics

def predict_obstacle_motion(obstacle_states, obstacle_controls, K, N, DT):
    prediction = {} # Dictionary to hold predictions for each obstacle
    for k in range(K):  # For each dynamic obstacle
        x_k = obstacle_states[k]  # Current state of obstacle k
        u_k = obstacle_controls[k]  # Control inputs for obstacle k
        preds = []  # List to hold predicted states, starting with current state. preds[0] = [mu_x, m_y, theta0]
        for t in range(N):  # For each time step in the prediction horizon
            x_k = obstacle_dyanmics(x_k, u_k, DT)  # Predict next state using dynamics function
            preds.append(x_k)  # Append predicted state to list
        preds = ca.hcat(preds)  # Concatenate list of states into a matrix
        uncertainty = ca.DM.eye(3) * 0.1 * (t + 1)  # Simple model: uncertainty grows linearly with time
        # Uncertainty is only in position (x, y), not in heading (theta)
        uncertainty[2, 2] = 0.0  # No uncertainty in heading
        prediction[k] = {'states': preds, 'uncertainty': uncertainty}  # Store predictions and uncertainty in dictionary
        return prediction