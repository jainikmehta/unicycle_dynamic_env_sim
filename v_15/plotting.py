"""
plotting.py

Recreates the multi-panel simulation plot.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Ellipse
import constants as const
import os

def plot_simulation_step(robot, goal, static_obstacles, dynamic_obstacles, 
                         predicted_trajectory, rrt_path, 
                         control_history, h_dyn_history, h_stat_history, 
                         t, save_path=None):
    
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])

    # --- Main Simulation Plot (ax1) ---
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_xlim(0, const.X_LIM)
    ax1.set_ylim(0, const.Y_LIM)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title(f'NMPC with 2nd-Order CBF (t={t*const.DT:.1f}s)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')

    # Plot RRT* Path
    if rrt_path is not None:
        ax1.plot(rrt_path[:, 0], rrt_path[:, 1], 'g--', lw=2, label='RRT* Path', zorder=7)

    # Plot Static Obstacles
    for obs in static_obstacles:
        # Core rectangle
        rect = Rectangle((obs.center[0]-obs.width/2, obs.center[1]-obs.height/2), 
                         obs.width, obs.height, color='gray', zorder=2)
        ax1.add_patch(rect)
        # Safety bubble
        bubble = Rectangle((obs.center[0]-(obs.width/2+const.D_SAFE), obs.center[1]-(obs.height/2+const.D_SAFE)), 
                           obs.width+2*const.D_SAFE, obs.height+2*const.D_SAFE, 
                           color='gray', ls='--', fill=False, zorder=1, alpha=0.5)
        ax1.add_patch(bubble)
    
    # Plot Dynamic Obstacles
    has_dyn_obs_labels = False
    for obs in dynamic_obstacles:
        # Measured position (circle) and velocity (arrow)
        x, y, theta = obs.measured_state
        circle = Circle((x, y), radius=obs.radius, color='red', zorder=5)
        ax1.add_patch(circle)
        ax1.arrow(x, y, 1.5*np.cos(theta), 1.5*np.sin(theta), head_width=0.8, fc='red', ec='red', zorder=5)

        # Trajectories
        true_traj = np.array(obs.true_trajectory)
        meas_traj = np.array(obs.measured_trajectory)
        ax1.plot(true_traj[:, 0], true_traj[:, 1], 'r-', lw=1.5, label='Actual Obs Path' if not has_dyn_obs_labels else "", zorder=6)
        ax1.plot(meas_traj[:, 0], meas_traj[:, 1], 'r:', lw=1.5, label='Measured Obs Path' if not has_dyn_obs_labels else "", zorder=6)
        has_dyn_obs_labels = True
        
        # Plot predicted uncertainty bubbles at key steps
        plot_indices = [0, 5, 10, 15, 20] 
        for i in plot_indices:
            if i < len(obs.predicted_path[0]):
                pred_pos = obs.predicted_path[:, i]
                cov = obs.predicted_cov[i]
                sigma_bound = obs.sigma_bounds[i]
                
                # Use trace for a simple circular bound
                uncertainty_radius = sigma_bound * np.sqrt(np.trace(cov))
                effective_radius = obs.radius + uncertainty_radius + const.D_SAFE
                
                bubble = Circle(pred_pos, radius=effective_radius, 
                                color='orange', ls='-', fill=False, zorder=4, alpha=0.8 - i*0.03)
                ax1.add_patch(bubble)
                
                # Plot covariance ellipse
                # width, height, angle = cov_to_ellipse(cov * (sigma_bound**2))
                # ellipse = Ellipse(xy=pred_pos, width=width, height=height, angle=angle,
                #                   edgecolor='red', fc='None', ls='--', zorder=4)
                # ax1.add_patch(ellipse)


    # Plot Robot
    traj = np.array(robot.trajectory)
    ax1.plot(traj[:, 0], traj[:, 1], 'b-', lw=1.5, label='Robot Path', zorder=8)
    rx, ry, r_theta = robot.state[:3]
    robot_patch = Circle((rx, ry), radius=robot.radius, color='blue', zorder=10)
    ax1.add_patch(robot_patch)
    ax1.arrow(rx, ry, 2.0*np.cos(r_theta), 2.0*np.sin(r_theta), head_width=0.8, fc='blue', ec='blue', zorder=10)
    
    # Plot Goal
    goal_patch = Circle(goal.state, radius=goal.radius, color='green', alpha=0.7, zorder=3)
    ax1.add_patch(goal_patch)
    
    # Plot NMPC Prediction
    if predicted_trajectory is not None:
        ax1.plot(predicted_trajectory[0, :], predicted_trajectory[1, :], 'm--', lw=2, label='NMPC Prediction', zorder=9)
    
    ax1.legend(loc="upper right")

    # --- Control Input Plot (ax2) ---
    ax2 = fig.add_subplot(gs[0, 1])
    if control_history:
        controls = np.array(control_history)
        time = np.arange(len(controls)) * const.DT
        ax2.plot(time, controls[:, 0], label='Linear Accel (m/s^2)')
        ax2.plot(time, controls[:, 1], label='Angular Accel (rad/s^2)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Control Input')
        ax2.set_title('Control Inputs vs. Time')
        ax2.legend()
        ax2.grid(True)

    # --- CBF h-value Plot (ax3) ---
    ax3 = fig.add_subplot(gs[1, 1])
    if h_dyn_history and h_stat_history:
        time = np.arange(len(h_dyn_history)) * const.DT
        min_h_dyn = [min(h) if h else 0 for h in h_dyn_history]
        min_h_stat = [min(h) if h else 0 for h in h_stat_history]
        
        ax3.plot(time, min_h_dyn, color='red', label='Min $h$ (Dynamic)')
        ax3.plot(time, min_h_stat, color='black', label='Min $h$ (Static)')
        ax3.axhline(0, color='r', ls='--', label='Safety Boundary ($h=0$)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Min $h$-value')
        ax3.set_title('Min CBF $h$-values vs. Time')
        ax3.legend()
        ax3.grid(True)
        ax3.set_ylim(bottom=min(np.min(min_h_dyn), np.min(min_h_stat), -1.0) - 1.0, 
                     top=max(np.max(min_h_dyn), np.max(min_h_stat), 1.0) + 1.0)


    plt.tight_layout()

    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
    else:
        plt.show()

# Helper function to convert covariance to ellipse parameters
def cov_to_ellipse(cov):
    """Convert 2x2 covariance matrix to ellipse parameters for plotting."""
    try:
        eigenvals, eigenvecs = np.linalg.eig(cov)
    except np.linalg.LinAlgError:
        return 0, 0, 0
        
    angle = np.degrees(np.arctan2(*eigenvecs[:,0][::-1]))
    # Width and height are 2*std_dev
    width, height = 2 * np.sqrt(eigenvals) 
    return width, height, angle