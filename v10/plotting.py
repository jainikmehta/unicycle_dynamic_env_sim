import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
import constants as const

def plot_environment(robot, goal, static_obstacles, dynamic_obstacles, 
                     predicted_trajectory=None, rrt_path=None, 
                     control_history=None, h_dyn_history=None, h_stat_history=None, 
                     intermediate_goal=None, save_path=None):
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])

    # --- Main Simulation Plot ---
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_xlim(0, const.X_LIM); ax1.set_ylim(0, const.Y_LIM)
    ax1.set_aspect('equal'); ax1.grid(True)
    ax1.set_title('NMPC with Smart Intermediate Goal'); ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')

    if rrt_path is not None:
        ax1.plot(rrt_path[:, 0], rrt_path[:, 1], 'g--', lw=2, label='RRT* Path', zorder=7)

    # --- MODIFICATION START ---
    # Plot Intermediate Goal
    if intermediate_goal is not None:
        ax1.plot(intermediate_goal[0], intermediate_goal[1], 'y*', markersize=15, label='Intermediate Goal', zorder=11)
    # --- MODIFICATION END ---

    for obs in static_obstacles:
        rect = Rectangle((obs.center[0]-obs.width/2, obs.center[1]-obs.height/2), obs.width, obs.height, color='gray', zorder=2)
        ax1.add_patch(rect)
        bubble = Rectangle((obs.center[0]-(obs.width/2+const.D_SAFE), obs.center[1]-(obs.height/2+const.D_SAFE)), 
                           obs.width+2*const.D_SAFE, obs.height+2*const.D_SAFE, 
                           color='gray', ls='--', fill=False, zorder=1, alpha=0.5)
        ax1.add_patch(bubble)
    
    has_dyn_obs_labels = False
    for obs in dynamic_obstacles:
        x, y, theta = obs.measured_state
        circle = Circle((x, y), radius=obs.radius, color='red', zorder=5)
        ax1.add_patch(circle)
        ax1.arrow(x, y, 1.5*np.cos(theta), 1.5*np.sin(theta), head_width=0.8, fc='red', ec='red', zorder=5)

        true_traj = np.array(obs.true_trajectory)
        meas_traj = np.array(obs.measured_trajectory)
        ax1.plot(true_traj[:, 0], true_traj[:, 1], 'r-', lw=1.5, label='Actual Obs Path' if not has_dyn_obs_labels else "", zorder=6)
        ax1.plot(meas_traj[:, 0], meas_traj[:, 1], 'r:', lw=1.5, label='Measured Obs Path' if not has_dyn_obs_labels else "", zorder=6)
        has_dyn_obs_labels = True
        
        plot_indices = [0, 10, 20]
        for i in plot_indices:
            if i < len(obs.predicted_path[0]):
                cov = obs.predicted_cov[i]
                uncertainty_radius_upper = const.SIGMA_BOUND_UPPER * np.sqrt(np.trace(cov))
                effective_radius_upper = obs.radius + uncertainty_radius_upper + const.D_SAFE
                disk = Circle(obs.predicted_path[:, i], radius=effective_radius_upper, 
                              color='orange', fill=True, zorder=3, alpha=0.7 - (i*0.02))
                ax1.add_patch(disk)

                uncertainty_radius_updated = obs.sigma_bounds[i] * np.sqrt(np.trace(cov))
                effective_radius_updated = obs.radius + uncertainty_radius_updated + const.D_SAFE
                bubble = Circle(obs.predicted_path[:, i], radius=effective_radius_updated, 
                                color='black', ls='-', fill=False, zorder=4)
                ax1.add_patch(bubble)

    traj = np.array(robot.trajectory)
    ax1.plot(traj[:, 0], traj[:, 1], 'b-', lw=1.5, label='Robot Path', zorder=8)
    rx, ry, r_theta = robot.state[:3]
    robot_patch = Circle((rx, ry), radius=robot.radius, color='blue', zorder=10)
    ax1.add_patch(robot_patch)
    ax1.arrow(rx, ry, 2.0*np.cos(r_theta), 2.0*np.sin(r_theta), head_width=0.8, fc='blue', ec='blue', zorder=10)
    goal_patch = Circle(goal.state, radius=goal.radius, color='green', alpha=0.7, zorder=3)
    ax1.add_patch(goal_patch)
    if predicted_trajectory is not None:
        ax1.plot(predicted_trajectory[0, :], predicted_trajectory[1, :], 'm--', lw=2, label='NMPC Prediction', zorder=9)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    if control_history:
        controls = np.array(control_history)
        time = np.arange(len(controls)) * const.DT
        ax2.plot(time, controls[:, 0], label='Linear Accel.')
        ax2.plot(time, controls[:, 1], label='Angular Accel.')
        ax2.set_title('Control Inputs vs. Time'); ax2.legend(); ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 1])
    if h_dyn_history and h_stat_history:
        time = np.arange(len(h_dyn_history)) * const.DT
        ax3.plot(time, h_dyn_history, color='red', label='Min h (Dynamic)')
        ax3.plot(time, h_stat_history, color='black', label='Min h (Static)')
        ax3.axhline(0, color='r', ls='--', label='Safety Boundary')
        ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Min h-value')
        ax3.set_title('CBF h-values vs. Time'); ax3.legend(); ax3.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100); plt.close(fig)
    else:
        plt.show()