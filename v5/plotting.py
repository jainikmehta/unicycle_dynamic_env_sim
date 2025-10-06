import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
import constants as const

def plot_environment(robot, goal, static_obstacles, dynamic_obstacles, 
                     predicted_trajectory=None, control_history=None, 
                     h_dyn_history=None, h_stat_history=None, save_path=None):
    """
    Plots the environment and auxiliary data, optionally saving the figure.
    """
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])

    # --- Main Simulation Plot ---
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_xlim(0, const.X_LIM)
    ax1.set_ylim(0, const.Y_LIM)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_title('NMPC Simulation with CBF Constraints')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')

    # Plot static obstacles and their safety bubbles
    for obs in static_obstacles:
        bottom_left_x = obs.center[0] - obs.width / 2
        bottom_left_y = obs.center[1] - obs.height / 2
        rect = Rectangle((bottom_left_x, bottom_left_y), obs.width, obs.height, color='gray', zorder=2)
        ax1.add_patch(rect)
        bubble_x = obs.center[0] - (obs.width / 2 + const.D_SAFE)
        bubble_y = obs.center[1] - (obs.height / 2 + const.D_SAFE)
        bubble_w = obs.width + 2 * const.D_SAFE
        bubble_h = obs.height + 2 * const.D_SAFE
        bubble = Rectangle((bubble_x, bubble_y), bubble_w, bubble_h, color='gray', linestyle='--', fill=False, zorder=1, alpha=0.5)
        ax1.add_patch(bubble)


    # Plot dynamic obstacles, their predicted paths, and safety bubbles
    for obs in dynamic_obstacles:
        x, y, theta = obs.state
        circle = Circle((x, y), radius=obs.radius, color='red', zorder=5)
        ax1.add_patch(circle)
        ax1.arrow(x, y, 1.5 * np.cos(theta), 1.5 * np.sin(theta), head_width=0.8, fc='red', ec='red', zorder=5)
        ax1.plot(obs.predicted_path[0, :], obs.predicted_path[1, :], 'r:', zorder=4)
        bubble = Circle((x, y), radius=obs.radius + const.D_SAFE, color='gray', linestyle='--', fill=False, zorder=1, alpha=0.5)
        ax1.add_patch(bubble)


    # Plot robot's full trajectory history
    traj = np.array(robot.trajectory)
    ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1.5, label='Robot Path', zorder=8)

    # Plot robot
    rx, ry, r_theta = robot.state
    robot_patch = Circle((rx, ry), radius=robot.radius, color='blue', zorder=10)
    ax1.add_patch(robot_patch)
    ax1.arrow(rx, ry, 2.0 * np.cos(r_theta), 2.0 * np.sin(r_theta), head_width=0.8, fc='blue', ec='blue', zorder=10)

    # Plot goal
    goal_patch = Circle(goal.state, radius=goal.radius, color='green', alpha=0.7, zorder=3)
    ax1.add_patch(goal_patch)
    
    # Plot predicted trajectory from NMPC
    if predicted_trajectory is not None:
        ax1.plot(predicted_trajectory[0, :], predicted_trajectory[1, :], 'm--', linewidth=2.0, label='NMPC Prediction', zorder=9)

    ax1.legend()

    # --- Control Input Plot ---
    ax2 = fig.add_subplot(gs[0, 1])
    if control_history:
        controls = np.array(control_history)
        time = np.arange(len(controls)) * const.DT
        ax2.plot(time, controls[:, 0], label='Linear Velocity')
        ax2.plot(time, controls[:, 1], label='Angular Velocity')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Control Input')
        ax2.set_title('Control Inputs vs. Time')
        ax2.legend()
        ax2.grid(True)

    # --- CBF h-value Plot ---
    ax3 = fig.add_subplot(gs[1, 1])
    if h_dyn_history and h_stat_history:
        time = np.arange(len(h_dyn_history)) * const.DT
        ax3.plot(time, h_dyn_history, label='Min h-value (Dynamic)')
        ax3.plot(time, h_stat_history, label='Min h-value (Static)')
        ax3.axhline(0, color='r', linestyle='--', label='Safety Boundary')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Min h-value')
        ax3.set_title('CBF h-values vs. Time')
        ax3.legend()
        ax3.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
    else:
        plt.show()