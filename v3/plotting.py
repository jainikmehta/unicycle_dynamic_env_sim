import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

def plot_environment(robot, goal, static_obstacles, dynamic_obstacles, predicted_trajectory=None, save_path=None):
    """
    Plots the environment, optionally including the predicted trajectory and saving the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_aspect('equal')
    ax.grid(True)

    # Plot static obstacles
    for obs in static_obstacles:
        bottom_left_x = obs.center[0] - obs.width / 2
        bottom_left_y = obs.center[1] - obs.height / 2
        rect = Rectangle((bottom_left_x, bottom_left_y), obs.width, obs.height, color='gray', zorder=2)
        ax.add_patch(rect)

    # Plot dynamic obstacles
    for obs in dynamic_obstacles:
        x, y, theta = obs.state
        circle = Circle((x, y), radius=obs.radius, color='red', zorder=5)
        ax.add_patch(circle)
        ax.arrow(x, y, 1.5 * np.cos(theta), 1.5 * np.sin(theta), head_width=0.8, fc='red', ec='red', zorder=5)

    # Plot robot's full trajectory history
    traj = np.array(robot.trajectory)
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1.5, label='Robot Path', zorder=8)

    # Plot robot
    rx, ry, r_theta = robot.state
    robot_patch = Circle((rx, ry), radius=robot.radius, color='blue', zorder=10)
    ax.add_patch(robot_patch)
    ax.arrow(rx, ry, 2.0 * np.cos(r_theta), 2.0 * np.sin(r_theta), head_width=0.8, fc='blue', ec='blue', zorder=10)

    # Plot goal
    goal_patch = Circle(goal.state, radius=goal.radius, color='green', alpha=0.7, zorder=3)
    ax.add_patch(goal_patch)
    
    # Plot predicted trajectory from NMPC
    if predicted_trajectory is not None:
        ax.plot(predicted_trajectory[0, :], predicted_trajectory[1, :], 'm--', linewidth=2.0, label='NMPC Prediction', zorder=9)

    ax.set_title('NMPC Simulation with CBF Constraints')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
    else:
        plt.show()