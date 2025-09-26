import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

def plot_environment(robot, goal, static_obstacles, dynamic_obstacles, duration=None):
    """
    Plots the environment with all objects.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_aspect('equal')
    ax.grid(True)

    # Plot static obstacles (rectangles)
    for obs in static_obstacles:
        # The stored x,y is the center, so we calculate the bottom-left corner for the patch
        bottom_left_x = obs.center[0] - obs.width / 2
        bottom_left_y = obs.center[1] - obs.height / 2
        rect = Rectangle((bottom_left_x, bottom_left_y), obs.width, obs.height, color='gray')
        ax.add_patch(rect)

    # Plot dynamic obstacles (red circles with heading)
    for obs in dynamic_obstacles:
        x, y, theta = obs.state
        circle = Circle((x, y), radius=obs.radius, color='red')
        ax.add_patch(circle)
        ax.arrow(x, y, 1.5 * np.cos(theta), 1.5 * np.sin(theta), head_width=0.8, fc='red', ec='red')

    # Plot robot (blue circle with heading)
    rx, ry, r_theta = robot.state
    robot_patch = Circle((rx, ry), radius=robot.radius, color='blue', zorder=10)
    ax.add_patch(robot_patch)
    ax.arrow(rx, ry, 1.5 * np.cos(r_theta), 1.5 * np.sin(r_theta), head_width=0.8, fc='blue', ec='blue', zorder=10)

    # Plot goal (green circle)
    goal_patch = Circle(goal.state, radius=goal.radius, color='green', alpha=0.7)
    ax.add_patch(goal_patch)

    plt.title('Simulation Environment')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    if duration:
        plt.show(block=False)
        plt.pause(duration)
        plt.close(fig)
    else:
        plt.show()