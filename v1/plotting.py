import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation


def plot_environment(robot, goal, static_obstacles, dynamic_obstacles, predictions=None, duration=None):
    """Plot the environment once.

    If duration (seconds) is provided, the plot will be shown non-blocking for that
    many seconds and then closed automatically. If duration is None, the plot will
    block until the window is closed by the user.
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_aspect('equal')

    # Plot static obstacles
    for obs in static_obstacles:
        rect = plt.Rectangle((obs.x, obs.y), obs.width, obs.height, color='gray')
        ax.add_patch(rect)

    # Plot dynamic obstacles
    dynamic_patches = []
    for obs in dynamic_obstacles:
        x = obs.state[0]
        y = obs.state[1]
        theta = obs.state[2]
        circle = Circle((x,y), radius=1.0, color='red')
        dynamic_patches.append(circle)
        ax.add_patch(circle)
        # Heading arrow
        ax.arrow(x, y, 1.5 * np.cos(theta), 1.5 * np.sin(theta), head_width=1.0, head_length=1.0, fc='red', ec='red')


    # Plot robot
    robot_patch = Circle((robot.state[0], robot.state[1]), radius=robot.radius, color='blue')
    ax.add_patch(robot_patch)
    # Robot heading arrow
    ax.arrow(robot.state[0], robot.state[1], 1.5 * np.cos(robot.state[2]), 1.5 * np.sin(robot.state[2]), head_width=1.0, head_length=1.0, fc='blue', ec='blue')

    # plot goal
    goal_patch = Circle((goal.x, goal.y), radius=goal.radius, color='green', alpha=0.5)
    ax.add_patch(goal_patch)

    # # Plot predictions if available
    # if predictions:
    #     for pred in predictions:
    #         pred_circle = Circle((pred['x'], pred['y']), radius=1.0, color='orange', alpha=0.5)
    #         ax.add_patch(pred_circle)

    plt.title('Simulation Environment')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid()

    if duration is None:
        # Blocking show until user closes the window
        plt.show()
    else:
        # Non-blocking show for `duration` seconds, then close
        plt.show(block=False)
        # Small pause to ensure the figure renders before the long pause
        plt.pause(0.001)
        plt.pause(duration)
        plt.close(fig)

