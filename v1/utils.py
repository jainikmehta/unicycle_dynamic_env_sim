import numpy as np
import random
import matplotlib.pyplot as plt
# Assume obstacle class and constants are defined as before

class grid_cell:
    def __init__(self, x_lim, y_lim, resolution):
        self.resolution = resolution # in meters
        self.x_lim = x_lim # in meters
        self.y_lim = y_lim # in meters
        # Calculate the number of grid cells in x and y directions
        self.grid_width = int(x_lim / resolution)
        self.grid_height = int(y_lim / resolution)
        self.num_cells = self.grid_width * self.grid_height
        # Create a 2D array to mark free cells as 0 (True) or occupied as 1 (False)
        self.grid_cells = np.zeros((self.grid_width, self.grid_height), dtype=float)
        # Mark border cells as occupied to provide a safety margin around the map
        # Use a border width of 5 cells
        border_width = 5
        bw = min(border_width, self.grid_width // 2, self.grid_height // 2)
        if bw > 0:
            # Left and right borders
            self.grid_cells[0:bw, :] = 1
            self.grid_cells[self.grid_width-bw:self.grid_width, :] = 1
            # Top and bottom borders
            self.grid_cells[:, 0:bw] = 1
            self.grid_cells[:, self.grid_height-bw:self.grid_height] = 1

    def update_grid(self, obstacle, obstacle_type = 'c'):
        # Mark the grid cells as 1 which are occupied by the circular obstacle
        
        if obstacle_type == 'c':  # circular obstacle
            x, y = obstacle.state[0], obstacle.state[1]
            radius = obstacle.radius
            # Determine the grid cell indices that the obstacle occupies
            x_start = max(0, int((x - radius) / self.resolution) - 1)
            x_end = min(self.grid_width - 1, int((x + radius) / self.resolution) + 1)
            y_start = max(0, int((y - radius) / self.resolution) - 1)
            y_end = min(self.grid_height - 1, int((y + radius) / self.resolution) + 1)
            
            for i in range(x_start, x_end + 1):
                for j in range(y_start, y_end + 1):
                        self.grid_cells[i, j] = 1  # Mark as occupied

        if obstacle_type == 'r':  # rectangular obstacle
            x, y = obstacle.x, obstacle.y
            width = obstacle.width
            height = obstacle.height
            # Determine the grid cell indices that the obstacle occupies
            x_start = max(0, int((x - width/2) / self.resolution) - 1)
            x_end = min(self.grid_width - 1, int((x + width/2) / self.resolution) + 1)
            y_start = max(0, int((y - height/2) / self.resolution) - 1)
            y_end = min(self.grid_height - 1, int((y + height/2) / self.resolution) + 1)
            
            for i in range(x_start, x_end + 1):
                for j in range(y_start, y_end + 1):
                        self.grid_cells[i, j] = 1  # Mark as occupied

    def plot_cells(self):
        plt.imshow(self.grid_cells.T, origin='lower', cmap='Greys', extent=(0, self.x_lim, 0, self.y_lim))
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Grid Cells (0: Free, 1: Occupied)')
        plt.colorbar(label='Cell Status')
        plt.grid(False)
        plt.show()