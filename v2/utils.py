import numpy as np
import random
import matplotlib.pyplot as plt

class Grid:
    def __init__(self, x_lim, y_lim, resolution, border_width=1):
        self.resolution = resolution
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.grid_width = int(x_lim / resolution)
        self.grid_height = int(y_lim / resolution)
        self.cells = np.zeros((self.grid_width, self.grid_height), dtype=int)
        
        # Initialize with a border buffer to prevent spawning near walls
        if border_width > 0:
            self.cells[0:border_width, :] = 1  # Left
            self.cells[-border_width:, :] = 1 # Right
            self.cells[:, 0:border_width] = 1  # Bottom
            self.cells[:, -border_width:] = 1 # Top

    def mark_rect_as_occupied(self, x_start, y_start, w_cells, h_cells, safety_dist):
        """Marks a rectangle plus a safety buffer as occupied."""
        buffer_cells = int(np.ceil(safety_dist / self.resolution))
        
        x_start_buf = max(0, x_start - buffer_cells)
        y_start_buf = max(0, y_start - buffer_cells)
        x_end_buf = min(self.grid_width, x_start + w_cells + buffer_cells)
        y_end_buf = min(self.grid_height, y_start + h_cells + buffer_cells)
        
        self.cells[x_start_buf:x_end_buf, y_start_buf:y_end_buf] = 1

    def mark_circle_as_occupied(self, x, y, radius, safety_dist):
        """Marks a circular area plus a safety buffer as occupied."""
        total_radius = radius + safety_dist
        x_min_cell = int((x - total_radius) / self.resolution)
        x_max_cell = int((x + total_radius) / self.resolution)
        y_min_cell = int((y - total_radius) / self.resolution)
        y_max_cell = int((y + total_radius) / self.resolution)

        for i in range(max(0, x_min_cell), min(self.grid_width, x_max_cell + 1)):
            for j in range(max(0, y_min_cell), min(self.grid_height, y_max_cell + 1)):
                self.cells[i, j] = 1
    
    def find_free_rect_space(self, w_cells, h_cells):
        """Finds a top-left cell index for a free rectangular area."""
        candidates = []
        for r in range(self.grid_width - w_cells + 1):
            for c in range(self.grid_height - h_cells + 1):
                if not np.any(self.cells[r:r+w_cells, c:c+h_cells]):
                    candidates.append((r, c))
        return random.choice(candidates) if candidates else None

    def find_random_free_cell(self):
        """Finds the indices of a random free cell."""
        free_cells = np.argwhere(self.cells == 0)
        return random.choice(free_cells) if len(free_cells) > 0 else None

    def plot(self):
        """Utility to visualize the grid occupancy."""
        plt.imshow(self.cells.T, origin='lower', cmap='Greys',
                   extent=(0, self.x_lim, 0, self.y_lim))
        plt.title('Grid Occupancy')
        plt.show()