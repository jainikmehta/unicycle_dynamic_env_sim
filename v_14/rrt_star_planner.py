"""
rrt_star_planner.py

A clean implementation of the RRT* pathfinding algorithm.
"""

import numpy as np
import random
import constants as const

class RRTStar:
    class Node:
        def __init__(self, p):
            self.p = np.array(p)  # Position [x, y]
            self.parent = None
            self.cost = 0.0

    def __init__(self, start, goal, obstacles_static, bounds,
                 max_iter=const.RRT_MAX_ITER, 
                 step_size=const.RRT_STEP_SIZE, 
                 search_radius=const.RRT_SEARCH_RADIUS, 
                 goal_sample_rate=const.RRT_GOAL_SAMPLE_RATE):
        
        self.start = self.Node(start)
        self.goal = self.Node(goal)
        self.bounds = [bounds[0], bounds[1], bounds[2], bounds[3]] # [xmin, xmax, ymin, ymax]
        self.static_obs = obstacles_static
        
        self.max_iter = max_iter
        self.step_size = step_size
        self.search_radius = search_radius
        self.goal_sample_rate = goal_sample_rate
        
        self.node_list = [self.start]

    def plan(self):
        """Run the RRT* planning algorithm."""
        for _ in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_node_idx = self.get_nearest_node_index(rnd_node)
            nearest_node = self.node_list[nearest_node_idx]
            
            new_node = self.steer(nearest_node, rnd_node)
            
            if self.is_collision_free(new_node.p):
                near_node_indices = self.find_near_node_indices(new_node)
                self.choose_parent(new_node, near_node_indices)
                
                if new_node.parent:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_node_indices)

            if self.is_near_goal(self.node_list[-1]):
                break
        
        return self.generate_final_path()

    def steer(self, from_node, to_node):
        """Create a new node by steering from `from_node` towards `to_node`."""
        dist = np.linalg.norm(to_node.p - from_node.p)
        
        new_node = self.Node(from_node.p.copy())
        
        if dist <= self.step_size:
            new_node.p = to_node.p
            new_node.cost = from_node.cost + dist
        else:
            direction = (to_node.p - from_node.p) / dist
            new_node.p = from_node.p + direction * self.step_size
            new_node.cost = from_node.cost + self.step_size
            
        new_node.parent = from_node
        return new_node

    def choose_parent(self, new_node, near_node_indices):
        """Find the best parent for `new_node` from the `near_node_indices`."""
        min_cost = new_node.cost
        best_parent = new_node.parent

        for idx in near_node_indices:
            near_node = self.node_list[idx]
            cost = near_node.cost + np.linalg.norm(new_node.p - near_node.p)
            
            if cost < min_cost and self.is_collision_free(new_node.p, near_node.p):
                min_cost = cost
                best_parent = near_node
        
        new_node.parent = best_parent
        new_node.cost = min_cost

    def rewire(self, new_node, near_node_indices):
        """Rewire the tree to check if `new_node` is a better parent for neighbors."""
        for idx in near_node_indices:
            near_node = self.node_list[idx]
            cost = new_node.cost + np.linalg.norm(new_node.p - near_node.p)
            
            if cost < near_node.cost and self.is_collision_free(new_node.p, near_node.p):
                near_node.parent = new_node
                near_node.cost = cost

    def generate_final_path(self):
        """Backtrack from the goal to the start to get the final path."""
        last_node = self.get_best_goal_node()
        if last_node is None:
            return None # No path found
        
        path = [self.goal.p]
        node = last_node
        while node.parent is not None:
            path.append(node.p)
            node = node.parent
        path.append(self.start.p)
        return np.array(path[::-1]) # Return in order [start...goal]

    def get_random_node(self):
        if random.random() > self.goal_sample_rate:
            rnd_p = [random.uniform(self.bounds[0], self.bounds[1]),
                     random.uniform(self.bounds[2], self.bounds[3])]
            rnd = self.Node(rnd_p)
        else:
            rnd = self.Node(self.goal.p)
        return rnd

    def get_nearest_node_index(self, rnd_node):
        dlist = [np.sum((node.p - rnd_node.p)**2) for node in self.node_list]
        return np.argmin(dlist)

    def find_near_node_indices(self, new_node):
        dlist = [np.sum((node.p - new_node.p)**2) for node in self.node_list]
        r = self.search_radius**2
        return [i for i, d in enumerate(dlist) if d <= r]

    def is_near_goal(self, node):
        return np.linalg.norm(node.p - self.goal.p) <= self.step_size

    def get_best_goal_node(self):
        goal_nodes = [node for node in self.node_list if self.is_near_goal(node)]
        if not goal_nodes:
            # If no node is near the goal, find the one closest to it
            dlist = [np.linalg.norm(node.p - self.goal.p) for node in self.node_list]
            return self.node_list[np.argmin(dlist)]
        
        # If nodes are near, pick the one with the lowest cost
        min_cost = float('inf')
        best_node = None
        for node in goal_nodes:
            cost = node.cost + np.linalg.norm(node.p - self.goal.p)
            if cost < min_cost:
                min_cost = cost
                best_node = node
        return best_node

    def is_collision_free(self, p_new, p_parent=None):
        """
        Check if a point `p_new` is collision-free.
        If `p_parent` is provided, checks the line segment between them.
        """
        # For simplicity, we only check the endpoint.
        # A full RRT* would check the path segment.
        for obs in self.static_obs:
            dx = abs(p_new[0] - obs.center[0]) - obs.width / 2
            dy = abs(p_new[1] - obs.center[1]) - obs.height / 2
            
            # Check if point is inside the "safe" rectangle
            if dx < const.D_SAFE and dy < const.D_SAFE:
                # Inside the core rectangle
                if dx < 0 and dy < 0: return False 
                # Inside the rounded corner safety margin
                if np.sqrt(max(0, dx)**2 + max(0, dy)**2) < const.D_SAFE:
                    return False
        return True