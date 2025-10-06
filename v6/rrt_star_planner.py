import numpy as np
import random

class RRTStar:
    class Node:
        def __init__(self, p):
            self.p = np.array(p)
            self.parent = None
            self.cost = 0.0

    def __init__(self, start, goal, obstacles_static, obstacles_dynamic, bounds, 
                 max_iter=500, step_size=2.0, search_radius=4.0, goal_sample_rate=0.1, safe_dist=3.0):
        self.start = self.Node(start)
        self.goal = self.Node(goal)
        self.bounds = bounds
        self.max_iter = max_iter
        self.step_size = step_size
        self.search_radius = search_radius
        self.goal_sample_rate = goal_sample_rate
        self.safe_dist = safe_dist

        self.static_obs = obstacles_static
        self.dynamic_obs = obstacles_dynamic
        self.node_list = [self.start]

    def plan(self):
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_node = self.get_nearest_node(rnd_node)
            new_node = self.steer(nearest_node, rnd_node)

            if self.is_collision_free(new_node.p, nearest_node.p):
                near_nodes_indices = self.find_near_nodes(new_node)
                self.choose_parent(new_node, near_nodes_indices)
                if new_node.parent:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_nodes_indices)

            if self.is_near_goal(self.node_list[-1]):
                break
        
        return self.generate_final_path()

    def steer(self, from_node, to_node):
        d, theta = self.calc_distance_and_angle(from_node, to_node)
        
        new_node = self.Node(from_node.p)
        new_node.p[0] += min(self.step_size, d) * np.cos(theta)
        new_node.p[1] += min(self.step_size, d) * np.sin(theta)
        
        new_node.parent = from_node
        new_node.cost = from_node.cost + self.line_cost(from_node, new_node)
        return new_node

    def choose_parent(self, new_node, near_nodes_indices):
        if not near_nodes_indices:
            return

        min_cost = new_node.cost
        best_parent = new_node.parent

        for i in near_nodes_indices:
            near_node = self.node_list[i]
            if self.is_collision_free(new_node.p, near_node.p):
                cost = near_node.cost + self.line_cost(near_node, new_node)
                if cost < min_cost:
                    min_cost = cost
                    best_parent = near_node
        
        new_node.parent = best_parent
        new_node.cost = min_cost

    def rewire(self, new_node, near_nodes_indices):
        for i in near_nodes_indices:
            near_node = self.node_list[i]
            if self.is_collision_free(new_node.p, near_node.p):
                cost = new_node.cost + self.line_cost(new_node, near_node)
                if cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = cost

    def generate_final_path(self):
        last_node = self.get_best_goal_node()
        if last_node is None:
            return None
        
        path = [self.goal.p]
        node = last_node
        while node.parent is not None:
            path.append(node.p)
            node = node.parent
        path.append(self.start.p)
        return np.array(path[::-1])

    def get_random_node(self):
        if random.random() > self.goal_sample_rate:
            rnd = self.Node([random.uniform(self.bounds[0], self.bounds[1]),
                             random.uniform(self.bounds[2], self.bounds[3])])
        else:
            rnd = self.Node(self.goal.p)
        return rnd

    def get_nearest_node(self, rnd_node):
        dlist = [np.sum((node.p - rnd_node.p)**2) for node in self.node_list]
        minind = dlist.index(min(dlist))
        return self.node_list[minind]

    def find_near_nodes(self, new_node):
        dlist = [np.sum((node.p - new_node.p)**2) for node in self.node_list]
        near_inds = [dlist.index(d) for d in dlist if d <= self.search_radius**2]
        return near_inds

    def is_near_goal(self, node):
        return np.linalg.norm(node.p - self.goal.p) <= self.step_size

    def get_best_goal_node(self):
        goal_nodes = [node for node in self.node_list if self.is_near_goal(node)]
        if not goal_nodes:
            return None
        
        min_cost = float('inf')
        best_node = None
        for node in goal_nodes:
            cost = node.cost + self.line_cost(node, self.goal)
            if cost < min_cost:
                min_cost = cost
                best_node = node
        return best_node

    def is_collision_free(self, p_new, p_parent):
        # Dynamic obstacles (circles)
        for obs in self.dynamic_obs:
            d = np.linalg.norm(obs.state[:2] - p_new)
            if d <= obs.radius + self.safe_dist:
                return False
        # Static obstacles (rectangles)
        for obs in self.static_obs:
            dx = abs(p_new[0] - obs.center[0]) - obs.width / 2
            dy = abs(p_new[1] - obs.center[1]) - obs.height / 2
            if dx < self.safe_dist and dy < self.safe_dist:
                if dx < 0 and dy < 0: return False # Inside
                if np.sqrt(max(0, dx)**2 + max(0, dy)**2) < self.safe_dist:
                    return False
        return True

    @staticmethod
    def line_cost(node1, node2):
        return np.linalg.norm(node1.p - node2.p)

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.p[0] - from_node.p[0]
        dy = to_node.p[1] - from_node.p[1]
        d = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        return d, theta