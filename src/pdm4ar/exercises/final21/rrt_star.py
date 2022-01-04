import math
import numpy as np
from typing import Sequence, List, Tuple

from dg_commons.planning import PolygonGoal
from dg_commons.sim.models.spacecraft import SpacecraftState
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry
from shapely.geometry import Point, LineString, Polygon
from tqdm import tqdm

# Point with parent info
class Node:
    def __init__(self, point:Point, parent=None):
        self.x = point.x
        self.y = point.y
        self.parent:Node = parent

class RrtStar:
    def __init__(self, 
                 start:SpacecraftState, 
                 goal:PolygonGoal,
                 static_obstacles: Sequence[Polygon], # List of static obstacles which are Polygons
                 max_size: int=100,
                 iter_max: int=20000,
                 eta: int=0.3,
                 gamma_rrg: int=20):   
        self.start = Node(Point(start.x, start.y)) # CoG of Spacecraft is start point
        self.goal = goal
        self.obstacles = static_obstacles
        self.max_size = max_size
        self.iter_max = iter_max
        self.eta = eta # eta in STEER function
        self.gamma_rrg = gamma_rrg # gamma_rrg in NEAR function

        self.path = []
        self.vertex_list = [self.start] 
        
    # main function for creating optimal path
    def planning(self) -> LineString:
        for i in tqdm(range(self.iter_max)):
            node_rand = self.generate_random_node()
            node_nearest = self.nearest_neighbor(self.vertex_list, node_rand)
            node_new = self.steer(node_nearest, node_rand)

            if self.obstacle_free(node_nearest, node_new):
                near_index = self.near(node_new)
                self.vertex_list.append(node_new)

                self.choose_parent(node_new, near_index)
                self.rewire(node_new, near_index)

        node_goal = self.search_goal_node()
        linestring_path = self.extract_path(node_goal)

        return linestring_path, self.path
    
    def generate_random_node(self):
        found = False
        while not found:
            rand_point = Point(np.random.uniform(0, self.max_size), np.random.uniform(0, self.max_size))
            found = True
            for obstacle in self.obstacles: 
                if obstacle.covers(rand_point):
                    found = False
            if found:
                break
        return Node(rand_point)

    def nearest_neighbor(self, vertex_list, node_rand):
        return vertex_list[int(np.argmin([math.hypot(vertex.x - node_rand.x, vertex.y - node_rand.y)
                                        for vertex in vertex_list]))]

    def steer(self, node_nearest, node_rand):
        dist, theta = self.get_distance_and_angle(node_nearest, node_rand)

        dist = min(self.eta, dist)
        node_new = Node(Point(node_nearest.x + dist * math.cos(theta),
                         node_nearest.y + dist * math.sin(theta)))

        node_new.parent = node_nearest

        return node_new

    def get_distance_and_angle(self, node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def obstacle_free(self, node_start:Node, node_end:Node):
        obs_free = True
        path = LineString([[node_start.x, node_start.y], [node_end.x, node_end.y]])
        for obstacle in self.obstacles:
            if obstacle.intersects(path):
                obs_free = False
        return obs_free
        
    def near(self, node_new:Node):
        n = len(self.vertex_list) + 1
        error_offset = 0.3 # Due to the calculation error such as 10.0000002
        r = min(self.gamma_rrg * math.sqrt((math.log(n) / n)), self.eta) + error_offset

        dist_table = [math.hypot(node.x - node_new.x, node.y - node_new.y) for node in self.vertex_list]
        near_index = [ind for ind in range(len(dist_table)) 
                                  if dist_table[ind] <= r and self.obstacle_free(node_new, self.vertex_list[ind])]
        return near_index

    def choose_parent(self, node_new:Node, near_index:List[int]):
        cost = [self.get_new_cost(self.vertex_list[i], node_new) if self.obstacle_free(self.vertex_list[i], node_new) else np.inf for i in near_index]
        cost_min_index = near_index[int(np.argmin(cost))]

        node_new.parent = self.vertex_list[cost_min_index]
    
    def get_new_cost(self, node_start, node_end):
        dist, _ = self.get_distance_and_angle(node_start, node_end)

        return self.cost(node_start) + dist

    def cost(self, node:Node):
        curr_node = node
        cost_ = 0.0

        while curr_node.parent:
            cost_ += math.hypot(curr_node.x - curr_node.parent.x, curr_node.y - curr_node.parent.y)
            curr_node = curr_node.parent

        return cost_

    def rewire(self, node_new:Node, near_index:List[int]):
        for i in near_index:
            node_near = self.vertex_list[i]

            if self.obstacle_free(node_new, node_near) and self.get_new_cost(node_new, node_near) < self.cost(node_near):
                node_near.parent = node_new

    def search_goal_node(self):
        node_goal_candidates = [node for node in self.vertex_list if self.goal.goal.covers(Point(node.x, node.y))]

        if len(node_goal_candidates) > 0:
            cost_list = [self.cost(node) for node in node_goal_candidates]
            return node_goal_candidates[int(np.argmin(cost_list))]

        return self.vertex_list[-1]

    def extract_path(self, node_goal:Node) -> LineString:
        node = node_goal

        while node is not None:
            self.path.append([node.x, node.y])
            node = node.parent

        self.path.reverse()
        return LineString(self.path)
