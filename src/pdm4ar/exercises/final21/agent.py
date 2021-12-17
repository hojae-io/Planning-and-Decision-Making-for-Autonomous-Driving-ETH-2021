from typing import Sequence
from decorator import decorator

from dg_commons import PlayerName
from dg_commons.planning import PolygonGoal
from dg_commons.sim import SimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftModel, SpacecraftCommands, SpacecraftState
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry

from pdm4ar.exercises_def.final21.scenario import get_dgscenario
from copy import deepcopy
from pdm4ar.exercises.final21.rrt_star import RrtStar
import math
import numpy as np

class Node:
    def __init__(self, point, passed=False):
        self.x = point[0]
        self.y = point[1]
        self.passed = passed

class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do NOT modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    def __init__(self,
                 goal: PolygonGoal,
                 static_obstacles: Sequence[StaticObstacle],
                 sg: SpacecraftGeometry,
                 sp: SpacecraftGeometry):
        self.goal = goal
        self.static_obstacles = static_obstacles
        self.sg = sg
        self.sp = sp
        self.name = None
        optimal_path, path_list, start = self.get_optimal_path()
        self.radius = 5
        self.path_nodes = [Node(point, True) if math.hypot(start.x-point[0], start.y-point[1]) < self.radius else Node(point) for point in path_list]

    def get_optimal_path(self):
        _, _, start = get_dgscenario()
        rrt_start = RrtStar(start=start, goal=self.goal, static_obstacles=deepcopy([s_obstacle.shape.buffer(3) for s_obstacle in self.static_obstacles]))
        optimal_path, path_list = rrt_start.planning()
        return optimal_path, path_list, start
        
    def on_episode_init(self, my_name: PlayerName):
        self.name = my_name

    def get_commands(self, sim_obs: SimObservations) -> SpacecraftCommands:
        """ This method is called by the simulator at each time step.

        This is how you can get your current state from the observations:
        my_current_state: SpacecraftState = sim_obs.players[self.name].state
        
        :param sim_obs:
        :return:
        """
        my_current_state: SpacecraftState = sim_obs.players[self.name].state

        psi = my_current_state.psi
        x = my_current_state.x
        y = my_current_state.y

        Kp = 1
        Kd = .5

        x_star, y_star = self.decide_target_point(my_current_state)
        dx, dy = x_star - x, y_star - y

        u_des = [math.cos(psi)*dx + math.sin(psi)*dy, -math.sin(psi)*dx + math.cos(psi)*dy]
        du_des = [0.1, 0]

        err = np.array(u_des) - np.array([0, 0])
        derr = np.array(du_des) - np.array([my_current_state.vx, my_current_state.vy]) 
        acc = Kp * err + Kd * derr

        acc_left = 0.5 * (acc[0] - (acc[1]/acc[0])/self.sg.w_half)
        acc_right = 0.5 * (acc[0] + (acc[1]/acc[0])/self.sg.w_half)

        return SpacecraftCommands(acc_left=acc_left, acc_right=acc_right)

    def decide_target_point(self, my_current_state):
        x = my_current_state.x
        y = my_current_state.y

        dist_table = [math.hypot(node.x - x, node.y - y) if node.passed==False else np.inf for node in self.path_nodes]
        index = int(np.argmin(dist_table))
        for node in self.path_nodes[:index+1]:
            node.passed = True
        target_node = self.path_nodes[index]
        return target_node.x, target_node.y



