from typing import Sequence
from decorator import decorator

from dg_commons import PlayerName
from dg_commons.planning import PolygonGoal
from dg_commons.sim import SimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftModel, SpacecraftCommands, SpacecraftState
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry, SpacecraftParameters
from numpy.core import multiarray
from shapely.geometry import point, LineString
from shapely.geometry.point import Point

from pdm4ar.exercises_def.final21.scenario import get_dgscenario
from copy import deepcopy
from pdm4ar.exercises.final21.rrt_star import RrtStar
import math
import numpy as np
from numpy.linalg import inv
import time

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
                 sp: SpacecraftParameters):
        self.goal = goal
        self.static_obstacles = static_obstacles
        self.sg = sg
        self.sp = sp
        self.name = None
        optimal_path, path_list, start = self.get_optimal_path()
        self.optimal_path = optimal_path
        self.radius = 1
        self.path_nodes = [Node(point, True) if math.hypot(start.x-point[0], start.y-point[1]) < self.radius else Node(point) for point in path_list]
        
        self.test = self.test_point_and_path()


    def test_point_and_path(self):
        point1 = Point(80,30).buffer(1)
        point2 = Point(30,60).buffer(1)
        point3 = Point(60,60).buffer(1)
        point4 = Point(60,30).buffer(1)
        point5 = Point(40, 20).buffer(1)
        path = LineString([[7, 4], [30, 30], [30, 60], [60, 60], [60, 30]])
        path2 = LineString([[20, 20], [20+20, 20+20*math.sqrt(3)]])
        path3 = LineString([[20, 20], [20+20, 20]])

        return [point1, point2, point3, point4, point5, path, path2, path3]

    def get_optimal_path(self):
        _, _, start = get_dgscenario()
        rrt_start = RrtStar(start=start, goal=self.goal, static_obstacles=deepcopy([s_obstacle.shape.buffer(2) for s_obstacle in self.static_obstacles]))
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

        current_time = float(sim_obs.time)

        acc_left, acc_right = self.pid_controller(my_current_state, current_time)

        return SpacecraftCommands(acc_left=acc_left, acc_right=acc_right)

    def brake(self, my_current_state:SpacecraftState):
        m = self.sg.m
        w_half = self.sg.w_half
        Iz = self.sg.Iz

        # Match psi with math.atan2(dy,dx)
        psi = my_current_state.psi
        psi = psi % (2*math.pi)
        if psi > math.pi:
            psi = -2*math.pi + psi

        psi = my_current_state.psi
        x = my_current_state.x
        y = my_current_state.y
        vx = my_current_state.vx
        vy = my_current_state.vy
        dpsi = my_current_state.dpsi  

        # Input: u = [x, y, phi] --> Output: a = [ax, ay, ddphi]
        KD = 1
        du_des = [0, 0, 0]
        du = [vx, vy, dpsi]

        derr = np.array(du_des) - np.array(du) 
        acc = KD * derr 

        acc_left = 0.5 * (acc[0] + vy/vx*acc[1] - acc[2]*Iz/(w_half*m))
        acc_right = 0.5 * (acc[0] + vy/vx*acc[1] + acc[2]*Iz/(w_half*m))
        return acc_left, acc_right

    def rotation_controller(self, my_current_state:SpacecraftState, wanted_angle):
        m = self.sg.m
        w_half = self.sg.w_half
        Iz = self.sg.Iz

        # Match psi with math.atan2(dy,dx)
        psi = my_current_state.psi
        psi = psi % (2*math.pi)
        if psi > math.pi:
            psi = -2*math.pi + psi

        x = my_current_state.x
        y = my_current_state.y
        vx = my_current_state.vx
        vy = my_current_state.vy
        dpsi = my_current_state.dpsi  

        KP = .3
        KD = .6

        u_des = wanted_angle
        u = psi
        
        du_des = 0
        du = dpsi
        err = np.array(u_des) - np.array(u)
        derr = np.array(du_des) - np.array(du) 

        ddpsi = KP * err + KD * derr 

        acc = Iz*ddpsi / (w_half*m)

        acc_left = -acc/2
        acc_right = acc/2
        return acc_left, acc_right

    def translation_controller(self, my_current_state:SpacecraftState, x_star=30, y_star=60):
        m = self.sg.m
        w_half = self.sg.w_half
        Iz = self.sg.Iz

        # Match psi with math.atan2(dy,dx)
        psi = my_current_state.psi
        psi = psi % (2*math.pi)
        if psi > math.pi:
            psi = -2*math.pi + psi

        x = my_current_state.x
        y = my_current_state.y
        vx = my_current_state.vx
        vy = my_current_state.vy
        dpsi = my_current_state.dpsi  

        KP = .2
        KD = .2
        dx, dy = x_star - x, y_star - y
        u_des = np.sqrt(dx**2 + dy**2)
        u = 0
        
        du_des = 0
        du = vx

        target_angle = math.atan2(dy,dx)
        if abs(target_angle - psi) < math.pi/2:
            err = np.array(u_des) - np.array(u)
        else:
            err = -(np.array(u_des) - np.array(u))    

        derr = np.array(du_des) - np.array(du) 

        acc = KP * err + KD * derr 

        acc_left = acc/2
        acc_right = acc/2
        
        return acc_left, acc_right  

    def pid_controller(self, my_current_state, current_time):
        print(current_time)
        m = self.sg.m
        w_half = self.sg.w_half
        Iz = self.sg.Iz

        # Match psi with math.atan2(dy,dx)
        psi = my_current_state.psi
        psi = psi % (2*math.pi)
        if psi > math.pi:
            psi = -2*math.pi + psi

        x = my_current_state.x
        y = my_current_state.y
        vx = my_current_state.vx
        vy = my_current_state.vy
        dpsi = my_current_state.dpsi  


        x_star, y_star, psi_star = self.decide_target_point(my_current_state)
        dx, dy = x_star - x, y_star - y
        wanted_angle = math.atan2(dy,dx)
        delta_angle = psi - wanted_angle


        # if vx > 1.0 or vy > 1.0 or dpsi > 0.1:
        #     acc_left, acc_right = self.brake(my_current_state)
        #     print(f'Brake activated acc_left: {acc_left:.8f}, acc_right: {acc_right:.8f}, '\
        #           f'vx: {vx:.8f}, vy: {vy:.8f}, dpsi: {dpsi:.8f}')   
        # else: 
        if -math.radians(5) < delta_angle and delta_angle < math.radians(5):
            acc_left, acc_right = self.translation_controller(my_current_state, x_star, y_star)
            print(f'Translation Controller acc_left: {acc_left:.8f}, acc_right: {acc_right:.8f}, vx: {vx:.8f}, vy: {vy:.8f}')
        else:
            acc_left, acc_right = self.rotation_controller(my_current_state, wanted_angle)
            print(f'Rotation Controller acc_left: {acc_left:.8f}, acc_right: {acc_right:.8f}, '\
                    f'delta angle: {delta_angle:.8f}, psi: {psi:.8f}, dpsi: {dpsi:.8f}')

        # acc_left, acc_right = self.rotation_controller(my_current_state, wanted_angle)
        # print(f'Rotation Controller acc_left: {acc_left:.8f}, acc_right: {acc_right:.8f}, '\
        #       f'delta angle: {delta_angle:.8f}, psi: {psi:.8f}, dpsi: {dpsi:.8f}')          
        # acc_left, acc_right = self.velocity_controller(my_current_state)
        # print(f'Rotation Controller acc_left: {acc_left:.8f}, acc_right: {acc_right:.8f}, '\
        #       f'vx: {vx:.8f}, vy: {vy:.8f}, dpsi: {dpsi:.8f}')                            
        # acc_left, acc_right = 0,0


        # if current_time < 1:
        #     acc_left, acc_right = self.velocity_controller(my_current_state)
        #     print(f'Rotation Controller acc_left: {acc_left:.8f}, acc_right: {acc_right:.8f}, '\
        #           f'vx: {vx:.8f}, vy: {vy:.8f}, dpsi: {dpsi:.8f}')    
        # if current_time > 40:
        #     print('TIME OVER!!')
        #     acc_left, acc_right = 9, 9


        return acc_left, acc_right

    def decide_target_point(self, my_current_state):
        x = my_current_state.x
        y = my_current_state.y

        dist_table = [math.hypot(node.x - x, node.y - y) if node.passed==False and math.hypot(node.x - x, node.y - y) > self.radius
                                                         else np.inf for node in self.path_nodes]
        index = int(np.argmin(dist_table))
        for node in self.path_nodes[:index+1]:
            node.passed = True
        target_node = self.path_nodes[index]

        # Calculate path's angle at the target node
        if index != len(self.path_nodes)-1:
            next_node = self.path_nodes[index+1]
        else:
            next_node = target_node
        dx = next_node.x - target_node.x
        dy = next_node.y - target_node.y
        return target_node.x, target_node.y, math.atan2(dy, dx)

