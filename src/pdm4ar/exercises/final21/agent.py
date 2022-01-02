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

import do_mpc

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

        self.x_ref = 0.0
        self.y_ref = 0.0
        self.psi_ref = 0.0
        
        self.previous_a_r = 0.0
        self.previous_a_l = 0.0

        self.model_type = 'discrete'
        self.mpc_model = do_mpc.model.Model(self.model_type)
        self.mpc_controller = None
        self.mpc_estimator = None
        self.mpc_simulator = None
    

    def mpc_model_init(self):
        """
        MPC model instantiation
        """
        # States struct
        x = self.mpc_model.set_variable('_x', 'x')
        y = self.mpc_model.set_variable('_x', 'y')
        psi = self.mpc_model.set_variable('_x', 'psi')
        dx = self.mpc_model.set_variable('_x', 'dx') # vx
        dy = self.mpc_model.set_variable('_x', 'dy') # vy
        dpsi = self.mpc_model.set_variable('_x', 'dpsi')
        # Input struct
        a_r = self.mpc_model.set_variable('_u', 'a_r') # right acceleration
        a_l = self.mpc_model.set_variable('_u', 'a_l') # left acceleration
        # Certain parameters
        m = self.sg.m
        L = self.sg.w_half
        I = self.sg.Iz
        # ODEs
        self.mpc_model.set_rhs('x', np.cos(psi)*dx-np.sin(psi)*dy)
        self.mpc_model.set_rhs('y', np.sin(psi)*dx+np.cos(psi)*dy)
        self.mpc_model.set_rhs('psi', dpsi)
        self.mpc_model.set_rhs('vx', dpsi*dy+a_r+a_l)
        self.mpc_model.set_rhs('vy', -dx*dpsi)
        self.mpc_model.set_rhs('dpsi', L*m/I*(a_r-a_l))
        # Build model
        self.mpc_model.setup()
    
    def mpc_controller_init(self):
        """
        MPC controller configuration
        """
        self.mpc_controller = do_mpc.controller.MPC(self.mpc_model)
        setup_mpc = {
            'n_horizon': 20,
            'n_robust': 1,
            'open_loop': 0,
            't_step': 0.01,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 2,
            'collocation_ni': 2,
            'store_full_solution': True,
            'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
        }
        self.mpc_controller.set_param(**setup_mpc)
        # Objective
        LAMBDA_1 = 1
        LAMBDA_2 = 1
        current_point = np.array((self.mpc_model.x['x'], self.mpc_model.x['y']))
        ref_point = np.array((self.x_ref, self.y_ref))
        dist = np.sum(np.square(ref_point - current_point))
        angle = np.square(self.psi_ref - self.mpc_model.x['psi'])
        cost = LAMBDA_1 * dist + LAMBDA_2 * angle
        self.mpc_controller.set_objective(mterm=cost, lterm=cost)
        self.mpc_controller.set_rterm(inp=1.0)
        # Constraints
        self.mpc_controller.bounds['lower', '_u', 'a_l'] = -10.0
        self.mpc_controller.bounds['upper', '_u', 'a_l'] = 10.0
        self.mpc_controller.bounds['lower', '_u', 'a_r'] = -10.0
        self.mpc_controller.bounds['upper', '_u', 'a_r'] = 10.0
        # setup
        self.mpc_controller.setup()
    
    def mpc_estimator_init(self):
        """
        MPC estimator setup
        """
        self.mpc_estimator = do_mpc.estimator.StateFeedback(self.mpc_model)
    
    def mpc_simulator_init(self):
        """
        MPC simulator setup
        """
        self.mpc_simulator = do_mpc.simulator.Simulator(self.mpc_model)
        params_simulator = {
            'integration_tool': 'cvodes',
            'abstol': 1e-10,
            'reltol': 1e-10,
            't_step': 0.01
        }
        self.mpc_simulator.set_param(**params_simulator)
        self.mpc_simulator.setup()

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

    def get_target_point(self, my_current_state):
        x = my_current_state.x
        y = my_current_state.y

        dist_table = [math.hypot(node.x - x, node.y - y) for node in self.path_nodes]
        index = int(np.argmin(dist_table))

        # Set target as 10 samples ahead
        LOOK_AHEAD = 10
        if index + LOOK_AHEAD < len(self.path_nodes):
            index += LOOK_AHEAD
        else:
            index = len(self.path_nodes - 1)

        target_node = self.path_nodes[index]

        # Calculate path's angle at the target node
        if index != len(self.path_nodes)-1:
            next_node = self.path_nodes[index+1]
        else:
            next_node = target_node
        dx = next_node.x - target_node.x
        dy = next_node.y - target_node.y
        return target_node.x, target_node.y, math.atan2(dy, dx)

