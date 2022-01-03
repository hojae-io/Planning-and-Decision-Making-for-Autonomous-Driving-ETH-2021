from typing import Sequence
from decorator import decorator

from dg_commons import PlayerName
from dg_commons.planning import PolygonGoal
from dg_commons.sim import SimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftModel, SpacecraftCommands, SpacecraftState
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry, SpacecraftParameters

from pdm4ar.exercises_def.final21.scenario import get_dgscenario
from copy import deepcopy
from pdm4ar.exercises.final21.rrt_star import RrtStar
import math
import numpy as np
import do_mpc

class Node:
    def __init__(self, point, psi=np.pi/2, passed=False):
        self.x = point[0]
        self.y = point[1]
        self.psi = psi
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
        optimal_path, path_list, start = self.get_optimal_path()
        self.optimal_path = optimal_path
        self.radius = 1.5
        self.path_nodes = []
        for idx, point in enumerate(path_list):
            if idx != len(path_list)-1:
                dx = path_list[idx+1][0] - point[0]
                dy = path_list[idx+1][1] - point[1]
                angle = math.atan2(dy, dx)
            else:
                angle = 0
            self.path_nodes.append(Node(point, angle))


        self.x_ref = 0
        self.y_ref = 0
        self.psi_ref = 0

        self.mpc_model = self.mpc_model_init()
        self.mpc_controller = self.mpc_controller_init()

    def get_optimal_path(self):
        _, _, start = get_dgscenario()
        rrt_start = RrtStar(start=start, goal=self.goal, static_obstacles=deepcopy([s_obstacle.shape.buffer(2.5) for s_obstacle in self.static_obstacles]))
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

        acc_left, acc_right = self.mpc(my_current_state)

        return SpacecraftCommands(acc_left=acc_left, acc_right=acc_right)

    def mpc(self, my_current_state:SpacecraftState):
        x, y, psi, vx, vy, dpsi = \
            my_current_state.x, my_current_state.y, my_current_state.psi, my_current_state.vx, my_current_state.vy, my_current_state.dpsi
        print(f'[Current State] x:{x:.5f}, y:{y:.5f}, psi:{psi:.5f}, vx:{vx:.5f}, vy:{vy:.5f}, dpsi:{dpsi:.5f}')

        psi = psi % (2*math.pi)
        if psi > math.pi:   
            psi = -2*math.pi + psi

        x_star, y_star, psi_star, index = self.decide_target_point(my_current_state)
        self.x_ref, self.y_ref, self.psi_ref = x_star, y_star, psi_star

        self.mpc_controller = self.mpc_controller_init()

        self.curr_state = np.array([x, y, psi, vx, vy, dpsi]).reshape(-1, 1)
        x0 = self.curr_state
        self.goal_state = np.array([x_star, y_star, psi_star, 0, 0, 0]).reshape(-1, 1)

        self.mpc_controller.x0 = x0

        self.mpc_controller.set_initial_guess()

        u0 = self.mpc_controller.make_step(x0)

        acc_left, acc_right = u0[1][0], u0[0][0]

        return acc_left, acc_right

    def mpc_model_init(self):
        """
        MPC model instantiation
        """
        model_type = 'continuous'
        mpc_model = do_mpc.model.Model(model_type)

        # States struct
        x = mpc_model.set_variable(var_type='_x', var_name='x', shape=(1,1))
        y = mpc_model.set_variable('_x', 'y')
        psi = mpc_model.set_variable('_x', 'psi')
        vx = mpc_model.set_variable('_x', 'vx') 
        vy = mpc_model.set_variable('_x', 'vy') 
        dpsi = mpc_model.set_variable('_x', 'dpsi')

        # Input struct
        u_r = mpc_model.set_variable('_u', 'u_r') # right acceleration
        u_l = mpc_model.set_variable('_u', 'u_l') # left acceleration

        # Certain parameters
        m = self.sg.m
        L = self.sg.w_half
        I = self.sg.Iz

        # ODEs
        mpc_model.set_rhs('x', np.cos(psi)*vx-np.sin(psi)*vy)
        mpc_model.set_rhs('y', np.sin(psi)*vx+np.cos(psi)*vy)
        mpc_model.set_rhs('psi', dpsi)
        mpc_model.set_rhs('vx', dpsi*vy+u_r+u_l)
        mpc_model.set_rhs('vy', -vx*dpsi)
        mpc_model.set_rhs('dpsi', L*m/I*(u_r-u_l))

        mpc_model.setup()

        return mpc_model

    def mpc_controller_init(self):
        """
        MPC controller configuration
        """
        mpc_controller = do_mpc.controller.MPC(self.mpc_model)
        setup_mpc = {
            'n_horizon': 20,
            'n_robust': 0,
            't_step': 0.1,
            'store_full_solution': True,
        }
        mpc_controller.set_param(**setup_mpc)
        LAMBDA_1 = 5   
        LAMBDA_2 = 10
        a = self.mpc_model.x['x']
        dist = (self.x_ref-self.mpc_model.x['x'])**2 + (self.y_ref-self.mpc_model.x['y'])**2
        angle = (self.psi_ref - self.mpc_model.x['psi'])**2
        cost = LAMBDA_1 * dist + LAMBDA_2 * angle
        # Control 
        mpc_controller.set_objective(mterm=cost, lterm=cost)
        mpc_controller.set_rterm(u_r=1, u_l=1)

        mpc_controller.bounds['lower', '_u', 'u_l'] = -10.0
        mpc_controller.bounds['upper', '_u', 'u_l'] = 10.0
        mpc_controller.bounds['lower', '_u', 'u_r'] = -10.0
        mpc_controller.bounds['upper', '_u', 'u_r'] = 10.0

        mpc_controller.bounds['lower', '_x', 'vx'] = -50
        mpc_controller.bounds['upper', '_x', 'vx'] = 50
        mpc_controller.bounds['lower', '_x', 'vy'] = -50
        mpc_controller.bounds['upper', '_x', 'vy'] = 50
        mpc_controller.bounds['lower', '_x', 'dpsi'] = -2*np.pi
        mpc_controller.bounds['upper', '_x', 'dpsi'] = 2*np.pi

        # setup
        mpc_controller.setup()
        return mpc_controller

    def decide_target_point(self, my_current_state):
        x = my_current_state.x
        y = my_current_state.y

        dist_table = [math.hypot(node.x - x, node.y - y) if node.passed==False and math.hypot(node.x - x, node.y - y) > self.radius
                                                         else np.inf for node in self.path_nodes]
        index = int(np.argmin(dist_table))

        if index == len(dist_table)-2:
            target_node = self.path_nodes[index+1]
        else:
            for node in self.path_nodes[:index+1]:
                node.passed = True
            target_node = self.path_nodes[index]

        return target_node.x, target_node.y, target_node.psi, index
