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
from shapely.geometry import Point, LineString, LinearRing, Polygon

from pdm4ar.exercises_def.final21.scenario import get_dgscenario
from copy import deepcopy
from pdm4ar.exercises.final21.rrt_star import RrtStar
import math
import numpy as np
import time

import do_mpc
import casadi as cas

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
        self.radius = 5
        self.path_nodes = []
        for idx, point in enumerate(path_list):
            if idx != len(path_list)-1:
                dx = path_list[idx+1][0] - point[0]
                dy = path_list[idx+1][1] - point[1]
                angle = math.atan2(dy, dx)
            else:
                angle = 0
            self.path_nodes.append(Node(point, angle))

        self.x_ref = 0.0
        self.y_ref = 0.0
        self.psi_ref = 0.0

        self.x_est = 0.0
        self.y_est = 0.0

        self.x_obs = []
        self.y_obs = []

        self.dynamic_obs_present = False
        self.x_static_obs = None
        self.y_static_obs = None
        self.x_dynamic_obs = None
        self.y_dynamic_obs = None

        self.mpc_model = None
        self.mpc_controller = None

        self.targets = []

    # Default Methods
    def get_optimal_path(self):
        _, _, start = get_dgscenario()
        rrt_start = RrtStar(
            start=start,
            goal=self.goal,
            static_obstacles=deepcopy(
                [s_obstacle.shape.buffer(3) for s_obstacle in self.static_obstacles]
            )
        )
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
        current_state: SpacecraftState = sim_obs.players[self.name].state

        # Static obstacles
        if self.x_static_obs is None:
            self.x_static_obs, self.y_static_obs = self.get_static_obs_coords()

        # Dynamic obstacles present
        if(len(sim_obs.players)) > 1:
            self.dynamic_obs_present = True
            # Update dynamic obstacles
            self.update_dynamic_obs_coords(sim_obs)
        else:
            self.dynamic_obs_present = False

        acc_left, acc_right = self.mpc(current_state)
        
        return SpacecraftCommands(acc_left=acc_left, acc_right=acc_right)
    
    def mpc(self, current_state):
        """
        Calculate required accelrations using MPC controller
        """
        self.mpc_model = self.mpc_model_init()
        # Update target point
        self.update_target_point(current_state)
        # Get state variable values
        x = current_state.x
        y = current_state.y
        psi = current_state.psi
        psi = psi % (2*math.pi)
        if psi > math.pi:   
            psi = -2*math.pi + psi
        vx = current_state.vx
        vy = current_state.vy
        dpsi = current_state.dpsi
        x0 = np.array([x, y, psi, vx, vy, dpsi]).reshape(-1,1)
        # Update MPC controller
        self.mpc_controller = self.mpc_controller_init()
        # Set up controller
        self.mpc_controller.x0 = x0
        self.mpc_controller.set_initial_guess()
        # MPC steps
        u0 = self.mpc_controller.make_step(x0)
        # Extract accelerations
        acc_left, acc_right = u0[1][0], u0[0][0]
        return acc_left, acc_right

    # MPC Controller
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
        #Auxiliary
        dist_static = mpc_model.set_expression(
            'dist_static',
            cas.mmin(cas.sqrt((self.x_static_obs-x)**2+(self.y_static_obs-y)**2))
        )  # Distance to static obstacles
        if self.dynamic_obs_present:
            dist_dynamic = mpc_model.set_expression(
                'dist_dynamic',
                cas.mmin(cas.sqrt((self.x_dynamic_obs-x)**2+(self.y_dynamic_obs-y)**2))
            )  # Distance to dynamic obstacles
        # Build model
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
        # Objective
        LAMBDA_1 = 8
        LAMBDA_2 = 6
        dist = (self.x_ref-self.mpc_model.x['x'])**2 + (self.y_ref-self.mpc_model.x['y'])**2
        angle = (self.psi_ref - self.mpc_model.x['psi'])**2
        cost = LAMBDA_1 * dist + LAMBDA_2 * angle
        mpc_controller.set_objective(mterm=cost, lterm=cost)
        mpc_controller.set_rterm(u_r=1, u_l=1)
        # Constraints
        # Accelerations
        mpc_controller.bounds['lower', '_u', 'u_l'] = -10.0
        mpc_controller.bounds['upper', '_u', 'u_l'] = 10.0
        mpc_controller.bounds['lower', '_u', 'u_r'] = -10.0
        mpc_controller.bounds['upper', '_u', 'u_r'] = 10.0
        # Distances to obstacles
        sp_l = self.sg.lf + self.sg.lr
        mpc_controller.set_nl_cons(
            'dist_static',
            -self.mpc_model.aux['dist_static'],
            ub=-sp_l
        )
        if self.dynamic_obs_present:
            mpc_controller.set_nl_cons(
                'dist_dynamic',
                -self.mpc_model.aux['dist_dynamic'],
                ub=-1.5*sp_l  # Further from dynamic obstacles
            )
        mpc_controller.bounds['lower', '_x', 'dpsi'] = -.2*np.pi
        mpc_controller.bounds['upper', '_x', 'dpsi'] = .2*np.pi
        # Setup
        mpc_controller.setup()
        return mpc_controller
    
    def mpc_estimator_init(self):
        """
        MPC estimator setup
        """
        mpc_estimator = do_mpc.estimator.StateFeedback(self.mpc_model)
        return mpc_estimator
    
    def mpc_simulator_init(self):
        """
        MPC simulator setup
        """
        mpc_simulator = do_mpc.simulator.Simulator(self.mpc_model)
        mpc_simulator.set_param(t_step=0.1)
        mpc_simulator.setup()
        return mpc_simulator

    def update_target_point(self, my_current_state):
        """
        Update the next target point
        """
        x = my_current_state.x
        y = my_current_state.y
        dist_table = [math.hypot(node.x - x, node.y - y) for node in self.path_nodes]
        index = int(np.argmin(dist_table))
        LOOK_AHEAD = 6
        if index + LOOK_AHEAD < len(self.path_nodes):
            index += LOOK_AHEAD
        else:
            index = len(self.path_nodes) - 1

        target_node = self.path_nodes[index]
        # Calculate path's angle at the target node
        next_index = len(self.path_nodes) - 1 \
            if index + 3 > len(self.path_nodes) - 1 \
                else index + 3
        next_node = self.path_nodes[next_index]
        previous_node = self.path_nodes[index - 3]
        dx = next_node.x - previous_node.x
        dy = next_node.y - previous_node.y
        # Update reference point
        self.x_ref = target_node.x
        self.y_ref = target_node.y
        self.psi_ref = np.arctan2(dy, dx)

        self.targets.append(Point(self.x_ref, self.y_ref))

    def update_dynamic_obs_coords(self, sim_obs):
        """
        Update coordinates of the dynamic obstacles
        """
        coords = None
        for player in sim_obs.players:
            # Update all other players
            if player != self.name:
                # Get obstacle state
                obs_state = sim_obs.players[player].state
                # Extract x, y values
                x, y = obs_state.x, obs_state.y
                _coords = np.array([x, y])
                if coords is None:
                    coords = _coords
                else:
                    coords = np.concatenate((coords, _coords))
        if coords.ndim > 1:
            self.x_dynamic_obs = cas.SX(coords[:, 0])
            self.y_dynamic_obs = cas.SX(coords[:, 1])
        else:
            self.x_dynamic_obs = cas.SX(coords[0])
            self.y_dynamic_obs = cas.SX(coords[1])
    
    def get_static_obs_coords(self):
        """
        Get the (exterior) coordinates of static obstacles
        """
        coords = None
        for obs in self.static_obstacles:
            if isinstance(obs.shape, LinearRing):
                _coords = np.array(obs.shape.coords)
            else:
                _coords = np.array(obs.shape.exterior.coords)
            if coords is None:
                coords = _coords
            else:
                coords = np.concatenate((coords, _coords))
        x = cas.SX(coords[:, 0])
        y = cas.SX(coords[:, 1])
        return x, y