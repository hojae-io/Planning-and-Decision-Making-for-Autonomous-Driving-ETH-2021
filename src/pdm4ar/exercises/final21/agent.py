from typing import Sequence
from decorator import decorator

from dg_commons import PlayerName
from dg_commons.planning import PolygonGoal
from dg_commons.sim import SimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.spacecraft import SpacecraftModel, SpacecraftCommands, SpacecraftState
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry, SpacecraftParameters
from numpy import linalg
from numpy.core import multiarray
from scipy.integrate._ivp.radau import P
from shapely.geometry import point, LineString
from shapely.geometry.point import Point

from pdm4ar.exercises_def.final21.scenario import get_dgscenario
from copy import deepcopy
from pdm4ar.exercises.final21.rrt_star import RrtStar
import math
import numpy as np
from numpy.linalg import inv
import control
from qpsolvers import solve_qp
from scipy import optimize
import sympy as sym
import scipy
import do_mpc
class Node:
    def __init__(self, point, psi=0, passed=False):
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
        self.name = None
        optimal_path, path_list, start = self.get_optimal_path()
        self.optimal_path = optimal_path
        self.radius = 5
        # self.path_nodes = [Node(point, True) if math.hypot(start.x-point[0], start.y-point[1]) < self.radius else Node(point) for point in path_list]
        self.path_nodes = []
        for idx, point in enumerate(path_list):
            if idx != len(path_list)-1:
                dx = path_list[idx+1][0] - point[0]
                dy = path_list[idx+1][1] - point[1]
                angle = math.atan2(dy, dx)
            else:
                angle = 0
            self.path_nodes.append(Node(point, angle))

        self.A = self.updateA()
        self.B = self.updateB()
        self.P = np.zeros((6,6))
        self.dt = 0.1
        self.ddt = 0.01
        self.curr_state = np.array([0,0,0,0,0,0])
        self.goal_state = np.array([0,0,0,0,0,0])

        self.x_ref = 20
        self.y_ref = 10
        self.psi_ref = np.pi/4

        self.mpc_model = self.mpc_model_init()
        self.mpc_controller = self.mpc_controller_init()
        self.mpc_estimator = self.mpc_estimator_init()
        self.mpc_simulator = self.mpc_simulator_init()



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

        # acc_left, acc_right = self.pid_controller(my_current_state, current_time)
        # acc_left, acc_right = self.integrated_ltvlqr_controller(my_current_state)
        # acc_left, acc_right = self.blog_ltvlqr_controller(my_current_state)
        # acc_left, acc_right = self.my_controller(my_current_state)
        acc_left, acc_right = self.mpc(my_current_state)

        # if len(self.path_nodes) != 1:
        #     self.path_nodes.pop(0)
        if current_time > 15:
            acc_left, acc_right = 9, 9
        return SpacecraftCommands(acc_left=acc_left, acc_right=acc_right)

    def mpc(self, my_current_state:SpacecraftState):
        x, y, psi, vx, vy, dpsi = \
            my_current_state.x, my_current_state.y, my_current_state.psi, my_current_state.vx, my_current_state.vy, my_current_state.dpsi
        print(f'[Current State] x:{x:.5f}, y:{y:.5f}, psi:{psi:.5f}, vx:{vx:.5f}, vy:{vy:.5f}, dpsi:{dpsi:.5f}')

        psi = psi % (2*math.pi)
        if psi > math.pi:   
            psi = -2*math.pi + psi

        # x_star, y_star, psi_star, index = self.decide_target_point(my_current_state)
        x_star, y_star, psi_star = 20, 10, math.pi/4
        self.curr_state = np.array([x, y, psi, vx, vy, dpsi]).reshape(-1, 1)
        x0 = self.curr_state
        self.goal_state = np.array([x_star, y_star, psi_star, 0, 0, 0]).reshape(-1, 1)

        self.mpc_controller.x0 = x0
        self.mpc_simulator.x0 = x0
        self.mpc_estimator.x0 = x0

        self.mpc_controller.set_initial_guess()

        n_steps = 20
        for i in range(n_steps):
            u0 = self.mpc_controller.make_step(x0)
            y_next = self.mpc_simulator.make_step(u0)
            x0 = self.mpc_estimator.make_step(y_next)

        acc_left = np.squeeze(self.mpc_controller.data['_u', 'u_l'][0])
        acc_right = np.squeeze(self.mpc_controller.data['_u', 'u_r'][0])

        return acc_left, acc_right

    def mpc_model_init(self):
        """
        MPC model instantiation
        """
        model_type = 'discrete'
        mpc_model = do_mpc.model.Model(model_type)

        # States struct
        x = mpc_model.set_variable(var_type='_x', var_name='x', shape=(1,1))
        y = mpc_model.set_variable('_x', 'y')
        psi = mpc_model.set_variable('_x', 'psi')
        vx = mpc_model.set_variable('_x', 'vx') # vx
        vy = mpc_model.set_variable('_x', 'vy') # vy
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
            'n_horizon': 5,
            'n_robust': 0,
            't_step': 0.01,
            'store_full_solution': True,
            'state_discretization': 'discrete'
        }
        mpc_controller.set_param(**setup_mpc)
        LAMBDA_1 = 10
        LAMBDA_2 = 3
        dist = (self.x_ref-self.mpc_model.x['x'])**2 + (self.y_ref-self.mpc_model.x['y'])**2
        angle = (self.psi_ref - self.mpc_model.x['psi'])**2
        cost = LAMBDA_1 * dist + LAMBDA_2 * angle
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

    def mpc_simulator_init(self):
        """
        MPC simulator setup
        """
        mpc_simulator = do_mpc.simulator.Simulator(self.mpc_model)
        mpc_simulator.set_param(t_step=0.1)
        mpc_simulator.setup()
        return mpc_simulator

    def mpc_estimator_init(self):
        """
        MPC estimator setup
        """
        mpc_estimator = do_mpc.estimator.StateFeedback(self.mpc_model)
        return mpc_estimator




    def my_controller(self, my_current_state:SpacecraftState):
        # Linear Time Varying Linear Quadratic Regulator
        x, y, psi, vx, vy, dpsi = \
            my_current_state.x, my_current_state.y, my_current_state.psi, my_current_state.vx, my_current_state.vy, my_current_state.dpsi

        print(f'[Current State] x:{x:.5f}, y:{y:.5f}, psi:{psi:.5f}, vx:{vx:.5f}, vy:{vy:.5f}, dpsi:{dpsi:.5f}')
        psi = psi % (2*math.pi)
        if psi > math.pi:
            psi = -2*math.pi + psi

        # x_star, y_star, psi_star, index = self.decide_target_point(my_current_state)
        x_star, y_star, psi_star = 20, 10, math.pi/4

        self.curr_state = np.array([x, y, psi, vx, vy, dpsi])
        self.goal_state = np.array([x_star, y_star, psi_star, 0, 0, 0])

        self.A = self.diffA(psi=psi, dpsi=dpsi)
        self.B = self.diffB()

        C = scipy.linalg.expm(self.A * 0.1)
        D = np.eye(6)
        for i in range(1,10):
            D = D + scipy.linalg.expm(self.A * (0.01*i))
        D = D @ self.B * 0.01

        acc = inv(D.T @ D) @ D.T @ (self.goal_state - C @ self.curr_state)

        acc_left, acc_right = acc[0], acc[1]
        max_val = max(abs(acc_left), abs(acc_right))
        acc_left = acc_left / max_val * 9 if max_val !=0 else 0
        acc_right = acc_right / max_val * 9 if max_val !=0 else 0
        print(f'[Acceleration State] acc_left:{acc_left:.5f}, acc_right:{acc_right:.5f}')


        return acc_left, acc_right

    def F(self, u):
        x0, y0, psi0, vx0, vy0, dpsi0 = \
            self.curr_state[0], self.curr_state[1], self.curr_state[2], self.curr_state[3], self.curr_state[4], self.curr_state[5]

        expected = np.array([x0 + vx0*(np.cos(psi0)/100 + np.cos(dpsi0/100 + psi0)/100 - (dpsi0*np.sin(dpsi0/100 + psi0))/10000) - vy0*(np.sin(dpsi0/100 + psi0)/100 + np.sin(psi0)/100 - (dpsi0*np.cos(dpsi0/100 + psi0))/10000) + (u[0]*np.cos(dpsi0/100 + psi0))/2000 + (u[1]*np.cos(dpsi0/100 + psi0))/2000,
                            y0 + vy0*(np.cos(psi0)/100 + np.cos(dpsi0/100 + psi0)/100 + (dpsi0*np.sin(dpsi0/100 + psi0))/10000) + vx0*(np.sin(dpsi0/100 + psi0)/100 + np.sin(psi0)/100 + (dpsi0*np.cos(dpsi0/100 + psi0))/10000) + (u[0]*np.sin(dpsi0/100 + psi0))/2000 + (u[1]*np.sin(dpsi0/100 + psi0))/2000,
                            dpsi0/50 + psi0 - u[0]/4000 + u[1]/4000,
                            u[0]/10 + u[1]/10 + vy0*(dpsi0/50 - u[0]/4000 + u[1]/4000) + vx0*((dpsi0*(dpsi0/100 - u[0]/4000 + u[1]/4000))/100 + 1),
                            u[0]*(dpsi0/2000 - u[0]/80000 + u[1]/80000) + u[1]*(dpsi0/2000 - u[0]/80000 + u[1]/80000) + vx0*(dpsi0/50 - u[0]/4000 + u[1]/4000) + vy0*((dpsi0*(dpsi0/100 - u[0]/4000 + u[1]/4000))/100 + 1),
                            dpsi0 - u[0]/20 + u[1]/20])

        return expected - self.goal_state

    def blog_ltvlqr_controller(self, my_current_state:SpacecraftState):
        # Linear Time Varying Linear Quadratic Regulator
        x, y, psi, vx, vy, dpsi = \
            my_current_state.x, my_current_state.y, my_current_state.psi, my_current_state.vx, my_current_state.vy, my_current_state.dpsi

        print(f'[Current State] x:{x:.5f}, y:{y:.5f}, psi:{psi:.5f}, vx:{vx:.5f}, vy:{vy:.5f}, dpsi:{dpsi:.5f}')
        psi = psi % (2*math.pi)
        if psi > math.pi:
            psi = -2*math.pi + psi

        # x_star, y_star, psi_star, index = self.decide_target_point(my_current_state)
        x_star, y_star, psi_star = 30, 10, math.pi/4

        curr_state = np.array([x, y, psi, vx, vy, dpsi])
        goal_state = np.array([x_star, y_star, psi_star, 1, 0, 0])

        x_error = curr_state - goal_state

        self.A = self.updateA(psi=psi, dpsi=dpsi, dt=self.dt)
        # self.A = self.updateA_old(vx=vx, vy=vy, psi=psi, dpsi=dpsi, dt=self.dt)
        self.B = self.updateB(dt=self.dt)

        N = 100
 
        # Create a list of N + 1 elements
        P = [None] * (N + 1)
        P[N] = self.Q


        # for i in range(N, 0, -1):
        #     P[i-1] = self.Q + self.A.T @ P[i] @ self.A - \
        #         (self.A.T @ P[i] @ self.B) @ np.linalg.pinv(self.R + self.B.T @ P[i] @ self.B) @ (self.B.T @ P[i] @ self.A)            

        K = [None] * N
        u = [None] * N

        # for i in range(N):
        #     K[i] = -np.linalg.pinv(self.R + self.B.T @ P[i] @ self.B) @ self.B.T @ P[i] @ self.A 
        #     u[i] = K[i] @ x_error

        for i in range(N,0,-1):
            K[i] = -np.linalg.pinv(self.R + self.B.T @ P[i] @ self.B) @ self.B.T @ P[i] @ self.A 
            P[i+1] = self.Q + K[i].T @ self.R @ K[i] +\
                   (self.A + self.B @ K[i]).T @ P[i] @ (self.A + self.B @ K[i])

            curr_state:Node = self.path_nodes[N-i-1] 
            self.A = self.updateA(psi=curr_state.psi, dt=self.dt)

        acc = u[N-1]

        acc_left, acc_right = acc[0], acc[1]

        return acc_left, acc_right

    def integrated_ltvlqr_controller(self, my_current_state:SpacecraftState):
        # Linear Time Varying Linear Quadratic Regulator
        x, y, psi, vx, vy, dpsi = \
            my_current_state.x, my_current_state.y, my_current_state.psi, my_current_state.vx, my_current_state.vy, my_current_state.dpsi

        psi = psi % (2*math.pi)
        if psi > math.pi:
            psi = -2*math.pi + psi

        # x_star, y_star, psi_star, index = self.decide_target_point(my_current_state)
        # x_star, y_star, psi_star = 10, 50, 3*math.pi/4
        # self.A = self.updateA(psi=psi, dt=self.dt)
        # self.A = self.updateA_old(vx=vx, vy=vy, psi=psi, dpsi=dpsi, dt=self.dt)
        self.B = self.updateB(dt=self.dt)

        N = len(self.path_nodes)
 
        # Create a list of N + 1 elements
        P = [None] * (N + 1)
        # P[N] = self.Q

        K = [None] * N
        u = [None] * N

        P[0] = np.zeros((6,6))
        for i in range(N):
            curr_state:Node = self.path_nodes[N-i-1] 
            self.A = self.updateA(psi=curr_state.psi, dt=self.dt)
            K[i] = -np.linalg.pinv(self.R + self.B.T @ P[i] @ self.B) @ self.B.T @ P[i] @ self.A 
            P[i+1] = self.Q + K[i].T @ self.R @ K[i] +\
                   (self.A + self.B @ K[i]).T @ P[i] @ (self.A + self.B @ K[i])


        curr_state = np.array([x, y, psi, vx, vy, dpsi])
        goal_Node:Node = self.path_nodes[0]
        goal_state = np.array([goal_Node.x, goal_Node.y, goal_Node.psi, 5, 0, 0])

        acc = K[N-1] @ (curr_state - goal_state)# + np.array([1,1])

        acc_left, acc_right = acc[0], acc[1]
        # max_val = max(abs(acc_left), abs(acc_right))
        # acc_left = acc_left / max_val * 9 if max_val !=0 else 0
        # acc_right = acc_right / max_val * 9 if max_val !=0 else 0


        return acc_left, acc_right


    def diffA(self, psi=0, dpsi=0):
        A = np.array([[0, 0, 0, np.cos(psi), -np.sin(psi),      0],
                      [0, 0, 0, np.sin(psi),  np.cos(psi),      0],
                      [0, 0, 0,           0,            0,      1],
                      [0, 0, 0,           0,         dpsi,      0],
                      [0, 0, 0,       -dpsi,            0,      0],
                      [0, 0, 0,           0,            0,      0]])

        return A


    def diffB(self):
        m = self.sg.m
        w_half = self.sg.w_half
        Iz = self.sg.Iz

        B = np.array([[0, 0],
                      [0, 0],
                      [0, 0],
                      [1, 1],
                      [0, 0],
                      [-w_half*m/Iz, w_half*m/Iz]])
        return B

    def updateA(self, psi=0, dpsi=0, dt=0):
        A = np.array([[1, 0, 0, np.cos(psi)*dt, -np.sin(psi)*dt,      0],
                      [0, 1, 0, np.sin(psi)*dt,  np.cos(psi)*dt,      0],
                      [0, 0, 1,              0,               0,     dt],
                      [0, 0, 0,              1,         dpsi*dt,      0],
                      [0, 0, 0,        dpsi*dt,               1,      0],
                      [0, 0, 0,              0,               0,      1]])

        return A

    def updateA_old(self, vx=5, vy=0, psi=0, dpsi=0, dt=0.1):
        A = np.array([[1, 0, (-vx*np.sin(psi)-vy*np.cos(psi))*dt, np.cos(psi)*dt, -np.sin(psi)*dt,      0],
                      [0, 1,  (vx*np.cos(psi)-vy*np.sin(psi))*dt, np.sin(psi)*dt,  np.cos(psi)*dt,      0],
                      [0, 0,                                   1,              0,               0,     dt],
                      [0, 0,                                   0,              1,         dpsi*dt,  vy*dt],
                      [0, 0,                                   0,       -dpsi*dt,               1, -vx*dt],
                      [0, 0,                                   0,              0,               0,      1]])

        return A

    def updateB(self, dt=0):
        m = self.sg.m
        w_half = self.sg.w_half
        Iz = self.sg.Iz

        B = np.array([[0, 0],
                      [0, 0],
                      [0, 0],
                      [dt, dt],
                      [0, 0],
                      [-w_half*m*dt/Iz, w_half*m*dt/Iz]])
        return B

    @property
    def Q(self):
        Q = np.eye(6,6)
        Q[1,1] = 2
        Q[2,2] = 2
        return Q

    @property
    def R(self):
        R = np.eye(2) 
        return R 

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

        return target_node.x, target_node.y, target_node.psi, index
