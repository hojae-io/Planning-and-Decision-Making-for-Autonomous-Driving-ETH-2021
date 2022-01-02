from copy import deepcopy
import matplotlib
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.sim.simulator_visualisation import ZOrders
from matplotlib import pyplot as plt
from dg_commons.sim.models.spacecraft import SpacecraftModel, SpacecraftState
from dg_commons import PlayerName
from numpy.lib.index_tricks import s_
from shapely.geometry.linestring import LineString
from pdm4ar.exercises.final21.agent import Pdm4arAgent
from shapely.geometry import Point

from pdm4ar.exercises_def.final21.scenario import get_dgscenario
from pdm4ar.exercises.final21.rrt_star import RrtStar
from pdm4ar.exercises_def.final21.ex import get_final21
import os

if __name__ == '__main__':
    # matplotlib.use('TkAgg')
    # dg_scenario, goal, x0 = get_dgscenario()
    # ax = plt.gca()
    # shapely_viz = ShapelyViz(ax)

    # PDM4AR = PlayerName("PDM4AR")

    # model = SpacecraftModel.default(x0)
    # models = {PDM4AR: model}
    # missions = {PDM4AR: goal}
    # players = {PDM4AR: Pdm4arAgent(
    #     static_obstacles=deepcopy(list(dg_scenario.static_obstacles.values())),
    #     goal=goal,
    #     sg=deepcopy(model.get_geometry()),
    #     sp=deepcopy(model.sp))
    # }
    # for i, s_obstacle in enumerate(dg_scenario.static_obstacles.values()):
    #     shapely_viz.add_shape(s_obstacle.shape, color=s_obstacle.geometry.color, zorder=ZOrders.ENV_OBSTACLE)
    #     # if i is not 0:
    #     #     shapely_viz.add_shape(s_obstacle.shape.buffer(2), color=s_obstacle.geometry.color, zorder=ZOrders.ENV_OBSTACLE)

    #     # print(s_obstacle.shape.boundary, s_obstacle.shape.covers(Point(50, 50)))
    #     # print(s_obstacle.shape.intersects(path))
    # shapely_viz.add_shape(goal.get_plottable_geometry(), color="orange", zorder=ZOrders.GOAL, alpha=0.5)
    # state = model.get_state()
    # shapely_viz.add_shape(Point(state.x, state.y))

    # rrt_start = RrtStar(start=x0, goal=goal, static_obstacles=deepcopy([s_obstacle.shape.buffer(2) for s_obstacle in dg_scenario.static_obstacles.values()]))
    # optimal_path, path_list = rrt_start.planning()
    # shapely_viz.add_shape(optimal_path)
    # print('Spacecraft:', model.sg.w_half)

    # ax = shapely_viz.ax
    # ax.autoscale()
    # ax.set_facecolor('k')
    # ax.set_aspect("equal")
    # plt.show()

    out = "/home/hjlee/ETH_2021_Fall/PDM4AR/PDM4AR_github/"
    ex = get_final21()
    for i, alg_in in enumerate(ex.test_values):
        try:
            i_str = alg_in.str_id() + str(i)
        except:
            i_str = str(i)
        alg_out = ex.algorithm(alg_in)
        report = ex.report(alg_in, alg_out)
        report_file = os.path.join(out, f"final21-{i_str}.html")
        report.to_html(report_file)



