from numpy.core.fromnumeric import choose
from shapely.geometry import Polygon
from pdm4ar.exercises_def.final21.scenario import get_dgscenario
import numpy as np
from typing import List
polygon = Polygon([(0,0), (1,1), (1,0)])
print(polygon.area, polygon.length)
print(np.random.random())
print((np.random.uniform(0, 100),
                         np.random.uniform(0, 100)))

a = [1, 3, 5, 2, 4, 6, 3, 5, 7]
print([ind if a[ind] < 5 else np.inf for ind in range(len(a)) ])

b = []
for i in b:
    print('hi')

def choose_parent(near_index:List[int]):
    for _ in near_index:
        print(_)

choose_parent([1,2,3,4,5])

print(np.array([1,2]) - np.array([2,3]))


