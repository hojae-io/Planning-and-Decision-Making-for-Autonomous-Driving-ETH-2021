from abc import abstractmethod, ABC
from typing import List, Optional

from pdm4ar.exercises.ex02.structures import AdjacencyList, X


class GraphSearch(ABC):
    @abstractmethod
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Optional[List[X]]:
        """
        :param graph: The given graph as an adjacency list
        :param start: The initial state (i.e. a node)
        :param goal: The goal state (i.e. a node)
        :return: The path from start to goal as a Sequence of states, None if a path does not exist
        """
        pass


class DepthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Optional[List[X]]:
        # todo implement here your solution
        return None


class BreadthFirst(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Optional[List[X]]:
        # todo implement here your solution
        return None


class IterativeDeepening(GraphSearch):
    def search(self, graph: AdjacencyList, start: X, goal: X) -> Optional[List[X]]:
        # todo implement here your solution
        return None
