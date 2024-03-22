import numpy as np
from MapEnvironment import MapEnvironment
from enum import Enum
import Robot
from typing import Set, Tuple

class TreeMode(Enum):
    MotionPlanning = 1
    Inspection = 2


class RRTree():
    def __init__(self, root_state, planning_env: MapEnvironment, mode: TreeMode = TreeMode.MotionPlanning):
        self.mode = mode
        self.env = planning_env # type: MapEnvironment
        self.root_state = tuple(root_state)
        self.root_node = RRNode(root_state, 0, None)
        self.vertices = {self.root_state: self.root_node}
        self.robot = Robot.Robot()
        self.max_coverage = 0
        self.max_coverage_state = 0
        self.max_step_size = 0.1

    def inspected_points_in_edge(self, start, end):
        num_interpolated_points = int(np.ceil(np.linalg.norm(np.array(end) - np.array(start)) / self.max_step_size))
        interpolated_path = np.linspace(start, end, num_interpolated_points + 1)
        path_inspected_points = [self.env.get_inspected_points(point) for point in interpolated_path]
        path_inspected_points = [points for points in path_inspected_points if len(points) > 0]
        if len(path_inspected_points) == 0:
            return set()
        path_inspected_points = np.concatenate(path_inspected_points, axis=0)
        path_inspected_points = set([tuple(point) for point in path_inspected_points])
        return path_inspected_points

    def insert_state(self, state: np.ndarray, parent_state: np.ndarray, inspected_points: np.ndarray = None):
        cost = self.robot.compute_distance(state, parent_state)
        state = tuple(state)
        if state in self.vertices.keys():
            return
        parent_state = tuple(parent_state)
        parent_node = self.vertices[parent_state]
        if self.mode == TreeMode.MotionPlanning:
            self.vertices[state] = RRNode(state, cost, parent_node)
        else:
            inspected_points = self.inspected_points_in_edge(parent_state, state)
            self.vertices[state] = RRNode(state, cost, parent_node, inspected_points)
            new_point_coverage = self.env.compute_coverage(self.vertices[state].total_inspected_points)
            if new_point_coverage > self.max_coverage:
                self.max_coverage = new_point_coverage
                self.max_coverage_state = state
    
    def distances_to_state(self, state, states):
        return np.linalg.norm(states - state, axis=1)
    
    def get_nearest_state(self, state: np.ndarray):
        vertices = list(self.vertices.keys())
        distances = self.distances_to_state(state, np.array(vertices))
        nearest_vertex_index = np.argmin(distances)
        return vertices[nearest_vertex_index]

    def cost_to_state(self, state):
        return self.vertices[tuple(state)].total_cost

    def get_state_parent(self, state):
        try:
            return self.vertices[tuple(state)].parent.state
        except:
            return None
    
    def path_to_state(self, state):
        path = []
        state = tuple(state)
        if state not in self.vertices.keys():
            return np.array([]), np.inf
        try:
            cost = self.vertices[state].total_cost
        except RecursionError:
            print(f"state: {state}")
            print(f"parent state: {self.vertices[state].parent.state}")
            print(f"parent parent state: {self.vertices[state].parent.parent.state}")
            exit()
        current_state = self.vertices[state]
        while current_state.parent:
            path.append(current_state.state)
            current_state = current_state.parent
        path.append(current_state.state)
        path.reverse()
        return np.array(path), cost

    def get_knn_states(self, pivot_state: np.ndarray, k: int):
        pivot_state_tuple = tuple(pivot_state)
        states_arr = list(self.vertices.keys())
        states_arr.remove(pivot_state_tuple)
        states_arr = np.array(states_arr)
        if len(states_arr) <= k:
            return states_arr

        distances = self.distances_to_state(pivot_state, states_arr)
        partitioned_states = states_arr[np.argpartition(distances, k).flatten()][:k]
        return partitioned_states

    def get_edges_as_states(self):
        return [(self.vertices[state].state, self.vertices[state].parent.state) for state in self.vertices.keys() if self.vertices[state].parent]

    def set_parent_for_state(self, state, new_parent):
        cost = self.robot.compute_distance(state, new_parent)
        state = tuple(state)
        new_parent = tuple(new_parent)
        # debug check: test that the new total inspected points matches expectations
        new_parent_node = self.vertices[new_parent]
        state_node = self.vertices[state]
        state_node.parent = new_parent_node
        state_node.inspected_points = self.inspected_points_in_edge(new_parent, state)
        assert self.vertices[state].parent == new_parent_node
        state_node.cost = cost
        if self.mode == TreeMode.Inspection:
            new_coverage = self.env.compute_coverage(self.vertices[state].total_inspected_points)
            if new_coverage > self.max_coverage:
                self.max_coverage = new_coverage
                self.max_coverage_state = state

    def get_total_inspected_points(self, state):
        return self.vertices[tuple(state)].total_inspected_points
    
    def get_inspected_points(self, state):
        return self.vertices[tuple(state)].inspected_points

class RRNode():
    def __init__(self, state: tuple[float, float], cost, parent_node: "RRNode | None" = None, inspected_points: set[Tuple[float, float]] = set()):
        self.cost = cost
        self.parent = parent_node # type: RRNode | None
        self.state = state
        self.inspected_points = inspected_points
        # self.total_inspected_points = self.inspected_points.union(self.parent.total_inspected_points) if self.parent else self.inspected_points

    @property
    def total_cost(self):
        if self.parent:
            return self.cost + self.parent.total_cost
        return 0

    @property
    def total_inspected_points(self):
        # return the total inspected points in the path to the node
        if not self.parent:
            return self.inspected_points
        return self.inspected_points.union(self.parent.total_inspected_points)