import numpy as np
from RRTTree import RRTTree
import time
from InspectionTree import RRTree, TreeMode
from MapEnvironment import MapEnvironment
from enum import Enum

class RRTMode(Enum):
    RRT = "RRT"
    RRTStar = "RRTStar"


class RRTMotionPlanner(object):

    def __init__(self, planning_env: MapEnvironment, ext_mode, goal_prob, max_step_size = 0.1):

        # set environment and search tree
        self.planning_env = planning_env
        self.max_step_size = max_step_size
        self.tree = RRTree(planning_env.start, planning_env, TreeMode.MotionPlanning)

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.goal_state = self.planning_env.goal

    def plan(self, rrt_mode: RRTMode = RRTMode.RRT, verbose=True, return_logs=False):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        start_time = time.perf_counter()
        # initialize an empty plan.
        plan = []
        goal_found = False
        num_iterations = 0
        while not goal_found:
            rand_state = self.get_random_configuration()
            near_state = self.tree.get_nearest_state(rand_state)
            new_state = self.extend(near_state, rand_state)
            if not self.planning_env.config_validity_checker(new_state):
                continue # new state is not valid
            if self.planning_env.edge_validity_checker(near_state, new_state):
                # if len(self.tree.vertices) % 10 == 0:
                #     print(f"iteration {num_iterations}, vertices: {len(self.tree.vertices)}")
                self.tree.insert_state(new_state, near_state)

                if rrt_mode == RRTMode.RRTStar:
                    near_states = self.tree.get_knn_states(new_state, int(np.log(len(self.tree.vertices))))
                    near_states = [state for state in near_states if self.planning_env.edge_validity_checker(state, new_state)]
                    for state in near_states:
                        self.rewire(state, new_state)
                    for state in near_states:
                        self.rewire(new_state, state)
                
                if np.array_equal(new_state, self.goal_state):
                    goal_found = True
            num_iterations += 1
        
        total_time = time.perf_counter() - start_time
        plan, cost = self.tree.path_to_state(self.goal_state)
        if verbose:
            print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
            print('Total time: {:.2f}'.format(total_time))
            print(f"cache stats: {self.planning_env.cache_hits} hits, {self.planning_env.cache_misses} misses")

        if return_logs:
            return np.array(plan), cost, total_time

        return np.array(plan)

    def get_random_configuration(self) -> np.ndarray:
        if np.random.uniform(0, 1) < self.goal_prob:
            return self.goal_state
        return np.random.uniform(-np.pi, +np.pi, 4)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        return np.sum(np.linalg.norm(np.diff(plan, axis=0), axis=1))

    def rewire(self, potential_parent, child):
        cost = np.linalg.norm(potential_parent - child)
        if self.tree.cost_to_state(potential_parent) + cost < self.tree.cost_to_state(child):
            self.tree.set_parent_for_state(child, potential_parent)

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        if self.ext_mode == 'E1':
            return rand_config
    
        difference_vec = rand_config - near_config
        distance = np.linalg.norm(difference_vec)
        if distance <= self.max_step_size:
            return rand_config
        
        return near_config + (difference_vec / distance) * self.max_step_size