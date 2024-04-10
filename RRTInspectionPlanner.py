import numpy as np
from RRTTree import RRTTree
import time
from InspectionTree import RRTree, TreeMode
from MapEnvironment import MapEnvironment
from enum import Enum

class OptimizationMode(Enum):
    NoOptimization = "NoOptimization"
    LocalDominance = "LocalDominance"
    GlobalDominance = "GlobalDominance"
    MultiTree = "MultiTree"
    MultiAdd = "MultiAdd"

class RRTInspectionPlanner(object):

    def __init__(self, planning_env: MapEnvironment, ext_mode, goal_prob, coverage):

        # set environment and search tree
        self.planning_env = planning_env
        # self.tree = RRTTree(self.planning_env, task="ip")
        self.tree = RRTree(planning_env.start, planning_env, TreeMode.Inspection)

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.coverage = coverage
        self.max_step_size = 0.1

    def plan(self, verbose=True, return_logs=False, optimization_mode: OptimizationMode = OptimizationMode.NoOptimization):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        self.optimization_mode = optimization_mode
        start_time = time.perf_counter()
        plan = []
        if self.optimization_mode == OptimizationMode.NoOptimization:
            plan, cost = self.normal_plan()
        elif self.optimization_mode == OptimizationMode.LocalDominance:
            plan, cost = self.local_dominance_plan()
        elif self.optimization_mode == OptimizationMode.GlobalDominance:
            plan, cost = self.global_dominance_plan()
        # elif self.optimization_mode == OptimizationMode.MultiTree:
        #     plan, cost = self.multi_tree_plan()
        elif self.optimization_mode == OptimizationMode.MultiAdd:
            plan, cost = self.multi_add_plan()
        
        total_time = time.perf_counter() - start_time
        goal_state = plan[-1]
        # print total path cost and time
        if verbose:
            print('Total cost of path: {:.2f}'.format(self.compute_cost(plan)))
            print('Total time: {:.2f}'.format(total_time))
            print(f'Total coverage: {self.planning_env.compute_coverage(self.tree.get_total_inspected_points(goal_state))}')
            print(plan)
        if return_logs:
            return plan, cost, self.tree.max_coverage, total_time
        return plan

    def normal_plan(self) -> tuple[np.ndarray, float]:
        num_iterations = 0
        max_coverage = 0
        while True:
            num_iterations += 1
            rand_state = self.get_random_configuration()
            near_state = self.tree.get_nearest_state(rand_state)
            new_state = self.extend(near_state, rand_state)
            if not self.planning_env.config_validity_checker(new_state):
                continue
            if self.planning_env.edge_validity_checker(near_state, new_state):
                inspected_points = self.planning_env.get_inspected_points(new_state)
                self.tree.insert_state(new_state, near_state, inspected_points)

            if self.tree.max_coverage > max_coverage:
                max_coverage = self.tree.max_coverage
            if self.tree.max_coverage >= self.coverage:
                goal_state = self.tree.max_coverage_state
                break

        return self.tree.path_to_state(goal_state)


    def multi_add_plan(self) -> tuple[np.ndarray, float]:
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        num_iterations = 0
        max_coverage = 0
        while True:
            num_iterations += 1
            rand_state = self.get_random_configuration()
            near_state = self.tree.get_nearest_state(rand_state)
            new_state = self.extend(near_state, rand_state)
            if not self.planning_env.config_validity_checker(new_state):
                continue
            if self.planning_env.edge_validity_checker(near_state, new_state):
                inspected_points = self.tree.inspected_points_in_edge(near_state, new_state)
                num_perturbations = 2 * (len(inspected_points) ** 2) # 2 * (n^2) perturbations
                # limit the number of perturbations to the log of the number of vertices in the tree
                # log_vertices = int(np.log(len(self.tree.vertices)))
                # num_perturbations = min(num_perturbations, log_vertices)
                # num_perturbations = min(num_perturbations, len(self.tree.vertices) - 1)
                num_perturbations = min(num_perturbations, len(self.tree.vertices) // 2)
                if num_perturbations <= 1:
                    self.tree.insert_state(new_state, near_state, inspected_points)
                    continue
                k_nearest_neighbors = self.tree.get_knn_states(near_state, num_perturbations)
                for i in range(num_perturbations):
                    parent_state = k_nearest_neighbors[i]
                    perturbed_state = new_state + np.random.normal(0, 0.00001, size=4)
                    if not self.planning_env.edge_validity_checker(parent_state, perturbed_state):
                        continue
                    self.tree.insert_state(perturbed_state, parent_state, inspected_points)

            if self.tree.max_coverage > max_coverage:
                max_coverage = self.tree.max_coverage
                print(f"max coverage: {max_coverage}, num iterations: {num_iterations}, num vertices: {len(self.tree.vertices)}")
            if self.tree.max_coverage >= self.coverage:
                goal_state = self.tree.max_coverage_state
                break
    
        return self.tree.path_to_state(goal_state)

    def local_dominance_plan(self) -> tuple[np.ndarray, float]:
        num_iterations = 0
        max_coverage = 0
        while True:
            num_iterations += 1
            rand_state = self.get_random_configuration()
            near_state = self.tree.get_nearest_state(rand_state)
            new_state = self.extend(near_state, rand_state)
            if not self.planning_env.config_validity_checker(new_state):
                continue # new state is not valid
            if self.planning_env.edge_validity_checker(near_state, new_state):
                inspected_points = self.planning_env.get_inspected_points(new_state)
                self.tree.insert_state(new_state, near_state, inspected_points)
                # local dominance optimization
                near_states = self.tree.get_knn_states(new_state, int(np.log(len(self.tree.vertices))))
                dominating_parents = self.local_dominated(new_state, near_states)
                if len(dominating_parents) > 0:
                    self.update_parent(new_state, dominating_parents)

            if self.tree.max_coverage > max_coverage:
                max_coverage = self.tree.max_coverage
            if self.tree.max_coverage >= self.coverage:
                goal_state = self.tree.max_coverage_state
                break
        
        return self.tree.path_to_state(goal_state)
    
    def global_dominance_plan(self, rewire_iters = 100) -> tuple[np.ndarray, float]:
        num_iterations = 0
        max_coverage = 0
        while True:
            num_iterations += 1
            rand_state = self.get_random_configuration()
            near_state = self.tree.get_nearest_state(rand_state)
            new_state = self.extend(near_state, rand_state)
            if not self.planning_env.config_validity_checker(new_state):
                continue
            if self.planning_env.edge_validity_checker(near_state, new_state):
                inspected_points = self.planning_env.get_inspected_points(new_state)
                self.tree.insert_state(new_state, near_state, inspected_points)
                # global dominance optimization
                if len(self.tree.vertices) % rewire_iters == 0:
                    self.rewire_tree()
            
            if self.tree.max_coverage > max_coverage:
                max_coverage = self.tree.max_coverage
            if self.tree.max_coverage >= self.coverage:
                goal_state = self.tree.max_coverage_state
                break

        return self.tree.path_to_state(goal_state)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        return np.sum(np.linalg.norm(np.diff(plan, axis=0), axis=1))

    def get_random_configuration(self) -> np.ndarray:
        return np.random.uniform(-np.pi, +np.pi, 4)

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
            
    def local_dominated(self, new_state, near_states):
        """
        Return the list of potential parents for new_state s.t. the path through them to it dominates the current path.
        """
        # filter out states for which the new path is longer
        near_states = [state for state in near_states if self.tree.cost_to_state(state) + np.linalg.norm(state - new_state) < self.tree.cost_to_state(new_state)]
        # filter out states for which the new path does not properly contain all the points inpected by the old path
        near_states = [state for state in near_states if 
                       self.tree.get_total_inspected_points(state).union(self.tree.inspected_points_in_edge(state, new_state)) > 
                       self.tree.get_total_inspected_points(new_state)
        ]
        # filter out states for which the edge is illegal
        near_states = [state for state in near_states if self.planning_env.edge_validity_checker(state, new_state)]
        return near_states

    def update_parent(self, new_state, dominating_states):
        """
        Return the best parent for new_state among the dominating states.
        Chooses the one with the largest inspected point set
        """
        # if self.optimization_mode == OptimizationMode.LocalDominanceByCost:
        costs = np.linalg.norm(dominating_states - new_state, axis=1)
        best_parent = dominating_states[np.argmin(costs)]
        # elif self.optimization_mode == OptimizationMode.LocalDominanceByCov:
        #     max_inspected_points = 0
        #     best_parent = None
        #     for potential_parent in dominating_states:
        #         potential_inspected_points = self.tree.get_total_inspected_points(potential_parent).union(self.tree.get_inspected_points(new_state))
        #         if len(potential_inspected_points) > max_inspected_points:
        #             max_inspected_points = len(potential_inspected_points)
        #             best_parent = potential_parent
        self.tree.set_parent_for_state(new_state, best_parent)

    def rewire_tree(self):
        for state in self.tree.vertices.keys():
            # other_states = list(self.tree.vertices.keys())
            other_states = self.tree.get_knn_states(state, int(np.log(len(self.tree.vertices))))
            # filter out states which are children of the current state
            other_states = [other_state for other_state in other_states if state not in self.tree.path_to_state(other_state)[0]]
            dominating_states = self.local_dominated(np.array(state), np.array(other_states))
            if len(dominating_states) > 0:
                self.update_parent(np.array(state), dominating_states)
