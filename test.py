from MapEnvironment import MapEnvironment
from RRTMotionPlanner import RRTMotionPlanner
from RRTInspectionPlanner import RRTInspectionPlanner, OptimizationMode
from time import perf_counter 
import cProfile
import numpy as np

env = MapEnvironment("map_ip.json", "ip")
# profiler = cProfile.Profile()
# profiler.enable()

planner = RRTInspectionPlanner(env, "E1", 0.05, 0.75)
planner.plan(optimization_mode=OptimizationMode.NoOptimization)
planner = RRTInspectionPlanner(env, "E1", 0.05, 0.75)
planner.plan(optimization_mode=OptimizationMode.LocalDominanceByCov)
planner = RRTInspectionPlanner(env, "E1", 0.05, 0.75)
planner.plan(optimization_mode=OptimizationMode.LocalDominanceByCost)

# profiler.disable()

# profiler.print_stats(sort='cumtime')
# random_configs = np.random.uniform(-np.pi, +np.pi, size=(10000, 4))


# start_time = perf_counter()
# for config in random_configs:
#     res_1 = env.get_inspected_points(config)
# print(f"Time elapsed: {perf_counter() - start_time}")

# planner = RRTMotionPlanner(env, "E1", 0.05)
# plan = planner.plan()

# start_time = perf_counter()
# for idx, (start, end) in enumerate(random_configs):
#     res_1 = env.edge_validity_checker(start, end)
#     # res_2 = env.backup_edge_validity_checker(start, end)
#     # assert(res_1 == res_2)
# print(f"Time elapsed: {perf_counter() - start_time}")
# print(f"cache hits: {env.cache_hits}")
# print(f"cache misses: {env.cache_misses}")

# start_time = perf_counter()
# random_configs = np.random.uniform(-np.pi, +np.pi, size=(15000, 4))
# for idx, start in enumerate(random_configs):
#     # res_1 = env.config_validity_checker(start)
#     res_2 = env.backup_config_validity_checker(start)
#     # assert(res_1 == res_2)
# print(f"Time elapsed: {perf_counter() - start_time}")