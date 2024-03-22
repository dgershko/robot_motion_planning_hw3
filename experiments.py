import multiprocessing
import numpy as np
import os
import pandas as pd
import csv

from MapEnvironment import MapEnvironment
from RRTMotionPlanner import RRTMotionPlanner
from RRTInspectionPlanner import RRTInspectionPlanner, OptimizationMode


def run_mp(args):
    planning_env = MapEnvironment(json_file=args.map, task=args.task)
    planner = RRTMotionPlanner(planning_env=planning_env, ext_mode=args.ext_mode, goal_prob=args.goal_prob, max_step_size=0.05)
    plan = planner.plan()
    return plan

def run_ip(args):
    map_file, task, ext_mode, goal_prob, coverage, optimization_mode = args
    np.random.seed(os.getpid())
    planning_env = MapEnvironment(json_file=map_file, task=task)
    planner = RRTInspectionPlanner(planning_env=planning_env, ext_mode=ext_mode, goal_prob=goal_prob, coverage=coverage)
    plan, cost, coverage, time = planner.plan(verbose=False, return_logs=True, optimization_mode=optimization_mode)
    return plan, cost, coverage, time

def do_ip_experiment(num_experiments, coverage_target, optimization_mode):
    csv_file = "ip_results.csv"
    frame = pd.DataFrame(columns=["mode", "plan_length", "cost", "coverage", "time", "coverage_target", "plan"]) # type: pd.DataFrame
    if not os.path.isfile(csv_file):
        frame.to_csv(csv_file, index=False)
    current_experiment = 1
    args = ["map_ip.json", "ip", "E1", 0.05, coverage_target, optimization_mode]
    with multiprocessing.Pool(processes=8) as pool:
        results = pool.imap_unordered(run_ip, [args] * num_experiments)
        for result in results:
            plan, cost, coverage, time = result
            print(f"#{current_experiment}: mode: {optimization_mode.value} plan length: {len(plan)}, cost: {cost:.3f}, coverage: {coverage:.3f}, time taken: {time:.3f}")
            current_experiment += 1
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                plan_list = [tuple(point) for point in plan]
                writer.writerow([optimization_mode.value, len(plan), cost, coverage, time, coverage_target, plan_list])

def parse_results(coverage_target=0.5):
    df = pd.read_csv("ip_results.csv")
    df = df[df["coverage_target"] == coverage_target]
    for mode in df["mode"].unique():
        mode_df = df[df["mode"] == mode]
        print(f"Mode: {mode}")
        print(f"Average plan length: {mode_df['plan_length'].mean()}")
        print(f"Average cost: {mode_df['cost'].mean()}")
        print(f"Average coverage: {mode_df['coverage'].mean()}")
        print(f"Average time: {mode_df['time'].mean()}")
        print(f"Number of experiments: {len(mode_df)}")
        # Find the plan that corresponds to the shortest path in this mode:
        min_cost = mode_df["cost"].min()
        best_plan = mode_df[mode_df["cost"] == min_cost].iloc[0]["plan"]
        best_plan_coverage = mode_df[mode_df["cost"] == min_cost].iloc[0]["coverage"]
        # convert plan to numpy array
        best_plan = np.array([np.array(point) for point in eval(best_plan)])
        env = MapEnvironment("map_ip.json", "ip")
        title = f"Best plan for mode: {mode}, target coverage: {coverage_target}\ncost: {min_cost:.3f} coverage: {best_plan_coverage:.3f}"
        env.visualize_plan(best_plan, title)

if __name__ == "__main__":
    # parse_results()
    while True:
        do_ip_experiment(50, 0.75, OptimizationMode.LocalDominanceByCov)
        do_ip_experiment(50, 0.75, OptimizationMode.LocalDominanceByCost)
        do_ip_experiment(50, 0.75, OptimizationMode.NoOptimization)
