import multiprocessing
import numpy as np
import os
import pandas as pd
import csv
from itertools import product
import matplotlib.pyplot as plt

from MapEnvironment import MapEnvironment
from RRTMotionPlanner import RRTMotionPlanner, RRTMode
from RRTInspectionPlanner import RRTInspectionPlanner, OptimizationMode
import seaborn as sns


def run_mp(args):
    map_file, task, ext_mode, goal_prob, rrt_mode = args
    np.random.seed(os.getpid())
    planning_env = MapEnvironment(json_file=map_file, task=task)
    planner = RRTMotionPlanner(planning_env=planning_env, ext_mode=ext_mode, goal_prob=goal_prob, max_step_size=0.05)
    plan, cost, time = planner.plan(rrt_mode=rrt_mode, verbose=False, return_logs=True)
    return plan, cost, time

def run_ip(args):
    map_file, task, ext_mode, goal_prob, coverage, optimization_mode = args
    np.random.seed(os.getpid())
    planning_env = MapEnvironment(json_file=map_file, task=task)
    planner = RRTInspectionPlanner(planning_env=planning_env, ext_mode=ext_mode, goal_prob=goal_prob, coverage=coverage)
    plan, cost, coverage, time = planner.plan(verbose=False, return_logs=True, optimization_mode=optimization_mode)
    return plan, cost, coverage, time

def do_mp_experiment(num_experiments, goal_bias=0.05, rrt_mode: RRTMode = RRTMode.RRTStar):
    csv_file = "mp_results.csv"
    frame = pd.DataFrame(columns=["rrt_mode", "plan_length", "cost", "goal_bias", "time", "plan"]) # type: pd.DataFrame
    if not os.path.isfile(csv_file):
        frame.to_csv(csv_file, index=False)
    current_experiment = 1
    args = ["map_mp.json", "mp", "E1", goal_bias, rrt_mode]
    with multiprocessing.Pool(processes=8) as pool:
        results = pool.imap_unordered(run_mp, [args] * num_experiments)
        for result in results:
            plan, cost, time = result
            print(f"#{current_experiment}: plan length: {len(plan)}, plan cost: {cost:.3f}, time taken: {time:.3f}")
            current_experiment += 1
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                plan_list = [tuple(point) for point in plan]
                writer.writerow([rrt_mode.value, len(plan), cost, goal_bias, time, plan_list])

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

def parse_ip_results(coverage_target=0.5, visualize=False):
    df = pd.read_csv("ip_results.csv")
    df = df[df["coverage_target"] == coverage_target]
    unique_modes = df["mode"].unique()
    table = pd.DataFrame(columns=["Mode", "Average Plan Length", "Average Cost", "Average Coverage", "Average Time", "Average Time of Top 90%", "Number of Experiments", "Cheapest Path"])
    for mode in unique_modes:
        mode_df = df[df["mode"] == mode]
        row = {
            "Mode": mode,
            "Average Plan Length": mode_df['plan_length'].mean(),
            "Average Cost": mode_df['cost'].mean(),
            "Average Coverage": mode_df['coverage'].mean(),
            "Average Time": mode_df['time'].mean(),
            "Number of Experiments": len(mode_df),
            "Cheapest Path": mode_df['cost'].min(),
            "Average Time of Top 90%": mode_df[mode_df['time'] <= mode_df['time'].quantile(0.8)]['time'].mean()
        }
        table = pd.concat([table, pd.DataFrame(row, index=[0])], ignore_index=True)
    print(table)
    if not visualize:
        return
    # Find the plan that corresponds to the shortest path in each mode:
    for mode in unique_modes:
        mode_df = df[df["mode"] == mode]
        min_cost = mode_df["cost"].min()
        best_plan = mode_df[mode_df["cost"] == min_cost].iloc[0]["plan"]
        best_plan_coverage = mode_df[mode_df["cost"] == min_cost].iloc[0]["coverage"]
        # convert plan to numpy array
        best_plan = np.array([np.array(point) for point in eval(best_plan)])
        env = MapEnvironment("map_ip.json", "ip")
        title = f"Best plan for mode: {mode}, target coverage: {coverage_target}\ncost: {min_cost:.3f} coverage: {best_plan_coverage:.3f}"
        filename = f"best_ip_plan_{mode}_{coverage_target}"
        env.visualize_plan(best_plan, title, filename=filename)


def parse_mp_results(visualize):
    df = pd.read_csv("mp_results.csv")
    unique_rrt_modes = df["rrt_mode"].unique()
    unique_goal_biases = df["goal_bias"].unique()
    table = pd.DataFrame(columns=["RRT Mode", "Goal Bias", "Average Plan Length", "Average Cost", "Average Time", "Average Time of Top 90%", "Cheapest Path", "Number of Experiments"])
    for rrt_mode in unique_rrt_modes:
        for goal_bias in unique_goal_biases:
            mode_df = df[(df["rrt_mode"] == rrt_mode) & (df["goal_bias"] == goal_bias)]
            row = {
                "RRT Mode": rrt_mode,
                "Goal Bias": goal_bias,
                "Average Plan Length": mode_df['plan_length'].mean(),
                "Average Cost": mode_df['cost'].mean(),
                "Average Time": mode_df['time'].mean(),
                "Number of Experiments": len(mode_df),
                "Cheapest Path": mode_df['cost'].min(),
                "Average Time of Top 90%": mode_df[mode_df['time'] <= mode_df['time'].quantile(0.9)]['time'].mean()
            }
            table = pd.concat([table, pd.DataFrame(row, index=[0])], ignore_index=True)
    print(table)
    if not visualize:
        return
    for mode, goal_bias in product(unique_rrt_modes, unique_goal_biases):
        mode_df = df[df["rrt_mode"] == mode]
        min_cost = mode_df["cost"].min()
        best_plan = mode_df[mode_df["cost"] == min_cost].iloc[0]["plan"]
        # convert plan to numpy array
        best_plan = np.array([np.array(point) for point in eval(best_plan)])
        env = MapEnvironment("map_mp.json", "mp")
        title = f"Best plan for mode: {mode}, goal bias: {goal_bias}\ncost: {min_cost:.3f}"
        filename = f"best_mp_plan_{mode}_{goal_bias}"
        env.visualize_plan(best_plan, title, filename=filename)


def plot_time_distribution(coverage_target=0.5):
    df = pd.read_csv("ip_results.csv")
    df = df[df["coverage_target"] == coverage_target]
    sns.histplot(df, x="time", hue="mode", kde=True, stat="probability", bins=1500, multiple='dodge')
    plt.xlabel("Time taken (s)")
    plt.ylabel("Probability")
    plt.xlim(0,300)
    # plt.legend()
    plt.title(f"Time distribution for target coverage: {coverage_target}")
    plt.show()

if __name__ == "__main__":
    # plot_time_distribution(0.75)
    # parse_ip_results(0.5, visualize=False)
    # print("=========================================================")
    # parse_ip_results(0.75, visualize=False)
    # print("=========================================================")
    # parse_mp_results(visualize=False)
    while True:
        do_ip_experiment(50, 0.7, OptimizationMode.MultiAdd)
        do_ip_experiment(50, 0.75, OptimizationMode.MultiAdd)
        do_ip_experiment(50, 0.8, OptimizationMode.MultiAdd)
        do_ip_experiment(50, 0.85, OptimizationMode.MultiAdd)
        do_ip_experiment(50, 0.7, OptimizationMode.NoOptimization)
        do_ip_experiment(50, 0.75, OptimizationMode.NoOptimization)
        do_ip_experiment(50, 0.8, OptimizationMode.NoOptimization)
        do_ip_experiment(50, 0.85, OptimizationMode.NoOptimization)
    #     do_ip_experiment(50, 0.5, OptimizationMode.LocalDominance)
    #     do_ip_experiment(50, 0.5, OptimizationMode.GlobalDominance100)
    #     do_ip_experiment(50, 0.5, OptimizationMode.NoOptimization)
    #     do_ip_experiment(50, 0.75, OptimizationMode.LocalDominance)
    #     do_ip_experiment(50, 0.75, OptimizationMode.GlobalDominance100)
    #     do_ip_experiment(50, 0.75, OptimizationMode.NoOptimization)

        # do_mp_experiment(50, 0.05, RRTMode.RRTStar)
        # do_mp_experiment(50, 0.05, RRTMode.RRT)
        # do_mp_experiment(50, 0.2, RRTMode.RRTStar)
        # do_mp_experiment(50, 0.2, RRTMode.RRT)