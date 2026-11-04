from baselines.heuristics.runtime import observe
import time
import math

from baselines.heuristics.utils.generate_init_solution import generate_multiweight_population
from baselines.heuristics.utils.load_data import *
import numpy as np
from baselines.heuristics.utils.operators import *
from baselines.heuristics.utils.tensorUtil import parse_vehicle_routes


# from Util.tensorUtil import parse_vehicle_routes

"""
算法随想
1. 
greedy 方法对于大实例的exploitation有帮助
但是对于小规模的实例，random对于求得全局最优解很重要，因此必须加入random的部分
2.
贪婪得初始解生成方法画地为牢，可能导致小规模实例难以求得最优解
3.
增大温度T，也会提高算法求解到小规模最优解的可能性
"""

destructOperatorList = [destroy_couple_random,
                        destroy_couple_worst_cost,
                        destroy_couple_worst_distance,
                        destroy_couple_shaw]
constructOperatorList = [repair_couple_greedy,
                         repair_couple_greedy_urgency_priority,
                         repair_couple_greedy_cost_priority]

d_operator_num = len(destructOperatorList)  # 破坏算子个数
c_operator_num = len(constructOperatorList)  # 重建算子个数
wDestruct = [1 for _ in range(d_operator_num)]  # 破坏算子初始权重
wConstruct = [1 for _ in range(c_operator_num)]  # 重建算子初始权重
last_diagnostics = {}


def initial_temperature(objective):
    """Initially accept a 2% deterioration with configured probability 0.5."""
    return max(float(objective), 1e-9) * ALNSConfig.initial_deterioration / -math.log(ALNSConfig.initial_acceptance)


def annealing_temperature(start_temperature, progress):
    progress = min(1., max(0., progress))
    return start_temperature * ALNSConfig.final_temperature_ratio ** progress


def destruct_construct(current_solution, d_num):
    destruct_index = np.random.choice(np.arange(len(wDestruct)), p=np.array(wDestruct) / sum(wDestruct))
    construct_index = np.random.choice(np.arange(len(wConstruct)), p=np.array(wConstruct) / sum(wConstruct))
    destroyed_info = destructOperatorList[destruct_index](current_solution, d_num)
    new_solution = constructOperatorList[construct_index](*destroyed_info, current_solution)
    return new_solution, destruct_index, construct_index


# 假设所有必要的库和配置都已导入
# import math, time, random
# from .utils import ...

def ALNS(instance_name, max_runtime=None, count=None):
    global wDestruct, wConstruct, last_diagnostics
    # --- 参数合法性检查 (保持不变) ---
    if max_runtime is not None and count is not None:
        raise ValueError("Error: max_runtime and count cannot be specified at the same time.")
    if max_runtime is None and count is None:
        raise ValueError("Error: You must specify either max_runtime or count as a stopping condition.")

    start_time = time.monotonic()
    instance = read_excel(instance_name)

    timesDestruct = [0 for _ in range(d_operator_num)]
    timesConstruct = [0 for _ in range(c_operator_num)]
    totalScoreDestruct = [0 for _ in range(d_operator_num)]
    totalScoreConstruct = [0 for _ in range(c_operator_num)]
    wDestruct = [1 for _ in range(d_operator_num)]
    wConstruct = [1 for _ in range(c_operator_num)]

    solution_table = set()

    task_num = len(instance)
    d_num = min(task_num, ALNSConfig.destroy_count)
    solution = generate_multiweight_population(instance)[0]
    init_solution = solution  # 保留初始解用于可能的返回
    greedy_time = time.monotonic() - start_time
    solution_table.add(solution.hash_key)
    current_fitness = solution.get_fitness()
    best_solution = solution
    best_fitness = current_fitness
    observe(best_solution, time.monotonic() - start_time, 0, "initial")
    start_temperature = initial_temperature(best_fitness)
    search_start = time.monotonic()
    search_budget = max(1e-9, max_runtime - (search_start - start_time)) if max_runtime is not None else None
    last_diagnostics = dict(destroy_count=d_num, initial_temperature=start_temperature,
                            final_temperature=start_temperature, completed_iterations=0,
                            worsening_candidates=0, accepted_worse=0, improving_moves=0,
                            equal_moves=0, weight_updates=0, update_interval=ALNSConfig.l_s)

    # --- 为停止条件和历史记录做准备 ---
    current_iteration = 0

    ### 关键改动 1: 初始化history列表，并记录初始解(t=0)的状态 ###
    fitness_history = [[best_fitness, time.monotonic() - start_time]]
    observe(best_solution, time.monotonic() - start_time, 0, "initialized")

    while max_runtime is None or time.monotonic() - start_time < max_runtime:
        # --- 核心算法逻辑 (保持不变) ---
        new_solution, destruct_index, construct_index = destruct_construct(solution, d_num)

        # ... (所有接受和评分逻辑保持不变) ...
        is_accept = False
        is_new = False
        if new_solution.hash_key not in solution_table:
            solution_table.add(new_solution.hash_key)
            is_new = True

        new_fitness = new_solution.get_fitness()
        progress = current_iteration / max(1, count) if count is not None else (time.monotonic() - search_start) / search_budget
        temperature = annealing_temperature(start_temperature, progress)
        last_diagnostics['final_temperature'] = temperature
        scoreDestruct, scoreConstruct = 0, 0

        if new_fitness < current_fitness:
            last_diagnostics['improving_moves'] += 1
            is_accept = True
            solution = new_solution
            current_fitness = new_fitness
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness

                ### 关键改动 2: 每当找到新的全局最优解时，记录时间和fitness ###
                elapsed_time = time.monotonic() - start_time
                fitness_history.append([best_fitness, elapsed_time])

                scoreDestruct = ALNSConfig.sigma_1
                scoreConstruct = ALNSConfig.sigma_1
            else:
                if is_new:
                    scoreDestruct = ALNSConfig.sigma_2
                    scoreConstruct = ALNSConfig.sigma_2
        # ... (elif 和 else 块保持不变) ...
        elif new_fitness == current_fitness:
            last_diagnostics['equal_moves'] += 1
            is_accept = True
            solution = new_solution
            if is_new:
                scoreDestruct = ALNSConfig.sigma_2
                scoreConstruct = ALNSConfig.sigma_2
        else:
            last_diagnostics['worsening_candidates'] += 1
            p_a = math.exp((current_fitness - new_fitness) / temperature)
            if random.random() < p_a:
                last_diagnostics['accepted_worse'] += 1
                is_accept = True
                solution = new_solution
                current_fitness = new_fitness
                if is_new:
                    scoreDestruct = ALNSConfig.sigma_3
                    scoreConstruct = ALNSConfig.sigma_3

        # ... (算子计分和权重更新逻辑保持不变) ...
        timesDestruct[destruct_index] += 1
        timesConstruct[construct_index] += 1
        totalScoreDestruct[destruct_index] += scoreDestruct
        totalScoreConstruct[construct_index] += scoreConstruct

        if (current_iteration + 1) % ALNSConfig.l_s == 0:
            last_diagnostics['weight_updates'] += 1
            for i in range(d_operator_num):
                if timesDestruct[i] != 0:
                    dTime = timesDestruct[i]
                else:
                    dTime = 1

                wDestruct[i] = wDestruct[i] * (1 - ALNSConfig.rho) + ALNSConfig.rho * totalScoreDestruct[i] / dTime
                totalScoreDestruct[i] = 0
                timesDestruct[i] = 0
            for i in range(c_operator_num):
                if timesConstruct[i] != 0:
                    cTime = timesConstruct[i]
                else:
                    cTime = 1
                wConstruct[i] = wConstruct[i] * (1 - ALNSConfig.rho) + ALNSConfig.rho * totalScoreConstruct[i] / cTime
                totalScoreConstruct[i] = 0
                timesConstruct[i] = 0

        # --- 在循环末尾检查停止条件 (保持不变) ---
        current_iteration += 1
        last_diagnostics['completed_iterations'] = current_iteration
        observe(best_solution, time.monotonic() - start_time, current_iteration, "search")

        if count is not None:
            if current_iteration >= count:
                # print(f"ALNS Stopping: Reached iteration limit ({count}).")
                break
        elif max_runtime is not None:
            elapsed_time = time.monotonic() - start_time
            if elapsed_time >= max_runtime:
                # print(f"ALNS Stopping: Reached time limit ({max_runtime:.2f}s).")
                break

    ### 关键改动 3: 修改返回值，同时返回最佳解、初始解和历史记录 ###
    # 为了与您 run_alns_task 中特殊的 greedy_package 处理保持兼容
    # 我们返回一个字典包含初始解和最终解，再加上历史记录
    return init_solution, best_solution, fitness_history, greedy_time


def _calculate_and_package_metrics(costs, dist_costs, delay_costs, avg_runtime_per_rep):
    """
    Calculates statistics and packages data for a given set of solution metrics.

    Args:
        costs (list): List of total costs (fitness) for each run.
        dist_costs (list): List of distance costs for each run.
        delay_costs (list): List of delay costs (tardiness) for each run.
        avg_runtime_per_rep (float): Average runtime per repetition.

    Returns:
        list: A data package containing calculated statistics.
    """
    costs_np = np.array(costs)
    delay_costs_np = np.array(delay_costs)
    dist_costs_np = np.array(dist_costs)

    min_cost_delay = delay_costs_np.min()
    min_cost_dist = dist_costs_np.min()
    mean_cost_delay = delay_costs_np.mean()
    mean_cost_dist = dist_costs_np.mean()
    mean_cost = costs_np.mean()
    overall_best_cost = costs_np.min()  # The overall best cost found across all runs

    data_package = [
        None,  # Placeholder (e.g., for best solution code, if needed in package)
        dist_costs,
        delay_costs,
        avg_runtime_per_rep,
        min_cost_delay,
        min_cost_dist,
        mean_cost_delay,
        mean_cost_dist,
        mean_cost,
        overall_best_cost,
    ]
    return data_package


def run_alns_experiment(instance_name, repeated, max_runtime=None, count=None):
    runtime_start_total = time.time()

    # Lists to store metrics for ALNS solutions
    alns_costs = []
    alns_delay_costs = []
    alns_dist_costs = []

    # Lists to store metrics for initial greedy solutions
    greedy_costs = []
    greedy_delay_costs = []
    greedy_dist_costs = []
    greedy_time_list = []
    worse_alns_cost = 0  # Initialize with a very large number
    worse_alns_solution_code = None  # Stores the actual code/representation of the best ALNS solution

    all_final_fitnesses = []
    all_histories = []

    print(f"Starting ALNS for instance: {instance_name} with {repeated} repetitions...")

    for i in range(repeated):
        print(f"--- Repetition {i + 1}/{repeated} ALNS ---")

        # Execute the ALNS algorithm for the current repetition
        init_solution, alns_solution, fitness_history, greedy_time = ALNS(instance_name, max_runtime, count)
        greedy_time_list.append(greedy_time)
        # Process ALNS solution metrics
        alns_costs.append(alns_solution.get_fitness())
        alns_dist_costs.append(alns_solution.distance)
        alns_delay_costs.append(alns_solution.tardiness)

        # Process initial greedy solution metrics
        greedy_costs.append(init_solution.get_fitness())
        greedy_dist_costs.append(init_solution.distance)
        greedy_delay_costs.append(init_solution.tardiness)

        # Update best_alns_cost and best_alns_solution_code if the current ALNS solution is better
        if alns_solution.get_fitness() > worse_alns_cost:
            worse_alns_cost = alns_solution.get_fitness()
            worse_alns_solution_code = alns_solution.get_code()  # Store the code of the best ALNS solution found so far

        if fitness_history:
            all_final_fitnesses.append(fitness_history[-1][0])
        all_histories.append(fitness_history)

    total_runtime = time.time() - runtime_start_total
    avg_runtime_per_repetition = total_runtime / repeated

    # Calculate and package metrics for ALNS solutions
    alns_data_package = _calculate_and_package_metrics(
        alns_costs,
        alns_dist_costs,
        alns_delay_costs,
        avg_runtime_per_repetition
    )
    # Update the first element of alns_data_package with the best solution code
    alns_data_package[0] = parse_vehicle_routes(worse_alns_solution_code).tolist()

    # Calculate and package metrics for initial greedy solutions
    greedy_data_package = _calculate_and_package_metrics(
        greedy_costs,
        greedy_dist_costs,
        greedy_delay_costs,
        sum(greedy_time_list) / len(greedy_time_list)  # Assuming greedy solution runtime is part of the total ALNS runtime
    )

    # 选择中位数的历史记录作为代表
    median_run_index = np.argsort(all_final_fitnesses)[len(all_final_fitnesses) // 2]
    representative_history = all_histories[median_run_index]


    greedy_data_package[0] = parse_vehicle_routes(init_solution.get_code()).tolist()  # Store the code of the initial greedy solution


    return worse_alns_solution_code, {'alns' : alns_data_package, "greedy" :greedy_data_package}, representative_history




if __name__ == "__main__":
    # 评估
    # code = [[[7, 3, 10, 15, 20, 23, 30, 29, 31, 38, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [2, 0, 8, 14, 17, 21, 25, 26, 32, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [1, 6, 9, 13, 16, 19, 27, 28, 37, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [4, 5, 11, 12, 18, 22, 24, 35, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [33, 39, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [36, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [34, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [7, 3, 10, 15, 20, 23, 30, 33, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [2, 0, 8, 14, 17, 21, 25, 31, 39, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [1, 6, 9, 13, 16, 19, 27, 28, 26, 29, 37, 32, 36, 38, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [4, 5, 11, 12, 18, 22, 24, 35, 34, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]]
    # code = convert_to_predefined_routes(code, 40)
    # code = [
    #     [0, 8, 4, 11, 16, 21, 24, 31, 34, 0, 3, 1, 9, 15, 18, 22, 26, 32, 40, 0, 2, 7, 10, 14, 17, 20, 28, 29, 27, 30, 38, 33, 37, 39, 0, 5, 6, 12, 13, 19, 23, 25, 36, 35],
    #     [0, 8, 4, 11, 16, 21, 24, 31, 30, 32, 39, 0, 3, 1, 9, 15, 18, 22, 26, 27, 33, 0, 2, 7, 10, 14, 17, 20, 28, 29, 38, 0, 5, 6, 12, 13, 19, 23, 25, 36, 0, 34, 40, 0, 37, 0, 35,0]
    # ]
    # code = [[0, 9, 2, 1, 5, 0, 20, 6, 13, 8, 17, 0, 4, 10, 3, 19, 12, 16, 0, 14, 11, 18, 15, 7], [0, 14, 10, 15, 7, 0, 3, 19, 0, 4, 2, 8, 17, 0, 20, 16, 0, 18, 0, 11, 1, 0, 6, 12, 0, 9, 13, 5]]
    # path_map = code2path_map(code)
    # sequence_map, path_init = path_map2sequence_map(path_map)
    # solution = Solution(instance, sequence_map, path_init)

    _, solution, _, _ = ALNS(instance_name=instance_name, max_runtime=60)
    print(solution.get_code())
    print(f"distance:{solution.distance}  tardiness:{solution.tardiness}")
    print(f"fitness = {solution.get_fitness()}")
    # print(parse_vehicle_routes(solution.get_code()))
    # pass
