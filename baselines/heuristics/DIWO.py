from baselines.heuristics.runtime import observe
from baselines.heuristics.utils.tensorUtil import parse_vehicle_routes
from baselines.heuristics.utils.util import *
from baselines.heuristics.utils.load_data import *
import random
from baselines.heuristics.utils.Solution import Solution
import math
from baselines.heuristics.utils.generate_init_solution import generate_multiweight_population
from baselines.heuristics.ALNS import _calculate_and_package_metrics

'''入侵杂草算法参数'''
POP_INITIAL_SIZE = 80     # 初始种群大小
POP_MAX_SIZE = 100         # 种群最大大小
S_MAX = 20                # seed最大数
S_MIN = 1                 # seed最小数



def neighbor_insertion(solution):
    task = random.randint(1, solution.task_num)
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()

    remove_(sequence_map, path_init_task_map, task)

    parent_positions = get_all_position(sequence_map, path_init_task_map, task, 'parent')

    parent_position = random.sample(list(parent_positions), 1)[0]
    insert_(sequence_map, path_init_task_map, task, parent_position, 'parent')

    child_positions = get_feasible_insert_position(sequence_map, path_init_task_map, task, 'child')
    child_position = random.sample(list(child_positions), 1)[0]
    insert_(sequence_map, path_init_task_map, task, child_position, 'child')
    return Solution(solution.instance, sequence_map, path_init_task_map)


def neighbor_swap(solution):
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()
    task_list = list(range(1, solution.task_num + 1))
    task_1 = random.sample(task_list, 1)[0]
    task_list.remove(task_1)
    task_2 = random.sample(task_list, 1)[0]
    '''如果任务是某个AGV的第一个任务，更新path_init_task_map'''
    if sequence_map[task_1]['parent_pre_task'] == 0:
        path_init_task_map[sequence_map[task_1]['parent']] = task_2

    if sequence_map[task_1]['child_pre_task'] == 0:
        path_init_task_map[sequence_map[task_1]['child']] = task_2

    if sequence_map[task_2]['parent_pre_task'] == 0:
        path_init_task_map[sequence_map[task_2]['parent']] = task_1

    if sequence_map[task_2]['child_pre_task'] == 0:
        path_init_task_map[sequence_map[task_2]['child']] = task_1

    '''交换task_1和task_2'''
    task_1_parent = sequence_map[task_1]['parent']
    task_1_parent_pre_task = sequence_map[task_1]['parent_pre_task']
    task_1_parent_next_task = sequence_map[task_1]['parent_next_task']
    task_1_child = sequence_map[task_1]['child']
    task_1_child_pre_task = sequence_map[task_1]['child_pre_task']
    task_1_child_next_task = sequence_map[task_1]['child_next_task']

    if sequence_map[task_1]['parent_pre_task'] != 0 and sequence_map[task_1]['parent_pre_task'] != task_2:
        sequence_map[sequence_map[task_1]['parent_pre_task']]['parent_next_task'] = task_2

    if sequence_map[task_1]['parent_next_task'] != 0 and sequence_map[task_1]['parent_next_task'] != task_2:
        sequence_map[sequence_map[task_1]['parent_next_task']]['parent_pre_task'] = task_2

    if sequence_map[task_1]['child_pre_task'] != 0 and sequence_map[task_1]['child_pre_task'] != task_2:
        sequence_map[sequence_map[task_1]['child_pre_task']]['child_next_task'] = task_2

    if sequence_map[task_1]['child_next_task'] != 0 and sequence_map[task_1]['child_next_task'] != task_2:
        sequence_map[sequence_map[task_1]['child_next_task']]['child_pre_task'] = task_2

    ''''''

    if sequence_map[task_2]['parent_pre_task'] != 0 and sequence_map[task_2]['parent_pre_task'] != task_1:
        sequence_map[sequence_map[task_2]['parent_pre_task']]['parent_next_task'] = task_1

    if sequence_map[task_2]['parent_next_task'] != 0 and sequence_map[task_2]['parent_next_task'] != task_1:
        sequence_map[sequence_map[task_2]['parent_next_task']]['parent_pre_task'] = task_1

    if sequence_map[task_2]['child_pre_task'] != 0 and sequence_map[task_2]['child_pre_task'] != task_1:
        sequence_map[sequence_map[task_2]['child_pre_task']]['child_next_task'] = task_1

    if sequence_map[task_2]['child_next_task'] != 0 and sequence_map[task_2]['child_next_task'] != task_1:
        sequence_map[sequence_map[task_2]['child_next_task']]['child_pre_task'] = task_1

    sequence_map[task_1]['parent'] = sequence_map[task_2]['parent']
    if sequence_map[task_2]['parent_pre_task'] == task_1:
        sequence_map[task_1]['parent_pre_task'] = task_2
    else:
        sequence_map[task_1]['parent_pre_task'] = sequence_map[task_2]['parent_pre_task']

    if sequence_map[task_2]['parent_next_task'] == task_1:
        sequence_map[task_1]['parent_next_task'] = task_2
    else:
        sequence_map[task_1]['parent_next_task'] = sequence_map[task_2]['parent_next_task']

    sequence_map[task_1]['child'] = sequence_map[task_2]['child']

    if sequence_map[task_2]['child_pre_task'] == task_1:
        sequence_map[task_1]['child_pre_task'] = task_2
    else:
        sequence_map[task_1]['child_pre_task'] = sequence_map[task_2]['child_pre_task']

    if sequence_map[task_2]['child_next_task'] == task_1:
        sequence_map[task_1]['child_next_task'] = task_2
    else:
        sequence_map[task_1]['child_next_task'] = sequence_map[task_2]['child_next_task']


    sequence_map[task_2]['parent'] = task_1_parent

    if task_1_parent_pre_task == task_2:
        sequence_map[task_2]['parent_pre_task'] = task_1
    else:
        sequence_map[task_2]['parent_pre_task'] = task_1_parent_pre_task

    if task_1_parent_next_task == task_2:
        sequence_map[task_2]['parent_next_task'] = task_1
    else:
        sequence_map[task_2]['parent_next_task'] = task_1_parent_next_task

    sequence_map[task_2]['child'] = task_1_child

    if task_1_child_pre_task == task_2:
        sequence_map[task_2]['child_pre_task'] = task_1
    else:
        sequence_map[task_2]['child_pre_task'] = task_1_child_pre_task

    if task_1_child_next_task == task_2:
        sequence_map[task_2]['child_next_task'] = task_1
    else:
        sequence_map[task_2]['child_next_task'] = task_1_child_next_task

    return Solution(solution.instance, sequence_map, path_init_task_map)

def neighbor_route_swap(solution):
    """Exchange short segments, rather than just relabel identical vehicles."""
    path_map = solution.get_path_map()
    vehicles = Config.M_parent_list if random.randint(0, 1) == 0 else Config.M_child_list
    if len(vehicles) < 2:
        return solution
    for _ in range(8):
        first, second = random.sample(vehicles, 2)
        a, b = path_map[first], path_map[second]
        if not a and not b:
            continue
        na = random.randint(1, min(3, len(a))) if a else 0
        nb = random.randint(1, min(3, len(b))) if b else 0
        ia, ib = random.randint(0, len(a) - na), random.randint(0, len(b) - nb)
        if na == len(a) and nb == len(b):
            continue
        candidate = copy_dict_int_list(path_map)
        candidate[first] = a[:ia] + b[ib:ib + nb] + a[ia + na:]
        candidate[second] = b[:ib] + a[ia:ia + na] + b[ib + nb:]
        sequence_map, initial = path_map2sequence_map(candidate)
        _, feasible = cal_fitness_feasible(solution.instance, sequence_map, initial)
        if feasible:
            return Solution(solution.instance, sequence_map, initial)
    return solution



def get_neighbor_solution(solution):
    neighbor_list = [neighbor_insertion, neighbor_swap, neighbor_route_swap]
    neighbor_index = random.randint(0,2)
    neighbor_solution = neighbor_list[neighbor_index](solution)
    return neighbor_solution


def DIWO(instance_name, max_runtime=None, count=None):
    # --- 参数合法性检查 (保持不变) ---
    if max_runtime is not None and count is not None:
        raise ValueError("Error: max_runtime and count cannot be specified at the same time.")
    if max_runtime is None and count is None:
        raise ValueError("Error: You must specify either max_runtime or count as a stopping condition.")

    # --- 初始化逻辑 (基本不变) ---
    start_time = time.monotonic()
    instance = read_excel(instance_name)
    weed_list = generate_multiweight_population(instance, POP_INITIAL_SIZE)
    best_solution = weed_list[0]
    best_fitness = best_solution.get_fitness()
    observe(best_solution, time.monotonic() - start_time, 0, "initial")

    # --- 为停止条件和历史记录做准备 ---
    current_iteration = 0

    ### 关键改动 1: 初始化history列表，并记录初始解(t=0)的状态 ###
    fitness_history = [[best_fitness, time.monotonic() - start_time]]
    observe(best_solution, time.monotonic() - start_time, 0, "initialized")

    while max_runtime is None or time.monotonic() - start_time < max_runtime:
        # --- 核心算法逻辑 ---
        fitness_list = [solution.get_fitness() for solution in weed_list]
        min_fitness = min(fitness_list)
        max_fitness = max(fitness_list)

        # 当种群中的最优个体比记录的全局最优更好时
        if min_fitness < best_fitness:
            best_fitness = min_fitness

            ### 关键改动 2: 每当找到新的全局最优解时，记录时间和fitness ###
            elapsed_time = time.monotonic() - start_time
            fitness_history.append([best_fitness, elapsed_time])

            # best_solution 会在循环末尾从排序后的列表中更新，这里无需操作

        seed_list = []
        for weed_solution in weed_list:
            weed_fitness = weed_solution.get_fitness()
            try:
                if abs(max_fitness - min_fitness) < 1e-5:
                    seed_num = random.randint(S_MIN, S_MAX)
                else:
                    seed_num = math.floor(S_MAX - (weed_fitness - min_fitness) / (max_fitness - min_fitness) * \
                                          (S_MAX - S_MIN))
            except:
                seed_num = random.randint(S_MIN, S_MAX)

            for seed_index in range(seed_num):
                new_solution = get_neighbor_solution(weed_solution)
                seed_list.append(new_solution)

        weed_seed_list = seed_list + weed_list
        weed_seed_list = sorted(weed_seed_list, key=lambda x: x.get_fitness())
        best_solution = weed_seed_list[0]
        weed_list = weed_seed_list[:POP_MAX_SIZE]

        # --- 停止条件检查 (保持不变) ---
        current_iteration += 1
        observe(best_solution, time.monotonic() - start_time, current_iteration, "search")
        if count is not None:
            if current_iteration >= count:
                # print(f"DIWO Stopping: Reached iteration limit ({count}).")
                break
        elif max_runtime is not None:
            elapsed_time = time.monotonic() - start_time
            if elapsed_time >= max_runtime:
                # print(f"DIWO Stopping: Reached time limit ({max_runtime:.2f}s).")
                break

    ### 关键改动 3: 修改返回值，同时返回最佳解和历史记录 ###
    return best_solution, fitness_history


def run_DIWO_experiment(instance_name, repeated, max_runtime=None, count=None):
    runtime_start_total = time.time()

    # Lists to store metrics for ALNS solutions
    costs = []
    delay_costs = []
    dist_costs = []
    worse_cost = 0  # Initialize with a very large number

    all_final_fitnesses = []
    all_histories = []

    worse_cost = 0  # Initialize with a very large number
    worse_solution_code = None  # Stores the actual code/representation of the best ALNS solution

    print(f"Starting DIWO for instance: {instance_name} with {repeated} repetitions...")

    for i in range(repeated):
        print(f"--- Repetition {i + 1}/{repeated} DIWO ---")

        # Execute the ALNS algorithm for the current repetition
        solution, fitness_history = DIWO(instance_name, max_runtime, count)

        # Process ALNS solution metrics
        costs.append(solution.get_fitness())
        dist_costs.append(solution.distance)
        delay_costs.append(solution.tardiness)

        # Update best_alns_cost and best_alns_solution_code if the current ALNS solution is better
        if solution.get_fitness() > worse_cost:
            worse_cost = solution.get_fitness()
            worse_solution_code = solution.get_code()  # Store the code of the best ALNS solution found so far

        if fitness_history:
            all_final_fitnesses.append(fitness_history[-1][0])
        all_histories.append(fitness_history)

    total_runtime = time.time() - runtime_start_total
    avg_runtime_per_repetition = total_runtime / repeated

    # Calculate and package metrics for ALNS solutions
    data_package = _calculate_and_package_metrics(
        costs,
        dist_costs,
        delay_costs,
        avg_runtime_per_repetition
    )
    median_run_index = np.argsort(all_final_fitnesses)[len(all_final_fitnesses) // 2]
    representative_history = all_histories[median_run_index]
    # Update the first element of alns_data_package with the best solution code
    data_package[0] = parse_vehicle_routes(worse_solution_code).tolist()
    return worse_solution_code, data_package, representative_history



if __name__ == "__main__":
    instance_name = "tasks_50"
    solution = DIWO(instance_name)
    print(f"fitness:{solution.get_fitness()}")
    print(f"distance:{solution.distance}  tardiness:{solution.tardiness}")
    pass
