from baselines.heuristics.runtime import observe

from baselines.heuristics.utils.generate_init_solution import generate_multiweight_population
from baselines.heuristics.utils.tensorUtil import parse_vehicle_routes
from baselines.heuristics.utils.util import *
import random
import numpy as np
from baselines.heuristics.utils.Solution import Solution
from baselines.heuristics.utils.load_data import *
from baselines.heuristics.ALNS import _calculate_and_package_metrics

POP_SIZE = 100  # 种群规模
CROSSOVER_RATE = 0.9  # 交叉概率
MUTATION_RATE = 0.1  # 变异概率


def get_task_position(path_map, task):
    for path_index, path in path_map.items():
        try:
            index = path.index(task)
            if path_index <= Config.M_parent:
                parent_path_index = path_index
                parent_path_task_index = index
            else:
                child_path_index = path_index
                child_path_task_index = index
        except ValueError:
            continue
    return parent_path_index, parent_path_task_index, child_path_index, child_path_task_index

def cross_over_task(path_map_1, path_map_2, task):
    """
    点交叉，把path_map_1中task的位置调整为task在path_map_2中的位置
    :param solution_1:
    :param solution_2:
    :param task:
    :return:
    """
    # path_map_1 = solution_1.get_path_map()
    # path_map_2 = solution_2.get_path_map()

    parent_path_index_1, parent_path_task_index_1, child_path_index_1, child_path_task_index_1 = get_task_position(path_map_1, task)
    parent_path_index_2, parent_path_task_index_2, child_path_index_2, child_path_task_index_2 = get_task_position(path_map_2, task)

    path_map_1[parent_path_index_1].remove(task)
    path_map_1[parent_path_index_2].insert(parent_path_task_index_2, task)

    path_map_1[child_path_index_1].remove(task)
    path_map_1[child_path_index_2].insert(child_path_task_index_2, task)

    return path_map_1

def crossover_and_mutation(pop):
    new_pop = []
    for father in pop:
        child = father
        if np.random.rand() < CROSSOVER_RATE:
            mother = random.choice(pop)
            child = cross_over_solution(father, mother)
        child = mutation(child)
        new_pop.append(child)
    return new_pop

def mutation(child):
    if np.random.rand() < MUTATION_RATE:
        return get_neighbor_solution(child)
    return child

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

def get_neighbor_solution(solution):
    neighbor_list = [neighbor_insertion, neighbor_swap]
    neighbor_index = random.randint(0,1)
    neighbor_solution = neighbor_list[neighbor_index](solution)
    return neighbor_solution
def cross_over_solution(solution_1, solution_2):
    """
    调整solution1的path_map，按照solution2的path_map的任务位置
    :param solution_1:
    :param solution_2:
    :return:
    """
    task_num = solution_1.task_num
    task_list = list(range(1, task_num + 1))
    random.shuffle(task_list)
    path_map_cross = solution_1.get_path_map()
    path_map_2 = solution_2.get_path_map()
    cross_task_num = 0
    for task in task_list:
        path_map_cross_temp = copy_dict_int_list(path_map_cross)
        path_map_cross_temp = cross_over_task(path_map_cross_temp, path_map_2, task)
        sequence_map, path_init_task_map = path_map2sequence_map(path_map_cross_temp)
        fitness, feasible = cal_fitness_feasible(solution_1.instance, sequence_map, path_init_task_map)
        if feasible:
            path_map_cross = path_map_cross_temp
            cross_task_num += 1
        if cross_task_num >= 1:
            break
    sequence_map, path_init_task_map = path_map2sequence_map(path_map_cross)
    return Solution(solution_1.instance, sequence_map, path_init_task_map)


def GA(instance_name, max_runtime=None, count=None):
    # --- 参数合法性检查 (保持不变) ---
    if max_runtime is not None and count is not None:
        raise ValueError("Error: max_runtime and count cannot be specified at the same time.")
    if max_runtime is None and count is None:
        raise ValueError("Error: You must specify either max_runtime or count as a stopping condition.")

    # --- 初始化逻辑 (基本不变) ---
    start_time = time.monotonic()
    instance = read_excel(instance_name)
    pop = generate_multiweight_population(instance, POP_SIZE)
    best_solution = pop[0]
    best_fitness = best_solution.get_fitness()
    observe(best_solution, time.monotonic() - start_time, 0, "initial")

    # --- 为停止条件和历史记录做准备 ---
    current_iteration = 0

    ### 关键改动 1: 初始化history列表，并记录初始解(t=0)的状态 ###
    fitness_history = [[best_fitness, time.monotonic() - start_time]]
    observe(best_solution, time.monotonic() - start_time, 0, "initialized")

    while max_runtime is None or time.monotonic() - start_time < max_runtime:
        # --- 核心算法逻辑 ---
        # 检查当前种群中是否有比记录的全局最优解更好的个体
        for solution in pop:
            fitness = solution.get_fitness()
            if fitness < best_fitness:
                best_solution = solution
                best_fitness = fitness

                ### 关键改动 2: 每当找到新的全局最优解时，记录时间和fitness ###
                elapsed_time = time.monotonic() - start_time
                fitness_history.append([best_fitness, elapsed_time])

        # 产生下一代
        new_pop = crossover_and_mutation(pop)
        pop = select(new_pop)

        # 精英保留策略：用至今为止找到的最好解替换一个随机个体
        pop[random.randint(0, len(pop) - 1)] = best_solution

        # --- 在循环末尾检查停止条件 ---
        current_iteration += 1
        observe(best_solution, time.monotonic() - start_time, current_iteration, "search")

        if count is not None:
            if current_iteration >= count:
                # print(f"GA Stopping: Reached iteration limit ({count}).")
                break
        elif max_runtime is not None:
            elapsed_time = time.monotonic() - start_time
            if elapsed_time >= max_runtime:
                # print(f"GA Stopping: Reached time limit ({max_runtime:.2f}s).")
                break

    ### 关键改动 3: 修改返回值，同时返回最佳解和历史记录 ###
    return best_solution, fitness_history

def select(pop):
    fitness_list = np.array([s.get_fitness() for s in pop])
    # hash_key_list = np.array([hash_key for hash_key in pop_map.keys()])
    # fitness_list = np.array([pop_map[hash_key].get_fitness() for hash_key in hash_key_list])
    for j in range(len(fitness_list)):
        fitness_list[j] = - fitness_list[j]

    fitness_list = (fitness_list - np.min(fitness_list)) + 1e-3
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness_list) / (fitness_list.sum()))
    return np.array(pop)[idx]


def run_GA_experiment(instance_name, repeated, max_runtime=None, count=None):
    runtime_start_total = time.time()

    # Lists to store metrics for ALNS solutions
    costs = []
    delay_costs = []
    dist_costs = []

    all_final_fitnesses = []
    all_histories = []

    worse_cost = 0  # Initialize with a very large number
    worse_solution_code = None  # Stores the actual code/representation of the best ALNS solution

    print(f"Starting GA for instance: {instance_name} with {repeated} repetitions...")

    for i in range(repeated):
        print(f"--- Repetition {i + 1}/{repeated} GA ---")

        # Execute the ALNS algorithm for the current repetition
        solution, fitness_history = GA(instance_name, max_runtime, count)

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

    # 选择中位数的历史记录作为代表
    median_run_index = np.argsort(all_final_fitnesses)[len(all_final_fitnesses) // 2]
    representative_history = all_histories[median_run_index]


    # Update the first element of alns_data_package with the best solution code
    data_package[0] = parse_vehicle_routes(worse_solution_code).tolist()
    return worse_solution_code, data_package, representative_history



# if __name__ == "__main__":
#     instance_name = "tasks_20"
#     solution = GA(instance_name)
#     print(f"fitness:{solution.get_fitness()}")
#     print(f"distance:{solution.distance}  tardiness:{solution.tardiness}")
#     pass
