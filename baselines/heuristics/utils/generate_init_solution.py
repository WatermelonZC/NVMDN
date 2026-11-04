import copy
import random
import time

from baselines.heuristics.utils import Config
from baselines.heuristics.utils.Solution import Solution
from baselines.heuristics.utils.util import *
from baselines.heuristics.utils.load_data import read_excel


MULTIWEIGHT_WEIGHTS = tuple(i / 100 for i in range(101))


def generate_multiweight_population(instance, population_size=1):
    """Use the same deterministic candidate pool and best seed for all searches.

    A single-solution search gets the best candidate. Population searches take
    the requested number from the same ranked pool. No algorithm RNG is used.
    """
    if not instance or population_size < 1:
        raise ValueError('Initialization needs tasks and a positive population size')
    candidates = [generate_solution_nearest(instance, weight) for weight in MULTIWEIGHT_WEIGHTS]
    candidates.sort(key=lambda solution: solution.get_fitness())
    if not all(solution.feasible for solution in candidates):
        raise RuntimeError('Multiweight initialization produced an infeasible solution')
    return [candidates[i % len(candidates)] for i in range(population_size)]



def generate_solution_random(instance):
    task_num = len(instance)
    sequence_map = {}
    path_init_task_map = {}
    # 初始时所有path均为空
    for path_index in range(1, Config.M + 1):
        path_init_task_map[path_index] = 0
    for task in range(1, task_num + 1):
        chain_list = random.sample(['parent', 'child'], 2)
        chain = chain_list[0]
        coupled_chain = chain_list[1]
        all_position_set = get_all_position(sequence_map, path_init_task_map, task, chain)
        position = random.sample(list(all_position_set), 1)[0]
        insert_(sequence_map, path_init_task_map, task, position, chain)
        feasible_position_set = get_feasible_insert_position(sequence_map, path_init_task_map, task, coupled_chain)
        coupled_position = random.sample(list(feasible_position_set), 1)[0]
        insert_(sequence_map, path_init_task_map, task, coupled_position, coupled_chain)
    solution = Solution(instance, sequence_map, path_init_task_map)
    return solution


def generate_solution_greedy(instance):
    """
    随机顺序插入task
    遍历所有的位置，找到cost最小的
    :param instance:
    :return:
    """
    task_num = len(instance)
    task_list = [task for task in range(1, task_num + 1)]
    task_time_list = [[task, instance[task - 1][5]] for task in task_list]
    task_time_list = sorted(task_time_list, key=lambda x: x[1], reverse=False)

    path_init_task_map = {}
    for path_index in range(1, Config.M + 1):
        path_init_task_map[path_index] = 0
    sequence_map = {}
    for task_time in task_time_list:
        task = task_time[0]
        best_fitness = float('inf')
        best_sequence_map = None
        best_initial_map = None
        parent_all_feasible_position_set = get_all_position(sequence_map, path_init_task_map, task, 'parent')
        for parent_position in parent_all_feasible_position_set:
            sequence_map_temp = copy_dict_int_dict(sequence_map)
            path_init_task_map_temp = copy_dict_int_int(path_init_task_map)
            insert_(sequence_map_temp, path_init_task_map_temp, task, parent_position, 'parent')
            child_all_feasible_position_set = get_feasible_insert_position(sequence_map_temp, path_init_task_map_temp, task, 'child')
            for child_position in child_all_feasible_position_set:
                sequence_map_temp_temp = copy_dict_int_dict(sequence_map_temp)
                path_init_task_map_temp_temp = copy_dict_int_int(path_init_task_map_temp)
                insert_(sequence_map_temp_temp, path_init_task_map_temp_temp, task, child_position, 'child')
                fitness, _, _ = cal_fitness(instance, sequence_map_temp_temp, path_init_task_map_temp_temp)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_sequence_map = sequence_map_temp_temp
                    best_initial_map = path_init_task_map_temp_temp
        if best_sequence_map is None:
            raise RuntimeError('No feasible joint insertion during initialization')
        sequence_map = best_sequence_map
        path_init_task_map = best_initial_map
    return Solution(instance, sequence_map, path_init_task_map)







def generate_solution_nearest(instance, weight_construct=None):
    """
    因为母车多，子车少
    母车完成某个任务后，执行下一个任务
    step1: 找到最早空闲的母车
    step2: 找到距离最早空闲母车，时空距离最近的子车
    step3: 从该子车开始，搜索cost最小的任务
    step4: 指定该任务，为子母车下一步执行的任务
    :param instance:
    :return:
    """
    if weight_construct is None:
        weight_construct = 0.8
    task_num = len(instance)
    task_list = [task for task in range(1, task_num + 1)]
    parent_map = {}  # 存储 母车: [空闲时间, 位置]
    child_map = {}   # 存储 子车: [空闲时间, 位置]

    path_map = {}
    for path_index in range(1, Config.M + 1):
        path_map[path_index] = []

    for parent in Config.M_parent_list:
        parent_map[parent] = [0, Config.M_position_map[parent]]
    for child in Config.M_child_list:
        child_map[child] = [0, Config.M_position_map[child]]

    while len(task_list) != 0:
        min_parent_index = min(parent_map, key=lambda k: parent_map[k][0])
        parent_info = parent_map[min_parent_index]
        parent_idle_time = parent_info[0]
        parent_position = parent_info[1]
        # 找子车，要求，时间相近，距离相近
        # temporal distance and spatial distance
        min_child_index = None
        distance_min = 1e5
        for child in Config.M_child_list:
            child_info = child_map[child]
            child_idle_time = child_info[0]
            child_position = child_info[1]
            spatial_distance = get_distance(parent_position, child_position)
            parent_arrive_time = parent_idle_time + spatial_distance / Config.V
            temporal_distance = max(0, child_idle_time - parent_arrive_time) * Config.V
            # temporal_distance = abs(child_idle_time - parent_arrive_time)  # 不要让母车去等待
            """
            注：此处temporal distance，如果child_idle_time < couple_time，即母车到达子车前，子车已经idle
            则temporal distance为0
            因为parent_idle_time已经是所有parent中最早的idle time了，由于子母车需要耦合执行任务，所以该child
            最早开始执行下一个任务的时间就是母车的idle time，无论差值大小
            """
            distance = spatial_distance + temporal_distance
            if distance < distance_min:
                min_child_index = child
                distance_min = distance
        '''找到了和母车时空距离最近的子车'''
        child_min_info = child_map[min_child_index]
        child_min_idle_time = child_min_info[0]
        child_min_position = child_min_info[1]
        couple_time = max(child_min_idle_time, parent_idle_time + get_distance(parent_position, child_min_position) / Config.V)
        couple_position = child_min_position
        # 找到距离couple_position距离最近，最紧急的任务
        task_min = None
        cost_min = 1e5
        t_d_min = None
        t_e_min = None
        idle_position = None
        for task in task_list:
            task_info = instance[task - 1]
            source_position = [task_info[1], task_info[2]]
            destination_position = [task_info[3], task_info[4]]
            t_operate = task_info[6]
            t_ddl = task_info[5]
            couple2source_distance = get_distance(couple_position, source_position)
            source2destination_distance = get_distance(source_position, destination_position)
            t_d = couple_time + Config.T_couple + couple2source_distance / Config.V + Config.T_load + source2destination_distance / Config.V + Config.T_decouple
            t_e = t_d + t_operate
            distance_cost = couple2source_distance
            urgent_cost = t_ddl - parent_idle_time     # 采用截止时间和耦合时间的差值，作为urgent的metric
            # 差值越小，说明该任务越urgent，越需要提早被执行
            """
            这么设置cost，会使得tardiness_cost大的任务一直被延后
            使得该任务的tardiness_cost滚雪球增大
            距离越短，越urgent，越要当前去做
            """
            cost = distance_cost * weight_construct + urgent_cost * (1 - weight_construct)
            if cost < cost_min:
                task_min = task
                cost_min = cost
                t_d_min = t_d
                t_e_min = t_e
                idle_position = destination_position
        path_map[min_parent_index].append(task_min)
        path_map[min_child_index].append(task_min)
        parent_map[min_parent_index] = [t_d_min, idle_position]
        child_map[min_child_index] = [t_e_min, idle_position]
        task_list.remove(task_min)

    sequence_map, path_init_task_map = path_map2sequence_map(path_map)
    solution = Solution(instance, sequence_map, path_init_task_map)
    return solution





#
# if __name__ == "__main__":
#     instance = read_excel("T30_I7.xlsx")
#     start_time = time.perf_counter()
#     greedy_solution = generate_solution_greedy(instance)
#     end_time = time.perf_counter()
#     nearest_solution = generate_solution_nearest2(instance)
#     print(f"初始解构建时间: {(end_time-start_time)} greedy_solution fitness: {greedy_solution.get_fitness()} nearest_solution fitness: {nearest_solution.get_fitness()}")


