from baselines.heuristics.utils.Config import Config
from baselines.heuristics.utils.ALNS_config import ALNSConfig
import time
import logging

# Logging is configured by the caller. Importing a utility must not create
# a log file in whichever directory the user launched the command from.


# %% copy 方法
def copy_set_int(original_set):
    """
    对set进行深拷贝
    set中元素的类型为int
    :param original_set:
    :return:
    """
    copied_set = set()
    for item in original_set:
        copied_set.add(item)

    return copied_set


def copy_list_int(original_list):
    """
    对set进行深拷贝
    set中元素的类型为int
    :param original_list:
    :return:
    """
    copied_list = []
    for item in copied_list:
        copied_list.append(item)

    return copied_list


def copy_dict_int_int(original_dict):
    return original_dict.copy()


def copy_dict_int_list(original_dict):
    return {key: value.copy() for key, value in original_dict.items()}


def copy_dict_int_dict(original_dict):
    return {key: value.copy() for key, value in original_dict.items()}


# %%
# def cal_distance(source_axis, destination_axis):
#     source_x = source_axis[0]
#     source_y = source_axis[1]
#     destination_x = destination_axis[0]
#     destination_y = destination_axis[1]
#
#     if source_x == destination_x:
#         # 同一条竖线
#         distance = abs(destination_y - source_y)
#         return distance
#     else:
#         if (source_y > 50 > destination_y) or (source_y < 50 < destination_y):
#             distance = abs(destination_x - source_x) + abs(destination_y - source_y)
#             return distance
#         else:
#             # 先得到离哪一个路口最近
#             distance_1 = min(abs(source_y - 0) + abs(destination_y - 0),
#                              abs(source_y - 50) + abs(destination_y - 50),
#                              abs(source_y - 100) + abs(destination_y - 100))
#             distance_2 = abs(destination_x - source_x)
#             return distance_1 + distance_2



def path_map2sequence_map(path_map):
    sequence_map = {}
    path_init_task_map = {}
    for agv_index in path_map.keys():
        path = path_map[agv_index]
        if len(path) == 0:
            path_init_task_map[agv_index] = 0
            continue
        for task_index, task in enumerate(path):
            if task in sequence_map.keys():
                task_info = sequence_map[task]
            else:
                task_info = {}
            if agv_index in Config.M_parent_list:
                task_info['parent'] = agv_index
                if task_index == 0:
                    task_info['parent_pre_task'] = 0
                    path_init_task_map[agv_index] = task
                else:
                    pre_task = path[task_index - 1]
                    task_info['parent_pre_task'] = pre_task
                if task_index == len(path) - 1:
                    # 最后一个任务
                    task_info['parent_next_task'] = 0
                else:
                    task_info['parent_next_task'] = path[task_index + 1]
            else:
                task_info['child'] = agv_index
                if task_index == 0:
                    task_info['child_pre_task'] = 0
                    path_init_task_map[agv_index] = task
                else:
                    pre_task = path[task_index - 1]
                    task_info['child_pre_task'] = pre_task
                if task_index == len(path) - 1:
                    # 最后一个任务
                    task_info['child_next_task'] = 0
                else:
                    task_info['child_next_task'] = path[task_index + 1]
            sequence_map[task] = task_info
    return sequence_map, path_init_task_map


# cal_fitness_time_list = []
def _cal_fitness_python(instance, sequence_map, path_init_task_map):
    # start_time = time.perf_counter()
    total_distance = 0  # 总的行驶距离
    total_tardiness = 0  # 总的迟到时间

    info_map = {task: dict() for task in sequence_map.keys()}
    task_num = len(sequence_map)

    task_parent_closed_set = set()
    task_child_closed_set = set()

    for path_index, first_task in path_init_task_map.items():
        if first_task != 0:
            if path_index in Config.M_parent_list:
                info_map[first_task]['parent_pre_d_time'] = 0
                task_parent_closed_set.add(first_task)
            else:
                info_map[first_task]['child_pre_e_time'] = 0
                task_child_closed_set.add(first_task)

    # task_closed = []
    task_closed_num = 0
    while task_closed_num < task_num:
        task_to_calculate = None
        for task in task_parent_closed_set:
            if task in task_child_closed_set:
                task_to_calculate = task
                break
        if task_to_calculate is None:
            print("无使能变迁，解非法")
            return 1e6
        task_closed_num += 1
        parent_pre_d_time = info_map[task_to_calculate]['parent_pre_d_time']
        child_pre_e_time = info_map[task_to_calculate]['child_pre_e_time']

        task_data = sequence_map[task_to_calculate]

        if task_data['parent_pre_task'] == 0:
            parent_pre_position = Config.M_position_map[task_data['parent']]
        else:
            parent_pre_position = [instance[task_data['parent_pre_task'] - 1][3],
                                   instance[task_data['parent_pre_task'] - 1][4]]
        if task_data['child_pre_task'] == 0:
            child_pre_position = Config.M_position_map[task_data['child']]
        else:
            child_pre_position = [instance[task_data['child_pre_task'] - 1][3],
                                  instance[task_data['child_pre_task'] - 1][4]]
        instance_task_data = instance[task_to_calculate - 1]
        source_position = [instance_task_data[1], instance_task_data[2]]
        destination_position = [instance_task_data[3], instance_task_data[4]]


        t_i_plus2i_minus = get_distance(parent_pre_position, child_pre_position) / Config.V  # 接子车距离
        t_i_minus2j_s = get_distance(child_pre_position, source_position) / Config.V  # 取货距离
        t_j_s2j_d = get_distance(source_position, destination_position) / Config.V  #

        t_idle_parent = max(0, child_pre_e_time - parent_pre_d_time - t_i_plus2i_minus)

        t_d = parent_pre_d_time + t_idle_parent + t_i_plus2i_minus + Config.T_couple + t_i_minus2j_s + Config.T_load + t_j_s2j_d + Config.T_decouple
        t_e = t_d + instance_task_data[6]

        parent_next_task = task_data['parent_next_task']
        child_next_task = task_data['child_next_task']

        task_distance = (t_i_plus2i_minus + t_i_minus2j_s + t_j_s2j_d) * Config.V
        task_tardiness = max(0, t_e - instance_task_data[5])

        # distance = get_distance(parent_pre_position, source_position) + get_distance(child_pre_position, source_position) + 2 * get_distance(source_position, destination_position)

        total_distance += task_distance
        total_tardiness += task_tardiness

        if parent_next_task != 0:
            info_map[parent_next_task]['parent_pre_d_time'] = t_d
            task_parent_closed_set.add(parent_next_task)
        if child_next_task != 0:
            info_map[child_next_task]['child_pre_e_time'] = t_e
            task_child_closed_set.add(child_next_task)

        task_parent_closed_set.remove(task_to_calculate)
        task_child_closed_set.remove(task_to_calculate)
    fitness = total_distance * Config.weight + total_tardiness * (1 - Config.weight)
    # end_time = time.perf_counter()
    # cal_fitness_time_list.append(end_time-start_time)
    # if len(cal_fitness_time_list) > 1000:
    #     print(sum(cal_fitness_time_list)/len(cal_fitness_time_list))
    return fitness, total_distance, total_tardiness

def _cal_fitness_feasible_python(instance, sequence_map, path_init_task_map):
    # start_time = time.perf_counter()
    total_distance = 0  # 总的行驶距离
    total_tardiness = 0  # 总的迟到时间

    info_map = {task: dict() for task in sequence_map.keys()}
    task_num = len(sequence_map)

    task_parent_closed_set = set()
    task_child_closed_set = set()

    for path_index, first_task in path_init_task_map.items():
        if first_task != 0:
            if path_index in Config.M_parent_list:
                info_map[first_task]['parent_pre_d_time'] = 0
                task_parent_closed_set.add(first_task)
            else:
                info_map[first_task]['child_pre_e_time'] = 0
                task_child_closed_set.add(first_task)

    # task_closed = []
    task_closed_num = 0
    while task_closed_num < task_num:
        task_to_calculate = None
        for task in task_parent_closed_set:
            if task in task_child_closed_set:
                task_to_calculate = task
                break
        if task_to_calculate is None:
            # print("无使能变迁，解非法")
            return 1e6, False
        task_closed_num += 1
        parent_pre_d_time = info_map[task_to_calculate]['parent_pre_d_time']
        child_pre_e_time = info_map[task_to_calculate]['child_pre_e_time']

        task_data = sequence_map[task_to_calculate]

        if task_data['parent_pre_task'] == 0:
            parent_pre_position = Config.M_position_map[task_data['parent']]
        else:
            parent_pre_position = [instance[task_data['parent_pre_task'] - 1][3],
                                   instance[task_data['parent_pre_task'] - 1][4]]
        if task_data['child_pre_task'] == 0:
            child_pre_position = Config.M_position_map[task_data['child']]
        else:
            child_pre_position = [instance[task_data['child_pre_task'] - 1][3],
                                  instance[task_data['child_pre_task'] - 1][4]]
        instance_task_data = instance[task_to_calculate - 1]
        source_position = [instance_task_data[1], instance_task_data[2]]
        destination_position = [instance_task_data[3], instance_task_data[4]]


        t_i_plus2i_minus = get_distance(parent_pre_position, child_pre_position) / Config.V  # 接子车距离
        t_i_minus2j_s = get_distance(child_pre_position, source_position) / Config.V  # 取货距离
        t_j_s2j_d = get_distance(source_position, destination_position) / Config.V  #

        t_idle_parent = max(0, child_pre_e_time - parent_pre_d_time - t_i_plus2i_minus)

        t_d = parent_pre_d_time + t_idle_parent + t_i_plus2i_minus + Config.T_couple + t_i_minus2j_s + Config.T_load + t_j_s2j_d + Config.T_decouple
        t_e = t_d + instance_task_data[6]

        parent_next_task = task_data['parent_next_task']
        child_next_task = task_data['child_next_task']

        task_distance = (t_i_plus2i_minus + t_i_minus2j_s + t_j_s2j_d) * Config.V
        task_tardiness = max(0, t_e - instance_task_data[5])

        # distance = get_distance(parent_pre_position, source_position) + get_distance(child_pre_position, source_position) + 2 * get_distance(source_position, destination_position)

        total_distance += task_distance
        total_tardiness += task_tardiness

        if parent_next_task != 0:
            info_map[parent_next_task]['parent_pre_d_time'] = t_d
            task_parent_closed_set.add(parent_next_task)
        if child_next_task != 0:
            info_map[child_next_task]['child_pre_e_time'] = t_e
            task_child_closed_set.add(child_next_task)

        task_parent_closed_set.remove(task_to_calculate)
        task_child_closed_set.remove(task_to_calculate)
    fitness = total_distance * Config.weight + total_tardiness * (1 - Config.weight)
    # end_time = time.perf_counter()
    # cal_fitness_time_list.append(end_time-start_time)
    # if len(cal_fitness_time_list) > 1000:
    #     print(sum(cal_fitness_time_list)/len(cal_fitness_time_list))
    return fitness, True

def cal_fitness(instance, sequence_map, path_init_task_map):
    from baselines.heuristics.utils.native_fitness import evaluate
    result = evaluate(instance, sequence_map)
    if result is not None:
        return result[:3]
    value = _cal_fitness_python(instance, sequence_map, path_init_task_map)
    return value if isinstance(value, tuple) else (value, 0.0, 0.0)


def cal_fitness_feasible(instance, sequence_map, path_init_task_map):
    from baselines.heuristics.utils.native_fitness import evaluate
    result = evaluate(instance, sequence_map)
    if result is not None:
        return result[0], result[3]
    return _cal_fitness_feasible_python(instance, sequence_map, path_init_task_map)


def get_all_position(destroyed_sequence_map, path_init_task_map, destroyed_task, chain):
    """
    获取所有的位置
    :param destroyed_sequence_map:
    :param path_init_task_map:
    :param chain:
    :return:
    所有任务前后位置，所有空path
    执行去重操作
    """

    '''
    destroyed_sequence_map[task][chain] is None，表示task还未插入到对应的chain中
    '''
    feasible_position_set = {(task, 0) for task in destroyed_sequence_map.keys() if
                             task != destroyed_task and destroyed_sequence_map[task][chain] is not None}  # 0表示该task之前
    feasible_position_set = feasible_position_set.union({(task, 1) for task in destroyed_sequence_map.keys() if
                                                         task != destroyed_task and destroyed_sequence_map[task][
                                                             chain] is not None})  # 1表示该task之后
    '''
        以下执行去重操作
        [1,2,3]
        (2,1) 和 (3, 0)位置等价
        此处默认去除(2, 1)
    '''
    to_delete_list = []
    for position in feasible_position_set:
        task = position[0]
        direction = position[1]
        if direction == 0:
            pre_task = destroyed_sequence_map[task][chain + '_pre_task']
            if pre_task != 0 and pre_task is not None:
                if (pre_task, 1) in feasible_position_set:
                    to_delete_list.append((pre_task, 1))
        else:
            next_task = destroyed_sequence_map[task][chain + '_next_task']
            if next_task != 0 and next_task is not None:
                if (next_task, 0) in feasible_position_set:
                    to_delete_list.append((task, 1))
    for position in to_delete_list:
        feasible_position_set.discard(position)

    # 加入空的path
    for path_index, first_task in path_init_task_map.items():
        if chain == 'child' and path_index in Config.M_child_list and first_task == 0:
            feasible_position_set.add((0, path_index))
        if chain == 'parent' and path_index in Config.M_parent_list and first_task == 0:
            feasible_position_set.add((0, path_index))
    return feasible_position_set


def get_feasible_insert_position(destroyed_sequence_map, path_init_task_map, destroyed_task, chain):
    """
    采用前后推导的方法，去除所有不可行的位置，得到可行位置
    注：空的path也可以插入
    注：调用该函数则已经插入了一半的链
    :param destroyed_sequence_map: 被破坏的sequence_map，已经在子链或者母链中插入了任务
    :param destroyed_task: 待插入的任务
    :param path_init_task_map: 记录每条路径的第一个任务，方便计算
    :param chain: 'parent'/'child' 在子链/母链中寻找  如果是parent，则destroyed_task已经被插入了child链
    :return: feasible_position_set
    example:
    (2, 0) is in feasible_position_set -> task 2's pre position is feasible
    (5, 1) is in feasible_position_set -> task 5's pre position is feasible
    (0, 3) is in feasible_position_set -> parent/child path 3 is an empty path
    """
    # start_time = time.perf_counter()
    # 得到所有的位置，再删去不合法的位置
    feasible_position_set = get_all_position(destroyed_sequence_map, path_init_task_map, destroyed_task, chain)
    # all_position_num = len(feasible_position_set)

    pre_to_explore_set = {destroyed_task}
    pre_explored_list = []
    while len(pre_to_explore_set) != 0:
        temp_set = copy_set_int(pre_to_explore_set)
        for task in temp_set:
            # 以下子车母车的两个前任务，在母/子链上，均需要在task之前
            child_pre_task = destroyed_sequence_map[task]['child_pre_task']
            parent_pre_task = destroyed_sequence_map[task]['parent_pre_task']
            if child_pre_task != 0 and child_pre_task is not None:
                feasible_position_set.discard((child_pre_task, 0))
                # couple_task x_{ij}^r j前和i后是同一个位置
                # 注：couple_task只需要考虑对应链上的
                if chain == 'child':
                    couple_task = destroyed_sequence_map[child_pre_task]['child_pre_task']
                else:
                    couple_task = destroyed_sequence_map[child_pre_task]['parent_pre_task']
                if couple_task != 0 and couple_task is not None:
                    feasible_position_set.discard((couple_task, 1))
                if child_pre_task not in pre_explored_list:
                    pre_to_explore_set.add(child_pre_task)
            if parent_pre_task != 0 and parent_pre_task is not None:
                feasible_position_set.discard((parent_pre_task, 0))
                if chain == 'child':
                    couple_task = destroyed_sequence_map[parent_pre_task]['child_pre_task']
                else:
                    couple_task = destroyed_sequence_map[parent_pre_task]['parent_pre_task']
                if couple_task != 0 and couple_task is not None:
                    feasible_position_set.discard((couple_task, 1))
                if parent_pre_task not in pre_explored_list:
                    pre_to_explore_set.add(parent_pre_task)
            pre_to_explore_set.remove(task)
            pre_explored_list.append(task)
    # print(f"pre_explored_list:{pre_explored_list}")

    next_to_explore_set = {destroyed_task}
    next_explored_list = []
    while len(next_to_explore_set) != 0:
        temp_set = copy_set_int(next_to_explore_set)
        for task in temp_set:
            child_next_task = destroyed_sequence_map[task]['child_next_task']
            parent_next_task = destroyed_sequence_map[task]['parent_next_task']
            if child_next_task != 0 and child_next_task is not None:
                feasible_position_set.discard((child_next_task, 1))
                if chain == 'child':
                    couple_task = destroyed_sequence_map[child_next_task]['child_next_task']
                else:
                    couple_task = destroyed_sequence_map[child_next_task]['parent_next_task']
                if couple_task != 0 and couple_task is not None:
                    feasible_position_set.discard((couple_task, 0))
                if child_next_task not in next_explored_list:
                    next_to_explore_set.add(child_next_task)
            if parent_next_task != 0 and parent_next_task is not None:
                feasible_position_set.discard((parent_next_task, 1))
                if chain == 'child':
                    couple_task = destroyed_sequence_map[parent_next_task]['child_next_task']
                else:
                    couple_task = destroyed_sequence_map[parent_next_task]['parent_next_task']
                if couple_task != 0 and couple_task is not None:
                    feasible_position_set.discard((couple_task, 0))
                if parent_next_task not in next_explored_list:
                    next_to_explore_set.add(parent_next_task)
            next_to_explore_set.remove(task)
            next_explored_list.append(task)

    return feasible_position_set



def insert_(destroyed_sequence_map, path_init_task_map, destroyed_task, insert_position, chain):
    '''
    将destroyed_task插入chain的insert_position位置
    注：这里的插入有可能插入一半，也有可能完全没有插入
    :param destroyed_sequence_map:
    :param path_init_task_map:
    :param destroyed_task:
    :param insert_position:
    :param chain:
    :return:
    '''
    if chain == 'parent':
        chain_opposite = 'child'
    else:
        chain_opposite = 'parent'
    if insert_position[0] != 0:
        insert_task = insert_position[0]
        direction = ['pre', 'next'][insert_position[1]]
        direction_opposite = ['pre', 'next'][insert_position[1] - 1]
        path_index = destroyed_sequence_map[insert_task][chain]
        direction_task = destroyed_sequence_map[insert_task][chain + '_' + direction + '_task']
        destroyed_sequence_map[insert_task][chain + '_' + direction + '_task'] = destroyed_task
        if direction_task != 0:
            destroyed_sequence_map[direction_task][
                chain + '_' + direction_opposite + '_task'] = destroyed_task
        else:
            if direction == 'pre':
                path_init_task_map[path_index] = destroyed_task
        if destroyed_task not in destroyed_sequence_map.keys():
            destroyed_sequence_map[destroyed_task] = {}
            destroyed_sequence_map[destroyed_task][chain] = path_index
            destroyed_sequence_map[destroyed_task][chain_opposite] = None
            destroyed_sequence_map[destroyed_task][chain + '_' + direction + '_task'] = direction_task
            destroyed_sequence_map[destroyed_task][chain_opposite + '_' + direction + '_task'] = None
            destroyed_sequence_map[destroyed_task][chain + '_' + direction_opposite + '_task'] = insert_task
            destroyed_sequence_map[destroyed_task][chain_opposite + '_' + direction_opposite + '_task'] = None
        else:
            destroyed_sequence_map[destroyed_task][chain] = path_index
            destroyed_sequence_map[destroyed_task][chain + '_' + direction + '_task'] = direction_task
            destroyed_sequence_map[destroyed_task][chain + '_' + direction_opposite + '_task'] = insert_task
    else:
        insert_path_index = insert_position[1]
        path_init_task_map[insert_path_index] = destroyed_task
        if destroyed_task not in destroyed_sequence_map.keys():
            destroyed_sequence_map[destroyed_task] = {}
            destroyed_sequence_map[destroyed_task][chain] = insert_path_index
            destroyed_sequence_map[destroyed_task][chain_opposite] = None
            destroyed_sequence_map[destroyed_task][chain + '_pre_task'] = 0
            destroyed_sequence_map[destroyed_task][chain_opposite + '_pre_task'] = None
            destroyed_sequence_map[destroyed_task][chain + '_next_task'] = 0
            destroyed_sequence_map[destroyed_task][chain_opposite + '_next_task'] = None
        else:
            destroyed_sequence_map[destroyed_task][chain] = insert_path_index
            destroyed_sequence_map[destroyed_task][chain + '_pre_task'] = 0
            destroyed_sequence_map[destroyed_task][chain + '_next_task'] = 0
    return destroyed_sequence_map, path_init_task_map


def remove_(sequence_map, path_init_task_map, task):
    """
    从sequence_map中，移除task（子链母链均移除）
    :param sequence_map:
    :param task:
    :return:
    """
    parent_pre_task = sequence_map[task]['parent_pre_task']
    child_pre_task = sequence_map[task]['child_pre_task']
    parent_next_task = sequence_map[task]['parent_next_task']
    child_next_task = sequence_map[task]['child_next_task']

    if parent_pre_task == 0:
        parent = sequence_map[task]['parent']
        path_init_task_map[parent] = parent_next_task
    else:
        sequence_map[parent_pre_task]['parent_next_task'] = parent_next_task

    if parent_next_task != 0:
        sequence_map[parent_next_task]['parent_pre_task'] = parent_pre_task

    if child_pre_task == 0:
        child = sequence_map[task]['child']
        path_init_task_map[child] = child_next_task
    else:
        sequence_map[child_pre_task]['child_next_task'] = child_next_task

    if child_next_task != 0:
        sequence_map[child_next_task]['child_pre_task'] = child_pre_task

    del sequence_map[task]

    return sequence_map, path_init_task_map


def get_path_map(sequence_map, path_init_task_map):
    path_map = {}
    for path_index, init_task in path_init_task_map.items():
        if init_task == 0:
            # 该path为空
            path_map[path_index] = []
            continue
        if path_index in Config.M_parent_list:
            path = [init_task]
            parent_next_task = sequence_map[init_task]['parent_next_task']
            while parent_next_task != 0:
                path.append(parent_next_task)
                parent_next_task = sequence_map[parent_next_task]['parent_next_task']
            path_map[path_index] = path
        else:
            path = [init_task]
            parent_next_task = sequence_map[init_task]['child_next_task']
            while parent_next_task != 0:
                path.append(parent_next_task)
                parent_next_task = sequence_map[parent_next_task]['child_next_task']
            path_map[path_index] = path

    return path_map


def get_T(instance):
    sum_dis = 0
    for info in instance:
        source_position = [info[1], info[2]]
        destination_position = [info[3], info[4]]
        sum_dis += get_distance(source_position, destination_position) / Config.V
    T = sum_dis / (len(instance) * (Config.M_parent + Config.M_child)) * ALNSConfig.T_coefficient
    return T


def get_distance(source_position, destination_position):
    distance = abs(source_position[0] - destination_position[0]) + abs(source_position[1] - destination_position[1])
    return distance
    # source_x = source_position[0]
    # source_y = source_position[1]
    # destination_x = destination_position[0]
    # destination_y = destination_position[1]
    #
    # if source_x == destination_x:
    #     # 同一条竖线
    #     distance = abs(destination_y - source_y)
    #     return distance
    # else:
    #     if (source_y > 50 > destination_y) or (source_y < 50 < destination_y):
    #         distance = abs(destination_x - source_x) + abs(destination_y - source_y)
    #         return distance
    #     else:
    #         # 先得到离哪一个路口最近
    #         distance_1 = min(abs(source_y - 0) + abs(destination_y - 0),
    #                          abs(source_y - 50) + abs(destination_y - 50),
    #                          abs(source_y - 100) + abs(destination_y - 100))
    #         distance_2 = abs(destination_x - source_x)
    #         return distance_1 + distance_2


def code2path_map(code):
    parent_code = code[0]
    child_code = code[1]
    path_map = {}
    parent_index = 1
    path = []
    for task_index, task in enumerate(parent_code):
        if task_index == 0:
            continue
        if task == 0:
            path_map[parent_index] = path
            path = []
            parent_index += 1
            continue
        path.append(task)
        if task_index == len(parent_code) - 1:
            path_map[parent_index] = path
    if parent_code[-1] == 0:
        path_map[parent_index] = []

    child_index = Config.M_parent + 1
    path = []
    for task_index, task in enumerate(child_code):
        if task_index == 0:
            continue
        if task == 0:
            path_map[child_index] = path
            path = []
            child_index += 1
            continue
        path.append(task)
        if task_index == len(child_code) - 1:
            path_map[child_index] = path
    if child_code[-1] == 0:
        path_map[child_index] = []
    return path_map

