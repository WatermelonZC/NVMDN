import torch

from baselines.heuristics.utils.Config import Config
# from util import *
# from Solution import Solution

from baselines.heuristics.utils.Solution import Solution
from baselines.heuristics.utils.load_data import *
from baselines.heuristics.utils.util import *
def parse_vehicle_routes(nested_list):
    """
    正确解析包含多个子车任务序列的嵌套列表
    规则：
    - 每个0代表一个子车的分界点
    - 连续的0表示空任务序列的子车
    - 0之间的数字是该子车的任务顺序
    """
    all_routes = []
    graph_size = Config.Graph_size
    temp_list = [nested_list[1], nested_list[0]]
    # result = torch.full((1, max_subvehicles, max_seq_len), -1, dtype=torch.long)
    for route in temp_list:
        last_route = None
        for num in route:
            if num == 0:
                last_route = []
                all_routes.append(last_route)
            else:
                last_route.append(num)
                # last_route.append(num+graph_size)
    vehicle_size = len(all_routes)
    # task_num = max(max(sublist) for sublist in nested_list)
    result = torch.full((1, vehicle_size, graph_size*2), -1, dtype=torch.long)
    for i, route in enumerate(all_routes):
        for j, seq in enumerate(route):
            result[0, i, j] = seq - 1
    return result



# # 测试数据
# input_data = [[0, 1,  6, 10,  4,  8,  5, 20, 15, 18, 12,0, 13, 7, 11,  3, 16, 17,  9,  2, 14, 19], [0,13,7, 11,  3, 16, 17,  9,  2, 14, 19,0,  1,  6, 10,  4,  8,  5, 20, 15, 18, 12,0,0]]
#
# path_map = code2path_map(input_data)
# sequence_map, path_init_task_map = path_map2sequence_map(path_map)
# solution = Solution(instance=read_excel("tasks_20.xlsx"), sequence_map=sequence_map,
#                     path_init_task_map=path_init_task_map)
# fitness = solution.get_fitness()
# fit = cal_fitness(read_excel("tasks_20.xlsx"),sequence_map, path_init_task_map)
# print(fit[0])
pass
# 测试填充版本
# padded_result = parse_vehicle_routes(input_data, 20)
# print("\n填充对齐后的Tensor:")
# print(padded_result)
