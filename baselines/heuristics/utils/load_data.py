import pandas as pd
from pathlib import Path
import numpy as np

from baselines.heuristics.utils.Config import Config


def read_excel(file_name):
    df = pd.read_excel(file_name)
    instance = [list(row) for index, row in df.iterrows()]
    size = len(instance)
    for i in range(size):
        instance[i][-2] = instance[i][-2]
    # 读取车辆数据
    # vehicle_df = pd.read_excel(file_name, sheet_name='Vehicles')
    # Config.T_couple = vehicle_df[vehicle_df['车辆类型'] == '母车']['tau_a'].to_list()[0]
    # Config.T_decouple = vehicle_df[vehicle_df['车辆类型'] == '母车']['tau_d'].to_list()[0]
    # Config.T_load = vehicle_df[vehicle_df['车辆类型'] == '母车']['tau_p'].to_list()[0]
    # Config.V = vehicle_df[vehicle_df['车辆类型'] == '母车']['v'].to_list()[0]
    Config.Graph_size = size
    return instance



def toTensor(data):
    coords = np.array([item[1:5] for item in data])  # shape: (20, 4)

    # 计算每列的最小值和最大值
    min_vals = coords.min(axis=0)  # [x_min, y_min, x2_min, y2_min]
    max_vals = coords.max(axis=0)  # [x_max, y_max, x2_max, y2_max]

    # 归一化函数
    def normalize(coords, min_vals, max_vals):
        return (coords - min_vals) / (max_vals - min_vals)

    # 应用归一化
    norm_coords = normalize(coords, min_vals, max_vals)

    # 将归一化后的坐标替换回原数据
    result = data.copy()
    for i in range(len(result)):
        result[i][1:5] = norm_coords[i]

    print(result)
