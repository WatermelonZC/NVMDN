import os
import pickle
from collections import deque

import numpy as np
from openpyxl.reader.excel import load_workbook
from openpyxl.styles import Font


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):

    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)

def normalize(tensor):
    time_min = tensor.min(dim=1, keepdim=True)[0]  # 每个 sample 的最小时间
    time_max = tensor.max(dim=1, keepdim=True)[0]
    time_norm = (tensor - time_min) / (time_max - time_min + 1e-8)
    return time_norm

import torch
import pandas as pd

# 测试tensor写入Excel文件
def write_tensor_to_excel(data, output_file="tasks_output.xlsx"):
    """
    将批处理后的单个样本数据（batch_size=1）写入到Excel文件的两个工作表中。
    此版本已更新以处理带有批处理维度的数据结构。

    Args:
        data (dict): 包含批处理后张量（batch_size=1）的单个字典。
        output_file (str): 输出的Excel文件名。
    """
    if not data or not isinstance(data, dict):
        print("输入数据必须是一个非空字典。")
        return

    # --- 1. 准备 'Tasks' 工作表 ---

    # 修改点: 使用 [0] 来移除批处理维度 (e.g., from [1, size, 2] to [size, 2])
    sor_loc = data['sor_loc'][0].numpy()
    tar_loc = data['tar_loc'][0].numpy()
    t_delivery = data['designated_time'][0].numpy()
    t_operation = data['tau_h'][0].numpy()
    num_tasks = sor_loc.shape[0]

    # 创建任务DataFrame (这部分逻辑不变)
    task_df = pd.DataFrame({
        'task_index': range(1, num_tasks + 1),
        'source_x': sor_loc[:, 0],
        'source_y': sor_loc[:, 1],
        'destination_x': tar_loc[:, 0],
        'destination_y': tar_loc[:, 1],
        't_delivery': t_delivery[:, 0],
        't_operation': t_operation[:, 0]
    })

    # --- 2. 准备 'Vehicles' 工作表 ---

    # 修改点: 使用 [0].item() 从 tensor([value]) 中提取纯数值
    mom_size = data['mom_size'][0].item()
    sub_size = data['sub_size'][0].item()

    # 修改点: 同样使用 [0] 移除批处理维度
    params = data['params'][0].numpy()

    # 修改点: v 的形状是 [1, 1], 需要两次索引或使用 .item()
    v = data['v'][0, 0].item()

    # 创建车辆DataFrame (这部分逻辑不变)
    vehicle_data = {
        '车辆类型': ['母车', '子车'],
        '车辆个数': [mom_size, sub_size],
        'tau_a': [params[0], None],
        'tau_d': [params[1], None],
        'tau_p': [params[2], None],
        'v': [v, None]
    }
    vehicle_df = pd.DataFrame(vehicle_data)

    # --- 3. 写入Excel文件 ---
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            task_df.to_excel(writer, sheet_name='Tasks', index=False)
            vehicle_df.to_excel(writer, sheet_name='Vehicles', index=False)
        print(f"数据已成功写入 {output_file}")
    except Exception as e:
        print(f"写入Excel文件时发生错误: {e}")


# 读取Excel文件到测试tensor
def read_excel_to_tensor(input_file="tasks.xlsx", designated_driver=False):
    # 读取任务数据
    task_df = pd.read_excel(input_file, sheet_name='Tasks', engine='openpyxl')

    # 重建sor_loc和tar_loc
    sor_loc = torch.tensor(
        task_df[['source_x', 'source_y']].values,
        dtype=torch.float32
    ).unsqueeze(0)  # (1, N, 2)

    tar_loc = torch.tensor(
        task_df[['destination_x', 'destination_y']].values,
        dtype=torch.float32
    ).unsqueeze(0)  # (1, N, 2)

    if designated_driver:
        designated_time = torch.tensor(
            task_df[['t_delivery']].values,
            dtype=torch.float32
        ).unsqueeze(0)  # (1, N, 2)
    else:
        designated_time = torch.tensor(
            task_df[['t_delivery']].values,
            dtype=torch.float32
        )[0, :].unsqueeze(0)

    # 读取车辆数据
    vehicle_df = pd.read_excel(input_file, sheet_name='Vehicles')

    # 分离母车和子车索引
    mom_size = vehicle_df[vehicle_df['车辆类型'] == '母车']['车辆个数'].to_list()[0]
    sub_size = vehicle_df[vehicle_df['车辆类型'] == '子车']['车辆个数'].to_list()[0]

    tau_a = vehicle_df[vehicle_df['车辆类型'] == '母车']['tau_a'].to_list()[0]
    tau_d = vehicle_df[vehicle_df['车辆类型'] == '母车']['tau_d'].to_list()[0]
    tau_p = vehicle_df[vehicle_df['车辆类型'] == '母车']['tau_p'].to_list()[0]
    tau_h = vehicle_df[vehicle_df['车辆类型'] == '母车']['tau_h'].to_list()[0]
    v = vehicle_df[vehicle_df['车辆类型'] == '母车']['v'].to_list()[0]
    params = torch.tensor([[tau_a, tau_d, tau_p, tau_h, v]])

    # 重建原始Tensor
    original_data = {
        'sor_loc': sor_loc,
        'tar_loc': tar_loc,
        'mom_size': mom_size,
        'sub_size': sub_size,
        'params': params,
        'designated_time': designated_time
    }

    return original_data


# 生成nvsp的数据
def generate_nvsp_data(num_samples=1, size=20, mom_size=1, sub_size=1, designated_driver=False, variable_instance=False):
    sor_loc = torch.cat([
        torch.rand(num_samples, size, 2),  # x, y 坐标仍然是连续的
        # torch.randint(0, 11, (num_samples, size, 1)) * 0.1  # z 轴取 {0.0, 0.1, ..., 1.0}
    ], dim=-1)  # (num_samples, size, 3)
    tar_loc = torch.cat([
        torch.rand(num_samples, size, 2),  # x, y 坐标仍然是连续的
        # torch.randint(0, 11, (num_samples, size, 1)) * 0.1  # z 轴取 {0.0, 0.1, ..., 1.0}
    ], dim=-1)  # (num_samples, size, 3)

    # # 生成tau_a, tau_d, tau_p, tau_h, v
    # tau_params = torch.full((num_samples, 3), 0.5, dtype=torch.float)
    # v = torch.full((num_samples, 1), 0.5, dtype=torch.float)
    # tau_h = torch.full((num_samples, 1), 2, dtype=torch.float)
    # tau_params = torch.cat((tau_params, tau_h), dim=1)
    # param_tensor = torch.cat((tau_params, v), dim=1)  # shape: [N, 5]

    variance_range = 0.1
    tau_adp = torch.full((num_samples, 3), 0.5)
    tau_h = torch.full((num_samples, 1), 2.0)
    v = torch.full((num_samples, 1), 0.5)
    if variable_instance:
        tau_adp = tau_adp + (torch.rand(num_samples, 3) * 2 * variance_range - variance_range)
        tau_h = tau_h + (torch.rand(num_samples, 1) * 2 * variance_range - variance_range)
        v = v + (torch.rand(num_samples, 1) * variance_range - variance_range)
    param_tensor = torch.cat((tau_adp, tau_h, v), dim=1)

    if not designated_driver:
        designated_time = torch.full((num_samples, 1), 1, dtype=torch.float)
    else:
        designated_time = torch.rand(num_samples, size, 1) * 1.2 + 0.2
    data = [
        {'sor_loc': sor_loc[i], 'tar_loc': tar_loc[i], 'designated_time': designated_time[i], 'sub_size': sub_size, 'mom_size': mom_size, 'params': param_tensor[i]}
        for i in range(num_samples)
    ]
    return data


# --- 全局配置：用于归一化的各项物理最大值 ---
MAX_VALUES = {
    'location': 100.0,
    'time': 2500.0,
    'tau_p': 60.0,  # params[2] (p=30) 的最大值预估
    'tau_ad': 15.0,  # params[0]和params[1] (a=8, d=8) 的最大值预估
    'velocity': 5.0,
    'tau_h': 100.0
}

normalization_tensor = torch.tensor([
    MAX_VALUES['location'], MAX_VALUES['location'],  # 对应 sor_loc
    MAX_VALUES['location'], MAX_VALUES['location'],  # 对应 tar_loc
    MAX_VALUES['time'],                              # 对应 designated_time
    MAX_VALUES['tau_h']                              # 对应 tau_h
])

_nvsp_feature_scale_cache = {}


def build_nvsp_request_features(input_data):
    """Build the normalized seven-dimensional NVSP request features.

    The first five columns preserve the oldest checkpoint layout. Later
    five- or six-dimensional projections are zero-padded to seven dimensions,
    preserving their output within floating-point tolerance.
    """
    designated_time = input_data['designated_time']
    batch_size, graph_size, _ = designated_time.size()
    tau_p = input_data['params'][:, 2].reshape(batch_size, 1, 1).expand(
        batch_size, graph_size, 1
    )
    features = torch.cat((
        input_data['sor_loc'],
        input_data['tar_loc'],
        designated_time,
        input_data['tau_h'],
        tau_p,
    ), dim=-1)
    scale_values = (
        MAX_VALUES['location'], MAX_VALUES['location'],
        MAX_VALUES['location'], MAX_VALUES['location'],
        MAX_VALUES['time'], MAX_VALUES['tau_h'], MAX_VALUES['tau_p'],
    )
    key = (features.device, features.dtype, scale_values)
    if key not in _nvsp_feature_scale_cache:
        _nvsp_feature_scale_cache[key] = features.new_tensor(scale_values)
    feature_scales = _nvsp_feature_scale_cache[key]
    return features / feature_scales


def collate_to_batch(list_of_dicts):
    """
    将一个字典列表（每个字典是一个样本）转换为一个批处理字典。
    """
    # 如果列表为空，返回空字典
    if not list_of_dicts:
        return {}

    # 获取第一个字典的所有键，作为批处理字典的键
    keys = list_of_dicts[0].keys()
    batched_dict = {}

    # 遍历所有键
    for key in keys:
        # 提取当前键在所有字典中的值，形成一个值的列表
        list_of_values = [d[key] for d in list_of_dicts]

        # 检查值是否为张量，如果不是（例如 'sub_size'），则先转换为张量
        if not isinstance(list_of_values[0], torch.Tensor):
            list_of_values = [torch.tensor(item) for item in list_of_values]

        # 使用 torch.stack 将值列表沿着新的维度（dim=0）堆叠起来
        # 这会创建批处理维度
        batched_dict[key] = torch.stack(list_of_values, dim=0)

    return batched_dict


def generate_nvsp_data_robustness(num_samples=1, size=20, mom_size=1, sub_size=1,
                      distribution_type='uniform', real_data_path=None, normalize=False, isEval=False,
                      return_batched=False):
    """
    生成NVSP数据。此版本确保 tau_adp ([8,8,30]) 对所有数据源都是固定的。
    """
    global normalization_tensor

    # --- 嵌套函数，仅用于程序化生成点 ---
    def _create_points_programmatically(dist_type):
        """
        Generates 2D points based on a specified distribution type,
        ensuring they are within a [0, 1] unit square before scaling.
        """
        num_total_points = num_samples * size
        points_unit_scale = None

        if dist_type == 'uniform':
            points_unit_scale = torch.rand(num_samples, size, 2)
            # Convert from unit scale [0, 1] to physical scale [0, 100] and round to nearest 5
            return points_unit_scale.mul_(100.0).div_(5.0).round_().mul_(5.0)

        elif dist_type == 'triangle':
            u = torch.rand(num_samples, size, 2)
            x = torch.min(u, dim=-1).values.unsqueeze(-1)
            y = torch.max(u, dim=-1).values.unsqueeze(-1)
            points_unit_scale = torch.cat([x, y], dim=-1)

        elif dist_type == 'ellipse':
            h, k = 0.5, 0.5  # Center
            a, b = 0.5, 0.25  # Radii
            r = torch.sqrt(torch.rand(num_total_points))
            theta = 2 * torch.pi * torch.rand(num_total_points)
            x = h + a * r * torch.cos(theta)
            y = k + b * r * torch.sin(theta)
            points = torch.stack([x, y], dim=-1)
            points_unit_scale = points.reshape(num_samples, size, 2)

        elif dist_type == 'semicircle':
            h, k = 0.5, 0.0  # Center at the bottom edge
            radius = 0.5
            r = radius * torch.sqrt(torch.rand(num_total_points))
            theta = torch.pi * torch.rand(num_total_points)  # Theta from 0 to pi
            x = h + r * torch.cos(theta)
            y = k + r * torch.sin(theta)
            points = torch.stack([x, y], dim=-1)
            points_unit_scale = points.reshape(num_samples, size, 2)

        elif dist_type == 'annulus':
            h, k = 0.5, 0.5  # Center
            r_outer, r_inner = 0.5, 0.2
            theta = 2 * torch.pi * torch.rand(num_total_points)
            r = torch.sqrt(torch.rand(num_total_points) * (r_outer ** 2 - r_inner ** 2) + r_inner ** 2)
            x = h + r * torch.cos(theta)
            y = k + r * torch.sin(theta)
            points = torch.stack([x, y], dim=-1)
            points_unit_scale = points.reshape(num_samples, size, 2)

        elif dist_type == 'gaussian_mixture':
            means = torch.tensor([[0.25, 0.75], [0.75, 0.25]])
            stds = torch.tensor([[0.1, 0.1], [0.1, 0.1]])

            # Generate all points at once
            cluster_assignments = torch.randint(0, 2, (num_total_points,))
            points = torch.normal(mean=means[cluster_assignments], std=stds[cluster_assignments])

            # Resample points that are out of bounds [0, 1]
            in_bounds = (points >= 0) & (points <= 1)
            all_in_bounds = torch.all(in_bounds, dim=1)

            while not torch.all(all_in_bounds):
                out_of_bounds_indices = torch.where(~all_in_bounds)[0]
                n_resample = len(out_of_bounds_indices)
                new_assignments = torch.randint(0, 2, (n_resample,))
                new_points = torch.normal(mean=means[new_assignments], std=stds[new_assignments])
                points[out_of_bounds_indices] = new_points

                # Recheck bounds
                in_bounds = (points >= 0) & (points <= 1)
                all_in_bounds = torch.all(in_bounds, dim=1)

            points_unit_scale = points.reshape(num_samples, size, 2)

        elif dist_type == 'spiral':
            noise = 0.02
            theta = torch.sqrt(torch.rand(num_total_points)) * 2.5 * 2 * torch.pi
            r_norm = theta / (2.5 * 2 * torch.pi)
            x = 0.5 + (r_norm * torch.cos(theta)) * 0.45 + torch.randn(num_total_points) * noise
            y = 0.5 + (r_norm * torch.sin(theta)) * 0.45 + torch.randn(num_total_points) * noise
            points = torch.stack([x, y], dim=-1)
            points_unit_scale = points.reshape(num_samples, size, 2)

        else:
            raise ValueError(f"无效的程序化 distribution_type: {dist_type}")

        # As a safety measure, clamp all points to be within [0, 1] before scaling
        points_unit_scale = torch.clamp(points_unit_scale, 0.0, 1.0)
        # Convert from unit scale [0, 1] to physical scale [0, 100] and round to nearest 5
        points_physical = points_unit_scale * 100.0
        return points_physical

    if distribution_type == 'real_data':
        if real_data_path is None:
            raise ValueError("使用 'real_data' 分布时必须提供 real_data_path。")
        try:
            df = pd.read_excel(real_data_path)
        except Exception as e:
            raise RuntimeError(f"读取真实数据文件时出错: {e}")

        sor_loc = torch.tensor(df[['source_x', 'source_y']].values, dtype=torch.float32).unsqueeze(0)
        tar_loc = torch.tensor(df[['destination_x', 'destination_y']].values, dtype=torch.float32).unsqueeze(0)

        # 2. Convert time columns to a tensor of shape (num_rows, 1)
        designated_time = torch.tensor(df['t_delivery'].values, dtype=torch.float32).reshape(1, -1, 1)
        tau_h = torch.tensor(df['t_operation'].values, dtype=torch.float32).reshape(1, -1, 1)

    else:
        # --- 程序化生成数据 ---
        sor_loc = _create_points_programmatically(distribution_type)
        tar_loc = _create_points_programmatically(distribution_type)

        ddl_base = torch.ones(num_samples, 1, 1) * 300
        task_indices = torch.arange(1, size + 1, dtype=torch.float32).reshape(1, size, 1)
        time_noise = torch.rand(num_samples, size, 1).mul_(2).sub_(1).mul_(40.0)
        designated_time = (ddl_base + task_indices * 40.0).add_(time_noise).round_()
        tau_h = torch.randint(0, 9, (num_samples, size, 1), dtype=torch.float32).mul_(5.0).add_(60.0)

    MAX_VALUES['time'] = 300 + size * 40.0 + 40.0
    normalization_tensor = torch.tensor([
        MAX_VALUES['location'], MAX_VALUES['location'],  # 对应 sor_loc
        MAX_VALUES['location'], MAX_VALUES['location'],  # 对应 tar_loc
        MAX_VALUES['time'],  # 对应 designated_time
        MAX_VALUES['tau_h']  # 对应 tau_h
    ])

    # --- 统一的固定参数 (对所有模式都适用) ---
    # tau_adp 现在总是在这里被定义为固定值
    tau_adp_fixed = torch.tensor([8.0, 8.0, 30.0])
    param_tensor = tau_adp_fixed.unsqueeze(0).repeat(num_samples, 1)

    # 速度 v 也是固定的
    # Match NVSPState/VectorizedNVSPState and the held-out physical instances.
    # This field is consumed by the task/robot encoder's travel features.
    v_fixed = torch.full((num_samples, 1), 1.2)

    if return_batched:
        if normalize:
            raise ValueError('return_batched requires physical (unnormalized) data')
        return {
            'sor_loc': sor_loc, 'tar_loc': tar_loc, 'designated_time': designated_time,
            'sub_size': torch.full((num_samples,), sub_size, dtype=torch.long),
            'mom_size': torch.full((num_samples,), mom_size, dtype=torch.long),
            'params': param_tensor, 'v': v_fixed, 'tau_h': tau_h,
        }

    # --- 组装成包含物理单位的数据集 ---
    physical_data = [
        {
            'sor_loc': sor_loc[i], 'tar_loc': tar_loc[i], 'designated_time': designated_time[i],
            'sub_size': sub_size, 'mom_size': mom_size,
            'params': param_tensor[i],  # 每个样本的params都是 [8, 8, 30]
            'v': v_fixed[i], 'tau_h': tau_h[i]
        }
        for i in range(num_samples)
    ]

    # --- 根据需要进行归一化处理 ---
    if not normalize:
        if isEval:
            physical_data = collate_to_batch(physical_data)
        return physical_data
    else:
        normalized_data = []
        max_tau = torch.tensor([MAX_VALUES['tau_ad'], MAX_VALUES['tau_ad'], MAX_VALUES['tau_p']])
        for sample in physical_data:
            normalized_sample = {
                'sor_loc': sample['sor_loc'] / MAX_VALUES['location'],
                'tar_loc': sample['tar_loc'] / MAX_VALUES['location'],
                'designated_time': sample['designated_time'] / MAX_VALUES['time'],
                'sub_size': sample['sub_size'], 'mom_size': sample['mom_size'],
                'params': sample['params'] / max_tau,
                'v': sample['v'] / MAX_VALUES['velocity'],
                'tau_h': sample['tau_h'] / MAX_VALUES['tau_h']
            }
            normalized_data.append(normalized_sample)
        return collate_to_batch(normalized_data)


def denormalize_output(normalized_output, key):
    # (此辅助函数无需改变)
    if key not in MAX_VALUES:
        raise ValueError(f"未知的key: {key}。可选: {list(MAX_VALUES.keys())}")
    return normalized_output * MAX_VALUES[key]


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


import pandas as pd
import os

def fileName_increase(file_name):
    # --- Determine the actual file name to write to ---
    actual_file_name = file_name
    base_name, ext = os.path.splitext(file_name)
    counter = 0

    while os.path.exists(actual_file_name):
        try:
            df_existing = pd.read_excel(actual_file_name)
            if len(df_existing) >= 10:
                # File exists and has enough rows, try next file name
                counter += 1
                actual_file_name = f"{base_name}({counter}){ext}"
                print(f"文件 '{actual_file_name}' 存在且数据完整 (>=30行)。正在尝试写入新文件：'{actual_file_name}'。")
            else:
                # File exists but has less than 30 rows, append to it
                break # Exit the loop, use the current actual_file_name
        except Exception as e:
            print(f"尝试读取文件 '{actual_file_name}' 时发生错误: {e}. 将尝试创建新文件。")
            counter += 1
            actual_file_name = f"{base_name}({counter}){ext}"
    return actual_file_name


def collect_result_to_excel_with_lock(write_file, idx, path, instance, package, only_id, lock):
    with lock:
        collect_result_to_excel(write_file, idx, path, instance, package, only_id)

def collect_result_to_excel(file_name, index, path, instance, data_package, only_id):
    """
    将结果收集到 Excel 文件中。如果文件不存在，则创建并添加表头。
    如果文件存在且数据行数小于30，则将新数据插入到现有数据的末尾。
    如果文件存在且数据行数大于等于30，则创建一个新文件，并在文件名后追加递增的数字。

    Args:
        file_name (str): Excel 文件的名称（例如 'results.xlsx'）。
        index (int/str): 数据的索引或标识符。
        data_package (tuple): 包含以下浮点数的元组：
            - cost_dist: 距离成本。
            - cost_delay: 延迟成本。
            - time: 时间。
            - min_cost_delay: 最小延迟成本。
            - min_cost_dist: 最小距离成本。
            - mean_cost_delay: 平均延迟成本。
            - mean_cost_dist: 平均距离成本。
            - mean_cost: 平均总成本。
            - best_cost: 最佳总成本。
    """
    instance = os.path.basename(instance)
    # 定义表头
    headers = [
        'Index', 'Path', 'Instance', 'Sol', 'Cost_Dist', 'Cost_Delay', 'Time',
        'Min_Cost_Delay', 'Min_Cost_Dist', 'Mean_Cost_Delay',
        'Mean_Cost_Dist', 'Mean_Cost', 'Best_Cost', 'Only_id'
    ]
    sol, cost_dist, cost_delay, time, \
        min_cost_delay, min_cost_dist, mean_cost_delay, \
        mean_cost_dist, mean_cost, best_cost = data_package
    # 创建新的数据行
    new_data = {
        'Index': index,
        'Path': path,
        'Instance': instance,
        'Sol': sol,
        'Cost_Dist': cost_dist,
        'Cost_Delay': cost_delay,
        'Time': time,
        'Min_Cost_Delay': min_cost_delay,
        'Min_Cost_Dist': min_cost_dist,
        'Mean_Cost_Delay': mean_cost_delay,
        'Mean_Cost_Dist': mean_cost_dist,
        'Mean_Cost': mean_cost,
        'Best_Cost': best_cost,
        'Only_id': only_id
    }
    new_df_row = pd.DataFrame([new_data])  # 将新数据转换为 DataFrame

    # --- Determine the actual file name to write to ---
    actual_file_name = file_name

    # --- Write the data ---
    if not os.path.exists(actual_file_name):
        # If the determined file name does not exist, create it with headers and data
        df = pd.DataFrame(columns=headers)
        df = pd.concat([df, new_df_row], ignore_index=True)
        df.to_excel(actual_file_name, index=False, engine='openpyxl')
        print(f"文件 '{actual_file_name}' 不存在，已创建并写入表头和第一行数据。")
    else:
        # If the determined file name exists (meaning it had <30 rows), append to it
        try:
            df = pd.read_excel(actual_file_name)
            # 检查现有 DataFrame 的列是否与预期表头一致
            if list(df.columns) != headers:
                print(f"警告：'{actual_file_name}' 的表头与预期不符。新数据仍将追加。")
            df_updated = pd.concat([df, new_df_row], ignore_index=True)
            df_updated.to_excel(actual_file_name, index=False, engine='openpyxl')
            # print(f"数据已成功插入到 '{actual_file_name}'。")
        except Exception as e:
            print(f"读取或写入文件 '{actual_file_name}' 时发生错误: {e}")


def data_sort_mean_with_formatting(fileName, other_frame=None, gap_name=None):
    """
    对 Excel 文件进行排序、计算平均值、重排特定列，并对平均值行的数值进行加粗标红格式化。
    新增功能：在文件顶部额外添加一个所有平均值行的汇总区域。
    """
    try:
        df = pd.read_excel(fileName, engine='openpyxl')
        print(f"文件 '{fileName}' 加载成功！")
    except FileNotFoundError:
        print(f"错误: 文件 '{fileName}' 未找到。请确保文件路径正确。")
        return
    except Exception as e:
        print(f"打开文件时发生错误: {e}")
        return

    # --- 您的原始数据处理逻辑 ---
    df_sorted = df.sort_values(by=['Only_id'])
    # df_sorted['block_id'] = df_sorted['Index'].apply(lambda x: x.rsplit('_', 1)[-1] if '_' in x else x)
    df_sorted['block_id'] = df_sorted['Only_id']
    heuristic_methods = ['alns', 'GA', 'DIWO', 'IGA', 'vms']
    df_sorted['effective_cost'] = np.where(
        df_sorted['Index'].isin(heuristic_methods),
        df_sorted['Mean_Cost'],
        df_sorted['Best_Cost']
    )
    df_sorted = df_sorted[df_sorted['Instance'].notna()].copy()  # 使用 .copy() 避免后续操作产生警告
    bks = df_sorted.groupby('Instance')['effective_cost'].min()
    df_sorted['BKS'] = df_sorted['Instance'].map(bks)
    # 计算Gap百分比
    # (effective_cost - BKS) / BKS * 100
    df_sorted['Gap_Percent'] = np.divide(
        (df_sorted['effective_cost'] - df_sorted['BKS']),
        df_sorted['BKS'],
        out=np.zeros_like(df_sorted['effective_cost'], dtype=float),  # 如果BKS为0，则结果为0
        where=df_sorted['BKS'] != 0
    ) * 100
    df_sorted['is_bks'] = (df_sorted['effective_cost'] == df_sorted['BKS'])
    processed_rows = []
    summary_avg_rows = []  ### 新增: 创建一个列表专门用于存放所有的平均值行 ###
    current_block_id = None
    current_block_data = []

    for _, row in df_sorted.iterrows():
        block_id = row['block_id']
        if current_block_id is None:
            current_block_id = block_id

        if block_id != current_block_id:
            if current_block_data:
                block_df = pd.DataFrame(current_block_data)
                avg_row = block_df.mean(numeric_only=True).to_dict()
                avg_row['Index'] = f"{current_block_id}_平均值"
                avg_row['Instance'] = block_df.loc[0, 'Instance']
                processed_rows.extend(current_block_data)
                processed_rows.append(avg_row)
                summary_avg_rows.append(avg_row)

            current_block_id = block_id
            current_block_data = [row.to_dict()]
        else:
            current_block_data.append(row.to_dict())

    if current_block_data:
        block_df = pd.DataFrame(current_block_data)
        avg_row = block_df.mean(numeric_only=True).to_dict()
        avg_row['Index'] = f"{current_block_id}_平均值"

        processed_rows.extend(current_block_data)
        processed_rows.append(avg_row)
        summary_avg_rows.append(avg_row)

        # 分别为汇总行和处理后的数据创建 DataFrame
    df_summary = pd.DataFrame(summary_avg_rows)
    df_details = pd.DataFrame(processed_rows)

    desired_column_order = df_details.columns
    blank_row_df = pd.DataFrame([{}], columns=df_details.columns)
    df_combined = pd.concat([df_summary, blank_row_df, df_details], ignore_index=True)
    df_final = df_combined[desired_column_order]

    # 从最终的 DataFrame 中删除临时列
    df_final = df_final.drop(columns=['block_id'], errors='ignore')  ### 修改: 在最终的Da

    if gap_name is not None:
        gap_value_series = df_final.loc[df_final['Index'] == gap_name, 'Gap_Percent']
        gap_value_series = gap_value_series.iloc[0] if not gap_value_series.empty else None
    else:
        gap_value_series = None

    # ==================================================================
    # 重新排列列
    cols = df_final.columns.tolist()
    if len(cols) > 1:
        # 将 is_bks 移到末尾，以免干扰现有重排逻辑
        if 'is_bks' in cols:
            cols.remove('is_bks')
            cols.append('is_bks')

        first_col, second_col, third_col = cols[0], cols[1], cols[2]
        other_cols = cols[3:-1] # 排除最后的 is_bks
        other_cols_reversed = other_cols[::-1]
        new_col_order = [first_col, second_col, third_col] + other_cols_reversed + ['is_bks']
        df_final = df_final[new_col_order]

    # --- 步骤 1: 先将处理和重排好的数据写入文件 ---
    output_fileName = add_suffix_to_filename_in_path(fileName, "_sort")
    # ### 修改: 将最终合并后的 DataFrame 写入 Excel ###
    with pd.ExcelWriter(output_fileName, engine='openpyxl') as writer:
        # 将主数据写入第一个名为 'Main_Results' 的Sheet
        df_final.to_excel(writer, sheet_name='Main_Results', index=False)
        if other_frame is not None:
            # 将分析结果写入第二个名为 'TimeToTarget_Analysis' 的Sheet
            other_frame.to_excel(writer, sheet_name='TimeToTarget_Analysis', index=False)

    # df_final.to_excel(output_fileName, index=False, engine='openpyxl')
    print(f"数据已处理并保存到 '{output_fileName}'。")

    # --- 步骤 2: 使用 openpyxl 进行格式化 ---
    # 这部分代码无需修改，因为它通过检查第一列是否为空来识别所有平均值行，
    # 所以它会自动格式化顶部的汇总区和数据内部的平均值行。
    print("开始应用格式...")
    bold_red_font = Font(bold=True, color="FF0000")
    red_font = Font(color="FF0000")  # 用于BKS的非粗体红字，以作区分
    wb = load_workbook(output_fileName)
    ws = wb.active
    header = [cell.value for cell in ws[1]]
    try:
        index_col_idx = header.index('Index')
        effective_cost_col_idx = header.index('effective_cost')
        is_bks_col_idx = header.index('is_bks')
    except ValueError as e:
        print(f"错误: Excel文件中缺少必要的列: {e}。无法进行格式化。")
        wb.save(output_fileName) # 保存未格式化的文件
        return
    # 遍历行来应用格式
    for row_idx, row in enumerate(ws.iter_rows(min_row=2), start=2):
        # 条件1: 格式化平均值行 (整行加粗标红)
        if "平均值" in str(row[index_col_idx].value) and 'greed' not in str(row[index_col_idx].value):
            cell_to_format = row[effective_cost_col_idx]
            if isinstance(cell_to_format.value, (int, float)):
                cell_to_format.font = bold_red_font
                cell_to_format.number_format = '0.000'

        # 条件2: 格式化BKS行 (仅将 effective_cost 单元格标红)
        elif row[is_bks_col_idx].value is True:
            cell_to_format = row[effective_cost_col_idx]
            cell_to_format.font = red_font

    first_col_letter = 'B'
    max_length = 0
    for cell in ws[first_col_letter]:
        try:
            if cell.value:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
        except:
            pass

    adjusted_width = max_length + 2
    ws.column_dimensions[first_col_letter].width = adjusted_width

    # --- 步骤 3: 保存带格式的文件 ---
    wb.save(output_fileName)
    print(f"格式化完成！最终文件已保存在 '{output_fileName}'。")
    return df_final, gap_value_series


def convert_to_predefined_routes(raw_data, num_tasks):
    shuttles = [f'S{i + 1}' for i in range(8)]
    carriers = [f'C{i + 1}' for i in range(4)]
    vehicle_names = shuttles + carriers

    depot_start = 0
    depot_end = num_tasks + 1

    if not raw_data or not isinstance(raw_data[0], list):
        print("Error: Input data is not in the expected format.")
        return {}

    routes_list = raw_data[0]

    predefined_routes = {}

    num_routes_to_process = min(len(routes_list), len(vehicle_names))

    for i in range(num_routes_to_process):
        vehicle_name = vehicle_names[i]
        raw_route = routes_list[i]

        # Filter out -1 placeholders and convert 0-indexed customer IDs to 1-based IDs
        customer_stops = [stop + 1 for stop in raw_route if stop != -1]

        # Assemble the full route: [start, ...customers..., end]
        full_route = [depot_start] + customer_stops

        predefined_routes[vehicle_name] = full_route

    return predefined_routes



def get_topological_sort(schedule_data: list[list[int]]):
    """
    Args:
        schedule_data (list[list[int]]): A list of lists representing vehicle schedules.
                                         Each inner list contains task IDs, with -1
                                         indicating empty slots.

    Returns:
        list[int] | str: A list of task IDs in topological order if successful,
                         or a string message indicating that a cycle was detected.
    """

    # Filter out -1 values from each schedule row
    schedule_np = np.array(schedule_data.to('cpu'))
    seq = [row[row != -1] for row in schedule_np]

    # --- Automatically determine total_num_tasks ---
    max_task_id = -1
    for vehicle_schedule in seq:
        if len(vehicle_schedule) > 0:
            current_max = np.max(vehicle_schedule)
            if current_max > max_task_id:
                max_task_id = int(current_max)  # Ensure it's an int

    if max_task_id == -1:  # No tasks found in any schedule
        return []  # Return empty list for no tasks or handle as an error/specific case

    total_num_tasks = max_task_id + 1
    # --- End automatic determination ---

    # Initialize graph (adjacency list) and in-degrees for all possible tasks
    all_tasks_ids = set(range(total_num_tasks))
    graph = {i: [] for i in all_tasks_ids}
    in_degree = {i: 0 for i in all_tasks_ids}

    # Construct the graph based on sequential dependencies in schedules
    for vehicle_schedule in seq:
        for i in range(len(vehicle_schedule) - 1):
            u = vehicle_schedule[i]
            v = vehicle_schedule[i + 1]

            # Basic validation: ensure tasks are within the determined range
            if u not in all_tasks_ids or v not in all_tasks_ids:
                return f"Error: Task ID {u} or {v} found which is outside the automatically determined range (0 to {total_num_tasks - 1}). This suggests an issue in task numbering or data."

            # Add edge (u -> v) only if it doesn't already exist to avoid duplicate in_degree increments
            if v not in graph[u]:
                graph[u].append(v)
                in_degree[v] += 1

    # Kahn's Algorithm for Topological Sort
    queue = deque()
    for task_id in all_tasks_ids:
        if in_degree[task_id] == 0:
            queue.append(task_id)

    topological_order = []
    visited_count = 0

    while queue:
        u = queue.popleft()
        topological_order.append(u)
        visited_count += 1

        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Check for cycles
    if visited_count == total_num_tasks:
        topological_order = [int(task_id) for task_id in topological_order]
        return topological_order
    else:
        cyclic_nodes = sorted([node for node in all_tasks_ids if in_degree[node] > 0])
        return f"A cycle was detected in the task dependencies. Topological sort is not possible. Cyclic nodes: {cyclic_nodes}"



def add_suffix_to_filename_in_path(file_path, suffix="_raw"):
    """
    为一个完整文件路径中的文件名部分添加后缀。

    Args:
        file_path (str): 原始文件路径。
        suffix (str): 要添加的后缀。

    Returns:
        str: 带有新文件名的完整路径。
    """
    # 分离目录和文件名
    directory, filename = os.path.split(file_path)
    # 分离文件名和扩展名
    base_name, extension = os.path.splitext(filename)
    # 创建新的文件名
    new_filename = f"{base_name}{suffix}{extension}"
    # 重新组合成完整路径
    return os.path.join(directory, new_filename)


def to_home_absolute_path(path_str):
    """
    将一个路径转换为基于用户主目录的绝对路径。
    如果路径本身已经是绝对路径，则直接返回。
    """
    # 检查输入路径是否已经是绝对路径
    if os.path.isabs(path_str):
        return path_str

    # 如果是相对路径，则和主目录拼接
    return os.path.join(os.path.expanduser('~'), path_str)
