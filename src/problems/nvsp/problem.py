import os
import pickle
from torch.utils.data import Dataset
import torch

from problems.nvsp.state import NVSPState, omega
from utils.beam_search import beam_search
from utils.data_utils import generate_nvsp_data, generate_nvsp_data_robustness


class NVSP(object):

    NAME = 'nvsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        # pi = pi.long()
        sub_vehicle_size = dataset['sub_size'][0].item()
        vehicle_size = sub_vehicle_size + dataset['mom_size'][0].item()
        sequences = NVSP.process_sequences_torch(pi, vehicle_size)
        # cost = NVSP.calculate_sequence_cost_2(sequences, torch.cat((dataset['sor_loc'], dataset['tar_loc']), 1))
        cost = NVSP.calculate_sequence_cost_with_pairing_tensor(sequences, torch.cat((dataset['sor_loc'], dataset['tar_loc']), 1), sub_vehicle_size)
        return cost, None



    @staticmethod
    def make_dataset(*args, **kwargs):
        return NVSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return NVSPState.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = NVSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)

    @staticmethod
    def process_sequences_torch(sequences, vehicle_size=5):
        """
        sequences: Tensor of shape (batch_size, n_step)
        task_size: The number of tasks
        vehicle_size: The number of vehicles
        """
        batch_size, n_step, _ = sequences.shape
        # 1. 计算任务索引和车辆索引
        tasks = sequences[:, :, 0].repeat(1, 1, 2)  # 任务索引 (batch_size, n_step)
        vehicle_subs = sequences[:, :, 1]
        vehicle_moms = sequences[:, :, 2]
        # vehicles = sequences // task_size  # 车辆索引 (batch_size, n_step)

        # 2. 创建存放任务的张量，初始化为 -1
        result = torch.full((batch_size, vehicle_size, n_step), -1, dtype=torch.long, device=sequences.device)
        vehicle_task = torch.cat((vehicle_subs, vehicle_moms), 1)
        # 3. 计算 batch 维度索引
        n_step = n_step * 2
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, n_step)

        # 4. 计算 task_positions (每辆车已分配的任务索引)
        seq1 = vehicle_task.unsqueeze(2)  # (B, N, 1)
        seq2 = vehicle_task.unsqueeze(1)  # (B, 1, N)

        # 构建相等矩阵 (B, N, N)：对于每个 (b, i, j)，判断第 b 个 batch 的第 i 和第 j 个元素是否相等
        equal_matrix = (seq1 == seq2).long()

        # 仅保留下三角（包含对角线），确保只统计前面（含当前）位置的重复项
        tril_mask = torch.tril(torch.ones((n_step, n_step), device=vehicle_task.device)).unsqueeze(0)  # (1, N, N)
        tril_equal = equal_matrix * tril_mask  # (B, N, N)

        # 对每个时间步 i，统计有多少个 j≤i 满足 seq[i] == seq[j]，再 -1 得到编号
        vehicle_counts = tril_equal.sum(dim=2) - 1  # (B, N)

        result[batch_indices, vehicle_task, vehicle_counts.long()] = tasks

        return result


    # 计算代价，这是母车、子车共同去完成任务的代价计算方式，并且一个任务作为单个任务来执行
    @staticmethod
    def calculate_sequence_cost(sequence, sor_loc, tar_loc, start_pos):
        """
        计算总曼哈顿距离成本，优化计算流程以减少冗余操作，提高运行效率。

        Args:
            sequence: (batch, vehicle_size, task_size) - 任务顺序，-1 表示填充
            sor_loc: (batch, task_size, 2) - 每个任务的起始坐标
            tar_loc: (batch, task_size, 2) - 每个任务的终点坐标
            start_pos: (batch, vehicle_size, 1) - 每辆车的起始任务索引

        Returns:
            (batch,) 形状的张量，包含每个批次的总距离成本
        """
        batch_size, vehicle_size, task_size = sequence.shape

        # 有效任务的掩码（True 表示真实任务，False 表示填充）
        valid_mask = (sequence != -1)

        # 获取每辆车的起始任务坐标 - (batch, vehicle_size, 2)
        start_points = sor_loc.gather(1, start_pos.expand(-1, -1, 2)).squeeze(2)

        # 任务坐标索引：将 -1 填充项裁剪为 0，以避免索引错误
        clamped_seq = sequence.clamp(min=0)

        # 直接通过索引获取任务的起点和终点坐标
        seq_start_points = sor_loc[:, None, :, :].expand(-1, vehicle_size, -1, -1).gather(
            2, clamped_seq.unsqueeze(-1).expand(-1, -1, -1, 2)
        )
        seq_end_points = tar_loc[:, None, :, :].expand(-1, vehicle_size, -1, -1).gather(
            2, clamped_seq.unsqueeze(-1).expand(-1, -1, -1, 2)
        )

        # 计算同一任务内的曼哈顿距离
        deltas_1 = torch.abs(seq_start_points - seq_end_points).sum(dim=-1)
        deltas_dist_1 = (deltas_1 * valid_mask).sum(dim=-1)

        # 计算任务间的曼哈顿距离（任务终点到下一任务起点）
        deltas_2 = torch.abs(seq_start_points[:, :, 1:] - seq_end_points[:, :, :-1]).sum(dim=-1)
        transition_mask = valid_mask[:, :, :-1] & valid_mask[:, :, 1:]
        deltas_dist_2 = (deltas_2 * transition_mask).sum(dim=-1)

        # 计算起始点到第一个任务的曼哈顿距离
        transition_mask_start = valid_mask[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        start_to_first_dist = (torch.abs(start_points[:, :, None, :] - seq_start_points[:, :, 0:1, :]) * transition_mask_start).sum(dim=[-1, -2])

        # 计算总距离
        total_distances = (deltas_dist_1 + deltas_dist_2 + start_to_first_dist).sum(dim=1)

        return total_distances

    # 带时间惩罚项的损失函数计算
    @staticmethod
    def calculate_sequence_cost_with_time(sequence, task_loc, designated_time, topologic_sort, params, sub_vehicle_size):
        B, V, T = sequence.shape
        N = task_loc.size(1)
        device = sequence.device
        half_T = T // 2

        # 扁平化数据
        task_flat = sequence.view(B, -1)  # [B, V*T]
        vehicle_flat = torch.arange(V, device=device).view(1, V, 1).repeat(B, 1, T).reshape(B, -1)  # [B, V*T]
        time_flat = torch.arange(T, device=device).repeat(B, V).view(B, -1)  # [B, V*T]
        batch_indices = torch.arange(B).unsqueeze(1).expand(B, N)  # [B, N]
        batch_idx = torch.arange(B)

        # 子车任务掩码
        sub_mask = (vehicle_flat < sub_vehicle_size) & (task_flat >= 0)
        b_idx_sub, flat_idx_sub = torch.nonzero(sub_mask, as_tuple=True)
        sub_task_idx = task_flat[b_idx_sub, flat_idx_sub]
        sub_vehicle_id = vehicle_flat[b_idx_sub, flat_idx_sub]
        sub_time_id = time_flat[b_idx_sub, flat_idx_sub]

        # 子车任务映射
        task_to_sub_vehicle = torch.full((B, N), -1, device=device, dtype=torch.long)
        task_to_sub_time = torch.full((B, N), -1, device=device, dtype=torch.long)
        task_to_sub_vehicle[b_idx_sub, sub_task_idx] = sub_vehicle_id
        task_to_sub_time[b_idx_sub, sub_task_idx] = sub_time_id

        # 母车任务掩码
        is_mother = torch.arange(V, device=device) >= sub_vehicle_size
        mother_mask = is_mother.view(1, V, 1).expand(B, V, T) & (sequence >= 0)
        b_idx, v_idx, t_idx = torch.nonzero(mother_mask, as_tuple=True)
        curr_task = sequence[b_idx, v_idx, t_idx]
        b_idx_mom, flat_idx_mom = torch.nonzero(mother_mask.view(B, -1), as_tuple=True)
        mom_time_id = time_flat[b_idx_mom, flat_idx_mom]

        # 母车上一任务位置, p_mom 对应的是母车执行当前节点的上一个节点
        t_prev = (t_idx - 1).clamp(min=0)
        prev_mother_task = sequence[b_idx, v_idx, t_prev]
        valid_mother_prev = (t_idx > 0) & (prev_mother_task >= 0)
        p_mother = torch.zeros(len(b_idx), 2, device=device)
        p_mother[valid_mother_prev] = task_loc[b_idx[valid_mother_prev], prev_mother_task[valid_mother_prev]]

        # 子车 id 和子车时间
        sub_v_idx = task_to_sub_vehicle[b_idx, curr_task]
        sub_t_idx = task_to_sub_time[b_idx, curr_task]
        sub_t_prev = (sub_t_idx - 1).clamp(min=0)

        # 子车上一任务, p_sub 对应的是母车移动到子车的那个节点
        sub_prev_task = sequence[b_idx, sub_v_idx, sub_t_prev]
        valid_sub_prev = (sub_t_idx > 0) & (sub_prev_task >= 0)
        p_sub = torch.zeros(len(b_idx), 2, device=device)
        p_sub[valid_sub_prev] = task_loc[b_idx[valid_sub_prev], sub_prev_task[valid_sub_prev]]

        # 当前任务位置
        p_task = task_loc[b_idx, curr_task]

        # 获得拓扑排序对应的位置顺序
        curr_task_bt = curr_task.view(B, N)
        # value_to_index = torch.full((B, N), -1, dtype=torch.long, device=device)
        # value_to_index[batch_indices, topologic_sort] = torch.arange(N).unsqueeze(0).expand(B, N)
        # topologic_index = value_to_index[batch_indices, curr_task_bt]  # [B, N]
        ref_expanded = topologic_sort.unsqueeze(2)  # shape: (2, 6, 1)
        new_expanded = curr_task_bt.unsqueeze(1)  # shape: (2, 1, 6)
        # 比较并找到匹配位置
        matches = (ref_expanded == new_expanded).int()  # shape: (2, 6, 6)
        topologic_index = torch.argmax(matches, dim=2)
        # 计算时间开销
        # topologic_index_add = topologic_index + 1
        ksi_s = torch.full((B, N), -1, dtype=torch.float, device=device)
        ksi_c = torch.full((B, N), -1, dtype=torch.float, device=device)
        T_s = torch.full((B, N), 0, dtype=torch.float, device=device)
        T_f_s = torch.full((B, N), 0, dtype=torch.float, device=device)
        T_f_c = torch.full((B, N), 0, dtype=torch.float, device=device)
        ksi_max = torch.full((B, N), 0, dtype=torch.float, device=device)
        vehicle_time = torch.full((B, V), 0, dtype=torch.float, device=device)
        # sub_time_id = sub_time_id.view(B, N)
        # mom_time_id = mom_time_id.view(B, N)
        tau_a, tau_d, tau_p, tau_h, v = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4]
        b_sub_v_idx = sub_v_idx.view(B, N)
        b_mom_v_idx = v_idx.view(B, N)
        sub_loc = p_sub.view(-1, T, 2)
        mom_loc = p_mother.view(-1, T, 2)
        curr_loc = p_task.view(-1, T, 2)
        for i in range(T):
            curr_index = topologic_index[:, i]
            task_index = curr_task_bt[batch_idx, curr_index]
            update_mask = task_index < half_T
            b_update_mask = ~update_mask
            t_p_d = torch.sum(torch.abs(curr_loc[batch_idx, curr_index] - sub_loc[batch_idx, curr_index]), dim=1) / v
            t_d_p = torch.sum(torch.abs(mom_loc[batch_idx, curr_index] - sub_loc[batch_idx, curr_index]), dim=1) / v
            # 对应哪台子车和母车
            sub_vehicle = b_sub_v_idx[batch_idx, curr_index]
            mom_vehicle = b_mom_v_idx[batch_idx, curr_index]
            ksi_s[batch_idx[update_mask], task_index[update_mask]] = vehicle_time[batch_idx[update_mask], sub_vehicle[update_mask]]
            ksi_c[batch_idx[update_mask], task_index[update_mask]] = vehicle_time[batch_idx[update_mask], mom_vehicle[update_mask]] + t_d_p[update_mask]
            ksi_max[batch_idx, task_index] = torch.max(ksi_s[batch_idx, task_index], ksi_c[batch_idx, task_index])
            T_s[batch_idx[update_mask], task_index[update_mask]] = ksi_max[batch_idx[update_mask], task_index[update_mask]] + tau_a[update_mask] + t_p_d[update_mask]
            # task_index = task_index - half_T
            T_f_s[batch_idx[b_update_mask], task_index[b_update_mask]] = T_s[batch_idx[b_update_mask], task_index[b_update_mask] - half_T] + tau_d[b_update_mask] + tau_p[
                b_update_mask] + tau_h[b_update_mask] + t_p_d[b_update_mask]
            T_f_c[batch_idx[b_update_mask], task_index[b_update_mask]] = T_s[batch_idx[b_update_mask], task_index[b_update_mask] - half_T] + tau_d[b_update_mask] + tau_p[
                b_update_mask] + t_p_d[b_update_mask]
            T_f_s[batch_idx[update_mask], task_index[update_mask]] = T_s[batch_idx[update_mask], task_index[update_mask]] + tau_p[update_mask]
            T_f_c[batch_idx[update_mask], task_index[update_mask]] = T_s[batch_idx[update_mask], task_index[update_mask]] + tau_p[update_mask]
            vehicle_time[batch_idx, sub_vehicle] = T_f_s[batch_idx, task_index]
            vehicle_time[batch_idx, mom_vehicle] = T_f_c[batch_idx, task_index]
        # 成本计算
        cost_to_pick = torch.sum(torch.abs(p_mother - p_sub), dim=1)
        cost_to_task = torch.sum(torch.abs(p_sub - p_task), dim=1)
        if designated_time.ndim == task_loc.ndim:
            designated_time = designated_time.repeat(1, T_f_s.size(1))
        else:
            designated_time = designated_time.squeeze(-1).repeat(1, 2)
        designated_time[..., :half_T] = 0
        T_f_s[..., :half_T] = 0
        cost_delay = torch.sum((T_f_s - designated_time).clamp(min=0), dim=1)
        dist_total_cost = torch.zeros(B, device=device)
        dist_total_cost = dist_total_cost.index_add(0, b_idx, cost_to_pick + cost_to_task)
        totol_cost = omega * dist_total_cost + (1 - omega) * cost_delay
        return totol_cost

    @staticmethod
    def calculate_cost_with_topology(
            routes: torch.Tensor,  # Shape: (B, V, T_max), 车辆的路线表，用于查找谁负责哪个任务
            coords: torch.Tensor,  # Shape: (B, 2*N, 2), 地点坐标
            topologic_sort: torch.Tensor,  # Shape: (B, N), 全局的【订单】执行顺序
            designated_time: torch.Tensor,  # Shape: (B, N)
            params: torch.Tensor,
            n_sub_vehicles: int,  # 子车数量
            omega: float = 0.5
    ):
        """
        【最终版成本计算函数 V4】
        严格按照给定的 topologic_sort (订单的全局执行顺序) 来计算总成本。
        """
        B, V, T_max = routes.shape
        N = topologic_sort.shape[1]
        device = routes.device
        batch_idx = torch.arange(B, device=device)
        n_orders = N

        # --- 步骤 1: 数据解析 - 建立 Order -> Vehicle 的映射关系 ---
        # 这个预处理步骤让我们能够快速地根据订单号找到负责的车辆。
        order_to_sub = torch.full((B, n_orders), -1, dtype=torch.long, device=device)
        order_to_mom = torch.full((B, n_orders), -1, dtype=torch.long, device=device)

        is_sub_mask = (torch.arange(V, device=device) < n_sub_vehicles).view(1, V, 1)

        # 找到所有子车的任务分配并记录
        sub_b, sub_v, sub_t = torch.nonzero((routes >= 0) & is_sub_mask, as_tuple=True)
        sub_orders = routes[sub_b, sub_v, sub_t]
        order_to_sub[sub_b, sub_orders] = sub_v

        # 找到所有母车的任务分配并记录
        mom_b, mom_v, mom_t = torch.nonzero((routes >= 0) & ~is_sub_mask, as_tuple=True)
        mom_orders = routes[mom_b, mom_v, mom_t]
        order_to_mom[mom_b, mom_orders] = mom_v

        # --- 步骤 2: 初始化模拟状态 ---
        # 车辆时间和位置都从0开始
        vehicle_time = torch.zeros((B, V), dtype=torch.float, device=device)
        vehicle_pos_coords = torch.zeros((B, V, 2), dtype=torch.float, device=device)  # 直接存储坐标，默认(0,0)

        total_dist_cost = torch.zeros(B, dtype=torch.float, device=device)
        order_completion_times = torch.zeros((B, n_orders), dtype=torch.float, device=device)

        # 解包参数
        tau_a, tau_d, tau_p, tau_h, v = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4]

        # --- 步骤 3: 严格按照拓扑排序，遍历订单并计算成本 ---
        for i in range(N):
            # a. 从拓扑排序中获取当前要处理的订单
            order_indices = topologic_sort[:, i]

            # b. 从映射中查找负责该订单的子车和母车
            # (请确保 topologic_sort 中的所有订单都能在 routes 中找到对应的车辆分配)
            sub_vehicles = order_to_sub[batch_idx, order_indices]
            mom_vehicles = order_to_mom[batch_idx, order_indices]

            # c. 获取车辆执行此任务前的出发位置坐标
            pre_sub_loc = vehicle_pos_coords[batch_idx, sub_vehicles]
            pre_mom_loc = vehicle_pos_coords[batch_idx, mom_vehicles]

            # d. 获取当前订单的取送货坐标
            pickup_loc = coords[batch_idx, order_indices]
            delivery_loc = coords[batch_idx, order_indices + n_orders]

            # e. 执行完整的原子化订单时间演进计算
            dist_mom_to_sub = torch.sum(torch.abs(pre_sub_loc - pre_mom_loc), dim=1)
            time_mom_to_sub = dist_mom_to_sub / v
            rendezvous_arrival_time = vehicle_time[batch_idx, mom_vehicles] + time_mom_to_sub
            departure_time_from_rendezvous = torch.max(vehicle_time[batch_idx, sub_vehicles], rendezvous_arrival_time)

            dist_rendezvous_to_pickup = torch.sum(torch.abs(pickup_loc - pre_sub_loc), dim=1)
            time_rendezvous_to_pickup = dist_rendezvous_to_pickup / v
            arrival_time_at_pickup = departure_time_from_rendezvous + time_rendezvous_to_pickup

            finish_time_at_pickup = arrival_time_at_pickup + tau_p

            dist_pickup_to_delivery = torch.sum(torch.abs(delivery_loc - pickup_loc), dim=1)
            time_pickup_to_delivery = dist_pickup_to_delivery / v
            arrival_time_at_delivery = finish_time_at_pickup + time_pickup_to_delivery

            start_time_at_delivery = arrival_time_at_delivery + tau_a
            final_finish_time = start_time_at_delivery + tau_d + tau_h
            final_mom_finish_time = start_time_at_delivery + tau_d

            # f. 更新车辆状态和统计数据，为下一个订单的计算做准备
            vehicle_time[batch_idx, sub_vehicles] = final_finish_time
            vehicle_time[batch_idx, mom_vehicles] = final_mom_finish_time

            vehicle_pos_coords[batch_idx, sub_vehicles] = delivery_loc
            vehicle_pos_coords[batch_idx, mom_vehicles] = delivery_loc

            total_dist_this_step = dist_mom_to_sub + dist_rendezvous_to_pickup + dist_pickup_to_delivery
            total_dist_cost += total_dist_this_step

            order_completion_times[batch_idx, order_indices] = final_finish_time

        # --- 步骤 4: 计算最终总成本 ---
        cost_delay = torch.sum((order_completion_times - designated_time.squeeze(-1)).clamp(min=0), dim=1)
        total_cost = omega * total_dist_cost + (1 - omega) * cost_delay

        return total_cost

    # 计算代价，这是母车接子车的代价计算方式
    @staticmethod
    def calculate_sequence_cost_with_pairing_tensor(sequence, task_loc, sub_vehicle_size):
        B, V, T = sequence.shape
        N = task_loc.size(1)
        device = sequence.device

        # 扁平化数据
        task_flat = sequence.view(B, -1)  # [B, V*T]
        vehicle_flat = torch.arange(V, device=device).view(1, V, 1).repeat(B, 1, T).reshape(B, -1)  # [B, V*T]
        time_flat = torch.arange(T, device=device).repeat(B, V).view(B, -1)  # [B, V*T]

        # 子车任务掩码
        sub_mask = (vehicle_flat < sub_vehicle_size) & (task_flat >= 0)
        b_idx_sub, flat_idx_sub = torch.nonzero(sub_mask, as_tuple=True)
        sub_task_idx = task_flat[b_idx_sub, flat_idx_sub]
        sub_vehicle_id = vehicle_flat[b_idx_sub, flat_idx_sub]
        sub_time_id = time_flat[b_idx_sub, flat_idx_sub]

        # 子车任务映射
        task_to_sub_vehicle = torch.full((B, N), -1, device=device, dtype=torch.long)
        task_to_sub_time = torch.full((B, N), -1, device=device, dtype=torch.long)
        task_to_sub_vehicle[b_idx_sub, sub_task_idx] = sub_vehicle_id
        task_to_sub_time[b_idx_sub, sub_task_idx] = sub_time_id

        # 母车任务掩码
        is_mother = torch.arange(V, device=device) >= sub_vehicle_size
        mother_mask = is_mother.view(1, V, 1).expand(B, V, T) & (sequence >= 0)
        b_idx, v_idx, t_idx = torch.nonzero(mother_mask, as_tuple=True)
        curr_task = sequence[b_idx, v_idx, t_idx]

        # 母车上一任务位置
        t_prev = (t_idx - 1).clamp(min=0)
        prev_mother_task = sequence[b_idx, v_idx, t_prev]
        valid_mother_prev = (t_idx > 0) & (prev_mother_task >= 0)
        p_mother = torch.zeros(len(b_idx), 2, device=device)
        p_mother[valid_mother_prev] = task_loc[b_idx[valid_mother_prev], prev_mother_task[valid_mother_prev]]

        # 子车 id 和子车时间
        sub_v_idx = task_to_sub_vehicle[b_idx, curr_task]
        sub_t_idx = task_to_sub_time[b_idx, curr_task]
        sub_t_prev = (sub_t_idx - 1).clamp(min=0)

        # 子车上一任务
        sub_prev_task = sequence[b_idx, sub_v_idx, sub_t_prev]
        valid_sub_prev = (sub_t_idx > 0) & (sub_prev_task >= 0)
        p_sub = torch.zeros(len(b_idx), 2, device=device)
        p_sub[valid_sub_prev] = task_loc[b_idx[valid_sub_prev], sub_prev_task[valid_sub_prev]]

        # 当前任务位置
        p_task = task_loc[b_idx, curr_task]

        # 成本计算
        cost_to_pick = torch.sum(torch.abs(p_mother - p_sub), dim=1)
        cost_to_task = torch.sum(torch.abs(p_sub - p_task), dim=1)
        total_cost = torch.zeros(B, device=device)
        total_cost = total_cost.index_add(0, b_idx, cost_to_pick + cost_to_task)

        return total_cost


    # 死锁检测，序列可行性判断
    # 示例 tensor
    # schedule = torch.tensor([[[2, 1, -1, -1, -1],
    #                           [-1, -1, -1, -1, -1],
    #                           [3, -1, -1, -1, -1],
    #                           [1, -1, -1, -1, -1],
    #                           [2, 3, -1, -1, -1]],
    #
    #                          [[0, -1, -1, -1, -1],
    #                           [2, 3, 6, -1, -1],
    #                           [4, 1, -1, -1, -1],
    #                           [0, 2, -1, -1, -1],
    #                           [1, 3, 4, -1, -1]]])
    @staticmethod
    def detect_deadlock(schedule):
        batch_size, num_vehicles, max_tasks = schedule.shape

        # 获取所有唯一任务（按 batch 计算）
        unique_tasks = torch.unique(schedule, sorted=True)
        unique_tasks = unique_tasks[unique_tasks != -1]  # 移除无效任务 -1
        num_tasks = unique_tasks.shape[0]  # 任务总数

        # 将任务转换为索引形式
        task_indices = torch.searchsorted(unique_tasks, schedule.clamp(min=0))  # 形状 (batch_size, num_vehicles, max_tasks)

        # 计算任务顺序矩阵（构建等待图）
        mask = schedule != -1
        next_task = torch.roll(schedule, shifts=-1, dims=2)  # 向右滚动模拟后继任务
        mask_shift = mask & (next_task != -1)  # 仅保留有效任务对

        src_tasks = task_indices[mask_shift]  # 任务执行顺序
        dst_tasks = torch.searchsorted(unique_tasks, next_task[mask_shift])  # 目标任务索引
        batch_indices = torch.nonzero(mask_shift, as_tuple=True)[0]  # 获取 batch 维度索引

        # 构建邻接矩阵
        adj_matrix = torch.zeros((batch_size, num_tasks, num_tasks), dtype=torch.int, device=schedule.device)
        adj_matrix[batch_indices, src_tasks, dst_tasks] = 1  # 记录任务间的等待关系

        # 传递闭包计算（更优于 Floyd-Warshall）
        reach = adj_matrix.clone()
        for _ in range(num_tasks - 1):
            reach = reach | reach.bmm(adj_matrix)  # 逐步扩展可达路径

        # 检测是否存在环路（对角线元素是否大于0）
        deadlock_exists = (torch.diagonal(reach, dim1=1, dim2=2) > 0).any(dim=1)

        return deadlock_exists


def make_instance(args):
    loc, sub_depot, mom_depot = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float),
        # 'sub_depot': torch.tensor(sub_depot, dtype=torch.int64).unsqueeze(1),
        # 'mom_depot': torch.tensor(mom_depot, dtype=torch.int64).unsqueeze(1),
    }

class NVSPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None, mom_vehicle_size=2, sub_vehicle_size=2, designated_driver=False,
                 variable_instance=False, compact=False):
        super(NVSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]
        else:
            self.data = generate_nvsp_data_robustness(
                num_samples, size, mom_vehicle_size, sub_vehicle_size, return_batched=compact)
        self.batched = isinstance(self.data, dict)
        self.size = len(self.data['sor_loc']) if self.batched else len(self.data)


    def __len__(self):
        return self.size


    def __getitem__(self, idx):
        if self.batched:
            device = self.data['sor_loc'].device
            if device.type == 'cuda' and isinstance(idx, (list, torch.Tensor)):
                idx = torch.as_tensor(idx, device=device)
            return {key: value[idx] for key, value in self.data.items()}
        return self.data[idx]



