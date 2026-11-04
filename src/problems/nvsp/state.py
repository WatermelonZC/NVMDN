import logging

import torch
from typing import NamedTuple, Tuple

from torch import Tensor
from torch.amp import autocast

from utils.ModelData import Start_T, End_T, isAmp
# from problems.nvsp.problem import NVSP
from utils.boolmask import mask_long2bool, mask_long_scatter
from utils.data_utils import normalize, write_tensor_to_excel, read_excel_to_tensor, get_topological_sort

tau_a = 8.0
tau_d = 8.0
tau_p = 30.0
# tau_h = 0.4
v = 1.2
omega = 0.1


# logger = logging.getLogger(__name__)

@torch.jit.script
def _calculate_new_state_jit(
        # --- 输入所有需要的状态张量 ---
        mask: torch.Tensor,
        T_f_s: torch.Tensor,  # <--- 新增输入
        T_f_c: torch.Tensor,  # <--- 新增输入 (可选)
        vehicle_pos: torch.Tensor,
        vehicle_time: torch.Tensor,
        sor_loc: torch.Tensor,
        tar_loc: torch.Tensor,
        tau_h: torch.Tensor,
        travel_cost: torch.Tensor,
        ids: torch.Tensor,
        # --- 输入动作 ---
        task_selected: torch.Tensor,
        sub_vehicle_selected: torch.Tensor,
        mom_vehicle_selected: torch.Tensor,
        tau_a: float,
        tau_d: float,
        tau_p: float,
        v: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    JIT优化的纯计算函数。
    接收旧的状态张量，返回一个包含所有完整新状态张量的元组。
    """
    sub_vehicles = sub_vehicle_selected
    mom_vehicles = mom_vehicle_selected
    order_indices = task_selected

    idx = ids.squeeze(dim=1)

    # --- 克隆输入张量以进行修改，保持函数无副作用 ---
    new_mask = mask.clone()
    new_vehicle_time = vehicle_time.clone()
    new_vehicle_pos = vehicle_pos.clone()
    new_T_f_s = T_f_s.clone()
    new_T_f_c = T_f_c.clone()

    # --- 更新 Mask ---
    new_mask[ids, sub_vehicles.unsqueeze(-1), order_indices.unsqueeze(-1)] = True
    new_mask[ids, mom_vehicles.unsqueeze(-1), order_indices.unsqueeze(-1)] = True

    # --- 核心计算逻辑 (与之前完全相同) ---
    pre_sub_loc = tar_loc[idx, vehicle_pos[idx, sub_vehicles].squeeze()]
    pre_mom_loc = tar_loc[idx, vehicle_pos[idx, mom_vehicles].squeeze()]
    pickup_loc = sor_loc[idx, order_indices]
    delivery_loc = tar_loc[idx, order_indices]
    tau_h_t = tau_h[idx, order_indices].squeeze(-1)

    dist_mom_to_sub = torch.sum(torch.abs(pre_sub_loc - pre_mom_loc), dim=1)
    time_mom_to_sub = dist_mom_to_sub / v
    rendezvous_arrival_time = vehicle_time[idx, mom_vehicles] + time_mom_to_sub
    departure_time_from_rendezvous = torch.max(vehicle_time[idx, sub_vehicles], rendezvous_arrival_time)

    dist_rendezvous_to_pickup = torch.sum(torch.abs(pickup_loc - pre_sub_loc), dim=1)
    time_rendezvous_to_pickup = dist_rendezvous_to_pickup / v
    arrival_time_at_pickup = departure_time_from_rendezvous + time_rendezvous_to_pickup
    finish_time_at_pickup = arrival_time_at_pickup + tau_p

    dist_pickup_to_delivery = torch.sum(torch.abs(delivery_loc - pickup_loc), dim=1)
    time_pickup_to_delivery = dist_pickup_to_delivery / v
    arrival_time_at_delivery = finish_time_at_pickup + time_pickup_to_delivery

    start_time_at_delivery = arrival_time_at_delivery + tau_a
    final_finish_time_sub = start_time_at_delivery + tau_d + tau_h_t
    final_finish_time_mom = start_time_at_delivery + tau_d

    # --- 用计算结果更新新的状态张量 ---
    new_T_f_s[idx, order_indices] = final_finish_time_sub
    new_T_f_c[idx, order_indices] = final_finish_time_mom

    new_vehicle_time[idx, sub_vehicles] = final_finish_time_sub
    new_vehicle_time[idx, mom_vehicles] = final_finish_time_mom

    new_vehicle_pos[idx, sub_vehicles] = order_indices.unsqueeze(-1)
    new_vehicle_pos[idx, mom_vehicles] = order_indices.unsqueeze(-1)

    total_dist_this_step = dist_mom_to_sub + dist_rendezvous_to_pickup + dist_pickup_to_delivery
    new_travel_cost = travel_cost + total_dist_this_step

    # Start_T[task_selected.item()] = vehicle_time[idx, mom_vehicles].item()
    # End_T[task_selected.item()] = final_finish_time_sub.item()
    # print(total_dist_this_step)
    # --- 返回所有更新后的张量组成的元组 ---
    return (
        new_mask, new_T_f_s, new_T_f_c, new_vehicle_time, new_vehicle_pos,
        new_travel_cost, order_indices.unsqueeze(-1)
    )


class NVSPState(NamedTuple):
    # Fixed input
    # 任务起始点
    sor_loc: torch.Tensor
    # 任务终止点
    tar_loc: torch.Tensor
    # 时间速度等参数
    params: torch.Tensor
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    i: torch.Tensor  # Keeps track of step
    vehicle_size: int
    # lengths: torch.Tensor
    sequence: list
    mask: torch.Tensor
    vehicle_pos: torch.Tensor
    # 上一次任务索引
    last_task_indices: torch.Tensor
    sub_vehicle_size: int
    mom_vehicle_size: int
    order_size: int
    vehicle_time: torch.Tensor
    ksi_s: torch.Tensor
    ksi_c: torch.Tensor
    T_s: torch.Tensor
    T_f_s: torch.Tensor
    T_f_c: torch.Tensor
    ksi_max: torch.Tensor
    tau_h: torch.Tensor
    # cost
    travel_cost: torch.Tensor
    time_cost: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
        )

    @staticmethod
    def initialize(input):
        sor_loc = input['sor_loc']
        tar_loc = torch.cat((input['tar_loc'], torch.zeros(input['tar_loc'].size(0), 1, 2, dtype=torch.float32, device=input['tar_loc'].device)), dim=1)
        sub_vehicle_size = input['sub_size'][0].item() if isinstance(input['sub_size'], Tensor) else input['sub_size']
        mom_vehicle_size = input['mom_size'][0].item() if isinstance(input['mom_size'], Tensor) else input['mom_size']
        params = input['params']
        tau_h = input['tau_h']
        vehicle_size = sub_vehicle_size + mom_vehicle_size
        batch_size = tar_loc.size(0)
        n_loc = sor_loc.size(1)
        device = tar_loc.device
        # all_loc = torch.cat((sor_loc, tar_loc, torch.zeros(batch_size, 1, 2, dtype=torch.float32, device=device)), dim=1)
        # lock_matrix = torch.zeros(batch_size, n_loc, n_loc, dtype=torch.bool, device=device)
        last_task_indices = torch.full((batch_size, 1), -1, dtype=torch.long, device=device, requires_grad=False)
        ksi_s = torch.full((batch_size, n_loc), -1, dtype=torch.float, device=device, requires_grad=False)
        ksi_c = torch.full((batch_size, n_loc), -1, dtype=torch.float, device=device, requires_grad=False)
        T_s = torch.full((batch_size, n_loc), 0, dtype=torch.float, device=device, requires_grad=False)
        T_f_s = torch.full((batch_size, n_loc), 0, dtype=torch.float, device=device, requires_grad=False)
        T_f_c = torch.full((batch_size, n_loc), 0, dtype=torch.float, device=device, requires_grad=False)
        ksi_max = torch.full((batch_size, n_loc), 0, dtype=torch.float, device=device, requires_grad=False)
        vehicle_time = torch.full((batch_size, vehicle_size), 0, dtype=torch.float, device=device, requires_grad=False)
        last_task_indices.fill_(n_loc)
        mask = torch.zeros(batch_size, vehicle_size, n_loc, dtype=torch.bool, device=device, requires_grad=False)
        vehicle_pos = torch.empty(batch_size, vehicle_size, 1, dtype=torch.int64, device=device, requires_grad=False)
        vehicle_pos.fill_(n_loc)
        travel_cost = torch.zeros(batch_size, device=device, requires_grad=False)
        time_cost = torch.zeros(batch_size, device=device, requires_grad=False)
        # assert vehicle_size < 2 * sub_vehicle_size
        return NVSPState(
            sor_loc=sor_loc,
            tar_loc=tar_loc,
            params=params,
            ids=torch.arange(batch_size, dtype=torch.long, device=device)[:, None],  # (batch_size, 1),  # Add steps dimension
            # lengths=torch.zeros(batch_size, 1, device=loc.device),
            i=torch.zeros(batch_size, 1, dtype=torch.int64, device=device),  # Vector with length num_steps
            vehicle_size=vehicle_size,
            sequence=[],
            # lock_matrix=lock_matrix,
            mask=mask,
            vehicle_pos=vehicle_pos,
            last_task_indices=last_task_indices,
            mom_vehicle_size=mom_vehicle_size,
            sub_vehicle_size=sub_vehicle_size,
            vehicle_time=vehicle_time,
            ksi_max=ksi_max,
            ksi_c=ksi_c,
            ksi_s=ksi_s,
            T_s=T_s,
            T_f_s=T_f_s,
            T_f_c=T_f_c,
            travel_cost=travel_cost,
            time_cost=time_cost,
            order_size=sor_loc.size(1),
            tau_h=tau_h,
        )

    def get_sub_state(self):
        return self.vehicle_pos[:, :self.sub_vehicle_size, :], (self.vehicle_time[:, :self.sub_vehicle_size]).unsqueeze(-1)

    def get_mom_state(self):
        return self.vehicle_pos[:, self.sub_vehicle_size:, :], (self.vehicle_time[:, self.sub_vehicle_size:]).unsqueeze(-1)

    def get_costs(self, dataset, pi):
        # from problems.nvsp.problem import NVSP
        # sub_vehicle_size = dataset['sub_size'][0].item()
        # vehicle_size = self.vehicle_size
        # sequences = NVSP.process_sequences_torch(pi, self.vehicle_size)
        # topologic_sort = get_topological_sort(sequences[0])
        # sort_list = torch.tensor(topologic_sort, device=dataset['designated_time'].device).unsqueeze(0)
        # cost = NVSP.calculate_cost_with_topology(sequences,  torch.cat((dataset['sor_loc'], dataset['tar_loc']), 1), sort_list, dataset["designated_time"] * 10, dataset["params"], 8)
        # transform_routes_with_empty(sequences, self.sub_vehicle_size)
        # cost = NVSP.calculate_sequence_cost_2(sequences, torch.cat((dataset['sor_loc'], dataset['tar_loc']), 1))
        # dist_cost = NVSP.calculate_sequence_cost_with_pairing_tensor(sequences, torch.cat((dataset['sor_loc'], dataset['tar_loc']), 1), sub_vehicle_size)
        if isAmp:
            # with autocast('cuda', enabled=False):
            time_limit = dataset['designated_time']  # 缩放
            if time_limit.ndim != self.tar_loc.ndim:
                time_limit = time_limit.repeat(1, self.T_f_s.size(1))
            else:
                time_limit = time_limit.squeeze(-1)
            # time_limit[..., :half_T] = 0
            # self.T_f_s[..., :half_T] = 0
            cost_delay = torch.sum((self.T_f_s - time_limit).clamp(min=0), dim=1)
        else:
            time_limit = dataset['designated_time']
            if time_limit.ndim != self.tar_loc.ndim:
                time_limit = time_limit.repeat(1, self.T_f_s.size(1))
            else:
                time_limit = time_limit.squeeze(-1)
            cost_delay = torch.sum((self.T_f_s - time_limit).clamp(min=0), dim=1)
        # cost_delay = torch.min(dist_cost, cost_delay)  # 假设延误代价不能大于路径代价，否则值回异常大不太正常
        # totol_cost = omega * self.travel_cost + (1 - omega) * cost_delay
        # totol_cost2 = NVSP.calculate_sequence_cost_with_time(sequences, torch.cat((dataset['sor_loc'], dataset['tar_loc']), 1), dataset['designated_time'] * half_T, pi[:, :, 0], dataset['params'], sub_vehicle_size)
        # print(totol_cost2 - totol_cost)

        return (self.travel_cost.detach(), cost_delay.detach()), None

    # 这是您类中的方法
    def update(self, task_selected, sub_vehicle_selected, mom_vehicle_selected):
        if isAmp:
            # with autocast('cuda', enabled=False):
                # 1. 调用JIT函数，传入所有需要的当前状态
            (
                new_mask, new_T_f_s, new_T_f_c, new_vehicle_time, new_vehicle_pos,
                new_travel_cost, new_last_task_indices
            ) = _calculate_new_state_jit(
                self.mask, self.T_f_s, self.T_f_c, self.vehicle_pos, self.vehicle_time,
                self.sor_loc, self.tar_loc, self.tau_h, self.travel_cost, self.ids,
                task_selected, sub_vehicle_selected, mom_vehicle_selected, tau_a, tau_d, tau_p, v
            )
        else:
            (
                new_mask, new_T_f_s, new_T_f_c, new_vehicle_time, new_vehicle_pos,
                new_travel_cost, new_last_task_indices
            ) = _calculate_new_state_jit(
                self.mask, self.T_f_s, self.T_f_c, self.vehicle_pos, self.vehicle_time,
                self.sor_loc, self.tar_loc, self.tau_h, self.travel_cost, self.ids,
                task_selected, sub_vehicle_selected, mom_vehicle_selected, tau_a, tau_d, tau_p, v
            )

        # 2. 更新 sequence (这是一个Python列表，在JIT外部处理)
        new_sequence = self.sequence + [torch.stack([task_selected, sub_vehicle_selected, mom_vehicle_selected], dim=1)]

        # 3. 使用 _replace() 创建并返回一个新的实例
        return NVSPState(
            sor_loc=self.sor_loc,
            tar_loc=self.tar_loc,
            params=self.params,
            ids=self.ids,
            vehicle_size=self.vehicle_size,
            sub_vehicle_size=self.sub_vehicle_size,
            mom_vehicle_size=self.mom_vehicle_size,
            order_size=self.order_size,
            ksi_s=self.ksi_s,
            ksi_c=self.ksi_c,
            T_s=self.T_s,
            ksi_max=self.ksi_max,
            tau_h=self.tau_h,
            time_cost=self.time_cost,
            i=self.i + 1,
            mask=new_mask,
            T_f_s=new_T_f_s,
            T_f_c=new_T_f_c,
            vehicle_pos=new_vehicle_pos,
            vehicle_time=new_vehicle_time,
            travel_cost=new_travel_cost,
            last_task_indices=new_last_task_indices,
            sequence=new_sequence
        )


    # def update(self, task_selected, sub_vehicle_selected, mom_vehicle_selected):
    #     """
    #     状态更新函数，精确实现“母车先接子车，再共同执行任务”的逻辑。
    #     """
    #     sub_vehicles = sub_vehicle_selected
    #     mom_vehicles = mom_vehicle_selected
    #     order_indices = task_selected
    #
    #     idx = self.ids.squeeze(dim=1)
    #     n_orders = self.order_size
    #
    #     # --- 1. 更新 Mask (与之前版本相同) ---
    #     self.mask[self.ids, sub_vehicles.unsqueeze(-1), order_indices.unsqueeze(-1)] = True
    #     self.mask[self.ids, mom_vehicles.unsqueeze(-1), order_indices.unsqueeze(-1)] = True
    #
    #     # --- 2. 获取所有相关位置的坐标 (与之前版本相同) ---
    #     pre_sub_loc = self.tar_loc[idx, self.vehicle_pos[idx, sub_vehicles].squeeze()]
    #     pre_mom_loc = self.tar_loc[idx, self.vehicle_pos[idx, mom_vehicles].squeeze()]
    #
    #     pickup_loc = self.sor_loc[idx, order_indices]
    #     delivery_loc = self.tar_loc[idx, order_indices]
    #
    #     # --- 3. 模拟完整的任务流程 (全新逻辑) ---
    #     # 解包参数
    #     tau_a, tau_d, tau_p, tau_h, v = self.params[:, 0], self.params[:, 1], self.params[:, 2], self.params[:, 3], self.params[:, 4]
    #
    #     # == 阶段一: 集合 (Rendezvous) 在子车的旧位置 ==
    #     dist_mom_to_sub = torch.sum(torch.abs(pre_sub_loc - pre_mom_loc), dim=1)
    #     time_mom_to_sub = dist_mom_to_sub / v
    #
    #     # 计算母车到达子车位置的时间点
    #     rendezvous_arrival_time = self.vehicle_time[idx, mom_vehicles] + time_mom_to_sub
    #
    #     # 同步点: 团队的出发时间，取决于“子车何时空闲”和“母车何时到达”，取最晚的那个
    #     departure_time_from_rendezvous = torch.max(self.vehicle_time[idx, sub_vehicles], rendezvous_arrival_time)
    #
    #     # == 阶段二: 团队一起前往取货点 ==
    #     dist_rendezvous_to_pickup = torch.sum(torch.abs(pickup_loc - pre_sub_loc), dim=1)
    #     time_rendezvous_to_pickup = dist_rendezvous_to_pickup / v
    #
    #     arrival_time_at_pickup = departure_time_from_rendezvous + time_rendezvous_to_pickup
    #
    #     # == 阶段三: 在取货点进行服务 ==
    #     # 假设取货服务在到达后立即开始
    #     finish_time_at_pickup = arrival_time_at_pickup + tau_p
    #
    #     # == 阶段四: 团队一起前往送货点 ==
    #     dist_pickup_to_delivery = torch.sum(torch.abs(delivery_loc - pickup_loc), dim=1)
    #     time_pickup_to_delivery = dist_pickup_to_delivery / v
    #
    #     arrival_time_at_delivery = finish_time_at_pickup + time_pickup_to_delivery
    #
    #     # == 阶段五: 在送货点进行服务，计算最终完成时间 ==
    #     start_time_at_delivery = arrival_time_at_delivery + tau_a
    #     final_finish_time_sub = start_time_at_delivery + tau_d + tau_h
    #     final_finish_time_mom = start_time_at_delivery + tau_d
    #
    #     # --- 4. 更新车辆的最终状态 ---
    #     self.vehicle_time[idx, sub_vehicles] = final_finish_time_sub
    #     self.vehicle_time[idx, mom_vehicles] = final_finish_time_mom
    #     self.vehicle_pos[idx, sub_vehicles] = order_indices.unsqueeze(-1)
    #     self.vehicle_pos[idx, mom_vehicles] = order_indices.unsqueeze(-1)
    #
    #     # --- 5. 更新统计数据并返回新状态 ---
    #     # 总行程包含三段：母车->子车，团队->取货点，团队->送货点
    #     self.T_f_s[idx, order_indices] = final_finish_time_sub
    #     total_dist_this_step = dist_mom_to_sub + dist_rendezvous_to_pickup + dist_pickup_to_delivery
    #     dist = self.travel_cost + total_dist_this_step
    #     self.sequence.append(torch.stack([order_indices, sub_vehicle_selected, mom_vehicle_selected], dim=1))
    #     update_i = self.i[:, :] + 1
    #     return self._replace(i=update_i, last_task_indices=order_indices.unsqueeze(-1), travel_cost=dist)


    def beam_update(self):
        pass

    def all_finished(self):
        # Exactly n steps
        return self.i[0].item() >= self.order_size

    def get_current_node(self):
        return self.vehicle_pos, normalize(self.vehicle_time).unsqueeze(-1), self.last_task_indices

    def get_mom_vehicle_mask(self):
        """
        【新版逻辑 - 选项A】
        在新模型下，任务是原子性的，子车完成任务后即为可用。
        因此，不再需要基于上一步任务类型的特殊mask。
        此函数返回一个全为False的mask，表示所有子车均可选。
        具体的可用性由解码器根据 vehicle_time 判断。
        """
        batch_size = self.mask.size(0)
        device = self.mask.device

        # 始终返回一个没有车辆被禁用的mask
        return torch.zeros(batch_size, 1, self.mom_vehicle_size, dtype=torch.bool, device=device)

    def get_sub_vehicle_mask(self):
        """
        【新版逻辑 - 选项A】
        在新模型下，任务是原子性的，子车完成任务后即为可用。
        因此，不再需要基于上一步任务类型的特殊mask。
        此函数返回一个全为False的mask，表示所有子车均可选。
        具体的可用性由解码器根据 vehicle_time 判断。
        """
        batch_size = self.mask.size(0)
        sub_size = self.sub_vehicle_size
        device = self.mask.device

        # 始终返回一个没有车辆被禁用的mask
        return torch.zeros(batch_size, 1, sub_size, dtype=torch.bool, device=device)

    def get_task_mask(self):
        """
        【新版逻辑】
        生成可执行任务的掩码。
        在新模型下，任务是原子性的(P-D一体)。因此，mask的逻辑非常简单：
        任何已经被完成的订单/任务，都应该被禁用。

        此函数依赖于 self.mask 的正确更新。
        self.mask 的维度应为 (batch, num_vehicles, num_orders)。
        """
        # self.mask 记录了 (B, V, N) 的完成状态
        # 我们需要一个 (B, 1, N) 的mask，表示哪些任务被任何车辆完成过
        # .any(dim=1) 会在 vehicle 维度上进行“或”操作。
        # 只要有一个车辆完成了这个任务，结果就为 True。
        # keepdim=True 保持维度为 (B, 1, N) 以匹配输出格式。

        task_done_mask = self.mask.any(dim=1, keepdim=True)

        return task_done_mask

    def get_mask(self):
        temp_mask = self.mask.contiguous()
        for vehicle in range(self.vehicle_size):
            for task in range(self.sor_loc.size(1)):
                # 任务已被访问，跳过
                if self.mask[:, vehicle, task].all():
                    continue

                # 当前车辆尝试执行任务 task
                prev_task = self.prev_acs[:, vehicle, 0]  # 获取该车上一次执行的任务

                # **如果该车还没有执行任何任务 (-1)，则不受 lock_matrix 约束**
                no_prev_task = prev_task == -1
                if no_prev_task.any():
                    # self.mask[no_prev_task, vehicle, task] = False  # 该车辆可以自由选择任务
                    continue  # 直接跳过后续逻辑

                temp_lock_matrix = self.lock_matrix.clone()
                temp_lock_matrix[self.ids, prev_task, task] = True  # 记录新依赖

                # 计算传递闭包
                temp_lock_matrix = self.transitive_closure_val(temp_lock_matrix)

                # **死锁检测**: 若对角线有 True，则该任务不可选
                deadlock_detected = torch.diagonal(temp_lock_matrix, dim1=1, dim2=2).any(dim=1)

                # 更新 mask
                # self.mask[self.ids[:, 0], vehicle, task] = deadlock_detected  # 不能选的任务设为 false
                temp_mask[deadlock_detected, vehicle, task] = True
        return temp_mask

    def get_nn_current(self, k=None):
        assert False, "Currently not implemented, look into which neighbours to use in step 0?"
        # Note: if this is called in step 0, it will have k nearest neighbours to node 0, which may not be desired
        # so it is probably better to use k = None in the first iteration
        if k is None:
            k = self.loc.size(-2)
        k = min(k, self.loc.size(-2) - self.i.item())  # Number of remaining
        return (
                self.dist[
                    self.ids,
                    self.prev_a
                ] +
                self.visited.float() * 1e6
        ).topk(k, dim=-1, largest=False)[1]

    def construct_solutions(self, actions):
        return actions

    def get_solution(self):
        sequence_tensor = torch.stack(self.sequence, dim=0)  # (list_length, batch, 1, 2)

        # **获取所有车辆索引**，shape: (batch_size, list_length, 1)
        vehicle_idx = sequence_tensor[:, :, :, 0].squeeze(2).permute(1, 0)  # (batch_size, list_length)

        # **获取所有任务编号**，shape: (batch_size, list_length, 1)
        task_idx = sequence_tensor[:, :, :, 1].squeeze(2).permute(1, 0)  # (batch_size, list_length)

        # **初始化 task_order**
        task_order = torch.full((self.sequence[0].size(0), self.vehicle_size, len(self.sequence)), -1, dtype=torch.long, device=self.loc.device)

        # **使用 scatter_ 填充 task_order**
        task_order.scatter_(1, vehicle_idx.unsqueeze(1).expand(-1, self.vehicle_size, -1), task_idx.unsqueeze(1))

        return task_order

    def getPij(self, i):
        sor_loc = torch.cat((self.sor_loc, torch.zeros(self.sor_loc.size(0), 1, 2, dtype=torch.float32, device=self.sor_loc.device)), dim=1)
        tar_loc = self.tar_loc
        return 0

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
    def detect_deadlock(self, schedule):
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
