"""Functional, batch-vectorized NVSP transition for fixed-length training.

All instances advance on the GPU. Only the outer sequence of policy decisions
is sequential, because each decision depends on the preceding vehicle state.
The public NVSPState remains available for beam search and route inspection.
"""
from typing import NamedTuple

import torch

from problems.nvsp.state import tau_a, tau_d, tau_p, v
from utils.data_utils import normalize


class VectorizedNVSPState(NamedTuple):
    sor_loc: torch.Tensor
    tar_loc: torch.Tensor
    ids: torch.Tensor
    vehicle_pos: torch.Tensor
    last_task_indices: torch.Tensor
    vehicle_time: torch.Tensor
    T_f_s: torch.Tensor
    tau_h: torch.Tensor
    travel_cost: torch.Tensor
    mask: torch.Tensor
    leg_distance: torch.Tensor
    sub_vehicle_size: int
    mom_vehicle_size: int
    order_size: int
    sequence: list

    @staticmethod
    def initialize(data, sub_size, mom_size):
        pickup, delivery = data['sor_loc'], data['tar_loc']
        batch, orders, _ = pickup.shape
        device = pickup.device
        vehicles = sub_size + mom_size
        return VectorizedNVSPState(
            pickup, torch.cat((delivery, delivery.new_zeros(batch, 1, 2)), dim=1),
            torch.arange(batch, device=device)[:, None],
            torch.full((batch, vehicles, 1), orders, dtype=torch.long, device=device),
            torch.full((batch, 1), orders, dtype=torch.long, device=device),
            pickup.new_zeros(batch, vehicles), pickup.new_zeros(batch, orders),
            data['tau_h'], pickup.new_zeros(batch),
            torch.zeros(batch, 1, orders, dtype=torch.bool, device=device),
            (delivery - pickup).abs().sum(-1), sub_size, mom_size, orders, [])

    def get_current_node(self):
        return self.vehicle_pos, normalize(self.vehicle_time).unsqueeze(-1), self.last_task_indices

    def get_task_mask(self):
        return self.mask

    def get_sub_vehicle_mask(self):
        return self.mask.new_zeros(self.mask.size(0), 1, self.sub_vehicle_size)

    def get_mom_vehicle_mask(self):
        return self.mask.new_zeros(self.mask.size(0), 1, self.mom_vehicle_size)

    def update(self, task, sub, mom):
        idx = self.ids[:, 0]
        sub_loc = self.tar_loc[idx, self.vehicle_pos[idx, sub, 0]]
        mom_loc = self.tar_loc[idx, self.vehicle_pos[idx, mom, 0]]
        pickup = self.sor_loc[idx, task]
        dist_join = (sub_loc - mom_loc).abs().sum(-1)
        dist_pickup = (pickup - sub_loc).abs().sum(-1)
        dist_delivery = self.leg_distance[idx, task]
        departure = torch.maximum(self.vehicle_time[idx, sub],
                                  self.vehicle_time[idx, mom] + dist_join / v)
        arrival_pickup = departure + dist_pickup / v
        finish_pickup = arrival_pickup + tau_p
        arrival_delivery = finish_pickup + dist_delivery / v
        start_delivery = arrival_delivery + tau_a
        finish_mom = start_delivery + tau_d
        finish_sub = finish_mom + self.tau_h[idx, task, 0]
        selected = torch.stack((sub, mom), dim=1)
        times = self.vehicle_time.scatter(1, selected, torch.stack((finish_sub, finish_mom), dim=1))
        positions = self.vehicle_pos.scatter(1, selected[:, :, None], task[:, None, None].expand(-1, 2, 1))
        completion = self.T_f_s.scatter(1, task[:, None], finish_sub[:, None])
        mask = self.mask.scatter(2, task[:, None, None], True)
        distance = self.travel_cost + (dist_join + dist_pickup + dist_delivery)
        # Trajectories are already collected by the decoder. Do not retain a
        # second Python history in every environment state / compiler input.
        return VectorizedNVSPState(self.sor_loc, self.tar_loc, self.ids, positions,
                            task[:, None], times, completion, self.tau_h,
                            distance, mask, self.leg_distance, self.sub_vehicle_size,
                            self.mom_vehicle_size, self.order_size, [])

    def get_costs(self, data, pi=None):
        deadline = data['designated_time']
        if deadline.ndim != self.tar_loc.ndim:
            deadline = deadline.repeat(1, self.order_size)
        else:
            deadline = deadline.squeeze(-1)
        delay = (self.T_f_s - deadline).clamp(min=0).sum(dim=1)
        return (self.travel_cost.detach(), delay.detach()), None
