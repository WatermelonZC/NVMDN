"""Separate task/robot self-attention followed by dynamic cross-attention."""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from nets.graph_encoder import MultiHeadAttention, SkipConnection
from utils.data_utils import MAX_VALUES


def task_features(x, data):
    pickup, delivery = data['sor_loc'].float(), data['tar_loc'].float()
    speed = data['v'].float().reshape(-1, 1, 1).clamp_min(1e-6)
    travel = (delivery - pickup).abs().sum(-1, keepdim=True) / speed
    earliest = (pickup.abs().sum(-1, keepdim=True) / speed + travel
                + data['params'].float().sum(-1)[:, None, None] + data['tau_h'].float())
    slack = data['designated_time'].float() - earliest
    return torch.cat((x, (delivery - pickup) / MAX_VALUES['location'],
                      travel / MAX_VALUES['time'], slack / MAX_VALUES['time']), -1)


class CrossAttentionLayer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(width, width, bias=False)
        self.key_value = nn.Linear(width, 2 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(),
                                nn.Linear(2 * width, width))

    def project_query(self, tasks):
        b, n, width = tasks.shape
        return self.query(tasks).reshape(b, n, self.heads, -1).transpose(1, 2)

    def forward(self, tasks, robots, projected_query=None):
        b, n, width = tasks.shape
        q = self.project_query(tasks) if projected_query is None else projected_query
        k, v = self.key_value(robots).reshape(b, robots.size(1), 2, self.heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        tasks = self.norm1(tasks + self.output(attended.transpose(1, 2).reshape(b, n, width)))
        return self.norm2(tasks + self.ff(tasks))


class TaskRobotCrossEncoder(nn.Module):
    architecture_revision = 'figure_ln_cross_res_ffn_v4'

    def __init__(self, input_dim=7, embed_dim=128, num_layers=1, num_heads=8,
                 daughters=8, mothers=4):
        super().__init__()
        self.daughters, self.mothers = daughters, mothers
        self.task_embedding = nn.Sequential(nn.Linear(input_dim + 4, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.robot_embedding = nn.Sequential(nn.Linear(7, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.role_embedding = nn.Parameter(torch.empty(2, embed_dim).normal_(std=.02))
        # Keep AM's head projections; normalize each token independently.
        # The restored residual FFN belongs to the cross-attention block.
        def attention_branch():
            return nn.Sequential(*[nn.Sequential(
                SkipConnection(MultiHeadAttention(num_heads, embed_dim, embed_dim)),
                nn.LayerNorm(embed_dim)) for _ in range(num_layers)])
        self.task_layers = attention_branch()
        self.robot_layers = attention_branch()
        self.cross_attention = CrossAttentionLayer(embed_dim, num_heads)

    @contextmanager
    def preserve_recomputed_robot_statistics(self):
        # LayerNorm has no running statistics. Keep the checkpoint context API
        # compatible with the dynamic-step caller and future stateful modules.
        buffers = [(buffer, buffer.clone()) for buffer in self.robot_layers.buffers()]
        try:
            yield
        finally:
            with torch.no_grad():
                for buffer, saved in buffers:
                    buffer.copy_(saved)

    def forward(self, x, prompt=None, data=None):
        if data is None:
            raise ValueError('Raw physical task data is required')
        tasks = self.task_layers(self.task_embedding(task_features(x, data)))
        # Only static task self-attention is cached. Robot and cross-attention
        # are evaluated inside every decision, using its current fleet state.
        return tasks, tasks.mean(1), tasks.new_zeros(tasks.size(0), 1, tasks.size(-1))

    def encode_robots(self, state, data):
        positions = state.tar_loc.gather(1, state.vehicle_pos.squeeze(-1)[..., None].expand(-1, -1, 2))
        count = self.daughters + self.mothers
        speed = data['v'].float().reshape(-1, 1, 1).expand(-1, count, -1) / 2.
        handling = data['params'].float()[:, None, :3].expand(-1, count, -1) / 60.
        features = torch.cat((positions / MAX_VALUES['location'],
                              state.vehicle_time[..., None] / MAX_VALUES['time'], speed, handling), -1)
        roles = torch.cat((self.role_embedding[:1].expand(self.daughters, -1),
                           self.role_embedding[1:].expand(self.mothers, -1)), 0)
        embedded = self.robot_embedding(features)
        return self.robot_layers(embedded + roles[None].to(embedded.dtype))

