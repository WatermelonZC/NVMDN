"""HA request encoder adapted to the seven NVSP request features."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class InputEmbedding(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.projection = nn.Linear(7, embed_dim)

    def forward(self, tasks_input):
        if tasks_input.size(-1) != 7:
            raise ValueError('HA-DRL requires [pickup xy, delivery xy, deadline, tau_h, tau_p]')
        pickup, delivery = tasks_input[..., :2], tasks_input[..., 2:4]
        times = tasks_input[..., 4:7]
        p = torch.cat((pickup, delivery, times), -1)
        d = torch.cat((delivery, torch.zeros_like(pickup), times), -1)
        return self.projection(torch.cat((p, d), 1))


class HeterogeneousAttentionLayer(nn.Module):
    def __init__(self, embed_dim=128, n_heads=8):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads, self.head_dim = n_heads, embed_dim // n_heads
        for name in ('K', 'V', 'Q_self', 'Q_pd', 'Q_pP', 'Q_pD', 'Q_dp', 'Q_dP', 'Q_dD', 'O'):
            setattr(self, 'W_'+name, nn.Linear(embed_dim, embed_dim, bias=False))

    def forward(self, h, p_mask, d_mask, pairing_matrix):
        batch, nodes, width = h.shape
        def heads(value):
            return value.view(batch, nodes, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = heads(self.W_K(h)), heads(self.W_V(h))

        def attention(projection, allowed=None):
            q = heads(projection(h))
            empty = None
            if allowed is not None:
                allowed = allowed[:, None].bool()
                empty = ~allowed.any(-1, keepdim=True)
                # No semantic branch is allowed to attend to an all-masked row.
                # A temporary unmasked row avoids NaNs; its result is zeroed.
                allowed = allowed | empty
            if getattr(self, 'use_sdpa', False):
                result = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.)
            else:
                scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
                if allowed is not None:
                    scores = scores.masked_fill(~allowed, float('-inf'))
                result = scores.softmax(-1) @ v
            return result if empty is None else result.masked_fill(empty, 0.)

        # These masks restrict KEY columns. Query rows are gated separately.
        p_keys = p_mask[:, None, :].expand(-1, nodes, -1)
        d_keys = d_mask[:, None, :].expand(-1, nodes, -1)
        pq, dq = p_mask[:, None, :, None], d_mask[:, None, :, None]
        branches = (
            attention(self.W_Q_self),
            attention(self.W_Q_pd, pairing_matrix) * pq,
            attention(self.W_Q_pP, p_keys) * pq,
            attention(self.W_Q_pD, d_keys) * pq,
            attention(self.W_Q_dp, pairing_matrix.transpose(1, 2)) * dq,
            attention(self.W_Q_dP, p_keys) * dq,
            attention(self.W_Q_dD, d_keys) * dq,
        )
        mixed = torch.stack(branches).mean(0)
        return self.W_O(mixed.transpose(1, 2).contiguous().view(batch, nodes, width))


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim=128, n_heads=8, ff_dim=512):
        super().__init__()
        self.attention = HeterogeneousAttentionLayer(embed_dim, n_heads)
        self.norm1, self.norm2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim))

    def forward(self, h, p_mask, d_mask, pairing_matrix):
        h = self.norm1(h + self.attention(h, p_mask, d_mask, pairing_matrix))
        return self.norm2(h + self.feed_forward(h))


class HAEncoder(nn.Module):

    def __init__(self, n_layers=3, d_model=128, n_heads=8, ff_dim=512, prompt_dim=5):
        super().__init__()
        self.embedding = InputEmbedding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, ff_dim) for _ in range(n_layers)])
        self.prompt_projector = nn.Linear(prompt_dim, d_model)
        self.depot_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        self.merge_layer = nn.Linear(d_model * 2, d_model)

    def forward(self, x, prompt):
        batch, tasks, _ = x.shape
        nodes = 2 * tasks + 1
        x = self.embedding(x)
        if prompt is not None:
            x = x + self.prompt_projector(prompt)[:, None]
        h = torch.cat((x, self.depot_embedding.expand(batch, -1, -1)), 1)
        indices = torch.arange(nodes, device=h.device)
        p_mask = (indices < tasks)[None].expand(batch, -1)
        d_mask = ((indices >= tasks) & (indices < 2*tasks))[None].expand(batch, -1)
        pairing = ((indices[:, None] - indices[None, :]).abs() == tasks)
        pairing = pairing & (indices[:, None] < 2*tasks) & (indices[None, :] < 2*tasks)
        pairing = pairing[None].expand(batch, -1, -1)
        for layer in self.layers:
            h = layer(h, p_mask, d_mask, pairing)
        requests = h[:, :tasks] + h[:, tasks:2*tasks]
        depot = h[:, -1:]
        return requests, torch.cat((requests, depot), 1).mean(1), depot
