import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import math

class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, x, y=None):
        if y is not None:
            return x + self.module(x, y)
        return x + self.module(x)

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, input_dim, embed_dim, val_dim=None, key_dim=None):
        super(MultiHeadAttention, self).__init__()
        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor = 1 / math.sqrt(key_dim)
        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q
        (batch_size, graph_size, input_dim) = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, 'Wrong embedding dimension of input'
        h_rows = h.contiguous().view(-1, input_dim)
        q_rows = q.contiguous().view(-1, input_dim)
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)
        Q = torch.matmul(q_rows, self.W_query).view(shp_q)
        K = torch.matmul(h_rows, self.W_key).view(shp)
        V = torch.matmul(h_rows, self.W_val).view(shp)
        if getattr(self, 'use_sdpa', False):
            allowed = None if mask is None else ~mask.reshape(batch_size, 1, n_query, graph_size)
            heads = F.scaled_dot_product_attention(Q.transpose(0, 1), K.transpose(0, 1), V.transpose(0, 1), attn_mask=allowed, dropout_p=0.0)
            return F.linear(heads.transpose(1, 2).reshape(batch_size, n_query, -1), self.W_out.reshape(-1, self.embed_dim).t())
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf
        attn = torch.softmax(compatibility, dim=-1)
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc
        heads = torch.matmul(attn, V)
        out = torch.mm(heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim), self.W_out.view(-1, self.embed_dim)).view(batch_size, n_query, self.embed_dim)
        return out

class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()
        normalizer_class = {'batch': nn.BatchNorm1d, 'instance': nn.InstanceNorm1d, 'layer': nn.LayerNorm}.get(normalization, None)
        if normalization == 'layer':
            self.normalizer = normalizer_class(embed_dim)
        else:
            self.normalizer = normalizer_class(embed_dim, affine=True)

    def init_parameters(self):
        for (name, param) in self.named_parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        elif isinstance(self.normalizer, nn.LayerNorm):
            return self.normalizer(input)
        else:
            assert self.normalizer is None, 'Unknown normalizer type'
            return input

class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(self, n_heads, embed_dim, feed_forward_hidden=512, normalization='batch'):
        super(MultiHeadAttentionLayer, self).__init__(SkipConnection(MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim)), Normalization(embed_dim, normalization), SkipConnection(nn.Sequential(nn.Linear(embed_dim, feed_forward_hidden), nn.ReLU(), nn.Linear(feed_forward_hidden, embed_dim)) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)), Normalization(embed_dim, normalization))

class VehicleAttentionEncoder(nn.Module):

    def __init__(self, n_heads, embed_dim, n_layers, robot_dim=None, normalization='batch', feed_forward_hidden=512):
        super(VehicleAttentionEncoder, self).__init__()
        self.init_embed = nn.Linear(robot_dim, embed_dim) if robot_dim is not None else None
        self.layers = nn.Sequential(*(MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization) for _ in range(n_layers)))

    def forward(self, x, mask=None):
        assert mask is None, 'Mask input is not used by this encoder.'
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x
        h = self.layers(h)
        return (h, h.mean(dim=1))

class RobotTypeEmbedderWithFilM(nn.Module):

    def __init__(self, embedding_dim, hyper_hidden_dim=512):
        super().__init__()
        self.robot_type_emb = nn.Embedding(paramet_ahasp.ROBOT_TYPE_NUM, embedding_dim)
        robot_dim = 3
        self.norm = nn.LayerNorm(embedding_dim)
        self.hyper = nn.Sequential(nn.Linear(embedding_dim, hyper_hidden_dim), nn.ReLU(), nn.Linear(hyper_hidden_dim, embedding_dim * 2))
        self.robot_init_linear = nn.Linear(robot_dim, embedding_dim, bias=False)

    def forward(self, state):
        batch_size = state.cur_coord.size(0)
        robot_init_info = torch.cat((state.cur_coord, state.cur_time.unsqueeze(-1) / paramet_ahasp.time_norm), -1)
        robot_init_embedding = self.robot_init_linear(robot_init_info)
        single_batch_type_index = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.long, device=robot_init_embedding.device)
        robot_type_index = single_batch_type_index.unsqueeze(0).expand(batch_size, -1)
        type_embed = self.robot_type_emb(robot_type_index)
        gamma_beta = self.hyper(type_embed)
        (gamma, beta) = gamma_beta.chunk(2, dim=-1)
        robot_embedding = gamma * robot_init_embedding + beta
        return self.norm(robot_embedding + robot_init_embedding)
