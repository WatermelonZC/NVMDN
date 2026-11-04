"""Hierarchical request, daughter-vehicle, and mother-vehicle selection."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from nets.graph_encoder import MultiHeadAttentionLayer


class StandardRequestEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, layers, heads):
        super().__init__()
        self.initial_embedding = nn.Linear(input_dim, embed_dim)
        self.layers = nn.Sequential(*[
            MultiHeadAttentionLayer(heads, embed_dim, normalization='batch')
            for _ in range(layers)])
        self.start_maker = nn.Parameter(torch.empty(embed_dim).uniform_(-1, 1))

    def forward(self, x, prompt=None):
        h = self.layers(self.initial_embedding(x))
        return h, h.mean(1), self.start_maker[None, None].expand(x.size(0), 1, -1)


class VehicleStateEncoder(nn.Module):
    def __init__(self, width, heads, independent=False):
        super().__init__()
        self.heads = heads
        self.independent = independent
        if independent:
            self.mlp = nn.Sequential(nn.Linear(width + 1, width), nn.ReLU(), nn.Linear(width, width))
        else:
            self.qkv = nn.Linear(width + 1, 3 * width, bias=False)
            self.out = nn.Linear(width, width, bias=False)

    def forward(self, times, embeddings):
        x = torch.cat((embeddings, times), -1)
        if self.independent:
            return self.mlp(x)
        b, n, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, -1).unbind(2)
        values = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                               v.transpose(1, 2), dropout_p=0.0)
        return self.out(values.transpose(1, 2).reshape(b, n, -1))


def selector(model, context, candidates, projection, mask, projected=None):
    b, n, width = candidates.shape
    heads = model.n_heads
    q = context.reshape(b, 1, heads, -1).transpose(1, 2)
    if projected is None:
        key, value, logit_key = projection(candidates).chunk(3, -1)
        k = key.reshape(b, n, heads, -1).transpose(1, 2)
        v = value.reshape(b, n, heads, -1).transpose(1, 2)
    else:
        k, v, logit_key = projected
    glimpse = F.scaled_dot_product_attention(q, k, v, attn_mask=(~mask)[:, None], dropout_p=0.0)
    glimpse = model.project_out(glimpse.transpose(1, 2).reshape(b, 1, width))
    logits = torch.matmul(glimpse, logit_key.transpose(1, 2)) / math.sqrt(width // heads)
    logits = model.tanh_clipping * torch.tanh(logits)
    logits = logits.masked_fill(mask, -math.inf)
    return torch.log_softmax(logits / model.temp, -1)


def decode_step(model, fixed, state, embeddings, ids, output_mode, fleet=None, nodes=None, selection_noise=None):
    positions, times, previous = state.get_current_node()
    if nodes is None:
        nodes = model.embeddings_with_depot
    if fleet is None:
        last = nodes.gather(1, positions.squeeze(-1).unsqueeze(-1).expand(-1, -1, model.embedding_dim))
        fleet = model.vehicle_state_att(times, last)
    cached = getattr(model, 'cache_request_context', False)
    global_request = fixed.context_node_projected if cached else embeddings.mean(1, keepdim=True)
    prior = nodes.gather(1, previous.reshape(-1, 1, 1).expand(-1, 1, model.embedding_dim))
    prior = torch.where((previous.reshape(-1, 1, 1) == embeddings.size(1)),
                        model.start_maker[None, None], prior)
    context = model.request_context(torch.cat((global_request, prior, fleet.mean(1, keepdim=True)), -1))
    task_mask = state.get_task_mask()
    projected = ((fixed.glimpse_key[:, :, 0].transpose(0, 1),
                  fixed.glimpse_val[:, :, 0].transpose(0, 1), fixed.logit_key[:, 0]) if cached else None)
    task_p = selector(model, context, embeddings, model.project_node_embeddings, task_mask, projected)
    task = ((task_p.exp()[:, 0].float() / selection_noise[0]).argmax(-1)
            if selection_noise is not None else model._select_node(task_p.exp()[:, 0], task_mask[:, 0]))
    chosen = embeddings[ids, task][:, None]
    sub_context = model.daughter_context(torch.cat((global_request, chosen), -1))
    sub_mask = state.get_sub_vehicle_mask()
    sub_p = selector(model, sub_context, fleet[:, :model.sub_vehicle_size], model.project_sub_context, sub_mask)
    sub = ((sub_p.exp()[:, 0].float() / selection_noise[1]).argmax(-1)
           if selection_noise is not None else model._select_node(sub_p.exp()[:, 0], sub_mask[:, 0]))
    selected_vehicle = fleet[ids, sub][:, None]
    mom_context = model.mother_context(torch.cat((global_request, chosen, selected_vehicle), -1))
    mom_mask = state.get_mom_vehicle_mask()
    mom_p = selector(model, mom_context, fleet[:, model.sub_vehicle_size:], model.project_mom_context, mom_mask)
    mom = (((mom_p.exp()[:, 0].float() / selection_noise[2]).argmax(-1)
            if selection_noise is not None else model._select_node(mom_p.exp()[:, 0], mom_mask[:, 0]))
           + model.sub_vehicle_size)
    state = state.update(task, sub, mom)
    if output_mode == 'selected':
        task_p = task_p.gather(2, task[:, None, None])
        sub_p = sub_p.gather(2, sub[:, None, None])
        mom_p = mom_p.gather(2, (mom - model.sub_vehicle_size)[:, None, None])
    elif output_mode == 'cost':
        task_p = sub_p = mom_p = None
    return task_p, sub_p, mom_p, task, sub, mom, state
