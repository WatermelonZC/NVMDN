"""AM and HA-DRL policies for the NVSP environment."""
import math
from typing import NamedTuple
import torch
import torch.nn.functional as F
from torch import nn
from nets.am_encoder import GraphAttentionEncoder
from nets.ha_encoder import HAEncoder
from utils.data_utils import build_nvsp_request_features, normalize
import time

def _baseline_decode_step(self, fixed, state, embeddings, ids, output_mode='full', data=None, projected_query=None):
    current_node = state.get_current_node()
    (task_log_p, mask) = self._get_log_p_nvsp(fixed, state, current_node=current_node, normalize=True, select_state='task', select_embedding=None)
    task_selected = self._select_node(task_log_p.exp()[:, 0, :], mask[:, 0, :])
    selected_embedding = embeddings[ids, task_selected]
    (sub_log_p, mask) = self._get_log_p_nvsp(fixed=fixed, state=state, current_node=current_node, select_state='sub_vehicle', select_embedding=selected_embedding.unsqueeze(1), embeddings=embeddings)
    sub_selected = self._select_node(sub_log_p.exp()[:, 0, :], mask[:, 0, :])
    sub_selected_task = state.vehicle_pos[ids, sub_selected, 0]
    sub_selected_node_embedding = self.embeddings_with_depot[ids, sub_selected_task].unsqueeze(1)
    sub_selected_time_state = current_node[1][ids, sub_selected].unsqueeze(-1)
    selected_embedding = self.vehicle_state_att(torch.cat((sub_selected_time_state, sub_selected_node_embedding), dim=-1))
    (mom_log_p, mask) = self._get_log_p_nvsp(fixed=fixed, state=state, current_node=current_node, select_state='mom_vehicle', select_embedding=selected_embedding, embeddings=embeddings)
    mom_selected = self._select_node(mom_log_p.exp()[:, 0, :], mask[:, 0, :])
    mom_selected = mom_selected + self.sub_vehicle_size
    state = state.update(task_selected, sub_selected, mom_selected)
    if output_mode == 'selected':
        task_log_p = task_log_p.gather(2, task_selected[:, None, None])
        sub_log_p = sub_log_p.gather(2, sub_selected[:, None, None])
        mom_log_p = mom_log_p.gather(2, (mom_selected - self.sub_vehicle_size)[:, None, None])
    elif output_mode == 'cost':
        task_log_p = sub_log_p = mom_log_p = None
    return (task_log_p, sub_log_p, mom_log_p, task_selected, sub_selected, mom_selected, state)

class LearningBaselineFixed(NamedTuple):
    """
    Context for LearningBaseline decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return LearningBaselineFixed(node_embeddings=self.node_embeddings[key], context_node_projected=self.context_node_projected[key], glimpse_key=self.glimpse_key[:, key], glimpse_val=self.glimpse_val[:, key], logit_key=self.logit_key[key])

class LearningBaseline(nn.Module):

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def __init__(self, embedding_dim, hidden_dim, problem, n_encode_layers=1, tanh_clipping=10.0, mask_inner=True, mask_logits=True, normalization='batch', n_heads=8, checkpoint_encoder=False, shrink_size=None, mom_vehicle_size=4, sub_vehicle_size=8, sychonization_cost=True, designated_driver=True, encoder_variant='am', **kwargs):
        if encoder_variant not in ('am', 'hadrl'):
            raise ValueError('Expected the AM or HA-DRL encoder')
        if not designated_driver or not sychonization_cost or shrink_size is not None:
            raise ValueError('AM and HA-DRL require the released NVSP state configuration')
        heterogeneous_attention = encoder_variant == 'hadrl'
        super(LearningBaseline, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.allow_partial = problem.NAME == 'sdvrp'
        self.sub_vehicle_size = sub_vehicle_size
        self.mom_vehicle_size = mom_vehicle_size
        self.vehicle_size = mom_vehicle_size + sub_vehicle_size
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.debug_checks = True
        self.use_sdpa = False
        assert designated_driver is not None, 'designated_driver must be set as parameter'
        self.designated_driver = designated_driver
        self.node_dim = 2
        self.prompt_dim = 6
        if designated_driver:
            self.node_dim = 7
            self.prompt_dim = 5
        step_context_dim = embedding_dim
        self.init_embed_depot = nn.Linear(self.node_dim, embedding_dim)
        self.project_sub_vehicle_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_mom_vehicle_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.node_step_project = nn.Linear(embedding_dim + 1, embedding_dim, bias=False)
        self.init_sor_embed = nn.Linear(self.node_dim, embedding_dim)
        self.init_tar_embed = nn.Linear(self.node_dim, embedding_dim)
        self.init_embeddings = None
        self.embeddings_with_depot = None
        self.start_maker = nn.Parameter(torch.Tensor(embedding_dim))
        self.start_maker.data.uniform_(-1, 1)
        assert sychonization_cost is not None, 'sychonization_cost must be set as parameter'
        assert heterogeneous_attention is not None, 'heterogeneous_attention must be set as parameter'
        self.is_sychonization_cost = sychonization_cost
        self.glimpse_norm = nn.LayerNorm(embedding_dim)
        self.sub_key_norm = nn.LayerNorm(embedding_dim)
        self.mom_key_norm = nn.LayerNorm(embedding_dim)
        self.node_query_norm = nn.LayerNorm(embedding_dim)
        self.uses_heterogeneous_attention = heterogeneous_attention
        self.init_embed = nn.Linear(self.node_dim, embedding_dim)
        self.vehicle_state_att = nn.Linear(embedding_dim + 1, embedding_dim)
        if self.uses_heterogeneous_attention:
            self.embedder = HAEncoder(n_layers=self.n_encode_layers, d_model=self.embedding_dim, n_heads=self.n_heads, prompt_dim=self.prompt_dim)
        else:
            self.embedder = GraphAttentionEncoder(n_heads=n_heads, embed_dim=self.embedding_dim, n_layers=self.n_encode_layers, normalization=normalization, prompt_dim=self.prompt_dim)
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_sub_context = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_mom_context = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.encoder_variant = encoder_variant
        self.decoder_variant = 'standard'
        self.request_encoder = 'standard'
        self.is_nvsp = True
        self.uses_cross_attention = False
        self.uses_vehicle_context = False

    def forward(self, input, return_pi=False, cost_only=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        (batch, graph_size, _) = input['designated_time'].size()
        prompt = None
        x = build_nvsp_request_features(input)
        if self.uses_heterogeneous_attention:
            (embeddings, _, self.init_embeddings) = self.embedder(x, prompt)
        else:
            (embeddings, _, self.init_embeddings) = self.embedder(self._init_embed(input), prompt)
        self.embeddings_with_depot = torch.cat((embeddings, self.init_embeddings), dim=1)
        from problems.nvsp.state_vectorized import VectorizedNVSPState
        state = VectorizedNVSPState.initialize(input, self.sub_vehicle_size, self.mom_vehicle_size)
        compact = True
        output_mode = 'cost' if cost_only else 'selected' if compact else 'full'
        (_log_p, pi, state) = self._inner(input, embeddings, state, output_mode=output_mode)
        (cost, mask) = state.get_costs(input, pi)
        if compact:
            self.init_embeddings = self.embeddings_with_depot = None
            self.last_query = self.last_glimpse = None
        if output_mode == 'cost':
            return cost
        ll = _log_p.mean(1) if output_mode == 'selected' else self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return (cost, ll, pi)
        return (cost, ll)

    def _calc_log_likelihood(self, _log_p, a, mask):
        (batch_size, seq_len, _) = a.shape
        likelihoods = torch.zeros(batch_size, seq_len, device=a.device)
        for (i, log_p_tensor) in enumerate(_log_p):
            indices = a[:, :, i]
            if i == 2:
                indices = indices - self.sub_vehicle_size
            selected_log_p = torch.gather(log_p_tensor, 2, indices.unsqueeze(-1)).squeeze(-1)
            likelihoods += selected_log_p
        return likelihoods.mean(1)

    def _init_embed(self, input):
        x = self.init_embed(build_nvsp_request_features(input))
        return torch.cat((x, self.start_maker[None, None, :].expand(x.size(0), 1, self.embedding_dim)), 1)
        return torch.cat((sor_embeddings, tar_embeddings, self.start_maker[None, None, :].expand(input['sor_loc'].size(0), 1, self.embedding_dim)), 1)
        return self.init_embed(input)

    def _inner(self, input, embeddings, state, output_mode='full'):
        chunk = 1
        outputs = []
        task_output = []
        mom_vehicle_output = []
        sub_vehicle_output = []
        sequences = []
        fixed = self._precompute(embeddings)
        batch_size = state.ids.size(0)
        i = 0
        ids = torch.arange(batch_size, device=embeddings.device)
        while not i >= state.order_size:
            values = _baseline_decode_step(self, fixed, state, embeddings, ids, output_mode, input)
            (task_log_p, sub_log_p, mom_log_p, task_selected, sub_selected, mom_selected, state) = values
            if output_mode == 'cost':
                i += 1
                continue
            if output_mode == 'selected':
                outputs.append(task_log_p[:, 0, 0].float() + sub_log_p[:, 0, 0].float() + mom_log_p[:, 0, 0].float())
            else:
                task_output.append(task_log_p[:, 0, :])
                sub_vehicle_output.append(sub_log_p[:, 0, :])
                mom_vehicle_output.append(mom_log_p[:, 0, :])
            sequences.append(torch.stack([task_selected, sub_selected, mom_selected], dim=1))
            i += 1
        if output_mode == 'cost':
            return (None, None, state)
        if output_mode == 'selected':
            return (torch.stack(outputs, 1), torch.stack(sequences, 1), state)
        return ((torch.stack(task_output, 1), torch.stack(sub_vehicle_output, 1), torch.stack(mom_vehicle_output, 1)), torch.stack(sequences, 1), state)
        return (torch.stack(outputs, 1), torch.stack(sequences, 1), state)

    def _select_node(self, probs, mask):
        if getattr(self, 'debug_checks', True):
            assert (probs == probs).all(), 'Probs should not contain any nans'
        if self.decode_type == 'greedy':
            (_, selected) = probs.max(1)
            if getattr(self, 'debug_checks', True):
                assert not mask.gather(1, selected.unsqueeze(-1)).data.any(), 'Decode greedy: infeasible action has maximum probability'
        elif self.decode_type == 'sampling':
            probabilities = probs.masked_fill(mask, 0)
            if getattr(self, 'fast_sampling', False):
                probabilities = probabilities.float()
                noise = torch.empty_like(probabilities).exponential_()
                selected = (probabilities / noise).argmax(dim=-1)
            else:
                selected = probabilities.multinomial(1).squeeze(1)
            if getattr(self, 'debug_checks', True):
                assert not mask.gather(1, selected.unsqueeze(-1)).any()
        else:
            assert False, 'Unknown decode type'
        return selected

    def _precompute(self, embeddings, num_steps=1):
        graph_embed = embeddings.mean(1)
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]
        (glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed) = self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)
        fixed_attention_node_data = (self._make_heads(glimpse_key_fixed, num_steps), self._make_heads(glimpse_val_fixed, num_steps), logit_key_fixed.contiguous())
        return LearningBaselineFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_nvsp(self, fixed, state, current_node, normalize=True, select_state='task', select_embedding=None, embeddings=None):
        if select_state == 'task':
            step_context = self._get_parallel_step_context(fixed.node_embeddings, current_node)
            query = self.node_query_norm(fixed.context_node_projected + self.project_step_context(step_context))
            (glimpse_K, glimpse_V, logit_K) = self._get_attention_node_data(fixed)
            mask = state.get_task_mask()
        elif select_state == 'sub_vehicle':
            query = self.sub_key_norm(self.project_sub_vehicle_query(self.last_glimpse + select_embedding) + self.last_glimpse + select_embedding)
            (glimpse_K, glimpse_V, logit_K) = self.get_attention_vehicle_data(embeddings, current_node, select_state)
            mask = state.get_sub_vehicle_mask()
        elif select_state == 'mom_vehicle':
            query = self.mom_key_norm(self.project_mom_vehicle_query(self.last_glimpse + select_embedding) + self.last_glimpse + select_embedding)
            (glimpse_K, glimpse_V, logit_K) = self.get_attention_vehicle_data(embeddings, current_node, select_state)
            mask = state.get_mom_vehicle_mask()
        (log_p, glimpse, attention) = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        self.last_query = query
        self.last_glimpse = glimpse
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)
        if getattr(self, 'debug_checks', True):
            assert not torch.isnan(log_p).any()
        return (log_p, mask)

    def _get_parallel_step_context(self, embeddings, current_node):
        (current_node, vehicle_time_state, last_order) = current_node
        (batch_size, vehicle_size, task_size) = current_node.size()
        current_node = current_node.long()
        if getattr(self, 'debug_checks', True) and (current_node.max() > embeddings.shape[1] or current_node.min() < 0):
            raise ValueError(f'gather 索引超出范围: min={current_node.min().item()}, max={current_node.max().item()}, 但 embeddings.shape[1]={embeddings.shape[1]}')
        current_node_embedding = torch.gather(self.embeddings_with_depot, 1, current_node.clone().view(batch_size, vehicle_size, 1).expand(batch_size, vehicle_size, embeddings.size(-1)))
        last_order_embedding = torch.gather(self.embeddings_with_depot, 1, last_order.clone().view(batch_size, 1, 1).expand(batch_size, 1, embeddings.size(-1)))
        current_node_embedding = self.vehicle_state_att(torch.cat((vehicle_time_state, current_node_embedding), dim=-1))
        current_node_embedding = current_node_embedding.mean(dim=1, keepdim=True)
        return current_node_embedding + last_order_embedding

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        (batch_size, num_steps, embed_dim) = query.size()
        key_size = val_size = embed_dim // self.n_heads
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        if getattr(self, 'use_sdpa', False):
            q = glimpse_Q.permute(1, 2, 0, 3, 4).reshape(batch_size * num_steps, self.n_heads, 1, key_size)
            k = glimpse_K.permute(1, 2, 0, 3, 4).reshape(batch_size * num_steps, self.n_heads, -1, key_size)
            v = glimpse_V.permute(1, 2, 0, 3, 4).reshape(batch_size * num_steps, self.n_heads, -1, val_size)
            allowed = (~mask).reshape(batch_size * num_steps, 1, 1, -1) if self.mask_inner else None
            heads = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
            heads = heads.reshape(batch_size, num_steps, self.n_heads, 1, val_size).permute(2, 0, 1, 3, 4)
            attention = None
        else:
            compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
            if self.mask_inner:
                assert self.mask_logits, 'Cannot mask inner without masking logits'
                compatibility = compatibility.masked_fill(mask[None, :, :, None, :], -math.inf)
            attention = torch.softmax(compatibility, dim=-1)
            heads = torch.matmul(attention, glimpse_V)
        glimpse = self.project_out(heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))
        final_Q = self.glimpse_norm(glimpse + query.unsqueeze(1))
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf
        return (logits, glimpse.squeeze(-2), attention)

    def _get_attention_node_data(self, fixed):
        return (fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key)

    def get_attention_vehicle_data(self, embeddings, current_node, type='sub_vehicle'):
        if type == 'sub_vehicle':
            (vehicle_state_loc, vehicle_time_state, last_order) = current_node
            vehicle_state_loc = vehicle_state_loc[:, :self.sub_vehicle_size, :]
            vehicle_time_state = vehicle_time_state[:, :self.sub_vehicle_size, :]
            (batch_size, vehicle_size, _) = vehicle_state_loc.size()
            project_context = self.project_sub_context
        else:
            (vehicle_state_loc, vehicle_time_state, last_order) = current_node
            vehicle_state_loc = vehicle_state_loc[:, self.sub_vehicle_size:, :]
            vehicle_time_state = vehicle_time_state[:, self.sub_vehicle_size:, :]
            (batch_size, vehicle_size, _) = vehicle_state_loc.size()
            project_context = self.project_mom_context
        vehicle_task_embedding = torch.gather(self.embeddings_with_depot, 1, vehicle_state_loc.contiguous().view(batch_size, vehicle_size, 1).expand(batch_size, vehicle_size, embeddings.size(-1))).view(batch_size, vehicle_size, embeddings.size(-1))
        msg_embedding = self.vehicle_state_att(torch.cat((vehicle_time_state, vehicle_task_embedding), dim=-1))
        (glimpse_key_step, glimpse_val_step, logit_key_step) = project_context(msg_embedding[:, None, :, :]).chunk(3, dim=-1)
        return (self._make_heads(glimpse_key_step), self._make_heads(glimpse_val_step), logit_key_step)

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1).expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1).permute(3, 0, 1, 2, 4)

    def sample_many(self, data, batch_rep=1, iter_rep=1):
        if data['sor_loc'].size(0) != 1:
            raise ValueError('Fixed-instance evaluation expects a single input instance')
        from problems.nvsp.problem import NVSP
        from problems.nvsp.state import omega
        begin = time.perf_counter()
        (distances, delays, trajectories) = ([], [], [])
        for _ in range(iter_rep):
            expanded = {k: v.repeat_interleave(batch_rep, 0) for (k, v) in data.items()}
            (costs, _, pi) = self(expanded, return_pi=True)
            distances.append(costs[0])
            delays.append(costs[1])
            trajectories.append(pi)
        (distance, delay) = (torch.cat(distances), torch.cat(delays))
        objective = omega * distance + (1 - omega) * delay
        best = int(objective.argmin().item())
        routes = NVSP.process_sequences_torch(torch.cat(trajectories)[best:best + 1], self.vehicle_size).tolist()
        package = [routes, distance.tolist(), delay.tolist(), time.perf_counter() - begin, delay[best].item(), distance[best].item(), delay.mean().item(), distance.mean().item(), objective.mean().item(), objective[best].item()]
        return (routes, objective[best].item(), package)
