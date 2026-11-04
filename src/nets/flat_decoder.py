"""Flat Decoder ablation with a factorized joint action distribution.

All three scores use the state before the action. The environment is updated
after each request/daughter/mother selection, as in the full model.
"""
import math

import torch
from torch.nn import functional as F

from nets.attention_model import AttentionModel, build_nvsp_request_features
from problems.nvsp.state_vectorized import VectorizedNVSPState


class FlatDecoder(AttentionModel):
    def __init__(self, *args, decoder_variant='flat_decoder', **kwargs):
        if decoder_variant != 'flat_decoder':
            raise ValueError('FlatDecoder requires decoder_variant=flat_decoder')
        super().__init__(*args, decoder_variant='nvmdn', **kwargs)
        self.decoder_variant = 'flat_decoder'

    def sample_many(self, *args, **kwargs):
        raise NotImplementedError('Use scripts/predict.py --variant flat_decoder --decode sample1024')

    def _unary_score(self, context, candidates, projection, mask):
        batch, count, width = candidates.shape
        query = context.reshape(batch, 1, self.n_heads, -1).transpose(1, 2)
        key, value, logit_key = projection(candidates).chunk(3, -1)
        key = key.reshape(batch, count, self.n_heads, -1).transpose(1, 2)
        value = value.reshape(batch, count, self.n_heads, -1).transpose(1, 2)
        glimpse = F.scaled_dot_product_attention(
            query, key, value, attn_mask=(~mask)[:, None], dropout_p=0.)
        glimpse = self.project_out(glimpse.transpose(1, 2).reshape(batch, 1, width))
        logits = torch.matmul(glimpse, logit_key.transpose(1, 2)) / math.sqrt(width // self.n_heads)
        logits = self.tanh_clipping * torch.tanh(logits)
        return logits.masked_fill(mask, -math.inf)[:, 0].float() / self.temp

    def _joint_log_p(self, data, embeddings, state):
        fleet = self.embedder.encode_robots(state, data)
        tasks = self.embedder.cross_attention(embeddings, fleet)
        global_request = tasks.mean(1, keepdim=True)
        last = state.last_task_indices[:, 0]
        previous = tasks.gather(1, last.clamp_max(tasks.size(1)-1)[:, None, None]
                                .expand(-1, 1, tasks.size(-1)))
        previous = torch.where((last == tasks.size(1))[:, None, None],
                               self.start_maker[None, None], previous)
        task_mask = state.get_task_mask()
        remaining = (~task_mask[:, 0]).to(tasks.dtype)
        remaining_pool = ((tasks * remaining[:, :, None]).sum(1, keepdim=True)
                          / remaining.sum(1, keepdim=True)[:, :, None].clamp_min(1))
        daughters = fleet[:, :self.sub_vehicle_size]
        mothers = fleet[:, self.sub_vehicle_size:]
        task_context = self.request_context(torch.cat(
            (global_request, previous, fleet.mean(1, keepdim=True)), -1))
        daughter_context = self.daughter_context(torch.cat(
            (global_request, remaining_pool), -1))
        mother_context = self.mother_context(torch.cat(
            (global_request, remaining_pool, daughters.mean(1, keepdim=True)), -1))
        task = self._unary_score(task_context, tasks, self.project_node_embeddings, task_mask)
        daughter = self._unary_score(daughter_context, daughters, self.project_sub_context,
                                     state.get_sub_vehicle_mask())
        mother = self._unary_score(mother_context, mothers, self.project_mom_context,
                                   state.get_mom_vehicle_mask())
        scores = task[:, :, None, None] + daughter[:, None, :, None] + mother[:, None, None, :]
        return F.log_softmax(scores.flatten(1), -1)

    def forward(self, data, return_pi=False, cost_only=False):
        embeddings, _, _ = self.embedder(build_nvsp_request_features(data), None, data)
        state = VectorizedNVSPState.initialize(data, self.sub_vehicle_size, self.mom_vehicle_size)
        actions, likelihoods = [], []
        for _ in range(embeddings.size(1)):
            log_p = self._joint_log_p(data, embeddings, state)
            if self.decode_type == 'greedy':
                selected = log_p.argmax(-1)
            elif self.decode_type == 'sampling':
                noise = torch.empty_like(log_p).exponential_().clamp_min(1e-12)
                selected = (log_p-noise.log()).argmax(-1)
            else:
                raise ValueError(self.decode_type)
            task = selected // (self.sub_vehicle_size*self.mom_vehicle_size)
            daughter = (selected // self.mom_vehicle_size) % self.sub_vehicle_size
            mother = selected % self.mom_vehicle_size + self.sub_vehicle_size
            state = state.update(task, daughter, mother)
            if not cost_only:
                actions.append(torch.stack((task, daughter, mother), -1))
                likelihoods.append(log_p.gather(1, selected[:, None]).squeeze(1))
        costs, _ = state.get_costs(data)
        if cost_only:
            return costs
        likelihood = torch.stack(likelihoods, 1).mean(1)
        return (costs, likelihood, torch.stack(actions, 1)) if return_pi else (costs, likelihood)
