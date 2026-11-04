"""TDRL adaptation with separate task and vehicle encoders."""
import math
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from nets.tdrl_encoders.graph_encoder import GraphAttentionEncoder
from nets.tdrl_encoders.vehicle_encoder import VehicleAttentionEncoder
from problems.nvsp.state_vectorized import VectorizedNVSPState
from utils.data_utils import MAX_VALUES

def _logits(model, query, cache, robot, mask=None, bias=None):
    (b, width) = query.shape
    heads = model.n_heads
    (k, v, logit_key) = cache
    reshape = lambda x: x.view(b, -1, heads, width // heads).transpose(1, 2)
    allowed = None if mask is None else (~mask)[:, None, None, :]
    glimpse = F.scaled_dot_product_attention(reshape(query), reshape(k), reshape(v), attn_mask=allowed, dropout_p=0.0)
    glimpse = glimpse.transpose(1, 2).reshape(b, width)
    output = model.project_out_robot if robot else model.project_out
    normalize = model.norm_glimpse_robot if robot else model.norm_glimpse_task
    final_query = normalize(output(glimpse).float() + query.float())
    logits = torch.matmul(final_query[:, None], logit_key.transpose(1, 2)).squeeze(1) / math.sqrt(width)
    if model.tanh_clipping > 0:
        logits = logits.tanh() * model.tanh_clipping
    if bias is not None:
        logits = logits + bias
    if mask is not None:
        logits = logits.masked_fill(mask, -math.inf)
    return F.log_softmax(logits / model.temp, -1)

def _initialize_direct(model, data):
    return model.encode_tasks(data)

def _direct_chunk(model, static_tasks, projected_query, data, state, noises, count, static_cache=None):
    (likelihoods, actions) = ([], [])
    ids = state.ids[:, 0]
    for step in range(count):
        fleet = model.input_encoder.encode_robots(state, data)
        tasks = model.encode_step_tasks(static_tasks, fleet, projected_query)
        graph = tasks.mean(1) if static_cache is None else static_cache[0]
        if model.direct_fast_decode:
            last = state.last_task_indices.reshape(-1, 1, 1)
            prior = tasks.gather(1, last.clamp_max(tasks.size(1) - 1).expand(-1, 1, tasks.size(-1)))[:, 0]
            prior = torch.where(last[:, 0] == tasks.size(1), model.start_maker[None], prior)
        else:
            nodes = torch.cat((tasks, model.start_maker[None, None].expand(tasks.size(0), 1, -1)), 1)
            prior = nodes.gather(1, state.last_task_indices.reshape(-1, 1, 1).expand(-1, 1, tasks.size(-1)))[:, 0]
        environment = model.environment_context(torch.cat((graph, prior, fleet.mean(1)), -1))
        carriers = fleet[:, model.sub_vehicle_size:]
        robot_cache = model.project_robot_embeddings(fleet).chunk(3, -1) if model.direct_fast_decode else None
        previous = model.empty_robots_embedding[None].expand(tasks.size(0), -1)
        query = model.norm_query_robot(model.project_robot_query(torch.cat((environment, carriers.mean(1), previous), -1)).float() + environment.float())
        carrier_cache = tuple((x[:, model.sub_vehicle_size:] for x in robot_cache)) if robot_cache is not None else model.project_robot_embeddings(carriers).chunk(3, -1)
        carrier_lp = _logits(model, query, carrier_cache, True)
        carrier = model.choose(carrier_lp, None if noises is None else noises[0][step])
        carrier_token = carriers[ids, carrier]
        workers = fleet[:, :model.sub_vehicle_size]
        query = model.norm_query_robot(model.project_robot_query(torch.cat((environment, workers.mean(1), carrier_token), -1)).float() + environment.float())
        worker_cache = tuple((x[:, :model.sub_vehicle_size] for x in robot_cache)) if robot_cache is not None else model.project_robot_embeddings(workers).chunk(3, -1)
        worker_lp = _logits(model, query, worker_cache, True)
        worker = model.choose(worker_lp, None if noises is None else noises[1][step])
        pair = (carrier_token + workers[ids, worker]) * 0.5
        query = model.norm_query_task(model.project_task_query(torch.cat((graph, pair), -1)).float() + environment.float())
        task_keys = model.project_node_embeddings(tasks).chunk(3, -1) if static_cache is None else static_cache[1]
        task_lp = _logits(model, query, task_keys, False, state.mask[:, 0])
        task = model.choose(task_lp, None if noises is None else noises[2][step])
        likelihoods.append(carrier_lp[ids, carrier].float() + worker_lp[ids, worker].float() + task_lp[ids, task].float())
        project_carrier = carrier + model.sub_vehicle_size
        actions.append(torch.stack((task, worker, project_carrier), -1))
        state = state.update(task, worker, project_carrier)
    return (torch.stack(likelihoods, 1), torch.stack(actions, 1), state)

class TDRLEncoder(nn.Module):

    def __init__(self, embedding_dim, n_heads, daughters, mothers):
        super().__init__()
        (self.daughters, self.mothers) = (daughters, mothers)
        self.graph_init_linear = nn.Linear(6, embedding_dim, bias=False)
        self.robot_init_linear = nn.Linear(3, embedding_dim, bias=False)
        self.graph_embedder = GraphAttentionEncoder(n_heads, embedding_dim, 1, normalization='batch', feed_forward_hidden=512)
        self.robot_embedder = VehicleAttentionEncoder(n_heads, embedding_dim, 1, normalization='layer', feed_forward_hidden=512)

    def forward(self, data):
        features = torch.cat((data['sor_loc'] / 100.0, data['tar_loc'] / 100.0, data['designated_time'] / MAX_VALUES['time'], data['tau_h'] / MAX_VALUES['time']), -1)
        return self.graph_embedder(self.graph_init_linear(features))[0]

    def encode_robots(self, state, data):
        positions = state.tar_loc.gather(1, state.vehicle_pos.squeeze(-1)[..., None].expand(-1, -1, 2))
        features = torch.cat((positions / 100.0, state.vehicle_time[..., None] / MAX_VALUES['time']), -1)
        return self.robot_embedder(self.robot_init_linear(features))[0]

class TDRLModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, problem, n_encode_layers=1, normalization='layer', n_heads=8, tanh_clipping=10.0, mom_vehicle_size=4, sub_vehicle_size=8, **kwargs):
        nn.Module.__init__(self)
        if (mom_vehicle_size, sub_vehicle_size) != (4, 8):
            raise ValueError('This TDRL comparison uses four carriers and eight workers')
        if embedding_dim != hidden_dim or n_encode_layers != 1:
            raise ValueError('This TDRL comparison uses equal widths and one encoder layer')
        (self.embedding_dim, self.hidden_dim, self.problem) = (embedding_dim, hidden_dim, problem)
        (self.n_encode_layers, self.n_heads) = (n_encode_layers, n_heads)
        (self.mom_vehicle_size, self.sub_vehicle_size) = (mom_vehicle_size, sub_vehicle_size)
        self.vehicle_size = mom_vehicle_size + sub_vehicle_size
        (self.tanh_clipping, self.temp) = (tanh_clipping, 1.0)
        self.mask_inner = self.mask_logits = True
        self.input_encoder = TDRLEncoder(embedding_dim, n_heads, sub_vehicle_size, mom_vehicle_size)
        self.environment_context = nn.Sequential(nn.Linear(3 * embedding_dim, embedding_dim), nn.LayerNorm(embedding_dim))
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_robot_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_robot_query = nn.Linear(3 * embedding_dim, embedding_dim, bias=False)
        self.project_task_query = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_out_robot = nn.Linear(embedding_dim, embedding_dim, bias=False)
        (self.norm_query_task, self.norm_query_robot) = (nn.LayerNorm(embedding_dim), nn.LayerNorm(embedding_dim))
        (self.norm_glimpse_task, self.norm_glimpse_robot) = (nn.LayerNorm(embedding_dim), nn.LayerNorm(embedding_dim))
        self.empty_robots_embedding = nn.Parameter(torch.empty(embedding_dim))
        self.start_maker = nn.Parameter(torch.empty(embedding_dim))
        nn.init.xavier_uniform_(self.empty_robots_embedding[None])
        nn.init.normal_(self.start_maker, std=0.02)
        (self.shrink_size, self.debug_checks) = (None, False)
        self.fast_state = self.fast_sampling = True
        self.compile_encoder = self.compile_steps = self.checkpoint_dynamic_steps = False
        (self.compile_mode, self.compile_chunk_size) = ('default', 1)
        (self.decoder_fp32, self.fused_recurrent, self.single_query_attention) = (False, False, False)
        self.direct_fast_decode = False
        self.cache_static_tasks = False
        self.decode_type = 'sampling'
        self.request_encoder = 'tdrl'
        self.encoder_variant = 'tdrl'
        self.decoder_variant = 'tdrl'
        self.is_nvsp = True
        self.uses_cross_attention = False
        self.uses_vehicle_context = False
        self.cache_static_tasks = True
        self.direct_fast_decode = True

    def forward(self, data, return_pi=False, cost_only=False):
        initialize = _initialize_direct
        static_tasks = initialize(self, data)
        projected_query = self.project_static_tasks(static_tasks)
        static_cache = (static_tasks.mean(1), self.project_node_embeddings(static_tasks).chunk(3, -1)) if self.cache_static_tasks else None
        state = VectorizedNVSPState.initialize(data, self.sub_vehicle_size, self.mom_vehicle_size)
        function = _direct_chunk
        (size, batch) = (static_tasks.size(1), static_tasks.size(0))
        (likelihoods, trajectories) = ([], [])
        for start in range(0, size, 1):
            count = min(1, size - start)
            noises = None if self.decode_type == 'greedy' else tuple((torch.empty(count, batch, choices, device=static_tasks.device).exponential_() for choices in (self.mom_vehicle_size, self.sub_vehicle_size, size)))
            arguments = (self, static_tasks, projected_query, data, state, noises, count, static_cache)
            if self.training and self.checkpoint_dynamic_steps and torch.is_grad_enabled():
                result = checkpoint(function, *arguments, use_reentrant=False, preserve_rng_state=False)
            else:
                result = function(*arguments)
            (ll, pi, state) = result
            likelihoods.append(ll)
            if return_pi:
                trajectories.append(pi)
        (costs, _) = state.get_costs(data)
        if cost_only:
            return costs
        likelihood = torch.cat(likelihoods, 1).mean(1)
        return (costs, likelihood, torch.cat(trajectories, 1)) if return_pi else (costs, likelihood)

    def encode_tasks(self, data):
        return self.input_encoder(data)

    def project_static_tasks(self, tasks):
        return None

    def encode_step_tasks(self, tasks, fleet, projected_query):
        return tasks

    def choose(self, log_p, noise):
        if self.decode_type == 'greedy':
            return log_p.argmax(-1)
        return (log_p.float().exp() / noise).argmax(-1)

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

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp
