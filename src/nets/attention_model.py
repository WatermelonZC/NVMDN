import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple

from utils.ModelData import Aij, Pij, End_T, Start_T
from utils.data_utils import build_nvsp_request_features, normalize, normalization_tensor
from utils.log_utils import createLogger
from utils.tensor_functions import compute_in_batches

from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many, sample_many_nvsp


# logger = createLogger('Attention')


def _fused_decoder_logits(query, glimpse_K, glimpse_V, logit_K, mask,
                          out_weight, norm_weight, norm_bias, norm_eps,
                          n_heads, tanh_clipping, mask_inner, mask_logits, temp):
    batch, steps, width = query.shape
    head_width = width // n_heads
    q = query.view(batch, steps, n_heads, 1, head_width).reshape(batch * steps, n_heads, 1, head_width)
    k = glimpse_K.permute(1, 2, 0, 3, 4).reshape(batch * steps, n_heads, -1, head_width)
    v = glimpse_V.permute(1, 2, 0, 3, 4).reshape(batch * steps, n_heads, -1, head_width)
    allowed = (~mask).reshape(batch * steps, 1, 1, -1) if mask_inner else None
    heads = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
    heads = heads.reshape(batch, steps, n_heads, 1, head_width).transpose(2, 3).reshape(batch, steps, 1, width)
    glimpse = F.linear(heads, out_weight)
    final_q = F.layer_norm(glimpse + query.unsqueeze(1), (width,), norm_weight, norm_bias, norm_eps)
    logits = torch.matmul(final_q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(width)
    if tanh_clipping > 0:
        logits = torch.tanh(logits) * tanh_clipping
    if mask_logits:
        logits = logits.masked_fill(mask, -math.inf)
    return torch.log_softmax(logits / temp, dim=-1), glimpse.squeeze(-2)


_compiled_decoder_logits = None


def get_compiled_decoder():
    global _compiled_decoder_logits
    if _compiled_decoder_logits is None:
        _compiled_decoder_logits = torch.compile(_fused_decoder_logits, dynamic=False)
    return _compiled_decoder_logits


def _dynamic_cross_step(self, state, embeddings, ids, output_mode, data, selection_noise=None, projected_query=None):
    from nets.hierarchical_decoder import decode_step
    fleet = self.embedder.encode_robots(state, data)
    tasks = self.embedder.cross_attention(embeddings, fleet, projected_query)
    fixed = self._precompute(tasks)
    nodes = torch.cat((tasks, self.start_maker[None, None].expand(tasks.size(0), 1, -1)), 1)
    return decode_step(self, fixed, state, tasks, ids, output_mode, fleet=fleet, nodes=nodes,
                       selection_noise=selection_noise)


def _cross_selection_noise(model, embeddings):
    if model.decode_type != 'sampling':
        return None
    return tuple(embeddings.new_empty(embeddings.size(0), count, dtype=torch.float32).exponential_()
                 for count in (embeddings.size(1), model.sub_vehicle_size, model.mom_vehicle_size))


_compiled_dynamic_steps = {}


def get_compiled_dynamic_step(mode='default'):
    if mode not in _compiled_dynamic_steps:
        _compiled_dynamic_steps[mode] = torch.compile(_dynamic_cross_step, dynamic=False, fullgraph=True, mode=mode)
    return _compiled_dynamic_steps[mode]


def _nvsp_decode_step(self, fixed, state, embeddings, ids, output_mode='full', data=None, projected_query=None):
    if getattr(self, 'request_encoder', 'vehicle_encoder') == 'task_robot_cross':
        return _dynamic_cross_step(self, state, embeddings, ids, output_mode, data,
                                   _cross_selection_noise(self, embeddings), projected_query)
    if getattr(self, 'decoder_variant', 'standard') == 'nvmdn':
        from nets.hierarchical_decoder import decode_step
        return decode_step(self, fixed, state, embeddings, ids, output_mode)
    current_node = state.get_current_node()
    task_log_p, mask = self._get_log_p_nvsp(fixed, state, current_node=current_node, normalize=True, select_state="task",
                                          select_embedding=None)
    task_selected = self._select_node(task_log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
    selected_embedding = embeddings[ids, task_selected]  # (batch_size, embedding_size)

    sub_log_p, mask = self._get_log_p_nvsp(fixed=fixed, state=state, current_node=current_node, select_state="sub_vehicle",
                                         select_embedding=selected_embedding.unsqueeze(1), embeddings=embeddings)
    sub_selected = self._select_node(sub_log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
    sub_selected_task = state.vehicle_pos[ids, sub_selected, 0]  # (batch,)
    sub_selected_node_embedding = self.embeddings_with_depot[ids, sub_selected_task].unsqueeze(1)  # (batch, 1, embedding_size)
    sub_selected_time_state = current_node[1][ids, sub_selected].unsqueeze(-1)  # (batch,)
    if self.uses_vehicle_context:
        selected_embedding = self.vehicle_state_att(sub_selected_time_state, sub_selected_node_embedding)
    else:
        selected_embedding = self.vehicle_state_att(torch.cat((sub_selected_time_state, sub_selected_node_embedding), dim=-1))

    mom_log_p, mask = self._get_log_p_nvsp(fixed=fixed, state=state, current_node=current_node, select_state="mom_vehicle",
                                         select_embedding=selected_embedding,
                                         embeddings=embeddings)
    mom_selected = self._select_node(mom_log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
    mom_selected = mom_selected + self.sub_vehicle_size
    state = state.update(task_selected, sub_selected, mom_selected)
    if output_mode == 'selected':
        task_log_p = task_log_p.gather(2, task_selected[:, None, None])
        sub_log_p = sub_log_p.gather(2, sub_selected[:, None, None])
        mom_log_p = mom_log_p.gather(2, (mom_selected - self.sub_vehicle_size)[:, None, None])
    elif output_mode == 'cost':
        task_log_p = sub_log_p = mom_log_p = None
    return task_log_p, sub_log_p, mom_log_p, task_selected, sub_selected, mom_selected, state



_compiled_nvsp_steps = {}


def get_compiled_nvsp_step(mode='default'):
    if mode not in _compiled_nvsp_steps:
        _compiled_nvsp_steps[mode] = torch.compile(
            _nvsp_decode_step, dynamic=False, fullgraph=True, mode=mode)
    return _compiled_nvsp_steps[mode]


def _nvsp_decode_chunk(model, fixed, state, embeddings, ids, output_mode, count, data=None, projected_query=None):
    # A small static loop lets AOTAutograd combine work across decisions.
    # Decisions still use the updated state from the preceding decision.
    outputs, sequences = [], []
    for _ in range(count):
        task_p, sub_p, mom_p, task, sub, mom, state = _nvsp_decode_step(
            model, fixed, state, embeddings, ids, output_mode, data, projected_query)
        if output_mode != 'cost':
            outputs.append(task_p[:, 0, 0].float() + sub_p[:, 0, 0].float() + mom_p[:, 0, 0].float())
            sequences.append(torch.stack((task, sub, mom), dim=1))
    return (None if output_mode == 'cost' else torch.stack(outputs, 1),
            None if output_mode == 'cost' else torch.stack(sequences, 1), state)


_compiled_nvsp_chunks = {}


def get_compiled_nvsp_chunk(mode='default'):
    if mode not in _compiled_nvsp_chunks:
        _compiled_nvsp_chunks[mode] = torch.compile(
            _nvsp_decode_chunk, dynamic=False, fullgraph=True, mode=mode)
    return _compiled_nvsp_chunks[mode]


_compiled_encoders = {}


def get_compiled_encoder(encoder):
    # Separate code objects prevent five different encoder families from
    # exhausting Dynamo's per-function specialization limit in one process.
    cls = type(encoder)
    if cls not in _compiled_encoders:
        _compiled_encoders[cls] = torch.compile(cls.forward, dynamic=False)
    return _compiled_encoders[cls]


def set_decode_type(model, decode_type, temp=None):
    # 如果模型是DataParallel或DistributedDataParallel类型，则将其转换为module类型
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module
    # 调用模型的set_decode_type函数，设置解码类型
    model.set_decode_type(decode_type, temp=temp)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )


class AttentionModel(nn.Module):

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 mom_vehicle_size=None,
                 sub_vehicle_size=None,
                 sychonization_cost=None,
                 use_vehicle_context=None,
                 designated_driver=None,
                 use_cross_attention=None,
                 dist_threshold=None,
                 time_threshold=None,
                 encoder_variant='nvmdn',
                 decoder_variant='nvmdn',
                 decoder_ablation='none',
                 request_encoder='task_robot_cross'):
        super(AttentionModel, self).__init__()
        if (problem.NAME != 'nvsp' or decoder_variant != 'nvmdn' or not use_cross_attention
                or request_encoder != 'task_robot_cross'):
            raise ValueError('This release supports the NVMDN architecture only')
        self.encoder_variant = encoder_variant
        self.decoder_variant = decoder_variant
        self.decoder_ablation = decoder_ablation
        self.request_encoder = request_encoder if use_cross_attention else 'standard'
        if request_encoder not in ('vehicle_encoder', 'task_robot', 'task_robot_cross', 'standard'):
            raise ValueError('Unknown request encoder: ' + request_encoder)
        if use_cross_attention and request_encoder in ('task_robot', 'task_robot_cross') and (decoder_ablation != 'none' or not designated_driver):
            raise ValueError('Task/robot encoding requires the full request input without old ablations')
        if decoder_variant == 'nvmdn' and not use_cross_attention:
            raise ValueError('The hierarchical decoder requires cross-attention')

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_nvsp = problem.NAME == 'nvsp'
        # Mother and daughter vehicle counts.
        self.sub_vehicle_size = sub_vehicle_size
        self.mom_vehicle_size = mom_vehicle_size

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.debug_checks = True
        self.use_sdpa = False

        # Problem specific context parameters (placeholder and step context dimension)
        if self.is_nvsp:
            assert designated_driver is not None, "designated_driver must be set as parameter"
            self.designated_driver = designated_driver
            self.node_dim = 2  # x, y
            self.prompt_dim = 6
            if designated_driver:
                self.node_dim = 7
                self.prompt_dim = 5
            step_context_dim = embedding_dim
            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(self.node_dim, embedding_dim)
            self.project_sub_vehicle_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.project_mom_vehicle_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.node_step_project = nn.Linear(embedding_dim + 1, embedding_dim, bias=False)
            self.init_sor_embed = nn.Linear(self.node_dim, embedding_dim)
            self.init_tar_embed = nn.Linear(self.node_dim, embedding_dim)
            self.init_embeddings = None
            self.embeddings_with_depot = None
            self.start_maker = nn.Parameter(torch.Tensor(embedding_dim))
            self.start_maker.data.uniform_(-1, 1)  # Placeholder should be in range of activations
            # self.sub_time_embedding = nn.Linear(1, embedding_dim)
            # self.mom_time_embedding = nn.Linear(1, embedding_dim)
            assert sychonization_cost is not None, "sychonization_cost must be set as parameter"
            assert use_vehicle_context is not None, "use_vehicle_context must be set as parameter"
            # assert not (use_vehicle_context and not use_cross_attention), "use_vehicle_context must be used without heterogeneous"
            assert use_cross_attention is not None, "use_cross_attention must be set as parameter"

            self.is_sychonization_cost = sychonization_cost
            self.uses_vehicle_context = use_vehicle_context
            self.glimpse_norm = nn.LayerNorm(embedding_dim)
            self.sub_key_norm = nn.LayerNorm(embedding_dim)
            self.mom_key_norm = nn.LayerNorm(embedding_dim)
            self.node_query_norm = nn.LayerNorm(embedding_dim)
            self.uses_cross_attention = use_cross_attention
            self.init_embed = nn.Linear(self.node_dim, embedding_dim)
            self.vehicle_state_att = nn.Identity()
            self.request_context = nn.Linear(3 * embedding_dim, embedding_dim, bias=False)
            self.daughter_context = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
            self.mother_context = nn.Linear(3 * embedding_dim, embedding_dim, bias=False)

        from nets.task_robot_cross_encoder import TaskRobotCrossEncoder
        self.embedder = TaskRobotCrossEncoder(self.node_dim, embedding_dim, n_encode_layers,
                                             n_heads, sub_vehicle_size, mom_vehicle_size)

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        # 补充，子母车异构问题
        self.project_sub_context = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_mom_context = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)

        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False, cost_only=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        if (getattr(self, 'compile_mode', 'default') != 'default'
                and torch.cuda.is_available() and hasattr(torch.compiler, 'cudagraph_mark_step_begin')):
            torch.compiler.cudagraph_mark_step_begin()
        if self.is_nvsp:
            if self.designated_driver:
                batch, graph_size, _ = input['designated_time'].size()
                prompt = None
                # 归一化：
                x = build_nvsp_request_features(input)
            else:
                prompt = torch.cat((input['params'], input['designated_time']), dim=1)
                x = torch.cat((input['sor_loc'], input['tar_loc']), dim=1)
            if self.uses_cross_attention:
                if self.request_encoder in ('task_robot', 'task_robot_cross'):
                    encoder = get_compiled_encoder(self.embedder) if getattr(self, 'compile_encoder', False) else type(self.embedder).forward
                    embeddings, _, self.init_embeddings = encoder(self.embedder, x, prompt, input)
                elif getattr(self, 'compile_encoder', False):
                    embeddings, _, self.init_embeddings = get_compiled_encoder(self.embedder)(self.embedder, x, prompt)
                else:
                    embeddings, _, self.init_embeddings = self.embedder(x, prompt)
            #     embeddings, _, self.init_embeddings = self.embedder(torch.cat((input['sor_loc'], input['tar_loc']), dim=1))
            else:
                if getattr(self, 'compile_encoder', False):
                    embeddings, _, self.init_embeddings = get_compiled_encoder(self.embedder)(self.embedder, self._init_embed(input), prompt)
                else:
                    embeddings, _, self.init_embeddings = self.embedder(self._init_embed(input), prompt)
            self.embeddings_with_depot = torch.cat((embeddings, self.init_embeddings), dim=1)
        else:
            if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
                embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
            else:
                embeddings, _ = self.embedder(self._init_embed(input))
        if (self.is_nvsp and getattr(self, 'vectorized_state', False)
                and self.shrink_size is None):
            from problems.nvsp.state_vectorized import VectorizedNVSPState
            state = VectorizedNVSPState.initialize(input, self.sub_vehicle_size, self.mom_vehicle_size)
        else:
            state = self.problem.make_state(input)
        compact = self.is_nvsp and getattr(self, 'vectorized_state', False) and self.shrink_size is None
        output_mode = 'cost' if cost_only and self.is_nvsp else ('selected' if compact else 'full')
        prepared_attention = None
        if self.is_nvsp and self.uses_vehicle_context and self.decoder_variant != 'nvmdn':
            attention = self.vehicle_state_att.attn.attention.module
            if getattr(attention, 'packed_vehicle_attention', False):
                prepared_attention = attention
                counts = (self.sub_vehicle_size + self.mom_vehicle_size, self.sub_vehicle_size, self.mom_vehicle_size)
                attention.prepare_packed_weights(counts if len(set(counts)) == len(counts) else ())
        _log_p, pi, state = self._inner(input, embeddings, state, output_mode=output_mode)
        if prepared_attention is not None:
            prepared_attention.clear_packed_weights()
        # cost, mask = self.problem.get_costs(input, pi)
        cost, mask = state.get_costs(input, pi)
        if compact:
            # Autograd retains the tensors it needs. Keeping a second copy on
            # the module would make deepcopy fail after a cached rollout
            # replay, because replay does not reassign these Python fields.
            self.init_embeddings = self.embeddings_with_depot = None
            self.last_query = self.last_glimpse = None
        if output_mode == 'cost':
            return cost

        ll = _log_p.mean(1) if output_mode == 'selected' else self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, ll, pi
        return cost, ll

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))

        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) // ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):
        if self.is_nvsp:
            # 以下已经验证过了：我需要计算likelihood，比如pi的第一组数据[1,0,3]，分别对饮log_p的第一个list的[-1.4247, -1.4173, -1.0772, -1.7342]的第1个值 -1.4173，第二个list[-0.7559, -1.5671, -1.1339]的第0个值-0.7559，第三个list[-4.1649e-01, -1.0769e+00]的3 - sub_size = 0个，也就是-4.1649e-01
            batch_size, seq_len, _ = a.shape  # 获取 batch 和 任务数量
            likelihoods = torch.zeros(batch_size, seq_len, device=a.device)  # 存储每个序列的 log-likelihood

            for i, log_p_tensor in enumerate(_log_p):
                indices = a[:, :, i]  # 取出 pi 的第 i 列索引
                if i == 2:  # 第三个 log_p，需要减去 sub_size
                    indices = indices - self.sub_vehicle_size

                # 进行索引：gather 需要 indices 维度与 log_p 维度匹配
                selected_log_p = torch.gather(log_p_tensor, 2, indices.unsqueeze(-1)).squeeze(-1)
                likelihoods += selected_log_p  # 累加 log-likelihood

            # return likelihoods.sum(1)
            return likelihoods.mean(1)
        else:
            # Get log_p corresponding to selected actions
            log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

            # Optional: mask out actions irrelevant to objective so they do not get reinforced
            if mask is not None:
                log_p[mask] = 0

            assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

            # Calculate log_likelihood
            return log_p.sum(1)

    def _init_embed(self, input):
        if self.is_nvsp:
            # depot_embeddings = self.init_embed_depot(input['depot'])[:, None, :]
            if self.designated_driver:
                x = self.init_embed(build_nvsp_request_features(input))
                return torch.cat(
                    (x, self.start_maker[None, None, :].expand(x.size(0), 1, self.embedding_dim)), 1)
            else:
                sor_embeddings = self.init_sor_embed(input['sor_loc'])
                tar_embeddings = self.init_tar_embed(input['tar_loc'])

            return torch.cat(
                (sor_embeddings, tar_embeddings, self.start_maker[None, None, :].expand(input['sor_loc'].size(0), 1, self.embedding_dim)), 1)
            # return s？elf.init_embed(torch.cat((input['sor_loc'], input['tar_loc']), 1))
            # return self.init_embed(torch.cat((input['tar_loc'], input['sor_loc']), -1))
        # TSP
        return self.init_embed(input)

    def _inner(self, input, embeddings, state, output_mode='full'):

        chunk = getattr(self, 'compile_chunk_size', 1)
        recompute_dynamic = (self.request_encoder == 'task_robot_cross' and self.training
                             and torch.is_grad_enabled() and getattr(self, 'checkpoint_dynamic_steps', False))
        if (chunk > 1 and self.is_nvsp and getattr(self, 'compile_steps', False)
                and getattr(self, 'vectorized_state', False) and self.shrink_size is None
                and output_mode in ('selected', 'cost') and not recompute_dynamic):
            return self._inner_compiled_chunks(embeddings, state, output_mode, chunk, input)

        outputs = []
        task_output = []
        mom_vehicle_output = []
        sub_vehicle_output = []

        sequences = []

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        batch_size = state.ids.size(0)

        # Perform decoding steps
        i = 0
        ids = torch.arange(batch_size, device=embeddings.device)
        # NVSP executes exactly one complete order at each step. Use the Python
        # counter instead of reading a CUDA tensor at every decision.
        while not (self.shrink_size is None and
                   (i >= state.order_size if self.is_nvsp else state.all_finished())):
            # shrink_size 用于动态调整 batch size，提高计算效率
            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16（如果未完成的样本过少（小于 16），则不进行 batch shrinking（因为批量归一化在 batch 过小时效果不佳））
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]
            if self.is_nvsp:
                if recompute_dynamic:
                    function = (get_compiled_dynamic_step(getattr(self, 'compile_mode', 'default'))
                                if getattr(self, 'compile_steps', False) else _dynamic_cross_step)
                    # Keep checkpoint outside torch.compile. AOT's compiled
                    # checkpoint wrapper queries RNG state even for a fully
                    # deterministic body, which cannot run in CUDA capture.
                    checkpoint_options = {}
                    if hasattr(self.embedder, 'preserve_recomputed_robot_statistics'):
                        from contextlib import nullcontext
                        checkpoint_options['context_fn'] = lambda: (
                            nullcontext(), self.embedder.preserve_recomputed_robot_statistics())
                    values = checkpoint(function, self, state, embeddings, ids, output_mode, input,
                                        _cross_selection_noise(self, embeddings),
                                        use_reentrant=False, preserve_rng_state=False, **checkpoint_options)
                    task_log_p, sub_log_p, mom_log_p, task_selected, sub_selected, mom_selected, state = values
                elif getattr(self, 'compile_steps', False) and self.is_nvsp:
                    history = state.sequence
                    values = get_compiled_nvsp_step(getattr(self, 'compile_mode', 'default'))(self, fixed, state._replace(sequence=[]), embeddings, ids, output_mode, input)
                    task_log_p, sub_log_p, mom_log_p, task_selected, sub_selected, mom_selected, state = values
                    state = state._replace(sequence=history + state.sequence)
                else:
                    values = _nvsp_decode_step(self, fixed, state, embeddings, ids, output_mode, input)
                    task_log_p, sub_log_p, mom_log_p, task_selected, sub_selected, mom_selected, state = values
            else:
                log_p, mask = self._get_log_p(fixed, state)  # batch, vehicle_size, n_loc
                # Select the indices of the next nodes in the sequences, result (batch_size) long
                selected = self._select_node(log_p.exp(), mask.view(batch_size, -1))  # Squeeze out steps dimension
                state = state.update(selected)

            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            if self.is_nvsp:
                if output_mode == 'cost':
                    i += 1
                    continue
                if output_mode == 'selected':
                    outputs.append(task_log_p[:, 0, 0].float() + sub_log_p[:, 0, 0].float() + mom_log_p[:, 0, 0].float())
                else:
                    task_output.append(task_log_p[:, 0, :])
                    sub_vehicle_output.append(sub_log_p[:, 0, :])
                    mom_vehicle_output.append(mom_log_p[:, 0, :])
                # outputs.append(torch.stack([task_log_p[:, 0, :], sub_log_p[:, 0, :], mom_log_p[:, 0, :]], dim=1))
                sequences.append(torch.stack([task_selected, sub_selected, mom_selected], dim=1))
            else:
                outputs.append(log_p[:, 0, :])
                sequences.append(selected)
            i += 1

        # Collected lists, return Tensor
        if output_mode == 'cost':
            return None, None, state
        if output_mode == 'selected':
            return torch.stack(outputs, 1), torch.stack(sequences, 1), state
        if self.is_nvsp:

            # return torch.stack(outputs, 1), state.process_sequences_torch(sequences, state.loc.size(1), state.vehicle_size)
            return ((torch.stack(task_output, 1), torch.stack(sub_vehicle_output, 1), torch.stack(mom_vehicle_output, 1)),
                    torch.stack(sequences, 1), state)
        return torch.stack(outputs, 1), torch.stack(sequences, 1), state

    def _inner_compiled_chunks(self, embeddings, state, output_mode, chunk, data=None):
        fixed = self._precompute(embeddings)
        ids = torch.arange(state.ids.size(0), device=embeddings.device)
        outputs, sequences = [], []
        compiled = get_compiled_nvsp_chunk(getattr(self, 'compile_mode', 'default'))
        # Task self-attention is static; only K/V change with the fleet state.
        # Keep this tensor local to one forward so weights and graphs stay fresh.
        projected_query = (self.embedder.cross_attention.project_query(embeddings)
                           if self.request_encoder == 'task_robot_cross'
                           and getattr(self, 'cache_cross_query', False) else None)
        for start in range(0, state.order_size, chunk):
            log_p, actions, state = compiled(self, fixed, state, embeddings, ids,
                                             output_mode, min(chunk, state.order_size - start), data, projected_query)
            if output_mode != 'cost':
                outputs.append(log_p)
                sequences.append(actions)
        return (None if output_mode == 'cost' else torch.cat(outputs, 1),
                None if output_mode == 'cost' else torch.cat(sequences, 1), state)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        global prompt, x
        if self.is_nvsp:
            prompt = None
            # x = torch.cat(
            #     (torch.cat((input['sor_loc'], input['designated_time']), dim=-1),
            #      torch.cat((input['tar_loc'], input['designated_time']), dim=-1)), dim=1)
            x = build_nvsp_request_features(input)

            if self.uses_cross_attention:
                return sample_many_nvsp(
                    lambda input: self._inner(*input),  # Need to unpack tuple into arguments
                    lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
                    (input, self.embedder(x, prompt, input) if self.request_encoder in ('task_robot', 'task_robot_cross') else self.embedder(x, prompt)),
                    # Pack input with embeddings (additional input)
                    batch_rep, iter_rep, self
                )
            else:
                return sample_many_nvsp(
                    lambda input: self._inner(*input),  # Need to unpack tuple into arguments
                    lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
                    (input, self.embedder(self._init_embed(input), prompt)),  # Pack input with embeddings (additional input)
                    batch_rep, iter_rep, self
                )

    def _select_node(self, probs, mask):

        if getattr(self, 'debug_checks', True):
            assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            if getattr(self, 'debug_checks', True):
                assert not mask.gather(1, selected.unsqueeze(
                    -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            # Explicitly zero masked entries. Multinomial cannot draw a zero
            # entry, so no host-side resampling loop is needed.
            probabilities = probs.masked_fill(mask, 0)
            if getattr(self, 'fast_sampling', False):
                # Exponential race samples the same categorical distribution.
                # Unlike multinomial, it has no per-call device assertions or
                # host synchronization. FP32 avoids overflow in FP16 division.
                probabilities = probabilities.float()
                noise = torch.empty_like(probabilities).exponential_()
                selected = (probabilities / noise).argmax(dim=-1)
            else:
                selected = probabilities.multinomial(1).squeeze(1)
            if getattr(self, 'debug_checks', True):
                assert not mask.gather(1, selected.unsqueeze(-1)).any()

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = (graph_embed[:, None, :] if getattr(self, 'cache_request_context', False)
                         and self.decoder_variant == 'nvmdn' else self.project_fixed_context(graph_embed)[:, None, :])

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p_nvsp(self, fixed, state, current_node, normalize=True, select_state="task", select_embedding=None, embeddings=None):
        if select_state == "task":
            # step_context = checkpoint(self._get_parallel_step_context, fixed.node_embeddings, current_node, use_reentrant=False)

            # 还需要补充上一服务订单
            step_context = self._get_parallel_step_context(fixed.node_embeddings, current_node)
            if self.is_sychonization_cost:
                query = self.node_query_norm(fixed.context_node_projected + \
                                             self.project_step_context(step_context))
            else:
                # query第二个维度决定了logit第二个维度的大小， K\V的倒数二个维度决定了logit的第三个维度的大小
                query = self.node_query_norm(fixed.context_node_projected + \
                                             self.project_step_context(step_context))
            glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed)
            # Compute the mask
            mask = state.get_task_mask()
        elif select_state == "sub_vehicle":
            # select_embedding 实际上为节点解码器所选中的节点的特征，
            query = self.sub_key_norm(self.project_sub_vehicle_query(self.last_glimpse + select_embedding) + self.last_glimpse + select_embedding)
            glimpse_K, glimpse_V, logit_K = self.get_attention_vehicle_data(embeddings, current_node, select_state)
            # Compute the mask
            mask = state.get_sub_vehicle_mask()
            # mask = torch.zeros(query.size(0), 1, self.sub_vehicle_size, dtype=torch.bool, device=query.device)
        elif select_state == "mom_vehicle":
            # select_embedding上为子车（shuttle）解码器所选中的车的其上一服务节点、空闲时间的特征经过MHA特征提取后得到
            query = self.mom_key_norm(self.project_mom_vehicle_query(self.last_glimpse + select_embedding) + self.last_glimpse + select_embedding)
            glimpse_K, glimpse_V, logit_K = self.get_attention_vehicle_data(embeddings, current_node, select_state)
            # Compute the mask
            mask = state.get_mom_vehicle_mask()
            # mask = torch.zeros(query.size(0), 1, self.mom_vehicle_size, dtype=torch.bool, device=query.device)
        if getattr(self, 'compile_decoder', False) and normalize:
            log_p, glimpse = get_compiled_decoder()(
                query, glimpse_K, glimpse_V, logit_K, mask,
                self.project_out.weight, self.glimpse_norm.weight, self.glimpse_norm.bias,
                self.glimpse_norm.eps, self.n_heads, self.tanh_clipping,
                self.mask_inner, self.mask_logits, self.temp)
            attention = None
        else:
            log_p, glimpse, attention = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        # if select_state == "task":
        #     i = current_node[2][0].tolist()[0]
        #     mean_attention = attention.mean(dim=0).squeeze(2)  # -> (batch_size, num_steps, graph_size)
        #     Aij[i] = mean_attention.tolist()[0][0]
        # Compute logits (unnormalized log_p)
        self.last_query = query
        self.last_glimpse = glimpse
        if normalize and not getattr(self, 'compile_decoder', False):
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)
        if getattr(self, 'debug_checks', True):
            assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, current_node):
        if self.is_sychonization_cost:
            current_node, vehicle_time_state, last_order = current_node
        else:
            current_node = current_node
        batch_size, vehicle_size, task_size = current_node.size()
        current_node = current_node.long()
        # 确保索引范围合法
        if getattr(self, 'debug_checks', True) and (current_node.max() > embeddings.shape[1] or current_node.min() < 0):
            raise ValueError(
                f"gather 索引超出范围: min={current_node.min().item()}, max={current_node.max().item()}, 但 embeddings.shape[1]={embeddings.shape[1]}")
        # current_node = current_node.clone()
        # size 为batch_size, vehicle_size, embeddings.size(-1)
        current_node_embedding = (torch.gather(
            self.embeddings_with_depot,
            1,
            current_node.clone()
            .view(batch_size, vehicle_size, 1)
            .expand(batch_size, vehicle_size, embeddings.size(-1))
        ))
        last_order_embedding = (torch.gather(
            self.embeddings_with_depot,
            1,
            last_order.clone()
            .view(batch_size, 1, 1)
            .expand(batch_size, 1, embeddings.size(-1))
        ))
        if self.uses_vehicle_context:
            current_node_embedding = self.vehicle_state_att(vehicle_time_state, current_node_embedding)
            current_node_embedding = current_node_embedding.mean(1, keepdim=True)
        else:
            current_node_embedding = self.vehicle_state_att(torch.cat((vehicle_time_state, current_node_embedding), dim=-1))
            current_node_embedding = current_node_embedding.mean(dim=1, keepdim=True)
        return current_node_embedding + last_order_embedding

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        if getattr(self, 'use_sdpa', False) and (not getattr(self, 'small_attention_math', False)
                                                or getattr(self, 'fused_vehicle_attention', False)):
            # Flatten batch and decoding-step axes for CUDA fused attention.
            q = glimpse_Q.permute(1, 2, 0, 3, 4).reshape(batch_size * num_steps, self.n_heads, 1, key_size)
            k = glimpse_K.permute(1, 2, 0, 3, 4).reshape(batch_size * num_steps, self.n_heads, -1, key_size)
            v = glimpse_V.permute(1, 2, 0, 3, 4).reshape(batch_size * num_steps, self.n_heads, -1, val_size)
            allowed = (~mask).reshape(batch_size * num_steps, 1, 1, -1) if self.mask_inner else None
            if (getattr(self, 'fused_vehicle_attention', False) and k.dtype == torch.float16 and q.is_cuda
                    and k.shape[-2] <= 128 and key_size in (16, 32)):
                from nets.vehicle_attention import short_attention
                # The query LayerNorm stays in FP32 under AMP. SDPA normally
                # autocasts it internally; the custom operation must do so too.
                heads = short_attention(q.to(k.dtype), k, v, allowed)
            else:
                heads = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
            heads = heads.reshape(batch_size, num_steps, self.n_heads, 1, val_size).permute(2, 0, 1, 3, 4)
            attention = None
        else:
            compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
            if self.mask_inner:
                assert self.mask_logits, "Cannot mask inner without masking logits"
                compatibility = compatibility.masked_fill(mask[None, :, :, None, :], -math.inf)
            attention = torch.softmax(compatibility, dim=-1)
            heads = torch.matmul(attention, glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = self.glimpse_norm(glimpse + query.unsqueeze(1))
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2), attention

    def _get_attention_node_data(self, fixed):

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def get_attention_vehicle_data(self, embeddings, current_node, type="sub_vehicle"):
        # decompose子车特征
        if type == "sub_vehicle":
            if self.is_sychonization_cost:
                vehicle_state_loc, vehicle_time_state, last_order = current_node
                vehicle_state_loc = vehicle_state_loc[:, :self.sub_vehicle_size, :]
                vehicle_time_state = vehicle_time_state[:, :self.sub_vehicle_size, :]
            else:
                vehicle_state_loc = current_node
                vehicle_state_loc = vehicle_state_loc[:, :self.sub_vehicle_size, :]
            batch_size, vehicle_size, _ = vehicle_state_loc.size()
            project_context = self.project_sub_context
            # time_context = self.sub_time_embedding
        else:
            if self.is_sychonization_cost:
                vehicle_state_loc, vehicle_time_state, last_order = current_node
                vehicle_state_loc = vehicle_state_loc[:, self.sub_vehicle_size:, :]
                vehicle_time_state = vehicle_time_state[:, self.sub_vehicle_size:, :]
            else:
                vehicle_state_loc = current_node
                vehicle_state_loc = vehicle_state_loc[:, self.sub_vehicle_size:, :]
            batch_size, vehicle_size, _ = vehicle_state_loc.size()
            project_context = self.project_mom_context
            # time_context = self.mom_time_embedding
        vehicle_task_embedding = torch.gather(
            self.embeddings_with_depot,
            1,
            vehicle_state_loc.contiguous()
            .view(batch_size, vehicle_size, 1)
            .expand(batch_size, vehicle_size, embeddings.size(-1))
        ).view(batch_size, vehicle_size, embeddings.size(-1))
        if self.uses_vehicle_context:
            # vehicle_time_state = normalize(vehicle_time_state)
            # vehicle_time_embedding = time_context(vehicle_time_state)
            # msg_embedding = vehicle_time_embedding + vehicle_task_embedding
            msg_embedding = self.vehicle_state_att(vehicle_time_state, vehicle_task_embedding)
        else:
            msg_embedding = self.vehicle_state_att(torch.cat((vehicle_time_state, vehicle_task_embedding), dim=-1))
        glimpse_key_step, glimpse_val_step, logit_key_step = \
            project_context(msg_embedding[:, None, :, :]).chunk(3, dim=-1)

        # Projection of concatenation is equivalent to addition of projections but this is more efficient
        return (
            self._make_heads(glimpse_key_step),
            self._make_heads(glimpse_val_step),
            logit_key_step,
        )

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
