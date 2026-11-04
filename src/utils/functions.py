import logging
import time
import warnings
import torch
import numpy as np
import os
import json

from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import torch.nn.functional as F

from utils.ModelData import End_T, Start_T, Aij


def load_problem(name):
    from problems.nvsp.problem import NVSP
    problem = {'nvsp': NVSP}.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage, weights_only=False)  # Load on CPU


def move_to(var, device, non_blocking=False):
    if isinstance(var, dict):
        return {k: move_to(v, device, non_blocking=non_blocking) for k, v in var.items()}
    return var.to(device, non_blocking=non_blocking)


def validate_encoder_checkpoint(model, checkpoint):
    """Require the architecture metadata and tensors for the released model."""
    for name in ('request_encoder', 'encoder_variant', 'decoder_variant'):
        if checkpoint.get(name) != getattr(model, name):
            raise ValueError('Checkpoint configuration mismatch for ' + name)
    if set(checkpoint.get('model', {})) != set(model.state_dict()):
        raise ValueError('Checkpoint parameter set does not match the selected model')


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage, weights_only=False)

    if isinstance(load_data, dict):
        validate_encoder_checkpoint(model, load_data)
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()
    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args


def load_model(path, epoch=None):
    from nets.attention_model import AttentionModel
    from nets.flat_decoder import FlatDecoder
    from nets.learning_baselines import LearningBaseline
    from nets.tdrl_model import TDRLModel

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None and os.path.isfile(os.path.join(path, 'model.pt')):
            return load_model(os.path.join(path, 'model.pt'))
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, 'args.json'))

    problem = load_problem(args['problem'])

    model_class = {
        'am': LearningBaseline, 'hadrl': LearningBaseline, 'tdrl': TDRLModel,
        'attention': FlatDecoder if args.get('decoder_variant') == 'flat_decoder' else AttentionModel,
    }.get(args.get('model', 'attention'), None)
    assert model_class is not None, "Unknown model: {}".format(model_class)

    model = model_class(
        args['embedding_dim'],
        args['hidden_dim'],
        problem,
        n_encode_layers=args['n_encode_layers'],
        mask_inner=True,
        mask_logits=True,
        normalization=args['normalization'],
        tanh_clipping=args['tanh_clipping'],
        checkpoint_encoder=args.get('checkpoint_encoder', False),
        shrink_size=args.get('shrink_size', None),
        mom_vehicle_size=args['mom_size'],
        sub_vehicle_size=args['sub_size'],
        sychonization_cost=args['sychonization_cost'],
        use_vehicle_context=args['use_vehicle_context'],
        designated_driver=args['designated_driver'],
        use_cross_attention=args['use_cross_attention'],
        dist_threshold=args['dist_threshold'] if 'dist_threshold' in args else None,
        time_threshold=args['time_threshold'] if 'time_threshold' in args else None,
        encoder_variant=args.get('encoder_variant', 'standard'),
        decoder_variant=args.get('decoder_variant', 'standard'),
        decoder_ablation=args.get('decoder_ablation', 'none'),
        request_encoder=args.get('request_encoder', 'vehicle_encoder'),
    )
    model.cache_cross_query = bool(args.get('cache_cross_query', False))
    model, *_ = _load_model_file(model_filename, model)

    model.eval()  # Put in eval mode

    return model, args


def parse_softmax_temperature(raw_temp):
    # Load from file
    if os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def repeat_nvsp_batch(v, n, device, batch_size):
    if isinstance(v, dict):
        return {k: repeat_nvsp_batch(v_, n, device, batch_size) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [repeat_nvsp_batch(v_, n, device, batch_size) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(repeat_nvsp_batch(v_, n, device, batch_size) for v_ in v)
    elif isinstance(v, int):
        return torch.full((batch_size*n, 1), fill_value=v, device=device)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    """
    :param input: (batch_size, graph_size, node_dim) input node features
    :return:
    """
    input = do_batch_rep(input, batch_rep)

    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)
        # pi.view(-1, batch_rep, pi.size(-1))
        cost, mask = get_cost_func(input, pi)

        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)
    # (batch_size * batch_rep, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat(
        [F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis],
        1
    )  # .view(embeddings.size(0), batch_rep * iter_rep, max_length)
    costs = torch.cat(costs, 1)

    # (batch_size)
    mincosts, argmincosts = costs.min(-1)
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]

    return minpis, mincosts



def _actions_to_vehicle_sequences(sequences, vehicle_size):
    batch_size, steps, _ = sequences.shape
    result = torch.full((batch_size, vehicle_size, steps), -1,
                        dtype=torch.long, device=sequences.device)
    counts = torch.zeros((batch_size, vehicle_size), dtype=torch.long, device=sequences.device)
    batch = torch.arange(batch_size, device=sequences.device)
    for step in range(steps):
        task, daughter, mother = sequences[:, step].unbind(-1)
        result[batch, daughter, counts[batch, daughter]] = task
        result[batch, mother, counts[batch, mother]] = task
        counts[batch, daughter] += 1
        counts[batch, mother] += 1
    return result


def sample_many_nvsp(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1, attention_model=None):
    """
    :param input: (batch_size, graph_size, node_dim) input node features
    :return:
    """
    from problems.nvsp.problem import NVSP
    from problems.nvsp.state import omega
    start = time.time()
    new_input = None
    if attention_model is not None and attention_model.is_nvsp:
            attention_model.embeddings_with_depot = torch.cat((input[1][0], input[1][2]), dim=1).repeat(batch_rep, 1, 1)
            new_input = (input[0], input[1][0])  # 取 dict 和 tuple 的第一个元素
        # else:
        #     attention_model.embeddings_with_depot = input[1][0].repeat(batch_rep, 1, 1)
        #     new_input = (input[0], input[1][0][:, :-1, :])  # 取 dict 和 tuple 的第一个元素
    input = repeat_nvsp_batch(new_input, batch_rep, batch_size=new_input[1].size(0), device=new_input[1].device)
    costs = []
    pis = []
    for i in range(iter_rep):
        state = NVSP.make_state(input[0])
        input = input + (state,)
        _log_p, pi, state = inner_func(input)
        # pi.view(-1, batch_rep, pi.size(-1))
        cost, mask = state.get_costs(input[0], pi)
        if attention_model.is_nvsp and attention_model.is_sychonization_cost:
            cost_dist = cost[0]
            cost_delay = cost[1]
            cost = omega * cost_dist + (1 - omega) * cost_delay
            # print("cost_dist: " + str(cost_dist.t()))
            # print("cost_delay: " + str(cost_delay.t()))
        vehicle_size = input[0]['sub_size'][0].item() + input[0]['mom_size'][0].item()
        sequences = _actions_to_vehicle_sequences(pi, vehicle_size)
        costs.append(cost.view(batch_rep, -1).t())
        pis.append(sequences.view(batch_rep, -1, sequences.size(1), sequences.size(-1)).transpose(0, 1))

    usgae_time = time.time() - start
    # print("sample_many time: ", usgae_time)
    max_length = max(pi.size(1) for pi in pis)
    # (batch_size * batch_rep, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat(
        [F.pad(pi, (0, max_length - pi.size(1))) for pi in pis],
        1
    )  # .view(embeddings.size(0), batch_rep * iter_rep, max_length)
    costs = torch.cat(costs, 1)

    # (batch_size)
    mincosts, argmincosts = costs.min(-1)
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]
    min_cost_delay = cost_delay[argmincosts]
    mim_cost_dist = cost_dist[argmincosts]
    mean_cost_delay = cost_delay.mean()
    mean_cost_dist = cost_dist.mean()
    mean_cost = costs.mean()
    best_cost = mincosts.item()
    data_package = [minpis.tolist(), cost_dist.tolist(), cost_delay.tolist(), usgae_time, min_cost_delay.item(), mim_cost_dist.item(), mean_cost_delay.item(), mean_cost_dist.item(), mean_cost.item(), mincosts.item()]
    # print("min cost delay : ", min_cost_delay)
    # print("min cost dist : ", mim_cost_dist)
    # print("mean cost delay : ", mean_cost_delay)
    # print("mean cost dist : ", mean_cost_dist)
    # print("mean cost : ", mean_cost)

    # from matplotlib import pyplot as plt
    # from seaborn.external.kde import gaussian_kde
    # from matplotlib.colors import LogNorm
    # import pandas as pd
    # length = input[0]['designated_time'].size(1) + 1
    # tar_loc = state.tar_loc
    # sor_loc = torch.cat(
    #     (state.sor_loc, torch.zeros(state.sor_loc.size(0), 1, 2, dtype=torch.float32, device=state.sor_loc.device)),
    #     dim=1
    # )
    # sor = np.array(sor_loc.tolist()[0])  # n x 2
    # tar = np.array(tar_loc.tolist()[0])  # n x 2
    # start_t = np.array(Start_T)  # n
    # end_t = np.array(End_T)  # n
    #
    # des_time_tensor = input[0]['designated_time']
    # des_time = des_time_tensor * des_time_tensor.size(1)
    # des_time = np.append(np.array(des_time.tolist()).reshape(-1), 0)
    #
    # # attention 分数预处理（统一长度）
    # Aij_processed = []
    # for elem in Aij:
    #     if isinstance(elem, list):
    #         arr = elem[:length] + [0] * (length - len(elem))
    #     elif isinstance(elem, (int, float)):
    #         arr = [float(elem)] + [0] * (length - 1)
    #     else:
    #         raise ValueError(f"Unexpected element type: {type(elem)}")
    #     Aij_processed.append(arr)
    #
    # aij = np.array(Aij_processed)
    #
    # # 构造 Pij 与 attention 列表
    # Pij_list, Aij_list = [], []
    #
    # for i in range(length):
    #     for j in range(length):
    #         if i == j:
    #             continue
    #
    #         dij = np.linalg.norm(tar[i] - sor[j])
    #         # pij = (des_time[j] - (end_t[i] - start_t[i]) - (end_t[j] - start_t[j])) - dij  # si, sj 设为 0
    #         pij = (des_time[i] - des_time[j]) / (dij / 0.4)
    #         aij_val = aij[i][j]
    #
    #         Pij_list.append(pij)
    #         Aij_list.append(aij_val)
    #
    # Pij_array = np.array(Pij_list)
    # Aij_array = np.array(Aij_list)
    #
    # df = pd.DataFrame({
    #     'Pij_Surplus_Time': Pij_array,
    #     'Aij_Attention_Score': Aij_array
    # })
    # df_filtered = df[df['Aij_Attention_Score'] != 0].copy()
    # # 3. 将 'Pij_Surplus_Time' 列的值归一化到 0~1 范围
    # pij_col = 'Pij_Surplus_Time'
    # min_val = df_filtered[pij_col].min()
    # max_val = df_filtered[pij_col].max()
    # if max_val - min_val == 0:
    #     df_filtered['Pij_Time_Normalized'] = 0.5  # 如果所有值都一样，则归一化为0.5
    # else:
    #     df_filtered['Pij_Time_Normalized'] = (df_filtered[pij_col] - min_val) / (max_val - min_val)
    # df_filtered.to_excel("exp/pij_aij_data.xlsx", sheet_name='Attention_Data', index=False)
    # print("数据保存成功。")
    #
    # if not os.path.exists("exp/pij_aij_data.xlsx"):
    #     print(f"错误：文件 'exp/pij_aij_data.xlsx' 不存在。")
    #     return
    #
    # df = pd.read_excel("exp/pij_aij_data.xlsx")
    # print("数据读取成功。")
    #
    # pij_data = df['Pij_Time_Normalized']
    # aij_data = df['Aij_Attention_Score']
    #
    # # --- 绘图部分 ---
    # # 计算点密度，用于颜色映射
    # xy = np.vstack([pij_data, aij_data])
    # z = gaussian_kde(xy)(xy)
    #
    # # 根据密度排序点，使得高密度点显示在前面，低密度点显示在后面
    # idx = z.argsort()
    # x, y, z = pij_data[idx], aij_data[idx], z[idx]
    #
    # plt.figure(figsize=(7, 5))
    # scatter = plt.scatter(x, y, c=z, cmap='Blues', alpha=0.6, s=15, edgecolors='none')
    # plt.yscale('log')  # 设置Y轴为对数刻度
    # plt.xlabel('$P_{i,j}$ (Surplus Time)', fontsize=12)
    # plt.ylabel('Attention Score $A_{i,j}$', fontsize=12)
    # plt.title('Distribution of $A_{i,j}$ over $P_{i,j}$', fontsize=14)
    # plt.grid(True, linestyle='--', alpha=0.3)
    # plt.colorbar(scatter, label='Density')
    # plt.tight_layout()
    # plt.show()

    return minpis.tolist(), best_cost, data_package
