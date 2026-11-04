#!/usr/bin/env python
import copy
import os
import json
import pprint as pp
import random
import datetime
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
try:
    from tensorboardX import SummaryWriter as TbLogger
except ModuleNotFoundError:
    try:
        from torch.utils.tensorboard import SummaryWriter as TbLogger
    except ModuleNotFoundError:
        TbLogger = None
from options import get_options
from problems.nvsp.state import _calculate_new_state_jit
from train import train_epoch, validate, get_inner_model
from utils.training_performance import configure_precision, configure_speed_profile
from utils.training_resume import checkpoint_epoch, create_epoch_scheduler, preserve_cpu_rng
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.flat_decoder import FlatDecoder
from nets.learning_baselines import LearningBaseline
from nets.tdrl_model import TDRLModel
from utils import (
    dist_utils,
    load_problem,
    torch_load_cpu,
)
import torch.distributed as dist
import gc

import logging


log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'train_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # 写入文件
    ]
)
# torch.set_printoptions(
#     profile="full",  # 完整显示（无省略）
#     edgeitems=3,     # 边缘显示 3 行/列（平衡可读性）
#     linewidth=120,   # 每行宽度（避免换行混乱）
#     precision=0,     # 小数位数（减少冗余输出）
# )

# torch.autograd.set_detect_anomaly(True)
def run(opts):
    configure_speed_profile(opts)
    if hasattr(opts, '_training_graph'):
        del opts._training_graph
    opts._used_training_graph = False
    # Keep Autograd's gradient accumulators on the same non-default stream
    # used for graph capture, including warmup epochs and later validation.
    try:
        if opts.use_cuda and (getattr(opts, 'cuda_graph_train', False) or getattr(opts, 'cuda_graph_rollout', False)):
            with torch.cuda.stream(torch.cuda.Stream()):
                return _run(opts)
        return _run(opts)
    finally:
        if hasattr(opts, '_training_graph'):
            del opts._training_graph


def _run(opts):
    # optimized_calculate_func = torch.compile(_calculate_new_state_jit, mode="max-autotune")
    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)
    if getattr(opts, 'cpu_threads', None) is not None:
        torch.set_num_threads(opts.cpu_threads)
    configure_precision(opts, reset=True)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard and TbLogger is not None:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))
    elif not opts.no_tensorboard:
        print("TensorBoard logging is unavailable; continuing with no_tensorboard=True.")
        opts.no_tensorboard = True

    if not os.path.isdir(opts.save_dir):
        os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Initialize model
    model_class = {
        'am': LearningBaseline, 'hadrl': LearningBaseline, 'tdrl': TDRLModel,
        'attention': FlatDecoder if opts.decoder_variant == 'flat_decoder' else AttentionModel,
    }.get(opts.model, None)
    assert model_class is not None, f"Unknown model: {model_class}"
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        mom_vehicle_size=opts.mom_size,
        sub_vehicle_size=opts.sub_size,
        sychonization_cost=opts.sychonization_cost,
        use_vehicle_context=opts.use_vehicle_context,
        designated_driver=opts.designated_driver,
        use_cross_attention=opts.use_cross_attention,
        dist_threshold=opts.dist_threshold,
        time_threshold=opts.time_threshold,
        encoder_variant=getattr(opts, 'encoder_variant', 'standard'),
        decoder_variant=getattr(opts, 'decoder_variant', 'standard'),
        decoder_ablation=getattr(opts, 'decoder_ablation', 'none'),
        request_encoder=getattr(opts, 'request_encoder', 'vehicle_encoder'),
    )
    model.debug_checks = getattr(opts, 'debug_checks', False)
    if opts.checkpoint_dynamic_steps is None:
        opts.checkpoint_dynamic_steps = model.request_encoder in ('task_robot_cross', 'tdrl', 'tdrl_icra') and opts.graph_size >= 50
    model.checkpoint_dynamic_steps = opts.checkpoint_dynamic_steps
    microbatch = getattr(opts, 'train_microbatch_size', None)
    if microbatch is not None:
        if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()):
            raise ValueError('BatchNorm encoders require the full training batch; disable train_microbatch_size')
        if microbatch <= 0 or opts.batch_size % microbatch:
            raise ValueError('Training microbatch must be positive and divide effective batch_size')
        if not (opts.request_encoder == 'task_robot_cross' and opts.precision == 'bf16'
                and opts.use_cuda and opts.cuda_graph_train):
            raise ValueError('Microbatch runtime requires task_robot_cross, BF16 and CUDA training graphs')
    model.compile_decoder = getattr(opts, 'compile_decoder', False)
    model.compile_steps = getattr(opts, 'compile_steps', False)
    model.compile_chunk_size = getattr(opts, 'compile_chunk_size', None) or 1
    model.compile_mode = getattr(opts, 'compile_mode', 'default')
    model.fast_sampling = getattr(opts, 'fast_sampling', False)
    model.vectorized_state = getattr(opts, 'vectorized_state', False)
    model.compile_encoder = getattr(opts, 'compile_encoder', False)
    model.cache_request_context = getattr(opts, 'fast_train', False) and model.decoder_variant == 'nvmdn'
    model.cache_cross_query = (bool(getattr(opts, 'cache_cross_query', False))
                              and model.request_encoder == 'task_robot_cross')
    if model.compile_steps:
        model.compile_decoder = False
    model.use_sdpa = getattr(opts, 'fast_train', False)
    for module in model.modules():
        module.use_sdpa = getattr(opts, 'fast_train', False)
        module.cache_relations = getattr(opts, 'fast_train', False)
    from utils.training_performance import configure_attention_performance
    configure_attention_performance(model, opts)
    if getattr(opts, 'cuda_graph_train', False) or getattr(opts, 'cuda_graph_rollout', False):
        from utils.cuda_graph_runtime import graph_compatible
        if not graph_compatible(model):
            print('CUDA graph disabled for this model configuration: requires vectorized_state, default compile mode, '
                  'and compatible fixed-size batches')
        elif getattr(opts, 'cuda_graph_train', False) and not model.fast_sampling:
            print('CUDA training graph disabled: enable fast_sampling')

    opts.distributed = False  # 默认关闭

    if opts.use_cuda and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()

        # 🚩 如果分布式训练
        if num_gpus > 1 and opts.launcher != 'none' and False:

            opts.local_rank = int(os.environ["LOCAL_RANK"])
            opts.rank = int(os.environ["RANK"])
            opts.world_size = int(os.environ["WORLD_SIZE"])
            opts.device = torch.device(f"cuda:{opts.local_rank}")
            torch.cuda.set_device(opts.local_rank)
            opts.distributed = True

            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=opts.local_rank,
                world_size=opts.world_size
            )


            if getattr(opts, 'sync_bn', True):
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if opts.local_rank == 0:
                    print(f'Using Synchronized BatchNorm ... world_size = {opts.world_size}')

            model = model.to(opts.device)

            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[opts.local_rank],
                output_device=opts.local_rank,
                find_unused_parameters=True
            )

            print(f"[rank{opts.local_rank}] Using Distributed Data Parallel on GPU {opts.local_rank}, rank {opts.rank}.")

        else:
            # 🚩 fallback 到单机多卡 DataParallel（不推荐，但可以保留）
            opts.device = torch.device("cuda:0")
            model = model.to(opts.device)
            if num_gpus > 1:
                model = nn.DataParallel(model)
                print("Using Data Parallel on multiple GPUs.")
            else:
                print("Using a single CUDA GPU.")

    else:
        opts.device = torch.device("cpu")
        model = model.to(opts.device)
        print("Using CPU mode.")
    # Overwrite model parameters by parameters to load
    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)
    model_ = get_inner_model(model)
    if load_data.get('model'):
        from utils.functions import validate_encoder_checkpoint
        validate_encoder_checkpoint(model_, load_data)
    loaded_model_state = load_data.get('model', {})
    model_.load_state_dict({**model_.state_dict(), **loaded_model_state})


    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        # Loading a rollout baseline performs an extra greedy evaluation. Keep
        # its loader seed draw from changing the original validation dataset.
        with preserve_cpu_rng():
            baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        ),
        **({'fused': True} if opts.use_cuda and getattr(opts, 'fast_train', False) else {})
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        if opts.use_cuda and getattr(opts, 'fast_train', False):
            for group in optimizer.param_groups:
                group['fused'] = True
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)
    from utils import ModelData
    if ModelData.use_scaler and load_data.get('scaler'):
        ModelData.scaler.load_state_dict(load_data['scaler'])

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = create_epoch_scheduler(optimizer, opts.lr_decay, opts.resume)

    # Start the actual training loop
    if opts.problem == 'nvsp':
        val_dataset = problem.make_dataset(
            size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution,
            mom_vehicle_size=opts.mom_size, sub_vehicle_size=opts.sub_size, designated_driver=opts.designated_driver,
            **({'compact': True} if opts.problem == 'nvsp' and getattr(opts, 'fast_train', False) else {}))
    else:
        val_dataset = problem.make_dataset(
            size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

    if opts.resume:
        epoch_resume = checkpoint_epoch(opts.resume)

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Checkpointing precedes validation and the baseline callback. Replay
        # both in their original order, including validation's loader seed draw.
        validate(model, val_dataset, opts)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1


    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )


if __name__ == "__main__":
    run(get_options())
