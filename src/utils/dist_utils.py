import os

import torch
import torch.multiprocessing as mp
from torch import distributed as dist


def init_dist(launcher, backend='nccl', **kwargs):
    """
    Initialize the distributed environment.
    Args:
        launcher (str): The launcher type ('torchrun', 'pytorch', or 'none').
        backend (str): The backend for communication, default is 'nccl'.
        **kwargs: Additional arguments that can be passed to init_process_group.
    """
    if launcher == 'torchrun':
        # Using torchrun (recommended for newer PyTorch versions)
        dist.init_process_group(backend=backend, **kwargs)

    elif launcher == 'pytorch' or launcher == 'distributed.launch':
        # Using the older distributed.launch or pytorch launcher (deprecated)
        dist.init_process_group(backend=backend, **kwargs)

    elif launcher == 'none':
        # No distributed training, do nothing
        print("Distributed training is not enabled")
        return

    else:
        raise ValueError(f"Unsupported launcher type: {launcher}")

    # Log the distributed environment details
    print(f"Rank {dist.get_rank()}/{dist.get_world_size()} initialized using {backend} backend.")


def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)
    print(f'init distributed in rank {torch.distributed.get_rank()}')


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def reduce_tensor(tensor, args):
    '''
        for acc kind, get the mean in each gpu
    '''
    rt = tensor.to(args.device)
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def gather_tensor(tensor, args):
    output_tensors = [tensor.clone() for _ in range(args.world_size)]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat


def test_distributed(opts):
    if not dist.is_initialized():
        print(f"Rank {opts.local_rank}: Distributed not initialized!")
        return

    try:
        # 打印 NCCL 信息
        print(f"Rank {opts.local_rank}: NCCL version = {torch.cuda.nccl.version()}")

        # 确保所有 rank 到达此处
        print(f"Rank {opts.local_rank}: Reached before barrier")
        dist.barrier()
        print(f"Rank {opts.local_rank}: barrier pass")

        # 点对点通信（仅 rank 0 和 1）
        if opts.local_rank == 0:
            data = torch.tensor([1.0]).cuda()
            torch.distributed.send(data, dst=1)
            print(f"Rank 0 sent data to Rank 1")
        elif opts.local_rank == 1:
            data = torch.tensor([0.0]).cuda()
            torch.distributed.recv(data, src=0)
            print(f"Rank 1 received data: {data.item()}")
        else:
            print(f"Rank {opts.local_rank}: No P2P communication")

        # 再次同步，确保点对点通信完成
        dist.barrier()

        # 测试 all_reduce
        dummy = torch.tensor([opts.local_rank]).cuda()
        dist.all_reduce(dummy)
        print(f"Rank {opts.local_rank}: all_reduce result = {dummy.item()}")

    except Exception as e:
        print(f"Rank {opts.local_rank} crashed: {e}")
        raise
