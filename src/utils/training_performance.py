"""Training runtime settings and vectorized loading for compact NVSP data."""
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler

from utils import ModelData
from utils.functions import move_to


def configure_speed_profile(opts):
    """Resolve defaults after the caller has selected its model flags."""
    requested = getattr(opts, 'speed_profile', 'standard')
    import sys
    supported = (opts.use_cuda and sys.platform != 'win32' and torch.cuda.device_count() == 1
                 and opts.problem == 'nvsp' and opts.model in ('attention', 'tdrl', 'tdrl_icra')
                 and getattr(opts, 'compile_mode', 'default') == 'default'
                 and not getattr(opts, 'debug_checks', False)
                 and getattr(opts, 'shrink_size', None) is None)
    if requested == 'throughput' and not supported:
        raise ValueError('throughput requires one visible CUDA GPU, NVSP attention, default compile mode, '
                         'no shrink_size and no debug_checks; select a GPU UUID or use --speed_profile standard')
    enabled = requested == 'throughput' or (requested == 'auto' and supported)
    opts.resolved_speed_profile = 'throughput' if enabled else 'standard'
    if enabled:
        # Training, greedy baseline and FP32 validation specialize the same
        # decoder for different batch sizes, precision and tail lengths.
        # Their bounded set can exceed Dynamo's default of eight variants.
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
        opts.fast_train = opts.vectorized_state = opts.fast_sampling = True
        opts.compile_steps = opts.compile_encoder = True
        opts.prefetch_batches = opts.cuda_graph_train = opts.cuda_graph_rollout = True
        if (getattr(opts, 'use_cross_attention', False) and getattr(opts, 'use_vehicle_context', False)
                and getattr(opts, 'encoder_variant', 'standard') == 'nvmdn'):
            opts.optimized_attention = True
        if getattr(opts, 'cpu_threads', None) is None:
            opts.cpu_threads = 4
    if getattr(opts, 'preload_mb', None) is None:
        opts.preload_mb = 1024 if enabled else 0
    if getattr(opts, 'tensor_loader', None) is None:
        opts.tensor_loader = enabled or getattr(opts, 'fast_train', False)
    if getattr(opts, 'reuse_baseline', None) is None:
        opts.reuse_baseline = enabled
    if getattr(opts, 'compile_chunk_size', None) is None:
        # Only select multi-decision compilation automatically on the measured
        # vehicle attention backend. Other devices and model families keep single steps.
        measured_attention = (enabled and getattr(opts, 'optimized_attention', False)
                        and getattr(opts, 'attention_backend', 'auto') in ('auto', 'triton')
                        and torch.cuda.get_device_capability() == (7, 5))
        opts.compile_chunk_size = 4 if measured_attention else 1
    if getattr(opts, 'cache_cross_query', None) is None:
        opts.cache_cross_query = bool(enabled and getattr(opts, 'use_cross_attention', False)
                                     and getattr(opts, 'request_encoder', None) == 'task_robot_cross')


def configure_attention_performance(model, opts):
    if not getattr(opts, 'optimized_attention', False):
        return
    if getattr(model, 'decoder_variant', 'standard') == 'nvmdn':
        for module in model.modules():
            module.fused_relation_masks = True
        opts.resolved_attention_backend = 'sdpa'
        return
    if not (getattr(model, 'uses_cross_attention', False) and getattr(model, 'uses_vehicle_context', False)):
        raise ValueError('optimized_attention requires vehicle attention with the vehicle attention aggregator')
    if getattr(model, 'encoder_variant', 'nvmdn') != 'nvmdn':
        raise ValueError('Optimized attention requires the NVMDN encoder')
    import importlib.util
    import sys
    backend = getattr(opts, 'attention_backend', 'auto')
    supported = (opts.use_cuda and sys.platform != 'win32' and hasattr(torch.library, 'custom_op')
                 and importlib.util.find_spec('triton') is not None)
    if backend == 'auto':
        # Other architectures retain PyTorch's own SDPA selection until measured.
        backend = ('triton' if supported and torch.cuda.get_device_capability(getattr(opts, 'device', None)) == (7, 5)
                   else 'sdpa')
    if backend == 'triton' and not supported:
        raise ValueError('vehicle attention Triton backend requires Linux CUDA, Triton and PyTorch with torch.library.custom_op')
    for module in model.modules():
        module.packed_vehicle_attention = module.singleton_vehicle_attention = True
        module.fused_relation_masks = True
        module.small_attention_math = backend in ('math', 'triton')
        module.fused_vehicle_attention = backend == 'triton'
    opts.resolved_attention_backend = backend
    print('vehicle attention optimized attention backend: ' + backend)


def identity_batch(batch):
    return batch


def _pin_batch(batch):
    if torch.is_tensor(batch):
        return batch.pin_memory()
    return {key: _pin_batch(value) for key, value in batch.items()}


class TensorBatchLoader:
    """One permutation per epoch; slices for ordered evaluation, tensors for shuffle."""
    def __init__(self, dataset, batch_size, shuffle, drop_last, pin_memory):
        self.dataset, self.batch_size = dataset, batch_size
        self.shuffle, self.drop_last, self.pin_memory = shuffle, drop_last, pin_memory
        inner = getattr(dataset, 'dataset', dataset)
        devices = {value.device for value in inner.data.values()}
        if torch.is_tensor(getattr(dataset, 'baseline', None)):
            devices.add(dataset.baseline.device)
        self.index_device = devices.pop() if len(devices) == 1 else torch.device('cpu')

    def __len__(self):
        count = len(self.dataset)
        return count // self.batch_size if self.drop_last else (count + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Match DataLoader's base seed draw and RandomSampler's independent RNG.
        # The ordering and subsequent global CPU RNG state remain unchanged.
        torch.empty((), dtype=torch.int64).random_().item()
        indices = None
        if self.shuffle:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(len(self.dataset), generator=generator).to(self.index_device)
        end = len(self.dataset) if not self.drop_last else len(self) * self.batch_size
        for start in range(0, end, self.batch_size):
            stop = min(start + self.batch_size, end)
            index = slice(start, stop) if indices is None else indices[start:stop]
            batch = self.dataset[index]
            yield _pin_batch(batch) if self.pin_memory else batch


def make_loader(dataset, batch_size, opts, shuffle=False, sampler=None, drop_last=False):
    inner = getattr(dataset, 'dataset', dataset)
    compact = getattr(inner, 'batched', False)
    resident = compact and any(value.is_cuda for value in inner.data.values())
    workers = 0 if resident else getattr(opts, 'num_workers', 0)
    kwargs = dict(num_workers=workers, pin_memory=opts.use_cuda and not resident)
    if workers:
        kwargs['persistent_workers'] = True
    if compact and workers == 0 and sampler is None and getattr(opts, 'tensor_loader', False):
        loader = TensorBatchLoader(dataset, batch_size, shuffle, drop_last, kwargs['pin_memory'])
    elif compact:
        if sampler is None:
            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        # Pass a whole list of indices to __getitem__, avoiding thousands of
        # Python dicts and tiny stack operations per batch.
        loader = DataLoader(dataset, batch_size=None,
                          sampler=BatchSampler(sampler, batch_size, drop_last),
                          collate_fn=identity_batch, **kwargs)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
                            sampler=sampler, drop_last=drop_last, **kwargs)
    if opts.use_cuda and getattr(opts, 'prefetch_batches', False):
        return DevicePrefetchLoader(loader, opts.device)
    return loader


def preload_dataset(dataset, opts):
    """Keep a compact epoch on GPU when the explicit budget leaves headroom.

    CPU source and sampling order are preserved. This function is called before
    baseline evaluation so both baseline and training reuse the same buffers.
    """
    budget = (getattr(opts, 'preload_mb', 0) or 0) * 1024 ** 2
    inner = getattr(dataset, 'dataset', dataset)
    if not opts.use_cuda or budget <= 0 or not getattr(inner, 'batched', False):
        return dataset
    needed = sum(x.numel() * x.element_size() for x in inner.data.values() if not x.is_cuda)
    labels = getattr(dataset, 'baseline', None)
    if torch.is_tensor(labels) and not labels.is_cuda:
        needed += labels.numel() * labels.element_size()
    if needed == 0:
        return dataset
    free, _ = torch.cuda.mem_get_info(opts.device)
    # Reserve at least half the currently free VRAM for activations/compiler.
    if needed > min(budget, free // 2):
        print(f'GPU preload skipped: dataset {needed / 1024**2:.1f} MiB exceeds budget/headroom')
        return dataset
    inner.data = {key: value.to(opts.device) for key, value in inner.data.items()}
    if hasattr(dataset, 'baseline') and torch.is_tensor(dataset.baseline):
        dataset.baseline = dataset.baseline.to(opts.device)
    print(f'GPU dataset resident: {needed / 1024**2:.1f} MiB')
    return dataset


def _record_stream(value, stream):
    if torch.is_tensor(value) and value.is_cuda:
        value.record_stream(stream)
    elif isinstance(value, dict):
        for item in value.values():
            _record_stream(item, stream)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _record_stream(item, stream)


class DevicePrefetchLoader:
    """Overlap the next pinned host batch copy with the current GPU work."""
    def __init__(self, loader, device):
        self.loader, self.device = loader, device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        stream = torch.cuda.Stream(device=self.device)
        iterator = iter(self.loader)

        def enqueue():
            try:
                host = next(iterator)
            except StopIteration:
                return None
            with torch.cuda.stream(stream):
                return move_to(host, self.device, non_blocking=True)

        pending = enqueue()
        while pending is not None:
            current = torch.cuda.current_stream(self.device)
            current.wait_stream(stream)
            batch = pending
            _record_stream(batch, current)
            pending = enqueue()
            yield batch


def configure_precision(opts, reset=False):
    precision = getattr(opts, 'precision', 'auto')
    enabled = opts.use_cuda and precision != 'fp32'
    bf16 = False
    if enabled:
        try:
            # Recent PyTorch also reports software-emulated BF16 support.
            # Turing should use native FP16 Tensor Cores instead.
            bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            # Older releases only expose the native-support query.
            bf16 = torch.cuda.is_bf16_supported()
    if precision == 'bf16' and enabled and not bf16:
        raise ValueError('This GPU does not support bfloat16. Use auto or fp16.')
    ModelData.amp_dtype = torch.bfloat16 if bf16 and precision in ('auto', 'bf16') else torch.float16
    ModelData.use_scaler = enabled and ModelData.amp_dtype == torch.float16
    if ModelData.use_scaler and (reset or ModelData.scaler is None):
        # torch.amp.GradScaler is only available in newer PyTorch releases.
        try:
            ModelData.scaler = torch.amp.GradScaler('cuda')
        except AttributeError:
            ModelData.scaler = torch.cuda.amp.GradScaler()
    return enabled


def autocast_context(opts, evaluation=False):
    enabled = opts.use_cuda and getattr(opts, 'precision', 'auto') != 'fp32'
    enabled = enabled and (not evaluation or getattr(opts, 'eval_amp', False))
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=ModelData.amp_dtype)
