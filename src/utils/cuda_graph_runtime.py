"""Optional whole-forward / forward-backward CUDA graphs for fixed NVSP shapes.

Graph objects stay outside nn.Module, so deepcopy and checkpoints contain only
the ordinary model. Optimizer, clipping and GradScaler updates remain eager.
"""
import torch
from weakref import WeakKeyDictionary

from problems.nvsp.state import omega
from utils.data_utils import MAX_VALUES


_rollout_graphs = WeakKeyDictionary()


def get_rollout_graph(model, batch, context_factory, amp_key):
    key = (tuple((name, tuple(value.shape), value.dtype, value.device) for name, value in sorted(batch.items())),
           tuple(sorted(MAX_VALUES.items())), amp_key, model.temp)
    cache = _rollout_graphs.setdefault(model, {})
    if key not in cache:
        cache[key] = RolloutGraph(model, batch, context_factory)
    return cache[key]


def clone_batch(batch):
    return {key: value.clone() for key, value in batch.items()}


def copy_batch(target, source):
    for key, value in source.items():
        target[key].copy_(value, non_blocking=True)


def graph_compatible(model):
    return (getattr(model, 'is_nvsp', False)
            and getattr(model, 'vectorized_state', False)
            and getattr(model, 'shrink_size', None) is None
            and not getattr(model, 'debug_checks', True)
            and getattr(model, 'compile_mode', 'default') == 'default')


class RolloutGraph:
    def __init__(self, model, batch, context_factory):
        self.batch = clone_batch(batch)
        stream = torch.cuda.current_stream(batch['sor_loc'].device)
        if stream.cuda_stream == 0:
            stream = torch.cuda.Stream(device=batch['sor_loc'].device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad(), torch.autocast(
                'cuda', enabled=False, cache_enabled=False):
            for _ in range(3):
                with context_factory():
                    model(self.batch, cost_only=True)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        # Outer evaluation autocast may already cache casts during warmup.
        # Disable its weight cache during capture, retaining the chosen dtype.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.no_grad(), torch.autocast(
                'cuda', enabled=False, cache_enabled=False), context_factory():
            self.cost = model(self.batch, cost_only=True)

    def __call__(self, batch):
        copy_batch(self.batch, batch)
        self.graph.replay()
        # Consumers must copy/use this result before the next replay.
        return self.cost


class TrainingGraph:
    def __init__(self, model, batch, baseline, opts, scaler, context_factory, loss_builder=None):
        self.batch, self.baseline = clone_batch(batch), baseline.clone()
        self.model, self.opts, self.scaler = model, opts, scaler

        def forward_backward():
            # Record weight casts in the graph so in-place optimizer updates
            # and baseline replacements cannot reuse an old autocast cache.
            with torch.autocast('cuda', enabled=False, cache_enabled=False), context_factory():
                costs, ll = model(self.batch)
                cost = omega * costs[0] + (1 - omega) * costs[1]
            reinforce = (((cost.float() - self.baseline) * ll.float()).mean()
                         if loss_builder is None else loss_builder(cost.float(), ll.float()))
            loss = reinforce
            if loss_builder is None and not (opts.use_cuda and opts.precision != 'fp32') and getattr(model, 'request_encoder', None) != 'tdrl_icra':
                loss = loss / opts.graph_size
            (scaler.scale(loss) if scaler is not None else loss).backward()
            return costs, cost, ll, reinforce, loss

        # Warmup must not alter BatchNorm statistics or consume training RNG.
        buffers = [(value, value.clone()) for value in model.buffers()]
        cpu_rng = torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state(self.batch['sor_loc'].device)
        stream = torch.cuda.current_stream(self.batch['sor_loc'].device)
        if stream.cuda_stream == 0:
            stream = torch.cuda.Stream(device=self.batch['sor_loc'].device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                forward_backward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for buffer, saved in buffers:
            buffer.copy_(saved)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng, self.batch['sor_loc'].device)
        model.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.outputs = forward_backward()
        self.grads = [(parameter, parameter.grad) for parameter in model.parameters()]
        self.routing_counts = [(module, module.routing_counts) for module in model.modules()
                               if getattr(module, 'dense_routing', False) and hasattr(module, 'routing_counts')]

    def __call__(self, batch, baseline):
        # Stream capture records kernels without executing the real batch.
        # Replay is required even for the first optimizer update.
        copy_batch(self.batch, batch)
        self.baseline.copy_(baseline, non_blocking=True)
        self.graph.replay()
        for parameter, gradient in self.grads:
            parameter.grad = gradient
        for module, counts in self.routing_counts:
            module.routing_counts = counts
        return self.outputs


class AccumulatedTrainingGraph:
    """Replay a smaller activation graph, then update the full batch once.

    Captured backward writes fresh gradients into the graph's fixed buffers.
    Separate accumulation buffers retain the weighted sum across replays.
    """
    def __init__(self, model, batch, baseline, opts, scaler, context_factory):
        self.microbatch = opts.train_microbatch_size
        if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()):
            raise ValueError('Microbatch accumulation changes BatchNorm statistics; use TrainingGraph')
        self.batch_size = baseline.size(0)
        if self.batch_size % self.microbatch:
            raise ValueError('Effective batch must be divisible by training microbatch')
        if getattr(model, 'request_encoder', None) != 'task_robot_cross':
            raise ValueError('Microbatch accumulation is qualified for task_robot_cross only')
        first = {key:value[:self.microbatch] for key,value in batch.items()}
        self.graph = TrainingGraph(model, first, baseline[:self.microbatch], opts, scaler, context_factory)
        self.parameters = [parameter for parameter,gradient in self.graph.grads if gradient is not None]
        self.accumulated = [torch.zeros_like(parameter) for parameter in self.parameters]

    def __call__(self, batch, baseline):
        torch._foreach_zero_(self.accumulated)
        distances, delays, likelihoods, reinforce_losses, losses = [], [], [], [], []
        count = self.batch_size // self.microbatch
        for start in range(0, self.batch_size, self.microbatch):
            end = start + self.microbatch
            values = self.graph({key:value[start:end] for key,value in batch.items()}, baseline[start:end])
            costs, cost, ll, reinforce, loss = values
            # Replay reuses output storage, so retain values before the next replay.
            distances.append(costs[0].detach().clone())
            delays.append(costs[1].detach().clone())
            likelihoods.append(ll.detach().clone())
            reinforce_losses.append(reinforce.detach().clone())
            losses.append(loss.detach().clone())
            torch._foreach_add_(self.accumulated, [parameter.grad for parameter in self.parameters], alpha=1./count)
        for parameter,gradient in zip(self.parameters,self.accumulated):
            parameter.grad = gradient
        costs = torch.cat(distances), torch.cat(delays)
        cost = omega * costs[0] + (1 - omega) * costs[1]
        return costs, cost, torch.cat(likelihoods), torch.stack(reinforce_losses).mean(), torch.stack(losses).mean()
