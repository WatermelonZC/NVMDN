"""Unbiased cross-instance control variate for the HA rollout advantage."""
import torch


def ha_reinforce_loss(cost, baseline, log_likelihood):
    if cost.shape != log_likelihood.shape or cost.ndim != 1:
        raise ValueError('HA policy loss requires aligned one-dimensional batch tensors')
    dtype = torch.float64 if log_likelihood.dtype == torch.float64 else torch.float32
    baseline = torch.as_tensor(baseline,device=cost.device,dtype=dtype).detach()
    if baseline.ndim == 0:
        baseline = baseline.expand_as(cost)
    if cost.shape != baseline.shape:
        raise ValueError('HA policy loss requires aligned one-dimensional batch tensors')
    advantage = cost.detach().to(dtype) - baseline
    count = advantage.numel()
    if count > 1:
        # Subtract the OTHER instances' mean residual. Their sampled actions
        # are independent of this instance's actions under HA's LayerNorm.
        # Including the B/(B-1) factor avoids batch-mean baseline bias.
        advantage = (count * advantage - advantage.sum()) / (count - 1)
    return (advantage * log_likelihood.to(dtype)).mean()
