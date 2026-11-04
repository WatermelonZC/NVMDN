"""Fused attention for vehicle attention's short vehicle sequences (optional CUDA/Triton).

No parameter or checkpoint format changes. CPU and unsupported installations
keep the ordinary PyTorch implementation. Import only when explicitly enabled.
"""
from typing import Optional
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(Q, K, V, M, O, P,
             qb: tl.constexpr, qh: tl.constexpr, qn: tl.constexpr, qd: tl.constexpr,
             kb: tl.constexpr, kh: tl.constexpr, kn: tl.constexpr, kd: tl.constexpr,
             vb: tl.constexpr, vh: tl.constexpr, vn: tl.constexpr, vd: tl.constexpr,
             mb: tl.constexpr, mh: tl.constexpr, mn: tl.constexpr, mk: tl.constexpr,
             H: tl.constexpr, N: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
             BN: tl.constexpr, BL: tl.constexpr, BD: tl.constexpr, MASK: tl.constexpr):
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    rows, cols, dims = tl.arange(0, BN), tl.arange(0, BL), tl.arange(0, BD)
    q = tl.load(Q + b * qb + h * qh + rows[:, None] * qn + dims[None, :] * qd,
                (rows[:, None] < N) & (dims[None, :] < D), 0)
    k = tl.load(K + b * kb + h * kh + cols[:, None] * kn + dims[None, :] * kd,
                (cols[:, None] < L) & (dims[None, :] < D), 0)
    v = tl.load(V + b * vb + h * vh + cols[:, None] * vn + dims[None, :] * vd,
                (cols[:, None] < L) & (dims[None, :] < D), 0)
    scores = tl.dot(q, tl.trans(k), input_precision='ieee') * (D ** -0.5)
    allowed = cols[None, :] < L
    if MASK:
        allowed = allowed & tl.load(M + b * mb + h * mh + rows[:, None] * mn + cols[None, :] * mk,
                                    (rows[:, None] < N) & (cols[None, :] < L), 0)
    scores = tl.where(allowed, scores, -float('inf'))
    p = tl.exp(scores - tl.max(scores, 1)[:, None])
    p = p / tl.sum(p, 1)[:, None]
    out = tl.dot(p.to(v.dtype), v, input_precision='ieee')
    tl.store(O + bh * N * D + rows[:, None] * D + dims[None, :], out,
             (rows[:, None] < N) & (dims[None, :] < D))
    tl.store(P + bh * N * L + rows[:, None] * L + cols[None, :], p,
             (rows[:, None] < N) & (cols[None, :] < L))


@triton.jit
def _backward(Q, K, V, P, DO, DQ, DK, DV,
              qb: tl.constexpr, qh: tl.constexpr, qn: tl.constexpr, qd: tl.constexpr,
              kb: tl.constexpr, kh: tl.constexpr, kn: tl.constexpr, kd: tl.constexpr,
              vb: tl.constexpr, vh: tl.constexpr, vn: tl.constexpr, vd: tl.constexpr,
              ob: tl.constexpr, oh: tl.constexpr, on: tl.constexpr, od: tl.constexpr,
              H: tl.constexpr, N: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
              BN: tl.constexpr, BL: tl.constexpr, BD: tl.constexpr):
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    rows, cols, dims = tl.arange(0, BN), tl.arange(0, BL), tl.arange(0, BD)
    q = tl.load(Q + b * qb + h * qh + rows[:, None] * qn + dims[None, :] * qd,
                (rows[:, None] < N) & (dims[None, :] < D), 0)
    k = tl.load(K + b * kb + h * kh + cols[:, None] * kn + dims[None, :] * kd,
                (cols[:, None] < L) & (dims[None, :] < D), 0)
    v = tl.load(V + b * vb + h * vh + cols[:, None] * vn + dims[None, :] * vd,
                (cols[:, None] < L) & (dims[None, :] < D), 0)
    p = tl.load(P + bh * N * L + rows[:, None] * L + cols[None, :],
                (rows[:, None] < N) & (cols[None, :] < L), 0)
    do = tl.load(DO + b * ob + h * oh + rows[:, None] * on + dims[None, :] * od,
                 (rows[:, None] < N) & (dims[None, :] < D), 0)
    dp = tl.dot(do, tl.trans(v), input_precision='ieee')
    ds = p * (dp - tl.sum(dp * p, 1)[:, None]) * (D ** -0.5)
    dq = tl.dot(ds.to(k.dtype), k, input_precision='ieee')
    dk = tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision='ieee')
    dv = tl.dot(tl.trans(p.to(do.dtype)), do, input_precision='ieee')
    tl.store(DQ + bh * N * D + rows[:, None] * D + dims[None, :], dq,
             (rows[:, None] < N) & (dims[None, :] < D))
    tl.store(DK + bh * L * D + cols[:, None] * D + dims[None, :], dk,
             (cols[:, None] < L) & (dims[None, :] < D))
    tl.store(DV + bh * L * D + cols[:, None] * D + dims[None, :], dv,
             (cols[:, None] < L) & (dims[None, :] < D))


@torch.library.custom_op('cat_kernels::short_forward', mutates_args=())
def _short_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   mask: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    b, h, n, d = q.shape
    length = k.shape[2]
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    probabilities = torch.empty((b, h, n, length), dtype=torch.float32, device=q.device)
    mask_strides = (0, 0, 0, 0) if mask is None else mask.expand(b, h, n, length).stride()
    _forward[(b * h,)](q, k, v, q if mask is None else mask, out, probabilities,
                        *q.stride(), *k.stride(), *v.stride(), *mask_strides,
                        h, n, length, d, max(16, triton.next_power_of_2(n)),
                        max(16, triton.next_power_of_2(length)), max(16, triton.next_power_of_2(d)),
                        mask is not None, num_warps=4)
    return out, probabilities


@_short_forward.register_fake
def _fake_forward(q, k, v, mask):
    return torch.empty_like(q, memory_format=torch.contiguous_format), q.new_empty(
        (*q.shape[:3], k.shape[2]), dtype=torch.float32)


@torch.library.custom_op('cat_kernels::short_backward', mutates_args=())
def _short_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    p: torch.Tensor, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, h, n, d = q.shape
    length = k.shape[2]
    dq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    _backward[(b * h,)](q, k, v, p, grad, dq, dk, dv,
                         *q.stride(), *k.stride(), *v.stride(), *grad.stride(),
                         h, n, length, d, max(16, triton.next_power_of_2(n)),
                         max(16, triton.next_power_of_2(length)), max(16, triton.next_power_of_2(d)), num_warps=4)
    return dq, dk, dv


@_short_backward.register_fake
def _fake_backward(q, k, v, p, grad):
    return tuple(torch.empty_like(x, memory_format=torch.contiguous_format) for x in (q, k, v))


def _setup_context(ctx, inputs, output):
    q, k, v, mask = inputs
    ctx.save_for_backward(q, k, v, output[1])
    ctx.mark_non_differentiable(output[1])


def _autograd(ctx, grad, ignored):
    q, k, v, p = ctx.saved_tensors
    return (*_short_backward(q, k, v, p, grad), None)


_short_forward.register_autograd(_autograd, setup_context=_setup_context)


def short_attention(q, k, v, mask=None):
    return _short_forward(q, k, v, mask)[0]
