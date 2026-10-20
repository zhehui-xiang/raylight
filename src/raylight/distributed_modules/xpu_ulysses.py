"""Ulysses sequence parallelism tuned for Intel XPU (Arc B-series over PCIe, XCCL).

Two measured problems with the generic yunchang path on 2-4x Arc Pro B70:
  * oneCCL's all-to-all for more than two ranks is 4-5x slower than running
    the same exchange as N-1 rounds of paired send/recv.
  * yunchang reshapes/copies around every all-to-all and its local attention
    cannot use OmniXPU's CUTE kernel, so per-layer cost exceeded one card.
"""

import os
import types

import torch
import torch.distributed as dist

try:
    from omni_xpu_kernel import cute as _cute

    if not (_cute.is_available() and _cute.supports_d128_bhld()):
        _cute = None
except Exception:
    _cute = None


def _use_rounds(group):
    if os.environ.get("RAYLIGHT_XPU_ALL_TO_ALL", "rounds") != "rounds":
        return False
    return dist.get_world_size(group) > 2 and dist.get_backend(group) == "xccl"


def all_to_all_single(output, input, group=None):
    """dist.all_to_all_single with equal splits along dim 0."""
    if not _use_rounds(group):
        dist.all_to_all_single(output, input, group=group)
        return
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    inputs, outputs = input.chunk(world), output.chunk(world)
    outputs[rank].copy_(inputs[rank])
    # Shift schedule: in round i every rank sends to rank+i and receives from rank-i.
    for step in range(1, world):
        dst = dist.get_global_rank(group, (rank + step) % world) if group is not None else (rank + step) % world
        src = dist.get_global_rank(group, (rank - step) % world) if group is not None else (rank - step) % world
        ops = [dist.P2POp(dist.isend, inputs[(rank + step) % world], dst, group),
               dist.P2POp(dist.irecv, outputs[(rank - step) % world], src, group)]
        for request in dist.batch_isend_irecv(ops):
            request.wait()


def patch_yunchang_all_to_all():
    """Route yunchang's Ulysses all-to-all through all_to_all_single above."""
    import yunchang.comm.all_to_all as yunchang_all_to_all

    if isinstance(yunchang_all_to_all.dist, types.SimpleNamespace):
        return
    proxy = types.SimpleNamespace(**{name: getattr(dist, name) for name in dir(dist) if not name.startswith("__")})
    proxy.all_to_all_single = lambda output, input, group=None, **kwargs: all_to_all_single(output, input, group)
    yunchang_all_to_all.dist = proxy


# oneCCL caches an IPC mapping for every buffer it exchanges and keeps it after
# PyTorch frees the memory, so empty_cache leaks device memory unless collectives
# keep reusing the same buffers.
_buffers = {}


def _buffer(name, shape, like):
    buffer = _buffers.get(name)
    if buffer is None or buffer.shape != shape or buffer.dtype != like.dtype or buffer.device != like.device:
        buffer = torch.empty(shape, dtype=like.dtype, device=like.device)
        _buffers[name] = buffer
    return buffer


def _local_attention(q, k, v):
    # q/k/v: (1, heads, S, d) backed by dense (1, S, heads, d) storage.
    if _cute is not None and q.dtype == torch.bfloat16 and q.shape[-1] == 128:
        return _cute.sdp_bhld_d128(q, k, v)
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def attention(q, k, v, group):
    """Full-sequence attention over a sequence-sharded batch-1 input.

    q/k/v: (s_local, heads, head_dim) per rank, sequence order by rank.
    Returns (s_local, heads * head_dim).
    """
    world = dist.get_world_size(group)
    s, heads, head_dim = q.shape
    heads_local = heads // world
    shape = (world, s, heads_local, head_dim)
    send = _buffer("send", shape, q)
    gathered = []
    for name, t in (("q", q), ("k", k), ("v", v)):
        send.copy_(t.unflatten(1, (world, heads_local)).transpose(0, 1))  # (N, s, h/N, d)
        recv = _buffer(name, shape, q)
        all_to_all_single(recv, send, group)
        # Received chunks concatenate to (S, h/N, d): BLHD storage, consumed without a copy.
        gathered.append(recv.view(1, world * s, heads_local, head_dim).transpose(1, 2))
    out = _local_attention(*gathered)
    send.view(1, world * s, heads_local, head_dim).copy_(out.transpose(1, 2))
    recv = _buffers["q"]  # q is consumed; reuse its buffer for the output exchange
    all_to_all_single(recv, send, group)
    return recv.transpose(0, 1).reshape(s, heads * head_dim)  # copies out of the reused buffer


def all_gather(x, group):
    """All-gather along dim 0 into a reused buffer, valid until the next call."""
    send = _buffer("gather_send", x.shape, x)
    send.copy_(x)
    recv = _buffer("gather_recv", (dist.get_world_size(group) * x.shape[0], *x.shape[1:]), x)
    dist.all_gather_into_tensor(recv, send, group=group)
    return recv
