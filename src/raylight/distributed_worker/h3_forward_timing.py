"""Per-forward MiniMax H3 timing inside Ray workers.

Enabled by RAYLIGHT_H3_FORWARD_TIMING=1 in the ComfyUI process (forwarded to the
workers). Prints the same "PIPELINE TIME: Time taken to H3_FORWARD/ ..." line as
headless_minimax_h3.py, with the worker rank added, because the DiT runs in the
worker processes where the runner's MiniMaxH3Model.forward patch does not exist.
"""

import functools
import json
import os
import sys
import time

ENV_NAME = "RAYLIGHT_H3_FORWARD_TIMING"


def _synchronize(x):
    import torch

    device = x.device
    if device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _fusion_stats():
    # ComfyUI-OmniXPU H3 route counters (cumulative), loaded in XPU workers by raylight_worker_setup.
    for name, module in list(sys.modules.items()):
        if name.endswith("adapters.h3_rms_modulation") and hasattr(module, "get_stats"):
            return module.get_stats()
    return None


def _delta(after, before):
    delta = {}
    for key, value in after.items():
        old = (before or {}).get(key)
        if isinstance(value, dict):
            old = old or {}
            delta[key] = {r: n - old.get(r, 0) for r, n in value.items() if n - old.get(r, 0)}
        else:
            delta[key] = value - (old or 0)
    return delta


def install(rank):
    if os.environ.get(ENV_NAME) != "1":
        return
    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model
    except ImportError:
        return
    original = MiniMaxH3Model.forward
    if getattr(original, "_raylight_timed", False):
        return
    count = 0

    @functools.wraps(original)
    def timed_forward(self, x, timestep, context, *args, **kwargs):
        nonlocal count
        count += 1
        _synchronize(x[0])
        stats_before = _fusion_stats()
        start = time.time_ns()
        try:
            return original(self, x, timestep, context, *args, **kwargs)
        finally:
            _synchronize(x[0])
            end = time.time_ns()
            elapsed = end - start
            print(
                f"PIPELINE TIME: Time taken to H3_FORWARD/ rank={rank} call={count} "
                f"video={tuple(x[0].shape)} audio={tuple(x[1].shape)} timestep={tuple(timestep.shape)} "
                f"context={tuple(context.shape)}: "
                f"{elapsed // 1_000_000_000}.{elapsed % 1_000_000_000:09d} sec, "
                f"begin: {start // 1_000_000_000}.{start % 1_000_000_000:09d}, "
                f"end: {end // 1_000_000_000}.{end % 1_000_000_000:09d}",
                flush=True,
            )
            stats_after = _fusion_stats()
            if stats_after is not None:
                print(f"H3_FUSION_STATS/ rank={rank} call={count} "
                      f"{json.dumps(_delta(stats_after, stats_before), sort_keys=True)}", flush=True)

    timed_forward._raylight_timed = True
    MiniMaxH3Model.forward = timed_forward
