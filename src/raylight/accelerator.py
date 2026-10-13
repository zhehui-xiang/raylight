"""Device helpers so Raylight runs on CUDA/ROCm (NCCL) and Intel XPU (XCCL)."""

import os

import torch


# oneCCL defaults for Ray workers, as in intel/llm-scaler-omni 0.1.0-b8.
# Values exported in the ComfyUI process take precedence.
_XPU_CCL_ENV_DEFAULTS = {
    "CCL_SYCL_ALLTOALL_ARC_LL": "1",
    "CCL_SYCL_CCL_BARRIER": "1",
}


def is_xpu():
    if torch.cuda.is_available():
        return False
    xpu = getattr(torch, "xpu", None)
    return xpu is not None and xpu.is_available()


def device_type():
    return "xpu" if is_xpu() else "cuda"


def device_module():
    return getattr(torch, device_type())


def device_count():
    return device_module().device_count()


def dist_backend():
    return "xccl" if is_xpu() else "nccl"


def current_device():
    return torch.device(device_type(), device_module().current_device())


def synchronize():
    device_module().synchronize()


def empty_cache():
    device_module().empty_cache()


def compute_capability():
    if is_xpu():
        # Only consumed by CUDA kernel selection; XPU never matches those paths.
        return 0
    return int("{}{}".format(*torch.cuda.get_device_capability()))


def actor_num_gpus():
    # Ray only detects NVIDIA/AMD GPUs, so XPU actors are placed without a GPU
    # resource and pinned to a device explicitly via actor_device_id().
    return 0 if is_xpu() else 1


def actor_device_id(worker_index, parallel_dict):
    """Device index a worker should bind to.

    CUDA workers see only their Ray-assigned GPU, so it is always 0. XPU workers
    see every card (oneCCL hangs under a narrow ZE_AFFINITY_MASK) and select one.
    """
    if not is_xpu():
        return 0
    device_ids = parallel_dict.get("device_ids") or list(range(device_count()))
    return device_ids[worker_index]


def bind_worker_device(device_id):
    if is_xpu():
        torch.xpu.set_device(device_id)
        return torch.device("xpu", device_id)
    return torch.device("cuda", device_id)


def worker_env_vars():
    """Environment the Ray workers need on top of Raylight's runtime_env."""
    if not is_xpu():
        return {}
    return {key: os.environ.get(key, value) for key, value in _XPU_CCL_ENV_DEFAULTS.items()}
