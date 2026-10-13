"""Bring ComfyUI-OmniXPU into Ray workers on Intel XPU.

Ray workers do not load ComfyUI custom nodes, so without this they miss the
XPU Comfy Kitchen provider (quantized weights) and OmniXPU's kernel patches.

This package is kept separate from ``raylight`` because ``setup`` runs as the
Ray ``worker_process_setup_hook``: OmniXPU's provider routing must happen
before torch or comfy_kitchen are imported, and importing raylight pulls in
both. Do not import torch at module level here.
"""

import importlib.util
import logging
import os
import sys

_OMNIXPU_PACKAGE = "ComfyUI-OmniXPU"


def _omnixpu_dir():
    if os.environ.get("RAYLIGHT_DISABLE_OMNIXPU") == "1":
        return None
    base = os.environ.get("COMFYUI_BASE_DIRECTORY")
    if not base:
        return None
    path = os.path.join(base, "custom_nodes", _OMNIXPU_PACKAGE)
    return path if os.path.isfile(os.path.join(path, "__init__.py")) else None


def _exec_file(name, path, package_dir=None):
    search = [package_dir] if package_dir is not None else None
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=search)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def setup():
    """Ray worker_process_setup_hook: route OmniXPU runtime providers."""
    path = _omnixpu_dir()
    if path is None:
        return
    prestartup = os.path.join(path, "prestartup_script.py")
    if os.path.isfile(prestartup):
        _exec_file("_raylight_omnixpu_prestartup", prestartup)


def apply_patches():
    """Apply OmniXPU's ComfyUI patches, as ComfyUI does when loading custom nodes."""
    path = _omnixpu_dir()
    if path is None or _OMNIXPU_PACKAGE in sys.modules:
        return
    try:
        _exec_file(_OMNIXPU_PACKAGE, os.path.join(path, "__init__.py"), package_dir=path)
    except Exception as exc:
        logging.warning(f"[Raylight] ComfyUI-OmniXPU patches unavailable in worker: {exc}")
