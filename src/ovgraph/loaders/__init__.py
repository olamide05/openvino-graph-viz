"""Loaders turn a model file into a Graph. Only the OpenVINO loader exists; new formats register themselves.
"""
from .base import Loader, available_loaders, load_graph, register_loader

# Importing the module registers it. It does NOT import openvino yet (that
# happens lazily inside .load), so the package works without openvino installed.
from . import openvino_loader  # noqa: F401

__all__ = ["Loader", "available_loaders", "load_graph", "register_loader"]
