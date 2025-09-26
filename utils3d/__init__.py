"""Lightweight compatibility layer emulating the subset of the utils3d API
required by the ComfyUI MoGe2 panorama nodes.

The original project ships a much larger helper library.  For environments
where that dependency is not pre-installed we provide minimal NumPy/Torch
implementations so the nodes can still import successfully.  The
implementations only cover the functions referenced inside this repository.
"""

from . import numpy
from . import torch as torch

__all__ = ["numpy", "torch"]
