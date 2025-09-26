from __future__ import annotations

from typing import Tuple, Union, Sequence

import torch as _torch

__all__ = [
    "image_pixel_center",
    "image_uv",
    "sliding_window_2d",
]


def image_uv(width: int, height: int, dtype: _torch.dtype = _torch.float32, device: Union[_torch.device, str, None] = None) -> _torch.Tensor:
    u = _torch.linspace(0.0, 1.0, width, dtype=dtype, device=device)
    v = _torch.linspace(0.0, 1.0, height, dtype=dtype, device=device)
    uu, vv = _torch.meshgrid(u, v, indexing="xy")
    return _torch.stack([uu, vv], dim=-1)


def image_pixel_center(width: int, height: int, dtype: _torch.dtype = _torch.float32, device: Union[_torch.device, str, None] = None) -> _torch.Tensor:
    x = _torch.linspace(0.5, width - 0.5, width, dtype=dtype, device=device)
    y = _torch.linspace(0.5, height - 0.5, height, dtype=dtype, device=device)
    xx, yy = _torch.meshgrid(x, y, indexing="xy")
    return _torch.stack([xx, yy], dim=-1)


def _normalize_dim(dim: Union[int, Sequence[int]], ndims: int) -> Tuple[int, ...]:
    if isinstance(dim, int):
        dim = (dim,)
    return tuple(d if d >= 0 else ndims + d for d in dim)


def sliding_window_2d(tensor: _torch.Tensor, window_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]] = 1, dim: Union[int, Tuple[int, int]] = (0, 1)) -> _torch.Tensor:
    dims = _normalize_dim(dim, tensor.dim())
    if isinstance(window_size, int):
        window_size = (window_size,) * len(dims)
    if isinstance(stride, int):
        stride = (stride,) * len(dims)

    if len(dims) != len(window_size) or len(dims) != len(stride):
        raise ValueError("dim, window_size and stride must have the same length")

    shape = list(tensor.shape)
    strides = list(tensor.stride())

    dims_map = {axis: (window_size[i], stride[i]) for i, axis in enumerate(dims)}
    ordered_axes = sorted(dims_map.keys())

    step_shape = []
    step_stride = []
    for axis in ordered_axes:
        win, st = dims_map[axis]
        if shape[axis] < win:
            raise ValueError("window larger than tensor dimension")
        step_shape.append((shape[axis] - win) // st + 1)
        step_stride.append(strides[axis] * st)

    remaining_shape = [shape[i] for i in range(tensor.dim()) if i not in dims_map]
    remaining_stride = [strides[i] for i in range(tensor.dim()) if i not in dims_map]

    window_shape = [dims_map[axis][0] for axis in ordered_axes]
    window_stride = [strides[axis] for axis in ordered_axes]

    out_shape = step_shape + remaining_shape + window_shape
    out_stride = step_stride + remaining_stride + window_stride

    return _torch.as_strided(tensor, size=tuple(out_shape), stride=tuple(out_stride))
