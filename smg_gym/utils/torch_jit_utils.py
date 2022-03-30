# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import numpy as np

from isaacgym.torch_utils import quat_mul
from isaacgym.torch_utils import quat_from_angle_axis
from isaacgym.torch_utils import torch_rand_float


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def lgsk_kernel(x: torch.Tensor, scale: float = 50.0, eps: float = 2) -> torch.Tensor:
    """Defines logistic kernel function to bound input

    Ref: https://arxiv.org/abs/1901.08652 (page 15)

    Args:
        x: Input tensor.
        scale: Scaling of the kernel function (controls how wide the 'bell' shape is')
        eps: Controls how 'tall' the 'bell' shape is.

    Returns:
        Output tensor computed using kernel.
    """
    scaled = x * scale
    return 1.0 / (scaled.exp() + eps + (-scaled).exp())


@torch.jit.script
def torch_random_dir(num_envs_to_reset: int, device: str) -> torch.Tensor:
    phi = torch_rand_float(0.0, 2*np.pi, (num_envs_to_reset, 1), device).squeeze(-1)
    costheta = torch_rand_float(-1.0, 1.0, (num_envs_to_reset, 1), device).squeeze(-1)
    theta = torch.arccos(costheta)

    return torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta)
    ], dim=-1)


@torch.jit.script
def torch_random_cardinal_dir(num_envs_to_reset: int, device: str) -> torch.Tensor:

    cardinal_dirs = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ], device=device)

    choice = torch.randint(low=0, high=6, size=(num_envs_to_reset, 1), device=device, dtype=torch.long)

    return cardinal_dirs[choice].squeeze()
