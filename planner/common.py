import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import math
import numpy as np

from einops import rearrange

def spatial_transform_map(p, x, invert=True, mode="bilinear"):
    """
    Inputs:
        p     - (bs, f, H, W) Tensor
        x     - (bs, 3) Tensor (x, y, theta) transforms to perform
    Outputs:
        p_trans - (bs, f, H, W) Tensor
    Conventions:
        Shift in X is rightward, and shift in Y is downward. Rotation is clockwise.

    Note: These denote transforms in an agent's position. Not the image directly.
    For example, if an agent is moving upward, then the map will be moving downward.
    To disable this behavior, set invert=False.
    """
    device = p.device
    H, W = p.shape[2:]

    trans_x = x[:, 0]
    trans_y = x[:, 1]
    # Convert translations to -1.0 to 1.0 range
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H / 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W / 2

    trans_x = trans_x / Wby2
    trans_y = trans_y / Hby2
    rot_t = x[:, 2]

    sin_t = torch.sin(rot_t)
    cos_t = torch.cos(rot_t)

    # This R convention means Y axis is downwards.
    A = torch.zeros(p.size(0), 3, 3).to(device)
    A[:, 0, 0] = cos_t
    A[:, 0, 1] = -sin_t
    A[:, 1, 0] = sin_t
    A[:, 1, 1] = cos_t
    A[:, 0, 2] = trans_x
    A[:, 1, 2] = trans_y
    A[:, 2, 2] = 1

    # Since this is a source to target mapping, and F.affine_grid expects
    # target to source mapping, we have to invert this for normal behavior.
    Ainv = torch.inverse(A)

    # If target to source mapping is required, invert is enabled and we invert
    # it again.
    if invert:
        Ainv = torch.inverse(Ainv)

    Ainv = Ainv[:, :2]
    grid = F.affine_grid(Ainv, p.size())
    p_trans = F.grid_sample(p, grid, mode=mode)

    return p_trans


def convert_world2map(world_coors, map_shape, map_scale):
    """
    World coordinate system:
        Agent starts at (0, 0) facing upward along X. Y is rightward.
    Map coordinate system:
        Agent starts at (W/2, H/2) with X rightward and Y downward.

    Inputs:
        world_coors: (bs, 2) --- (x, y) in world coordinates
        map_shape: tuple with (H, W)
        map_scale: scalar indicating the cell size in the map
    """
    H, W = map_shape
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2

    x_world = world_coors[:, 0]
    y_world = world_coors[:, 1]

    x_map = torch.clamp((Wby2 + y_world / map_scale), 0, W - 1).round()
    y_map = torch.clamp((Hby2 - x_world / map_scale), 0, H - 1).round()

    map_coors = torch.stack([x_map, y_map], dim=1)  # (bs, 2)

    return map_coors


def convert_map2world(map_coors, map_shape, map_scale):
    """
    World coordinate system:
        Agent starts at (0, 0) facing upward along X. Y is rightward.
    Map coordinate system:
        Agent starts at (W/2, H/2) with X rightward and Y downward.

    Inputs:
        map_coors: (bs, 2) --- (x, y) in map coordinates
        map_shape: tuple with (H, W)
        map_scale: scalar indicating the cell size in the map
    """
    H, W = map_shape
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2

    x_map = map_coors[:, 0]
    y_map = map_coors[:, 1]

    x_world = (Hby2 - y_map) * map_scale
    y_world = (x_map - Wby2) * map_scale

    world_coors = torch.stack([x_world, y_world], dim=1)  # (bs, 2)

    return world_coors


def subtract_pose(pose_a, pose_b):
    """
    Compute pose of pose_b in the egocentric coordinate frame of pose_a.
    Inputs:
        pose_a - (bs, 3) --- (x, y, theta)
        pose_b - (bs, 3) --- (x, y, theta)

    Conventions:
        The origin is at the center of the map.
        X is upward with agent's forward direction
        Y is rightward with agent's rightward direction
    """

    x_a, y_a, theta_a = torch.unbind(pose_a, dim=1)
    x_b, y_b, theta_b = torch.unbind(pose_b, dim=1)

    r_ab = torch.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2)  # (bs, )
    phi_ab = torch.atan2(y_b - y_a, x_b - x_a) - theta_a  # (bs, )
    theta_ab = theta_b - theta_a  # (bs, )
    theta_ab = torch.atan2(torch.sin(theta_ab), torch.cos(theta_ab))

    x_ab = torch.stack(
        [r_ab * torch.cos(phi_ab), r_ab * torch.sin(phi_ab), theta_ab,], dim=1
    )  # (bs, 3)

    return x_ab


def add_pose(pose_a, pose_ab):
    """
    Add pose_ab (in ego-coordinates of pose_a) to pose_a
    Inputs:
        pose_a - (bs, 3) --- (x, y, theta)
        pose_b - (bs, 3) --- (x, y, theta)

    Conventions:
        The origin is at the center of the map.
        X is upward with agent's forward direction
        Y is rightward with agent's rightward direction
    """

    x_a, y_a, theta_a = torch.unbind(pose_a, dim=1)
    x_ab, y_ab, theta_ab = torch.unbind(pose_ab, dim=1)

    r_ab = torch.sqrt(x_ab ** 2 + y_ab ** 2)
    phi_ab = torch.atan2(y_ab, x_ab)

    x_b = x_a + r_ab * torch.cos(phi_ab + theta_a)
    y_b = y_a + r_ab * torch.sin(phi_ab + theta_a)
    theta_b = theta_a + theta_ab
    theta_b = torch.atan2(torch.sin(theta_b), torch.cos(theta_b))

    pose_b = torch.stack([x_b, y_b, theta_b], dim=1)  # (bs, 3)

    return pose_b


def crop_map(h, x, crop_size, mode="bilinear"):
    """
    Crops a tensor h centered around location x with size crop_size

    Inputs:
        h - (bs, F, H, W)
        x - (bs, 2) --- (x, y) locations
        crop_size - scalar integer

    Conventions for x:
        The origin is at the top-left, X is rightward, and Y is downward.
    """

    bs, _, H, W = h.size()
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2
    start = -(crop_size - 1) / 2 if crop_size % 2 == 1 else -(crop_size // 2)
    end = start + crop_size - 1
    x_grid = (
        torch.arange(start, end + 1, step=1)
        .unsqueeze(0)
        .expand(crop_size, -1)
        .contiguous()
        .float()
    )
    y_grid = (
        torch.arange(start, end + 1, step=1)
        .unsqueeze(1)
        .expand(-1, crop_size)
        .contiguous()
        .float()
    )
    center_grid = torch.stack([x_grid, y_grid], dim=2).to(
        h.device
    )  # (crop_size, crop_size, 2)

    x_pos = x[:, 0] - Wby2  # (bs, )
    y_pos = x[:, 1] - Hby2  # (bs, )

    crop_grid = center_grid.unsqueeze(0).expand(
        bs, -1, -1, -1
    )  # (bs, crop_size, crop_size, 2)
    crop_grid = crop_grid.contiguous()

    # Convert the grid to (-1, 1) range
    crop_grid[:, :, :, 0] = (
        crop_grid[:, :, :, 0] + x_pos.unsqueeze(1).unsqueeze(2)
    ) / Wby2
    crop_grid[:, :, :, 1] = (
        crop_grid[:, :, :, 1] + y_pos.unsqueeze(1).unsqueeze(2)
    ) / Hby2

    h_cropped = F.grid_sample(h, crop_grid, mode=mode)

    return h_cropped

