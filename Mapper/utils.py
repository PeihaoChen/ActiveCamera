import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt

def process_image(img, img_mean, img_std):
    """
    Convert HWC -> CHW, normalize image.
    Inputs:
        img - (bs, H, W, C) torch Tensor
        img_mean - list of per-channel means
        img_std - list of per-channel stds

    Outputs:
        img_p - (bs, C, H, W)
    """
    C = img.shape[3]
    device = img.device

    img_p = rearrange(img.float(), "b h w c -> b c h w")
    img_p = img_p / 255.0  # (bs, C, H, W)

    if type(img_mean) == type([]):
        img_mean_t = rearrange(torch.Tensor(img_mean), "c -> () c () ()").to(device)
        img_std_t = rearrange(torch.Tensor(img_std), "c -> () c () ()").to(device)
    else:
        img_mean_t = img_mean.to(device)
        img_std_t = img_std.to(device)

    img_p = (img_p - img_mean_t) / img_std_t

    return img_p

def transpose_image(img):
    """
    Inputs:
        img - (bs, H, W, C) torch Tensor
    """
    img_p = img.permute(0, 3, 1, 2)  # (bs, C, H, W)
    return img_p

def padded_resize(x, size):
    """For an image tensor of size (bs, c, h, w), resize it such that the
    larger dimension (h or w) is scaled to `size` and the other dimension is
    zero-padded on both sides to get `size`.
    """
    h, w = x.shape[2:]
    top_pad = 0
    bot_pad = 0
    left_pad = 0
    right_pad = 0
    if h > w:
        left_pad = (h - w) // 2
        right_pad = (h - w) - left_pad
    elif w > h:
        top_pad = (w - h) // 2
        bot_pad = (w - h) - top_pad
    x = F.pad(x, (left_pad, right_pad, top_pad, bot_pad))
    x = F.interpolate(x, size, mode="bilinear", align_corners=False)
    return x

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

def dilate_tensor(x, size, iterations=1):
    """
    x - (bs, C, H, W)
    size - int / tuple of intes

    Assumes a kernel of ones with size 'size'.
    """
    if type(size) == int:
        padding = size // 2
    else:
        padding = tuple([v // 2 for v in size])
    for i in range(iterations):
        x = F.max_pool2d(x, size, stride=1, padding=padding)

    return x



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

def bottom_row_padding(p):
    V = p.shape[2]
    Vby2 = (V - 1) / 2 if V % 2 == 1 else V // 2
    left_h_pad = 0
    right_h_pad = int(V - 1)
    if V % 2 == 1:
        left_w_pad = int(Vby2)
        right_w_pad = int(Vby2)
    else:
        left_w_pad = int(Vby2) - 1
        right_w_pad = int(Vby2)

    # Pad so that the origin is at the center
    p_pad = F.pad(p, (left_w_pad, right_w_pad, left_h_pad, right_h_pad), "constant", 0)

    return p_pad

def bottom_row_cropping(p, map_size):
    bs = p.shape[0]
    V = map_size
    Vby2 = (V - 1) / 2 if V % 2 == 1 else V // 2
    device = p.device

    x_crop_center = torch.zeros(bs, 2).to(device)
    x_crop_center[:, 0] = V - 1
    x_crop_center[:, 1] = Vby2
    x_crop_size = V

    p_cropped = crop_map(p, x_crop_center, x_crop_size)

    return p_cropped

def bottom_row_center_cropping(p, map_size):
    bs = p.shape[0]
    V = p.shape[2]
    Vby2 = int((V - 1) / 2 if V % 2 == 1 else V // 2)

    map_sizeby2 = int((map_size - 1) / 2 if map_size % 2 == 1 else map_size // 2)
    croped_map = p[:,:, Vby2-map_sizeby2:Vby2+map_sizeby2+1, V-map_size:]

    return croped_map

def convert_gt2channel_to_gtrgb(gts):
    """
    Inputs:
        gts   - (H, W, 2) numpy array with values between 0.0 to 1.0
              - channel 0 is 1 if occupied space
              - channel 1 is 1 if explored space
    """
    H, W, _ = gts.shape

    exp_mask = (gts[..., 1] >= 0.5).astype(np.float32)
    occ_mask = (gts[..., 0] >= 0.5).astype(np.float32) * exp_mask
    free_mask = (gts[..., 0] < 0.5).astype(np.float32) * exp_mask
    unk_mask = 1 - exp_mask

    gt_imgs = np.stack(
        [
            0.0 * occ_mask + 0.0 * free_mask + 255.0 * unk_mask,
            0.0 * occ_mask + 255.0 * free_mask + 255.0 * unk_mask,
            255.0 * occ_mask + 0.0 * free_mask + 255.0 * unk_mask,
        ],
        axis=2,
    ).astype(
        np.float32
    )  # (H, W, 3)

    return gt_imgs

def mapper_debugger_plot(_gt,_pt,rgb):
    def convert_gt2channel_to_gtrgb(gts):
        H, W, _ = gts.shape

        exp_mask = (gts[..., 1] >= 0.5).astype(np.float32)
        occ_mask = (gts[..., 0] >= 0.5).astype(np.float32) * exp_mask
        free_mask = (gts[..., 0] < 0.5).astype(np.float32) * exp_mask
        unk_mask = 1 - exp_mask

        gt_imgs = np.stack(
            [
                0.0 * occ_mask + 0.0 * free_mask + 255.0 * unk_mask,
                0.0 * occ_mask + 255.0 * free_mask + 255.0 * unk_mask,
                255.0 * occ_mask + 0.0 * free_mask + 255.0 * unk_mask,
            ],
            axis=2,
        ).astype(
            np.float32
        )  # (H, W, 3)

        return gt_imgs

    gt = convert_gt2channel_to_gtrgb(_gt)
    pt = convert_gt2channel_to_gtrgb(_pt)
    import matplotlib.pyplot as plt
    plt.subplot(221)
    plt.title('depth projection')
    plt.imshow(gt,interpolation='none')
    
    plt.subplot(222)
    plt.title('predict local map')
    plt.imshow(pt,interpolation='none')

    plt.subplot(223)
    plt.title('rgb')
    plt.imshow(rgb,interpolation='none')
    
    import os
    path = '/mnt/cephfs/home/linkunyang/Projects/multiON/debugger_result/'
    count = len(os.listdir(path))
    plt.savefig(f'{path}{count+1}.png')

    np.save(f'{path}rgb_{count+1}',rgb)
    np.save(f'{path}_pt_{count+1}',_pt)
    np.save(f'{path}_gt_{count+1}',_gt)
