
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import spaces
# import utils
# import mapnet
# import model

from Mapper.utils import (
    process_image,
    transpose_image,
    padded_resize,
    crop_map
)

from Mapper.mapnet import DepthProjectionNet
from Mapper.model import OccupancyAnticipator
import cv2
import numpy as np
from einops import rearrange, asnumpy
from habitat.config import Config as CN

class Mapper(nn.Module):
    def __init__(self, config, projection_unit):
        super().__init__()
        self.config = config
        self.img_mean_t = rearrange(
            torch.Tensor(self.config.NORMALIZATION.img_mean), "c -> () c () ()"
        )
        self.img_std_t = rearrange(
            torch.Tensor(self.config.NORMALIZATION.img_std), "c -> () c () ()"
        )
        self.projection_unit = projection_unit
        if self.config.freeze_projection_unit:
            for p in self.projection_unit.parameters():
                p.requires_grad = False

        # Cache to store pre-computed information
        self._cache = {}

    def forward(self, x, masks=None):
        outputs = self.predict_deltas(x, masks=masks)
        return outputs

    def predict_deltas(self, x, masks=None):
        """输入RGB、Depth，输出预测的local map（pt）
        """
        # Transpose multichannel inputs
        st = process_image(x["rgb_at_t"], self.img_mean_t, self.img_std_t)
        ego_map_gt_at_t = transpose_image(x["ego_map_gt_at_t"])

        # Compute past and current egocentric maps
        bs = st.size(0)

        pu_inputs = {
            "rgb": st,
            "ego_map_gt": ego_map_gt_at_t
        }

        pu_outputs = self.projection_unit(pu_inputs)    # 前向传播occupancy anticipation model
        pt = pu_outputs["occ_estimate"][:bs]

        outputs = {
            "pt": pt,           # 当前时刻的anticipated local map（即map predictor的预测局部地图）
        }

        outputs["register_ego_map"] = pu_outputs["register_ego_map"]
        return outputs

class OccupancyAnticipationWrapper(nn.Module):
    def __init__(self, model, V, input_hw):
        super().__init__()
        self.main = model
        self.V = V
        self.input_hw = input_hw
        self.keys_to_interpolate = [
            "ego_map_hat",
            "occ_estimate",
            "depth_proj_estimate",  # specific to RGB Model V2
        ]

    def forward(self, x):
        x["rgb"] = padded_resize(x["rgb"], self.input_hw[0])
        if "ego_map_gt" in x:
            x["ego_map_gt"] = F.interpolate(x["ego_map_gt"], size=self.input_hw)
        x_full = self.main(x)

        for k in x_full.keys():
            if k in self.keys_to_interpolate:
                x_full[k] = F.interpolate(
                    x_full[k], size=(self.V, self.V), mode="bilinear"
                )


        return x_full
        
def load_state_dict(model,loaded_state_dict):
    """Intelligent state dict assignment. Load state-dict only for keysload_state_dict(strict=False)
    that are available and have matching parameter sizes.
    """
    src_state_dict = model.state_dict()
    matching_state_dict = {}
    offending_keys = []
    for k, v in loaded_state_dict.items():
        if k in src_state_dict.keys() and v.shape == src_state_dict[k].shape:
            matching_state_dict[k] = v
        else:
            offending_keys.append(k)
    src_state_dict.update(matching_state_dict)
    model.load_state_dict(src_state_dict)
    if len(offending_keys) > 0:
        print("=======> Update: list of offending keys in load_state_dict")
        for k in offending_keys:
            if 'mapper_copy' in k or 'pose_estimator' in k:
                continue
            print(k)
    return model
