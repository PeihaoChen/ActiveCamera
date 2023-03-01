import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels

from Mapper.unet import (
    UNetEncoder,
    UNetDecoder,
    MiniUNetEncoder,
    LearnedRGBProjection,
    MergeMultimodal,
    ResNetRGBEncoder,
)

class BaseModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.config = cfg

        if cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_0 == "sigmoid":
            self.normalize_channel_0 = torch.sigmoid
        elif cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_0 == "softmax":
            self.normalize_channel_0 = softmax_0d # test

        if cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_1 == "sigmoid":
            self.normalize_channel_1 = torch.sigmoid
        elif cfg.GP_ANTICIPATION.OUTPUT_NORMALIZATION.channel_1 == "softmax":
            self.normalize_channel_1 = softmax_2d

        self._create_gp_models()

    def forward(self, x):
        final_outputs = {}
        gp_outputs = self._do_gp_anticipation(x)
        final_outputs.update(gp_outputs)

        return final_outputs

    def _create_gp_models(self):
        raise NotImplementedError

    def _do_gp_anticipation(self, x):
        raise NotImplementedError

    def _normalize_decoder_output(self, x_dec):
        x_dec_c0 = self.normalize_channel_0(x_dec[:, 0])
        x_dec_c1 = self.normalize_channel_1(x_dec[:, 1])
        return torch.stack([x_dec_c0, x_dec_c1], dim=1)
    
class OccAntRGBD(BaseModel):
    """
    Anticipated using rgb and depth projection.
    """

    def _create_gp_models(self):
        nmodes = 2
        nmodes_x4 = nmodes
        nmodes_x5 = nmodes
        gp_cfg = self.config.GP_ANTICIPATION

        # Compute constants
        resnet_type = (
            gp_cfg.resnet_type if hasattr(gp_cfg, "resnet_type") else "resnet50"
        )
        infeats = 768 if resnet_type == "resnet50" else 192
        nsf = gp_cfg.unet_nsf # 16
        unet_encoder = UNetEncoder(2, nsf=nsf)
        unet_decoder = UNetDecoder(gp_cfg.nclasses, nsf=nsf)
        unet_feat_size = nsf * 8 # 16 * 8 = 128

        # RGB encoder branch
        self.gp_rgb_encoder = ResNetRGBEncoder(resnet_type)
        self.gp_rgb_projector = LearnedRGBProjection(mtype="upsample", infeats=infeats)
        self.gp_rgb_unet = MiniUNetEncoder(infeats, unet_feat_size)

        # Depth encoder branch
        self.gp_depth_proj_encoder = unet_encoder

        # Merge modules
        self.gp_merge_x5 = MergeMultimodal(unet_feat_size, nmodes=nmodes_x5)
        self.gp_merge_x4 = MergeMultimodal(unet_feat_size, nmodes=nmodes_x4)
        self.gp_merge_x3 = MergeMultimodal(unet_feat_size // 2, nmodes=nmodes)

        # Decoder module
        self.gp_decoder = unet_decoder

        if gp_cfg.freeze_features:
            for p in self.gp_rgb_encoder.parameters():
                p.requires_grad = False

    def _do_gp_anticipation(self, x):
        """
        Inputs:
            x is a dictionary containing the following keys:
                'rgb' - (bs, 3, H, W) RGB input
                'ego_map_gt' - (bs, 2, H, W) probabilities
        """
        x_rgb = self.gp_rgb_encoder(x["rgb"])  # (bs, infeats, H/8, W/8)
        x_gp = self.gp_rgb_projector(x_rgb)  # (bs, infeats, H/4, W/4)


        fuse_rgb_f = x_gp
        depth_projection_input = x["ego_map_gt"]

        x_rgb_enc = self.gp_rgb_unet(fuse_rgb_f)  # {'x3p', 'x4p', 'x5p'}
        x_depth_proj_enc = self.gp_depth_proj_encoder(
            depth_projection_input
        )  # {'x1', 'x2', 'x3', 'x4', 'x5'}

        # Replace x_depth_proj_enc with merged features
        x5_inputs = [x_rgb_enc["x5p"], x_depth_proj_enc["x5"]]
        x4_inputs = [x_rgb_enc["x4p"], x_depth_proj_enc["x4"]]
        x3_inputs = [x_rgb_enc["x3p"], x_depth_proj_enc["x3"]]

        x5_enc = self.gp_merge_x5(*x5_inputs)  # (unet_feat_size  , H/16, H/16) # 8*8
        x4_enc = self.gp_merge_x4(*x4_inputs)  # (unet_feat_size  , H/8 , H/8 ) # 16*16
        x3_enc = self.gp_merge_x3(*x3_inputs)  # (unet_feat_size/2, H/4 , H/4 ) # 32*32
        x_depth_proj_enc["x5"] = x5_enc
        x_depth_proj_enc["x4"] = x4_enc
        x_depth_proj_enc["x3"] = x3_enc

        x_dec = self.gp_decoder(x_depth_proj_enc)  # (bs, 2, H, W)
        x_dec = self._normalize_decoder_output(x_dec)

        outputs = {"occ_estimate": x_dec}

        outputs["register_ego_map"] = depth_projection_input
        return outputs

class OccupancyAnticipator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.main = OccAntRGBD(cfg)



    def forward(self, x):
        return self.main(x)

    @property
    def use_gp_anticipation(self):
        return self.main.use_gp_anticipation

    @property
    def model_type(self):
        return self._model_type
