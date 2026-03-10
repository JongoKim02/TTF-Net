#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:15:47 2024

@author: Md Mostafijur Rahman
"""

import sys
from typing import Tuple
import numpy as np
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Union
from lib.utils.tools.logger import Logger as Log
from lib.models.tools.module_helper import ModuleHelper
from networks.UXNet_3D.uxnet_encoder import uxnet_conv

import logging

logger = logging.getLogger(__name__)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def set_text_fusion(self, enabled: bool = True):
    self.enable_text_fusion = enabled


class InlineTriPlaneTextFuser(nn.Module):
    def __init__(self, in_ch: int, text_dim: int, d: int = 64):
        super().__init__()
        self.ax_proj = nn.Conv2d(in_ch, d, kernel_size=1, bias=False)  # (H,W)
        self.sa_proj = nn.Conv2d(in_ch, d, kernel_size=1, bias=False)  # (D,W)
        self.co_proj = nn.Conv2d(in_ch, d, kernel_size=1, bias=False)  # (D,H)
        self.text_proj = nn.Linear(text_dim, d, bias=False)

    def forward(self, feat3d: torch.Tensor, sel_text: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "_dbg_once"):
            with torch.no_grad():
                print(f"[TriFuse] feat={tuple(feat3d.shape)}, text={tuple(sel_text.shape)}")
            self._dbg_once = True

        B, C, D, H, W = feat3d.shape


        axial_summary = feat3d.amax(dim=2)
        sagittal_summary = feat3d.amax(dim=3)
        coronal_summary = feat3d.amax(dim=4)

        ax = self.ax_proj(axial_summary).flatten(2)
        sa = self.sa_proj(sagittal_summary).flatten(2)
        co = self.co_proj(coronal_summary).flatten(2)

        k = F.normalize(self.text_proj(sel_text), dim=-1)

        ax_ker = F.softmax(torch.einsum("bdn,bd->bn", ax, k), dim=-1).view(B, 1, H, W)
        sa_ker = F.softmax(torch.einsum("bdn,bd->bn", sa, k), dim=-1).view(B, 1, D, W)
        co_ker = F.softmax(torch.einsum("bdn,bd->bn", co, k), dim=-1).view(B, 1, D, H)


        ax_w_3d = feat3d * ax_ker.unsqueeze(2)  # [B,C,D,H,W] * [B,1,1,H,W]
        sa_w_3d = feat3d * sa_ker.unsqueeze(3)  # [B,C,D,H,W] * [B,1,D,1,W]
        co_w_3d = feat3d * co_ker.unsqueeze(4)  # [B,C,D,H,W] * [B,1,D,H,1]


        return (1*feat3d + 1*ax_w_3d + 1*sa_w_3d + 1*co_w_3d) / 4.00


class ModifiedUnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size,
            upsample_kernel_size,
            norm_name,
            res_block=False,
            skip_aggregation='concatenation',
            text_fuser: nn.Module | None = None,
            enable_text_fusion: bool = False
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.
            skip_aggregation: type of skip aggregation, addition or concatenation
        """

        super().__init__()
        self.skip_aggregation = skip_aggregation
        in_out_channels = out_channels
        if self.skip_aggregation == 'concatenation':
            in_out_channels = out_channels + out_channels
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                in_out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                in_out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        self.text_fuser = text_fuser
        self.enable_text_fusion = enable_text_fusion

    def forward(self, inp, skip, sel_text: torch.Tensor | None = None):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        if self.skip_aggregation == 'concatenation':
            out = torch.cat((out, skip), dim=1)
        else:
            out = out + skip
        if self.enable_text_fusion and (self.text_fuser is not None) and (sel_text is not None):
            out = self.text_fuser(out, sel_text)
        out = self.conv_block(out)
        return out


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

        Log.info('proj_dim: {}'.format(proj_dim))

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1),
                ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.Conv3d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


# class ResBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self,
#                  in_planes: int,
#                  planes: int,
#                  spatial_dims: int = 3,
#                  stride: int = 1,
#                  downsample: Union[nn.Module, partial, None] = None,
#     ) -> None:
#         """
#         Args:
#             in_planes: number of input channels.
#             planes: number of output channels.
#             spatial_dims: number of spatial dimensions of the input image.
#             stride: stride to use for first conv layer.
#             downsample: which downsample layer to use.
#         """
#
#         super().__init__()
#
#         conv_type: Callable = Conv[Conv.CONV, spatial_dims]
#         norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
#
#         self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
#         self.bn1 = norm_type(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
#         self.bn2 = norm_type(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         residual = x
#
#         out: torch.Tensor = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out


class UXNET(nn.Module):

    def __init__(
            self,
            in_chans=1,
            out_chans=13,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
            text_dim: int = 1024,
            triplane_dim: int = 64,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        # in_channels: int,
        # out_channels: int,
        # img_size: Union[Sequence[int], int],
        # feature_size: int = 16,
        # if not (0 <= dropout_rate <= 1):
        #     raise ValueError("dropout_rate should be between 0 and 1.")
        #
        # if hidden_size % num_heads != 0:
        #     raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        # self.feature_size = feature_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        self.text_dim = text_dim
        self.triplane_dim = triplane_dim
        self.enable_text_fusion = True
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims

        # self.classification = False
        # self.vit = ViT(
        #     in_channels=in_channels,
        #     img_size=img_size,
        #     patch_size=self.patch_size,
        #     hidden_size=hidden_size,
        #     mlp_dim=mlp_dim,
        #     num_layers=self.num_layers,
        #     num_heads=num_heads,
        #     pos_embed=pos_embed,
        #     classification=self.classification,
        #     dropout_rate=dropout_rate,
        #     spatial_dims=spatial_dims,
        # )
        self.uxnet_3d = uxnet_conv(
            in_chans=self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.out_chans)
        self.fuse5 = InlineTriPlaneTextFuser(in_ch=self.feat_size[3], text_dim=self.text_dim, d=self.triplane_dim)
        self.fuse4 = InlineTriPlaneTextFuser(in_ch=self.feat_size[2], text_dim=self.text_dim, d=self.triplane_dim)
        self.fuse3 = InlineTriPlaneTextFuser(in_ch=self.feat_size[1], text_dim=self.text_dim, d=self.triplane_dim)
        self.fuse2 = InlineTriPlaneTextFuser(in_ch=self.feat_size[0], text_dim=self.text_dim, d=self.triplane_dim)
        # self.fuse1 = InlineTriPlaneTextFuser(in_ch=self.feat_size[0], text_dim=self.text_dim, d=self.triplane_dim)
        # self.conv_proj = ProjectionHead(dim_in=hidden_size)

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in, sel_text=None):

        def _align_sel(sel, B_feat):
            if sel is None:
                return None

            if sel.dtype != torch.float32:
                sel = sel.float()
            if sel.device != x_in.device:
                sel = sel.to(x_in.device, non_blocking=True)
            B_txt = sel.shape[0]
            if B_txt == B_feat:
                return sel

            if B_feat % B_txt == 0:
                r = B_feat // B_txt
                return sel.repeat_interleave(r, dim=0)

            return None
        outs = self.uxnet_3d(x_in)
        # print([outs[0].shape,outs[1].shape,outs[2].shape,outs[3].shape])
        # print(outs[0].size())
        # print(outs[1].size())
        # print(outs[2].size())
        # print(outs[3].size())
        enc1 = self.encoder1(x_in)
        # print('enc1:', enc1.size())
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        # print('enc2:', enc2.size())
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        # print('enc3:', enc3.size())
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        # print('enc4:', enc4.size())
        # dec4 = self.proj_feat(outs[3], self.hidden_size, self.feat_size)
        enc_hidden = self.encoder5(outs[3])
        # print('enc_hidden:', enc_hidden.size())
        dec3 = self.decoder5(enc_hidden, enc4)
        if sel_text is not None:
            sel5 = _align_sel(sel_text, dec3.shape[0])
            if sel5 is not None:

                #     print(f"[TriFuse] feat={tuple(dec3.shape)}, text={tuple(sel5.shape)}")
                dec3 = self.fuse5(dec3, sel5)

        dec2 = self.decoder4(dec3, enc3)
        if sel_text is not None:
            sel4 = _align_sel(sel_text, dec2.shape[0])
            if sel4 is not None:
                # if __debug__:
                #     print(f"[TriFuse] feat={tuple(dec2.shape)}, text={tuple(sel4.shape)}")
                dec2 = self.fuse4(dec2, sel4)

        dec1 = self.decoder3(dec2, enc2)
        if sel_text is not None:
            sel3 = _align_sel(sel_text, dec1.shape[0])
            if sel3 is not None:
                # if __debug__:
                #     print(f"[TriFuse] feat={tuple(dec1.shape)}, text={tuple(sel3.shape)}")
                dec1 = self.fuse3(dec1, sel3)

        dec0 = self.decoder2(dec1, enc1)
        if sel_text is not None:
            sel2 = _align_sel(sel_text, dec0.shape[0])
            if sel2 is not None:
                # if __debug__:
                #     print(f"[TriFuse] feat={tuple(dec0.shape)}, text={tuple(sel2.shape)}")
                dec0 = self.fuse2(dec0, sel2)
        out = self.decoder1(dec0)

        # if sel_text is not None:
        #     sel1 = _align_sel(sel_text, out.shape[0])
        #     if sel1 is not None:

        # print('out:', out.size())
        # feat = self.conv_proj(dec4)

        return self.out(out)


class UXNET_EffiDec3D(nn.Module):

    def __init__(
            self,
            in_chans=1,
            out_chans=13,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            n_decoder_channels=48,
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = True,
            res_block: bool = True,
            skip_aggregation: str = 'concatenation',
            resolution_factor: int = 2,
            spatial_dims=3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        # in_channels: int,
        # out_channels: int,
        # img_size: Union[Sequence[int], int],
        # feature_size: int = 16,
        # if not (0 <= dropout_rate <= 1):
        #     raise ValueError("dropout_rate should be between 0 and 1.")
        #
        # if hidden_size % num_heads != 0:
        #     raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        # self.feature_size = feature_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.n_decoder_channels = n_decoder_channels
        self.resolution_factor = resolution_factor
        self.cls_head_in_channels = n_decoder_channels
        self.n_channels_enc2_dec3 = min(self.n_decoder_channels, self.feat_size[0])
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims

        # self.classification = False
        # self.vit = ViT(
        #     in_channels=in_channels,
        #     img_size=img_size,
        #     patch_size=self.patch_size,
        #     hidden_size=hidden_size,
        #     mlp_dim=mlp_dim,
        #     num_layers=self.num_layers,
        #     num_heads=num_heads,
        #     pos_embed=pos_embed,
        #     classification=self.classification,
        #     dropout_rate=dropout_rate,
        #     spatial_dims=spatial_dims,
        # )
        self.uxnet_3d = uxnet_conv(
            in_chans=self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        if self.resolution_factor <= 1:
            self.encoder1 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.in_chans,
                out_channels=self.n_channels_enc2_dec3,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
        if self.resolution_factor <= 2:
            self.encoder2 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[0],
                out_channels=self.n_channels_enc2_dec3,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
        if self.resolution_factor <= 4:
            self.encoder3 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[1],
                out_channels=self.n_decoder_channels,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
        if self.resolution_factor <= 8:
            self.encoder4 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[2],
                out_channels=self.n_decoder_channels,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
        if self.resolution_factor <= 16:
            self.encoder5 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[3],
                out_channels=self.n_decoder_channels,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.cls_head_in_channels = self.n_decoder_channels
        if self.resolution_factor <= 8:
            self.decoder5 = ModifiedUnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_decoder_channels,
                out_channels=self.n_decoder_channels,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                skip_aggregation=skip_aggregation,
            )
            self.cls_head_in_channels = self.n_decoder_channels
        if self.resolution_factor <= 4:
            self.decoder4 = ModifiedUnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_decoder_channels,
                out_channels=self.n_decoder_channels,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                skip_aggregation=skip_aggregation,
            )
            self.cls_head_in_channels = self.n_decoder_channels
        if self.resolution_factor <= 2:
            self.decoder3 = ModifiedUnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_decoder_channels,
                out_channels=self.n_channels_enc2_dec3,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                skip_aggregation=skip_aggregation
            )
            self.cls_head_in_channels = self.n_channels_enc2_dec3
        if self.resolution_factor <= 1:
            self.decoder2 = ModifiedUnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_channels_enc2_dec3,
                out_channels=self.n_channels_enc2_dec3,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
                skip_aggregation=skip_aggregation
            )
            self.decoder1 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=self.n_channels_enc2_dec3,
                out_channels=self.n_channels_enc2_dec3,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.cls_head_in_channels = self.n_channels_enc2_dec3
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.cls_head_in_channels,
                                out_channels=self.out_chans)
        # self.conv_proj = ProjectionHead(dim_in=hidden_size)

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in):

        # Check for invalid resolution_factor
        if self.resolution_factor > 16:
            print("Invalid resolution_factor for this model. Must be <= 16.")
            return sys.exit()

        outs = self.uxnet_3d(x_in)
        # print([outs[0].shape,outs[1].shape,outs[2].shape,outs[3].shape])
        # print(outs[0].size())
        # print(outs[1].size())
        # print(outs[2].size())
        # print(outs[3].size())
        # enc1 = self.encoder1(x_in)
        # print(enc1.size())
        # Encoder Pass
        enc1, enc2, enc3, enc4, enc_hidden, result = None, None, None, None, None, None

        if self.resolution_factor <= 1:
            enc1 = self.encoder1(x_in)
        if self.resolution_factor <= 2:
            x2 = outs[0]  # Highest resolution from backbone
            enc2 = self.encoder2(x2) if hasattr(self, 'encoder2') else x2
        if self.resolution_factor <= 4:
            x3 = outs[1]
            enc3 = self.encoder3(x3) if hasattr(self, 'encoder3') else x3
        if self.resolution_factor <= 8:
            x4 = outs[2]
            enc4 = self.encoder4(x4) if hasattr(self, 'encoder4') else x4
        if self.resolution_factor <= 16:
            enc_hidden = self.encoder5(outs[3])
            result = enc_hidden  # Temporary variable to store the output result

        # Decoder Pass (start from 8x resolution)

        if self.resolution_factor <= 8:
            dec3 = self.decoder5(enc_hidden, enc4 if hasattr(self, 'encoder4') else None)
            result = dec3
        if self.resolution_factor <= 4:
            dec2 = self.decoder4(dec3, enc3 if hasattr(self, 'encoder3') else None)
            result = dec2
        if self.resolution_factor <= 2:
            dec1 = self.decoder3(dec2, enc2 if hasattr(self, 'encoder2') else None)
            result = dec1
        if self.resolution_factor <= 1:
            dec0 = self.decoder2(dec1, enc1 if hasattr(self, 'encoder1') else None)
            result = self.decoder1(dec0)

        ## feat = self.conv_proj(dec4)
        # Return the final result, passed through the output layer
        return self.out(result)
