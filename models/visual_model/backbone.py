# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import math
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from functools import partial
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import PatchEmbed

from utils.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, name:str, backbone: nn.Module, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 return_interm_layers: bool,
                 dilation: bool):

        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=FrozenBatchNorm2d)
            # pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        assert name in ('resnet50', 'resnet101')
        num_channels = 2048
        super().__init__(name, backbone, num_channels, return_interm_layers)


class CustomPatchEmbed(PatchEmbed):
    """ 
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__(img_size, patch_size, in_chans, embed_dim, flatten=flatten)

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class ViTBackbone(nn.Module):
    """ ViT backbone."""
    def __init__(self, name: str, token_spatial_size: int):
        super().__init__()

        assert name in ['vit_tiny', \
                        'vit_small', \
                        'vit_base']

        if name == 'vit_tiny':
            embed_dim, num_heads, distilled = 192, 3, True
        elif name == 'vit_small':
            embed_dim, num_heads, distilled = 384, 6, True
        elif name == 'vit_base':
            embed_dim, num_heads, distilled = 768, 12, True

        backbone = VisionTransformer(
            patch_size=16, embed_dim=embed_dim, depth=12, num_heads=num_heads,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            distilled=distilled, embed_layer=partial(CustomPatchEmbed, flatten=False)
        )

        self.patch_embed = backbone.patch_embed
        self.pos_drop    = backbone.pos_drop
        self.blocks      = backbone.blocks
        self.norm        = backbone.norm

        self.pos_embed   = nn.Parameter(torch.zeros(1, embed_dim, token_spatial_size, token_spatial_size))

        # excluding cls and distill tokens
        # pos_embed   = backbone.pos_embed
        # self.pos_embed = pos_embed[:, 2:, :] if distilled is True \
        #                                         else pos_embed[:, 1:, :]
        self.num_channels = embed_dim #* 4
        
        self.window_size = [token_spatial_size // 2**(2-i) for i in range(3)]
        self.num_blocks_each_stage = [3, 6, 3]
        
        # # freeze parameters of the first stage
        # for p in [self.patch_embed, self.pos_embed]:
        #     p.requires_grad_(False)

        # for idx in range(self.num_blocks_each_stage[0]):
        #     for p in self.blocks[idx].parameters():
        #         p.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        out: Dict[str, NestedTensor] = {}
        data = tensor_list.tensors
        mask = tensor_list.mask[None].float()
        # pos_embed = self._resize_pos_embed((data.shape[2]//16, data.shape[3]//16))

        x = self.patch_embed(data)
        x = self.pos_drop(x + self.pos_embed)
        x = self.forward_by_windows(x)
        # x = self.blocks(x)
        x = self.norm(x)
        out_size = int(math.sqrt(x.shape[1]))
        x = x.reshape(x.shape[0], out_size, out_size, -1).permute(0, 3, 1, 2)
        
        # out_h, out_w = x.shape[2], x.shape[3]
        # out_h, out_w  = x.shape[2] // 2, x.shape[3] // 2
        # x = F.unfold(x, (2, 2), stride=(2, 2))
        # x = x.reshape(x.shape[0], x.shape[1], out_h, out_w)
        
        mask = F.interpolate(mask, size=(out_size, out_size)).to(torch.bool)[0]
        out['out'] = NestedTensor(x.contiguous(), mask)

        return out
    
    def forward_by_windows(self, x):
        N, C, H, W = x.shape
        num_stages = len(self.window_size)

        start_block = 0
        for idx in range(num_stages):
            win_shape = (self.window_size[idx], self.window_size[idx])
            win_size = win_shape[0] * win_shape[1]
            win_num = H * W // win_size

            x = F.unfold(x, win_shape, stride=win_shape) # (N, C*win_size, win_num)
            x = x.permute(0, 2, 1).contiguous() # (N, win_num, C*win_size)
            x = x.reshape(N*win_num, C, win_size)
            x = x.permute(0, 2, 1).contiguous() # (N*win_num, win_size, C)

            x = self.blocks[start_block:start_block+self.num_blocks_each_stage[idx]](x)
            
            x = x.permute(0, 2, 1).contiguous() # (N*win_num, C, win_size)
            x = x.reshape(N, win_num, C*win_size) # (N, win_num, C*win_size)
            x = x.permute(0, 2, 1).contiguous() # (N, C*win_size, win_num)
            x = F.fold(x, (H, W), win_shape, stride=win_shape)

            start_block += self.num_blocks_each_stage[idx]
        
        x = x.flatten(2).transpose(1, 2).contiguous()
        
        return x
    
    def _resize_pos_embed(self, tgt_size):
        resized_pos_embed = F.interpolate(self.pos_embed, size=tgt_size, mode='bicubic', align_corners=False)
        self.pos_embed = nn.Parameter(resized_pos_embed)
        

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    if args.backbone in ['resnet50', 'resnet101']:
        return_interm_layers = False
        backbone = Backbone(args.backbone, return_interm_layers, args.dilation)
    elif args.backbone in ['vit_tiny', 'vit_small', 'vit_base']:
        token_spatial_size = args.imsize // 16
        backbone = ViTBackbone(args.backbone, token_spatial_size)
        # backbone._resize_pos_embed((tgt_size, tgt_size))

    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
