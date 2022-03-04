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


class VisualModel(nn.Module):
    """ Visual Model with Transformers (ViT)."""
    def __init__(self, vit, embed_dim: int, token_spatial_size: int):
        super().__init__()

        self.patch_embed = vit.patch_embed
        self.pos_drop    = vit.pos_drop
        self.blocks      = vit.blocks
        self.norm        = vit.norm

        self.pos_embed   = nn.Parameter(torch.zeros(1, embed_dim, token_spatial_size, token_spatial_size))

        self.num_channels = embed_dim
        
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

        x = self.patch_embed(data)
        x = self.pos_drop(x + self.pos_embed)
        x = self.forward_by_windows(x)
        x = self.norm(x)
        out_size = int(math.sqrt(x.shape[1]))
        # [B, C, out_size, out_size]
        x = x.reshape(x.shape[0], out_size, out_size, -1).permute(0, 3, 1, 2)
        
        # out_h, out_w = x.shape[2], x.shape[3]
        # out_h, out_w  = x.shape[2] // 2, x.shape[3] // 2
        # x = F.unfold(x, (2, 2), stride=(2, 2))
        # x = x.reshape(x.shape[0], x.shape[1], out_h, out_w)
        
        mask = F.interpolate(mask, size=(out_size, out_size)).to(torch.bool)[0]
        out = NestedTensor(x.contiguous(), mask)

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


def build_visual_branch(args):
    if args.vit_model == 'vit_tiny':
        embed_dim, num_heads = 192, 3
    elif args.vit_model == 'vit_small':
        embed_dim, num_heads = 384, 6
    elif args.vit_model == 'vit_base':
        embed_dim, num_heads = 768, 12
    
    vit = VisionTransformer(patch_size=16, embed_dim=embed_dim, depth=12, \
                            num_heads=num_heads, mlp_ratio=4, qkv_bias=True, \
                            norm_layer=partial(nn.LayerNorm, eps=1e-6), \
                            embed_layer=partial(CustomPatchEmbed, flatten=False), \
                            distilled=True
                            )
    token_spatial_size = int(args.imsize // 16)

    model = VisualModel(vit, embed_dim, token_spatial_size)

    return model