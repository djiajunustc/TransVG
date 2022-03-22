import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List
from functools import partial
# from timm.models.layers import PatchEmbed
from utils.misc import NestedTensor
from timm.models.layers import Mlp, DropPath #, trunc_normal_, lecun_normal_
from .layers import AttentionV2, PatchEmbed


# class VisualBlock(nn.Module):

#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x


class VisualBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionV2(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        norm_x = self.norm1(x)
        x = x + self.drop_path(self.attn(norm_x, norm_x, norm_x, attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformers """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, \
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None, 
                 VL_LOC=[3, 6, 9]):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_channels = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_heads = num_heads

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, flatten=False)
        self.embed_shape = self.patch_embed.embed_shape
        
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.embed_shape, self.embed_shape))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            VisualBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], 
                norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
    
    def forward(self, img_data: NestedTensor):
        visu_src, visu_mask = img_data.decompose()
        batch_size = visu_src.shape[0]

        visu_src = self.patch_embed(visu_src)
        visu_src = self.pos_drop(visu_src + self.pos_embed)
        visu_src = visu_src.flatten(2).transpose(1, 2)
        # visu_size = int(math.sqrt(visu_src.shape[1]))
        
        visu_mask = visu_mask[None].float()
        visu_mask = F.interpolate(visu_mask, size=(self.embed_shape, self.embed_shape)).to(torch.bool)[0]
        visu_mask_expanded = visu_mask.view(batch_size, 1, 1, self.embed_shape**2). \
            expand(-1, self.num_heads, -1, -1)
        
        for i, block in enumerate(self.blocks):
            visu_src = block(visu_src, visu_mask_expanded)

        if self.norm is not None:
            visu_src = self.norm(visu_src)
        
        visu_src = visu_src.reshape(batch_size, self.embed_shape, self.embed_shape, -1).permute(0, 3, 1, 2)
        out = NestedTensor(visu_src.contiguous(), visu_mask)

        return out

    def _resize_pos_embed(self, tgt_size):
        resized_pos_embed = F.interpolate(self.pos_embed, size=tgt_size, mode='bicubic', align_corners=False)
        self.pos_embed = nn.Parameter(resized_pos_embed)


def build_visual_branch(args):
    if args.vit_model == 'tiny':
        embed_dim, num_heads = 192, 3
    elif args.vit_model == 'small':
        embed_dim, num_heads = 384, 6
    elif args.vit_model == 'base':
        embed_dim, num_heads = 768, 12
    
    # embed_shape2d = [int(args.imsize//16) for _ in range(2)]
    model = VisionTransformer(img_size=args.imsize, patch_size=16, \
                        embed_dim=embed_dim, depth=12, num_heads=num_heads, \
                        mlp_ratio=4, qkv_bias=True, \
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), \
                        embed_layer=partial(PatchEmbed, flatten=False), \
                        VL_LOC=[3,6,9])

    return model