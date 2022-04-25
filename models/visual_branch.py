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


class ConcatLinearModulation(nn.Module):
    
    def __init__(self,
                 dim,
                 mlp_ratio=4,
                 drop=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        concat_dim = int(dim * 2)
        hidden_dim = int(dim * mlp_ratio)

        self.lang_mapping = nn.Sequential(
            nn.Linear(dim, dim),
            norm_layer(dim),
            # act_layer(dim),
            # nn.Dropout(drop),
            # nn.Linear(dim, dim),
            # norm_layer(dim),
            # act_layer(dim)
        )
        
        self.fc1 = nn.Linear(concat_dim, hidden_dim)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x, y):
        # get cls token
        y = y[:, 0] 
        y = self.lang_mapping(y)
        x = torch.cat([x, y], dim=-1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class ClsTokenModulation(nn.Module):
    
    def __init__(self,
                 dim,
                 mlp_ratio=4,
                 drop=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.norm = norm_layer(hidden_dim)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        # get cls token
        x = x[:, 0] 
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class Block_v1(nn.Module):

    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 drop=0., 
                 attn_drop=0., 
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,
                 language_modulation=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionV2(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.language_modulation= language_modulation
        if self.language_modulation is not None:
            if self.language_modulation == 'cross_attn':
                self.lang_modulation = AttentionV2(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif self.language_modulation == 'concat_linear':
                self.lang_modulation = ConcatLinearModulation(dim=dim, mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer)
            elif self.language_modulation == 'cls_token':
                self.lang_modulation = ClsTokenModulation(dim=dim, mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer)
            else:
                raise ValueError('language_modulation can only be one of ["cross_attn", "concat_linear", "cls_token"]')

    def forward(self, x, y=None, x_attn_mask=None, y_attn_mask=None):
        norm_x = self.norm1(x)
        x = x + self.drop_path(self.attn(norm_x, norm_x, norm_x, x_attn_mask))
        
        if self.language_modulation is not None:
            assert y is not None
        
        if self.language_modulation == 'cross_attn':
            x = x + self.drop_path(self.lang_modulation(x, y, y, y_attn_mask))
        elif self.language_modulation == 'concat_linear':
            x = x + self.drop_path(self.lang_modulation(x, y))
        elif self.language_modulation == 'cls_token':
            x = x + self.drop_path(self.lang_modulation(y))
        
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class Block_v2(nn.Module):

    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 drop=0., 
                 attn_drop=0., 
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,
                 language_modulation=None
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionV2(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.language_modulation= language_modulation
        if self.language_modulation is not None:
            if self.language_modulation == 'cross_attn':
                self.norm3 = norm_layer(dim)
                self.lang_modulation = AttentionV2(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif self.language_modulation == 'concat_linear':
                self.norm3 = norm_layer(dim)
                self.lang_modulation = ConcatLinearModulation(dim=dim, mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer)
            elif self.language_modulation == 'cls_token':
                self.lang_modulation = ClsTokenModulation(dim=dim, mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer)
            else:
                raise ValueError('language_modulation can only be one of ["cross_attn", "concat_linear", "cls_token"]')

    def forward(self, x, y=None, x_attn_mask=None, y_attn_mask=None):
        norm_x = self.norm1(x)
        x = x + self.drop_path(self.attn(norm_x, norm_x, norm_x, x_attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        if self.language_modulation:
            assert y is not None
        
        if self.language_modulation == 'cross_attn':
            x = x + self.drop_path(self.lang_modulation(self.norm3(x), y, y, y_attn_mask))
        elif self.language_modulation == 'concat_linear':
            x = x + self.drop_path(self.lang_modulation(self.norm3(x), y))
        elif self.language_modulation == 'cls_token':
            x = x + self.drop_path(self.lang_modulation(y))

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformers """
    def __init__(self, 
                 img_size=640, 
                 patch_size=16, 
                 in_chans=3, 
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 use_block_v2=False,
                 reg_out_type='reg_token',
                 language_modulation='cross_attn',
                 modulation_loc=[8, 9, 10, 11], 
                 without_visual_mask=False
                 ):
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
        self.modulation_loc = modulation_loc
        self.reg_out_type = reg_out_type
        self.without_visual_mask = without_visual_mask

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, flatten=False)
        self.embed_shape = self.patch_embed.embed_shape
        
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.embed_shape, self.embed_shape))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.reg_out_type == 'reg_token':
            self.reg_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # nn.init.normal_(self.reg_token, std=0.02)

        if use_block_v2:
            _block = Block_v2
        else:
            _block = Block_v1

        basic_block = partial(_block, 
                              dim=embed_dim,
                              num_heads=num_heads,
                              mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias,
                              drop=drop_rate,
                              attn_drop=attn_drop_rate,
                              norm_layer=norm_layer,
                              act_layer=act_layer
                              )
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_list = []
        for i in range(depth):
            if i in self.modulation_loc:
                block_list.append(basic_block(language_modulation=language_modulation))
            else:
                block_list.append(basic_block())    
        self.blocks = nn.ModuleList(block_list)

        self.norm = norm_layer(embed_dim)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _forward_with_reg_token(self, visu_src, ling_src, visu_mask, ling_mask):
        batch_size = visu_src.shape[0]

        visu_src = self.patch_embed(visu_src)
        visu_src = self.pos_drop(visu_src + self.pos_embed)
        
        visu_src = visu_src.flatten(2).transpose(1, 2)
        
        visu_mask = visu_mask[None].float()
        visu_mask = F.interpolate(visu_mask, size=(self.embed_shape, self.embed_shape)).to(torch.bool)[0]
        visu_mask = visu_mask.flatten(1)

        # concatenate [REG] token and visual tokens
        reg_src = self.reg_token.expand(batch_size, -1, -1)
        visu_src = torch.cat([reg_src, visu_src], dim=1)

        reg_mask = torch.zeros(batch_size, 1, dtype=reg_src.dtype, device=reg_src.device)
        visu_mask = torch.cat([reg_mask, visu_mask], dim=1)

        visu_mask_expanded = visu_mask.view(batch_size, 1, 1, -1).expand(-1, self.num_heads, -1, -1)
        ling_mask_expanded = ling_mask.view(batch_size, 1, 1, -1).expand(-1, self.num_heads, -1, -1)
        
        if self.without_visual_mask:
            visu_mask_expanded = None

        for i, block in enumerate(self.blocks):
            if i in self.modulation_loc:
                visu_src = block(visu_src, ling_src, visu_mask_expanded, ling_mask_expanded)
            else:
                visu_src = block(visu_src, visu_mask_expanded)
        reg_src = visu_src[:, 0, :]

        if self.norm is not None:
            reg_src = self.norm(reg_src)

        reg_src = reg_src.squeeze(1)

        return reg_src


    def _forward_with_avg_out(self, visu_src, ling_src, visu_mask, ling_mask):
        batch_size = visu_src.shape[0]

        visu_src = self.patch_embed(visu_src)
        visu_src = self.pos_drop(visu_src + self.pos_embed)
        
        visu_src = visu_src.flatten(2).transpose(1, 2)
        
        visu_mask = visu_mask[None].float()
        visu_mask = F.interpolate(visu_mask, size=(self.embed_shape, self.embed_shape)).to(torch.bool)[0]
        
        visu_mask_expanded = visu_mask.view(batch_size, 1, 1, -1).expand(-1, self.num_heads, -1, -1)
        ling_mask_expanded = ling_mask.view(batch_size, 1, 1, -1).expand(-1, self.num_heads, -1, -1)

        if self.without_visual_mask:
            visu_mask_expanded = None
            
        for i, block in enumerate(self.blocks):
            if i in self.modulation_loc:
                visu_src = block(visu_src, ling_src, visu_mask_expanded, ling_mask_expanded)
            else:
                visu_src = block(visu_src, visu_mask_expanded)

        if self.norm is not None:
            visu_src = self.norm(visu_src)
        
        reg_src = visu_src.mean(dim=1)

        return reg_src


    def forward(self, visu_src, ling_src, visu_mask, ling_mask):

        if self.reg_out_type == 'reg_token':
            return self._forward_with_reg_token(visu_src, ling_src, visu_mask, ling_mask)
        elif self.reg_out_type == 'avg_out':
            return self._forward_with_avg_out(visu_src,ling_src, visu_mask, ling_mask)
        else:
            raise ValueError('reg_out_type should be one of ["reg_token", "avg_out"]')

    def _resize_pos_embed(self, tgt_size):
        resized_pos_embed = F.interpolate(self.pos_embed, size=tgt_size, mode='bicubic', align_corners=False)
        self.pos_embed = nn.Parameter(resized_pos_embed)


def build_visual_branch(args):
    if args.vit_model == 'tiny':
        embed_dim, num_heads, mlp_ratio = 192, 3, 4
    elif args.vit_model == 'small':
        embed_dim, num_heads, mlp_ratio = 384, 6, 4
    elif args.vit_model == 'base':
        embed_dim, num_heads, mlp_ratio = 768, 12, 4
    elif args.vit_model == 'detr_like':
        embed_dim, num_heads, mlp_ratio = 256, 8, 8
    
    model = VisionTransformer(img_size=args.imsize, 
                              patch_size=16, 
                              embed_dim=embed_dim, 
                              depth=12, 
                              num_heads=num_heads, 
                              mlp_ratio=mlp_ratio, 
                              qkv_bias=True, 
                              reg_out_type=args.reg_out_type,
                              use_block_v2=args.use_block_v2,
                              modulation_loc=args.modulation_loc,
                              language_modulation=args.language_modulation,
                              without_visual_mask=args.without_visual_mask
                              )

    return model