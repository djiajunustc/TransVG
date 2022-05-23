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


class Block(nn.Module):

    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 drop=0., 
                 attn_drop=0., 
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionV2(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_attn_mask=None):
        norm_x = self.norm1(x)
        x = x + self.drop_path(self.attn(norm_x, norm_x, norm_x, x_attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformers """
    def __init__(self, 
                 img_size=640, 
                 text_token_dim=768, 
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
                 reg_out_type='reg_token',
                 language_query_max_len=20,
                 num_modulation=4,
                 modulate_in_last_blocks=False,
                 reg_token_in_last_blocks=False,
                 without_visual_mask=False,
                 num_vpt=0
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
        # self.modulation_loc = modulation_loc
        self.reg_out_type = reg_out_type
        self.reg_token_in_last_blocks = reg_token_in_last_blocks
        self.modulate_in_last_blocks = modulate_in_last_blocks
        self.without_visual_mask = without_visual_mask
        self.num_vpt = num_vpt
        self.query_len = language_query_max_len

        if self.reg_token_in_last_blocks:
            assert self.modulate_in_last_blocks, \
                    'when adding reg_token in last blocks, v-l modulation should be also conducted in last blocks'

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.text_proj = nn.Linear(text_token_dim, embed_dim)
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, flatten=False)
        self.embed_shape = self.patch_embed.embed_shape
        
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.embed_shape, self.embed_shape))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.reg_out_type == 'reg_token':
            self.reg_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # nn.init.normal_(self.reg_token, std=0.02)

        _block = Block

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
        if self.modulate_in_last_blocks:
            self.modulation_loc = [depth - num_modulation + i for i in range(num_modulation)]
        else:
            self.modulation_loc = [(depth//num_modulation)*(i+1)-1 for i in range(num_modulation)]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_list = []
        for i in range(depth):
            block_list.append(basic_block())    
        self.blocks = nn.ModuleList(block_list)

        self.norm = norm_layer(embed_dim)
        
        self.lang_modulation_embedding = nn.Parameter(torch.zeros(num_modulation, self.query_len, embed_dim))
        pre_fusion_blocks_list = []
        for i in range(num_modulation):
            pre_fusion_blocks_list.append(basic_block())
        self.lang_modulation_pre_blocks = nn.ModuleList(pre_fusion_blocks_list)

        if self.num_vpt > 0:
            assert self.modulate_in_last_blocks and self.reg_token_in_last_blocks, \
                'When using prompt tuning, vl-modulation conducted in last blocks and reg_token added in last blocks'
            self.prompt_tokens = nn.Parameter(torch.zeros(depth - num_modulation, self.num_vpt, embed_dim))

        self._reset_parameters()
        self._freeze_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _freeze_parameters(self):
        if self.num_vpt > 0:
            self.pos_drop.eval()
            self.patch_embed.eval()
            self.pos_embed.requires_grad = False
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        
            for i in range(len(self.blocks) - len(self.modulation_loc)):
                self.blocks[i].eval()
                for param in self.blocks[i].parameters():
                    param.requires_grad = False

    def _forward_with_reg_token(self, visu_src, ling_src, visu_mask, ling_mask):
        batch_size = visu_src.shape[0]

        visu_src = self.patch_embed(visu_src)
        visu_src = self.pos_drop(visu_src + self.pos_embed)
        visu_src = visu_src.flatten(2).transpose(1, 2)
        reg_src  = self.reg_token.expand(batch_size, -1, -1)

        ling_mask = ling_mask.view(batch_size, 1, 1, -1).expand(-1, self.num_heads, -1, -1)
        visu_mask = torch.zeros(batch_size, 1+visu_src.shape[1], dtype=visu_src.dtype, device=visu_src.device)
        mask = visu_mask.view(batch_size, 1, 1, -1).expand(-1, self.num_heads, -1, -1)

        src = torch.cat([reg_src, visu_src], dim=1)

        count_modulation = 0
        for i, block in enumerate(self.blocks):
            if i not in self.modulation_loc:
                src = block(src, mask)
            else:
                this_lang_pos = self.lang_modulation_embedding[count_modulation].unsqueeze(0).expand(batch_size, -1, -1)
                this_lang_src = ling_src + this_lang_pos
                this_lang_src = self.lang_modulation_pre_blocks[count_modulation](this_lang_src, ling_mask)
                this_src = torch.cat([src, this_lang_src], dim=1)
                this_mask = torch.cat([mask, ling_mask], dim=-1)
                this_src = block(this_src, this_mask)
                src = this_src[:, :-1*self.query_len, :]
                count_modulation += 1
                
        reg_src = src[:, 0, :]

        if self.norm is not None:
            reg_src = self.norm(reg_src)

        reg_src = reg_src.squeeze(1)

        return reg_src
        
    def forward(self, visu_src, text_src, visu_mask, text_mask):
        # language features projection
        text_src = self.text_proj(text_src)

        if self.reg_out_type == 'reg_token':
            return self._forward_with_reg_token(visu_src, text_src, visu_mask, text_mask)
        else:
            raise ValueError('reg_out_type should be one of ["reg_token", "avg_out"]')

    def _resize_pos_embed(self, tgt_size):
        resized_pos_embed = F.interpolate(self.pos_embed, size=tgt_size, mode='bicubic', align_corners=False)
        self.pos_embed = nn.Parameter(resized_pos_embed)


def build_visual_branch_prompt_fusion(args, text_token_dim):
    if args.vit_model == 'tiny':
        embed_dim, num_heads, mlp_ratio = 192, 3, 4
    elif args.vit_model == 'small':
        embed_dim, num_heads, mlp_ratio = 384, 6, 4
    elif args.vit_model == 'base':
        embed_dim, num_heads, mlp_ratio = 768, 12, 4
    elif args.vit_model == 'detr_like':
        embed_dim, num_heads, mlp_ratio = 256, 8, 8
    
    model = VisionTransformer(img_size=args.imsize, 
                              text_token_dim=text_token_dim,
                              embed_dim=embed_dim, 
                              num_heads=num_heads, 
                              mlp_ratio=mlp_ratio, 
                              reg_out_type=args.reg_out_type,
                            #   use_block_v2=args.use_block_v2,
                              language_query_max_len=args.max_query_len,
                              num_modulation=args.num_modulation,
                              modulate_in_last_blocks=args.modulate_in_last_blocks,
                              reg_token_in_last_blocks=args.reg_token_in_last_blocks,
                              without_visual_mask=args.without_visual_mask,
                              num_vpt=args.num_vpt
                              )

    return model