from locale import normalize
import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List
from functools import partial
# from timm.models.layers import PatchEmbed
# from timm.models.layers import Mlp, DropPath #, trunc_normal_, lecun_normal_
from .layers import Attention, AttentionV2, PatchEmbed


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0., drop_path=0., activation='relu', 
                 norm_layer=nn.LayerNorm, normalize_before=False):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.attn = AttentionV2(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
                                attn_drop=attn_drop, proj_drop=0.)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.dropout = nn.Dropout(drop)
        self.linear2 = nn.Linear(hidden_dim, dim)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward_post(self, q_src, k_src, v_src, qk_attn_mask):
        q_src2 = self.attn(q_src, k_src, v_src, attn_mask=qk_attn_mask)
        q_src = q_src + self.dropout1(q_src2)
        q_src = self.norm1(q_src)
        q_src2 = self.linear2(self.dropout(self.activation(self.linear1(q_src))))
        q_src = q_src + self.dropout2(q_src2)
        q_src = self.norm2(q_src)
        return q_src
    
    def forward_pre(self, q_src, k_src, v_src, qk_attn_mask):
        q_src2 = self.norm1(q_src)
        k_src2 = self.norm1(k_src)
        v_src2 = self.norm1(v_src)
        q_src2 = self.attn(q_src2, k_src2, v_src2, attn_mask=qk_attn_mask)
        q_src = q_src + self.dropout1(q_src2)
        q_src2 = self.norm2(q_src)
        q_src2 = self.linear2(self.dropout(self.activation(self.linear1(q_src2))))
        q_src = q_src + self.dropout2(q_src2)
        return q_src

    def forward(self, q_src, k_src, v_src, attn_mask):
        if self.normalize_before:
            return self.forward_pre(q_src, k_src, v_src, attn_mask)
        else:
            return self.forward_post(q_src, k_src, v_src, attn_mask)


class VLTransformer(nn.Module):
    """ Visual-Linguistic Transformer """
    def __init__(self, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 num_vtoken=400, num_ltoken=40, norm_layer=None, activation='relu', 
                 normalize_before=False, sparse_type=None):
        
        super().__init__()
        self.num_channels = self.embed_dim = embed_dim
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.num_heads = num_heads
        self.num_token = 1 + num_vtoken + num_ltoken
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_token, embed_dim))
        # self.reg_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.pos_embed = nn.Embedding(self.num_token, embed_dim)
        self.reg_token = nn.Embedding(1, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_list = [
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, \
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, \
                drop_path=dpr[i], norm_layer=norm_layer, activation=activation, \
                normalize_before=normalize_before)
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(block_list)
        # self.blocks = nn.ModuleList(*[
        #     Block(
        #         dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, \
        #         qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, \
        #         drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, \
        #         normalize_before=normalize_before)
        #     for i in range(depth)])
        self.norm = norm_layer(embed_dim) if normalize_before else None

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, visu_src, visu_mask, lang_src, lang_mask):
        batch_size = visu_src.shape[0]

        reg_src = self.reg_token.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        reg_mask = torch.zeros((batch_size, 1)).to(reg_src.device).to(torch.bool)
        rvl_src  = torch.cat([reg_src, visu_src, lang_src], dim=1)
        rvl_mask = torch.cat([reg_mask, visu_mask, lang_mask], dim=1)
        # pos_embed = self.pos_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        rvl_src = rvl_src + self.pos_embed.weight.unsqueeze(0)

        # import pdb
        # pdb.set_trace()

        # reg_src = self.reg_token.weight.unsqueeze(1).repeat(1, batch_size, 1)
        # reg_mask = torch.zeros((batch_size, 1)).to(reg_src.device).to(torch.bool)
        # rvl_src  = torch.cat([reg_src, visu_src, lang_src], dim=0)
        # rvl_mask = torch.cat([reg_mask, visu_mask, lang_mask], dim=1)
        # pos_embed = self.pos_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        # rvl_src = rvl_src + pos_embed

        rvl_mask = rvl_mask.view(batch_size, 1, 1, self.num_token). \
            expand(-1, self.num_heads, -1, -1)
            # . \
            # reshape(batch_size * self.num_heads, 1, self.num_token)
        
        for block in self.blocks:
            rvl_src = block(rvl_src, rvl_src, rvl_src, rvl_mask)
        reg_hs = rvl_src[:, 0, :]
        # import pdb
        # pdb.set_trace()
        if self.norm is not None:
            reg_hs = self.norm(reg_hs)

        return reg_hs
        

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



def build_vl_module(args, num_vtoken, num_ltoken):
    if args.vit_model == 'tiny':
        embed_dim, num_heads = 192, 3
    elif args.vit_model == 'small':
        embed_dim, num_heads = 384, 6
    elif args.vit_model == 'base':
        embed_dim, num_heads = 768, 12

    model = VLTransformer(embed_dim=embed_dim, depth=6, \
                          num_heads=num_heads, mlp_ratio=4, qkv_bias=True, \
                          num_vtoken=num_vtoken, num_ltoken=num_ltoken, \
                          norm_layer=partial(nn.LayerNorm, eps=1e-5), \
                          normalize_before=args.vl_normalize_before, \
                          activation=args.vl_activation, 
                          drop_rate=0.1
                          )
    
    return model