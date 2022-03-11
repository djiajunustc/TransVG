import math
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from functools import partial
from timm.models.vision_transformer import VisionTransformer
# from timm.models.layers import PatchEmbed
from utils.misc import NestedTensor, is_main_process
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AttentionSparseKV(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def complexity(self, num_inputs, num_queries):
        num_channels = self.dim
        comp = num_queries * num_channels ** 2  # q embed
        comp += (num_inputs * num_channels ** 2) * 2  # kv embed
        comp += (num_queries * num_inputs * num_channels) * 2  # attention
        comp += num_queries * num_inputs * self.num_heads * 3  # softmax
        comp += num_queries * num_channels ** 2  # proj
        return comp

    def forward(self, x, q, q_lengths):
        B, N, C = x.shape
        q = self.q(q).reshape(-1, self.num_heads, C // self.num_heads).permute(1, 0, 2)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 3, 0, 1, 4)
        k, v = kv[0], kv[1]   # make torchscript happy (cannot use tensor as tuple)

        if not self.training:
            x = batched_sparse_attention(q, k, v, q_lengths, self.scale)
            x = x.transpose(0, 1).reshape(-1, C)
        else:
            if (q_lengths.max() - q_lengths.min()) == 0:
                q = q.reshape(self.num_heads, B, -1, C // self.num_heads)
                attn = (q @ k.transpose(-1, -2)) * self.scale
                attn = attn.softmax(dim=-1, dtype=v.dtype)
                attn = self.attn_drop(attn)
                x = (attn @ v).permute(1, 2, 0, 3).reshape(-1, C)
            else:
                kv_lengths = q_lengths.new_full([B], kv.shape[3])
                k = k.reshape(self.num_heads, -1, C // self.num_heads)
                v = v.reshape(self.num_heads, -1, C // self.num_heads)
                attn = batched_sparse_gemm(q, k, q_lengths, kv_lengths, False, True) * self.scale
                attn = attn.softmax(dim=-1, dtype=v.dtype)
                attn = self.attn_drop(attn)
                x = batched_sparse_gemm(attn, v, q_lengths, kv_lengths, False, False)
                x = x.transpose(0, 1).reshape(-1, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockSparseKV(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, split_sizes=[2, 1]):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.dge = DynamicGrainedEncoder(in_channels=dim, split_sizes=split_sizes, complexity_handler=self.complexity)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        k = self.dge.compress(x, H, W)
        x = x + self.drop_path(self.atten(self.norm1(x), self.norm1(k), self.dge.states["batches"]))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ 
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.embed_shape = int(img_size//patch_size)
        self.num_patches = self.embed_shape ** 2
        self.flatten = flatten
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class WindowedViT(nn.Module):
    """ Windowed Vision Transformers """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None, 
                 sparse=False):
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
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, flatten=False)
        embed_shape = self.patch_embed.embed_shape

        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, embed_shape, embed_shape))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.window_size = [embed_shape//4, embed_shape//2, embed_shape]
        self.num_blocks_each_stage = [3, 6, 3]


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


def build_visual_branch(args):
    if args.vit_model == 'tiny':
        embed_dim, num_heads = 192, 3
    elif args.vit_model == 'small':
        embed_dim, num_heads = 384, 6
    elif args.vit_model == 'base':
        embed_dim, num_heads = 768, 12
    
    # embed_shape2d = [int(args.imsize//16) for _ in range(2)]
    model = WindowedViT(img_size=args.imsize, patch_size=16, embed_dim=embed_dim, \
                        depth=12, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, \
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), \
                        embed_layer=partial(PatchEmbed, flatten=False), \
                        sparse=args.sparse_vit)

    return model