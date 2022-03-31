import torch
import torch.nn as nn
import torch.nn.functional as F

# from .comp_visual_branch import build_visual_branch
from .visual_branch import build_visual_branch
from .linguistic_branch import build_linguistic_branch
# from .vl_transformer import build_vl_transformer
from .vl_module import build_vl_module
from .vl_module_no_rt import build_vl_module_no_rt


class TransVG(nn.Module):
    def __init__(self, args):
        super(TransVG, self).__init__()
        hidden_dim = args.vl_hidden_dim
        
        self.linguistic_branch = build_linguistic_branch(args)
        self.visual_branch = build_visual_branch(args)

        self.text_proj = nn.Linear(self.linguistic_branch.num_channels, hidden_dim)

        self.bbox_embed = MLP(hidden_dim, 256, 4, 3)

    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]

        # Language branch
        ling_out = self.linguistic_branch(text_data)
        ling_src, ling_mask = ling_out.decompose()
        ling_src = self.text_proj(ling_src)

        # Visual-Linguistic module
        visu_src, visu_mask = img_data.decompose()
        reg_src = self.visual_branch(visu_src, ling_src, visu_mask, ling_mask)

        pred_box = self.bbox_embed(reg_src).sigmoid()

        return pred_box


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
