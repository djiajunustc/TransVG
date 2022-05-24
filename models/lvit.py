import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_branch import build_visual_branch
from .language_branch import build_language_branch

# experiments
from .visual_branch_prompt_fusion import build_visual_branch_prompt_fusion


class LViT(nn.Module):
    def __init__(self, args):
        super(LViT, self).__init__()

        self.language_branch = build_language_branch(args)

        if args.language_modulation in ['deep_prompt', 'shallow_prompt']:
            self.visual_branch = build_visual_branch_prompt_fusion(args)
        else:
            self.visual_branch = build_visual_branch(args)

        embed_dim = self.visual_branch.embed_dim
        self.text_proj = nn.Linear(self.language_branch.num_channels, embed_dim)

        self.prediction_head = MLP(self.visual_branch.num_channels, 256, 4, 3)

    def forward(self, img_data, text_data):
        visu_src, visu_mask = img_data.decompose()
        text_src, text_mask = text_data.decompose()
        
        # Language branch
        text_src, text_mask = self.language_branch(text_src, text_mask)
        text_src = self.text_proj(text_src)

        # Language conditioned vision branch
        reg_src = self.visual_branch(visu_src, text_src, visu_mask, text_mask)

        # Prediction head
        pred_box = self.prediction_head(reg_src).sigmoid()

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
