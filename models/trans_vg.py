import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertModel
from .visual_model.detr import build_detr
from .language_model.bert import build_bert
from .vl_transformer import build_vl_transformer
from utils.box_utils import xywh2xyxy


class TransVG(nn.Module):
    def __init__(self, args):
        super(TransVG, self).__init__()
        hidden_dim = args.vl_hidden_dim
        divisor = 16 if args.dilation else 32
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        self.num_text_token = args.max_query_len

        self.visumodel = build_detr(args)
        self.textmodel = build_bert(args)

        num_total = self.num_visu_token + self.num_text_token + 1
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)

        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

        self.vl_transformer = build_vl_transformer(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)


    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]

        # visual backbone
        visu_mask, visu_src = self.visumodel(img_data)
        visu_src = self.visu_proj(visu_src) # (N*B)xC

        # language bert
        text_fea = self.textmodel(text_data)
        text_src, text_mask = text_fea.decompose()
        assert text_mask is not None
        text_src = self.text_proj(text_src)
        # permute BxLenxC to LenxBxC
        text_src = text_src.permute(1, 0, 2)
        text_mask = text_mask.flatten(1)

        # target regression token
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt_mask = torch.zeros((bs, 1)).to(tgt_src.device).to(torch.bool)
        
        vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)
        vl_mask = torch.cat([tgt_mask, text_mask, visu_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos) # (1+L+N)xBxC
        vg_hs = vg_hs[0]

        pred_box = self.bbox_embed(vg_hs).sigmoid()

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
