import torch
import torch.nn.functional as F

from torch import nn
from utils.misc import NestedTensor
from transformers import AutoModel
from transformers import RobertaModel, BertModel


def mlp_mapping(input_dim, output_dim):
    return torch.nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(output_dim, output_dim),
        nn.LayerNorm(output_dim),
        nn.ReLU(),
    )


class LanguageModel(nn.Module):
    """ Linguistic Model with Transformers (BERT)."""
    def __init__(self, 
                 pretrained_backbone, 
                 frozen_embedding=False,
                 frozen_encoder=False,
                 ):
        super().__init__()
        self.frozen_encoder = frozen_encoder
        self.frozen_embedding = frozen_embedding if not self.frozen_encoder else True
        self.num_channels = 768

        self.language_backbone = pretrained_backbone
        
        # encoder_layers = []
        # for i in range(len(pretrained_backbone.encoder.layer)):
        #     encoder_layers.append(pretrained_backbone.encoder.layer[i])
        # self.encoder = nn.ModuleList(encoder_layers)

        # self.text_proj = nn.Linear(self.num_channels, embed_dim)
        self._freeze_params()

    def _freeze_params(self):
        if self.frozen_embedding:
            self.language_backbone.embeddings.eval()
            for param in self.language_backbone.embeddings.parameters():
                param.requires_grad = False
        
        if self.frozen_encoder:
            self.language_backbone.encoder.eval()
            for param in self.language_backbone.encoder.parameters():
                param.requires_grad = False

    def forward(self, text_src, text_mask):

        text_tokens = self.language_backbone(text_src, token_type_ids=None, attention_mask=text_mask)[0]
        text_mask = ~text_mask.to(torch.bool)

        return text_tokens, text_mask

def build_language_branch(args):
    
    if args.bert_model == 'bert-base-uncased':
        bert_model = BertModel
    elif args.bert_model == 'roberta-base':
        bert_model = RobertaModel
    else:
        raise ValueError('Only support bert-base-uncased or roberta-base')
    # assert args.bert_model == 'roberta-base'
    # bert_model = RobertaModel

    bert_model = bert_model.from_pretrained(args.pretrained_lm_path + '/' + args.bert_model)
        
    model = LanguageModel(bert_model, 
                          frozen_embedding=args.language_frozen_embedding,
                          frozen_encoder=args.langauge_frozen_encoder,
                          )
    
    return model
