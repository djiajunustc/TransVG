import torch
import torch.nn.functional as F

from torch import nn
from utils.misc import NestedTensor
from transformers import AutoModel
from transformers import RobertaModel, BertModel


class LinguisticModel(nn.Module):
    """ Linguistic Model with Transformers (BERT)."""
    def __init__(self, 
                 pretrained_backbone, 
                 frozen_embedding=False,
                 frozen_encoder=False,
                 embed_dim=384
                 ):
        super().__init__()
        self.frozen_encoder = frozen_encoder
        self.frozen_embedding = frozen_embedding if not self.frozen_encoder else True
        self.num_channels = 768

        self.embeddings = pretrained_backbone.embeddings
        
        encoder_layers = []
        for i in range(len(pretrained_backbone.encoder.layer)):
            encoder_layers.append(pretrained_backbone.encoder.layer[i])
        self.encoder = nn.ModuleList(encoder_layers)

        self.text_proj = nn.Linear(self.num_channels, embed_dim)
        self._freeze_params()

    def _freeze_params(self):
        if self.frozen_embedding:
            self.embeddings.eval()
            for param in self.embeddings.parameters():
                param.requires_grad = False
        
        if self.frozen_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, tensor_list: NestedTensor):
        ling_src, ling_mask = tensor_list.decompose()
        ling_src = self.embeddings(ling_src)
        
        extended_mask = ling_mask[:, None, None, :]
        outputs = ling_src
        for layer in self.encoder:
            outputs = layer(outputs, attention_mask=extended_mask)[0]
        # outputs = self.encoder(ling_src, attention_mask=extended_mask)

        text_src = self.text_proj(outputs)
        text_mask = ~ling_mask.to(torch.bool)

        return text_src, text_mask

def build_linguistic_branch(args):
    
    if args.bert_model == 'bert-base-uncased':
        bert_model = BertModel
    elif args.bert_model == 'roberta-base':
        bert_model = RobertaModel
    else:
        raise ValueError('Only support bert-base-uncased or roberta-base')
    # assert args.bert_model == 'roberta-base'
    # bert_model = RobertaModel

    bert_model = bert_model.from_pretrained(args.pretrained_lm_path + '/' + args.bert_model)

    if args.vit_model == 'tiny':
        embed_dim = 192
    elif args.vit_model == 'small':
        embed_dim = 384
    elif args.vit_model == 'base':
        embed_dim = 768
        
    model = LinguisticModel(bert_model, 
                            frozen_embedding=args.language_frozen_embedding,
                            frozen_encoder=args.langauge_frozen_encoder,
                            embed_dim=embed_dim
                            )
    
    return model
