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
                 prompt_tuning=False
                 ):
        super().__init__()
        self.prompt_tuning = prompt_tuning
        self.num_channels = 768

        self.embeddings = pretrained_backbone.embeddings
        self.encoder = pretrained_backbone.encoder
        # self.pooler = pretrained_backbone.pooler
        
        if self.prompt_tuning:
            self.prompt_tokens = nn.Parameter(torch.zeros(1, 20, self.num_channels))
            nn.init.normal_(self.prompt_tokens, std=1e-6)
        
        self._freeze_params()

    def _freeze_params(self):
        self.embeddings.eval()
        for param in self.embeddings.parameters():
            param.requires_grad = False
        
        if self.prompt_tuning:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, tensor_list: NestedTensor):
        ling_src, ling_mask = tensor_list.decompose()
        ling_src = self.embeddings(ling_src)

        if self.prompt_tuning:
            batch_size = ling_src.shape[0]
            prompt_src  = self.prompt_tokens.expand(batch_size, -1, -1)
            prompt_mask = torch.ones(batch_size, 20, dtype=ling_mask.dtype, device=ling_src.device)

            ling_src  = torch.cat([ling_src, prompt_src], dim=1)
            ling_mask = torch.cat([ling_mask, prompt_mask], dim=1)
        
        extended_mask = ling_mask[:, None, None, :]
        outputs = self.encoder(ling_src, attention_mask=extended_mask)

        xs = outputs.last_hidden_state

        # mask = tensor_list.mask.to(torch.bool)
        mask = ling_mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)

        return out

def build_linguistic_branch(args):
    
    if args.bert_model == 'bert-base-uncased':
        bert_model = BertModel
    elif args.bert_model == 'roberta-base':
        bert_model = RobertaModel
    else:
        raise ValueError('Only support bert-base-uncased or roberta-base')

    bert_model = bert_model.from_pretrained(args.pretrained_lm_path + '/' + args.bert_model)
    # is_freeze = args.lr_bert == 0
    model = LinguisticModel(bert_model, args.prompt_tuning)
    return model
