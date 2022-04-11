import torch
import torch.nn.functional as F

from torch import nn
from utils.misc import NestedTensor
from transformers import AutoModel
from transformers import RobertaModel, BertModel


class LinguisticModel(nn.Module):
    """ Linguistic Model with Transformers (BERT)."""
    def __init__(self, pretrained_backbone, is_freeze: bool):
        super().__init__()

        self.bert = pretrained_backbone
        self.num_channels = 768
        
        if is_freeze:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):

        outputs = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask)
        xs = outputs.last_hidden_state

        mask = tensor_list.mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)

        return out

def build_linguistic_branch(args):
    
    if args.bert_model == 'bert-base-uncased':
        bert_model = BertModel
    elif args.bert_model == 'roberta-base':
        bert_model = RobertaModel

    bert_model = bert_model.from_pretrained(args.pretrained_lm_path + '/' + args.bert_model)
    is_freeze = args.lr_bert == 0
    model = LinguisticModel(bert_model, is_freeze)
    return model
