import torch
import torch.nn as nn

from attention import Attention

from adapters import (
    FfnAdapter, 
    CrossAttnAdapter, 
    AdapterConfig
)

from transformers import BertModel

bert = BertModel.from_pretrained(config.enc_bert_id)

class Transformer(nn.Module):

    def __init(self, bert, config: dict, dec_bert=None):
        super().__init__()

        if dec_bert is None:
            dec_bert = bert

    def forward(self,)