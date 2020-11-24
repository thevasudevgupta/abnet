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

        self.enc_bert = bert
        self.dec_bert = bert if dec_bert is None else dec_bert

        self.enc_ffn_adapter = FfnAdapter(config["encoder"]["ffn_adapter_config"])
        self.cross_attn_adapter CrossAttnAdapter(config["decoder"]["cross_attn_adapter_config"])
        self.dec_ffn_adapter = FffnAdapter(config["decoder"]["ffn_adapter_config"])

    def forward(self, src_tokens, tgt_tokens, attn_mask=None, head_mask=None, output_attentions=False):

        # encoder
        x = self.enc_bert(src_tokens, return_dict=True)
        x = x["logits"]
        enc_out = self.enc_ffn_adapter(x)

        # decoder
        x = self.dec_bert(tgt_tokens, return_dict=True)
        x = x["logits"]
        x = self.cross_attn_adapter(x
                                attention_mask=attn_mask,
                                head_mask=head_mask,
                                encoder_hidden_states=enc_out,
                                output_attentions=output_attentions)
        x = self.dec_ffn_adapter(x)

        
