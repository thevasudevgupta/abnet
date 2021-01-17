# __author__ = "Vasudev Gupta"

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from modeling.multihead_attention import MultiheadAttention


class AdapterBasedDecoder(nn.Module):

    def __init__(self, bert, adapter_config):
        super(AdapterBasedDecoder, self).__init__()

        self.bert = bert

        n = len(self.bert.encoder.layer)
        for i in range(n):
            self.bert.encoder.layer[i].add_decoder_adapter_(adapter_config)

    def forward(self, **kwargs):
        return self.bert(**kwargs)


class MixAdapterBL(object):

    def __init__(self):
        """
            Inherit BertLayer from this class to add support for adapters
        """
        self.add_encoder_adapter = False
        self.add_decoder_adapter = False

    def add_decoder_adapter_(self, adapter_config):
        self.add_decoder_adapter = True

        layer_norm_eps = adapter_config.layer_norm_eps
        dropout_prob = adapter_config.dropout_prob
        num_attention_heads = adapter_config.num_attention_heads
        hidden_size = adapter_config.hidden_size
        intermediate_size = adapter_config.intermediate_size

        self.encoder_attn =  MultiheadAttention(hidden_size,
                                            num_attention_heads,
                                            kdim=hidden_size,
                                            vdim=hidden_size,
                                            dropout=dropout_prob, 
                                            encoder_decoder_attention=True)
        self.encoder_attn_layer_norm = nn.LayerNorm(hidden_size)

        self.encoder_attn_fc1 = nn.Linear(hidden_size, intermediate_size)
        self.encoder_attn_fc2 = nn.Linear(intermediate_size, hidden_size)
        self.encoder_attn_final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        return "ADDED"

    # TODO : fix this
    def decoder_adapter_forward(self,  
                                hidden_states,
                                attention_mask=None,
                                head_mask=None,
                                encoder_hidden_states=None,
                                encoder_attention_mask=None,
                                output_attentions=False):

        hidden_states = self.encoder_attn(hidden_states,
                                        attention_mask=attention_mask,
                                        head_mask=head_mask,
                                        encoder_hidden_states=encoder_hidden_states,
                                        encoder_attention_mask=encoder_attention_mask,
                                        output_attentions=output_attentions)


        return hidden_states[0]

    def add_encoder_adapter_(self, adapter_config):
        self.add_encoder_adapter = True

        hidden_size = adapter_config["hidden_size"]
        intermediate_size = adapter_config["intermediate_size"]
        layer_norm_eps = adapter_config["layer_norm_eps"]

        self.adapter_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.adapter_w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.adapter_w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

        return "adapter ADDED"

    def encoder_adapter_forward(self, bert_out):
        x = self.adapter_ln(bert_out)
        x = self.adapter_w2(F.relu(self.adapter_w1(x)))
        return x + bert_out

    def encoder_adapter_requires_grad_(self, encoder_adapter):

        if self.add_encoder_adapter:
            for param in self.adapter_ln.parameters():
                param.requires_grad_(encoder_adapter)
            for param in self.adapter_w1.parameters():
                param.requires_grad_(encoder_adapter)
            for param in self.adapter_w2.parameters():
                param.requires_grad_(encoder_adapter)
        else:
            raise ValueError("ADD encoder adapters first")

    def decoder_adapter_requires_grad_(self, decoder_adapter):

        if self.add_decoder_adapter:
            for param in self.encoder_attn.parameters():
                param.requires_grad_(decoder_adapter)
            for param in self.encoder_attn_layer_norm.parameters():
                param.requires_grad_(decoder_adapter)
            for param in self.encoder_attn_fc1.parameters():
                param.requires_grad_(decoder_adapter)
            for param in self.encoder_attn_fc2.parameters():
                param.requires_grad_(decoder_adapter)
            for param in self.encoder_attn_final_layer_norm.parameters():
                param.requires_grad_(decoder_adapter)
        else:
            raise ValueError("ADD decoder adapters first")
