# __author__ = "Vasudev Gupta"

import torch
from torch import nn

from transformers.activations import ACT2FN
from dataclasses import dataclass

from layers import BertAttention, BertIntermediate, BertOutput

@dataclass
class AdapterConfig:
    hidden_size: int
    intermediate_size: int
    hidden_act: str
    layer_norm_eps: float
    hidden_dropout_prob: float


class FfnAdapter(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, attn_out)
        inter_out = self.intermediate(attn_out)
        x = self.output(inter_out, attn_out)


class CrossAttnAdapter(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.attn = BertAttention(config)

    def forward(self, 
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False
    ):
        out = self.attn(hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False)

        return out


class MixAdapterLayer(object):

    def __init__(self):
        self.add_cross_attn_adapter = False

    def add_cross_attn_adapter_(self, add, adapter_config=None)
        self.add_cross_attn_adapter = add
        self.cross_attn_adapter = CrossAttnAdapter(adapter_config)

    def cross_attn_adapter_forward(self,  
                            hidden_states,
                            attention_mask=None,
                            head_mask=None,
                            encoder_hidden_states=None,
                            encoder_attention_mask=None,
                            output_attentions=False):

        hidden_states = self.cross_attn_adapter(hidden_states,
                                        attention_mask=attention_mask,
                                        head_mask=head_mask,
                                        encoder_hidden_states=encoder_hidden_states,
                                        encoder_attention_mask=encoder_attention_mask,
                                        output_attentions=output_attentions)

        return hidden_states[0]


class MixAdapterBert(object):

    def __init__(self):
        """Inherit BertModel from this class"""

    def add_adapter_(cross_attn_adap, cross_attn_adap_config):

        if cross_attn_adap:
            n = len(self.encoder.layer)
            for i in range(n):
                self.encoder.layer[i].add_cross_attn_adapter_(cross_attn_adap_config)

    def adapter_requires_grad(cross_attn_adap=None):
        n = len(self.encoder.layer)
        for i in range(n):
            self.encoder.layer[i].cross_attn_adapter.adapter_requires_grad_(cross_attn_adap)


class MixAdapterEncDec(object):

    def __init__(self):
        """Inherit EncDec from this this class"""

    def adapter_requires_grad_(self,
                    enc_ffn_adapter: bool,
                    dec_ffn_adapter: bool,
                    cross_attn_adapter: bool,
                ):

        num = len(self.model.encoder.layers)
        for i in range(num):
            m1, m2, _ = self.model.encoder.layer[i].adapter_requires_grad_(enc_ffn_adapter)
            m1, m2 = "encoder " + m1, "encoder " + m2

        num = len(self.model.decoder.layers)
        for i in range(num):
            m3, m4, m5 = self.model.decoder.layer[i].adapter_requires_grad_(dec_ffn_adapter, cross_attn_adapter)
            m3, m4, m5 = "decoder " + m3, "decoder " + m4, m5

        print("==========Adapter activation status==========")
        print(m1, "\n", m2, "\n", m3, "\n", m4, "\n", m5, "\n", m6, "\n", m7)
        print("=============================================")

    def save_adapter(self,
                    path: str,
                    enc_ffn_adapter: bool,
                    dec_ffn_adapter: bool,
                    cross_attn_adapter: bool):

        state_dict = self.state_dict()
        saving_keys = []

        if enc_ffn_adapter:
            num = len(self.model.encoder.layers)
            for i in range(num):
                k = f"model.encoder.layer.{i}.adapter_ffn"
                saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        if dec_ffn_adapter:
            num = len(self.model.decoder.layers)
            for i in range(num):
                k = f"model.decoder.layer.{i}.adapter_ffn"
                saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])
        
        if cross_attn_adapter:
            num = len(self.model.decoder.layers)
            for i in range(num):
                k = f"model.decoder.layers.{i}.cross_attn_adapter"
                saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        saving = {}
        for k in saving_keys:
            saving.update({k: state_dict[k]})

        if path:
            print(f"saving: {saving.keys()}")
            torch.save(saving, path)

    def load_adapter(self, path: str = None, map_location="cuda:0"):
        # simple loading; saving is very important
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict, strict=False)
        print(f'Whatever weights were in {path} are loaded')
