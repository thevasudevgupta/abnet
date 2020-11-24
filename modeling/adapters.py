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


class Activation_Function_Class(nn.Module):

    def __init__(self, hidden_act):
        super().__init__()

        self.f = ACT2FN[hidden_act.lower()]

    def forward(self, x):
        return self.f(x)


class AdapterLayerMix(object):
    # Inherit encoder-layer / decoder-layer from this class

    def __init__(self):
        # call this method at end of `__init__`
        self.add_adapter_ffn = False
        self.add_adapter_cross_attn = False

    def adapter_ffn_forward(self, adapter_in, adapter_residual_conn):
        # call this method at end of layer `forward`

        adapter_out = self.adapter_ffn(adapter_in, adapter_residual_conn)
        adapter_out = adapter_out[0]

        return adapter_out

    def adapter_cross_attn_forward(self, adapter_in, adapter_residual_conn):

        adapter_out = self.adapter_cross_attn(adapter_in, adapter_residual_conn)
        adapter_out = adapter_out[0]

        return adapter_out

    def add_adapter_ffn_(self, config_ffn):

        self.add_adapter_ffn = True
        self.adapter_ffn = Adapter(config_ffn)

        return "ffn adapter added"

    def add_adapter_cross_attn_(self, config_cross_attn):
        # remember to call it only with decoder

        self.add_adapter_cross_attn = True
        self.adapter_cross_attn = Adapter(config_cross_attn)

        return "cross-attn adapter added"

    def adapter_requires_grad_(self, ffn_grad, self_attn_grad, cross_attn_grad=None):

        m1 = "ffn NOT activated"
        m2 = "self-attn NOT activated"
        m3 = "cross-attn NOT activated"

        if self.add_adapter_ffn:
            m1 = "ffn adapter not activated"
            for param in self.adapter_ffn.parameters():
                param.requires_grad_(ffn_grad)
            if ffn_grad:
                m1 = "ffn adapter activated"
        else:
            m1 = "ffn adapter ADD first"

        if self.add_adapter_cross_attn:
            m3 = "cross-attn adapter not activated"
            for param in self.adapter_cross_attn.parameters():
                param.requires_grad_(cross_attn_grad)
            if cross_attn_grad:
                m3 = "cross-attn adapter activated"
        else:
            m3 = "cross-attn adapter ADD first"

        return m1, m2, m3


class MixAdapterBFCG(object):

    def __init__(self):
        """Inherit BFCG from this this class"""

    def add_adapter_(self,
                    enc_ffn_adapter: bool, 
                    dec_ffn_adapter: bool,
                    enc_self_attn_adapter: bool,
                    dec_self_attn_adapter: bool,
                    cross_attn_adapter: bool,
                    enc_tok_embed_adapter: bool,
                    dec_tok_embed_adapter: bool,
                    enc_ffn_adapter_config: AdapterConfig,
                    dec_ffn_adapter_config: AdapterConfig,
                    enc_self_attn_adapter_config: AdapterConfig,
                    dec_self_attn_adapter_config: AdapterConfig,
                    cross_attn_adapter_config: AdapterConfig,
                    enc_tok_embed_adapter_config: AdapterConfig,
                    dec_tok_embed_adapter_config: AdapterConfig):

        m1 = "encoder ffn adapter NOT added"
        m2 = "encoder self-attn adapter NOT added"
        m3 = "decoder ffn adapter NOT added"
        m4 = "decoder self-attn adapter NOT added"
        m5 = "cross-attn adapter NOT added"
        m6 = "encoder tok-embed adapter NOT added"
        m7 = "decoder tok-embed adapter NOT added"

        if enc_ffn_adapter:
            num = len(self.model.encoder.layers)
            for i in range(num):
                m1 = self.model.encoder.layers[i].add_adapter_ffn_(enc_ffn_adapter_config)
                m1 = "encoder " + m1

        if enc_self_attn_adapter:
            num = len(self.model.encoder.layers)
            for i in range(num):
                m2 = self.model.encoder.layers[i].add_adapter_self_attn_(enc_self_attn_adapter_config)
                m2 = "encoder " + m2

        if dec_ffn_adapter:
            num = len(self.model.decoder.layers)
            for i in range(num):
                m3 = self.model.decoder.layers[i].add_adapter_ffn_(dec_ffn_adapter_config)
                m3 = "decoder " + m3

        if dec_self_attn_adapter:
            num = len(self.model.decoder.layers)
            for i in range(num):
                m4 = self.model.decoder.layers[i].add_adapter_self_attn_(dec_self_attn_adapter_config)
                m4 = "decoder " + m4

        if cross_attn_adapter:
            num = len(self.model.decoder.layers)
            for i in range(num):
                m5 = self.model.decoder.layers[i].add_adapter_cross_attn_(cross_attn_adapter_config)

        if enc_tok_embed_adapter:
            m6 = self.model.encoder.add_adapter_tok_embed_(enc_tok_embed_adapter_config)
            m6 = "encoder " + m6

        if dec_tok_embed_adapter:
            m7 = self.model.decoder.add_adapter_tok_embed_(dec_tok_embed_adapter_config)
            m7 = "decoder " + m7

        print("==========Adapter status============================")
        print(m1, "\n", m2, "\n", m3, "\n", m4, "\n", m5, "\n", m6, "\n", m7)
        print("====================================================")

    def adapter_requires_grad_(self,
                    enc_ffn_adapter: bool,
                    dec_ffn_adapter: bool,
                    cross_attn_adapter: bool,
                    enc_self_attn_adapter: bool,
                    dec_self_attn_adapter: bool,
                    enc_tok_embed_adapter: bool,
                    dec_tok_embed_adapter: bool):

        num = len(self.model.encoder.layers)
        for i in range(num):
            m1, m2, _ = self.model.encoder.layers[i].adapter_requires_grad_(enc_ffn_adapter, enc_self_attn_adapter)
            m1, m2 = "encoder " + m1, "encoder " + m2

        num = len(self.model.decoder.layers)
        for i in range(num):
            m3, m4, m5 = self.model.decoder.layers[i].adapter_requires_grad_(dec_ffn_adapter, dec_self_attn_adapter, cross_attn_adapter)
            m3, m4, m5 = "decoder " + m3, "decoder " + m4, m5

        m6 = self.model.encoder.adapter_requires_grad_(enc_tok_embed_adapter)
        m7 = self.model.decoder.adapter_requires_grad_(dec_tok_embed_adapter)

        print("==========Adapter activation status==========")
        print(m1, "\n", m2, "\n", m3, "\n", m4, "\n", m5, "\n", m6, "\n", m7)
        print("=============================================")

    def save_adapter(self,
                    path: str,
                    enc_ffn_adapter: bool,
                    dec_ffn_adapter: bool,
                    cross_attn_adapter: bool,
                    enc_self_attn_adapter: bool,
                    dec_self_attn_adapter: bool,
                    enc_tok_embed_adapter: bool,
                    dec_tok_embed_adapter: bool):

        state_dict = self.state_dict()
        saving_keys = []

        if enc_ffn_adapter:
            num = len(self.model.encoder.layers)
            for i in range(num):
                k = f"model.encoder.layers.{i}.adapter_ffn"
                saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        if dec_ffn_adapter:
            num = len(self.model.decoder.layers)
            for i in range(num):
                k = f"model.decoder.layers.{i}.adapter_ffn"
                saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])
        
        if cross_attn_adapter:
            num = len(self.model.decoder.layers)
            for i in range(num):
                k = f"model.decoder.layers.{i}.adapter_cross_attn"
                saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        if enc_self_attn_adapter:
            num = len(self.model.encoder.layers)
            for i in range(num):
                k = f"model.encoder.layers.{i}.adapter_self_attn"
                saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        if dec_self_attn_adapter:
            num = len(self.model.decoder.layers)
            for i in range(num):
                k = f"model.decoder.layers.{i}.adapter_self_attn"
                saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        if enc_tok_embed_adapter:
            k = f"model.encoder.adapter_tok_embed"
            saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        if dec_tok_embed_adapter:
            k = f"model.decoder.adapter_tok_embed"
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
