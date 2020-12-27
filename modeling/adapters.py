# __author__ = "Vasudev Gupta"

import torch
from torch import nn
from transformers.activations import ACT2FN

from modeling.bert_layers import BertAttention


class FFNAdapter(nn.Module):

    def __init__(self, config):
        super().__init__()

        hidden_size = config["hidden_size"]
        intermediate_size = config.get("intermediate_size", None)
        layer_norm_eps = config["layer_norm_eps"]

        if intermediate_size is None:
            intermediate_size = hidden_size

        layers = []

        layers.append(nn.Linear(hidden_size, intermediate_size))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(intermediate_size, hidden_size))

        self.ffn = nn.Sequential(*layers)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, input_tensor):
        x = self.ffn(input_tensor)
        x = self.LayerNorm(x + input_tensor)        
        return x


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
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions)

        return out


class MixAdapterBL(object):

    def __init__(self):
        """
            Inherit BertLayer from this class to add support for adapters
        """
        self.add_cross_attn_adapter = False
        self.add_ffn_adapter = False

    def add_cross_attn_adapter_(self, adapter_config):
        self.add_cross_attn_adapter = True
        self.cross_attn_adapter = CrossAttnAdapter(adapter_config)
        return "ADDED"

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

    def add_ffn_adapter_(self, adapter_config):
        self.add_ffn_adapter = True
        self.ffn_adapter = FFNAdapter(adapter_config)
        return "adapter ADDED"

    def ffn_adapter_forward(self, x):
        return self.ffn_adapter(x)

    def adapter_requires_grad_(self, ffn_adapter, cross_attn_adapter=None):

        m1 = "ffn_adapter ADD first"
        m2 = "cross-attn_adapter ADD first"

        if self.add_ffn_adapter:
            m1 = "ffn_adapter NOT trainable"
            for param in self.ffn_adapter.parameters():
                param.requires_grad_(ffn_adapter)
            if ffn_adapter:
                m1 = "ffn_adapter trainable"

        if self.add_cross_attn_adapter:
            m2 = "cross-attn_adapter NOT trainable"
            for param in self.cross_attn_adapter.parameters():
                param.requires_grad_(cross_attn_adapter)
            if cross_attn_adapter:
                m2 = "cross-attn_adapter trainable"

        return m1, m2


class MixAdapterTMP(object):

    def __init__(self):
        """
            Inherit TransformerMaskPredict from this class to add support for adapters
        """

    def add_adapter_(self, 
                enc_ffn_adapter, 
                dec_ffn_adapter,
                cross_attn_adapter,
                enc_ffn_adapter_config,
                dec_ffn_adapter_config, 
                cross_attn_adapter_config):

        m1 = "cross-attn_adapter NOT added"
        m2 = "decoder ffn_adapter NOT added"
        m3 = "encoder ffn_adapter NOT added"

        if cross_attn_adapter:
            n = len(self.decoder.encoder.layer)
            for i in range(n):
                m1 = self.decoder.encoder.layer[i].add_cross_attn_adapter_(cross_attn_adapter_config)
            m1 = "Cross-attn " + m1

        if dec_ffn_adapter:
            n = len(self.decoder.encoder.layer)
            for i in range(n):
                m2 = self.decoder.encoder.layer[i].add_ffn_adapter_(dec_ffn_adapter_config)
            m2 = "decoder " + m2

        if enc_ffn_adapter:
            n = len(self.encoder.encoder.layer)
            for i in range(n):
                m3 = self.encoder.encoder.layer[i].add_ffn_adapter_(enc_ffn_adapter_config)
            m3 = "encoder " + m3

        print("==========Adapter ADDN status==========")
        print(m1, "\n", m2, "\n", m3)
        print("=============================================")


    def adapter_requires_grad_(self,
                    enc_ffn_adapter: bool,
                    dec_ffn_adapter: bool,
                    cross_attn_adapter: bool,
                ):

        m1 = "cross-attn_adapter NOT activated"
        m2 = "decoder ffn_adapter NOT activated"
        m3 = "encoder ffn_adapter NOT activated"

        n = len(self.encoder.encoder.layer)
        for i in range(n):
            m1, _ = self.encoder.encoder.layer[i].adapter_requires_grad_(enc_ffn_adapter)
            m1 = "encoder " + m1

        n = len(self.decoder.encoder.layer)
        for i in range(n):
            m2, m3 = self.decoder.encoder.layer[i].adapter_requires_grad_(dec_ffn_adapter, cross_attn_adapter)
            m2 = "decoder " + m2

        print("==========Adapter activation status==========")
        print(m1, "\n", m2, "\n", m3)
        print("=============================================")

    def layers_requires_grad_(self, length_embed:bool):
        for p in self.encoder.embeddings.length_embedding.parameters():
            p.requires_grad_(length_embed)

    def save_finetuned(self,
                    path:str,
                    enc_ffn_adapter:bool=True,
                    dec_ffn_adapter:bool=True,
                    cross_attn_adapter:bool=True,
                    length_embed:bool=True,
                    print_status:bool=True):

        state_dict = self.state_dict()
        saving_keys = []

        if enc_ffn_adapter:
            num = len(self.encoder.encoder.layer)
            for i in range(num):
                k = f"encoder.encoder.layer.{i}.ffn_adapter"
                saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        if dec_ffn_adapter:
            num = len(self.decoder.encoder.layer)
            for i in range(num):
                k = f"decoder.encoder.layer.{i}.ffn_adapter"
                saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        if cross_attn_adapter:
            num = len(self.decoder.encoder.layer)
            for i in range(num):
                k = f"decoder.encoder.layer.{i}.cross_attn_adapter"
                saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        if length_embed:
            k = "encoder.embeddings.length_embedding"
            saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        saving = {}
        for k in saving_keys:
            saving.update({k: state_dict[k]})

        if path:
            if print_status: print(f"saving: {saving.keys()}")
            torch.save(saving, path)

    def load_finetuned(self, path:str=None, map_location="cuda:0"):
        # simple loading; saving is very important
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict, strict=False)
        print(f'Whatever weights were in {path} are loaded')
