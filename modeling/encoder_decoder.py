import torch
import torch.nn as nn

from adapters import MixAdapterTransformer
from modeling_bert import BertModel


class TransformerMaskPredict(MixAdapterTransformer):

    def __init(self, config):
        super().__init__()

        self.encoder = BertModel.from_pretrained(config["encoder_id"])
        self.decoder = BertModel.from_pretrained(config["decoder_id"])

        self.register_buffer("final_layer_bias", torch.zeros(1, self.dec_bert.embeddings.num_embeddings))

        for param in self.parameters():
            param.requires_grad_(False)

        self.add_adapter_(config["enc_ffn_adapter"],
                        config["dec_ffn_adapter"],
                        config["cross_attn_adapter"],
                        config["enc_ffn_adapter_config"],
                        config["dec_ffn_adapter_config"],
                        config["cross_attn_adapter_config"])

        self.adapter_requires_grad_(config["enc_ffn_adapter"],
                                config["dec_ffn_adapter"],
                                config["cross_attn_adapter"])

    def forward(self, input_ids, attention_mask, decoder_input_ids):

        # encoder
        x = self.encoder(input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True)
        x = x["logits"]

        # decoder
        x = self.decoder(input_ids=decoder_input_ids,
                    encoder_hidden_states=x,
                    encoder_attention_mask=attention_mask,
                    return_dict=True)
        x = x["logits"]

        # -> (bz, seqlen, 768)        
        x = F.linear(x, self.dec_bert.embeddings, bias=self.final_layer_bias)

        return x

    def generate(self):
        """This is based on mask-predict as suggested in paper"""
        raise NotImplementedError


