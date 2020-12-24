# __author__ = "Vasudev Gupta"

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.adapters import MixAdapterTMP
from modeling.modeling_bert import BertModel

class TransformerMaskPredict(MixAdapterTMP):

    def __init(self, config):
        super().__init__()

        self.encoder = BertModel.from_pretrained(config["encoder_id"], num_lengths=config.num_lengths, add_length_embedding=True)
        self.decoder = BertModel.from_pretrained(config["decoder_id"])

        # self.register_buffer("final_layer_bias", torch.zeros(1, self.decoder.embeddings.word_embeddings.num_embeddings))

        for param in self.parameters():
            param.requires_grad_(False)

        self.add_adapter_(config.enc_ffn_adapter,
                        config.dec_ffn_adapter,
                        config.cross_attn_adapter,
                        config.enc_ffn_adapter_config,
                        config.dec_ffn_adapter_config,
                        config.cross_attn_adapter_config)

        # now encoder will have ffn-adapter
        # decoder will have ffn-adapter & cross-attn-adapter

        self.adapter_requires_grad_(config.enc_ffn_adapter,
                                config.dec_ffn_adapter,
                                config.cross_attn_adapter)
        self.layers_requires_grad_(True)

    def forward(self, input_ids, encoder_attention_mask, decoder_input_ids=None, decoder_attention_mask=None, labels=None, return_dict=True, **kwargs):
        """
        Input View:
            input_ids :: torch.tensor : [LENGTH], [CLS], ........., [SEP], [PAD] ...... [PAD]
            decoder_input_ids :: torch.tensor : [CLS], ........, [PAD] ...... [PAD]
            labels: torch.tensor : ............, [SEP], [PAD] ...... [PAD]
        """

        loss = None
        length_loss = None
        translation_loss = None

        # encoder
        x = self.encoder(input_ids=input_ids,
                    attention_mask=encoder_attention_mask,
                    return_dict=True)
        length_logits = x["length_logits"]
        x = x["last_hidden_state"]

        # decoder
        x = self.decoder(input_ids=decoder_input_ids,
                        attention_mask=decoder_attention_mask,
                        encoder_hidden_states=x,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=True)
        x = x["last_hidden_state"]
        x = F.linear(x, self.decoder.embeddings.word_embeddings.weight, bias=None)

        if labels is not None:
            loss, length_loss, translation_loss = self.loss_fn(x, labels, length_logits)

        if not return_dict:
            return x, length_logits, loss

        return {
            "logits": x,
            "length_logits": length_logits,
            "loss": loss,
            "length_loss": length_loss,
            "translation_loss": translation_loss
            }

    @staticmethod
    def _pad(ls:list, max_len:int, pad:int):
        while len(ls) < max_len:
            ls.append(pad)
        return ls
    
    def save(self, path:str):
        saving = self.state_dict()
        torch.save(saving, path)

    def load(self, path:str, map_location=torch.device("cuda")):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

    def loss_fn(self, final_logits, labels, length_logits):
        
        loss_fn = nn.CrossEntropyLoss()
        translation_loss = loss_fn(final_logits, labels)
        
        length_loss = ## put something

        return loss, length_loss, translation_loss
