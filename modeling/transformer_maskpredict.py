# __author__ = "Vasudev Gupta"

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.adapters import MixAdapterTMP
from modeling.modeling_bert import BertModel

class TransformerMaskPredict(MixAdapterTMP):

    def __init__(self, config):
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

    def forward(self, input_ids, encoder_attention_mask, decoder_input_ids=None, decoder_attention_mask=None, labels=None, return_dict=True):
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
        length_logits = x.pop("length_logits")
        x = torch.cat([length_logits, x.pop("last_hidden_state")], dim=1)
        # decoder
        x = self.decoder(input_ids=decoder_input_ids,
                        attention_mask=decoder_attention_mask,
                        encoder_hidden_states=x,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=True)
        x = x["last_hidden_state"]
        x = F.linear(x, self.decoder.embeddings.word_embeddings.weight, bias=None)

        if labels is not None:
            loss, length_loss, translation_loss = self.compute_loss(x, labels, length_logits)

        if not return_dict:
            return x, length_logits, loss, length_loss, translation_loss

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

    def compute_loss(self, final_logits, labels, length_logits, eps=0.1, reduction="sum"):
        # loss_fn = LossFunc(eps=eps, reduction=reduction)
        # return loss_fn(final_logits, labels, length_logits)
        final_logits = final_logits.view(-1, final_logits.size(-1))
        labels = labels.view(-1)
        return nn.CrossEntropyLoss()(final_logits, labels), None, None

class LossFunc(nn.Module):

    def __init__(self, eps=0.1, reduction="sum"):
        super().__init__()
        # TODO
        # think of padding
        self.eps = eps
        self.reduction = reduction

    def compute_length_loss(self, length_logits, length_labels):
        length_logits = F.log_softmax(length_logits, dim=-1)
        length_loss = F.nll_loss(length_logits, length_labels, reduction=self.reduction)
        return length_loss

    def compute_translation_loss(self, final_logits, labels):

        final_logits = F.log_softmax(final_logits, dim=-1)
        nll_loss = F.nll_loss(final_logits, labels, reduction=self.reduction)
        smooth_loss = final_logits.mean(-1)
        
        if self.reduction == "sum":
            smooth_loss = smooth_loss.sum()

        return (1.-self.eps)*nll_loss + self.eps*smooth_loss

    def forward(self, final_logits, labels, length_logits):

        # TODO
        length_labels = labels.size(-1)

        length_loss = self.compute_length_loss(length_logits, length_labels.size(-1))
        translation_loss = self.compute_translation_loss(final_logits, labels)

        loss = 0.1*length_loss + translation_loss

        return {
            "loss": loss,
            "length_loss": length_loss,
            "translation_loss": translation_loss
            }
