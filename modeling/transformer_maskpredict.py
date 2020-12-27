# __author__ = "Vasudev Gupta"

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.adapters import MixAdapterTMP
from modeling.modeling_bert import BertModel
from modeling.decoding import MaskPredict
from modeling.utils import Dict

# setting up `from_pretrained` method
from transformers.file_utils import hf_bucket_url, cached_path
import json

class TransformerMaskPredict(nn.Module, MixAdapterTMP):

    def __init__(self, config):
        super().__init__()
        MixAdapterTMP.__init__(self)

        self.config = config

        self.encoder = BertModel.from_pretrained(self.config["encoder_id"], num_lengths=self.config.num_lengths, add_length_embedding=True)
        self.decoder = BertModel.from_pretrained(self.config["decoder_id"])

        # self.register_buffer("final_layer_bias", torch.zeros(1, self.decoder.embeddings.word_embeddings.num_embeddings))

        for param in self.parameters():
            param.requires_grad_(False)

        self.add_adapter_(self.config.enc_ffn_adapter,
                        self.config.dec_ffn_adapter,
                        self.config.cross_attn_adapter,
                        self.config.enc_ffn_adapter_config,
                        self.config.dec_ffn_adapter_config,
                        self.config.cross_attn_adapter_config)

        # now encoder will have ffn-adapter
        # decoder will have ffn-adapter & cross-attn-adapter

        self.adapter_requires_grad_(self.config.enc_ffn_adapter,
                                self.config.dec_ffn_adapter,
                                self.config.cross_attn_adapter)
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

    def compute_loss(self, final_logits, labels, length_logits, eps=0.1, reduction="sum"):
        # loss_fn = LossFunc(eps=eps, reduction=reduction)
        # return loss_fn(final_logits, labels, length_logits)
        final_logits = final_logits.view(-1, final_logits.size(-1))
        labels = labels.view(-1)
        return nn.CrossEntropyLoss()(final_logits, labels), None, None

    def save_pretrained(self, save_directory:str):
        """
            We are saving only the finetuned weights ; bert-weights in encoder and decoder are not getting saved 
            and can be loaded directly from huggingface hub
        """

        if save_directory not in os.listdir(): 
            os.makedirs(save_directory)

        # saving config
        path = os.path.join(save_directory, "config.json")
        with open(path, "w") as f:
            json.dump(self.config, f)

        # saving only the adapter weights and length embedding
        path = os.path.join(save_directory, "pytorch_model.bin")
        self.save_finetuned(path, print_status=False)

        return True

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path:str):
        """
            Setting up this method will enable to load directly from huggingface hub just like other HF models are loaded
        """
        model_id = pretrained_model_name_or_path

        config_url = hf_bucket_url(model_id, filename="config.json")
        config_file = cached_path(config_url)
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        config = Dict.from_nested_dict(config)

        # downloading & load only the adapter weights from huggingface hub
        # and corresponding bert weights will be loaded when class is getting initiated
        model_url = hf_bucket_url(model_id, filename="pytorch_model.bin")
        model_file = cached_path(model_url)
        state_dict = torch.load(model_file, map_location="cpu")

        # randomly initializing model from given config with bert weights restored
        model = cls(config)
        # now restoring adapter weights
        model.load_state_dict(state_dict, strict=False)

        return model

    def generate(self, **kwargs):
        """
            This method is not available and MaskPredict class should be used instead
        """
        raise NotImplementedError

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
