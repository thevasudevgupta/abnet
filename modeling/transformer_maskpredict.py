# __author__ = "Vasudev Gupta"

import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.file_utils import hf_bucket_url, cached_path

from modeling.adapters import AdapterBasedDecoder
from modeling.modeling_bert import BertModel
from modeling.decoding import MaskPredict
from modeling.utils import Dict


class TransformerMaskPredict(nn.Module, MaskPredict):

    def __init__(self, config):
        super().__init__()
        MaskPredict.__init__(self)

        self.config = config

        # encoder with adapter
        self.encoder = BertModel.from_pretrained(config.encoder_id, num_lengths=config.num_lengths, add_length_embedding=True)
        self._add_encoder_adapter_(config.encoder_adapter_config)

        # decoder with adapters
        bert = BertModel.from_pretrained(self.config["decoder_id"])
        self.decoder = AdapterBasedDecoder(bert, config.decoder_adapter_config)

        # lm head
        hidden_size = config.decoder_adapter_config.hidden_size
        self.final_dense = nn.Linear(hidden_size, hidden_size)
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=config.decoder_adapter_config.layer_norm_eps)
        self.final_bias = nn.Parameter(torch.zeros(self.decoder.bert.embeddings.word_embeddings.weight.size(0)))

        # handling what to train
        self._requires_grad_(config.bert_requires_grad,
                            config.encoder_adapter_requires_grad,
                            config.decoder_adapter_requires_grad,
                            config.length_embed_requires_grad,
                            config.lm_head_requires_grad)

    def _requires_grad_(self,
                    bert_requires_grad: bool,
                    encoder_adapter_requires_grad: bool,
                    decoder_adapter_requires_grad: bool,
                    length_embed_requires_grad: bool,
                    lm_head_requires_grad:bool):

        for p in self.encoder.parameters():
            p.requires_grad_(bert_requires_grad)
        for p in self.decoder.parameters():
            p.requires_grad_(bert_requires_grad)

        n = len(self.encoder.encoder.layer)
        for i in range(n):
            self.encoder.encoder.layer[i].encoder_adapter_requires_grad_(encoder_adapter_requires_grad)

        n = len(self.decoder.bert.encoder.layer)
        for i in range(n):
            self.decoder.bert.encoder.layer[i].decoder_adapter_requires_grad_(decoder_adapter_requires_grad)

        for p in self.encoder.embeddings.length_embedding.parameters():
            p.requires_grad_(length_embed_requires_grad)

        for p in self.dense.parameters():
            p.requires._grad_(lm_head_requires_grad)
        for p in self.final_layer_norm.parameters():
            p.requires_grad_(lm_head_requires_grad)
        for p in self.final_bias.parameters():
            p.requires_grad_(lm_head_requires_grad)

    def _add_encoder_adapter_(self, adapter_config):

        n = len(self.encoder.encoder.layer)
        for i in range(n):
            self.encoder.encoder.layer[i].add_encoder_adapter_(adapter_config)

    def forward(self, input_ids, encoder_attention_mask, decoder_input_ids=None, decoder_attention_mask=None, labels=None, return_dict=True):
        """
        Input View:
            input_ids :: torch.tensor : [LENGTH], [CLS], ........., [SEP], [PAD] ...... [PAD]
            decoder_input_ids :: torch.tensor : [CLS], ........, [PAD] ...... [PAD]
            labels: torch.tensor : ............, [SEP], [PAD] ...... [PAD]
        """

        # encoder
        x = self.encoder(input_ids=input_ids,
                    attention_mask=encoder_attention_mask,
                    return_dict=True)
        length_logits = x.pop("length_logits")
        x = torch.cat([length_logits, x.pop("last_hidden_state")], dim=1)

        # adding head over length logits
        length_logits = F.linear(length_logits, self.encoder.embeddings.length_embedding.weight, bias=None)

        # decoder
        x = self.decoder(input_ids=decoder_input_ids,
                        attention_mask=decoder_attention_mask,
                        encoder_hidden_states=x,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=True)

        # language modeling head
        x = self.final_layer_norm(F.relu(self.final_dense(x["last_hidden_state"])))
        x = F.linear(x, self.decoder.bert.embeddings.word_embeddings.weight, bias=self.final_bias)

        if not return_dict:
            return x, length_logits

        return {
            "logits": x,
            "length_logits": length_logits
            }

    def compute_loss(self, final_logits, labels, length_logits, loss_mask, pad_id=0, eps=0.1, reduction="sum"):
        loss_fn = LossFunc(eps=eps, pad_id=pad_id, reduction=reduction)
        # TODO fix rn reduction is "sum"
        return loss_fn(final_logits, labels, length_logits, loss_mask)

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
        self._save_finetuned(path, verbose=False)

        return True

    def _save_finetuned(self, path:str,
                        encoder_adapter:bool=True,
                        decoder_adapter:bool=True,
                        length_embed:bool=True,
                        lm_head:bool=True,
                        verbose:bool=True):

        state_dict = self.state_dict()
        saving_keys = []

        if encoder_adapter:
            num = len(self.encoder.encoder.layer)
            for i in range(num):
                k1 = f"encoder.encoder.layer.{i}.adapter_ln"
                k2 = f"encoder.encoder.layer.{i}.adapter_w1"
                k3 = f"encoder.encoder.layer.{i}.adapter_w2"
                saving_keys.extend([key for key in state_dict.keys() if (key.startswith(k1) or key.startswith(k2) or key.startswith(k3))])

        if decoder_adapter:
            num = len(self.decoder.bert.encoder.layer)
            for i in range(num):
                k1 = f"decoder.bert.encoder.layer.{i}.encoder_attn"
                k2 = f"decoder.bert.encoder.layer.{i}.encoder_attn_layer_norm"
                k3 = f"decoder.bert.encoder.layer.{i}.encoder_attn_fc1"
                k4 = f"decoder.bert.encoder.layer.{i}.encoder_attn_fc2"
                k5 = f"decoder.bert.encoder.layer.{i}.encoder_attn_final_layer_norm"
                saving_keys.extend([key for key in state_dict.keys() if (key.startswith(k1) or key.startswith(k2) or key.startswith(k3) or key.startswith(k4) or key.startswith(k5))])

        if length_embed:
            k = "encoder.embeddings.length_embedding"
            saving_keys.extend([key for key in state_dict.keys() if key.startswith(k)])

        if lm_head:
            k1 = "final_dense"
            k2 = "final_layer_norm"
            k3 = "final_bias"
            saving_keys.extend([key for key in state_dict.keys() if (key.startswith(k1) or key.startswith(k2) or key.startswith(k3))])

        saving = {}
        for k in saving_keys:
            saving.update({k: state_dict[k]})

        if path:
            if verbose: print(f"saving: {saving.keys()}")
            torch.save(saving, path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path:str):
        """
            Setting up this method will enable to load directly from huggingface hub just like other HF models are loaded
        """
        model_id = pretrained_model_name_or_path

        if len(model_id.split("/")) == 1:
            name = model_id
        else:
            username, name = model_id.split("/")

        if name in os.listdir():
            print("LOADING config & model weights from local directory")
            config_file = os.path.join(name, "config.json")
            model_file = os.path.join(name, "pytorch_model.bin")
        else:
            config_url = hf_bucket_url(model_id, filename="config.json")
            config_file = cached_path(config_url)
            # downloading & load only the adapter weights from huggingface hub
            # and corresponding bert weights will be loaded when class is getting initiated
            model_url = hf_bucket_url(model_id, filename="pytorch_model.bin")
            model_file = cached_path(model_url)

        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        config = Dict.from_nested_dict(config)

        state_dict = torch.load(model_file, map_location="cpu")
        # randomly initializing model from given config with bert weights restored
        model = cls(config)
        # now restoring adapter weights
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return model

class LossFunc(nn.Module):

    def __init__(self, eps=0.1, pad_id=0, reduction="sum"):
        super().__init__()
        self.eps = eps
        self.pad_id = pad_id
        self.reduction = reduction

    def compute_length_loss(self, length_logits, length_labels, pad_mask):
        
        num_pads = pad_mask.long().sum(dim=-1)
        length_labels = length_labels - num_pads
        length_logits = F.log_softmax(length_logits.squeeze(), dim=-1)        
        length_loss = F.nll_loss(length_logits, length_labels.long(), reduction="sum")

        return length_loss

    def compute_translation_loss(self, logits, labels, pad_mask, loss_mask):
        """
            loss_mask is tensor with 1s on positions contributing loss
        """

        logits = F.log_softmax(logits, dim=-1)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1, 1)

        pad_mask = pad_mask.view(-1)
        nll_loss = -logits.gather(dim=-1, index=labels).squeeze()
        nll_loss[pad_mask] = 0.

        smooth_loss = -logits.sum(-1).squeeze()
        smooth_loss[pad_mask] = 0.

        loss_mask = loss_mask.eq(1).view(-1)
  
        nll_loss = nll_loss[loss_mask].sum()
        smooth_loss = smooth_loss[loss_mask].sum()

        return (1.-self.eps)*nll_loss + (self.eps/logits.size(-1))*smooth_loss

    def forward(self, logits, labels, length_logits, loss_mask):

        pad_mask = labels.eq(self.pad_id)
        length_labels = torch.ones(labels.size(), device=labels.device).sum(1)

        length_loss = self.compute_length_loss(length_logits, length_labels, pad_mask)
        translation_loss = self.compute_translation_loss(logits, labels, pad_mask, loss_mask)
        loss = 0.1*length_loss + translation_loss

        if self.reduction == "mean":
            loss_mask.view(-1)[pad_mask.view(-1)] = 0
            num_tokens = loss_mask.sum()
            loss = loss/num_tokens
            length_loss = length_loss/num_tokens
            translation_loss = translation_loss/num_tokens

        return {
            "loss": loss,
            "length_loss": length_loss,
            "translation_loss": translation_loss
            }
