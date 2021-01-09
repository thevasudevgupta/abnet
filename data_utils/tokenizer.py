# __author__ = 'Vasudev Gupta'

import numpy as np
import random
import torch
from transformers import BertTokenizer


class Tokenizer(object):

    def __init__(self, encoder_id, decoder_id, length_token=0):

        self.encoder_tokenizer = BertTokenizer.from_pretrained(encoder_id)
        self.decoder_tokenizer = BertTokenizer.from_pretrained(decoder_id)

        self.length_token = length_token
        self.sep_id = self.decoder_tokenizer.convert_tokens_to_ids(self.decoder_tokenizer.sep_token)
        self.src_pad_token = self.encoder_tokenizer.pad_token

        # some useful args
        self.tgt_cls_id = self.decoder_tokenizer.convert_tokens_to_ids(self.decoder_tokenizer.cls_token)
        self.tgt_pad_id = self.decoder_tokenizer.convert_tokens_to_ids(self.decoder_tokenizer.pad_token)
        self.tgt_mask_id = self.decoder_tokenizer.convert_tokens_to_ids(self.decoder_tokenizer.mask_token)
        self.tgt_sep_id = self.decoder_tokenizer.convert_tokens_to_ids(self.decoder_tokenizer.sep_token)

    def prepare_seq2seq_batch(self, src_texts:list, tgt_texts:list=None, max_length:int=32, max_target_length:int=32):

        src_batch = self.encoder_tokenizer(src_texts, padding=True, max_length=max_length, truncation=True)

        input_ids = torch.tensor(src_batch["input_ids"])
        bz = len(input_ids)
        input_ids = torch.cat([self.length_token*torch.ones(bz, 1), input_ids], dim=1)
        
        encoder_attention_mask = torch.tensor(src_batch["attention_mask"])
        encoder_attention_mask = torch.cat([torch.ones(bz, 1), encoder_attention_mask], dim=1)

        out = {
            "input_ids": input_ids.long(),
            "encoder_attention_mask": encoder_attention_mask.long()
        }

        if tgt_texts is not None:
            tgt_batch = self.decoder_tokenizer(tgt_texts, padding=True, max_length=max_target_length, truncation=True)

            decoder_input_ids = torch.tensor(tgt_batch["input_ids"])
            labels = torch.tensor(tgt_batch["input_ids"])

            decoder_attention_mask = torch.tensor(tgt_batch["attention_mask"])

            out.update({
                "decoder_input_ids": decoder_input_ids,
                "labels": labels,
                "decoder_attention_mask": decoder_attention_mask
            })

        return out

    def batch_decode(self, batch:torch.Tensor, is_src_txt=False, is_tgt_txt=False, skip_special_tokens=False):
        if is_src_txt:
            out = self.encoder_tokenizer.batch_decode(batch, skip_special_tokens=skip_special_tokens)
            if not skip_special_tokens:
                out = [sent.replace(self.src_pad_token, "[LENGTH]", 1) for sent in out]
        elif is_tgt_txt:
            out = self.decoder_tokenizer.batch_decode(batch, skip_special_tokens=skip_special_tokens)
        else:
            raise ValueError("specify either of encoder or decoder to be True")
        return out

    def mask_decoder_ids(self, decoder_input_ids):
        
        mask_id = self.tgt_mask_id 
        pad_id = self.tgt_pad_id
 
        masked_decoder_ids, mask_ids = self.mask_linearly(decoder_input_ids, mask_id, pad_id)

        return {
            "masked_decoder_ids": masked_decoder_ids,
            "mask_ids": mask_ids
        }

    @staticmethod
    def mask_linearly(tensor:torch.Tensor, mask_id:int, pad_id:int):
        """
            Masking is done linearly such that any seq looses any # tokens from 1 to seqlen .

            mask_locations is excluding <pad> tokens automatically by putting 0s at padded positions

            returns
                masked_tensor, mask_locations
        """

        bz, seqlens = tensor.size()
        pad_mask = tensor.eq(pad_id)
        seqlens = seqlens - pad_mask.float().sum(1)

        num_masks = torch.tensor([random.randint(1, seqlen) for seqlen in seqlens.squeeze().tolist()], device=tensor.device)
        probs = num_masks / seqlens.squeeze()
        mask_bools = [[np.random.rand()>(1-probs[i].item()) for t in tensor[i]] for i in range(bz)]
        mask_bools = torch.tensor(mask_bools, device=tensor.device)
        tensor.view(-1)[mask_bools.view(-1)] = mask_id
        tensor.view(-1)[pad_mask.view(-1)] = pad_id

        mask_ids = mask_bools.long()
        mask_ids.view(-1)[pad_mask.view(-1)] = 0

        return tensor, mask_ids
