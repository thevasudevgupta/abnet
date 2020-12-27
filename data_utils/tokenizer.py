# __author__ = 'Vasudev Gupta'

import torch
from transformers import BertTokenizer


class Tokenizer(object):

    def __init__(self, encoder_id, decoder_id, length_token=0):

        self.encoder_tokenizer = BertTokenizer.from_pretrained(encoder_id)
        self.decoder_tokenizer = BertTokenizer.from_pretrained(decoder_id)

        self.length_token = length_token
        self.sep_id = self.decoder_tokenizer.convert_tokens_to_ids(self.decoder_tokenizer.sep_token)

        # some useful args
        self.tgt_pad_token = self.decoder_tokenizer.convert_tokens_to_ids(self.decoder_tokenizer.pad_token)
        self.tgt_mask_token = self.decoder_tokenizer.convert_tokens_to_ids(self.decoder_tokenizer.mask_token)

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
            labels = decoder_input_ids[:, 1:]
            decoder_input_ids = torch.tensor([m[m!=self.sep_id].tolist() for m in decoder_input_ids])

            decoder_attention_mask = torch.tensor(tgt_batch["attention_mask"])
            decoder_attention_mask = decoder_attention_mask[:, 1:]

            out.update({
                "decoder_input_ids": decoder_input_ids,
                "labels": labels,
                "decoder_attention_mask": decoder_attention_mask
            })

        return out

    def batch_decode(self, batch:torch.Tensor, is_src_txt=False, is_tgt_txt=False, skip_special_tokens=False):
        if is_src_txt:
            out = self.encoder_tokenizer.batch_decode(batch, skip_special_tokens=skip_special_tokens)
        elif is_tgt_txt:
            out = self.decoder_tokenizer.batch_decode(batch, skip_special_tokens=skip_special_tokens)
        else:
            raise ValueError("specify either of encoder or decoder to be True")
        return out
