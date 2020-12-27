import torch
import torch.nn.functional as F

class MaskPredict(object):

    def __init__(self):
        """
            This class will be helpful during interence time.
            Inherit TransformerMaskPredict from this class
        """

    @torch.no_grad()
    def generate(self, 
                input_ids:torch.tensor,
                encoder_attention_mask:torch.tensor,
                tokenizer,
                iterations=10,
                k=1):

        # TODO fix K

        # TODO check device of tensor

        self.iterations = iterations
        self.pad_token = tokenizer.tgt_pad_token
        self.mask_token = tokenizer.tgt_mask_token

        x = self.encoder(input_ids=input_ids,
                    attention_mask=encoder_attention_mask,
                    return_dict=True)
        length_logits = x.pop("length_logits")
        x = torch.cat([length_logits, x.pop("last_hidden_state")], dim=1)
        
        _, lengths = length_logits.max(dim=1)

        tgt_tokens = [[mask_token]*t for t in lengths.squeeze()]
        tgt_tokens = [self._pad(ls, lengths.max(), pad_token) for ls in tgt_tokens]

        decoder_attention_mask = [[1]*t for t in lengths.squeeze()]
        decoder_attention_mask = [self._pad(ls, lengths.max(), 0) for ls in tgt_tokens]

        tgt_tokens, lprobs = self._generate(x, encoder_attention_mask, tgt_tokens, decoder_attention_mask)

        output = {
            "tgt_tokens": tgt_tokens,
            "sequence_logprobs": lprobs,
            "tgt_text": tokenizer.batch_decode(tgt_tokens, skip_special_tokens=True)
        }
        return output

    @staticmethod
    def _pad(ls:list, max_len:int, pad:int):
        while len(ls) < max_len:
            ls.append(pad)
        return ls

    def _generate(self, encoder_out, encoder_attention_mask, tgt_tokens, decoder_attention_mask):
        pad_mask = decoder_attention_mask.eq(0)
        seqlens = tgt_tokens.size(1) - pad_mask.sum(dim=1)

        # starting 0th iteration
        tgt_tokens, token_probs = self._generate_non_autoregressive(encoder_out, encoder_attention_mask, tgt_tokens, decoder_attention_mask)
        
        # [PADDING]
        tgt_tokens.view(-1)[pad_mask.view(-1)] = self.pad_token
        token_probs.view(-1)[pad_mask.view(-1)] = 1.0

        for counter in range(1, self.iterations):

            num_mask = (seqlens.float() * (1.0 - (counter / self.iterations))).long()
            mask_ind = self.select_worst(token_probs, num_mask)

            # [INPUT MASKING]
            assign_single_value_long(tgt_tokens, mask_ind, self.mask_token)
            tgt_tokens.view(-1)[pad_mask.view(-1)] = self.pad_token

            new_tgt_tokens, new_token_probs = self._generate_non_autoregressive(encoder_out, encoder_attention_mask, tgt_tokens, decoder_attention_mask)

            # insert new values in places where masking was done in input
            assign_multi_value_long(token_probs, mask_ind, new_token_probs)            
            token_probs.view(-1)[pad_mask.view(-1)] = 1.0

            assign_multi_value_long(tgt_tokens, mask_ind, new_tgt_tokens)
            tgt_tokens.view(-1)[pad_mask.view(-1)] = self.pad_token

        # getting log-probability of sequence
        lprobs = token_probs.log().sum(-1)
        return tgt_tokens, lprobs

    @torch.no_grad()
    def _generate_non_autoregressive(self, encoder_out, encoder_attention_mask, tgt_tokens, decoder_attention_mask):
        out = self.decoder(input_ids=tgt_tokens,
                        attention_mask=decoder_attention_mask,
                        encoder_hidden_states=encoder_out,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=True)
        probs = F.softmax(out["last_hidden_state"], dim=-1)
        tgt_probs, tgt_tokens = probs.max(dim=-1)
        return tgt_tokens, tgt_probs

    def select_worst(self, token_probs, num_mask):
        bz, seqlen = token_probs.size()
        masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bz)]
        masks = [torch.cat([mask, mask.new(seqlen - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        return torch.stack(masks, dim=0)

def assign_single_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y

def assign_multi_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y.view(-1)[i.view(-1)]
