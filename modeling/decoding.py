import torch

def generate_step_with_prob(out):
    probs = F.softmax(out[0], dim=-1)
    max_probs, idx = probs.max(dim=-1)
    return idx, max_probs, probs

def assign_single_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y

def assign_single_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y

def assign_multi_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y.view(-1)[i.view(-1)]


# call encoder as regular encoder


class MaskPredict(object):

    def __init__(self, model, iterations=None):
        self.model = model
        self.iterations = iterations

    def generate(self, src_text):
        src_text
        self.model.encoder()

        tgt_tokens = ["mask"]

        self._generate()

    def _generate(self, encoder_out, tgt_tokens, pad_token, mask_token):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(pad_token)
        seq_lens = seq_len - pad_mask.sum(dim=1)

        iterations = seq_len if self.iterations is None else self.iterations

        tgt_tokens, token_probs = self._generate_non_autoregressive(encoder_out, tgt_tokens)
        assign_single_value_byte(tgt_tokens, pad_mask, pad_token)
        assign_single_value_byte(token_probs, pad_mask, 1.0)
        
        for counter in range(1, iterations):
            num_mask = (seq_lens.float() * (1.0 - (counter / iterations))).long()

            assign_single_value_byte(token_probs, pad_mask, 1.0)
            mask_ind = self.select_worst(token_probs, num_mask)
            assign_single_value_long(tgt_tokens, mask_ind, mask_token)
            assign_single_value_byte(tgt_tokens, pad_mask, pad_token)

            decoder_out = self.model.decoder(tgt_tokens, encoder_out)
            new_tgt_tokens, new_token_probs, _ = generate_step_with_prob(decoder_out)

            assign_multi_value_long(token_probs, mask_ind, new_token_probs)
            assign_single_value_byte(token_probs, pad_mask, 1.0)

            assign_multi_value_long(tgt_tokens, mask_ind, new_tgt_tokens)
            assign_single_value_byte(tgt_tokens, pad_mask, pad_token)

        lprobs = token_probs.log().sum(-1)
        return tgt_tokens, lprobs
    
    def _generate_non_autoregressive(self, encoder_out, tgt_tokens):
        decoder_out = self.model.decoder(tgt_tokens, encoder_out)
        tgt_tokens, token_probs, _ = generate_step_with_prob(decoder_out)
        return tgt_tokens, token_probs

    def select_worst(self, token_probs, num_mask):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        return torch.stack(masks, dim=0)

    @torch.no_grad()
    def fetch_translations(self, src_texts, tgt_texts, dl):

        self.model.eval()
        data = []

        bar = tqdm(zip(src_texts, tgt_texts), desc="predicting ... ", leave=False)
        for s, t in bar:
            batch = dl.prepare_seq2seq_batch(src_texts=s)

            for k in batch:
                batch[k] = batch[k].to(self.device)

            out = self.model.generate(**batch, decoder_start_token_id=dl.sep_token, max_length=dl.max_target_length)
            pred = tokenizer.batch_decode(out, skip_special_tokens=True)

            data.extend(list(zip(s, t, pred)))

        # (src, tgt, prediction)
        return data