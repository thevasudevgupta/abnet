# __author__ = "Vasudev Gupta"
import torch

from adapters import MixAdapterTransformer
from modeling_bert import BertModel

class TransformerMaskPredict(MixAdapterTransformer):

    def __init(self, config):
        super().__init__()

        self.encoder = BertModel.from_pretrained(config["encoder_id"])
        self.encoder.resize_token_embeddings(config.vocab_size)

        self.decoder = BertModel.from_pretrained(config["decoder_id"])

        self.register_buffer("final_layer_bias", torch.zeros(1, self.decoder.embeddings.word_embeddings.num_embeddings))

        for param in self.encoder.parameters():
            param.requires_grad_(False)

        for param in self.decoder.parameters():
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

    def forward(self, input_ids, encoder_attention_mask, decoder_input_ids=None, decoder_attention_mask=None, labels=None, mode="training", return_dict=True, **kwargs):
        """
        input_ids :: torch.tensor : [LENGTH], ........., [SEP], [PAD] ...... [PAD]
        decoder_input_ids :: torch.tensor : [CLS], ........, [PAD] ...... [PAD]
        labels: torch.tensor : ............, [SEP], [PAD] ...... [PAD]
        """
        T = 1

        # encoder
        x = self.encoder(input_ids=input_ids,
                    attention_mask=encoder_attention_mask,
                    return_dict=True)
        x = x["last_hidden_state"]
        length_repr = x[:, 0, :]
        x = x[:, 1:, :]

        if mode == "inference":
            T = kwargs["T"]
            K = kwargs["K"]
            pad_id = kwargs["pad_id"]
            mask_id = kwargs["mask_id"]

            print("Prediction mode")
            tgt_lengths = torch.topk(x[:, -1, :], k=k, dim=-1).index

            batch_lengths = ?
            # -> bz
            decoder_input_ids = [[mask_id for i in range(l)] for l in batch_lengths]
            decoder_input_ids = torch.tensor([self._pad(t, max(batch_lengths), pad_id) for t in decoder_input_ids])
            decoder_attention_mask = torch.ne(decoder_input_ids, pad_id)

        for t in range(T):
            # decoder
            x = self.decoder(input_ids=decoder_input_ids,
                        attention_mask=decoder_attention_mask,
                        encoder_hidden_states=x,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=True)
            x = x["last_hidden_state"]
            x = F.linear(x, self.decoder.embeddings.word_embeddings.weight, bias=self.final_layer_bias)

            if mode == "inference":
                num_masks = [torch.floor(len(tensor)*((T-t)/T)).item() for tensor in x]
                mask_posn = [torch.topk(tensor, n, dim=-1, largest=False).index.tolist() for n, tensor in zip(num_masks, x)]
                for i, mask_pos in mask_posn:
                    decoder_input_ids[i, mask_pos] = mask_id
                print(decoder_input_ids)

        if return_dict:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(x, labels)

            return {
                "logits": x,
                "loss": loss.mean()
            }

        return x

    @staticmethod
    def _pad(ls:list, max_len:int, pad:int):
        while len(ls) < max_len:
            ls.append(pad)
        return ls

    @torch.no_grad()
    def generate(self, input_ids, T=10, K=4, tokenizer=None, **kwargs):
        # if extra arguments are specified, then they won't be used
        """This is based on mask-predict as suggested in paper"""
        self.eval()
        out = self(input_ids, encoder_attention_mask, mode="inference", K=K, T=T, pad_id=pad_id, mask_id=mask_id)
        out = torch.max(a, dim=-1).indices
        if tokenizer:
            out = tokenizer.batch_decode(out)
        return out # -> (bz, seqlen)

