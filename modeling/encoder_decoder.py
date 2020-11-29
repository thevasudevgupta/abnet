# __author__ = "Vasudev Gupta"
import torch

from adapters import MixAdapterTransformer
from modeling_bert import BertModel

class TransformerMaskPredict(MixAdapterTransformer):

    def __init(self, config):
        super().__init__()

        self.encoder = BertModel.from_pretrained(config["encoder_id"])
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

    def forward(self, input_ids, decoder_input_ids, encoder_attention_mask, decoder_attention_mask, labels, return_dict=False):
        """
        input_ids :: (torch.tensor) : [CLS], ........., [SEP], [PAD] ...... [PAD]
        decoder_input_ids :: (torch.tensor) : [CLS], ........, [PAD] ...... [PAD]
        labels: (torch.tensor) : ............, [SEP], [PAD] ...... [PAD]
        """

        # encoder
        x = self.encoder(input_ids=input_ids,
                    attention_mask=encoder_attention_mask,
                    return_dict=True)
        x = x["last_hidden_state"]

        # decoder
        x = self.decoder(input_ids=decoder_input_ids,
                    attention_mask=decoder_attention_mask,
                    encoder_hidden_states=x,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=True)
        x = x["last_hidden_state"]

        # -> (bz, seqlen, 768)        
        x = F.linear(x, self.decoder.embeddings.word_embeddings.weight, bias=self.final_layer_bias)
        # -> (bz, seqlen, vocab_size)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(x, labels)

        if return_dict:
            return {
                "logits": x,
                "loss": loss.mean()
            }

        return x

    def generate(self, input_ids, decoder_start_token_id, max_length):
        """This is based on mask-predict as suggested in paper"""
        raise NotImplementedError


