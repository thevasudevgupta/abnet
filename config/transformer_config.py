# __author__ = 'Vasudev Gupta'

class Dict(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, k):
        return getattr(self, k)

    def __repr__(self):
        return f"ModelConfig({self.kwargs})"

    def get(self, k, op=None):
        if not hasattr(self, k):
            return op
        return getattr(self, k)

# IWSLT14 De-En
IWSLT14 = Dict(encoder_id="bert-base-german-cased",
            decoder_id="bert-base-uncased",
            enc_ffn_adapter=True,
            dec_ffn_adapter=True,
            cross_attn_adapter=True,
            mask_token="[MASK]",
            pad_token="[PAD]",
            # its fine to overlap with pad token since embedding layer is different in both cases
            length_id=0,
            num_lengths=512,
            enc_ffn_adapter_config=Dict(hidden_size=768,
                                    intermediate_size=512,
                                    layer_norm_eps=1e-12
            ),
            dec_ffn_adapter_config=Dict(hidden_size=768,
                                    intermediate_size=2048,
                                    layer_norm_eps=1e-12
            ),
            cross_attn_adapter_config=Dict(hidden_size=768,
                                        layer_norm_eps=1e-12,
                                        hidden_dropout_prob=0.1,
                                        num_attention_heads=12,
                                        attention_probs_dropout_prob=0.1
            )
)
