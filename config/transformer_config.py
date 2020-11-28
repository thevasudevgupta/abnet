# __author__ = 'Vasudev Gupta'


class Dict(dict):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattribute__(self, k):
        return self[k]


# IWSLT14 De-En
IWSLT14 = Dict(encoder_id="bert-base-german-cased",
            decoder_id="bert-base-uncased",
            enc_ffn_adapter=True,
            dec_ffn_adapter=True,
            cross_attn_adapter=True,
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
