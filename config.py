
model_config = {
    "encoder":  {
        "ffn_adapter_config":
    },
    "decoder":  {
        "ffn_adapter_config": ,
        "attn_adapter_config":
    }

}


class ModelConfig:

    enc_bert_id: str = "bert-base-uncased"
    dec_bert_id: str = "bert-base-uncased"

    adapter_config: AdapterConfig = AdapterConfig(input_size=768)