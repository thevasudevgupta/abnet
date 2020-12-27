# __author__ = "Vasudev Gupta"
import yaml
import argparse
from data_utils import DataLoader, Tokenizer
from utils import fetch_translations_and_bleu
import wandb
from modeling import TransformerMaskPredict

PROJECT_NAME = "parallel-decoder-paper"

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument("--tr_max_samples", type=int, default=2000)
    parser.add_argument("--val_max_samples", type=int, default=2000)
    parser.add_argument("--tst_max_samples", type=int, default=2000)
    parser.add_argument("--tr_tgt_file", type=str)
    parser.add_argument("--tr_src_file", type=str)
    parser.add_argument("--val_tgt_file", type=str)
    parser.add_argument("--val_src_file", type=str)
    parser.add_argument("--tst_tgt_file", type=str)
    parser.add_argument("--tst_src_file", type=str)
    parser.add_argument("--model_id", type=str)

    args = parser.parse_args()
    return args

# TODO
from modeling import Dict
# rm model.config.length_token

if __name__ == "__main__":

    args = get_args()
    wandb.init(project=PROJECT_NAME, config=args)

    # TODO rm this
    config = yaml.safe_load(open("transformer_config/iwslt14_de_en.yaml", "r"))
    config = Dict.from_nested_dict(config)
    # 

    # TODO change this
    # model = TransformerMaskPredict.from_pretrained(args.model_id)
    model = TransformerMaskPredict(config)
    model.config.length_token = 0
    model.config.num_lengths = 768
# TODO rm this
    for k in ["enc_ffn_adapter_config", "dec_ffn_adapter_config", "cross_attn_adapter_config"]:
      model.config[k]["layer_norm_eps"] = 1.e-5
# 

    tokenizer = Tokenizer(model.config.encoder_id, model.config.decoder_id, model.config.length_token)

    dl = DataLoader(args, tokenizer)
    tr_dataset, val_dataset, tst_dataset = dl()

    data, columns = dl.build_seqlen_table()
    wandb.log({'Sequence-Lengths': wandb.Table(data=data, columns=columns)})

    # add upper limit on dataset
    for mode, dataset in zip(["tr", "val", "tst"], [tr_dataset, val_dataset, tst_dataset]):

        out = fetch_translations_and_bleu(model, dataset, tokenizer, args.iterations, args.k)

        wandb.log({mode+'_bleu': out["bleu"]})
        wandb.log({mode+'_predictions': wandb.Table(data=[out["src"], out["tgt"], out["pred"]], columns=['src', 'tgt', 'tgt_pred'])})
