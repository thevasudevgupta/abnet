# __author__ = "Vasudev Gupta"

import yaml
import wandb
import os
import argparse

from data_utils import DataLoader, Tokenizer
import training
from training import Trainer, TrainerConfig
from modeling import TransformerMaskPredict, Dict
from utils import fetch_translations_and_bleu

TRAINING_ID = "iwslt14_de_en" # "wmt16_ro_en"
TRANSFORMER_CONFIG_FILE = os.path.join("transformer_config", f"{TRAINING_ID}.yaml")
TRAINER_CONFIG = getattr(training, TRAINING_ID)

# TODO
# rm model.config.length_token

# TODO
# initialize adapters weights

# remember lengths embedding must be shuch that its max value is max val of tgt seqlen

if __name__ == "__main__":

    # setting config of transformer
    config = yaml.safe_load(open(TRANSFORMER_CONFIG_FILE, "r"))
    config = Dict.from_nested_dict(config)

# TODO rm this
    for k in ["enc_ffn_adapter_config", "dec_ffn_adapter_config", "cross_attn_adapter_config"]:
      config[k]["layer_norm_eps"] = 1.e-5

    # setup training config
    args = TrainerConfig.from_default()
    args.update(TRAINER_CONFIG.__dict__)

    # setup transformer for mask-predict
    model = TransformerMaskPredict(config)
    model.config.length_token = 0
    tokenizer = Tokenizer(model.config.encoder_id, model.config.decoder_id, model.config.length_token)

    # preparing data
    dl = DataLoader(args, tokenizer)
    tr_dataset, val_dataset, test_dataset = dl()

    # setting-up system for training
    trainer = Trainer(model, args)
    trainer.fit(tr_dataset, val_dataset)
    trainer.model.save_pretrained(args.save_finetuned_path)

    # testing on test-data
    tst_loss = trainer.evaluate(test_dataset)
    wandb.log({"tst_loss": tst_loss})

    # TODO
    # pass input twice and check whether same output

    # TODO
    # mask the decoder_input_ids

    data, columns = dl.build_seqlen_table()
    wandb.log({'Sequence-Lengths': wandb.Table(data=data, columns=columns)})

    # TODO
    # add upper limit on dataset
    for mode, dataset in zip(["tr", "val", "tst"], [tr_dataset, val_dataset, tst_dataset]):

        out = fetch_translations_and_bleu(model, dataset, tokenizer, args.iterations, args.k)

        wandb.log({mode+'_bleu': out["bleu"]})
        wandb.log({mode+'_predictions': wandb.Table(data=[out["src"], out["tgt"], out["pred"]], columns=['src', 'tgt', 'tgt_pred'])})

