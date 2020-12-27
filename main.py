# __author__ = "Vasudev Gupta"

import yaml
import wandb
import os
import argparse

from data_utils import DataLoader, Tokenizer
import training
from training import Trainer, TrainerConfig, Logger
from modeling import TransformerMaskPredict, Dict

TRAINING_ID = "iwslt14_de_en"
TRANSFORMER_CONFIG_FILE = os.path.join("transformer_config", f"{TRAINING_ID}.yaml")
TRAINER_CONFIG = getattr(training, TRAINING_ID)

if __name__ == "__main__":

    # setting config of transformer
    config = yaml.safe_load(open(TRANSFORMER_CONFIG_FILE, "r"))
    config = Dict.from_nested_dict(config)

    # setup training config
    args = TrainerConfig.from_default()
    args.update(TRAINER_CONFIG.__dict__)

    # setup transformer for mask-predict
    model = TransformerMaskPredict(config)
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
