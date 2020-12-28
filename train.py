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

# just change this to switch dataset and model config
TRAINING_ID = "iwslt14_de_en" # "wmt16_ro_en"

TRANSFORMER_CONFIG_FILE = os.path.join("transformer_config", f"{TRAINING_ID}.yaml")
TRAINER_CONFIG = getattr(training, TRAINING_ID)

if __name__ == "__main__":

  config = yaml.safe_load(open(TRANSFORMER_CONFIG_FILE, "r"))
  config = Dict.from_nested_dict(config)

  args = TrainerConfig.from_default()
  args.update(TRAINER_CONFIG.__dict__)

  if not args.load_pretrained_path:
    model = TransformerMaskPredict(config)
  else:
    model = TransformerMaskPredict.from_pretrained(args.load_pretrained_path)
  tokenizer = Tokenizer(model.config.encoder_id, model.config.decoder_id, model.config.length_token)

  dl = DataLoader(args, tokenizer)
  tr_dataset, val_dataset, test_dataset = dl()

  trainer = Trainer(model, args)
  if args.load_training_state:
    trainer.load_training_state_dict(args.base_dir)

  trainer.fit(tr_dataset, val_dataset)
    
  if args.save_pretrained_path:
    trainer.model.save_pretrained(args.save_pretrained_path)

  tst_loss = trainer.evaluate(test_dataset)
  wandb.log({"tst_loss": tst_loss})

  for mode, dataset in zip(["tr", "val", "tst"], [tr_dataset, val_dataset, test_dataset]):

    out = fetch_translations_and_bleu(model, dataset, tokenizer, args.iterations, args.B, args.bleu_num_samples)
    data = list(zip(out["src"], out["tgt"], out["pred"]))
    wandb.log({
          mode+'_bleu': out["bleu"],
          mode+'_predictions': wandb.Table(data=data, columns=['src', 'tgt', 'tgt_pred'])
        })
