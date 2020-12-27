# __author__ = "Vasudev Gupta"

import yaml
import wandb
import os

from dataloader import DataLoader
import training
from training import Trainer, TrainerConfig, Logger
from modeling import TransformerMaskPredict, Dict

TRAINING_ID = "iwslt14_de_en"

TRANSFORMER_CONFIG_FILE = os.path.join("transformer_config", f"{TRAINING_ID}.yaml")
TRAINER_CONFIG = getattr(training, TRAINING_ID)

if __name__ == "__main__":

    # setting config of transformer
    transformer_config = yaml.safe_load(open(TRANSFORMER_CONFIG_FILE, "r"))
    transformer_config = Dict.from_nested_dict(transformer_config)
    print(transformer_config)

    # setup training config
    trainer_config = TrainerConfig.from_default()
    trainer_config.update(TRAINER_CONFIG.__dict__)

    print(trainer_config)

    # setup transformer for mask-predict
    model = TransformerMaskPredict(transformer_config)

    # preparing data
    dl = DataLoader(transformer_config, trainer_config)
    tr_dataset, val_dataset, test_dataset = dl()

    # setting-up system for training
    trainer = Trainer(model, trainer_config)
    trainer.fit(tr_dataset, val_dataset)

    # TODO
    # pass input twice and check whether same output

    # testing on test-data
    tst_loss = trainer.evaluate(test_dataset)
    wandb.log({"tst_loss": tst_loss})

    finetuned_path = trainer_config.save_finetuned_path
    if finetuned_path:
        save_path = os.path.join(trainer_config.base_dir, finetuned_path)
        trainer.model.save_finetuned(save_path)

    # if args.mode == "infer":
    #     model.from_pretrained("vasudevgupta/abnet-iwslt14")
    #     predictor = MaskPredict(model, iterations=args.iterations)
        
    #     predictor.generate()

    # logging bleu and predictions
    logger = Logger(trainer, dl, trainer_config.batch_size, len(dl.val_src))
    logger.log_length_table()
    logger.log_translations_and_bleu(dl.tr_src, dl.tr_tgt, mode="tr")
    logger.log_translations_and_bleu(dl.val_src, dl.val_tgt, mode="val")
    logger.log_translations_and_bleu(dl.tst_src, dl.tst_tgt, mode="tst")
