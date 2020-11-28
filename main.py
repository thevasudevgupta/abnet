
import torch
import argparse

from dataloader import DataLoader
from trainer import Trainer
from modeling import TransformerMaskPredict
import config
from utils import Logger

if __name__ == "__main__":

    # setup config
    trainer_config = config.main
    transformer_config = config.IWSLT14

    # setup transformer for mask-predict
    model = TransformerMaskPredict(transformer_config)

    # preparing data
    dl = DataLoader(transformer_config, trainer_config)
    tr_dataset, val_dataset = dl()

    # setting-up system for training
    trainer = Trainer(model, trainer_config)
    trainer.fit(tr_dataset, val_dataset)

    if trainer_config.save_adapter_path:
        trainer.model.save_adapter(f"{trainer_config.base_dir}/{trainer_config.save_adapter_path}", 
                    trainer_config.enc_ffn_adapter, 
                    trainer_config.dec_ffn_adapter,
                    trainer_config.cross_attn_adapter)

    logger = Logger(trainer, dl, trainer_config.batch_size, len(dl.val_src))
    logger.log_length_table()
    logger.log_translations_and_bleu(dl.tr_src, dl.tr_tgt, mode="tr")
    logger.log_translations_and_bleu(dl.val_src, dl.val_tgt, mode="val")
