
import torch
import argparse
from sacrebleu import corpus_bleu

from dataloader import DataLoader
from trainer import Trainer
from modeling import TransformerMaskPredict
import config

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

    # seqlen logging
    data, columns = dl.build_seqlen_table()
    wandb.log({'Sequence-Lengths': wandb.Table(data=data, columns=columns)})

    # bleu keeping number of samples in training and validation same
    indices = range(0, len(dl.val_src), trainer_config.batch_size)

    src = [dl.tr_src[start:trainer_config.batch_size+start] for start in indices]
    tgt = [dl.tr_tgt[start:trainer_config.batch_size+start] for start in indices]
    print(f"Calculating bleu over training data", end=" ")
    tr_data, pred, tgt = predictor(trainer.model, tokenizer, src, tgt, trainer_config.max_pred_length, trainer_config.src_lang)
    wandb.log({'tr_predictions': wandb.Table(data=tr_data, columns=['src', 'tgt', 'tgt_pred'])})
    # bleu score
    tr_bleu = corpus_bleu(pred, [tgt]).score
    wandb.log({'tr_bleu': tr_bleu})
    print("||DONE||")

    src = [dl.val_src[start:trainer_config.batch_size+start] for start in indices]
    tgt = [dl.val_tgt[start:trainer_config.batch_size+start] for start in indices]
    print(f"Calculating bleu over val data", end=" ")
    val_data, pred, tgt = predictor(trainer.model, tokenizer, src, tgt, trainer_config.max_pred_length, trainer_config.src_lang)
    wandb.log({'val_predictions': wandb.Table(data=val_data, columns=['src', 'tgt', 'tgt_pred'])})
    # bleu score
    val_bleu = corpus_bleu(pred, [tgt]).score
    wandb.log({'val_bleu': val_bleu})
    print("||DONE||")
