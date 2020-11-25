
from transformers import (
    BertModel,
    BertTokenizer
)
import torch
import argparse
from sacrebleu import corpus_bleu

from dataloader import DataLoader
from trainer import Trainer
from modeling import Transformer
from utils import read_prepare_data
import config

if __name__ == "__main__":

    # setup config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="main")
    args = parser.parse_args()
    args = getattr(config, args.config)

    # preparing tokenizers
    enc_tokenizer = BertTokenizer.from_pretrained(args.enc_bert_id)
    dec_tokenizer = BertTokenizer.from_pretrained(args.dec_bert_id)

    # preparing encoder
    enc_bert = BertModel.from_pretrained(args.enc_bert_id)
    enc_bert.add_adapter_(args.enc_ffn_adapter, args.enc_ffn_adapter_config)

    # preparing decoder
    dec_bert = BertModel.from_pretrained(args.dec_bert_id)
    dec_bert.add_adapter_(args.dec_ffn_adapter, args.cross_attn_adapter, args.dec_ffn_adapter_config, args.cross_attn_adapter_config)

    # TODO
    # requires_grad ?

    # integrating encoder, decoder
    model = Transformer(encoder=enc_bert, decoder=dec_bert)

    # preparing data
    dl = DataLoader(enc_tokenizer, dec_tokenizer, args)
    tr_dataset, val_dataset = dl()

    # setting-up system for training
    trainer = Trainer(model, args)
    trainer.fit(tr_dataset, val_dataset)

    if args.save_adapter_path:
        bart.save_adapter(f"{args.base_dir}/{args.save_adapter_path}", 
                    args.enc_ffn_adapter, 
                    args.dec_ffn_adapter,
                    args.cross_attn_adapter)

    # seqlen logging
    data, columns = dl.build_seqlen_table()
    wandb.log({'Sequence-Lengths': wandb.Table(data=data, columns=columns)})

    # bleu keeping number of samples in training and validation same
    indices = range(0, len(dl.val_src), args.batch_size)

    src = [dl.tr_src[start:args.batch_size+start] for start in indices]
    tgt = [dl.tr_tgt[start:args.batch_size+start] for start in indices]
    print(f"Calculating bleu over training data", end=" ")
    tr_data, pred, tgt = predictor(trainer.model, tokenizer, src, tgt, args.max_pred_length, args.src_lang)
    wandb.log({'tr_predictions': wandb.Table(data=tr_data, columns=['src', 'tgt', 'tgt_pred'])})
    # bleu score
    tr_bleu = corpus_bleu(pred, [tgt]).score
    wandb.log({'tr_bleu': tr_bleu})
    print("||DONE||")

    src = [dl.val_src[start:args.batch_size+start] for start in indices]
    tgt = [dl.val_tgt[start:args.batch_size+start] for start in indices]
    print(f"Calculating bleu over val data", end=" ")
    val_data, pred, tgt = predictor(trainer.model, tokenizer, src, tgt, args.max_pred_length, args.src_lang)
    wandb.log({'val_predictions': wandb.Table(data=val_data, columns=['src', 'tgt', 'tgt_pred'])})
    # bleu score
    val_bleu = corpus_bleu(pred, [tgt]).score
    wandb.log({'val_bleu': val_bleu})
    print("||DONE||")
