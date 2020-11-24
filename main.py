from transformers import (
    BertModel,
    BertTokenizer
)
import torch

from dataloader import DataLoader
from trainer import Trainer
from modeling import Transformer
from utils import read_prepare_data
from config import transformer_config, args

if __name__ == "__main__":

    enc_tokenizer = BertTokenizer.from_pretrained(args.enc_bert_id)
    dec_tokenizer = BertTokenizer.from_pretrained(args.dec_bert_id)

    enc_bert = BertModel.from_pretrained(args.enc_bert_id)
    dec_bert = BertModel.from_pretrained(args.dec_bert_id)
    model = Transformer(enc_bert, transformer_config, dec_bert=dec_bert)

    tr_src, tr_tgt, val_src, val_tgt, src, tgt = read_prepare_data(args)
    dataloader = DataLoader(tr_src, tr_tgt, val_src, val_tgt, enc_tokenizer, dec_tokenizer, args)
    tr_dataset, val_dataset = dataloader()

    trainer = Trainer(model, args)
    trainer.fit(tr_dataset, val_dataset)
