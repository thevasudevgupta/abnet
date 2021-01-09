# __author__ = 'Vasudev Gupta'

import wandb
from sacrebleu import corpus_bleu
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from data_utils.tokenizer import Tokenizer
from data_utils.dataloader import DataLoader

@torch.no_grad()
def fetch_translations_and_bleu(model:nn.Module,
                            dataset:DataLoader, 
                            tokenizer:Tokenizer,
                            iterations=10,
                            k=1,
                            num_samples=6000):
    """
        BLEU keeping number of samples in training and validation same
    """
    model.eval()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.cuda()

    pred = []
    tgt = []
    src = []

    for batch in tqdm(dataset, desc="predicting ... ", leave=False):

        for k in batch:
            batch[k] = batch[k].to(device)

        out = model.generate(**batch, iterations=iterations, tokenizer=tokenizer, k=k)
        pred.extend(out["tgt_text"])
        src.extend(tokenizer.batch_decode(batch["input_ids"], is_src_txt=True))
        tgt.extend(tokenizer.batch_decode(batch["labels"], is_tgt_txt=True))

        if len(pred) > num_samples:
            break

    # bleu score
    bleu = corpus_bleu(pred, [tgt]).score

    return {
        "bleu": bleu,
        "src": src,
        "tgt": tgt,
        "pred": pred
    }

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument("--tr_max_samples", type=int, default=-1, help="Don't change it")
    parser.add_argument("--val_max_samples", type=int, default=-1, help="Don't change it")
    parser.add_argument("--tst_max_samples", type=int, default=-1, help="Don't change it")
    parser.add_argument("--bleu_num_samples", type=int)
    parser.add_argument("--tr_tgt_file", type=str)
    parser.add_argument("--tr_src_file", type=str)
    parser.add_argument("--val_tgt_file", type=str)
    parser.add_argument("--val_src_file", type=str)
    parser.add_argument("--tst_tgt_file", type=str)
    parser.add_argument("--tst_src_file", type=str)
    parser.add_argument("--model_id", type=str)

    args = parser.parse_args()
    return args
