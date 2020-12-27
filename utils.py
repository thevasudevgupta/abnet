# __author__ = 'Vasudev Gupta'

import wandb
from sacrebleu import corpus_bleu
from tqdm import tqdm

import torch
import torch.nn as nn
from data_utils.tokenizer import Tokenizer
from data_utils.dataloader import DataLoader

@torch.no_grad()
def fetch_translations_and_bleu(model:nn.Module,
                            dataset:DataLoader, 
                            tokenizer:Tokenizer,
                            iterations=10,
                            k=1):
    """
        BLEU keeping number of samples in training and validation same
    """

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.cuda()

    pred = []
    tgt = []
    src = []

    for batch in tqdm(dataset, desc="predicting ... "):

        for k in batch:
            batch[k] = batch[k].to(device)

        out = model.generate(**batch, iterations=iterations, tokenizer=tokenizer, k=k)
        pred.extend(out["tgt_text"])
        src.extend(tokenizer.batch_decode(batch["input_ids"], is_src_txt=True))
        tgt.extend(tokenizer.batch_decode(batch["labels"], is_tgt_txt=True))

    # bleu score
    bleu = corpus_bleu(pred, [tgt]).score

    return {
        "bleu": bleu,
        "src": src,
        "tgt": tgt,
        "pred": pred
    }
