# __author__ = 'Vasudev Gupta'

import wandb
from sacrebleu import corpus_bleu


class Logger(object):

    def __init__(self, trainer, dl, batch_size, num_samples):
        self.trainer = trainer
        self.dl = dl
        self.batch_size = batch_size
        self.num_samples = num_samples

    def log_length_table(self):
        # seqlen logging
        data, columns = self.dl.build_seqlen_table()
        wandb.log({'Sequence-Lengths': wandb.Table(data=data, columns=columns)})

    def log_translations_and_bleu(self, src_texts, tgt_texts, mode="val"):
        # bleu keeping number of samples in training and validation same
        indices = range(0, self.num_samples, self.batch_size)

        # batching the data
        src_texts = [src_texts[start:self.batch_size+start] for start in indices]
        tgt_texts = [tgt_texts[start:self.batch_size+start] for start in indices]

        print(f"Calculating bleu over {mode} data", end=" ")
        data, pred, tgt = self.trainer.fetch_translations(src_texts, tgt_texts, self.dl)
        wandb.log({mode+'_predictions': wandb.Table(data=data, columns=['src', 'tgt', 'tgt_pred'])})

        tgt = [d[1] for d in data]
        pred = [d[2] for d in data]

        # bleu score
        bleu = corpus_bleu(pred, [tgt]).score
        wandb.log({mode+'_bleu': bleu})
        print("||DONE||")
