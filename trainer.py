# __author__ = 'Vasudev Gupta'

import torch
from torch_trainer import TorchTrainer
from tqdm import tqdm
import os


class Trainer(TorchTrainer):

    def __init__(self, model, args):

        self.model = model
        self.lr = args.lr
        self.args = args

        super().__init__(args)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        for k in batch:
            batch[k] = batch[k].to(self.device)
        # with torch.cuda.amp.autocast((self.precision=='mixed_16')):
        out = self.model(**batch, return_dict=True)
        return out["loss"]

    @torch.no_grad()
    def validation_step(self, batch):
        for k in batch:
            batch[k] = batch[k].to(self.device)
        out = self.model(**batch, return_dict=True)
        return out["loss"]

    def training_epoch_end(self, epoch, losses):

        finetuned_path = self.args.save_finetuned_path
        if finetuned_path:
            save_path = os.path.join(self.args.base_dir, f"e{epoch}-{finetuned_path}")
            self.model.save_finetuned(save_path)

            self.save_training_state_dict(self.args.base_dir)
