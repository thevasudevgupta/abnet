# __author__ = 'Vasudev Gupta'

import torch
from training.torch_trainer import TorchTrainer
from tqdm import tqdm
import os


class Trainer(TorchTrainer):

    def __init__(self, model, tokenizer, args):

        self.model = model
        self.tokenizer = tokenizer
        self.lr = args.lr
        self.args = args

        super().__init__(args)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        # dynamic masking over decoder_input_ids
        out = self.tokenizer.mask_decoder_ids(batch["decoder_input_ids"])
        batch["decoder_input_ids"] = out.pop("masked_decoder_ids")

        for k in batch:
            batch[k] = batch[k].to(self.device)

        # TODO: complete it whenever loss_fn is ready
        loss_mask = out.pop("mask_ids")

        # with torch.cuda.amp.autocast((self.precision=='mixed_16')):
        out = self.model(**batch, return_dict=True)
        self.log(length_loss=out["length_loss"], translation_loss=out["translation_loss"])

        return out["loss"]

    @torch.no_grad()
    def validation_step(self, batch):
        # dynamic masking over decoder_input_ids
        out = self.tokenizer.mask_decoder_ids(batch["decoder_input_ids"])
        batch["decoder_input_ids"] = out.pop("masked_decoder_ids")

        for k in batch:
            batch[k] = batch[k].to(self.device)

        # TODO: complete it whenever loss_fn is ready
        loss_mask = out.pop("mask_ids")

        out = self.model(**batch, return_dict=True)
        self.log(length_loss=out["length_loss"], translation_loss=out["translation_loss"])
        return out["loss"]

    def training_epoch_end(self, epoch, losses):

        save_pretrained_path = self.args.save_pretrained_path
        if save_pretrained_path:
            self.model.save_pretrained(save_pretrained_path)
        if self.args.save_training_state:
            self.save_training_state_dict(self.args.base_dir)
