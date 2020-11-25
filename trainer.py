# __author__ = 'Vasudev Gupta'
import torch
from torch_trainer import TorchTrainer

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
            batch[k] = torch.tensor(batch[k])
            batch[k] = batch[k].to(self.device)

        # with torch.cuda.amp.autocast((self.precision=='mixed_16')):
        out = self.model(**batch, return_dict=True)

        loss = out["loss"].mean()

        return loss

    def validation_step(self, batch):

        for k in batch:
            batch[k] = torch.tensor(batch[k])
            batch[k] = batch[k].to(self.device)

        with torch.no_grad():
            # with torch.cuda.amp.autocast((self.precision=='mixed_16')):
            out = self.model(**batch, return_dict=True)
            loss = out["loss"].mean()

        return loss

    def training_epoch_end(self, epoch, losses):

       if self.args.save_adapter_path:
            self.model.save_adapter(f"{self.args.base_dir}/{self.args.save_adapter_path}-{epoch}.pt", 
                        self.args.enc_ffn_adapter,
                        self.args.dec_ffn_adapter,
                        self.args.cross_attn_adapter)

            self.save_training_state_dict(self.base_dir)
