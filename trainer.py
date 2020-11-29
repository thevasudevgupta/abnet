# __author__ = 'Vasudev Gupta'
import torch
from torch_trainer import TorchTrainer
from tqdm import tqdm


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

       if self.args.save_adapter_path:
            self.model.save_adapter(f"{self.args.base_dir}/{self.args.save_adapter_path}-{epoch}.pt", 
                        self.args.enc_ffn_adapter,
                        self.args.dec_ffn_adapter,
                        self.args.cross_attn_adapter)

            self.save_training_state_dict(self.args.base_dir)

    @torch.no_grad()
    def fetch_translations(self, src_texts, tgt_texts, dl):

        self.model.eval()
        data = []

        bar = tqdm(zip(src_texts, tgt_texts), desc="predicting ... ", leave=False)
        for s, t in bar:
            batch = dl.prepare_seq2seq_batch(src_texts=s)

            for k in batch:
                batch[k] = batch[k].to(self.device)

            out = self.model.generate(**batch, decoder_start_token_id=dl.sep_token, max_length=dl.max_target_length)
            pred = tokenizer.batch_decode(out, skip_special_tokens=True)

            data.extend(list(zip(s, t, pred)))

        # (src, tgt, prediction)
        return data
