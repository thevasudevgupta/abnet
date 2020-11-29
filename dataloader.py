# __author__ = 'Vasudev Gupta'
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, src: list, tgt: list):

        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return {
            'src': self.src[idx],
            'tgt': self.tgt[idx]
        }

class DataLoader(object):

    def __init__(self, transformer_config, args):

        self.enc_tokenizer = BertTokenizer.from_pretrained(transformer_config["encoder_id"])
        self.dec_tokenizer = BertTokenizer.from_pretrained(transformer_config["decoder_id"])

        # decoder based
        self.sep_token = self.dec_tokenizer.convert_tokens_to_ids(self.dec_tokenizer.sep_token)
        self.cls_token = self.dec_tokenizer.convert_tokens_to_ids(self.dec_tokenizer.cls_token)

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.max_length = args.max_length
        self.max_target_length = args.max_target_length

        # prepare_data args
        self.tr_max_samples = args.tr_max_samples
        self.val_max_samples = args.val_max_samples
        self.tst_max_samples = args.tst_max_samples

        self.tr_tgt_file = args.tr_tgt_file
        self.tr_src_file = args.tr_src_file

        self.val_tgt_file = args.val_tgt_file
        self.val_src_file = args.val_src_file

        self.tst_tgt_file = args.tst_tgt_file
        self.tst_src_file = args.tst_src_file

    def __call__(self):

        self.tr_src, self.tr_tgt = self.prepare_data(self.tr_tgt_file, self.tr_src_file, self.tr_max_samples, "tr")
        self.val_src, self.val_tgt = self.prepare_data(self.val_tgt_file, self.val_src_file, self.val_max_samples, "val")
        self.tst_src, self.tst_tgt = self.prepare_data(self.tst_tgt_file, self.tst_src_file, self.tst_max_samples, "tst")

        self.setup()

        tr_dataset = self.train_dataloader()
        val_dataset = self.val_dataloader()
        test_dataset = self.test_dataloader()

        return tr_dataset, val_dataset, test_dataset

    def prepare_data(self, tgt_file, src_file, max_samples, mode="tr"):

        with open(tgt_file) as file1, open(src_file) as file2:
            tgt = file1.readlines()
            src = file2.readlines()
        print(f'total size of {mode} data (src, tgt): ', f'({len(src)}, {len(tgt)})')
        
        src = src[:max_samples]
        tgt = tgt[:max_samples]

        return src, tgt

    def setup(self):
        self.tr_dataset = CustomDataset(self.tr_src, self.tr_tgt)
        self.val_dataset = CustomDataset(self.val_src, self.val_tgt)
        self.test_dataset = CustomDataset(self.tr_src, self.tr_tgt)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.tr_dataset,
                                            pin_memory=True,
                                            shuffle=True,
                                            batch_size=self.batch_size,
                                            collate_fn=self.collate_fn,
                                            num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                            pin_memory=True,
                                            shuffle=False,
                                            batch_size=self.batch_size,
                                            collate_fn=self.collate_fn,
                                            num_workers=self.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                            pin_memory=True,
                                            shuffle=False,
                                            batch_size=self.batch_size,
                                            collate_fn=self.collate_fn,
                                            num_workers=self.num_workers)

    def collate_fn(self, features):

        src = [f['src'] for f in features]
        tgt = [f['tgt'] for f in features]

        batch =  self.prepare_seq2seq_batch(src_texts=src, tgt_texts=tgt)

        return batch

    def build_seqlen_table(self):
    
        # src train data
        lens = [len(self.enc_tokenizer.tokenize(s)) for s in self.tr_src]
        src_tr = {'max': np.max(lens), 'avg': np.mean(lens), 'min': np.min(lens)}
        
        # src val data
        lens = [len(self.enc_tokenizer.tokenize(s)) for s in self.val_src]
        src_val = {'max': np.max(lens), 'avg': np.mean(lens), 'min': np.min(lens)}
        
        # tgt train data
        lens = [len(self.dec_tokenizer.tokenize(t)) for t in self.tr_tgt]
        tgt_tr = {'max': np.max(lens), 'avg': np.mean(lens), 'min': np.min(lens)}
        
        # tgt val data
        lens = [len(self.dec_tokenizer.tokenize(t)) for t in self.val_tgt]
        tgt_val = {'max': np.max(lens), 'avg': np.mean(lens), 'min': np.min(lens)}

        columns = ['src-train', 'src-val', 'tgt-train', 'tgt-val']
        data = [[src_tr[k], src_val[k], tgt_tr[k], tgt_val[k]] for k in ['max', 'avg', 'min']]

        return data, columns

    def prepare_seq2seq_batch(self, src_texts: list, tgt_texts: list = None):

        src_batch = self.enc_tokenizer(src_texts, padding=True, max_length=self.max_length, truncation=True)

        out = {
            "input_ids": torch.tensor(src_batch["input_ids"]),
            "encoder_attention_mask": torch.tensor(src_batch["attention_mask"])
        }

        if tgt_texts is not None:
            tgt_batch = self.dec_tokenizer(tgt_texts, padding=True, max_length=self.max_target_length, truncation=True)

            decoder_input_ids = torch.tensor(tgt_batch["input_ids"])
            labels = decoder_input_ids[:, 1:]
            decoder_input_ids = torch.tensor([m[m!=self.sep_token].tolist() for m in decoder_input_ids])

            decoder_attention_mask = torch.tensor(tgt_batch["attention_mask"])
            decoder_attention_mask = decoder_attention_mask[:, 1:]

            out.update({
                "decoder_input_ids": decoder_input_ids,
                "labels": labels,
                "decoder_attention_mask": decoder_attention_mask
            })

        return out
