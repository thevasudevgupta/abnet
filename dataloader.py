# __author__ = 'Vasudev Gupta'
import numpy as np
import torch
from sklearn.model_selection import train_test_split

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, src: list, tgt: list):

        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return {
            'input_ids': self.src[idx],
            'labels': self.tgt[idx]
        }

class DataLoader(object):

    def __init__(self, enc_tokenizer, dec_tokenizer, args):

        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.max_length = args.max_length
        self.max_target_length = args.max_target_length

        self.src_lang = args.src_lang
        self.tgt_lang = args.tgt_lang

        # prepare_data args
        self.tr_max_samples = args.tr_max_samples
        self.val_max_samples = args.val_max_samples
        self.random_state = args.random_seed
        self.test_size = args.test_size
        self.tgt_file = args.tgt_file
        self.src_file = args.src_file

    def __call__(self):
        self.prepare_data()
        self.setup()
        tr_dataset = self.train_dataloader()
        val_dataset = self.val_dataloader()
        return tr_dataset, val_dataset

    def prepare_data(self):
        # with open("data/itr.txt") as file1:
        #     data = file1.readlines()

        # tgt = [d.split("\t")[0] for d in data]
        # src = [d.split("\t")[1] for d in data]

        with open(self.tgt_file) as file1, open(self.src_file) as file2:
            tgt = file1.readlines()
            src = file2.readlines()
        print('total size of data (src, tgt): ', f'({len(src)}, {len(tgt)})')
        tr_src, val_src, tr_tgt, val_tgt = train_test_split(src, tgt, test_size=self.test_size, random_state=self.random_seed, shuffle=True)
        
        self.tr_src = tr_src[:self.tr_max_samples]
        self.tr_tgt = tr_tgt[:self.tr_max_samples]
        self.val_src = val_src[:self.val_max_samples]
        self.val_tgt = val_tgt[:self.val_max_samples]

    def setup(self):
        self.tr_dataset = CustomDataset(self.tr_src, self.tr_tgt)
        self.val_dataset = CustomDataset(self.val_src, self.val_tgt)

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

    def collate_fn(self, features):
        
        raise ValueError("fix tokenizer")

        inputs = [f['input_ids'] for f in features]
        labels = [f['labels'] for f in features]
        
        batch =  self.tokenizer.prepare_seq2seq_batch(
            src_texts=inputs, src_lang=self.src_lang, tgt_lang=self.tgt_lang, tgt_texts=labels,
            max_length=self.max_length, max_target_length=self.max_target_length)
        
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
