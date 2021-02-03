# __author__ = 'Vasudev Gupta'

from dataclasses import dataclass, replace


@dataclass
class Config:

    tr_tgt_file: str
    tr_src_file: str

    val_tgt_file: str
    val_src_file: str

    tst_tgt_file: str
    tst_src_file: str

    max_length: int
    max_target_length: int

    tr_max_samples: int = -1
    val_max_samples: int = -1
    tst_max_samples: int = -1
    bleu_num_samples: int = 6000
    batch_size: int = 16
    accumulation_steps: int = 1
    lr: float = 7e-4
    max_epochs: int = 10

    iterations: int = 10
    B: int = 4

    save_training_state: str = False # or True
    load_training_state: str = False # or True
    load_pretrained_path: str = None
    save_pretrained_path: str = None

    base_dir: str = "base_dir"
    num_workers: int = 4
    # save_epoch_dir: str = None
    project_name: str = 'parallel-decoder-paper'
    wandb_run_name: str = None

iwslt14_de_en = Config(tr_src_file="data/iwslt14/iwslt14.tokenized.de-en/train.de",
                tr_tgt_file="data/iwslt14/iwslt14.tokenized.de-en/train.en",
                val_src_file="data/iwslt14/iwslt14.tokenized.de-en/valid.de",
                val_tgt_file="data/iwslt14/iwslt14.tokenized.de-en/valid.en",
                tst_src_file="data/iwslt14/iwslt14.tokenized.de-en/test.de",
                tst_tgt_file="data/iwslt14/iwslt14.tokenized.de-en/test.en",
                max_length=48,
                max_target_length=48,
                wandb_run_name="iwslt14-de-en",
                base_dir="iwslt14-de-en",
                save_pretrained_path="abnet-iwslt14-de-en",
                load_pretrained_path=None)
