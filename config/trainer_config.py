# __author__ = 'Vasudev Gupta'

from dataclasses import dataclass, replace
import torch

from torch_trainer import DefaultArgs


@dataclass
class TrainerConfig(DefaultArgs):

    tr_tgt_file: str = 'data/something'
    tr_src_file: str = 'data/something'

    val_tgt_file: str = ""
    val_src_file: str = ""

    tst_tgt_file: str = ""
    tst_src_file: str = ""

    max_length: int = 32
    max_target_length: int = 32
    
    tr_max_samples: int = -1
    val_max_samples: int = -1
    tst_max_samples: int = -1

    batch_size: int = 16
    lr: float = 1e-4

    base_dir: str = "base_dir"
    num_workers: int = 2

    # control adapter from here
    # manually switch off layers in case you want to freeze
    load_adapter_path: str = None
    save_adapter_path: str = None

    # args used in torch_trainer
    max_epochs: int = 3
    accumulation_steps: int = 1
    save_epoch_dir: str = None
    early_stop_n: int = None
    map_location: torch.device = torch.device("cuda:0")
    save_dir: str = None
    load_dir: str = None
    tpus: int = 0
    precision: str = 'float32'
    fast_dev_run: bool = False

    # all these args will be invalid if you run sweep
    project_name: str = 'parallel-decoder-paper'
    wandb_run_name: str = None
    wandb_off: bool = False
    wandb_resume: bool = False
    wandb_run_id: str = None

main = TrainerConfig()

IWSLT14 = TrainerConfig(tr_src_file="data/iwslt14/iwslt14.tokenized.de-en/train.de",
                    tr_tgt_file="iwslt14/iwslt14.tokenized.de-en/train.en",
                    val_src_file="data/iwslt14/iwslt14.tokenized.de-en/valid.de",
                    val_tgt_file="iwslt14/iwslt14.tokenized.de-en/valid.en",
                    tst_src_file="data/iwslt14/iwslt14.tokenized.de-en/test.de",
                    tst_tgt_file="iwslt14/iwslt14.tokenized.de-en/test.en",
                    wandb_run_name="iwslt14-157K,7K",
                    base_dir="iwslt14-157K,7K",
                    tr_max_samples=-1,
                    val_max_samples=-1,
                    tst_max_samples=-1)
