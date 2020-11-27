# __author__ = 'Vasudev Gupta'

from dataclasses import dataclass, replace
import torch

from torch_trainer import DefaultArgs


@dataclass
class TrainerConfig(DefaultArgs):

    tgt_file: str = 'data/something'
    src_file: str = 'data/something'
    single_file: bool = False

    max_length: int = 32
    max_target_length: int = 32
    tr_max_samples: int = 100
    val_max_samples: int = 20

    batch_size: int = 16
    lr: float = 1e-4

    base_dir: str = "base_dir"

    test_size: float = .25
    random_seed:  int = 7232114
    num_workers: int = 2
    max_pred_length: int = 40

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