# __author__ = 'Vasudev Gupta'

from dataclasses import dataclass, replace, field
import torch
from adapters import AdapterConfig

from torch_trainer import DefaultArgs

@dataclass
class TrainerConfig(DefaultArgs):

    tgt_file: str = 'data/something'
    src_file: str = 'data/something'
    single_file: bool = False

    enc_bert_id: str = "bert-base-uncased"
    dec_bert_id: str = "bert-base-uncased"

    src_lang: str = 'hi_IN'
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

    tgt_lang: str = 'en_XX'

    # control adapter from here
    # manually switch off layers in case you want to freeze
    load_adapter_path: str = None
    save_adapter_path: str = None
    enc_ffn_adapter: bool = True
    dec_ffn_adapter: bool = True
    cross_attn_adapter: bool = True

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

    # adapter-inside config
    enc_ffn_adapter_config: FfnAdapterConfig = field(repr=False, default=FfnAdapterConfig)
    dec_ffn_adapter_config: FfnAdapterConfig = field(repr=False, default=FfnAdapterConfig)
    cross_attn_adapter_config: CrossAttnAdapterConfig = field(repr=False, default=CrossAttnAdapterConfig)

main = TrainerConfig()