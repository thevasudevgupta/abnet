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
    
    tr_max_samples: int = 20000
    val_max_samples: int = -1
    tst_max_samples: int = -1
    batch_size: int = 16
    accumulation_steps: int = 1
    lr: float = 0.001
    max_epochs: int = 5

    iterations: int = 10
    B: int = 4

    save_training_state: str = False # or True
    load_training_state: str = False # or True
    load_pretrained_path: str = None
    save_pretrained_path: str = None

    bleu_num_samples: int = 2000
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
                load_pretrained_path=None,
                bleu_num_samples=6000)

wmt16_ro_en = Config(tr_src_file="data/wmt16_ro_en/europarl-v8.ro-en.ro",
                tr_tgt_file="data/wmt16_ro_en/europarl-v8.ro-en.en",
                val_src_file="data/wmt16_ro_en/newsdev2016-roen-src.ro.sgm",
                val_tgt_file="data/wmt16_ro_en/newsdev2016-roen-ref.en.sgm",
                tst_src_file="data/wmt16_ro_en/newstest2016-enro-ref.ro.sgm",
                tst_tgt_file="data/wmt16_ro_en/newstest2016-enro-src.en.sgm",
                max_length=56,
                max_target_length=40,
                wandb_run_name="wmt16-ro-en",
                base_dir="wmt16-ro-en",
                save_pretrained_path="abnet-wmt16-ro-en",
                load_pretrained_path=None,
                bleu_num_samples=2200)
