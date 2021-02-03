import yaml
import wandb

from data_utils import DataLoader, Tokenizer
from utils import fetch_translations_and_bleu, get_args
from modeling import TransformerMaskPredict

PROJECT_NAME = "parallel-decoder-paper"


if __name__ == "__main__":

    args = get_args()
    wandb.init(project=PROJECT_NAME, config=args)

    model = TransformerMaskPredict.from_pretrained(args.model_id)
    tokenizer = Tokenizer(model.config.encoder_id, model.config.decoder_id, model.config.length_token)

    dl = DataLoader(args, tokenizer)
    tr_dataset, val_dataset, tst_dataset = dl()

    data, columns = dl.build_seqlen_table()
    wandb.log({'Sequence-Lengths': wandb.Table(data=data, columns=columns)})

    for mode, dataset in zip(["tr", "val", "tst"], [tr_dataset, val_dataset, tst_dataset]):
        out = fetch_translations_and_bleu(model, dataset, tokenizer, args.iterations, args.B, num_samples=args.bleu_num_samples)
        data = list(zip(out["src"], out["tgt"], out["pred"]))
        wandb.log({
            mode+'_bleu': out["bleu"],
            mode+'_predictions': wandb.Table(data=data, columns=['src', 'tgt', 'tgt_pred'])
        })
