# __author__ = 'Vasudev Gupta'
import torch
from tqdm import tqdm

@torch.no_grad()
def predictor(model, tokenizer, lists_src, lists_tgt, pred_max_length, src_lang='hi_IN', device=torch.device("cuda")):
    pred = []
    val_data = []
    tgt = []

    model.to(device)
    model.eval()

    raise ValueError("fix tokenizer")  
    bar = tqdm(zip(lists_src, lists_tgt), desc="predicting ... ", leave=False)
    for s, t in bar:
         batch =  tokenizer.prepare_seq2seq_batch(src_texts=s, src_lang=src_lang)

         for k in batch:
            batch[k] = torch.tensor(batch[k])
            batch[k] = batch[k].to(device)

        raise ValueError("fix .generate method")

        out = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"], max_length=pred_max_length)
        translation = tokenizer.batch_decode(out, skip_special_tokens=True)
         
        val = list(zip(s, t, translation))
        val_data.extend(val)

        pred.extend(translation)
        tgt.extend(t)

    return val_data, pred, tgt
