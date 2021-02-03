# Incorporating BERT into Parallel Sequence Decoding with Adapters

Unofficial implementation of [paper](https://arxiv.org/abs/2010.06138).

## Setting Up

**Setting up dataset and training scripts**

```shell
# download & prepare IWSLT14 dataset 
cd data/iwslt14
bash prepare-iwslt14.sh

# start training
python train.py --training_id "iwslt14_de_en"
```

**Using trained model directly**

```python
from modeling import TransformerMaskPredict

# lets load model-weights from huggingface hub
model_id = "vasudevgupta/abnet-iwslt14-de-en"
model = TransformerMaskPredict.from_pretrained(model_id)
```

## Note:

- `prepare-iwslt14.sh` is directly taken from fairseq (as per mentioned in paper)
- `modeling/multihead_attention.py`, `modeling/modeling_bert.py` are taken from huggingface repositary; but small changes are introduced for adding adapters.
- Rest of all scripts are completely written by us.
- We are hosting our weights in `huggingface_hub` to ease process of loading model.

## Submitted by

**Vasudev Gupta (ME18B182) & Rishabh Shah (ME18B029)**