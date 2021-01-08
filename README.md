# Incorporating BERT into Parallel Sequence Decoding with Adapters

Unofficial implementation of [paper](https://arxiv.org/abs/2010.06138).

## Setting Up

```python
# cloning github repo
git clone https://github.com/vasudevgupta7/cs6910-project.git
cd cs6910-project
```

**Setting up dataset and training scripts**

```python
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

# you can directly run existing script for inference
sh scripts/infer_iwslt14_de_en.sh
```

**Feel free to raise an issue incase you find some problem in this implementation.**
