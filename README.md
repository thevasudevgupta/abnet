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
python main.py
```

**Using trained model directly**

```python
# Using model for inference
import config
from modeling import TransformerMaskPredict

transformer_config = config.model_iwslt14
model = TransformerMaskPredict(transformer_config)
# lets load model weights from huggingface hub
model.from_pretrained("vasudevgupta/abnet-iwslt14-de-en")
```

**Feel free to raise an issue incase you find some problem in this implementation.**
