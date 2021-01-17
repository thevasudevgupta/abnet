__author__ = "Vasudev Gupta"

from modeling.transformer_maskpredict import TransformerMaskPredict
from modeling.utils import Dict

"""
bert code is taken from HuggingFace directly (below files)
  - multihead_attention.py
  - modeling_bert.py

Small changes are added in above files to add adapters in BERT

Below files are completely written by us
  - adapters.py
  - transformer_maskpredict.py
  - utils.py
  - decoding.py
"""

# [k for k in s["model"].keys() if not(k.startswith('decoder.bert.encoder.layer') or k.startswith("encoder.encoder.layer"))]

# TODO : remove commented
# switched from postlayer norm to prelayer norm in encoder-adapter
# length embedding initialized properly
# added extra dense-layer, layer-norm before shared-embed in lm head
# added bias in shared-embed in lm head
# 