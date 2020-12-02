# Incorporating BERT into Parallel Sequence Decoding with Adapters

This is my implementation of this [paper](https://arxiv.org/abs/2010.06138).

## Setting Up

```python
# preparing data 
cd data/iwslt14
bash prepare-iwslt14.sh

```


## TODO
- [ ] Take top B length prediction and select one with best prob (B = 4)
- [ ] upper bound of iterative decoding is 10
- [ ] how we will be training embedding for LENGTH token?
- [ ] tokens may repeat from 1st round to 2nd round, when 2nd round is done and moving to 3rd round of masking.

**Feel free to raise an issue incase you find any problems in this implementation.**
