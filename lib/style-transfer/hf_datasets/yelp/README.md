---
dataset_info:
  config_name: yelp_dataset
  features:
    - name: user_id
      dtype: string
    - name: text_id
      list: int32
    - name: keywords
      list: string
    - name: text
      list: string
  splits:
    - name: train
      num_bytes: 124717795
      num_examples: 16000
    - name: validation
      num_bytes: 15294998
      num_examples: 2000
    - name: test
      num_bytes: 15004163
      num_examples: 2000
  download_size: 161804969
  dataset_size: 155016956
---

# Dataset Card for Yelp Style Transfer

## Dataset Description

- **Homepage:** https://github.com/arkhn/ai-lembic
- **Public:** True

## Citation Information

This dataset is a collection of yelp reviews that have been preprocessed to be used for style
transfer.

```
@article{DBLP:journals/corr/Asghar16,
  author       = {Nabiha Asghar},
  title        = {Yelp Dataset Challenge: Review Rating Prediction},
  journal      = {CoRR},
  volume       = {abs/1605.05362},
  year         = {2016},
  url          = {http://arxiv.org/abs/1605.05362},
  eprinttype    = {arXiv},
  eprint       = {1605.05362},
  timestamp    = {Mon, 13 Aug 2018 16:49:17 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/Asghar16.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
