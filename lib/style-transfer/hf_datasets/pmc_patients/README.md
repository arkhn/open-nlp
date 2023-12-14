---
dataset_info:
  config_name: pmc_dataset
  features:
    - name: user_id
      dtype: string
    - name: text_id
      list: int32
    - name: title
      list: string
    - name: keywords
      list: string
    - name: text
      list: string
  splits:
    - name: train
      num_bytes: 13766458
      num_examples: 2684
    - name: validation
      num_bytes: 1654623
      num_examples: 335
    - name: test
      num_bytes: 1702935
      num_examples: 336
  download_size: 17834606
  dataset_size: 17124016
---

# Dataset Card for PMC Patients Style Transfer

## Dataset Description

- **Homepage:** https://github.com/arkhn/ai-lembic
- **Public:** True

## Citation Information

This dataset is a collection of clinical cases that have been preprocessed to be used for style
transfer.

```
@misc{zhao2023pmcpatients,
      title={PMC-Patients: A Large-scale Dataset of Patient Summaries and Relations for Benchmarking Retrieval-based Clinical Decision Support Systems},
      author={Zhengyun Zhao and Qiao Jin and Fangyuan Chen and Tuorui Peng and Sheng Yu},
      year={2023},
      eprint={2202.13876},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
