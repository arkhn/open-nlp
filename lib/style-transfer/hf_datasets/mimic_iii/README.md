---
dataset_info:
  config_name: mimic_iii_dataset
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
      num_bytes: 4104183
      num_examples: 1020
    - name: validation
      num_bytes: 534543
      num_examples: 127
    - name: test
      num_bytes: 508129
      num_examples: 129
  download_size: 5292469
  dataset_size: 5146855
---

# Dataset Card for Mimic III Style Transfer

## Dataset Description

- **Homepage:** https://github.com/arkhn/ai-lembic
- **Public:** True

## Citation Information

This dataset is a collection of clinical reports that have been preprocessed
to be used for style transfer.

```
@inproceedings{10.1145/3368555.3384469,
author = {Wang, Shirly and McDermott, Matthew B. A. and Chauhan, Geeticka and Ghassemi, Marzyeh and Hughes, Michael C. and Naumann, Tristan},
title = {MIMIC-Extract: A Data Extraction, Preprocessing, and Representation Pipeline for MIMIC-III},
year = {2020},
isbn = {9781450370462},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3368555.3384469},
doi = {10.1145/3368555.3384469},
abstract = {Machine learning for healthcare researchers face challenges to progress and reproducibility due to a lack of standardized processing frameworks for public datasets. We present MIMIC-Extract, an open source pipeline for transforming the raw electronic health record (EHR) data of critical care patients from the publicly-available MIMIC-III database into data structures that are directly usable in common time-series prediction pipelines. MIMIC-Extract addresses three challenges in making complex EHR data accessible to the broader machine learning community. First, MIMIC-Extract transforms raw vital sign and laboratory measurements into usable hourly time series, performing essential steps such as unit conversion, outlier handling, and aggregation of semantically similar features to reduce missingness and improve robustness. Second, MIMIC-Extract extracts and makes prediction of clinically-relevant targets possible, including outcomes such as mortality and length-of-stay as well as comprehensive hourly intervention signals for ventilators, vasopressors, and fluid therapies. Finally, the pipeline emphasizes reproducibility and extensibility to future research questions. We demonstrate the pipeline's effectiveness by developing several benchmark tasks for outcome and intervention forecasting and assessing the performance of competitive models.},
booktitle = {Proceedings of the ACM Conference on Health, Inference, and Learning},
pages = {222–235},
numpages = {14},
keywords = {Machine learning, MIMIC-III, Healthcare, Time series data, Reproducibility},
location = {Toronto, Ontario, Canada},
series = {CHIL '20}
}
```
