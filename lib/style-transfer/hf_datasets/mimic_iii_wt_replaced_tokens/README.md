---
dataset_info:
  - config_name: default
    features:
      - name: text_id
        list: int32
      - name: keywords
        list: string
      - name: text
        list: string
    splits:
      - name: train
        num_bytes: 14199719
        num_examples: 5605
      - name: validation
        num_bytes: 1786643
        num_examples: 714
      - name: test
        num_bytes: 1694637
        num_examples: 665
    download_size: 9064900
    dataset_size: 17680999
  - config_name: mimic-iii-gpt4o-tokens
    features:
      - name: text_id
        list: int32
      - name: keywords
        list: string
      - name: text
        list: string
    splits:
      - name: train
        num_bytes: 14199719
        num_examples: 5605
      - name: validation
        num_bytes: 1786643
        num_examples: 714
      - name: test
        num_bytes: 1694637
        num_examples: 665
    download_size: 9064900
    dataset_size: 17680999
---
