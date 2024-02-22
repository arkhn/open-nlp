---
dataset_info:
  config_name: mimicoracle
  features:
    - name: subject_id
      dtype: string
    - name: document_id
      dtype: string
    - name: chartdate
      dtype: string
    - name: text
      dtype: string
    - name: section_title
      dtype: string
    - name: section_content
      dtype: string
    - name: section_start
      dtype: int32
    - name: section_end
      dtype: int32
  splits:
    - name: train
      num_bytes: 191330120
      num_examples: 15036
    - name: test
      num_bytes: 52895350
      num_examples: 4129
  download_size: 244108855
  dataset_size: 244225470
---
