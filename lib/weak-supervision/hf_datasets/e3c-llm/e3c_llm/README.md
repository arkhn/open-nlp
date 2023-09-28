---
dataset_info:
  features:
    - name: text
      dtype: string
    - name: tokens_offsets
      sequence:
        sequence: int32
    - name: clinical_entity_tags
      sequence:
        class_label:
          names:
            "0": O
            "1": B-CLINENTITY
            "2": I-CLINENTITY
  config_name: e3c-llm
  splits:
    - name: en_layer1
      num_bytes: 768555
      num_examples: 1520
    - name: en_layer2_validation
      num_bytes: 175089
      num_examples: 334
    - name: fr_layer1
      num_bytes: 758368
      num_examples: 1109
    - name: eu_layer2
      num_bytes: 503182
      num_examples: 1594
    - name: eu_layer2_validation
      num_bytes: 131870
      num_examples: 468
    - name: it_layer2
      num_bytes: 1590730
      num_examples: 2436
    - name: es_layer2_validation
      num_bytes: 166201
      num_examples: 261
    - name: fr_layer2_validation
      num_bytes: 170233
      num_examples: 293
    - name: es_layer2
      num_bytes: 1506040
      num_examples: 2347
    - name: en_layer2
      num_bytes: 1539228
      num_examples: 2873
    - name: fr_layer2
      num_bytes: 1583560
      num_examples: 2389
    - name: eu_layer1
      num_bytes: 910983
      num_examples: 3126
    - name: it_layer1
      num_bytes: 768769
      num_examples: 1145
    - name: es_layer1
      num_bytes: 754628
      num_examples: 1134
    - name: it_layer2_validation
      num_bytes: 172651
      num_examples: 275
  download_size: 0
  dataset_size: 11500087
---

# Dataset Card for E3C

## Dataset Description

- **Public:** True
- **Tasks:** NER

This dataset is an annotated corpus of clinical texts from E3C using Large Language Models (LLM).
