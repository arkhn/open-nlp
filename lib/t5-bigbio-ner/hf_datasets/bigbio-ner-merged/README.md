# BIGBIO-NER

This repository contains the codebase to create a dataset by merging (almost) all NER datasets in the BIGBIO collection. The resulting dataset is formatted in IOB2 and the InstructionNER format.

Default config uses _xx_ent_wiki_sm_ as a multilingual tokenizer and uploads the dataset to HuggingFace as _bio-datasets/bigbio-ner-merged_.

## Usage

```bash
python -m spacy download xx_ent_wiki_sm
python create_and_upload.py
```
