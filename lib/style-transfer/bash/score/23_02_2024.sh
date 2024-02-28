#!/bin/bash

python style_transfer/score.py -m \
gen_dataset=clinical-dream-team/gen-style-transfer/run-fqnnb5dr-gen_dataset:v0 \
test_dataset=clinical-dream-team/gen-style-transfer/run-fqnnb5dr-test_dataset:v0 \
sft_ratio=0.06 \
gen_ratio=0.7 \
sem_model.train_size=0.30 \
sem_model.epochs=1 \
sem_model.name=sentence-transformers/all-distilroberta-v1 \
sem_model.use_ground_truth=true \
sem_model.is_logged=true \
sem_model.loss._target_=sentence_transformers.losses.ContrastiveLoss
