#!/bin/bash

python style_transfer/score.py -m \
gen_dataset=clinical-dream-team/gen-style-transfer/run-o8bvb9xv-gen_dataset:v0 \
test_dataset=clinical-dream-team/gen-style-transfer/run-o8bvb9xv-test_dataset:v0 \
sft_ratio=0.06 \
gen_ratio=0.7 \
sem_model.train_size=0.30 \
sem_model.epochs=1 \
sem_model.name=sentence-transformers/all-distilroberta-v1 \
sem_model.use_ground_truth=true,false \
sem_model.is_logged=true \
sem_model.loss._target_=sentence_transformers.losses.ContrastiveLoss \
sem_model.is_trainable=false,true \
sem_model.checkpoint=clinical-dream-team/score-style-transfer/run-ehtjzcb9-sem_model:v0 \
dpo_gen=2

python style_transfer/score.py -m \
gen_dataset=clinical-dream-team/gen-style-transfer/run-onrfz5tw-gen_dataset:v0 \
test_dataset=clinical-dream-team/gen-style-transfer/run-onrfz5tw-test_dataset:v0 \
sft_ratio=0.04 \
gen_ratio=0.7 \
sem_model.train_size=0.30 \
sem_model.epochs=1 \
sem_model.name=sentence-transformers/all-distilroberta-v1 \
sem_model.use_ground_truth=true,false \
sem_model.is_logged=true \
sem_model.loss._target_=sentence_transformers.losses.ContrastiveLoss \
sem_model.is_trainable=false,true \
dpo_gen=0
