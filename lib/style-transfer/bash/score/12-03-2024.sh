#!/bin/bash
python style_transfer/score.py -m \
gen_dataset=clinical-dream-team/gen-style-transfer/run-i9nuptpk-gen_dataset:v0 \
test_dataset=clinical-dream-team/gen-style-transfer/run-i9nuptpk-test_dataset:v0 \
sft_ratio=0.04 \
gen_ratio=0.7 \
sem_model.train_size=0.30 \
sem_model.epochs=1 \
sem_model.name=sentence-transformers/all-distilroberta-v1 \
sem_model.use_ground_truth=true,false \
sem_model.is_logged=true \
sem_model.loss._target_=sentence_transformers.losses.ContrastiveLoss \
sem_model.is_trainable=false,true \
sem_model.checkpoint=clinical-dream-team/score-style-transfer/run-mru97w7c-sem_model:v0 \
dpo_gen=2
