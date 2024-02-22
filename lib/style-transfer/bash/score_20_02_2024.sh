#!/bin/bash

python style_transfer/score.py -m \
gen_dataset=clinical-dream-team/gen-style-transfer/run-qmopfb2e-gen_dataset:v0 \
test_dataset=clinical-dream-team/gen-style-transfer/run-qmopfb2e-test_dataset:v0 \
sft_ratio=0.006 \
gen_ratio=0.7 \
sem_model.train_size=0.15,0.25,0.30 \
sem_model.epochs=1,3 \
sem_model.name=sentence-transformers/all-mpnet-base-v2,sentence-transformers/all-distilroberta-v1 \
sem_model.use_ground_truth=false,true \
sem_model.loss._target_=sentence_transformers.losses.OnlineContrastiveLoss
