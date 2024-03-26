#!/bin/bash

accelerate launch --num_cpu_threads_per_process=16 --config_file=accelerate-config.yaml style_transfer/dpo.py \
training_args.per_device_train_batch_size=1 \
training_args.gradient_accumulation_steps=16 \
training_args.learning_rate=4e-6 \
training_args.num_train_epochs=80 \
beta=0.3 \
dpo_gen=2 \
checkpoint=clinical-dream-team/dpo-style-transfer/checkpoint-sft-ratio-0.04_gen-ratio-0.7_dpo1:v127 \
dataset=clinical-dream-team/score-style-transfer/run-mh67il8t-gen_score_dataset:v0 \
sft_ratio=0.04 \
gen_ratio=0.7 \
percentile=80
