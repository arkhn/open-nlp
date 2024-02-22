#!/bin/bash

accelerate launch --num_cpu_threads_per_process=16 --config_file=accelerate-config.yaml style_transfer/dpo.py -m \
training_args.per_device_train_batch_size=1 \
training_args.gradient_accumulation_steps=16 \
training_args.learning_rate=2e-6 \
beta=0.1,0.3 \
threshold=0.6,0.7,0.8 \
dpo_gen=1 \
checkpoint=clinical-dream-team/sft-style-transfer/checkpoint-sft-ratio-0.006_gen-ratio-0.7:v7 \
dataset=clinical-dream-team/score-style-transfer/run-026wi6py-gen_score_dataset:v0 \
sft_ratio=0.006 \
gen_ratio=0.7
