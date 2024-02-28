#!/bin/bash

accelerate launch --num_cpu_threads_per_process=16 --config_file=accelerate-config.yaml style_transfer/dpo.py \
training_args.per_device_train_batch_size=1 \
training_args.gradient_accumulation_steps=16 \
training_args.learning_rate=2e-6 \
beta=0.1 \
dpo_gen=2 \
checkpoint=clinical-dream-team/dpo-style-transfer/checkpoint-zy4fln6j:v37 \
dataset=clinical-dream-team/score-style-transfer/run-sbgme69h-gen_score_dataset:v0 \
sft_ratio=0.06 \
gen_ratio=0.7
