#!/bin/bash

accelerate_sft="accelerate launch --num_cpu_threads_per_process=16 --config_file=accelerate-config.yaml style_transfer/sft.py"

$accelerate_sft sft_ratio=0.06 gen_ratio=0.7 training_args.num_train_epochs=20
$accelerate_sft sft_ratio=0.04 gen_ratio=0.7 training_args.num_train_epochs=20
$accelerate_sft sft_ratio=0.02 gen_ratio=0.7 training_args.num_train_epochs=20
