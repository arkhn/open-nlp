#!/bin/bash

accelerate_sft="accelerate launch --num_cpu_threads_per_process=16 --config_file=accelerate-config.yaml style_transfer/sft.py"

$accelerate_sft sft_ratio=0.03 gen_ratio=0.7 training_args.num_train_epochs=40
