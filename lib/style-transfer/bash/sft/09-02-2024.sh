#!/bin/bash

accelerate launch --num_cpu_threads_per_process=16 --config_file=accelerate-config.yaml style_transfer/sft.py -m sft_ratio=0.01,0.008,0.006,0.004,0.002 gen_ratio=0.7 training_args.num_train_epochs=50
