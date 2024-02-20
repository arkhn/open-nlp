#!/bin/bash

python style_transfer/gen.py checkpoint=clinical-dream-team/sft-style-transfer/checkpoint-sft-ratio-0.004_gen-ratio-0.7:v5 sft_ratio=0.004 gen_ratio=0.7 &&
python style_transfer/gen.py checkpoint=clinical-dream-team/sft-style-transfer/checkpoint-sft-ratio-0.006_gen-ratio-0.7:v7 sft_ratio=0.006 gen_ratio=0.7
python style_transfer/gen.py checkpoint=clinical-dream-team/sft-style-transfer/checkpoint-sft-ratio-0.01_gen-ratio-0.7:v7 sft_ratio=0.01 gen_ratio=0.7 &&
python style_transfer/gen.py checkpoint=clinical-dream-team/sft-style-transfer/checkpoint-sft-ratio-0.99_gen-ratio-0.7:v281 sft_ratio=0.99 gen_ratio=0.7
