#!/bin/bash

python style_transfer/gen.py checkpoint=clinical-dream-team/sft-style-transfer/checkpoint-sft-ratio-0.99_gen-ratio-0.7:v281 sft_ratio=0.99 gen_ratio=0.7 &&
python style_transfer/gen.py checkpoint=clinical-dream-team/sft-style-transfer/checkpoint-sft-ratio-0.02_gen-ratio-0.7:v11 sft_ratio=0.02 gen_ratio=0.7
