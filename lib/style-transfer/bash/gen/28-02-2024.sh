#!/bin/bash
python style_transfer/gen.py checkpoint=clinical-dream-team/dpo-style-transfer/checkpoint-sft-ratio-0.06_gen-ratio-0.7_dpo1:v159 dpo_gen=1 sft_ratio=0.06 gen_ratio=0.7
