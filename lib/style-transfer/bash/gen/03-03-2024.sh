#!/bin/bash
python style_transfer/gen.py checkpoint=clinical-dream-team/sft-style-transfer/checkpoint-sft-ratio-0.04_gen-ratio-0.7:v13 dpo_gen=0 sft_ratio=0.04 gen_ratio=0.7
