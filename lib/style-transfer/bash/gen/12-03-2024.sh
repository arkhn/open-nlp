#!/bin/bash
python style_transfer/gen.py checkpoint=clinical-dream-team/dpo-style-transfer/checkpoint-sft-ratio-0.04_gen-ratio-0.7_dpo3:v134 dpo_gen=3 sft_ratio=0.04 gen_ratio=0.7
