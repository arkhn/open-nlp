#!/bin/bash
python style_transfer/gen.py checkpoint=clinical-dream-team/dpo-style-transfer/checkpoint-rl7g2p97:v83 dpo_gen=1 sft_ratio=0.006 gen_ratio=0.7
