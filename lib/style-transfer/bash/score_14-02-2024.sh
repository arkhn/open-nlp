#!/bin/bash

python style_transfer/score.py \
gen_dataset=clinical-dream-team/gen-style-transfer/run-qmopfb2e-gen_dataset:v0 \
test_dataset=clinical-dream-team/gen-style-transfer/run-qmopfb2e-test_dataset:v0 \
sft_ratio=0.006 \
gen_ratio=0.7
