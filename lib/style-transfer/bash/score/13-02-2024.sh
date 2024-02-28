#!/bin/bash

python style_transfer/score.py \
  gen_dataset=clinical-dream-team/gen-style-transfer/run-byeksy0m-gen_dataset:v0 \
  test_dataset=clinical-dream-team/gen-style-transfer/run-byeksy0m-test_dataset:v0 \
  sft_ratio=0.004 \
  gen_ratio=0.7

python style_transfer/score.py \
  gen_dataset=clinical-dream-team/gen-style-transfer/run-2vkesygs-gen_dataset:v0 \
  test_dataset=clinical-dream-team/gen-style-transfer/run-2vkesygs-test_dataset:v0 \
  sft_ratio=0.006 \
  gen_ratio=0.7

python style_transfer/score.py \
gen_dataset=clinical-dream-team/gen-style-transfer/run-ka1qr4r7-gen_dataset:v0 \
test_dataset=clinical-dream-team/gen-style-transfer/run-ka1qr4r7-test_dataset:v0 \
sft_ratio=0.01 \
gen_ratio=0.7
