#!/bin/bash

python _create_dataset.py \
  clinical-dream-team/score-style-transfer/run-ofzh3aqu-gen_score_dataset:v0 \
  clinical-dream-team/score-style-transfer/run-cnipr0cb-gen_score_dataset:v0 \
  clinical-dream-team/score-style-transfer/run-3ahhn8iu-gen_score_dataset:v0 \
  clinical-dream-team/score-style-transfer/run-mru97w7c-gen_score_dataset:v0 \
  clinical-dream-team/score-style-transfer/run-mh67il8t-gen_score_dataset:v0 \
  clinical-dream-team/score-style-transfer/run-b52npjc2-gen_score_dataset:v0 \
  clinical-dream-team/score-style-transfer/run-apjlkkyz-gen_score_dataset:v0
