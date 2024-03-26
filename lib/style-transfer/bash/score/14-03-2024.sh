#!/bin/bash
#python style_transfer/score.py -m \
#gen_dataset=clinical-dream-team/gen-style-transfer/run-n23e6736-gen_dataset:v0 \
#test_dataset=clinical-dream-team/gen-style-transfer/run-n23e6736-test_dataset:v0 \
#sft_ratio=0.04 \
#gen_ratio=0.7 \
#sem_model.name=sentence-transformers/all-distilroberta-v1 \
#sem_model.use_ground_truth=false \
#sem_model.is_logged=false \
#sem_model.is_trainable=false \
#sem_model.checkpoint=clinical-dream-team/score-style-transfer/run-ken3c452-sem_model:v0 \
#dpo_gen=1

python style_transfer/score.py -m \
gen_dataset=clinical-dream-team/gen-style-transfer/run-i9nuptpk-gen_dataset:v0 \
test_dataset=clinical-dream-team/gen-style-transfer/run-i9nuptpk-test_dataset:v0 \
sft_ratio=0.04 \
gen_ratio=0.7 \
sem_model.name=sentence-transformers/all-distilroberta-v1 \
sem_model.use_ground_truth=false \
sem_model.is_logged=false \
sem_model.is_trainable=false \
sem_model.checkpoint=clinical-dream-team/score-style-transfer/run-ken3c452-sem_model:v0 \
dpo_gen=2

#python style_transfer/score.py -m \
#gen_dataset=clinical-dream-team/gen-style-transfer/run-b52npjc2-gen_dataset:v0 \
#test_dataset=clinical-dream-team/gen-style-transfer/run-b52npjc2-test_dataset:v0 \
#sft_ratio=0.06 \
#gen_ratio=0.7 \
#sem_model.name=sentence-transformers/all-distilroberta-v1 \
#sem_model.use_ground_truth=false \
#sem_model.is_logged=false \
#sem_model.is_trainable=false \
#sem_model.checkpoint=clinical-dream-team/score-style-transfer/run-mru97w7c-sem_model:v0 \
#dpo_gen=0
#
#python style_transfer/score.py -m \
#gen_dataset=clinical-dream-team/gen-style-transfer/run-0u9i94v0-gen_dataset:v0 \
#test_dataset=clinical-dream-team/gen-style-transfer/run-0u9i94v0-test_dataset:v0 \
#sft_ratio=0.06 \
#gen_ratio=0.7 \
#sem_model.name=sentence-transformers/all-distilroberta-v1 \
#sem_model.use_ground_truth=false \
#sem_model.is_logged=false \
#sem_model.is_trainable=false \
#sem_model.checkpoint=clinical-dream-team/score-style-transfer/run-mru97w7c-sem_model:v0 \
#dpo_gen=1
#
#python style_transfer/score.py -m \
#gen_dataset=clinical-dream-team/gen-style-transfer/run-o8bvb9xv-gen_dataset:v0 \
#test_dataset=clinical-dream-team/gen-style-transfer/run-o8bvb9xv-test_dataset:v0 \
#sft_ratio=0.06 \
#gen_ratio=0.7 \
#sem_model.name=sentence-transformers/all-distilroberta-v1 \
#sem_model.use_ground_truth=false \
#sem_model.is_logged=false \
#sem_model.is_trainable=false \
#sem_model.checkpoint=clinical-dream-team/score-style-transfer/run-mru97w7c-sem_model:v0 \
#dpo_gen=2
