#!/bin/bash
# Split the dataset into train, dev, and test sets
mkdir -p post-processed
dataset_path=post-processed/filtered_entries-utf8.jsonl
total_lines=$(wc -l < $dataset_path)
train_end=$((total_lines * 80 / 100))
dev_end=$((train_end + total_lines * 10 / 100))
sed -n "1,${train_end}p" $dataset_path > post-processed/train.jsonl
sed -n "$((train_end + 1)),$((dev_end))p" $dataset_path > post-processed/dev.jsonl
sed -n "$((dev_end + 1)),$((total_lines))p" $dataset_path > post-processed/test.jsonl
