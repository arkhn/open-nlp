# Conflict Detection Baselines Makefile

.PHONY: help install test train example clean

help:  ## Show this help message
	@echo "Conflict Detection Baselines"
	@echo "=========================="
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

train:  ## Train the baseline model
	python train.py

train-debug:  ## Train with debug settings
	python train.py training.epochs=2 training.eval_steps=10 training.logging_steps=5

train-bert:  ## Train with BERT base model
	python train.py model.model_name_or_path=bert-base-uncased

train-biobert:  ## Train with BioBERT model
	python train.py model.model_name_or_path=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext

clean:  ## Clean output files
	rm -rf outputs/
	rm -f example_predictions.json
	rm -f *.log

setup: install  ## Complete setup (install dependencies)

all: setup train  ## Complete pipeline (setup + train)
