.PHONY: help install clean run-zero run-one run-few run-all

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package and dependencies
	pip install -e .

run-zero: ## Run zero-shot baseline
	python run_baseline.py --config-name=zero_shot

run-one: ## Run one-shot baseline
	python run_baseline.py --config-name=one_shot

run-few: ## Run few-shot baseline
	python run_baseline.py --config-name=few_shot

run-qwen: ## Run with Qwen model
	python run_baseline.py --config-name=qwen_models

run-all: run-zero run-one run-few ## Run all baselines

clean: ## Clean output files
	rm -rf outputs/
	rm -rf wandb/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

setup-env: ## Setup environment variables (requires manual API key)
	@echo "Please set your Groq API key:"
	@echo "export GROQ_API_KEY='your_api_key_here'"
	@echo "Then run: make install"
