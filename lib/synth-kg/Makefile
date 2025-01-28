ADAPTER_PATHS := /lora/adapter1 /lora/adapter2
DATASET_PATH := datasets/health/model=llama-3.3-70b-versatile_t=0.7_size=1500/private.parquet
DOMAIN := health
ENV_FILE := .env
EVALUATOR_PATH := sentence-transformers-paraphrase-MiniLM-L6-v2
FINANCE_EVAL_OUTPUT_PATH := datasets/finance/eval/model_outputs/model=$(MODEL_ID)_size=$(SIZE)_step=$(STEP)
HEALTH_EVAL_OUTPUT_PATH := datasets/health/eval/model_outputs/model=$(MODEL_ID)_size=$(SIZE)_step=$(STEP)
EVAL_SFT_PATH := datasets/$(DOMAIN)/model=$(MODEL_ID)_size=$(SIZE)_step=$(STEP)/model=$(EVALUATOR_PATH)_eval.parquet
HYDRA_CONFIG := "test"
MODEL_ID := mrx4y3h2
N := 4
OUTPUT_PATH := datasets/$(DOMAIN)/model=$(MODEL_ID)_size=$(SIZE)_step=$(STEP)
PRIVATE_PATH := datasets/health/model=llama-3.3-70b-versatile_t=0.7_size=1500/private.parquet
RESULTS := /path/to/results/folder
SIZE := 1500
STEP := sft
WB_RUN_ID := 1234567890

.PHONY: sft \
        generation \
        merge-fingpt \
        finance-preprocessing \
        health-preprocessing \
        score \
        alignment \
        health-evaluation \
        finance-evaluation \
        log-results

# Supervised Fine-Tuning
sft:
		sky launch -c sft -y --env-file .env --env HYDRA_CONFIG=$(HYDRA_CONFIG) launch/sft.yaml
		sky down -y sft

# Generate synthetic data using a specific model
generation:
		# Launch cluster for data generation
		sky launch -c generation -y \
			--env DATASET=$(DATASET_PATH)/private.parquet \
			--env ADAPTERS_PATHS=$(ADAPTERS_PATHS) \
			--env OUTPUT_PATH=$(OUTPUT_PATH) \
			--env-file $(ENV_FILE) \
			launch/generation.yaml

		# Prepare local output directory
		mkdir -p $(OUTPUT_PATH)

		# Download generated data from cluster
		scp generation:~/sky_workdir/$(OUTPUT_PATH)/public_generated.parquet $(OUTPUT_PATH)/public_generated.parquet

		# Cleanup cluster
		sky down -y generation

# Merge FinGPT data
merge-fingpt:
		sky launch -c merge-fingpt -y launch/merge-fingpt.yaml
		sky down -y merge-fingpt

# Data preprocessing tasks
finance-preprocessing:
		python datasets/preprocessing/finance/run.py

health-preprocessing:
		python datasets/preprocessing/health/run.py

# Scoring Step
score:
		python training_steps/score/run.py \
			--evaluator_path $(EVALUATOR_PATH) \
			--public_dataset $(OUTPUT_PATH)/public_generated.parquet \
			--private_dataset $(PRIVATE_PATH)/private.parquet \
			--output_path $(OUTPUT_PATH) \
			--n $(N)

# Alignment Step
alignment:
		sky launch -c dpo -y --env-file $(ENV_FILE) --env HYDRA_CONFIG=$(HYDRA_CONFIG) launch/dpo.yaml
		sky down -y dpo

# Health domain evaluation
health-evaluation:
		## Launch evaluation cluster
		sky launch -c health-evaluation -y \
			--env-file $(ENV_FILE) \
			--env HYDRA_CONFIG=$(HYDRA_CONFIG) \
			--env DATASET=$(EVAL_SFT_PATH) \
			--env OUTPUT_PATH=$(HEALTH_EVAL_OUTPUT_PATH) \
		launch/health-evaluation.yaml

		## Download evaluation results
		mkdir -p $(HEALTH_EVAL_OUTPUT_PATH)
		scp health-evaluation:~/sky_workdir/$(HEALTH_EVAL_OUTPUT_PATH)/evaluation.parquet $(HEALTH_EVAL_OUTPUT_PATH)/evaluation.parquet
		scp health-evaluation:~/sky_workdir/$(HEALTH_EVAL_OUTPUT_PATH)/evaluation_alpacare_sft.parquet $(HEALTH_EVAL_OUTPUT_PATH)/evaluation_alpacare_sft.parquet

		## Cleanup and run evaluation script
		sky down -y health-evaluation
		./eval/health/evaluation.sh $(MODEL_ID) $(HEALTH_EVAL_OUTPUT_PATH)/evaluation.parquet

# Finance domain evaluation
finance-evaluation:
		python eval/finance/join_datasets.py --evaluation $(EVAL_SFT_PATH) --private_seed $(PRIVATE_PATH) --output_path $(FINANCE_EVAL_OUTPUT_PATH)
		cd eval/finance/data && python download.py
		sky launch -c finance-evaluation -y \
			--env-file $(ENV_FILE) \
			--env HYDRA_CONFIG=$(HYDRA_CONFIG) \
			--env FINANCE_EVAL_OUTPUT_PATH=$(FINANCE_EVAL_OUTPUT_PATH) \
			--env OUTPUT_PATH=$(FINANCE_EVAL_OUTPUT_PATH) \
			launch/finance-evaluation.yaml

		# Cleanup and run evaluation script
		sky down -y finance-evaluation

log-results:
	python eval/log_results.py --run-id $(WB_RUN_ID) --results $(RESULTS)